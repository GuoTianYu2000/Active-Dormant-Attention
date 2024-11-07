import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from typing import *
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXFlashAttention2, GPTNeoXMLP, GPTNeoXModel, GPTNeoXForCausalLM, apply_rotary_pos_emb, GPTNeoXLayer, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging

logger = logging.get_logger(__name__)
from src.hook import Hook


def gpt_neox_modified_mlp_forward(mlp: GPTNeoXMLP, hook: Hook, context: Dict[str, Any], hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = mlp.dense_h_to_4h(hidden_states)
    hidden_states = hook("mlp_up_proj", hidden_states, context)
    hidden_states = mlp.act(hidden_states)
    hidden_states = hook("mlp_activations", hidden_states, context)
    hidden_states = mlp.dense_4h_to_h(hidden_states)
    hidden_states = hook("mlp_down_proj", hidden_states, context)
    return hidden_states


def _attn_projections_and_rope(
    attn: GPTNeoXAttention,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_past: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
):
    has_layer_past = layer_past is not None
    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = attn.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (attn.num_attention_heads, 3 * attn.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : attn.head_size].permute(0, 2, 1, 3)
    key = qkv[..., attn.head_size : 2 * attn.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * attn.head_size :].permute(0, 2, 1, 3)

    query = hook("q_proj", query, context)
    key = hook("k_proj", key, context)
    value = hook("v_proj", value, context)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : attn.rotary_ndims]
    query_pass = query[..., attn.rotary_ndims :]
    key_rot = key[..., : attn.rotary_ndims]
    key_pass = key[..., attn.rotary_ndims :]

    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = attn.rotary_emb(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    key = torch.cat((key, key_pass), dim=-1)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    present = (key, value) if use_cache else None

    query = hook("q_proj_postrope", query, context)
    key = hook("k_proj_postrope", key, context)
    value = hook("v_proj_postrope", value, context)

    return query, key, value, present


def _attn_eager(attn: GPTNeoXAttention, hook: Hook, context: Dict[str, Any], query, key, value, attention_mask=None, head_mask=None):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    # compute causal mask from causal mask buffer
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    # dynamically increase the causal mask with the key length, if needed.
    if key_length > attn.bias.shape[-1]:
        attn._init_bias(key_length, device=key.device)
    causal_mask = attn.bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=attn.norm_factor,
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    attn_scores = hook("attn_logits", attn_scores, context)

    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.to(value.dtype)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = hook("attn_weights", attn_weights, context)

    attn_weights = attn.attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)

    attn_output = hook("attn_qkt_v", attn_output, context)

    return attn_output, attn_weights


def gpt_neox_modified_attn_eager_forward(
    attn: GPTNeoXAttention,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    # Apply attention-specific projections and rope
    query, key, value, present = _attn_projections_and_rope(
        attn,
        hook,
        context,
        hidden_states=hidden_states,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    # Compute attention
    attn_output, attn_weights = _attn_eager(attn, hook, context, query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = attn._merge_heads(attn_output, attn.num_attention_heads, attn.head_size)
    attn_output = attn.dense(attn_output)

    attn_output = hook("attn_output", attn_output, context)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def gpt_neox_modified_attn_flash_forward(
    attn: GPTNeoXFlashAttention2,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    # Apply attention-specific projections and rope
    query, key, value, present = _attn_projections_and_rope(
        attn,
        hook,
        context,
        hidden_states=hidden_states,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    query_length = query.shape[-2]

    # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
    target_dtype = value.dtype
    if query.dtype != target_dtype:
        query = query.to(target_dtype)
    if key.dtype != target_dtype:
        key = key.to(target_dtype)

    # Permute to get the expected shape for Flash Attention
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 / bfloat16 just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    input_dtype = query.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(attn.config, "_pre_quantization_dtype"):
            target_dtype = attn.config._pre_quantization_dtype
        else:
            target_dtype = attn.query_key_value.weight.dtype

        logger.warning_once(f"The input hidden states seems to be silently casted in float32, this might be related to" f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in" f" {target_dtype}.")

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    attention_dropout = attn.config.attention_dropout if attn.training else 0.0

    # Compute attention
    attn_weights = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length,
        dropout=attention_dropout,
        softmax_scale=attn.norm_factor,
        is_causal=attn.is_causal,
        use_top_left_mask=attn._flash_attn_uses_top_left_mask,
    )

    # Reshape outputs
    attn_output = attn_weights.reshape(attn_weights.shape[0], attn_weights.shape[1], attn.num_attention_heads * attn.head_size)

    attn_output = hook("attn_qkt_v", attn_output, context)

    attn_output = attn.dense(attn_output)

    attn_output = hook("attn_output", attn_output, context)

    outputs = (attn_output, layer_past)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def gpt_neox_modified_attn_forward(
    attn: GPTNeoXAttention,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    if isinstance(attn, GPTNeoXFlashAttention2):
        attention_layer_outputs = gpt_neox_modified_attn_flash_forward(
            attn,
            hook,
            context,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
    elif isinstance(attn, GPTNeoXAttention):
        attention_layer_outputs = gpt_neox_modified_attn_eager_forward(
            attn,
            hook,
            context,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
    else:
        raise ValueError(f"Unsupported attention type: {type(attn)}")

    return attention_layer_outputs


def gpt_neox_modified_layer_forward(
    layer: GPTNeoXLayer,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
):
    hidden_states = hook("layer_input_prenorm", hidden_states, context)
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = hook("layer_input_postnorm", hidden_states, context)

    attention_layer_outputs = gpt_neox_modified_attn_forward(
        layer.attention,
        hook,
        context,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
    attn_output = layer.post_attention_dropout(attn_output)
    outputs = attention_layer_outputs[1:]

    if layer.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        hook("post_attn_residual", hidden_states + attn_output, context)
        ln_hidden_states = layer.post_attention_layernorm(hidden_states)
        hook("post_attn_postnorm", ln_hidden_states, context)
        mlp_output = layer.mlp(ln_hidden_states)
        mlp_output = layer.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
        hook("post_mlp_residual", hidden_states, context)
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        hidden_states = hook("post_attn_residual", hidden_states, context)
        hidden_states = layer.post_attention_layernorm(attn_output)
        hidden_states = hook("post_attn_postnorm", hidden_states, context)
        mlp_output = gpt_neox_modified_mlp_forward(layer.mlp, hook, context, hidden_states)
        mlp_output = layer.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output
        hidden_states = hook("post_mlp_residual", hidden_states, context)

    if use_cache:
        outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
    else:
        outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

    return outputs


def gpt_neox_modified_model_forward(
    model: GPTNeoXModel,
    hook: Hook,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict
    use_cache = use_cache if use_cache is not None else model.config.use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * model.config.num_hidden_layers)
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = model.embed_in(input_ids)

    # Attention mask.
    attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
    if model._attn_implementation == "flash_attention_2":
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif model._attn_implementation == "sdpa" and not output_attentions and head_mask is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = model.get_head_mask(head_mask, model.config.num_hidden_layers)

    hidden_states = model.emb_dropout(inputs_embeds)

    if model.gradient_checkpointing and model.training:
        if use_cache:
            logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    presents = () if use_cache else None
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (layer, layer_past) in enumerate(zip(model.layers, past_key_values)):
        layer_context = {"layer_idx": i}
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if model.gradient_checkpointing and model.training:
            outputs = model._gradient_checkpointing_func(
                gpt_neox_modified_layer_forward,
                layer,
                hook,
                layer_context,
                hidden_states,
                attention_mask,
                position_ids,
                head_mask[i],
                use_cache,
                None,
                output_attentions,
            )
        else:
            outputs = gpt_neox_modified_layer_forward(
                layer,
                hook,
                layer_context,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)
        if output_attentions:
            all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = model.final_layer_norm(hidden_states)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )


def gpt_neox_modified_causal_lm_forward(
    causal_lm: GPTNeoXForCausalLM,
    hook: Hook,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
        only required when the model is used as a decoder in a Sequence to Sequence model.

        Contains pre-computed hidden-states (key and values in the causal_lm-attention blocks that can be used (see
        `past_key_values` input) to speed up sequential decoding.

        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
        `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config.is_decoder = True
    >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    ```"""
    return_dict = return_dict if return_dict is not None else causal_lm.config.use_return_dict

    outputs = gpt_neox_modified_model_forward(
        causal_lm.gpt_neox,
        hook,
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    lm_logits = causal_lm.embed_out(hidden_states)

    lm_loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    return CausalLMOutputWithPast(
        loss=lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


__all__ = [
    "gpt_neox_modified_attn_forward",
    "gpt_neox_modified_mlp_forward",
    "gpt_neox_modified_layer_forward",
    "gpt_neox_modified_model_forward",
    "gpt_neox_modified_causal_lm_forward",
]
