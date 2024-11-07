import math
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.olmo.modeling_olmo import BaseModelOutputWithPast, CausalLMOutputWithPast, OlmoAttention, OlmoDecoderLayer, OlmoFlashAttention2, OlmoForCausalLM, OlmoMLP, OlmoModel, apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging, is_torchdynamo_compiling

logger = logging.get_logger(__name__)

from src.hook import Hook


def olmo_modified_mlp_forward(mlp: OlmoMLP, hook: Hook, context: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    gate_proj = mlp.gate_proj(x)
    gate_proj = hook("mlp_gate_proj", gate_proj, context)
    up_proj = mlp.up_proj(x)
    up_proj = hook("mlp_up_proj", up_proj, context)
    activations = mlp.act_fn(gate_proj)
    activations = hook("mlp_activations", activations, context)
    intermediate_states = activations * up_proj
    intermediate_states = hook("mlp_intermediate_states", intermediate_states, context)
    down_proj = mlp.down_proj(intermediate_states)
    down_proj = hook("mlp_down_proj", down_proj, context)
    return down_proj


def olmo_modified_self_attn_eager_forward(
    attn: OlmoAttention,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    if attn.config.clip_qkv is not None:
        query_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)
        key_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)
        value_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)

    query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    query_states = hook("q_proj", query_states, context)
    key_states = hook("k_proj", key_states, context)
    value_states = hook("v_proj", value_states, context)

    cos, sin = attn.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, attn.num_key_value_groups)
    value_states = repeat_kv(value_states, attn.num_key_value_groups)

    query_states = hook("q_proj_postrope", query_states, context)
    key_states = hook("k_proj_postrope", key_states, context)
    value_states = hook("v_proj_postrope", value_states, context)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

    attn_weights = hook("attn_logits", attn_weights, context)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = hook("attn_weights", attn_weights, context)
    attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
        raise ValueError(f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = hook("attn_softmax_qkt_v", attn_output, context)

    attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

    attn_output = attn.o_proj(attn_output)

    attn_output = hook("attn_output", attn_output, context)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def olmo_modified_self_attn_flash_forward(
    attn: OlmoFlashAttention2,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    if attn.config.clip_qkv is not None:
        query_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)
        key_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)
        value_states.clamp_(min=-attn.config.clip_qkv, max=attn.config.clip_qkv)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    query_states = hook("q_proj", query_states, context)
    key_states = hook("k_proj", key_states, context)
    value_states = hook("v_proj", value_states, context)

    cos, sin = attn.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

    query_states = hook("q_proj_postrope", query_states, context)
    key_states = hook("k_proj_postrope", key_states, context)
    value_states = hook("v_proj_postrope", value_states, context)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = attn.attention_dropout if attn.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (OlmoRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(attn.config, "_pre_quantization_dtype"):
            target_dtype = attn.config._pre_quantization_dtype
        else:
            target_dtype = attn.q_proj.weight.dtype

        logger.warning_once(f"The input hidden states seems to be silently casted in float32, this might be related to" f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in" f" {target_dtype}.")

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        use_top_left_mask=attn._flash_attn_uses_top_left_mask,
        is_causal=attn.is_causal,
    )

    attn_output = hook("attn_softmax_qkt_v", attn_output, context)

    attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size).contiguous()
    attn_output = attn.o_proj(attn_output)

    attn_output = hook("attn_output", attn_output, context)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def olmo_modified_self_attn_forward(
    attn: OlmoFlashAttention2,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(attn, OlmoFlashAttention2):
        return olmo_modified_self_attn_flash_forward(
            attn,
            hook,
            context,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )
    elif isinstance(attn, OlmoAttention):
        return olmo_modified_self_attn_eager_forward(
            attn,
            hook,
            context,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )
    else:
        raise ValueError(f"Unsupported attention implementation: {attn.config.attn_implementation}")


def olmo_modified_decoder_layer_forward(
    layer: OlmoDecoderLayer,
    hook: Hook,
    context: Dict[str, Any],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states
    hidden_states = hook("layer_input_prenorm", hidden_states, context)

    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = hook("layer_input_postnorm", hidden_states, context)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = olmo_modified_self_attn_forward(
        layer.self_attn,
        hook,
        context,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    hidden_states = hook("post_attn_residual", hidden_states, context)

    # Fully Connected
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hook("post_attn_postnorm", hidden_states, context)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    hidden_states = hook("post_mlp_residual", hidden_states, context)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def olmo_modified_model_forward(
        model: OlmoModel,
        hook: Hook,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if model.gradient_checkpointing and model.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not model.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for i, decoder_layer in enumerate(model.layers):
        layer_context = {"layer_idx": i}
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if model.gradient_checkpointing and model.training:
            layer_outputs = model._gradient_checkpointing_func(
                olmo_modified_decoder_layer_forward,
                decoder_layer,
                hook,
                layer_context,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = olmo_modified_decoder_layer_forward(
                decoder_layer,
                hook,
                layer_context,
                hidden_states,
                causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = model.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def olmo_modified_causal_lm_forward(
        causal_lm: OlmoForCausalLM,
        hook: Hook,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoForCausalLM

        >>> model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else causal_lm.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else causal_lm.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else causal_lm.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = olmo_modified_model_forward(
            causal_lm.model,
            hook,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = causal_lm.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, causal_lm.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "olmo_modified_self_attn_forward",
    "olmo_modified_mlp_forward",
    "olmo_modified_decoder_layer_forward",
    "olmo_modified_model_forward",
    "olmo_modified_causal_lm_forward",
]