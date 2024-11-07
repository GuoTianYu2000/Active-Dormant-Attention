import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, OlmoForCausalLM, GPTNeoXForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.modified_forwards.llama import llama_modified_causal_lm_forward
from src.modified_forwards.olmo import olmo_modified_causal_lm_forward
from src.modified_forwards.gpt_neox import gpt_neox_modified_causal_lm_forward
from src.hook import Hook, IdentityHook


class ModelWrapper:
	def __init__(self, model: AutoModelForCausalLM):
		assert isinstance(model, (LlamaForCausalLM, OlmoForCausalLM, GPTNeoXForCausalLM))
		self.model: AutoModelForCausalLM = model
		if isinstance(model, LlamaForCausalLM):
			self.lm_forward = llama_modified_causal_lm_forward
		elif isinstance(model, OlmoForCausalLM):
			self.lm_forward = olmo_modified_causal_lm_forward
		elif isinstance(model, GPTNeoXForCausalLM):
			self.lm_forward = gpt_neox_modified_causal_lm_forward
		else:
			raise NotImplementedError("Model not implemented")

	def forward(
			self,
			input_ids: torch.LongTensor,
			hook: Hook = IdentityHook(),
	) -> CausalLMOutputWithPast:
		return self.lm_forward(
			causal_lm=self.model,
			input_ids=input_ids,
			hook=hook,
		)



__all__ = ["ModelWrapper"]
