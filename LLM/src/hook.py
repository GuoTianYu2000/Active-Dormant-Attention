import torch
from typing import Callable, Dict, Any, Tuple, List
from collections.abc import Collection
from .entropy import row_headwise_average_normalized_entropy, column_headwise_maximum_normalized_entropy


class Hook:
	def __init__(
			self, target_name: str, 
			target_condition: Callable[[torch.Tensor, Dict[str, Any]], bool]
	) -> None:
		self.target_name: str = target_name
		self.target_condition: Callable[[torch.Tensor, Dict[str, Any]], bool] = target_condition

	def __call__(
			self, name: str, tensor: torch.Tensor, context: Dict[str, Any],
	) -> torch.Tensor:
		if name == self.target_name and self.target_condition(tensor, context):
			return self.intervention(tensor, context)
		else:
			return tensor

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		return tensor


class CompositeHook(Hook):
	def __init__(self, hooks: List[Hook]) -> None:
		self.hooks: List[Hook] = hooks

	def __call__(
			self, name: str, tensor: torch.Tensor, context: Dict[str, Any],
	) -> Any:
		output = tensor
		for hook in self.hooks:
			output = hook(name, output, context)
		return output


class IdentityHook(Hook):
	def __init__(self) -> None:
		super().__init__(target_name="n/a", target_condition=lambda tensor, context: False)


class RecordHook(Hook):
	def __init__(self, target_name: str, record_buffer: List, target_layers: Collection[int]) -> None:
		self.record_buffer: List = record_buffer
		self.target_layers: Collection[int] = target_layers

		def target_condition(tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
			layer_idx = context.get("layer_idx", -1)
			return layer_idx in self.target_layers

		super().__init__(target_name=target_name, target_condition=target_condition)

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		self.record_buffer.append(tensor)
		return tensor



class ZeroOutAttentionHeadHook(Hook):
	def __init__(self, target_layer_head_indices: Dict[int, Collection[int]]) -> None:
		self.target_layer_head_indices: Dict[int, Collection[int]] = target_layer_head_indices

		def target_condition(tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
			layer_idx = context.get("layer_idx", -1)
			return layer_idx in self.target_layer_head_indices

		super().__init__(target_name="attn_weights", target_condition=target_condition)

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		attn_weights = tensor
		layer_idx = context.get("layer_idx")
		if layer_idx is not None:
			for head_idx in self.target_layer_head_indices[layer_idx]:
				attn_weights[..., head_idx, :, :] = 0
		return attn_weights


class MoveAttentionToBOSHook(Hook):
	def __init__(self, target_layer_head_indices: Dict[int, Collection[int]]) -> None:
		self.target_layer_head_indices: Dict[int, Collection[int]] = target_layer_head_indices

		def target_condition(tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
			layer_idx = context.get("layer_idx", -1)
			return layer_idx in self.target_layer_head_indices

		super().__init__(target_name="attn_weights", target_condition=target_condition)

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		attn_weights = tensor
		layer_idx = context.get("layer_idx")
		if layer_idx is not None:
			for head_idx in self.target_layer_head_indices[layer_idx]:
				attn_weights[..., head_idx, :, :] = 0
				attn_weights[..., head_idx, :, 0] = 1
		return attn_weights
	

class ZeroOutAttentionSinkHeadWithEntropyDetectionHook(Hook):
	def __init__(self, row_entropy_threshold: float, column_entropy_threshold: float) -> None:
		self.row_entropy_threshold = row_entropy_threshold
		self.column_entropy_threshold = column_entropy_threshold

		def target_condition(tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
			return True

		super().__init__(target_name="attn_weights", target_condition=target_condition)

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		attn_weights = tensor
		headwise_row_entropy = row_headwise_average_normalized_entropy(attn_weights)
		headwise_column_entropy = column_headwise_maximum_normalized_entropy(attn_weights)
		condition = (headwise_row_entropy < self.row_entropy_threshold) & (headwise_column_entropy > self.column_entropy_threshold)
		attn_weights[condition, :, :] = 0
		return attn_weights
	

class MoveAttentionToBOSWithEntropyDetectionHook(Hook):
	def __init__(self, row_entropy_threshold: float, column_entropy_threshold: float) -> None:
		self.row_entropy_threshold = row_entropy_threshold
		self.column_entropy_threshold = column_entropy_threshold

		def target_condition(tensor: torch.Tensor, context: Dict[str, Any]) -> bool:
			return True

		super().__init__(target_name="attn_weights", target_condition=target_condition)

	def intervention(
			self, tensor: torch.Tensor, context: Dict[str, Any]
	) -> torch.Tensor:
		attn_weights = tensor
		headwise_row_entropy = row_headwise_average_normalized_entropy(attn_weights)
		headwise_column_entropy = column_headwise_maximum_normalized_entropy(attn_weights)
		condition = (headwise_row_entropy < self.row_entropy_threshold) & (headwise_column_entropy > self.column_entropy_threshold)
		attn_weights[condition, :, :] = 0
		attn_weights[condition, :, 0] = 1
		return attn_weights



__all__ = ["Hook", "IdentityHook", "ZeroOutAttentionHeadHook", "MoveAttentionToBOSHook", "ZeroOutAttentionSinkHeadWithEntropyDetectionHook", "MoveAttentionToBOSWithEntropyDetectionHook"]
