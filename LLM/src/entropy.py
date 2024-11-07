import torch


def row_tokenwise_normalized_entropy(attn_weights):
    attn_probs = torch.where(
        attn_weights > 0, attn_weights / attn_weights.sum(dim=-1, keepdim=True), 0
    )
    entropy = torch.sum(
        torch.where(
            attn_probs > 0,
            -attn_probs * torch.log(attn_probs),
            0,
        ),
        dim=-1,
    )
    normalization = torch.log(
        torch.arange(1, attn_weights.shape[-1] + 1, device=attn_weights.device)
    )
    normalization = torch.reshape(
        normalization, tuple([1 for _ in range(len(entropy.shape) - 1)] + [-1])
    )
    return torch.where(normalization > 0, entropy / normalization, 0)


def row_headwise_average_normalized_entropy(attn_weights):
    return torch.mean(row_tokenwise_normalized_entropy(attn_weights), dim=-1)


def column_tokenwise_normalized_entropy(attn_weights):
    attn_probs = torch.where(
        attn_weights > 0, attn_weights / attn_weights.sum(dim=-2, keepdim=True), 0
    )
    entropy = torch.sum(
        torch.where(
            attn_probs > 0,
            -attn_probs * torch.log(attn_probs),
            0,
        ),
        dim=-2,
    )
    normalization = torch.log(
        attn_weights.shape[-2]
        - torch.arange(0, attn_weights.shape[-2], device=attn_weights.device)
    )
    normalization = torch.reshape(
        normalization, tuple([1 for _ in range(len(entropy.shape) - 1)] + [-1])
    )
    return torch.where(normalization > 0, entropy / normalization, 0)


def column_headwise_maximum_normalized_entropy(attn_weights):
    return torch.max(column_tokenwise_normalized_entropy(attn_weights), dim=-1)[0]


__all__ = [
    "row_tokenwise_normalized_entropy",
    "row_headwise_average_normalized_entropy",
    "column_tokenwise_normalized_entropy",
    "column_headwise_maximum_normalized_entropy",
]
