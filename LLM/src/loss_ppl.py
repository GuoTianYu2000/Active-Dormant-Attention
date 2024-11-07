import torch
from torch.nn.functional import cross_entropy


def ce_loss(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = cross_entropy(shift_logits, shift_labels)
    return loss


def perplexity(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    loss = ce_loss(logits, labels)
    return perplexity_from_loss(loss)


def perplexity_from_loss(loss: torch.FloatTensor) -> torch.FloatTensor:
    ppl = torch.exp(loss)
    return ppl


__all__ = [
    "ce_loss", "perplexity", "perplexity_from_loss"
]
