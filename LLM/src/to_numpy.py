import torch


def to_numpy(tensor):
    return tensor.detach().cpu().to(torch.float).numpy()


__all__ = ["to_numpy"]
