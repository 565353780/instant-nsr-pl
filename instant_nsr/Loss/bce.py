import torch


def binary_cross_entropy(input, target):
    """Numerically stable binary cross entropy."""
    return -(target * torch.log(input + 1e-8) + (1 - target) * torch.log(1 - input + 1e-8)).mean()
