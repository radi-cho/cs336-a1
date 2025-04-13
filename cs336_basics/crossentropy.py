import torch

from cs336_basics.softmax import softmax


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = softmax(inputs, -1, return_log=True)
    selected_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    return -torch.mean(selected_probs)
