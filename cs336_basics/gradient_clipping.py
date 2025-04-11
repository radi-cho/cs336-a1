import torch
import torch.nn as nn
from typing import Iterable


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    total_norm = torch.sqrt(sum(
        (p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None)
    ))

    clip_coef = max_l2_norm / (total_norm + eps)

    if total_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
