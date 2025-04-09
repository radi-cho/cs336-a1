import torch
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    x = Q @ K.transpose(-2, -1)
    x /= torch.sqrt(torch.tensor(K.size(-1), dtype=K.dtype, device=K.device))

    if mask is not None:
        x = x.masked_fill(~mask, float("-inf"))

    x = softmax(x, dim=-1)
    x = x @ V

    return x
