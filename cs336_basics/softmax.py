import torch


def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals, _ = tensor.max(dim=dim, keepdim=True)
    exp_tensor = torch.exp(tensor - max_vals)
    sum_exp = exp_tensor.sum(dim=dim, keepdim=True)

    return exp_tensor / sum_exp
