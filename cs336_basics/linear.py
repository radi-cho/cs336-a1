import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty(in_features, out_features, **kwargs))
        std = 2/(in_features + out_features)
        torch.nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
