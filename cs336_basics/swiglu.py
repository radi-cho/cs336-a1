import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SiLU(nn.Module):
    def __init__(self) -> None:
        super(SiLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super(SwiGLU, self).__init__()
        kwargs = {'device': device, 'dtype': dtype}

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)

        d_ff = (d_ff // 64) * 64

        self.w1: Linear = Linear(d_model, d_ff, **kwargs)
        self.w2: Linear = Linear(d_ff, d_model, **kwargs)
        self.w3: Linear = Linear(d_model, d_ff, **kwargs)
        self.silu: SiLU = SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        
        result = self.w2(self.silu(x1) * x2)
        
        return result
