import torch
import torch.nn as nn

from cs336_basics.multihead_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: str | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}

        self.attn = MultiHeadSelfAttention(d_model, num_heads, True, theta, max_seq_len, **kwargs)
        self.ln1 = RMSNorm(d_model, **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)
        self.ffn = SwiGLU(d_model, d_ff)
        print(d_ff) # outputs 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions=torch.arange(x.size(1)))
        z = y + self.ffn(self.ln2(y))
        return z
