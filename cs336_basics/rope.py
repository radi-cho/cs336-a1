import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, device=device).float()

        x = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        x = torch.einsum('i,j->ij', positions, x)
        sin, cos = x.sin(), x.cos()

        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd = x1 * sin + x2 * cos

        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)
        return x_rotated
