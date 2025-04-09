import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: torch.Tensor,
        num_heads: torch.Tensor,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None
    ):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

        self.linear_q = Linear(self.d_k * num_heads, d_model, **kwargs)
        self.linear_k = Linear(self.d_k * num_heads, d_model, **kwargs)
        self.linear_v = Linear(self.d_v * num_heads, d_model, **kwargs)
        self.linear_o = Linear(d_model, self.d_v * num_heads, **kwargs)

    def combine_head_dim(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        i_dim = dim // self.num_heads
        x = x.view(batch_size, seq_len, self.num_heads, i_dim)
        x = torch.einsum("bshd->bhsd", x)
        return x.reshape(batch_size * self.num_heads, seq_len, -1)

    def concat_heads(self, x: torch.Tensor):
        bh, seq_len, in_feature = x.size()
        batch_size = bh // self.num_heads
        i_dim = in_feature
        x = x.view(batch_size, self.num_heads, seq_len, i_dim)
        x = torch.einsum("bhsd->bshd", x.contiguous())
        return x.reshape(batch_size, seq_len, i_dim * self.num_heads)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = self.combine_head_dim(self.linear_q(x))
        K = self.combine_head_dim(self.linear_k(x))
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = self.combine_head_dim(self.linear_v(x))

        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1) == 0
        x = scaled_dot_product_attention(Q, K, V, mask)
        x = self.concat_heads(x)

        x = self.linear_o(x)
        return x
