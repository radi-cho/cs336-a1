import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.multihead_attention import MultiHeadSelfAttention


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
        self.ffn = SwiGLU(d_model, d_ff, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions=torch.arange(x.size(1)))
        z = y + self.ffn(self.ln2(y))
        return z


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: str | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}

        self.token_embeddings = Embedding(vocab_size, d_model, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    **kwargs
                )
            )

        self.ln_final = RMSNorm(d_model, **kwargs)
        self.lm_head = Linear(d_model, vocab_size, **kwargs)
        self.token_embeddings.weight = self.lm_head.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
