import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **kwargs))
        torch.nn.init.trunc_normal_(self.weight, 0.0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        try:
            return self.weight[token_ids]
        except:
            print(token_ids.shape)
            print(torch.max(token_ids))
            raise
