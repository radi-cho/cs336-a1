import torch
import numpy as np
import numpy.typing as npt
from typing import Tuple


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    begin = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    input = np.stack([dataset[bi : bi + context_length] for bi in begin])
    label = np.stack([dataset[bi + 1 : bi + context_length + 1] for bi in begin])

    input_tensor = torch.tensor(input, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label, dtype=torch.long, device=device)

    return input_tensor, label_tensor
