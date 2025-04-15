import torch
from cs336_basics import softmax


END_OF_TEXT_TOKEN = 256


def top_p_sample(logits: torch.Tensor, p: float) -> int:
    logits, ind = torch.sort(logits, descending=True)
    probs = softmax(logits, dim=-1)

    cutoff = torch.cumsum(probs, dim=-1) > p
    if torch.any(cutoff):
        last_included = torch.argmax(cutoff).item()
        logits, ind = logits[:last_included + 1], ind[:last_included + 1]
        probs = softmax(logits, dim=-1)

    return ind[torch.multinomial(probs, num_samples=1)].item()


@torch.no_grad()
def decode(
    generator,
    inp_seq: list[int],
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int = END_OF_TEXT_TOKEN,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> list[int]:
    seq = list(inp_seq)
    for _ in range(max_tokens):
        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)

        logits = generator(in_indices=input_tensor, **kwargs)
        logits = logits[0, -1, :]
        logits = logits / temperature

        next_token = top_p_sample(logits, top_p)
        seq.append(next_token)

        if next_token == eos_token_id:
            break

    return seq
