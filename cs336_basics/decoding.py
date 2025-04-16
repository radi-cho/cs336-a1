import torch
import pickle
from tqdm import tqdm

from cs336_basics.softmax import softmax
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.transformer import Transformer
from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT_TOKEN = 256


def top_p_sample(logits: torch.Tensor, p: float) -> int:
    logits, ind = torch.sort(logits, descending=True)
    probs = softmax(logits, dim=-1)

    cutoff = torch.cumsum(probs, dim=-1) > p
    if torch.any(cutoff):
        last_included = torch.nonzero(cutoff, as_tuple=False)[0].item()
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list[int]:
    seq = list(inp_seq)
    for _ in tqdm(range(max_tokens)):
        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)

        logits = generator(input_tensor)
        logits = logits[0, -1, :]
        logits = logits / temperature

        next_token = top_p_sample(logits, top_p)
        seq.append(next_token)

        if next_token == eos_token_id:
            break

    return seq


if __name__ == "__main__":
    # with open("../archive/open_tokenizer_vocab.pickle", "rb") as f:
    with open("../archive/tiny_tokenizer_vocab.pickle", "rb") as f:
        vocab = pickle.load(f)

    # with open("../archive/open_tokenizer_merges.pickle", "rb") as f:
    with open("../archive/tiny_tokenizer_merges.pickle", "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    model = Transformer(10000, 256, 512, 4, 16, 1344, 10000, "cuda")
    # model = Transformer(10000, 256, 512, 4, 16, 1344, 10000, "cuda")
    load_checkpoint("checkpoint_lr1e-3.pt", model)
    inp = tokenizer.encode("Once ")
    out = decode(model, inp, max_tokens=256, device="cuda")

    print(tokenizer.decode(out))
