import time
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
import numpy as np

from cs336_basics.constants import PAT

def pretokenize_text(
    text: str,
    naive: bool = False
) -> List[Tuple[bytes]]:
    if naive:
        token_list = text.split()
    else:
        token_list = [m.group() for m in PAT.finditer(text)]

    return [tuple(bytes([b]) for b in token.encode()) for token in token_list]


def merge(sequence: Tuple[bytes], pair: Tuple[bytes]) -> Tuple[bytes]:
    merged, i = [], 0
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i+1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(sequence[i])
            i += 1

    return tuple(merged)


def get_pairs(sequence: Tuple[bytes]) -> set:
    return set((sequence[i], sequence[i+1]) for i in range(len(sequence) - 1))


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ) -> None:
        self.vocab = vocab
        self.reverse_lookup = {token: id for id, token in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens.sort(key=lambda token: (-len(token), token), reverse=False)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return Tokenizer(vocab, merges, special_tokens)

    def encode_helper(self, text: str) -> List[int]:
        result = []
        pretokenized = pretokenize_text(text)

        for pretoken in pretokenized:
            pairs = get_pairs(pretoken)
            if pairs:
                while True:
                    min_rank = float("inf")
                    best_pair = None
                    for pair in pairs:
                        rank = self.merge_ranks.get(pair)
                        if rank is not None and rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                    if best_pair is None:
                        break

                    pretoken = merge(pretoken, best_pair)
                    pairs = get_pairs(pretoken)

            for item in pretoken:
                result.append(self.reverse_lookup[item])

        return result

    def encode(self, text: str) -> List[int]:
        result = []
        start_idx = 0
        for special_token in self.special_tokens:
            special_idx = text.find(special_token, start_idx)

            while special_idx != -1:
                before_special = text[start_idx:special_idx]
                if before_special:
                    result.extend(self.encode_helper(before_special))

                result.append(self.reverse_lookup[special_token.encode()])
                start_idx = special_idx + len(special_token)
                special_idx = text.find(special_token, start_idx)

        if start_idx < len(text):
            result.extend(self.encode_helper(text[start_idx:]))

        return result

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            # yield from self.encode(text)
            yield len(text.encode()), self.encode(text)

    def decode(
        self,
        ids: List[int]
    ) -> str:
        byte_sequence = b""
        for id in ids:
            byte_sequence += self.vocab[id]

        return byte_sequence.decode(errors="replace")

if __name__ == "__main__":
    with open("../archive/open_tokenizer_vocab.pickle", "rb") as f:
    # with open("../archive/tiny_tokenizer_vocab.pickle", "rb") as f:
        vocab = pickle.load(f)

    with open("../archive/open_tokenizer_merges.pickle", "rb") as f:
    # with open("../archive/tiny_tokenizer_merges.pickle", "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    total_lines = 2301019 # 94568885
    # total_lines = 15600057 # 157832
    byte_sum = 0
    token_count = 0
    counter = 0

    token_list = []

    start = time.time()
    with open("../data/owt_valid.txt", "r") as f:
    # with open("../data/TinyStoriesV2-GPT4-train.txt", "r") as f:
        for byte_length, encoding in tqdm(tokenizer.encode_iterable(f), total=total_lines):
            # if counter >= total_lines:
            #     break
            # byte_sum += byte_length
            # token_count += len(encoding)
            token_list.extend(encoding)
            counter += 1

    end = time.time()
    print(f"Time: {end - start:.4f}s")
    # print(f"Throughput: {byte_sum / (end - start):.4f}s")
    # print(f"Compression: {byte_sum / token_count:.4f}")
    token_array = np.array(token_list, dtype=np.uint16)  # or np.uint32 depending on tokenizer vocab size

    # Save to file
    np.save("owt_valid.npy", token_array)
    # np.save("tiny_train.npy", token_array)
