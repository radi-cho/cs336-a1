# This file is used to run tokenizer experiments and measurements
import pickle
from typing import Tuple, List

def find_longest_tokens(path: str, top_n: int = 20) -> List[Tuple[int, bytes]]:
    vocab = pickle.load(open(path, "rb"))
    sorted_tokens = sorted(vocab.items(), key=lambda item: len(item[1]), reverse=True)
    return sorted_tokens[:top_n]

if __name__ == "__main__":
    longest_tokens = find_longest_tokens("../archive/open_tokenizer_vocab.pickle")
    for tid, token in longest_tokens:
        print(f"id: {tid}, length: {len(token)}, token: {token}")
