import regex as re
from typing import List, Tuple
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


def merge(
    sequence: Tuple[bytes],
    pair: Tuple[bytes]
) -> Tuple[bytes]:
    merged, i = [], 0
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i:i+2] == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(sequence[i])
            i += 1

    return tuple(merged)


def split_by_delimiters(
    delimiters: List[str],
    text: str
) -> List[str]:
    pattern = f"[{''.join(map(re.escape, delimiters))}]"
    return re.split(pattern, text)
