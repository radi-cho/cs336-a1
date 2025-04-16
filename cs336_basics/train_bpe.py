import pickle
import regex as re
import multiprocessing
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Iterable

from cs336_basics.constants import CHARACTER_ACCUMULATION
from cs336_basics.tokenizer import pretokenize_text


def read_text_lines(file_path: str) -> Iterable[str]:
    with open(file_path, "r") as file:
        for line in file:
            yield line


def pretokenize_chunk(args) -> List[Tuple[bytes]]:
    text_chunk, naive = args
    return pretokenize_text(text_chunk, naive)


def pretokenize(
    text_lines: Iterable[str],
    naive: bool = False,
    num_proc: int = 4,
    special_tokens: List[str] = []
) -> Iterable[Tuple[bytes]]:
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_special_tokens)

    def process_lines():
        current_text = ""
        for line in text_lines:
            current_text += line
            if len(current_text) > CHARACTER_ACCUMULATION:
                segments = re.split(pattern, current_text)
                yield segments
                current_text = ""
        if current_text:
            segments = re.split(pattern, current_text)
            yield segments

    args_gen = ((doc, naive) for chunk in process_lines() for doc in chunk)

    with multiprocessing.Pool(num_proc) as pool:
        imap_obj = pool.imap_unordered(pretokenize_chunk, args_gen)
        for token_list in tqdm(imap_obj):
            yield from token_list


def get_index(
    frequency_table: Dict[Tuple[bytes, ...], int]
) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], set]]:
    pair_frequencies = defaultdict(int)
    pair_sequences = defaultdict(set)

    for seq in tqdm(frequency_table):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_frequencies[pair] += frequency_table[seq]
            pair_sequences[pair].add(seq)
            
    return pair_frequencies, pair_sequences


def apply_merges_update_index(
    frequency_table: Dict[Tuple[bytes, ...], int],
    pair: Tuple[bytes, bytes],
    pair_frequencies: Dict[Tuple[bytes, bytes], int],
    pair_sequences: Dict[Tuple[bytes, bytes], set]
) -> None:
    pair_first, pair_second = pair
    merged_pair = pair_first + pair_second

    for token_seq in list(pair_sequences[pair]):
        freq = frequency_table[token_seq]
        if freq <= 0:
            continue

        i, new_seq = 0, []

        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == pair_first and token_seq[i + 1] == pair_second:
                new_seq.append(merged_pair)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1

        new_seq_tuple = tuple(new_seq)
        seen_pairs = set()

        for i in range(len(token_seq) - 1):
            old_pair = (token_seq[i], token_seq[i + 1])
            pair_frequencies[old_pair] -= freq
            if pair_frequencies[old_pair] <= 0:
                del pair_frequencies[old_pair]

        for i in range(len(new_seq) - 1):
            new_pair = (new_seq[i], new_seq[i + 1])
            pair_frequencies[new_pair] += freq

            if new_pair not in seen_pairs:
                pair_sequences[new_pair].add(new_seq_tuple)
                seen_pairs.add(new_pair)

        if new_seq_tuple in frequency_table:
            frequency_table[new_seq_tuple] += freq
        else:
            frequency_table[new_seq_tuple] = freq

        frequency_table[token_seq] = 0

    pair_frequencies.pop(pair, None)
    pair_sequences.pop(pair, None)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    merges = []
    vocab = {}
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    vocab.update({(256 + i): item.encode() for i, item in enumerate(special_tokens)})

    text_lines = read_text_lines(input_path)
    pretokenized_sequence = pretokenize(text_lines, special_tokens=special_tokens)
    frequency_table = dict(Counter(pretokenized_sequence))

    with multiprocessing.Pool(4) as pool:
        for result in pool.imap_unordered(pretokenize, text_lines):
            for token, count in result.items():
                frequency_table[tuple(bytes([b]) for b in token)] += count

    print("Building index")
    pair_frequencies, pair_sequences = get_index(frequency_table)
    print("Training")

    progress = tqdm(total=vocab_size - len(vocab))
    while len(vocab) < vocab_size:
        if not pair_frequencies:
            break

        # Sort by count, then key
        candidate_pair = max(pair_frequencies.items(), key=lambda x: (x[1], x[0]))[0]
        vocab[len(vocab)] = candidate_pair[0] + candidate_pair[1]
        merges.append(candidate_pair)

        apply_merges_update_index(frequency_table, candidate_pair, pair_frequencies, pair_sequences)
        progress.update(1)
    progress.close()

    return vocab, merges


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    name: str
) -> None:
    with open(name + "_vocab.pickle", "wb") as file:
        pickle.dump(vocab, file)

    with open(name + "_merges.pickle", "wb") as file:
        pickle.dump(merges, file)


if __name__ == "__main__":
    # vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-train-small.txt", 10000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    vocab, merges = train_bpe("../data/owt_train.txt", 32000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("./fixtures/sample1.txt", 257 + 8, ["<|endoftext|>"])
    # vocab, merges = train_bpe("./fixtures/sample2.txt", 257 + 8, ["<|endoftext|>"])
    save_tokenizer(vocab, merges, "tokenizer")
    # print(merges)
