import pickle
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

from cs336_basics.common import pretokenize_text, merge

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        self.reverse_lookup = {token: id for id, token in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens or []
        # TODO: Is this the right idea?
        self.special_tokens.sort(key=lambda token: (-len(token), token), reverse=False)
        self.special_token_bytes = [tok.encode("utf-8") for tok in self.special_tokens]

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

    def encode_helper(self, text: str):
        result = []
        pretokenized = pretokenize_text(text)

        for pretoken in pretokenized:
            for pair in self.merges: # TODO: Might be too slow
                pretoken = merge(pretoken, pair)

            for item in pretoken:
                result.append(self.reverse_lookup[item])

        return result

    def encode(
        self,
        text: str
    ) -> List[int]:
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
            yield from self.encode(text)

    def decode(
        self,
        ids: List[int]
    ) -> str:
        byte_sequence = b""
        for id in ids:
            byte_sequence += self.vocab[id]
        
        # TODO: FIX, Ask about what happens if this cannot be decoded?
        try:
            return byte_sequence.decode()
        except:
            return byte_sequence
