import json
import regex as re
from typing import Iterable, Iterator
from basics.BPE import PAT, bpe_tokenizer


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = [tuple(pair) for pair in merges]
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.special_token_set = set(self.special_tokens)

        existing = set(self.vocab.values())
        next_id = max(self.vocab) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing:
                self.vocab[next_id] = token_bytes
                existing.add(token_bytes)
                next_id += 1

        self.token_to_id = {tok: i for i, tok in self.vocab.items()}
        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)

        if self.special_tokens:
            self.special_pattern = "(" + "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
        else:
            self.special_pattern = None

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        u2b = {u: b for b, u in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = {
                token_id: bytes(u2b[ch] for ch in token_str)
                for token_str, token_id in json.load(f).items()
            }

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                pieces = line.rstrip("\n").split(" ")
                if len(pieces) != 2:
                    continue
                merges.append(tuple(bytes(u2b[ch] for ch in p) for p in pieces))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        cached = self.cache.get(pretoken)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in pretoken]
        while len(parts) > 1:
            best_rank = None
            best_pos = -1
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_pos < 0:
                break
            parts[best_pos:best_pos + 2] = [parts[best_pos] + parts[best_pos + 1]]

        ids = [self.token_to_id[p] for p in parts]
        self.cache[pretoken] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_pattern is None:
            parts = [text]
        else:
            parts = re.split(self.special_pattern, text)

        for part in parts:
            if not part:
                continue
            if part in self.special_token_set:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue
            for match in re.finditer(PAT, part):
                ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")