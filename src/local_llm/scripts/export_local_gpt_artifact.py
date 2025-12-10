from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Keep regex fully consistent with src/training/data_loader.py
TOKEN_PATTERN = re.compile(r"\w+|\S", re.UNICODE)

# Special tokens consistent with CorpusDataLoader
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


class SimpleTokenizer:
    """
    Minimal tokenizer/decoder that mirrors the training DataLoader behavior:
    - Regex tokenization: r"\\w+|\\S"
    - Optional lowercasing (True by default, same as loader)
    - Uses a persisted vocab.json of the form:
        {
          "vocab": ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "alice", ...],
          "token_to_id": {"<PAD>": 0, "<BOS>": 1, ...},
          "id_to_token": {"0": "<PAD>", "1": "<BOS>", ...},
          "lowercase": true
        }
    Features:
    - encode/encode_with_bos/encode_with_bos_eos
    - batch_encode / batch_decode
    - improved detokenization rules (compact punctuation)
    - strict special token validation
    """

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab: List[str] = data["vocab"]
        # token_to_id may contain ints or strings; coerce to int
        self.token_to_id: Dict[str, int] = {}
        for k, v in data["token_to_id"].items():
            self.token_to_id[k] = int(v) if isinstance(v, str) else v
        # id_to_token is string keys in file; coerce to int
        self.id_to_token: Dict[int, str] = {
            int(k): v for k, v in data["id_to_token"].items()
        }
        self.lowercase: bool = bool(data.get("lowercase", True))

        # Validate special tokens ordering and ids
        for i, st in enumerate(SPECIAL_TOKENS):
            if i >= len(self.vocab) or self.vocab[i] != st:
                raise ValueError(
                    f"Vocab special token mismatch at index {i}: expected {st}, got {self.vocab[i] if i < len(self.vocab) else '<out of range>'}"
                )
            if (
                self.token_to_id.get(st, None) != i
                or self.id_to_token.get(i, None) != st
            ):
                raise ValueError(
                    f"Special token mapping mismatch for {st}: token_to_id={self.token_to_id.get(st)} id_to_token[{i}]={self.id_to_token.get(i)}"
                )

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return TOKEN_PATTERN.findall(text)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(t, UNK_ID) for t in self.tokenize(text)]

    def encode_with_bos(self, text: str) -> List[int]:
        return [BOS_ID] + self.encode(text)

    def encode_with_bos_eos(self, text: str) -> List[int]:
        return [BOS_ID] + self.encode(text) + [EOS_ID]

    def batch_encode(
        self, texts: Iterable[str], add_bos: bool = True, add_eos: bool = False
    ) -> List[List[int]]:
        out: List[List[int]] = []
        for t in texts:
            if add_bos and add_eos:
                out.append(self.encode_with_bos_eos(t))
            elif add_bos:
                out.append(self.encode_with_bos(t))
            elif add_eos:
                out.append(self.encode(t) + [EOS_ID])
            else:
                out.append(self.encode(t))
        return out

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        toks = []
        for i in ids:
            if strip_special and i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            toks.append(self.id_to_token.get(i, "<UNK>"))

        # Naive detokenization: join with space then fix common punctuation spacing
        text = " ".join(toks)
        # Remove space before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # Tighten brackets/quotes
        text = re.sub(r"\s+([\)\]\}])", r"\1", text)
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)
        # Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def batch_decode(
        self, batch_ids: Iterable[List[int]], strip_special: bool = True
    ) -> List[str]:
        return [self.decode(ids, strip_special=strip_special) for ids in batch_ids]


def _cli():
    ap = argparse.ArgumentParser(description="SimpleTokenizer smoke test")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json")
    ap.add_argument("--text", default="Hello, world!", help="Text to encode/decode")
    ap.add_argument(
        "--bos-eos", action="store_true", default=False, help="Add BOS/EOS in encoding"
    )
    args = ap.parse_args()

    tok = SimpleTokenizer(args.vocab)
    if args.bos_eos:
        ids = tok.encode_with_bos_eos(args.text)
    else:
        ids = tok.encode_with_bos(args.text)
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))


if __name__ == "__main__":
    _cli()
