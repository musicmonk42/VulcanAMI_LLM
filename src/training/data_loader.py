"""
High-performance text corpus loader and tokenizer for self-awareness training.

Features:
- Deterministic vocabulary build with frequency threshold & max vocab size.
- Preserves special tokens: <BOS>, <EOS>, <PAD>, <UNK>.
- Configurable insertion of BOS/EOS markers (per-line or single-stream).
- Optional deduplication of consecutive <EOS> tokens.
- Creates train/validation token streams with configurable split ratio.
- Efficient contiguous batch extraction (supports causal language modeling).
- Optional memory mapping for large corpora (placeholder hooks; disabled by default).
- Provides batch iterators returning (input_ids, target_ids) tuples.

Usage:
    loader = CorpusDataLoader(
        corpus_dir="data/corpus",
        max_vocab_size=50000,
        min_freq=2,
        seq_len=128,
        val_ratio=0.01,
        line_level_markers=True,
        deduplicate_eos=True
    )
    train_batch = loader.sample_train_batch(batch_size=32)
    val_batch = loader.sample_val_batch(batch_size=32)

Revision (robust tiny-corpus handling):
- Added graceful degradation when validation (or train) token stream too short for requested seq_len.
- Clamps effective sequence length per split to (stream_len - 1).
- Returns informative fields: sequences, seq_len, split, warning (if applicable).
- Never raises RuntimeError for tiny validation; instead returns empty batch with warning.
- Maintains original interface returning "sequences" list so downstream code using prior format still works.
"""

from __future__ import annotations
import os
import math
import random
import re
from typing import Dict, List, Tuple, Iterable, Optional, Any
from collections import Counter

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

TOKEN_PATTERN = re.compile(r"\w+|\S", re.UNICODE)


class CorpusDataLoader:
    """
    CorpusDataLoader builds a vocabulary and provides sampled batches for causal LM training.

    Improvements over previous version:
    - Added flags `line_level_markers` and `deduplicate_eos` to control BOS/EOS insertion behavior.
    - Avoids excessive inflation of BOS/EOS frequency (optional single-stream mode).
    - Provides method `tokens_info()` for deeper diagnostics.
    - Guard against pathological tiny corpora still creating at least one validation token.
    - Robust batch sampling that auto-clamps seq_len per split and returns warnings instead of crashing.
    """

    def __init__(
        self,
        corpus_dir: str,
        max_vocab_size: int = 50000,
        min_freq: int = 1,
        seq_len: int = 128,
        val_ratio: float = 0.01,
        seed: int = 42,
        shuffle_files: bool = True,
        lowercase: bool = True,
        line_level_markers: bool = True,
        deduplicate_eos: bool = True,
    ):
        """
        Args:
            corpus_dir: Directory containing text files.
            max_vocab_size: Maximum vocabulary size including special tokens.
            min_freq: Minimum frequency for a token to be included (after special tokens).
            seq_len: Requested sequence length for batches (used as upper bound; may be clamped).
            val_ratio: Fraction of total tokens reserved for validation (temporal split).
            seed: RNG seed for deterministic shuffling & sampling.
            shuffle_files: Shuffle file order before reading (affects token ordering).
            lowercase: Lowercase all text lines.
            line_level_markers: If True, insert <BOS> and <EOS> around EACH non-empty line.
                                If False, treat entire corpus as one stream: single leading <BOS>, trailing <EOS>.
            deduplicate_eos: If True, collapse consecutive <EOS> tokens to a single one.
        """
        self.corpus_dir = corpus_dir
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.seq_len = seq_len  # requested seq_len (will clamp dynamically per split)
        self.val_ratio = val_ratio
        self.seed = seed
        self.lowercase = lowercase
        self.line_level_markers = line_level_markers
        self.deduplicate_eos = deduplicate_eos

        self.rng = random.Random(seed)

        self._files = self._list_text_files()
        if shuffle_files:
            self.rng.shuffle(self._files)

        raw_tokens = self._read_all_tokens()
        if self.deduplicate_eos:
            raw_tokens = self._dedup_eos(raw_tokens)

        self.raw_token_count = len(raw_tokens)

        self.vocab, self.token_to_id, self.id_to_token = self._build_vocab(raw_tokens)
        self.vocab_size = len(self.vocab)

        encoded = [self.token_to_id.get(t, UNK_ID) for t in raw_tokens]
        self.train_tokens, self.val_tokens = self._split_tokens(encoded)

        self._train_len = len(self.train_tokens)
        self._val_len = len(self.val_tokens)

        # Internal indices for optional sequential streaming (not used in random window sampling)
        self._train_index = 0
        self._val_index = 0

    # ------------------------------------------------------------------ #
    # File Listing
    # ------------------------------------------------------------------ #
    def _list_text_files(self) -> List[str]:
        if not os.path.isdir(self.corpus_dir):
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        out: List[str] = []
        for root, _, files in os.walk(self.corpus_dir):
            for f in files:
                if f.lower().endswith((".txt", ".log", ".md")):
                    out.append(os.path.join(root, f))
        if not out:
            raise RuntimeError(f"No text files found in: {self.corpus_dir}")
        return out

    # ------------------------------------------------------------------ #
    # Token Reading
    # ------------------------------------------------------------------ #
    def _read_all_tokens(self) -> List[str]:
        if self.line_level_markers:
            return self._read_line_level_tokens()
        else:
            return self._read_stream_tokens()

    def _read_line_level_tokens(self) -> List[str]:
        """
        Insert <BOS> / <EOS> per non-empty line (legacy behavior).
        """
        tokens: List[str] = []
        for path in self._files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if self.lowercase:
                        line = line.lower()
                    line = line.strip()
                    if not line:
                        continue
                    line_tokens = TOKEN_PATTERN.findall(line)
                    if not line_tokens:
                        continue
                    tokens.append("<BOS>")
                    tokens.extend(line_tokens)
                    tokens.append("<EOS>")
        return tokens

    def _read_stream_tokens(self) -> List[str]:
        """
        Treat entire corpus as a single stream:
        - Add one leading <BOS>
        - Add one trailing <EOS>
        - Do not insert markers per line
        """
        stream: List[str] = ["<BOS>"]
        for path in self._files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if self.lowercase:
                        line = line.lower()
                    line = line.strip()
                    if not line:
                        continue
                    line_tokens = TOKEN_PATTERN.findall(line)
                    if line_tokens:
                        stream.extend(line_tokens)
        stream.append("<EOS>")
        return stream

    def _dedup_eos(self, toks: List[str]) -> List[str]:
        """
        Collapse consecutive <EOS> tokens into a single one to avoid frequency inflation.
        """
        if not toks:
            return toks
        deduped: List[str] = []
        prev_was_eos = False
        for t in toks:
            if t == "<EOS>":
                if not prev_was_eos:
                    deduped.append(t)
                prev_was_eos = True
            else:
                deduped.append(t)
                prev_was_eos = False
        return deduped

    # ------------------------------------------------------------------ #
    # Vocabulary Construction
    # ------------------------------------------------------------------ #
    def _build_vocab(
        self, tokens: List[str]
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        freq = Counter(tokens)
        # Remove special tokens if they appear naturally to avoid duplication
        for st in SPECIAL_TOKENS:
            if st in freq:
                del freq[st]

        # Filter by min_freq
        filtered = [(tok, c) for tok, c in freq.items() if c >= self.min_freq]
        # Sort by frequency desc then alphabetical for determinism
        filtered.sort(key=lambda x: (-x[1], x[0]))
        # Truncate to remaining slots
        remaining_slots = self.max_vocab_size - len(SPECIAL_TOKENS)
        trimmed = filtered[: max(0, remaining_slots)]

        vocab = SPECIAL_TOKENS + [t for t, _ in trimmed]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return vocab, token_to_id, id_to_token

    # ------------------------------------------------------------------ #
    # Train / Validation Split
    # ------------------------------------------------------------------ #
    def _split_tokens(self, encoded: List[int]) -> Tuple[List[int], List[int]]:
        total = len(encoded)
        val_count = max(1, int(total * self.val_ratio))
        # Reserve last val_count tokens for validation (temporal split)
        if val_count >= total:
            # Edge case: extremely small corpus; put a single token into validation if possible
            train = encoded
            val = encoded[-1:] if encoded else []
        else:
            train = encoded[:-val_count]
            val = encoded[-val_count:]
        return train, val

    # ------------------------------------------------------------------ #
    # Effective Sequence Length Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _max_possible_seq_len(stream_len: int) -> int:
        """
        Maximum usable causal LM sequence length for a stream of length L:
        We need x of length seq_len and y shifted by +1, so seq_len <= L - 1.
        """
        return max(0, stream_len - 1)

    def _clamp_seq_len_for_stream(self, requested_seq_len: int, stream_len: int) -> int:
        """
        Clamp the requested sequence length to what the stream can support.
        Ensures at least 1 if any tokens beyond BOS exist.
        """
        max_possible = self._max_possible_seq_len(stream_len)
        if max_possible <= 0:
            return 0
        return max(1, min(requested_seq_len, max_possible))

    # ------------------------------------------------------------------ #
    # Batch Sampling (ROBUST)
    # ------------------------------------------------------------------ #
    def sample_train_batch(self, batch_size: int) -> Dict[str, Any]:
        return self._sample_batch(
            token_stream=self.train_tokens,
            length=self._train_len,
            batch_size=batch_size,
            split_name="train",
        )

    def sample_val_batch(self, batch_size: int) -> Dict[str, Any]:
        return self._sample_batch(
            token_stream=self.val_tokens,
            length=self._val_len,
            batch_size=batch_size,
            split_name="val",
        )

    def _sample_batch(
        self, token_stream: List[int], length: int, batch_size: int, split_name: str
    ) -> Dict[str, Any]:
        """
        Robust batch sampling:
        - Clamps effective sequence length per split.
        - If stream too small (< 2 tokens), returns empty batch with a warning field.
        - Returns dict with: sequences (List[List[int]]), seq_len (effective), split, warning (optional).
        - Each sequence length is (seq_len + 1) because we include the next token (classic LM format).
        """
        requested_seq_len = self.seq_len
        eff_seq_len = self._clamp_seq_len_for_stream(requested_seq_len, length)

        # Not enough tokens to form even a single (input, target) pair
        if eff_seq_len == 0 or length < 2:
            return {
                "sequences": [],
                "seq_len": 0,
                "split": split_name,
                "warning": f"{split_name} stream too short (length={length}) for any sequence sampling.",
            }

        # We will extract windows of size (eff_seq_len + 1) from token_stream
        window_size = eff_seq_len + 1
        max_start = length - window_size

        # If max_start < 0 (should not happen due to clamping), adjust gracefully
        if max_start < 0:
            # Reduce seq_len further
            eff_seq_len = self._clamp_seq_len_for_stream(eff_seq_len - 1, length)
            if eff_seq_len <= 0:
                return {
                    "sequences": [],
                    "seq_len": 0,
                    "split": split_name,
                    "warning": f"{split_name} stream cannot provide even minimal sequence.",
                }
            window_size = eff_seq_len + 1
            max_start = max(0, length - window_size)

        sequences: List[List[int]] = []
        if max_start <= 0:
            # Only one possible window: start at 0
            base_seq = token_stream[:window_size]
            for _ in range(batch_size):
                sequences.append(base_seq)
        else:
            for _ in range(batch_size):
                start = self.rng.randint(0, max_start)
                seq = token_stream[start : start + window_size]
                sequences.append(seq)

        return {"sequences": sequences, "seq_len": eff_seq_len, "split": split_name}

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def get_vocab_info(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "sample_vocab": self.vocab[:50],
        }

    def tokens_info(self) -> Dict[str, Any]:
        """
        Extended token stream diagnostics.
        """
        return {
            "raw_token_count": self.raw_token_count,
            "train_token_count": self._train_len,
            "val_token_count": self._val_len,
            "val_ratio_effective": self._val_len / max(1, self.raw_token_count),
            "requested_seq_len": self.seq_len,
            "max_train_seq_possible": self._max_possible_seq_len(self._train_len),
            "max_val_seq_possible": self._max_possible_seq_len(self._val_len),
            "line_level_markers": self.line_level_markers,
            "deduplicate_eos": self.deduplicate_eos,
        }

    # ------------------------------------------------------------------ #
    # Properties for Special Token IDs
    # ------------------------------------------------------------------ #
    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    # ------------------------------------------------------------------ #
    # Optional Sequential Iterators (not used by main code paths)
    # ------------------------------------------------------------------ #
    def iter_train_stream(self) -> Iterable[List[int]]:
        """
        Yield non-overlapping sequential windows over train_tokens.
        Each yielded window size is (seq_len + 1) respecting clamping.
        """
        eff_seq_len = self._clamp_seq_len_for_stream(self.seq_len, self._train_len)
        if eff_seq_len <= 0:
            return
        step = eff_seq_len + 1
        for start in range(0, self._train_len - step + 1, step):
            yield self.train_tokens[start : start + step]

    def iter_val_stream(self) -> Iterable[List[int]]:
        """
        Yield non-overlapping sequential windows over val_tokens.
        """
        eff_seq_len = self._clamp_seq_len_for_stream(self.seq_len, self._val_len)
        if eff_seq_len <= 0:
            return
        step = eff_seq_len + 1
        for start in range(0, self._val_len - step + 1, step):
            yield self.val_tokens[start : start + step]
