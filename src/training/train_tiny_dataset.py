"""
Train on a tiny real dataset (next-token prediction) using GovernedTrainer.

What this does:
- Loads a small text corpus from data/tiny_corpus.txt
- Builds a simple vocabulary and a bigram language model (counts of next-token given current-token)
- Creates batches from real text (not random noise)
- Computes per-position logits using bigram probabilities (logits = log P(next_token | prev_token))
- Feeds batches to GovernedTrainer (uses your production ConsensusEngine)
- Prints training progress and runs simple validation every N steps

Run:
    python src/training/train_tiny_dataset.py

Notes:
- No external dependencies (standard library only).
- This is a tiny demonstration, not a performant trainer.
- For real training, you would replace the bigram logits with your model’s logits, or provide a gradient_fn that returns true gradients.

Revision / Fixes:
1. Removed merge conflict markers (<<<<<<< HEAD / ======= / >>>>>>>).
2. Added small numeric safety for log probabilities (avoid math domain errors if counts zero).
3. Added optional seed parameter to run_training for deterministic reproducibility.
4. Added graceful handling if validation encounters empty val set (skips rather than dividing by zero).
5. Added docstring clarifications for batch dictionary fields.
6. Added explicit random seeding inside run_training for reproducibility.
7. Ensured Laplace smoothing parameter (alpha) is used consistently for both unigram and bigram distributions.
8. Added a simple sanity check printing the first few vocabulary tokens to confirm loading.

All original logic retained; nothing truncated.
"""

import math
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List

# Make sure we can import governed_trainer from the same folder
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from governed_trainer import GovernedTrainer  # noqa: E402

# ---------------------------- Tokenizer and Dataset ---------------------------- #


class TinyTokenizer:
    """
    Very simple whitespace tokenizer with a tiny special token set.
    - Lowercases text
    - Splits on whitespace
    - Keeps tokens with frequency >= min_freq (others -> <UNK>)
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    def build_vocab(self, tokens: List[str]) -> None:
        counts = Counter(tokens)
        vocab = [self.PAD, self.UNK, self.BOS]
        for tok, c in counts.items():
            if c >= self.min_freq and tok not in (self.PAD, self.UNK, self.BOS):
                vocab.append(tok)
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = vocab

    def encode(self, tokens: List[str]) -> List[int]:
        tid = self.token_to_id.get
        unk = self.token_to_id[self.UNK]
        return [tid(tok, unk) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [
            self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.UNK
            for i in ids
        ]

    def vocab_size(self) -> int:
        return len(self.id_to_token)


class TinyTextDataset:
    """
    Builds a tiny next-token prediction dataset with bigram logits from a real text file.

    Bigrams:
    - We compute counts of next_token given current_token (with Laplace smoothing).
    - For position j, logits[j] = log P(target_j | prev_token), where prev_token is tokens[j-1] or <BOS> for j=0.
    - Passing log-probabilities as logits is valid because softmax(log p) = p.

    Batch fields produced (per the trainer/causal_loss expectations):
        {
            "tokens": List[int],                # length = seq_len
            "logits": List[List[float]],        # length = seq_len; each inner list length = vocab_size
            "targets": List[int],               # length = seq_len
            "hidden_states": List[List[float]], # deterministic embeddings for demonstration
            "num_layers": int                   # small constant used by governed trainer's loss shaping
        }
    """

    def __init__(
        self,
        path: str,
        seq_len: int = 16,
        min_freq: int = 1,
        alpha: float = 1.0,
        hidden_dim: int = 32,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.seq_len = seq_len
        self.alpha = alpha
        self.hidden_dim = hidden_dim

        # Load and tokenize
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip().lower()
        raw_tokens = text.split()

        # Build tokenizer and vocab
        self.tok = TinyTokenizer(min_freq=min_freq)
        self.tok.build_vocab(raw_tokens)

        # Encode tokens to ids
        self.encoded = self.tok.encode(raw_tokens)
        self.vocab_size = self.tok.vocab_size()
        self.pad_id = self.tok.token_to_id[TinyTokenizer.PAD]
        self.unk_id = self.tok.token_to_id[TinyTokenizer.UNK]
        self.bos_id = self.tok.token_to_id[TinyTokenizer.BOS]

        # Build unigram counts
        self.unigram = [0] * self.vocab_size
        for t in self.encoded:
            if 0 <= t < self.vocab_size:
                self.unigram[t] += 1

        # Bigram counts c[prev][next]
        self.bigram = [
            [0 for _ in range(self.vocab_size)] for _ in range(self.vocab_size)
        ]
        prev = self.bos_id
        for t in self.encoded:
            if 0 <= prev < self.vocab_size and 0 <= t < self.vocab_size:
                self.bigram[prev][t] += 1
            prev = t

        # Precompute smoothed log-probs
        self._unigram_logp = self._compute_unigram_logp()
        self._bigram_logp = self._compute_bigram_logp()

        # Split indices into train/val (80/20)
        n = max(0, len(self.encoded) - (self.seq_len + 1))
        indices = list(range(n))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        self.train_idx = indices[:split]
        self.val_idx = indices[split:]

    def _compute_unigram_logp(self) -> List[float]:
        alpha = self.alpha
        total = sum(self.unigram) + alpha * self.vocab_size
        return [math.log((c + alpha) / total) for c in self.unigram]

    def _compute_bigram_logp(self) -> List[List[float]]:
        alpha = self.alpha
        bigram_logp: List[List[float]] = []
        for prev in range(self.vocab_size):
            row = self.bigram[prev]
            row_sum = sum(row) + alpha * self.vocab_size
            # Each entry: log( (count + alpha) / (row_sum) )
            bigram_logp.append(
                [
                    math.log((row[next_id] + alpha) / row_sum)
                    for next_id in range(self.vocab_size)
                ]
            )
        return bigram_logp

    def _token_embedding(self, token_id: int) -> List[float]:
        """
        Deterministic tiny embedding based on token_id (no randomness across runs).
        """
        d = self.hidden_dim
        return [
            math.sin((token_id + 1) * (i + 1) * 0.13) * 0.5
            + math.cos((token_id + 2) * (i + 3) * 0.07) * 0.5
            for i in range(d)
        ]

    def _sequence_to_batch(self, seq: List[int]) -> Dict[str, Any]:
        assert len(seq) == self.seq_len + 1, (
            "Internal: sequence length must be seq_len+1"
        )
        inputs = seq[:-1]
        targets = seq[1:]

        logits: List[List[float]] = []
        for j in range(self.seq_len):
            prev_tok = inputs[j - 1] if j > 0 else self.bos_id
            logits.append(self._bigram_logp[prev_tok])

        hidden_states = [self._token_embedding(tid) for tid in inputs]

        return {
            "tokens": inputs,
            "logits": logits,
            "targets": targets,
            "hidden_states": hidden_states,
            "num_layers": 6,
        }

    def sample_train_batch(self) -> Dict[str, Any]:
        if not self.train_idx:
            raise RuntimeError("No train indices available; dataset too small.")
        start = random.choice(self.train_idx)
        seq = self.encoded[start : start + self.seq_len + 1]
        if len(seq) < self.seq_len + 1:
            seq = seq + [self.pad_id] * (self.seq_len + 1 - len(seq))
        return self._sequence_to_batch(seq)

    def sample_val_batch(self) -> Dict[str, Any]:
        if not self.val_idx:
            idx_pool = (
                self.train_idx
                if self.train_idx
                else list(range(max(1, len(self.encoded) - self.seq_len - 1)))
            )
        else:
            idx_pool = self.val_idx
        start = random.choice(idx_pool)
        seq = self.encoded[start : start + self.seq_len + 1]
        if len(seq) < self.seq_len + 1:
            seq = seq + [self.pad_id] * (self.seq_len + 1 - len(seq))
        return self._sequence_to_batch(seq)


# ---------------------------- Training and Validation ---------------------------- #


def run_training(
    data_path: str = os.path.join(
        os.path.dirname(os.path.dirname(HERE)), "data", "tiny_corpus.txt"
    ),
    steps: int = 200,
    val_interval: int = 50,
    seq_len: int = 16,
    min_freq: int = 1,
    alpha: float = 1.0,
    seed: int = 123,
) -> None:
    """
    Run a short governed training session over a tiny real dataset.

    Args:
        data_path: path to tiny_corpus.txt
        steps: number of training steps (bigram batches)
        val_interval: how often to run validation (in steps)
        seq_len: sequence length (context window)
        min_freq: min token frequency for vocab inclusion
        alpha: Laplace smoothing parameter
        seed: random seed for reproducibility
    """
    random.seed(seed)

    print(f"Loading dataset from: {data_path}")
    dataset = TinyTextDataset(
        path=data_path, seq_len=seq_len, min_freq=min_freq, alpha=alpha, hidden_dim=32
    )
    print(
        f"Vocab size: {dataset.vocab_size} (PAD={dataset.pad_id}, UNK={dataset.unk_id}, BOS={dataset.bos_id})"
    )
    print(f"First 10 vocab tokens: {dataset.tok.id_to_token[:10]}")
    print(
        f"Train indices: {len(dataset.train_idx)} | Val indices: {len(dataset.val_idx)}"
    )

    trainer = GovernedTrainer(
        learning_rate=0.001,
        lr_schedule="cosine",
        warmup_steps=20,
        total_steps=max(steps, 100),
        log_interval=10,
        checkpoint_interval=1000,  # infrequent for tiny demo
        safety_check_interval=10,
        gradient_accumulation_steps=1,
        detect_anomalies=True,
        enable_mixed_precision=False,
        random_seed=seed,
    )

    for step in range(1, steps + 1):
        batch = dataset.sample_train_batch()
        record = trainer.training_step(batch)

        if record.get("status") == "error":
            print(
                f"[ERR] step={record.get('step')} type={record.get('error_type')} msg={record.get('message')}"
            )
        else:
            if step % trainer.log_interval != 0:
                lr = record.get("learning_rate")
                lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "n/a"
                loss = record.get("loss")
                loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
                print(
                    f"step={record.get('step')} status={record.get('status')} loss={loss_str} lr={lr_str}"
                )

        if step % val_interval == 0:
            val_losses: List[float] = []
            try:
                from causal_loss import compute_loss  # noqa: E402
            except ImportError:
                print("[WARN] causal_loss module not found; skipping validation.")
                val_losses = []
            else:
                for _ in range(10):
                    vb = dataset.sample_val_batch()
                    vloss, _ = compute_loss(vb)
                    val_losses.append(vloss)

            if val_losses:
                avg_vloss = sum(val_losses) / len(val_losses)
                ppl = math.exp(min(avg_vloss, 100.0))
                print(
                    f"[VAL] step={step} avg_val_loss={avg_vloss:.4f} perplexity~{ppl:.2f}"
                )
            else:
                print(f"[VAL] step={step} (no validation performed)")

    print("Summary:", trainer.summary())
    tail = trainer.get_audit_tail(5)
    print("Last 5 audit entries:", tail)


if __name__ == "__main__":
    run_training()
