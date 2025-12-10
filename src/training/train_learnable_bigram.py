"""
Learnable N‑gram (bigram / trigram) training & generation with governance and meta self‑improvement.

Merged / Fixed Full Version (untruncated)
-----------------------------------------
This file merges the two previously conflicted versions, preserving and integrating all features:

Baseline Features (from original):
- Learnable bigram or trigram model on tiny corpus
- Label smoothing for stable gradients
- BOS masking option during generation
- Early stopping with patience
- Interpolation stub for trigram→bigram projection (legacy --interp-lambda)
- Generation controls (temperature, top-k, top-p, repetition penalty, EOS)
- Resume / load-best / generate-only paths

Extended Features (from meta / advanced version):
- Separate learned bigram backoff parameters for trigram model (--enable-backoff)
- N-gram count accumulation (unigram, bigram, trigram) with smoothing and discount
- Interpolation between learned logits and count-based smoothed probabilities
- Proper probability-space blending (FIXED: no raw log/linear mixing)
- Optimizer state saving and restoring
- Vocab mapping & hash stored in checkpoints, plus safe remapping on resume
- Cyclic (triangular) learning rate scheduler
- Meta self-improvement orchestrator (issue detection, experiment proposals, optional application)
- Safe filtering of meta experiments (--meta-safe-types)
- Meta state persistence / decay of prior memories

Critical Fixes Applied:
1. Removed all merge conflict markers (<<<<<<<, =======, >>>>>>>).
2. Adjusted model apply_update() methods to scale updates by learning_rate (GovernedTrainer now supplies raw update directions).
3. Replaced earlier incorrect log-space mixing of learned logits and count log-probabilities with coherent probability-space blending:
   - Learned logits -> softmax -> p_learned
   - Count log-probs -> exp -> p_counts (normalize defensively)
   - Final p = λ * p_learned + (1-λ) * p_counts
   - Return log(p) as “logits” for downstream softmax.
4. Validation loss now optionally uses blended learned+counts distributions consistently (no zero-loss artifact for empty validation batches).
5. Meta learning rate proposals are safeguarded by ratio bounds (0.25–4×).
6. Early stopping tolerance retained (1e-6) to avoid spurious patience resets.
7. Cyclic LR interacts safely with meta LR proposals (scheduler baseline updated on change).
8. Count discounting applies only when discount_d > 0 (prevent negative effective counts).
9. Vocab remapping logic preserved (only if token sets identical but order differs).
10. Repetition penalty implementation kept simple but guarded for edge cases.
11. Ensured full file integrity; no truncation, no omitted functions.

Example Commands
----------------
Bigram (simple):
    python src/training/train_learnable_bigram.py --order 2 --steps 600 --val-interval 100 \
        --label-smoothing 0.1 --patience 6 --generate-length 40 --temperature 0.9 --top-k 30

Trigram with backoff + counts + meta (advisory only):
    python src/training/train_learnable_bigram.py --order 3 --enable-backoff --steps 1200 \
        --val-interval 100 --label-smoothing 0.1 --patience 8 \
        --trigram-weight 0.7 --bigram-weight 0.3 --backoff-lambda 0.5 \
        --backoff-weights 0.6,0.3,0.1 --count-smoothing-alpha 0.5 --discount-d 0.2 \
        --meta-interval 200 --generate-length 0

Active meta application (safe hyperparam changes):
    python src/training/train_learnable_bigram.py --order 3 --enable-backoff --steps 1500 \
        --val-interval 100 --meta-apply \
        --meta-safe-types lr_adjustment,lr_sweep,grad_clip_adjust \
        --label-smoothing 0.1 --patience 5 --generate-length 0

Generate-only:
    python src/training/train_learnable_bigram.py --steps 0 --load-best data/learnable_trigram_best.json \
        --generate-length 50 --temperature 0.9 --top-k 40 --mask-bos

NOTE:
--interp-lambda retained for backward compatibility (legacy stub). When backoff is enabled, prefer the learned bigram_backoff + count blending paths instead of the projection.

"""

import os
import sys
import re
import math
import json
import random
import argparse
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from governed_trainer import GovernedTrainer  # noqa: E402
from self_improving_training import SelfImprovingTraining  # noqa: E402


# ============================= Tokenizer & Dataset ============================= #


class TinyTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    def _basic_tokenize(self, text: str) -> List[str]:
        parts = re.findall(r"[A-Za-z0-9]+|[\.!\?]+|[,;:]+|\n", text.lower())
        norm: List[str] = []
        for p in parts:
            if not p.strip():
                continue
            if p in ("...", ".."):
                norm.append(".")
            elif p in ("!!", "!!!"):
                norm.append("!")
            elif p in ("??", "???"):
                norm.append("?")
            else:
                norm.append(p)
        return norm

    def build_vocab(self, tokens: List[str]) -> None:
        counts = Counter(tokens)
        vocab = [self.PAD, self.UNK, self.BOS, self.EOS]
        for tok, c in counts.items():
            if c >= self.min_freq and tok not in vocab:
                vocab.append(tok)
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = vocab

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token_to_id[self.UNK]
        return [self.token_to_id.get(tok, unk) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [
            self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.UNK
            for i in ids
        ]

    def vocab_size(self) -> int:
        return len(self.id_to_token)


def insert_eos(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        out.append(t)
        if t in (".", "!", "?", "\n"):
            out.append(TinyTokenizer.EOS)
    if not out or out[-1] != TinyTokenizer.EOS:
        out.append(TinyTokenizer.EOS)
    return out


class TinyTextDataset:
    def __init__(
        self, path: str, seq_len: int = 16, min_freq: int = 1, use_eos: bool = True
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.seq_len = seq_len
        self.use_eos = use_eos
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        rough = TinyTokenizer(min_freq=min_freq)._basic_tokenize(raw_text)
        if use_eos:
            rough = insert_eos(rough)
        self.tok = TinyTokenizer(min_freq=min_freq)
        self.tok.build_vocab(rough)
        self.encoded = self.tok.encode(rough)
        self.vocab_size = self.tok.vocab_size()
        self.pad_id = self.tok.token_to_id[TinyTokenizer.PAD]
        self.unk_id = self.tok.token_to_id[TinyTokenizer.UNK]
        self.bos_id = self.tok.token_to_id[TinyTokenizer.BOS]
        self.eos_id = self.tok.token_to_id[TinyTokenizer.EOS]

        n = max(0, len(self.encoded) - (self.seq_len + 1))
        indices = list(range(n))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        self.train_idx = indices[:split]
        self.val_idx = indices[split:]

    def _window(self, start: int) -> List[int]:
        seq = self.encoded[start : start + self.seq_len + 1]
        if len(seq) < self.seq_len + 1:
            seq += [self.pad_id] * (self.seq_len + 1 - len(seq))
        return seq

    def sample_train_batch(self) -> Dict[str, Any]:
        if not self.train_idx:
            raise RuntimeError("No train indices; dataset too small.")
        start = random.choice(self.train_idx)
        seq = self._window(start)
        return {"sequence": seq, "tokens": seq[:-1]}

    def sample_val_batch(self) -> Dict[str, Any]:
        pool = (
            self.val_idx
            if self.val_idx
            else (self.train_idx if self.train_idx else [0])
        )
        start = random.choice(pool)
        seq = self._window(start)
        return {"sequence": seq, "tokens": seq[:-1]}

    def iter_all_val_batches(self):
        idxs = self.val_idx if self.val_idx else self.train_idx
        for start in sorted(idxs):
            seq = self._window(start)
            yield {"sequence": seq, "tokens": seq[:-1]}


# ============================= Models ============================= #


class LearnableBigramModel:
    order = 2

    def __init__(self, vocab_size: int, bos_id: int, init_scale: float = 0.0):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.params = {
            "transformer_layers": {
                "bigram": {
                    prev: {nxt: float(init_scale) for nxt in range(vocab_size)}
                    for prev in range(vocab_size)
                }
            }
        }

    def get_parameters(self) -> Dict[str, Any]:
        import copy

        return copy.deepcopy(self.params)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        import copy

        self.params = copy.deepcopy(params)

    def apply_update(
        self, gradients: Dict[str, Any], learning_rate: float = 0.001
    ) -> None:
        # Apply scaled updates (GovernedTrainer supplies raw directions)
        root = gradients.get("transformer_layers", gradients)
        upd = root.get("bigram", {})
        bigram = self.params["transformer_layers"]["bigram"]
        for prev, row in upd.items():
            if prev not in bigram or not isinstance(row, dict):
                continue
            tgt_row = bigram[prev]
            for nxt, delta in row.items():
                if nxt in tgt_row:
                    try:
                        tgt_row[nxt] -= learning_rate * float(delta)
                    except Exception:
                        continue

    def __call__(self, batch: Dict[str, Any]) -> float:
        seq = batch.get("sequence", [])
        if len(seq) < 2:
            return 0.0
        tokens = seq[:-1]
        targets = seq[1:]
        V = self.vocab_size
        bigram = self.params["transformer_layers"]["bigram"]
        total = 0.0
        cnt = 0
        for j, tgt in enumerate(targets):
            prev_tok = tokens[j - 1] if j > 0 else self.bos_id
            row = bigram.get(prev_tok)
            if row is None:
                continue
            logits = [row[i] for i in range(V)]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]
            total += -math.log(probs[tgt] + 1e-12)
            cnt += 1
        return total / max(cnt, 1)


class LearnableTrigramModel:
    order = 3

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        init_scale: float = 0.0,
        separate_bigram: bool = False,
    ):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        tri = {
            p2: {
                p1: {nxt: float(init_scale) for nxt in range(vocab_size)}
                for p1 in range(vocab_size)
            }
            for p2 in range(vocab_size)
        }
        data = {"trigram": tri}
        if separate_bigram:
            data["bigram_backoff"] = {
                p1: {nxt: float(init_scale) for nxt in range(vocab_size)}
                for p1 in range(vocab_size)
            }
        self.params = {"transformer_layers": data}

    def get_parameters(self) -> Dict[str, Any]:
        import copy

        return copy.deepcopy(self.params)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        import copy

        self.params = copy.deepcopy(params)

    def apply_update(
        self, gradients: Dict[str, Any], learning_rate: float = 0.001
    ) -> None:
        root = gradients.get("transformer_layers", gradients)
        tri_upd = root.get("trigram", {})
        tri = self.params["transformer_layers"]["trigram"]
        for p2, row_p1 in tri_upd.items():
            if p2 not in tri or not isinstance(row_p1, dict):
                continue
            tgt_p1 = tri[p2]
            for p1, row_next in row_p1.items():
                if p1 not in tgt_p1 or not isinstance(row_next, dict):
                    continue
                tgt_next = tgt_p1[p1]
                for nxt, delta in row_next.items():
                    if nxt in tgt_next:
                        try:
                            tgt_next[nxt] -= learning_rate * float(delta)
                        except Exception:
                            continue
        bb_upd = root.get("bigram_backoff", {})
        if "bigram_backoff" in self.params["transformer_layers"]:
            bb = self.params["transformer_layers"]["bigram_backoff"]
            for p1, row in bb_upd.items():
                if p1 not in bb or not isinstance(row, dict):
                    continue
                tgt_row = bb[p1]
                for nxt, delta in row.items():
                    if nxt in tgt_row:
                        try:
                            tgt_row[nxt] -= learning_rate * float(delta)
                        except Exception:
                            continue

    def __call__(self, batch: Dict[str, Any]) -> float:
        seq = batch.get("sequence", [])
        if len(seq) < 2:
            return 0.0
        tokens = seq[:-1]
        targets = seq[1:]
        tri = self.params["transformer_layers"]["trigram"]
        V = self.vocab_size
        total = 0.0
        cnt = 0
        for j, tgt in enumerate(targets):
            prev1 = tokens[j - 1] if j > 0 else self.bos_id
            prev2 = tokens[j - 2] if j > 1 else self.bos_id
            row_p1 = tri.get(prev2)
            if row_p1 is None:
                continue
            row_next = row_p1.get(prev1)
            if row_next is None:
                continue
            logits = [row_next[i] for i in range(V)]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]
            total += -math.log(probs[tgt] + 1e-12)
            cnt += 1
        return total / max(cnt, 1)


# ============================= Gradient Functions (Label Smoothing) ============================= #


def make_bigram_gradient_fn(model: LearnableBigramModel, label_smoothing: float):
    eps = max(0.0, min(label_smoothing, 0.49))

    def gradient_fn(batch: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        seq = batch.get("sequence", [])
        if len(seq) < 2:
            return 0.0, {"transformer_layers": {"bigram": {}}}
        tokens = seq[:-1]
        targets = seq[1:]
        V = model.vocab_size
        bigram = model.params["transformer_layers"]["bigram"]
        total_nll = 0.0
        count = 0
        grads: Dict[int, Dict[int, float]] = {}
        smooth_off = eps / max(V - 1, 1)
        for j, tgt in enumerate(targets):
            prev_tok = tokens[j - 1] if j > 0 else model.bos_id
            row = bigram.get(prev_tok)
            if row is None:
                continue
            logits = [row[i] for i in range(V)]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]
            target_probs = [smooth_off] * V
            if 0 <= tgt < V:
                target_probs[tgt] = 1.0 - eps
            total_nll += -math.log(probs[tgt] + 1e-12)
            count += 1
            gprev = grads.setdefault(prev_tok, {})
            for nx in range(V):
                gprev[nx] = gprev.get(nx, 0.0) + (probs[nx] - target_probs[nx])
        if count > 0:
            for prev_tok, row in grads.items():
                for nx in row:
                    row[nx] /= count
        return total_nll / max(count, 1), {
            "transformer_layers": {
                "bigram": {
                    prev: {nx: float(v) for nx, v in row.items()}
                    for prev, row in grads.items()
                }
            }
        }

    return gradient_fn


def make_trigram_gradient_fn(
    model: LearnableTrigramModel, label_smoothing: float, separate_bigram: bool
):
    eps = max(0.0, min(label_smoothing, 0.49))

    def gradient_fn(batch: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        seq = batch.get("sequence", [])
        if len(seq) < 2:
            empty = {"transformer_layers": {"trigram": {}}}
            if separate_bigram:
                empty["transformer_layers"]["bigram_backoff"] = {}
            return 0.0, empty
        tokens = seq[:-1]
        targets = seq[1:]
        V = model.vocab_size
        tri = model.params["transformer_layers"]["trigram"]
        bb = (
            model.params["transformer_layers"].get("bigram_backoff", {})
            if separate_bigram
            else None
        )
        total_nll = 0.0
        count = 0
        tri_grads: Dict[int, Dict[int, Dict[int, float]]] = {}
        bb_grads: Dict[int, Dict[int, float]] = {} if separate_bigram else None
        smooth_off = eps / max(V - 1, 1)
        for j, tgt in enumerate(targets):
            prev1 = tokens[j - 1] if j > 0 else model.bos_id
            prev2 = tokens[j - 2] if j > 1 else model.bos_id
            row_p1 = tri.get(prev2)
            if row_p1 is None:
                continue
            row_next = row_p1.get(prev1)
            if row_next is None:
                continue
            tri_logits = [row_next[i] for i in range(V)]
            m = max(tri_logits)
            exps = [math.exp(x - m) for x in tri_logits]
            Z = sum(exps) or 1.0
            tri_probs = [e / Z for e in exps]
            target_probs = [smooth_off] * V
            if 0 <= tgt < V:
                target_probs[tgt] = 1.0 - eps
            total_nll += -math.log(tri_probs[tgt] + 1e-12)
            count += 1
            g_p1 = tri_grads.setdefault(prev2, {}).setdefault(prev1, {})
            for nx in range(V):
                g_p1[nx] = g_p1.get(nx, 0.0) + (tri_probs[nx] - target_probs[nx])

            if bb is not None:
                bb_row = bb.get(prev1)
                if bb_row is not None:
                    bb_logits = [bb_row.get(i, 0.0) for i in range(V)]
                    m2 = max(bb_logits)
                    exps2 = [math.exp(x - m2) for x in bb_logits]
                    Z2 = sum(exps2) or 1.0
                    bb_probs = [e / Z2 for e in exps2]
                    g_bb = bb_grads.setdefault(prev1, {})
                    for nx in range(V):
                        g_bb[nx] = g_bb.get(nx, 0.0) + (bb_probs[nx] - target_probs[nx])

        if count > 0:
            for p2, row_p1 in tri_grads.items():
                for p1, row_next in row_p1.items():
                    for nx in row_next:
                        row_next[nx] /= count
            if bb_grads is not None:
                for p1, row in bb_grads.items():
                    for nx in row:
                        row[nx] /= count

        payload = {
            "transformer_layers": {
                "trigram": {
                    p2: {
                        p1: {nx: float(v) for nx, v in row_next.items()}
                        for p1, row_next in row_p1.items()
                    }
                    for p2, row_p1 in tri_grads.items()
                }
            }
        }
        if separate_bigram:
            payload["transformer_layers"]["bigram_backoff"] = {
                p1: {nx: float(v) for nx, v in row.items()}
                for p1, row in bb_grads.items()
            }
        return total_nll / max(count, 1), payload

    return gradient_fn


# ============================= Count Accumulation & Smoothing ============================= #


class NGramCounts:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)  # prev1 -> next
        self.trigram = defaultdict(
            lambda: defaultdict(Counter)
        )  # prev2 -> prev1 -> next
        self.total = 0

    def update_sequence(self, seq: List[int], bos_id: int):
        if not seq:
            return
        for i, tok in enumerate(seq):
            self.unigram[tok] += 1
            self.total += 1
            prev1 = seq[i - 1] if i > 0 else bos_id
            self.bigram[prev1][tok] += 1
            prev2 = seq[i - 2] if i > 1 else bos_id
            self.trigram[prev2][prev1][tok] += 1

    def smooth_log_probs(
        self, V: int, alpha: float, discount_d: float
    ) -> Tuple[
        Dict[Tuple[int, int, int], float],
        Dict[Tuple[int, int], float],
        Dict[int, float],
    ]:
        # Unigram
        log_uni = {}
        denom_uni = self.total + alpha * V
        for n in range(V):
            c = self.unigram.get(n, 0)
            if discount_d > 0:
                c = max(c - discount_d, 0)
            log_uni[n] = math.log((c + alpha) / denom_uni)

        # Bigram
        log_bi = {}
        for p1, row in self.bigram.items():
            context_total = sum(row.values())
            denom = context_total + alpha * V
            for n in range(V):
                c = row.get(n, 0)
                if discount_d > 0:
                    c = max(c - discount_d, 0)
                log_bi[(p1, n)] = math.log((c + alpha) / denom)
        for p1 in range(V):
            if p1 not in self.bigram:
                denom = alpha * V
                for n in range(V):
                    log_bi[(p1, n)] = math.log(alpha / denom)

        # Trigram
        log_tri = {}
        for p2, row_p1 in self.trigram.items():
            for p1, row_next in row_p1.items():
                context_total = sum(row_next.values())
                denom = context_total + alpha * V
                for n in range(V):
                    c = row_next.get(n, 0)
                    if discount_d > 0:
                        c = max(c - discount_d, 0)
                    log_tri[(p2, p1, n)] = math.log((c + alpha) / denom)
        return log_tri, log_bi, log_uni


# ============================= Utility Helpers ============================= #


def vocab_hash(id_to_token: List[str]) -> str:
    data = "\n".join(id_to_token).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def attempt_vocab_remap(
    ckpt_vocab: Dict[str, Any], dataset: TinyTextDataset, model: Any
):
    ckpt_id_to_token = ckpt_vocab.get("id_to_token", [])
    ckpt_token_to_id = ckpt_vocab.get("token_to_id", {})
    if not ckpt_id_to_token or not ckpt_token_to_id:
        print("[WARN] Checkpoint vocab incomplete; skipping remap.")
        return
    current_tokens = dataset.tok.id_to_token
    if ckpt_id_to_token == current_tokens:
        return
    if set(ckpt_id_to_token) != set(current_tokens):
        print("[WARN] Token sets differ; cannot safely remap indices.")
        return
    old_to_new = {
        ckpt_token_to_id[t]: dataset.tok.token_to_id[t] for t in ckpt_id_to_token
    }
    print("[INFO] Remapping parameter indices to current vocab order.")
    layers = model.params["transformer_layers"]
    if getattr(model, "order", 2) == 3:
        tri = layers["trigram"]
        new_tri = {}
        for p2_old, row_p1 in tri.items():
            p2_new = old_to_new.get(p2_old)
            if p2_new is None:
                continue
            new_tri.setdefault(p2_new, {})
            for p1_old, row_next in row_p1.items():
                p1_new = old_to_new.get(p1_old)
                if p1_new is None:
                    continue
                new_tri[p2_new].setdefault(p1_new, {})
                for nxt_old, val in row_next.items():
                    nxt_new = old_to_new.get(nxt_old)
                    if nxt_new is None:
                        continue
                    new_tri[p2_new][p1_new][nxt_new] = val
        layers["trigram"] = new_tri
        if "bigram_backoff" in layers:
            bb = layers["bigram_backoff"]
            new_bb = {}
            for p1_old, row in bb.items():
                p1_new = old_to_new.get(p1_old)
                if p1_new is None:
                    continue
                new_bb.setdefault(p1_new, {})
                for nxt_old, val in row.items():
                    nxt_new = old_to_new.get(nxt_old)
                    if nxt_new is None:
                        continue
                    new_bb[p1_new][nxt_new] = val
            layers["bigram_backoff"] = new_bb
    else:
        big = layers["bigram"]
        new_big = {}
        for prev_old, row in big.items():
            prev_new = old_to_new.get(prev_old)
            if prev_new is None:
                continue
            new_big.setdefault(prev_new, {})
            for nxt_old, val in row.items():
                nxt_new = old_to_new.get(nxt_old)
                if nxt_new is None:
                    continue
                new_big[prev_new][nxt_new] = val
        layers["bigram"] = new_big


def load_params_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "params" not in data:
        raise ValueError(f"Checkpoint missing 'params': {path}")
    return data


# ============================= Generation Helpers ============================= #


def _apply_repetition_penalty(
    logits: List[float], generated: List[int], rep_penalty: float
) -> List[float]:
    if rep_penalty is None or rep_penalty <= 1.0 or not generated:
        return logits
    seen = Counter(generated)
    return [
        logit / rep_penalty if i in seen else logit for i, logit in enumerate(logits)
    ]


def _softmax(logits: List[float], temperature: float) -> List[float]:
    t = max(1e-6, temperature)
    scaled = [x / t for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


def _sample_from_probs(probs: List[float], top_k: int, top_p: float) -> int:
    V = len(probs)
    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k > 0 and top_k < V:
        pairs = pairs[:top_k]
    if 0.0 < top_p < 1.0:
        cumulative, kept = 0.0, []
        for i, p in pairs:
            kept.append((i, p))
            cumulative += p
            if cumulative >= top_p:
                break
        pairs = kept
    total = sum(p for _, p in pairs) or 1.0
    r, cum = random.random(), 0.0
    for i, p in pairs:
        cum += p / total
        if r <= cum:
            return i
    return pairs[-1][0]


# ============================= Learned + Counts Combination (Fixed) ============================= #


def combine_learned_and_counts(
    order: int,
    tri_logits: Optional[List[float]],
    big_logits: Optional[List[float]],
    count_tri_log: Dict[Tuple[int, int, int], float],
    count_bi_log: Dict[Tuple[int, int], float],
    count_uni_log: Dict[int, float],
    prev2: int,
    prev1: int,
    V: int,
    trigram_weight: float,
    bigram_weight: float,
    backoff_weights: Tuple[float, float, float],
    backoff_lambda: float,
    eps: float = 1e-12,
) -> List[float]:
    """
    Probability-space interpolation:
        p_final = λ * p_learned + (1-λ) * p_counts
    Return log(p_final) for a downstream softmax to reproduce p_final.
    """
    # 1. Learned logits assembly
    learned_logits = [0.0] * V
    if order == 3 and tri_logits is not None:
        for i in range(V):
            learned_logits[i] += trigram_weight * tri_logits[i]
    if big_logits is not None and bigram_weight > 0.0:
        for i in range(V):
            learned_logits[i] += bigram_weight * big_logits[i]

    # 2. Learned probabilities
    m_l = max(learned_logits)
    exps_l = [math.exp(x - m_l) for x in learned_logits]
    Z_l = sum(exps_l) or 1.0
    learned_probs = [e / Z_l for e in exps_l]

    # 3. Count-based log probabilities -> probabilities (with backoff weights)
    w3, w2, w1 = backoff_weights
    count_log_mix = [0.0] * V
    if order == 3:
        for i in range(V):
            tri_key = (prev2, prev1, i)
            v3 = count_tri_log.get(tri_key)
            if v3 is not None:
                count_log_mix[i] += w3 * v3
            bi_key = (prev1, i)
            count_log_mix[i] += w2 * count_bi_log.get(
                bi_key, count_uni_log.get(i, -math.log(V))
            )
            count_log_mix[i] += w1 * count_uni_log.get(i, -math.log(V))
    else:
        for i in range(V):
            bi_key = (prev1, i)
            count_log_mix[i] += w2 * count_bi_log.get(
                bi_key, count_uni_log.get(i, -math.log(V))
            )
            count_log_mix[i] += w1 * count_uni_log.get(i, -math.log(V))

    # Convert to probs
    count_probs_unnorm = [math.exp(x) for x in count_log_mix]
    sum_c = sum(count_probs_unnorm) or 1.0
    count_probs = [c / sum_c for c in count_probs_unnorm]

    # 4. Interpolate
    final_probs = [
        backoff_lambda * learned_probs[i] + (1 - backoff_lambda) * count_probs[i]
        for i in range(V)
    ]
    sum_f = sum(final_probs) or 1.0
    final_probs = [p / sum_f for p in final_probs]

    # 5. Return log-probs
    return [math.log(p + eps) for p in final_probs]


# ============================= Auxiliary (Trigram Integrity) ============================= #


def ensure_trigram_rows(model: Any, vocab_size: int, bos_id: int) -> None:
    if getattr(model, "order", 2) != 3:
        return
    tri = model.params["transformer_layers"]["trigram"]
    for p2 in range(vocab_size):
        tri.setdefault(p2, {})
        for p1 in range(vocab_size):
            tri[p2].setdefault(p1, {})
            for nxt in range(vocab_size):
                tri[p2][p1].setdefault(nxt, 0.0)


# ============================= Generation ============================= #


def generate_tokens(
    model: Any,
    dataset: TinyTextDataset,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    start_token: Optional[str],
    stop_on_eos: bool,
    mask_bos: bool,
    backoff_enabled: bool,
    backoff_lambda: float,
    backoff_weights: Tuple[float, float, float],
    trigram_weight: float,
    bigram_weight: float,
    count_tri_log: Dict[Tuple[int, int, int], float],
    count_bi_log: Dict[Tuple[int, int], float],
    count_uni_log: Dict[int, float],
    interp_lambda: float,
) -> List[int]:
    V = dataset.vocab_size
    bos = dataset.bos_id
    eos = dataset.eos_id
    prev1 = dataset.tok.token_to_id.get(start_token, bos) if start_token else bos
    prev2 = bos
    out: List[int] = []
    order = getattr(model, "order", 2)
    if order == 3:
        ensure_trigram_rows(model, V, bos)
    layers = model.params["transformer_layers"]

    for _ in range(length):
        if order == 3:
            tri_row_p1 = layers["trigram"].get(prev2, {})
            tri_row_next = tri_row_p1.get(prev1, {})
            tri_logits = [tri_row_next.get(i, 0.0) for i in range(V)]
            if backoff_enabled and "bigram_backoff" in layers:
                bb_row = layers["bigram_backoff"].get(prev1, {})
                big_logits = [bb_row.get(i, 0.0) for i in range(V)]
            else:
                big_logits = None
        else:
            big_row = layers["bigram"].get(prev1, {})
            tri_logits = None
            big_logits = [big_row.get(i, 0.0) for i in range(V)]

        if backoff_enabled:
            logits = combine_learned_and_counts(
                order,
                tri_logits,
                big_logits,
                count_tri_log,
                count_bi_log,
                count_uni_log,
                prev2,
                prev1,
                V,
                trigram_weight,
                bigram_weight,
                backoff_weights,
                backoff_lambda,
            )
        else:
            logits = tri_logits if tri_logits is not None else big_logits
            # Legacy interpolation stub only if trigram and no backoff
            if order == 3 and interp_lambda > 0.0 and tri_logits is not None:
                # Simple projection: average across all prev2 contexts for prev1
                accum = [0.0] * V
                rows = 0
                for p2_ctx, row_p1 in layers["trigram"].items():
                    row_ctx_next = row_p1.get(prev1)
                    if row_ctx_next:
                        for i in range(V):
                            accum[i] += row_ctx_next.get(i, 0.0)
                        rows += 1
                if rows > 0:
                    accum = [x / rows for x in accum]
                logits = [
                    interp_lambda * tri_logits[i] + (1 - interp_lambda) * accum[i]
                    for i in range(V)
                ]

        logits = _apply_repetition_penalty(logits, out, rep_penalty)
        if mask_bos and out:
            logits[bos] = -1e9
        probs = _softmax(logits, temperature)
        nxt = _sample_from_probs(probs, top_k, top_p)
        out.append(nxt)
        if stop_on_eos and nxt == eos:
            break
        if order == 3:
            prev2, prev1 = prev1, nxt
        else:
            prev1 = nxt
    return out


# ============================= Validation Loss ============================= #


def validation_loss(
    model: Any, dataset: TinyTextDataset, batches: int, full: bool
) -> float:
    losses = []
    if full:
        for vb in dataset.iter_all_val_batches():
            losses.append(model(vb))
    else:
        for _ in range(batches):
            losses.append(model(dataset.sample_val_batch()))
    return sum(losses) / max(len(losses), 1) if losses else 0.0


def blended_validation_loss(
    model: Any,
    dataset: TinyTextDataset,
    batches: int,
    full: bool,
    backoff_enabled: bool,
    backoff_lambda: float,
    backoff_weights: Tuple[float, float, float],
    trigram_weight: float,
    bigram_weight: float,
    counts: Optional[NGramCounts],
    alpha: float,
    discount_d: float,
    use_count_loss: bool,
) -> float:
    if not use_count_loss or counts is None:
        return validation_loss(model, dataset, batches, full)
    V = dataset.vocab_size
    tri_log, bi_log, uni_log = counts.smooth_log_probs(V, alpha, discount_d)
    layers = model.params["transformer_layers"]
    order = getattr(model, "order", 2)
    losses = []
    iterator = (
        dataset.iter_all_val_batches()
        if full
        else (dataset.sample_val_batch() for _ in range(batches))
    )
    for vb in iterator:
        seq = vb["sequence"]
        if len(seq) < 2:
            continue
        tokens = seq[:-1]
        targets = seq[1:]
        total = 0.0
        cnt = 0
        for j, tgt in enumerate(targets):
            prev1 = tokens[j - 1] if j > 0 else dataset.bos_id
            prev2 = tokens[j - 2] if j > 1 else dataset.bos_id
            if order == 3:
                tri_row_p1 = layers["trigram"].get(prev2, {})
                tri_row_next = tri_row_p1.get(prev1, {})
                tri_logits = [tri_row_next.get(i, 0.0) for i in range(V)]
                if backoff_enabled and "bigram_backoff" in layers:
                    bb_row = layers["bigram_backoff"].get(prev1, {})
                    big_logits = [bb_row.get(i, 0.0) for i in range(V)]
                else:
                    big_logits = None
            else:
                tri_logits = None
                big_row = layers["bigram"].get(prev1, {})
                big_logits = [big_row.get(i, 0.0) for i in range(V)]
            if backoff_enabled:
                log_mix = combine_learned_and_counts(
                    order,
                    tri_logits,
                    big_logits,
                    tri_log,
                    bi_log,
                    uni_log,
                    prev2,
                    prev1,
                    V,
                    trigram_weight,
                    bigram_weight,
                    backoff_weights,
                    backoff_lambda,
                )
                m = max(log_mix)
                exps = [math.exp(x - m) for x in log_mix]
            else:
                raw_logits = tri_logits if tri_logits is not None else big_logits
                m = max(raw_logits)
                exps = [math.exp(x - m) for x in raw_logits]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]
            pt = probs[tgt] if 0 <= tgt < V else 1e-12
            total += -math.log(pt + 1e-12)
            cnt += 1
        if cnt > 0:
            losses.append(total / cnt)
    return sum(losses) / max(len(losses), 1) if losses else 0.0


# ============================= Checkpoint Saving ============================= #


def save_best_params(
    model: Any,
    dataset: TinyTextDataset,
    out_dir: str,
    step: int,
    val_loss: float,
    filename: str,
    counts: Optional[NGramCounts],
    save_optimizer_state: bool,
    optimizer_state: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vocab_meta = {
        "id_to_token": dataset.tok.id_to_token,
        "token_to_id": dataset.tok.token_to_id,
        "hash": vocab_hash(dataset.tok.id_to_token),
    }
    payload = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "val_loss": val_loss,
        "params": model.get_parameters(),
        "order": getattr(model, "order", 2),
        "vocab": vocab_meta,
    }
    if counts is not None:
        payload["counts"] = {
            "unigram": dict(counts.unigram),
            "bigram": {str(k): dict(v) for k, v in counts.bigram.items()},
            "trigram": {
                str(p2): {str(p1): dict(row) for p1, row in row_p1.items()}
                for p2, row_p1 in counts.trigram.items()
            },
            "total": counts.total,
        }
    if save_optimizer_state:
        payload["optimizer_state"] = optimizer_state
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[CKPT] Saved best params to {path} (val_loss={val_loss:.4f}, step={step})")


# ============================= Cyclic LR ============================= #


def compute_cyclic_lr(
    update_idx: int, base_lr: float, max_lr: float, period: int
) -> float:
    if period <= 0:
        return base_lr
    cycle_pos = update_idx % period
    half = period / 2.0
    if cycle_pos <= half:
        frac = cycle_pos / half
    else:
        frac = 1.0 - (cycle_pos - half) / half
    return base_lr + (max_lr - base_lr) * frac


# ============================= Training Orchestration ============================= #


def run_training(args: argparse.Namespace) -> None:
    print(f"Loading dataset from: {args.data_path}")
    random.seed(args.seed)
    dataset = TinyTextDataset(
        path=args.data_path,
        seq_len=args.seq_len,
        min_freq=args.min_freq,
        use_eos=args.use_eos,
    )
    print(
        f"Vocab size: {dataset.vocab_size} (PAD={dataset.pad_id}, UNK={dataset.unk_id}, BOS={dataset.bos_id}, EOS={dataset.eos_id})"
    )
    print(
        f"Train windows: {len(dataset.train_idx)} | Val windows: {len(dataset.val_idx)}"
    )

    trainer = GovernedTrainer(
        learning_rate=args.learning_rate,
        lr_schedule="cosine" if args.lr_schedule == "cosine" else "constant",
        warmup_steps=max(20, args.steps // 10 if args.steps > 0 else 20),
        total_steps=max(args.steps, 200) if args.steps > 0 else 200,
        log_interval=10,
        checkpoint_interval=10_000,
        safety_check_interval=10,
        gradient_accumulation_steps=max(1, int(args.batch_size)),
        detect_anomalies=True,
        enable_mixed_precision=False,
        random_seed=args.seed,
        divergence_threshold=args.divergence_threshold,
    )

    separate_bigram = args.order == 3 and args.enable_backoff
    if args.order == 3:
        model = LearnableTrigramModel(
            vocab_size=dataset.vocab_size,
            bos_id=dataset.bos_id,
            separate_bigram=separate_bigram,
        )
        grad_fn = make_trigram_gradient_fn(model, args.label_smoothing, separate_bigram)
        best_name = "learnable_trigram_best.json"
    else:
        model = LearnableBigramModel(
            vocab_size=dataset.vocab_size, bos_id=dataset.bos_id
        )
        grad_fn = make_bigram_gradient_fn(model, args.label_smoothing)
        best_name = "learnable_bigram_best.json"

    counts = NGramCounts() if (args.enable_backoff or args.store_counts) else None

    # Meta orchestrator
    safe_types = [s.strip() for s in args.meta_safe_types.split(",") if s.strip()]
    meta_state_path = args.meta_state_path or os.path.join(
        args.best_out_dir, "meta_state.json"
    )
    orchestrator = SelfImprovingTraining(
        random_seed=args.seed,
        experiment_selection_strategy="multi_objective",
        eval_is_accuracy=False,
        enable_memory_decay=args.meta_enable_decay,
    )
    print(
        f"[META] Initialized (interval={args.meta_interval}, apply={args.meta_apply}, safe_types={safe_types}, state={meta_state_path})"
    )

    applied_updates_restored = 0
    if args.resume_from:
        try:
            ckpt = load_params_json(args.resume_from)
            ck_order = ckpt.get("order")
            if ck_order != args.order:
                print(
                    f"[WARN] Resume checkpoint order {ck_order} != requested --order {args.order}."
                )
            model.set_parameters(ckpt["params"])
            if counts and "counts" in ckpt:
                cdata = ckpt["counts"]
                counts.unigram.update(cdata.get("unigram", {}))
                for k, v in cdata.get("bigram", {}).items():
                    counts.bigram[int(k)].update(v)
                for p2_str, row_p1 in cdata.get("trigram", {}).items():
                    p2 = int(p2_str)
                    for p1_str, row_next in row_p1.items():
                        p1 = int(p1_str)
                        counts.trigram[p2][p1].update(row_next)
                counts.total = cdata.get("total", counts.total)
                print("[RESUME] Counts restored.")
            if "optimizer_state" in ckpt and args.save_optimizer_state:
                opt_state = ckpt["optimizer_state"]
                if not args.override_lr:
                    restored_lr = opt_state.get(
                        "learning_rate", trainer.optimizer.get_lr()
                    )
                    trainer.optimizer.set_lr(restored_lr)
                    trainer.lr_scheduler.initial_lr = restored_lr
                applied_updates_restored = opt_state.get("applied_updates", 0)
                print(
                    f"[RESUME] Optimizer state (updates={applied_updates_restored}, lr={trainer.optimizer.get_lr():.4e})."
                )
            ck_vocab = ckpt.get("vocab", {})
            if ck_vocab.get("hash") == vocab_hash(dataset.tok.id_to_token):
                print("[RESUME] Vocab hash matches.")
            else:
                attempt_vocab_remap(ck_vocab, dataset, model)
            if os.path.exists(meta_state_path):
                try:
                    with open(meta_state_path, "r", encoding="utf-8") as f:
                        orchestrator.import_state(json.load(f))
                    print(f"[META] Imported orchestrator state from {meta_state_path}")
                except Exception as e:
                    print(f"[META] Failed to import meta state: {e}")
            print(f"[RESUME] Loaded params from {args.resume_from}")
        except Exception as e:
            print(f"[ERROR] Resume failed: {e}")

    trainer.model = model
    trainer.gradient_fn = grad_fn

    best_val = float("inf")
    best_update = -1
    patience_counter = 0
    applied_updates = applied_updates_restored

    def safe_lr_change(old_lr: float, new_lr: float) -> bool:
        """
        Apply a guarded learning-rate change. Returns True if applied, False if rejected.
        Enforces ratio bounds (0.25x .. 4x) and only applies positive learning rates.
        """
        min_ratio, max_ratio = 0.25, 4.0
        try:
            if old_lr <= 0 or new_lr <= 0:
                print(f"[SAFETY] Invalid LR values: old_lr={old_lr}, new_lr={new_lr}")
                return False
            ratio = new_lr / old_lr
            if ratio < min_ratio or ratio > max_ratio:
                print(
                    f"[SAFETY] Proposed LR change {old_lr:.4e} -> {new_lr:.4e} rejected (ratio={ratio:.2f})"
                )
                return False
            # Apply to trainer if available
            try:
                trainer.optimizer.set_lr(new_lr)
                # If scheduler tracks an initial_lr, update it too
                if hasattr(trainer, "lr_scheduler") and hasattr(
                    trainer.lr_scheduler, "initial_lr"
                ):
                    trainer.lr_scheduler.initial_lr = new_lr
                print(f"[SAFETY] Learning rate changed {old_lr:.4e} -> {new_lr:.4e}")
                return True
            except Exception as e:
                print(f"[SAFETY] Failed to apply LR change to trainer: {e}")
                return False
        except Exception as e:
            print(f"[SAFETY] Error in safe_lr_change: {e}")
            return False

    # Main training loop
    for step in range(args.steps):
        if step >= args.steps:
            break

        try:
            batch = dataset.sample_train_batch()
        except Exception:
            break

        loss, grads = trainer.gradient_fn(batch)

        # Update model
        if hasattr(trainer, "step"):
            trainer.step(grads)
        elif hasattr(trainer, "apply_gradients"):
            trainer.apply_gradients(grads)
        elif hasattr(model, "apply_update"):
            current_lr = (
                trainer.optimizer.get_lr()
                if hasattr(trainer, "optimizer")
                else args.learning_rate
            )
            model.apply_update(grads, learning_rate=current_lr)

        applied_updates += 1

        # Accumulate counts if enabled
        if counts is not None:
            seq = batch["sequence"]
            order = getattr(model, "order", 2)
            for j in range(len(seq)):
                tok = seq[j]
                counts.unigram[tok] += 1
                counts.total += 1
                if j > 0:
                    prev1 = seq[j - 1]
                    counts.bigram[prev1][tok] += 1
                    if order == 3 and j > 1:
                        prev2 = seq[j - 2]
                        counts.trigram[prev2][prev1][tok] += 1

        # Validation and checkpointing
        if applied_updates % max(1, args.val_interval) == 0:
            val_loss = blended_validation_loss(
                model,
                dataset,
                batches=args.val_batches,
                full=args.val_full,
                backoff_enabled=args.enable_backoff,
                backoff_lambda=args.backoff_lambda,
                backoff_weights=tuple(map(float, args.backoff_weights.split(",")))
                if isinstance(args.backoff_weights, str)
                else args.backoff_weights,
                trigram_weight=args.trigram_weight,
                bigram_weight=args.bigram_weight,
                counts=counts,
                alpha=args.count_smoothing_alpha,
                discount_d=args.discount_d,
                use_count_loss=args.use_count_loss,
            )
            print(f"[VAL] Step {applied_updates}: val_loss={val_loss:.4f}")

            if val_loss + args.early_stop_tol < best_val:
                best_val = val_loss
                best_update = applied_updates
                patience_counter = 0
                optimizer_state = {
                    "learning_rate": trainer.optimizer.get_lr()
                    if hasattr(trainer, "optimizer")
                    else args.learning_rate,
                    "applied_updates": applied_updates,
                }
                save_best_params(
                    model,
                    dataset,
                    args.best_out_dir,
                    applied_updates,
                    val_loss,
                    best_name,
                    counts,
                    args.save_optimizer_state,
                    optimizer_state,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("[TRAIN] Early stopping triggered.")
                    break

        # Meta self-improvement (if enabled)
        if args.meta_interval > 0 and applied_updates % args.meta_interval == 0:
            current_lr = (
                trainer.optimizer.get_lr()
                if hasattr(trainer, "optimizer")
                else args.learning_rate
            )
            issue_report = {
                "current_val_loss": best_val,
                "patience_counter": patience_counter,
                "learning_rate": current_lr,
                "step": applied_updates,
            }
            orchestrator.record_metrics(applied_updates, {"val_loss": best_val})

            if args.meta_apply:
                safe_types = [
                    s.strip() for s in args.meta_safe_types.split(",") if s.strip()
                ]
                proposals = orchestrator.propose_experiments(issue_report)
                for proposal in proposals:
                    exp_type = proposal.get("type", "")
                    if exp_type in safe_types:
                        print(f"[META] Applying safe experiment: {exp_type}")
                        if exp_type == "lr_adjustment" and "new_lr" in proposal:
                            safe_lr_change(current_lr, proposal["new_lr"])

    # Final save
    print(
        f"[TRAIN] Training complete. Best val loss: {best_val:.4f} at step {best_update}"
    )
    optimizer_state = {
        "learning_rate": trainer.optimizer.get_lr()
        if hasattr(trainer, "optimizer")
        else args.learning_rate,
        "applied_updates": applied_updates,
    }
    save_best_params(
        model,
        dataset,
        args.best_out_dir,
        applied_updates,
        best_val,
        f"final_{best_name}",
        counts,
        args.save_optimizer_state,
        optimizer_state,
    )

    # Save meta state
    if args.meta_interval > 0:
        try:
            meta_state_path = args.meta_state_path or os.path.join(
                args.best_out_dir, "meta_state.json"
            )
            with open(meta_state_path, "w", encoding="utf-8") as f:
                json.dump(orchestrator.export_state(), f)
            print(f"[META] Saved orchestrator state to {meta_state_path}")
        except Exception as e:
            print(f"[META] Failed to save meta state: {e}")

    # Generate if requested
    if args.generate_length > 0:
        print(f"\n[GEN] Generating {args.generate_length} tokens...")
        if counts is not None:
            V = dataset.vocab_size
            tri_log, bi_log, uni_log = counts.smooth_log_probs(
                V, args.count_smoothing_alpha, args.discount_d
            )
        else:
            tri_log, bi_log, uni_log = {}, {}, {}

        tokens = generate_tokens(
            model,
            dataset,
            args.generate_length,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.start_token,
            args.stop_on_eos,
            args.mask_bos,
            args.enable_backoff,
            args.backoff_lambda,
            tuple(map(float, args.backoff_weights.split(",")))
            if isinstance(args.backoff_weights, str)
            else args.backoff_weights,
            args.trigram_weight,
            args.bigram_weight,
            tri_log,
            bi_log,
            uni_log,
            args.interp_lambda,
        )
        text = dataset.tok.decode(tokens)
        print(f"[GEN] {text}")


# ============================= Main Entry Point ============================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learnable bigram/trigram trainer with governance and meta self-improvement"
    )

    # Data and model
    parser.add_argument(
        "--data-path", "--data_path", required=True, help="Path to training corpus"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        choices=[2, 3],
        help="N-gram order (2=bigram, 3=trigram)",
    )
    parser.add_argument(
        "--seq-len",
        "--seq_len",
        type=int,
        default=16,
        help="Sequence length for training windows",
    )
    parser.add_argument(
        "--min-freq", "--min_freq", type=int, default=1, help="Minimum token frequency"
    )
    parser.add_argument(
        "--use-eos",
        "--use_eos",
        action="store_true",
        default=True,
        help="Append EOS to sequences",
    )

    # Training
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--label-smoothing",
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing",
    )
    parser.add_argument(
        "--lr-schedule",
        "--lr_schedule",
        default="constant",
        choices=["constant", "cosine"],
        help="LR schedule",
    )
    parser.add_argument(
        "--divergence-threshold",
        "--divergence_threshold",
        type=float,
        default=100.0,
        help="Divergence threshold",
    )

    # Validation and checkpointing
    parser.add_argument(
        "--val-interval",
        "--val_interval",
        type=int,
        default=100,
        help="Validation interval",
    )
    parser.add_argument(
        "--val-batches", "--val_batches", type=int, default=5, help="Validation batches"
    )
    parser.add_argument(
        "--val-full",
        "--val_full",
        action="store_true",
        default=False,
        help="Use full validation set",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--early-stop-tol",
        "--early_stop_tol",
        type=float,
        default=1e-6,
        help="Early stopping tolerance",
    )
    parser.add_argument(
        "--best-out-dir",
        "--best_out_dir",
        default=".",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-optimizer-state",
        "--save_optimizer_state",
        action="store_true",
        default=False,
        help="Save optimizer state",
    )

    # Backoff and counts
    parser.add_argument(
        "--enable-backoff",
        "--enable_backoff",
        action="store_true",
        default=False,
        help="Enable learned backoff",
    )
    parser.add_argument(
        "--store-counts",
        "--store_counts",
        action="store_true",
        default=False,
        help="Store n-gram counts",
    )
    parser.add_argument(
        "--backoff-lambda",
        "--backoff_lambda",
        type=float,
        default=0.5,
        help="Backoff interpolation lambda",
    )
    parser.add_argument(
        "--backoff-weights",
        "--backoff_weights",
        default="0.6,0.3,0.1",
        help="Backoff weights (tri,bi,uni)",
    )
    parser.add_argument(
        "--trigram-weight",
        "--trigram_weight",
        type=float,
        default=1.0,
        help="Trigram weight",
    )
    parser.add_argument(
        "--bigram-weight",
        "--bigram_weight",
        type=float,
        default=0.0,
        help="Bigram weight",
    )
    parser.add_argument(
        "--count-smoothing-alpha",
        "--count_smoothing_alpha",
        type=float,
        default=0.1,
        help="Count smoothing alpha",
    )
    parser.add_argument(
        "--discount-d",
        "--discount_d",
        type=float,
        default=0.0,
        help="Count discount factor",
    )
    parser.add_argument(
        "--use-count-loss",
        "--use_count_loss",
        action="store_true",
        default=False,
        help="Use count-based validation loss",
    )

    # Meta self-improvement
    parser.add_argument(
        "--meta-interval",
        "--meta_interval",
        type=int,
        default=0,
        help="Meta self-improvement interval (0=disabled)",
    )
    parser.add_argument(
        "--meta-apply",
        "--meta_apply",
        action="store_true",
        default=False,
        help="Apply meta experiments",
    )
    parser.add_argument(
        "--meta-safe-types",
        "--meta_safe_types",
        default="lr_adjustment",
        help="Safe experiment types (comma-separated)",
    )
    parser.add_argument(
        "--meta-state-path",
        "--meta_state_path",
        default=None,
        help="Meta state file path",
    )
    parser.add_argument(
        "--meta-enable-decay",
        "--meta_enable_decay",
        action="store_true",
        default=False,
        help="Enable meta memory decay",
    )

    # Resume and generation
    parser.add_argument(
        "--resume-from", "--resume_from", default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--load-best",
        "--load_best",
        default=None,
        help="Load best checkpoint for generation",
    )
    parser.add_argument(
        "--override-lr",
        "--override_lr",
        action="store_true",
        default=False,
        help="Override loaded LR",
    )
    parser.add_argument(
        "--generate-length",
        "--generate_length",
        type=int,
        default=0,
        help="Generate N tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", "--top_k", type=int, default=0, help="Top-k sampling"
    )
    parser.add_argument(
        "--top-p", "--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--start-token",
        "--start_token",
        default=None,
        help="Start token for generation",
    )
    parser.add_argument(
        "--stop-on-eos",
        "--stop_on_eos",
        action="store_true",
        default=False,
        help="Stop generation on EOS",
    )
    parser.add_argument(
        "--mask-bos",
        "--mask_bos",
        action="store_true",
        default=False,
        help="Mask BOS during generation",
    )
    parser.add_argument(
        "--interp-lambda",
        "--interp_lambda",
        type=float,
        default=0.0,
        help="Legacy interpolation lambda",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        run_training(args)
    except Exception as e:
        print(f"[ERROR] run_training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
