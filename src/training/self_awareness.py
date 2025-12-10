"""
Self-Awareness Metrics Module for VULCAN-AMI Training
=====================================================

Provides a unified, extensible set of introspection / self-awareness metrics
to quantify model uncertainty, calibration, reliability, and diversity
during validation intervals.

Core Dimensions
---------------
1. Uncertainty / Confidence
   - Token entropy (bits)
   - Max-probability confidence
   - Negative log-likelihood (NLL) in nats
   - Perplexity (exp(NLL))

2. Calibration
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
   - Adaptive (quantile) ECE
   - Brier score (binary variant; optional)
   - (Optional) Multi-class ECE (micro / macro)

3. Accuracy
   - Token-level correctness ratio

4. Diversity / Novelty
   - Distinct-n (n=1,2,3) over generated sample (micro)
   - Macro distinct-n (average per sequence)
   - Optional per-sequence distinct-n distribution summary

5. Aggregate Summaries
   - Mean / std entropy
   - Combined calibration pack
   - Composite awareness dict for logging and orchestration gating

Integration Pattern
-------------------
Example (inside your validation interval in train_learnable_bigram.py):

    from training.self_awareness import (
        compute_validation_self_awareness,
        generate_sample_for_diversity,
        calculate_distinct_n,
        awareness_summary
    )

    # After computing validation loss `vloss`
    awareness = compute_validation_self_awareness(
        model=model,
        dataset=dataset,
        counts=counts if counts else None,
        batches=10,
        max_tokens_per_batch=seq_len,
        backoff_enabled=args.enable_backoff,
        backoff_lambda=args.backoff_lambda,
        backoff_weights=args.backoff_weights_tuple,
        trigram_weight=args.trigram_weight,
        bigram_weight=args.bigram_weight,
        alpha=args.count_smoothing_alpha,
        discount_d=args.discount_d,
        adaptive_ece_bins=15,
        fixed_ece_bins=15
    )

    # Optional: novelty from a short generation
    sample_ids = generate_sample_for_diversity(
        model=model,
        dataset=dataset,
        counts=counts,
        length=min(64, args.seq_len * 2),
        temperature=max(0.8, args.temperature),
        top_k=max(10, args.top_k),
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        start_token=args.start_token,
        mask_bos=args.mask_bos,
        backoff_enabled=args.enable_backoff,
        backoff_lambda=args.backoff_lambda,
        backoff_weights=args.backoff_weights_tuple,
        trigram_weight=args.trigram_weight,
        bigram_weight=args.bigram_weight
    )

    awareness["distinct_1"] = calculate_distinct_n([sample_ids], 1)
    awareness["distinct_2"] = calculate_distinct_n([sample_ids], 2)
    awareness["distinct_3"] = calculate_distinct_n([sample_ids], 3)

    composite = awareness_summary(awareness)
    orchestrator.record_telemetry(
        loss=loss,
        eval_score=vloss,
        novelty_score=awareness.get("distinct_3", 0.0),
        **{"meta_awareness": composite}
    )

Design Choices
--------------
- Uses pure Python for portability (no numpy required).
- Backoff blending hooks remain compatible with existing n-gram + counts logic.
- Clean separation between probability extraction and metric computation.
- All public functions are pure or side-effect free except generation helper.
- Extended with optional multi-class calibration and diversity macro metrics.

Neural Model Adaptation
-----------------------
If you later switch to a neural model:
- Replace `_softmax` with torch.softmax (detach + CPU).
- Provide logits directly rather than constructing from layered dictionaries.

Revision Notes (Fix Applied)
----------------------------
Previous implementation of `_combine_learned_and_counts` mixed learned "logits" (possibly raw scores)
with count-based LOG probabilities directly and then softmaxed the linear combination. This merged
log-space values and linear-space values incorrectly.

Fix:
- Convert learned logits to a probability distribution via softmax.
- Convert count log-probabilities back to probabilities (exp).
- Interpolate in probability space: p_final = λ * p_learned + (1-λ) * p_counts.
- Convert back to log-space (log(p_final + ε)) so downstream softmax recovers p_final exactly.

This produces mathematically coherent interpolation between learned distribution and count-based distribution.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ============================================================
# Public Export Control
# ============================================================

__all__ = [
    "compute_validation_self_awareness",
    "generate_sample_for_diversity",
    "calculate_distinct_n",
    "calculate_distinct_n_per_sequence",
    "calculate_macro_distinct_n",
    "calculate_distinct_all",
    "awareness_summary",
    "multi_class_ece",
    "compute_multi_class_calibration",
    "calculate_confidence",
    "calculate_token_nll",
    "calculate_perplexity",
    "summarize_entropies",
    "calculate_ece",
    "calculate_mce",
    "calculate_adaptive_ece",
    "calculate_brier_score",
    "trainer_reaction",
    "build_extended_awareness",
]

# ============================================================
# 1. Basic Probability Utilities
# ============================================================


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    Z = sum(exps)
    if Z <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / Z for e in exps]


def _normalize(probabilities: Sequence[float], eps: float = 1e-12) -> List[float]:
    s = sum(probabilities)
    if s <= eps:
        n = len(probabilities)
        return [1.0 / n] * n if n > 0 else []
    return [p / s for p in probabilities]


def _entropy_bits(
    probabilities: Sequence[float], assume_normalized: bool = True
) -> float:
    """
    Shannon entropy in bits.
    If assume_normalized=False, will normalize first.
    """
    if not assume_normalized:
        probabilities = _normalize(probabilities)
    h = 0.0
    for p in probabilities:
        if p > 0.0:
            h -= p * math.log2(p)
    return h


# ============================================================
# 2. Core Metrics: Confidence, NLL, Perplexity, Entropy Summary
# ============================================================


def calculate_confidence(probabilities: Sequence[float]) -> float:
    return max(probabilities) if probabilities else 0.0


def calculate_token_nll(
    true_token_probabilities: Sequence[float],
    log_base: float = math.e,
    eps: float = 1e-12,
) -> float:
    """
    Average negative log-likelihood over token correct probabilities.
    log_base = e -> nats; log_base = 2 -> bits.
    """
    if not true_token_probabilities:
        return 0.0
    adjusted = [max(p, eps) for p in true_token_probabilities]
    logs = [-math.log(p, log_base) for p in adjusted]
    return sum(logs) / len(logs)


def calculate_perplexity(
    true_token_probabilities: Sequence[float],
    log_base: float = math.e,
    eps: float = 1e-12,
) -> float:
    """
    Perplexity derived from average NLL.
    """
    if not true_token_probabilities:
        return 0.0
    nll = calculate_token_nll(true_token_probabilities, log_base=log_base, eps=eps)
    if log_base == math.e:
        return math.exp(nll)
    elif log_base == 2.0:
        return 2.0**nll
    else:
        return log_base**nll


def summarize_entropies(
    prob_distributions: Sequence[Sequence[float]], assume_normalized: bool = True
) -> Dict[str, float]:
    entropies: List[float] = []
    for probs in prob_distributions:
        if probs:
            entropies.append(_entropy_bits(probs, assume_normalized=assume_normalized))
    if not entropies:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    mean = statistics.mean(entropies)
    std = statistics.pstdev(entropies) if len(entropies) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(entropies),
        "max": max(entropies),
        "count": len(entropies),
    }


# ============================================================
# 3. Calibration Metrics (ECE, MCE, Adaptive ECE, Brier + Multi-class)
# ============================================================


def _init_bins(n_bins: int) -> Dict[int, Tuple[float, float, int]]:
    return {i: (0.0, 0.0, 0) for i in range(n_bins)}


def compute_calibration_bins(
    predictions: Sequence[int],
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    if len(predictions) != len(confidences) or len(predictions) != len(labels):
        raise ValueError("Lengths mismatch for predictions/confidences/labels.")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")

    bins = _init_bins(n_bins)
    boundaries = [i / n_bins for i in range(n_bins + 1)]

    for pred, conf, label in zip(predictions, confidences, labels):
        conf = max(0.0, min(conf, 1.0))
        idx = min(int(conf * n_bins), n_bins - 1)
        total_conf, total_acc, count = bins[idx]
        count += 1
        total_conf += conf
        total_acc += 1.0 if pred == label else 0.0
        bins[idx] = (total_conf, total_acc, count)

    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        total_conf, total_acc, count = bins[i]
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if count > 0:
            avg_conf = total_conf / count
            avg_acc = total_acc / count
        else:
            avg_conf = avg_acc = 0.0
        out.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "count": count,
            }
        )
    return out


def calculate_ece(
    predictions: Sequence[int],
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    bins = compute_calibration_bins(predictions, confidences, labels, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        w = b["count"] / total
        ece += w * abs(b["avg_accuracy"] - b["avg_confidence"])
    return ece


def calculate_mce(
    predictions: Sequence[int],
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    bins = compute_calibration_bins(predictions, confidences, labels, n_bins=n_bins)
    max_gap = 0.0
    for b in bins:
        gap = abs(b["avg_accuracy"] - b["avg_confidence"])
        if gap > max_gap:
            max_gap = gap
    return max_gap


def calculate_adaptive_ece(
    predictions: Sequence[int],
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    if len(predictions) != len(confidences) or len(predictions) != len(labels):
        raise ValueError("Lengths mismatch for adaptive ECE.")
    total = len(predictions)
    if total == 0:
        return 0.0
    data = sorted(zip(confidences, predictions, labels))
    target_bin_size = total / n_bins
    ece = 0.0
    for i in range(n_bins):
        start = int(i * target_bin_size)
        end = int((i + 1) * target_bin_size) if i < n_bins - 1 else total
        subset = data[start:end]
        if not subset:
            continue
        bin_confs = [c for c, _, _ in subset]
        bin_preds = [p for _, p, _ in subset]
        bin_labels = [l for _, _, l in subset]
        avg_conf = sum(bin_confs) / len(bin_confs)
        acc = sum(1.0 for p, l in zip(bin_preds, bin_labels) if p == l) / len(subset)
        w = len(subset) / total
        ece += w * abs(acc - avg_conf)
    return ece


def calculate_brier_score(
    confidences: Sequence[float],
    labels: Sequence[int],
    positive_label: int = 1,
) -> float:
    if len(confidences) != len(labels):
        raise ValueError("Lengths mismatch for Brier score.")
    if not confidences:
        return 0.0
    errors = []
    for p, y in zip(confidences, labels):
        p = max(0.0, min(p, 1.0))
        y_bin = 1.0 if y == positive_label else 0.0
        errors.append((p - y_bin) ** 2)
    return sum(errors) / len(errors)


# ---------- Multi-class Calibration (Optional) ----------


def compute_multi_class_calibration(
    all_class_probs: Sequence[Sequence[float]],
    true_labels: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes per-bin calibration using top-class confidence for multi-class predictions,
    plus class-wise accuracy distribution.

    Args:
        all_class_probs: list of per-token probability vectors.
        true_labels: list of ground-truth class indices.
        n_bins: number of bins for ECE/MCE.

    Returns:
        {
            "ece": float,
            "mce": float,
            "bins": [...],
            "class_accuracy": {class_idx: accuracy},
            "macro_accuracy": float,
            "micro_accuracy": float
        }
    """
    if len(all_class_probs) != len(true_labels):
        raise ValueError("Length mismatch between probability vectors and labels.")
    predictions: List[int] = []
    confidences: List[float] = []
    for probs in all_class_probs:
        if not probs:
            predictions.append(-1)
            confidences.append(0.0)
            continue
        conf = max(probs)
        pred = probs.index(conf)
        predictions.append(pred)
        confidences.append(conf)

    bins = compute_calibration_bins(
        predictions, confidences, true_labels, n_bins=n_bins
    )
    ece = calculate_ece(predictions, confidences, true_labels, n_bins=n_bins)
    mce = calculate_mce(predictions, confidences, true_labels, n_bins=n_bins)

    # Class-wise accuracy
    class_correct: Dict[int, int] = {}
    class_total: Dict[int, int] = {}
    for p, t in zip(predictions, true_labels):
        class_total[t] = class_total.get(t, 0) + 1
        if p == t:
            class_correct[t] = class_correct.get(t, 0) + 1

    class_accuracy: Dict[int, float] = {
        c: (class_correct.get(c, 0) / class_total[c]) for c in class_total
    }
    macro_accuracy = (
        (sum(class_accuracy.values()) / len(class_accuracy)) if class_accuracy else 0.0
    )
    micro_accuracy = (
        (sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(true_labels))
        if true_labels
        else 0.0
    )

    return {
        "ece": ece,
        "mce": mce,
        "bins": bins,
        "class_accuracy": class_accuracy,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
    }


def multi_class_ece(
    all_class_probs: Sequence[Sequence[float]],
    true_labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """
    Convenience function returning only ECE for multi-class.
    """
    cal = compute_multi_class_calibration(all_class_probs, true_labels, n_bins=n_bins)
    return cal["ece"]


# ============================================================
# 4. Diversity / Distinct-n (Micro & Macro)
# ============================================================


def _get_ngrams(tokens: Sequence[Any], n: int) -> Iterable[Tuple[Any, ...]]:
    length = len(tokens)
    if length < n:
        return
    for i in range(length - n + 1):
        yield tuple(tokens[i : i + n])


def calculate_distinct_n(
    sequences: Sequence[Sequence[Any]],
    n: int,
) -> float:
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not sequences:
        return 0.0
    total_ngrams = 0
    unique: set = set()
    for seq in sequences:
        for ng in _get_ngrams(seq, n):
            unique.add(ng)
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(unique) / total_ngrams


def calculate_distinct_n_per_sequence(
    sequences: Sequence[Sequence[Any]], n: int
) -> List[float]:
    if n <= 0:
        raise ValueError("n must be > 0.")
    scores: List[float] = []
    for seq in sequences:
        total = 0
        unique: set = set()
        for ng in _get_ngrams(seq, n):
            unique.add(ng)
            total += 1
        scores.append((len(unique) / total) if total > 0 else 0.0)
    return scores


def calculate_macro_distinct_n(sequences: Sequence[Sequence[Any]], n: int) -> float:
    per = calculate_distinct_n_per_sequence(sequences, n)
    return statistics.mean(per) if per else 0.0


def calculate_distinct_all(
    sequences: Sequence[Sequence[Any]],
    ns: Sequence[int] = (1, 2, 3),
    macro: bool = False,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for n in ns:
        micro_score = calculate_distinct_n(sequences, n)
        result[f"distinct_{n}"] = micro_score
        if macro:
            result[f"macro_distinct_{n}"] = calculate_macro_distinct_n(sequences, n)
    return result


# ============================================================
# 5. Probability Extraction for N-Gram + Backoff Model
# ============================================================


def _combine_learned_and_counts(
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
    Properly combine learned logits and count-based log probabilities.

    Steps:
    1. Build learned logits (raw scores) for the current context.
    2. Convert learned logits -> learned_probs via softmax.
    3. Build count-based log probabilities (already in log space), convert to probabilities via exp.
    4. Weighted interpolation in probability space:
          p_final = backoff_lambda * learned_probs + (1-backoff_lambda) * count_probs
    5. Return log(p_final + eps) so downstream softmax reproduces p_final.

    This avoids mixing raw score space and log-probability space directly.
    """
    # 1. Learned logits
    learned_logits = [0.0] * V
    if order == 3 and tri_logits is not None:
        for i in range(V):
            learned_logits[i] += trigram_weight * tri_logits[i]
    if big_logits is not None and bigram_weight > 0.0:
        for i in range(V):
            learned_logits[i] += bigram_weight * big_logits[i]

    # 2. Learned probabilities
    learned_probs = _softmax(learned_logits) if learned_logits else [1.0 / V] * V

    # 3. Count-based probabilities
    w3, w2, w1 = backoff_weights
    count_log_values = [0.0] * V
    if order == 3:
        for i in range(V):
            tri_key = (prev2, prev1, i)
            v3 = count_tri_log.get(tri_key)
            if v3 is not None:
                count_log_values[i] += w3 * v3
            bi_key = (prev1, i)
            count_log_values[i] += w2 * count_bi_log.get(
                bi_key, count_uni_log.get(i, -math.log(V))
            )
            count_log_values[i] += w1 * count_uni_log.get(i, -math.log(V))
    else:
        for i in range(V):
            bi_key = (prev1, i)
            count_log_values[i] += w2 * count_bi_log.get(
                bi_key, count_uni_log.get(i, -math.log(V))
            )
            count_log_values[i] += w1 * count_uni_log.get(i, -math.log(V))

    # Convert log probabilities to probabilities
    count_probs = [math.exp(cv) for cv in count_log_values]
    # Normalize counts distribution defensively
    count_probs = _normalize(count_probs, eps)

    # 4. Interpolate distributions
    final_probs = [
        backoff_lambda * learned_probs[i] + (1 - backoff_lambda) * count_probs[i]
        for i in range(V)
    ]
    # Final normalization (may be slightly off due to numeric issues)
    final_probs = _normalize(final_probs, eps)

    # 5. Return log probabilities as "logits"
    return [math.log(p + eps) for p in final_probs]


def extract_batch_probabilities(
    model: Any,
    dataset: Any,
    vb: Dict[str, Any],
    counts: Optional[Any],
    *,
    backoff_enabled: bool,
    backoff_lambda: float,
    backoff_weights: Tuple[float, float, float],
    trigram_weight: float,
    bigram_weight: float,
    alpha: float,
    discount_d: float,
) -> Tuple[List[List[float]], List[int]]:
    """
    Returns per-position probability vectors and ground-truth targets for one validation batch.
    """
    seq = vb.get("sequence")
    if not seq or len(seq) < 2:
        return [], []
    tokens = seq[:-1]
    targets = seq[1:]
    V = dataset.vocab_size
    order = getattr(model, "order", 2)
    layers = model.params["transformer_layers"]

    tri_log, bi_log, uni_log = {}, {}, {}
    if counts is not None:
        tri_log, bi_log, uni_log = counts.smooth_log_probs(V, alpha, discount_d)

    probs_list: List[List[float]] = []
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
            log_mix = _combine_learned_and_counts(
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
            probs = _softmax(
                log_mix
            )  # converts log probabilities back to probabilities
        else:
            raw_logits = tri_logits if tri_logits is not None else big_logits
            probs = _softmax(raw_logits) if raw_logits else [1.0 / V] * V

        probs_list.append(probs)

    return probs_list, list(targets)


# ============================================================
# 6. Self-Awareness Aggregation
# ============================================================


def compute_validation_self_awareness(
    model: Any,
    dataset: Any,
    counts: Optional[Any],
    *,
    batches: int = 5,
    max_tokens_per_batch: Optional[int] = None,
    backoff_enabled: bool,
    backoff_lambda: float,
    backoff_weights: Tuple[float, float, float],
    trigram_weight: float,
    bigram_weight: float,
    alpha: float,
    discount_d: float,
    fixed_ece_bins: int = 15,
    adaptive_ece_bins: int = 15,
    brier_positive_label: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes a suite of self-awareness metrics over several sampled validation batches.

    Returns dict containing:
        - avg_entropy_bits
        - entropy_std_bits
        - avg_confidence
        - accuracy
        - nll_nats
        - perplexity
        - ece
        - mce
        - adaptive_ece
        - brier_score (if positive label provided)
        - token_count
    """
    all_probs: List[List[float]] = []
    all_targets: List[int] = []
    all_confidences: List[float] = []
    all_correct_flags: List[bool] = []
    true_token_probs: List[float] = []
    predictions: List[int] = []

    for _ in range(batches):
        vb = dataset.sample_val_batch()
        probs_list, targets = extract_batch_probabilities(
            model,
            dataset,
            vb,
            counts,
            backoff_enabled=backoff_enabled,
            backoff_lambda=backoff_lambda,
            backoff_weights=backoff_weights,
            trigram_weight=trigram_weight,
            bigram_weight=bigram_weight,
            alpha=alpha,
            discount_d=discount_d,
        )
        if not probs_list:
            continue

        if max_tokens_per_batch is not None and len(probs_list) > max_tokens_per_batch:
            probs_list = probs_list[:max_tokens_per_batch]
            targets = targets[:max_tokens_per_batch]

        for probs, tgt in zip(probs_list, targets):
            if not probs:
                continue
            conf = calculate_confidence(probs)
            pred = probs.index(conf)
            correct = pred == tgt
            true_prob = probs[tgt] if 0 <= tgt < len(probs) else 0.0

            all_probs.append(probs)
            all_targets.append(tgt)
            all_confidences.append(conf)
            all_correct_flags.append(correct)
            true_token_probs.append(true_prob)
            predictions.append(pred)

    token_count = len(all_targets)
    if token_count == 0:
        return {
            "avg_entropy_bits": 0.0,
            "entropy_std_bits": 0.0,
            "avg_confidence": 0.0,
            "accuracy": 0.0,
            "nll_nats": 0.0,
            "perplexity": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "adaptive_ece": 0.0,
            "brier_score": 0.0,
            "token_count": 0,
        }

    entropy_summary = summarize_entropies(all_probs)
    accuracy = sum(1.0 for c in all_correct_flags if c) / token_count
    nll_nats = calculate_token_nll(true_token_probs, log_base=math.e)
    ppl = calculate_perplexity(true_token_probs, log_base=math.e)

    ece = calculate_ece(
        predictions, all_confidences, all_targets, n_bins=fixed_ece_bins
    )
    mce = calculate_mce(
        predictions, all_confidences, all_targets, n_bins=fixed_ece_bins
    )
    adaptive = calculate_adaptive_ece(
        predictions, all_confidences, all_targets, n_bins=adaptive_ece_bins
    )

    if brier_positive_label is not None:
        brier = calculate_brier_score(
            all_confidences, all_targets, positive_label=brier_positive_label
        )
    else:
        brier = 0.0

    return {
        "avg_entropy_bits": entropy_summary["mean"],
        "entropy_std_bits": entropy_summary["std"],
        "avg_confidence": sum(all_confidences) / token_count,
        "accuracy": accuracy,
        "nll_nats": nll_nats,
        "perplexity": ppl,
        "ece": ece,
        "mce": mce,
        "adaptive_ece": adaptive,
        "brier_score": brier,
        "token_count": token_count,
    }


# ============================================================
# 7. Generation Helper for Diversity Sampling
# ============================================================


def generate_sample_for_diversity(
    model: Any,
    dataset: Any,
    counts: Optional[Any],
    *,
    length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    start_token: Optional[int],
    mask_bos: bool,
    backoff_enabled: bool,
    backoff_lambda: float,
    backoff_weights: Tuple[float, float, float],
    trigram_weight: float,
    bigram_weight: float,
    alpha: float = 0.5,
    discount_d: float = 0.2,
) -> List[int]:
    """
    Lightweight token generation for diversity metrics.

    If a generic `generate_tokens` function exists (in surrounding training code),
    delegate to it for consistency. Otherwise use a simple greedy fallback,
    which is intentionally NOT diverse (only structural placeholder).
    """
    if "generate_tokens" in globals():
        tri_log, bi_log, uni_log = ({}, {}, {})
        if counts is not None:
            tri_log, bi_log, uni_log = counts.smooth_log_probs(
                dataset.vocab_size, alpha, discount_d
            )
        return globals()["generate_tokens"](
            model=model,
            dataset=dataset,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            start_token=start_token,
            stop_on_eos=True,
            mask_bos=mask_bos,
            backoff_enabled=backoff_enabled,
            backoff_lambda=backoff_lambda,
            backoff_weights=backoff_weights,
            trigram_weight=trigram_weight,
            bigram_weight=bigram_weight,
            count_tri_log=tri_log,
            count_bi_log=bi_log,
            count_uni_log=uni_log,
        )

    generated: List[int] = []
    current_prev1 = start_token if start_token is not None else dataset.bos_id
    current_prev2 = dataset.bos_id

    tri_log, bi_log, uni_log = {}, {}, {}
    if counts is not None:
        tri_log, bi_log, uni_log = counts.smooth_log_probs(
            dataset.vocab_size, alpha, discount_d
        )

    order = getattr(model, "order", 2)
    layers = model.params["transformer_layers"]
    V = dataset.vocab_size

    for _ in range(length):
        if order == 3:
            tri_row_p1 = layers["trigram"].get(current_prev2, {})
            tri_row_next = tri_row_p1.get(current_prev1, {})
            tri_logits = [tri_row_next.get(i, 0.0) for i in range(V)]
            big_logits = None
        else:
            tri_logits = None
            big_row = layers["bigram"].get(current_prev1, {})
            big_logits = [big_row.get(i, 0.0) for i in range(V)]

        if backoff_enabled:
            log_mix = _combine_learned_and_counts(
                order,
                tri_logits,
                big_logits,
                tri_log,
                bi_log,
                uni_log,
                current_prev2,
                current_prev1,
                V,
                trigram_weight,
                bigram_weight,
                backoff_weights,
                backoff_lambda,
            )
            probs = _softmax(log_mix)
        else:
            raw_logits = tri_logits if tri_logits is not None else big_logits
            probs = _softmax(raw_logits) if raw_logits else [1.0 / V] * V

        nxt = probs.index(max(probs))
        generated.append(nxt)
        current_prev2, current_prev1 = current_prev1, nxt
        if nxt == dataset.eos_id:
            break

    return generated


# ============================================================
# 8. Composite Awareness Summary / Thresholding
# ============================================================


def awareness_summary(awareness: Dict[str, float]) -> Dict[str, Any]:
    """
    Provide extended summary with sanity checks & derived metrics.
    """
    out = dict(awareness)

    if awareness.get("perplexity", 0.0) > 0.0 and awareness.get("nll_nats", 0.0) > 0.0:
        out["ppl_nll_ratio"] = awareness["perplexity"] / max(
            awareness["nll_nats"], 1e-12
        )
    else:
        out["ppl_nll_ratio"] = 0.0

    gap = awareness.get("avg_confidence", 0.0) - awareness.get("accuracy", 0.0)
    out["confidence_accuracy_gap"] = gap

    ece = awareness.get("ece", 0.0)
    mce = awareness.get("mce", 0.0)
    out["calibration_risk"] = ece * 0.6 + mce * 0.4

    entropy_mean = awareness.get("avg_entropy_bits", 0.0)
    if entropy_mean < 0.2:
        out["entropy_flag"] = "low_entropy"
    else:
        out["entropy_flag"] = "ok"

    out["is_calibration_degrading"] = ece > 0.15 and gap > 0.1
    out["is_overconfident"] = gap > 0.2 and ece > 0.05
    out["is_uncertain"] = (
        awareness.get("avg_confidence", 0.0) < 0.3 and entropy_mean > 2.0
    )

    return out


# ============================================================
# 9. OPTIONAL: Threshold-based Trainer Reaction (Example API)
# ============================================================


def trainer_reaction(
    awareness: Dict[str, float], lr: float, min_lr: float = 1e-5, max_lr: float = 1e-1
) -> Dict[str, Any]:
    """
    Example heuristic adaptation logic.
    Returns a dict of recommended adjustments.
    """
    adjustments: Dict[str, Any] = {}
    ece = awareness.get("ece", 0.0)
    acc = awareness.get("accuracy", 0.0)
    conf = awareness.get("avg_confidence", 0.0)
    entropy_mean = awareness.get("avg_entropy_bits", 0.0)

    new_lr = lr
    if ece > 0.2:
        new_lr = max(min_lr, lr * 0.9)
        adjustments["lr_reason"] = "high_ece_reduce_lr"
    if acc < 0.5 and conf > 0.8:
        adjustments["curriculum_action"] = "increase_hard_examples"
    if entropy_mean < 0.25:
        adjustments["sampling_adjustment"] = "increase_temperature"
    if entropy_mean > 4.0 and conf < 0.4:
        adjustments["sampling_adjustment"] = "decrease_temperature"

    adjustments["recommended_lr"] = min(max_lr, max(min_lr, new_lr))
    return adjustments


# ============================================================
# 10. Multi-Class + Diversity Integrated Awareness (Aggregator)
# ============================================================


def build_extended_awareness(
    base_awareness: Dict[str, float],
    sequences: Optional[Sequence[Sequence[Any]]] = None,
    multi_class_probs: Optional[Sequence[Sequence[float]]] = None,
    multi_class_labels: Optional[Sequence[int]] = None,
    distinct_ns: Sequence[int] = (1, 2, 3),
    include_macro_distinct: bool = True,
    multi_class_bins: int = 10,
) -> Dict[str, Any]:
    """
    Extend a base awareness dictionary with:
      - Distinct-n micro/macro scores (if sequences provided)
      - Multi-class calibration pack (if multi-class probabilities provided)
    """
    extended = dict(base_awareness)

    if sequences:
        diversity = calculate_distinct_all(
            sequences, ns=distinct_ns, macro=include_macro_distinct
        )
        extended.update(diversity)
        for n in distinct_ns:
            per_scores = calculate_distinct_n_per_sequence(sequences, n)
            extended[f"distinct_{n}_per_sequence_mean"] = (
                statistics.mean(per_scores) if per_scores else 0.0
            )
            extended[f"distinct_{n}_per_sequence_std"] = (
                statistics.pstdev(per_scores) if len(per_scores) > 1 else 0.0
            )

    if multi_class_probs and multi_class_labels is not None:
        mc_cal = compute_multi_class_calibration(
            multi_class_probs, multi_class_labels, n_bins=multi_class_bins
        )
        extended["multi_class_ece"] = mc_cal["ece"]
        extended["multi_class_mce"] = mc_cal["mce"]
        extended["multi_class_macro_accuracy"] = mc_cal["macro_accuracy"]
        extended["multi_class_micro_accuracy"] = mc_cal["micro_accuracy"]

    extended_summary = awareness_summary(extended)
    extended.update({f"summary_{k}": v for k, v in extended_summary.items()})

    return extended


# ============================================================
# END OF MODULE
# ============================================================
