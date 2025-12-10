from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossMetrics:
    # All losses are in natural-log units (nats)
    loss_sum_valid: float  # Sum of cross-entropy over valid targets
    valid_tokens: int  # Count of targets that contributed
    loss_per_token: float  # Average cross-entropy over valid targets
    ppl: float  # Perplexity = exp(loss_per_token)


def _flatten_logits_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten (B, T, V) logits and (B, T) targets to (N, V) and (N,),
    or pass-through if already flat.
    """
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
    elif logits.dim() == 2:
        logits_flat = logits
    else:
        raise ValueError(f"logits must be (B,T,V) or (N,V), got {tuple(logits.shape)}")

    if targets.dim() == 2:
        targets_flat = targets.reshape(-1)
    elif targets.dim() == 1:
        targets_flat = targets
    else:
        raise ValueError(f"targets must be (B,T) or (N,), got {tuple(targets.shape)}")

    if logits_flat.size(0) != targets_flat.size(0):
        raise ValueError(
            f"Flattened logits and targets must have same N. "
            f"Got logits N={logits_flat.size(0)} vs targets N={targets_flat.size(0)}"
        )
    return logits_flat, targets_flat


@torch.no_grad()
def compute_loss_metrics_eval(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None
) -> LossMetrics:
    """
    Compute evaluation CE loss metrics in nats:
    - loss_sum_valid: sum over valid positions only
    - loss_per_token: average over valid positions
    - ppl: exp(loss_per_token)

    This function does NOT backprop. It is safe under torch.no_grad().
    """
    logits_flat, targets_flat = _flatten_logits_targets(logits, targets)

    # Compute per-position loss in nats, do not reduce
    loss_vec = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
        ignore_index=ignore_index
        if ignore_index is not None
        else -100,  # default HF ignore_index
    )

    # Build valid mask
    if ignore_index is None:
        # all positions valid
        valid_mask = torch.ones_like(targets_flat, dtype=torch.bool)
    else:
        valid_mask = targets_flat != ignore_index

    valid_loss = loss_vec[valid_mask]
    valid_tokens = int(valid_mask.sum().item())

    loss_sum_valid = float(valid_loss.sum().item())
    loss_per_token = float(loss_sum_valid / max(valid_tokens, 1))

    ppl = math.exp(loss_per_token)  # natural log base

    return LossMetrics(
        loss_sum_valid=loss_sum_valid,
        valid_tokens=valid_tokens,
        loss_per_token=loss_per_token,
        ppl=ppl,
    )


def compute_loss_metrics_train(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None
) -> Tuple[torch.Tensor, LossMetrics]:
    """
    Compute training CE loss and metrics:
    - Returns a scalar loss suitable for backprop (loss_per_token),
      and a LossMetrics object for logging.
    - Uses the same normalization as eval: divide by number of valid targets ONCE.
    """
    logits_flat, targets_flat = _flatten_logits_targets(logits, targets)

    loss_vec = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
        ignore_index=ignore_index if ignore_index is not None else -100,
    )

    if ignore_index is None:
        valid_mask = torch.ones_like(targets_flat, dtype=torch.bool)
    else:
        valid_mask = targets_flat != ignore_index

    valid_loss = loss_vec[valid_mask]
    valid_tokens = int(valid_mask.sum().item())

    # Normalize once by valid token count
    loss_sum_valid = valid_loss.sum()
    loss_per_token = loss_sum_valid / max(valid_tokens, 1)

    # For metrics logging
    metrics = LossMetrics(
        loss_sum_valid=float(loss_sum_valid.detach().item()),
        valid_tokens=valid_tokens,
        loss_per_token=float(loss_per_token.detach().item()),
        ppl=float(math.exp(loss_per_token.detach().item())),
    )

    return loss_per_token, metrics


@torch.no_grad()
def random_label_sanity(
    vocab_size: int,
    targets: torch.Tensor,
    device: Optional[torch.device] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Sanity test: replace targets by random labels in [0, vocab_size).
    With uniform logits (or random logits with mean ~0), expected:
      loss_per_token ≈ ln(vocab_size), ppl ≈ vocab_size.

    Returns dict with computed loss and ppl and the theoretical ln(V), V for comparison.
    """
    if device is None:
        device = targets.device

    # Create random labels, keeping ignore positions as-is if ignore_index is set
    if ignore_index is None:
        rand_tgts = torch.randint(
            low=0, high=vocab_size, size=targets.shape, device=device
        )
    else:
        rand_tgts = torch.randint(
            low=0, high=vocab_size, size=targets.shape, device=device
        )
        rand_tgts = torch.where(targets == ignore_index, targets, rand_tgts)

    # Uniform logits = zeros
    if targets.dim() == 2:
        B, T = targets.shape
        logits = torch.zeros((B, T, vocab_size), device=device)
    elif targets.dim() == 1:
        N = targets.shape[0]
        logits = torch.zeros((N, vocab_size), device=device)
    else:
        raise ValueError(f"targets must be (B,T) or (N,), got {tuple(targets.shape)}")

    metrics = compute_loss_metrics_eval(logits, rand_tgts, ignore_index=ignore_index)

    return {
        "loss_per_token": metrics.loss_per_token,
        "ppl": metrics.ppl,
        "expected_ln_vocab": math.log(vocab_size),
        "expected_vocab": float(vocab_size),
        "valid_tokens": float(metrics.valid_tokens),
    }


@torch.no_grad()
def uniform_logits_sanity(
    vocab_size: int,
    targets: torch.Tensor,
    logits_shape_like: Tuple[int, ...],
    device: Optional[torch.device] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Sanity test: given real targets but force logits to be uniform (zeros).
    Expected: loss_per_token ≈ ln(vocab_size), ppl ≈ vocab_size.
    """
    if device is None:
        device = targets.device

    logits = torch.zeros(logits_shape_like, device=device)
    metrics = compute_loss_metrics_eval(logits, targets, ignore_index=ignore_index)
    return {
        "loss_per_token": metrics.loss_per_token,
        "ppl": metrics.ppl,
        "expected_ln_vocab": math.log(vocab_size),
        "expected_vocab": float(vocab_size),
        "valid_tokens": float(metrics.valid_tokens),
    }
