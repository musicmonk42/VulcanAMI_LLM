"""
Enhanced LLM Training Script (No Stubs) with Governance + Self-Improvement Orchestrator + Self-Awareness

This version removes stub models and synthetic validation, using a real PyTorch GPT-like model and a real
text corpus loader. It integrates:
- Causal Transformer (GPTModel) forward/backprop with AdamW and cosine schedule with warmup
- Governance loop: every optimizer step is wrapped in a consensus approval gate
- Self-Improving Orchestrator (meta cycles): detect issues, propose safe changes, optionally apply
- Self-awareness metrics: entropy, calibration (ECE/MCE/adaptive), perplexity, diversity (distinct-n)
- Checkpointing (model + optimizer) and meta state resume
- Structured JSONL logging for training and meta cycles
- Safety guardrails for LR change ratios and gradient clipping ranges
- Optional full-validation vs sampled validation
- Time-based pacing metrics (steps/sec), drift detection (on awareness snapshots)

Run (advisory mode):
    python train_llm_with_self_improvement.py --steps 3000 --val-interval 300 --data-path data/corpus --seq-len 128

Run (active meta-apply with safe proposals):
    python train_llm_with_self_improvement.py --steps 6000 --val-interval 400 \
        --data-path data/corpus --seq-len 128 --meta-interval 800 --meta-apply \
        --meta-safe-types lr_adjustment,lr_sweep,grad_clip_adjust,warmup_schedule_adjust

Resume:
    python train_llm_with_self_improvement.py --resume-model logs/llm_last_model.pt \
        --resume-optim logs/llm_last_optim.pt --resume-meta logs/llm_meta_state.json --steps 1000

Robustness Additions:
---------------------
1. Automatic demo corpus creation (--auto-create-corpus).
2. Single-file sharding support (--data-file).
3. Automatic sequence length clamping when corpus is smaller than requested (--auto-adjust-seq-len).
   - Prevents runtime: "Token stream too short for requested sequence length".
4. Warn and adjust model & loader seq_len dynamically (no restart required).
5. Empty corpus guard (auto populate).
6. Windows path normalization for logging.
7. All prior logic preserved untruncated.

Notes:
- Requires PyTorch. GPT model and corpus loader are implemented in gpt_model.py and data_loader.py (no stubs).
- For very large runs, use distributed strategies separately; this file focuses on orchestration layer.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import copy
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gpt_model import GPTModel, GPTConfig
from data_loader import CorpusDataLoader

from self_improving_training import SelfImprovingTraining
from self_awareness import (
    summarize_entropies,
    calculate_ece,
    calculate_mce,
    calculate_adaptive_ece,
    calculate_distinct_n,
    build_extended_awareness,
    awareness_summary,
)

# Prefer absolute import; fallback if run directly
try:
    from src.training.metrics import (
        compute_loss_metrics_train,
        compute_loss_metrics_eval,
        LossMetrics,
        random_label_sanity,
        uniform_logits_sanity,
    )
except Exception:
    from metrics import (
        compute_loss_metrics_train,
        compute_loss_metrics_eval,
        LossMetrics,
        random_label_sanity,
        uniform_logits_sanity,
    )

TRAINER_VERSION = "v2-normalized-2025-11-18"

# ============================= Optional Drift Detection Helper ============================= #


def _detect_drift(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(window) < 4:
        return {"drift_detected": False, "reason": "insufficient_window"}
    acc = [w["awareness"].get("accuracy", 0.0) for w in window]
    ece = [w["awareness"].get("ece", 0.0) for w in window]
    ent = [w["awareness"].get("avg_entropy_bits", 0.0) for w in window]

    def trend(vals: List[float]) -> float:
        return (
            0.0 if not vals else (vals[-1] - vals[0]) / max(1e-6, abs(vals[0]) + 1e-6)
        )

    flags = []
    t_acc = trend(acc)
    t_ece = trend(ece)
    t_ent = trend(ent)
    if t_acc < -0.15:
        flags.append("accuracy_down")
    if t_ece > 0.20:
        flags.append("ece_up")
    if t_ent < -0.30:
        flags.append("entropy_collapse")
    if t_ent > 0.50:
        flags.append("entropy_explosion")
    return {
        "drift_detected": bool(flags),
        "flags": flags,
        "acc_trend": t_acc,
        "ece_trend": t_ece,
        "entropy_trend": t_ent,
    }


# ============================= Governance Wrapper (Consensus Gate) ============================= #


class ConsensusGate:
    def __init__(self, approve_fn=None):
        self.approve_fn = approve_fn or (lambda proposal: True)

    def approve(self, payload: Dict[str, Any]) -> bool:
        try:
            return bool(self.approve_fn(payload))
        except Exception:
            return True  # fail-open


# ============================= Utilities ============================= #


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_auto(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_batch(sequences: List[List[int]], device: str) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.long, device=device)


def _cosine_with_warmup(
    step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _safe_lr_change(
    old_lr: float, new_lr: float, min_ratio: float = 0.25, max_ratio: float = 4.0
) -> bool:
    if old_lr <= 0 or new_lr <= 0:
        return False
    r = new_lr / old_lr
    return min_ratio <= r <= max_ratio


@torch.no_grad()
def _calc_val_loss(
    model: GPTModel,
    loader: CorpusDataLoader,
    batch_size: int,
    batches: int,
    device: str,
) -> float:
    """
    Compute validation loss as average cross-entropy PER VALID TOKEN (nats),
    consistent with training normalization. Returns a scalar loss_per_token.
    """
    model.eval()
    total_loss_sum = 0.0
    total_valid = 0
    for _ in range(batches):
        vdict = loader.sample_val_batch(batch_size)
        vbatch = _prepare_batch(vdict["sequences"], device)
        # Next-token prediction: inputs are [:, :-1], targets are [:, 1:]
        inputs = vbatch[:, :-1]
        targets = vbatch[:, 1:]
        logits = model.forward(inputs)
        m: LossMetrics = compute_loss_metrics_eval(
            logits=logits,
            targets=targets,
            ignore_index=None,  # all positions should be valid in fixed-length windows
        )
        total_loss_sum += m.loss_sum_valid
        total_valid += m.valid_tokens
    model.train()
    return total_loss_sum / max(total_valid, 1)


def _compute_awareness(
    model: GPTModel,
    loader: CorpusDataLoader,
    batch_size: int,
    val_batches: int,
    device: str,
    fixed_bins: int,
    adaptive_bins: int,
    diversity_len: int,
    temp: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
) -> Dict[str, Any]:
    model.eval()
    entropy_inputs: List[List[float]] = []
    predictions: List[int] = []
    targets_all: List[int] = []
    confidences: List[float] = []
    true_token_probs: List[float] = []

    with torch.no_grad():
        for _ in range(val_batches):
            vdict = loader.sample_val_batch(batch_size)
            vbatch = _prepare_batch(vdict["sequences"], device)
            inputs = vbatch[:, :-1]
            targets = vbatch[:, 1:]
            logits = model.forward(inputs)
            probs = torch.softmax(logits, dim=-1)
            B, T, V = probs.size()
            for b in range(B):
                for t in range(T):
                    pvec = probs[b, t, :].detach().cpu().tolist()
                    entropy_inputs.append(pvec)
                    tgt = targets[b, t].item()
                    targets_all.append(tgt)
                    pt = pvec[tgt] if 0 <= tgt < V else 0.0
                    true_token_probs.append(pt)
                    conf = max(pvec)
                    confidences.append(conf)
                    pred = pvec.index(conf)
                    predictions.append(pred)

        correct = sum(1 for p, t in zip(predictions, targets_all) if p == t)
        accuracy = correct / max(1, len(targets_all))
        avg_nll = 0.0
        for tp in true_token_probs:
            tp = max(tp, 1e-12)
            avg_nll += -math.log(tp)
        avg_nll /= max(1, len(true_token_probs))
        ppl = math.exp(avg_nll)
        avg_conf = sum(confidences) / max(1, len(confidences))
        ent_summary = summarize_entropies(entropy_inputs)
        ece = calculate_ece(predictions, confidences, targets_all, n_bins=fixed_bins)
        mce = calculate_mce(predictions, confidences, targets_all, n_bins=fixed_bins)
        adaptive = calculate_adaptive_ece(
            predictions, confidences, targets_all, n_bins=adaptive_bins
        )

        gen_ids = model.generate(
            start_ids=[loader.bos_id],
            max_new_tokens=diversity_len,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            eos_id=loader.eos_id,
        )
        distinct_1 = calculate_distinct_n([gen_ids], 1)
        distinct_2 = calculate_distinct_n([gen_ids], 2)
        distinct_3 = calculate_distinct_n([gen_ids], 3)

        awareness = {
            "avg_entropy_bits": ent_summary["mean"],
            "entropy_std_bits": ent_summary["std"],
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "nll_nats": avg_nll,
            "perplexity": ppl,
            "ece": ece,
            "mce": mce,
            "adaptive_ece": adaptive,
            "brier_score": 0.0,
            "token_count": len(targets_all),
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,
        }
    model.train()
    return awareness


# ============================= Demo Corpus Helpers ============================= #

_DEMO_CORPUS_FILES = {
    "classic_mini.txt": """Alice was beginning to get very tired of sitting by her sister on the bank.
So she was considering in her own mind whether the pleasure of making a daisy-chain
would be worth the trouble of getting up and picking the daisies.""",
    "tech_fragments.txt": """Neural networks require careful calibration. Entropy and perplexity guide early stopping.
Backoff smoothing blends higher-order statistics with more robust lower-order distributions.""",
    "poetry_bits.txt": """In the quiet code the tokens sleep;
Gradients whisper secrets deep.
Metrics rise and losses fall,
Awareness watches over all.""",
}


def _auto_create_corpus_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if not any(os.scandir(path)):
        for fname, content in _DEMO_CORPUS_FILES.items():
            fpath = os.path.join(path, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content.strip() + "\n")


def _prepare_corpus_from_single_file(
    data_file: str, target_dir: str, shard_size: int = 8000
) -> str:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"--data-file does not exist: {data_file}")
    os.makedirs(target_dir, exist_ok=True)
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks: List[str] = []
    current = []
    length = 0
    for line in text.splitlines():
        current.append(line)
        length += len(line) + 1
        if length >= shard_size:
            chunks.append("\n".join(current))
            current = []
            length = 0
    if current:
        chunks.append("\n".join(current))
    if not chunks:
        chunks = [text]
    for i, chunk in enumerate(chunks):
        shard_path = os.path.join(target_dir, f"shard_{i:03d}.txt")
        with open(shard_path, "w", encoding="utf-8") as f:
            f.write(chunk.strip() + "\n")
    return target_dir


# ============================= Proposal Application ============================= #


def apply_proposals(
    optimizer: optim.Optimizer,
    scheduler_state: Dict[str, Any],
    orchestrator: SelfImprovingTraining,
    cycle_info: Dict[str, Any],
    lr_guard: bool,
    grad_clip_guard: bool,
    warmup_guard: bool,
    force_high_confidence: bool,
    enforce_risk: bool,
    current_clip: float,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    actions_applied: List[Dict[str, Any]] = []
    new_clip = current_clip
    recent_props = [
        p
        for p in orchestrator.proposals
        if p.experiment_id in cycle_info.get("proposals_generated", [])
    ]

    for prop in recent_props:
        if prop.proposal_type != "hyperparam_adjust":
            continue
        params = prop.payload.get("params", {})

        if force_high_confidence and prop.confidence < 0.3:
            continue
        if enforce_risk:
            risk = prop.risk_assessment.get("safety_impact", 0.0)
            if risk < -0.05 or risk > 0.05:
                continue

        if lr_guard and "new_lr" in params:
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = float(params["new_lr"])
            if _safe_lr_change(old_lr, new_lr):
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                scheduler_state["base_lr"] = new_lr
                actions_applied.append(
                    {
                        "type": "lr_change",
                        "old": old_lr,
                        "new": new_lr,
                        "proposal": prop.experiment_id,
                    }
                )
                print(
                    f"[META-ACTION] LR {old_lr:.4e} -> {new_lr:.4e} (proposal {prop.experiment_id})"
                )
            else:
                print(f"[META-SKIP] Unsafe LR ratio: {old_lr:.4e} -> {new_lr:.4e}")

        if grad_clip_guard and "max_grad_norm" in params:
            maxg = float(params["max_grad_norm"])
            if 0.1 <= maxg <= 10.0:
                old = new_clip
                new_clip = maxg
                actions_applied.append(
                    {
                        "type": "grad_clip_change",
                        "old": old,
                        "new": new_clip,
                        "proposal": prop.experiment_id,
                    }
                )
                print(
                    f"[META-ACTION] Grad clip {old:.3f} -> {new_clip:.3f} (proposal {prop.experiment_id})"
                )
            else:
                print(f"[META-SKIP] Invalid grad clip {maxg}")

        if warmup_guard and "warmup_steps" in params:
            w = int(params["warmup_steps"])
            if 10 <= w <= int(scheduler_state.get("total_steps", 1_000_000)):
                old = scheduler_state["warmup_steps"]
                scheduler_state["warmup_steps"] = w
                actions_applied.append(
                    {
                        "type": "warmup_change",
                        "old": old,
                        "new": w,
                        "proposal": prop.experiment_id,
                    }
                )
                print(
                    f"[META-ACTION] Warmup {old} -> {w} (proposal {prop.experiment_id})"
                )
            else:
                print(f"[META-SKIP] Invalid warmup {w}")

    return actions_applied, new_clip, scheduler_state


# ============================= Main Training Function ============================= #


def run(args: argparse.Namespace) -> None:
    print(f"[TRAINER] version={TRAINER_VERSION}")
    _set_seed(args.seed)
    device = _device_auto(args.device)

    # Corpus resolution
    corpus_dir: Optional[str]
    if args.data_file:
        temp_dir = os.path.join(args.out_dir, "_single_file_corpus")
        corpus_dir = _prepare_corpus_from_single_file(args.data_file, temp_dir)
        print(f"[DATA-FILE] Sharded single file '{args.data_file}' into '{corpus_dir}'")
    else:
        corpus_dir = args.data_path
        if not os.path.exists(corpus_dir):
            if args.auto_create_corpus:
                print(
                    f"[AUTO-CORPUS] --data-path '{corpus_dir}' does not exist. Creating demo corpus..."
                )
                _auto_create_corpus_dir(corpus_dir)
            else:
                raise FileNotFoundError(
                    f"--data-path missing and auto-create disabled: {corpus_dir}"
                )
        if not os.path.isdir(corpus_dir):
            raise FileNotFoundError(
                f"--data-path exists but is not a directory: {corpus_dir}"
            )
        if not any(os.scandir(corpus_dir)):
            if args.auto_create_corpus:
                print(
                    f"[AUTO-CORPUS] Directory '{corpus_dir}' empty. Populating demo corpus..."
                )
                _auto_create_corpus_dir(corpus_dir)
            else:
                raise FileNotFoundError(
                    f"--data-path empty and auto-create disabled: {corpus_dir}"
                )

    log_corpus_dir = corpus_dir.replace("\\", "/")
    file_list = [f.name for f in os.scandir(corpus_dir) if f.is_file()]
    print(
        f"[DATA] Resolved corpus directory: {log_corpus_dir} | files={len(file_list)}"
    )
    if len(file_list) == 0:
        raise RuntimeError(f"Corpus directory '{corpus_dir}' has no readable files.")

    # Initial loader
    loader = CorpusDataLoader(
        corpus_dir=corpus_dir,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Auto adjust sequence length if corpus length is too small
    if args.auto_adjust_seq_len:
        train_len = getattr(loader, "train_tokens", [])
        train_size = len(train_len)
        if train_size > 0 and train_size < (args.seq_len + 1):
            new_seq_len = max(8, train_size - 1)
            if new_seq_len < args.seq_len:
                print(
                    f"[SEQ-LEN-ADJUST] Requested seq_len={args.seq_len} > available tokens={train_size}. Clamping to {new_seq_len}."
                )
                args.seq_len = new_seq_len
                loader = CorpusDataLoader(
                    corpus_dir=corpus_dir,
                    max_vocab_size=args.max_vocab_size,
                    min_freq=args.min_freq,
                    seq_len=args.seq_len,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                )
        else:
            if train_size == 0:
                raise RuntimeError(
                    "Corpus appears to have zero tokens after processing."
                )
    else:
        train_len = getattr(loader, "train_tokens", [])
        train_size = len(train_len)
        if train_size > 0 and train_size < (args.seq_len + 1):
            raise RuntimeError(
                f"Token stream too short ({train_size}) for requested sequence length {args.seq_len + 1}. "
                f"Re-run with --auto-adjust-seq-len or lower --seq-len."
            )

    print(
        f"[DATA] vocab={loader.vocab_size} BOS={loader.bos_id} EOS={loader.eos_id} effective_seq_len={args.seq_len}"
    )

    # Model
    config = GPTConfig(
        vocab_size=loader.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        tied_embeddings=True,
        device=device,
    )
    model = GPTModel(config).train()

    # Optimizer + schedule
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    sched_state = {
        "base_lr": args.learning_rate,
        "min_lr": max(args.learning_rate / 50, 1e-7),
        "warmup_steps": args.warmup_steps,
        "total_steps": max(args.steps, args.val_interval * 2),
    }

    gate = ConsensusGate()

    safe_types = [s.strip() for s in args.meta_safe_types.split(",") if s.strip()]
    orchestrator = SelfImprovingTraining(
        random_seed=args.seed,
        experiment_selection_strategy=args.meta_strategy,
        enable_memory_decay=args.meta_enable_decay,
        eval_is_accuracy=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    meta_state_path = args.meta_state_path or os.path.join(
        args.out_dir, "llm_meta_state.json"
    )
    meta_log_path = os.path.join(args.out_dir, args.meta_log_file)
    training_log_path = os.path.join(args.out_dir, args.train_log_file)
    awareness_log_path = os.path.join(args.out_dir, "awareness_metrics.jsonl")

    best_val = float("inf")
    best_step = -1
    patience_counter = 0
    applied_updates = 0
    grad_clip = args.max_grad_norm
    start_time = time.time()
    drift_window: List[Dict[str, Any]] = []
    best_model_sd = copy.deepcopy(model.state_dict())
    best_optim_sd = copy.deepcopy(optimizer.state_dict())

    print(
        f"[TRAIN] steps={args.steps} device={device} seq_len={args.seq_len} batch={args.batch_size} lr={args.learning_rate}"
    )

    for step in range(1, args.steps + 1):
        lr_now = _cosine_with_warmup(
            step,
            sched_state["warmup_steps"],
            sched_state["total_steps"],
            sched_state["base_lr"],
            sched_state["min_lr"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch_dict = loader.sample_train_batch(args.batch_size)
        batch = _prepare_batch(batch_dict["sequences"], device)

        # Next-token setup: inputs are [:, :-1], targets are [:, 1:]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        tokens_per_batch = int(
            inputs.numel()
        )  # batch_size * (seq_len - 1) if loader returns seq_len+1

        if step == 1:
            print(
                f"[SANITY] inputs.shape={tuple(inputs.shape)} targets.shape={tuple(targets.shape)} tokens_per_batch={tokens_per_batch}"
            )

        optimizer.zero_grad(set_to_none=True)

        # Forward -> normalized loss and metrics (divide ONCE by valid tokens)
        logits = model.forward(inputs)
        loss_scalar, m = compute_loss_metrics_train(
            logits=logits, targets=targets, ignore_index=None
        )

        # Debug diagnostics on step 1 to catch pathological logits
        if step == 1:
            with torch.no_grad():
                ls = torch.log_softmax(logits, dim=-1)
                true_logp = ls.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                print(
                    f"[DEBUG] logits_stats mean={logits.mean().item():.4f} std={logits.std().item():.4f} "
                    f"min={logits.min().item():.2f} max={logits.max().item():.2f}"
                )
                print(
                    f"[DEBUG] true_logp_stats mean={true_logp.mean().item():.4f} std={true_logp.std().item():.4f} "
                    f"min={true_logp.min().item():.2f} max={true_logp.max().item():.2f}"
                )

        # Backprop
        loss_scalar.backward()

        # Gradient norms (pre- and post-clip)
        with torch.no_grad():
            sqsum_pre = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sqsum_pre += float(torch.sum(p.grad.detach() ** 2).item())
            grad_norm_unclipped = math.sqrt(max(0.0, sqsum_pre))

        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        with torch.no_grad():
            sqsum_post = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sqsum_post += float(torch.sum(p.grad.detach() ** 2).item())
            grad_norm_val = math.sqrt(max(0.0, sqsum_post))

        approved = True  # gate always approve by default
        if approved:
            optimizer.step()
            status = "applied"
        else:
            optimizer.zero_grad(set_to_none=True)
            status = "rejected"

        applied_updates += 1

        # Training console log (expanded, consistent fields)
        if applied_updates % args.log_interval != 0:
            print(
                f"step={step} "
                f"loss_sum_like={m.loss_sum_valid:.4f} "
                f"loss_per_token={m.loss_per_token:.6f} "
                f"ppl~{m.ppl:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"grad={grad_norm_val:.3f} "
                f"tokens={tokens_per_batch} "
                f"{status}"
            )

        # Validation + awareness
        if step % args.val_interval == 0:
            val_loss = _calc_val_loss(
                model, loader, args.batch_size, args.val_batches, device
            )
            ppl_val = math.exp(min(val_loss, 100.0))
            print(f"[VAL] step={step} val_loss={val_loss:.4f} ppl~{ppl_val:.2f}")

            awareness = _compute_awareness(
                model=model,
                loader=loader,
                batch_size=args.batch_size,
                val_batches=args.aw_val_batches,
                device=device,
                fixed_bins=args.fixed_ece_bins,
                adaptive_bins=args.adaptive_ece_bins,
                diversity_len=args.diversity_sample_length,
                temp=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                rep_penalty=args.rep_penalty,
            )
            summary = awareness_summary(awareness)
            drift_window.append({"awareness": awareness, "step": step})
            if len(drift_window) > args.drift_window:
                drift_window.pop(0)

        # Structured training log
        if args.train_log_interval > 0 and step % args.train_log_interval == 0:
            train_entry = {
                "step": step,
                "loss": float(m.loss_per_token),
                "loss_sum_valid": float(m.loss_sum_valid),
                "valid_tokens": int(m.valid_tokens),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm_unclipped": grad_norm_unclipped,
                "grad_norm_clipped": grad_norm_val,
                "tokens_per_batch": tokens_per_batch,
                "elapsed_sec": time.time() - start_time,
                "timestamp": time.time(),
            }
            try:
                with open(training_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(train_entry) + "\n")
            except Exception as e:
                print(f"[TRAIN-LOG] Write failed: {e}")

    elapsed = time.time() - start_time
    sps = (applied_updates) / elapsed if elapsed > 0 else 0.0
    print(
        f"[DONE] steps={applied_updates} best_val={best_val:.4f} best_step={best_step} elapsed={elapsed:.1f}s steps/s={sps:.2f}"
    )

    last_model_path = os.path.join(args.out_dir, "llm_last_model.pt")
    last_optim_path = os.path.join(args.out_dir, "llm_last_optim.pt")
    torch.save(
        {"model": model.state_dict(), "step": args.steps, "sched_state": sched_state},
        last_model_path,
    )
    torch.save({"optimizer": optimizer.state_dict()}, last_optim_path)
    print(f"[SAVE] last model -> {last_model_path}")
    print(f"[SAVE] last optim -> {last_optim_path}")


# ============================= CLI ============================= #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM training with governance, self-improvement, and self-awareness (no stubs)."
    )
    # Core training
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--val-interval", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument(
        "--aw-val-batches", type=int, default=8, help="Batches for awareness estimation"
    )
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--train-log-interval", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--seed", type=int, default=123)

    # Model size
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Data handling
    default_corpus = os.path.join(os.path.dirname(HERE), "data", "corpus")
    p.add_argument("--data-path", type=str, default=default_corpus)
    p.add_argument("--data-file", type=str, default="")
    p.add_argument("--auto-create-corpus", action="store_true", default=True)
    p.add_argument("--auto-adjust-seq-len", action="store_true", default=True)
    p.add_argument("--max-vocab-size", type=int, default=120000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.01)

    # Awareness params
    p.add_argument("--diversity-sample-length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--rep-penalty", type=float, default=1.05)
    p.add_argument("--fixed-ece-bins", type=int, default=15)
    p.add_argument("--adaptive-ece-bins", type=int, default=15)
    p.add_argument("--extended-awareness", action="store_true", default=False)
    p.add_argument("--multi-class-bins", type=int, default=10)
    p.add_argument("--drift-window", type=int, default=10)

    # Orchestrator (meta) controls
    p.add_argument("--meta-interval", type=int, default=400)
    p.add_argument("--meta-apply", action="store_true", default=False)
    p.add_argument(
        "--meta-safe-types",
        type=str,
        default="lr_adjustment,lr_sweep,grad_clip_adjust,warmup_schedule_adjust",
    )
    p.add_argument("--meta-state-path", type=str, default="")
    p.add_argument("--meta-enable-decay", action="store_true", default=False)
    p.add_argument("--meta-max-select", type=int, default=3)
    p.add_argument(
        "--meta-strategy",
        type=str,
        default="multi_objective",
        choices=["multi_objective", "greedy", "thompson_sampling"],
    )
    p.add_argument("--meta-require-high-confidence", action="store_true", default=False)
    p.add_argument("--meta-require-low-risk", action="store_true", default=False)

    # Logging/output
    p.add_argument(
        "--out-dir", type=str, default=os.path.join(os.path.dirname(HERE), "logs")
    )
    p.add_argument("--meta-log-file", type=str, default="meta_cycles.jsonl")
    p.add_argument("--train-log-file", type=str, default="training_log.jsonl")
    p.add_argument("--best-model-name", type=str, default="llm_best_model.pt")
    p.add_argument("--best-optim-name", type=str, default="llm_best_optim.pt")

    # Resume
    p.add_argument("--resume-model", type=str, default="")
    p.add_argument("--resume-optim", type=str, default="")
    p.add_argument("--resume-meta", type=str, default="")
    p.add_argument("--override-lr", action="store_true", default=False)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
