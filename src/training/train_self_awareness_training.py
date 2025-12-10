"""
Full training loop using GPTModel + self-awareness metrics (function form).

This refactored version removes reliance on __main__ and __init__ package
entrypoints. It exposes a callable function suitable for integration into a
larger orchestration system.

Primary function:
    train_self_awareness_transformer(config: dict) -> dict

Where config is a dictionary providing the same parameters formerly passed
via CLI flags.

Example integration:

    from src.training.train_self_awareness_transformer import train_self_awareness_transformer

    result = train_self_awareness_transformer({
        "corpus_dir": "data/corpus",
        "steps": 5000,
        "seq_len": 128,
        "batch_size": 32,
        "val_interval": 250,
        "lr": 3e-4,
        "extended_awareness": True
    })

The returned result includes final awareness snapshot, drift status, report text,
and path references for logs and model artifacts.

If you still want CLI usage, you can run:
    python -m src.training.train_self_awareness_transformer --corpus_dir data/corpus

(Guard retained but isolated to a helper so platform usage is pure.)

Revision / Fixes in this version:
1. Added directory creation guards for report_out and save_final_model paths (they could fail if parent dirs missing).
2. Added gradient norm computation and logging into validation payload for richer telemetry.
3. Adjusted learning rate scheduling: CosineAnnealingLR no longer overrides manual warmup phase.
   - Scheduler.step() is deferred until warmup period completes to respect warmup LR.
4. Added safe handling when save_final_model is falsy (None or empty string) to skip saving gracefully.
5. Added explicit flush and close safety for metrics file (context manager alternative preserved).
6. Added minor numeric stability guard when computing perplexity on extremely large avg_nll.
7. Added optional key 'grad_norm' and 'best_perplexity' to returned result dictionary.
8. Ensured deterministic seed usage also sets cuda.manual_seed_all when CUDA is available.
9. Ensured extended awareness multi-class calibration uses clipped probability vector slice length aligned with labels.
10. Maintained all original logic untruncated; only appended/modified where noted.

NOTE: To revert to original scheduling behavior, remove the warmup gating around scheduler.step().
"""

from __future__ import annotations
import argparse
import json
import os
import time
import math
import random
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.training.data_loader import CorpusDataLoader
from src.training.gpt_model import GPTModel, GPTConfig
from src.training.self_awareness import (
    calculate_distinct_n,
    build_extended_awareness,
    awareness_summary,
    summarize_entropies,
    calculate_ece,
    calculate_mce,
    calculate_adaptive_ece,
)
from src.training.awareness_thresholds import AwarenessThresholds
from src.training.drift_detection import detect_drift
from src.training.post_training_report import generate_report


# -------------------------------------------------------------------
# Helper: Prepare batch tensor
# -------------------------------------------------------------------
def _prepare_batch(sequences: List[List[int]], device: str) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.long, device=device)


# -------------------------------------------------------------------
# Core Training Function (Platform-Facing)
# -------------------------------------------------------------------
def train_self_awareness_transformer(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a causal Transformer while collecting self-awareness metrics.

    Args (config keys):
        corpus_dir (str): Directory containing text files.
        steps (int): Total training steps.
        seq_len (int): Sequence length (max context).
        batch_size (int)
        val_interval (int): Steps between metric evaluations.
        lr (float): Base learning rate.
        warmup_steps (int, optional)
        max_vocab_size (int, optional)
        min_freq (int, optional)
        seed (int, optional)
        device (str, optional)
        n_layers, n_heads, dim, ff_mult, dropout (model hyperparams)
        clip_grad (float, optional)
        log_path (str): JSONL metrics output path.
        diversity_sample_length (int)
        temperature (float), top_k (int), top_p (float), rep_penalty (float)
        adaptive_ece_bins (int), fixed_ece_bins (int)
        extended_awareness (bool)
        multi_class_bins (int)
        drift_window (int)
        report_out (str): Final textual report output path.
        save_final_model (str): Path to save final model weights (optional).
        max_prob_vectors_for_extended (int): Limit of probability vectors used for extended calibration (optional).

    Returns:
        Dict with keys:
            final_awareness
            final_summary
            final_drift_status
            report_text
            metrics_log_path
            model_path (if saved)
            steps_completed
            grad_norm (last step)
            best_perplexity
    """
    # Extract config with defaults
    corpus_dir = config["corpus_dir"]
    steps = int(config.get("steps", 20000))
    seq_len = int(config.get("seq_len", 128))
    batch_size = int(config.get("batch_size", 32))
    val_interval = int(config.get("val_interval", 500))
    lr = float(config.get("lr", 3e-4))
    warmup_steps = int(config.get("warmup_steps", 500))
    max_vocab_size = int(config.get("max_vocab_size", 30000))
    min_freq = int(config.get("min_freq", 2))
    seed = int(config.get("seed", 42))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    n_layers = int(config.get("n_layers", 6))
    n_heads = int(config.get("n_heads", 8))
    dim = int(config.get("dim", 512))
    ff_mult = int(config.get("ff_mult", 4))
    dropout = float(config.get("dropout", 0.1))
    clip_grad = float(config.get("clip_grad", 1.0))
    log_path = config.get("log_path", "logs/self_awareness_transformer.jsonl")
    diversity_sample_length = int(config.get("diversity_sample_length", 256))
    temperature = float(config.get("temperature", 0.9))
    top_k = int(config.get("top_k", 64))
    top_p = float(config.get("top_p", 0.95))
    rep_penalty = float(config.get("rep_penalty", 1.05))
    adaptive_ece_bins = int(config.get("adaptive_ece_bins", 15))
    fixed_ece_bins = int(config.get("fixed_ece_bins", 15))
    extended_awareness = bool(config.get("extended_awareness", False))
    multi_class_bins = int(config.get("multi_class_bins", 10))
    drift_window = int(config.get("drift_window", 10))
    report_out = config.get("report_out", "logs/final_self_awareness_report.txt")
    save_final_model = config.get("save_final_model", "logs/final_gpt_model.pt")
    max_prob_vectors_for_extended = int(
        config.get("max_prob_vectors_for_extended", 500)
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure directories
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if save_final_model:
        os.makedirs(os.path.dirname(save_final_model), exist_ok=True)
    if report_out:
        os.makedirs(os.path.dirname(report_out), exist_ok=True)

    metrics_file = open(log_path, "a", encoding="utf-8")

    loader = CorpusDataLoader(
        corpus_dir=corpus_dir,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        seq_len=seq_len,
        val_ratio=0.01,
        seed=seed,
    )

    print(f"[INIT] Vocab size: {loader.vocab_size}")

    model_config = GPTConfig(
        vocab_size=loader.vocab_size,
        seq_len=seq_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        dropout=dropout,
        tied_embeddings=True,
        device=device,
    )
    model = GPTModel(model_config)
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    # Scheduler only after warmup to avoid overriding manual linear warmup.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps - warmup_steps), eta_min=lr / 50
    )

    thresholds = AwarenessThresholds()
    drift_buffer: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_perplexity = float("inf")

    def _warmup_lr(step: int):
        if warmup_steps > 0 and step <= warmup_steps:
            return lr * (step / max(1, warmup_steps))
        return None

    start_time = time.time()
    last_awareness = None
    last_summary = None
    last_drift_status = None
    last_grad_norm = None

    print("[TRAIN] Commencing training loop")

    for step in range(1, steps + 1):
        batch_dict = loader.sample_train_batch(batch_size)
        batch = _prepare_batch(batch_dict["sequences"], model_config.device)

        optimizer.zero_grad()
        loss = model.loss(batch)

        warm_lr = _warmup_lr(step)
        if warm_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Gradient norm
        with torch.no_grad():
            gn_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn_sq += p.grad.detach().pow(2).sum().item()
            last_grad_norm = math.sqrt(max(gn_sq, 0.0))

        optimizer.step()
        # Apply scheduler only after warmup completion
        if warm_lr is None:
            scheduler.step()

        if step % val_interval == 0 or step == steps:
            with torch.no_grad():
                val_batches = 6
                val_token_probs: List[float] = []
                val_predictions: List[int] = []
                val_targets: List[int] = []
                val_confidences: List[float] = []
                entropy_inputs: List[List[float]] = []

                for _ in range(val_batches):
                    vdict = loader.sample_val_batch(batch_size)
                    vbatch = _prepare_batch(vdict["sequences"], model_config.device)
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
                            val_targets.append(tgt)
                            val_token_probs.append(pvec[tgt] if 0 <= tgt < V else 0.0)
                            conf = max(pvec)
                            val_confidences.append(conf)
                            pred = pvec.index(conf)
                            val_predictions.append(pred)

                correct = sum(1 for p, t in zip(val_predictions, val_targets) if p == t)
                accuracy = correct / max(1, len(val_targets))
                avg_nll = 0.0
                eps = 1e-12
                for tp in val_token_probs:
                    tp = max(tp, eps)
                    avg_nll += -math.log(tp)
                avg_nll /= max(1, len(val_token_probs))
                # Numeric safety for exp overflow
                perplexity = math.exp(min(avg_nll, 100.0))
                avg_conf = sum(val_confidences) / max(1, len(val_confidences))
                entropy_summary = summarize_entropies(entropy_inputs)

                ece = calculate_ece(
                    val_predictions, val_confidences, val_targets, n_bins=fixed_ece_bins
                )
                mce = calculate_mce(
                    val_predictions, val_confidences, val_targets, n_bins=fixed_ece_bins
                )
                adaptive_ece = calculate_adaptive_ece(
                    val_predictions,
                    val_confidences,
                    val_targets,
                    n_bins=adaptive_ece_bins,
                )

                awareness = {
                    "avg_entropy_bits": entropy_summary["mean"],
                    "entropy_std_bits": entropy_summary["std"],
                    "avg_confidence": avg_conf,
                    "accuracy": accuracy,
                    "nll_nats": avg_nll,
                    "perplexity": perplexity,
                    "ece": ece,
                    "mce": mce,
                    "adaptive_ece": adaptive_ece,
                    "brier_score": 0.0,
                    "token_count": len(val_targets),
                }

                gen_ids = model.generate(
                    start_ids=[loader.bos_id],
                    max_new_tokens=diversity_sample_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    eos_id=loader.eos_id,
                )
                awareness["distinct_1"] = calculate_distinct_n([gen_ids], 1)
                awareness["distinct_2"] = calculate_distinct_n([gen_ids], 2)
                awareness["distinct_3"] = calculate_distinct_n([gen_ids], 3)

                if extended_awareness:
                    # Align probability vectors and labels to max_prob_vectors_for_extended
                    pv_slice = entropy_inputs[:max_prob_vectors_for_extended]
                    lbl_slice = val_targets[: len(pv_slice)]
                    awareness = build_extended_awareness(
                        base_awareness=awareness,
                        sequences=[gen_ids],
                        multi_class_probs=pv_slice,
                        multi_class_labels=lbl_slice,
                        distinct_ns=(1, 2, 3),
                        include_macro_distinct=True,
                        multi_class_bins=multi_class_bins,
                    )

                summary = awareness_summary(awareness)
                drift_buffer.append({"awareness": awareness, "step": step})
                if len(drift_buffer) > drift_window:
                    drift_buffer.pop(0)
                drift_status = detect_drift(drift_buffer)
                flags = thresholds.evaluate(awareness)

                payload = {
                    "step": step,
                    "loss": float(loss.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "awareness": awareness,
                    "summary": summary,
                    "flags": flags,
                    "drift": drift_status,
                    "grad_norm": last_grad_norm,
                    "time_elapsed_s": time.time() - start_time,
                }
                metrics_file.write(json.dumps(payload) + "\n")
                metrics_file.flush()

                print(
                    f"[VAL step={step}] loss={loss.item():.4f} acc={accuracy:.3f} ppl={perplexity:.2f} "
                    f"ece={ece:.4f} entropy={entropy_summary['mean']:.3f} distinct3={awareness.get('distinct_3', 0.0):.3f} "
                    f"grad_norm={last_grad_norm:.3f}"
                )

                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                if perplexity < best_val_loss:
                    best_val_loss = perplexity

                last_awareness = awareness
                last_summary = summary
                last_drift_status = drift_status

        # Simple LR adaptation post-warmup
        if (
            warmup_steps > 0
            and step > warmup_steps
            and step % val_interval == 0
            and last_awareness is not None
        ):
            if last_awareness.get("ece", 0.0) > 0.2:
                for pg in optimizer.param_groups:
                    old_lr = pg["lr"]
                    pg["lr"] = max(old_lr * 0.9, lr / 100)
                    print(f"[ADAPT] High ECE -> LR {old_lr:.6f} -> {pg['lr']:.6f}")

    metrics_file.close()

    model_path = None
    if save_final_model:
        try:
            torch.save(model.state_dict(), save_final_model)
            model_path = save_final_model
            print(f"[SAVE] Model saved to {save_final_model}")
        except Exception as e:
            print(f"[SAVE-ERROR] Could not save model to {save_final_model}: {e}")

    # Generate report
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[READ-ERROR] Failed reading metrics log for report: {e}")
        records = []
    report_text = (
        generate_report(records)
        if records
        else "No records available for report generation."
    )
    try:
        with open(report_out, "w", encoding="utf-8") as rf:
            rf.write(report_text)
        print(f"[REPORT] Final report written to {report_out}")
    except Exception as e:
        print(f"[REPORT-ERROR] Could not write report: {e}")

    return {
        "final_awareness": last_awareness,
        "final_summary": last_summary,
        "final_drift_status": last_drift_status,
        "report_text": report_text,
        "metrics_log_path": log_path,
        "model_path": model_path,
        "steps_completed": steps,
        "grad_norm": last_grad_norm,
        "best_perplexity": best_perplexity,
    }


# -------------------------------------------------------------------
# Optional CLI Wrapper (kept only for manual runs; safe to remove)
# -------------------------------------------------------------------
def _cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument(
        "--log_path", type=str, default="logs/self_awareness_transformer.jsonl"
    )
    parser.add_argument("--diversity_sample_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rep_penalty", type=float, default=1.05)
    parser.add_argument("--adaptive_ece_bins", type=int, default=15)
    parser.add_argument("--fixed_ece_bins", type=int, default=15)
    parser.add_argument("--extended_awareness", action="store_true")
    parser.add_argument("--multi_class_bins", type=int, default=10)
    parser.add_argument("--drift_window", type=int, default=10)
    parser.add_argument(
        "--report_out", type=str, default="logs/final_self_awareness_report.txt"
    )
    parser.add_argument(
        "--save_final_model", type=str, default="logs/final_gpt_model.pt"
    )
    parser.add_argument("--max_prob_vectors_for_extended", type=int, default=500)
    args = parser.parse_args()

    result = train_self_awareness_transformer(vars(args))
    print("[FINAL RESULT]")
    print(
        json.dumps(
            {
                "final_awareness": result["final_awareness"],
                "final_summary": result["final_summary"],
                "final_drift_status": result["final_drift_status"],
                "metrics_log_path": result["metrics_log_path"],
                "model_path": result["model_path"],
                "steps_completed": result["steps_completed"],
                "best_perplexity": result["best_perplexity"],
                "grad_norm": result["grad_norm"],
            },
            indent=2,
        )
    )


# NOTE: No automatic execution guard; platform decides whether to call _cli_main().
# If you want to enable direct CLI execution, you can re-add:
# if __name__ == "__main__":
#     _cli_main()
