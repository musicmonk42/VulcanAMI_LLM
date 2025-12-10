"""
Patched Minimal causal Transformer (GPT-like) for token-level language modeling (Fully Expanded, Untruncated).

Purpose:
- Provide a non-stub neural model for integrating self-awareness metrics and adaptive training.
- Lightweight single-file implementation: pure PyTorch, easily extended.
- This patched version adds configurable loss reduction (mean/sum), optional label smoothing,
  explicit per-token loss return, perplexity helper, gradient-friendly utilities, improved generation
  controls, consistent device handling, and explicit stable parameter initialization to prevent
  pathological logits at startup.

Architecture:
- Token embedding + positional embedding
- N-layer stack: (Multi-Head Self-Attention + FeedForward), Pre-LN residual architecture
- Optional weight tying between input embeddings and output projection layer
- Dynamic causal mask caching per sequence length (no fixed size limit)

Generation Features:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Repetition penalty (logarithmic subtractive approach)
- Greedy fallback if temperature <= 1e-8 or if top_k == 1
- Context cropping to config.seq_len (positional embedding limit)
- Safe guards against NaNs in probability distributions

Training:
- Next-token prediction objective
- Configurable reduction mode: 'mean' or 'sum'
- Optional label smoothing
- Returns a dictionary from loss() if requested, with:
  {
    'loss': scalar tensor,
    'avg_loss_per_token': float,
    'perplexity': float,
    'total_tokens': int
  }
  plus ability to just return scalar for backward compatibility
- Supports automatic mixed precision usage outside (no internal AMP dependency)
- Clear error if input sequence exceeds configured seq_len

Patch Additions / Revisions:
1. Added GPTConfig fields: loss_reduction, label_smoothing, return_loss_dict to control loss behavior.
2. Added enforce_safe_softmax to guard against numerical instability after filtering.
3. Added generate(..., greedy_threshold=...) parameter; if temperature <= greedy_threshold, uses argmax.
4. Added method `compute_logits(inputs)` separated from forward() for clarity (forward delegates).
5. Added helper `prepare_inputs` to centralize positional embedding logic.
6. Added perplexity computation inside loss when using reduction='mean' (if reduction='sum' perplexity is based on averaged raw loss).
7. Provided `infer_perplexity_from_avg_loss(avg_loss)` static method for external usage.
8. Added explicit device property and `to_device()` method for chaining.
9. Kept all previous functionality fully intact (no truncation).
10. Added explicit docstrings for clarity at each public method.
11. Provided an optional `return_dict` parameter in loss() to maintain backward compatibility with existing training script.
12. CRITICAL: Added explicit, GPT-2–style stable parameter initialization (Normal(0, 0.02) for Linear/Embedding,
    zeros biases, pos_emb std=0.01) to prevent extreme logits and astronomically large perplexities at startup.

NOTE:
- If you had previously observed astronomically large perplexity values, it was likely due to treating a summed loss
  as if it were already averaged and/or unstable default initializations accumulating through residual paths.
  Set loss_reduction='mean' for conventional per-token average cross entropy and use this file’s initialization.
- Label smoothing > 0.0 modifies targets distribution; set label_smoothing=0.05 for minor smoothing.

Usage Example:

    from gpt_model import GPTModel, GPTConfig
    import torch

    config = GPTConfig(
        vocab_size=5000,
        seq_len=256,
        dim=512,
        n_layers=6,
        n_heads=8,
        ff_mult=4,
        dropout=0.1,
        tied_embeddings=True,
        loss_reduction="mean",
        label_smoothing=0.0,
        return_loss_dict=True,
        device="cuda"
    )
    model = GPTModel(config)
    batch = torch.randint(0, config.vocab_size, (16, 257), device=config.device)  # (B, T+1)
    out = model.loss(batch)  # returns dict since return_loss_dict=True

    print(out['loss'], out['avg_loss_per_token'], out['perplexity'])

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================
# GPT Configuration Dataclass
# ======================================


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    tied_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Patch additions:
    loss_reduction: str = "mean"  # 'mean' or 'sum'
    label_smoothing: float = 0.0  # 0.0 disables smoothing
    return_loss_dict: bool = False  # If True, model.loss returns dict instead of scalar
    enforce_safe_softmax: bool = True  # Guard against NaNs in generation softmax
    generation_pad_token_id: Optional[int] = (
        None  # Optional pad token id for future use
    )


# ======================================
# Multi-Head Self-Attention
# ======================================


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Dynamic causal mask cache; key by (T, device_str) to avoid cross-device reuse
        self._mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Stable initialization for attention projections
        # Use GPT-2 style Normal(0, 0.02) and zero biases to prevent large logits at init
        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns (1, 1, T, T) lower-triangular mask of booleans (True for allowed positions), cached by (T, device).
        """
        key = (T, str(device))
        if key not in self._mask_cache:
            # Boolean mask; True where allowed (lower triangle)
            mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
            self._mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, T, T)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, h, T, T)
        causal = self._get_causal_mask(T, x.device)
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))  # causal masking

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # (B, h, T, d)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


# ======================================
# FeedForward Block
# ======================================


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        # Stable initialization for FFN Linear layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================
# Transformer Block (Pre-LN)
# ======================================


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, ff_mult: int, dropout: float, ln_eps: float
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim, eps=ln_eps)
        self.ff = FeedForward(dim, ff_mult, dropout)

        # LayerNorm defaults are already stable (weight=1.0, bias=0.0).
        # No change needed unless customizing.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ======================================
# GPT Model
# ======================================


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                    ln_eps=config.layer_norm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

        if config.tied_embeddings:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
            self.head.weight = self.token_emb.weight  # weight tying
        else:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=True)
            # Stable initialization if not tied
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.head.bias)

        # Stable embedding initialization (GPT-2 style)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

        # Move to device
        self.to(config.device)

    # ---------------------------
    # Device helpers
    # ---------------------------
    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    def to_device(self, device: Union[str, torch.device]) -> "GPTModel":
        self.config.device = str(device)
        return self.to(device)

    # ---------------------------
    # Positional Embedding Application
    # ---------------------------
    def prepare_inputs(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Prepare token + positional embeddings for input IDs tensor of shape (B, T).
        Enforces sequence length constraint.
        """
        B, T = idx.size()
        if T > self.config.seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds configured maximum {self.config.seq_len}. "
                f"Increase config.seq_len before model initialization."
            )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.token_emb(idx)  # (B, T, C)
        posv = self.pos_emb(pos).unsqueeze(0).expand(B, T, -1)
        return self.drop(tok + posv)

    # ---------------------------
    # Forward (Logits Computation)
    # ---------------------------
    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from already prepared embedding tensor x of shape (B, T, C).
        """
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from token IDs -> logits.
        """
        x = self.prepare_inputs(idx)
        return self.compute_logits(x)

    # ---------------------------
    # Loss Function
    # ---------------------------
    def loss(
        self, batch: torch.Tensor, return_dict: Optional[bool] = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute next-token prediction loss.

        Args:
            batch: Tensor of shape (B, T+1) containing token IDs.
                   Inputs are batch[:, :-1]; targets are batch[:, 1:].
            return_dict: If True (or config.return_loss_dict), returns detailed dict.
                         Otherwise returns just scalar loss tensor.

        Behavior:
            - If config.loss_reduction == 'mean': uses averaged cross entropy across all tokens.
            - If 'sum': uses summed cross entropy; avg_loss_per_token is then scaled by total token count.
            - Label smoothing applied if config.label_smoothing > 0.0

        Returns:
            Scalar loss tensor if return_dict False,
            else dictionary:
                {
                  'loss': scalar tensor (suitable for backward),
                  'avg_loss_per_token': float,
                  'perplexity': float,
                  'total_tokens': int,
                  'reduction': str
                }
        """
        if return_dict is None:
            return_dict = self.config.return_loss_dict

        inputs = batch[:, :-1]  # (B, T)
        targets = batch[:, 1:]  # (B, T)
        logits = self.forward(inputs)  # (B, T, vocab)
        B, T, V = logits.size()
        total_tokens = B * T

        # Flatten for CE
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.reshape(B * T)

        # Label smoothing
        if self.config.label_smoothing > 0.0:
            eps = self.config.label_smoothing
            # Convert targets to one-hot
            with torch.no_grad():
                true_dist = torch.zeros_like(logits_flat)
                true_dist.fill_(eps / (V - 1))
                true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - eps)
            if self.config.loss_reduction == "mean":
                ce = (
                    -(true_dist * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1).mean()
                )
            elif self.config.loss_reduction == "sum":
                ce = -(true_dist * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1).sum()
            else:
                raise ValueError(f"Invalid loss_reduction {self.config.loss_reduction}")
        else:
            # Standard CE
            if self.config.loss_reduction == "mean":
                ce = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
            elif self.config.loss_reduction == "sum":
                ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
            else:
                raise ValueError(f"Invalid loss_reduction {self.config.loss_reduction}")

        if not return_dict:
            return ce

        if self.config.loss_reduction == "mean":
            avg_loss_per_token = float(ce.item())
        else:
            # Sum reduction -> divide by total tokens for avg metric
            avg_loss_per_token = float(ce.item()) / max(1, total_tokens)

        perplexity = math.exp(min(avg_loss_per_token, 100.0))  # safe guard

        return {
            "loss": ce,
            "avg_loss_per_token": avg_loss_per_token,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "reduction": self.config.loss_reduction,
        }

    @staticmethod
    def infer_perplexity_from_avg_loss(avg_loss: float) -> float:
        """
        Given an average loss per token, return perplexity.
        """
        return math.exp(min(avg_loss, 100.0))

    # ---------------------------
    # Generation / Sampling
    # ---------------------------
    @torch.no_grad()
    def generate(
        self,
        start_ids: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: Optional[int] = None,
        greedy_threshold: float = 1e-8,
    ) -> List[int]:
        """
        Autoregressive token generation.

        Args:
            start_ids: Initial context tokens.
            max_new_tokens: Maximum number of new tokens to append.
            temperature: Softmax temperature (<= greedy_threshold triggers greedy argmax).
            top_k: Keep only the top_k logits (0 disables top-k).
            top_p: Nucleus sampling threshold (if <1.0).
            repetition_penalty: >1.0 penalizes tokens already generated (logarithmic subtract).
            eos_id: If provided, stops when generated token equals eos_id.
            greedy_threshold: If temperature <= this value, greedy sampling is used.

        Notes:
            - Context is cropped to last config.seq_len tokens.
            - Position embedding length is fixed at initialization; increase config.seq_len for longer contexts.
        """
        device = self.device
        generated = list(start_ids)

        for _ in range(max_new_tokens):
            ctx = generated[-self.config.seq_len :]
            idx = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
            logits = self.forward(idx)[:, -1, :]  # (1, vocab)

            # Temperature / Greedy
            if temperature <= greedy_threshold:
                # Greedy argmax ignoring penalties
                if repetition_penalty > 1.0 and generated:
                    # Apply repetition penalty before argmax
                    counts: Dict[int, int] = {}
                    for t in generated:
                        counts[t] = counts.get(t, 0) + 1
                    penalty_log = math.log(repetition_penalty)
                    logits = logits.clone()
                    for token, c in counts.items():
                        logits[0, token] -= penalty_log * c
                next_id = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break
                continue

            logits = logits / max(temperature, 1e-8)

            # Repetition penalty (logarithmic subtract)
            if repetition_penalty > 1.0 and generated:
                counts: Dict[int, int] = {}
                for t in generated:
                    counts[t] = counts.get(t, 0) + 1
                penalty_log = math.log(repetition_penalty)
                logits = logits.clone()
                for token, c in counts.items():
                    logits[0, token] -= penalty_log * c

            # Top-k filtering
            if top_k > 0 and top_k < logits.size(1):
                kth_vals, kth_idx = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, kth_idx, kth_vals)
                logits = mask

            # Top-p filtering (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                cutoff_mask = cumulative > top_p
                if torch.any(cutoff_mask):
                    first_exceed = torch.where(cutoff_mask)[1].min()
                    sorted_logits[:, first_exceed + 1 :] = float("-inf")
                # Scatter back
                new_logits = torch.full_like(logits, float("-inf"))
                new_logits.scatter_(1, sorted_indices, sorted_logits)
                logits = new_logits

            # Final probabilities
            probs = F.softmax(logits, dim=-1)
            if self.config.enforce_safe_softmax:
                if torch.isnan(probs).any():
                    probs = torch.nan_to_num(probs, nan=0.0)
                    s = float(probs.sum().item())
                    if s <= 0.0:
                        # Fallback to uniform
                        probs = torch.ones_like(probs) / probs.size(1)
                    else:
                        probs = probs / s

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return generated

    # ---------------------------
    # Utility for manual perplexity on arbitrary logits & targets
    # ---------------------------
    @staticmethod
    def compute_token_level_nll(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token negative log-likelihood given logits and target IDs (no reduction).
        logits: (B, T, V)
        targets: (B, T)
        Returns: (B, T) tensor of NLL values.
        """
        B, T, V = logits.size()
        log_probs = F.log_softmax(logits, dim=-1)
        flat = log_probs.view(B * T, V)
        tgt_flat = targets.view(B * T)
        idx = torch.arange(B * T, device=logits.device)
        nll_flat = -flat[idx, tgt_flat]
        return nll_flat.view(B, T)

    @staticmethod
    def compute_perplexity_from_logits(
        logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> float:
        """
        Convenience: compute perplexity directly from logits and targets.
        reduction: 'mean' averages over all tokens; 'sum' sums then divides by total tokens.
        """
        nll = GPTModel.compute_token_level_nll(logits, targets)
        if reduction == "sum":
            avg_nll = float(nll.sum().item()) / max(1, nll.numel())
        else:
            avg_nll = float(nll.mean().item())
        return math.exp(min(avg_nll, 100.0))


# End of patched gpt_model.py (fully unabridged)
