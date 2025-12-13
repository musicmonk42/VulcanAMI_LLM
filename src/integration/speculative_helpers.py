from __future__ import annotations

import asyncio
import logging
import random  # Needed for random token choice on low entropy fallback
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Initialize logger
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the appropriate device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)

# CRITICAL ENHANCEMENT: Add KL_THRESHOLD for the guard
KL_THRESHOLD = 0.5


@dataclass
class SpeculativeStats:
    """Statistics tracking for speculative decoding performance."""

    drafted: int = 0
    accepted: int = 0
    rejected_early: int = 0
    total_kl: float = 0.0  # E. Enhancement: Track total KL for avg calculation
    # Store per-token KL divergences for detailed analysis
    kl_divergences: List[float] = field(default_factory=list)
    total_steps: int = 0
    max_lookahead: int = 0

    # C. Enhancement: Track final sequence lengths for batch output
    final_lengths: List[int] = field(default_factory=list)

    # ENHANCEMENT: Add rejection reason
    rejection_reason: str = ""

    @property
    def efficiency_gain(self) -> float:
        """
        Efficiency gain calculated as (Total Accepted Tokens) / (Total Tokens Drafted).
        A proxy for the probability of acceptance.
        """
        return self.accepted / self.drafted if self.drafted > 0 else 0.0

    # E. Enhancement: Derived metrics
    @property
    def acceptance_rate(self) -> float:
        """Rate of drafted tokens that were accepted."""
        return float(self.accepted) / max(1, self.drafted)

    @property
    def avg_kl(self) -> float:
        """Average KL divergence over all drafted tokens."""
        # Use total_kl / count of kl_divergences for a more accurate average of measured KL events
        return float(self.total_kl) / max(1, len(self.kl_divergences))


class LowRankDraftTransformer(nn.Module):
    """
    Real LoRA-style lightweight draft model.
    Projects full hidden states -> low-dim -> reconstruct logits.
    """

    def __init__(
        self,
        parent: nn.Module,
        rank: int = 16,
        shrink_factor: float = 0.5,
        entropy_threshold: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.parent = parent
        self.shrink = shrink_factor
        self.rank = rank
        self.entropy_threshold = entropy_threshold
        self._logger = logger or log

        if not hasattr(parent, "config"):
            raise ValueError("Parent model must have a 'config' attribute.")

        hidden_size = parent.config.hidden_size
        draft_size = int(hidden_size * shrink_factor)
        self.vocab_size = parent.config.vocab_size

        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, draft_size, bias=False)
        self.lm_head = nn.Linear(draft_size, self.vocab_size, bias=False)

        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)

        # Optimization: Apply torch.compile if available
        if torch.__version__ >= "2.0":
            try:
                self.forward = torch.compile(self.forward)
            except Exception as e:
                logger.debug(f"Operation failed: {e}")

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Runs the main model's forward pass (without gradient) and compresses the hidden state.
        Returns: compressed low-rank hidden state for the *last* token.
        """
        # B. Enhancement: Handle batching / ensure device
        if input_ids.dim() == 1:  # ENHANCEMENT: Batch encode if 1D
            input_ids = input_ids.unsqueeze(0)

        assert input_ids.device == next(self.parent.parameters()).device, (
            "Input and parent model device mismatch in encode."
        )

        self.parent.eval()
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.float16,
            enabled=(input_ids.device.type == "cuda"),
        ):
            parent_output = self.parent(input_ids, output_hidden_states=True)
            if hasattr(parent_output, "hidden_states"):
                hidden = parent_output.hidden_states[-1]
            else:
                hidden = parent_output[0]

        # Select the hidden state of the last token in the sequence
        last_hidden = hidden[:, -1, :]
        return self.down_proj(last_hidden)

    def get_logits(
        self, low_rank_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Uses the compressed hidden state to predict logits.
        Returns: Logits for the next token (shape [B, V])
        """
        expanded = self.up_proj(low_rank_hidden)
        expanded = F.gelu(expanded)
        return self.lm_head(expanded)  # [B, V]

    # FIX: Implement full training loop
    def train_distill(
        self, dataloader, target_model: nn.Module, scheduler=None, epochs: int = 1
    ):
        """
        Distillation training method for the Low-Rank Draft Model.
        Uses Mean Squared Error (MSE) loss on logits against the full target model.
        """
        device = next(self.parameters()).device
        self.to(device)
        self.train()
        target_model.to(device)
        target_model.eval()

        # ENHANCEMENT: Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        n_steps = 0

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    # 1. Get Target Logits from the Full Model (Teacher)
                    with torch.no_grad():
                        target_output = target_model(input_ids)
                        target_logits = target_output.logits[:, :-1, :]  # (B, S-1, V)

                    # 2. Get Draft Logits from the Low-Rank Model (Student)
                    # NOTE: This assumes the parent model call can run the full forward pass to get hidden states
                    parent_output = self.parent(input_ids, output_hidden_states=True)
                    hidden_states = parent_output.hidden_states[-1]  # (B, S, H)

                    low_rank_hidden_all = self.down_proj(hidden_states)  # (B, S, rank)
                    expanded_all = self.up_proj(
                        low_rank_hidden_all
                    )  # (B, S, draft_size)
                    expanded_all = F.gelu(expanded_all)
                    draft_logits_all = self.lm_head(expanded_all)  # (B, S, V)

                    draft_logits = draft_logits_all[:, :-1, :]  # (B, S-1, V)

                    # 3. Compute Distillation Loss (MSE on Logits)
                    loss = F.mse_loss(draft_logits, target_logits)

                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

                if n_steps % 100 == 0:
                    self._logger.info(
                        f"Epoch {epoch}, Step {n_steps}, MSE Loss: {loss.item():.4f}"
                    )

                n_steps += 1
                # ENHANCEMENT: Limit total steps if needed, here we let it run based on epochs/dataloader
                # if n_steps >= 1000:
                #     return

    @torch.no_grad()
    def check_entropy_and_fallback(
        self, logits: torch.Tensor, entropy_threshold: Optional[float] = None
    ) -> bool:
        """
        Checks the entropy of the draft model's logit distribution.
        If entropy is too low (i.e., model is overconfident), suggests fallback.
        """
        threshold = (
            entropy_threshold
            if entropy_threshold is not None
            else self.entropy_threshold
        )

        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Entropy H = - sum(p * log_p) over the vocabulary (dim=-1)
        entropy_per_sample = -(p * log_p).sum(dim=-1)

        min_entropy = entropy_per_sample.min().item()

        if min_entropy < threshold:
            self._logger.warning(
                "Min draft logit entropy (%.4f) < threshold (%.4f); suggesting fallback.",
                min_entropy,
                threshold,
            )
            return True

        return False


def build_draft_transformer(
    main_transformer: Any,
    shrink_factor: float = 0.5,
    rank: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    entropy_threshold: float = 0.1,
) -> LowRankDraftTransformer:
    """
    Build a real, trainable LoRA draft model.
    """
    if not hasattr(main_transformer, "config"):
        raise ValueError("main_transformer must have .config")

    draft = LowRankDraftTransformer(
        main_transformer,
        rank=rank,
        shrink_factor=shrink_factor,
        entropy_threshold=entropy_threshold,
    )
    draft.to(device)
    draft.eval()
    return draft


# --------------------- CRITICAL ENHANCEMENT: SPECULATIVE DECODING LOOP ---------------------


@torch.no_grad()
def speculative_sampling_and_verify(
    main_model: nn.Module,
    draft_model: LowRankDraftTransformer,
    input_ids: torch.Tensor,
    stats: SpeculativeStats,
    max_lookahead: int = 5,
    kl_threshold: float = KL_THRESHOLD,
    temperature: float = 1.0,
    entropy_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, SpeculativeStats]:
    """
    Performs one step of speculative decoding: drafts tokens, verifies them with
    the main model, and performs the acceptance sampling logic.

    Returns:
        The updated input_ids (padded sequence with accepted tokens) and updated stats.
    """
    # ENHANCEMENT: Handle edges (empty input)
    if input_ids.shape[-1] == 0:
        stats.rejection_reason = "EmptyInput"
        # Return empty sequence, None next token (represented by a dummy tensor)
        return input_ids, stats

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    B, L = input_ids.shape
    device = input_ids.device

    # FIX 1: Remove repeated .to(device) calls
    # assert input_ids.device == next(main_model.parameters()).device, "Input and main model device mismatch."
    # assert input_ids.device == next(draft_model.parameters()).device, "Input and draft model device mismatch."
    # We assume models are already on the correct device via setup (e.g., build_draft_transformer)

    stats.final_lengths = []

    # OPTIMIZATION: Use AMP (autocast) for mixed precision
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
    ):
        # 0. Entropy Check (Implementation in speculative_sampling_and_verify)
        low_rank_hidden_init = draft_model.encode(input_ids)
        draft_logits_init = draft_model.get_logits(low_rank_hidden_init, input_ids)

        # Compute entropy for the initial draft logits
        probs_init = F.softmax(draft_logits_init, dim=-1)
        entropy_init = (
            -torch.sum(probs_init * torch.log(probs_init + 1e-10), dim=-1).mean().item()
        )

        if entropy_init < (entropy_threshold or draft_model.entropy_threshold):
            # Fallback due to low entropy (overconfidence)
            stats.rejection_reason = "LowEntropy"

            full_output_init = main_model(input_ids)
            main_logits_init = full_output_init.logits[:, -1, :]  # [B, V]

            # Sample next token from main model's distribution (not just greedy)
            main_p = F.softmax(main_logits_init / temperature, dim=-1)
            next_token = torch.multinomial(main_p, num_samples=1)  # [B, 1]

            stats.total_steps += 1
            stats.accepted += B
            stats.drafted += B
            stats.final_lengths = [L + 1] * B

            return torch.cat([input_ids, next_token], dim=1), stats

        # 1. Draft Phase: Generate speculative tokens
        current_ids = input_ids.clone()
        drafted_tokens_batch: List[torch.Tensor] = []
        draft_logits_list: List[torch.Tensor] = []

        stats.total_steps += 1

        # ENHANCEMENT: Ensure len(draft_logits_list) == max_lookahead
        for i in range(max_lookahead):
            low_rank_hidden = draft_model.encode(current_ids)
            draft_logits = draft_model.get_logits(low_rank_hidden, current_ids)
            draft_logits_list.append(draft_logits)

            p_draft = F.softmax(draft_logits / temperature, dim=-1)
            next_token = torch.multinomial(p_draft, num_samples=1).squeeze(-1)

            drafted_tokens_batch.append(next_token)
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            stats.drafted += B

        drafted_tokens = torch.stack(drafted_tokens_batch, dim=1)  # [B, K]
        K = max_lookahead

        # 2. Verification Phase
        full_output = main_model(current_ids.detach())
        main_logits_seq = full_output.logits[:, L:]

        # 3. Acceptance/Rejection Phase (Vectorized for B)
        acceptance_indices = K * torch.ones(B, dtype=torch.long, device=device)
        accepted_count_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        for k in range(K):
            main_logits_k = main_logits_seq[:, k]  # [B, V]
            p_main = F.softmax(main_logits_k / temperature, dim=-1)  # [B, V]

            draft_logits_k = draft_logits_list[k]  # [B, V]
            q_draft = F.softmax(draft_logits_k / temperature, dim=-1)  # [B, V]

            drafted_token_ids_k = drafted_tokens[:, k].unsqueeze(1)  # [B, 1]

            p_draft_token = q_draft.gather(dim=-1, index=drafted_token_ids_k).squeeze(
                -1
            )  # [B]
            p_main_token = p_main.gather(dim=-1, index=drafted_token_ids_k).squeeze(
                -1
            )  # [B]

            rejected_at_k = p_main_token < p_draft_token

            update_mask = (acceptance_indices == K) & rejected_at_k
            acceptance_indices[update_mask] = k

            accepted_count_per_sample += (acceptance_indices == K).long()

        total_accepted = accepted_count_per_sample.sum().item()
        stats.accepted += total_accepted

        stats.max_lookahead = max(stats.max_lookahead, acceptance_indices.min().item())
        stats.rejected_early += (K * B) - total_accepted

        # 4. Final Token Sampling and KL Guard
        new_input_ids_list: List[torch.Tensor] = []

        for i in range(B):
            acceptance_idx = acceptance_indices[i].item()
            accepted_len_i = acceptance_idx

            # --- Per-Token KL Guard Check ---
            if accepted_len_i > 0:
                main_logits_accepted = main_logits_seq[i, :accepted_len_i]
                draft_logits_accepted = torch.stack(draft_logits_list[:accepted_len_i])[
                    :, i, :
                ]

                kl_div_per_token = F.kl_div(
                    F.log_softmax(draft_logits_accepted, dim=-1),
                    F.softmax(main_logits_accepted, dim=-1),
                    reduction="none",
                    log_target=False,
                ).sum(dim=-1)

                # ENHANCEMENT: Update total KL
                stats.total_kl += kl_div_per_token.sum().item()  # FIX 2: Accumulate KL
                stats.kl_divergences.extend(kl_div_per_token.cpu().tolist())

                if (kl_div_per_token > kl_threshold).any():
                    stats.rejection_reason = "KLGuard"
                    stats.rejected_early += acceptance_idx - 1
                    stats.accepted -= acceptance_idx - 1
                    accepted_len_i = 1

            # --- Sample the next token (Residual Sampling) ---
            sample_dist_idx = accepted_len_i  # Index for the distribution to sample from (L + accepted_len_i)

            # Logits for the rejection point (L + accepted_len_i)
            sample_logits = (
                main_logits_seq[i, sample_dist_idx, :]
                if sample_dist_idx < K
                else full_output.logits[i, L + K - 1, :]
            )

            p_main_k = F.softmax(sample_logits / temperature, dim=-1)

            # Draft logits for the position *after* the last accepted token
            q_draft_k = (
                F.softmax(draft_logits_list[sample_dist_idx][i] / temperature, dim=-1)
                if sample_dist_idx < K
                else torch.zeros_like(p_main_k)
            )

            p_res = torch.clamp(p_main_k - q_draft_k, min=0.0)
            Z = p_res.sum()

            if Z > 1e-6:
                p_res /= Z
            else:
                p_res = p_main_k

            next_token_i = torch.multinomial(p_res, num_samples=1).squeeze(-1)

            # --- Finalize Sequence ---
            accepted_tokens_i = drafted_tokens[i, :accepted_len_i]

            final_sequence_i = torch.cat(
                [input_ids[i], accepted_tokens_i, next_token_i.unsqueeze(0)], dim=0
            )
            new_input_ids_list.append(final_sequence_i)

            stats.final_lengths.append(final_sequence_i.shape[0])

        # Pad to max length in the batch
        max_len = max(stats.final_lengths)
        padded_sequences = [
            F.pad(seq, (0, max_len - seq.shape[0]), value=0)
            for seq in new_input_ids_list
        ]

        return torch.stack(padded_sequences), stats


# Helper function to maintain train/distill method consistency with the original class structure
LowRankDraftTransformer.train = LowRankDraftTransformer.train_distill


# D. Enhancement: Add async wrapper for integration with CognitiveLoop
async def speculative_sampling_and_verify_async(
    main_model: nn.Module,
    draft_model: LowRankDraftTransformer,
    input_ids: torch.Tensor,
    stats: SpeculativeStats,
    max_lookahead: int = 5,
    kl_threshold: float = KL_THRESHOLD,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, SpeculativeStats]:
    """
    Asynchronous wrapper for speculative_sampling_and_verify, running the synchronous
    PyTorch logic in a thread pool executor.
    """
    loop = asyncio.get_running_loop()

    # We use a lambda to capture the arguments for execution in the thread
    return await loop.run_in_executor(
        None,
        lambda: speculative_sampling_and_verify(
            main_model,
            draft_model,
            input_ids,
            stats,
            max_lookahead=max_lookahead,
            kl_threshold=kl_threshold,
            temperature=temperature,
        ),
    )
