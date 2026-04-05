"""
Tool Selection Bandit - Adaptive bandit-based tool selection learning.

Integrates the AdaptiveBanditOrchestrator for online learning of
tool selection policies based on execution feedback.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import pickle  # SECURITY: Internal data only, never deserialize untrusted data
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Optional dependency: contextual bandit ---
try:
    from ..contextual_bandit import (
        AdaptiveBanditOrchestrator,
        BanditAction,
        BanditContext,
        BanditFeedback,
    )

    BANDIT_AVAILABLE = True
    logger.info("Contextual bandit imported successfully")
except ImportError as e:
    logger.warning(f"Contextual bandit not available: {e}")
    BANDIT_AVAILABLE = False
    # Create placeholders
    AdaptiveBanditOrchestrator = None
    BanditContext = None
    BanditFeedback = None
    BanditAction = None

# ==============================================================================
# Learning Reward Penalties
# ==============================================================================
# Penalty factor for unverified high-confidence results.
# Prevents learning from potentially wrong but confident answers.
UNVERIFIED_QUALITY_PENALTY = 0.7  # Reduce to 70% of claimed confidence

# Penalty factor for fallback results.
# Heavily penalizes fallback paths to prevent reinforcing failures.
FALLBACK_QUALITY_PENALTY = 0.3  # Reduce to 30% of quality


class ToolSelectionBandit:
    """
    Integrates the full AdaptiveBanditOrchestrator for tool selection learning.
    This replaces the minimal stub interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.is_enabled = BANDIT_AVAILABLE
        config = config or {}
        # Tool names registered with the bandit learning system
        # FIX #2: Added 'philosophical' and 'mathematical' tools (previous change)
        # FIX #3: Added 'world_model' for meta-cognitive self-introspection (this change)
        # Without registration, bandit updates fail with "Unknown tool name 'X' in bandit update"
        self.tool_names = [
            "symbolic",
            "probabilistic",
            "causal",
            "analogical",
            "multimodal",
            "philosophical",  # FIX #2: Register philosophical reasoning tool
            "mathematical",   # FIX #2: Register mathematical reasoning tool
            "world_model",    # FIX #3: Register world_model for meta-cognitive self-introspection
        ]

        # **************************************************************************
        # START CRITICAL FIX: Add lock for thread-safe updates to prevent crash
        self.update_lock = threading.RLock()
        # END CRITICAL FIX
        # **************************************************************************

        # CRITICAL FIX: Add fallback attributes for when bandit is disabled
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.statistics = {}

        if not self.is_enabled:
            logger.warning(
                "ToolSelectionBandit is disabled; contextual_bandit module not found."
            )
            self.orchestrator = None
            return

        feature_dim = config.get("feature_dim", 128)
        num_tools = len(self.tool_names)

        # Instantiate the full bandit orchestrator
        self.orchestrator = AdaptiveBanditOrchestrator(
            n_actions=num_tools, context_dim=feature_dim
        )

    def select_tool(self, features: np.ndarray, constraints: Dict[str, float]) -> str:
        """Select a tool using the adaptive bandit orchestrator."""
        if not self.is_enabled:
            # Use deterministic fallback instead of random selection.
            # Random selection causes non-deterministic results and inconsistent tool health.
            # Default to "probabilistic" as a reasonable general-purpose fallback.
            logger.info("[ToolSelectionBandit] Using deterministic fallback: probabilistic")
            return "probabilistic"

        context = BanditContext(
            features=features, problem_type="tool_selection", constraints=constraints
        )
        action = self.orchestrator.select_action(context)
        return action.tool_name

    def update_from_execution(
        self,
        features: np.ndarray,
        tool_name: str,
        quality: float,
        time_ms: float,
        energy_mj: float,
        constraints: Dict[str, float],
        is_verified: bool = False,
        is_fallback: bool = False,
    ):
        """
        Update the bandit orchestrator with execution results.

        Args:
            features: Feature vector for the context.
            tool_name: Name of the tool that was executed.
            quality: Quality score of the result (0-1).
            time_ms: Execution time in milliseconds.
            energy_mj: Energy consumption in millijoules.
            constraints: Constraint dictionary for context.
            is_verified: If True, the result was mathematically verified as correct.
                Unverified results receive reduced rewards.
            is_fallback: If True, the result came from a fallback mechanism.
                Fallback results receive heavily reduced rewards.
        """

        # **************************************************************************
        # START CRITICAL FIX: Wrap entire method in lock to prevent race conditions
        with self.update_lock:
            if not self.is_enabled:
                # Update fallback statistics even when disabled
                if tool_name not in self.statistics:
                    self.statistics[tool_name] = {"pulls": 0, "rewards": []}
                self.statistics[tool_name]["pulls"] += 1
                reward = self._compute_reward(
                    quality, time_ms, energy_mj, constraints,
                    is_verified=is_verified, is_fallback=is_fallback
                )
                self.statistics[tool_name]["rewards"].append(reward)
                return

            try:
                # 1. Compute reward from the outcome (with verification and fallback status)
                reward = self._compute_reward(
                    quality, time_ms, energy_mj, constraints,
                    is_verified=is_verified, is_fallback=is_fallback
                )

                # 2. Create the context and action objects
                context = BanditContext(
                    features=features,
                    problem_type="tool_selection",
                    constraints=constraints,
                )
                try:
                    action_id = self.tool_names.index(tool_name)
                except ValueError:
                    logger.error(f"Unknown tool name '{tool_name}' in bandit update.")
                    return

                # A full implementation would log the probability from the active policy at selection time.
                # Here we use a simplification for the update.
                action = BanditAction(
                    tool_name=tool_name,
                    action_id=action_id,
                    expected_reward=0,
                    probability=1.0 / len(self.tool_names),
                )

                # 3. Create the feedback object
                feedback = BanditFeedback(
                    context=context,
                    action=action,
                    reward=reward,
                    execution_time=time_ms,
                    energy_used=energy_mj,
                    success=quality > constraints.get("min_confidence", 0.5),
                )

                # 4. Update the orchestrator (now thread-safe)
                self.orchestrator.update(feedback)
            except Exception as e:
                # Add error handling for robustness
                logger.error(f"Error during bandit update: {e}", exc_info=True)
        # END CRITICAL FIX
        # **************************************************************************

    def _compute_reward(
        self,
        quality: float,
        time_ms: float,
        energy_mj: float,
        constraints: Dict[str, float],
        is_verified: bool = False,
        is_fallback: bool = False,
    ) -> float:
        """
        Computes a reward score between 0 and 1.

        Considers whether the answer was verified as correct before rewarding.
        Unverified answers with high confidence should NOT receive full reward -
        confidence != correctness. Fallback results receive reduced rewards to
        prevent learning from potentially incorrect LLM responses.

        Args:
            quality: Confidence score from the tool (0-1)
            time_ms: Execution time in milliseconds
            energy_mj: Energy used in millijoules
            constraints: Dict with time_budget_ms, energy_budget_mj
            is_verified: Whether the result was mathematically verified
            is_fallback: Whether this result came from a fallback mechanism

        Returns:
            Reward score between 0 and 1
        """
        time_budget = constraints.get("time_budget_ms", 1000)
        energy_budget = constraints.get("energy_budget_mj", 1000)

        time_score = max(0, 1 - (time_ms / time_budget))
        energy_score = max(0, 1 - (energy_mj / energy_budget))

        # Penalize unverified high-confidence results.
        # If result is not verified, reduce the effective quality score
        # to prevent learning from potentially wrong but confident answers.
        effective_quality = quality
        if not is_verified and quality > 0.7:
            # Reduce confidence for unverified high-confidence answers
            effective_quality = quality * UNVERIFIED_QUALITY_PENALTY
            logger.debug(
                f"[ToolSelector] Reduced reward for unverified high-confidence result: "
                f"{quality:.2f} -> {effective_quality:.2f}"
            )

        # Significantly reduce reward for fallback results.
        # Fallback typically means primary engine failed, so we shouldn't
        # strongly reinforce this path.
        if is_fallback:
            effective_quality = effective_quality * FALLBACK_QUALITY_PENALTY
            logger.debug(
                f"[ToolSelector] Reduced reward for fallback result: "
                f"{quality:.2f} -> {effective_quality:.2f}"
            )

        # Weighted combination, prioritizing quality
        reward = 0.6 * effective_quality + 0.3 * time_score + 0.1 * energy_score
        return float(np.clip(reward, 0.0, 1.0))

    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_enabled:
            return {
                "status": "disabled",
                "reason": "contextual_bandit module not found",
                "exploration_rate": self.exploration_rate,
                "arm_stats": self.statistics,
            }
        return self.orchestrator.get_statistics()

    def save_model(self, path: str):
        if self.is_enabled and self.orchestrator:
            self.orchestrator.save_model(path)
        else:
            # CRITICAL FIX: Save fallback statistics when disabled
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(
                Path(path, encoding="utf-8") / "bandit_statistics.pkl", "wb"
            ) as f:
                pickle.dump(self.statistics, f)

    def load_model(self, path: str):
        if self.is_enabled and self.orchestrator:
            self.orchestrator.load_model(path)
        else:
            # CRITICAL FIX: Load fallback statistics when disabled
            stats_path = Path(path) / "bandit_statistics.pkl"
            if stats_path.exists():
                with open(stats_path, "rb") as f:
                    self.statistics = pickle.load(
                        f
                    )  # nosec B301 - Internal data structure

    def increase_exploration(self):
        """Increase exploration rate (delegated)."""
        if not self.is_enabled:
            # CRITICAL FIX: Update exploration_rate even when disabled
            self.exploration_rate = min(0.3, self.exploration_rate * 1.5)
            return
        # This function would need to be implemented in the AdaptiveBanditOrchestrator
        # For now, it's a placeholder call.
        logger.info("Increasing exploration rate for bandit.")
