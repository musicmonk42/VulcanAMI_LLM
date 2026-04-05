"""
intervention_orchestrator.py - High-level intervention orchestration for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Note: This is distinct from intervention_manager.py, which contains the low-level
intervention types (Correlation, InterventionCandidate, InterventionExecutor, etc.).
This module contains the InterventionManager class that orchestrates interventions
at the WorldModel level.

Contains:
- InterventionManager: Schedules and executes interventions using the world model.
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .observation_types import Observation

logger = logging.getLogger(__name__)


class InterventionManager:
    """Manages intervention testing and processing"""

    def __init__(self, world_model):
        self.world_model = world_model
        self.intervention_queue = deque()
        self.intervention_history = deque(maxlen=100)
        self.intervention_count = 0

    def schedule_interventions(
        self, correlations: List[Any], budget: float
    ) -> List[Any]:
        """Schedule interventions based on correlations and budget"""
        # Lazy import to avoid circular dependencies
        from . import world_model_core

        if not world_model_core.INTERVENTION_MANAGER_AVAILABLE:
            logger.error(
                "Cannot schedule interventions - InterventionManager not available"
            )
            return []

        # EXAMINE: Prioritize interventions
        candidates = self.world_model.intervention_prioritizer.prioritize_interventions(
            correlations, budget
        )

        # SELECT: Choose which to execute
        selected = []
        budget_used = 0

        for candidate in candidates:
            # Validate intervention with safety validator
            if self.world_model.safety_validator:
                try:
                    if hasattr(
                        self.world_model.safety_validator, "validate_intervention"
                    ):
                        intervention_validation = (
                            self.world_model.safety_validator.validate_intervention(
                                candidate.correlation.var_a,
                                candidate.correlation.var_b,
                                (
                                    candidate.intervention_type
                                    if hasattr(candidate, "intervention_type")
                                    else "simulated"
                                ),
                                (
                                    candidate.metadata
                                    if hasattr(candidate, "metadata")
                                    else {}
                                ),
                            )
                        )

                        if not intervention_validation.get("safe", True):
                            logger.warning(
                                "Unsafe intervention blocked: %s -> %s: %s",
                                candidate.correlation.var_a,
                                candidate.correlation.var_b,
                                intervention_validation.get("reason", "unknown"),
                            )
                            continue
                except Exception as e:
                    logger.error(
                        "Safety validator error in validate_intervention: %s", e
                    )
                    # Fail-safe: block on validator error
                    continue

            if budget_used + candidate.cost > budget:
                break
            if len(selected) >= self.world_model.max_interventions_per_cycle:
                break
            selected.append(candidate)
            budget_used += candidate.cost

        # APPLY: Queue for execution
        self.intervention_queue.extend(selected)

        return selected

    def execute_next_intervention(self) -> Optional[Any]:
        """Execute the next queued intervention"""
        from . import world_model_core

        if not world_model_core.INTERVENTION_MANAGER_AVAILABLE:
            logger.error(
                "Cannot execute intervention - InterventionManager not available"
            )
            return None

        if not self.intervention_queue:
            return None

        candidate = self.intervention_queue.popleft()

        try:
            # Execute intervention
            result = self.world_model.intervention_executor.execute_intervention(
                candidate
            )

            # Process result
            self._process_intervention_result(candidate, result)

            # REMEMBER: Track intervention
            self.intervention_history.append(
                {
                    "timestamp": time.time(),
                    "correlation": candidate.correlation,
                    "result": result.type,
                    "cost": result.cost_actual,
                }
            )
            self.intervention_count += 1

            return result

        except Exception as e:
            logger.error("Intervention execution failed: %s", e)
            return None

    def process_intervention_observation(
        self, intervention_data: Dict[str, Any], observation: Observation
    ):
        """Process observation from an intervention"""
        from . import world_model_core

        if not world_model_core.CAUSAL_GRAPH_AVAILABLE:
            logger.error(
                "Cannot process intervention observation - CausalDAG not available"
            )
            return

        if "intervened_variable" not in intervention_data:
            return

        cause = intervention_data["intervened_variable"]

        # EXAMINE: Check each effect variable
        causal_edges_added = []

        for effect_var, effect_value in observation.variables.items():
            if effect_var == cause:
                continue

            # Check if effect is significant
            if self._is_significant_effect(
                cause, effect_var, effect_value, intervention_data
            ):
                # Calculate strength
                strength = self._calculate_causal_strength(
                    cause, effect_var, intervention_data, observation
                )

                if strength > self.world_model.min_causal_strength:
                    # Validate causal edge with safety validator
                    if self.world_model.safety_validator:
                        try:
                            if hasattr(
                                self.world_model.safety_validator,
                                "validate_causal_edge",
                            ):
                                edge_validation = self.world_model.safety_validator.validate_causal_edge(
                                    cause, effect_var, strength
                                )
                                if not edge_validation.get("safe", True):
                                    logger.warning(
                                        "Unsafe causal edge blocked: %s -> %s: %s",
                                        cause,
                                        effect_var,
                                        edge_validation.get("reason", "unknown"),
                                    )
                                    continue
                        except Exception as e:
                            logger.error(
                                "Safety validator error in validate_causal_edge: %s", e
                            )
                            continue

                    causal_edges_added.append((cause, effect_var, strength))

        # APPLY: Add causal edges
        with self.world_model.lock:
            for cause, effect, strength in causal_edges_added:
                self.world_model.causal_graph.add_edge(
                    cause, effect, strength=strength, evidence_type="intervention"
                )
                logger.info(
                    "Added causal edge: %s -> %s (strength=%.2f)",
                    cause,
                    effect,
                    strength,
                )

    def _process_intervention_result(
        self, candidate: Any, result: Any
    ):
        """Process result from intervention execution"""
        from . import world_model_core

        if not world_model_core.CAUSAL_GRAPH_AVAILABLE:
            return

        if result.type == "success" and result.causal_strength:
            if result.causal_strength > self.world_model.min_causal_strength:
                # Validate causal edge with safety validator
                if self.world_model.safety_validator:
                    try:
                        if hasattr(
                            self.world_model.safety_validator, "validate_causal_edge"
                        ):
                            edge_validation = (
                                self.world_model.safety_validator.validate_causal_edge(
                                    candidate.correlation.var_a,
                                    candidate.correlation.var_b,
                                    result.causal_strength,
                                )
                            )
                            if not edge_validation.get("safe", True):
                                logger.warning(
                                    "Unsafe causal edge from intervention blocked: %s -> %s: %s",
                                    candidate.correlation.var_a,
                                    candidate.correlation.var_b,
                                    edge_validation.get("reason", "unknown"),
                                )
                                return
                    except Exception as e:
                        logger.error(
                            "Safety validator error in validate_causal_edge: %s", e
                        )
                        return

                with self.world_model.lock:
                    self.world_model.causal_graph.add_edge(
                        candidate.correlation.var_a,
                        candidate.correlation.var_b,
                        strength=result.causal_strength,
                        evidence_type="intervention",
                    )
                logger.info(
                    "Added causal edge from intervention: %s -> %s (strength=%.2f)",
                    candidate.correlation.var_a,
                    candidate.correlation.var_b,
                    result.causal_strength,
                )
        elif result.type == "inconclusive":
            self.world_model.intervention_executor.handle_intervention_failure(
                candidate, result
            )

    def _is_significant_effect(
        self,
        cause: str,
        effect: str,
        effect_value: Any,
        intervention_data: Dict[str, Any],
    ) -> bool:
        """Determine if an effect is statistically significant"""
        from . import world_model_core

        if world_model_core.CorrelationTracker is None:
            return False

        baseline = self.world_model.correlation_tracker.get_baseline(effect)
        if baseline is None:
            return False

        if isinstance(effect_value, (int, float)) and isinstance(
            baseline, (int, float)
        ):
            deviation = abs(effect_value - baseline)
            threshold = self.world_model.correlation_tracker.get_noise_level(effect) * 3
            return deviation > threshold

        return False

    def _calculate_causal_strength(
        self,
        cause: str,
        effect: str,
        intervention_data: Dict[str, Any],
        observation: Observation,
    ) -> float:
        """Calculate strength of causal relationship"""
        from . import world_model_core

        if world_model_core.CorrelationTracker is None:
            return 0.0

        baseline = self.world_model.correlation_tracker.get_baseline(effect)
        actual = observation.variables.get(effect)

        if baseline is None or actual is None:
            return 0.0

        if isinstance(baseline, (int, float)) and isinstance(actual, (int, float)):
            # Normalized effect size
            effect_size = abs(actual - baseline) / (abs(baseline) + 1e-10)

            # Adjust by confidence
            confidence = observation.confidence

            # Calculate final strength
            strength = min(1.0, effect_size * confidence)

            return strength

        return 0.0
