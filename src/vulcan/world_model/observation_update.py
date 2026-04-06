"""
observation_update.py - Observation processing and world model updates.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

from .observation_types import Observation
from . import world_model_core as _core

logger = logging.getLogger(__name__)


def process_observation(wm, observation: Observation, constraints=None):
    """Main entrypoint from the rest of the system."""
    if not wm.router:
        logger.error("Router not available. Cannot process observation.")
        return {"plan": None, "results": "Router unavailable, observation dropped."}

    plan = wm.router.route(observation, constraints or {})
    results = wm.router.execute(plan)
    return {"plan": plan, "results": results}


def update_from_observation(wm, observation: Observation) -> Dict[str, Any]:
    """
    Update world model from new observation
    FIXED: Refactored locking to prevent deadlock with router.
    INTEGRATED: Schema validation for observations
    ENHANCED: Performance tracking for observation processing
    """
    from src.utils.performance_metrics import PerformanceTimer

    with PerformanceTimer("world_model_update", "observation_processing"):
        start_time = time.time()

        # --- Part 1: Validation and Planning (Locked) ---
        with wm.lock:
            # Schema validation (if enabled)
            validation_errors = []
            if wm.validate_observations and wm.schema_registry:
                try:
                    # Convert observation to dict for validation
                    obs_dict = {
                        "timestamp": observation.timestamp,
                        "domain": observation.domain,
                        "variables": observation.variables,
                        "confidence": observation.confidence,
                    }
                    if observation.metadata:
                        obs_dict["metadata"] = observation.metadata

                    # Validate against observation schema
                    validation_result = wm.schema_registry.validate(obs_dict, "observation")
                    if not validation_result.valid:
                        validation_errors = [err.to_dict() for err in validation_result.errors]
                        logger.warning(
                            f"Observation schema validation failed: {len(validation_errors)} error(s). "
                            "Processing will continue but data quality may be compromised."
                        )
                        for err in validation_result.errors[:3]:  # Log first 3 errors
                            logger.debug(f"  - {err.message} at {err.path}")
                except Exception as e:
                    logger.error(f"Schema validation error: {e}", exc_info=True)

            # EXAMINE: Validate and analyze observation
            is_valid, error_msg = wm.observation_processor.validate_observation(
                observation
            )
            if not is_valid:
                logger.warning("Invalid observation: %s", error_msg)
                return {"status": "rejected", "reason": error_msg}

            # Extract components
            variables = wm.observation_processor.extract_variables(observation)
            intervention_data = wm.observation_processor.detect_intervention_data(
                observation
            )
            temporal_patterns = wm.observation_processor.extract_temporal_patterns(
                observation
            )

            # --- START NEW LINGUISTIC PROCESSING ---
            linguistic_data = wm.observation_processor.extract_linguistic_data(
                observation
            )
            if linguistic_data:
                wm.update_from_text(linguistic_data, {})  # Use the new method
            # --- END NEW LINGUISTIC PROCESSING ---

            # SELECT: Use router to determine which updates to run
            if wm.router:
                constraints = {"time_budget_ms": 1000, "priority_threshold": "normal"}
                update_plan = wm.router.route(observation, constraints)
            else:
                # Fallback: run all updates sequentially
                update_plan = None
                execution_results = sequential_update(wm, observation)

        # --- Part 2: Execution (Unlocked) ---
        # The router executes its own updates, which manage their own locks.
        # This MUST be called outside the main lock to prevent deadlock.
        if wm.router and update_plan:
            try:
                execution_results = wm.router.execute(update_plan)
            except Exception as e:
                logger.error(f"Router execution failed: {e}", exc_info=True)
                return {"status": "error", "reason": f"Router execution failed: {e}"}

        # --- Part 3: Finalization (Locked) ---
        with wm.lock:
            # Bootstrap mode: check for testable correlations
            if wm.bootstrap_mode and _core.INTERVENTION_MANAGER_AVAILABLE:
                wm._check_bootstrap_opportunities()

            # REMEMBER: Update state and validate periodically
            wm.observation_count += 1
            wm.last_observation_time = observation.timestamp

            # Periodic validation
            validation_result = wm.consistency_validator.validate_if_needed()

        # Prepare response
        execution_time = (time.time() - start_time) * 1000

        response = {
            "status": "success",
            "variables_extracted": len(variables),
            "patterns_detected": len(temporal_patterns.get("trends", {})),
            "intervention_processed": intervention_data is not None,
            "updates_executed": execution_results.get("updates_executed", []),
            "execution_time_ms": execution_time,
            "validation": validation_result,
            "safety_checks": wm.safety_mode,
            "meta_reasoning_enabled": wm.meta_reasoning_enabled,
            "self_improvement_enabled": wm.self_improvement_enabled,
        }

        # Add schema validation results if applicable
        if wm.validate_observations and validation_errors:
            response["schema_validation"] = {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
            }

        return response


def update_from_text(wm, text: str, predictions: Dict[str, Any]):
    """
    Update world model from language observations
    NEW METHOD for linguistic observations.
    """
    if not _core.CAUSAL_GRAPH_AVAILABLE or wm.dynamics is None:
        logger.warning(
            "Causal DAG or Dynamics Model unavailable, skipping update from text."
        )
        return

    with wm.lock:
        wm.linguistic_observations.append(
            {"timestamp": time.time(), "text": text, "predictions": predictions}
        )

        # Extract causal relationships from text (MOCK)
        causal_relations = extract_causal_relations(wm, text)

        # Update causal DAG
        for rel in causal_relations:
            # Mock structure for extracted relation
            cause = rel.get("cause", "unknown_cause")
            effect = rel.get("effect", "unknown_effect")
            strength = rel.get("strength", 0.5)

            if wm.safety_validator:
                try:
                    if hasattr(wm.safety_validator, "validate_causal_edge"):
                        edge_validation = (
                            wm.safety_validator.validate_causal_edge(
                                cause, effect, strength
                            )
                        )
                        if not edge_validation.get("safe", True):
                            logger.warning(
                                "Unsafe causal edge from text blocked: %s -> %s",
                                cause,
                                effect,
                            )
                            continue
                except Exception as e:
                    logger.error(
                        "Safety validator error in validate_causal_edge (text): %s",
                        e,
                    )
                    continue

            wm.causal_graph.add_edge(
                cause, effect, strength=strength, evidence_type="linguistic"
            )
            logger.info(
                f"Linguistic update: Added edge {cause} -> {effect} (strength={strength:.2f})"
            )

        # Update dynamics tracker prediction accuracy (MOCK)
        if wm.dynamics and hasattr(wm.dynamics, "update"):
            # Pass a simplified observation for dynamics update
            mock_obs = MagicMock(spec=Observation)
            mock_obs.variables = predictions
            mock_obs.timestamp = time.time()
            wm.dynamics.update(mock_obs)  # Update dynamics model


def validate_generation(wm, proposed_token: str, context: Dict[str, Any]) -> bool:
    """
    Check if language generation (proposed token/action) violates the causal model.
    NEW METHOD for linguistic generation validation.
    """
    if not _core.CAUSAL_GRAPH_AVAILABLE:
        return True  # Fail-safe to allow generation if model is missing

    with wm.lock:
        # MOCK IMPLEMENTATION: Check if this token would create impossible causal chain
        if would_create_contradiction(wm, proposed_token, context):
            logger.warning(
                f"Generation blocked: Proposed token '{proposed_token}' creates causal contradiction."
            )
            return False

        # MOCK: Check against known safety invariants related to generation
        if wm.invariants and wm.invariants.check_invariant_violations(
            {"generated_token": proposed_token, **context}
        ):
            logger.warning(
                f"Generation blocked: Proposed token '{proposed_token}' violates an invariant."
            )
            return False

        return True


def extract_causal_relations(wm, text: str) -> List[Dict[str, Any]]:
    """MOCK: Placeholder for NLP/LLM-based causal relation extraction"""
    if "causes" in text.lower():
        return [
            {
                "cause": "text_mention",
                "effect": "hypothesized_effect",
                "strength": 0.65,
            },
            {
                "cause": "linguistic_topic",
                "effect": "sentiment_variable",
                "strength": 0.8,
            },
        ]
    return []


def would_create_contradiction(
    wm, proposed_token: str, context: Dict[str, Any]
) -> bool:
    """MOCK: Placeholder for checking causal cycle/contradiction"""
    if proposed_token.lower() == "error" and wm.causal_graph.has_edge("A", "B"):
        return True
    return False


def sequential_update(wm, observation: Observation) -> Dict[str, Any]:
    """Fallback sequential update when router unavailable"""

    updates_executed = []

    # Update correlation tracker
    if wm.correlation_tracker:
        wm.correlation_tracker.update(observation)
        updates_executed.append("correlation")

    # Update dynamics model
    if wm.dynamics:
        wm.dynamics.update(observation)
        updates_executed.append("dynamics")

    # Update confidence tracker
    if wm.confidence_tracker:
        wm.confidence_tracker.update(observation=observation)
        updates_executed.append("confidence")

    # Detect invariants
    if wm.invariant_detector:
        wm.invariant_detector.check([observation])
        updates_executed.append("invariants")

    return {"status": "sequential", "updates_executed": updates_executed}
