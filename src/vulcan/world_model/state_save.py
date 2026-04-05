"""
state_save.py - Save world model state to disk.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import json
import logging
import time
from pathlib import Path as FilePath

from . import world_model_core as _core

logger = logging.getLogger(__name__)


def save_state(wm, path: str):
    """Save world model state to disk"""

    # Note: Use the 'FilePath' alias for pathlib.Path
    save_path = FilePath(path)
    save_path.mkdir(parents=True, exist_ok=True)

    state = {
        "model_version": wm.model_version,
        "observation_count": wm.observation_count,
        "intervention_count": wm.intervention_manager.intervention_count,
        "causal_structure": wm.get_causal_structure(),
        "config": {
            "min_correlation_strength": wm.min_correlation_strength,
            "min_causal_strength": wm.min_causal_strength,
            "bootstrap_mode": wm.bootstrap_mode,
            "meta_reasoning_enabled": wm.meta_reasoning_enabled,
            "self_improvement_enabled": wm.self_improvement_enabled,
        },
        "component_availability": {
            "causal_graph": _core.CAUSAL_GRAPH_AVAILABLE,
            "correlation_tracker": _core.CorrelationTracker is not None,
            "intervention_manager": _core.INTERVENTION_MANAGER_AVAILABLE,
            "prediction_engine": _core.PREDICTION_ENGINE_AVAILABLE,
            "dynamics_model": _core.DynamicsModel is not None,
            "invariant_detector": _core.INVARIANT_DETECTOR_AVAILABLE,
            "confidence_calibrator": _core.CONFIDENCE_CALIBRATOR_AVAILABLE,
            "router": _core.ROUTER_AVAILABLE,
            "meta_reasoning": _core.META_REASONING_AVAILABLE,
            "self_improvement": _core.META_REASONING_AVAILABLE
            and _core.SelfImprovementDrive is not None,
            "safety_validator": _core.EnhancedSafetyValidator is not None,
        },
    }

    # Add versioning and make write atomic
    state["version"] = getattr(wm, "model_version", 1.0)
    state["timestamp"] = time.time()
    temp_path = save_path / "world_model_state.tmp.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    temp_path.rename(save_path / "world_model_state.json")

    # Save router state
    if wm.router:
        wm.router.save_state(str(save_path))

    # Save safety validator state if available
    if wm.safety_validator:
        try:
            safety_state = {}
            if hasattr(wm.safety_validator, "get_safety_stats"):
                safety_state["safety_stats"] = (
                    wm.safety_validator.get_safety_stats()
                )
            if hasattr(wm.safety_validator, "constraint_manager"):
                if hasattr(
                    wm.safety_validator.constraint_manager, "get_constraint_stats"
                ):
                    safety_state["constraint_stats"] = (
                        wm.safety_validator.constraint_manager.get_constraint_stats()
                    )

            with open(save_path / "safety_state.json", "w", encoding="utf-8") as f:
                json.dump(safety_state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Error saving safety state: %s", e)

    # Save meta-reasoning state if available
    if wm.meta_reasoning_enabled:
        try:
            meta_reasoning_state = {
                "objectives": wm.motivational_introspection.explain_motivation_structure(),
                "statistics": wm.motivational_introspection.get_statistics(),
                "validation_history_size": len(
                    wm.motivational_introspection.validation_history
                ),
            }

            with open(
                save_path / "meta_reasoning_state.json", "w", encoding="utf-8"
            ) as f:
                json.dump(meta_reasoning_state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Error saving meta-reasoning state: %s", e)

    # Save self-improvement state (handled by the drive itself)
    if wm.self_improvement_enabled:
        try:
            # The drive saves its own state automatically
            logger.info("Self-improvement drive state saved by drive itself")
        except Exception as e:
            logger.error("Error with self-improvement state: %s", e)

    logger.info("World model state saved to %s", save_path)
    wm.model_version += 0.1
