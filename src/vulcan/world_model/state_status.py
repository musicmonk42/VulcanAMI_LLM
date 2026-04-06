"""
state_status.py - System status reporting for World Model.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

from typing import Any, Dict

from . import world_model_core as _core


def get_system_status(wm) -> Dict[str, Any]:
    """Get comprehensive system status"""

    return {
        "model_version": wm.model_version,
        "observation_count": wm.observation_count,
        "last_observation_time": wm.last_observation_time,
        "intervention_count": wm.intervention_manager.intervention_count,
        "safety_mode": wm.safety_mode,
        "bootstrap_mode": wm.bootstrap_mode,
        "meta_reasoning_enabled": wm.meta_reasoning_enabled,
        "full_meta_reasoning_enabled": getattr(wm, "full_meta_reasoning_enabled", False),
        "self_improvement_enabled": wm.self_improvement_enabled,
        "improvement_running": (
            wm.improvement_running if wm.self_improvement_enabled else False
        ),
        "components": {
            "causal_graph": {
                "available": _core.CAUSAL_GRAPH_AVAILABLE,
                "nodes": (
                    len(wm.causal_graph.nodes) if _core.CAUSAL_GRAPH_AVAILABLE else 0
                ),
                "edges": (
                    len(wm.causal_graph.edges) if _core.CAUSAL_GRAPH_AVAILABLE else 0
                ),
            },
            "correlation_tracker": {"available": _core.CorrelationTracker is not None},
            "intervention_manager": {
                "available": _core.INTERVENTION_MANAGER_AVAILABLE,
                "queued": len(wm.intervention_manager.intervention_queue),
            },
            "prediction_engine": {"available": _core.PREDICTION_ENGINE_AVAILABLE},
            "dynamics_model": {"available": _core.DynamicsModel is not None},
            "invariant_detector": {"available": _core.INVARIANT_DETECTOR_AVAILABLE},
            "confidence_calibrator": {"available": _core.CONFIDENCE_CALIBRATOR_AVAILABLE},
            "router": {"available": _core.ROUTER_AVAILABLE},
            "meta_reasoning": {
                "available": _core.META_REASONING_AVAILABLE,
                "enabled": wm.meta_reasoning_enabled,
                # Note: Include full meta-reasoning component status
                "components": {
                    # Use getattr consistently for all attributes to ensure safety
                    "motivational_introspection": getattr(wm, "motivational_introspection", None) is not None,
                    "validation_tracker": getattr(wm, "validation_tracker", None) is not None,
                    "transparency_interface": getattr(wm, "transparency_interface", None) is not None,
                    "internal_critic": getattr(wm, "internal_critic", None) is not None,
                    "curiosity_reward_shaper": getattr(wm, "curiosity_reward_shaper", None) is not None,
                    "ethical_boundary_monitor": getattr(wm, "ethical_boundary_monitor", None) is not None,
                    "preference_learner": getattr(wm, "preference_learner", None) is not None,
                    "value_evolution_tracker": getattr(wm, "value_evolution_tracker", None) is not None,
                    "counterfactual_reasoner": getattr(wm, "counterfactual_reasoner", None) is not None,
                    "goal_conflict_detector": getattr(wm, "goal_conflict_detector", None) is not None,
                    "objective_negotiator": getattr(wm, "objective_negotiator", None) is not None,
                },
            },
            "self_improvement": {
                "available": _core.META_REASONING_AVAILABLE
                and _core.SelfImprovementDrive is not None,
                "enabled": getattr(wm, "self_improvement_enabled", False),
            },
            "safety_validator": {
                "available": _core.EnhancedSafetyValidator is not None,
                "enabled": getattr(wm, "safety_mode", "disabled") == "enabled",
            },
        },
    }
