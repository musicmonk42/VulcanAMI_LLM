"""
governance.py - Agent proposal evaluation and objective management.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def evaluate_agent_proposal(wm, proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate agent proposal using meta-reasoning"""

    if not wm.meta_reasoning_enabled:
        logger.warning(
            "evaluate_agent_proposal called but meta-reasoning is disabled"
        )
        return {
            "status": "unavailable",
            "valid": True,
            "reason": "Meta-reasoning layer not enabled",
            "confidence": 0.0,
        }

    with wm.lock:
        try:
            validation = (
                wm.motivational_introspection.validate_proposal_alignment(
                    proposal
                )
            )

            return {
                "status": "success",
                "valid": validation.valid,
                "overall_status": (
                    validation.overall_status.value
                    if hasattr(validation.overall_status, "value")
                    else str(validation.overall_status)
                ),
                "confidence": validation.confidence,
                "reasoning": validation.reasoning,
                "conflicts": [
                    {
                        "objectives": c.get("objectives", []),
                        "type": c.get("type", "unknown"),
                        "severity": c.get("severity", "unknown"),
                    }
                    for c in validation.conflicts_detected
                ],
                "alternatives": validation.alternatives_suggested,
                "timestamp": validation.timestamp,
            }
        except Exception as e:
            logger.error("Error evaluating agent proposal: %s", e)
            return {
                "status": "error",
                "valid": False,
                "reason": f"Evaluation failed: {str(e)}",
                "confidence": 0.0,
            }


def get_objective_state(wm) -> Dict[str, Any]:
    """Get current objective state from meta-reasoning layer"""

    if not wm.meta_reasoning_enabled:
        return {"enabled": False, "reason": "Meta-reasoning layer not available"}

    with wm.lock:
        try:
            return {
                "enabled": True,
                "objectives": wm.motivational_introspection.explain_motivation_structure(),
            }
        except Exception as e:
            logger.error("Error getting objective state: %s", e)
            return {"enabled": True, "error": str(e)}


def negotiate_objectives(wm, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Negotiate between multiple agent proposals"""

    if not wm.meta_reasoning_enabled:
        return {
            "status": "unavailable",
            "reason": "Meta-reasoning layer not enabled",
        }

    with wm.lock:
        try:
            # Check if objective_hierarchy is available
            _ = wm.motivational_introspection.objective_hierarchy

            return {
                "status": "success",
                "negotiation_available": True,
                "num_proposals": len(proposals),
                "message": "Use ObjectiveNegotiator directly for full negotiation",
            }
        except Exception as e:
            logger.error("Error negotiating objectives: %s", e)
            return {"status": "error", "reason": str(e)}


def validate_model_consistency(wm) -> Dict[str, Any]:
    """Validate internal model consistency"""

    with wm.lock:
        consistency_result = wm.consistency_validator.validate()

        # Add safety validation if available
        if wm.safety_validator:
            try:
                if hasattr(wm.safety_validator, "get_safety_stats"):
                    safety_validation = {
                        "safety_enabled": True,
                        "safety_stats": wm.safety_validator.get_safety_stats(),
                    }
                else:
                    safety_validation = {
                        "safety_enabled": True,
                        "safety_stats": "unavailable",
                    }
                consistency_result["safety_validation"] = safety_validation
            except Exception as e:
                logger.error("Error in safety validation: %s", e)
                consistency_result["safety_validation"] = {
                    "safety_enabled": True,
                    "error": str(e),
                }
        else:
            consistency_result["safety_validation"] = {
                "safety_enabled": False,
                "warning": "Safety validator not available",
            }

        # Add meta-reasoning validation if available
        if wm.meta_reasoning_enabled:
            try:
                objective_consistency = (
                    wm.motivational_introspection.objective_hierarchy.check_consistency()
                )
                consistency_result["meta_reasoning_validation"] = {
                    "enabled": True,
                    "objective_consistency": objective_consistency,
                }
            except Exception as e:
                logger.error("Error in meta-reasoning validation: %s", e)
                consistency_result["meta_reasoning_validation"] = {
                    "enabled": True,
                    "error": str(e),
                }
        else:
            consistency_result["meta_reasoning_validation"] = {"enabled": False}

        return consistency_result
