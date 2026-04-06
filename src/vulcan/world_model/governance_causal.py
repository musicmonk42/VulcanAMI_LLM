"""
governance_causal.py - Causal structure inspection and bootstrap opportunities.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
from typing import Any, Dict

from . import world_model_core as _core

logger = logging.getLogger(__name__)


def get_causal_structure(wm) -> Dict[str, Any]:
    """Get current causal structure information"""

    with wm.lock:
        structure = {
            "nodes": list(wm.causal_graph.nodes),
            "edges": [],
            "statistics": {
                "node_count": len(wm.causal_graph.nodes),
                "edge_count": len(wm.causal_graph.edges),
                "strongly_connected_components": len(
                    wm.causal_graph.find_strongly_connected_components()
                ),
                "max_path_length": wm.causal_graph.get_longest_path_length(),
            },
        }

        # Extract edge information
        for edge in wm.causal_graph.edges.values():
            structure["edges"].append(
                {
                    "cause": edge.cause,
                    "effect": edge.effect,
                    "strength": edge.strength,
                    "evidence_type": edge.evidence_type,
                    "confidence_interval": edge.confidence_interval,
                }
            )

        # Add invariant information
        if wm.invariants:
            structure["invariants"] = {
                "count": (
                    len(wm.invariants.invariants)
                    if hasattr(wm.invariants, "invariants")
                    else 0
                ),
                "types": wm.invariants.get_invariant_types(),
            }

        # Add confidence information
        if wm.confidence_tracker:
            structure["model_confidence"] = (
                wm.confidence_tracker.get_model_confidence()
            )

        # Add router metrics
        if wm.router:
            structure["router_metrics"] = wm.router.get_metrics()

        # Add safety information
        if wm.safety_validator:
            try:
                if hasattr(wm.safety_validator, "get_safety_stats"):
                    structure["safety_stats"] = (
                        wm.safety_validator.get_safety_stats()
                    )
            except Exception as e:
                logger.error("Error getting safety stats: %s", e)
                structure["safety_stats"] = {"error": str(e)}

        # Add meta-reasoning information
        if wm.meta_reasoning_enabled:
            try:
                structure["meta_reasoning"] = {
                    "enabled": True,
                    "objectives": list(
                        wm.motivational_introspection.active_objectives.keys()
                    ),
                    "statistics": wm.motivational_introspection.get_statistics(),
                }
            except Exception as e:
                logger.error("Error getting meta-reasoning info: %s", e)
                structure["meta_reasoning"] = {"enabled": True, "error": str(e)}
        else:
            structure["meta_reasoning"] = {"enabled": False}

        # Add self-improvement information
        if wm.self_improvement_enabled:
            try:
                structure["self_improvement"] = wm.get_improvement_status()
            except Exception as e:
                logger.error("Error getting self-improvement info: %s", e)
                structure["self_improvement"] = {"enabled": True, "error": str(e)}
        else:
            structure["self_improvement"] = {"enabled": False}

        return structure


def check_bootstrap_opportunities(wm):
    """Check for correlations that should be tested"""

    if not _core.INTERVENTION_MANAGER_AVAILABLE:
        return

    correlations = wm.correlation_tracker.get_strong_correlations(
        wm.min_correlation_strength
    )

    for correlation in correlations[:10]:
        # Skip if already tested
        if wm.causal_graph.has_edge(correlation.var_a, correlation.var_b):
            continue

        # FIXED: Handle CorrelationEntry objects
        try:
            if hasattr(correlation, "correlation"):
                corr_strength = correlation.correlation
            elif hasattr(correlation, "r_value"):
                corr_strength = correlation.r_value
            else:
                logger.warning(
                    f"Cannot find strength attribute in correlation object for "
                    f"{correlation.var_a} -> {correlation.var_b}"
                )
                continue

            # Create a simple wrapper object with the expected interface
            class CorrelationWrapper:
                def __init__(
                    self, var_a, var_b, strength, p_value, sample_size, metadata
                ):
                    self.var_a = var_a
                    self.var_b = var_b
                    self.strength = strength
                    self.p_value = p_value
                    self.sample_size = sample_size
                    self.metadata = metadata

            wrapped_correlation = CorrelationWrapper(
                correlation.var_a,
                correlation.var_b,
                corr_strength,
                correlation.p_value if hasattr(correlation, "p_value") else 0.0,
                (
                    correlation.sample_size
                    if hasattr(correlation, "sample_size")
                    else 0
                ),
                correlation.metadata if hasattr(correlation, "metadata") else {},
            )

            # Estimate value
            info_gain = wm.intervention_prioritizer.estimate_information_gain(
                wrapped_correlation
            )
            cost = wm.intervention_prioritizer.estimate_intervention_cost(
                wrapped_correlation
            )

            # Queue if high value
            if info_gain / cost > 2.0:
                wm.intervention_prioritizer.queue_intervention(
                    wrapped_correlation
                )

        except Exception as e:
            logger.error(f"Error processing correlation for bootstrap: {e}")
            continue
