"""
performance_introspect.py - Performance self-knowledge and capability assessment.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

from typing import Any, Dict, List

from . import world_model_core as _core


def introspect_performance(wm) -> Dict[str, Any]:
    """
    Provide self-knowledge about system performance.

    This allows the system to answer questions like:
    - "What's your current accuracy?"
    - "Which engines are working well?"
    - "What issues have you encountered?"

    Returns:
        Dict with performance metrics and known issues
    """
    if wm.observation_count < _core.MIN_OBSERVATIONS_FOR_RECOMMENDATIONS:
        return {
            'status': 'insufficient_data',
            'message': 'Need more observations for meaningful performance analysis',
            'observation_count': wm.observation_count,
        }

    stats = compute_performance_stats(wm)
    issues = identify_known_issues(wm)
    capabilities = assess_engine_capabilities(wm)

    return {
        'status': 'operational',
        'performance': stats,
        'known_issues': issues,
        'capabilities': capabilities,
        'confidence': 0.95,
        'observation_count': wm.observation_count,
        'model_version': wm.model_version,
    }


def compute_performance_stats(wm) -> Dict[str, Any]:
    """Compute performance statistics from observation history."""
    history_size = len(wm.observation_processor.observation_history)

    if history_size == 0:
        return {'message': 'No observations recorded yet'}

    stats = {
        'total_observations': history_size,
        'model_version': wm.model_version,
    }

    if hasattr(wm, 'prediction_manager') and wm.prediction_manager:
        pred_history = list(wm.prediction_manager.prediction_history)
        if pred_history:
            confidences = [
                p['prediction'].confidence
                for p in pred_history
                if hasattr(p.get('prediction'), 'confidence')
            ]
            if confidences:
                stats['avg_prediction_confidence'] = sum(confidences) / len(confidences)
                stats['prediction_count'] = len(pred_history)

    if _core.CAUSAL_GRAPH_AVAILABLE and wm.causal_graph:
        stats['causal_nodes'] = len(wm.causal_graph.nodes) if hasattr(wm.causal_graph, 'nodes') else 0
        stats['causal_edges'] = len(wm.causal_graph.edges) if hasattr(wm.causal_graph, 'edges') else 0

    return stats


def identify_known_issues(wm) -> List[Dict[str, Any]]:
    """Identify known issues from observation patterns."""
    issues = []

    if _core.CAUSAL_GRAPH_AVAILABLE and wm.causal_graph:
        try:
            if wm.causal_graph.has_cycles():
                issues.append({
                    'type': 'causal_cycles',
                    'severity': 'MEDIUM',
                    'description': 'Causal graph contains cycles that may affect prediction accuracy',
                })
        except Exception:
            pass

    if _core.CONFIDENCE_CALIBRATOR_AVAILABLE and wm.confidence_calibrator:
        try:
            cal_error = wm.confidence_calibrator.calculate_expected_calibration_error()
            if cal_error > 0.15:
                issues.append({
                    'type': 'calibration_error',
                    'severity': 'MEDIUM',
                    'description': f'Confidence calibration error is high: {cal_error:.3f}',
                })
        except Exception:
            pass

    if not wm.meta_reasoning_enabled:
        issues.append({
            'type': 'meta_reasoning_disabled',
            'severity': 'HIGH',
            'description': 'Meta-reasoning is not enabled - self-introspection limited',
        })

    return issues


def assess_engine_capabilities(wm) -> Dict[str, Any]:
    """Assess capabilities based on component availability."""
    capabilities = {}

    capabilities['causal_reasoning'] = {
        'available': _core.CAUSAL_GRAPH_AVAILABLE,
        'status': 'working' if _core.CAUSAL_GRAPH_AVAILABLE else 'unavailable',
    }

    capabilities['prediction'] = {
        'available': _core.PREDICTION_ENGINE_AVAILABLE,
        'status': 'working' if _core.PREDICTION_ENGINE_AVAILABLE else 'unavailable',
    }

    capabilities['intervention_testing'] = {
        'available': _core.INTERVENTION_MANAGER_AVAILABLE,
        'status': 'working' if _core.INTERVENTION_MANAGER_AVAILABLE else 'unavailable',
    }

    capabilities['confidence_calibration'] = {
        'available': _core.CONFIDENCE_CALIBRATOR_AVAILABLE,
        'status': 'working' if _core.CONFIDENCE_CALIBRATOR_AVAILABLE else 'unavailable',
    }

    capabilities['meta_reasoning'] = {
        'available': wm.meta_reasoning_enabled,
        'status': 'working' if wm.meta_reasoning_enabled else 'limited',
    }

    capabilities['self_improvement'] = {
        'available': wm.self_improvement_enabled,
        'status': 'active' if wm.self_improvement_enabled else 'disabled',
    }

    return capabilities
