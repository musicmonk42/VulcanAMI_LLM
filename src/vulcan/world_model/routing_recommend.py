"""
routing_recommend.py - Routing recommendations based on learned patterns.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

from typing import Any, Dict, List

from . import world_model_core as _core


def has_sufficient_history(wm, min_observations: int = _core.MIN_OBSERVATIONS_FOR_RECOMMENDATIONS) -> bool:
    """
    Check if world model has enough data to make recommendations.

    Args:
        wm: WorldModel instance
        min_observations: Minimum number of observations needed

    Returns:
        True if sufficient history exists for recommendations
    """
    return wm.observation_count >= min_observations


def recommend_routing(
    wm,
    query: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recommend routing based on learned patterns.

    Consults the world model's causal understanding and observation history
    to provide routing recommendations.

    Args:
        wm: WorldModel instance
        query: Query text
        classification: Initial classification from query router

    Returns:
        Dict with tools, confidence, reasoning, warnings, alternative_routing
    """
    if not has_sufficient_history(wm):
        return {
            'tools': classification.get('tools', []),
            'confidence': classification.get('confidence', 0.5),
            'reasoning': 'Insufficient historical data for recommendations',
            'warnings': [],
            'alternative_routing': None,
        }

    query_features = extract_query_features(wm, query)

    tools = classification.get('tools', [])
    if predicts_failure(wm, query_features, tools):
        warnings = get_failure_warnings(wm, query_features, tools)
        alternatives = suggest_alternative_tools(wm, query_features)

        return {
            'tools': alternatives.get('tools', tools),
            'confidence': alternatives.get('confidence', 0.6),
            'reasoning': f"Similar queries failed with {tools}. Recommending alternatives.",
            'warnings': warnings,
            'alternative_routing': alternatives,
        }

    return {
        'tools': tools,
        'confidence': classification.get('confidence', 0.7),
        'reasoning': 'Classification confirmed by world model patterns',
        'warnings': [],
        'alternative_routing': None,
    }


def extract_query_features(wm, query: str) -> Dict[str, Any]:
    """Extract features from query for pattern matching."""
    if not query:
        return {'type': 'unknown'}

    query_lower = query.lower()

    return {
        'type': infer_query_type(wm, query_lower),
        'has_formal_logic': any(sym in query for sym in _core.FORMAL_LOGIC_SYMBOLS),
        'has_probability': any(kw in query_lower for kw in _core.PROBABILITY_KEYWORDS),
        'is_self_referential': any(kw in query_lower for kw in _core.SELF_REFERENTIAL_KEYWORDS),
        'complexity': len(query) / 100.0,
    }


def infer_query_type(wm, query_lower: str) -> str:
    """Infer query type from content using shared constants."""
    if any(sym in query_lower for sym in _core.FORMAL_LOGIC_SYMBOLS) or any(
        kw in query_lower for kw in _core.FORMAL_LOGIC_KEYWORDS
    ):
        return 'formal_logic'
    elif any(kw in query_lower for kw in _core.PROBABILITY_KEYWORDS):
        return 'probabilistic'
    elif any(kw in query_lower for kw in ['compute', 'calculate', 'integral']):
        return 'mathematical'
    elif any(kw in query_lower for kw in _core.SELF_REFERENTIAL_KEYWORDS):
        return 'self_referential'
    elif any(kw in query_lower for kw in ['cause', 'effect', 'intervention']):
        return 'causal'
    else:
        return 'general'


def predicts_failure(wm, query_features: Dict[str, Any], tools: List[str]) -> bool:
    """Predict if routing is likely to fail based on causal patterns."""
    if not _core.CAUSAL_GRAPH_AVAILABLE or not wm.causal_graph:
        return False

    query_type = query_features.get('type', 'unknown')

    for tool in tools:
        pattern_node = f'{tool}_on_{query_type}'
        try:
            if wm.causal_graph.has_node(pattern_node):
                paths = wm.causal_graph.find_all_paths(pattern_node, 'validation_failure')
                if paths:
                    for path in paths:
                        if hasattr(path, 'total_strength') and path.total_strength > wm.PATH_STRENGTH_THRESHOLD:
                            return True
        except Exception:
            pass

    return False


def get_failure_warnings(wm, query_features: Dict[str, Any], tools: List[str]) -> List[str]:
    """Get warnings about potential failures."""
    warnings = []
    query_type = query_features.get('type', 'unknown')

    for tool in tools:
        if query_type == 'formal_logic' and 'math' in tool.lower():
            warnings.append(f"{tool} may produce incorrect results for formal logic queries")
        if query_type == 'probabilistic' and 'symbolic' in tool.lower():
            warnings.append(f"{tool} may not handle probabilistic reasoning well")

    return warnings


def suggest_alternative_tools(wm, query_features: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest alternative tools based on query features."""
    query_type = query_features.get('type', 'unknown')

    suggestions = {
        'formal_logic': {'tools': ['logic_engine', 'symbolic'], 'confidence': 0.8},
        'probabilistic': {'tools': ['probabilistic', 'bayesian'], 'confidence': 0.8},
        'mathematical': {'tools': ['mathematical', 'computational'], 'confidence': 0.8},
        'causal': {'tools': ['causal', 'counterfactual'], 'confidence': 0.8},
        'self_referential': {'tools': ['world_model', 'meta_reasoning'], 'confidence': 0.9},
    }

    return suggestions.get(query_type, {'tools': ['hybrid'], 'confidence': 0.6})
