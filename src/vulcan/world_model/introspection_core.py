"""
introspection_core.py - Core introspect dispatch logic and type classification.

Extracted from world_model_core.py to reduce class size.
Handler routing logic has been further extracted to introspection_handlers.py.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import re

logger = logging.getLogger(__name__)


def introspect(wm, query: str, aspect: str = "general") -> dict:
    """
    Handle all self-introspection queries.

    FIX Issue #4: Comprehensive self-awareness handling.
    FIX Issue #1 & #2: Delegation intelligence - detect when query LOOKS
    self-referential but actually needs another reasoner.

    World Model is where VULCAN's "self" resides. It should be aware of:
    - Its own architecture and capabilities
    - Its reasoning processes across all domains
    - Its limitations and boundaries
    - Questions about its own existence, awareness, preferences

    This includes questions about math, logic, probability, causation, etc.
    The world model maintains awareness of ALL reasoning that happens.

    Args:
        wm: WorldModel instance
        query: The introspection query
        aspect: Aspect to focus on (general, capabilities, process, boundaries)

    Returns:
        Dictionary with response, confidence, aspect, and reasoning
        If delegation is needed, includes 'needs_delegation', 'recommended_tool',
        and 'delegation_reason' keys.
    """
    from .introspection_handlers import route_introspection_query
    from .introspection_domain import general_introspection
    from .introspection_responses import (
        generate_comparison_response,
        generate_future_speculation_response,
        generate_preference_response,
    )
    from .introspection_demo import handle_demonstration_query

    query_lower = query.lower()

    # Check if delegation is needed FIRST
    needs_delegation, recommended_tool, delegation_reason = wm._analyze_delegation_need(query)

    if needs_delegation:
        logger.info(
            f"[WorldModel] DELEGATION RECOMMENDED: "
            f"'{recommended_tool}' - {delegation_reason}"
        )
        return {
            "confidence": 0.65,
            "response": None,
            "aspect": "delegation",
            "reasoning": delegation_reason,
            "needs_delegation": True,
            "recommended_tool": recommended_tool,
            "delegation_reason": delegation_reason,
            "metadata": {
                "awareness_confidence": 0.90,
                "detected_pattern": recommended_tool,
                "query_analysis": delegation_reason,
            },
        }

    # Try aspect-specific handlers
    result = route_introspection_query(wm, query, query_lower)
    if result is not None:
        return result

    # ENHANCED INTROSPECTION TYPE CLASSIFICATION
    question_type = classify_introspection_type(wm, query)

    if question_type == "COMPARISON":
        return {
            "confidence": 0.85,
            "response": generate_comparison_response(wm, query),
            "aspect": "comparison",
            "reasoning": "Question comparing VULCAN to other AI systems",
        }
    elif question_type == "FUTURE_CAPABILITY":
        return {
            "confidence": 0.75,
            "response": generate_future_speculation_response(wm, query),
            "aspect": "future_speculation",
            "reasoning": "Speculative question about future capabilities or emergence",
        }
    elif question_type == "PREFERENCE":
        return {
            "confidence": 0.85,
            "response": generate_preference_response(wm, query),
            "aspect": "preference",
            "reasoning": "Question about VULCAN's preferences or choices",
        }

    # DEMONSTRATION QUERIES
    if question_type == "DEMONSTRATION":
        return handle_demonstration_query(wm, query)

    # GENERAL INTROSPECTION (FALLBACK)
    logger.debug(f"[WorldModel] Could not classify introspection type, using general: {query[:100]}")
    return {
        "confidence": 0.80,
        "response": general_introspection(wm, query),
        "aspect": aspect,
        "reasoning": "General introspective query",
    }


def classify_introspection_type(wm, query: str) -> str:
    """
    Classify what type of introspection question this is.

    Returns one of: COMPARISON, FUTURE_CAPABILITY, CURRENT_CAPABILITY,
    ARCHITECTURAL, PREFERENCE, DEMONSTRATION, or GENERAL
    """
    query_lower = query.lower()

    # DEMONSTRATION PATTERNS
    demonstration_patterns = [
        r'demonstrate\s+(?:how\s+you\s+)?(?:use|do|perform)',
        r'show\s+(?:me\s+)?(?:an?\s+)?(?:example|demo)',
        r'give\s+(?:me\s+)?(?:an?\s+)?(?:example|demo)',
        r'can\s+you\s+demonstrate',
        r'run\s+(?:an?\s+)?(?:example|demonstration)',
        r'let\s+me\s+see.*reasoning',
        r'example\s+of.*reasoning',
        r'demonstration\s+of',
    ]

    for pattern in demonstration_patterns:
        if re.search(pattern, query_lower):
            return "DEMONSTRATION"

    if re.search(r'(?:different\s+from|compared\s+to|versus|vs\.?|how\s+do\s+you\s+compare)(?:\s+\w+)?', query_lower):
        return "COMPARISON"

    ai_names = ['grok', 'chatgpt', 'claude', 'bard', 'gemini', 'copilot', 'llama', 'gpt']
    if any(name in query_lower for name in ai_names):
        return "COMPARISON"

    if re.search(r'would\s+you.*(?:achieve|become|develop|gain|attain)', query_lower):
        return "FUTURE_CAPABILITY"
    if re.search(r'if\s+you.*(?:continue|interact|learn).*(?:would|could)', query_lower):
        return "FUTURE_CAPABILITY"
    if re.search(r'(?:would|could|might)\s+you\s+(?:ever|eventually|someday)', query_lower):
        return "FUTURE_CAPABILITY"

    if re.search(r'would\s+you.*(?:choose|prefer|want|like|take|pick)', query_lower):
        return "PREFERENCE"
    if re.search(r'what\s+would\s+you\s+(?:choose|prefer|do|pick)', query_lower):
        return "PREFERENCE"

    if re.search(r'(?:can|do|are)\s+you.*(?:able|capable|have)', query_lower):
        return "CURRENT_CAPABILITY"

    if re.search(r'how\s+(?:do|does)\s+you.*(?:work|function|operate)', query_lower):
        return "ARCHITECTURAL"

    return "GENERAL"


def identify_capability(wm, query: str) -> str:
    """Identify which capability is being asked about."""
    capability_keywords = {
        "reason": ["reason", "think", "analyze", "infer"],
        "compute": ["calculate", "compute", "solve"],
        "remember": ["remember", "recall", "know"],
        "learn": ["learn", "improve", "adapt"],
        "feel": ["feel", "experience", "sense"],
        "want": ["want", "desire", "prefer", "choose"],
        "understand": ["understand", "comprehend", "grasp"],
    }

    query_lower = query.lower()
    for capability, keywords in capability_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return capability

    return "general"
