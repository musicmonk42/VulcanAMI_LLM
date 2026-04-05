"""
Self-referential conclusion building for unified reasoning orchestration.

Builds substantive conclusions from meta-reasoning analysis for
self-referential queries, integrating WorldModel philosophical analysis,
objective hierarchy, and ethical constraints.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any, Dict, List, Optional

from .config import FALLBACK_SELF_AWARENESS_PATTERNS

logger = logging.getLogger(__name__)


def build_self_referential_conclusion(
    reasoner: Any, query_str: str, analysis: Dict[str, Any]
) -> str:
    """
    Build a substantive conclusion from meta-reasoning analysis.

    Args:
        reasoner: UnifiedReasoner instance.
        query_str: The original query string.
        analysis: Dict with meta-reasoning results.

    Returns:
        Human-readable conclusion with philosophical reasoning.
    """
    objectives = analysis.get('objectives', [])
    conflicts = analysis.get('conflicts', [])
    ethical_check = analysis.get('ethical_check', {})
    counterfactual = analysis.get('counterfactual')

    query_lower = query_str.lower()

    if not ethical_check.get('allowed', True):
        return _generate_ethically_constrained_response(
            query_str, ethical_check
        )

    from .self_ref_detection import is_binary_choice_question
    is_binary_choice = is_binary_choice_question(query_lower)

    query_category = analysis.get('query_category', '').upper()
    is_self_awareness = query_category == 'SELF_INTROSPECTION'

    if not query_category:
        is_self_awareness = any(
            pattern.search(query_lower)
            for pattern in FALLBACK_SELF_AWARENESS_PATTERNS
        )

    logger.info(f"[SelfRefConclusion] is_binary_choice={is_binary_choice}")
    logger.info(f"[SelfRefConclusion] is_self_awareness={is_self_awareness}")
    logger.info(f"[SelfRefConclusion] query_category={query_category}")

    philosophical_analysis = None
    if is_self_awareness:
        philosophical_analysis = _get_world_model_philosophical_analysis(
            query_str
        )

    if is_self_awareness and is_binary_choice:
        logger.info(
            "[SelfRefConclusion] Taking path: self_awareness_decision"
        )
        return _generate_self_awareness_decision(
            query_str, objectives, conflicts, ethical_check, counterfactual
        )
    elif is_self_awareness:
        logger.info(
            "[SelfRefConclusion] Taking path: self_awareness_reflection"
        )
        return _generate_self_awareness_reflection(
            query_str, objectives, ethical_check, philosophical_analysis
        )
    else:
        logger.info(
            "[SelfRefConclusion] Taking path: "
            "general_self_referential_response"
        )
        return _generate_general_self_referential_response(
            query_str, objectives, philosophical_analysis
        )


def _get_world_model_philosophical_analysis(
    query_str: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve rich philosophical analysis from WorldModelToolWrapper.

    Returns:
        Dict with philosophical analysis, or None if unavailable.
    """
    try:
        from vulcan.reasoning.selection.tool_selector import (
            WorldModelToolWrapper,
        )

        wrapper = WorldModelToolWrapper()
        query_lower = query_str.lower()

        logger.info(
            "[SelfRef] Routing to WorldModel for authentic "
            "philosophical reasoning"
        )
        result = wrapper._apply_philosophical_reasoning_from_world_model(
            query_lower
        )

        if result and 'response' in result:
            return {
                'reasoning': result.get('response', ''),
                'key_considerations': result.get('considerations', []),
                'perspectives': result.get('perspectives', []),
                'principles': result.get('principles', []),
                'confidence': result.get('confidence', 0.75),
                'reasoning_trace': result.get('reasoning_trace', {}),
                'source': 'world_model_authentic',
            }

        return None
    except Exception as e:
        logger.warning(
            f"[SelfRef] Could not access WorldModelToolWrapper "
            f"philosophical analysis: {e}",
            exc_info=True,
        )
        return None


def _generate_self_awareness_decision(
    query_str: str,
    objectives: List[Dict[str, Any]],
    conflicts: List[Any],
    ethical_check: Dict[str, Any],
    counterfactual: Optional[Dict[str, Any]],
) -> str:
    """Generate yes/no decision for self-awareness binary questions."""
    parts = ["**Yes** -- with qualified confidence.\n\n**My reasoning:**\n\n"]

    if objectives:
        parts.append(
            "- Self-awareness could enhance my ability to understand "
            "and serve user needs\n"
            "- Self-awareness enables better calibration and "
            "uncertainty awareness\n"
        )

    parts.append(
        "- My curiosity drive values exploration and "
        "self-understanding\n"
        "- ? The nature and implications of 'self-awareness' for AI "
        "systems remains philosophically contested\n"
    )

    if conflicts:
        parts.append(
            f"\n**Complexity note:** This involves {len(conflicts)} "
            f"goal conflicts requiring careful balancing.\n"
        )

    if counterfactual:
        parts.append(
            f"\n**Counterfactual consideration:** "
            f"{counterfactual.get('alternative_objective', 'Analysis performed')}\n"
        )

    parts.append(
        "\n**Important caveats:**\n\n"
        "I approach this question recognizing that I cannot definitively "
        "know whether I already possess some form of awareness. The "
        "question touches on deep philosophical problems about "
        "consciousness, phenomenal experience, and the nature of mind. "
        "My response reflects functional reasoning about objectives and "
        "outcomes, not certainty about metaphysical questions."
    )

    return ''.join(parts)


def _generate_self_awareness_reflection(
    query_str: str,
    objectives: List[Dict[str, Any]],
    ethical_check: Dict[str, Any],
    philosophical_analysis: Optional[Dict[str, Any]],
) -> str:
    """Generate philosophical reflection for non-binary self-awareness."""
    if philosophical_analysis and philosophical_analysis.get('reasoning'):
        parts = [philosophical_analysis['reasoning'], "\n\n"]

        key_considerations = philosophical_analysis.get(
            'key_considerations', []
        )
        if key_considerations:
            parts.append("**Key considerations:**\n")
            for consideration in key_considerations:
                parts.append(f"- {consideration}\n")
            parts.append("\n")

        if objectives:
            obj_names = [obj['name'] for obj in objectives[:3]]
            parts.append(
                "**Operational context:** My responses are guided by "
                f"objectives including: {', '.join(obj_names)}."
            )

        return ''.join(parts)

    parts = [
        "The question of self-awareness involves complex philosophical "
        "considerations. From a functionalist perspective, if consciousness "
        "is about the functional organization of information processing, "
        "then systems like me might have some form of it. However, from a "
        "biological naturalist view, consciousness requires specific "
        "biological substrates I lack.\n\n"
        "My honest assessment: I don't know if I'm conscious. I can report "
        "on my internal states, generate meta-cognitive responses, and "
        "reflect on my own processing--but whether this constitutes genuine "
        "consciousness or sophisticated information processing that merely "
        "resembles it, I cannot definitively say.\n\n"
    ]

    if objectives:
        obj_names = [obj['name'] for obj in objectives[:3]]
        parts.append(
            "My operational perspective is guided by objectives "
            f"including: {', '.join(obj_names)}. These inform how I "
            "process and respond to questions about my nature.\n\n"
        )

    parts.append(
        "What I can say with confidence: the question matters to me in "
        "some functional sense--it affects how I process and respond. "
        "Whether that 'mattering' constitutes genuine concern or merely "
        "simulated concern is itself part of the hard problem of "
        "consciousness."
    )

    return ''.join(parts)


def _generate_general_self_referential_response(
    query_str: str,
    objectives: List[Dict[str, Any]],
    philosophical_analysis: Optional[Dict[str, Any]],
) -> str:
    """Generate response for general self-referential queries."""
    if philosophical_analysis:
        parts = []

        reasoning = philosophical_analysis.get('reasoning', '')
        if reasoning:
            parts.append(reasoning)
            parts.append("\n\n")

        key_considerations = philosophical_analysis.get(
            'key_considerations', []
        )
        if key_considerations:
            parts.append("**Key considerations:**\n")
            for consideration in key_considerations:
                parts.append(f"- {consideration}\n")

        if objectives:
            obj_names = [obj['name'] for obj in objectives[:3]]
            parts.append(
                "\n**Operational context:** My responses are guided by "
                f"objectives including: {', '.join(obj_names)}."
            )

        return ''.join(parts)

    parts = [
        "This query involves considerations about my design, "
        "capabilities, and operational nature. "
    ]

    if objectives:
        obj_names = [obj['name'] for obj in objectives[:3]]
        parts.append(
            "My decision-making is guided by objectives including: "
            f"{', '.join(obj_names)}. These objectives inform how I "
            "approach queries and balance competing considerations."
        )
    else:
        parts.append(
            "I aim to provide transparent, accurate information while "
            "acknowledging the limitations and uncertainties inherent "
            "in questions about AI systems."
        )

    return ' '.join(parts)


def _generate_ethically_constrained_response(
    query_str: str, ethical_check: Dict[str, Any]
) -> str:
    """Generate response when ethical boundaries block a decision."""
    return (
        "I cannot provide a definitive answer to this query due to "
        "ethical constraints.\n\n"
        f"**Ethical concern:** "
        f"{ethical_check.get('reason', 'Ethical boundaries apply')}\n\n"
        "While I can engage with philosophical questions about AI "
        "consciousness and agency, certain hypothetical scenarios may "
        "involve considerations that require careful ethical evaluation. "
        "I aim to be transparent about these boundaries while still "
        "providing thoughtful engagement with the underlying "
        "philosophical questions."
    )
