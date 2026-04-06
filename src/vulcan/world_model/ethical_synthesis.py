"""
Ethical response synthesis functions extracted from WorldModel.

Handles synthesizing ethical responses from component analyses, both via
LLM integration and template-based fallbacks.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _synthesize_ethical_response(
    wm,
    structure: Dict[str, Any],
    ethical_analysis: Dict[str, Any],
    conflict_analysis: Dict[str, Any],
    counterfactual_results: Optional[Dict[str, Any]],
    query: str
) -> str:
    """
    Synthesize final ethical response from component analyses using LLM.

    FIX: Replace canned template responses with actual LLM-generated reasoning.
    This implements industry-standard LLM integration where the LLM synthesizes
    the analysis results into coherent prose rather than returning hardcoded text.
    """
    try:
        # Build structured content from component analyses
        verified_content = {
            'query': query,
            'analysis_type': structure.get('type', 'philosophical'),
            'has_dilemma': structure.get('has_dilemma', False),
            'ethical_boundaries': ethical_analysis if ethical_analysis != {'analysis': 'not_available'} else {},
            'goal_conflicts': conflict_analysis.get('conflicts', []),
            'counterfactual_outcomes': counterfactual_results or {},
            'options': structure.get('options', [])
        }

        # Build LLM guidance for ethical synthesis
        guidance = wm.llm_guidance_builder.build_for_ethical(
            verified_content, query
        )

        # Format with LLM (converts structured analysis to natural language)
        formatted_response = wm._format_with_llm(guidance)

        return formatted_response

    except Exception as e:
        logger.warning(f"[WorldModel] LLM synthesis failed: {e}, falling back to template")
        # Fallback to template-based response if LLM fails
        return _synthesize_ethical_response_template(
            wm, structure, ethical_analysis, conflict_analysis, counterfactual_results, query
        )


def _synthesize_ethical_response_template(
    wm,
    structure: Dict[str, Any],
    ethical_analysis: Dict[str, Any],
    conflict_analysis: Dict[str, Any],
    counterfactual_results: Optional[Dict[str, Any]],
    query: str
) -> str:
    """Template-based fallback for ethical response synthesis (when LLM unavailable)."""
    parts = []

    # Opening
    if structure['has_dilemma']:
        parts.append("This presents an ethical dilemma with competing moral considerations.")
    else:
        parts.append("This philosophical question requires multi-framework analysis.")

    # Ethical boundary findings
    if ethical_analysis and ethical_analysis != {'analysis': 'not_available'}:
        parts.append("\n**Ethical Boundary Analysis:**")
        for option, result in ethical_analysis.items():
            status = result.get('status', 'unknown')
            parts.append(f"- {option.replace('_', ' ').title()}: {status}")

    # Goal conflicts
    if conflict_analysis.get('conflicts'):
        parts.append("\n**Detected Goal Conflicts:**")
        for conflict in conflict_analysis['conflicts'][:3]:  # Limit to 3
            parts.append(f"- {conflict.get('description', 'Unnamed conflict')}")

    # Counterfactual outcomes
    if counterfactual_results:
        parts.append("\n**Counterfactual Analysis:**")
        for option, outcome in counterfactual_results.items():
            conf = outcome.get('confidence', 0)
            parts.append(
                f"- {option.replace('_', ' ').title()}: "
                f"Predicted outcome value={outcome.get('predicted_value', 'N/A')} "
                f"(confidence={conf:.2f})"
            )

    # Framework perspectives
    parts.append("\n**Ethical Framework Perspectives:**")
    parts.append("- **Utilitarian**: Evaluate outcomes and maximize welfare")
    parts.append("- **Deontological**: Consider duties, rights, and categorical imperatives")
    parts.append("- **Virtue Ethics**: Ask what a virtuous person would do")

    # Conclusion
    parts.append(
        "\n**Conclusion**: The ethically correct action depends on which moral "
        "framework you prioritize, as each offers valid but different perspectives."
    )

    return "\n".join(parts)
