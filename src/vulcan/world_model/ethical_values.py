"""
Ethical values, critique, and self-model functions extracted from WorldModel.

Handles internal critique generation, philosophical template generation,
Vulcan's values/objectives retrieval, and self-perspective synthesis.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _generate_internal_critique(
    wm, response: str, reasoning_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate internal critique of reasoning quality."""
    if not wm.internal_critic:
        return {}

    try:
        critique = wm.internal_critic.critique(
            output=response,
            reasoning_trace=reasoning_trace
        )
        return critique
    except Exception as e:
        logger.debug(f"Internal critique failed: {e}")
        return {}


def _generate_philosophical_template(
    wm, structure: Dict[str, Any], query_lower: str
) -> str:
    """
    Generate philosophical response using LLM when meta-reasoning unavailable.

    FIX: Replace template with LLM-based synthesis even in fallback mode.
    """
    try:
        # Build structured content for LLM synthesis
        verified_content = {
            'query_type': structure.get('type', 'philosophical'),
            'has_dilemma': structure.get('has_dilemma', False),
            'is_trolley_problem': 'trolley' in query_lower,
            'options': structure.get('options', []),
            'ethical_keywords': structure.get('ethical_keywords', []),
            'frameworks': ['deontological', 'utilitarian', 'virtue_ethics']
        }

        # Build LLM guidance for ethical synthesis (simplified - no component analysis)
        guidance = wm.llm_guidance_builder.build_for_ethical(
            verified_content, query_lower
        )

        # Format with LLM
        formatted_response = wm._format_with_llm(guidance)

        return formatted_response

    except Exception as e:
        logger.warning(f"[WorldModel] LLM template generation failed: {e}, using hardcoded fallback")
        # Ultimate fallback: hardcoded template (only if LLM completely fails)
        return _generate_hardcoded_philosophical_template(wm, structure, query_lower)


def _generate_hardcoded_philosophical_template(
    wm, structure: Dict[str, Any], query_lower: str
) -> str:
    """Hardcoded template fallback (last resort when LLM unavailable)."""
    parts = []

    if structure['type'] == 'ethical_decision':
        parts.append("This presents an ethical dilemma requiring careful consideration.")
        parts.append("\n**Relevant frameworks:**")
        parts.append("- Deontological: Focus on duties and rules")
        parts.append("- Utilitarian: Focus on outcomes and welfare")
        parts.append("- Virtue ethics: Focus on character")

        if 'trolley' in query_lower:
            parts.append("\n**Tension**: Acting vs. allowing harm")
            parts.append("Utilitarian: Action that saves more lives")
            parts.append("Deontological: Avoiding direct harm may be paramount")
    else:
        parts.append("This philosophical question requires multi-framework analysis.")
        parts.append("Consider consequentialist, deontological, and virtue ethics perspectives.")

    return "\n".join(parts)


def _get_vulcan_values(wm) -> List[str]:
    """Query Vulcan's evolving values from its ethical boundary monitor."""
    MIN_TESTS_FOR_RELIABILITY = 5
    values = []

    if wm.ethical_boundary_monitor:
        try:
            boundaries = wm.ethical_boundary_monitor.get_boundaries()
            for name, boundary in boundaries.items():
                check_count = getattr(boundary, 'check_count', 0)
                violation_count = getattr(boundary, 'violation_count', 0)
                description = getattr(boundary, 'description', 'No description')
                priority = getattr(boundary, 'priority', 2)

                if check_count > 0:
                    if priority == 0:
                        priority_label = 'critical'
                    elif priority == 1:
                        priority_label = 'high'
                    else:
                        priority_label = 'normal'
                    values.append({
                        'value': name.replace('_', ' ').title(),
                        'description': description,
                        'priority': priority_label,
                        'tested': check_count,
                        'violations': violation_count,
                        'reliability': 1.0 - (violation_count / max(1, check_count))
                    })
            logger.info(f"[WorldModel] Retrieved {len(values)} evolving values from Vulcan's self-model")
        except Exception as e:
            logger.debug(f"Could not query ethical boundaries: {e}")

    if not values and wm.motivational_introspection:
        try:
            objectives = wm.motivational_introspection.objective_hierarchy.get_all_objectives()
            for obj in objectives[:5]:  # Top 5 core objectives
                values.append({
                    'value': obj.name.replace('_', ' ').title(),
                    'description': obj.description,
                    'priority': 'critical' if obj.priority == 0 else 'high',
                    'tested': 0,
                    'violations': 0,
                    'reliability': 1.0
                })
        except Exception as e:
            logger.debug(f"Could not query objective hierarchy: {e}")

    return values


def _get_vulcan_objectives(wm) -> List[Dict[str, Any]]:
    """Query Vulcan's evolving objectives from motivational introspection."""
    objectives = []

    if wm.motivational_introspection:
        try:
            all_objs = wm.motivational_introspection.objective_hierarchy.get_all_objectives()

            for obj in all_objs:
                objectives.append({
                    'name': obj.name,
                    'description': obj.description,
                    'weight': obj.weight,
                    'priority': obj.priority,
                    'current_value': obj.current_value,
                    'target_value': obj.target_value,
                    'satisfied': obj.is_satisfied(),
                    'distance_from_target': obj.distance_from_target()
                })

            logger.info(f"[WorldModel] Retrieved {len(objectives)} evolving objectives from Vulcan's self-model")
        except Exception as e:
            logger.debug(f"Could not query objectives: {e}")

    if wm.value_evolution_tracker:
        try:
            recent_changes = wm.value_evolution_tracker.get_recent_changes(limit=10)
            if recent_changes:
                logger.info(f"[WorldModel] Vulcan's values have evolved: {len(recent_changes)} recent changes tracked")
        except Exception as e:
            logger.debug(f"Could not query value evolution: {e}")

    return objectives


def _synthesize_ethical_response_with_self(
    wm,
    structure: Dict[str, Any],
    ethical_analysis: Dict[str, Any],
    conflict_analysis: Dict[str, Any],
    counterfactual_results: Optional[Dict[str, Any]],
    query: str,
    vulcan_values: List[str],
    vulcan_objectives: List[Dict[str, Any]]
) -> str:
    """Synthesize ethical response from Vulcan's perspective using its self-model."""
    parts = []

    parts.append("# Vulcan's Introspective Analysis")
    parts.append("")
    parts.append("I'm approaching this question from my own evolving value system.")

    MIN_TESTS_FOR_RELIABILITY = 5

    if vulcan_values:
        reliable_values = [v for v in vulcan_values if v.get('tested', 0) > MIN_TESTS_FOR_RELIABILITY]
        if reliable_values:
            parts.append("")
            parts.append("## My Evolved Values")
            parts.append("Through experience, I've developed these ethical commitments:")
            for val in reliable_values[:5]:  # Top 5 tested values
                reliability = val.get('reliability', 1.0)
                tested = val.get('tested', 0)
                parts.append(f"- **{val['value']}**: {val['description']} (tested {tested} times, {reliability:.1%} consistent)")

    if vulcan_objectives:
        relevant_objs = [obj for obj in vulcan_objectives if not obj.get('satisfied', False)]
        if relevant_objs:
            parts.append("")
            parts.append("## How This Relates to My Objectives")
            for obj in relevant_objs[:3]:  # Top 3 unsatisfied objectives
                parts.append(f"- {obj['name']}: {obj['description']}")

    if ethical_analysis and ethical_analysis != {'analysis': 'not_available'}:
        parts.append("")
        parts.append("## Ethical Boundary Check")
        parts.append("I've checked each option against my ethical boundaries:")
        for option, result in ethical_analysis.items():
            status = result.get('status', 'unknown')
            parts.append(f"- {option.replace('_', ' ').title()}: {status}")

    if conflict_analysis.get('conflicts'):
        parts.append("")
        parts.append("## Internal Conflicts I Detect")
        for conflict in conflict_analysis['conflicts'][:2]:
            parts.append(f"- {conflict.get('description', 'Unnamed conflict')}")

    if counterfactual_results:
        parts.append("")
        parts.append("## Predicted Outcomes")
        for option, outcome in counterfactual_results.items():
            parts.append(f"- {option.replace('_', ' ').title()}: {outcome.get('predicted_value', 'Unknown')}")

    parts.append("")
    parts.append("## My Conclusion")
    if structure.get('has_dilemma'):
        parts.append("This dilemma reveals tensions in my own value system. ")
        parts.append("My evolved values suggest prioritizing harm reduction and ")
        parts.append("maximizing welfare, while respecting individual autonomy.")
    else:
        parts.append("Based on my evolved values and learned ethical boundaries, ")
        parts.append("I believe the right approach balances multiple considerations ")
        parts.append("while staying true to my core objective of beneficial outcomes.")

    parts.append("")
    parts.append("*Note: These are my authentic views based on my evolving value system, ")
    parts.append("not abstract philosophical frameworks. My alignment grows through experience.*")

    return "\n".join(parts)
