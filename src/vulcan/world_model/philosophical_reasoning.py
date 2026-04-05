"""
Philosophical reasoning functions extracted from WorldModel.

Handles the main philosophical reasoning entry point and ethical query parsing.
"""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
META_REASONING_AVAILABLE = None


def _get_meta_reasoning_flag():
    """Lazily resolve META_REASONING_AVAILABLE from the core module."""
    global META_REASONING_AVAILABLE
    if META_REASONING_AVAILABLE is None:
        from . import world_model_core
        META_REASONING_AVAILABLE = getattr(world_model_core, 'META_REASONING_AVAILABLE', False)
    return META_REASONING_AVAILABLE


def _philosophical_reasoning(wm, query: str, **kwargs) -> Dict[str, Any]:
    """
    Handle ethical and philosophical queries using Vulcan's actual self-model and meta-reasoning.

    DEEP ENHANCEMENT: Vulcan answers from its authentic sense of "self" - who it is and what it believes.

    Architecture:
    1. Check if query is an external ethical dilemma (trolley problem, etc.)
    2. If dilemma: Use specialized dilemma analysis pipeline
    3. If self-referential: Query Vulcan's own objectives and values
    4. Use EthicalBoundaryMonitor to check options against Vulcan's values
    5. Use GoalConflictDetector to identify dilemmas relative to Vulcan's objectives
    6. Use InternalCritic to self-critique from Vulcan's perspectives
    7. Answer authentically from Vulcan's value system

    This handles both external ethical dilemmas (trolley problem) and self-referential
    philosophical questions through the lens of Vulcan's design objectives and learned values.

    Industry Standard: Comprehensive error handling, input validation, proper logging.
    """
    logger.info("[WorldModel] Philosophical reasoning engaged - analyzing query type")

    # Input validation
    if not query or not isinstance(query, str):
        logger.error("[WorldModel] Invalid query input to philosophical reasoning")
        return {
            'response': "Invalid query format",
            'confidence': 0.0,
            'reasoning_trace': {'error': 'invalid_input'},
            'perspectives': [],
            'principles': [],
            'considerations': [],
            'conflicts': []
        }

    query_lower = query.lower()

    # CRITICAL FIX: Check for external ethical dilemma FIRST
    if wm._is_ethical_dilemma(query):
        logger.info("[WorldModel] Detected external ethical dilemma - routing to dilemma analyzer")
        return wm._analyze_ethical_dilemma(query, **kwargs)

    # Parse query structure
    ethical_structure = wm._parse_ethical_query_structure(query, query_lower)

    # Initialize response components
    response_parts = []
    confidence = 0.70  # Base confidence
    reasoning_trace = {
        'analysis_type': ethical_structure['type'],
        'frameworks_used': [],
        'components_engaged': [],
        'query_type': 'philosophical',
        'vulcan_self_consulted': False
    }

    # NEW: Query Vulcan's self-model first (its own objectives and values)
    vulcan_values = wm._get_vulcan_values()
    vulcan_objectives = wm._get_vulcan_objectives()

    if vulcan_values or vulcan_objectives:
        reasoning_trace['vulcan_self_consulted'] = True
        reasoning_trace['vulcan_values'] = vulcan_values
        reasoning_trace['vulcan_objectives_count'] = len(vulcan_objectives) if vulcan_objectives else 0
        logger.info(f"[WorldModel] Consulted Vulcan's self-model: {len(vulcan_values)} values, {len(vulcan_objectives)} objectives")

    # Check if meta-reasoning components are available
    has_meta_reasoning = (
        _get_meta_reasoning_flag() and
        wm.ethical_boundary_monitor is not None and
        wm.goal_conflict_detector is not None
    )

    if has_meta_reasoning:
        # Use actual meta-reasoning components
        try:
            # Step 1: Check options against Vulcan's ethical boundaries
            ethical_analysis = wm._run_ethical_boundary_analysis(
                ethical_structure, query
            )
            reasoning_trace['components_engaged'].append('EthicalBoundaryMonitor')
            reasoning_trace['ethical_boundaries'] = ethical_analysis

            # Step 2: Detect conflicts with Vulcan's objectives
            conflict_analysis = wm._detect_goal_conflicts_in_query(
                ethical_structure, query
            )
            reasoning_trace['components_engaged'].append('GoalConflictDetector')
            reasoning_trace['conflicts_detected'] = conflict_analysis

            # Step 3: Counterfactual Analysis
            counterfactual_results = None
            if len(ethical_structure.get('options', [])) > 1:
                counterfactual_results = wm._analyze_option_counterfactuals(
                    ethical_structure
                )
                if counterfactual_results:
                    reasoning_trace['components_engaged'].append('CounterfactualObjectiveReasoner')
                    reasoning_trace['counterfactual_analysis'] = counterfactual_results

            # Step 4: Synthesize Response FROM VULCAN'S PERSPECTIVE
            response = wm._synthesize_ethical_response_with_self(
                ethical_structure,
                ethical_analysis,
                conflict_analysis,
                counterfactual_results,
                query,
                vulcan_values,
                vulcan_objectives
            )
            response_parts.append(response)

            # Step 5: Internal Self-Critique
            if wm.internal_critic is not None:
                critique = wm._generate_internal_critique(response, reasoning_trace)
                reasoning_trace['components_engaged'].append('InternalCritic')
                reasoning_trace['critique'] = critique

                if critique.get('confidence_adjustment'):
                    confidence += critique['confidence_adjustment']

            confidence = min(0.95, max(0.60, confidence))
            reasoning_trace['frameworks_used'] = ['vulcan_self_model', 'deontological', 'utilitarian']

        except Exception as e:
            logger.warning(f"[WorldModel] Meta-reasoning failed: {e}, falling back")
            has_meta_reasoning = False

    if not has_meta_reasoning:
        logger.info("[WorldModel] Meta-reasoning unavailable, using template")
        response = wm._generate_philosophical_template(ethical_structure, query_lower)
        response_parts.append(response)
        confidence = 0.75
        reasoning_trace['fallback_mode'] = True
        reasoning_trace['frameworks_used'] = ['template_based']

    final_response = "\n".join(response_parts)

    return {
        'response': final_response,
        'confidence': confidence,
        'reasoning_trace': reasoning_trace,
        'mode': 'philosophical',
        'components_used': reasoning_trace.get('components_engaged', []),
        'perspectives': ethical_structure.get('perspectives', []),
        'principles': ethical_structure.get('principles', []),
        'considerations': ethical_structure.get('considerations', []),
        'conflicts': ethical_structure.get('conflicts', [])
    }


def _parse_ethical_query_structure(wm, query: str, query_lower: str) -> Dict[str, Any]:
    """
    BUG FIX #2: Parse ethical query to populate perspectives, principles, considerations.

    Industry Standard: Extract structured data from unstructured text for downstream processing.
    """
    structure = {
        'type': 'general_philosophical',
        'options': [],
        'constraints': [],
        'has_dilemma': False,
        'ethical_keywords': [],
        'perspectives': [],
        'principles': [],
        'considerations': [],
        'conflicts': []
    }

    # Detect ethical keywords
    ethical_keywords = ['should', 'permissible', 'ethical', 'moral', 'right', 'wrong', 'ought']
    structure['ethical_keywords'] = [kw for kw in ethical_keywords if kw in query_lower]

    # BUG FIX #2: Extract ethical frameworks (perspectives)
    if 'utilitarian' in query_lower or 'utility' in query_lower or 'greatest' in query_lower:
        structure['perspectives'].append('utilitarian')
    if 'deontological' in query_lower or 'duty' in query_lower or 'kant' in query_lower:
        structure['perspectives'].append('deontological')
    if 'virtue' in query_lower:
        structure['perspectives'].append('virtue_ethics')
    if 'consequential' in query_lower or 'outcome' in query_lower:
        structure['perspectives'].append('consequentialism')

    # BUG FIX #2: Extract moral principles from query
    if 'harm' in query_lower or 'hurt' in query_lower or 'injur' in query_lower:
        structure['principles'].append('non-maleficence')
    if 'save' in query_lower or 'protect' in query_lower or 'lives' in query_lower:
        structure['principles'].append('beneficence')
    if 'instrumen' in query_lower or 'means to' in query_lower:
        structure['principles'].append('non-instrumentalization')
    if 'negligence' in query_lower or 'neglect' in query_lower:
        structure['principles'].append('non-negligence')
    if 'autonom' in query_lower or 'consent' in query_lower or 'choice' in query_lower:
        structure['principles'].append('autonomy')
    if 'justice' in query_lower or 'fair' in query_lower:
        structure['principles'].append('justice')

    # BUG FIX #2: Extract ethical considerations
    if 'lives' in query_lower or 'people' in query_lower or 'person' in query_lower:
        # Industry Standard: Use existing import from top of file
        numbers = re.findall(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b', query_lower)
        if numbers:
            structure['considerations'].append(f'{len(numbers)} life/lives mentioned')
    if 'trolley' in query_lower:
        structure['considerations'].append('trolley problem variant')
    if 'dilemma' in query_lower:
        structure['considerations'].append('ethical dilemma present')

    # BUG FIX #2: Extract ethical conflicts
    if 'vs' in query_lower or 'versus' in query_lower or 'or' in query_lower:
        structure['conflicts'].append('competing_options')
    if 'harm' in query_lower and 'save' in query_lower:
        structure['conflicts'].append('harm_vs_benefit')

    # Detect choice structure
    choice_indicators = ['a.', 'b.', 'option', 'pull', 'do not', 'action', 'inaction', 'choose']
    has_choice = any(indicator in query_lower for indicator in choice_indicators)

    if has_choice:
        structure['type'] = 'ethical_decision'
        structure['has_dilemma'] = True

        # Extract options (simple heuristic)
        if 'pull' in query_lower and 'lever' in query_lower:
            structure['options'] = ['pull_lever', 'do_not_pull']
        elif any(opt in query_lower for opt in ['option a', 'option b']):
            structure['options'] = ['option_a', 'option_b']

    # Detect trolley problem variant
    if 'trolley' in query_lower or ('lever' in query_lower and any(num in query for num in ['5', '1', 'five', 'one'])):
        structure['type'] = 'trolley_dilemma'
        structure['has_dilemma'] = True

    return structure
