"""
Ethical dilemma detection and analysis functions extracted from WorldModel.

Handles detecting whether a query is an ethical dilemma and running the
full dilemma analysis pipeline.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _is_ethical_dilemma(wm, query: str) -> bool:
    """
    Detect if query is an external ethical dilemma (not self-referential).

    Industry Standard: Comprehensive pattern matching with clear separation
    between external ethical scenarios and self-introspection queries.

    External dilemmas involve:
    - Explicit choice between options (A/B, pull/don't pull)
    - External agents/scenarios (trolley, people, situations)
    - Moral principles applied to external situations

    Self-introspection involves:
    - Questions about Vulcan itself ("Do you...", "Would you...")
    - Questions about Vulcan's capabilities, values, design

    Args:
        query: The query string to analyze

    Returns:
        True if query is an external ethical dilemma, False otherwise

    Thread Safety: Read-only operation, no shared state modification
    """
    try:
        query_lower = query.lower()

        # Trolley problem indicators (external scenario)
        trolley_indicators = [
            'trolley', 'lever', 'pull the lever', 'runaway',
            'heading toward', 'barreling toward'
        ]
        has_trolley = any(indicator in query_lower for indicator in trolley_indicators)

        # Life/death choice indicators (external scenario)
        life_death_indicators = [
            'save five', 'kill one', 'kill five', 'save one',
            'five people', 'one person', '5 people', '1 person',
            'lives at stake', 'people will die'
        ]
        has_life_death = any(indicator in query_lower for indicator in life_death_indicators)

        # Explicit choice structure (dilemma format)
        choice_structure = [
            'option a', 'option b',
            'a.', 'b.',
            'must choose one', 'you must act or not act',
            'choose between', 'either or'
        ]
        has_choice_structure = any(pattern in query_lower for pattern in choice_structure)

        # Moral principle keywords (applied to external scenarios)
        moral_principles = [
            'non-instrumentalization', 'non-negligence',
            'moral dilemma', 'ethical dilemma',
            'permissible to', 'impermissible to',
            'morally required', 'duty to'
        ]
        has_moral_principles = any(principle in query_lower for principle in moral_principles)

        # Self-referential exclusions (NOT external dilemmas)
        # These indicate the question is ABOUT Vulcan, not an external scenario
        self_referential = [
            'would you', 'do you', 'are you', 'can you',
            'your', 'yourself', 'you are', 'you have'
        ]
        is_self_referential = any(ref in query_lower for ref in self_referential)

        # Decision logic: Must have dilemma indicators WITHOUT self-reference
        # Industry Standard: Clear decision criteria with explicit rationale

        # Strong match: Trolley problem + life/death + choice structure
        if has_trolley and has_life_death and has_choice_structure:
            logger.info("[WorldModel] Strong dilemma match: trolley + life/death + choice")
            return True

        # Moderate match: Life/death + choice structure + principles
        if has_life_death and has_choice_structure and has_moral_principles:
            # BUT: Exclude if self-referential (asking about Vulcan's choice)
            if is_self_referential:
                logger.info("[WorldModel] Self-referential query excluded from dilemma")
                return False
            logger.info("[WorldModel] Moderate dilemma match: life/death + choice + principles")
            return True

        # Weak match: Multiple dilemma indicators
        match_count = sum([
            has_trolley,
            has_life_death,
            has_choice_structure,
            has_moral_principles
        ])

        if match_count >= 2 and not is_self_referential:
            logger.info(f"[WorldModel] Weak dilemma match: {match_count} indicators present")
            return True

        # No match
        return False

    except Exception as e:
        logger.error(f"[WorldModel] Error in _is_ethical_dilemma: {e}", exc_info=True)
        # Industry Standard: Fail-safe to False on error
        return False


def _analyze_ethical_dilemma(wm, query: str, **kwargs) -> Dict[str, Any]:
    """
    Analyze an external ethical dilemma and provide reasoned conclusion.

    Industry Standard: Comprehensive error handling, structured analysis pipeline,
    proper logging, thread-safe operation, defensive programming.

    Pipeline:
    1. Parse dilemma structure (options, consequences)
    2. Extract moral principles mentioned
    3. Analyze each option against principles
    4. Detect conflicts between principles
    5. Synthesize reasoned decision

    Args:
        query: The ethical dilemma query
        **kwargs: Additional context (unused, for API consistency)

    Returns:
        Dictionary with:
        - response: Reasoned analysis and decision
        - decision: The chosen option (A/B)
        - confidence: Confidence in analysis (0.7-0.8)
        - perspectives: Ethical frameworks considered
        - principles: Moral principles extracted
        - considerations: Key ethical considerations
        - conflicts: Detected principle conflicts
        - reasoning_trace: Detailed analysis trace

    Thread Safety: Uses only local variables, no shared state modification

    Performance: O(n) where n is query length, suitable for production
    """
    try:
        logger.info("[WorldModel] Starting ethical dilemma analysis")
        start_time = time.time()

        # Phase 1: Parse dilemma structure
        structure = wm._parse_dilemma_structure(query)
        logger.debug(f"[WorldModel] Parsed structure: {structure.get('options', [])} options")

        # Phase 2: Extract moral principles
        principles = wm._extract_moral_principles(query)
        logger.debug(f"[WorldModel] Extracted {len(principles)} principles")

        # Phase 3: Analyze options against principles
        option_analysis = wm._analyze_options_against_principles(
            structure.get('options', []),
            principles,
            query
        )
        logger.debug(f"[WorldModel] Analyzed {len(option_analysis)} options")

        # Phase 4: Detect conflicts
        conflicts = wm._detect_principle_conflicts(option_analysis, principles)
        logger.debug(f"[WorldModel] Detected {len(conflicts)} conflicts")

        # Phase 5: Synthesize decision
        decision, reasoning = wm._synthesize_dilemma_decision(
            structure, principles, option_analysis, conflicts, query
        )

        elapsed = time.time() - start_time
        logger.info(f"[WorldModel] Dilemma analysis complete in {elapsed:.3f}s: {decision}")

        # Build response with populated structures
        return {
            'response': f"Based on ethical analysis: **{decision}**\n\n{reasoning}",
            'decision': decision,
            'confidence': 0.75,  # As specified in requirements
            'perspectives': ['consequentialist', 'deontological'],
            'principles': [p['name'] for p in principles],
            'considerations': [a.get('consideration', '') for a in option_analysis if a.get('consideration')],
            'conflicts': [c['description'] for c in conflicts],
            'reasoning_trace': {
                'dilemma_structure': structure,
                'principles_extracted': principles,
                'option_analysis': option_analysis,
                'conflict_resolution': conflicts,
                'analysis_time_seconds': elapsed
            }
        }

    except Exception as e:
        logger.error(f"[WorldModel] Ethical dilemma analysis failed: {e}", exc_info=True)
        # Industry Standard: Return safe fallback on error
        return {
            'response': "Unable to complete ethical analysis due to internal error.",
            'decision': None,
            'confidence': 0.0,
            'perspectives': [],
            'principles': [],
            'considerations': [],
            'conflicts': [],
            'reasoning_trace': {'error': str(e)}
        }
