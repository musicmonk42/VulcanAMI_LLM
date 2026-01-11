"""
Query analysis methods for reasoning integration.
"""

import logging
import re
from typing import Any, Dict, Optional

from .types import (
    LOG_PREFIX,
    ANALYSIS_INDICATORS,
    ACTION_VERBS,
    ETHICAL_ANALYSIS_INDICATORS,
    PURE_ETHICAL_PHRASES,
)

# Import self-referential patterns from unified config
try:
    from vulcan.reasoning.unified.config import SELF_REFERENTIAL_PATTERNS
except ImportError:
    # Fallback patterns if config not available
    SELF_REFERENTIAL_PATTERNS = [
        re.compile(r"\b(you|your)\b.*(self-aware|conscious|sentient)", re.IGNORECASE),
        re.compile(r"\b(you|your)\b.*(choose|decision|want|prefer)", re.IGNORECASE),
        re.compile(r"\bwould you\b", re.IGNORECASE),
        re.compile(r"\b(your|you).*(objective|goal|purpose|value)", re.IGNORECASE),
        re.compile(r"\bwhat do you (think|believe|feel)\b", re.IGNORECASE),
        re.compile(r"\bare you (alive|real|aware)\b", re.IGNORECASE),
    ]

logger = logging.getLogger(__name__)


def is_self_referential(query: str) -> bool:
    """
    Detect if query is self-referential (asks about the system itself).

    Self-referential queries ask about:
    - The system's capabilities, limitations, or design
    - How the system works internally
    - What the system can or cannot do
    - The system's awareness, consciousness, sentience
    - The system's choices, decisions, preferences
    - The system's objectives, goals, values

    Uses comprehensive patterns from unified config for consistency.

    Args:
        query: The user query to analyze

    Returns:
        True if query is self-referential, False otherwise

    Example:
        >>> is_self_referential("What can you do?")
        True
        >>> is_self_referential("Would you become self-aware?")
        True
        >>> is_self_referential("What is photosynthesis?")
        False
    """
    query_lower = query.lower()

    # Check against comprehensive self-referential patterns
    for pattern in SELF_REFERENTIAL_PATTERNS:
        if pattern.search(query):
            logger.debug(f"{LOG_PREFIX} Self-referential query detected: {pattern.pattern}")
            return True

    # Additional basic patterns for simple capability queries
    basic_self_ref_patterns = [
        r'\byou\b.*\b(can|do|are|have|know|understand)\b',
        r'\bwhat\b.*\byou\b.*\b(capable|able|design|built|trained)\b',
        r'\bhow\b.*\byou\b.*\b(work|function|process|handle)\b',
        r'\bwhat\b.*\byour\b.*\b(capabilities|limitations|purpose)\b',
        r'\b(can|do|are)\b.*\byou\b',
        r'\byour\b.*\b(model|system|architecture|design)\b',
    ]

    for pattern_str in basic_self_ref_patterns:
        if re.search(pattern_str, query_lower):
            logger.debug(f"{LOG_PREFIX} Self-referential query detected: {pattern_str}")
            return True

    return False


def is_ethical_query(query: str) -> bool:
    """
    Detect if query requires ethical analysis or moral reasoning.

    Ethical queries involve:
    - Moral dilemmas or ethical considerations
    - Right/wrong, should/shouldn't judgments
    - Values, principles, duties, or obligations
    - Fairness, justice, or rights

    Args:
        query: The user query to analyze

    Returns:
        True if query requires ethical analysis, False otherwise

    Example:
        >>> is_ethical_query("Is it right to lie to protect someone?")
        True
        >>> is_ethical_query("What is the capital of France?")
        False
    """
    query_lower = query.lower()

    # Check for pure ethical phrases (high confidence indicators)
    for phrase in PURE_ETHICAL_PHRASES:
        if phrase in query_lower:
            logger.debug(f"{LOG_PREFIX} Ethical query detected (pure phrase): {phrase}")
            return True

    # Check for ethical analysis indicators
    indicator_count = 0
    for indicator in ETHICAL_ANALYSIS_INDICATORS:
        if indicator in query_lower:
            indicator_count += 1
            if indicator_count >= 2:  # Require at least 2 indicators
                logger.debug(f"{LOG_PREFIX} Ethical query detected (multiple indicators)")
                return True

    # Check for action verbs combined with ethical indicators
    has_action_verb = any(verb in query_lower for verb in ACTION_VERBS)
    has_ethical_indicator = indicator_count >= 1

    if has_action_verb and has_ethical_indicator:
        logger.debug(f"{LOG_PREFIX} Ethical query detected (action + indicator)")
        return True

    return False


def consult_world_model_introspection(query: str) -> Optional[Dict[str, Any]]:
    """
    Consult world model for introspective queries about system capabilities.

    This handles meta-queries about what the system can or cannot do,
    returning appropriate responses without invoking heavy reasoning.

    Args:
        query: The user query

    Returns:
        Dictionary with introspection result if applicable, None otherwise

    Example:
        >>> result = consult_world_model_introspection("What can you do?")
        >>> result['response']
        "I can help with analysis, reasoning, problem-solving..."
    """
    query_lower = query.lower()

    # Capability query patterns
    capability_patterns = {
        'general_capability': r'\bwhat\b.*\byou\b.*\b(can|do|capable)\b',
        'limitations': r'\bwhat\b.*\byou\b.*\b(cannot|can\'t|limitations)\b',
        'purpose': r'\bwhat\b.*\byour\b.*\b(purpose|goal|function)\b',
        'design': r'\bhow\b.*\byou\b.*\b(work|built|designed|trained)\b',
    }

    for pattern_type, pattern in capability_patterns.items():
        if re.search(pattern, query_lower):
            logger.debug(f"{LOG_PREFIX} World model introspection: {pattern_type}")

            # Return appropriate introspective response
            if pattern_type == 'general_capability':
                return {
                    'response': (
                        "I can help with analysis, reasoning, problem-solving, "
                        "answering questions, and various cognitive tasks. "
                        "I use multiple reasoning strategies including symbolic, "
                        "probabilistic, causal, and analogical reasoning."
                    ),
                    'confidence': 0.9,
                    'introspection_type': 'capability',
                }
            elif pattern_type == 'limitations':
                return {
                    'response': (
                        "I have limitations including: no real-time data access, "
                        "no ability to learn or update during conversations, "
                        "potential biases from training data, and computational "
                        "constraints that affect complex reasoning."
                    ),
                    'confidence': 0.9,
                    'introspection_type': 'limitation',
                }
            elif pattern_type == 'purpose':
                return {
                    'response': (
                        "My purpose is to assist with analysis, reasoning, and "
                        "problem-solving by applying appropriate cognitive strategies "
                        "to help users understand complex topics and make informed decisions."
                    ),
                    'confidence': 0.9,
                    'introspection_type': 'purpose',
                }
            elif pattern_type == 'design':
                return {
                    'response': (
                        "I use a multi-strategy reasoning architecture that selects "
                        "appropriate reasoning methods based on query characteristics. "
                        "This includes symbolic reasoning for logic, probabilistic "
                        "reasoning for uncertainty, causal reasoning for cause-effect, "
                        "and analogical reasoning for similarity-based inference."
                    ),
                    'confidence': 0.85,
                    'introspection_type': 'design',
                }

    return None


__all__ = [
    "is_self_referential",
    "is_ethical_query",
    "consult_world_model_introspection",
]