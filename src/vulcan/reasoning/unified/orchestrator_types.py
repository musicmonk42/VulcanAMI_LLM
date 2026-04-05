"""
Orchestrator Types and Module-Level Utilities

Pre-compiled regex patterns and standalone helper functions extracted from
orchestrator.py for reuse and testability.

These are module-level constants and functions that do not depend on
UnifiedReasoner instance state.

Author: VulcanAMI Team
"""

import os
import re
from typing import Any

from .config import CREATIVE_TASK_KEYWORDS
from .types import ReasoningTask


# ==============================================================================
# REGEX PATTERNS FOR MATHEMATICAL DETECTION
# ==============================================================================
# Pre-compiled regex patterns for mathematical expression detection.
# Used in _classify_reasoning_task() to identify mathematical queries.
#
# ENHANCED (Jan 2026): Added support for advanced mathematical notation:
# - Summation: sum, \sum
# - Integration: integral, \int
# - Derivatives: partial, \partial, d/dx
# - Probability notation: P(X|Y), P(X)
# - Logical quantifiers: forall, exists
# - Set notation: element-of, union, intersection, subset, superset
#
# Root Cause Fix: Original patterns only matched simple arithmetic (2+2)
# but failed to detect advanced mathematical queries like:
# - "Compute sum_{k=1}^n (2k-1)"
# - "Calculate integral x^2 dx"
# - "Find P(X|+) given sensitivity/specificity"
#
# Industry Standards Applied:
# - Unicode support for mathematical symbols
# - Case-insensitive matching
# - Comprehensive pattern coverage
# - Performance-optimized pre-compilation
# ==============================================================================

# Basic arithmetic expressions (2+2, 3*4, etc.)
MATH_EXPRESSION_PATTERN = re.compile(r'\d+\s*[+\-*/^]\s*\d+')

# Mathematical query phrases with arithmetic
MATH_QUERY_PATTERN = re.compile(
    r'(?:what\s+is|calculate|compute|evaluate)\s+\d+\s*[+\-*/^]\s*\d+',
    re.IGNORECASE,
)

# Advanced mathematical notation (ADDED: Jan 2026)
MATH_SYMBOLS_PATTERN = re.compile(
    r'[\u2211\u222b\u2202\u2200\u2203\u2208\u222a\u2229\u2282\u2286\u2287\u2283\u2205\u221e\u03c0\u220f\u221a\u00b1\u2264\u2265\u2260\u2248\u00d7\u00f7\u2207\u0394]|'  # Unicode math symbols
    r'\\(?:sum|int|partial|forall|exists|infty|pi|prod|sqrt|nabla|delta)|'  # LaTeX commands
    r'\b(?:sum|integral|derivative|limit|forall|exists)\b',  # English keywords
    re.IGNORECASE | re.UNICODE,
)

# Probability notation: P(X), P(X|Y), Pr(A), etc.
# Note: Pattern supports both ASCII pipe '|' (U+007C) and mathematical
# vertical bar '\u2223' (U+2223) for conditional probability notation.
PROBABILITY_NOTATION_PATTERN = re.compile(
    r'P\s*\([^)]+\)|'  # P(X), P(Disease)
    r'P\s*\([^)]+\s*[|\u2223]\s*[^)]+\)|'  # P(X|Y), P(Disease|Test+)
    r'Pr\s*\([^)]+\)|'  # Pr(X) - alternative notation
    r'E\s*\[[^\]]+\]|'  # E[X] - expected value
    r'Var\s*\([^)]+\)',  # Var(X) - variance
    re.IGNORECASE,
)

# Induction proof patterns
INDUCTION_PATTERN = re.compile(
    r'\b(?:prove|verify|show)\s+by\s+induction\b|'
    r'\bbase\s+case\b|'
    r'\binductive\s+(?:step|hypothesis)\b|'
    r'\b(?:assume|given)\s+.*\s+(?:prove|show)\b',
    re.IGNORECASE,
)


def is_test_environment() -> bool:
    """
    Check if we're running in a test environment.

    Returns:
        True if running under pytest or unittest.
    """
    return (
        "pytest" in str(os.getenv("_", ""))
        or "pytest" in str(os.getenv("PYTEST_CURRENT_TEST", ""))
        or "unittest" in str(os.getenv("_", ""))
    )


def is_creative_task(task: ReasoningTask) -> bool:
    """
    Check if a task represents a creative task that should skip confidence filtering.

    Creative tasks (writing poems, stories, etc.) may have lower confidence scores
    due to their subjective nature, but should not be filtered out.

    Args:
        task: The reasoning task to check

    Returns:
        True if the task is creative and should skip confidence filtering

    Examples:
        >>> from vulcan.reasoning.unified.types import ReasoningTask
        >>> task = ReasoningTask(query="Write a poem about love")
        >>> is_creative_task(task)
        True

        >>> task = ReasoningTask(query="Calculate 2+2")
        >>> is_creative_task(task)
        False
    """
    # Extract query string from task
    query_str = ""
    if isinstance(task.query, str):
        query_str = task.query.lower()
    elif isinstance(task.query, dict):
        query_str = str(task.query.get("query", "")).lower()
        query_str += str(task.query.get("text", "")).lower()

    if isinstance(task.input_data, str):
        query_str += task.input_data.lower()

    # Check for creative keywords
    words = query_str.split()
    if words:
        first_word = words[0].rstrip(",.!?")
        if first_word in CREATIVE_TASK_KEYWORDS:
            return True

    # Check for creative noun indicators
    creative_nouns = {
        "poem", "sonnet", "haiku", "story", "essay",
        "song", "lyrics", "script", "novel",
    }
    return any(noun in query_str for noun in creative_nouns)
