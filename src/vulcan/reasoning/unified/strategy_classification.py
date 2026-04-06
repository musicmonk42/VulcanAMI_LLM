"""
Reasoning task classification for unified reasoning orchestration.

Heuristic-based classifier to select the most appropriate reasoning type
based on features of the input data and query.

Extracted from orchestrator.py for modularity.
Sub-module: strategy_classification_heuristics.

Author: VulcanAMI Team
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .component_loader import _load_reasoning_components
from .strategy_classification_heuristics import (
    apply_keyword_scores,
    apply_query_key_boosts,
    apply_specific_pattern_boosts,
    map_tool_name_to_reasoning_type,
    select_portfolio_reasoners,
)
from ..reasoning_types import ReasoningType

# Re-export for backward compatibility (used by strategy_planning)
__all__ = [
    "classify_reasoning_task",
    "determine_reasoning_type",
    "map_tool_name_to_reasoning_type",
    "select_portfolio_reasoners",
]

# Pre-compiled regex patterns (module-level for performance)
MATH_EXPRESSION_PATTERN = re.compile(r'\d+\s*[+\-*/^]\s*\d+')
MATH_QUERY_PATTERN = re.compile(
    r'(?:what\s+is|calculate|compute|evaluate)\s+\d+\s*[+\-*/^]\s*\d+',
    re.IGNORECASE,
)
MATH_SYMBOLS_PATTERN = re.compile(
    r'[+\u2211\u222B\u2202\u2200\u2203\u2208\u222A\u2229\u2282\u2286'
    r'\u2287\u2283\u2205\u221E\u03C0\u220F\u221A\u00B1\u2264\u2265'
    r'\u2260\u2248\u00D7\u00F7\u2207\u0394]|'
    r'\\(?:sum|int|partial|forall|exists|infty|pi|prod|sqrt|nabla|delta)|'
    r'\b(?:sum|integral|derivative|limit|forall|exists)\b',
    re.IGNORECASE | re.UNICODE,
)
PROBABILITY_NOTATION_PATTERN = re.compile(
    r'P\s*\([^)]+\)|'
    r'P\s*\([^)]+\s*[|\u2223]\s*[^)]+\)|'
    r'Pr\s*\([^)]+\)|'
    r'E\s*\[[^\]]+\]|'
    r'Var\s*\([^)]+\)',
    re.IGNORECASE,
)
INDUCTION_PATTERN = re.compile(
    r'\b(?:prove|verify|show)\s+by\s+induction\b|'
    r'\bbase\s+case\b|'
    r'\binductive\s+(?:step|hypothesis)\b|'
    r'\b(?:assume|given)\s+.*\s+(?:prove|show)\b',
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def classify_reasoning_task(
    input_data: Any, query: Dict[str, Any]
) -> ReasoningType:
    """
    Heuristic-based classifier to select the most appropriate reasoning type.

    Checks keywords in BOTH input_data AND query dict.

    Args:
        input_data: The input data to classify.
        query: Query dictionary with metadata.

    Returns:
        Best-matching ReasoningType.
    """
    scores = defaultdict(float)
    query_str = str(query).lower()

    input_str = ""
    if isinstance(input_data, str):
        input_str = input_data.lower()
    elif isinstance(input_data, dict):
        for key in ['query', 'text', 'problem', 'question', 'input']:
            if key in input_data and isinstance(input_data[key], str):
                input_str = input_data[key].lower()
                break

    combined_str = query_str + " " + input_str

    if isinstance(input_data, (list, tuple, np.ndarray)):
        try:
            arr = np.array(input_data)
            if np.issubdtype(arr.dtype, np.number):
                scores[ReasoningType.PROBABILISTIC] += 0.6
        except Exception as e:
            logger.debug(f"Failed to check numeric data type: {e}")

    if isinstance(input_data, str):
        scores[ReasoningType.SYMBOLIC] += 0.2
        if any(op in input_data for op in [" AND ", " OR ", " NOT ", "=>"]):
            scores[ReasoningType.SYMBOLIC] += 0.4

        if MATH_EXPRESSION_PATTERN.search(input_data):
            scores[ReasoningType.MATHEMATICAL] += 0.8
        if MATH_QUERY_PATTERN.search(input_data):
            scores[ReasoningType.MATHEMATICAL] += 0.9
        if MATH_SYMBOLS_PATTERN.search(input_data):
            scores[ReasoningType.MATHEMATICAL] += 1.0
            logger.debug("[Classifier] Advanced math notation detected")
        if INDUCTION_PATTERN.search(input_data):
            scores[ReasoningType.MATHEMATICAL] += 0.7
            logger.debug("[Classifier] Induction pattern detected")
        if PROBABILITY_NOTATION_PATTERN.search(input_data):
            bayes_indicators = [
                "sensitivity", "specificity", "prevalence",
                "test", "diagnostic",
            ]
            is_bayesian = any(
                ind in input_data.lower() for ind in bayes_indicators
            )
            if not is_bayesian:
                scores[ReasoningType.MATHEMATICAL] += 0.6
    elif isinstance(input_data, dict):
        if any(
            key in input_data
            for key in ["graph", "nodes", "edges", "evidence"]
        ):
            scores[ReasoningType.CAUSAL] += 0.5
            scores[ReasoningType.PROBABILISTIC] += 0.2

    # Delegate to heuristics helpers
    apply_keyword_scores(combined_str, scores)
    apply_query_key_boosts(query, scores)

    if (
        isinstance(input_data, str)
        and len(input_data) > 200
        and "generate" in combined_str
    ):
        scores[ReasoningType.SYMBOLIC] += 0.5

    # Enhanced detection for specific problem types
    apply_specific_pattern_boosts(combined_str, scores)

    if not scores or max(scores.values()) < 0.3:
        return ReasoningType.PROBABILISTIC

    best_type = max(scores, key=scores.get)
    logger.debug(
        f"Reasoning type classifier scores: {dict(scores)}. "
        f"Selected: {best_type}"
    )
    return best_type


def determine_reasoning_type(
    reasoner: Any, input_data: Any, query: Optional[Dict[str, Any]]
) -> ReasoningType:
    """
    Automatically determine appropriate reasoning type.

    Args:
        reasoner: UnifiedReasoner instance (for accessing reasoners dict).
        input_data: The input data.
        query: Optional query dictionary.

    Returns:
        Best-matching ReasoningType.
    """
    reasoning_components = _load_reasoning_components()
    ModalityType = reasoning_components.get("ModalityType")

    if isinstance(input_data, dict) and ModalityType:
        modality_types = [
            k for k in input_data.keys() if isinstance(k, ModalityType)
        ]
        if len(modality_types) > 1:
            return ReasoningType.MULTIMODAL

    return classify_reasoning_task(input_data, query or {})
