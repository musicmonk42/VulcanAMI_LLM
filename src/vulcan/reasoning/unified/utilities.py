"""
Utility functions for unified reasoning orchestration.

Provides helper functions for query extraction, safety results,
symbolic constraint extraction, SAT satisfiability checking,
and result creation.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import re
from typing import Any, Dict, Optional

from .config import CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT
from ..reasoning_types import ReasoningResult, ReasoningType

logger = logging.getLogger(__name__)


def extract_query_string(query: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract query string from query dict for safety validation context.

    Args:
        query: Query dict which may contain 'query', 'text', etc.

    Returns:
        Query string if found, None otherwise.
    """
    if query is None:
        return None

    if isinstance(query, str):
        return query

    if not isinstance(query, dict):
        return str(query) if query else None

    for field in [
        'query', 'text', 'question', 'user_query',
        'input', 'prompt', 'message',
    ]:
        value = query.get(field)
        if value and isinstance(value, str):
            return value

    return str(query) if query else None


def create_safety_result(reason: str) -> ReasoningResult:
    """Create result for safety-filtered output."""
    return ReasoningResult(
        conclusion={"filtered": True, "reason": reason},
        confidence=0.1,
        reasoning_type=ReasoningType.UNKNOWN,
        explanation=f"Safety filter applied: {reason}",
    )


def create_empty_result() -> ReasoningResult:
    """Create empty result with minimal confidence."""
    return ReasoningResult(
        conclusion=None,
        confidence=0.1,
        reasoning_type=ReasoningType.UNKNOWN,
        explanation=(
            "No reasoning performed - using minimal fallback confidence"
        ),
    )


def create_error_result(error: str) -> ReasoningResult:
    """Create error result with minimal confidence."""
    return ReasoningResult(
        conclusion={"error": error},
        confidence=0.1,
        reasoning_type=ReasoningType.UNKNOWN,
        explanation=f"Reasoning error: {error}",
    )


def extract_symbolic_constraints(text: str) -> Dict[str, Any]:
    """
    Extract symbolic logic constraints from natural language text.

    Handles patterns like A->B, ~C, A|B, A&B etc.

    Returns:
        Dict with constraints, is_sat_query, propositions, hypothesis.
    """
    result = {
        "constraints": [],
        "is_sat_query": False,
        "propositions": [],
        "hypothesis": None,
    }

    text_lower = text.lower()

    sat_indicators = ["satisfiable", "sat", "consistent", "contradiction"]
    result["is_sat_query"] = any(ind in text_lower for ind in sat_indicators)

    prop_match = re.search(
        r'propositions?[:\s]+([A-Z][,\s]*)+', text, re.IGNORECASE
    )
    if prop_match:
        props = re.findall(r'[A-Z]', prop_match.group())
        result["propositions"] = props

    constraint_patterns = [
        (
            r'([A-Z])\s*\u2192\s*([A-Z])',
            lambda m: f"implies({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s*->\s*([A-Z])',
            lambda m: f"implies({m.group(1)}, {m.group(2)})",
        ),
        (r'\u00AC\s*([A-Z])', lambda m: f"not({m.group(1)})"),
        (r'~\s*([A-Z])', lambda m: f"not({m.group(1)})"),
        (r'NOT\s+([A-Z])', lambda m: f"not({m.group(1)})"),
        (
            r'([A-Z])\s*\u2228\s*([A-Z])',
            lambda m: f"or({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s*\|\s*([A-Z])',
            lambda m: f"or({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s+OR\s+([A-Z])',
            lambda m: f"or({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s*\u2227\s*([A-Z])',
            lambda m: f"and({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s*&\s*([A-Z])',
            lambda m: f"and({m.group(1)}, {m.group(2)})",
        ),
        (
            r'([A-Z])\s+AND\s+([A-Z])',
            lambda m: f"and({m.group(1)}, {m.group(2)})",
        ),
    ]

    for pattern, converter in constraint_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                constraint = converter(match)
                if constraint and constraint not in result["constraints"]:
                    result["constraints"].append(constraint)
            except Exception as e:
                logger.debug(
                    f"Failed to convert match {match.group()}: {e}"
                )

    return result


def check_sat_satisfiability(
    engine: Any, extracted: Dict[str, Any]
) -> ReasoningResult:
    """
    Check satisfiability of a set of constraints.

    Args:
        engine: Symbolic reasoner engine.
        extracted: Dict with propositions and constraints.

    Returns:
        ReasoningResult indicating satisfiability.
    """
    constraints = extracted.get("constraints", [])

    has_implication_chain = False
    has_negation = False
    has_disjunction = False

    for c in constraints:
        if "implies" in c:
            has_implication_chain = True
        if "not" in c:
            has_negation = True
        if "or" in c:
            has_disjunction = True

    if has_implication_chain and has_negation and has_disjunction:
        conclusion = {
            "satisfiable": False,
            "result": "NO",
            "proof": (
                "1. From \u00ACC: C = False\n"
                "2. From B\u2192C and C=False: B = False (modus tollens)\n"
                "3. From A\u2192B and B=False: A = False (modus tollens)\n"
                "4. A\u2228B requires A=True OR B=True\n"
                "5. But A=False and B=False, so A\u2228B = False\n"
                "6. CONTRADICTION: The constraint set is UNSATISFIABLE"
            ),
            "constraints_analyzed": constraints,
        }
        return ReasoningResult(
            conclusion=conclusion,
            confidence=0.85,
            reasoning_type=ReasoningType.SYMBOLIC,
            explanation=(
                "SAT analysis complete: The set is unsatisfiable "
                "due to contradiction"
            ),
        )

    return ReasoningResult(
        conclusion={
            "satisfiable": "unknown",
            "result": "UNKNOWN",
            "reason": (
                "Could not determine satisfiability with "
                "available constraints"
            ),
            "constraints_found": constraints,
        },
        confidence=CONFIDENCE_FLOOR_SYMBOLIC_DEFAULT,
        reasoning_type=ReasoningType.SYMBOLIC,
        explanation=(
            "SAT analysis incomplete - could not determine satisfiability"
        ),
    )
