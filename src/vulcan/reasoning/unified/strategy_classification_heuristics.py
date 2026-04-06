"""
Heuristic helpers for reasoning task classification.

Keyword-based scoring, tool-name-to-ReasoningType mapping,
and portfolio reasoner selection logic.
Extracted from strategy_classification.py for modularity.

Author: VulcanAMI Team
"""

import logging
from typing import Any, Dict, List, Optional

from ..reasoning_types import ReasoningType

logger = logging.getLogger(__name__)

# Keyword map used by the classifier to score reasoning types.
KEYWORD_MAP: Dict[ReasoningType, List[str]] = {
    ReasoningType.PROBABILISTIC: [
        "probability", "likelihood", "chance", "distribution", "threshold",
    ],
    ReasoningType.CAUSAL: [
        "cause", "effect", "why", "impact", "influence", "reason",
    ],
    ReasoningType.SYMBOLIC: [
        "prove", "logic", "valid", "theorem", "deduce", "consistent",
        "generate", "summarize", "explain", "text", "narrative", "story",
    ],
    ReasoningType.ANALOGICAL: [
        "similar", "analogy", "like", "resembles", "comparison",
    ],
    ReasoningType.COUNTERFACTUAL: [
        "what if", "counterfactual", "had not",
    ],
    ReasoningType.MULTIMODAL: [
        "image", "video", "audio", "multimodal",
    ],
    ReasoningType.MATHEMATICAL: [
        "calculate", "compute", "solve", "evaluate", "simplify",
        "factor", "integrate", "differentiate", "derivative",
        "integral", "equation", "expression", "formula",
        "arithmetic", "algebra", "calculus", "math", "sum",
        "product", "divide", "multiply", "add", "subtract",
        "plus", "minus", "times", "equals", "+", "-", "*", "/",
        "^", "**", "sqrt", "square root", "power", "exponent",
        "logarithm", "log", "sin", "cos", "tan", "matrix",
        "determinant", "eigenvalue", "polynomial", "quadratic",
        "linear", "numerical",
    ],
    ReasoningType.PHILOSOPHICAL: [
        "ethical", "moral", "permissible", "obligatory", "forbidden",
        "duty", "ought", "should", "right", "wrong", "virtue",
        "justice", "fairness", "deontological", "utilitarian",
        "consequentialist", "kantian", "dilemma", "normative",
        "deontic",
    ],
}

# Tool-name-to-ReasoningType mapping dictionary.
TOOL_NAME_MAPPING: Dict[str, ReasoningType] = {
    'mathematical': ReasoningType.MATHEMATICAL,
    'math': ReasoningType.MATHEMATICAL,
    'mathematical_computation': ReasoningType.MATHEMATICAL,
    'symbolic': ReasoningType.SYMBOLIC,
    'logic': ReasoningType.SYMBOLIC,
    'symbolic_reasoning': ReasoningType.SYMBOLIC,
    'fol_solver': ReasoningType.SYMBOLIC,
    'probabilistic': ReasoningType.PROBABILISTIC,
    'probability': ReasoningType.PROBABILISTIC,
    'probabilistic_reasoning': ReasoningType.PROBABILISTIC,
    'causal': ReasoningType.CAUSAL,
    'cause': ReasoningType.CAUSAL,
    'causal_reasoning': ReasoningType.CAUSAL,
    'dag_analyzer': ReasoningType.CAUSAL,
    'analogical': ReasoningType.ANALOGICAL,
    'analogy': ReasoningType.ANALOGICAL,
    'analogical_reasoning': ReasoningType.ANALOGICAL,
    'multimodal': ReasoningType.MULTIMODAL,
    'multi_modal': ReasoningType.MULTIMODAL,
    'philosophical': ReasoningType.PHILOSOPHICAL,
    'philosophy': ReasoningType.PHILOSOPHICAL,
    'ethical': ReasoningType.PHILOSOPHICAL,
    'world_model': ReasoningType.PHILOSOPHICAL,
    'worldmodel': ReasoningType.PHILOSOPHICAL,
    'meta_reasoning': ReasoningType.PHILOSOPHICAL,
    'ethics': ReasoningType.PHILOSOPHICAL,
    'moral': ReasoningType.PHILOSOPHICAL,
}


def apply_keyword_scores(
    combined_str: str, scores: Dict[ReasoningType, float]
) -> None:
    """Apply keyword-based scoring from KEYWORD_MAP."""
    for r_type, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in combined_str:
                scores[r_type] += 0.3


def apply_query_key_boosts(
    query: Dict[str, Any], scores: Dict[ReasoningType, float]
) -> None:
    """Apply score boosts based on specific query dictionary keys."""
    if any(key in query for key in ["treatment", "intervention", "action"]):
        scores[ReasoningType.CAUSAL] += 0.5
        if "outcome" in query:
            scores[ReasoningType.CAUSAL] += 0.2
    if "hypothesis" in query:
        scores[ReasoningType.SYMBOLIC] += 0.4
    if "source_problem" in query or "target_problem" in query:
        scores[ReasoningType.ANALOGICAL] += 0.5
    if "factual_state" in query and "intervention" in query:
        scores[ReasoningType.COUNTERFACTUAL] += 0.7


def apply_specific_pattern_boosts(
    combined_str: str, scores: Dict[ReasoningType, float]
) -> None:
    """Apply pattern-based score boosts for specific problem types."""
    sat_patterns = [
        "satisfiable", "satisfiability", "sat", "propositions",
        "constraints", "a \u2192 b", "a->b", "\u00AC", "\u2228", "\u2227",
    ]
    if any(p in combined_str for p in sat_patterns):
        scores[ReasoningType.SYMBOLIC] += 0.5

    bayes_patterns = [
        "sensitivity", "specificity", "prevalence", "p(x|", "bayes",
        "posterior", "prior probability", "base rate",
    ]
    if any(p in combined_str for p in bayes_patterns):
        scores[ReasoningType.PROBABILISTIC] += 0.8
        scores[ReasoningType.MATHEMATICAL] -= 0.5

    causal_patterns = [
        "confound", "randomize", "causal effect", "treatment effect",
        "intervention", "s\u2192d", "s->d", "causal graph",
    ]
    if any(p in combined_str for p in causal_patterns):
        scores[ReasoningType.CAUSAL] += 0.6

    analogy_patterns = [
        "map the", "structure mapping", "domain s", "domain t",
        "analogs", "map from", "mapping between", "deep structure",
    ]
    if any(p in combined_str for p in analogy_patterns):
        scores[ReasoningType.ANALOGICAL] += 0.7

    proof_patterns = [
        "verify each step", "valid or invalid", "proof sketch",
        "claim:", "step 1", "step 2",
    ]
    if any(p in combined_str for p in proof_patterns):
        scores[ReasoningType.SYMBOLIC] += 0.5

    ethics_patterns = [
        "permissible", "forbidden", "harm to innocents", "deontic",
        "nonzero probability of", "rule:",
    ]
    if any(p in combined_str for p in ethics_patterns):
        scores[ReasoningType.PHILOSOPHICAL] += 0.6

    fol_patterns = [
        "first-order logic", "quantifier scope", "formalization",
        "\u2200", "\u2203", "forall", "exists"]
    if any(p in combined_str for p in fol_patterns):
        scores[ReasoningType.SYMBOLIC] += 0.5


def map_tool_name_to_reasoning_type(
    tool_name: str,
) -> Optional[ReasoningType]:
    """
    Map tool name string to ReasoningType enum.

    Args:
        tool_name: Tool name string.

    Returns:
        Corresponding ReasoningType, or None if not found.
    """
    tool_name_lower = tool_name.lower().strip()

    if tool_name_lower in TOOL_NAME_MAPPING:
        return TOOL_NAME_MAPPING[tool_name_lower]

    try:
        for reasoning_type in ReasoningType:
            if reasoning_type.value == tool_name_lower:
                return reasoning_type
    except Exception as e:
        logger.debug(
            f"Failed to match tool name '{tool_name}' to "
            f"ReasoningType: {e}"
        )

    logger.warning(
        f"[Orchestrator] Unknown tool name: '{tool_name}' - "
        "no ReasoningType mapping found"
    )
    return None


def select_portfolio_reasoners(
    reasoner: Any, task: Any
) -> List[ReasoningType]:
    """
    Select complementary reasoners for portfolio.

    Args:
        reasoner: UnifiedReasoner instance.
        task: ReasoningTask.

    Returns:
        List of ReasoningType to include in portfolio.
    """
    portfolio = []

    if ReasoningType.PROBABILISTIC in reasoner.reasoners:
        portfolio.append(ReasoningType.PROBABILISTIC)

    if task.query:
        query_str = str(task.query).lower()

        if "cause" in query_str or "effect" in query_str:
            if ReasoningType.CAUSAL in reasoner.reasoners:
                portfolio.append(ReasoningType.CAUSAL)

        if "prove" in query_str or "logic" in query_str:
            if ReasoningType.SYMBOLIC in reasoner.reasoners:
                portfolio.append(ReasoningType.SYMBOLIC)

        if "similar" in query_str or "analogy" in query_str:
            if ReasoningType.ANALOGICAL in reasoner.reasoners:
                portfolio.append(ReasoningType.ANALOGICAL)

        if any(
            kw in query_str
            for kw in ("generate", "summarize", "explain")
        ):
            if ReasoningType.SYMBOLIC in reasoner.reasoners:
                portfolio.append(ReasoningType.SYMBOLIC)

    max_size = 3
    if task.constraints.get("time_budget_ms", float("inf")) < 2000:
        max_size = 2

    return portfolio[:max_size]
