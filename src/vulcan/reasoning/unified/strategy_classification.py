"""
Reasoning task classification for unified reasoning orchestration.

Heuristic-based classifier to select the most appropriate reasoning type
based on features of the input data and query. Also includes tool name
mapping and portfolio reasoner selection.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .component_loader import _load_reasoning_components
from ..reasoning_types import ReasoningType

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

    keyword_map = {
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
    for r_type, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in combined_str:
                scores[r_type] += 0.3

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

    if (
        isinstance(input_data, str)
        and len(input_data) > 200
        and "generate" in combined_str
    ):
        scores[ReasoningType.SYMBOLIC] += 0.5

    # Enhanced detection for specific problem types
    _apply_specific_pattern_boosts(combined_str, scores)

    if not scores or max(scores.values()) < 0.3:
        return ReasoningType.PROBABILISTIC

    best_type = max(scores, key=scores.get)
    logger.debug(
        f"Reasoning type classifier scores: {dict(scores)}. "
        f"Selected: {best_type}"
    )
    return best_type


def _apply_specific_pattern_boosts(
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
        "\u2200", "\u2203", "forall", "exists",
    ]
    if any(p in combined_str for p in fol_patterns):
        scores[ReasoningType.SYMBOLIC] += 0.5


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

    tool_mapping = {
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

    if tool_name_lower in tool_mapping:
        return tool_mapping[tool_name_lower]

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
