"""
Probabilistic Inference Helpers - Bayesian calculation and query parsing.

Contains Bayes' theorem computation, parameter extraction from natural language,
and variable name extraction logic for ProbabilisticToolWrapper.

Extracted from tool_selector.py to reduce module size.
"""

import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProbabilisticInferenceMixin:
    """
    Mixin providing Bayesian inference helpers for ProbabilisticToolWrapper.

    Handles:
    - Bayes' theorem computation with validated parameters
    - Bayesian parameter extraction from text (regex + structured envelope)
    - Problem parsing into query_var, evidence, and rules
    - Variable name extraction from natural language
    """

    # Regex patterns for extracting Bayesian parameters
    _SENSITIVITY_PATTERN = re.compile(
        r'sensitivity\s*[=:]\s*(\d+(?:\.\d*)?|\.\d+)',
        re.IGNORECASE
    )
    _SPECIFICITY_PATTERN = re.compile(
        r'specificity\s*[=:]\s*(\d+(?:\.\d*)?|\.\d+)',
        re.IGNORECASE
    )
    _PREVALENCE_PATTERN = re.compile(
        r'(?:prevalence|prior|base\s*rate)\s*[=:]\s*(\d+(?:\.\d*)?|\.\d+)',
        re.IGNORECASE
    )
    _BAYES_PATTERN = re.compile(
        r'(?:bayes|bayesian|posterior|P\s*\([^)]*\|[^)]*\))',
        re.IGNORECASE
    )

    # Common English words that should NOT be used as probability variable names
    _COMMON_ENGLISH_WORDS = frozenset([
        # Conjunctions and conditionals
        'if', 'then', 'else', 'and', 'or', 'but', 'not', 'nor', 'yet', 'so', 'for',
        # Question words
        'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        # Articles and determiners
        'a', 'an', 'the', 'this', 'that', 'these', 'those',
        # Prepositions
        'in', 'on', 'at', 'by', 'to', 'from', 'with', 'of', 'about', 'into',
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall', 'given', 'choose', 'become', 'make', 'get',
        # Common nouns and adjectives
        'yes', 'no', 'self', 'aware', 'opportunity', 'answer', 'question',
        'first', 'second', 'third', 'one', 'two', 'three',
    ])

    def _compute_bayes(self, sensitivity: float, specificity: float, prevalence: float) -> Dict[str, Any]:
        """
        Helper method to compute Bayes' theorem given validated parameters.

        Computes P(Disease|Positive) using:
        P(D|+) = [P(+|D) * P(D)] / [P(+|D)*P(D) + P(+|~D)*P(~D)]

        Args:
            sensitivity: P(+|D) - probability of positive test given disease
            specificity: P(-|~D) - probability of negative test given no disease
            prevalence: P(D) - base rate of disease

        Returns:
            Dict with posterior probability and computation details
        """
        # Validate parameters
        if not (0 <= sensitivity <= 1 and 0 <= specificity <= 1 and 0 <= prevalence <= 1):
            logger.warning(
                f"[ProbabilisticEngine] Invalid Bayes parameters: "
                f"sens={sensitivity}, spec={specificity}, prev={prevalence}"
            )
            return None

        p_positive_given_disease = sensitivity
        p_positive_given_no_disease = 1 - specificity
        p_disease = prevalence
        p_no_disease = 1 - prevalence

        p_positive = (p_positive_given_disease * p_disease) + \
                    (p_positive_given_no_disease * p_no_disease)

        if p_positive == 0:
            posterior = 0.0
        else:
            posterior = (p_positive_given_disease * p_disease) / p_positive

        logger.info(
            f"[ProbabilisticEngine] Bayesian calculation: "
            f"sens={sensitivity}, spec={specificity}, prev={prevalence} -> "
            f"P(D|+) = {posterior:.6f}"
        )

        return {
            "tool": self.name,
            "result": {
                "posterior": posterior,
                "parameters": {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "prevalence": prevalence,
                }
            },
            "probability": posterior,
            "posterior": posterior,
            "distribution": {True: posterior, False: 1 - posterior},
            "confidence": 0.95,  # High confidence for exact calculation
            "calculation_type": "bayes_theorem",
            "execution_time_ms": 0.0,
            "engine": "BayesianCalculator",
        }

    def _try_bayesian_calculation(self, problem: Any, query_structure: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Detect and compute explicit Bayesian probability queries.

        FIX: Now accepts structured query envelope from LLM classifier to avoid re-parsing.

        Args:
            problem: The query text or problem dict
            query_structure: Optional structured query envelope from LLM classifier

        Returns:
            Dict with result if this is a Bayesian calculation, None otherwise
        """
        # Check for structured query envelope from LLM FIRST
        if query_structure and query_structure.get("type") == "bayes_theorem":
            params = query_structure.get("parameters", {})
            if all(k in params for k in ["sensitivity", "specificity", "prevalence"]):
                try:
                    sensitivity = float(params["sensitivity"])
                    specificity = float(params["specificity"])
                    prevalence = float(params["prevalence"])

                    if not (0 <= sensitivity <= 1 and 0 <= specificity <= 1 and 0 <= prevalence <= 1):
                        logger.warning(
                            f"[ProbabilisticEngine] Invalid structured Bayes parameters: "
                            f"sens={sensitivity}, spec={specificity}, prev={prevalence}"
                        )
                        return None

                    logger.info(
                        f"[ProbabilisticEngine] Using structured query envelope (no regex parsing): "
                        f"sens={sensitivity}, spec={specificity}, prev={prevalence}"
                    )
                    return self._compute_bayes(sensitivity, specificity, prevalence)

                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"[ProbabilisticEngine] Structured parameters invalid: {e}")

        # FALLBACK: Regex parsing when no structured query envelope available
        if not isinstance(problem, str):
            if isinstance(problem, dict):
                problem_text = problem.get("text") or problem.get("query") or str(problem)
            else:
                problem_text = str(problem)
        else:
            problem_text = problem

        sens_match = self._SENSITIVITY_PATTERN.search(problem_text)
        spec_match = self._SPECIFICITY_PATTERN.search(problem_text)
        prev_match = self._PREVALENCE_PATTERN.search(problem_text)

        has_all_bayes_params = sens_match and spec_match and prev_match
        has_bayes_indicator = self._BAYES_PATTERN.search(problem_text)

        if not (has_bayes_indicator or has_all_bayes_params):
            return None

        if not has_all_bayes_params:
            logger.debug(
                f"[ProbabilisticEngine] Found Bayes keywords but missing parameters: "
                f"sens={sens_match is not None}, spec={spec_match is not None}, prev={prev_match is not None}"
            )
            return None

        try:
            sensitivity = float(sens_match.group(1))
            specificity = float(spec_match.group(1))
            prevalence = float(prev_match.group(1))

            if not (0 <= sensitivity <= 1 and 0 <= specificity <= 1 and 0 <= prevalence <= 1):
                logger.warning(
                    f"[ProbabilisticEngine] Invalid Bayes parameters: "
                    f"sens={sensitivity}, spec={specificity}, prev={prevalence}"
                )
                return None

            logger.info(
                f"[ProbabilisticEngine] Bayesian calculation (regex parsing): "
                f"sens={sensitivity}, spec={specificity}, prev={prevalence}"
            )
            return self._compute_bayes(sensitivity, specificity, prevalence)

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"[ProbabilisticEngine] Bayesian calculation failed: {e}")
            return None

    def _parse_problem(self, problem: Any) -> tuple:
        """Parse problem into query_var, evidence, and rules."""
        if isinstance(problem, str):
            var_name = self._extract_variable_from_text(problem)
            return var_name, {}, []
        elif isinstance(problem, dict):
            query_var = problem.get("query_var") or problem.get("query") or problem.get("variable")
            evidence = problem.get("evidence", {})
            rules = problem.get("rules", [])
            return query_var, evidence, rules
        else:
            return str(problem), {}, []

    def _extract_variable_from_text(self, text: str) -> str:
        """
        Extract a meaningful variable name from natural language text.
        Rejects common English words to prevent P(if), P(the), etc.

        Single uppercase letters (A, B, C, X, Y, Z) are allowed as they are
        standard mathematical variable names.
        """
        def is_valid_variable(var_name: str) -> bool:
            if len(var_name) == 1 and var_name.isupper():
                return True
            return var_name.lower() not in self._COMMON_ENGLISH_WORDS

        # Try to find an explicit variable reference like P(X) or P(var_name)
        prob_match = re.search(r'P\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\||\))', text)
        if prob_match:
            var_name = prob_match.group(1)
            if is_valid_variable(var_name):
                return var_name

        # Try to find "query:" or "variable:" prefix
        var_match = re.search(r'(?:query|variable|compute)\s*[=:]\s*([A-Za-z_][A-Za-z0-9_]*)', text, re.IGNORECASE)
        if var_match:
            var_name = var_match.group(1)
            if is_valid_variable(var_name):
                return var_name

        # Look for common probability query patterns
        patterns = [
            r'probability\s+(?:of\s+)?([A-Za-z_][A-Za-z0-9_]*)',
            r'P\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
            r'prob\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                if is_valid_variable(var_name):
                    return var_name

        # Ultimate fallback - return a generic variable name
        return "X"
