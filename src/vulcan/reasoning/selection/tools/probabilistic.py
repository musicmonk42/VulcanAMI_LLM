"""
Probabilistic Tool Wrapper - Adapts ProbabilisticReasoner to the common reason() interface.

The ProbabilisticReasoner uses query() method for Bayesian inference.
This wrapper:
1. Extracts query variable and evidence from problem
2. Executes Bayesian inference
3. Returns probability distribution

Extracted from tool_selector.py to reduce module size.
"""

import re
import logging
import time
from typing import Any, Dict, Optional

from .probabilistic_inference import ProbabilisticInferenceMixin

logger = logging.getLogger(__name__)


class ProbabilisticToolWrapper(ProbabilisticInferenceMixin):
    """
    Wrapper for ProbabilisticReasoner that exposes reason() method.

    The ProbabilisticReasoner uses query() method for Bayesian inference.
    Now properly extracts probability parameters from natural language queries
    instead of passing the query title as the variable.
    """

    # Keywords that indicate probability/statistical queries
    # Used by gate check to reject non-probability queries and prevent P(if) errors
    _PROBABILITY_KEYWORDS = (
        'probability', 'bayes', 'bayesian', 'posterior', 'prior',
        'likelihood', 'sensitivity', 'specificity', 'prevalence',
        'conditional', 'p(', 'e[', 'distribution', 'odds', 'ratio',
        '%', 'percent', 'chance', 'risk', 'uncertainty',
    )

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        if isinstance(engine, str):
            raise TypeError(
                f"ProbabilisticToolWrapper received string '{engine}' instead of ProbabilisticReasoner instance. "
                f"Expected: ProbabilisticReasoner instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "ProbabilisticToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine."
            )
        if not hasattr(engine, 'query'):
            raise AttributeError(
                f"ProbabilisticToolWrapper engine must have 'query()' method. "
                f"Got type: {type(engine).__name__}"
            )
        self.engine = engine
        self.name = "probabilistic"
        self.config = config or {}
        logger.debug(f"[ProbabilisticToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute probabilistic reasoning on the problem.

        FIX #1: Now includes gate check for probability applicability to avoid
        wasting computation on non-probability queries.

        Args:
            problem: Dict with query_var, evidence, and optional rules

        Returns:
            Dict with probability distribution and confidence
        """
        start_time = time.time()

        try:
            # FIX #1: Gate check - is this actually a probability query?
            query_str = self._extract_query_text(problem)

            is_probability_query = False

            if query_str and hasattr(self.engine, '_is_probability_query'):
                is_probability_query = self.engine._is_probability_query(query_str)
            else:
                query_lower = query_str.lower() if query_str else ''
                is_probability_query = any(kw in query_lower for kw in self._PROBABILITY_KEYWORDS)

            if query_str and not is_probability_query:
                logger.info(
                    f"[ProbabilisticEngine] Gate check: Query does not appear to be a probability question "
                    f"(prevents P(if) style errors)"
                )
                return {
                    "tool": self.name,
                    "applicable": False,
                    "reason": "Query does not involve probability concepts",
                    "confidence": 0.0,
                    "engine": "ProbabilisticReasoner",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                }

            # Clear state before each query
            if hasattr(self.engine, 'clear_state'):
                self.engine.clear_state()

            # Extract structured query envelope from problem if available
            query_structure = None
            if isinstance(problem, dict):
                query_structure = problem.get('query_structure')
                if not query_structure and 'context' in problem:
                    context = problem.get('context', {})
                    if isinstance(context, dict):
                        query_structure = context.get('query_structure')

            # Try Bayesian calculation first for explicit probability queries
            bayes_result = self._try_bayesian_calculation(problem, query_structure=query_structure)
            if bayes_result is not None:
                return bayes_result

            # Parse the problem
            query_var, evidence, rules = self._parse_problem(problem)

            if not query_var:
                return self._error_result("No query variable provided")

            logger.info(
                f"[ProbabilisticEngine] Computing P({query_var} | evidence={evidence})"
            )

            # Add any rules to the engine
            for rule in rules:
                confidence = rule.get("confidence", 0.9) if isinstance(rule, dict) else 0.9
                rule_str = rule.get("rule", rule) if isinstance(rule, dict) else rule
                self.engine.add_rule(rule_str, confidence)

            # Execute the Bayesian inference
            result = self.engine.query(query_var, evidence)

            execution_time = (time.time() - start_time) * 1000

            # Extract probability
            if isinstance(result, dict):
                prob_true = result.get(True, 0.5)
                prob_false = result.get(False, 0.5)
            else:
                prob_true = float(result) if result else 0.5
                prob_false = 1.0 - prob_true

            logger.info(
                f"[ProbabilisticEngine] Result: P({query_var}=True)={prob_true:.4f}, "
                f"time={execution_time:.0f}ms"
            )

            return {
                "tool": self.name,
                "result": result,
                "probability": prob_true,
                "posterior": prob_true,
                "distribution": {True: prob_true, False: prob_false},
                "confidence": max(prob_true, prob_false),
                "query_var": query_var,
                "evidence": evidence,
                "execution_time_ms": execution_time,
                "engine": "ProbabilisticReasoner",
            }

        except Exception as e:
            logger.error(f"[ProbabilisticEngine] Reasoning failed: {e}", exc_info=True)
            return self._error_result(str(e))

    def _extract_query_text(self, problem: Any) -> str:
        """
        Extract meaningful query text from problem for keyword detection.

        Handles various problem formats:
        - String: return as-is
        - Dict: extract text/query/content fields
        - Object: try to get text representation from known attributes

        Returns:
            Extracted query text, or empty string if no meaningful text found
        """
        if isinstance(problem, str):
            return problem

        if isinstance(problem, dict):
            text_fields = ['text', 'query', 'content', 'message', 'question', 'input']
            for field_name in text_fields:
                if field_name in problem and isinstance(problem[field_name], str):
                    return problem[field_name]

            if 'problem' in problem:
                return self._extract_query_text(problem['problem'])

            skip_keys = {'id', 'timestamp', 'metadata', 'config', 'settings'}
            text_parts = []
            for key, value in problem.items():
                if key not in skip_keys and isinstance(value, str):
                    text_parts.append(value)
            return ' '.join(text_parts) if text_parts else ''

        text_attrs = ['text', 'query', 'content', 'message', 'question']
        for attr in text_attrs:
            if hasattr(problem, attr):
                value = getattr(problem, attr)
                if isinstance(value, str):
                    return value

        str_repr = str(problem)
        if not str_repr.startswith('<') or 'object at 0x' not in str_repr:
            return str_repr

        return ''

    def _error_result(self, error: str) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "result": None,
            "confidence": 0.1,
            "error": error,
            "engine": "ProbabilisticReasoner",
        }
