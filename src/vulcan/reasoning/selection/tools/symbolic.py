"""
Symbolic Tool Wrapper - Adapts SymbolicReasoner to the common reason() interface.

The SymbolicReasoner uses query() method for theorem proving.
This wrapper:
1. Extracts query from problem (string or dict)
2. Calls the SAT/theorem prover
3. Returns result in expected format

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

from .symbolic_helpers import SymbolicPreprocessingMixin

logger = logging.getLogger(__name__)


class SymbolicToolWrapper(SymbolicPreprocessingMixin):
    """
    Wrapper for SymbolicReasoner that exposes reason() method.

    The SymbolicReasoner uses query() method for theorem proving.
    This wrapper:
    1. Extracts query from problem (string or dict)
    2. Calls the SAT/theorem prover
    3. Returns result in expected format
    """

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        # FAIL-FAST: Raise explicit error if wrong type is passed instead of silent failure
        if isinstance(engine, str):
            raise TypeError(
                f"SymbolicToolWrapper received string '{engine}' instead of SymbolicReasoner instance. "
                f"This indicates a bug in the engine initialization code. "
                f"Expected: SymbolicReasoner instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "SymbolicToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine. "
                "Check engine initialization in _initialize_real_engines()."
            )
        # Type safety: Verify engine has required query() method
        if not hasattr(engine, 'query'):
            raise AttributeError(
                f"SymbolicToolWrapper engine must have 'query()' method. "
                f"Got type: {type(engine).__name__}, "
                f"Available methods: {[m for m in dir(engine) if not m.startswith('_')]}"
            )
        self.engine = engine
        self.name = "symbolic"
        self.config = config or {}
        logger.debug(f"[SymbolicToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute symbolic reasoning on the problem.

        Args:
            problem: Query string or dict with 'query' key

        Returns:
            Dict with tool, result, confidence, and proof details
        """
        start_time = time.time()

        try:
            # Clear state before each query to prevent cross-contamination.
            if hasattr(self.engine, 'clear_state'):
                self.engine.clear_state()

            # Extract query string from problem
            query_str = self._extract_query(problem)

            if not query_str:
                return self._error_result("No query provided")

            logger.info(f"[SymbolicEngine] Processing query: {query_str[:100]}...")

            # ================================================================
            # CRITICAL FIX: Check if QueryPreprocessor already extracted formal input
            # If preprocessing was done by reasoning_integration.py, use that result!
            # ================================================================
            preprocessed_query = query_str  # Default to original

            if isinstance(problem, dict):
                preprocessing_result = problem.get('preprocessing') or problem.get('formal_input')

                if preprocessing_result:
                    formal_input = self._extract_formal_input(preprocessing_result)
                    if formal_input:
                        preprocessed_query = formal_input
                        logger.info(
                            f"[SymbolicEngine] Using preprocessed input from QueryPreprocessor: "
                            f"'{preprocessed_query[:50]}...'"
                        )

            # If no preprocessing was provided, try to extract formal logic ourselves
            if preprocessed_query == query_str:
                preprocessed_query = self._preprocess_query(query_str)
                if preprocessed_query != query_str:
                    logger.info(
                        f"[SymbolicEngine] Preprocessed query locally: "
                        f"'{query_str[:50]}...' -> '{preprocessed_query[:50]}...'"
                    )

            # Check if problem contains rules/facts to add to knowledge base
            if isinstance(problem, dict):
                rules = problem.get("rules", [])
                facts = problem.get("facts", [])
                for rule in rules:
                    self.engine.add_rule(rule)
                for fact in facts:
                    self.engine.add_fact(fact)

            # Execute the symbolic reasoning query with preprocessed input
            result = self.engine.query(preprocessed_query)

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"[SymbolicEngine] Query complete: proven={result.get('proven')}, "
                f"confidence={result.get('confidence', 0):.3f}, time={execution_time:.0f}ms"
            )

            return {
                "tool": self.name,
                "result": result,
                "proven": result.get("proven", False),
                "confidence": result.get("confidence", 0.5),
                "proof": result.get("proof"),
                "method": result.get("method", "symbolic"),
                "execution_time_ms": execution_time,
                "engine": "SymbolicReasoner",
                "preprocessed": preprocessed_query != query_str,
            }

        except Exception as e:
            logger.error(f"[SymbolicEngine] Reasoning failed: {e}", exc_info=True)
            return self._error_result(str(e))

    def _extract_query(self, problem: Any) -> str:
        """Extract query string from problem."""
        if isinstance(problem, str):
            return problem
        elif isinstance(problem, dict):
            return problem.get("query") or problem.get("text") or problem.get("formula") or ""
        else:
            return str(problem)

    def _extract_formal_input(self, preprocessing_result: Any) -> Optional[str]:
        """
        Extract formal input from various preprocessing result structures.

        Args:
            preprocessing_result: Can be:
                - PreprocessingResult dataclass with formal_input attribute
                - Dict with 'formal_input' key
                - Direct string

        Returns:
            Extracted formal input string, or None if not found/empty
        """
        formal_input = None

        # Try dataclass with formal_input attribute
        if hasattr(preprocessing_result, 'formal_input'):
            formal_input = preprocessing_result.formal_input
        # Try dict with formal_input key
        elif isinstance(preprocessing_result, dict):
            formal_input = preprocessing_result.get('formal_input')
        # Direct string
        elif isinstance(preprocessing_result, str):
            formal_input = preprocessing_result

        # Validate and convert to string
        if formal_input and len(str(formal_input)) > 0:
            return str(formal_input) if not isinstance(formal_input, str) else formal_input

        return None

    def _error_result(self, error: str) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "result": None,
            "confidence": 0.1,
            "error": error,
            "engine": "SymbolicReasoner",
        }
