"""
Mathematical Tool Wrapper - Adapts MathematicalComputationTool to the common reason() interface.

FIX #1: Missing Engine Registration
Evidence from logs: "Tool 'mathematical' not available, using fallback: symbolic"

The MathematicalComputationTool performs symbolic math computation using:
- SymPy for symbolic algebra, calculus, etc.
- LLM-based code generation for complex problems
- Template-based generation for common operations
- Safe sandboxed execution via RestrictedPython

Extracted from tool_selector.py to reduce module size.
"""

import re
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MathematicalToolWrapper:
    """
    Wrapper for MathematicalComputationTool that exposes reason() method.

    FIX #1: Missing Engine Registration
    FIX #4: Added early gate check to reject non-mathematical queries.
    """

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with a MathematicalComputationTool instance.

        Args:
            engine: MathematicalComputationTool instance
            config: Optional configuration dict (for warm pool compatibility).
        """
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        if isinstance(engine, str):
            raise TypeError(
                f"MathematicalToolWrapper received string '{engine}' instead of MathematicalComputationTool instance. "
                f"This indicates a bug in _initialize_real_engines(). "
                f"Expected: MathematicalComputationTool instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "MathematicalToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine. "
                "Ensure MathematicalComputationTool is properly instantiated."
            )
        # BUG FIX #4: Add gate check capability validation
        if not (hasattr(engine, 'solve') or hasattr(engine, 'compute') or hasattr(engine, 'reason')):
            raise AttributeError(
                f"MathematicalToolWrapper engine must have 'solve()', 'compute()', or 'reason()' method. "
                f"Got type: {type(engine).__name__}, "
                f"Available methods: {[m for m in dir(engine) if not m.startswith('_')][:10]}"
            )
        self.engine = engine
        self.name = "mathematical"
        self.config = config or {}
        logger.debug(f"[MathematicalToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute mathematical computation on the problem.

        BUG FIX #4: Added early gate check to reject non-mathematical queries.

        Args:
            problem: Dict with query, or string query

        Returns:
            Dict with computation result and confidence, or rejection if not mathematical
        """
        start_time = time.perf_counter()

        try:
            # Extract query string from problem
            if isinstance(problem, str):
                query = problem
            elif isinstance(problem, dict):
                query = problem.get("query") or problem.get("text") or problem.get("problem", "")
            else:
                query = str(problem)

            # BUG FIX #4: Early Gate Check - Reject non-mathematical queries
            if not self._is_mathematical_query(query):
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"[MathematicalEngine] Gate check REJECTED query (not mathematical): "
                    f"'{query[:100]}...' - Returning low confidence (0.0) to trigger tool re-selection"
                )
                return {
                    "tool": self.name,
                    "result": None,
                    "confidence": 0.0,
                    "success": False,
                    "applicable": False,
                    "reason": "Query does not contain mathematical content",
                    "engine": "MathematicalComputationTool",
                    "execution_time_ms": execution_time,
                }

            logger.info(f"[MathematicalEngine] Computing: {query[:100]}...")

            # Execute mathematical computation
            if hasattr(self.engine, "compute"):
                result = self.engine.compute(query)
            elif hasattr(self.engine, "solve"):
                result = self.engine.solve(query)
            elif hasattr(self.engine, "reason"):
                result = self.engine.reason(problem)
            else:
                result = {
                    "analysis": "Mathematical computation requested",
                    "query": query,
                    "success": False,
                    "confidence": 0.3
                }

            execution_time = (time.perf_counter() - start_time) * 1000

            # Extract confidence and success from result
            confidence = 0.7
            success = True
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.7)
                success = result.get("success", True)
                if not success:
                    confidence = min(confidence, 0.3)

            logger.info(f"[MathematicalEngine] Computation complete: success={success}, confidence={confidence:.3f}, time={execution_time:.0f}ms")

            return {
                "tool": self.name,
                "result": result,
                "confidence": confidence,
                "success": success,
                "execution_time_ms": execution_time,
                "engine": "MathematicalComputationTool",
            }

        except Exception as e:
            logger.error(f"[MathematicalEngine] Computation failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "confidence": 0.1,
                "success": False,
                "error": str(e),
                "engine": "MathematicalComputationTool",
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
            }

    def _is_mathematical_query(self, query: str) -> bool:
        """
        BUG FIX #4: Gate check to determine if query contains mathematical content.

        Defense in Depth - Multiple detection layers:
        - Layer 1: Explicit math keywords (calculate, compute, solve, etc.)
        - Layer 2: Mathematical symbols (=, +, -, *, /, ^, etc.)
        - Layer 3: Mathematical notation (integral, sum, partial, sqrt, etc.)
        - Layer 4: Numbers with operations

        Returns:
            True if query appears to be mathematical, False otherwise
        """
        if not query or not isinstance(query, str):
            return False

        query_lower = query.lower()

        # Layer 1: Mathematical keywords
        math_keywords = (
            'calculate', 'compute', 'solve', 'evaluate', 'simplify',
            'integrate', 'derivative', 'differentiate', 'limit',
            'sum', 'product', 'matrix', 'vector', 'equation',
            'formula', 'algebra', 'calculus', 'geometry',
            'trigonometry', 'logarithm', 'exponential',
            'probability', 'statistics', 'factorial', 'permutation',
            'what is', 'how much', 'how many',
        )
        has_math_keyword = any(kw in query_lower for kw in math_keywords)

        # Layer 2: Mathematical operators and symbols
        math_symbols = ('=', '+', '-', '*', '/', '^', '**', '%', '\u221a', '\u00b1')
        has_math_symbol = any(sym in query for sym in math_symbols)

        # Layer 3: Mathematical notation (Unicode)
        math_notation = ('\u222b', '\u2211', '\u220f', '\u2202', '\u2207', '\u221e', '\u2248', '\u2260', '\u2264', '\u2265', '\u03c0', '\u03b8', '\u03b1', '\u03b2')
        has_math_notation = any(sym in query for sym in math_notation)

        # Layer 4: Numbers with context
        has_number_with_operation = bool(re.search(r'\d+\s*[\+\-\*/\^]|\d*\.\d+|x\^?\d+|\d+x', query))

        # Anti-pattern: Logic/philosophical keywords that indicate NOT math
        logic_keywords = (
            'nonmonotonic', 'exception', 'if and only if', 'implies',
            'therefore', 'necessarily', 'possibly', 'forall', 'exists',
            'entails', 'satisfiable', 'valid', 'sound', 'complete',
            'belief', 'knowledge', 'ethical', 'moral', 'ought',
        )
        has_logic_keyword = any(kw in query_lower for kw in logic_keywords)

        is_math = (has_math_keyword or has_math_symbol or has_math_notation or has_number_with_operation)

        if has_logic_keyword and not (has_math_symbol or has_math_notation):
            return False

        return is_math
