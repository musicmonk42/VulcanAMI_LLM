"""
Causal Tool Wrapper - Adapts CausalReasoner to the common reason() interface.

The CausalReasoner performs causal DAG analysis and counterfactual reasoning.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CausalToolWrapper:
    """
    Wrapper for CausalReasoner that exposes reason() method.

    The CausalReasoner performs causal DAG analysis and counterfactual reasoning.
    """

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        # FAIL-FAST: Raise explicit error if wrong type is passed
        if isinstance(engine, str):
            raise TypeError(
                f"CausalToolWrapper received string '{engine}' instead of CausalReasoner instance. "
                f"Expected: CausalReasoner instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "CausalToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine."
            )
        # CausalReasoner may have different method names, so check for common ones
        if not (hasattr(engine, 'query') or hasattr(engine, 'analyze') or hasattr(engine, 'reason')):
            raise AttributeError(
                f"CausalToolWrapper engine must have 'query()', 'analyze()', or 'reason()' method. "
                f"Got type: {type(engine).__name__}"
            )
        self.engine = engine
        self.name = "causal"
        self.config = config or {}
        logger.debug(f"[CausalToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute causal reasoning on the problem.
        """
        start_time = time.time()

        try:
            # Parse problem
            if isinstance(problem, str):
                query = problem
                data = None
                intervention = None
            elif isinstance(problem, dict):
                query = problem.get("query") or problem.get("text", "")
                data = problem.get("data")
                intervention = problem.get("intervention")
            else:
                query = str(problem)
                data = None
                intervention = None

            logger.info(f"[CausalEngine] Analyzing causal query: {query[:100]}...")

            # Execute causal reasoning
            if hasattr(self.engine, "analyze_causality"):
                result = self.engine.analyze_causality(query, data=data)
            elif hasattr(self.engine, "reason"):
                result = self.engine.reason(problem)
            elif hasattr(self.engine, "query"):
                result = self.engine.query(query)
            else:
                result = {"analysis": "Causal analysis requested", "query": query}

            execution_time = (time.time() - start_time) * 1000

            confidence = 0.7
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.7)

            logger.info(f"[CausalEngine] Analysis complete: confidence={confidence:.3f}, time={execution_time:.0f}ms")

            return {
                "tool": self.name,
                "result": result,
                "confidence": confidence,
                "execution_time_ms": execution_time,
                "engine": "CausalReasoner",
            }

        except Exception as e:
            logger.error(f"[CausalEngine] Reasoning failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "confidence": 0.1,
                "error": str(e),
                "engine": "CausalReasoner",
            }
