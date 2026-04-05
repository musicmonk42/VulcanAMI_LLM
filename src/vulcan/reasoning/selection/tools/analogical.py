"""
Analogical Tool Wrapper - Adapts AnalogicalReasoner to the common reason() interface.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnalogicalToolWrapper:
    """
    Wrapper for AnalogicalReasoner that exposes reason() method.
    """

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        # Industry Standard: Fail-fast with clear error messages
        if isinstance(engine, str):
            raise TypeError(
                f"AnalogicalToolWrapper received string '{engine}' instead of AnalogicalReasoner instance. "
                f"This indicates a bug in engine initialization. "
                f"Expected: AnalogicalReasoner instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "AnalogicalToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine. "
                "Check _initialize_real_engines() for proper engine instantiation."
            )
        # Verify engine has required methods
        if not (hasattr(engine, 'find_analogies') or hasattr(engine, 'reason')):
            raise AttributeError(
                f"AnalogicalToolWrapper engine must have 'find_analogies()' or 'reason()' method. "
                f"Got type: {type(engine).__name__}, "
                f"Available methods: {[m for m in dir(engine) if not m.startswith('_')][:10]}"
            )
        self.engine = engine
        self.name = "analogical"
        self.config = config or {}
        logger.debug(f"[AnalogicalToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """Execute analogical reasoning on the problem."""
        start_time = time.time()

        try:
            # Parse problem
            if isinstance(problem, str):
                query = problem
            elif isinstance(problem, dict):
                query = problem.get("query") or problem.get("text", "")
            else:
                query = str(problem)

            logger.info(f"[AnalogicalEngine] Finding analogies for: {query[:100]}...")

            # Execute analogical reasoning
            if hasattr(self.engine, "find_analogies"):
                result = self.engine.find_analogies(query)
            elif hasattr(self.engine, "reason"):
                result = self.engine.reason(problem)
            else:
                result = {"analogies": [], "query": query}

            execution_time = (time.time() - start_time) * 1000

            confidence = 0.6
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.6)

            logger.info(f"[AnalogicalEngine] Complete: confidence={confidence:.3f}, time={execution_time:.0f}ms")

            return {
                "tool": self.name,
                "result": result,
                "confidence": confidence,
                "execution_time_ms": execution_time,
                "engine": "AnalogicalReasoner",
            }

        except Exception as e:
            logger.error(f"[AnalogicalEngine] Reasoning failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "confidence": 0.1,
                "error": str(e),
                "engine": "AnalogicalReasoner",
            }
