"""
Multimodal Tool Wrapper - Adapts MultimodalReasoner to the common reason() interface.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MultimodalToolWrapper:
    """
    Wrapper for MultimodalReasoner that exposes reason() method.
    """

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        # Industry Standard: Type safety with comprehensive validation
        if isinstance(engine, str):
            raise TypeError(
                f"MultimodalToolWrapper received string '{engine}' instead of MultimodalReasoner instance. "
                f"Expected: MultimodalReasoner instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "MultimodalToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine."
            )
        if not (hasattr(engine, 'process') or hasattr(engine, 'reason')):
            raise AttributeError(
                f"MultimodalToolWrapper engine must have 'process()' or 'reason()' method. "
                f"Got type: {type(engine).__name__}"
            )
        self.engine = engine
        self.name = "multimodal"
        self.config = config or {}
        logger.debug(f"[MultimodalToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """Execute multimodal reasoning on the problem."""
        start_time = time.time()

        try:
            logger.info(f"[MultimodalEngine] Processing multimodal input...")

            # Execute multimodal reasoning
            if hasattr(self.engine, "process"):
                result = self.engine.process(problem)
            elif hasattr(self.engine, "reason"):
                result = self.engine.reason(problem)
            else:
                result = {"processed": True, "input_type": type(problem).__name__}

            execution_time = (time.time() - start_time) * 1000

            confidence = 0.65
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.65)

            logger.info(f"[MultimodalEngine] Complete: confidence={confidence:.3f}, time={execution_time:.0f}ms")

            return {
                "tool": self.name,
                "result": result,
                "confidence": confidence,
                "execution_time_ms": execution_time,
                "engine": "MultimodalReasoner",
            }

        except Exception as e:
            logger.error(f"[MultimodalEngine] Reasoning failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "confidence": 0.1,
                "error": str(e),
                "engine": "MultimodalReasoner",
            }
