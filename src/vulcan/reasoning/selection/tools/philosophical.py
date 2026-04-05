"""
Philosophical Tool Wrapper - Routes philosophical/ethical reasoning to World Model.

DEPRECATED: The PhilosophicalReasoner has been removed. This wrapper now routes
philosophical/ethical queries to World Model's _philosophical_reasoning method.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PhilosophicalToolWrapper:
    """
    DEPRECATED: Wrapper that delegates philosophical reasoning to World Model.

    The PhilosophicalReasoner has been removed. This wrapper now routes
    philosophical/ethical queries to World Model's _philosophical_reasoning method.
    """

    def __init__(self, engine=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize wrapper. Engine parameter is ignored - uses World Model.

        Args:
            engine: Ignored (for API compatibility)
            config: Optional configuration dict (for warm pool compatibility).
        """
        self._world_model = None
        self.name = "philosophical"
        self.config = config or {}

    def _get_world_model(self):
        """Lazy-load World Model using singleton."""
        if self._world_model is None:
            try:
                # FIX: Use singleton to avoid repeated 10-15s initialization overhead
                from vulcan.reasoning.singletons import get_world_model
                self._world_model = get_world_model()
                if self._world_model is None:
                    # Fallback to direct instantiation if singleton not available
                    from vulcan.world_model.world_model_core import WorldModel
                    self._world_model = WorldModel()
            except ImportError:
                logger.warning("[PhilosophicalToolWrapper] World Model not available")
        return self._world_model

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute philosophical/ethical reasoning via World Model.

        Args:
            problem: Dict with query, or string query

        Returns:
            Dict with reasoning result and confidence
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

            logger.info(f"[PhilosophicalToolWrapper] Routing to World Model: {query[:100]}...")

            # Route to World Model's philosophical reasoning
            world_model = self._get_world_model()
            if world_model:
                result = world_model.reason(query, mode='philosophical')
            else:
                # Fallback if World Model not available
                result = {
                    "response": "Philosophical analysis requested but World Model not available",
                    "confidence": 0.3
                }

            execution_time = (time.perf_counter() - start_time) * 1000

            # Extract confidence from result
            confidence = result.get("confidence", 0.7) if isinstance(result, dict) else 0.7

            logger.info(f"[PhilosophicalToolWrapper] Analysis complete: confidence={confidence:.3f}, time={execution_time:.0f}ms")

            return {
                "tool": self.name,
                "result": result,
                "confidence": confidence,
                "execution_time_ms": execution_time,
                "engine": "WorldModel.philosophical",
            }

        except Exception as e:
            logger.error(f"[PhilosophicalToolWrapper] Reasoning failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "confidence": 0.1,
                "error": str(e),
                "engine": "WorldModel.philosophical",
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
            }
