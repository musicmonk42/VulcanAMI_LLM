"""
Integration layer for reasoning module.

Part of the VULCAN-AGI system.

This module wires the ToolSelector and reasoning strategies into the query
processing flow. It provides a clean interface for applying reasoning-based
tool selection to queries.

Key Features:
- Lazy initialization of reasoning components
- Graceful degradation when components unavailable
- Strategy selection based on query type and complexity
- Portfolio execution for complex queries
- Comprehensive logging for observability

Components Integrated:
- ToolSelector: Intelligent tool selection using multi-armed bandits
- PortfolioExecutor: Multi-tool execution strategies

Usage:
    # Apply reasoning to select tools
    from vulcan.reasoning.reasoning_integration import apply_reasoning
    result = apply_reasoning(
        query="Explain causal relationship...",
        query_type="reasoning",
        complexity=0.75,
    )

    # Use result in routing
    if result.reasoning_strategy == "causal_reasoning":
        # Route to causal reasoning pipeline
        pass
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_WORKERS = 4  # Default number of workers for PortfolioExecutor
DEFAULT_TIME_BUDGET_MS = 5000  # Default time budget in milliseconds
DEFAULT_ENERGY_BUDGET_MJ = 1000  # Default energy budget in millijoules
DEFAULT_MIN_CONFIDENCE = 0.5  # Default minimum confidence threshold


@dataclass
class ReasoningResult:
    """
    Result from reasoning module.

    Contains tool selection and strategy information for query processing.

    Attributes:
        selected_tools: List of tools selected for query
        reasoning_strategy: Strategy name for reasoning
        confidence: Confidence in selection (0.0 to 1.0)
        rationale: Human-readable explanation
        metadata: Additional context information
    """

    selected_tools: List[str]
    reasoning_strategy: str
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningIntegration:
    """
    Integrates reasoning module into query processing.

    Provides a unified interface for applying reasoning-based tool selection
    and strategy determination. Handles lazy initialization and graceful
    degradation when components are unavailable.

    Thread Safety:
        All methods are thread-safe. Internal components are initialized
        lazily with proper locking.

    Usage:
        integration = ReasoningIntegration()
        result = integration.apply_reasoning(query, query_type, complexity)
    """

    def __init__(self):
        """Initialize reasoning integration with lazy component loading."""
        self._tool_selector = None
        self._portfolio_executor = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._stats_lock = threading.RLock()

        # Statistics
        self._invocations = 0
        self._tool_selections = 0
        self._portfolio_executions = 0
        self._errors = 0

        logger.info("[ReasoningIntegration] Initialized (lazy loading enabled)")

    def _init_components(self) -> None:
        """
        Lazy initialization of reasoning components.

        Attempts to import and initialize ToolSelector and PortfolioExecutor.
        Failures are logged but don't prevent basic operation.
        """
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            # Try to initialize ToolSelector
            try:
                from vulcan.reasoning.selection.tool_selector import ToolSelector

                self._tool_selector = ToolSelector()
                logger.info("[ReasoningIntegration] ToolSelector initialized")
            except ImportError as e:
                logger.warning(f"[ReasoningIntegration] ToolSelector not available: {e}")
            except Exception as e:
                logger.error(f"[ReasoningIntegration] ToolSelector init failed: {e}")

            # Try to initialize PortfolioExecutor
            try:
                from vulcan.reasoning.selection.portfolio_executor import (
                    PortfolioExecutor,
                )

                # Create with empty tools dict - will use mock tools
                self._portfolio_executor = PortfolioExecutor(
                    tools={},
                    max_workers=DEFAULT_MAX_WORKERS
                )
                logger.info("[ReasoningIntegration] PortfolioExecutor initialized")
            except ImportError as e:
                logger.warning(
                    f"[ReasoningIntegration] PortfolioExecutor not available: {e}"
                )
            except Exception as e:
                logger.error(
                    f"[ReasoningIntegration] PortfolioExecutor init failed: {e}"
                )

            self._initialized = True

    def apply_reasoning(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Apply reasoning to select tools and strategy.

        Analyzes the query and uses the ToolSelector (if available) to
        determine the best tools and reasoning strategy.

        Args:
            query: The user query
            query_type: Type from router (general, reasoning, execution, etc.)
            complexity: Complexity score (0.0 to 1.0)
            context: Optional context dict with conversation_id, history, etc.

        Returns:
            ReasoningResult with selected tools and strategy
        """
        self._init_components()

        with self._stats_lock:
            self._invocations += 1

        # Fast path for simple queries
        if complexity < 0.3:
            return ReasoningResult(
                selected_tools=["general"],
                reasoning_strategy="direct",
                confidence=0.9,
                rationale="Simple query, direct response",
                metadata={"fast_path": True, "complexity": complexity},
            )

        # Default values
        selected_tools = ["general"]
        reasoning_strategy = "default"
        confidence = 0.7
        rationale = "Default reasoning"

        # Use tool selector if available
        if self._tool_selector:
            try:
                # Build selection request
                from vulcan.reasoning.selection.tool_selector import (
                    SelectionRequest,
                    SelectionMode,
                )

                # Map complexity to selection mode
                if complexity > 0.7:
                    mode = SelectionMode.ACCURATE
                elif complexity < 0.4:
                    mode = SelectionMode.FAST
                else:
                    mode = SelectionMode.BALANCED

                # Create request
                request = SelectionRequest(
                    problem=query,
                    constraints={
                        "time_budget_ms": DEFAULT_TIME_BUDGET_MS,
                        "energy_budget_mj": DEFAULT_ENERGY_BUDGET_MJ,
                        "min_confidence": DEFAULT_MIN_CONFIDENCE,
                    },
                    mode=mode,
                    context=context or {},
                )

                # Get selection
                result = self._tool_selector.select_and_execute(request)

                selected_tools = result.tools_used if result.tools_used else ["general"]
                reasoning_strategy = result.strategy_used.value
                confidence = result.calibrated_confidence
                rationale = f"ToolSelector selected via {result.strategy_used.value}"

                with self._stats_lock:
                    self._tool_selections += 1

                logger.info(
                    f"[ToolSelector] Selected: {selected_tools} "
                    f"using {reasoning_strategy} (confidence: {confidence:.2f})"
                )

            except Exception as e:
                logger.warning(f"[ToolSelector] Selection failed: {e}")
                with self._stats_lock:
                    self._errors += 1

        # Determine reasoning strategy based on query type if not set by selector
        if reasoning_strategy == "default":
            reasoning_strategy = self._select_reasoning_strategy(query_type, complexity)

        return ReasoningResult(
            selected_tools=selected_tools,
            reasoning_strategy=reasoning_strategy,
            confidence=confidence,
            rationale=rationale,
            metadata={
                "query_type": query_type,
                "complexity": complexity,
                "tool_selector_available": self._tool_selector is not None,
            },
        )

    def _select_reasoning_strategy(self, query_type: str, complexity: float) -> str:
        """
        Select appropriate reasoning strategy based on query characteristics.

        Args:
            query_type: Type of query (reasoning, perception, planning, etc.)
            complexity: Query complexity (0.0 to 1.0)

        Returns:
            Strategy name string
        """
        # High complexity reasoning queries use causal reasoning
        if query_type == "reasoning" and complexity > 0.6:
            return "causal_reasoning"

        # Execution tasks use planning
        if query_type == "execution":
            return "planning"

        # Medium-high complexity uses probabilistic reasoning
        if complexity > 0.5:
            return "probabilistic_reasoning"

        # Perception tasks use analogical reasoning
        if query_type == "perception":
            return "analogical_reasoning"

        # Planning tasks use deliberative reasoning
        if query_type == "planning":
            return "deliberative"

        # Learning tasks use meta-reasoning
        if query_type == "learning":
            return "meta_reasoning"

        # Default to direct for simple queries
        return "direct"

    def run_portfolio(
        self,
        query: str,
        tools: List[str],
        strategy: str,
    ) -> Dict[str, Any]:
        """
        Run portfolio execution for complex queries.

        Executes multiple tools using the specified strategy for queries
        that benefit from diverse reasoning approaches.

        Args:
            query: The user query
            tools: List of tools to use
            strategy: Execution strategy (parallel, sequential, etc.)

        Returns:
            Portfolio execution result dictionary
        """
        self._init_components()

        if not self._portfolio_executor:
            logger.warning("[ReasoningIntegration] PortfolioExecutor not available")
            return {"status": "skipped", "reason": "executor_unavailable"}

        try:
            from vulcan.reasoning.selection.portfolio_executor import (
                ExecutionStrategy,
                ExecutionMonitor,
            )

            # Map strategy string to enum
            strategy_map = {
                "causal_reasoning": ExecutionStrategy.SEQUENTIAL_REFINEMENT,
                "probabilistic_reasoning": ExecutionStrategy.SPECULATIVE_PARALLEL,
                "analogical_reasoning": ExecutionStrategy.CASCADE,
                "planning": ExecutionStrategy.SEQUENTIAL_REFINEMENT,
                "deliberative": ExecutionStrategy.COMMITTEE_CONSENSUS,
                "direct": ExecutionStrategy.SINGLE,
            }

            exec_strategy = strategy_map.get(strategy, ExecutionStrategy.ADAPTIVE_MIX)

            # Create monitor
            monitor = ExecutionMonitor(
                time_budget_ms=DEFAULT_TIME_BUDGET_MS,
                energy_budget_mj=DEFAULT_ENERGY_BUDGET_MJ,
                min_confidence=DEFAULT_MIN_CONFIDENCE,
            )

            # Execute
            result = self._portfolio_executor.execute(
                strategy=exec_strategy,
                tool_names=tools,
                problem=query,
                constraints={
                    "time_budget_ms": DEFAULT_TIME_BUDGET_MS,
                    "energy_budget_mj": DEFAULT_ENERGY_BUDGET_MJ,
                    "min_confidence": DEFAULT_MIN_CONFIDENCE,
                },
                monitor=monitor,
            )

            with self._stats_lock:
                self._portfolio_executions += 1

            logger.info(f"[PortfolioExecutor] Completed with strategy: {strategy}")

            return {
                "status": "success",
                "strategy_used": result.strategy.value,
                "tools_used": result.tools_used,
                "execution_time_ms": result.execution_time * 1000,
                "confidence": (
                    result.consensus_confidence
                    if result.consensus_confidence
                    else result.confidence
                ),
            }

        except Exception as e:
            logger.error(f"[PortfolioExecutor] Execution failed: {e}")
            with self._stats_lock:
                self._errors += 1

            return {"status": "error", "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics for monitoring.

        Returns:
            Dictionary with invocation counts, success rates, etc.
        """
        with self._stats_lock:
            total = self._invocations
            return {
                "initialized": self._initialized,
                "tool_selector_available": self._tool_selector is not None,
                "portfolio_executor_available": self._portfolio_executor is not None,
                "invocations": self._invocations,
                "tool_selections": self._tool_selections,
                "portfolio_executions": self._portfolio_executions,
                "errors": self._errors,
                "success_rate": (
                    (total - self._errors) / total if total > 0 else 0.0
                ),
            }

    def shutdown(self) -> None:
        """Shutdown reasoning components gracefully."""
        if self._tool_selector and hasattr(self._tool_selector, "shutdown"):
            try:
                self._tool_selector.shutdown(timeout=5.0)
            except Exception as e:
                logger.warning(f"[ReasoningIntegration] ToolSelector shutdown: {e}")

        if self._portfolio_executor and hasattr(self._portfolio_executor, "shutdown"):
            try:
                self._portfolio_executor.shutdown(timeout=5.0)
            except Exception as e:
                logger.warning(f"[ReasoningIntegration] PortfolioExecutor shutdown: {e}")

        logger.info("[ReasoningIntegration] Shutdown complete")


# Global instance (singleton)
_reasoning_integration: Optional[ReasoningIntegration] = None
_integration_lock = threading.Lock()


def get_reasoning_integration() -> ReasoningIntegration:
    """
    Get or create reasoning integration instance.

    Returns:
        ReasoningIntegration singleton instance
    """
    global _reasoning_integration

    if _reasoning_integration is None:
        with _integration_lock:
            if _reasoning_integration is None:
                _reasoning_integration = ReasoningIntegration()

    return _reasoning_integration


def apply_reasoning(
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Convenience function to apply reasoning.

    Args:
        query: The user query
        query_type: Type from router
        complexity: Complexity score
        context: Optional context dict

    Returns:
        ReasoningResult with selected tools and strategy
    """
    return get_reasoning_integration().apply_reasoning(
        query, query_type, complexity, context
    )


def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str,
) -> Dict[str, Any]:
    """
    Convenience function to run portfolio execution.

    Args:
        query: The user query
        tools: List of tools to use
        strategy: Execution strategy

    Returns:
        Portfolio execution result
    """
    return get_reasoning_integration().run_portfolio(query, tools, strategy)


def get_reasoning_statistics() -> Dict[str, Any]:
    """
    Get reasoning integration statistics.

    Returns:
        Statistics dictionary
    """
    return get_reasoning_integration().get_statistics()


def shutdown_reasoning() -> None:
    """Shutdown reasoning integration."""
    global _reasoning_integration

    if _reasoning_integration:
        _reasoning_integration.shutdown()
        _reasoning_integration = None
