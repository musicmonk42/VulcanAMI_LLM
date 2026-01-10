"""
Utility functions and conveniences for reasoning integration.

This module contains:
- Convenience functions (apply_reasoning, run_portfolio_reasoning, etc.)
- Observer integration functions
- Statistics tracking
- Singleton management

Module: vulcan.reasoning.integration.utils
Author: Vulcan AI Team
"""

import atexit
import threading
from typing import Any, Dict, List, Optional

from .types import ReasoningResult
from .orchestrator import ReasoningIntegration

# Global singleton instance
_reasoning_integration: Optional[ReasoningIntegration] = None
_integration_lock = threading.Lock()


def get_reasoning_integration(
    config: Optional[Dict[str, Any]] = None
) -> ReasoningIntegration:
    """
    Get or create the global reasoning integration singleton.

    Uses double-checked locking pattern for thread-safe lazy initialization.

    Args:
        config: Optional configuration dictionary (only used on first call)

    Returns:
        ReasoningIntegration singleton instance

    Example:
        >>> integration = get_reasoning_integration()
        >>> result = integration.apply_reasoning(...)
    """
    global _reasoning_integration

    if _reasoning_integration is None:
        with _integration_lock:
            if _reasoning_integration is None:
                _reasoning_integration = ReasoningIntegration(config)

    return _reasoning_integration


def _shutdown_on_exit() -> None:
    """Atexit handler to shutdown integration gracefully."""
    global _reasoning_integration

    if _reasoning_integration is not None:
        try:
            _reasoning_integration.shutdown(timeout=2.0)
        except Exception:
            pass  # Ignore errors during exit shutdown


# Register atexit handler for graceful shutdown
atexit.register(_shutdown_on_exit)


def apply_reasoning(
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Convenience function to apply reasoning using the global singleton.

    This is the primary entry point for most use cases. It handles singleton
    management automatically.

    Args:
        query: The user query to process
        query_type: Type from router (general, reasoning, execution, etc.)
        complexity: Complexity score (0.0 to 1.0)
        context: Optional context dict with conversation_id, history, etc.

    Returns:
        ReasoningResult with selected tools and strategy

    Example:
        >>> from vulcan.reasoning.integration import apply_reasoning
        >>> result = apply_reasoning(
        ...     query="What causes climate change?",
        ...     query_type="reasoning",
        ...     complexity=0.7
        ... )
        >>> print(f"Tools: {result.selected_tools}")
    """
    return get_reasoning_integration().apply_reasoning(
        query, query_type, complexity, context
    )


def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run portfolio execution using the global singleton.

    Args:
        query: The user query
        tools: List of tools to use
        strategy: Execution strategy
        constraints: Optional execution constraints

    Returns:
        Portfolio execution result dictionary

    Example:
        >>> result = run_portfolio_reasoning(
        ...     query="Complex problem",
        ...     tools=["symbolic", "causal"],
        ...     strategy="causal_reasoning"
        ... )
    """
    return get_reasoning_integration().run_portfolio(
        query, tools, strategy, constraints
    )


def get_reasoning_statistics() -> Dict[str, Any]:
    """
    Convenience function to get reasoning integration statistics.

    Returns:
        Dictionary with statistics (success_rate, avg_confidence, etc.)

    Example:
        >>> stats = get_reasoning_statistics()
        >>> print(f"Success rate: {stats['success_rate']:.1%}")
    """
    return get_reasoning_integration().get_statistics()


def shutdown_reasoning(timeout: float = 5.0) -> None:
    """
    Convenience function to shutdown reasoning integration.

    Args:
        timeout: Maximum time to wait for shutdown (seconds)

    Example:
        >>> shutdown_reasoning(timeout=10.0)
    """
    integration = get_reasoning_integration()
    integration.shutdown(timeout=timeout)


# Observer integration functions
def observe_query_start(query: str, query_type: str, complexity: float) -> None:
    """
    Observe the start of query processing.

    Args:
        query: The user query
        query_type: Type from router
        complexity: Complexity score
    """
    integration = get_reasoning_integration()
    if hasattr(integration, '_system_observer') and integration._system_observer:
        integration._system_observer.observe_query_start(query, query_type, complexity)


def observe_engine_result(
    engine: str,
    result: Any,
    confidence: float,
    execution_time_ms: float
) -> None:
    """
    Observe reasoning engine result.

    Args:
        engine: Engine name
        result: Engine result
        confidence: Confidence score
        execution_time_ms: Execution time in milliseconds
    """
    integration = get_reasoning_integration()
    if hasattr(integration, '_system_observer') and integration._system_observer:
        integration._system_observer.observe_engine_result(
            engine, result, confidence, execution_time_ms
        )


def observe_outcome(success: bool, selected_tools: List[str], final_confidence: float) -> None:
    """
    Observe final reasoning outcome.

    Args:
        success: Whether reasoning succeeded
        selected_tools: Tools that were selected
        final_confidence: Final confidence score
    """
    integration = get_reasoning_integration()
    if hasattr(integration, '_system_observer') and integration._system_observer:
        integration._system_observer.observe_outcome(success, selected_tools, final_confidence)


def observe_validation_failure(reason: str) -> None:
    """
    Observe validation failure.

    Args:
        reason: Reason for validation failure
    """
    integration = get_reasoning_integration()
    if hasattr(integration, '_system_observer') and integration._system_observer:
        integration._system_observer.observe_validation_failure(reason)


def observe_error(error: Exception, context: str) -> None:
    """
    Observe error during reasoning.

    Args:
        error: The exception that occurred
        context: Context where error occurred
    """
    integration = get_reasoning_integration()
    if hasattr(integration, '_system_observer') and integration._system_observer:
        integration._system_observer.observe_error(error, context)


__all__ = [
    "get_reasoning_integration",
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
]


def observe_reasoning_selection(query: str, tools: List[str], strategy: str) -> None:
    """
    Observe and log reasoning selection event.
    
    Args:
        query: The query being processed
        tools: List of selected tools
        strategy: Selected reasoning strategy
    """
    logger.info(f"Reasoning selection: strategy={strategy}, tools={tools}")


def observe_reasoning_execution(query: str, tools: List[str], duration_ms: float) -> None:
    """
    Observe and log reasoning execution event.
    
    Args:
        query: The query being processed
        tools: List of tools executed
        duration_ms: Execution duration in milliseconds
    """
    logger.info(f"Reasoning execution: tools={tools}, duration={duration_ms:.2f}ms")


def observe_reasoning_success(query: str, result: Any) -> None:
    """
    Observe and log reasoning success event.
    
    Args:
        query: The query that was processed
        result: The successful result
    """
    logger.info("Reasoning success")


def observe_reasoning_failure(query: str, error: str) -> None:
    """
    Observe and log reasoning failure event.
    
    Args:
        query: The query that failed
        error: Error message
    """
    logger.warning(f"Reasoning failure: {error}")


def observe_reasoning_degradation(query: str, reason: str) -> None:
    """
    Observe and log reasoning degradation event.
    
    Args:
        query: The query being processed
        reason: Reason for degradation
    """
    logger.warning(f"Reasoning degradation: {reason}")
