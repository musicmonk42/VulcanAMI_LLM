"""
Reasoning integration subpackage for VULCAN.

Re-exports all public API for the reasoning integration layer.
"""

from .types import (
    ReasoningStrategyType,
    RoutingDecision,
    ReasoningResult,
    IntegrationStatistics,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
    LOG_PREFIX,
    MAX_FALLBACK_ATTEMPTS,
)
from .safety_checker import (
    is_false_positive_safety_block,
    is_result_safety_filtered,
    get_safety_filtered_fallback_tools,
)
from .query_router import get_reasoning_type_from_route
from .orchestrator import ReasoningIntegration
from .utils import (
    get_reasoning_integration,
    apply_reasoning,
    run_portfolio_reasoning,
    get_reasoning_statistics,
    shutdown_reasoning,
    observe_reasoning_selection,
    observe_reasoning_execution,
    observe_reasoning_success,
    observe_reasoning_failure,
    observe_reasoning_degradation,
    # SystemObserver integration functions
    observe_query_start,
    observe_engine_result,
    observe_outcome,
    observe_validation_failure,
    observe_error,
)

__all__ = [
    # Types
    "ReasoningStrategyType",
    "RoutingDecision",
    "ReasoningResult",
    "IntegrationStatistics",
    # Constants
    "DECOMPOSITION_COMPLEXITY_THRESHOLD",
    "LOG_PREFIX",
    "MAX_FALLBACK_ATTEMPTS",
    # Safety checker
    "is_false_positive_safety_block",
    "is_result_safety_filtered",
    "get_safety_filtered_fallback_tools",
    # Query router
    "get_reasoning_type_from_route",
    # Main class
    "ReasoningIntegration",
    # Utilities
    "get_reasoning_integration",
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    # Observers
    "observe_reasoning_selection",
    "observe_reasoning_execution",
    "observe_reasoning_success",
    "observe_reasoning_failure",
    "observe_reasoning_degradation",
    # SystemObserver integration functions
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
]
