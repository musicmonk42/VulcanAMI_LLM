"""
Reasoning integration subpackage for VULCAN.

This package provides the integration layer between the query processing pipeline
and the reasoning subsystem.

Refactored modules:
    - types: Core dataclasses, enums, and constants ✅
    - safety_checker: Safety validation and false positive detection ✅
    - query_router: Query type routing and classification ✅

Module: vulcan.reasoning.integration
Author: Vulcan AI Team
"""

# Re-export types
from .types import (
    # Constants
    LOG_PREFIX,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIME_BUDGET_MS,
    DEFAULT_ENERGY_BUDGET_MJ,
    DEFAULT_MIN_CONFIDENCE,
    MAX_FALLBACK_ATTEMPTS,
    MIN_CONFIDENCE_FLOOR,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_GOOD_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
    # Query Analysis Constants
    ANALYSIS_INDICATORS,
    ACTION_VERBS,
    ETHICAL_ANALYSIS_INDICATORS,
    PURE_ETHICAL_PHRASES,
    # Enums
    ReasoningStrategyType,
    # Mappings
    QUERY_TYPE_STRATEGY_MAP,
    ROUTE_TO_REASONING_TYPE,
    # Dataclasses
    RoutingDecision,
    ReasoningResult,
    IntegrationStatistics,
)

# Re-export safety checker
from .safety_checker import (
    is_false_positive_safety_block,
    is_result_safety_filtered,
    get_safety_filtered_fallback_tools,
)

# Re-export query router
from .query_router import (
    get_reasoning_type_from_route,
)

# Re-export from parent module for full API (temporary during refactoring)
try:
    from ..reasoning_integration import (
        ReasoningIntegration,
        apply_reasoning,
        run_portfolio_reasoning,
        get_reasoning_integration,
        get_reasoning_statistics,
        shutdown_reasoning,
        observe_query_start,
        observe_engine_result,
        observe_outcome,
        observe_validation_failure,
        observe_error,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Failed to import from parent reasoning_integration module: {e}"
    )
    ReasoningIntegration = None
    apply_reasoning = None
    run_portfolio_reasoning = None
    get_reasoning_integration = None
    get_reasoning_statistics = None
    shutdown_reasoning = None
    observe_query_start = None
    observe_engine_result = None
    observe_outcome = None
    observe_validation_failure = None
    observe_error = None


__all__ = [
    # Types
    "LOG_PREFIX",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIME_BUDGET_MS",
    "DEFAULT_ENERGY_BUDGET_MJ",
    "DEFAULT_MIN_CONFIDENCE",
    "MAX_FALLBACK_ATTEMPTS",
    "MIN_CONFIDENCE_FLOOR",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_GOOD_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "CONFIDENCE_LOW_THRESHOLD",
    "DECOMPOSITION_COMPLEXITY_THRESHOLD",
    "ANALYSIS_INDICATORS",
    "ACTION_VERBS",
    "ETHICAL_ANALYSIS_INDICATORS",
    "PURE_ETHICAL_PHRASES",
    "ReasoningStrategyType",
    "QUERY_TYPE_STRATEGY_MAP",
    "ROUTE_TO_REASONING_TYPE",
    "RoutingDecision",
    "ReasoningResult",
    "IntegrationStatistics",
    # Safety checker
    "is_false_positive_safety_block",
    "is_result_safety_filtered",
    "get_safety_filtered_fallback_tools",
    # Query router
    "get_reasoning_type_from_route",
    # Main API (from parent)
    "ReasoningIntegration",
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_integration",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
]

__version__ = "2.0.0"
__author__ = "Vulcan AI Team"
