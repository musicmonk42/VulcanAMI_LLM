"""
Reasoning integration subpackage for VULCAN.

This package provides the integration layer between the query processing pipeline
and the reasoning subsystem.

Refactored modules:
    - types: Core dataclasses, enums, and constants ✅
    - safety_checker: Safety validation and false positive detection ✅
    - query_router: Query type routing and classification ✅
    - orchestrator: Main ReasoningIntegration class ✅ (stub - delegates to parent)
    - component_init: Component initialization methods ✅ (stub)
    - selection_strategies: Tool selection logic ✅ (stub)
    - query_analysis: Query analysis methods ✅ (stub)
    - utils: Convenience functions and observers ✅ (stub - delegates to parent)

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

# Re-export orchestrator (stub - currently delegates to parent)
from .orchestrator import (
    ReasoningIntegration,
)

# Re-export utils (stub - currently delegates to parent)
from .utils import (
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
    # Orchestrator
    "ReasoningIntegration",
    # Utils
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
