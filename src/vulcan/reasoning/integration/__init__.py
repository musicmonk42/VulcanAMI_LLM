"""
Reasoning integration subpackage for VULCAN.

This package provides the integration layer between the query processing pipeline
and the reasoning subsystem. It wires together ToolSelector, PortfolioExecutor,
and reasoning strategies into a unified interface.

The package is organized into focused modules:
    - types: Core dataclasses for results and statistics
    - constants: Configuration constants and thresholds
    - safety: Safety filtering and validation
    - query_analysis: Query type detection and analysis
    - tool_selection: Tool selector integration
    - decomposition: Problem decomposition
    - world_model_bridge: World model integration
    - arena_delegation: Arena delegation logic
    - statistics: Statistics tracking
    - observer: SystemObserver integration
    - orchestrator: Main ReasoningIntegration class

Usage:
    >>> from vulcan.reasoning.integration import apply_reasoning
    >>> result = apply_reasoning(query="Explain X", query_type="reasoning")

Module: vulcan.reasoning.integration
Author: Vulcan AI Team
"""

# TODO: Complete refactoring - Currently re-exporting from parent module
# This maintains backward compatibility while refactoring is in progress

try:
    from ..reasoning_integration import (
        ReasoningIntegration,
        ReasoningResult,
        IntegrationStatistics,
        RoutingDecision,
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
    ReasoningResult = None
    IntegrationStatistics = None
    RoutingDecision = None
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
    "ReasoningIntegration",
    "ReasoningResult",
    "IntegrationStatistics",
    "RoutingDecision",
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
