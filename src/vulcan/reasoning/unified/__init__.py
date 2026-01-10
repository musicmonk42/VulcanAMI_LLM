"""
Unified reasoning subpackage for VULCAN.

This package provides the main unified reasoning orchestrator that coordinates
multiple reasoning paradigms (symbolic, causal, probabilistic, analogical, etc.)
through adaptive strategy selection and portfolio execution.

The package is organized into focused modules:
    - types: Core dataclasses for reasoning tasks and plans
    - config: Configuration constants and environment loading
    - component_loader: Component initialization and dependency injection
    - strategies: Reasoning strategy implementations
    - cache: Result caching with LRU eviction
    - orchestrator: Main UnifiedReasoner class
    - multimodal_handler: Multimodal and counterfactual reasoning
    - persistence: State persistence and model serialization

Usage:
    >>> from vulcan.reasoning.unified import UnifiedReasoner
    >>> reasoner = UnifiedReasoner()
    >>> result = reasoner.reason(query="What causes X?", strategy="causal")

Module: vulcan.reasoning.unified
Author: Vulcan AI Team
"""

# TODO: Complete refactoring - Currently re-exporting from parent module
# This maintains backward compatibility while refactoring is in progress

try:
    from ..unified_reasoning import (
        UnifiedReasoner,
        ReasoningTask,
        ReasoningPlan,
        ToolWeightManager,
        get_weight_manager,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Failed to import from parent unified_reasoning module: {e}"
    )
    UnifiedReasoner = None
    ReasoningTask = None
    ReasoningPlan = None
    ToolWeightManager = None
    get_weight_manager = None


__all__ = [
    "UnifiedReasoner",
    "ReasoningTask",
    "ReasoningPlan",
    "ToolWeightManager",
    "get_weight_manager",
]

__version__ = "2.0.0"
__author__ = "Vulcan AI Team"
