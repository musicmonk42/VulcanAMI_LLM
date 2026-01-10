"""
Unified Reasoning Package

This package provides a unified interface for reasoning across multiple reasoning engines
and strategies. It orchestrates probabilistic, symbolic, causal, and analogical reasoning
with sophisticated tool selection, learning, and adaptation mechanisms.

Main Components:
- UnifiedReasoner: Main orchestrator class
- ReasoningTask, ReasoningPlan: Core data structures
- ToolWeightManager: Weight management for ensemble reasoning
- Strategy functions: Sequential, parallel, ensemble, adaptive, etc.

Submodules:
- types: Core dataclasses and types
- config: Configuration constants
- component_loader: Lazy component loading
- cache: Tool weight management and query hashing
- strategies: Reasoning execution strategies
- orchestrator: Main UnifiedReasoner class
- multimodal_handler: Multimodal reasoning methods
- persistence: State save/load functionality

Usage:
    from vulcan.reasoning.unified import UnifiedReasoner
    
    reasoner = UnifiedReasoner(enable_learning=True)
    result = reasoner.reason({"query": "What causes rain?"})
"""

# Import core types
from .types import ReasoningTask, ReasoningPlan

# Import configuration constants
from .config import *

# Import component loader functions
from .component_loader import (
    _load_reasoning_components,
    _load_selection_components,
    _load_optional_components,
)

# Import cache management
from .cache import ToolWeightManager, compute_query_hash, get_weight_manager

# Re-export ReasoningStrategy for backward compatibility
from ..reasoning_types import ReasoningStrategy

# Import strategy functions
from .strategies import (
    execute_sequential_reasoning as _sequential_reasoning,
    execute_parallel_reasoning as _parallel_reasoning,
    execute_ensemble_reasoning as _ensemble_reasoning,
    execute_adaptive_reasoning as _adaptive_reasoning,
    execute_hybrid_reasoning as _hybrid_reasoning,
    execute_hierarchical_reasoning as _hierarchical_reasoning,
    execute_portfolio_reasoning as _portfolio_reasoning,
    execute_utility_based_reasoning as _utility_based_reasoning,
    weighted_voting as _weighted_voting,
    combine_parallel_results as _combine_parallel_results,
    topological_sort as _topological_sort,
    merge_dependency_results as _merge_dependency_results,
)

# Import main orchestrator
from .orchestrator import UnifiedReasoner

# Import multimodal methods
from .multimodal_handler import (
    reason_multimodal,
    reason_counterfactual,
    reason_by_analogy,
)

# Import persistence functions
from .persistence import save_state, load_state

__all__ = [
    # Core types
    "ReasoningTask",
    "ReasoningPlan",
    "ReasoningStrategy",  # Re-exported from reasoning_types for backward compatibility
    # Component loaders
    "_load_reasoning_components",
    "_load_selection_components",
    "_load_optional_components",
    # Cache management
    "ToolWeightManager",
    "compute_query_hash",
    "get_weight_manager",
    # Strategy functions
    "_sequential_reasoning",
    "_parallel_reasoning",
    "_ensemble_reasoning",
    "_adaptive_reasoning",
    "_hybrid_reasoning",
    "_hierarchical_reasoning",
    "_portfolio_reasoning",
    "_utility_based_reasoning",
    "_weighted_voting",
    "_combine_parallel_results",
    "_topological_sort",
    "_merge_dependency_results",
    # Main class
    "UnifiedReasoner",
    # Multimodal methods
    "reason_multimodal",
    "reason_counterfactual",
    "reason_by_analogy",
    # Persistence
    "save_state",
    "load_state",
]
