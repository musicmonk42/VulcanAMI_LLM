"""
Unified Reasoning Package
=========================

This package provides a unified interface for reasoning across multiple reasoning engines
and strategies. It orchestrates probabilistic, symbolic, causal, and analogical reasoning
with sophisticated tool selection, learning, and adaptation mechanisms.

Main Components:
    - UnifiedReasoner: Main orchestrator class
    - ReasoningTask, ReasoningPlan: Core data structures
    - ToolWeightManager: Weight management for ensemble reasoning
    - Strategy functions: Sequential, parallel, ensemble, adaptive, etc.

Planning Integration:
    The orchestrator integrates with vulcan.planning module for:
    - Plan creation and optimization using Plan class
    - Cost and duration estimation using Plan.total_cost and Plan.expected_duration
    - Topological sorting using Plan.optimize()

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
    >>> from vulcan.reasoning.unified import UnifiedReasoner
    >>> reasoner = UnifiedReasoner(enable_learning=True)
    >>> result = reasoner.reason({"query": "What causes rain?"})

Author: VulcanAMI Team
Version: 2.1 (Planning integration)
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
    topological_sort_using_plan as _topological_sort_using_plan,
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

# ============================================================================
# BACKWARDS COMPATIBILITY LAYER (Phase 1)
# ============================================================================
# These functions provide compatibility with the legacy integration package.
# They emit deprecation warnings to help track usage during migration.
# All observer functions and type conversion utilities are preserved here.
# ============================================================================

import warnings
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import ReasoningType enum for type conversion
try:
    from ..reasoning_types import ReasoningType
    REASONING_TYPE_AVAILABLE = True
except ImportError:
    ReasoningType = None
    REASONING_TYPE_AVAILABLE = False
    logger.warning("ReasoningType enum not available - type conversion disabled")


def get_reasoning_integration(config: Optional[Dict[str, Any]] = None):
    """
    DEPRECATED: Use UnifiedReasoner directly instead.
    
    Compatibility wrapper that returns a UnifiedReasoner instance.
    This function exists for backwards compatibility with code that
    previously used vulcan.reasoning.integration.get_reasoning_integration().
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        UnifiedReasoner instance
        
    Example:
        >>> # OLD (deprecated):
        >>> integration = get_reasoning_integration()
        >>> 
        >>> # NEW (preferred):
        >>> from vulcan.reasoning.unified import UnifiedReasoner
        >>> reasoner = UnifiedReasoner()
    """
    warnings.warn(
        "get_reasoning_integration() is deprecated. "
        "Use 'from vulcan.reasoning.unified import UnifiedReasoner' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to get singleton first to avoid creating multiple instances
    try:
        from ..singletons import get_unified_reasoner
        reasoner = get_unified_reasoner()
        if reasoner is not None:
            return reasoner
    except ImportError:
        pass
    
    # Fallback to creating new instance
    return UnifiedReasoner(config=config)


def apply_reasoning(
    query: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
    selected_tools: Optional[List[str]] = None,
    skip_tool_selection: bool = False,
) -> Any:
    """
    DEPRECATED: Use UnifiedReasoner.reason() directly instead.
    
    Compatibility wrapper for the legacy apply_reasoning() function.
    
    **SINGLE AUTHORITY PATTERN (Chain of Command Fix):**
    If `selected_tools` is provided with `skip_tool_selection=True`,
    the tools are used WITHOUT re-selection. This honors ToolSelector's
    authoritative decision and prevents competing tool selections.
    
    Args:
        query: The user query to process
        query_type: Type from router (general, reasoning, execution, etc.)
        complexity: Complexity score (0.0 to 1.0)
        context: Optional context dict
        selected_tools: Pre-selected tools from ToolSelector (authoritative)
        skip_tool_selection: If True, use selected_tools without re-selecting
        
    Returns:
        ReasoningResult from UnifiedReasoner
        
    Example:
        >>> # OLD (deprecated):
        >>> from vulcan.reasoning.integration import apply_reasoning
        >>> result = apply_reasoning("query", "reasoning", 0.7)
        >>> 
        >>> # NEW (preferred):
        >>> from vulcan.reasoning.unified import UnifiedReasoner
        >>> from vulcan.reasoning.singletons import get_unified_reasoner
        >>> reasoner = get_unified_reasoner() or UnifiedReasoner()
        >>> result = reasoner.reason({"query": "query", "type": query_type})
        >>>
        >>> # AUTHORITY PATTERN (with pre-selected tools):
        >>> result = reasoner.reason(
        ...     {"query": "query"},
        ...     pre_selected_tools=["symbolic"],
        ...     skip_tool_selection=True
        ... )
    """
    warnings.warn(
        "apply_reasoning() is deprecated. "
        "Use 'UnifiedReasoner().reason()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to get singleton unified reasoner
    try:
        from ..singletons import get_unified_reasoner
        reasoner = get_unified_reasoner()
        if reasoner is None:
            reasoner = UnifiedReasoner()
    except ImportError:
        reasoner = UnifiedReasoner()
    
    # Convert to UnifiedReasoner input format
    # Pass pre-selected tools to honor ToolSelector authority
    return reasoner.reason(
        input_data={
            "query": query,
            "type": query_type,
            "complexity": complexity,
            "context": context or {},
        },
        pre_selected_tools=selected_tools,
        skip_tool_selection=skip_tool_selection,
    )


def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    DEPRECATED: Use UnifiedReasoner portfolio strategy instead.
    
    Compatibility wrapper for portfolio execution.
    
    Args:
        query: The user query
        tools: List of tools to use
        strategy: Execution strategy
        constraints: Optional execution constraints
        
    Returns:
        Portfolio execution result dictionary
    """
    warnings.warn(
        "run_portfolio_reasoning() is deprecated. "
        "Use UnifiedReasoner with PORTFOLIO strategy instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        from ..singletons import get_unified_reasoner
        reasoner = get_unified_reasoner()
        if reasoner is None:
            reasoner = UnifiedReasoner()
    except ImportError:
        reasoner = UnifiedReasoner()
    
    # Use portfolio strategy
    return reasoner.reason(
        {"query": query, "tools": tools},
        strategy=ReasoningStrategy.PORTFOLIO,
        constraints=constraints
    )


def get_reasoning_statistics() -> Dict[str, Any]:
    """
    DEPRECATED: Use UnifiedReasoner.get_statistics() instead.
    
    Get reasoning statistics from the unified reasoner.
    
    Returns:
        Statistics dictionary
    """
    warnings.warn(
        "get_reasoning_statistics() is deprecated. "
        "Use 'reasoner.get_statistics()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        from ..singletons import get_unified_reasoner
        reasoner = get_unified_reasoner()
        if reasoner is not None:
            return reasoner.get_statistics()
    except (ImportError, AttributeError):
        pass
    
    return {}


def shutdown_reasoning(timeout: float = 5.0) -> None:
    """
    DEPRECATED: Use UnifiedReasoner.shutdown() instead.
    
    Shutdown reasoning system gracefully.
    
    Args:
        timeout: Shutdown timeout in seconds
    """
    warnings.warn(
        "shutdown_reasoning() is deprecated. "
        "Use 'reasoner.shutdown()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        from ..singletons import get_unified_reasoner
        reasoner = get_unified_reasoner()
        if reasoner is not None and hasattr(reasoner, 'shutdown'):
            reasoner.shutdown(timeout=timeout)
    except (ImportError, AttributeError):
        pass


# ============================================================================
# OBSERVER FUNCTIONS - SystemObserver Integration
# ============================================================================
# These functions integrate with vulcan.world_model.system_observer for
# tracking reasoning events. They are preserved for backward compatibility.
# ============================================================================

def observe_query_start(
    query_id: str,
    query: str,
    classification: Dict[str, Any]
) -> None:
    """
    Observe and record the start of a query.
    
    Integrates with SystemObserver if available, otherwise logs locally.
    
    Args:
        query_id: Unique query identifier
        query: The query text
        classification: Query classification metadata
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer is not None:
            observer.observe_query_start(query_id, query, classification)
    except ImportError:
        logger.debug("SystemObserver not available - query start not recorded")
    except Exception as e:
        logger.debug(f"observe_query_start error: {e}")


def observe_engine_result(
    query_id: str,
    engine_name: str,
    result: Any,
    success: bool,
    execution_time_ms: float
) -> None:
    """
    Observe and record an engine execution result.
    
    Args:
        query_id: Unique query identifier
        engine_name: Name of the reasoning engine
        result: The execution result
        success: Whether execution succeeded
        execution_time_ms: Execution time in milliseconds
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer is not None:
            # Convert non-dict results to dict format
            result_dict = result if isinstance(result, dict) else {'value': result}
            observer.observe_engine_result(
                query_id, engine_name, result_dict, success, execution_time_ms
            )
    except ImportError:
        logger.debug("SystemObserver not available - engine result not recorded")
    except Exception as e:
        logger.debug(f"observe_engine_result error: {e}")


def observe_outcome(
    query_id: str,
    response: Dict[str, Any],
    user_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Observe and record query outcome with optional user feedback.
    
    Args:
        query_id: Unique query identifier
        response: The response sent to the user
        user_feedback: Optional user feedback
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer is not None:
            observer.observe_outcome(query_id, response, user_feedback)
    except ImportError:
        logger.debug("SystemObserver not available - outcome not recorded")
    except Exception as e:
        logger.debug(f"observe_outcome error: {e}")


def observe_validation_failure(
    query_id: str,
    engine_name: str,
    reason: str,
    query: str,
    result: Any
) -> None:
    """
    Observe and record a validation failure.
    
    Args:
        query_id: Unique query identifier
        engine_name: Name of the reasoning engine
        reason: Failure reason
        query: The original query
        result: The failed result
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer is not None:
            observer.observe_validation_failure(
                query_id, engine_name, reason, query, result
            )
    except ImportError:
        logger.debug("SystemObserver not available - validation failure not recorded")
    except Exception as e:
        logger.debug(f"observe_validation_failure error: {e}")


def observe_error(
    query_id: str,
    error_type: str,
    error_message: str,
    component: str
) -> None:
    """
    Observe and record an error.
    
    Args:
        query_id: Unique query identifier
        error_type: Type of error
        error_message: Error message
        component: Component where error occurred
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer is not None:
            observer.observe_error(query_id, error_type, error_message, component)
    except ImportError:
        logger.debug("SystemObserver not available - error not recorded")
    except Exception as e:
        logger.debug(f"observe_error error: {e}")


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


def observe_reasoning_success(query: str, result: Any, confidence: float) -> None:
    """
    Observe and log successful reasoning.
    
    Args:
        query: The query being processed
        result: The reasoning result
        confidence: Result confidence score
    """
    logger.info(f"Reasoning success: confidence={confidence:.2f}")


def observe_reasoning_failure(query: str, error: str, tools: List[str]) -> None:
    """
    Observe and log reasoning failure.
    
    Args:
        query: The query being processed
        error: Error message
        tools: List of tools that failed
    """
    logger.warning(f"Reasoning failure: error={error}, tools={tools}")


def observe_reasoning_degradation(query: str, expected: float, actual: float) -> None:
    """
    Observe and log reasoning performance degradation.
    
    Args:
        query: The query being processed
        expected: Expected performance metric
        actual: Actual performance metric
    """
    logger.warning(f"Reasoning degradation: expected={expected}, actual={actual}")


# ============================================================================
# TYPE CONVERSION UTILITIES
# ============================================================================
# These functions ensure type safety when converting between string and enum
# reasoning types. Critical for preventing pipeline result dropping.
# ============================================================================

def convert_reasoning_type_to_enum(
    reasoning_type: Union[str, "ReasoningType", None],
    context: str = "unknown"
) -> Optional["ReasoningType"]:
    """
    Convert reasoning_type string to ReasoningType enum with comprehensive alias support.
    
    CRITICAL: Pipeline Dropping Bug Fix - The orchestrator requires Enum values and will
    crash or discard valid answers on type mismatch. This function provides safe
    conversion with comprehensive logging.
    
    Args:
        reasoning_type: The reasoning type as string, Enum, or None
        context: Context string for logging (e.g., "agent_pool", "orchestrator")
        
    Returns:
        ReasoningType enum value if conversion successful, None otherwise
        
    Examples:
        >>> convert_reasoning_type_to_enum("philosophical", "agent_pool")
        ReasoningType.PHILOSOPHICAL
        
        >>> convert_reasoning_type_to_enum("meta_reasoning", "orchestrator")
        ReasoningType.PHILOSOPHICAL  # Alias mapped to PHILOSOPHICAL
        
        >>> convert_reasoning_type_to_enum(ReasoningType.CAUSAL, "orchestrator")
        ReasoningType.CAUSAL  # Already enum, pass through
    """
    if not REASONING_TYPE_AVAILABLE or ReasoningType is None:
        logger.warning(
            f"[{context}] Cannot convert reasoning_type - ReasoningType enum not available"
        )
        return None
    
    # Already an enum - pass through
    if isinstance(reasoning_type, ReasoningType):
        return reasoning_type
    
    # None value - return None
    if reasoning_type is None:
        return None
    
    # String value - attempt conversion
    if isinstance(reasoning_type, str):
        # Try direct attribute access first (e.g., "PHILOSOPHICAL" -> ReasoningType.PHILOSOPHICAL)
        try:
            return ReasoningType[reasoning_type.upper()]
        except KeyError:
            pass
        
        # Try value matching (e.g., "philosophical" -> ReasoningType.PHILOSOPHICAL)
        reasoning_type_lower = reasoning_type.lower()
        for member in ReasoningType:
            if member.value.lower() == reasoning_type_lower:
                logger.debug(
                    f"[{context}] Converted reasoning_type string '{reasoning_type}' "
                    f"to enum {member}"
                )
                return member
        
        # Special case mappings for common aliases
        alias_map = {
            "meta_reasoning": ReasoningType.PHILOSOPHICAL,
            "philosophical_reasoning": ReasoningType.PHILOSOPHICAL,
            "ethical_reasoning": ReasoningType.PHILOSOPHICAL,
            "math": ReasoningType.MATHEMATICAL,
            "causal_reasoning": ReasoningType.CAUSAL,
            "world_model": ReasoningType.PHILOSOPHICAL,
        }
        
        if reasoning_type_lower in alias_map:
            converted = alias_map[reasoning_type_lower]
            logger.debug(
                f"[{context}] Converted reasoning_type alias '{reasoning_type}' "
                f"to enum {converted}"
            )
            return converted
        
        # Conversion failed - log detailed error
        logger.error(
            f"[{context}] CRITICAL: Failed to convert reasoning_type string '{reasoning_type}' "
            f"to ReasoningType enum. This may cause result dropping!"
        )
        return None
    
    # Unknown type
    logger.error(
        f"[{context}] CRITICAL: reasoning_type has invalid type {type(reasoning_type).__name__}. "
        f"Expected str or ReasoningType enum."
    )
    return None


def ensure_reasoning_type_enum(
    result: Union[Dict[str, Any], Any],
    context: str = "unknown"
) -> Union[Dict[str, Any], Any]:
    """
    Ensure reasoning_type field in result is a ReasoningType enum.
    
    This function modifies the result in-place if it contains a string reasoning_type,
    converting it to the proper enum value. Critical for preventing result dropping.
    
    Args:
        result: Result dictionary or object with reasoning_type field
        context: Context string for logging
        
    Returns:
        Modified result with enum reasoning_type (or original if no conversion needed)
        
    Example:
        >>> result = {"reasoning_type": "philosophical", "confidence": 0.9}
        >>> ensure_reasoning_type_enum(result, "agent_pool")
        {"reasoning_type": ReasoningType.PHILOSOPHICAL, "confidence": 0.9}
    """
    if result is None:
        return result
    
    # Handle dictionary format
    if isinstance(result, dict):
        reasoning_type = result.get("reasoning_type")
        if reasoning_type is not None:
            converted = convert_reasoning_type_to_enum(reasoning_type, context)
            if converted is not None:
                result["reasoning_type"] = converted
            elif isinstance(reasoning_type, str):
                # Conversion failed - log full result for debugging
                logger.error(
                    f"[{context}] DISCARDED RESULT: reasoning_type conversion failed. "
                    f"Result will likely be dropped by orchestrator. "
                    f"reasoning_type='{reasoning_type}', "
                    f"full_result_keys={list(result.keys())}"
                )
        return result
    
    # Handle object format (e.g., ReasoningResult)
    if hasattr(result, "reasoning_type"):
        reasoning_type = getattr(result, "reasoning_type", None)
        if reasoning_type is not None:
            converted = convert_reasoning_type_to_enum(reasoning_type, context)
            if converted is not None and not isinstance(reasoning_type, ReasoningType):
                try:
                    setattr(result, "reasoning_type", converted)
                except AttributeError:
                    # Frozen dataclass or read-only attribute
                    logger.warning(
                        f"[{context}] Cannot modify reasoning_type attribute "
                        f"(frozen/read-only). Type: {type(result).__name__}"
                    )
            elif isinstance(reasoning_type, str):
                # Conversion failed
                logger.error(
                    f"[{context}] DISCARDED RESULT: reasoning_type conversion failed on object"
                )
    
    return result


# Alias for backwards compatibility
ReasoningIntegration = UnifiedReasoner


# ============================================================================
# END BACKWARDS COMPATIBILITY LAYER
# ============================================================================

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
    "_topological_sort_using_plan",  # Alternative using Plan.optimize()
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
    # ===== BACKWARDS COMPATIBILITY LAYER =====
    # Legacy integration package compatibility
    "get_reasoning_integration",
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    "ReasoningIntegration",  # Alias for UnifiedReasoner
    # Observer functions
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
    "observe_reasoning_selection",
    "observe_reasoning_execution",
    "observe_reasoning_success",
    "observe_reasoning_failure",
    "observe_reasoning_degradation",
    # Type conversion utilities
    "convert_reasoning_type_to_enum",
    "ensure_reasoning_type_enum",
]
