"""
Utility functions and conveniences for reasoning integration.

This module contains:
- Convenience functions (apply_reasoning, run_portfolio_reasoning, etc.)
- Observer integration functions
- Statistics tracking
- Singleton management
- Type safety helpers for reasoning_type enum conversion

Module: vulcan.reasoning.integration.utils
Author: Vulcan AI Team
"""

import atexit
import logging
import threading
from typing import Any, Dict, List, Optional, Union

from .types import ReasoningResult
from .orchestrator import ReasoningIntegration

logger = logging.getLogger(__name__)

# Import ReasoningType enum for type conversion
try:
    from vulcan.reasoning.reasoning_types import ReasoningType
    REASONING_TYPE_AVAILABLE = True
except ImportError:
    ReasoningType = None
    REASONING_TYPE_AVAILABLE = False
    logger.warning("ReasoningType enum not available - type conversion disabled")

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
def observe_query_start(
    query_id: str,
    query: str,
    classification: Dict[str, Any]
) -> None:
    """
    Observe the start of query processing.

    Args:
        query_id: Unique query identifier
        query: Query text
        classification: Query classification (category, complexity, tools)
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer:
            observer.observe_query_start(query_id, query, classification)
    except ImportError:
        pass  # SystemObserver not available
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
    Observe reasoning engine result.

    Args:
        query_id: Query identifier
        engine_name: Name of reasoning engine
        result: Reasoning result dictionary
        success: Whether execution succeeded
        execution_time_ms: Execution time in milliseconds
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer:
            # Ensure result is a dict
            result_dict = result if isinstance(result, dict) else {'value': result}
            observer.observe_engine_result(query_id, engine_name, result_dict, success, execution_time_ms)
    except ImportError:
        pass  # SystemObserver not available
    except Exception as e:
        logger.debug(f"observe_engine_result error: {e}")


def observe_outcome(
    query_id: str,
    response: Dict[str, Any],
    user_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Observe final query outcome.

    Args:
        query_id: Query identifier
        response: Final response dictionary
        user_feedback: Optional user feedback (rating, etc)
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer:
            observer.observe_outcome(query_id, response, user_feedback)
    except ImportError:
        pass  # SystemObserver not available
    except Exception as e:
        logger.debug(f"observe_outcome error: {e}")


def observe_validation_failure(
    query_id: str,
    engine_name: str,
    reason: str,
    query: str,
    result: Dict[str, Any]
) -> None:
    """
    Observe validation failure.

    Args:
        query_id: Query identifier
        engine_name: Engine that produced invalid result
        reason: Why validation failed
        query: Original query
        result: Invalid result
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer:
            observer.observe_validation_failure(query_id, engine_name, reason, query, result)
    except ImportError:
        pass  # SystemObserver not available
    except Exception as e:
        logger.debug(f"observe_validation_failure error: {e}")


def observe_error(
    query_id: str,
    error_type: str,
    error_message: str,
    component: str
) -> None:
    """
    Observe error during reasoning.

    Args:
        query_id: Query identifier
        error_type: Type of error
        error_message: Error message
        component: Component where error occurred
    """
    try:
        from vulcan.world_model.system_observer import get_system_observer
        observer = get_system_observer()
        if observer:
            observer.observe_error(query_id, error_type, error_message, component)
    except ImportError:
        pass  # SystemObserver not available
    except Exception as e:
        logger.debug(f"observe_error error: {e}")


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
    "observe_reasoning_selection",
    "observe_reasoning_execution",
    "observe_reasoning_success",
    "observe_reasoning_failure",
    "observe_reasoning_degradation",
    "convert_reasoning_type_to_enum",
    "ensure_reasoning_type_enum",
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


# ============================================================
# TYPE SAFETY HELPERS - Enum Conversion for reasoning_type
# ============================================================

def convert_reasoning_type_to_enum(
    reasoning_type: Union[str, 'ReasoningType', None],
    context: str = "unknown"
) -> Optional['ReasoningType']:
    """
    Convert reasoning_type from string to ReasoningType enum safely.
    
    This function addresses the pipeline-dropping bug where philosophical/ethical
    results from world_model are discarded when reasoning_type is passed as a
    string instead of as a ReasoningType Enum value.
    
    The orchestrator (agent pool, collective, etc) requires Enum values and will
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
        None  # Not in enum, logs warning
        
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
                logger.info(
                    f"[{context}] Converted reasoning_type string '{reasoning_type}' "
                    f"to enum {member}"
                )
                return member
        
        # Special case mappings for common aliases
        alias_map = {
            "meta_reasoning": ReasoningType.PHILOSOPHICAL,  # Meta-reasoning uses philosophical
            "philosophical_reasoning": ReasoningType.PHILOSOPHICAL,
            "ethical_reasoning": ReasoningType.PHILOSOPHICAL,
            "math": ReasoningType.MATHEMATICAL,
            "causal_reasoning": ReasoningType.CAUSAL,
            "world_model": ReasoningType.PHILOSOPHICAL,  # World model uses philosophical
        }
        
        if reasoning_type_lower in alias_map:
            converted = alias_map[reasoning_type_lower]
            logger.info(
                f"[{context}] Converted reasoning_type alias '{reasoning_type}' "
                f"to enum {converted}"
            )
            return converted
        
        # Conversion failed - log detailed error
        logger.error(
            f"[{context}] CRITICAL: Failed to convert reasoning_type string '{reasoning_type}' "
            f"to ReasoningType enum. Valid values: {[m.value for m in ReasoningType]}. "
            f"This may cause result dropping!"
        )
        return None
    
    # Unknown type
    logger.error(
        f"[{context}] CRITICAL: reasoning_type has invalid type {type(reasoning_type).__name__}. "
        f"Expected str or ReasoningType enum. Value: {reasoning_type}"
    )
    return None


def ensure_reasoning_type_enum(
    result: Union[Dict[str, Any], Any],
    context: str = "unknown"
) -> Union[Dict[str, Any], Any]:
    """
    Ensure reasoning_type field in result is a ReasoningType enum.
    
    This function modifies the result in-place if it contains a string reasoning_type,
    converting it to the proper enum value. If conversion fails, logs full result
    context for debugging.
    
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
                    f"confidence={result.get('confidence', 'unknown')}, "
                    f"conclusion_present={('conclusion' in result) or ('response' in result)}, "
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
                # Conversion failed - log full result for debugging
                logger.error(
                    f"[{context}] DISCARDED RESULT: reasoning_type conversion failed. "
                    f"Result will likely be dropped by orchestrator. "
                    f"reasoning_type='{reasoning_type}', "
                    f"confidence={getattr(result, 'confidence', 'unknown')}, "
                    f"has_conclusion={hasattr(result, 'conclusion')}, "
                    f"result_type={type(result).__name__}"
                )
    
    return result
