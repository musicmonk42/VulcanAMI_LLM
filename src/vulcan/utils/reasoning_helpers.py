"""
Reasoning Helpers Module

Helper utilities for working with reasoning results from various reasoning engines.
This module provides safe accessors and converters for handling polymorphic
reasoning result types (dataclasses, dictionaries, objects).

Extracted from original main.py (11,316 lines) during modular refactoring (PR #704).

Author: VULCAN-AGI Team
Version: 1.0.0
"""

from typing import Any, Dict, Optional, TypeVar, Union

# Type variable for generic return types
T = TypeVar('T')


def _get_reasoning_attr(
    result: Optional[Union[Dict[str, Any], object]],
    attr: str,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Safely extract an attribute from a polymorphic reasoning result.
    
    This helper handles the diversity of reasoning result types across the platform:
    1. Dictionary results from JSON-based engines
    2. Dataclass results from structured reasoning engines (e.g., ReasoningResult)
    3. Named tuple results from legacy engines
    4. Custom objects with attributes
    5. None/null results from failed reasoning attempts
    
    The function provides a unified interface for attribute access, eliminating
    the need for type checking and error handling at every call site.
    
    Args:
        result: The reasoning result to extract from. Can be:
            - Dict[str, Any]: Dictionary with string keys
            - object: Any object with attributes (dataclass, named tuple, etc.)
            - None: Represents a failed or missing result
        attr: The attribute/key name to extract (e.g., "confidence", "conclusion")
        default: Default value to return if attribute not found (default: None)
        
    Returns:
        The attribute value if found, otherwise the default value.
        Type matches the default parameter if provided.
        
    Examples:
        Extract from dictionary result:
        >>> result = {"conclusion": "Proven", "confidence": 0.95}
        >>> _get_reasoning_attr(result, "conclusion")
        'Proven'
        >>> _get_reasoning_attr(result, "confidence")
        0.95
        
        Extract from dataclass result:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class ReasoningResult:
        ...     conclusion: str
        ...     confidence: float
        >>> result = ReasoningResult(conclusion="Verified", confidence=0.88)
        >>> _get_reasoning_attr(result, "conclusion")
        'Verified'
        
        Handle missing attributes safely:
        >>> result = {"conclusion": "Test"}
        >>> _get_reasoning_attr(result, "missing_key", "fallback")
        'fallback'
        
        Handle None results:
        >>> _get_reasoning_attr(None, "any_attr", "default_value")
        'default_value'
        
    Thread Safety:
        This function is thread-safe as it performs only read operations.
        
    Performance:
        O(1) for dictionary access, O(1) for attribute access.
        No expensive operations or allocations.
        
    See Also:
        - vulcan.reasoning.types.ReasoningResult: Primary dataclass result type
        - tests/test_reasoning_content_propagation.py: Usage examples in tests
    """
    # Handle None/null results early
    if result is None:
        return default
    
    # Handle dictionary results (most common case)
    if isinstance(result, dict):
        return result.get(attr, default)
    
    # Handle objects with attributes (dataclasses, named tuples, custom objects)
    # Use hasattr for safety to avoid AttributeError
    if hasattr(result, attr):
        return getattr(result, attr, default)
    
    # Attribute not found in any supported structure
    return default


# Module exports
__all__ = [
    "_get_reasoning_attr",
]
