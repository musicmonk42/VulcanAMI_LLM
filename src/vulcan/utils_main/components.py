"""
Component Initialization Module - DEPRECATED

This module is deprecated. Use vulcan.reasoning.singletons instead.

All functions now redirect to the centralized singleton registry.
"""

import warnings
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__author__ = "VULCAN-AGI Team"

# Emit deprecation warning on import
warnings.warn(
    "vulcan.utils_main.components is deprecated. "
    "Use vulcan.reasoning.singletons instead.",
    DeprecationWarning,
    stacklevel=2
)


def initialize_component(name: str, func: Callable[[], Any]) -> Any:
    """
    DEPRECATED: Use vulcan.reasoning.singletons.get_or_create() instead.
    """
    warnings.warn(
        "initialize_component() is deprecated. "
        "Use singletons.get_or_create() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from vulcan.reasoning.singletons import get_or_create
    return get_or_create(name, func)


def get_initialized_components() -> Dict[str, Any]:
    """DEPRECATED: Returns empty dict for backwards compatibility."""
    warnings.warn(
        "get_initialized_components() is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    return {}


def set_component(name: str, component: Any) -> None:
    """DEPRECATED: No-op for backwards compatibility."""
    warnings.warn(
        "set_component() is deprecated. Use singletons registry instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.debug(f"set_component('{name}') called but deprecated - no action taken")


def get_component(name: str, default: Any = None) -> Any:
    """DEPRECATED: Use singletons.get_singleton() instead."""
    warnings.warn(
        "get_component() is deprecated. Use singletons.get_singleton() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from vulcan.reasoning.singletons import get_singleton
        return get_singleton(name)
    except (ImportError, ValueError):
        return default


def clear_components() -> int:
    """DEPRECATED: Use singletons.reset_all() instead."""
    warnings.warn(
        "clear_components() is deprecated. Use singletons.reset_all() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from vulcan.reasoning.singletons import reset_all
        reset_all()
    except ImportError:
        pass
    return 0


# Maintain backward compatibility for other functions that may be in use
def has_component(name: str) -> bool:
    """DEPRECATED: Check if a component exists."""
    warnings.warn(
        "has_component() is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    return False


def remove_component(name: str) -> Optional[Any]:
    """DEPRECATED: Remove a component."""
    warnings.warn(
        "remove_component() is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    return None


def list_components() -> List[str]:
    """DEPRECATED: List components."""
    warnings.warn(
        "list_components() is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    return []


def shutdown_components() -> None:
    """DEPRECATED: Shutdown components."""
    warnings.warn(
        "shutdown_components() is deprecated.",
        DeprecationWarning,
        stacklevel=2
    )
    pass


__all__ = [
    "initialize_component",
    "get_initialized_components",
    "set_component",
    "get_component",
    "clear_components",
    "has_component",
    "remove_component",
    "list_components",
    "shutdown_components",
]
