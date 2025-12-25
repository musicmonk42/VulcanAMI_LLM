# ============================================================
# VULCAN-AGI Component Initialization Module
# Utilities for tracking component initialization state
# ============================================================
#
# This module provides:
#     - Singleton component initialization tracking
#     - Lazy initialization support
#     - Component lifecycle management
#
# USAGE:
#     from vulcan.utils_main.components import initialize_component, get_component
#     
#     # Initialize a component once
#     db = initialize_component("database", lambda: Database())
#     
#     # Retrieve later
#     db = get_component("database")
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and thread safety
# ============================================================

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# THREAD-SAFE COMPONENT REGISTRY
# ============================================================

# Dictionary to track initialized components
_initialized_components: Dict[str, Any] = {}

# Lock for thread-safe access
_components_lock = threading.RLock()


# ============================================================
# COMPONENT MANAGEMENT FUNCTIONS
# ============================================================


def initialize_component(name: str, func: Callable[[], Any]) -> Any:
    """
    Ensure a component is initialized only once per process.
    
    Thread-safe initialization that ensures the factory function
    is called exactly once, even with concurrent access.
    
    Args:
        name: Unique name for the component
        func: Factory function to create the component (called with no arguments)
        
    Returns:
        The initialized component (either newly created or cached)
        
    Example:
        >>> def create_db():
        ...     return DatabaseConnection()
        >>> db = initialize_component("database", create_db)
    """
    with _components_lock:
        if name not in _initialized_components:
            logger.info(f"Initializing component: {name}")
            try:
                _initialized_components[name] = func()
                logger.debug(f"Component '{name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize component '{name}': {e}")
                raise
        return _initialized_components[name]


def get_initialized_components() -> Dict[str, Any]:
    """
    Get a copy of the dictionary of all initialized components.
    
    Returns:
        Copy of the initialized components dictionary
    """
    with _components_lock:
        return dict(_initialized_components)


def set_component(name: str, component: Any) -> None:
    """
    Directly set a component in the initialized components dict.
    
    Use this when you have a pre-created component instance.
    
    Args:
        name: Unique name for the component
        component: The component instance to store
    """
    with _components_lock:
        _initialized_components[name] = component
        logger.debug(f"Component '{name}' set directly")


def get_component(name: str, default: Any = None) -> Any:
    """
    Get a specific initialized component by name.
    
    Args:
        name: Name of the component to retrieve
        default: Value to return if component not found
        
    Returns:
        The component if found, otherwise the default value
    """
    with _components_lock:
        return _initialized_components.get(name, default)


def has_component(name: str) -> bool:
    """
    Check if a component has been initialized.
    
    Args:
        name: Name of the component to check
        
    Returns:
        True if component exists, False otherwise
    """
    with _components_lock:
        return name in _initialized_components


def remove_component(name: str) -> Optional[Any]:
    """
    Remove a component from the registry.
    
    Args:
        name: Name of the component to remove
        
    Returns:
        The removed component, or None if not found
    """
    with _components_lock:
        component = _initialized_components.pop(name, None)
        if component is not None:
            logger.info(f"Component '{name}' removed from registry")
        return component


def clear_components() -> int:
    """
    Clear all initialized components (for testing or shutdown).
    
    Returns:
        Number of components cleared
    """
    with _components_lock:
        count = len(_initialized_components)
        _initialized_components.clear()
        logger.info(f"Cleared {count} components from registry")
        return count


def list_components() -> List[str]:
    """
    List all initialized component names.
    
    Returns:
        List of component names
    """
    with _components_lock:
        return list(_initialized_components.keys())


# ============================================================
# COMPONENT LIFECYCLE UTILITIES
# ============================================================


def shutdown_components() -> None:
    """
    Shutdown all components that have a shutdown/close method.
    
    Iterates through all components and calls their shutdown, close,
    or cleanup methods if available.
    """
    with _components_lock:
        for name, component in list(_initialized_components.items()):
            try:
                # Try common shutdown method names
                if hasattr(component, 'shutdown'):
                    logger.info(f"Shutting down component: {name}")
                    component.shutdown()
                elif hasattr(component, 'close'):
                    logger.info(f"Closing component: {name}")
                    component.close()
                elif hasattr(component, 'cleanup'):
                    logger.info(f"Cleaning up component: {name}")
                    component.cleanup()
            except Exception as e:
                logger.warning(f"Error shutting down component '{name}': {e}")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Core functions
    "initialize_component",
    "get_initialized_components",
    "set_component",
    "get_component",
    "has_component",
    "remove_component",
    "clear_components",
    "list_components",
    # Lifecycle
    "shutdown_components",
]


# Log module initialization
logger.debug(f"Components module v{__version__} loaded")
