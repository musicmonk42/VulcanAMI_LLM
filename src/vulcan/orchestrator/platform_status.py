"""
Platform Integration Status

This module provides functions to check the availability and health of all
Vulcan platform modules. Used for monitoring which components are successfully
loaded and integrated.

Functions:
    get_platform_integration_status - Get status of all platform modules
    log_platform_integration_status - Log platform integration summary
    get_module_registry - Get the registry of available modules
"""

import logging
from typing import Any, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Module availability flags
UTILS_MODULE_AVAILABLE = False
LLM_MODULE_AVAILABLE = False
DISTILLATION_MODULE_AVAILABLE = False
ARENA_MODULE_AVAILABLE = False
METRICS_MODULE_AVAILABLE = False
API_MODULE_AVAILABLE = False

# Module info getter functions
_get_utils_module_info: Optional[Callable[[], Dict[str, Any]]] = None
_get_llm_module_info: Optional[Callable[[], Dict[str, Any]]] = None
_get_distillation_module_info: Optional[Callable[[], Dict[str, Any]]] = None
_get_arena_module_info: Optional[Callable[[], Dict[str, Any]]] = None
_get_metrics_module_info: Optional[Callable[[], Dict[str, Any]]] = None
_get_api_module_info: Optional[Callable[[], Dict[str, Any]]] = None

# Module registry - maps module name to (available_flag, info_getter)
_MODULE_REGISTRY: Dict[str, Tuple[bool, Optional[Callable[[], Dict[str, Any]]]]] = {}


def initialize_module_registry() -> None:
    """
    Initialize the module registry by checking which modules are available.
    
    This function attempts to import each Vulcan module and registers
    successful imports with their info getter functions. Should be called
    once at application startup.
    
    Note:
        This function modifies module-level global variables. It's designed
        to be called once during initialization.
    """
    global UTILS_MODULE_AVAILABLE, LLM_MODULE_AVAILABLE
    global DISTILLATION_MODULE_AVAILABLE, ARENA_MODULE_AVAILABLE
    global METRICS_MODULE_AVAILABLE, API_MODULE_AVAILABLE
    global _get_utils_module_info, _get_llm_module_info
    global _get_distillation_module_info, _get_arena_module_info
    global _get_metrics_module_info, _get_api_module_info
    global _MODULE_REGISTRY
    
    # Try to import utils_main module
    try:
        from vulcan.utils_main import (
            get_module_info as get_utils_module_info,
            validate_utils,
        )
        UTILS_MODULE_AVAILABLE = True
        _get_utils_module_info = get_utils_module_info
    except ImportError as e:
        logger.debug(f"Utils module not available: {e}")
        UTILS_MODULE_AVAILABLE = False
    
    # Try to import LLM module
    try:
        from vulcan.llm import (
            get_module_info as get_llm_module_info,
            validate_llm_module,
        )
        LLM_MODULE_AVAILABLE = True
        _get_llm_module_info = get_llm_module_info
    except ImportError as e:
        logger.debug(f"LLM module not available: {e}")
        LLM_MODULE_AVAILABLE = False
    
    # Try to import distillation module
    try:
        from vulcan.distillation import (
            get_module_info as get_distillation_module_info,
            validate_distillation_module,
            get_knowledge_distiller,
            initialize_knowledge_distiller,
        )
        DISTILLATION_MODULE_AVAILABLE = True
        _get_distillation_module_info = get_distillation_module_info
    except ImportError as e:
        logger.debug(f"Distillation module not available: {e}")
        DISTILLATION_MODULE_AVAILABLE = False
    
    # Try to import arena module
    try:
        from vulcan.arena import (
            get_module_info as get_arena_module_info,
            validate_arena_module,
        )
        ARENA_MODULE_AVAILABLE = True
        _get_arena_module_info = get_arena_module_info
    except ImportError as e:
        logger.debug(f"Arena module not available: {e}")
        ARENA_MODULE_AVAILABLE = False
    
    # Try to import metrics module
    try:
        from vulcan.metrics import (
            get_module_info as get_metrics_module_info,
            validate_metrics_module,
        )
        METRICS_MODULE_AVAILABLE = True
        _get_metrics_module_info = get_metrics_module_info
    except ImportError as e:
        logger.debug(f"Metrics module not available: {e}")
        METRICS_MODULE_AVAILABLE = False
    
    # Try to import API module
    try:
        from vulcan.api import (
            get_module_info as get_api_module_info,
            validate_api_module,
        )
        API_MODULE_AVAILABLE = True
        _get_api_module_info = get_api_module_info
    except ImportError as e:
        logger.debug(f"API module not available: {e}")
        API_MODULE_AVAILABLE = False
    
    # Build module registry
    _MODULE_REGISTRY = {
        "utils_main": (UTILS_MODULE_AVAILABLE, _get_utils_module_info),
        "llm": (LLM_MODULE_AVAILABLE, _get_llm_module_info),
        "distillation": (DISTILLATION_MODULE_AVAILABLE, _get_distillation_module_info),
        "arena": (ARENA_MODULE_AVAILABLE, _get_arena_module_info),
        "metrics": (METRICS_MODULE_AVAILABLE, _get_metrics_module_info),
        "api": (API_MODULE_AVAILABLE, _get_api_module_info),
    }


def get_module_registry() -> Dict[str, Tuple[bool, Optional[Callable[[], Dict[str, Any]]]]]:
    """
    Get the module registry.
    
    Returns:
        Dictionary mapping module names to (available, info_getter) tuples
    
    Note:
        If the registry hasn't been initialized, this will return an empty dict.
        Call initialize_module_registry() first at application startup.
    """
    return _MODULE_REGISTRY.copy()


def get_platform_integration_status(include_details: bool = True) -> Dict[str, Any]:
    """
    Get the status of all extracted modules for platform integration.
    
    This function provides a comprehensive view of which modules are available
    and fully integrated with the platform. Useful for health checks and
    debugging module loading issues.
    
    Args:
        include_details: Whether to include detailed module info (default True).
                        Set to False for faster status checks that only return
                        availability without calling each module's info getter.
    
    Returns:
        Dictionary containing:
            - modules_available: Dict mapping module name to availability bool
            - all_modules_available: Bool indicating if all modules loaded
            - module_details: (if include_details=True) Detailed info from each module
    
    Example:
        >>> status = get_platform_integration_status(include_details=False)
        >>> if not status["all_modules_available"]:
        ...     print(f"Missing modules: {[k for k, v in status['modules_available'].items() if not v]}")
    """
    if not _MODULE_REGISTRY:
        logger.warning("Module registry not initialized. Call initialize_module_registry() first.")
        return {
            "modules_available": {},
            "all_modules_available": False,
            "error": "Module registry not initialized",
        }
    
    status = {name: available for name, (available, _) in _MODULE_REGISTRY.items()}
    
    result = {
        "modules_available": status,
        "all_modules_available": all(status.values()),
    }
    
    # Get detailed info from each module if requested
    if include_details:
        detailed_info = {}
        for module_name, (available, info_getter) in _MODULE_REGISTRY.items():
            if available and info_getter:
                try:
                    detailed_info[module_name] = info_getter()
                except Exception as e:
                    detailed_info[module_name] = {
                        "error": f"Failed to get {module_name} module info: {type(e).__name__}: {e}"
                    }
            else:
                detailed_info[module_name] = {
                    "available": False,
                    "error": "Module not loaded or info getter not available"
                }
        result["module_details"] = detailed_info
    
    return result


def log_platform_integration_status() -> None:
    """
    Log platform integration status summary.
    
    Logs a concise summary of which modules are available and which are missing.
    Called lazily to avoid slowing down imports. Typically called after all
    modules have been initialized.
    
    Note:
        This function uses INFO level for success and WARNING level for
        missing modules, making it suitable for production monitoring.
    """
    status = get_platform_integration_status(include_details=False)
    
    if "error" in status:
        logger.warning(f"Platform status check failed: {status['error']}")
        return
    
    if status["all_modules_available"]:
        logger.info("✓ All extracted modules are available for platform integration")
    else:
        unavailable = [k for k, v in status["modules_available"].items() if not v]
        logger.warning(f"Some extracted modules unavailable: {unavailable}")
        
        # Log which modules ARE available for debugging
        available = [k for k, v in status["modules_available"].items() if v]
        if available:
            logger.info(f"Available modules: {available}")
