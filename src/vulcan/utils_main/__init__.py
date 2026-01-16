# ============================================================
# VULCAN-AGI Utils Package
# Utility functions and classes for the VULCAN-AGI system
# ============================================================
#
# MODULES:
#     process_lock  - Cross-platform file-based process lock (uses filelock)
#     timing        - Performance instrumentation decorators
#     sanitize      - JSON sanitization utilities
#     components    - DEPRECATED: Use vulcan.reasoning.singletons instead
#     network       - Network utilities (port scanning)
#
# USAGE:
#     from vulcan.utils_main import ProcessLock, timed_async, sanitize_payload
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
import sys
from typing import Any, Callable, Dict, Optional

# Module metadata
__version__ = "2.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CORE IMPORTS WITH AVAILABILITY TRACKING
# ============================================================

_imports_successful = True

try:
    from vulcan.utils_main.process_lock import (
        ProcessLock,
        FCNTL_AVAILABLE,
        FILELOCK_AVAILABLE,
        get_process_lock,
        set_process_lock,
    )
except ImportError as e:
    logger.warning(f"process_lock module not available: {e}")
    _imports_successful = False
    ProcessLock = None
    FCNTL_AVAILABLE = False
    FILELOCK_AVAILABLE = False
    get_process_lock = None
    set_process_lock = None

try:
    from vulcan.utils_main.timing import (
        timed_async,
        timed_sync,
        run_tasks_in_parallel,
        SLOW_OPERATION_THRESHOLD_MS,
    )
except ImportError as e:
    logger.warning(f"timing module not available: {e}")
    _imports_successful = False
    timed_async = None
    timed_sync = None
    run_tasks_in_parallel = None
    SLOW_OPERATION_THRESHOLD_MS = 100

try:
    from vulcan.utils_main.sanitize import (
        sanitize_payload,
        deep_sanitize_for_json,
        DEEP_SANITIZE_MAX_DEPTH,
        # Backward compatibility aliases
        _sanitize_payload,
        _deep_sanitize_for_json,
    )
except ImportError as e:
    logger.warning(f"sanitize module not available: {e}")
    _imports_successful = False
    sanitize_payload = None
    deep_sanitize_for_json = None
    DEEP_SANITIZE_MAX_DEPTH = 50
    _sanitize_payload = None
    _deep_sanitize_for_json = None

try:
    from vulcan.utils_main.components import (
        initialize_component,
        get_initialized_components,
        set_component,
        get_component,
        clear_components,
    )
except ImportError as e:
    logger.warning(f"components module not available: {e}")
    _imports_successful = False
    initialize_component = None
    get_initialized_components = None
    set_component = None
    get_component = None
    clear_components = None

try:
    from vulcan.utils_main.network import find_available_port
except ImportError as e:
    logger.warning(f"network module not available: {e}")
    _imports_successful = False
    find_available_port = None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Process Lock
    "ProcessLock",
    "FCNTL_AVAILABLE",  # Backward compatibility
    "FILELOCK_AVAILABLE",
    "get_process_lock",
    "set_process_lock",
    # Timing
    "timed_async",
    "timed_sync",
    "run_tasks_in_parallel",
    "SLOW_OPERATION_THRESHOLD_MS",
    # Sanitize
    "sanitize_payload",
    "deep_sanitize_for_json",
    "DEEP_SANITIZE_MAX_DEPTH",
    "_sanitize_payload",  # Backward compatibility
    "_deep_sanitize_for_json",  # Backward compatibility
    # Components (DEPRECATED - use vulcan.reasoning.singletons)
    "initialize_component",
    "get_initialized_components",
    "set_component",
    "get_component",
    "clear_components",
    # Network
    "find_available_port",
    # Module info
    "get_module_info",
    "validate_utils",
]


# ============================================================
# MODULE UTILITIES
# ============================================================


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the utils module.
    
    Returns:
        Dictionary with module information and availability status
    """
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "python_version": sys.version,
        "components": {
            "process_lock": ProcessLock is not None,
            "timing": timed_async is not None,
            "sanitize": sanitize_payload is not None,
            "components": initialize_component is not None,
            "network": find_available_port is not None,
        },
        "fcntl_available": FCNTL_AVAILABLE,  # Backward compatibility
        "filelock_available": FILELOCK_AVAILABLE,
    }


def validate_utils() -> bool:
    """
    Validate that all utility modules are properly loaded.
    
    Returns:
        True if all modules loaded successfully, False otherwise
    """
    try:
        info = get_module_info()
        all_available = all(info["components"].values())
        
        if all_available:
            logger.info("All utils modules validated successfully")
        else:
            missing = [k for k, v in info["components"].items() if not v]
            logger.warning(f"Some utils modules unavailable: {missing}")
        
        return all_available
        
    except Exception as e:
        logger.error(f"Utils validation failed: {e}")
        return False


# ============================================================
# MODULE INITIALIZATION
# ============================================================

if _imports_successful:
    logger.info(f"VULCAN-AGI Utils module v{__version__} loaded successfully")
else:
    logger.warning(f"VULCAN-AGI Utils module v{__version__} loaded with some components unavailable")
