# ============================================================
# VULCAN-AGI API Package
# FastAPI application components for VULCAN-AGI
# ============================================================
#
# This package provides:
#     - Request/Response models (Pydantic)
#     - Middleware (API key validation, rate limiting, security headers)
#     - Rate limiting storage and cleanup
#     - Utility functions for API operations
#
# USAGE:
#     from vulcan.api import (
#         StepRequest,
#         StepResponse,
#         validate_api_key_middleware,
#         rate_limiting_middleware,
#     )
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
from typing import Any, Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CORE IMPORTS
# ============================================================

_imports_successful = True

try:
    from vulcan.api.models import (
        # Enums
        HealthStatus,
        ErrorType,
        # Request models
        StepRequest,
        ChatRequest,
        # Response models
        StepResponse,
        ChatMessage,
        ChatResponse,
        VulcanResponse,
        StatusResponse,
        ConfigResponse,
        ImprovementApproval,
        HealthResponse,
        MetricsResponse,
        ErrorResponse,
    )
except ImportError as e:
    logger.warning(f"API models not available: {e}")
    _imports_successful = False
    HealthStatus = None
    ErrorType = None
    StepRequest = None
    ChatRequest = None
    StepResponse = None
    ChatMessage = None
    ChatResponse = None
    VulcanResponse = None
    StatusResponse = None
    ConfigResponse = None
    ImprovementApproval = None
    HealthResponse = None
    MetricsResponse = None
    ErrorResponse = None

try:
    from vulcan.api.rate_limiting import (
        rate_limit_storage,
        rate_limit_lock,
        rate_limit_cleanup_thread,
        cleanup_rate_limits,
        start_rate_limit_cleanup,
        stop_rate_limit_cleanup,
        check_rate_limit,
    )
except ImportError as e:
    logger.warning(f"Rate limiting module not available: {e}")
    _imports_successful = False
    rate_limit_storage = {}
    rate_limit_lock = None
    rate_limit_cleanup_thread = None
    cleanup_rate_limits = None
    start_rate_limit_cleanup = None
    stop_rate_limit_cleanup = None
    check_rate_limit = None

try:
    from vulcan.api.middleware import (
        validate_api_key_middleware,
        rate_limiting_middleware,
    )
except ImportError as e:
    logger.warning(f"Middleware module not available: {e}")
    _imports_successful = False
    validate_api_key_middleware = None
    rate_limiting_middleware = None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Enums
    "HealthStatus",
    "ErrorType",
    # Request models
    "StepRequest",
    "ChatRequest",
    # Response models
    "StepResponse",
    "ChatMessage",
    "ChatResponse",
    "VulcanResponse",
    "StatusResponse",
    "ConfigResponse",
    "ImprovementApproval",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
    # Rate limiting
    "rate_limit_storage",
    "rate_limit_lock",
    "rate_limit_cleanup_thread",
    "cleanup_rate_limits",
    "start_rate_limit_cleanup",
    "stop_rate_limit_cleanup",
    "check_rate_limit",
    # Middleware
    "validate_api_key_middleware",
    "rate_limiting_middleware",
    # Module utilities
    "get_module_info",
    "validate_api_module",
]


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the API module.
    
    Returns:
        Dictionary containing module information and component availability
    """
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "components": {
            "models": StepRequest is not None,
            "rate_limiting": check_rate_limit is not None,
        },
        "models_available": [
            name for name in [
                "HealthStatus", "ErrorType",
                "StepRequest", "StepResponse", "ChatMessage", "ChatRequest",
                "ChatResponse", "VulcanResponse", "StatusResponse", "ConfigResponse", 
                "ImprovementApproval", "HealthResponse", "MetricsResponse", "ErrorResponse"
            ] if globals().get(name) is not None
        ],
    }


def validate_api_module() -> bool:
    """
    Validate that the API module is properly loaded and functional.
    
    Returns:
        True if module is functional, False otherwise
    """
    try:
        info = get_module_info()
        
        # Core components must be available
        all_available = all(info["components"].values())
        
        if all_available:
            logger.info("API module validated successfully")
        else:
            missing = [k for k, v in info["components"].items() if not v]
            logger.warning(f"Some API components unavailable: {missing}")
        
        return all_available
        
    except Exception as e:
        logger.error(f"API module validation failed: {e}")
        return False


# Log module initialization
if _imports_successful:
    logger.info(f"VULCAN-AGI API module v{__version__} loaded successfully")
else:
    logger.warning(f"VULCAN-AGI API module v{__version__} loaded with some components unavailable")
