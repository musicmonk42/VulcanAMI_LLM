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
        StepRequest,
        StepResponse,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        StatusResponse,
        ConfigResponse,
        ImprovementApproval,
    )
except ImportError as e:
    logger.warning(f"API models not available: {e}")
    _imports_successful = False
    StepRequest = None
    StepResponse = None
    ChatMessage = None
    ChatRequest = None
    ChatResponse = None
    StatusResponse = None
    ConfigResponse = None
    ImprovementApproval = None

try:
    from vulcan.api.rate_limiting import (
        rate_limit_storage,
        rate_limit_lock,
        rate_limit_cleanup_thread,
        cleanup_rate_limits,
        start_rate_limit_cleanup,
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
    check_rate_limit = None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Models
    "StepRequest",
    "StepResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "StatusResponse",
    "ConfigResponse",
    "ImprovementApproval",
    # Rate limiting
    "rate_limit_storage",
    "rate_limit_lock",
    "rate_limit_cleanup_thread",
    "cleanup_rate_limits",
    "start_rate_limit_cleanup",
    "check_rate_limit",
]


# Log module initialization
if _imports_successful:
    logger.debug(f"API module v{__version__} loaded successfully")
else:
    logger.debug(f"API module v{__version__} loaded with some components unavailable")
