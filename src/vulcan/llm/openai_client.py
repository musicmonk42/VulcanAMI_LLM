# ============================================================
# VULCAN-AGI OpenAI Client Module
# OpenAI API client management with lazy initialization
# ============================================================
#
# This module provides:
#     - Lazy initialization of OpenAI client
#     - Error tracking for initialization failures
#     - Environment variable configuration
#
# USAGE:
#     from vulcan.llm.openai_client import get_openai_client, OPENAI_AVAILABLE
#     
#     client = get_openai_client()
#     if client:
#         response = client.chat.completions.create(...)
#
# CONFIGURATION:
#     Environment Variable: OPENAI_API_KEY
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
import os
from typing import Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# OPENAI AVAILABILITY CHECK
# ============================================================

OPENAI_AVAILABLE = False
OpenAI = None

try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
    OPENAI_AVAILABLE = True
    logger.debug("OpenAI package available")
except ImportError:
    logger.info("OpenAI package not installed - install with: pip install openai")
    OPENAI_AVAILABLE = False


# ============================================================
# SKIP_OPENAI CONFIGURATION
# ============================================================
# The SKIP_OPENAI environment variable controls whether OpenAI fallback is enabled.
# When SKIP_OPENAI='true', OpenAI will be disabled even if OPENAI_API_KEY is set.
# When SKIP_OPENAI='false' (default), OpenAI fallback is enabled.
# This is used in CI workflows like scalability_test.yml to ensure OpenAI fallback
# is available when the internal LLM times out during stress tests.

_skip_openai_env = os.environ.get("SKIP_OPENAI", "false").lower()
SKIP_OPENAI = _skip_openai_env in ("true", "1", "yes")

if SKIP_OPENAI:
    logger.info("SKIP_OPENAI=true - OpenAI fallback is DISABLED")
else:
    logger.debug("SKIP_OPENAI=false - OpenAI fallback is ENABLED (if API key is set)")


# ============================================================
# GLOBAL CLIENT STATE
# ============================================================

# Lazy-loaded OpenAI client
_openai_client: Optional[object] = None
_openai_init_error: Optional[str] = None
_openai_initialized: bool = False


# ============================================================
# CLIENT MANAGEMENT FUNCTIONS
# ============================================================


def get_openai_client():
    """
    Get the OpenAI client instance (lazy initialization).
    
    The client is initialized on first call and cached for subsequent calls.
    If initialization fails, the error is cached and None is returned.
    
    Returns:
        OpenAI client instance if available and initialized, None otherwise
        
    Note:
        The client will return None in the following cases:
        - OpenAI package is not installed
        - SKIP_OPENAI environment variable is set to 'true'
        - OPENAI_API_KEY environment variable is not set
        - OpenAI client initialization fails
        
    Example:
        >>> client = get_openai_client()
        >>> if client:
        ...     response = client.chat.completions.create(
        ...         model="gpt-3.5-turbo",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
    """
    global _openai_client, _openai_init_error, _openai_initialized
    
    if not OPENAI_AVAILABLE:
        return None
    
    # Check SKIP_OPENAI environment variable
    # FIX: Support SKIP_OPENAI env var from workflow to control OpenAI fallback
    if SKIP_OPENAI:
        if not _openai_initialized:
            _openai_init_error = "OpenAI skipped due to SKIP_OPENAI=true environment variable"
            _openai_initialized = True
            logger.info("OpenAI client disabled by SKIP_OPENAI=true")
        return None
    
    if _openai_initialized:
        return _openai_client
    
    # Attempt initialization
    _openai_initialized = True
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _openai_init_error = "OPENAI_API_KEY environment variable not set"
        logger.warning(
            "OPENAI_API_KEY not set - OpenAI integration disabled. "
            "Set OPENAI_API_KEY in environment or repository secrets to enable OpenAI fallback."
        )
        return None
    
    # Log that we found an API key (without exposing any part of it for security)
    logger.info("OPENAI_API_KEY found, attempting initialization...")
    
    try:
        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully - fallback is AVAILABLE")
        return _openai_client
    except Exception as e:
        _openai_init_error = str(e)
        logger.error(
            f"Failed to initialize OpenAI client: {e}. "
            "This may indicate an invalid API key. "
            "Verify OPENAI_API_KEY is correct in repository secrets."
        )
        return None


def get_openai_init_error() -> Optional[str]:
    """
    Return any error from OpenAI initialization for diagnostics.
    
    Returns:
        Error message string if initialization failed, None if successful
    """
    if not OPENAI_AVAILABLE:
        return "OpenAI package not installed - install with: pip install openai"
    return _openai_init_error


def initialize_openai_client(api_key: Optional[str] = None) -> bool:
    """
    Explicitly initialize the OpenAI client.
    
    Args:
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    global _openai_client, _openai_init_error, _openai_initialized
    
    if not OPENAI_AVAILABLE:
        _openai_init_error = "OpenAI package not installed"
        return False
    
    # Reset state
    _openai_client = None
    _openai_init_error = None
    _openai_initialized = True
    
    # Get API key
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        _openai_init_error = "No API key provided and OPENAI_API_KEY not set"
        logger.warning(_openai_init_error)
        return False
    
    try:
        _openai_client = OpenAI(api_key=key)
        logger.info("OpenAI client initialized successfully")
        return True
    except Exception as e:
        _openai_init_error = str(e)
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False


def reset_openai_client() -> None:
    """
    Reset the OpenAI client state for re-initialization.
    
    Useful for testing or when credentials change.
    """
    global _openai_client, _openai_init_error, _openai_initialized
    _openai_client = None
    _openai_init_error = None
    _openai_initialized = False
    logger.debug("OpenAI client state reset")


def is_openai_initialized() -> bool:
    """
    Check if OpenAI client initialization has been attempted.
    
    Returns:
        True if initialization was attempted (regardless of success)
    """
    return _openai_initialized


def is_openai_ready() -> bool:
    """
    Check if OpenAI client is ready for use.
    
    Returns:
        True if client is initialized and ready
    """
    return _openai_initialized and _openai_client is not None


def verify_openai_configuration() -> dict:
    """
    Verify OpenAI configuration and return detailed diagnostics.
    
    This function provides comprehensive information about the OpenAI
    configuration status, useful for debugging CI failures.
    
    Returns:
        Dictionary with diagnostic information:
        - available: Whether OpenAI package is installed
        - skip_openai: Whether SKIP_OPENAI env var is set to true
        - api_key_set: Whether OPENAI_API_KEY env var is set (non-empty)
        - client_ready: Whether OpenAI client is initialized and ready
        - initialization_error: Any error that occurred during initialization
        - status: Overall status ("READY", "DISABLED", "ERROR", "NOT_CONFIGURED")
        - message: Human-readable status message
        
    Example:
        >>> status = verify_openai_configuration()
        >>> if status["status"] != "READY":
        ...     print(f"OpenAI not available: {status['message']}")
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_set = bool(api_key and api_key.strip())
    
    result = {
        "available": OPENAI_AVAILABLE,
        "skip_openai": SKIP_OPENAI,
        "api_key_set": api_key_set,
        "client_ready": is_openai_ready(),
        "initialization_error": get_openai_init_error(),
        "status": "UNKNOWN",
        "message": "",
    }
    
    # Determine status
    if not OPENAI_AVAILABLE:
        result["status"] = "ERROR"
        result["message"] = "OpenAI package not installed - run: pip install openai"
    elif SKIP_OPENAI:
        result["status"] = "DISABLED"
        result["message"] = "OpenAI is disabled (SKIP_OPENAI=true). Set SKIP_OPENAI=false to enable."
    elif not api_key_set:
        result["status"] = "NOT_CONFIGURED"
        result["message"] = (
            "OPENAI_API_KEY not set. Set the OPENAI_API_KEY repository secret "
            "or provide via workflow input to enable OpenAI fallback."
        )
    elif is_openai_ready():
        result["status"] = "READY"
        result["message"] = "OpenAI client ready and available"
    else:
        # API key is set but client failed to initialize
        result["status"] = "ERROR"
        error = get_openai_init_error() or "Unknown initialization error"
        result["message"] = (
            f"OpenAI initialization failed: {error}. "
            "The API key may be invalid or expired."
        )
    
    return result


def log_openai_status() -> None:
    """
    Log the current OpenAI configuration status.
    
    This is a convenience function that logs the output of verify_openai_configuration()
    at appropriate log levels. Call this during application startup to provide
    visibility into OpenAI availability.
    """
    status = verify_openai_configuration()
    
    if status["status"] == "READY":
        logger.info(f"✓ OpenAI fallback READY: {status['message']}")
    elif status["status"] == "DISABLED":
        logger.warning(f"⚠ OpenAI fallback DISABLED: {status['message']}")
    elif status["status"] == "NOT_CONFIGURED":
        logger.warning(f"⚠ OpenAI fallback NOT CONFIGURED: {status['message']}")
    else:  # ERROR
        logger.error(f"❌ OpenAI fallback ERROR: {status['message']}")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "get_openai_client",
    "get_openai_init_error",
    "initialize_openai_client",
    "reset_openai_client",
    "is_openai_initialized",
    "is_openai_ready",
    "verify_openai_configuration",
    "log_openai_status",
    "OPENAI_AVAILABLE",
    "SKIP_OPENAI",
]


# Log module status
logger.debug(f"OpenAI client module v{__version__} loaded (available: {OPENAI_AVAILABLE}, skip: {SKIP_OPENAI})")
