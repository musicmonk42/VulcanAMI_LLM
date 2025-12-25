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
    
    if _openai_initialized:
        return _openai_client
    
    # Attempt initialization
    _openai_initialized = True
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _openai_init_error = "OPENAI_API_KEY environment variable not set"
        logger.warning("OPENAI_API_KEY not set - OpenAI integration disabled")
        return None
    
    try:
        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return _openai_client
    except Exception as e:
        _openai_init_error = str(e)
        logger.error(f"Failed to initialize OpenAI client: {e}")
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
    "OPENAI_AVAILABLE",
]


# Log module status
logger.debug(f"OpenAI client module v{__version__} loaded (available: {OPENAI_AVAILABLE})")
