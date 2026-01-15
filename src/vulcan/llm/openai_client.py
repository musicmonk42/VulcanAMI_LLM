# ============================================================
# VULCAN-AGI OpenAI Client Module
# OpenAI API client management with lazy initialization
# ============================================================
#
# This module provides:
#     - Lazy initialization of OpenAI client
#     - Error tracking for initialization failures
#     - Environment variable configuration
#     - P2 FIX: Robust retry logic with exponential backoff for API calls
#
# USAGE:
#     from vulcan.llm.openai_client import get_openai_client, OPENAI_AVAILABLE
#     
#     client = get_openai_client()
#     if client:
#         response = client.chat.completions.create(...)
#
# P2 FIX USAGE (with retry wrapper):
#     from vulcan.llm.openai_client import call_openai_with_retry
#     
#     response = call_openai_with_retry(
#         lambda client: client.chat.completions.create(...),
#         max_retries=3
#     )
#
# CONFIGURATION:
#     Environment Variables:
#     - OPENAI_API_KEY: API key for authentication
#     - OPENAI_MAX_RETRIES: Maximum retry attempts (default: 3)
#     - OPENAI_BASE_DELAY: Base delay in seconds for backoff (default: 1.0)
#     - OPENAI_MAX_DELAY: Maximum delay in seconds (default: 60.0)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.1.0 - P2 FIX: Added robust retry logic with exponential backoff
# ============================================================

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Module metadata
__version__ = "1.1.0"  # P2 FIX: Added retry logic with exponential backoff
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# ============================================================
# P2 FIX: RETRY CONFIGURATION
# ============================================================

# Default retry configuration (can be overridden via environment variables)
DEFAULT_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
DEFAULT_BASE_DELAY = float(os.environ.get("OPENAI_BASE_DELAY", "1.0"))
DEFAULT_MAX_DELAY = float(os.environ.get("OPENAI_MAX_DELAY", "60.0"))

# P2 FIX: Track retry statistics globally
_retry_stats: Dict[str, int] = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "total_retries": 0,
    "rate_limit_errors": 0,
    "other_errors": 0,
}


@dataclass
class RetryStats:
    """
    Statistics for a single API call attempt.
    
    P2 FIX: Provides detailed information about retry behavior for
    monitoring and debugging.
    """
    success: bool = False
    attempts: int = 0
    total_delay_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    final_error: Optional[str] = None
    rate_limit_encountered: bool = False


def get_retry_stats() -> Dict[str, int]:
    """
    Get global retry statistics.
    
    P2 FIX: Exposes retry metrics for monitoring and alerting.
    
    Returns:
        Dictionary with retry statistics
    """
    return _retry_stats.copy()


def reset_retry_stats() -> None:
    """Reset global retry statistics."""
    global _retry_stats
    _retry_stats = {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "total_retries": 0,
        "rate_limit_errors": 0,
        "other_errors": 0,
    }


# ============================================================
# OPENAI AVAILABILITY CHECK
# ============================================================

OPENAI_AVAILABLE = False
OpenAI = None
RateLimitError = None
APIError = None
APIConnectionError = None
APITimeoutError = None

try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
    OPENAI_AVAILABLE = True
    logger.debug("OpenAI package available")
    
    # P2 FIX: Import error types for retry logic
    try:
        from openai import RateLimitError as _RateLimitError
        from openai import APIError as _APIError
        from openai import APIConnectionError as _APIConnectionError
        from openai import APITimeoutError as _APITimeoutError
        RateLimitError = _RateLimitError
        APIError = _APIError
        APIConnectionError = _APIConnectionError
        APITimeoutError = _APITimeoutError
        logger.debug("OpenAI error types imported for retry handling")
    except ImportError:
        logger.debug("OpenAI error types not available (older package version)")
        
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
# P2 FIX: RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================

T = TypeVar('T')


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    P2 FIX: Identifies transient errors that may succeed on retry.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    error_name = type(error).__name__
    error_str = str(error).lower()
    
    # Check for known retryable error types
    if RateLimitError and isinstance(error, RateLimitError):
        return True
    if APIConnectionError and isinstance(error, APIConnectionError):
        return True
    if APITimeoutError and isinstance(error, APITimeoutError):
        return True
    
    # Check error message for rate limit indicators
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "quota exceeded",
        "capacity",
    ]
    for indicator in rate_limit_indicators:
        if indicator in error_str:
            return True
    
    # Check for transient network errors
    transient_indicators = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "503",
        "502",
        "504",
    ]
    for indicator in transient_indicators:
        if indicator in error_str:
            return True
    
    return False


def _calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> float:
    """
    Calculate exponential backoff delay with jitter.
    
    P2 FIX: Implements industry-standard exponential backoff with jitter
    to prevent thundering herd problems.
    
    Formula: min(max_delay, base_delay * 2^attempt) + random_jitter
    
    Args:
        attempt: The current retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        
    Returns:
        Calculated delay in seconds
    """
    # Exponential backoff
    delay = min(max_delay, base_delay * (2 ** attempt))
    
    # Add jitter (10-30% of delay) to prevent thundering herd
    jitter = delay * random.uniform(0.1, 0.3)
    
    return delay + jitter


def call_openai_with_retry(
    api_call: Callable[[Any], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    client: Optional[Any] = None,
) -> Optional[T]:
    """
    Execute an OpenAI API call with automatic retry on transient errors.
    
    P2 FIX: This function provides robust retry handling for OpenAI API calls,
    implementing exponential backoff with jitter for rate limit errors.
    
    Args:
        api_call: A callable that takes an OpenAI client and returns a result.
                  Example: lambda client: client.chat.completions.create(...)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        client: Optional OpenAI client (uses get_openai_client() if not provided)
        
    Returns:
        The result of the API call, or None if all retries fail
        
    Raises:
        Exception: Re-raises non-retryable errors immediately
        
    Example:
        >>> result = call_openai_with_retry(
        ...     lambda client: client.chat.completions.create(
        ...         model="gpt-3.5-turbo",
        ...         messages=[{"role": "user", "content": "Hello"}]
        ...     ),
        ...     max_retries=3
        ... )
    """
    global _retry_stats
    
    _retry_stats["total_calls"] += 1
    
    # Get client
    openai_client = client or get_openai_client()
    if not openai_client:
        logger.error("[P2 FIX] OpenAI client not available for retry call")
        _retry_stats["failed_calls"] += 1
        return None
    
    last_error: Optional[Exception] = None
    total_delay = 0.0
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            result = api_call(openai_client)
            _retry_stats["successful_calls"] += 1
            
            if attempt > 0:
                logger.info(
                    f"[P2 FIX] OpenAI call succeeded on attempt {attempt + 1} "
                    f"after {total_delay:.2f}s total delay"
                )
            
            return result
            
        except Exception as e:
            last_error = e
            error_name = type(e).__name__
            
            # Check if this is a rate limit error
            is_rate_limit = (
                RateLimitError and isinstance(e, RateLimitError)
            ) or "rate" in str(e).lower()
            
            if is_rate_limit:
                _retry_stats["rate_limit_errors"] += 1
            else:
                _retry_stats["other_errors"] += 1
            
            # Check if error is retryable
            if not _is_retryable_error(e):
                logger.error(
                    f"[P2 FIX] Non-retryable OpenAI error: {error_name}: {e}"
                )
                _retry_stats["failed_calls"] += 1
                raise  # Re-raise non-retryable errors
            
            # Check if we have retries left
            if attempt >= max_retries:
                logger.error(
                    f"[P2 FIX] OpenAI call failed after {max_retries + 1} attempts. "
                    f"Last error: {error_name}: {e}"
                )
                _retry_stats["failed_calls"] += 1
                break
            
            # Calculate backoff delay
            delay = _calculate_backoff_delay(attempt, base_delay, max_delay)
            total_delay += delay
            _retry_stats["total_retries"] += 1
            
            logger.warning(
                f"[P2 FIX] OpenAI error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{error_name}: {e}. Retrying in {delay:.2f}s..."
            )
            
            time.sleep(delay)
    
    # All retries exhausted
    return None


async def call_openai_with_retry_async(
    api_call: Callable[[Any], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    client: Optional[Any] = None,
) -> Optional[T]:
    """
    Async version of call_openai_with_retry.
    
    P2 FIX: Provides async retry handling for use in async contexts.
    Uses asyncio.sleep instead of time.sleep to avoid blocking.
    
    Args:
        api_call: A callable that takes an OpenAI client and returns a result
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay cap in seconds
        client: Optional OpenAI client
        
    Returns:
        The result of the API call, or None if all retries fail
    """
    import asyncio
    
    global _retry_stats
    
    _retry_stats["total_calls"] += 1
    
    # Get client
    openai_client = client or get_openai_client()
    if not openai_client:
        logger.error("[P2 FIX] OpenAI client not available for async retry call")
        _retry_stats["failed_calls"] += 1
        return None
    
    last_error: Optional[Exception] = None
    total_delay = 0.0
    
    for attempt in range(max_retries + 1):
        try:
            result = api_call(openai_client)
            _retry_stats["successful_calls"] += 1
            
            if attempt > 0:
                logger.info(
                    f"[P2 FIX] Async OpenAI call succeeded on attempt {attempt + 1} "
                    f"after {total_delay:.2f}s total delay"
                )
            
            return result
            
        except Exception as e:
            last_error = e
            error_name = type(e).__name__
            
            is_rate_limit = (
                RateLimitError and isinstance(e, RateLimitError)
            ) or "rate" in str(e).lower()
            
            if is_rate_limit:
                _retry_stats["rate_limit_errors"] += 1
            else:
                _retry_stats["other_errors"] += 1
            
            if not _is_retryable_error(e):
                logger.error(
                    f"[P2 FIX] Non-retryable OpenAI error (async): {error_name}: {e}"
                )
                _retry_stats["failed_calls"] += 1
                raise
            
            if attempt >= max_retries:
                logger.error(
                    f"[P2 FIX] Async OpenAI call failed after {max_retries + 1} attempts. "
                    f"Last error: {error_name}: {e}"
                )
                _retry_stats["failed_calls"] += 1
                break
            
            delay = _calculate_backoff_delay(attempt, base_delay, max_delay)
            total_delay += delay
            _retry_stats["total_retries"] += 1
            
            logger.warning(
                f"[P2 FIX] Async OpenAI error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{error_name}: {e}. Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    return None


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
    
    P2 FIX: Now includes retry statistics in the output.
    
    Returns:
        Dictionary with diagnostic information:
        - available: Whether OpenAI package is installed
        - skip_openai: Whether SKIP_OPENAI env var is set to true
        - api_key_set: Whether OPENAI_API_KEY env var is set (non-empty)
        - client_ready: Whether OpenAI client is initialized and ready
        - initialization_error: Any error that occurred during initialization
        - status: Overall status ("READY", "DISABLED", "ERROR", "NOT_CONFIGURED")
        - message: Human-readable status message
        - retry_stats: P2 FIX - Retry statistics
        
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
        "retry_stats": get_retry_stats(),  # P2 FIX
        "retry_config": {  # P2 FIX
            "max_retries": DEFAULT_MAX_RETRIES,
            "base_delay": DEFAULT_BASE_DELAY,
            "max_delay": DEFAULT_MAX_DELAY,
        },
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
    
    # P2 FIX: Log retry configuration
    logger.info(
        f"[P2 FIX] Retry config: max_retries={DEFAULT_MAX_RETRIES}, "
        f"base_delay={DEFAULT_BASE_DELAY}s, max_delay={DEFAULT_MAX_DELAY}s"
    )


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
    # P2 FIX: Retry utilities
    "call_openai_with_retry",
    "call_openai_with_retry_async",
    "get_retry_stats",
    "reset_retry_stats",
    "RetryStats",
    # P2 FIX: Configuration constants
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
]


# Log module status
logger.debug(
    f"OpenAI client module v{__version__} loaded "
    f"(available: {OPENAI_AVAILABLE}, skip: {SKIP_OPENAI}, "
    f"retry_enabled: True, max_retries: {DEFAULT_MAX_RETRIES})"
)
