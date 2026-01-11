# ============================================================
# VULCAN-AGI Arena HTTP Session Module
# HTTP connection pool management for Arena API calls
# ============================================================
#
# HTTP CONNECTION POOL FIX: Global session for connection reuse
# Creating a new ClientSession per request causes overhead and
# connection exhaustion. The session is initialized in lifespan
# context and used by all HTTP operations.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added atexit cleanup registration for graceful shutdown
# ============================================================

import asyncio
import atexit
import logging
import os
from typing import Any, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# AIOHTTP AVAILABILITY CHECK
# ============================================================

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False
    logger.info("aiohttp not available - HTTP session functionality disabled")

# ============================================================
# CONFIGURATION
# ============================================================

# HTTP CONNECTION POOL FIX: Constants for connection pool configuration
# These can be overridden via environment variables for different deployment scenarios
HTTP_POOL_LIMIT = int(os.environ.get("VULCAN_HTTP_POOL_LIMIT", "100"))
HTTP_POOL_LIMIT_PER_HOST = int(os.environ.get("VULCAN_HTTP_POOL_LIMIT_PER_HOST", "30"))
HTTP_DNS_CACHE_TTL = int(os.environ.get("VULCAN_HTTP_DNS_CACHE_TTL", "300"))
HTTP_TOTAL_TIMEOUT = float(os.environ.get("VULCAN_HTTP_TOTAL_TIMEOUT", "60.0"))
HTTP_CONNECT_TIMEOUT = float(os.environ.get("VULCAN_HTTP_CONNECT_TIMEOUT", "10.0"))
HTTP_READ_TIMEOUT = float(os.environ.get("VULCAN_HTTP_READ_TIMEOUT", "30.0"))

# ============================================================
# GLOBAL SESSION
# ============================================================

# HTTP CONNECTION POOL FIX: Global aiohttp session for connection reuse
# Creating a new ClientSession per request causes overhead and connection exhaustion
# The session is initialized in lifespan context and used by all HTTP operations
_http_session: Optional[Any] = None  # Type as Any to handle aiohttp not being installed


async def get_http_session():
    """
    Get the global HTTP session for connection pooling.
    
    HTTP CONNECTION POOL FIX: This ensures HTTP connections are reused
    rather than creating new connections for each request, which was
    causing latency and potential connection exhaustion.
    
    Returns:
        aiohttp.ClientSession with connection pooling
        
    Raises:
        RuntimeError: If aiohttp is not available
    """
    global _http_session
    if not AIOHTTP_AVAILABLE or aiohttp is None:
        raise RuntimeError("aiohttp not available - cannot create HTTP session")
    
    if _http_session is None or _http_session.closed:
        # Create session with configurable connection pool limits
        connector = aiohttp.TCPConnector(
            limit=HTTP_POOL_LIMIT,
            limit_per_host=HTTP_POOL_LIMIT_PER_HOST,
            ttl_dns_cache=HTTP_DNS_CACHE_TTL,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=HTTP_TOTAL_TIMEOUT,
            connect=HTTP_CONNECT_TIMEOUT,
            sock_read=HTTP_READ_TIMEOUT,
        )
        _http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )
        logger.info(
            f"HTTP connection pool initialized (limit={HTTP_POOL_LIMIT}, "
            f"per_host={HTTP_POOL_LIMIT_PER_HOST})"
        )
    return _http_session


async def close_http_session():
    """Close the global HTTP session gracefully."""
    global _http_session
    if _http_session is not None and not _http_session.closed:
        await _http_session.close()
        _http_session = None
        logger.info("HTTP connection pool closed")


def is_session_active() -> bool:
    """Check if HTTP session is active."""
    return _http_session is not None and not _http_session.closed


def get_pool_config() -> dict:
    """Get current pool configuration."""
    return {
        "pool_limit": HTTP_POOL_LIMIT,
        "pool_limit_per_host": HTTP_POOL_LIMIT_PER_HOST,
        "dns_cache_ttl": HTTP_DNS_CACHE_TTL,
        "total_timeout": HTTP_TOTAL_TIMEOUT,
        "connect_timeout": HTTP_CONNECT_TIMEOUT,
        "read_timeout": HTTP_READ_TIMEOUT,
        "session_active": is_session_active(),
    }


def _sync_cleanup():
    """
    Synchronous cleanup wrapper for atexit.
    
    Ensures HTTP session is properly closed on application shutdown.
    Handles cases where event loop may or may not be running.
    """
    global _http_session
    if _http_session is not None and not _http_session.closed:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule cleanup as a task
                loop.create_task(close_http_session())
            else:
                # If loop is not running, run cleanup synchronously
                loop.run_until_complete(close_http_session())
        except RuntimeError:
            # No event loop available - skip cleanup gracefully
            pass


# Register cleanup handler to run on application exit
atexit.register(_sync_cleanup)


__all__ = [
    "get_http_session",
    "close_http_session",
    "is_session_active",
    "get_pool_config",
    "AIOHTTP_AVAILABLE",
    "HTTP_POOL_LIMIT",
    "HTTP_POOL_LIMIT_PER_HOST",
    "HTTP_DNS_CACHE_TTL",
    "HTTP_TOTAL_TIMEOUT",
    "HTTP_CONNECT_TIMEOUT",
    "HTTP_READ_TIMEOUT",
]
