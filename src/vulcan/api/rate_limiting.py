# ============================================================
# VULCAN-AGI Rate Limiting Module
# Thread-safe storage and utilities for API rate limiting
# ============================================================
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     2.0.0 - Added Redis-based rate limiting for distributed deployments
# ============================================================

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Module metadata
__version__ = "2.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# IN-MEMORY RATE LIMITING (Single Instance)
# ============================================================

# Thread-safe storage for simple rate limiting
rate_limit_storage: Dict[str, List[float]] = {}
rate_limit_lock = threading.RLock()
rate_limit_cleanup_thread: Optional[threading.Thread] = None
rate_limit_cleanup_stop_event = threading.Event()  # Stop signal for graceful shutdown


# ============================================================
# RATE LIMITING FUNCTIONS
# ============================================================


def cleanup_rate_limits(
    cleanup_interval: int = 300,
    window_seconds: int = 60
) -> None:
    """
    Periodically cleanup old rate limit entries (in-memory only).
    
    This function runs in a background thread to prevent memory
    growth from accumulating rate limit timestamps.
    
    Args:
        cleanup_interval: Seconds between cleanup runs
        window_seconds: Rate limit window size in seconds
    """
    while not rate_limit_cleanup_stop_event.is_set():
        try:
            # Use wait() instead of sleep() for immediate response to stop signal
            if rate_limit_cleanup_stop_event.wait(timeout=cleanup_interval):
                # Stop event was set, exit gracefully
                logger.info("Rate limit cleanup thread stopping gracefully")
                break
                
            current_time = time.time()
            window_start = current_time - window_seconds

            with rate_limit_lock:
                for client_id in list(rate_limit_storage.keys()):
                    rate_limit_storage[client_id] = [
                        t for t in rate_limit_storage[client_id] if t > window_start
                    ]
                    if not rate_limit_storage[client_id]:
                        del rate_limit_storage[client_id]

            logger.debug("Rate limit storage cleaned up")
        except Exception as e:
            logger.error(f"Rate limit cleanup error: {e}")


def start_rate_limit_cleanup(
    cleanup_interval: int = 300,
    window_seconds: int = 60
) -> threading.Thread:
    """
    Start the rate limit cleanup background thread.
    
    Args:
        cleanup_interval: Seconds between cleanup runs
        window_seconds: Rate limit window size in seconds
        
    Returns:
        The cleanup thread
    """
    global rate_limit_cleanup_thread
    
    if rate_limit_cleanup_thread is not None and rate_limit_cleanup_thread.is_alive():
        logger.debug("Rate limit cleanup thread already running")
        return rate_limit_cleanup_thread
    
    # Clear stop event in case it was set previously
    rate_limit_cleanup_stop_event.clear()
    
    rate_limit_cleanup_thread = threading.Thread(
        target=cleanup_rate_limits,
        args=(cleanup_interval, window_seconds),
        daemon=True,
        name="RateLimitCleanup"
    )
    rate_limit_cleanup_thread.start()
    logger.info("Rate limit cleanup thread started")
    
    return rate_limit_cleanup_thread


def stop_rate_limit_cleanup(timeout: float = 5.0) -> bool:
    """
    Stop the rate limit cleanup background thread gracefully.
    
    Args:
        timeout: Maximum seconds to wait for thread to stop
        
    Returns:
        True if thread stopped successfully, False otherwise
    """
    global rate_limit_cleanup_thread
    
    if rate_limit_cleanup_thread is None or not rate_limit_cleanup_thread.is_alive():
        logger.debug("Rate limit cleanup thread not running")
        return True
    
    # Signal the thread to stop
    rate_limit_cleanup_stop_event.set()
    
    # Wait for thread to finish
    rate_limit_cleanup_thread.join(timeout=timeout)
    
    if rate_limit_cleanup_thread.is_alive():
        logger.warning(f"Rate limit cleanup thread did not stop within {timeout}s")
        return False
    
    logger.info("Rate limit cleanup thread stopped successfully")
    rate_limit_cleanup_thread = None
    return True


def check_rate_limit(
    client_id: str,
    max_requests: int,
    window_seconds: int
) -> Tuple[bool, int]:
    """
    Check rate limit using in-memory storage.
    
    WARNING: This only works for single-instance deployments.
    Use check_rate_limit_redis() for multi-worker/distributed deployments.
    
    Args:
        client_id: Client identifier (IP or API key hash)
        max_requests: Maximum allowed requests in window
        window_seconds: Window size in seconds
        
    Returns:
        Tuple of (allowed: bool, remaining: int)
    """
    current_time = time.time()
    window_start = current_time - window_seconds
    
    with rate_limit_lock:
        bucket = rate_limit_storage.setdefault(client_id, [])
        # Evict old timestamps
        rate_limit_storage[client_id] = [t for t in bucket if t > window_start]
        
        current_count = len(rate_limit_storage[client_id])
        
        if current_count >= max_requests:
            return False, 0
        
        rate_limit_storage[client_id].append(current_time)
        return True, max_requests - current_count - 1


# ============================================================
# REDIS-BASED RATE LIMITING (Distributed)
# ============================================================

async def check_rate_limit_redis(
    client_id: str,
    max_requests: int,
    window_seconds: int,
    redis_client: Optional[Any] = None
) -> Tuple[bool, int]:
    """
    Check rate limit using Redis for distributed deployments.
    
    Uses Redis sorted sets with timestamps as scores for efficient
    sliding window rate limiting that works across multiple workers
    and server instances.
    
    Falls back to in-memory rate limiting if Redis is unavailable.
    
    Args:
        client_id: Client identifier (IP or API key hash)
        max_requests: Maximum allowed requests in window
        window_seconds: Window size in seconds
        redis_client: Redis client instance (async)
        
    Returns:
        Tuple of (allowed: bool, remaining: int)
        
    Example:
        >>> from vulcan.server import state
        >>> allowed, remaining = await check_rate_limit_redis(
        ...     client_id="user_abc",
        ...     max_requests=100,
        ...     window_seconds=60,
        ...     redis_client=state.redis_client
        ... )
    """
    # Fallback to in-memory if Redis unavailable
    if redis_client is None:
        logger.debug("Redis unavailable, using in-memory rate limiting")
        return check_rate_limit(client_id, max_requests, window_seconds)
    
    try:
        key = f"vulcan:rate_limit:{client_id}"
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Use pipeline for atomic operations
        pipe = redis_client.pipeline()
        
        # Remove timestamps outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Add current timestamp
        pipe.zadd(key, {str(current_time): current_time})
        
        # Count requests in window
        pipe.zcard(key)
        
        # Set expiry to auto-cleanup old keys
        pipe.expire(key, window_seconds + 10)
        
        # Execute pipeline
        results = await pipe.execute()
        
        count = results[2]  # zcard result
        
        if count > max_requests:
            # Remove the timestamp we just added (over limit)
            await redis_client.zrem(key, str(current_time))
            return False, 0
        
        remaining = max_requests - count
        return True, remaining
        
    except Exception as e:
        logger.warning(f"Redis rate limit check failed: {e}, falling back to in-memory")
        return check_rate_limit(client_id, max_requests, window_seconds)


def check_rate_limit_sync_redis(
    client_id: str,
    max_requests: int,
    window_seconds: int,
    redis_client: Optional[Any] = None
) -> Tuple[bool, int]:
    """
    Synchronous version of Redis rate limiting.
    
    For use in synchronous middleware or non-async contexts.
    """
    if redis_client is None:
        return check_rate_limit(client_id, max_requests, window_seconds)
    
    try:
        key = f"vulcan:rate_limit:{client_id}"
        current_time = time.time()
        window_start = current_time - window_seconds
        
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(current_time): current_time})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 10)
        results = pipe.execute()
        
        count = results[2]
        
        if count > max_requests:
            redis_client.zrem(key, str(current_time))
            return False, 0
        
        return True, max_requests - count
        
    except Exception as e:
        logger.warning(f"Redis rate limit failed: {e}, using in-memory")
        return check_rate_limit(client_id, max_requests, window_seconds)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def get_client_id_from_request(
    host: str,
    api_key: Optional[str] = None
) -> str:
    """
    Get a client ID for rate limiting.
    
    If an API key is provided, use its hash. Otherwise, use the host.
    
    Args:
        host: Client's host/IP address
        api_key: Optional API key
        
    Returns:
        Client identifier string
    """
    if api_key:
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return host


def clear_rate_limits() -> int:
    """
    Clear all in-memory rate limit entries (for testing).
    
    Returns:
        Number of entries cleared
    """
    with rate_limit_lock:
        count = len(rate_limit_storage)
        rate_limit_storage.clear()
        return count


async def clear_rate_limits_redis(redis_client: Any) -> int:
    """
    Clear all Redis rate limit entries.
    
    Args:
        redis_client: Redis client instance (async)
        
    Returns:
        Number of keys deleted
    """
    if redis_client is None:
        return clear_rate_limits()
    
    try:
        keys = await redis_client.keys("vulcan:rate_limit:*")
        if keys:
            await redis_client.delete(*keys)
        return len(keys)
    except Exception as e:
        logger.error(f"Failed to clear Redis rate limits: {e}")
        return 0


def get_rate_limit_stats() -> Dict[str, int]:
    """
    Get rate limiting statistics (in-memory only).
    
    Returns:
        Dictionary with statistics
    """
    with rate_limit_lock:
        total_clients = len(rate_limit_storage)
        total_entries = sum(len(v) for v in rate_limit_storage.values())
        return {
            "total_clients": total_clients,
            "total_entries": total_entries,
        }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # In-memory
    "rate_limit_storage",
    "rate_limit_lock",
    "rate_limit_cleanup_thread",
    "check_rate_limit",
    "cleanup_rate_limits",
    "start_rate_limit_cleanup",
    "stop_rate_limit_cleanup",
    "clear_rate_limits",
    # Redis-based
    "check_rate_limit_redis",
    "check_rate_limit_sync_redis",
    "clear_rate_limits_redis",
    # Utilities
    "get_client_id_from_request",
    "get_rate_limit_stats",
]
