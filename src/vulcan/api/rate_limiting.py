# ============================================================
# VULCAN-AGI Rate Limiting Module
# Thread-safe storage and utilities for API rate limiting
# ============================================================
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import hashlib
import logging
import threading
import time
from typing import Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# RATE LIMITING STORAGE
# ============================================================

# Thread-safe storage for simple rate limiting
rate_limit_storage: Dict[str, List[float]] = {}
rate_limit_lock = threading.RLock()
rate_limit_cleanup_thread: Optional[threading.Thread] = None


# ============================================================
# RATE LIMITING FUNCTIONS
# ============================================================


def cleanup_rate_limits(
    cleanup_interval: int = 300,
    window_seconds: int = 60
) -> None:
    """
    Periodically cleanup old rate limit entries.
    
    This function runs in a background thread to prevent memory
    growth from accumulating rate limit timestamps.
    
    Args:
        cleanup_interval: Seconds between cleanup runs
        window_seconds: Rate limit window size in seconds
    """
    while True:
        try:
            time.sleep(cleanup_interval)
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
    
    rate_limit_cleanup_thread = threading.Thread(
        target=cleanup_rate_limits,
        args=(cleanup_interval, window_seconds),
        daemon=True,
        name="RateLimitCleanup"
    )
    rate_limit_cleanup_thread.start()
    logger.info("Rate limit cleanup thread started")
    
    return rate_limit_cleanup_thread


def check_rate_limit(
    client_id: str,
    max_requests: int,
    window_seconds: int
) -> tuple:
    """
    Check if a client has exceeded the rate limit.
    
    Args:
        client_id: Identifier for the client (IP or API key hash)
        max_requests: Maximum allowed requests in the window
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
    Clear all rate limit entries (for testing).
    
    Returns:
        Number of entries cleared
    """
    with rate_limit_lock:
        count = len(rate_limit_storage)
        rate_limit_storage.clear()
        return count


def get_rate_limit_stats() -> Dict[str, int]:
    """
    Get rate limiting statistics.
    
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
    "rate_limit_storage",
    "rate_limit_lock",
    "rate_limit_cleanup_thread",
    "cleanup_rate_limits",
    "start_rate_limit_cleanup",
    "check_rate_limit",
    "get_client_id_from_request",
    "clear_rate_limits",
    "get_rate_limit_stats",
]
