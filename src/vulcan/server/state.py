"""
Vulcan Server Shared State Module

This module provides centralized storage for module-level globals shared across
server components, specifically designed to avoid circular import issues between
app.py and manager.py.

Architecture:
    This module follows the "Shared Kernel" pattern from Domain-Driven Design,
    providing a minimal shared context that prevents circular dependencies while
    maintaining clean separation of concerns. Both app.py and manager.py can
    safely import this lightweight module without creating import cycles.

Thread Safety:
    All state variables are designed to be set once during initialization or
    accessed through thread-safe mechanisms (locks are provided where needed).
    The module itself is thread-safe for read operations after initialization.
    Write operations should use the provided locks where applicable.

Security Considerations:
    - No secrets or credentials are stored in this module
    - Redis client connection is established with proper authentication
    - Process lock prevents unauthorized concurrent instances

Performance:
    - Minimal overhead: only stores references, no computations
    - Lock contention is minimal (only during thread startup)
    - No global state mutations during normal operation

Module Attributes:
    process_lock (Optional[ProcessLock]): 
        Process-level lock for split-brain prevention when Redis is unavailable.
        Ensures only one orchestrator instance runs to prevent conflicting state
        modifications. Set during lifespan startup if Redis is not available.
        
    rate_limit_cleanup_thread (Optional[Thread]): 
        Background daemon thread for rate limit cleanup. Periodically removes
        expired rate limit entries. Should be started once during initialization
        and checked for liveness before attempting to restart. Access must be
        protected by rate_limit_thread_lock.
        
    rate_limit_thread_lock (Lock): 
        Reentrant lock for thread-safe access to rate_limit_cleanup_thread.
        Prevents race conditions when multiple workers or initialization paths
        attempt to start or check the cleanup thread simultaneously. Must be
        acquired before reading or writing rate_limit_cleanup_thread.
        
    redis_client (Optional[redis.Redis]): 
        Redis client instance for distributed state synchronization. May be None
        if Redis is not configured, connection failed during initialization, or
        running in standalone/development mode. The client itself is thread-safe.

Usage Example:
    ```python
    # In app.py - initialize Redis client
    from vulcan.server import state
    import redis
    
    try:
        state.redis_client = redis.Redis.from_url(settings.redis_url)
        state.redis_client.ping()
    except Exception as e:
        state.redis_client = None
        
    # In manager.py - start rate limit cleanup thread
    from vulcan.server import state
    
    with state.rate_limit_thread_lock:
        if (state.rate_limit_cleanup_thread is None or 
            not state.rate_limit_cleanup_thread.is_alive()):
            thread = Thread(target=cleanup_rate_limits, daemon=True)
            thread.start()
            state.rate_limit_cleanup_thread = thread
    ```

See Also:
    - vulcan.server.app: Main application lifespan management
    - vulcan.server.startup.manager: Startup phase orchestration
    - vulcan.utils_main.process_lock: ProcessLock implementation

Notes:
    - This module must remain dependency-free except for standard library
    - Do not add business logic to this module
    - Keep module size minimal to avoid import overhead
    
Version: 2.0.0
Author: VULCAN-AGI Team
License: MIT
"""

from typing import Optional, Any
from threading import Thread, Lock

__all__ = [
    "process_lock",
    "rate_limit_cleanup_thread", 
    "rate_limit_thread_lock",
    "redis_client",
]

# ============================================================
# Split-Brain Prevention
# ============================================================

process_lock: Optional[Any] = None
"""
Process lock for split-brain prevention when Redis is unavailable.

When Redis is not configured, only one orchestrator instance should run
to prevent conflicting state modifications. This lock ensures mutual exclusion.

Type: Optional[ProcessLock] (using Any to avoid import cycles)
"""

# ============================================================
# Background Thread Management
# ============================================================

rate_limit_cleanup_thread: Optional[Thread] = None
"""
Background daemon thread for rate limit cleanup.

This thread periodically cleans up expired rate limit entries. It should
be started once during initialization and checked for liveness before
attempting to restart.

Type: Optional[Thread]
Thread-Safety: Access should be protected by rate_limit_thread_lock
"""

rate_limit_thread_lock: Lock = Lock()
"""
Lock for thread-safe access to rate_limit_cleanup_thread.

Prevents race conditions when multiple workers or initialization paths
attempt to start or check the cleanup thread simultaneously.

Type: threading.Lock
Usage: 
    with rate_limit_thread_lock:
        if rate_limit_cleanup_thread is None or not rate_limit_cleanup_thread.is_alive():
            rate_limit_cleanup_thread = Thread(...)
"""

# ============================================================
# Redis Connection
# ============================================================

redis_client: Optional[Any] = None
"""
Redis client instance for distributed state synchronization.

May be None if:
- Redis is not configured (redis_url not provided in settings)
- Redis connection failed during initialization
- Running in standalone/development mode

Type: Optional[redis.Redis] (using Any to avoid import dependency)
Thread-Safety: Redis client itself is thread-safe for operations
"""
