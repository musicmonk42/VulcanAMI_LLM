# ============================================================
# VULCAN-AGI Arena Package
# Graphix Arena API client for agent coordination
# ============================================================
#
# Arena is the FastAPI-based coordination surface for:
#     - Agent training via tournaments
#     - Graph language runtime (Graphix IR graphs)
#     - Language evolution with pattern registry
#     - Feedback integration (RLHF)
#
# USAGE:
#     from vulcan.arena import execute_via_arena, submit_arena_feedback
#     
#     result = await execute_via_arena(query, routing_plan)
#     await submit_arena_feedback(proposal_id, score=0.8, rationale="Good response")
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
from typing import Any, Dict, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CORE IMPORTS
# ============================================================

try:
    from vulcan.arena.client import (
        execute_via_arena,
        submit_arena_feedback,
        select_arena_agent,
        build_arena_payload,
        AIOHTTP_AVAILABLE,
        GENERATOR_TIMEOUT,
        SIMPLE_TASK_TIMEOUT,
        ArenaCircuitBreaker,
        DistributedCircuitBreaker,
        FeedbackRetryQueue,
        get_circuit_breaker_stats,
        reset_circuit_breaker,
    )
    _imports_successful = True
except ImportError as e:
    logger.warning(f"Arena client module not available: {e}")
    _imports_successful = False
    execute_via_arena = None
    submit_arena_feedback = None
    select_arena_agent = None
    build_arena_payload = None
    AIOHTTP_AVAILABLE = False
    GENERATOR_TIMEOUT = 90.0
    SIMPLE_TASK_TIMEOUT = 30.0
    ArenaCircuitBreaker = None
    DistributedCircuitBreaker = None
    FeedbackRetryQueue = None
    get_circuit_breaker_stats = None
    reset_circuit_breaker = None

try:
    from vulcan.arena.http_session import (
        get_http_session,
        close_http_session,
        HTTP_POOL_LIMIT,
        HTTP_POOL_LIMIT_PER_HOST,
    )
except ImportError as e:
    logger.warning(f"HTTP session module not available: {e}")
    _imports_successful = False
    get_http_session = None
    close_http_session = None
    HTTP_POOL_LIMIT = 100
    HTTP_POOL_LIMIT_PER_HOST = 30


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Arena client
    "execute_via_arena",
    "submit_arena_feedback",
    "select_arena_agent",
    "build_arena_payload",
    "AIOHTTP_AVAILABLE",
    # Timeouts and constants
    "GENERATOR_TIMEOUT",
    "SIMPLE_TASK_TIMEOUT",
    # Resilience classes
    "ArenaCircuitBreaker",
    "DistributedCircuitBreaker",
    "FeedbackRetryQueue",
    "get_circuit_breaker_stats",
    "reset_circuit_breaker",
    # HTTP session
    "get_http_session",
    "close_http_session",
    "HTTP_POOL_LIMIT",
    "HTTP_POOL_LIMIT_PER_HOST",
    # Module utilities
    "get_module_info",
    "validate_arena_module",
]


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the arena module.
    
    Returns:
        Dictionary containing:
        - version: Module version
        - author: Module author
        - imports_successful: Whether all imports succeeded
        - aiohttp_available: Whether aiohttp is available for HTTP calls
        - components: Status of each component
    """
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "aiohttp_available": AIOHTTP_AVAILABLE if AIOHTTP_AVAILABLE is not None else False,
        "components": {
            "execute_via_arena": execute_via_arena is not None,
            "submit_arena_feedback": submit_arena_feedback is not None,
            "get_http_session": get_http_session is not None,
            "close_http_session": close_http_session is not None,
        },
    }


def validate_arena_module() -> bool:
    """
    Validate that the arena module is properly loaded and functional.
    
    Returns:
        True if module is functional, False otherwise
    """
    try:
        info = get_module_info()
        
        # Core functions must be available
        core_available = all(info["components"].values())
        
        if core_available:
            logger.info("Arena module validated successfully")
        else:
            missing = [k for k, v in info["components"].items() if not v]
            logger.warning(f"Some arena components unavailable: {missing}")
        
        return core_available
        
    except Exception as e:
        logger.error(f"Arena module validation failed: {e}")
        return False


# Log module initialization
if _imports_successful:
    logger.info(f"VULCAN-AGI Arena module v{__version__} loaded successfully")
else:
    logger.warning(f"VULCAN-AGI Arena module v{__version__} loaded with some components unavailable")
