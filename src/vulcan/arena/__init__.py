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
    # HTTP session
    "get_http_session",
    "close_http_session",
    "HTTP_POOL_LIMIT",
    "HTTP_POOL_LIMIT_PER_HOST",
    # Module utilities
    "get_module_info",
]


def get_module_info() -> Dict[str, Any]:
    """Get information about the arena module."""
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "aiohttp_available": AIOHTTP_AVAILABLE if AIOHTTP_AVAILABLE is not None else False,
    }


# Log module initialization
if _imports_successful:
    logger.info(f"VULCAN-AGI Arena module v{__version__} loaded successfully")
else:
    logger.warning(f"VULCAN-AGI Arena module v{__version__} loaded with some components unavailable")
