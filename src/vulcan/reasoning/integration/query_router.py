"""
Query routing and strategy determination for reasoning integration.

This module handles query type classification and routing decisions to
appropriate reasoning types based on query characteristics and fast-path routes.

Module: vulcan.reasoning.integration.query_router
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
from typing import Optional

from .types import (
    LOG_PREFIX,
    ROUTE_TO_REASONING_TYPE,
)

logger = logging.getLogger(__name__)


def get_reasoning_type_from_route(
    query_type: str, route: Optional[str] = None
) -> str:
    """
    Get the appropriate reasoning type from query route or query type.
    
    This function ensures proper reasoning type classification instead of
    returning UNKNOWN with confidence=0.1.
    
    Args:
        query_type: The query type (e.g., "reasoning", "philosophical", "mathematical")
        route: Optional route string (e.g., "PHILOSOPHICAL-FAST-PATH")
        
    Returns:
        Reasoning type string (e.g., "symbolic", "causal", "mathematical")
    
    Example:
        >>> get_reasoning_type_from_route("philosophical", "PHILOSOPHICAL-FAST-PATH")
        'world_model'
        >>> get_reasoning_type_from_route("reasoning", None)
        'causal'
    """
    # Try route first (more specific)
    if route:
        route_upper = route.upper()
        if route_upper in ROUTE_TO_REASONING_TYPE:
            return ROUTE_TO_REASONING_TYPE[route_upper]
    
    # Try query_type (case-insensitive)
    if query_type:
        query_type_lower = query_type.lower()
        if query_type_lower in ROUTE_TO_REASONING_TYPE:
            return ROUTE_TO_REASONING_TYPE[query_type_lower]
    
    # Default to hybrid for unknown types
    logger.debug(
        f"{LOG_PREFIX} Unknown query_type='{query_type}' route='{route}', "
        f"defaulting to 'hybrid' reasoning type"
    )
    return "hybrid"


__all__ = [
    "get_reasoning_type_from_route",
]
