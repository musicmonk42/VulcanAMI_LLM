"""
Response formatting and sanitization for reasoning integration.

Handles context sanitization, Arena delegation, and response utilities.
"""

import dataclasses
import json
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from .types import LOG_PREFIX

logger = logging.getLogger(__name__)

# Arena configuration
ARENA_REASONING_URL = os.environ.get(
    "ARENA_REASONING_URL",
    "http://localhost:8001/api/v1/reason"
)
ARENA_DELEGATION_TIMEOUT = float(os.environ.get("ARENA_DELEGATION_TIMEOUT", "30.0"))


def sanitize_context_for_json(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize context dictionary to make it JSON serializable.
    
    The context may contain objects like PreprocessingResult that have
    to_dict() methods. This function recursively converts such objects
    to plain dictionaries.
    
    Args:
        context: Original context dictionary
        
    Returns:
        Sanitized context dictionary that is JSON serializable
    """
    if not context:
        return {}
    
    def sanitize_value(value: Any) -> Any:
        """Recursively sanitize a value for JSON serialization."""
        # Handle None
        if value is None:
            return None
        
        # Handle primitives
        if isinstance(value, (bool, int, float, str)):
            return value
        
        # Handle objects with to_dict() method (e.g., PreprocessingResult)
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            try:
                return value.to_dict()
            except Exception as e:
                logger.warning(
                    f"{LOG_PREFIX} Failed to serialize object with to_dict(): {e}"
                )
                return str(value)
        
        # Handle dataclasses with __dataclass_fields__
        if hasattr(value, '__dataclass_fields__'):
            try:
                return dataclasses.asdict(value)
            except Exception as e:
                logger.warning(
                    f"{LOG_PREFIX} Failed to serialize dataclass: {e}"
                )
                return str(value)
        
        # Handle dictionaries recursively
        if isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        
        # Handle lists and tuples recursively
        if isinstance(value, (list, tuple)):
            return [sanitize_value(item) for item in value]
        
        # Handle sets (convert to list)
        if isinstance(value, (set, frozenset)):
            return [sanitize_value(item) for item in value]
        
        # Handle Enum
        if isinstance(value, Enum):
            return value.value
        
        # Fallback: convert to string
        try:
            return str(value)
        except Exception:
            return repr(value)
    
    return sanitize_value(context)


def delegate_to_arena(
    query: str,
    original_tool: str,
    query_type: str,
    complexity: float,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Delegate reasoning task to Arena when local tools fail.
    
    Arena provides a full reasoning pipeline with evolution/tournaments
    that may succeed where individual tools fail. This is the final
    fallback before returning a low-confidence result.
    
    Args:
        query: The original query text
        original_tool: The tool that initially failed
        query_type: Type of query (reasoning, symbolic, etc.)
        complexity: Query complexity score (0.0 to 1.0)
        context: Optional context dictionary
        
    Returns:
        Dictionary with Arena result if successful, None otherwise
        
    Note:
        This method uses httpx for synchronous HTTP requests to Arena.
        It will not block the event loop in async contexts.
    """
    try:
        import httpx
        
        logger.info(
            f"{LOG_PREFIX} Delegating to Arena: tool={original_tool}, "
            f"query_type={query_type}, complexity={complexity:.2f}"
        )
        
        # Build request payload
        sanitized_context = sanitize_context_for_json(context or {})
        
        arena_payload = {
            "query": query,
            "selected_tools": [original_tool],
            "query_type": query_type,
            "complexity": complexity,
            "context": {
                **sanitized_context,
                'vulcan_fallback': True,
                'original_tool': original_tool,
            },
        }
        
        # Get API key from environment
        api_key = os.environ.get("GRAPHIX_API_KEY")
        if not api_key:
            # For internal delegation, use a special bypass key
            api_key = "internal-vulcan-delegation"
            logger.debug(
                f"{LOG_PREFIX} Using internal delegation key for Arena"
            )
        
        # Make request to Arena
        response = httpx.post(
            ARENA_REASONING_URL,
            json=arena_payload,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=ARENA_DELEGATION_TIMEOUT,
        )
        
        if response.status_code == 200:
            try:
                arena_result = response.json()
            except json.JSONDecodeError as e:
                logger.error(
                    f"{LOG_PREFIX} Arena returned invalid JSON: {e}. "
                    f"Response: {response.text[:200]}"
                )
                return None
                
            result_data = arena_result.get('result', {})
            
            logger.info(
                f"{LOG_PREFIX} Arena delegation successful: "
                f"confidence={result_data.get('confidence', 'N/A')}"
            )
            
            return {
                'conclusion': result_data.get('conclusion'),
                'confidence': result_data.get('confidence', 0.5),
                'explanation': result_data.get('explanation'),
                'arena_fallback': True,
                'original_tool': original_tool,
            }
        else:
            logger.error(
                f"{LOG_PREFIX} Arena HTTP error {response.status_code}: "
                f"{response.text[:200]}"
            )
            return None
            
    except ImportError:
        logger.warning(
            f"{LOG_PREFIX} httpx not available for Arena delegation. "
            f"Install with: pip install httpx"
        )
        return None
    except Exception as e:
        # Catch specific httpx exceptions if available
        try:
            import httpx
            if isinstance(e, (httpx.ConnectTimeout, httpx.ReadTimeout)):
                logger.error(
                    f"{LOG_PREFIX} Arena timed out after {ARENA_DELEGATION_TIMEOUT}s"
                )
                return None
        except ImportError:
            pass
        
        logger.error(f"{LOG_PREFIX} Arena delegation failed: {e}", exc_info=True)
        return None
