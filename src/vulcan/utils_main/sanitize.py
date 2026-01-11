# ============================================================
# VULCAN-AGI Sanitization Utilities Module
# Functions for sanitizing data for JSON serialization
# ============================================================
#
# This module provides utilities for:
#     - Removing None keys from dictionaries
#     - Converting non-serializable types to strings
#     - Handling circular references
#     - Processing datetime and enum types
#
# USAGE:
#     from vulcan.utils_main.sanitize import sanitize_payload, deep_sanitize_for_json
#     
#     # Basic sanitization
#     clean_data = sanitize_payload({None: "bad", "good": "value"})
#     
#     # Deep sanitization (more aggressive)
#     clean_data = deep_sanitize_for_json(complex_object)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and type hints
# ============================================================

import base64
import datetime
import logging
from enum import Enum as EnumBase
from typing import Any, Dict, List, Tuple, Union

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum recursion depth for deep sanitization (prevents stack overflow on circular refs)
DEEP_SANITIZE_MAX_DEPTH = 50

# Marker for unserializable values
UNSERIALIZABLE_MARKER = "__unserializable__"
MAX_DEPTH_MARKER = "__max_depth_exceeded__"


# ============================================================
# SANITIZATION FUNCTIONS
# ============================================================


def sanitize_payload(data: Any) -> Any:
    """
    Sanitize payload for JSON serialization by handling None keys and values.
    
    JSON does not support None as dictionary keys. This function:
    - Removes dictionary entries with None keys (which would cause serialization failure)
    - Recursively processes nested dicts and lists
    - Converts non-serializable values to strings
    - Detects and warns about key collisions when converting non-string keys
    
    Args:
        data: The data to sanitize (dict, list, or any value)
        
    Returns:
        Sanitized data safe for JSON serialization
        
    Example:
        >>> sanitize_payload({None: "value", "key": "value"})
        {'key': 'value'}
        >>> sanitize_payload({"nested": {None: "bad", "ok": 1}})
        {'nested': {'ok': 1}}
    """
    if isinstance(data, dict):
        # Filter out None keys and recursively sanitize values
        # Note: str(k) handles non-string keys (e.g., integers) for JSON compatibility
        sanitized = {}
        for k, v in data.items():
            if k is None:
                continue  # Remove entries with None keys entirely
            str_key = str(k) if not isinstance(k, str) else k
            if str_key in sanitized:
                logger.warning(f"Key collision during sanitization: '{str_key}'")
            sanitized[str_key] = sanitize_payload(v)
        return sanitized
    elif isinstance(data, list):
        # Recursively sanitize list elements
        return [sanitize_payload(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuples to lists for JSON compatibility
        return [sanitize_payload(item) for item in data]
    elif data is None:
        return None  # None values are fine, just not None keys
    elif isinstance(data, (str, int, float, bool)):
        return data
    else:
        # For any other type, try to convert to string
        try:
            return str(data)
        except Exception:
            return "__unserializable__"


def deep_sanitize_for_json(data: Any, _depth: int = 0) -> Any:
    """
    Deep sanitization for JSON serialization - more aggressive than sanitize_payload.
    
    This is a fallback when standard sanitization fails. It:
    - Removes ALL non-string keys (not just None)
    - Converts any objects to their string representations
    - Handles circular references by tracking depth
    - Handles special types like datetime, enum, etc.
    
    Args:
        data: The data to sanitize
        _depth: Current recursion depth (to prevent infinite recursion)
        
    Returns:
        JSON-serializable data
    """
    if _depth > DEEP_SANITIZE_MAX_DEPTH:
        return "__max_depth_exceeded__"
    
    if data is None:
        return None
    
    if isinstance(data, (str, int, float, bool)):
        return data
    
    # Handle bytes - try UTF-8 decode first, fall back to base64
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return base64.b64encode(data).decode('ascii')
    
    # Handle bytearray - convert to bytes first
    if isinstance(data, bytearray):
        return deep_sanitize_for_json(bytes(data), _depth)
    
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Skip None keys entirely
            if k is None:
                continue
            # Convert non-string keys to strings
            str_key = str(k) if not isinstance(k, str) else k
            result[str_key] = deep_sanitize_for_json(v, _depth + 1)
        return result
    
    if isinstance(data, (list, tuple)):
        return [deep_sanitize_for_json(item, _depth + 1) for item in data]
    
    # Handle Enum types (more precise than checking for 'value' attribute)
    if isinstance(data, EnumBase):
        try:
            # Recursively sanitize enum value in case it's a complex type
            return deep_sanitize_for_json(data.value, _depth + 1)
        except Exception:
            return str(data)  # Fallback to string representation
    
    # Handle datetime types (more precise than checking for 'isoformat' attribute)
    if isinstance(data, (datetime.datetime, datetime.date, datetime.time)):
        return data.isoformat()
    
    # Custom objects: try to convert via __dict__
    if hasattr(data, '__dict__'):
        try:
            return deep_sanitize_for_json(data.__dict__, _depth + 1)
        except (TypeError, ValueError, RecursionError):
            # Log at debug level since this is an expected fallback path
            pass  # Fall through to string conversion
    
    # Last resort: convert to string
    try:
        return str(data)
    except Exception:
        return UNSERIALIZABLE_MARKER


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def is_json_serializable(data: Any) -> bool:
    """
    Check if data is JSON serializable without modification.
    
    Args:
        data: The data to check
        
    Returns:
        True if data can be serialized to JSON as-is
    """
    import json
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely convert data to JSON string, sanitizing if necessary.
    
    Args:
        data: The data to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation of the data
    """
    import json
    try:
        return json.dumps(data, **kwargs)
    except (TypeError, ValueError):
        sanitized = deep_sanitize_for_json(data)
        return json.dumps(sanitized, **kwargs)


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

# Original function names had underscore prefix in main.py
_sanitize_payload = sanitize_payload
_deep_sanitize_for_json = deep_sanitize_for_json


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Main functions
    "sanitize_payload",
    "deep_sanitize_for_json",
    # Convenience functions
    "is_json_serializable",
    "safe_json_dumps",
    # Constants
    "DEEP_SANITIZE_MAX_DEPTH",
    "UNSERIALIZABLE_MARKER",
    "MAX_DEPTH_MARKER",
    # Backward compatibility
    "_sanitize_payload",
    "_deep_sanitize_for_json",
]


# Log module initialization
logger.debug(f"Sanitize module v{__version__} loaded (max depth: {DEEP_SANITIZE_MAX_DEPTH})")
