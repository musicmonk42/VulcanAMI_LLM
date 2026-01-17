"""
Vulcan Utilities Package

Common utilities for the Vulcan system.
"""

from .safe_execution import (
    SafeCodeExecutor,
    execute_math_code,
    get_executor,
    is_safe_execution_available,
    reset_executor,
)

from .numeric_utils import (
    is_close,
    clamp,
    safe_divide,
    normalize_weights,
    DEFAULT_EPSILON,
)

__version__ = "1.2.0"
__all__ = [
    # Safe execution
    "SafeCodeExecutor",
    "execute_math_code",
    "get_executor",
    "is_safe_execution_available",
    "reset_executor",
    # Numeric utilities
    "is_close",
    "clamp",
    "safe_divide",
    "normalize_weights",
    "DEFAULT_EPSILON",
]
