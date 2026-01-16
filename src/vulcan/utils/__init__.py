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
    float_equals,
    is_close,
    clamp,
    is_in_range,
    safe_divide,
    normalize_weights,
    check_finite,
    validate_probability,
    DEFAULT_EPSILON,
)

__version__ = "1.1.0"
__all__ = [
    # Safe execution
    "SafeCodeExecutor",
    "execute_math_code",
    "get_executor",
    "is_safe_execution_available",
    "reset_executor",
    # Numeric utilities
    "float_equals",
    "is_close",
    "clamp",
    "is_in_range",
    "safe_divide",
    "normalize_weights",
    "check_finite",
    "validate_probability",
    "DEFAULT_EPSILON",
]
