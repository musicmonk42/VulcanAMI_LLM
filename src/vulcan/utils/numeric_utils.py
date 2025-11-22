# src/vulcan/utils/numeric_utils.py
"""
Numeric Utilities for Safe Numerical Comparisons

Provides utilities for:
- Safe float comparisons with epsilon tolerance
- Bounds checking
- Range validation
- Overflow protection

Security: LOW-MEDIUM (Correctness)
"""

import math
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Default epsilon for float comparisons
DEFAULT_EPSILON = 1e-9

# Absolute max epsilon (safety)
MAX_EPSILON = 1e-3


def float_equals(a: float, b: float, epsilon: float = DEFAULT_EPSILON) -> bool:
    """
    Check if two floats are equal within epsilon tolerance
    
    Args:
        a: First value
        b: Second value
        epsilon: Tolerance (default 1e-9)
        
    Returns:
        True if |a - b| < epsilon
        
    Note:
        For very large or small numbers, consider using is_close() instead,
        which handles both relative and absolute tolerance.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    
    # Handle infinity and NaN
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    
    return abs(a - b) < epsilon


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Check if two values are close using both relative and absolute tolerance.
    
    This is more robust than float_equals for very large or very small numbers.
    Uses the same algorithm as math.isclose from Python 3.5+.
    
    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)
        
    Returns:
        True if abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        
    Example:
        >>> is_close(1e10, 1e10 + 1)  # Relative tolerance for large numbers
        True
        >>> is_close(1e-10, 2e-10, abs_tol=1e-10)  # Absolute for small numbers
        True
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]"""
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return max(min_val, min(max_val, value))


def is_in_range(value: float, min_val: float, max_val: float, 
                epsilon: float = DEFAULT_EPSILON) -> bool:
    """Check if value is in range [min_val, max_val] with epsilon tolerance"""
    return (value >= min_val - epsilon) and (value <= max_val + epsilon)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for division by zero"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def normalize_weights(weights: list, epsilon: float = DEFAULT_EPSILON) -> list:
    """Normalize weights to sum to 1.0"""
    if not weights:
        return []
    
    total = sum(weights)
    if total <= epsilon:
        raise ValueError(f"Cannot normalize: total weight {total} too small")
    
    return [w / total for w in weights]


def check_finite(value: Union[float, list, tuple], name: str = "value") -> bool:
    """Check if value(s) are finite (not inf, not nan)"""
    if isinstance(value, (list, tuple)):
        return all(check_finite(v, f"{name}[{i}]") for i, v in enumerate(value))
    
    if not isinstance(value, (int, float)):
        return False
    
    if math.isnan(value) or math.isinf(value):
        return False
    
    return True


def validate_probability(p: float, epsilon: float = DEFAULT_EPSILON) -> bool:
    """Validate that p is a valid probability in [0, 1]"""
    return check_finite(p) and is_in_range(p, 0.0, 1.0, epsilon)
