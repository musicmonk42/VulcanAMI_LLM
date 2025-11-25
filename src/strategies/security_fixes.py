"""
Security fixes module for strategies package.
Re-exports safe_pickle_load and RestrictedUnpickler from vulcan.security_fixes.
"""

from src.vulcan.security_fixes import (
    safe_pickle_load,
    RestrictedUnpickler,
)

__all__ = ['safe_pickle_load', 'RestrictedUnpickler']
