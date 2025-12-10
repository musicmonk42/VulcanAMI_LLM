"""
Security fixes module for integration package.
Re-exports safe_pickle_load and RestrictedUnpickler from vulcan.security_fixes.
"""

from src.vulcan.security_fixes import (
    safe_pickle_load,
    RestrictedUnpickler,
)

__all__ = ["safe_pickle_load", "RestrictedUnpickler"]
