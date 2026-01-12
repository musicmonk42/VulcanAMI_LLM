"""
Security fixes module for conformal package.
Re-exports safe_pickle_load and RestrictedUnpickler from vulcan.security_fixes.
"""

# Try relative import first (when running as package), then absolute import
try:
    from vulcan.security_fixes import RestrictedUnpickler, safe_pickle_load
except ImportError:
    # Fallback for different import contexts
    try:
        from src.vulcan.security_fixes import RestrictedUnpickler, safe_pickle_load
    except ImportError:
        # Final fallback - raise informative error
        raise ImportError(
            "Cannot import security_fixes. Ensure vulcan.security_fixes is available. "
            "Tried: 'vulcan.security_fixes' and 'src.vulcan.security_fixes'"
        )

__all__ = ["safe_pickle_load", "RestrictedUnpickler"]
