#!/usr/bin/env python3
# =============================================================================
# Platform Utilities
# =============================================================================
# Extracted from full_platform.py:
# - _get_vulcan_module (cached VULCAN module import)
# - _check_vulcan_deployment (deployment state check)
# =============================================================================

import importlib
import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger("unified_platform")

# Cache the VULCAN module to avoid repeated imports
_vulcan_module_cache = None


def _get_vulcan_module():
    """
    Get the VULCAN module, using a cached version if available.

    Returns:
        tuple: (module, error_response) - module if successful, None and JSONResponse if not
    """
    global _vulcan_module_cache

    # Return cached module if available and still valid
    if _vulcan_module_cache is not None:
        return _vulcan_module_cache, None

    try:
        vulcan_module = importlib.import_module("src.vulcan.main")
        if not hasattr(vulcan_module, "app"):
            return None, JSONResponse(
                status_code=503,
                content={"error": "VULCAN module not available"}
            )
        _vulcan_module_cache = vulcan_module
        return vulcan_module, None
    except ImportError as e:
        logger.error(f"Failed to import VULCAN module: {e}")
        return None, JSONResponse(
            status_code=503,
            content={"error": "VULCAN module unavailable", "detail": str(e)}
        )


def _check_vulcan_deployment(vulcan_module):
    """
    Check if the VULCAN deployment is initialized.

    Returns:
        JSONResponse or None: error response if not initialized, None if OK
    """
    if not hasattr(vulcan_module.app, "state") or not hasattr(vulcan_module.app.state, "deployment"):
        return JSONResponse(
            status_code=503,
            content={
                "error": "VULCAN not initialized",
                "detail": "The VULCAN deployment has not been initialized yet."
            }
        )
    return None
