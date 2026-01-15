"""
Endpoint utilities for VULCAN.

Provides common utilities for endpoint handlers, including deployment
access helpers that handle both standalone and mounted sub-app scenarios.
"""

import importlib
import logging
import os
from typing import Optional, TYPE_CHECKING

from fastapi import Request, HTTPException

if TYPE_CHECKING:
    from vulcan.orchestrator.deployment import ProductionDeployment

logger = logging.getLogger(__name__)


def get_deployment(request: Request) -> Optional["ProductionDeployment"]:
    """
    Get the ProductionDeployment instance from any valid location.
    
    Handles both standalone VULCAN and mounted sub-app scenarios.
    
    This function addresses the sub-app state isolation bug where:
    1. Deployment is attached to vulcan_module.app.state.deployment
    2. VULCAN is mounted as sub-app: app.mount("/vulcan", vulcan_module.app)
    3. Request.app references the parent app, not vulcan_module.app
    4. Parent app never has deployment set on its state
    
    When deployment is found via module import (fallback path), this function
    also propagates it to request.app.state.deployment for faster access on
    subsequent requests from the same app instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ProductionDeployment instance or None if not found
    """
    # Try 1: Direct app.state (standalone mode or properly propagated)
    if hasattr(request.app.state, "deployment") and request.app.state.deployment is not None:
        logger.debug("Deployment found in request.app.state")
        return request.app.state.deployment
    
    # Log which app we're checking (useful for debugging sub-app issues)
    app_title = getattr(request.app, "title", "unknown")
    logger.debug(f"Deployment not in request.app.state (app.title={app_title}), trying module fallback")
    
    # Try 2: vulcan.main module (when imported as vulcan.main)
    # Try 3: src.vulcan.main module (full_platform mounting pattern)
    for module_path in ["vulcan.main", "src.vulcan.main"]:
        try:
            vulcan_module = importlib.import_module(module_path)
            if (
                hasattr(vulcan_module, "app") 
                and hasattr(vulcan_module.app, "state")
                and hasattr(vulcan_module.app.state, "deployment")
                and vulcan_module.app.state.deployment is not None
            ):
                deployment = vulcan_module.app.state.deployment
                logger.info(f"Deployment found via module {module_path}")
                
                # FIX: Propagate deployment to request.app.state for faster
                # access on subsequent requests. This avoids repeated module
                # imports on every request when running as mounted sub-app.
                try:
                    request.app.state.deployment = deployment
                    logger.info(
                        f"Propagated deployment to request.app.state "
                        f"(app.title={app_title}) for faster subsequent access"
                    )
                except Exception as prop_err:
                    # Non-fatal: propagation failure just means slower subsequent lookups
                    logger.debug(f"Could not propagate deployment to request.app: {prop_err}")
                
                return deployment
        except ImportError as e:
            logger.debug(f"Could not import {module_path}: {e}")
            continue
        except AttributeError as e:
            logger.debug(f"Module {module_path} exists but deployment not accessible: {e}")
            continue
    
    # Log detailed diagnostic information when deployment is not found
    logger.warning(
        f"Deployment not found. Diagnostics: "
        f"request.app.title={app_title}, "
        f"request.app.state attrs={list(vars(request.app.state).keys()) if hasattr(request.app, 'state') else 'no state'}"
    )
    
    return None


def require_deployment(request: Request) -> "ProductionDeployment":
    """
    Get deployment or raise HTTPException if not available.
    
    This function should be used in endpoint handlers to ensure
    deployment is available before processing requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ProductionDeployment instance
        
    Raises:
        HTTPException: 503 if deployment not found
    """
    deployment = get_deployment(request)
    if deployment is None:
        # Get process ID for multi-worker debugging
        pid = os.getpid()
        app_title = getattr(request.app, "title", "unknown")
        
        logger.error(
            f"CRITICAL: deployment not found (pid={pid}, app.title={app_title}). "
            f"Checked: request.app.state, vulcan.main.app.state, src.vulcan.main.app.state. "
            f"This usually means the startup phase has not completed or failed."
        )
        raise HTTPException(
            status_code=503,
            detail="System initializing - deployment not ready. Please retry in a few seconds."
        )
    return deployment
