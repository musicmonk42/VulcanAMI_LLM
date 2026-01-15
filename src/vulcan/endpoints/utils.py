"""
Endpoint utilities for VULCAN.

Provides common utilities for endpoint handlers, including deployment
access helpers that handle both standalone and mounted sub-app scenarios.
"""

import importlib
import logging
import os
from typing import Optional, Union, TYPE_CHECKING

from fastapi import Request, HTTPException

if TYPE_CHECKING:
    from vulcan.orchestrator.deployment import ProductionDeployment

logger = logging.getLogger(__name__)


def get_deployment_from_module() -> Optional["ProductionDeployment"]:
    """
    Get ProductionDeployment instance from module-level app state.
    
    This function provides a fallback for getting deployment when a Request
    object is not available (e.g., when called from proxy functions in
    full_platform.py that bypass FastAPI's request injection).
    
    Returns:
        ProductionDeployment instance or None if not found
        
    Note:
        This should only be used when a Request object is unavailable.
        Prefer get_deployment(request) when possible as it can propagate
        the deployment to the request's app state for faster subsequent access.
    """
    for module_path in ["vulcan.main", "src.vulcan.main"]:
        try:
            vulcan_module = importlib.import_module(module_path)
            # Use getattr chain with None defaults for cleaner access
            app = getattr(vulcan_module, "app", None)
            if app is None:
                continue
            state = getattr(app, "state", None)
            if state is None:
                continue
            deployment = getattr(state, "deployment", None)
            if deployment is not None:
                logger.debug(f"Deployment found via module {module_path}")
                return deployment
        except ImportError as e:
            logger.debug(f"Could not import {module_path}: {e}")
            continue
        except AttributeError as e:
            logger.debug(f"Module {module_path} exists but deployment not accessible: {e}")
            continue
    
    logger.warning("Deployment not found via module fallback")
    return None


def get_deployment(request: Optional[Request] = None) -> Optional["ProductionDeployment"]:
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
        request: Optional FastAPI request object. If None, falls back to
                 module-level lookup only.
        
    Returns:
        ProductionDeployment instance or None if not found
    """
    # If no request provided, use module fallback only
    if request is None:
        return get_deployment_from_module()
    
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
            # Use getattr chain with None defaults for cleaner access
            app = getattr(vulcan_module, "app", None)
            if app is None:
                continue
            state = getattr(app, "state", None)
            if state is None:
                continue
            deployment = getattr(state, "deployment", None)
            if deployment is not None:
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
                except (AttributeError, TypeError) as prop_err:
                    # Non-fatal: propagation failure just means slower subsequent lookups
                    # AttributeError: state doesn't exist or is read-only
                    # TypeError: state object doesn't support attribute assignment
                    logger.debug(f"Could not propagate deployment to request.app: {type(prop_err).__name__}: {prop_err}")
                
                return deployment
        except ImportError as e:
            logger.debug(f"Could not import {module_path}: {e}")
            continue
        except AttributeError as e:
            logger.debug(f"Module {module_path} exists but deployment not accessible: {e}")
            continue
    
    # Log detailed diagnostic information when deployment is not found
    # Use getattr with default to safely get state attributes
    try:
        state_attrs = list(dir(request.app.state)) if hasattr(request.app, 'state') else []
        # Filter to only show deployment-related or custom attributes (not dunder methods)
        state_attrs = [a for a in state_attrs if not a.startswith('_')]
    except (TypeError, AttributeError):
        state_attrs = ['<unable to inspect>']
    
    logger.warning(
        f"Deployment not found. Diagnostics: "
        f"request.app.title={app_title}, "
        f"request.app.state attrs={state_attrs}"
    )
    
    return None


def require_deployment(request: Optional[Request] = None) -> "ProductionDeployment":
    """
    Get deployment or raise HTTPException if not available.
    
    This function should be used in endpoint handlers to ensure
    deployment is available before processing requests.
    
    Args:
        request: Optional FastAPI request object. If None, falls back to
                 module-level lookup only.
        
    Returns:
        ProductionDeployment instance
        
    Raises:
        HTTPException: 503 if deployment not found
    """
    deployment = get_deployment(request)
    if deployment is None:
        # Get process ID for multi-worker debugging
        pid = os.getpid()
        app_title = getattr(request.app, "title", "unknown") if request else "no_request"
        
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
