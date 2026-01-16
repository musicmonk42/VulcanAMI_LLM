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
    Get the ProductionDeployment instance from app state.
    
    Simplified version that relies on proper initialization rather than
    module hacking. This implements proper dependency injection by:
    1. Primary: Using request.app.state (proper DI)
    2. Fallback: Module import for backwards compatibility
    3. Auto-propagation: Caching deployment in app.state when found
    
    Args:
        request: Optional FastAPI request object. If None, falls back to
                 module-level lookup only.
        
    Returns:
        ProductionDeployment instance or None if not found
    """
    if request is None:
        # No request context - use module fallback only
        return get_deployment_from_module()
    
    # Primary path: get from request.app.state (proper DI)
    if hasattr(request.app.state, "deployment") and request.app.state.deployment is not None:
        logger.debug("Deployment found in request.app.state")
        return request.app.state.deployment
    
    # Log warning if not found (indicates initialization issue)
    app_title = getattr(request.app, "title", "unknown")
    logger.warning(
        f"Deployment not found in request.app.state (app.title={app_title}). "
        "This indicates startup may not have completed properly. Trying module fallback."
    )
    
    # Fallback to module import (for backwards compatibility during transition)
    deployment = get_deployment_from_module()
    
    # Propagate deployment to app.state if found (for faster subsequent access)
    if deployment is not None:
        try:
            request.app.state.deployment = deployment
            logger.info(
                f"Propagated deployment to request.app.state (app.title={app_title}) "
                "for faster subsequent access"
            )
        except (AttributeError, TypeError) as e:
            # Non-fatal: propagation failure just means slower subsequent lookups
            logger.debug(f"Could not propagate deployment to request.app: {type(e).__name__}: {e}")
    
    return deployment


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
