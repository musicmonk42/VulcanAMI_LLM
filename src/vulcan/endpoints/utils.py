"""
Endpoint utilities for VULCAN.

Provides common utilities for endpoint handlers, including deployment
access helpers that handle both standalone and mounted sub-app scenarios.
"""

import importlib
import logging
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
    
    Args:
        request: FastAPI request object
        
    Returns:
        ProductionDeployment instance or None if not found
    """
    # Try 1: Direct app.state (standalone mode or properly propagated)
    if hasattr(request.app.state, "deployment") and request.app.state.deployment is not None:
        return request.app.state.deployment
    
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
                return vulcan_module.app.state.deployment
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not get deployment from {module_path}: {e}")
            continue
    
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
        logger.error(
            "CRITICAL: deployment not found. Checked: request.app.state, "
            "vulcan.main.app.state, src.vulcan.main.app.state"
        )
        raise HTTPException(
            status_code=503,
            detail="System initializing - deployment not ready. Please retry in a few seconds."
        )
    return deployment
