#!/usr/bin/env python3
# =============================================================================
# VULCAN Proxy Routes
# =============================================================================
# Extracted from full_platform.py:
# - vulcan_chat_redirect (backward compatibility redirect)
# - vulcan_health_proxy (VULCAN health check proxy)
# - debug_parent_deployment (deployment state debug endpoint)
# - vulcan_chat_proxy (VULCAN chat API proxy)
# - v1_chat_proxy (alternative chat API endpoint)
# =============================================================================

import importlib
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

logger = logging.getLogger("unified_platform")

router = APIRouter()


@router.get("/vulcan_chat.html")
async def vulcan_chat_redirect():
    """Redirect /vulcan_chat.html to root for backward compatibility."""
    return RedirectResponse(url="/", status_code=301)


@router.get("/vulcan/health", response_model=None)
async def vulcan_health_proxy():
    """
    Proxy endpoint for VULCAN health check.

    This endpoint calls the VULCAN health check directly, providing a reliable
    way to check VULCAN's status even if the /vulcan mount has issues.
    """
    try:
        # Try to import and call VULCAN's health check directly
        vulcan_module = importlib.import_module("src.vulcan.main")
        if hasattr(vulcan_module, "app") and hasattr(vulcan_module.app, "state"):
            vulcan_app = vulcan_module.app

            # Check if deployment is initialized
            if not hasattr(vulcan_app.state, "deployment"):
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "unhealthy",
                        "error": "VULCAN deployment not initialized",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            deployment = vulcan_app.state.deployment
            if deployment is None:
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "unhealthy",
                        "error": "VULCAN deployment is None",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            # Get status from deployment
            try:
                status = deployment.get_status()
                health_checks = {
                    "error_rate": status.get("health", {}).get("error_rate", 0) < 0.1,
                    "memory_usage": status.get("health", {}).get("memory_usage_mb", 0) < 2000,
                    "latency": status.get("health", {}).get("latency_ms", 0) < 1000,
                }
                healthy = all(health_checks.values())

                return {
                    "status": "healthy" if healthy else "unhealthy",
                    "checks": health_checks,
                    "details": status,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                logger.warning(f"VULCAN health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "unhealthy",
                    "error": "VULCAN module not properly initialized",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
    except ImportError as e:
        logger.error(f"Failed to import VULCAN module: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "error": f"VULCAN module import failed: {e}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"VULCAN health proxy error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


@router.get("/debug/deployment", response_model=None)
async def debug_parent_deployment(request: Request):
    """
    Debug endpoint to verify deployment state on the PARENT app.

    Use this endpoint to confirm that app.state.deployment is correctly set
    on the parent app that hosts the VULCAN sub-app. Compare with
    /vulcan/debug/deployment to verify both apps have deployment set.

    If this returns {"deployment": "None"}, the deployment was not
    properly set on the parent app during startup in full_platform.py.

    Returns:
        Dict containing:
            - deployment: String representation of the deployment object (or "None")
            - deployment_type: Type name of the deployment object
            - app_title: Title of the parent app
            - worker_id: Process ID of the worker handling this request
            - startup_time: Timestamp when the app started (if available)
            - has_deployment_attr: Whether app.state has a deployment attribute
    """
    pid = os.getpid()
    deployment = getattr(request.app.state, "deployment", None)

    return {
        "deployment": str(deployment) if deployment is not None else "None",
        "deployment_type": type(deployment).__name__ if deployment is not None else "NoneType",
        "app_title": getattr(request.app, "title", "unknown"),
        "worker_id": pid,
        "startup_time": getattr(request.app.state, "startup_time", None),
        "has_deployment_attr": hasattr(request.app.state, "deployment"),
    }


@router.post("/vulcan/v1/chat", response_model=None)
async def vulcan_chat_proxy(request: Request):
    """
    Proxy endpoint for VULCAN chat API.

    This endpoint forwards chat requests to VULCAN's /v1/chat endpoint,
    providing a reliable way to access the chat API even if the /vulcan
    mount has issues.
    """
    # Parse request body first
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid JSON",
                "detail": str(e),
            }
        )

    try:
        # Try to import VULCAN's chat handler
        vulcan_module = importlib.import_module("src.vulcan.main")
        if hasattr(vulcan_module, "app"):
            vulcan_app = vulcan_module.app

            # Check if deployment is initialized
            if not hasattr(vulcan_app.state, "deployment"):
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "VULCAN not initialized",
                        "detail": "The VULCAN deployment has not been initialized yet. Please try again later.",
                    }
                )

            # Import and call the chat endpoint handler directly
            try:
                from src.vulcan.main import UnifiedChatRequest, unified_chat
            except ImportError as e:
                logger.error(f"Failed to import VULCAN chat handler: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Chat handler unavailable",
                        "detail": str(e),
                    }
                )

            # Validate and create request object
            try:
                chat_request = UnifiedChatRequest(**body)
            except Exception as e:
                # Handle Pydantic validation errors
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Invalid request",
                        "detail": str(e),
                    }
                )

            # Call the chat endpoint function directly
            result = await unified_chat(request, chat_request)
            return result
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "VULCAN module not available",
                    "detail": "The VULCAN module could not be loaded.",
                }
            )
    except Exception as e:
        logger.error(f"VULCAN chat proxy error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal error",
                "detail": str(e),
            }
        )


@router.post("/v1/chat", response_model=None)
async def v1_chat_proxy(request: Request):
    """
    Alternative proxy endpoint for chat API at /v1/chat.

    This provides compatibility with clients that expect the chat API
    at /v1/chat instead of /vulcan/v1/chat.
    """
    # Delegate to the VULCAN chat proxy
    return await vulcan_chat_proxy(request)
