"""
Vulcan status proxy route handlers extracted from full_platform.py.

Provides:
- _proxy_vulcan_status_endpoint (internal utility)
- v1_status_proxy (GET /v1/status)
- v1_cognitive_status_proxy (GET /v1/cognitive/status)
- v1_llm_status_proxy (GET /v1/llm/status)
- v1_routing_status_proxy (GET /v1/routing/status)
- safety_status_proxy (GET /safety/status)
- safety_audit_recent_proxy (GET /safety/audit/recent)
- world_model_status_proxy (GET /world-model/status)
- memory_status_proxy (GET /memory/status)
- hardware_status_proxy (GET /hardware/status)
"""

import importlib
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def _proxy_vulcan_status_endpoint(
    endpoint_name: str,
    handler_name: str,
    fallback_response: Optional[Dict[str, Any]] = None,
    **handler_kwargs
) -> Any:
    """
    Helper function to proxy status requests to VULCAN endpoints.

    Reduces code duplication across all status proxy endpoints by handling:
    - VULCAN module loading and validation
    - Dynamic import of the handler function
    - Error handling and fallback responses

    Args:
        endpoint_name: Name of the endpoint for logging (e.g., "v1/status")
        handler_name: Name of the handler function in src.vulcan.main
        fallback_response: Response to return if handler import fails
        **handler_kwargs: Keyword arguments to pass to the handler

    Returns:
        Response from the VULCAN handler or error response
    """
    from src.full_platform import _check_vulcan_deployment, _get_vulcan_module

    if fallback_response is None:
        fallback_response = {"status": "unavailable"}

    try:
        vulcan_module, error_response = _get_vulcan_module()
        if error_response:
            return error_response

        deployment_error = _check_vulcan_deployment(vulcan_module)
        if deployment_error:
            return deployment_error

        try:
            # Dynamic import of the handler function
            vulcan_main = importlib.import_module("src.vulcan.main")
            handler = getattr(vulcan_main, handler_name, None)
            if handler is None:
                logger.warning(f"Handler {handler_name} not found in src.vulcan.main")
                return fallback_response
            result = await handler(**handler_kwargs)
            return result
        except ImportError as e:
            logger.warning(f"Could not import {handler_name}: {e}")
            return fallback_response
    except Exception as e:
        logger.error(f"{endpoint_name} proxy error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(e)}
        )


@router.get("/v1/status", response_model=None)
async def v1_status_proxy():
    """
    Proxy endpoint for VULCAN system status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="v1/status",
        handler_name="system_status"
    )


@router.get("/v1/cognitive/status", response_model=None)
async def v1_cognitive_status_proxy():
    """
    Proxy endpoint for VULCAN cognitive status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="v1/cognitive/status",
        handler_name="cognitive_status"
    )


@router.get("/v1/llm/status", response_model=None)
async def v1_llm_status_proxy():
    """
    Proxy endpoint for VULCAN LLM status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="v1/llm/status",
        handler_name="llm_status"
    )


@router.get("/v1/routing/status", response_model=None)
async def v1_routing_status_proxy():
    """
    Proxy endpoint for VULCAN routing status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="v1/routing/status",
        handler_name="routing_status"
    )


@router.get("/safety/status", response_model=None)
async def safety_status_proxy():
    """
    Proxy endpoint for VULCAN safety status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="safety/status",
        handler_name="safety_status"
    )


@router.get("/safety/audit/recent", response_model=None)
async def safety_audit_recent_proxy(limit: int = 10):
    """
    Proxy endpoint for VULCAN safety audit recent logs.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="safety/audit/recent",
        handler_name="safety_audit_recent",
        fallback_response={"logs": []},
        limit=limit
    )


@router.get("/world-model/status", response_model=None)
async def world_model_status_proxy():
    """
    Proxy endpoint for VULCAN world model status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="world-model/status",
        handler_name="world_model_status"
    )


@router.get("/memory/status", response_model=None)
async def memory_status_proxy():
    """
    Proxy endpoint for VULCAN memory status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="memory/status",
        handler_name="memory_status"
    )


@router.get("/hardware/status", response_model=None)
async def hardware_status_proxy():
    """
    Proxy endpoint for VULCAN hardware status.
    Public endpoint - no authentication required for dashboard display.
    """
    return await _proxy_vulcan_status_endpoint(
        endpoint_name="hardware/status",
        handler_name="hardware_status"
    )
