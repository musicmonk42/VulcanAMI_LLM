#!/usr/bin/env python3
# =============================================================================
# VULCAN Feedback Proxy Routes
# =============================================================================
# Extracted from full_platform.py:
# - v1_feedback_thumbs_proxy (thumbs up/down feedback proxy)
# - v1_feedback_proxy (general feedback proxy)
# - v1_feedback_stats_proxy (feedback stats proxy)
# =============================================================================

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from platform.utils import _check_vulcan_deployment, _get_vulcan_module

logger = logging.getLogger("unified_platform")

router = APIRouter()


@router.post("/v1/feedback/thumbs", response_model=None)
async def v1_feedback_thumbs_proxy(request: Request):
    """
    Proxy endpoint for VULCAN feedback thumbs API.

    This forwards thumbs up/down feedback requests to VULCAN's /v1/feedback/thumbs
    endpoint, providing a consistent API surface for the chat interface.
    """
    # Parse request body
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "detail": str(e)}
        )

    try:
        # Get VULCAN module (cached)
        vulcan_module, error_response = _get_vulcan_module()
        if error_response:
            return error_response

        # Check if deployment is initialized
        deployment_error = _check_vulcan_deployment(vulcan_module)
        if deployment_error:
            return deployment_error

        # Import the handler and request model
        try:
            from src.vulcan.main import ThumbsFeedbackRequest, submit_thumbs_feedback
        except ImportError as e:
            logger.error(f"Failed to import VULCAN feedback handler: {e}")
            return JSONResponse(
                status_code=503,
                content={"error": "Feedback handler unavailable", "detail": str(e)}
            )

        # Validate and create request object
        try:
            feedback_request = ThumbsFeedbackRequest(**body)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "detail": str(e)}
            )

        # Call the feedback endpoint function directly
        result = await submit_thumbs_feedback(feedback_request)
        return result

    except Exception as e:
        logger.error(f"Feedback thumbs proxy error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(e)}
        )


@router.post("/v1/feedback", response_model=None)
async def v1_feedback_proxy(request: Request):
    """
    Proxy endpoint for VULCAN feedback API.

    This forwards general feedback requests to VULCAN's /v1/feedback endpoint.
    """
    # Parse request body
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON", "detail": str(e)}
        )

    try:
        # Get VULCAN module (cached)
        vulcan_module, error_response = _get_vulcan_module()
        if error_response:
            return error_response

        # Check if deployment is initialized
        deployment_error = _check_vulcan_deployment(vulcan_module)
        if deployment_error:
            return deployment_error

        # Import the handler and request model
        try:
            from src.vulcan.main import FeedbackRequest, submit_feedback
        except ImportError as e:
            logger.error(f"Failed to import VULCAN feedback handler: {e}")
            return JSONResponse(
                status_code=503,
                content={"error": "Feedback handler unavailable", "detail": str(e)}
            )

        # Validate and create request object
        try:
            feedback_request = FeedbackRequest(**body)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "detail": str(e)}
            )

        # Call the feedback endpoint function directly
        result = await submit_feedback(feedback_request)
        return result

    except Exception as e:
        logger.error(f"Feedback proxy error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(e)}
        )


@router.get("/v1/feedback/stats", response_model=None)
async def v1_feedback_stats_proxy():
    """
    Proxy endpoint for VULCAN feedback stats API.

    This forwards feedback stats requests to VULCAN's /v1/feedback/stats endpoint.
    """
    try:
        # Get VULCAN module (cached)
        vulcan_module, error_response = _get_vulcan_module()
        if error_response:
            return error_response

        # Check if deployment is initialized
        deployment_error = _check_vulcan_deployment(vulcan_module)
        if deployment_error:
            return deployment_error

        # Import the handler
        try:
            from src.vulcan.main import get_feedback_stats
        except ImportError as e:
            logger.error(f"Failed to import VULCAN feedback stats handler: {e}")
            return JSONResponse(
                status_code=503,
                content={"error": "Feedback stats handler unavailable", "detail": str(e)}
            )

        # Call the feedback stats endpoint function directly
        result = await get_feedback_stats()
        return result

    except Exception as e:
        logger.error(f"Feedback stats proxy error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(e)}
        )
