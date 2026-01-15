"""
Self-Improvement Endpoints

This module provides endpoints for managing VULCAN's autonomous self-improvement
drive, including starting/stopping, status monitoring, error reporting, and
approval management.

Endpoints:
    POST /v1/improvement/start           - Start autonomous improvement drive
    POST /v1/improvement/stop            - Stop improvement drive
    GET  /v1/improvement/status          - Get improvement status and statistics
    POST /v1/improvement/report-error    - Report error for improvement analysis
    POST /v1/improvement/approve         - Approve/reject pending improvement
    GET  /v1/improvement/pending         - Get pending approvals
    POST /v1/improvement/update-metric   - Update performance metric
"""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import error_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["self-improvement"])


@router.post("/v1/improvement/start", response_model=None)
async def start_self_improvement(request: Request) -> dict:
    """
    Start the autonomous self-improvement drive.
    
    Initiates VULCAN's self-improvement system which continuously monitors
    performance, identifies improvement opportunities, and proposes changes.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "started" or "already_running"
            - message: Human-readable status message
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or world model not initialized
        HTTPException: 400 if self-improvement not enabled in configuration
        HTTPException: 500 if start fails
    
    Note:
        The improvement drive runs asynchronously in the background and
        will continue until explicitly stopped or system shutdown.
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model:
            raise HTTPException(status_code=503, detail="World model not available")

        if (
            not hasattr(world_model, "self_improvement_enabled")
            or not world_model.self_improvement_enabled
        ):
            raise HTTPException(
                status_code=400, detail="Self-improvement not enabled in configuration"
            )

        if (
            hasattr(world_model, "improvement_running")
            and world_model.improvement_running
        ):
            return {
                "status": "already_running",
                "message": "Self-improvement drive is already running",
            }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, world_model.start_autonomous_improvement)

        logger.info("🚀 Self-improvement drive started via API")

        return {
            "status": "started",
            "message": "Self-improvement drive started successfully",
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start self-improvement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/improvement/stop", response_model=None)
async def stop_self_improvement(request: Request) -> dict:
    """
    Stop the autonomous self-improvement drive.
    
    Gracefully stops the self-improvement system. Any pending improvements
    remain available for approval but no new improvements will be generated.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "stopped" or "not_running"
            - message: Human-readable status message
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or world model not initialized
        HTTPException: 500 if stop fails
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model:
            raise HTTPException(status_code=503, detail="World model not available")

        if (
            not hasattr(world_model, "improvement_running")
            or not world_model.improvement_running
        ):
            return {
                "status": "not_running",
                "message": "Self-improvement drive is not running",
            }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, world_model.stop_autonomous_improvement)

        logger.info("🛑 Self-improvement drive stopped via API")

        return {
            "status": "stopped",
            "message": "Self-improvement drive stopped successfully",
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop self-improvement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/improvement/status", response_model=None)
async def get_improvement_status(request: Request) -> dict:
    """
    Get current self-improvement status and statistics.
    
    Returns comprehensive status including whether the drive is running,
    metrics being tracked, pending approvals, and improvement history.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing status from world model's get_improvement_status()
        which typically includes:
            - enabled: Whether self-improvement is enabled
            - running: Whether drive is currently active
            - state: Current state with pending approvals
            - metrics: Performance metrics being tracked
            - history: Recent improvement actions
    
    Raises:
        HTTPException: 503 if system or world model not initialized
        HTTPException: 500 if status retrieval fails
    
    Note:
        This endpoint also updates Prometheus metrics for monitoring:
        - improvement_approvals_pending: Gauge of pending approvals
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model:
            raise HTTPException(status_code=503, detail="World model not available")

        if not hasattr(world_model, "self_improvement_enabled"):
            return {"enabled": False, "message": "Self-improvement not available"}

        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, world_model.get_improvement_status)

        # Update Prometheus metrics if available
        try:
            from prometheus_client import Gauge
            improvement_approvals_pending = Gauge(
                "improvement_approvals_pending",
                "Pending self-improvement approvals"
            )
            if status.get("enabled") and "state" in status:
                state = status["state"]
                improvement_approvals_pending.set(len(state.get("pending_approvals", [])))
        except ImportError:
            pass

        return status

    except Exception as e:
        logger.error(f"Failed to get improvement status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/improvement/report-error", response_model=None)
async def report_error(request: Request) -> dict:
    """
    Report an error to trigger self-improvement analysis.
    
    Submits an error report to the self-improvement system which will
    analyze the error and potentially propose fixes or configuration changes.
    
    Args:
        request: FastAPI request with ErrorReportRequest body containing:
            - error_message: Description of the error
            - error_type: Type/category of error
            - severity: Error severity level
            - context: Optional context dict with additional info
    
    Returns:
        Dict containing:
            - status: "reported"
            - error_type: Type of error reported
            - severity: Severity level
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or world model not initialized
        HTTPException: 500 if report fails
    
    Note:
        This endpoint increments the error_counter Prometheus metric.
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model:
            raise HTTPException(status_code=503, detail="World model not available")

        # Get request body
        from vulcan.api.models import ErrorReportRequest
        body = await request.json()
        error_request = ErrorReportRequest(**body)

        # Create exception object from request
        error = Exception(error_request.error_message)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, world_model.report_error, error, error_request.context
        )

        # Update Prometheus metrics
        error_counter.labels(error_type=error_request.error_type).inc()

        logger.info(f"Error reported: {error_request.error_type} - {error_request.error_message}")

        return {
            "status": "reported",
            "error_type": error_request.error_type,
            "severity": error_request.severity,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/improvement/approve", response_model=None)
async def approve_improvement(request: Request) -> dict:
    """
    Approve or reject a pending improvement action.
    
    Processes approval/rejection for a pending improvement proposal.
    Approved improvements are executed, rejected ones are logged for analysis.
    
    Args:
        request: FastAPI request with ApprovalRequest body containing:
            - approval_id: ID of the pending approval
            - approved: Boolean - true to approve, false to reject
            - notes: Optional notes explaining the decision
    
    Returns:
        Dict containing:
            - status: "success"
            - approval_id: ID of processed approval
            - approved: Boolean indicating approval/rejection
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or self-improvement drive not available
        HTTPException: 404 if approval ID not found
        HTTPException: 500 if processing fails
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model or not hasattr(world_model, "self_improvement_drive"):
            raise HTTPException(
                status_code=503, detail="Self-improvement drive not available"
            )

        drive = world_model.self_improvement_drive

        # Get request body
        from vulcan.api.models import ApprovalRequest
        body = await request.json()
        approval_request = ApprovalRequest(**body)

        loop = asyncio.get_running_loop()

        if approval_request.approved:
            result = await loop.run_in_executor(
                None, drive.approve_pending, approval_request.approval_id
            )
        else:
            result = await loop.run_in_executor(
                None,
                drive.reject_pending,
                approval_request.approval_id,
                approval_request.notes or "Rejected via API",
            )

        if result:
            logger.info(
                f"Improvement {approval_request.approval_id} "
                f"{'approved' if approval_request.approved else 'rejected'}"
            )

            return {
                "status": "success",
                "approval_id": approval_request.approval_id,
                "approved": approval_request.approved,
                "timestamp": time.time(),
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Approval {approval_request.approval_id} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process approval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/improvement/pending", response_model=None)
async def get_pending_approvals(request: Request) -> dict:
    """
    Get list of pending improvement approvals.
    
    Returns all improvement proposals that are awaiting approval or rejection.
    Each pending approval includes details about the proposed change and
    estimated impact.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - pending_approvals: List of pending approval objects
            - count: Number of pending approvals
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or self-improvement drive not available
        HTTPException: 500 if retrieval fails
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model or not hasattr(world_model, "self_improvement_drive"):
            raise HTTPException(
                status_code=503, detail="Self-improvement drive not available"
            )

        drive = world_model.self_improvement_drive

        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, drive.get_status)

        pending = status.get("state", {}).get("pending_approvals", [])

        return {
            "pending_approvals": pending,
            "count": len(pending),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/improvement/update-metric", response_model=None)
async def update_performance_metric(request: Request) -> dict:
    """
    Update a performance metric.
    
    Reports a performance metric value to the self-improvement system.
    If the metric indicates degraded performance, triggers improvement analysis.
    
    Args:
        request: FastAPI request with query parameters:
            - metric: Name of the performance metric
            - value: Numeric value of the metric
    
    Returns:
        Dict containing:
            - status: "updated"
            - metric: Name of updated metric
            - value: New metric value
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if system or world model not initialized
        HTTPException: 500 if update fails
    
    Example:
        POST /v1/improvement/update-metric?metric=latency_p99&value=250.5
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        world_model = deployment.collective.deps.world_model

        if not world_model:
            raise HTTPException(status_code=503, detail="World model not available")

        # Get query parameters
        metric = request.query_params.get("metric")
        value_str = request.query_params.get("value")
        
        if not metric or not value_str:
            raise HTTPException(
                status_code=400,
                detail="Both 'metric' and 'value' query parameters required"
            )
        
        try:
            value = float(value_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value: '{value_str}' is not a number"
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, world_model.update_performance_metric, metric, value
        )

        return {
            "status": "updated",
            "metric": metric,
            "value": value,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update metric: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
