"""
Health Check Endpoints

This module provides health, liveness, and readiness endpoints for monitoring
and orchestration systems (Kubernetes, Docker, etc.).

Endpoints:
    GET /health        - Comprehensive health check with subsystem status
    GET /health/live   - Lightweight liveness probe (< 100ms)
    GET /health/ready  - Fast readiness probe for essential components
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Performs deep health checks across all major subsystems including:
    - Deployment initialization status
    - Error rate thresholds
    - Memory usage limits
    - Latency requirements
    - LLM availability
    - Self-improvement drive status (if enabled)
    
    Returns:
        Dict containing:
            - status: "healthy" or "unhealthy"
            - checks: Dict of individual health check results
            - details: Full system status from deployment
            - timestamp: Current Unix timestamp
    
    Raises:
        Returns unhealthy status with error details if checks fail
    """
    try:
        app = request.app
        
        if not hasattr(app.state, "deployment"):
            return {
                "status": "unhealthy",
                "error": "Deployment not initialized",
                "timestamp": time.time(),
            }

        deployment = app.state.deployment
        status = deployment.get_status()
        
        # Get settings from app state or use defaults
        settings = getattr(app.state, "settings", None)
        max_memory_mb = getattr(settings, "max_memory_mb", 2000) if settings else 2000

        health_checks = {
            "error_rate": status["health"].get("error_rate", 0) < 0.1,
            "energy_budget": status["health"].get("energy_budget_left_nJ", 0) > 1000,
            "memory_usage": status["health"].get("memory_usage_mb", 0) < max_memory_mb * 0.9,
            "latency": status["health"].get("latency_ms", 0) < 1000,
        }

        # Add self-improvement health check if available
        try:
            world_model = deployment.collective.deps.world_model
            if (
                world_model
                and hasattr(world_model, "self_improvement_enabled")
                and world_model.self_improvement_enabled
            ):
                health_checks["self_improvement"] = hasattr(
                    world_model, "improvement_running"
                )
        except Exception as e:
            logger.debug(f"Failed to check self-improvement status: {e}")

        # Add LLM availability check
        health_checks["llm_available"] = hasattr(app.state, "llm")

        healthy = all(health_checks.values())

        return {
            "status": "healthy" if healthy else "unhealthy",
            "checks": health_checks,
            "details": status,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Lightweight liveness check endpoint.
    
    Provides a fast (< 100ms) liveness check suitable for Kubernetes
    and Docker health probes. This endpoint only verifies the application
    process is running and can respond to requests.
    
    FIX Issue #41: The main /health endpoint was taking 5+ seconds due to
    collecting comprehensive status from multiple subsystems. This endpoint
    provides instant response for liveness detection.
    
    Returns:
        Dict containing:
            - status: Always "ok" if endpoint responds
            - timestamp: Current Unix timestamp
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> Dict[str, Any]:
    """
    Fast readiness check endpoint.
    
    Validates essential components are initialized without collecting
    full system metrics. Suitable for Kubernetes readiness probes.
    
    FIX Issue #41: Provides a quick readiness check that only validates
    essential components without the overhead of comprehensive health checks.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "ready" or "not_ready"
            - deployment_initialized: Boolean
            - llm_initialized: Boolean
            - timestamp: Current Unix timestamp
    
    Note:
        Returns 200 OK even if not ready (check status field).
        This allows orchestrators to distinguish between
        process failure and initialization delay.
    """
    try:
        app = request.app
        has_deployment = hasattr(app.state, "deployment")
        has_llm = hasattr(app.state, "llm")
        
        ready = has_deployment and has_llm
        
        return {
            "status": "ready" if ready else "not_ready",
            "deployment_initialized": has_deployment,
            "llm_initialized": has_llm,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": time.time(),
        }
