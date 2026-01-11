"""
Monitoring Endpoints

This module provides Prometheus metrics and system monitoring endpoints.

Endpoints:
    GET /metrics - Prometheus-compatible metrics endpoint
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """
    Prometheus metrics endpoint.
    
    Exports system metrics in Prometheus text format for scraping by
    monitoring systems. Includes counters, gauges, and histograms for:
    - Request rates and latencies
    - Error rates
    - Resource usage (memory, CPU)
    - Subsystem-specific metrics
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Plain text response with Prometheus metrics
    
    Raises:
        HTTPException: 404 if Prometheus metrics are disabled in settings
    
    Note:
        This endpoint is typically scraped by Prometheus at regular intervals.
        For security, consider restricting access in production environments.
    """
    try:
        # Import prometheus_client here to avoid hard dependency
        from prometheus_client import generate_latest
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Prometheus client not available"
        )
    
    # Check if metrics are enabled in settings
    app = request.app
    settings = getattr(app.state, "settings", None)
    prometheus_enabled = getattr(settings, "prometheus_enabled", True) if settings else True
    
    if not prometheus_enabled:
        raise HTTPException(
            status_code=404,
            detail="Metrics disabled"
        )

    return Response(
        generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )
