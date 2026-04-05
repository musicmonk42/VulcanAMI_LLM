"""
Platform middleware extracted from full_platform.py.

Provides:
- request_size_limit_middleware: Header-based request size limiting
- security_headers_middleware: Security headers applied globally
- log_requests: Request logging and Prometheus metrics
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def request_size_limit_middleware(request: Request, call_next, *, settings: Any):
    """
    Request size limiting middleware (header-based).

    Args:
        request: The incoming request.
        call_next: The next middleware/handler in the chain.
        settings: Platform settings object with ``max_request_size_bytes``.
    """
    try:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                cl = int(content_length)
                if cl > settings.max_request_size_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request too large",
                            "max_bytes": settings.max_request_size_bytes,
                        },
                    )
            except ValueError:
                # Invalid header; deny to be safe
                return JSONResponse(
                    status_code=400, content={"error": "Invalid Content-Length"}
                )
        # For chunked requests, rely on upstream reverse proxy limits
    except Exception as e:
        logger.error(f"Error checking request size: {e}", exc_info=True)
    return await call_next(request)


async def security_headers_middleware(request: Request, call_next):
    """
    Security headers middleware (applied globally).

    Adds the following headers to every response:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - Referrer-Policy: no-referrer
    - Content-Security-Policy (relaxed for chat interface CDN scripts)
    - Strict-Transport-Security
    - Cache-Control: no-store
    - Vary: Origin
    """
    response = await call_next(request)
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Relaxed CSP for chat interface - allows CDN scripts and inline styles
    # NOTE: 'unsafe-inline' and 'unsafe-eval' are required for:
    # - marked.js (Markdown rendering) which may use eval internally
    # - highlight.js (syntax highlighting) for code blocks
    # - Inline event handlers in the chat HTML
    # For production, consider moving to nonce-based CSP if security requirements increase
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https:"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Origin"
    return response


async def log_requests(
    request: Request,
    call_next,
    *,
    service_manager: Any,
    prometheus_available: bool = False,
    request_counter: Any = None,
    request_duration: Any = None,
):
    """
    Log all requests and track metrics.

    Args:
        request: The incoming request.
        call_next: The next middleware/handler in the chain.
        service_manager: The platform ServiceManager instance.
        prometheus_available: Whether Prometheus client is available.
        request_counter: Prometheus Counter for request counts.
        request_duration: Prometheus Histogram for request durations.
    """
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        f"{request.method} {request.url.path} [{response.status_code}] {duration:.3f}s"
    )

    if prometheus_available and request_counter and request_duration:
        service = "platform"
        service_status = await service_manager.get_service_status()
        for name, svc in service_status.items():
            if svc.get("mount_path") and request.url.path.startswith(svc["mount_path"]):
                service = name
                break

        request_counter.labels(
            service=service, method=request.method, endpoint=request.url.path
        ).inc()

        request_duration.labels(service=service, endpoint=request.url.path).observe(
            duration
        )

    return response
