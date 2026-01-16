"""
API Middleware

This module provides FastAPI middleware for authentication, rate limiting,
and security headers.

Middleware:
    validate_api_key   - API key authentication
    rate_limiting      - In-process rate limiting
    security_headers   - Security header injection
"""

import hashlib
import hmac
import logging
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


async def validate_api_key_middleware(
    request: Request,
    call_next: Callable,
    settings,
    auth_failures_counter=None
) -> Response:
    """
    API key validation middleware.
    
    Validates API keys for all routes except public endpoints. Supports
    multiple authentication headers for compatibility:
    - X-API-Key (recommended)
    - X-API-KEY (alternative)
    - Authorization: Bearer <key> (OAuth-style)
    
    Public routes are identified by suffix matching to support mounting
    under different paths (e.g., /vulcan/).
    
    Args:
        request: FastAPI request object
        call_next: Next middleware or route handler
        settings: Application settings with api_key configuration
        auth_failures_counter: Optional Prometheus counter for tracking failures
    
    Returns:
        Response from next middleware/handler or 401 error response
    
    Note:
        If no API key is configured in settings, validation is skipped.
        This allows for easy development/testing without authentication.
    """
    # Define public routes that don't require authentication
    public_suffixes = (
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json"
    )
    
    path = request.url.path or ""
    
    # Allow access to public routes
    if any(path.endswith(suffix) for suffix in public_suffixes):
        return await call_next(request)

    # Skip validation if no API key is configured
    if not settings.api_key:
        return await call_next(request)

    # Extract API key from various header formats
    headers = request.headers
    provided_key = (
        headers.get("X-API-Key")
        or headers.get("X-API-KEY")
        or headers.get("x-api-key")
        or (
            headers.get("Authorization", "")[7:]  # Remove "Bearer " prefix
            if headers.get("Authorization", "").startswith("Bearer ")
            else None
        )
    )

    # Validate API key using constant-time comparison
    if not provided_key or not hmac.compare_digest(provided_key, settings.api_key):
        # Increment failure counter if available (Prometheus metric)
        if auth_failures_counter is not None:
            try:
                auth_failures_counter.inc()
            except Exception as e:
                logger.debug(f"Failed to increment auth failures counter: {e}")
        
        client_host = getattr(request.client, "host", "unknown") if request.client else "unknown"
        logger.warning(
            f"Invalid or missing API key from {client_host}. "
            f"Path: {path}"
        )
        
        return JSONResponse(
            status_code=401,
            content={
                "error": "Invalid or missing API key",
                "accepted_headers": [
                    "X-API-Key",
                    "X-API-KEY",
                    "Authorization: Bearer <key>",
                ],
                "how_to_fix": "Send one of the accepted headers with the configured API key.",
            },
        )

    return await call_next(request)


async def rate_limiting_middleware(
    request: Request,
    call_next: Callable,
    settings,
    rate_limit_storage: dict,
    rate_limit_lock
) -> Response:
    """
    Simple in-process rate limiting middleware.
    
    Implements a sliding window rate limiter that tracks requests per client.
    Client identification uses:
    1. API key hash (if provided) - more accurate for authenticated requests
    2. Client IP address (fallback) - for unauthenticated requests
    
    Public routes are exempt from rate limiting.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware or route handler
        settings: Application settings with rate limiting configuration
        rate_limit_storage: Dict mapping client IDs to request timestamps
        rate_limit_lock: Threading lock for storage access
    
    Returns:
        Response from next middleware/handler or 429 error response
    
    Note:
        This is an in-process implementation. For distributed deployments,
        consider using Redis-based rate limiting.
    """
    # Skip if rate limiting is disabled
    if not settings.rate_limit_enabled:
        return await call_next(request)

    # Define routes exempt from rate limiting
    public_suffixes = ("/", "/health", "/metrics")
    path = request.url.path or ""
    
    if any(path.endswith(suffix) for suffix in public_suffixes):
        return await call_next(request)

    # Determine client identifier
    client_id = request.client.host if request.client else "unknown"

    # If API key is provided, use its hash for more accurate tracking
    if settings.api_key:
        api_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-API-KEY")
            or (
                request.headers.get("Authorization", "")[7:]
                if request.headers.get("Authorization", "").startswith("Bearer ")
                else None
            )
        )
        if api_key:
            # Use hash of API key as client ID (privacy-preserving)
            client_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    current_time = time.time()
    window_start = current_time - settings.rate_limit_window_seconds

    # Thread-safe access to rate limit storage
    with rate_limit_lock:
        # Get or create bucket for this client
        bucket = rate_limit_storage.setdefault(client_id, [])
        
        # Remove timestamps outside the current window (sliding window)
        rate_limit_storage[client_id] = [t for t in bucket if t > window_start]

        # Check if limit exceeded
        if len(rate_limit_storage[client_id]) >= settings.rate_limit_requests:
            logger.warning(
                f"Rate limit exceeded for client {client_id}: "
                f"{len(rate_limit_storage[client_id])} requests in "
                f"{settings.rate_limit_window_seconds}s window"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": settings.rate_limit_window_seconds,
                    "limit": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window_seconds,
                },
            )

        # Add current request timestamp
        rate_limit_storage[client_id].append(current_time)

    return await call_next(request)


async def security_headers_middleware(
    request: Request,
    call_next: Callable
) -> Response:
    """
    Add security headers to all responses.
    
    Implements defense-in-depth security with strict CSP.
    """
    response = await call_next(request)

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Enable XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Enforce HTTPS
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    
    # Strict Content Security Policy
    # NOTE: Removed 'unsafe-eval' - if marked.js/highlight.js break,
    # consider using DOMPurify for markdown or a Web Worker for syntax highlighting
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    
    # Additional security headers
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()"
    )

    return response
