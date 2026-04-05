"""
Authentication route handlers extracted from full_platform.py.

Provides:
- get_token (POST /auth/token)
- protected_endpoint (GET /api/protected)
"""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.platform.auth import verify_authentication

router = APIRouter()


@router.post("/auth/token", response_model=None)
async def get_token(request: Request, sub: Optional[str] = None):
    """
    Secure token issuance endpoint.
    Requires:
    - AuthMethod is JWT
    - Valid API key via X-API-Key
    """
    from src.platform.auth import (
        JWT_AVAILABLE,
        AuthMethod,
        AuthenticationError,
        JWTAuth,
        _safe_compare,
    )
    from src.platform.globals import get_settings
    settings = get_settings()

    if not JWT_AVAILABLE:
        raise HTTPException(status_code=501, detail="JWT not available")

    if settings.auth_method != AuthMethod.JWT:
        raise HTTPException(
            status_code=400,
            detail="Token issuance available only when AuthMethod is JWT",
        )

    if not settings.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")

    configured_key = settings.api_key or ""
    presented_key = request.headers.get("X-API-Key")
    if not _safe_compare(presented_key, configured_key):
        raise AuthenticationError("Invalid API key for token issuance")

    subject = (sub or "api_key").strip()
    if not subject or len(subject) > 128:
        subject = "api_key"

    token = JWTAuth.create_access_token({"sub": subject}, settings)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.jwt_expire_minutes * 60,
    }


@router.get("/api/protected", response_model=None)
async def protected_endpoint(auth: Dict = Depends(verify_authentication)):
    """Example protected endpoint."""
    return {"message": "Access granted!", "auth": auth}
