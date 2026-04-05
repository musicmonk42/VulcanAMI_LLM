"""Authentication system for the Vulcan unified platform.

Provides AuthMethod enum, JWT authentication, API key verification,
and the unified verify_authentication dependency.
"""

import hmac
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger("unified_platform")

# Optional JWT dependency with graceful degradation
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception  # Fallback type for except clauses


# =============================================================================
# AUTH METHOD ENUM
# =============================================================================


class AuthMethod(str, Enum):
    """Supported authentication methods."""

    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"


# =============================================================================
# AUTHENTICATION ERROR
# =============================================================================


class AuthenticationError(HTTPException):
    """Custom authentication error."""

    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# CONSTANT-TIME COMPARISON
# =============================================================================


def _safe_compare(a: Optional[str], b: Optional[str]) -> bool:
    """Constant-time string comparison that handles None safely."""
    if a is None or b is None:
        return False
    try:
        # Ensure both are str
        if not isinstance(a, str):
            a = str(a)
        if not isinstance(b, str):
            b = str(b)
        return hmac.compare_digest(a, b)
    except Exception:
        return False


# =============================================================================
# JWT AUTHENTICATION
# =============================================================================


class JWTAuth:
    """JWT authentication handler."""

    @staticmethod
    def create_access_token(data: dict, settings) -> str:
        """Create JWT access token.

        Args:
            data: Token payload data.
            settings: UnifiedPlatformSettings instance with jwt_secret,
                      jwt_expire_minutes, and jwt_algorithm.
        """
        if not JWT_AVAILABLE or not settings.jwt_secret:
            raise ValueError("JWT not available or secret not configured")

        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(
            to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm
        )
        return encoded_jwt

    @staticmethod
    def verify_token(token: str, settings) -> Dict[str, Any]:
        """Verify JWT token and return payload.

        Args:
            token: The JWT token string.
            settings: UnifiedPlatformSettings instance with jwt_secret
                      and jwt_algorithm.
        """
        if not JWT_AVAILABLE or not settings.jwt_secret:
            raise AuthenticationError("JWT not configured")

        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {e}")


# =============================================================================
# SECURITY SCHEMES
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# UNIFIED AUTHENTICATION VERIFICATION
# =============================================================================


async def verify_authentication(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Dict[str, Any]:
    """
    Unified authentication verification.
    Supports API Key, JWT, and OAuth2.
    Security-hardening:
    - Constant-time comparison for API keys
    - Strict checks depending on configured method

    Note: This function imports settings lazily to avoid circular imports.
    """
    # Lazy import to avoid circular dependency with settings module
    from src.platform.settings import get_settings

    settings = get_settings()

    if settings.auth_method == AuthMethod.NONE:
        from src.env_utils import is_dev_env
        if not is_dev_env():
            raise AuthenticationError("Authentication not configured")
        return {"authenticated": True, "method": "none", "warning": "dev-mode-no-auth"}

    # API Key authentication (strict)
    if settings.auth_method == AuthMethod.API_KEY:
        configured_key = settings.api_key
        if not configured_key:
            raise AuthenticationError("API key not configured")
        # Prefer X-API-Key header; optionally allow Bearer to carry API key for compatibility
        presented_key = api_key or (bearer.credentials if bearer else None)
        if _safe_compare(presented_key, configured_key):
            return {"authenticated": True, "method": "api_key"}
        raise AuthenticationError("Invalid API key")

    # JWT authentication
    if settings.auth_method == AuthMethod.JWT:
        if not bearer or not bearer.credentials:
            raise AuthenticationError("Bearer token required")
        payload = JWTAuth.verify_token(bearer.credentials, settings)
        return {
            "authenticated": True,
            "method": "jwt",
            "user": payload.get("sub"),
            "payload": payload,
        }

    # OAuth2 (placeholder - extend as needed)
    if settings.auth_method == AuthMethod.OAUTH2:
        # Implement OAuth2 flow here
        raise AuthenticationError("OAuth2 not yet implemented")

    raise AuthenticationError("Unknown authentication method")
