"""
Admin service management route handlers extracted from full_platform.py.

Provides:
- _verify_admin_access (utility function)
- admin_list_services (GET /admin/services)
- admin_get_service (GET /admin/services/{service_name})
- admin_stop_service (POST /admin/services/{service_name}/stop)
- admin_start_service (POST /admin/services/{service_name}/start)
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()
logger = logging.getLogger(__name__)


def _verify_admin_access(
    api_key: Optional[str],
    credentials: Optional[HTTPAuthorizationCredentials]
) -> bool:
    """
    Verify if the request has valid admin access credentials.

    Checks both API key and JWT token authentication methods.

    Args:
        api_key: API key from X-API-Key header
        credentials: JWT credentials from Authorization header

    Returns:
        True if authenticated, False otherwise
    """
    from src.platform.auth import JWT_AVAILABLE, _safe_compare
    from src.platform.globals import get_settings
    settings = get_settings()

    # Check API key authentication
    configured_key = settings.api_key
    if configured_key and api_key:
        if _safe_compare(api_key, configured_key):
            return True

    # Check JWT authentication
    if JWT_AVAILABLE and credentials and settings.jwt_secret:
        try:
            from jose import jwt, JWTError

            token = credentials.credentials
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=["HS256"]
            )
            # Token is valid - allow access
            # Optionally, check for admin role/scope in payload
            return True
        except Exception:
            pass

    return False


@router.get("/admin/services", response_model=None)
async def admin_list_services(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """
    List all registered services with their current status.

    Requires authentication via API key (X-API-Key header) or JWT token.

    Returns:
        List of all services with their status, mount paths, and health info.
    """
    from src.platform.globals import get_service_manager
    service_manager = get_service_manager()

    # Verify authentication
    api_key = request.headers.get("X-API-Key")
    if not _verify_admin_access(api_key, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide X-API-Key or Bearer token."
        )

    services = await service_manager.get_service_status()
    return {
        "services": services,
        "total_count": len(services),
        "mounted_count": sum(1 for s in services.values() if s.get("mounted")),
        "stopped_count": sum(1 for s in services.values() if not s.get("mounted")),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/admin/services/{service_name}", response_model=None)
async def admin_get_service(
    service_name: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """
    Get detailed information about a specific service.

    Args:
        service_name: Name of the service to query

    Returns:
        Detailed service information including status, paths, and timestamps.
    """
    from src.platform.globals import get_service_manager
    service_manager = get_service_manager()

    # Verify authentication
    api_key = request.headers.get("X-API-Key")
    if not _verify_admin_access(api_key, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide X-API-Key or Bearer token."
        )

    service_details = await service_manager.get_service_details(service_name)
    if not service_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found"
        )

    return {
        "service": service_details,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/admin/services/{service_name}/stop", response_model=None)
async def admin_stop_service(
    service_name: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """
    Stop (unmount) a specific service.

    This endpoint removes the service's routes from the application,
    making it unavailable until it is started again. The service
    configuration is preserved.

    Args:
        service_name: Name of the service to stop

    Returns:
        Result of the stop operation with status details.
    """
    from src.platform.globals import get_app, get_service_manager
    app = get_app()
    service_manager = get_service_manager()

    # Verify authentication
    api_key = request.headers.get("X-API-Key")
    if not _verify_admin_access(api_key, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide X-API-Key or Bearer token."
        )

    result = await service_manager.stop_service(app, service_name)

    if not result.get("success"):
        status_code = status.HTTP_404_NOT_FOUND if "not found" in result.get("error", "").lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(
            status_code=status_code,
            detail=result.get("error", "Failed to stop service")
        )

    logger.info(f"Admin stopped service: {service_name}")
    return result


@router.post("/admin/services/{service_name}/start", response_model=None)
async def admin_start_service(
    service_name: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """
    Start (re-mount) a previously stopped service.

    This endpoint re-mounts a service that was previously stopped,
    making it available again at its original mount path.

    Args:
        service_name: Name of the service to start

    Returns:
        Result of the start operation with status details.
    """
    from src.platform.globals import get_app, get_service_manager
    app = get_app()
    service_manager = get_service_manager()

    # Verify authentication
    api_key = request.headers.get("X-API-Key")
    if not _verify_admin_access(api_key, credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Provide X-API-Key or Bearer token."
        )

    result = await service_manager.start_service(app, service_name)

    if not result.get("success"):
        status_code = status.HTTP_404_NOT_FOUND if "not found" in result.get("error", "").lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(
            status_code=status_code,
            detail=result.get("error", "Failed to start service")
        )

    logger.info(f"Admin started service: {service_name}")
    return result
