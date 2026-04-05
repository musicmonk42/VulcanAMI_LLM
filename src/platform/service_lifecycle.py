"""Service lifecycle operations (stop/start) for the Vulcan unified platform.

Provides standalone async functions for stopping and starting services,
designed to be called by AsyncServiceManager delegation methods.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI

logger = logging.getLogger("unified_platform")

# Optional dependency for WSGI middleware
try:
    from starlette.middleware.wsgi import WSGIMiddleware

    WSGI_AVAILABLE = True
except ImportError:
    WSGI_AVAILABLE = False


async def stop_service(
    manager, app: FastAPI, name: str, flash_manager=None
) -> Dict[str, Any]:
    """
    Stop (unmount) a service from the main app.

    This removes the service's route from the FastAPI application,
    effectively making it unavailable. The service configuration
    is preserved so it can be restarted later.

    Args:
        manager: The AsyncServiceManager instance
        app: The FastAPI application instance
        name: Name of the service to stop
        flash_manager: Optional FlashMessageManager for status messages

    Returns:
        Dict with success status and details
    """
    async with manager._lock:
        service = manager.services.get(name)
        if not service:
            return {
                "success": False,
                "error": f"Service '{name}' not found",
                "available_services": list(manager.services.keys())
            }

        if not service.get("mounted"):
            return {
                "success": False,
                "error": f"Service '{name}' is not currently mounted",
                "status": "already_stopped"
            }

        try:
            # Get the mount path for the service
            mount_path = service.get("mount_path")
            if not mount_path:
                return {
                    "success": False,
                    "error": f"Service '{name}' has no mount path configured"
                }

            # Find and remove the route from FastAPI app
            # FastAPI stores mounted apps in app.routes
            route_removed = False
            for i, route in enumerate(app.routes):
                if hasattr(route, 'path') and route.path == mount_path:
                    app.routes.pop(i)
                    route_removed = True
                    break

            if not route_removed:
                # Route not found, but mark as stopped anyway
                logger.warning(
                    f"Route for {name} at {mount_path} not found in app.routes, "
                    "but marking as stopped"
                )

            # Update service status
            service["mounted"] = False
            service["stopped_at"] = datetime.utcnow().isoformat()

            logger.info(f"Stopped service {name} (was mounted at {mount_path})")

            if flash_manager:
                await flash_manager.add_message(
                    "warning",
                    f"Stopped {name}",
                    f"Service at {mount_path} is now unavailable"
                )

            return {
                "success": True,
                "service": name,
                "mount_path": mount_path,
                "status": "stopped",
                "stopped_at": service["stopped_at"]
            }

        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


async def start_service(
    manager, app: FastAPI, name: str, flash_manager=None
) -> Dict[str, Any]:
    """
    Start (re-mount) a previously stopped service.

    This re-mounts a service that was previously stopped, making it
    available again at its original mount path.

    Args:
        manager: The AsyncServiceManager instance
        app: The FastAPI application instance
        name: Name of the service to start
        flash_manager: Optional FlashMessageManager for status messages

    Returns:
        Dict with success status and details
    """
    async with manager._lock:
        service = manager.services.get(name)
        if not service:
            return {
                "success": False,
                "error": f"Service '{name}' not found",
                "available_services": list(manager.services.keys())
            }

        if service.get("mounted"):
            return {
                "success": False,
                "error": f"Service '{name}' is already running",
                "status": "already_running"
            }

        if not service.get("import_success"):
            return {
                "success": False,
                "error": f"Service '{name}' failed to import and cannot be started",
                "import_error": service.get("error")
            }

        try:
            mount_path = service.get("mount_path")
            service_app = service.get("app")

            if not mount_path or not service_app:
                return {
                    "success": False,
                    "error": f"Service '{name}' is missing mount_path or app configuration"
                }

            # Mount the service back to the app
            # Check if it's a WSGI or ASGI app based on previous mount style
            use_wsgi = service.get("use_wsgi", False)
            if use_wsgi and WSGI_AVAILABLE:
                from starlette.middleware.wsgi import WSGIMiddleware
                app.mount(mount_path, WSGIMiddleware(service_app))
            else:
                app.mount(mount_path, service_app)

            # Update service status
            service["mounted"] = True
            service["started_at"] = datetime.utcnow().isoformat()
            if "stopped_at" in service:
                del service["stopped_at"]

            logger.info(f"Started service {name} at {mount_path}")

            if flash_manager:
                await flash_manager.add_message(
                    "success",
                    f"Started {name}",
                    f"Service now available at {mount_path}"
                )

            return {
                "success": True,
                "service": name,
                "mount_path": mount_path,
                "status": "running",
                "started_at": service["started_at"],
                "docs_url": service.get("docs_url")
            }

        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
