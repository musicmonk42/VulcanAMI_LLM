"""Service management for the Vulcan unified platform.

Provides ServiceImportResult, import_service_async, check_service_health_async,
and AsyncServiceManager for managing the lifecycle of mounted sub-applications.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

logger = logging.getLogger("unified_platform")

# Optional dependencies with graceful degradation
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from prometheus_client import Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from starlette.middleware.wsgi import WSGIMiddleware

    WSGI_AVAILABLE = True
except ImportError:
    WSGI_AVAILABLE = False


# =============================================================================
# SERVICE IMPORT RESULT
# =============================================================================


@dataclass
class ServiceImportResult:
    """Result of attempting to import a service."""

    name: str
    success: bool
    app: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    import_path: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# =============================================================================
# ASYNC SERVICE IMPORT
# =============================================================================


async def import_service_async(
    name: str,
    module_path: str,
    attr_name: str,
    expected_type: Optional[str] = None,
    flash_manager=None,
    settings=None,
) -> ServiceImportResult:
    """
    Asynchronously and robustly import a service with detailed error reporting.
    Supports both relative and absolute imports (e.g., 'main' or 'src.vulcan.main').

    Args:
        name: Human-readable service name.
        module_path: Python module path to import from.
        attr_name: Attribute name to retrieve from the module.
        expected_type: Optional expected type name for validation.
        flash_manager: Optional FlashMessageManager for status messages.
        settings: Optional settings object; if None, will be imported lazily.
    """
    if settings is None:
        from src.platform.settings import get_settings
        settings = get_settings()

    # Logger might not be fully configured yet, use print for critical path
    print(f"Attempting to import {name} from {module_path}.{attr_name}")

    # Try multiple import strategies
    import_strategies = [module_path]

    # If auto-detect is enabled, try src.* variants
    if settings.auto_detect_src and "." not in module_path:
        import_strategies.extend(
            [
                f"src.{module_path}",
                f"src.vulcan.{module_path}" if name == "VULCAN" else None,
                f"src.arena.{module_path}" if name == "Arena" else None,
                f"src.registry.{module_path}" if name == "Registry" else None,
            ]
        )

    # Remove None entries
    import_strategies = [s for s in import_strategies if s]

    for strategy in import_strategies:
        try:
            # Try to import the module
            module = __import__(strategy, fromlist=[attr_name])
            print(f"  ✓ Module '{strategy}' imported successfully")

            # Check if attribute exists
            if not hasattr(module, attr_name):
                error = f"Module '{strategy}' has no attribute '{attr_name}'"
                print(f"  ✗ {error}")
                continue

            # Get the app object
            app_obj = getattr(module, attr_name)
            actual_type = type(app_obj).__name__
            print(f"  ✓ Found '{attr_name}' with type '{actual_type}'")

            # Validate type if expected
            if expected_type and actual_type != expected_type:
                print(f"  ⚠ Expected type '{expected_type}' but got '{actual_type}'")

            print(f"  ✅ Successfully imported {name} from {strategy}")
            if flash_manager:
                await flash_manager.add_message(
                    "success",
                    f"Successfully imported {name}",
                    f"Import path: {strategy}.{attr_name}",
                )

            return ServiceImportResult(
                name=name,
                success=True,
                app=app_obj,
                import_path=f"{strategy}.{attr_name}",
            )

        except ImportError as e:
            print(f"  ✗ ImportError for '{strategy}': {e}")
            continue
        except Exception as e:
            print(f"  ✗ {type(e).__name__} for '{strategy}': {e}")
            # Re-raise critical exceptions (like the SyntaxError)
            if isinstance(e, (SyntaxError, NameError, TypeError)):
                raise e
            continue

    # All strategies failed
    error = f"Cannot import {name} from any of: {import_strategies}"
    print(f"  ✗ {error}")

    if flash_manager:
        await flash_manager.add_message(
            "error", f"Failed to import {name}", f"Tried: {', '.join(import_strategies)}"
        )

    return ServiceImportResult(
        name=name, success=False, error=error, error_type="ImportError"
    )


# =============================================================================
# ASYNC SERVICE HEALTH CHECKS
# =============================================================================


async def check_service_health_async(
    service_name: str,
    base_url: str,
    path: str,
    health_check_timeout: float = 5.0,
    service_health_gauge=None,
) -> Dict[str, Any]:
    """
    Perform actual HTTP health check on a service asynchronously.

    Args:
        service_name: Name of the service for labeling.
        base_url: Base URL to check against.
        path: Health check endpoint path.
        health_check_timeout: Timeout in seconds.
        service_health_gauge: Optional Prometheus Gauge for metrics.
    """
    if not HTTPX_AVAILABLE:
        return {"status": "unknown", "message": "httpx not available for health checks"}

    try:
        async with httpx.AsyncClient(timeout=health_check_timeout) as client:
            start_time = datetime.utcnow()
            response = await client.get(f"{base_url}{path}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            is_healthy = response.status_code == 200

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and service_health_gauge:
                service_health_gauge.labels(service=service_name).set(1 if is_healthy else 0)

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "status_code": response.status_code,
                "response": response.json() if is_healthy else None,
                "latency_ms": latency,
            }
    except httpx.TimeoutException:
        if PROMETHEUS_AVAILABLE and service_health_gauge:
            service_health_gauge.labels(service=service_name).set(0)
        return {"status": "timeout", "message": "Health check timed out"}
    except Exception as e:
        if PROMETHEUS_AVAILABLE and service_health_gauge:
            service_health_gauge.labels(service=service_name).set(0)
        return {"status": "error", "message": str(e)}


# =============================================================================
# ASYNC SERVICE MANAGER
# =============================================================================


class AsyncServiceManager:
    """Fully async service lifecycle manager."""

    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.import_results: List[ServiceImportResult] = []
        self._lock = asyncio.Lock()

    async def register_service(
        self,
        name: str,
        import_result: ServiceImportResult,
        mount_path: str,
        health_path: str,
    ):
        """Register a service asynchronously."""
        async with self._lock:
            self.import_results.append(import_result)

            if import_result.success:
                self.services[name] = {
                    "name": name,
                    "mounted": False,
                    "mount_path": mount_path,
                    "health_path": health_path,
                    "app": import_result.app,
                    "import_success": True,
                    "import_path": import_result.import_path,
                    "docs_url": f"{mount_path}/docs",
                }
            else:
                self.services[name] = {
                    "name": name,
                    "mounted": False,
                    "import_success": False,
                    "error": import_result.error,
                    "error_type": import_result.error_type,
                }

    async def mount_service(self, app: FastAPI, name: str, use_wsgi: bool = False, flash_manager=None):
        """Mount a service to the main app asynchronously."""
        async with self._lock:
            service = self.services.get(name)
            if not service or not service["import_success"]:
                return False

            try:
                if use_wsgi and WSGI_AVAILABLE:
                    from starlette.middleware.wsgi import WSGIMiddleware
                    app.mount(service["mount_path"], WSGIMiddleware(service["app"]))
                else:
                    app.mount(service["mount_path"], service["app"])

                service["mounted"] = True
                logger.info(f"Mounted {name} at {service['mount_path']}")

                if flash_manager:
                    await flash_manager.add_message(
                        "success",
                        f"Mounted {name}",
                        f"Available at {service['mount_path']}",
                    )

                return True
            except Exception as e:
                logger.error(f"Failed to mount {name}: {e}")
                service["mount_error"] = str(e)

                if flash_manager:
                    await flash_manager.add_message(
                        "error", f"Failed to mount {name}", str(e)
                    )

                return False

    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services asynchronously."""
        async with self._lock:
            return {
                name: {
                    "mounted": service.get("mounted", False),
                    "mount_path": service.get("mount_path"),
                    "health_path": service.get("health_path"),
                    "import_path": service.get("import_path"),
                    "docs_url": (
                        service.get("docs_url") if service.get("mounted") else None
                    ),
                    "error": service.get("error"),
                }
                for name, service in self.services.items()
            }

    async def check_all_health(
        self, base_url: str, service_health_gauge=None
    ) -> Dict[str, Any]:
        """Check health of all mounted services asynchronously."""
        health_results = {}

        async with self._lock:
            services = dict(self.services)  # Copy to avoid holding lock

        # Check health in parallel
        tasks = []
        service_names = []

        for name, service in services.items():
            if service.get("mounted") and service.get("health_path"):
                tasks.append(
                    check_service_health_async(
                        name,
                        base_url,
                        service["health_path"],
                        service_health_gauge=service_health_gauge,
                    )
                )
                service_names.append(name)
            elif service.get("mounted"):
                health_results[name] = {
                    "status": "mounted",
                    "message": "No health endpoint",
                }
            else:
                health_results[name] = {"status": "not_mounted"}

        # Execute all health checks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(service_names, results):
                if isinstance(result, Exception):
                    health_results[name] = {"status": "error", "message": str(result)}
                else:
                    health_results[name] = result

        return health_results

    async def stop_service(self, app: FastAPI, name: str, flash_manager=None) -> Dict[str, Any]:
        """
        Stop (unmount) a service from the main app.

        This removes the service's route from the FastAPI application,
        effectively making it unavailable. The service configuration
        is preserved so it can be restarted later.

        Args:
            app: The FastAPI application instance
            name: Name of the service to stop
            flash_manager: Optional FlashMessageManager for status messages

        Returns:
            Dict with success status and details
        """
        async with self._lock:
            service = self.services.get(name)
            if not service:
                return {
                    "success": False,
                    "error": f"Service '{name}' not found",
                    "available_services": list(self.services.keys())
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

    async def start_service(self, app: FastAPI, name: str, flash_manager=None) -> Dict[str, Any]:
        """
        Start (re-mount) a previously stopped service.

        This re-mounts a service that was previously stopped, making it
        available again at its original mount path.

        Args:
            app: The FastAPI application instance
            name: Name of the service to start
            flash_manager: Optional FlashMessageManager for status messages

        Returns:
            Dict with success status and details
        """
        async with self._lock:
            service = self.services.get(name)
            if not service:
                return {
                    "success": False,
                    "error": f"Service '{name}' not found",
                    "available_services": list(self.services.keys())
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

    async def get_service_details(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific service.

        Args:
            name: Name of the service

        Returns:
            Dict with service details or None if not found
        """
        async with self._lock:
            service = self.services.get(name)
            if not service:
                return None

            return {
                "name": service.get("name"),
                "mounted": service.get("mounted", False),
                "mount_path": service.get("mount_path"),
                "health_path": service.get("health_path"),
                "import_path": service.get("import_path"),
                "import_success": service.get("import_success"),
                "docs_url": service.get("docs_url") if service.get("mounted") else None,
                "error": service.get("error"),
                "stopped_at": service.get("stopped_at"),
                "started_at": service.get("started_at"),
            }
