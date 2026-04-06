"""Service management for the Vulcan unified platform.

Provides AsyncServiceManager for managing the lifecycle of mounted sub-applications.
Import utilities and health checking are delegated to service_imports;
stop/start lifecycle operations are delegated to service_lifecycle.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from src.platform.service_imports import (
    ServiceImportResult,
    check_service_health_async,
    import_service_async,
)

logger = logging.getLogger("unified_platform")

# Optional dependencies with graceful degradation
try:
    from starlette.middleware.wsgi import WSGIMiddleware

    WSGI_AVAILABLE = True
except ImportError:
    WSGI_AVAILABLE = False


# Re-export for backward compatibility
__all__ = [
    "ServiceImportResult",
    "import_service_async",
    "check_service_health_async",
    "AsyncServiceManager",
]


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
        """Stop (unmount) a service. Delegates to service_lifecycle module."""
        from src.platform.service_lifecycle import stop_service
        return await stop_service(self, app, name, flash_manager)

    async def start_service(self, app: FastAPI, name: str, flash_manager=None) -> Dict[str, Any]:
        """Start (re-mount) a previously stopped service. Delegates to service_lifecycle module."""
        from src.platform.service_lifecycle import start_service
        return await start_service(self, app, name, flash_manager)

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
