"""Service import utilities and health checking for the Vulcan unified platform.

Provides ServiceImportResult, import_service_async, and check_service_health_async
for robustly importing and monitoring sub-application services.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

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
