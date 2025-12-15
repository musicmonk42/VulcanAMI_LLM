#!/usr/bin/env python3
# ============================================================
# Graphix + Vulcan Unified Platform v2.1 (PRODUCTION-HARDENED)
# ============================================================
# Enterprise-grade unified server with additional production enhancements:
# - Absolute import resolution (src.vulcan.main support)
# - Flash messaging system for errors/warnings
# - Multi-worker awareness and warnings
# - Extended auth (JWT, OAuth2 ready)
# - Secrets management integration
# - Fully async ServiceManager
# - Enhanced documentation aggregation
# - Security hardening: stricter CORS, auth defaults, token issuance, headers,
#   constant-time secret comparisons, request size limiting
# ============================================================

# NOTE: Subprocess management now uses subprocess.Popen instead of asyncio.create_subprocess_exec
# This avoids issues with Windows event loop policy when using uvicorn --reload

# Now proceed with all imports
import argparse
import asyncio
import hmac
import importlib
import json  # For Arena API endpoints
import logging
import os
import subprocess  # For background process management
import sys
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.wsgi import WSGIMiddleware

# NOTE: Windows event loop policy is now set at the very top of the file (line 17)
# before any imports that might trigger asyncio initialization.
# 
# NOTE: The encoding fix and .env loader have been MOVED
# into the lifespan function to ensure they run in the
# Uvicorn worker process.


# Optional dependencies with graceful degradation
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  httpx not available - health checks will be limited")

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    print("⚠️  prometheus-client not available - metrics disabled")

try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("⚠️  python-jose not available - JWT auth disabled")

# =============================================================================

# Arena components (for integrated API endpoints)
try:
    # Import dynamically to avoid issues
    arena_module_available = True
except ImportError:
    arena_module_available = False
    print("⚠️  Arena components not available - Arena API endpoints disabled")

# SECRETS MANAGEMENT
# =============================================================================


class SecretsManager:
    """
    Unified secrets management supporting multiple backends.
    Supports: environment variables, .env files, AWS Secrets Manager, etc.
    """

    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment with fallback support."""
        # Try direct env var
        value = os.environ.get(key)
        if value:
            return value

        # Try with UNIFIED_ prefix
        value = os.environ.get(f"UNIFIED_{key}")
        if value:
            return value

        # Try AWS Secrets Manager (if boto3 available)
        try:
            pass

            import boto3

            secrets_client = boto3.client("secretsmanager")
            secret_value = secrets_client.get_secret_value(SecretId=key)

            if "SecretString" in secret_value:
                return secret_value["SecretString"]
        except Exception as e:
            logger.debug(f"Failed to initialize platform component: {e}")
        return default


secrets = SecretsManager()

# =============================================================================
# CONFIGURATION
# =============================================================================


class AuthMethod(str, Enum):
    """Supported authentication methods."""

    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"


class UnifiedPlatformSettings(BaseSettings):
    """Centralized configuration with secrets support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="UNIFIED_", case_sensitive=False, extra="ignore"
    )

    # Server configuration
    # Security: Default to localhost binding for safety
    # Set UNIFIED_HOST=0.0.0.0 in environment to bind to all interfaces
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1  # Default to 1 for safety
    reload: bool = False

    # Service mount paths (configurable!)
    vulcan_mount: str = "/vulcan"
    arena_mount: str = "/arena"
    registry_mount: str = "/registry"
    api_gateway_mount: str = "/api-gateway"
    dqs_mount: str = "/dqs"
    pii_mount: str = "/pii"

    # Service import paths (support absolute imports like src.vulcan.main)
    vulcan_module: str = "src.vulcan.main"
    vulcan_attr: str = "app"
    arena_module: str = "src.graphix_arena"
    arena_attr: str = "app"
    registry_module: str = "app"
    registry_attr: str = "app"
    api_gateway_module: str = "src.api_gateway"
    api_gateway_attr: str = "app"
    dqs_module: str = "src.dqs_service"
    dqs_attr: str = "app"
    pii_module: str = "src.pii_service"
    pii_attr: str = "app"

    # Standalone service ports (for services that can't be mounted as sub-apps)
    api_server_port: int = 8001
    registry_grpc_port: int = 50051
    listener_port: int = 8084

    # Enable/disable individual services
    enable_api_gateway: bool = True
    enable_dqs_service: bool = True
    enable_pii_service: bool = True
    enable_api_server: bool = True
    enable_registry_grpc: bool = True
    enable_listener: bool = True

    # Auto-detect src structure
    auto_detect_src: bool = True

    # Authentication
    # Default will be auto-selected based on configured secrets in __init__
    auth_method: AuthMethod = AuthMethod.NONE
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None

    # CORS
    cors_enabled: bool = True
    # Tightened default origins; wildcard is no longer the default for security
    # Note: "null" origin is needed for file:// URLs (local HTML files)
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ]

    # Health checks
    enable_health_checks: bool = True
    health_check_timeout: float = 5.0

    # Monitoring
    enable_metrics: bool = True
    metrics_path: str = "/metrics"

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Flash messaging
    flash_message_max: int = 10  # Max recent messages to show

    # Multi-worker awareness
    warn_on_multi_worker: bool = True

    # Request limits
    max_request_size_bytes: int = 10 * 1024 * 1024  # 10 MiB

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load secrets from SecretsManager
        if not self.api_key:
            self.api_key = secrets.get_secret("API_KEY")
        if not self.jwt_secret:
            self.jwt_secret = secrets.get_secret("JWT_SECRET")

        # Sanitize CORS origins: drop empty strings and whitespace-only entries
        if isinstance(self.cors_origins, list):
            self.cors_origins = [
                o.strip() for o in self.cors_origins if isinstance(o, str) and o.strip()
            ]
            # Remove '*' from defaults to enforce explicit allowlist unless user sets it intentionally
            self.cors_origins = [o for o in self.cors_origins if o != "*"]

        # Auto-select auth method if not explicitly set: prefer JWT if configured, else API key, else NONE
        if self.auth_method == AuthMethod.NONE:
            if self.jwt_secret:
                self.auth_method = AuthMethod.JWT
            elif self.api_key:
                self.auth_method = AuthMethod.API_KEY
            else:
                self.auth_method = AuthMethod.NONE


settings = UnifiedPlatformSettings()

# =============================================================================
# FLASH MESSAGING SYSTEM
# =============================================================================


class FlashMessage:
    """Flash message for displaying errors/warnings."""

    def __init__(self, level: str, message: str, details: Optional[str] = None):
        self.level = level  # error, warning, info, success
        self.message = message
        self.details = details
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class FlashMessageManager:
    """Thread-safe flash message manager."""

    def __init__(self, max_messages: int = 10):
        self.messages = deque(maxlen=max_messages)
        self._lock = asyncio.Lock()

    async def add_message(
        self, level: str, message: str, details: Optional[str] = None
    ):
        """Add a new flash message."""
        async with self._lock:
            self.messages.append(FlashMessage(level, message, details))

    async def get_recent_messages(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent flash messages."""
        async with self._lock:
            return [msg.to_dict() for msg in list(self.messages)[-limit:]]

    async def clear_messages(self):
        """Clear all messages."""
        async with self._lock:
            self.messages.clear()


flash_manager = FlashMessageManager(max_messages=settings.flash_message_max)

# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_unified_logging():
    """Configure unified logging for all services with UTF-8 support."""
    # Clear existing handlers to avoid duplicates from the watcher process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create handlers with explicit UTF-8 encoding
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Ensure stdout is UTF-8 on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to cleanup platform resource: {e}")

    # File handler with explicit UTF-8 encoding
    file_handler = logging.FileHandler("unified_platform.log", encoding="utf-8")

    # Configure logging with UTF-8 handlers
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[stdout_handler, file_handler],
        force=True,  # Force re-configuration
    )

    # Set log levels for sub-apps
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("vulcan").setLevel(logging.INFO)
    logging.getLogger("arena").setLevel(logging.INFO)
    logging.getLogger("registry").setLevel(logging.INFO)


# NOTE: setup_unified_logging() is NO LONGER called here.
# It is now called inside the lifespan() function.
logger = logging.getLogger("unified_platform")

# Multi-worker warning (this will still log from the watcher process)
if settings.workers > 1 and settings.warn_on_multi_worker:
    # We must configure logging here *just for this warning*
    # It will be re-configured by the worker later.
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.warning("=" * 70)
    logger.warning("⚠️  MULTI-WORKER MODE DETECTED")
    logger.warning("=" * 70)
    logger.warning("Running with workers > 1 has implications:")
    logger.warning("  • In-memory state is NOT shared between workers")
    logger.warning("  • Lifespan events run once per worker")
    logger.warning("  • Flash messages are per-worker")
    logger.warning("  • For shared state, use Redis, PostgreSQL, etc.")
    logger.warning("  • Health checks may show inconsistent results")
    logger.warning("")
    logger.warning("Recommendation:")
    logger.warning("  • Use workers=1 for development")
    logger.warning("  • Use external stores (Redis) for production multi-worker")
    logger.warning("=" * 70)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Initialize metrics only once to avoid duplicate registration errors
request_counter = None
request_duration = None
service_health = None
active_workers = None

if PROMETHEUS_AVAILABLE:
    try:
        # Try to create metrics - if they already exist, prometheus will raise an error
        # which we'll catch and skip creation
        request_counter = Counter(
            "unified_platform_requests_total",
            "Total requests to unified platform",
            ["service", "method", "endpoint"],
        )
        request_duration = Histogram(
            "unified_platform_request_duration_seconds",
            "Request duration",
            ["service", "endpoint"],
        )
        service_health = Gauge(
            "unified_platform_service_health",
            "Service health status (1=healthy, 0=unhealthy)",
            ["service"],
        )
        active_workers = Gauge(
            "unified_platform_active_workers", "Number of active workers"
        )

        # Set worker count
        active_workers.set(settings.workers)
    except (ValueError, Exception) as e:
        # Metrics already registered (module reimported) or other error
        # This is not critical - metrics will just not be collected
        if "Duplicated timeseries" in str(e):
            print("⚠️  Prometheus metrics already registered, skipping creation")
        else:
            print(f"⚠️  Failed to initialize Prometheus metrics: {e}")
        request_counter = None
        request_duration = None
        service_health = None
        active_workers = None
        request_duration = None
        service_health = None
        active_workers = None

# =============================================================================
# PATH SETUP WITH ABSOLUTE IMPORT SUPPORT
# =============================================================================


def setup_python_path():
    """
    Ensure proper Python path for imports.
    Supports both flat and src/ directory structures.
    """
    script_dir = Path(__file__).resolve().parent

    # Add current directory
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Add src directory if it exists
    src_dir = script_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        # Use print here as logger isn't configured yet
        print(f"✓ Added src directory to path: {src_dir}")

    # Add parent directory
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    print(f"Debug: Python path configured: {sys.path[:3]}...")


setup_python_path()

# =============================================================================
# ENHANCED AUTHENTICATION SYSTEM
# =============================================================================


class AuthenticationError(HTTPException):
    """Custom authentication error."""

    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


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


class JWTAuth:
    """JWT authentication handler."""

    @staticmethod
    def create_access_token(data: dict) -> str:
        """Create JWT access token."""
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
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        if not JWT_AVAILABLE or not settings.jwt_secret:
            raise AuthenticationError("JWT not configured")

        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {e}")


# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


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
    """
    if settings.auth_method == AuthMethod.NONE:
        return {"authenticated": True, "method": "none"}

    # API Key authentication (strict)
    if settings.auth_method == AuthMethod.API_KEY:
        configured_key = settings.api_key or ""
        # Prefer X-API-Key header; optionally allow Bearer to carry API key for compatibility
        presented_key = api_key or (bearer.credentials if bearer else None)
        if _safe_compare(presented_key, configured_key):
            return {"authenticated": True, "method": "api_key"}
        raise AuthenticationError("Invalid API key")

    # JWT authentication
    if settings.auth_method == AuthMethod.JWT:
        if not bearer or not bearer.credentials:
            raise AuthenticationError("Bearer token required")
        payload = JWTAuth.verify_token(bearer.credentials)
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


# =============================================================================
# ROBUST IMPORT SYSTEM WITH ABSOLUTE PATH SUPPORT
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


async def import_service_async(
    name: str, module_path: str, attr_name: str, expected_type: Optional[str] = None
) -> ServiceImportResult:
    """
    Asynchronously and robustly import a service with detailed error reporting.
    Supports both relative and absolute imports (e.g., 'main' or 'src.vulcan.main').
    """
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
    service_name: str, base_url: str, path: str
) -> Dict[str, Any]:
    """
    Perform actual HTTP health check on a service asynchronously.
    """
    if not HTTPX_AVAILABLE:
        return {"status": "unknown", "message": "httpx not available for health checks"}

    try:
        async with httpx.AsyncClient(timeout=settings.health_check_timeout) as client:
            start_time = datetime.utcnow()
            response = await client.get(f"{base_url}{path}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            is_healthy = response.status_code == 200

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and service_health:
                service_health.labels(service=service_name).set(1 if is_healthy else 0)

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "status_code": response.status_code,
                "response": response.json() if is_healthy else None,
                "latency_ms": latency,
            }
    except httpx.TimeoutException:
        if PROMETHEUS_AVAILABLE and service_health:
            service_health.labels(service=service_name).set(0)
        return {"status": "timeout", "message": "Health check timed out"}
    except Exception as e:
        if PROMETHEUS_AVAILABLE and service_health:
            service_health.labels(service=service_name).set(0)
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

    async def mount_service(self, app: FastAPI, name: str, use_wsgi: bool = False):
        """Mount a service to the main app asynchronously."""
        async with self._lock:
            service = self.services.get(name)
            if not service or not service["import_success"]:
                return False

            try:
                if use_wsgi:
                    app.mount(service["mount_path"], WSGIMiddleware(service["app"]))
                else:
                    app.mount(service["mount_path"], service["app"])

                service["mounted"] = True
                # Use logger here, as it's guaranteed to be configured
                logger.info(f"✓ Mounted {name} at {service['mount_path']}")

                await flash_manager.add_message(
                    "success",
                    f"Mounted {name}",
                    f"Available at {service['mount_path']}",
                )

                return True
            except Exception as e:
                logger.error(f"✗ Failed to mount {name}: {e}")
                service["mount_error"] = str(e)

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

    async def check_all_health(self, base_url: str) -> Dict[str, Any]:
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
                    check_service_health_async(name, base_url, service["health_path"])
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


# Global async service manager
service_manager = AsyncServiceManager()

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the unified platform.
    Handles startup and graceful shutdown.
    """
    worker_id = os.getpid()

    # ====================================================================
    # [!!!] MOVED BLOCKS HERE TO RUN IN WORKER PROCESS [!!!]
    # ====================================================================

    # NOTE: Windows event loop policy is now set at module level (line 42)
    # to ensure it's applied before uvicorn creates the event loop

    # Fix Windows console encoding issues
    if sys.platform == "win32":
        try:
            # Fix stdout/stderr streams
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")

            # CRITICAL FIX: Reconfigure ALL existing logging handlers
            # This prevents UnicodeEncodeError when logging Unicode characters
            for handler in logging.root.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    try:
                        # Try to reconfigure the handler's stream
                        if hasattr(handler.stream, "reconfigure"):
                            handler.stream.reconfigure(encoding="utf-8")
                        elif hasattr(handler.stream, "name"):
                            # For file handlers, recreate stream with UTF-8
                            old_stream = handler.stream
                            handler.stream = open(
                                old_stream.name, "w", encoding="utf-8"
                            )
                    except (AttributeError, OSError):
                        # Some streams can't be reconfigured (e.g., StringIO)
                        # Skip them gracefully
                        pass

            print("✅ Reconfigured stdout/stderr and logging handlers to UTF-8")
        except Exception as e:
            print(f"⚠️  Could not reconfigure encoding: {e}")

    # LOAD ENVIRONMENT VARIABLES
    try:
        from dotenv import load_dotenv

        # Load .env file from project root (go up from src/ to Graphix/)
        # Assumes this file is in a 'src' directory
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"✅ Loaded environment variables from: {env_path}")

            # Verify critical keys
            if os.getenv("OPENAI_API_KEY"):
                print("✅ OPENAI_API_KEY loaded successfully")
            else:
                print("⚠️ OPENAI_API_KEY not found in .env")

            if os.getenv("ANTHROPIC_API_KEY"):
                print("✅ ANTHROPIC_API_KEY loaded successfully")

            if os.getenv("GRAPHIX_API_KEY"):
                print("✅ GRAPHIX_API_KEY loaded successfully")
        else:
            print(f"⚠️ .env file not found at: {env_path}")
            print("💡 Create a .env file with your API keys to enable LLM features")
    except ImportError:
        print("⚠️ python-dotenv not installed. Run: pip install python-dotenv")
        print("💡 API keys will need to be set as system environment variables")
    except Exception as e:
        print(f"❌ Error loading .env: {e}")
    # ====================================================================

    # [!!!] MOVED LOGGING SETUP HERE [!!!]
    # This ensures logging is configured *after* encoding is set
    # and *inside* the worker process.
    setup_unified_logging()

    # STARTUP
    try:
        logger.info("=" * 70)
        logger.info(f"Starting Unified Platform (Worker {worker_id})")
        logger.info("=" * 70)

        # Log configuration
        logger.info(f"Host: {settings.host}:{settings.port}")
        logger.info(f"Workers: {settings.workers}")
        logger.info(f"Auth: {settings.auth_method.value}")
        logger.info(f"Metrics: {'Enabled' if settings.enable_metrics else 'Disabled'}")
        logger.info(
            f"Health Checks: {'Enabled' if settings.enable_health_checks else 'Disabled'}"
        )

        # Enforce security-critical configuration
        if settings.auth_method == AuthMethod.JWT and not settings.jwt_secret:
            logger.error(
                "JWT auth is enabled but jwt_secret is not configured. Failing startup."
            )
            raise RuntimeError("JWT secret is required when AuthMethod is JWT")
        if settings.cors_enabled and any(o == "*" for o in settings.cors_origins):
            logger.warning(
                "Wildcard '*' detected in CORS origins; this is discouraged in production. Consider restricting to specific origins."
            )

        # Import and mount services
        logger.info("=" * 70)
        logger.info("Importing services...")
        logger.info("=" * 70)

        # Import VULCAN
        vulcan_result = await import_service_async(
            "VULCAN", settings.vulcan_module, settings.vulcan_attr, "FastAPI"
        )
        await service_manager.register_service(
            "vulcan",
            vulcan_result,
            settings.vulcan_mount,
            f"{settings.vulcan_mount}/health",
        )

        # Import Arena
        arena_result = await import_service_async(
            "Arena", settings.arena_module, settings.arena_attr, "FastAPI"
        )
        await service_manager.register_service(
            "arena",
            arena_result,
            settings.arena_mount,
            f"{settings.arena_mount}/health",
        )

        # Import Registry
        registry_result = await import_service_async(
            "Registry", settings.registry_module, settings.registry_attr, "Flask"
        )
        await service_manager.register_service(
            "registry",
            registry_result,
            settings.registry_mount,
            f"{settings.registry_mount}/health",
        )

        # Mount services
        logger.info("=" * 70)
        logger.info("Mounting services...")
        logger.info("=" * 70)

        # Explicit VULCAN mount per user request
        try:
            vulcan_module = importlib.import_module("src.vulcan.main")
            if not hasattr(vulcan_module, "app") or not isinstance(
                vulcan_module.app, FastAPI
            ):
                raise RuntimeError("src.vulcan.main does not expose a FastAPI 'app'")
            app.mount("/vulcan", vulcan_module.app)
            logger.info("✓ Mounted vulcan at /vulcan")
            # Manually update service manager state if it was registered
            if "vulcan" in service_manager.services:
                service_manager.services["vulcan"]["mounted"] = True

            # ================================================================
            # VULCAN DEPLOYMENT INITIALIZATION
            # Since VULCAN is mounted as a sub-app, its lifespan doesn't run.
            # We must manually initialize the deployment here.
            # ================================================================
            try:
                logger.info("Initializing VULCAN deployment...")
                from vulcan.config import AgentConfig, get_config
                from vulcan.orchestrator import ProductionDeployment

                # Load configuration profile
                vulcan_config = get_config("development")
                if not isinstance(vulcan_config, AgentConfig):
                    logger.warning(
                        "get_config returned invalid type, creating default AgentConfig"
                    )
                    vulcan_config = AgentConfig()

                # Create the deployment
                vulcan_deployment = ProductionDeployment(vulcan_config)

                # Attach to VULCAN's app.state so health checks pass
                vulcan_module.app.state.deployment = vulcan_deployment
                vulcan_module.app.state.startup_time = __import__("time").time()
                vulcan_module.app.state.worker_id = worker_id

                # ================================================================
                # ACTIVATE ALL VULCAN SUBSYSTEMS (from main.py lifespan logic)
                # ================================================================
                def _activate_subsystem(deps, attr_name: str, display_name: str, needs_init: bool = False):
                    """Helper to activate a subsystem with optional initialization."""
                    if hasattr(deps, attr_name) and getattr(deps, attr_name):
                        subsystem = getattr(deps, attr_name)
                        if needs_init and hasattr(subsystem, 'initialize'):
                            subsystem.initialize()
                        logger.info(f"✓ {display_name} activated")
                        return True
                    return False

                try:
                    logger.info("Activating all Vulcan subsystem modules...")
                    
                    # Initialize subsystems that need explicit initialization
                    _activate_subsystem(vulcan_deployment.collective.deps, 'curiosity', 'Curiosity Engine', needs_init=True)
                    _activate_subsystem(vulcan_deployment.collective.deps, 'crystallizer', 'Knowledge Crystallizer', needs_init=True)
                    _activate_subsystem(vulcan_deployment.collective.deps, 'decomposer', 'Problem Decomposer', needs_init=True)
                    _activate_subsystem(vulcan_deployment.collective.deps, 'semantic_bridge', 'Semantic Bridge', needs_init=True)
                    
                    # Initialize all Reasoning subsystems (no explicit init needed)
                    _activate_subsystem(vulcan_deployment.collective.deps, 'symbolic', 'Symbolic Reasoning')
                    _activate_subsystem(vulcan_deployment.collective.deps, 'probabilistic', 'Probabilistic Reasoning')
                    _activate_subsystem(vulcan_deployment.collective.deps, 'causal', 'Causal Reasoning')
                    _activate_subsystem(vulcan_deployment.collective.deps, 'analogical', 'Analogical Reasoning')
                    
                    # Initialize Memory subsystems
                    _activate_subsystem(vulcan_deployment.collective.deps, 'ltm', 'Long-term Memory')
                    _activate_subsystem(vulcan_deployment.collective.deps, 'am', 'Associative Memory')
                    
                    # Initialize Learning subsystems
                    _activate_subsystem(vulcan_deployment.collective.deps, 'continual', 'Continual Learning')
                    _activate_subsystem(vulcan_deployment.collective.deps, 'meta', 'Meta-Learning')
                    
                    # Initialize Safety subsystems
                    if hasattr(vulcan_deployment.collective.deps, 'safety') and vulcan_deployment.collective.deps.safety:
                        safety_validator = vulcan_deployment.collective.deps.safety
                        if hasattr(safety_validator, 'activate_all_constraints'):
                            try:
                                safety_validator.activate_all_constraints()
                                logger.info("✓ Safety Validator with all constraints activated")
                            except Exception as e:
                                logger.warning(f"Failed to activate all constraints: {e}")
                                logger.info("✓ Safety Validator activated (without all constraints)")
                        else:
                            logger.info("✓ Safety Validator activated")
                    
                    logger.info("✅ All Vulcan subsystem modules activation complete")
                    
                except Exception as subsys_err:
                    logger.error(f"Error during subsystem activation: {subsys_err}", exc_info=True)
                    logger.warning("Continuing with partial subsystem activation")

                # Start self-improvement drive if enabled
                if vulcan_config.enable_self_improvement:
                    try:
                        world_model = vulcan_deployment.collective.deps.world_model
                        
                        if world_model:
                            from vulcan.world_model.meta_reasoning import MotivationalIntrospection
                            
                            world_model_config = vulcan_config.world_model
                            config_path = getattr(
                                world_model_config,
                                "meta_reasoning_config",
                                "configs/intrinsic_drives.json",
                            )
                            
                            introspection = MotivationalIntrospection(world_model, config_path=config_path)
                            logger.info("✓ MotivationalIntrospection initialized (modern mode)")
                        
                        if world_model and hasattr(world_model, "start_autonomous_improvement"):
                            world_model.start_autonomous_improvement()
                            logger.info("🚀 Autonomous self-improvement drive started")
                        else:
                            logger.warning("Self-improvement enabled but world model doesn't support it")
                    except Exception as si_err:
                        logger.error(f"Failed to start self-improvement drive: {si_err}")
                # ================================================================

                # Initialize LLM component if available
                try:
                    from graphix_vulcan_llm import GraphixVulcanLLM

                    llm_instance = GraphixVulcanLLM(
                        config_path="configs/llm_config.yaml"
                    )
                    vulcan_module.app.state.llm = llm_instance
                    logger.info("✓ VULCAN LLM initialized")
                except ImportError:
                    logger.info("GraphixVulcanLLM not available, using mock")
                    from unittest.mock import MagicMock

                    vulcan_module.app.state.llm = MagicMock()
                except Exception as llm_err:
                    logger.warning(f"LLM initialization failed: {llm_err}, using mock")
                    from unittest.mock import MagicMock

                    vulcan_module.app.state.llm = MagicMock()

                logger.info("✓ VULCAN deployment initialized successfully")

            except Exception as init_err:
                logger.error(
                    f"⚠️ VULCAN deployment initialization failed: {init_err}",
                    exc_info=True,
                )
                logger.warning(
                    "VULCAN health checks will report unhealthy until manually initialized"
                )
            # ================================================================

        except Exception as e:
            logger.error(f"❌ Failed to mount vulcan at /vulcan: {e}", exc_info=True)
            logger.info("vulcan: ❌ FAILED")

        await service_manager.mount_service(app, "arena")
        await service_manager.mount_service(app, "registry", use_wsgi=True)

        # ================================================================
        # ADDITIONAL FASTAPI SERVICES (API Gateway, DQS, PII)
        # ================================================================
        logger.info("=" * 70)
        logger.info("Importing additional FastAPI services...")
        logger.info("=" * 70)

        # Import and mount API Gateway
        if settings.enable_api_gateway:
            try:
                api_gateway_result = await import_service_async(
                    "API Gateway", settings.api_gateway_module, settings.api_gateway_attr, "FastAPI"
                )
                await service_manager.register_service(
                    "api_gateway",
                    api_gateway_result,
                    settings.api_gateway_mount,
                    f"{settings.api_gateway_mount}/health",
                )
                await service_manager.mount_service(app, "api_gateway")
                logger.info(f"✓ Mounted API Gateway at {settings.api_gateway_mount}")
            except Exception as e:
                logger.error(f"❌ Failed to mount API Gateway: {e}", exc_info=True)
        else:
            logger.info("⊘ API Gateway disabled via configuration")

        # Import and mount DQS Service
        if settings.enable_dqs_service:
            try:
                dqs_result = await import_service_async(
                    "DQS Service", settings.dqs_module, settings.dqs_attr, "FastAPI"
                )
                await service_manager.register_service(
                    "dqs",
                    dqs_result,
                    settings.dqs_mount,
                    f"{settings.dqs_mount}/health",
                )
                await service_manager.mount_service(app, "dqs")
                logger.info(f"✓ Mounted DQS Service at {settings.dqs_mount}")
            except Exception as e:
                logger.error(f"❌ Failed to mount DQS Service: {e}", exc_info=True)
        else:
            logger.info("⊘ DQS Service disabled via configuration")

        # Import and mount PII Service
        if settings.enable_pii_service:
            try:
                pii_result = await import_service_async(
                    "PII Service", settings.pii_module, settings.pii_attr, "FastAPI"
                )
                await service_manager.register_service(
                    "pii",
                    pii_result,
                    settings.pii_mount,
                    f"{settings.pii_mount}/health",
                )
                await service_manager.mount_service(app, "pii")
                logger.info(f"✓ Mounted PII Service at {settings.pii_mount}")
            except Exception as e:
                logger.error(f"❌ Failed to mount PII Service: {e}", exc_info=True)
        else:
            logger.info("⊘ PII Service disabled via configuration")

        # ================================================================
        # STANDALONE SERVICES (API Server, Registry gRPC, Listener)
        # These services need to run as separate processes because they
        # use non-FastAPI frameworks (custom HTTP server, gRPC, etc.)
        # ================================================================
        logger.info("=" * 70)
        logger.info("Starting standalone services as background processes...")
        logger.info("=" * 70)

        # Track background processes for cleanup
        background_processes = []

        # Start API Server (custom HTTP server)
        if settings.enable_api_server:
            try:
                logger.info(f"Starting API Server on port {settings.api_server_port}...")
                # Use subprocess.Popen instead of asyncio.create_subprocess_exec
                # This avoids issues with Windows event loop policy and uvicorn --reload
                api_server_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m", "src.api_server",
                    ],
                    env={**os.environ, "API_SERVER_PORT": str(settings.api_server_port)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                background_processes.append(("api_server", api_server_proc))
                logger.info(f"✓ API Server started (PID: {api_server_proc.pid})")
                
                # Register in service manager for status tracking
                await service_manager.register_service(
                    "api_server",
                    ServiceImportResult(
                        name="API Server",
                        success=True,
                        import_path="src.api_server (standalone)",
                    ),
                    f"http://localhost:{settings.api_server_port}",
                    f"http://localhost:{settings.api_server_port}/health",
                )
            except Exception as e:
                logger.error(f"❌ Failed to start API Server: {e}", exc_info=True)
        else:
            logger.info("⊘ API Server disabled via configuration")

        # Start Registry gRPC Server
        if settings.enable_registry_grpc:
            try:
                logger.info(f"Starting Registry gRPC Server on port {settings.registry_grpc_port}...")
                # Use subprocess.Popen instead of asyncio.create_subprocess_exec
                registry_grpc_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m", "src.governance.registry_api_server",
                    ],
                    env={**os.environ, "REGISTRY_PORT": str(settings.registry_grpc_port)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                background_processes.append(("registry_grpc", registry_grpc_proc))
                logger.info(f"✓ Registry gRPC Server started (PID: {registry_grpc_proc.pid})")
                
                # Register in service manager
                await service_manager.register_service(
                    "registry_grpc",
                    ServiceImportResult(
                        name="Registry gRPC",
                        success=True,
                        import_path="src.governance.registry_api_server (standalone)",
                    ),
                    f"grpc://localhost:{settings.registry_grpc_port}",
                    None,  # gRPC services don't have HTTP health endpoints
                )
            except Exception as e:
                logger.error(f"❌ Failed to start Registry gRPC Server: {e}", exc_info=True)
        else:
            logger.info("⊘ Registry gRPC Server disabled via configuration")

        # Start Listener Service
        if settings.enable_listener:
            try:
                logger.info(f"Starting Listener Service on port {settings.listener_port}...")
                # Use subprocess.Popen instead of asyncio.create_subprocess_exec
                listener_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m", "src.listener",
                        "--port", str(settings.listener_port),
                        "--host", settings.host,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                background_processes.append(("listener", listener_proc))
                logger.info(f"✓ Listener Service started (PID: {listener_proc.pid})")
                
                # Register in service manager
                await service_manager.register_service(
                    "listener",
                    ServiceImportResult(
                        name="Listener",
                        success=True,
                        import_path="src.listener (standalone)",
                    ),
                    f"http://localhost:{settings.listener_port}",
                    f"http://localhost:{settings.listener_port}/health",
                )
            except Exception as e:
                logger.error(f"❌ Failed to start Listener Service: {e}", exc_info=True)
        else:
            logger.info("⊘ Listener Service disabled via configuration")

        # Store background processes in app state for cleanup
        app.state.background_processes = background_processes

        # Summary
        logger.info("=" * 70)
        logger.info("Service Status Summary")
        logger.info("=" * 70)
        service_status = await service_manager.get_service_status()
        for name, service in service_status.items():
            status_txt = "✅ MOUNTED" if service.get("mounted") else "❌ FAILED"
            logger.info(f"{name}: {status_txt}")
            if service.get("mounted"):
                logger.info(f"  → {service['mount_path']}")
                logger.info(f"  → Import: {service.get('import_path', 'N/A')}")

        logger.info("=" * 70)
        logger.info(f"Platform Ready! (Worker {worker_id})")
        logger.info("=" * 70)

    except asyncio.CancelledError:
        logger.warning(f"Unified Platform (Worker {worker_id}) startup cancelled")
        raise
    except Exception as e:
        logger.error(f"Failed to start Unified Platform: {e}", exc_info=True)
        raise

    try:
        yield
    except asyncio.CancelledError:
        logger.info(
            f"Unified Platform (Worker {worker_id}) received cancellation signal"
        )
    finally:
        # SHUTDOWN
        logger.info("=" * 70)
        logger.info(f"Shutting down Unified Platform (Worker {worker_id})...")
        logger.info("=" * 70)

        # Cleanup background processes
        if hasattr(app.state, "background_processes"):
            logger.info("Terminating background services...")
            for service_name, process in app.state.background_processes:
                try:
                    if process.poll() is None:  # Process is still running
                        logger.info(f"Terminating {service_name} (PID: {process.pid})...")
                        process.terminate()
                        try:
                            # Wait up to 5 seconds for graceful shutdown
                            process.wait(timeout=5.0)
                            logger.info(f"✓ {service_name} terminated gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if graceful shutdown fails
                            logger.warning(f"Force killing {service_name} (PID: {process.pid})...")
                            process.kill()
                            process.wait()
                            logger.info(f"✓ {service_name} killed")
                except Exception as e:
                    logger.error(f"Error terminating {service_name}: {e}")

        # Cleanup tasks
        logger.info("Shutdown complete")


# =============================================================================
# FASTAPI APP CREATION
# =============================================================================

app = FastAPI(
    title="Graphix Vulcan Unified Platform",
    description="Enterprise-grade unified platform for AI agent services (v2.1 - Production-Hardened)",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Request size limiting middleware (header-based)
@app.middleware("http")
async def request_size_limit_middleware(request: Request, call_next):
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


# Security headers middleware (applied globally)
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Origin"
    return response


# CORS middleware (tightened)
if settings.cors_enabled:
    allowed_origins = [
        o
        for o in settings.cors_origins
        if isinstance(o, str) and o.strip() and o != "*"
    ]
    if not allowed_origins:
        allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


# Request logging and metrics middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        f"{request.method} {request.url.path} [{response.status_code}] {duration:.3f}s"
    )

    if PROMETHEUS_AVAILABLE and request_counter and request_duration:
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


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Enhanced root endpoint with flash messaging, live service health, and API explorer.
    """
    base_url = f"http://{request.url.hostname}:{request.url.port or settings.port}"

    service_status = await service_manager.get_service_status()
    recent_messages = await flash_manager.get_recent_messages(limit=5)

    health_checks = {}
    if settings.enable_health_checks:
        health_checks = await service_manager.check_all_health(base_url)

    flash_html = ""
    if recent_messages:
        flash_html = '<div class="flash-section">'
        for msg in reversed(recent_messages):
            level_colors = {
                "error": "#dc3545",
                "warning": "#ffc107",
                "info": "#17a2b8",
                "success": "#28a745",
            }
            color = level_colors.get(msg["level"], "#6c757d")
            flash_html += f"""
            <div class="flash-message" style="border-left: 4px solid {color};">
                <strong style="color: {color};">{msg["level"].upper()}</strong>: {msg["message"]}
                {f'<div class="flash-details">{msg["details"]}</div>' if msg.get("details") else ""}
                <div class="flash-timestamp">{msg["timestamp"]}</div>
            </div>
            """
        flash_html += "</div>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Graphix Vulcan Unified Platform v2.1</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .flash-section {{
                margin: 20px 0;
            }}
            .flash-message {{
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .flash-details {{
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            .flash-timestamp {{
                font-size: 0.8em;
                color: #999;
                margin-top: 5px;
            }}
            .service-card {{
                background: white;
                padding: 20px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .status-mounted {{ color: #28a745; }}
            .status-failed {{ color: #dc3545; }}
            .status-healthy {{ color: #28a745; }}
            .status-unhealthy {{ color: #ffc107; }}
            .links {{ margin-top: 10px; }}
            .links a {{
                display: inline-block;
                margin-right: 15px;
                padding: 8px 15px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .error {{ color: #dc3545; font-size: 0.9em; }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .badge-success {{ background: #28a745; color: white; }}
            .badge-danger {{ background: #dc3545; color: white; }}
            .badge-warning {{ background: #ffc107; color: black; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Graphix Vulcan Unified Platform</h1>
            <p>Production-Hardened Enterprise Platform (v2.1)</p>
            <p><strong>Worker PID:</strong> {os.getpid()} | <strong>Workers:</strong> {settings.workers}</p>
            <p><strong>Auth:</strong> {settings.auth_method.value.upper()} |
               <strong>Metrics:</strong> {"✅ Enabled" if settings.enable_metrics and PROMETHEUS_AVAILABLE else "❌ Disabled"}</p>
        </div>

        {flash_html}

        <div class="service-card">
            <h2>📊 Platform Status</h2>
            <p><strong>Status:</strong> <span class="badge badge-success">ACTIVE</span></p>
            <div class="links">
                <a href="/docs">📚 API Documentation</a>
                <a href="/health">🏥 Health Check</a>
                <a href="/api/status">📋 JSON Status</a>
                {f'<a href="{settings.metrics_path}">📈 Metrics</a>' if settings.enable_metrics and PROMETHEUS_AVAILABLE else ""}
                <a href="/auth/token">🔑 Get Token</a>
            </div>
        </div>
    """

    for name, status in service_status.items():
        mounted = status.get("mounted", False)
        health = health_checks.get(name, {})

        status_class = "status-mounted" if mounted else "status-failed"
        status_text = "✅ MOUNTED" if mounted else "❌ FAILED"

        health_status = ""
        if mounted and health:
            health_class = f"status-{health.get('status', 'unknown')}"
            health_status = f'<p class="{health_class}"><strong>Health:</strong> {health.get("status", "unknown").upper()}</p>'
            if health.get("latency_ms"):
                health_status += (
                    f"<p><strong>Latency:</strong> {health['latency_ms']:.2f}ms</p>"
                )

        html_content += f"""
        <div class="service-card">
            <h2 class="{status_class}">{name.upper()}</h2>
            <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        """

        if mounted:
            html_content += f"""
            <p><strong>Mount Path:</strong> <code>{status.get("mount_path")}</code></p>
            <p><strong>Import Path:</strong> <code>{status.get("import_path", "N/A")}</code></p>
            {health_status}
            <div classs="links">
                <a href="{status.get("mount_path")}">🔗 Service Root</a>
                {f'<a href="{status.get("docs_url")}">📚 API Docs</a>' if status.get("docs_url") else ""}
                {f'<a href="{status.get("health_path")}">🏥 Health</a>' if status.get("health_path") else ""}
            </div>
            """
        elif status.get("error"):
            html_content += f'<p class="error">Error: {status["error"]}</p>'

        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check(request: Request):
    """Comprehensive health check for all services."""
    base_url = f"http://{request.url.hostname}:{request.url.port or settings.port}"

    service_status = await service_manager.get_service_status()

    health_checks = {}
    if settings.enable_health_checks:
        health_checks = await service_manager.check_all_health(base_url)

    all_healthy = all(
        h.get("status") == "healthy"
        for h in health_checks.values()
        if isinstance(h, dict) and "status" in h
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_pid": os.getpid(),
        "services": {
            name: {
                "mounted": status.get("mounted", False),
                "health": health_checks.get(name, {"status": "unknown"}),
            }
            for name, status in service_status.items()
        },
    }


@app.get("/api/status")
async def api_status():
    """JSON API for service status."""
    return {
        "platform": {
            "name": "Graphix Vulcan Unified Platform",
            "version": "2.1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_pid": os.getpid(),
            "workers": settings.workers,
        },
        "services": await service_manager.get_service_status(),
        "configuration": {
            "auth_method": settings.auth_method.value,
            "metrics_enabled": settings.enable_metrics,
            "health_checks_enabled": settings.enable_health_checks,
            "auto_detect_src": settings.auto_detect_src,
        },
    }


@app.post("/auth/token")
async def get_token(request: Request, sub: Optional[str] = None):
    """
    Secure token issuance endpoint.
    Requires:
    - AuthMethod is JWT
    - Valid API key via X-API-Key
    """
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

    token = JWTAuth.create_access_token({"sub": subject})
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.jwt_expire_minutes * 60,
    }


@app.get("/api/protected")
async def protected_endpoint(auth: Dict = Depends(verify_authentication)):
    """Example protected endpoint."""
    return {"message": "Access granted!", "auth": auth}


if PROMETHEUS_AVAILABLE:

    @app.get(settings.metrics_path)
    async def metrics():
        """Aggregated Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# ARENA API ENDPOINTS (INTEGRATED)
# =============================================================================

_arena_instance_cache = None
_arena_instance_initialized = False


def get_arena_instance():
    """
    Get or create Arena instance for proxy endpoints.
    """
    global _arena_instance_cache, _arena_instance_initialized

    if _arena_instance_cache is not None:
        return _arena_instance_cache

    if _arena_instance_initialized:
        return None

    _arena_instance_initialized = True

    try:
        from src.graphix_arena import _ARENA_INSTANCE, GraphixArena, register_routes

        if _ARENA_INSTANCE is not None:
            logger.info("✅ Using existing Arena instance from src.graphix_arena")
            _arena_instance_cache = _ARENA_INSTANCE
            return _arena_instance_cache

        logger.info("🔧 Creating new Arena instance for platform proxy endpoints")
        arena = GraphixArena()

        try:
            register_routes(arena)
        except Exception as e:
            logger.warning(f"Could not register Arena routes: {e}")

        _arena_instance_cache = arena
        logger.info("✅ Arena instance initialized successfully")
        return _arena_instance_cache

    except ImportError as e:
        logger.error(f"❌ Could not import Arena components: {e}")
        logger.error("   Arena API endpoints will return 503 Service Unavailable")
        return None
    except Exception as e:
        logger.error(f"❌ Could not initialize Arena instance: {e}")
        return None


@app.post("/api/arena/run/{agent_id}")
async def arena_run_agent(
    agent_id: str, request: Request, auth: Dict = Depends(verify_authentication)
):
    """Run agent task via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.run_agent_task(request)
        return result
    except Exception as e:
        logger.error(f"Arena agent task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arena/feedback")
async def arena_feedback(request: Request, auth: Dict = Depends(verify_authentication)):
    """Submit feedback via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.feedback_ingestion(request)
        return result
    except Exception as e:
        logger.error(f"Arena feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arena/tournament")
async def arena_tournament(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """Run tournament via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        result = await arena.run_tournament_task(request)
        return result
    except Exception as e:
        logger.error(f"Arena tournament failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arena/feedback_dispatch")
async def arena_feedback_dispatch(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """Dispatch feedback protocol via Arena API."""
    arena = get_arena_instance()
    if arena is None:
        raise HTTPException(status_code=503, detail="Arena not available")

    try:
        from src.graphix_arena import MAX_PAYLOAD_SIZE, dispatch_feedback_protocol

        data = await request.json()

        payload_size = len(json.dumps(data))
        if payload_size > MAX_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: {payload_size} > {MAX_PAYLOAD_SIZE}",
            )

        context = {"audit_log": []}
        result = dispatch_feedback_protocol(data, context)
        return result
    except Exception as e:
        logger.error(f"Arena feedback dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CLI & MAIN
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Graphix Vulcan Unified Platform v2.1")

    parser.add_argument("--host", default=settings.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    parser.add_argument(
        "--workers", type=int, default=settings.workers, help="Number of workers"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument("--mount-vulcan", default=settings.vulcan_mount)
    parser.add_argument("--mount-arena", default=settings.arena_mount)
    parser.add_argument("--mount-registry", default=settings.registry_mount)

    parser.add_argument(
        "--vulcan-module",
        default=settings.vulcan_module,
        help="VULCAN import module (e.g., 'main' or 'src.vulcan.main')",
    )
    parser.add_argument(
        "--arena-module", default=settings.arena_module, help="Arena import module"
    )
    parser.add_argument(
        "--registry-module",
        default=settings.registry_module,
        help="Registry import module",
    )

    parser.add_argument(
        "--auth-method",
        choices=["none", "api_key", "jwt", "oauth2"],
        default=settings.auth_method.value,
        help="Authentication method",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--jwt-secret", help="JWT secret key")

    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-health-checks", action="store_true")
    parser.add_argument(
        "--no-auto-detect-src",
        action="store_true",
        help="Disable automatic src/ directory detection",
    )

    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    settings.host = args.host
    settings.port = args.port
    settings.workers = args.workers
    settings.reload = args.reload
    settings.vulcan_mount = args.mount_vulcan
    settings.arena_mount = args.mount_arena
    settings.registry_mount = args.mount_registry
    settings.vulcan_module = args.vulcan_module
    settings.arena_module = args.arena_module
    settings.registry_module = args.registry_module
    settings.auth_method = AuthMethod(args.auth_method)
    settings.log_level = args.log_level
    settings.auto_detect_src = not args.no_auto_detect_src

    if args.api_key:
        settings.api_key = args.api_key
    if args.jwt_secret:
        settings.jwt_secret = args.jwt_secret
    if args.disable_metrics:
        settings.enable_metrics = False
    if args.disable_health_checks:
        settings.enable_health_checks = False

    setup_unified_logging()

    import uvicorn

    logger.info("=" * 70)
    logger.info("Launching Unified Platform Server v2.1")
    logger.info("=" * 70)
    logger.info(f"URL: http://{settings.host}:{settings.port}")
    logger.info(f"Docs: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Workers: {settings.workers}")
    logger.info(f"Auth: {settings.auth_method.value}")
    logger.info("=" * 70)

    uvicorn.run(
        "full_platform:app",
        host=settings.host,
        port=settings.port,
        workers=1 if settings.reload else settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
