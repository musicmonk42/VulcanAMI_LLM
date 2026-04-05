"""Platform settings and LLM component validation for the Vulcan unified platform.

Provides UnifiedPlatformSettings (Pydantic BaseSettings) and the
validate_llm_components diagnostic function.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.platform.auth import AuthMethod
from src.platform.secrets import secrets

logger = logging.getLogger("unified_platform")


# =============================================================================
# UNIFIED PLATFORM SETTINGS
# =============================================================================


class UnifiedPlatformSettings(BaseSettings):
    """Centralized configuration with secrets support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="UNIFIED_", case_sensitive=False, extra="ignore"
    )

    # Server configuration
    # When PORT env var is set (Railway/Heroku), bind to 0.0.0.0 to accept external connections
    # When running locally (no PORT env var), default to localhost for security
    host: str = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    port: int = int(os.environ.get("PORT", 8080))
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
    registry_module: str = "src.governance.app"
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

    # Standalone service host bindings (default to localhost for security)
    listener_host: str = os.environ.get("LISTENER_HOST", "127.0.0.1")

    # Enable/disable individual services
    enable_api_gateway: bool = True
    enable_dqs_service: bool = True
    enable_pii_service: bool = True
    enable_api_server: bool = True
    enable_registry_grpc: bool = True
    enable_listener: bool = True

    # Cloud platform detection (for logging purposes)
    _is_cloud_platform: bool = bool(
        os.environ.get("RAILWAY_ENVIRONMENT")  # Railway detection
        or os.environ.get("RAILWAY_SERVICE_NAME")  # Alternative Railway detection
        or os.environ.get("RENDER")  # Render.com detection
        or os.environ.get("HEROKU_APP_NAME")  # Heroku detection
    )

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
                from src.env_utils import is_dev_env
                if not is_dev_env():
                    raise ValueError(
                        "No authentication configured. Set JWT_SECRET or API_KEY, "
                        "or set VULCAN_ENV=development|test to run without auth."
                    )


# Module-level settings accessor (lazy singleton to match original behavior)
_settings_instance: Optional[UnifiedPlatformSettings] = None


def get_settings() -> UnifiedPlatformSettings:
    """Get or create the singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = UnifiedPlatformSettings()
    return _settings_instance


def set_settings(settings: UnifiedPlatformSettings) -> None:
    """Set the settings instance (for use during initialization)."""
    global _settings_instance
    _settings_instance = settings


# =============================================================================
# LLM COMPONENT VALIDATION
# =============================================================================


def validate_llm_components() -> Dict[str, Any]:
    """
    Validate that real LLM components loaded, not fallbacks.

    This function checks the CognitiveLoop, GraphixTransformer, and GraphixVulcanBridge
    components to ensure they are the real implementations, not fallback stubs.

    Returns:
        Dictionary with validation results:
        - issues: List of issues found
        - status: "PASS" or "FAIL"
        - components: Dict of component status
    """
    issues = []
    components = {}

    # Check CognitiveLoop
    try:
        from src.integration.cognitive_loop import CognitiveLoop
        components["CognitiveLoop"] = {
            "loaded": True,
            "is_fallback": getattr(CognitiveLoop, '_is_fallback', False),
            "type": type(CognitiveLoop).__name__
        }
        if components["CognitiveLoop"]["is_fallback"]:
            issues.append("CognitiveLoop is using FALLBACK implementation")
    except ImportError as e:
        components["CognitiveLoop"] = {"loaded": False, "error": str(e)}
        issues.append(f"CognitiveLoop import failed: {e}")

    # Check GraphixTransformer
    try:
        from src.llm_core.graphix_transformer import GraphixTransformer
        components["GraphixTransformer"] = {
            "loaded": True,
            "is_fallback": getattr(GraphixTransformer, '_is_fallback', False),
            "type": type(GraphixTransformer).__name__
        }
        if components["GraphixTransformer"]["is_fallback"]:
            issues.append("GraphixTransformer is using FALLBACK implementation")
    except ImportError as e:
        components["GraphixTransformer"] = {"loaded": False, "error": str(e)}
        issues.append(f"GraphixTransformer import failed: {e}")

    # Check GraphixVulcanBridge
    try:
        from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge
        components["GraphixVulcanBridge"] = {
            "loaded": True,
            "is_fallback": getattr(GraphixVulcanBridge, '_is_fallback', False),
            "type": type(GraphixVulcanBridge).__name__
        }
        if components["GraphixVulcanBridge"]["is_fallback"]:
            issues.append("GraphixVulcanBridge is using FALLBACK implementation")
    except ImportError as e:
        components["GraphixVulcanBridge"] = {"loaded": False, "error": str(e)}
        issues.append(f"GraphixVulcanBridge import failed: {e}")

    # Check HybridLLMExecutor
    try:
        from src.vulcan.llm.hybrid_executor import get_hybrid_executor, verify_hybrid_executor_setup
        executor = get_hybrid_executor()
        if executor is not None:
            verification = verify_hybrid_executor_setup()
            components["HybridLLMExecutor"] = {
                "loaded": True,
                "has_internal_llm": verification.get("has_internal_llm", False),
                "internal_llm_type": verification.get("internal_llm_type"),
                "status": verification.get("status")
            }
            if not verification.get("has_internal_llm"):
                issues.append("HybridLLMExecutor has no internal LLM - will fallback to OpenAI")
        else:
            components["HybridLLMExecutor"] = {"loaded": False, "error": "Not initialized"}
    except ImportError as e:
        components["HybridLLMExecutor"] = {"loaded": False, "error": str(e)}

    # Generate result
    result = {
        "issues": issues,
        "status": "FAIL" if issues else "PASS",
        "components": components
    }

    # Log results
    if issues:
        logger.error("=" * 60)
        logger.error("LLM COMPONENT ISSUES DETECTED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("=" * 60)
    else:
        logger.info("All LLM components loaded successfully (no fallbacks)")

    return result
