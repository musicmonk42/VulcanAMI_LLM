# ============================================================
# VULCAN-AGI Settings Module
# Centralized configuration using Pydantic BaseSettings
# Supports environment variable overrides for all settings
# ============================================================
#
# USAGE:
#     from vulcan.settings import settings, Settings
#     
#     # Access settings
#     print(settings.api_host)
#     print(settings.api_port)
#     
#     # Create custom settings instance
#     custom = Settings(api_port=9000)
#
# ENVIRONMENT VARIABLES:
#     All settings can be overridden via environment variables.
#     See each field's 'env' parameter for the variable name.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation
# ============================================================

import logging
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    VULCAN-AGI application settings.
    
    All settings can be overridden via environment variables.
    """
    # API key for VULCAN service (checked by middleware)
    api_key: Optional[str] = Field(default=None, env=["API_KEY", "VULCAN_API_KEY"])

    # JWT (if used by any endpoints)
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60, env="JWT_EXPIRE_MINUTES")

    # Simple rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=60, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, env="RATE_LIMIT_WINDOW_SECONDS")

    # Self-improvement knobs (read by config and runtime)
    improvement_max_cost_usd: float = Field(
        default=10.0, env="IMPROVEMENT_MAX_COST_USD"
    )
    improvement_check_interval_seconds: int = Field(
        default=120, env="IMPROVEMENT_CHECK_INTERVAL_SECONDS"
    )
    
    # FIX: Self-Improvement Git Commit Control
    # Disable auto-commits to GitHub - prevents "Cannot commit: /app is not a Git repository" errors
    # Default to False to prevent unintended commits in production environments
    self_improvement_auto_commit: bool = Field(
        default=False, env="VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT"
    )

    # --- Fields from old Settings class, preserved ---
    max_graph_size: int = 1000
    max_execution_time_s: float = 30.0
    max_memory_mb: int = 2000
    enable_code_execution: bool = False
    enable_sandboxing: bool = True
    allowed_modules: List[str] = ["numpy", "pandas", "scipy", "sklearn"]

    # API server defaults to localhost for security; override with environment variable
    # Railway assigns PORT dynamically, so we read from environment with fallback to 8080
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8080, env="PORT")
    api_workers: int = 4
    api_title: str = "VULCAN-AGI API"
    api_version: str = "2.0.0"

    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    rate_limit_cleanup_interval: int = 300

    reasoning_service_url: Optional[str] = None
    planning_service_url: Optional[str] = None
    learning_service_url: Optional[str] = None
    memory_service_url: Optional[str] = None
    safety_service_url: Optional[str] = None

    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    prometheus_enabled: bool = True
    jaeger_enabled: bool = False
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831

    encryption_key: Optional[str] = None

    deployment_mode: str = "standalone"
    checkpoint_path: Optional[str] = None
    auto_checkpoint_interval: int = 100

    # Self-improvement configuration
    enable_self_improvement: bool = False
    self_improvement_config: str = "configs/intrinsic_drives.json"
    self_improvement_state: str = "data/agent_state.json"
    self_improvement_approval_required: bool = True
    # self.improvement_max_cost_usd is duplicated, using the new one
    self_improvement_check_interval_s: int = 60  # duplicated, using new name

    # LLM Execution Mode Configuration
    # Modes: "local_first" (default), "openai_first", "parallel", "ensemble"
    # - local_first: Try Vulcan's local LLM first, fallback to OpenAI
    # - openai_first: Try OpenAI first, fallback to local LLM
    # - parallel: Run both simultaneously, use first successful response
    # - ensemble: Run both, combine/select best response based on quality
    llm_execution_mode: str = Field(default="parallel", env="LLM_EXECUTION_MODE")
    # Timeout for parallel/ensemble execution (seconds)
    llm_parallel_timeout: float = Field(default=30.0, env="LLM_PARALLEL_TIMEOUT")
    # PERFORMANCE FIX: Skip local LLM entirely when it consistently returns None
    # When True: Skips local LLM attempts entirely, goes directly to OpenAI
    # ISSUE-001: Local LLM returns None 100% of the time, wasting 5-30s per request
    # Default changed to True to reduce response time from 37-73s to 5-10s
    skip_local_llm: bool = Field(default=True, env="SKIP_LOCAL_LLM")
    # For ensemble mode: minimum confidence threshold for response selection
    llm_ensemble_min_confidence: float = Field(
        default=0.7, env="LLM_ENSEMBLE_MIN_CONFIDENCE"
    )
    # Maximum tokens for OpenAI API calls (increased to 2000 for diagnostic purposes)
    llm_openai_max_tokens: int = Field(default=2000, env="LLM_OPENAI_MAX_TOKENS")

    # Knowledge Distillation Configuration
    # When enabled, captures OpenAI responses and uses them to train Vulcan's local LLM
    enable_knowledge_distillation: bool = Field(
        default=True, env="ENABLE_KNOWLEDGE_DISTILLATION"
    )
    # Path to store distillation training examples
    distillation_storage_path: str = Field(
        default="data/distillation_examples.json", env="DISTILLATION_STORAGE_PATH"
    )
    # Number of examples before triggering training
    distillation_batch_size: int = Field(default=32, env="DISTILLATION_BATCH_SIZE")
    # Time interval for periodic training (seconds)
    distillation_training_interval_s: int = Field(
        default=300, env="DISTILLATION_TRAINING_INTERVAL_S"
    )
    # Learning rate for distillation training
    distillation_learning_rate: float = Field(
        default=0.0001, env="DISTILLATION_LEARNING_RATE"
    )
    # Whether to automatically trigger training when batch is full
    distillation_auto_train: bool = Field(default=True, env="DISTILLATION_AUTO_TRAIN")

    # ================================================================
    # GRAPHIX ARENA CONFIGURATION
    # Arena is the FastAPI-based coordination surface for agent collaboration,
    # tournament selection, graph evolution, and feedback integration.
    # ================================================================
    # Base URL for Graphix Arena service
    arena_base_url: str = Field(
        default="http://localhost:8080/arena", env="ARENA_BASE_URL"
    )
    # API key for Arena authentication - must be set via env var in production
    # Default matches Arena server's default (graphix_arena.py line 346) to enable
    # internal service-to-service calls when GRAPHIX_API_KEY is not explicitly set
    arena_api_key: Optional[str] = Field(
        default="default-secret-key-for-dev", env="GRAPHIX_API_KEY"
    )
    # PERFORMANCE FIX: Arena timeout reduced to 60s based on production analysis
    # Evidence from logs shows Arena circuit breaker causes wasted work
    # while 120s is too long. 60s balances completion vs. responsiveness.
    # Circuit breaker in client.py has GENERATOR_TIMEOUT=45s - this should be higher
    # to allow Arena operations to complete before we give up entirely.
    arena_timeout: float = Field(default=60.0, env="ARENA_TIMEOUT")
    # PERFORMANCE FIX: Disable Arena by default due to 30-53 second timeouts
    # ISSUE-005: Arena operations timeout at 30-53 seconds with 66% timeout rate
    # Default changed to False to reduce response time from 37-73s to 5-10s
    # Set ARENA_ENABLED=true only for complex multi-agent tournament scenarios
    arena_enabled: bool = Field(default=False, env="ARENA_ENABLED")
    # PERFORMANCE FIX: Complexity threshold for Arena fast-path skip
    # Queries with complexity < this value skip Arena entirely for faster response
    # This prevents unnecessary Arena overhead for very simple queries
    # FIX: Lowered default from 0.3 to 0.1 - most queries should go through Arena
    # Set ARENA_COMPLEXITY_THRESHOLD=0.0 to disable fast-path skip entirely
    arena_complexity_threshold: float = Field(default=0.1, env="ARENA_COMPLEXITY_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Added from old model_config


# ============================================================
# SINGLETON INSTANCE
# ============================================================

# Create the singleton settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        The singleton Settings instance
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    
    Useful when environment variables change at runtime.
    
    Returns:
        New Settings instance with updated values
    """
    global settings
    settings = Settings()
    logger.info("Settings reloaded from environment")
    return settings


def validate_settings() -> bool:
    """
    Validate current settings.
    
    Returns:
        True if settings are valid, False otherwise
    """
    try:
        # Check critical settings
        if settings.api_port < 1 or settings.api_port > 65535:
            logger.error(f"Invalid api_port: {settings.api_port}")
            return False
        
        if settings.api_workers < 1:
            logger.error(f"Invalid api_workers: {settings.api_workers}")
            return False
        
        if settings.rate_limit_requests < 1:
            logger.error(f"Invalid rate_limit_requests: {settings.rate_limit_requests}")
            return False
        
        logger.info("Settings validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Settings validation failed: {e}")
        return False


def print_settings_summary():
    """Print a summary of current settings for debugging."""
    print("\n" + "=" * 60)
    print("VULCAN-AGI Settings Summary")
    print("=" * 60)
    print(f"API Host:           {settings.api_host}")
    print(f"API Port:           {settings.api_port}")
    print(f"API Workers:        {settings.api_workers}")
    print(f"Deployment Mode:    {settings.deployment_mode}")
    print(f"CORS Enabled:       {settings.cors_enabled}")
    print(f"Rate Limiting:      {settings.rate_limit_enabled}")
    print(f"Self-Improvement:   {settings.enable_self_improvement}")
    print(f"Distillation:       {settings.enable_knowledge_distillation}")
    print(f"Arena Enabled:      {settings.arena_enabled}")
    print(f"LLM Mode:           {settings.llm_execution_mode}")
    print("=" * 60 + "\n")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "validate_settings",
    "print_settings_summary",
]


# Log module initialization
logger.info(f"Settings module v{__version__} loaded successfully")
