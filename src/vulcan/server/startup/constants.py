"""
Startup Constants

Centralized configuration constants for the VULCAN-AGI startup process.
All magic numbers are extracted here with clear documentation.
"""

# ============================================================
# Deployment Mode Configuration (P2 Fix: Issue #9)
# ============================================================

class DeploymentMode:
    """
    Deployment mode constants for configuration profiles.
    
    Defines valid deployment modes and provides validation.
    Each mode corresponds to a configuration profile with
    different resource limits and feature flags.
    """
    
    PRODUCTION = "production"
    """Production mode: optimized for performance and stability"""
    
    TESTING = "testing"
    """Testing mode: optimized for test execution and mocking"""
    
    DEVELOPMENT = "development"
    """Development mode: verbose logging and debug features enabled"""
    
    VALID_MODES = {PRODUCTION, TESTING, DEVELOPMENT}
    """Set of all valid deployment mode strings"""
    
    DEFAULT = DEVELOPMENT
    """Default mode when invalid mode is specified"""
    
    @classmethod
    def is_valid(cls, mode: str) -> bool:
        """
        Check if a deployment mode is valid.
        
        Args:
            mode: Deployment mode string to validate
            
        Returns:
            True if mode is valid, False otherwise
        """
        return mode in cls.VALID_MODES
    
    @classmethod
    def normalize(cls, mode: str) -> str:
        """
        Normalize a deployment mode to valid value.
        
        If the mode is invalid, returns the default mode.
        
        Args:
            mode: Deployment mode string to normalize
            
        Returns:
            Normalized deployment mode string
        """
        return mode if cls.is_valid(mode) else cls.DEFAULT


# ============================================================
# Logging Emoji Constants (P3 Fix: Issue #15)
# ============================================================

class LogEmoji:
    """
    Standardized emoji constants for consistent logging.
    
    Provides visual indicators in logs for quick status recognition.
    Use consistently across all logging statements.
    """
    
    SUCCESS = "✓"
    """Minor success indicator (e.g., component initialized)"""
    
    SUCCESS_MAJOR = "✅"
    """Major success indicator (e.g., full startup complete)"""
    
    WARNING = "⚠️"
    """Warning indicator for degraded state or recoverable issues"""
    
    ERROR = "❌"
    """Error indicator for failures"""
    
    INFO = "ℹ️"
    """Information indicator for status updates"""
    
    ROCKET = "🚀"
    """Launch/start indicator for significant events"""
    
    CHART = "📊"
    """Metrics/statistics indicator"""
    
    STOP = "🛑"
    """Stop/shutdown indicator"""


# ============================================================
# Thread Pool Configuration
# ============================================================

DEFAULT_THREAD_POOL_SIZE = 32
"""
Default ThreadPoolExecutor worker count for parallel cognitive tasks.

Rationale: Sufficient for parallel execution of multiple reasoning subsystems
(memory, reasoning, planning, world_model) without thread pool exhaustion.
The default Python ThreadPoolExecutor size is min(32, os.cpu_count() + 4),
which can be too small for VULCAN's concurrent cognitive processing needs.
"""

THREAD_NAME_PREFIX = "vulcan_"
"""Prefix for thread names to identify VULCAN threads in debugging."""


# ============================================================
# Memory Management Configuration
# ============================================================

MEMORY_GUARD_THRESHOLD_PERCENT = 85.0
"""
Memory usage threshold percentage that triggers garbage collection.

Rationale: Triggers GC at 85% memory usage to prevent OOM conditions
while allowing headroom for temporary allocations during model loading.
"""

MEMORY_GUARD_CHECK_INTERVAL_SECONDS = 5.0
"""
Interval in seconds between memory guard checks.

Rationale: Frequent enough to catch memory spikes, infrequent enough
to avoid excessive CPU overhead from monitoring.
"""


# ============================================================
# Redis Configuration
# ============================================================

REDIS_WORKER_TTL_SECONDS = 3600
"""
Time-to-live for worker registration in Redis (1 hour).

Rationale: Workers refresh their registration periodically. 1 hour TTL
ensures stale worker entries are cleaned up if workers crash without
graceful shutdown.
"""


# ============================================================
# Self-Optimization Configuration
# ============================================================

SELF_OPTIMIZER_TARGET_LATENCY_MS = 100
"""Target P95 latency in milliseconds for autonomous performance tuning."""

SELF_OPTIMIZER_TARGET_MEMORY_MB = 2000
"""Target memory usage in MB for autonomous performance tuning."""

SELF_OPTIMIZER_INTERVAL_SECONDS = 60
"""Interval in seconds between self-optimization iterations."""


# ============================================================
# Shutdown Configuration
# ============================================================

SHUTDOWN_TIMEOUT_SECONDS = 2.0
"""
Maximum time in seconds to wait for thread joins during shutdown.

Rationale: Allows daemon threads to complete current work items while
preventing indefinite hangs during shutdown.
"""


# ============================================================
# Logging Configuration
# ============================================================

MAX_STARTUP_INFO_LOGS = 10
"""
Maximum number of INFO-level logs during startup per phase.

Rationale: Reduces log flooding in production with multiple workers
while maintaining visibility into startup progress.
"""


# ============================================================
# Model Registry Configuration
# ============================================================

DEFAULT_CONFIG_DIR = "configs"
"""Default directory for configuration files."""

DEFAULT_DATA_DIR = "data"
"""Default directory for persistent data storage."""

DEFAULT_CHECKPOINT_DIR = "checkpoints"
"""Default directory for model checkpoints."""

LLM_CONFIG_PATH = "configs/llm_config.yaml"
"""Default path to LLM configuration file."""


# ============================================================
# Startup Phase Configuration
# ============================================================

class StartupPhaseConfig:
    """
    Configuration for startup phase timeouts and dependencies.
    
    Defines timeout values for each startup phase and which phases
    are critical (must complete for server to be operational).
    """
    
    # Phase timeout values in seconds
    CONFIGURATION_TIMEOUT = 30
    """Timeout for configuration loading phase."""
    
    CORE_SERVICES_TIMEOUT = 45
    """Timeout for core services initialization."""
    
    REASONING_SYSTEMS_TIMEOUT = 60
    """Timeout for reasoning systems initialization."""
    
    MEMORY_SYSTEMS_TIMEOUT = 60
    """Timeout for memory systems initialization."""
    
    PRELOADING_TIMEOUT = 120
    """Timeout for model preloading phase (longest due to model loading)."""
    
    MONITORING_TIMEOUT = 30
    """Timeout for monitoring systems initialization."""
    
    # Critical phases that must complete for server to start
    CRITICAL_PHASES = {"CONFIGURATION", "CORE_SERVICES"}
    """Set of phase names that are critical for startup."""
    
    # Phases that can run in parallel
    PARALLEL_GROUPS = {
        "services": ["REASONING_SYSTEMS", "MEMORY_SYSTEMS"],
        "optional": ["PRELOADING", "MONITORING"],
    }
    """Groups of phases that can run in parallel."""
