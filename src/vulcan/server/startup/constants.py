"""
Startup Constants

Centralized configuration constants for the VULCAN-AGI startup process.
All magic numbers are extracted here with clear documentation.
"""

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
    """Configuration metadata for startup phases."""
    
    # Phase timeout limits (seconds)
    CONFIGURATION_TIMEOUT = 30
    CORE_SERVICES_TIMEOUT = 60
    REASONING_SYSTEMS_TIMEOUT = 120
    MEMORY_SYSTEMS_TIMEOUT = 60
    PRELOADING_TIMEOUT = 180
    MONITORING_TIMEOUT = 30
    
    # Phase criticality (determines if failure stops startup)
    CRITICAL_PHASES = {
        "CONFIGURATION",
        "CORE_SERVICES",
    }
    
    # Phase parallel groups (can be initialized concurrently)
    PARALLEL_GROUPS = {
        "models": [
            "bert_model",
            "sentence_transformer",
            "hierarchical_memory",
            "unified_learning_system",
        ],
        "singletons": [
            "reasoning_singletons",
            "tool_selector",
            "problem_decomposer",
        ],
    }
