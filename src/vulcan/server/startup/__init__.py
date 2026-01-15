"""
VULCAN-AGI Startup Management

Modular startup system with phased initialization, error isolation,
and health validation.
"""

from .manager import StartupManager
from .phases import StartupPhase, PhaseMetadata, get_phase_metadata
from .subsystems import SubsystemManager, SubsystemConfig
from .health import HealthCheck, HealthStatus, ComponentHealth
from .constants import (
    DEFAULT_THREAD_POOL_SIZE,
    MEMORY_GUARD_THRESHOLD_PERCENT,
    MEMORY_GUARD_CHECK_INTERVAL_SECONDS,
    REDIS_WORKER_TTL_SECONDS,
    PROCESS_LOCK_HEARTBEAT_INTERVAL_SECONDS,
    PROCESS_LOCK_TTL_SECONDS,
    SELF_OPTIMIZER_TARGET_LATENCY_MS,
    SELF_OPTIMIZER_TARGET_MEMORY_MB,
    SELF_OPTIMIZER_INTERVAL_SECONDS,
    DeploymentMode,
    LogEmoji,
)

__all__ = [
    # Main manager
    "StartupManager",
    # Phases
    "StartupPhase",
    "PhaseMetadata",
    "get_phase_metadata",
    # Subsystems
    "SubsystemManager",
    "SubsystemConfig",
    # Health
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
    # Constants
    "DEFAULT_THREAD_POOL_SIZE",
    "MEMORY_GUARD_THRESHOLD_PERCENT",
    "MEMORY_GUARD_CHECK_INTERVAL_SECONDS",
    "REDIS_WORKER_TTL_SECONDS",
    "PROCESS_LOCK_HEARTBEAT_INTERVAL_SECONDS",
    "PROCESS_LOCK_TTL_SECONDS",
    "SELF_OPTIMIZER_TARGET_LATENCY_MS",
    "SELF_OPTIMIZER_TARGET_MEMORY_MB",
    "SELF_OPTIMIZER_INTERVAL_SECONDS",
    "DeploymentMode",
    "LogEmoji",
]
