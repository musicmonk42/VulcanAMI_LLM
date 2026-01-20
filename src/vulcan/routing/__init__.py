# ============================================================
# VULCAN-AGI Routing Module - Simplified Architecture
# ============================================================
# This module has been simplified as part of the tool-based architecture.
# Complex regex routing has been replaced by LLM function calling.
#
# KEPT COMPONENTS:
# - TelemetryRecorder: Dual-mode telemetry for meta-learning
# - GovernanceLogger: Audit.db integration
# - ExperimentTrigger: Meta-learning experiment generation
# - SystemHealthMonitor: System health monitoring
#
# REMOVED COMPONENTS (replaced by LLM function calling):
# - QueryRouter: Regex classification (now LLM decides)
# - LLMRouter: Redundant LLM call for routing
# - EmbeddingCache: Routing similarity (LLM has embeddings)
# - AgentCollaboration: Multi-agent orchestration
# - RoutingPrompts: Classification prompts
# ============================================================

"""
VULCAN Routing Module - Simplified Architecture

The complex regex-based routing has been replaced by LLM function calling.
The LLM now decides which tools to use based on natural language understanding.

Remaining Components:
    - TelemetryRecorder: Production telemetry and metrics
    - GovernanceLogger: Audit logging for compliance
    - ExperimentTrigger: Meta-learning experiments
    - SystemHealthMonitor: System health monitoring

Usage:
    from vulcan.routing import record_telemetry, log_to_governance
    
    # Record telemetry
    record_telemetry(query="...", response="...", tools_used=["hash_compute"])
    
    # Log governance event
    log_to_governance(action="query_processed", data={...})
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# ============================================================
# SINGLETON GLOBALS FOR THREAD-SAFE COMPONENT ACCESS
# ============================================================
_ROUTING_SINGLETON_LOCK = threading.RLock()
_ROUTING_COMPONENTS_INITIALIZED = False

# ============================================================
# REMOVED COMPONENTS - Now handled by LLM function calling
# ============================================================
# Query Router - REMOVED (LLM decides via function calling)
QUERY_ROUTER_AVAILABLE = False
QueryAnalyzer = None
QueryPlan = None
route_query = None
route_query_async = None

# Agent Collaboration - REMOVED (simplified architecture)
COLLABORATION_AVAILABLE = False
AgentCollaborationManager = None
trigger_agent_collaboration = None

# Embedding Cache - REMOVED (LLM has native embeddings)
EMBEDDING_CACHE_AVAILABLE = False
get_embedding_cached = None
is_simple_query = None

# LLM Router - REMOVED (redundant)
LLM_ROUTER_AVAILABLE = False
LLMQueryRouter = None

# ============================================================
# KEPT COMPONENTS
# ============================================================

# Telemetry Recorder
try:
    from .telemetry_recorder import (
        TelemetryRecorder,
        TelemetryEntry,
        AIInteractionEntry,
        InteractionSource,
        record_telemetry,
        record_interaction,
        record_ai_interaction,
        get_telemetry_recorder,
    )
    TELEMETRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Telemetry recorder module not available: {e}")
    TELEMETRY_AVAILABLE = False
    TelemetryRecorder = None
    TelemetryEntry = None
    AIInteractionEntry = None
    InteractionSource = None
    record_telemetry = None
    record_interaction = None
    record_ai_interaction = None
    get_telemetry_recorder = None

# Governance Logger
try:
    from .governance_logger import (
        GovernanceLogger,
        ActionType,
        SeverityLevel,
        AuditEntry,
        log_to_governance,
        log_to_governance_async,
        log_to_governance_fire_and_forget,
        get_governance_logger,
        shutdown_governance_logger,
        BufferedGovernanceLogger,
        get_buffered_governance_logger,
        shutdown_buffered_governance_logger,
        log_routing_result,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Governance logger module not available: {e}")
    GOVERNANCE_AVAILABLE = False
    GovernanceLogger = None
    ActionType = None
    SeverityLevel = None
    AuditEntry = None
    log_to_governance = None
    log_to_governance_async = None
    log_to_governance_fire_and_forget = None
    get_governance_logger = None
    shutdown_governance_logger = None
    BufferedGovernanceLogger = None
    get_buffered_governance_logger = None
    shutdown_buffered_governance_logger = None
    log_routing_result = None

# Experiment Trigger
try:
    from .experiment_trigger import (
        ExperimentTrigger,
        ExperimentProposal,
        ExperimentCondition,
        should_run_experiment,
        get_experiment_trigger,
        generate_experiments_from_interactions,
    )
    EXPERIMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Experiment trigger module not available: {e}")
    EXPERIMENT_AVAILABLE = False
    ExperimentTrigger = None
    ExperimentProposal = None
    ExperimentCondition = None
    should_run_experiment = None
    get_experiment_trigger = None
    generate_experiments_from_interactions = None

# System Health Monitor
try:
    from .system_health_monitor import (
        SystemHealthMonitor,
        HealthStatus,
        get_system_health,
        check_system_health,
    )
    HEALTH_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"System health monitor module not available: {e}")
    HEALTH_MONITOR_AVAILABLE = False
    SystemHealthMonitor = None
    HealthStatus = None
    get_system_health = None
    check_system_health = None


# ============================================================
# MODULE INITIALIZATION
# ============================================================

def initialize_routing_components() -> dict:
    """
    Initialize all routing components with proper error handling.
    
    Returns:
        dict: Status of each component initialization
    """
    global _ROUTING_COMPONENTS_INITIALIZED
    
    with _ROUTING_SINGLETON_LOCK:
        if _ROUTING_COMPONENTS_INITIALIZED:
            return get_routing_status()
        
        status = {
            "telemetry": TELEMETRY_AVAILABLE,
            "governance": GOVERNANCE_AVAILABLE,
            "experiment": EXPERIMENT_AVAILABLE,
            "health_monitor": HEALTH_MONITOR_AVAILABLE,
        }
        
        # Initialize components that need explicit initialization
        if TELEMETRY_AVAILABLE:
            try:
                get_telemetry_recorder()
                logger.info("✓ Telemetry recorder initialized")
            except Exception as e:
                logger.error(f"✗ Telemetry recorder initialization failed: {e}")
                status["telemetry"] = False
        
        if GOVERNANCE_AVAILABLE:
            try:
                get_governance_logger()
                logger.info("✓ Governance logger initialized")
            except Exception as e:
                logger.error(f"✗ Governance logger initialization failed: {e}")
                status["governance"] = False
        
        _ROUTING_COMPONENTS_INITIALIZED = True
        
        available_count = sum(1 for v in status.values() if v)
        logger.info(f"Routing components initialized: {available_count}/{len(status)} available")
        
        return status


def get_routing_status() -> dict:
    """
    Get the current status of all routing components.
    
    Returns:
        dict: Component availability and statistics
    """
    return {
        "initialized": _ROUTING_COMPONENTS_INITIALIZED,
        "components": {
            "telemetry": {"available": TELEMETRY_AVAILABLE},
            "governance": {"available": GOVERNANCE_AVAILABLE},
            "experiment": {"available": EXPERIMENT_AVAILABLE},
            "health_monitor": {"available": HEALTH_MONITOR_AVAILABLE},
        },
        "removed_components": [
            "query_router",
            "llm_router", 
            "embedding_cache",
            "agent_collaboration",
            "routing_prompts",
        ],
        "architecture": "tool_based_llm_function_calling",
    }


# ============================================================
# PUBLIC API EXPORTS
# ============================================================

__all__ = [
    # Telemetry
    "TelemetryRecorder",
    "TelemetryEntry",
    "AIInteractionEntry",
    "InteractionSource",
    "record_telemetry",
    "record_interaction",
    "record_ai_interaction",
    "get_telemetry_recorder",
    "TELEMETRY_AVAILABLE",
    # Governance
    "GovernanceLogger",
    "ActionType",
    "SeverityLevel",
    "AuditEntry",
    "log_to_governance",
    "log_to_governance_async",
    "log_to_governance_fire_and_forget",
    "get_governance_logger",
    "shutdown_governance_logger",
    "BufferedGovernanceLogger",
    "get_buffered_governance_logger",
    "shutdown_buffered_governance_logger",
    "log_routing_result",
    "GOVERNANCE_AVAILABLE",
    # Experiments
    "ExperimentTrigger",
    "ExperimentProposal",
    "ExperimentCondition",
    "should_run_experiment",
    "get_experiment_trigger",
    "generate_experiments_from_interactions",
    "EXPERIMENT_AVAILABLE",
    # Health Monitor
    "SystemHealthMonitor",
    "HealthStatus",
    "get_system_health",
    "check_system_health",
    "HEALTH_MONITOR_AVAILABLE",
    # Module functions
    "initialize_routing_components",
    "get_routing_status",
    # Removed component flags (for backward compatibility)
    "QUERY_ROUTER_AVAILABLE",
    "COLLABORATION_AVAILABLE",
    "EMBEDDING_CACHE_AVAILABLE",
    "LLM_ROUTER_AVAILABLE",
]
