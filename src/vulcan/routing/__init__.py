# ============================================================
# VULCAN-AGI Query Routing and Dual-Mode Learning Integration Layer
# ============================================================
# Enterprise-grade query routing with dual-mode learning support:
# - MODE 1: User Interactions (human queries, feedback, real-world problems)
# - MODE 2: AI-to-AI Interactions (agent collaboration, tournaments, debates)
#
# PRODUCTION-READY: Thread-safe, graceful degradation, comprehensive logging
# FULLY TESTED: All components validated against platform standards
# ============================================================

"""
VULCAN Query Routing and Dual-Mode Learning Integration Layer

This module provides intelligent query routing to VULCAN's cognitive systems
with support for dual-mode learning:

MODE 1: User Interactions
    - Human queries and feedback
    - Real-world problem solving
    - Production telemetry
    - Utility memory population

MODE 2: AI-to-AI Interactions
    - Agent-to-agent collaboration
    - Arena tournaments (competitive learning)
    - Inter-agent debates and deliberation
    - Multi-agent problem solving
    - Success/risk memory population

Components:
    - QueryRouter: Dual-mode detection and task decomposition
    - AgentCollaborationManager: Multi-agent problem solving
    - TelemetryRecorder: Dual-mode telemetry for meta-learning
    - GovernanceLogger: Audit.db integration
    - ExperimentTrigger: Meta-learning experiment generation

Usage:
    from vulcan.routing import route_query, trigger_agent_collaboration

    # Route a user query
    plan = route_query("Analyze this pattern", source="user")

    # Route an agent-to-agent query
    plan = route_query("Collaborate on solution", source="agent")

    # Trigger multi-agent collaboration
    session = trigger_agent_collaboration(query, ["perception", "reasoning"])
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# SINGLETON GLOBALS FOR THREAD-SAFE COMPONENT ACCESS
# ============================================================
_ROUTING_SINGLETON_LOCK = threading.RLock()
_ROUTING_COMPONENTS_INITIALIZED = False

# ============================================================
# LAZY IMPORTS WITH GRACEFUL DEGRADATION
# ============================================================

# Query Router
try:
    from .query_router import (
        QueryAnalyzer,
        QueryPlan,
        AgentTask,
        QueryType,
        LearningMode,
        ProcessingPlan,
        GovernanceSensitivity,
        analyze_query,
        decompose_to_agent_tasks,
        route_query,
        route_query_async,
        get_query_analyzer,
        shutdown_blocking_executor,
        # FIX 2: Query Router Timeout constant
        QUERY_ROUTING_TIMEOUT_SECONDS,
    )

    QUERY_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Query router module not available: {e}")
    QUERY_ROUTER_AVAILABLE = False
    QueryAnalyzer = None
    QueryPlan = None
    AgentTask = None
    QueryType = None
    LearningMode = None
    ProcessingPlan = None
    GovernanceSensitivity = None
    analyze_query = None
    decompose_to_agent_tasks = None
    route_query = None
    route_query_async = None
    get_query_analyzer = None
    shutdown_blocking_executor = None
    QUERY_ROUTING_TIMEOUT_SECONDS = 5.0  # Default fallback

# Agent Collaboration
try:
    from .agent_collaboration import (
        AgentMessage,
        AgentCollaborationManager,
        CollaborationSession,
        MessageType,
        CollaborationStatus,
        trigger_agent_collaboration,
        get_collaboration_manager,
    )

    COLLABORATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent collaboration module not available: {e}")
    COLLABORATION_AVAILABLE = False
    AgentMessage = None
    AgentCollaborationManager = None
    CollaborationSession = None
    MessageType = None
    CollaborationStatus = None
    trigger_agent_collaboration = None
    get_collaboration_manager = None

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
        # PERFORMANCE FIX: Non-blocking buffered logging
        BufferedGovernanceLogger,
        get_buffered_governance_logger,
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
    BufferedGovernanceLogger = None
    get_buffered_governance_logger = None
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

# Embedding Cache (Performance Fix for Slow Query Routing)
try:
    from .embedding_cache import (
        get_embedding_cached,
        get_embeddings_batch_cached,
        is_simple_query,
        get_cache_stats as get_embedding_cache_stats,
        clear_cache as clear_embedding_cache,
    )

    EMBEDDING_CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Embedding cache module not available: {e}")
    EMBEDDING_CACHE_AVAILABLE = False
    get_embedding_cached = None
    get_embeddings_batch_cached = None
    is_simple_query = None
    get_embedding_cache_stats = None
    clear_embedding_cache = None


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
            "query_router": QUERY_ROUTER_AVAILABLE,
            "collaboration": COLLABORATION_AVAILABLE,
            "telemetry": TELEMETRY_AVAILABLE,
            "governance": GOVERNANCE_AVAILABLE,
            "experiment": EXPERIMENT_AVAILABLE,
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
        logger.info(
            f"Routing components initialized: {available_count}/{len(status)} available"
        )

        return status


def get_routing_status() -> dict:
    """
    Get the current status of all routing components.

    Returns:
        dict: Component availability and statistics
    """
    status = {
        "initialized": _ROUTING_COMPONENTS_INITIALIZED,
        "components": {
            "query_router": {
                "available": QUERY_ROUTER_AVAILABLE,
                "stats": None,
            },
            "collaboration": {
                "available": COLLABORATION_AVAILABLE,
                "stats": None,
            },
            "telemetry": {
                "available": TELEMETRY_AVAILABLE,
                "stats": None,
            },
            "governance": {
                "available": GOVERNANCE_AVAILABLE,
                "stats": None,
            },
            "experiment": {
                "available": EXPERIMENT_AVAILABLE,
                "stats": None,
            },
        },
    }

    # Get stats from available components
    if QUERY_ROUTER_AVAILABLE and get_query_analyzer:
        try:
            analyzer = get_query_analyzer()
            status["components"]["query_router"]["stats"] = analyzer.get_stats()
        except Exception:
            pass

    if COLLABORATION_AVAILABLE and get_collaboration_manager:
        try:
            manager = get_collaboration_manager()
            status["components"]["collaboration"]["stats"] = manager.get_stats()
        except Exception:
            pass

    if TELEMETRY_AVAILABLE and get_telemetry_recorder:
        try:
            recorder = get_telemetry_recorder()
            status["components"]["telemetry"]["stats"] = recorder.get_stats()
        except Exception:
            pass

    if GOVERNANCE_AVAILABLE and get_governance_logger:
        try:
            gov_logger = get_governance_logger()
            status["components"]["governance"]["stats"] = gov_logger.get_stats()
        except Exception:
            pass

    if EXPERIMENT_AVAILABLE and get_experiment_trigger:
        try:
            trigger = get_experiment_trigger()
            status["components"]["experiment"]["stats"] = trigger.get_stats()
        except Exception:
            pass

    return status


# ============================================================
# PUBLIC API EXPORTS
# ============================================================

__all__ = [
    # Query Router
    "QueryAnalyzer",
    "QueryPlan",
    "AgentTask",
    "QueryType",
    "LearningMode",
    "ProcessingPlan",
    "GovernanceSensitivity",
    "analyze_query",
    "decompose_to_agent_tasks",
    "route_query",
    "route_query_async",
    "get_query_analyzer",
    "shutdown_blocking_executor",
    "QUERY_ROUTING_TIMEOUT_SECONDS",  # FIX 2: Query Router Timeout constant
    # Agent Collaboration
    "AgentMessage",
    "AgentCollaborationManager",
    "CollaborationSession",
    "MessageType",
    "CollaborationStatus",
    "trigger_agent_collaboration",
    "get_collaboration_manager",
    # Telemetry
    "TelemetryRecorder",
    "TelemetryEntry",
    "AIInteractionEntry",
    "InteractionSource",
    "record_telemetry",
    "record_interaction",
    "record_ai_interaction",
    "get_telemetry_recorder",
    # Governance
    "GovernanceLogger",
    "ActionType",
    "SeverityLevel",
    "AuditEntry",
    "log_to_governance",
    "log_to_governance_async",
    "log_to_governance_fire_and_forget",
    "get_governance_logger",
    # PERFORMANCE FIX: Non-blocking buffered logging
    "BufferedGovernanceLogger",
    "get_buffered_governance_logger",
    "log_routing_result",
    # Experiments
    "ExperimentTrigger",
    "ExperimentProposal",
    "ExperimentCondition",
    "should_run_experiment",
    "get_experiment_trigger",
    "generate_experiments_from_interactions",
    # Embedding Cache (Performance Fix)
    "get_embedding_cached",
    "get_embeddings_batch_cached",
    "is_simple_query",
    "get_embedding_cache_stats",
    "clear_embedding_cache",
    "EMBEDDING_CACHE_AVAILABLE",
    # Module functions
    "initialize_routing_components",
    "get_routing_status",
    # Availability flags
    "QUERY_ROUTER_AVAILABLE",
    "COLLABORATION_AVAILABLE",
    "TELEMETRY_AVAILABLE",
    "GOVERNANCE_AVAILABLE",
    "EXPERIMENT_AVAILABLE",
]
