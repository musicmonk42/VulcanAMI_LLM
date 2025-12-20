"""
VULCAN-AGI: Volumetric Unified Learning Cognitive Architecture Network

A production-grade AGI system integrating multiple cognitive architectures:
- Memory: Hierarchical, distributed memory systems
- Reasoning: Multi-paradigm reasoning (symbolic, probabilistic, causal, analogical)
- Learning: Continual, curriculum, meta-learning with RLHF
- Safety: Comprehensive safety validation and governance
- Routing: Dual-mode query routing with collaboration support
- Curiosity Engine: Autonomous knowledge gap exploration
- Problem Decomposer: Hierarchical problem decomposition

All modules export their key classes and provide availability flags for
graceful degradation when dependencies are missing.
"""

import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "2.0.0"
__author__ = "Vulcan AI Team"

# ============================================================================
# MEMORY MODULE
# ============================================================================
try:
    from .memory import (
        Memory,
        MemoryConfig,
        MemoryType,
        HierarchicalMemory,
        DistributedMemory,
        MemoryIndex,
        VectorMemoryStore,
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        WorkingMemory,
    )

    MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory module not available: {e}")
    MEMORY_AVAILABLE = False

# ============================================================================
# REASONING MODULE
# ============================================================================
try:
    from .reasoning import (
        UnifiedReasoner,
        ReasoningType,
        ReasoningStrategy,
        ProbabilisticReasoner,
        CausalReasoner,
        SymbolicReasoner,
        AnalogicalReasoner,
        MultimodalReasoner,
        UNIFIED_AVAILABLE,
        PROBABILISTIC_AVAILABLE,
        CAUSAL_AVAILABLE,
        SYMBOLIC_AVAILABLE,
        ANALOGICAL_AVAILABLE,
        MULTIMODAL_AVAILABLE,
    )

    REASONING_AVAILABLE = UNIFIED_AVAILABLE
except ImportError as e:
    logger.warning(f"Reasoning module not available: {e}")
    REASONING_AVAILABLE = False
    UNIFIED_AVAILABLE = False
    PROBABILISTIC_AVAILABLE = False
    CAUSAL_AVAILABLE = False
    SYMBOLIC_AVAILABLE = False
    ANALOGICAL_AVAILABLE = False
    MULTIMODAL_AVAILABLE = False

# ============================================================================
# LEARNING MODULE
# ============================================================================
try:
    from .learning import (
        UnifiedLearningSystem,
        CurriculumLearner,
        LearningConfig,
        LearningMode,
    )

    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Learning module not available: {e}")
    LEARNING_AVAILABLE = False

# ============================================================================
# ROUTING MODULE - Query Routing and Dual-Mode Learning
# ============================================================================
try:
    from .routing import (
        route_query,
        analyze_query,
        QueryAnalyzer,
        QueryPlan,
        ProcessingPlan,
        QueryType,
        LearningMode as RoutingLearningMode,
        trigger_agent_collaboration,
        record_telemetry,
        log_to_governance,
        initialize_routing_components,
        get_routing_status,
        QUERY_ROUTER_AVAILABLE,
        COLLABORATION_AVAILABLE,
        TELEMETRY_AVAILABLE,
        GOVERNANCE_AVAILABLE,
    )

    ROUTING_AVAILABLE = QUERY_ROUTER_AVAILABLE
except ImportError as e:
    logger.warning(f"Routing module not available: {e}")
    ROUTING_AVAILABLE = False
    QUERY_ROUTER_AVAILABLE = False
    COLLABORATION_AVAILABLE = False
    TELEMETRY_AVAILABLE = False
    GOVERNANCE_AVAILABLE = False

# ============================================================================
# CURIOSITY ENGINE MODULE
# ============================================================================
try:
    from .curiosity_engine import (
        CuriosityEngine,
        GapAnalyzer,
        ExperimentGenerator,
        KnowledgeGap,
        Experiment,
        CURIOSITY_ENGINE_AVAILABLE,
        GAP_ANALYZER_AVAILABLE,
        EXPERIMENT_GENERATOR_AVAILABLE,
    )

    CURIOSITY_AVAILABLE = CURIOSITY_ENGINE_AVAILABLE
except ImportError as e:
    logger.warning(f"Curiosity engine module not available: {e}")
    CURIOSITY_AVAILABLE = False
    CURIOSITY_ENGINE_AVAILABLE = False
    GAP_ANALYZER_AVAILABLE = False
    EXPERIMENT_GENERATOR_AVAILABLE = False

# ============================================================================
# PROBLEM DECOMPOSER MODULE
# ============================================================================
try:
    from .problem_decomposer import (
        ProblemDecomposer,
        DecompositionMode,
        DecompositionPlan,
        ProblemExecutor,
        FallbackChain,
        DecompositionLibrary,
        PROBLEM_DECOMPOSER_AVAILABLE,
        STRATEGIES_AVAILABLE,
        EXECUTOR_AVAILABLE,
    )

    DECOMPOSER_AVAILABLE = PROBLEM_DECOMPOSER_AVAILABLE
except ImportError as e:
    logger.warning(f"Problem decomposer module not available: {e}")
    DECOMPOSER_AVAILABLE = False
    PROBLEM_DECOMPOSER_AVAILABLE = False
    STRATEGIES_AVAILABLE = False
    EXECUTOR_AVAILABLE = False

# ============================================================================
# SAFETY MODULE
# ============================================================================
try:
    from .safety import (
        SAFETY_VALIDATOR_AVAILABLE,
        GOVERNANCE_ORCHESTRATOR_AVAILABLE,
        get_safety_validator,
        SafetyUnavailable,
    )

    SAFETY_AVAILABLE = SAFETY_VALIDATOR_AVAILABLE
except ImportError as e:
    logger.warning(f"Safety module not available: {e}")
    SAFETY_AVAILABLE = False
    SAFETY_VALIDATOR_AVAILABLE = False
    GOVERNANCE_ORCHESTRATOR_AVAILABLE = False

# ============================================================================
# CONFIG MODULE
# ============================================================================
try:
    from .config import AgentConfig, get_config

    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Config module not available: {e}")
    CONFIG_AVAILABLE = False

# ============================================================================
# ORCHESTRATOR MODULE
# ============================================================================
try:
    from .orchestrator import ProductionDeployment

    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Orchestrator module not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

# ============================================================================
# MODULE STATUS
# ============================================================================


def get_vulcan_status() -> dict:
    """Get availability status of all VULCAN components."""
    return {
        "version": __version__,
        "memory": MEMORY_AVAILABLE,
        "reasoning": REASONING_AVAILABLE,
        "learning": LEARNING_AVAILABLE,
        "routing": ROUTING_AVAILABLE,
        "curiosity": CURIOSITY_AVAILABLE,
        "decomposer": DECOMPOSER_AVAILABLE,
        "safety": SAFETY_AVAILABLE,
        "config": CONFIG_AVAILABLE,
        "orchestrator": ORCHESTRATOR_AVAILABLE,
    }


def print_vulcan_status():
    """Print a formatted status report of all VULCAN components."""
    status = get_vulcan_status()
    print("\n" + "=" * 60)
    print(f"VULCAN-AGI Module Status (v{status['version']})")
    print("=" * 60)

    for component, available in status.items():
        if component == "version":
            continue
        status_icon = "✓" if available else "✗"
        status_text = "Available" if available else "Not Available"
        print(f"{status_icon} {component.capitalize():15s}: {status_text}")

    print("=" * 60)

    available_count = sum(1 for k, v in status.items() if k != "version" and v)
    total_count = len(status) - 1  # Exclude version
    print(f"Total: {available_count}/{total_count} components available")
    print("=" * 60 + "\n")


# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Status functions
    "get_vulcan_status",
    "print_vulcan_status",
    # Availability flags
    "MEMORY_AVAILABLE",
    "REASONING_AVAILABLE",
    "LEARNING_AVAILABLE",
    "ROUTING_AVAILABLE",
    "CURIOSITY_AVAILABLE",
    "DECOMPOSER_AVAILABLE",
    "SAFETY_AVAILABLE",
    "CONFIG_AVAILABLE",
    "ORCHESTRATOR_AVAILABLE",
]

# Log initialization
available_count = sum(
    [
        MEMORY_AVAILABLE,
        REASONING_AVAILABLE,
        LEARNING_AVAILABLE,
        ROUTING_AVAILABLE,
        CURIOSITY_AVAILABLE,
        DECOMPOSER_AVAILABLE,
        SAFETY_AVAILABLE,
    ]
)

logger.info(
    f"VULCAN-AGI v{__version__} initialized: {available_count}/7 core modules available"
)
