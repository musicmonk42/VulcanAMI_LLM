# ============================================================
# VULCAN LLM Integration Package
# Core integration components connecting LLM subsystems
# ============================================================
#
# MODULES:
#     cognitive_loop         - Main cognitive generation loop
#     graphix_vulcan_bridge  - Bridge between Graphix IR and VULCAN
#     distillation_integration - Knowledge distillation system
#     parallel_candidate_scorer - Parallel token scoring
#     speculative_helpers    - Speculative decoding utilities
#     token_consensus_adapter - Token consensus mechanisms
#     gvulcan_integration    - Unified gvulcan component integration
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Core components
try:
    from src.integration.cognitive_loop import (
        CognitiveLoop,
        LoopRuntimeConfig,
        LoopSamplingConfig,
    )
    COGNITIVE_LOOP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CognitiveLoop not available: {e}")
    COGNITIVE_LOOP_AVAILABLE = False
    CognitiveLoop = None
    LoopRuntimeConfig = None
    LoopSamplingConfig = None

try:
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge
    BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphixVulcanBridge not available: {e}")
    BRIDGE_AVAILABLE = False
    GraphixVulcanBridge = None

try:
    from src.integration.distillation_integration import DistillationSystem
    DISTILLATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DistillationSystem not available: {e}")
    DISTILLATION_SYSTEM_AVAILABLE = False
    DistillationSystem = None

try:
    from src.integration.parallel_candidate_scorer import ParallelCandidateScorer
    PARALLEL_SCORER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ParallelCandidateScorer not available: {e}")
    PARALLEL_SCORER_AVAILABLE = False
    ParallelCandidateScorer = None

try:
    from src.integration.token_consensus_adapter import TokenConsensusAdapter
    CONSENSUS_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TokenConsensusAdapter not available: {e}")
    CONSENSUS_ADAPTER_AVAILABLE = False
    TokenConsensusAdapter = None

# GVulcan Integration
try:
    from src.integration.gvulcan_integration import (
        GVulcanIntegration,
        GVULCAN_AVAILABLE,
        GVULCAN_COMPONENTS,
        get_integration,
        get_component_status,
        is_component_available,
        reset_integration,
    )
except ImportError as e:
    logger.warning(f"gvulcan_integration not available: {e}")
    GVULCAN_AVAILABLE = False
    GVulcanIntegration = None
    GVULCAN_COMPONENTS = {}
    get_integration = None
    get_component_status = None
    is_component_available = None
    reset_integration = None

# Memory Bridge - Unified memory system integration
try:
    from src.integration.memory_bridge import (
        MemoryBridge,
        MemoryBridgeConfig,
        create_memory_bridge,
        PERSISTENT_MEMORY_AVAILABLE,
        HIERARCHICAL_MEMORY_AVAILABLE,
        GOVERNED_UNLEARNING_AVAILABLE,
        COST_OPTIMIZER_AVAILABLE,
    )
    MEMORY_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MemoryBridge not available: {e}")
    MEMORY_BRIDGE_AVAILABLE = False
    MemoryBridge = None
    MemoryBridgeConfig = None
    create_memory_bridge = None
    PERSISTENT_MEMORY_AVAILABLE = False
    HIERARCHICAL_MEMORY_AVAILABLE = False
    GOVERNED_UNLEARNING_AVAILABLE = False
    COST_OPTIMIZER_AVAILABLE = False

# GVulcan Bridge - Data quality and policy utilities
try:
    from src.integration.gvulcan_bridge import (
        GVulcanBridge,
        create_gvulcan_bridge,
        DQS_AVAILABLE,
        OPA_AVAILABLE,
    )
    GVULCAN_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"GVulcanBridge not available: {e}")
    GVULCAN_BRIDGE_AVAILABLE = False
    GVulcanBridge = None
    create_gvulcan_bridge = None
    DQS_AVAILABLE = False
    OPA_AVAILABLE = False

__all__ = [
    "__version__",
    "__author__",
    # Core Loop
    "CognitiveLoop",
    "LoopRuntimeConfig", 
    "LoopSamplingConfig",
    "COGNITIVE_LOOP_AVAILABLE",
    # Bridge
    "GraphixVulcanBridge",
    "BRIDGE_AVAILABLE",
    # Distillation
    "DistillationSystem",
    "DISTILLATION_SYSTEM_AVAILABLE",
    # Parallel Scoring
    "ParallelCandidateScorer",
    "PARALLEL_SCORER_AVAILABLE",
    # Consensus
    "TokenConsensusAdapter",
    "CONSENSUS_ADAPTER_AVAILABLE",
    # GVulcan Integration
    "GVulcanIntegration",
    "GVULCAN_AVAILABLE",
    "GVULCAN_COMPONENTS",
    "get_integration",
    "get_component_status",
    "is_component_available",
    "reset_integration",
    # Memory Bridge
    "MemoryBridge",
    "MemoryBridgeConfig",
    "create_memory_bridge",
    "MEMORY_BRIDGE_AVAILABLE",
    "PERSISTENT_MEMORY_AVAILABLE",
    "HIERARCHICAL_MEMORY_AVAILABLE",
    "GOVERNED_UNLEARNING_AVAILABLE",
    "COST_OPTIMIZER_AVAILABLE",
    # GVulcan Bridge
    "GVulcanBridge",
    "create_gvulcan_bridge",
    "GVULCAN_BRIDGE_AVAILABLE",
    "DQS_AVAILABLE",
    "OPA_AVAILABLE",
]

logger.debug(f"VULCAN Integration package v{__version__} loaded")
