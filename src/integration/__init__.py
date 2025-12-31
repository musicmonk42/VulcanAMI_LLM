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
]

logger.debug(f"VULCAN Integration package v{__version__} loaded")
