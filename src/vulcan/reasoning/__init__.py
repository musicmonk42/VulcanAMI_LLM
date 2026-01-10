"""
Vulcan Reasoning Module

Exports all reasoning components and types for the unified reasoning system.
This module provides access to multiple reasoning paradigms and orchestration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Core Types - Always available
# CIRCULAR IMPORT FIX: ReasoningStrategy is now in reasoning_types.py to
# prevent circular imports when agent_pool.py imports from src.vulcan.reasoning
# ============================================================================
try:
    from .reasoning_types import (
        ModalityType,
        PortfolioStrategy,
        ReasoningChain,
        ReasoningResult,
        ReasoningStep,
        ReasoningType,
        ReasoningStrategy,  # CIRCULAR IMPORT FIX: Now from reasoning_types
        SelectionMode,
        UtilityContext,
    )

    TYPES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"Core types import failed: {e}")
    TYPES_AVAILABLE = False
    # Create minimal fallbacks
    from enum import Enum

    class ReasoningType(Enum):
        SYMBOLIC = "symbolic"
        CAUSAL = "causal"
        PROBABILISTIC = "probabilistic"
        ANALOGICAL = "analogical"
        MULTIMODAL = "multimodal"

    class ReasoningStrategy(Enum):
        """Fallback ReasoningStrategy enum.
        
        Note: This fallback is kept for defensive programming in case
        reasoning_types.py fails to import (e.g., due to missing dependencies).
        In normal operation, ReasoningStrategy is imported from reasoning_types.py.
        """

        SEQUENTIAL = "sequential"
        PARALLEL = "parallel"
        ENSEMBLE = "ensemble"
        HIERARCHICAL = "hierarchical"
        ADAPTIVE = "adaptive"
        HYBRID = "hybrid"
        PORTFOLIO = "portfolio"
        UTILITY_BASED = "utility_based"


# ============================================================================
# Probabilistic Reasoning - Core component
# ============================================================================
try:
    from .probabilistic_reasoning import ProbabilisticReasoner

    PROBABILISTIC_AVAILABLE = True
except ImportError as e:
    logger.error(f"Probabilistic reasoning import failed: {e}")
    ProbabilisticReasoner = None
    PROBABILISTIC_AVAILABLE = False

# ============================================================================
# Causal Reasoning - Core component
# ============================================================================
try:
    from .causal_reasoning import CausalEdge, CausalReasoner, EnhancedCausalReasoning

    CAUSAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Causal reasoning import failed: {e}")
    CausalReasoner = None
    EnhancedCausalReasoning = None
    CausalEdge = None
    CAUSAL_AVAILABLE = False

# ============================================================================
# Symbolic Reasoning - Core component (FIXED: Ultra-defensive imports)
# ============================================================================
try:
    # Primary import from subdirectory - most critical
    from .symbolic.reasoner import SymbolicReasoner

    SYMBOLIC_AVAILABLE = True
    logger.info("Symbolic reasoning loaded successfully from symbolic.reasoner")

    # Try to import core types - but don't fail if they're not there
    # Import each one individually with its own try/except
    try:
        from .symbolic.core import Clause
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Clause: {e}")
        Clause = None

    try:
        from .symbolic.core import Literal
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Literal: {e}")
        Literal = None

    try:
        from .symbolic.core import Constant
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Constant: {e}")
        Constant = None

    try:
        from .symbolic.core import Variable
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Variable: {e}")
        Variable = None

    # Try extended symbolic reasoners - optional
    try:
        from .symbolic.reasoner import BayesianReasoner
    except (ImportError, AttributeError) as e:
        logger.debug(f"BayesianReasoner not available: {e}")
        BayesianReasoner = None

    try:
        from .symbolic.reasoner import EnhancedSymbolicReasoner
    except (ImportError, AttributeError) as e:
        logger.debug(f"EnhancedSymbolicReasoner not available: {e}")
        EnhancedSymbolicReasoner = None

    try:
        from .symbolic.reasoner import HybridReasoner
    except (ImportError, AttributeError) as e:
        logger.debug(f"HybridReasoner not available: {e}")
        HybridReasoner = None

except ImportError as e:
    logger.error(f"Symbolic reasoning import failed: {e}")
    SymbolicReasoner = None
    BayesianReasoner = None
    EnhancedSymbolicReasoner = None
    HybridReasoner = None
    Clause = None
    Literal = None
    Constant = None
    Variable = None
    SYMBOLIC_AVAILABLE = False
except Exception as e:
    logger.error(f"Unexpected error loading symbolic reasoning: {e}")
    SymbolicReasoner = None
    BayesianReasoner = None
    EnhancedSymbolicReasoner = None
    HybridReasoner = None
    Clause = None
    Literal = None
    Constant = None
    Variable = None
    SYMBOLIC_AVAILABLE = False

# ============================================================================
# Analogical Reasoning - Core component
# ============================================================================
try:
    from .analogical_reasoning import (
        AnalogicalReasoner,
        AnalogicalReasoningEngine,
        Entity,
        Relation,
    )

    ANALOGICAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Analogical reasoning import failed: {e}")
    AnalogicalReasoningEngine = None
    AnalogicalReasoner = None
    Entity = None
    Relation = None
    ANALOGICAL_AVAILABLE = False

# ============================================================================
# Multimodal Reasoning - Core component
# ============================================================================
try:
    from .multimodal_reasoning import (
        MultimodalReasoner,
        MultiModalReasoningEngine,
        CrossModalReasoner,
    )

    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Multimodal reasoning import failed: {e}")
    MultiModalReasoningEngine = None
    MultimodalReasoner = None
    CrossModalReasoner = None
    MULTIMODAL_AVAILABLE = False

# ============================================================================
# Philosophical Reasoning - DEPRECATED: Now handled by World Model
# ============================================================================
# The PhilosophicalReasoner wrapper has been removed. Ethical/philosophical
# queries are now routed directly to the World Model which has:
# - Causal prediction via predict_interventions()
# - Multi-framework evaluation via InternalCritic
# - Goal conflict detection via GoalConflictDetector
# - Ethical boundary monitoring via EthicalBoundaryMonitor
#
# The wrapper added complexity without value - World Model already has
# all the sophisticated machinery for ethical reasoning.
# ============================================================================
PhilosophicalReasoner = None  # Deprecated - use World Model
is_philosophical_query = None  # Deprecated
PHILOSOPHICAL_AVAILABLE = False  # Philosophical queries route to World Model
logger.info("Philosophical reasoning: Routed to World Model (wrapper removed)")

# ============================================================================
# Note: Cryptographic Engine - Deterministic hash/encoding computations
# ============================================================================
try:
    from .cryptographic_engine import (
        CryptographicEngine,
        CryptoOperation,
        CryptoResult,
        compute_crypto,
        get_crypto_engine,
    )

    CRYPTOGRAPHIC_AVAILABLE = True
    logger.info("Note: Cryptographic engine loaded successfully")
except ImportError as e:
    logger.warning(f"Cryptographic engine import failed: {e}")
    CryptographicEngine = None
    CryptoOperation = None
    CryptoResult = None
    compute_crypto = None
    get_crypto_engine = None
    CRYPTOGRAPHIC_AVAILABLE = False

# ============================================================================
# Unified Reasoner - Main orchestrator (critical)
# ============================================================================
try:
    from .unified import UnifiedReasoner

    UNIFIED_AVAILABLE = True
except ImportError as e:
    logger.critical(f"UnifiedReasoner import failed: {e}")
    UnifiedReasoner = None
    UNIFIED_AVAILABLE = False

# ============================================================================
# Explainability - Optional enhancement
# ============================================================================
try:
    from .reasoning_explainer import ReasoningExplainer, SafetyAwareReasoning

    EXPLAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Reasoning explainer import failed: {e}")
    ReasoningExplainer = None
    SafetyAwareReasoning = None
    EXPLAINER_AVAILABLE = False

# ============================================================================
# Tool Selection Components - Optional but recommended
# ============================================================================
try:
    from .selection.admission_control import (
        AdmissionControlIntegration,
        RequestPriority,
    )
    from .selection.tool_selector import SelectionRequest, SelectionResult, ToolSelector

    SELECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Selection components import failed: {e}")
    ToolSelector = None
    SelectionRequest = None
    SelectionResult = None
    AdmissionControlIntegration = None
    RequestPriority = None
    SELECTION_AVAILABLE = False

# ============================================================================
# Contextual Bandit - Advanced feature
# ============================================================================
try:
    from .contextual_bandit import (
        AdaptiveBanditOrchestrator,
        BanditAction,
        BanditContext,
        BanditFeedback,
    )

    BANDIT_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Contextual bandit import failed: {e}")
    AdaptiveBanditOrchestrator = None
    BanditContext = None
    BanditFeedback = None
    BanditAction = None
    BANDIT_AVAILABLE = False

# ============================================================================
# Public API - What users can import
# ============================================================================
__all__ = [
    # ===== Core Types =====
    "ReasoningType",
    "ReasoningStep",
    "ReasoningChain",
    "ReasoningResult",
    "ModalityType",
    "SelectionMode",
    "PortfolioStrategy",
    "UtilityContext",
    "ReasoningStrategy",
    # ===== Main Orchestrator =====
    "UnifiedReasoner",
    # ===== Individual Reasoners =====
    "ProbabilisticReasoner",
    "CausalReasoner",
    "EnhancedCausalReasoning",
    "SymbolicReasoner",
    "BayesianReasoner",
    "EnhancedSymbolicReasoner",
    "HybridReasoner",  # ADDED: Export HybridReasoner for hybrid reasoning support
    "AnalogicalReasoningEngine",
    "AnalogicalReasoner",
    "MultiModalReasoningEngine",
    "MultimodalReasoner",
    "CrossModalReasoner",  # ADDED: Export CrossModalReasoner
    "PhilosophicalReasoner",  # DEPRECATED: Returns None, use World Model
    "is_philosophical_query",  # DEPRECATED: Returns None
    # ===== Note: Cryptographic Engine =====
    "CryptographicEngine",
    "CryptoOperation",
    "CryptoResult",
    "compute_crypto",
    "get_crypto_engine",
    # ===== Symbolic Types =====
    "Clause",
    "Literal",
    "Constant",
    "Variable",
    # ===== Analogical Types =====
    "Entity",
    "Relation",
    # ===== Causal Types =====
    "CausalEdge",
    # ===== Explainability =====
    "ReasoningExplainer",
    "SafetyAwareReasoning",
    # ===== Tool Selection (Optional) =====
    "ToolSelector",
    "SelectionRequest",
    "SelectionResult",
    "AdmissionControlIntegration",
    "RequestPriority",
    # ===== Bandit Learning (Optional) =====
    "AdaptiveBanditOrchestrator",
    "BanditContext",
    "BanditFeedback",
    "BanditAction",
    # ===== Reasoning Integration (Query Flow Fix) =====
    "ReasoningIntegration",
    "IntegrationReasoningResult",
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_integration",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    "INTEGRATION_AVAILABLE",
    # ===== SystemObserver Integration (BUG #3 FIX) =====
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
    # ===== Availability Flags =====
    "PROBABILISTIC_AVAILABLE",
    "CAUSAL_AVAILABLE",
    "SYMBOLIC_AVAILABLE",
    "ANALOGICAL_AVAILABLE",
    "MULTIMODAL_AVAILABLE",
    "PHILOSOPHICAL_AVAILABLE",  # Deprecated - always False, use World Model
    "CRYPTOGRAPHIC_AVAILABLE",  # Note: Add availability flag
    "UNIFIED_AVAILABLE",
    "EXPLAINER_AVAILABLE",
    "SELECTION_AVAILABLE",
    "BANDIT_AVAILABLE",
    # ===== Mathematical Verification =====
    "MathematicalVerificationEngine",
    "MathematicalToolOrchestrator",
    "MathErrorType",
    "MathVerificationStatus",
    "MathProblem",
    "MathSolution",
    "VerificationResult",
    "BayesianProblem",
    "MATHEMATICAL_VERIFICATION_AVAILABLE",
    # ===== Mathematical Computation =====
    "MathematicalComputationTool",
    "ProblemType",
    "SolutionStrategy",
    "ComputationResult",
    "ProblemClassification",
    "ProblemClassifier",
    "CodeTemplates",
    "create_mathematical_computation_tool",
    "MATHEMATICAL_COMPUTATION_AVAILABLE",
    "MATH_EXECUTION_AVAILABLE",
]


# ============================================================================
# Module Status Report
# ============================================================================
def get_module_status() -> dict:
    """Get availability status of all reasoning components."""
    # Note: "language" reasoning is provided by the symbolic reasoner
    # which handles natural language parsing and logical formalization
    language_available = SYMBOLIC_AVAILABLE or UNIFIED_AVAILABLE
    return {
        "probabilistic": PROBABILISTIC_AVAILABLE,
        "causal": CAUSAL_AVAILABLE,
        "symbolic": SYMBOLIC_AVAILABLE,
        "analogical": ANALOGICAL_AVAILABLE,
        "multimodal": MULTIMODAL_AVAILABLE,
        "philosophical": False,  # Deprecated - use World Model for ethical reasoning
        "unified": UNIFIED_AVAILABLE,
        "explainer": EXPLAINER_AVAILABLE,
        "selection": SELECTION_AVAILABLE,
        "bandit": BANDIT_AVAILABLE,
        "mathematical_verification": MATHEMATICAL_VERIFICATION_AVAILABLE,
        "mathematical_computation": MATHEMATICAL_COMPUTATION_AVAILABLE,
        "language": language_available,  # Language reasoning via symbolic/unified
        "world_model_ethical": True,  # NEW: World Model handles ethical reasoning
    }


def print_module_status():
    """Print a formatted status report of all components."""
    status = get_module_status()
    print("\n" + "=" * 60)
    print("VULCAN Reasoning Module Status")
    print("=" * 60)

    for component, available in status.items():
        status_icon = "✓" if available else "✗"
        status_text = "Available" if available else "Not Available"
        print(f"{status_icon} {component.capitalize():15s}: {status_text}")

    print("=" * 60)

    available_count = sum(status.values())
    total_count = len(status)
    print(f"Total: {available_count}/{total_count} components available")
    print("=" * 60 + "\n")


def create_unified_reasoner(
    config: Optional[dict] = None,
    enable_learning: bool = True,
    enable_safety: bool = True,
) -> Optional["UnifiedReasoner"]:
    """
    Convenience function to get or create a UnifiedReasoner with error handling.

    Note: Now uses singleton pattern to prevent re-initialization per query.
    The first call with specific config creates the singleton instance; subsequent
    calls return the cached instance (ignoring config).

    Args:
        config: Configuration dictionary (only used on first call)
        enable_learning: Enable learning components (only used on first call)
        enable_safety: Enable safety validation (only used on first call)

    Returns:
        UnifiedReasoner instance (singleton) or None if unavailable
    """
    if not UNIFIED_AVAILABLE:
        logger.error("Cannot create UnifiedReasoner - component not available")
        return None

    # Helper function to create fallback instance
    def _create_fallback_instance():
        logger.warning("Singleton UnifiedReasoner unavailable - creating new instance")
        return UnifiedReasoner(
            config=config, enable_learning=enable_learning, enable_safety=enable_safety
        )

    try:
        # Note: Use singleton to prevent re-initialization per query
        from .singletons import get_unified_reasoner
        reasoner = get_unified_reasoner(
            config=config, enable_learning=enable_learning, enable_safety=enable_safety
        )
        if reasoner is not None:
            return reasoner
        
        # Fallback to direct instantiation if singleton returns None
        return _create_fallback_instance()
    except ImportError:
        # singletons module not available, fall back to direct instantiation
        return _create_fallback_instance()
    except Exception as e:
        logger.error(f"Failed to create UnifiedReasoner: {e}")
        return None


# Export helpers
__all__.extend(["get_module_status", "print_module_status", "create_unified_reasoner"])

# ============================================================================
# Validation - Warn if critical components are missing
# ============================================================================
if not UNIFIED_AVAILABLE:
    logger.critical(
        "UnifiedReasoner not available! This is the main orchestrator. "
        "Most functionality will be unavailable."
    )

# Log successful initialization
available_reasoners = sum(
    [
        PROBABILISTIC_AVAILABLE,
        CAUSAL_AVAILABLE,
        SYMBOLIC_AVAILABLE,
        ANALOGICAL_AVAILABLE,
        MULTIMODAL_AVAILABLE,
    ]
)

logger.info(
    f"Vulcan Reasoning Module initialized: {available_reasoners}/5 reasoners available"
)

# ============================================================================
# Reasoning Integration - Query Flow Integration (FIX: Wire into Query Flow)
# ============================================================================
try:
    from .reasoning_integration import (
        ReasoningIntegration,
        ReasoningResult as IntegrationReasoningResult,
        apply_reasoning,
        run_portfolio_reasoning,
        get_reasoning_integration,
        get_reasoning_statistics,
        shutdown_reasoning,
        # BUG #3 FIX: SystemObserver integration functions
        observe_query_start,
        observe_engine_result,
        observe_outcome,
        observe_validation_failure,
        observe_error,
    )

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Reasoning integration import failed: {e}")
    ReasoningIntegration = None
    IntegrationReasoningResult = None
    apply_reasoning = None
    run_portfolio_reasoning = None
    get_reasoning_integration = None
    get_reasoning_statistics = None
    shutdown_reasoning = None
    # BUG #3 FIX: No-op fallbacks for observer functions
    observe_query_start = None
    observe_engine_result = None
    observe_outcome = None
    observe_validation_failure = None
    observe_error = None
    INTEGRATION_AVAILABLE = False

# ============================================================================
# Mathematical Verification - SOTA Mathematical Reasoning
# ============================================================================
try:
    from .mathematical_verification import (
        MathematicalVerificationEngine,
        MathematicalToolOrchestrator,
        MathErrorType,
        MathVerificationStatus,
        MathProblem,
        MathSolution,
        VerificationResult,
        BayesianProblem,
    )

    MATHEMATICAL_VERIFICATION_AVAILABLE = True
    logger.info("Mathematical verification engine loaded successfully")
except ImportError as e:
    logger.warning(f"Mathematical verification import failed: {e}")
    MathematicalVerificationEngine = None
    MathematicalToolOrchestrator = None
    MathErrorType = None
    MathVerificationStatus = None
    MathProblem = None
    MathSolution = None
    VerificationResult = None
    BayesianProblem = None
    MATHEMATICAL_VERIFICATION_AVAILABLE = False

# ============================================================================
# Mathematical Computation Tool - SOTA Symbolic Computation
# ============================================================================
try:
    from .mathematical_computation import (
        MathematicalComputationTool,
        ProblemType,
        SolutionStrategy,
        ComputationResult,
        ProblemClassification,
        ProblemClassifier,
        CodeTemplates,
        create_mathematical_computation_tool,
        SAFE_EXECUTION_AVAILABLE as MATH_EXECUTION_AVAILABLE,
    )

    MATHEMATICAL_COMPUTATION_AVAILABLE = True
    logger.info("Mathematical computation tool loaded successfully")
except ImportError as e:
    logger.warning(f"Mathematical computation import failed: {e}")
    MathematicalComputationTool = None
    ProblemType = None
    SolutionStrategy = None
    ComputationResult = None
    ProblemClassification = None
    ProblemClassifier = None
    CodeTemplates = None
    create_mathematical_computation_tool = None
    MATH_EXECUTION_AVAILABLE = False
    MATHEMATICAL_COMPUTATION_AVAILABLE = False


# ============================================================================
# REASONING_ENGINES Registry - Maps engine names to classes
# ============================================================================
# This registry provides a central mapping of reasoning engine names to their
# implementation classes. It's used by the ToolSelector and PortfolioExecutor
# to instantiate the appropriate reasoning engine for a given query type.
#
# FIX TASK 3: Register all available reasoning engines including previously
# missing ones (analogical, language, multimodal, world_model).

def _build_reasoning_engines_registry():
    """Build the REASONING_ENGINES registry dynamically based on availability."""
    engines = {}
    
    # Core reasoning engines
    if PROBABILISTIC_AVAILABLE and ProbabilisticReasoner is not None:
        engines['probabilistic'] = ProbabilisticReasoner
        
    if CAUSAL_AVAILABLE and CausalReasoner is not None:
        engines['causal'] = CausalReasoner
        
    if SYMBOLIC_AVAILABLE and SymbolicReasoner is not None:
        engines['symbolic'] = SymbolicReasoner
        
    # FIX TASK 3: Register previously missing engines
    if ANALOGICAL_AVAILABLE and AnalogicalReasoningEngine is not None:
        engines['analogical'] = AnalogicalReasoningEngine
        
    if MULTIMODAL_AVAILABLE and MultiModalReasoningEngine is not None:
        engines['multimodal'] = MultiModalReasoningEngine
        
    # PHILOSOPHICAL ENGINE REMOVED: Ethical reasoning now handled by World Model
    # The World Model has: predict_interventions(), InternalCritic, GoalConflictDetector,
    # EthicalBoundaryMonitor - all the machinery needed for ethical reasoning.
    # Route 'philosophical' queries to 'world_model' instead.
        
    if MATHEMATICAL_COMPUTATION_AVAILABLE and MathematicalComputationTool is not None:
        engines['mathematical'] = MathematicalComputationTool
        
    if CRYPTOGRAPHIC_AVAILABLE and CryptographicEngine is not None:
        engines['cryptographic'] = CryptographicEngine
    
    # World model will be added as an adapter when available
    # (world_model is imported separately from vulcan.world_model)
    
    return engines

# Build the registry
REASONING_ENGINES = _build_reasoning_engines_registry()

# Export the registry
__all__.extend(["REASONING_ENGINES"])

# Log registered engines
logger.info(
    f"REASONING_ENGINES registry: {len(REASONING_ENGINES)} engines registered: "
    f"{list(REASONING_ENGINES.keys())}"
)

# ============================================================================
# Version Info
# ============================================================================
__version__ = "1.0.0"
__author__ = "Vulcan AI Team"
__description__ = (
    "Unified reasoning system with multiple paradigms and adaptive tool selection"
)
