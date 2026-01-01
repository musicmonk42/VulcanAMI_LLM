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
        """Fallback ReasoningStrategy enum"""

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
# Unified Reasoner - Main orchestrator (critical)
# ============================================================================
try:
    from .unified_reasoning import UnifiedReasoner

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
    # ===== Availability Flags =====
    "PROBABILISTIC_AVAILABLE",
    "CAUSAL_AVAILABLE",
    "SYMBOLIC_AVAILABLE",
    "ANALOGICAL_AVAILABLE",
    "MULTIMODAL_AVAILABLE",
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
    return {
        "probabilistic": PROBABILISTIC_AVAILABLE,
        "causal": CAUSAL_AVAILABLE,
        "symbolic": SYMBOLIC_AVAILABLE,
        "analogical": ANALOGICAL_AVAILABLE,
        "multimodal": MULTIMODAL_AVAILABLE,
        "unified": UNIFIED_AVAILABLE,
        "explainer": EXPLAINER_AVAILABLE,
        "selection": SELECTION_AVAILABLE,
        "bandit": BANDIT_AVAILABLE,
        "mathematical_verification": MATHEMATICAL_VERIFICATION_AVAILABLE,
        "mathematical_computation": MATHEMATICAL_COMPUTATION_AVAILABLE,
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

    ISSUE #2 FIX: Now uses singleton pattern to prevent re-initialization per query.
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
        # ISSUE #2 FIX: Use singleton to prevent re-initialization per query
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
# Version Info
# ============================================================================
__version__ = "1.0.0"
__author__ = "Vulcan AI Team"
__description__ = (
    "Unified reasoning system with multiple paradigms and adaptive tool selection"
)
