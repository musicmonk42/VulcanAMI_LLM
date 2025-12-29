"""
VULCAN-AGI Problem Decomposer Module
Hierarchical problem decomposition and solution execution

Components:
- ProblemDecomposer: Main orchestrator for problem decomposition
- DecompositionStrategies: Various strategies for decomposition
- ProblemExecutor: Solution execution engine
- FallbackChain: Graceful degradation and fallback handling
- DecompositionLibrary: Pattern library for known decomposition patterns
- PrincipleLearner: Learning decomposition principles from experience

Note: This module uses graceful degradation. When dependencies like numpy
are not available, classes will be None but availability flags will indicate
the status. Always check availability flags (e.g., PROBLEM_DECOMPOSER_AVAILABLE)
before using the classes in this module.
"""

import logging

logger = logging.getLogger(__name__)

# Core types and classes from problem_decomposer_core
try:
    from .problem_decomposer_core import (
        DecompositionMode,
        DecompositionPlan,
        DecompositionStep,
        DomainDataCategory,
        DomainSelector,
        ExecutionOutcome,
        LearningGap,
        PerformanceTracker,
        ProblemComplexity,
        ProblemDecomposer,
        ProblemGraph,
        ProblemSignature,
        StrategyProfiler,
    )

    PROBLEM_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Problem decomposer core not available: {e}")
    PROBLEM_DECOMPOSER_AVAILABLE = False
    ProblemDecomposer = None
    DecompositionMode = None
    ProblemComplexity = None
    DomainDataCategory = None
    ProblemSignature = None
    DecompositionStep = None
    ProblemGraph = None
    DecompositionPlan = None
    ExecutionOutcome = None
    LearningGap = None
    PerformanceTracker = None
    StrategyProfiler = None
    DomainSelector = None

# Decomposition strategies
try:
    from .decomposition_strategies import (
        AnalogicalDecomposition,
        BruteForceSearch,
        DecompositionResult,
        DecompositionStrategy,
        ExactDecomposition,
        PatternMatch,
        SemanticDecomposition,
        StrategyType,
        StructuralDecomposition,
        SyntheticBridging,
    )

    STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Decomposition strategies not available: {e}")
    STRATEGIES_AVAILABLE = False
    DecompositionStrategy = None
    StrategyType = None
    DecompositionResult = None
    PatternMatch = None
    ExactDecomposition = None
    SemanticDecomposition = None
    StructuralDecomposition = None
    SyntheticBridging = None
    AnalogicalDecomposition = None
    BruteForceSearch = None

# Problem executor
try:
    from .problem_executor import (
        ExecutionStrategy,
        ProblemExecutor,
        SolutionResult,
        SolutionType,
    )

    EXECUTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Problem executor not available: {e}")
    EXECUTOR_AVAILABLE = False
    ProblemExecutor = None
    SolutionType = None
    ExecutionStrategy = None
    SolutionResult = None

# Fallback chain
try:
    from .fallback_chain import (
        ComponentType,
        DecompositionComponent,
        DecompositionFailure,
        ExecutionPlan,
        FallbackChain,
        FailureType,
        StrategyStatus,
    )

    FALLBACK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Fallback chain not available: {e}")
    FALLBACK_AVAILABLE = False
    FallbackChain = None
    StrategyStatus = None
    FailureType = None
    ComponentType = None
    DecompositionComponent = None
    DecompositionFailure = None
    ExecutionPlan = None

# Decomposition library
try:
    from .decomposition_library import (
        Context,
        DecompositionLibrary,
        DecompositionPrinciple,
        DomainCategory,
        Pattern,
        PatternPerformance,
        PatternStatus,
        StratifiedDecompositionLibrary,
    )

    LIBRARY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Decomposition library not available: {e}")
    LIBRARY_AVAILABLE = False
    DecompositionLibrary = None
    StratifiedDecompositionLibrary = None
    PatternStatus = None
    DomainCategory = None
    Pattern = None
    Context = None
    DecompositionPrinciple = None
    PatternPerformance = None

# Principle learner
try:
    from .principle_learner import (
        DecompositionToTraceConverter,
        DictObject,
        PrincipleLearner,
        PrinciplePromoter,
        PromotionCandidate,
    )

    PRINCIPLE_LEARNER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Principle learner not available: {e}")
    PRINCIPLE_LEARNER_AVAILABLE = False
    PrincipleLearner = None
    PrinciplePromoter = None
    PromotionCandidate = None
    DecompositionToTraceConverter = None
    DictObject = None

# Adaptive thresholds
try:
    from .adaptive_thresholds import (
        AdaptiveThresholds,
        PerformanceRecord,
        StrategyProfile,
        ThresholdConfig,
        ThresholdType,
    )

    ADAPTIVE_THRESHOLDS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive thresholds not available: {e}")
    ADAPTIVE_THRESHOLDS_AVAILABLE = False
    AdaptiveThresholds = None
    ThresholdType = None
    ThresholdConfig = None
    PerformanceRecord = None
    StrategyProfile = None

# Learning integration
try:
    from .learning_integration import (
        DecompositionDifficultyEstimator,
        IntegratedLearningCoordinator,
        ProblemToExperienceConverter,
        RLHFFeedbackRouter,
        UnifiedDecomposerLearner,
    )

    LEARNING_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Learning integration not available: {e}")
    LEARNING_INTEGRATION_AVAILABLE = False
    UnifiedDecomposerLearner = None
    IntegratedLearningCoordinator = None
    RLHFFeedbackRouter = None
    DecompositionDifficultyEstimator = None
    ProblemToExperienceConverter = None

# Decomposer bootstrap
try:
    from .decomposer_bootstrap import DecomposerBootstrap

    BOOTSTRAP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Decomposer bootstrap not available: {e}")
    BOOTSTRAP_AVAILABLE = False
    DecomposerBootstrap = None

# Mathematical Decomposer - SOTA mathematical verification integration
try:
    from .mathematical_decomposer import (
        EnhancedMathematicalDecomposer,
        MathematicalProblemContext,
        detect_mathematical_problem,
        MATH_VERIFICATION_AVAILABLE,
    )

    MATHEMATICAL_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mathematical decomposer not available: {e}")
    MATHEMATICAL_DECOMPOSER_AVAILABLE = False
    EnhancedMathematicalDecomposer = None
    MathematicalProblemContext = None
    detect_mathematical_problem = None
    MATH_VERIFICATION_AVAILABLE = False

__all__ = [
    # Core Problem Decomposer
    "ProblemDecomposer",
    "DecompositionMode",
    "ProblemComplexity",
    "DomainDataCategory",
    "ProblemSignature",
    "DecompositionStep",
    "ProblemGraph",
    "DecompositionPlan",
    "ExecutionOutcome",
    "LearningGap",
    "PerformanceTracker",
    "StrategyProfiler",
    "DomainSelector",
    # Strategies
    "DecompositionStrategy",
    "StrategyType",
    "DecompositionResult",
    "PatternMatch",
    "ExactDecomposition",
    "SemanticDecomposition",
    "StructuralDecomposition",
    "SyntheticBridging",
    "AnalogicalDecomposition",
    "BruteForceSearch",
    # Executor
    "ProblemExecutor",
    "SolutionType",
    "ExecutionStrategy",
    "SolutionResult",
    # Fallback
    "FallbackChain",
    "StrategyStatus",
    "FailureType",
    "ComponentType",
    "DecompositionComponent",
    "DecompositionFailure",
    "ExecutionPlan",
    # Library
    "DecompositionLibrary",
    "StratifiedDecompositionLibrary",
    "PatternStatus",
    "DomainCategory",
    "Pattern",
    "Context",
    "DecompositionPrinciple",
    "PatternPerformance",
    # Principle Learner
    "PrincipleLearner",
    "PrinciplePromoter",
    "PromotionCandidate",
    "DecompositionToTraceConverter",
    "DictObject",
    # Adaptive Thresholds
    "AdaptiveThresholds",
    "ThresholdType",
    "ThresholdConfig",
    "PerformanceRecord",
    "StrategyProfile",
    # Learning Integration
    "UnifiedDecomposerLearner",
    "IntegratedLearningCoordinator",
    "RLHFFeedbackRouter",
    "DecompositionDifficultyEstimator",
    "ProblemToExperienceConverter",
    # Bootstrap
    "DecomposerBootstrap",
    # Mathematical Decomposer
    "EnhancedMathematicalDecomposer",
    "MathematicalProblemContext",
    "detect_mathematical_problem",
    # Availability Flags
    "PROBLEM_DECOMPOSER_AVAILABLE",
    "STRATEGIES_AVAILABLE",
    "EXECUTOR_AVAILABLE",
    "FALLBACK_AVAILABLE",
    "LIBRARY_AVAILABLE",
    "PRINCIPLE_LEARNER_AVAILABLE",
    "ADAPTIVE_THRESHOLDS_AVAILABLE",
    "LEARNING_INTEGRATION_AVAILABLE",
    "BOOTSTRAP_AVAILABLE",
    "MATHEMATICAL_DECOMPOSER_AVAILABLE",
    "MATH_VERIFICATION_AVAILABLE",
]

# Version info
__version__ = "1.0.0"
