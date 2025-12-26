"""
VULCAN-AGI Curiosity Engine Module
Autonomous curiosity-driven learning and knowledge gap exploration

Components:
- CuriosityEngine: Main orchestrator for curiosity-driven learning
- GapAnalyzer: Knowledge gap detection and analysis
- ExperimentGenerator: Experiment design for addressing gaps
- DependencyGraph: Knowledge dependency tracking
- ExplorationBudget: Resource management for exploration

Note: This module uses graceful degradation. When dependencies like numpy
are not available, classes will be None but availability flags will indicate
the status. Always check availability flags (e.g., CURIOSITY_ENGINE_AVAILABLE)
before using the classes in this module.
"""

import logging

logger = logging.getLogger(__name__)

# Core types and classes from curiosity_engine_core
try:
    from .curiosity_engine_core import (
        CuriosityEngine,
        ExperimentManager,
        ExperimentResult,
        ExplorationFrontier,
        ExplorationValueEstimator,
        GapPrioritizer,
        KnowledgeIntegrator,
        KnowledgeRegion,
        LearningPriority,
        RegionManager,
        SafeExperimentExecutor,
        StrategySelector,
    )

    CURIOSITY_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Curiosity engine core not available: {e}")
    CURIOSITY_ENGINE_AVAILABLE = False
    CuriosityEngine = None
    ExperimentManager = None
    ExperimentResult = None
    ExplorationFrontier = None
    ExplorationValueEstimator = None
    GapPrioritizer = None
    KnowledgeIntegrator = None
    KnowledgeRegion = None
    LearningPriority = None
    RegionManager = None
    SafeExperimentExecutor = None
    StrategySelector = None

# Gap analyzer components
try:
    from .gap_analyzer import (
        AnomalyAnalyzer,
        DecompositionAnalyzer,
        FailureTracker,
        GapAnalyzer,
        GapRegistry,
        GapType,
        KnowledgeGap,
        LatentGap,
        LatentGapDetector,
        Pattern,
        PatternTracker,
        PredictionAnalyzer,
        SimpleAnomalyDetector,
        TransferAnalyzer,
    )

    GAP_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Gap analyzer not available: {e}")
    GAP_ANALYZER_AVAILABLE = False
    GapAnalyzer = None
    KnowledgeGap = None
    GapType = None
    Pattern = None
    LatentGap = None
    GapRegistry = None
    SimpleAnomalyDetector = None
    FailureTracker = None
    PatternTracker = None
    DecompositionAnalyzer = None
    PredictionAnalyzer = None
    TransferAnalyzer = None
    AnomalyAnalyzer = None
    LatentGapDetector = None

# Experiment generator components
try:
    from .experiment_generator import (
        Constraint,
        DomainSimilarityCalculator,
        Experiment,
        ExperimentBuilder,
        ExperimentCache,
        ExperimentGenerator,
        ExperimentTemplates,
        ExperimentTracker,
        ExperimentType,
        FailureAnalysis,
        FailureAnalyzer,
        FailureType,
        IterativeExperimentDesigner,
        ParameterAdjuster,
        SyntheticDataGenerator,
    )

    EXPERIMENT_GENERATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Experiment generator not available: {e}")
    EXPERIMENT_GENERATOR_AVAILABLE = False
    ExperimentGenerator = None
    Experiment = None
    ExperimentType = None
    FailureType = None
    Constraint = None
    FailureAnalysis = None
    ExperimentTemplates = None
    ExperimentCache = None
    ExperimentTracker = None
    SyntheticDataGenerator = None
    DomainSimilarityCalculator = None
    ExperimentBuilder = None
    FailureAnalyzer = None
    ParameterAdjuster = None
    IterativeExperimentDesigner = None

# Dependency graph components
try:
    from .dependency_graph import (
        CacheManager,
        CycleAwareDependencyGraph,
        CycleDetector,
        DependencyAnalyzer,
        DependencyEdge,
        DependencyType,
        EvictionManager,
        GraphStorage,
        PathFinder,
        ROICalculator,
        TopologicalSorter,
    )

    DEPENDENCY_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dependency graph not available: {e}")
    DEPENDENCY_GRAPH_AVAILABLE = False
    CycleAwareDependencyGraph = None
    DependencyAnalyzer = None
    DependencyType = None
    DependencyEdge = None
    GraphStorage = None
    PathFinder = None
    CycleDetector = None
    TopologicalSorter = None
    CacheManager = None
    EvictionManager = None
    ROICalculator = None

# Exploration budget components
try:
    from .exploration_budget import (
        BudgetRecovery,
        BudgetTracker,
        CostCalibrator,
        CostEstimator,
        CostHistory,
        DynamicBudget,
        EfficiencyTracker,
        LoadAdjuster,
        ResourceAdvisor,
        ResourceMonitor,
        ResourcePredictor,
        ResourceSampler,
        ResourceSnapshot,
        ResourceType,
    )

    EXPLORATION_BUDGET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Exploration budget not available: {e}")
    EXPLORATION_BUDGET_AVAILABLE = False
    DynamicBudget = None
    ResourceMonitor = None
    ResourceType = None
    ResourceSnapshot = None
    CostHistory = None
    BudgetTracker = None
    BudgetRecovery = None
    LoadAdjuster = None
    EfficiencyTracker = None
    ResourceSampler = None
    ResourcePredictor = None
    ResourceAdvisor = None
    CostCalibrator = None
    CostEstimator = None

# Curiosity driver components (process-isolated active driver)
try:
    from .curiosity_driver import (
        CuriosityDriver,
        CuriosityDriverConfig,
        CycleOutcome,
        CycleResult,
        CycleStatisticsTracker,
        DriverState,
        ProcessPoolManager,
        create_curiosity_driver,
    )

    CURIOSITY_DRIVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Curiosity driver not available: {e}")
    CURIOSITY_DRIVER_AVAILABLE = False
    CuriosityDriver = None
    CuriosityDriverConfig = None
    CycleOutcome = None
    CycleResult = None
    CycleStatisticsTracker = None
    DriverState = None
    ProcessPoolManager = None
    create_curiosity_driver = None

# Query outcome and buffer for data flow to curiosity engine
# BUG #3 FIX: Enable data flow from main process to curiosity engine
try:
    from .query_outcome import QueryOutcome, OutcomeStatus
    from .outcome_queue import (
        OutcomeBuffer,
        get_outcome_buffer,
        reset_outcome_buffer,
        record_outcome,
        get_pending_outcome_count,
        get_outcome_statistics,
        consume_outcomes,
    )
    OUTCOME_QUEUE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Outcome queue not available: {e}")
    OUTCOME_QUEUE_AVAILABLE = False
    QueryOutcome = None
    OutcomeStatus = None
    OutcomeBuffer = None
    get_outcome_buffer = None
    reset_outcome_buffer = None
    record_outcome = None
    get_pending_outcome_count = None
    get_outcome_statistics = None
    consume_outcomes = None

# SQLite outcome bridge for cross-process data sharing (Subprocess Isolation Fix)
try:
    from .outcome_bridge import (
        record_query_outcome,
        get_recent_outcomes,
        get_unprocessed_outcomes,
        mark_outcomes_processed,
        get_outcome_statistics as get_bridge_outcome_statistics,
        analyze_outcomes_for_gaps,
        cleanup_old_outcomes,
        OutcomeStatistics,
    )
    OUTCOME_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Outcome bridge not available: {e}")
    OUTCOME_BRIDGE_AVAILABLE = False
    record_query_outcome = None
    get_recent_outcomes = None
    get_unprocessed_outcomes = None
    mark_outcomes_processed = None
    get_bridge_outcome_statistics = None
    analyze_outcomes_for_gaps = None
    cleanup_old_outcomes = None
    OutcomeStatistics = None

__all__ = [
    # Core Engine
    "CuriosityEngine",
    "ExperimentManager",
    "ExperimentResult",
    "ExplorationFrontier",
    "ExplorationValueEstimator",
    "GapPrioritizer",
    "KnowledgeIntegrator",
    "KnowledgeRegion",
    "LearningPriority",
    "RegionManager",
    "SafeExperimentExecutor",
    "StrategySelector",
    # Gap Analysis
    "GapAnalyzer",
    "KnowledgeGap",
    "GapType",
    "Pattern",
    "LatentGap",
    "GapRegistry",
    "SimpleAnomalyDetector",
    "FailureTracker",
    "PatternTracker",
    "DecompositionAnalyzer",
    "PredictionAnalyzer",
    "TransferAnalyzer",
    "AnomalyAnalyzer",
    "LatentGapDetector",
    # Experiment Generation
    "ExperimentGenerator",
    "Experiment",
    "ExperimentType",
    "FailureType",
    "Constraint",
    "FailureAnalysis",
    "ExperimentTemplates",
    "ExperimentCache",
    "ExperimentTracker",
    "SyntheticDataGenerator",
    "DomainSimilarityCalculator",
    "ExperimentBuilder",
    "FailureAnalyzer",
    "ParameterAdjuster",
    "IterativeExperimentDesigner",
    # Dependency Graph
    "CycleAwareDependencyGraph",
    "DependencyAnalyzer",
    "DependencyType",
    "DependencyEdge",
    "GraphStorage",
    "PathFinder",
    "CycleDetector",
    "TopologicalSorter",
    "CacheManager",
    "EvictionManager",
    "ROICalculator",
    # Exploration Budget
    "DynamicBudget",
    "ResourceMonitor",
    "ResourceType",
    "ResourceSnapshot",
    "CostHistory",
    "BudgetTracker",
    "BudgetRecovery",
    "LoadAdjuster",
    "EfficiencyTracker",
    "ResourceSampler",
    "ResourcePredictor",
    "ResourceAdvisor",
    "CostCalibrator",
    "CostEstimator",
    # Curiosity Driver (Active Heartbeat)
    "CuriosityDriver",
    "CuriosityDriverConfig",
    "CycleOutcome",
    "CycleResult",
    "CycleStatisticsTracker",
    "DriverState",
    "ProcessPoolManager",
    "create_curiosity_driver",
    # Query Outcome (BUG #3 FIX: Data flow to curiosity engine)
    "QueryOutcome",
    "OutcomeStatus",
    "OutcomeBuffer",
    "get_outcome_buffer",
    "reset_outcome_buffer",
    "record_outcome",
    "get_pending_outcome_count",
    "get_outcome_statistics",
    "consume_outcomes",
    # Outcome Bridge (Subprocess Isolation Fix)
    "record_query_outcome",
    "get_recent_outcomes",
    "get_unprocessed_outcomes",
    "mark_outcomes_processed",
    "get_bridge_outcome_statistics",
    "analyze_outcomes_for_gaps",
    "cleanup_old_outcomes",
    "OutcomeStatistics",
    "OUTCOME_BRIDGE_AVAILABLE",
    # Availability Flags
    "CURIOSITY_ENGINE_AVAILABLE",
    "GAP_ANALYZER_AVAILABLE",
    "EXPERIMENT_GENERATOR_AVAILABLE",
    "DEPENDENCY_GRAPH_AVAILABLE",
    "EXPLORATION_BUDGET_AVAILABLE",
    "CURIOSITY_DRIVER_AVAILABLE",
    "OUTCOME_QUEUE_AVAILABLE",
]

# Version info
__version__ = "1.0.0"
