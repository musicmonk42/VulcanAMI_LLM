"""
VULCAN-AGI Knowledge Crystallizer Module
Extracts, validates, stores, and applies crystallized knowledge principles

Components:
- KnowledgeCrystallizer: Main orchestrator for knowledge crystallization
- PrincipleExtractor: Extract principles from execution traces
- KnowledgeValidator: Validate crystallized knowledge
- VersionedKnowledgeBase: Versioned storage for knowledge
- CrystallizationSelector: Select appropriate crystallization methods
- ContraindicationTracker: Track negative knowledge (what NOT to do)

Note: This module uses graceful degradation. When dependencies like numpy
are not available, classes will be None but availability flags will indicate
the status. Always check availability flags (e.g., KNOWLEDGE_CRYSTALLIZER_AVAILABLE)
before using the classes in this module.
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# KNOWLEDGE CRYSTALLIZER CORE
# ============================================================================
try:
    from .knowledge_crystallizer_core import (
        ApplicationMode,
        ApplicationResult,
        CrystallizationMode,
        CrystallizationResult,
        ExecutionTrace,
        ImbalanceHandler,
        KnowledgeApplicator,
        KnowledgeCrystallizer,
    )

    KNOWLEDGE_CRYSTALLIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge crystallizer core not available: {e}")
    KNOWLEDGE_CRYSTALLIZER_AVAILABLE = False
    KnowledgeCrystallizer = None
    KnowledgeApplicator = None
    CrystallizationMode = None
    ApplicationMode = None
    CrystallizationResult = None
    ApplicationResult = None
    ExecutionTrace = None
    ImbalanceHandler = None

# ============================================================================
# PRINCIPLE EXTRACTOR
# ============================================================================
try:
    from .principle_extractor import (
        AbstractionEngine,
        CrystallizedPrinciple,
        ExtractionStrategy,
        Metric,
        MetricType,
        Pattern,
        PatternDetector,
        PatternType,
        PrincipleCandidate,
        PrincipleExtractor,
        SuccessAnalyzer,
        SuccessFactor,
    )

    PRINCIPLE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Principle extractor not available: {e}")
    PRINCIPLE_EXTRACTOR_AVAILABLE = False
    PrincipleExtractor = None
    PatternDetector = None
    SuccessAnalyzer = None
    AbstractionEngine = None
    CrystallizedPrinciple = None
    PrincipleCandidate = None
    SuccessFactor = None
    Pattern = None
    Metric = None
    PatternType = None
    MetricType = None
    ExtractionStrategy = None

# ============================================================================
# VALIDATION ENGINE
# ============================================================================
try:
    from .validation_engine import (
        DomainCategory,
        DomainTestCase,
        DomainValidator,
        FailureAnalysis,
        KnowledgeValidator,
        Principle,
        TestResult,
        ValidationLevel,
        ValidationResult,
        ValidationResults,
    )

    VALIDATION_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Validation engine not available: {e}")
    VALIDATION_ENGINE_AVAILABLE = False
    KnowledgeValidator = None
    DomainValidator = None
    ValidationLevel = None
    DomainCategory = None
    TestResult = None
    Principle = None
    FailureAnalysis = None
    ValidationResult = None
    ValidationResults = None
    DomainTestCase = None

# ============================================================================
# KNOWLEDGE STORAGE
# ============================================================================
try:
    from .knowledge_storage import (
        CompressionType,
        IndexEntry,
        KnowledgeIndex,
        KnowledgePruner,
        PrincipleVersion,
        PruneCandidate,
        QueryResult,
        SimpleVectorIndex,
        StorageBackend,
        VersionedKnowledgeBase,
    )

    KNOWLEDGE_STORAGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge storage not available: {e}")
    KNOWLEDGE_STORAGE_AVAILABLE = False
    VersionedKnowledgeBase = None
    KnowledgeIndex = None
    KnowledgePruner = None
    StorageBackend = None
    CompressionType = None
    PrincipleVersion = None
    IndexEntry = None
    PruneCandidate = None
    QueryResult = None
    SimpleVectorIndex = None

# ============================================================================
# CRYSTALLIZATION SELECTOR
# ============================================================================
try:
    from .crystallization_selector import (
        AdaptiveStrategy,
        BatchStrategy,
        CascadeAwareStrategy,
        CrystallizationMethod,
        CrystallizationSelector,
        DomainType,
        HybridStrategy,
        IncrementalStrategy,
        MethodSelection,
        SelectionStrategy,
        StandardStrategy,
        TraceCharacteristics,
        TraceComplexity,
    )

    CRYSTALLIZATION_SELECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Crystallization selector not available: {e}")
    CRYSTALLIZATION_SELECTOR_AVAILABLE = False
    CrystallizationSelector = None
    CrystallizationMethod = None
    TraceComplexity = None
    DomainType = None
    TraceCharacteristics = None
    MethodSelection = None
    SelectionStrategy = None
    StandardStrategy = None
    CascadeAwareStrategy = None
    IncrementalStrategy = None
    BatchStrategy = None
    AdaptiveStrategy = None
    HybridStrategy = None

# ============================================================================
# CONTRAINDICATION TRACKER
# ============================================================================
try:
    from .contraindication_tracker import (
        CascadeAnalyzer,
        CascadeImpact,
        Contraindication,
        ContraindicationDatabase,
        ContraindicationGraph,
        FailureMode,
        Severity,
        SimpleGraph,
    )

    CONTRAINDICATION_TRACKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Contraindication tracker not available: {e}")
    CONTRAINDICATION_TRACKER_AVAILABLE = False
    ContraindicationDatabase = None
    ContraindicationGraph = None
    CascadeAnalyzer = None
    Contraindication = None
    CascadeImpact = None
    FailureMode = None
    Severity = None
    SimpleGraph = None

__all__ = [
    # Core Knowledge Crystallizer
    "KnowledgeCrystallizer",
    "KnowledgeApplicator",
    "CrystallizationMode",
    "ApplicationMode",
    "CrystallizationResult",
    "ApplicationResult",
    "ExecutionTrace",
    "ImbalanceHandler",
    # Principle Extractor
    "PrincipleExtractor",
    "PatternDetector",
    "SuccessAnalyzer",
    "AbstractionEngine",
    "CrystallizedPrinciple",
    "PrincipleCandidate",
    "SuccessFactor",
    "Pattern",
    "Metric",
    "PatternType",
    "MetricType",
    "ExtractionStrategy",
    # Validation Engine
    "KnowledgeValidator",
    "DomainValidator",
    "ValidationLevel",
    "DomainCategory",
    "TestResult",
    "Principle",
    "FailureAnalysis",
    "ValidationResult",
    "ValidationResults",
    "DomainTestCase",
    # Knowledge Storage
    "VersionedKnowledgeBase",
    "KnowledgeIndex",
    "KnowledgePruner",
    "StorageBackend",
    "CompressionType",
    "PrincipleVersion",
    "IndexEntry",
    "PruneCandidate",
    "QueryResult",
    "SimpleVectorIndex",
    # Crystallization Selector
    "CrystallizationSelector",
    "CrystallizationMethod",
    "TraceComplexity",
    "DomainType",
    "TraceCharacteristics",
    "MethodSelection",
    "SelectionStrategy",
    "StandardStrategy",
    "CascadeAwareStrategy",
    "IncrementalStrategy",
    "BatchStrategy",
    "AdaptiveStrategy",
    "HybridStrategy",
    # Contraindication Tracker
    "ContraindicationDatabase",
    "ContraindicationGraph",
    "CascadeAnalyzer",
    "Contraindication",
    "CascadeImpact",
    "FailureMode",
    "Severity",
    "SimpleGraph",
    # Availability Flags
    "KNOWLEDGE_CRYSTALLIZER_AVAILABLE",
    "PRINCIPLE_EXTRACTOR_AVAILABLE",
    "VALIDATION_ENGINE_AVAILABLE",
    "KNOWLEDGE_STORAGE_AVAILABLE",
    "CRYSTALLIZATION_SELECTOR_AVAILABLE",
    "CONTRAINDICATION_TRACKER_AVAILABLE",
]

# Version info
__version__ = "1.0.0"
