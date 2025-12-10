"""
test_knowledge_crystallizer_integration.py - Integration tests for Knowledge Crystallizer
Tests the full pipeline: Extract → Validate → Store → Apply → Track

Run with: pytest src/vulcan/tests/test_knowledge_crystallizer_integration.py -v
"""

import pytest
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import all components with correct paths
from src.vulcan.knowledge_crystallizer.principle_extractor import (
    PrincipleExtractor,
    ExecutionTrace,
    Metric,
    MetricType,
    Pattern,
    PatternType,
    ExtractionStrategy,
)
from src.vulcan.knowledge_crystallizer.validation_engine import (
    KnowledgeValidator,
    Principle,
    ValidationLevel,
    DomainTestCase,
)
from src.vulcan.knowledge_crystallizer.contraindication_tracker import (
    ContraindicationDatabase,
    ContraindicationGraph,
    CascadeAnalyzer,
    Contraindication,
    FailureMode,
    Severity,
)
from src.vulcan.knowledge_crystallizer.knowledge_storage import (
    VersionedKnowledgeBase,
    KnowledgeIndex,
    KnowledgePruner,
    StorageBackend,
    CompressionType,
)
from src.vulcan.knowledge_crystallizer.crystallization_selector import (
    CrystallizationSelector,
    CrystallizationMethod,
    TraceCharacteristics,
)
from src.vulcan.knowledge_crystallizer.knowledge_crystallizer_core import (
    KnowledgeCrystallizer,
    ExecutionTrace as CoreExecutionTrace,
    CrystallizationMode,
    ApplicationMode,
)


# ============================================================================
# MODULE-LEVEL FUNCTION (REQUIRED FOR PICKLING)
# ============================================================================


def simple_execution(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple execution function for testing

    CRITICAL: This MUST be at module level (not inside a fixture) to be picklable.
    Python's pickle module cannot serialize functions defined inside other functions.
    """
    x = inputs.get("x", 0)
    return {"result": x * 2, "success": True}


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_execution_trace():
    """Create a sample execution trace for testing"""
    return ExecutionTrace(
        trace_id="test_trace_001",
        actions=[
            {"type": "initialize", "params": {"x": 10}},
            {"type": "process", "params": {"method": "optimize"}},
            {"type": "validate", "params": {"threshold": 0.9}},
            {"type": "finalize", "params": {"output": "result"}},
        ],
        outcomes={"success": True, "value": 42, "execution_time": 0.5},
        context={"domain": "optimization", "environment": "test"},
        metrics=[
            Metric(
                name="accuracy",
                metric_type=MetricType.ACCURACY,
                value=0.95,
                threshold=0.9,
            ),
            Metric(
                name="latency", metric_type=MetricType.LATENCY, value=0.5, threshold=1.0
            ),
        ],
        success=True,
        domain="optimization",
    )


@pytest.fixture
def sample_principle():
    """
    Create a sample principle for testing

    FIX: Uses module-level simple_execution function instead of local function
    to ensure the principle can be pickled for storage tests.
    """
    test_pattern = Pattern(
        pattern_type=PatternType.SEQUENTIAL,
        components=["step1", "step2", "step3"],
        confidence=0.8,
        complexity=3,
    )

    return Principle(
        id="test_principle_001",
        core_pattern=test_pattern,
        confidence=0.75,
        applicable_domains=["optimization", "general"],
        measurement_requirements=["accuracy", "latency"],
        execution_logic=simple_execution,  # FIX: Reference module-level function
        execution_type="function",
    )


# ============================================================================
# TEST SUITE: Principle Extractor
# ============================================================================


class TestPrincipleExtractor:
    """Test principle extraction pipeline"""

    def test_create_extractor(self):
        """Test creating principle extractor"""
        extractor = PrincipleExtractor(
            min_evidence_count=2,
            min_confidence=0.5,
            strategy=ExtractionStrategy.BALANCED,
        )
        assert extractor is not None
        assert extractor.min_confidence == 0.5
        assert extractor.strategy == ExtractionStrategy.BALANCED

    def test_extract_from_trace(self, sample_execution_trace):
        """Test extracting principles from trace"""
        extractor = PrincipleExtractor(
            min_evidence_count=1,
            min_confidence=0.3,
            strategy=ExtractionStrategy.AGGRESSIVE,
        )

        principles = extractor.extract_from_trace(sample_execution_trace)
        assert isinstance(principles, list)

    def test_pattern_detection(self, sample_execution_trace):
        """Test pattern detection"""
        extractor = PrincipleExtractor()
        patterns = extractor.pattern_detector.detect_patterns(sample_execution_trace)
        assert isinstance(patterns, list)
        if patterns:
            assert any(p.pattern_type == PatternType.SEQUENTIAL for p in patterns)

    def test_success_analysis(self, sample_execution_trace):
        """Test success factor analysis"""
        extractor = PrincipleExtractor()
        factors = extractor.success_analyzer.analyze(sample_execution_trace)
        assert isinstance(factors, list)
        if factors:
            assert all(hasattr(f, "importance") for f in factors)


# ============================================================================
# TEST SUITE: Validation Engine
# ============================================================================


class TestValidationEngine:
    """Test validation engine"""

    def test_create_validator(self):
        """Test creating validator"""
        validator = KnowledgeValidator(min_confidence=0.6, consistency_threshold=0.7)
        assert validator is not None
        assert validator.min_confidence == 0.6

    def test_basic_validation(self, sample_principle):
        """Test basic validation"""
        validator = KnowledgeValidator()
        result = validator.validate(sample_principle)
        assert result is not None
        assert hasattr(result, "is_valid")
        assert hasattr(result, "confidence")
        assert result.is_valid

    def test_consistency_validation(self, sample_principle):
        """Test consistency validation"""
        validator = KnowledgeValidator()
        result = validator.validate_consistency(sample_principle)
        assert result is not None
        assert result.confidence > 0.0

    def test_execute_principle(self, sample_principle):
        """Test executing principle logic"""
        test_inputs = {"x": 5, "domain": "optimization"}
        output = sample_principle.execute(test_inputs)
        assert output is not None
        assert output.get("result") == 10
        assert output.get("success") is True

    def test_multilevel_validation(self, sample_principle):
        """Test multilevel validation selection"""
        validator = KnowledgeValidator()
        results = validator.validate_principle_multilevel(
            sample_principle,
            context={"time_budget_ms": 5000, "quality_requirement": "standard"},
        )
        assert "basic" in results


# ============================================================================
# TEST SUITE: Contraindication Tracking
# ============================================================================


class TestContraindicationTracking:
    """Test contraindication tracking"""

    def test_create_database(self):
        """Test creating contraindication database"""
        db = ContraindicationDatabase()
        assert db is not None
        assert hasattr(db, "contraindications")

    def test_register_contraindication(self):
        """Test registering contraindication"""
        db = ContraindicationDatabase()
        contra = Contraindication(
            condition="high_memory_usage",
            failure_mode=FailureMode.RESOURCE.value,
            frequency=1,
            severity=0.7,
            workaround="Reduce batch size",
            domain="optimization",
        )
        db.register("test_principle_001", contra)

        contras = db.get_contraindications("test_principle_001")
        assert len(contras) > 0
        assert contras[0].condition == "high_memory_usage"

    def test_create_graph(self):
        """Test creating contraindication graph"""
        graph = ContraindicationGraph()
        assert graph is not None
        assert hasattr(graph, "graph")

    def test_cascade_analyzer(self):
        """Test cascade analyzer"""
        db = ContraindicationDatabase()
        graph = ContraindicationGraph()
        analyzer = CascadeAnalyzer(db, graph)
        assert analyzer is not None

    def test_statistics(self):
        """Test getting statistics"""
        db = ContraindicationDatabase()
        stats = db.get_statistics()
        assert "total_contraindications" in stats
        assert isinstance(stats["total_contraindications"], int)


# ============================================================================
# TEST SUITE: Knowledge Storage
# ============================================================================


class TestKnowledgeStorage:
    """Test knowledge storage"""

    def test_create_knowledge_base(self):
        """Test creating knowledge base"""
        kb = VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY,
            compression=CompressionType.NONE,
            auto_save=False,
        )
        assert kb is not None
        assert kb.backend == StorageBackend.MEMORY

    def test_store_principle(self, sample_principle):
        """Test storing principle"""
        kb = VersionedKnowledgeBase(backend=StorageBackend.MEMORY, auto_save=False)
        stored_id = kb.store(
            sample_principle, author="test_user", message="Initial version"
        )
        assert stored_id == sample_principle.id

    def test_retrieve_principle(self, sample_principle):
        """Test retrieving principle"""
        kb = VersionedKnowledgeBase(backend=StorageBackend.MEMORY, auto_save=False)
        kb.store(sample_principle, author="test_user")

        retrieved = kb.get(sample_principle.id)
        assert retrieved is not None
        assert retrieved.id == sample_principle.id

    def test_version_control(self, sample_principle):
        """Test version control"""
        kb = VersionedKnowledgeBase(backend=StorageBackend.MEMORY, auto_save=False)
        kb.store(sample_principle, author="test_user", message="Initial")

        sample_principle.confidence = 0.85
        kb.store_versioned(sample_principle, author="test_user", message="Updated")

        versions = kb.get_history(sample_principle.id)
        assert len(versions) >= 2

    def test_search(self, sample_principle):
        """Test search functionality"""
        kb = VersionedKnowledgeBase(backend=StorageBackend.MEMORY, auto_save=False)
        kb.store(sample_principle)

        results = kb.search({"domain": "general", "min_confidence": 0.5}, limit=10)
        assert results is not None
        assert hasattr(results, "total_count")

    def test_index(self, sample_principle):
        """Test indexing"""
        index = KnowledgeIndex(embedding_dim=128)
        index.index_principle(sample_principle)
        stats = index.get_statistics()
        assert stats["total_indexed"] > 0

    def test_pruning(self, sample_principle):
        """Test pruning"""
        pruner = KnowledgePruner()
        candidates = pruner.identify_low_confidence(
            [sample_principle], confidence_threshold=0.9
        )
        assert isinstance(candidates, list)


# ============================================================================
# TEST SUITE: Crystallization Selector
# ============================================================================


class TestCrystallizationSelector:
    """Test crystallization method selection"""

    def test_create_selector(self):
        """Test creating selector"""
        selector = CrystallizationSelector()
        assert selector is not None
        assert len(selector.strategies) > 0

    def test_select_method_simple(self):
        """Test selecting method for simple trace"""
        selector = CrystallizationSelector()
        trace = CoreExecutionTrace(
            trace_id="selector_test_001",
            actions=[{"type": "action1"}, {"type": "action2"}],
            outcomes={"success": True},
            context={"domain": "general"},
            success=True,
        )

        selection = selector.select_method(trace, context={})
        assert selection.method in CrystallizationMethod
        assert 0 <= selection.confidence <= 1

    def test_select_cascade_aware(self):
        """Test selecting CASCADE_AWARE for failures"""
        selector = CrystallizationSelector()
        trace = CoreExecutionTrace(
            trace_id="selector_test_002",
            actions=[{"type": "action1"}] * 5,
            outcomes={"success": False, "error": "timeout"},
            context={"domain": "optimization"},
            success=False,
            metadata={"failure_mode": "timeout"},
        )

        selection = selector.select_method(
            trace, context={"cascade_failures_detected": True}
        )
        assert selection.method is not None

    def test_update_performance(self):
        """Test tracking performance"""
        selector = CrystallizationSelector()
        selector.update_performance(CrystallizationMethod.STANDARD, success=True)
        selector.update_performance(CrystallizationMethod.STANDARD, success=True)

        stats = selector.get_statistics()
        assert "method_performance" in stats


# ============================================================================
# TEST SUITE: Full Integration
# ============================================================================


class TestFullIntegration:
    """Test full integration through KnowledgeCrystallizer"""

    def test_create_crystallizer(self):
        """Test creating crystallizer"""
        crystallizer = KnowledgeCrystallizer(vulcan_memory=None, semantic_bridge=None)
        assert crystallizer is not None
        assert crystallizer.extractor is not None
        assert crystallizer.validator is not None

    def test_crystallize_trace(self):
        """Test crystallizing a trace (full pipeline)"""
        crystallizer = KnowledgeCrystallizer()

        trace = CoreExecutionTrace(
            trace_id="integration_test_001",
            actions=[
                {"type": "load", "params": {"data": "dataset.csv"}},
                {"type": "preprocess", "params": {"method": "normalize"}},
                {"type": "train", "params": {"epochs": 10}},
                {"type": "evaluate", "params": {"metric": "accuracy"}},
            ],
            outcomes={"success": True, "accuracy": 0.92, "execution_time": 2.5},
            context={
                "domain": "classification",
                "environment": "production",
                "resources": {"memory": 512, "cpu": 2},
            },
            success=True,
            domain="classification",
            metadata={"model": "random_forest", "dataset_size": 1000},
        )

        result = crystallizer.crystallize(trace, context={"time_budget_ms": 10000})
        assert result is not None
        assert hasattr(result, "confidence")
        assert hasattr(result, "principles")

    def test_knowledge_storage(self):
        """Test that principles are stored"""
        crystallizer = KnowledgeCrystallizer()
        all_principles = crystallizer.knowledge_base.get_all_principles()
        assert isinstance(all_principles, list)

    def test_system_statistics(self):
        """Test getting system statistics"""
        crystallizer = KnowledgeCrystallizer()
        stats = crystallizer.knowledge_base.get_statistics()
        assert "total_principles" in stats
        assert isinstance(stats["total_principles"], int)


# ============================================================================
# INTEGRATION TEST: End-to-End Pipeline
# ============================================================================


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline"""
    crystallizer = KnowledgeCrystallizer()

    traces = []
    for i in range(3):
        trace = CoreExecutionTrace(
            trace_id=f"e2e_test_{i:03d}",
            actions=[{"type": "initialize"}, {"type": "process"}, {"type": "finalize"}],
            outcomes={"success": True, "value": i * 10},
            context={"domain": "general", "iteration": i},
            success=True,
        )
        traces.append(trace)

    results = []
    for trace in traces:
        result = crystallizer.crystallize(trace)
        results.append(result)

    assert len(results) == 3
    assert all(hasattr(r, "confidence") for r in results)

    all_principles = crystallizer.knowledge_base.get_all_principles()
    assert isinstance(all_principles, list)
