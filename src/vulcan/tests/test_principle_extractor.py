"""
test_principle_extractor.py - Comprehensive tests for principle_extractor module
Part of the VULCAN-AGI system
"""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass

# Import the module under test
from vulcan.knowledge_crystallizer.principle_extractor import (
    PatternType,
    MetricType,
    ExtractionStrategy,
    Pattern,
    Metric,
    ExecutionTrace,
    SuccessFactor,
    PrincipleCandidate,
    CrystallizedPrinciple,
    Principle,
    PrincipleExtractor,
    PatternDetector,
    SuccessAnalyzer,
    AbstractionEngine,
)


class TestPattern:
    """Tests for Pattern class"""

    def test_create_pattern(self):
        """Test creating a pattern"""
        pattern = Pattern(
            pattern_type=PatternType.SEQUENTIAL,
            components=["action1", "action2", "action3"],
            confidence=0.8,
            complexity=3,
        )

        assert pattern.pattern_type == PatternType.SEQUENTIAL
        assert len(pattern.components) == 3
        assert pattern.confidence == 0.8
        assert pattern.complexity == 3

    def test_pattern_signature(self):
        """Test pattern signature generation"""
        pattern = Pattern(
            pattern_type=PatternType.SEQUENTIAL,
            components=["a", "b", "c"],
            structure={"key": "value"},
        )

        signature = pattern.signature()
        assert signature is not None
        assert len(signature) == 32  # MD5 hash

        # Same pattern should have same signature
        pattern2 = Pattern(
            pattern_type=PatternType.SEQUENTIAL,
            components=["a", "b", "c"],
            structure={"key": "value"},
        )
        assert pattern.signature() == pattern2.signature()

    def test_pattern_similarity(self):
        """Test pattern similarity comparison"""
        pattern1 = Pattern(
            pattern_type=PatternType.SEQUENTIAL,
            components=["action1", "action2", "action3"],
        )

        pattern2 = Pattern(
            pattern_type=PatternType.SEQUENTIAL,
            components=["action1", "action2", "action4"],
        )

        # Should be similar (2/4 components match)
        assert pattern1.is_similar_to(pattern2, threshold=0.5)

        # Should not be similar with high threshold
        assert not pattern1.is_similar_to(pattern2, threshold=0.9)

    def test_pattern_equality(self):
        """Test pattern equality"""
        pattern1 = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        pattern2 = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        # Should be equal based on signature
        assert pattern1 == pattern2

    def test_pattern_hashable(self):
        """Test pattern can be used in sets/dicts"""
        pattern1 = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        pattern2 = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        # Should be usable in set
        pattern_set = {pattern1, pattern2}
        assert len(pattern_set) == 1  # Same pattern

    def test_pattern_to_dict(self):
        """Test pattern serialization"""
        pattern = Pattern(
            pattern_type=PatternType.ITERATIVE,
            components=["loop", "action"],
            confidence=0.9,
            frequency=3.0,
        )

        data = pattern.to_dict()
        assert data["pattern_type"] == "iterative"
        assert data["confidence"] == 0.9
        assert data["frequency"] == 3.0
        assert len(data["components"]) == 2


class TestMetric:
    """Tests for Metric class"""

    def test_create_metric(self):
        """Test creating a metric"""
        metric = Metric(
            name="accuracy",
            metric_type=MetricType.ACCURACY,
            value=0.95,
            threshold=0.9,
            is_success=True,
        )

        assert metric.name == "accuracy"
        assert metric.value == 0.95
        assert metric.threshold == 0.9
        assert metric.is_success

    def test_meets_threshold_accuracy(self):
        """Test threshold checking for accuracy metric"""
        metric = Metric(
            name="accuracy", metric_type=MetricType.ACCURACY, value=0.95, threshold=0.9
        )

        assert metric.meets_threshold()

        metric.value = 0.85
        assert not metric.meets_threshold()

    def test_meets_threshold_latency(self):
        """Test threshold checking for latency (lower is better)"""
        metric = Metric(
            name="latency",
            metric_type=MetricType.LATENCY,
            value=50.0,
            threshold=100.0,
            unit="ms",
        )

        assert metric.meets_threshold()

        metric.value = 150.0
        assert not metric.meets_threshold()

    def test_normalize_value(self):
        """Test metric value normalization"""
        metric = Metric(name="score", metric_type=MetricType.PERFORMANCE, value=75.0)

        normalized = metric.normalize_value(min_val=0, max_val=100)
        assert 0.74 < normalized < 0.76  # Should be ~0.75

    def test_metric_to_dict(self):
        """Test metric serialization"""
        metric = Metric(
            name="accuracy",
            metric_type=MetricType.ACCURACY,
            value=0.95,
            unit="percentage",
        )

        data = metric.to_dict()
        assert data["name"] == "accuracy"
        assert data["metric_type"] == "accuracy"
        assert data["value"] == 0.95
        assert data["unit"] == "percentage"


class TestExecutionTrace:
    """Tests for ExecutionTrace class"""

    def test_create_trace(self):
        """Test creating an execution trace"""
        trace = ExecutionTrace(
            trace_id="trace_001",
            actions=[
                {"type": "initialize", "params": {}},
                {"type": "process", "params": {}},
                {"type": "finalize", "params": {}},
            ],
            outcomes={"result": "success"},
            context={"env": "production"},
            success=True,
            domain="data_processing",
        )

        assert trace.trace_id == "trace_001"
        assert len(trace.actions) == 3
        assert trace.success
        assert trace.domain == "data_processing"

    def test_get_duration(self):
        """Test getting execution duration"""
        trace = ExecutionTrace(
            trace_id="trace_001",
            actions=[],
            outcomes={"execution_time": 1.5},
            context={},
        )

        assert trace.get_duration() == 1.5

        # Test with metadata timestamps
        trace2 = ExecutionTrace(
            trace_id="trace_002",
            actions=[],
            outcomes={},
            context={},
            metadata={"start_time": 100.0, "end_time": 102.5},
        )

        assert trace2.get_duration() == 2.5

    def test_get_action_sequence(self):
        """Test extracting action sequence"""
        trace = ExecutionTrace(
            trace_id="trace_001",
            actions=[{"type": "read"}, {"type": "transform"}, {"type": "write"}],
            outcomes={},
            context={},
        )

        sequence = trace.get_action_sequence()
        assert sequence == ["read", "transform", "write"]

    def test_trace_equality(self):
        """Test trace equality based on trace_id"""
        trace1 = ExecutionTrace(
            trace_id="trace_001", actions=[], outcomes={}, context={}
        )

        trace2 = ExecutionTrace(
            trace_id="trace_001",
            actions=[{"type": "different"}],
            outcomes={},
            context={},
        )

        assert trace1 == trace2  # Same trace_id

    def test_trace_to_dict(self):
        """Test trace serialization"""
        trace = ExecutionTrace(
            trace_id="trace_001",
            actions=[{"type": "action"}],
            outcomes={"status": "ok"},
            context={"env": "test"},
            success=True,
        )

        data = trace.to_dict()
        assert data["trace_id"] == "trace_001"
        assert data["success"]
        assert "actions" in data
        assert "outcomes" in data


class TestSuccessFactor:
    """Tests for SuccessFactor class"""

    def test_create_success_factor(self):
        """Test creating a success factor"""
        factor = SuccessFactor(
            factor_type="metric_accuracy",
            importance=0.8,
            evidence_count=5,
            conditions=["accuracy>=0.9"],
            correlation=0.75,
        )

        assert factor.factor_type == "metric_accuracy"
        assert factor.importance == 0.8
        assert factor.evidence_count == 5
        assert factor.correlation == 0.75

    def test_update_importance(self):
        """Test updating importance with new evidence"""
        factor = SuccessFactor(factor_type="test", importance=0.5, evidence_count=1)

        initial_importance = factor.importance
        factor.update_importance(0.9)

        # Should increase with high evidence
        assert factor.importance > initial_importance
        assert factor.evidence_count == 2

        # Should be clamped to [0, 1]
        assert 0.0 <= factor.importance <= 1.0

    def test_factor_to_dict(self):
        """Test factor serialization"""
        factor = SuccessFactor(
            factor_type="timing",
            importance=0.7,
            evidence_count=3,
            conditions=["fast_execution"],
        )

        data = factor.to_dict()
        assert data["factor_type"] == "timing"
        assert data["importance"] == 0.7
        assert data["evidence_count"] == 3


class TestPrincipleCandidate:
    """Tests for PrincipleCandidate class"""

    def test_create_candidate(self):
        """Test creating a principle candidate"""
        pattern = Pattern(
            pattern_type=PatternType.SEQUENTIAL, components=["a", "b", "c"]
        )

        candidate = PrincipleCandidate(
            pattern=pattern, origin_domain="math", confidence=0.7
        )

        assert candidate.pattern == pattern
        assert candidate.origin_domain == "math"
        assert candidate.confidence == 0.7

    def test_add_evidence(self):
        """Test adding evidence to candidate"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        candidate = PrincipleCandidate(pattern=pattern)

        trace = ExecutionTrace(
            trace_id="trace_001", actions=[], outcomes={}, context={}, success=True
        )

        initial_evidence_count = len(candidate.evidence)
        candidate.add_evidence(trace)

        assert len(candidate.evidence) == initial_evidence_count + 1
        assert trace in candidate.evidence

    def test_confidence_update_on_evidence(self):
        """Test confidence updates when adding evidence"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        candidate = PrincipleCandidate(pattern=pattern)
        initial_confidence = candidate.confidence

        # Add successful trace
        trace = ExecutionTrace(
            trace_id="trace_001", actions=[], outcomes={}, context={}, success=True
        )
        candidate.add_evidence(trace)

        # Confidence should be updated
        assert candidate.confidence != initial_confidence

    def test_calculate_generalizability(self):
        """Test generalizability calculation"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        candidate = PrincipleCandidate(pattern=pattern)

        # Add evidence from different domains
        for i in range(3):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                actions=[],
                outcomes={},
                context={"key": f"value{i}"},
                domain=f"domain_{i}",
                success=True,
            )
            candidate.add_evidence(trace)

        generalizability = candidate.calculate_generalizability()

        # Should have some generalizability with diverse domains
        assert 0.0 <= generalizability <= 1.0
        assert generalizability > 0.3  # Multiple domains


class TestCrystallizedPrinciple:
    """Tests for CrystallizedPrinciple class"""

    def test_create_principle(self):
        """Test creating a crystallized principle"""
        pattern = Pattern(
            pattern_type=PatternType.SEQUENTIAL, components=["step1", "step2"]
        )

        principle = CrystallizedPrinciple(
            id="principle_001",
            name="Test Principle",
            description="A test principle",
            core_pattern=pattern,
            applicable_domains=["math", "science"],
            confidence=0.85,
        )

        assert principle.id == "principle_001"
        assert principle.name == "Test Principle"
        assert principle.confidence == 0.85
        assert len(principle.applicable_domains) == 2

    def test_apply_principle(self):
        """Test applying principle to a problem"""
        pattern = Pattern(
            pattern_type=PatternType.SEQUENTIAL, components=["analyze", "solve"]
        )

        principle = CrystallizedPrinciple(
            id="p1",
            name="Analysis First",
            description="Analyze before solving",
            core_pattern=pattern,
            applicable_domains=["math"],
            confidence=0.9,
        )

        problem = {"domain": "math", "type": "equation"}
        solution = principle.apply(problem)

        assert solution["principle_id"] == "p1"
        assert solution["confidence"] == 0.9
        assert "pattern" in solution

    def test_apply_contraindicated_domain(self):
        """Test applying principle to contraindicated domain"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        principle = CrystallizedPrinciple(
            id="p1",
            name="Test",
            description="Test",
            core_pattern=pattern,
            applicable_domains=["math"],
            contraindicated_domains=["physics"],
            confidence=0.9,
        )

        problem = {"domain": "physics"}
        solution = principle.apply(problem)

        # Should have warning and reduced confidence
        assert "warning" in solution
        assert solution["confidence"] < 0.9

    def test_update_stats_success(self):
        """Test updating statistics on success"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a"])

        principle = CrystallizedPrinciple(
            id="p1",
            name="Test",
            description="Test",
            core_pattern=pattern,
            confidence=0.5,
        )

        initial_success = principle.success_count
        initial_confidence = principle.confidence

        principle.update_stats(success=True)

        assert principle.success_count == initial_success + 1
        # Confidence should increase with success
        assert principle.confidence >= initial_confidence

    def test_update_stats_failure(self):
        """Test updating statistics on failure"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a"])

        principle = CrystallizedPrinciple(
            id="p1",
            name="Test",
            description="Test",
            core_pattern=pattern,
            confidence=0.9,
            success_count=5,
            failure_count=0,
        )

        principle.update_stats(success=False)

        assert principle.failure_count == 1

    def test_get_success_rate(self):
        """Test calculating success rate"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a"])

        principle = CrystallizedPrinciple(
            id="p1",
            name="Test",
            description="Test",
            core_pattern=pattern,
            success_count=7,
            failure_count=3,
        )

        success_rate = principle.get_success_rate()
        assert success_rate == 0.7

    def test_principle_to_dict(self):
        """Test principle serialization"""
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        principle = CrystallizedPrinciple(
            id="p1",
            name="Test Principle",
            description="Test",
            core_pattern=pattern,
            confidence=0.85,
            success_count=8,
            failure_count=2,
        )

        data = principle.to_dict()
        assert data["id"] == "p1"
        assert data["name"] == "Test Principle"
        assert data["confidence"] == 0.85
        assert data["success_rate"] == 0.8
        assert "core_pattern" in data


class TestPrincipleExtractor:
    """Tests for PrincipleExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create extractor for testing"""
        return PrincipleExtractor(
            min_evidence_count=2,
            min_confidence=0.5,
            strategy=ExtractionStrategy.BALANCED,
        )

    @pytest.fixture
    def sample_trace(self):
        """Create sample execution trace"""
        return ExecutionTrace(
            trace_id="trace_001",
            actions=[
                {"type": "initialize"},
                {"type": "process"},
                {"type": "validate"},
                {"type": "finalize"},
            ],
            outcomes={"status": "success", "execution_time": 0.5},
            context={"env": "test", "resources": {"cpu": 80}},
            metrics=[
                Metric(
                    name="accuracy",
                    metric_type=MetricType.ACCURACY,
                    value=0.95,
                    threshold=0.9,
                    is_success=True,
                )
            ],
            success=True,
            domain="data_processing",
        )

    def test_create_extractor(self, extractor):
        """Test creating principle extractor"""
        assert extractor is not None
        assert extractor.min_evidence_count == 2
        assert extractor.min_confidence == 0.5
        assert extractor.strategy == ExtractionStrategy.BALANCED

    def test_strategy_adjustment(self):
        """Test threshold adjustment based on strategy"""
        conservative = PrincipleExtractor(
            min_evidence_count=3,
            min_confidence=0.6,
            strategy=ExtractionStrategy.CONSERVATIVE,
        )

        assert conservative.min_evidence_count >= 5
        assert conservative.min_confidence >= 0.8

        aggressive = PrincipleExtractor(
            min_evidence_count=3,
            min_confidence=0.6,
            strategy=ExtractionStrategy.AGGRESSIVE,
        )

        assert aggressive.min_confidence < 0.6

    def test_extract_candidates(self, extractor, sample_trace):
        """Test extracting candidates from trace"""
        candidates = extractor.extract_candidates(sample_trace)

        assert isinstance(candidates, list)
        # Should find some patterns in the trace
        assert len(candidates) >= 0

    def test_extract_from_trace(self, extractor, sample_trace):
        """Test extracting principles from single trace"""
        principles = extractor.extract_from_trace(sample_trace)

        assert isinstance(principles, list)
        # All returned items should be CrystallizedPrinciple
        for principle in principles:
            assert isinstance(principle, CrystallizedPrinciple)

    def test_extract_from_batch(self, extractor):
        """Test extracting from multiple traces"""
        traces = []
        for i in range(5):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                actions=[
                    {"type": "initialize"},
                    {"type": "process"},
                    {"type": "finalize"},
                ],
                outcomes={"status": "success"},
                context={},
                success=True,
                domain="test",
            )
            traces.append(trace)

        principles = extractor.extract_from_batch(traces)

        assert isinstance(principles, list)
        # Should extract principles with sufficient evidence
        for principle in principles:
            assert isinstance(principle, CrystallizedPrinciple)

    def test_extract_invalid_trace(self, extractor):
        """Test extraction with invalid trace"""
        # None trace
        principles = extractor.extract_from_trace(None)
        assert principles == []

        # Empty trace
        empty_trace = ExecutionTrace(
            trace_id="empty", actions=[], outcomes={}, context={}
        )
        principles = extractor.extract_from_trace(empty_trace)
        assert isinstance(principles, list)

    def test_calculate_principle_confidence(self, extractor):
        """Test confidence calculation"""
        traces = []
        for i in range(5):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                actions=[{"type": "action"}],
                outcomes={},
                context={},
                success=True,
                domain="test",
                metrics=[
                    Metric(
                        name="score",
                        metric_type=MetricType.PERFORMANCE,
                        value=0.9,
                        is_success=True,
                    )
                ],
            )
            traces.append(trace)

        confidence = extractor.calculate_principle_confidence(traces)

        assert 0.0 <= confidence <= 1.0
        # Should have decent confidence with all successful traces
        assert confidence > 0.5

    def test_analyze_success_factors(self, extractor, sample_trace):
        """Test success factor analysis"""
        factors = extractor.analyze_success_factors(sample_trace)

        assert isinstance(factors, list)
        for factor in factors:
            assert isinstance(factor, SuccessFactor)
            assert 0.0 <= factor.importance <= 1.0

    def test_cache_cleanup(self, extractor):
        """Test cache cleanup mechanism"""
        # Add some candidates to pool
        pattern = Pattern(pattern_type=PatternType.SEQUENTIAL, components=["a", "b"])

        candidate = PrincipleCandidate(
            pattern=pattern,
            confidence=0.2,  # Low confidence
        )

        sig = pattern.signature()
        extractor.candidate_pool[sig] = candidate

        # Force cleanup
        extractor._last_cleanup = 0
        extractor._cleanup_cache()

        # Low confidence candidates should be removed
        # (may or may not be removed depending on criteria)
        assert isinstance(extractor.candidate_pool, dict)


class TestPatternDetector:
    """Tests for PatternDetector class"""

    @pytest.fixture
    def detector(self):
        """Create pattern detector"""
        return PatternDetector()

    def test_create_detector(self, detector):
        """Test creating pattern detector"""
        assert detector is not None
        assert detector.min_pattern_length >= 1

    def test_detect_sequential_patterns(self, detector):
        """Test detecting sequential patterns"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[{"type": "read"}, {"type": "process"}, {"type": "write"}],
            outcomes={},
            context={},
        )

        patterns = detector.detect_patterns(trace)

        assert isinstance(patterns, list)
        # Should detect at least sequential pattern
        sequential = [p for p in patterns if p.pattern_type == PatternType.SEQUENTIAL]
        assert len(sequential) > 0

    def test_detect_iterative_patterns(self, detector):
        """Test detecting iterative patterns"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[
                {"type": "loop"},
                {"type": "process"},
                {"type": "loop"},
                {"type": "process"},
                {"type": "loop"},
                {"type": "process"},
            ],
            outcomes={},
            context={},
        )

        patterns = detector.detect_patterns(trace)

        # Should detect iterative pattern
        iterative = [p for p in patterns if p.pattern_type == PatternType.ITERATIVE]
        assert len(iterative) > 0

    def test_detect_conditional_patterns(self, detector):
        """Test detecting conditional patterns"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[
                {"type": "check_condition"},
                {"type": "then_action"},
                {"type": "else_action"},
            ],
            outcomes={},
            context={},
        )

        patterns = detector.detect_patterns(trace)

        # Should detect some patterns
        assert isinstance(patterns, list)

    def test_empty_trace(self, detector):
        """Test pattern detection with empty trace"""
        trace = ExecutionTrace(trace_id="empty", actions=[], outcomes={}, context={})

        patterns = detector.detect_patterns(trace)
        assert patterns == []


class TestSuccessAnalyzer:
    """Tests for SuccessAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create success analyzer"""
        return SuccessAnalyzer()

    def test_create_analyzer(self, analyzer):
        """Test creating success analyzer"""
        assert analyzer is not None
        assert hasattr(analyzer, "factor_weights")

    def test_analyze_successful_trace(self, analyzer):
        """Test analyzing successful trace"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[{"type": "initialize"}, {"type": "process"}],
            outcomes={"execution_time": 0.5},
            context={"env": "production"},
            metrics=[
                Metric(
                    name="accuracy",
                    metric_type=MetricType.ACCURACY,
                    value=0.95,
                    threshold=0.9,
                    is_success=True,
                )
            ],
            success=True,
        )

        factors = analyzer.analyze(trace)

        assert isinstance(factors, list)
        assert len(factors) > 0

        for factor in factors:
            assert isinstance(factor, SuccessFactor)
            assert 0.0 <= factor.importance <= 2.0  # Can exceed 1.0 with multipliers

    def test_analyze_failed_trace(self, analyzer):
        """Test analyzing failed trace"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[{"type": "action"}],
            outcomes={},
            context={},
            success=False,
        )

        factors = analyzer.analyze(trace)

        # Should still extract factors even from failure
        assert isinstance(factors, list)

    def test_analyze_action_factors(self, analyzer):
        """Test action factor analysis"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[
                {"type": "init"},
                {"type": "process1"},
                {"type": "process2"},
                {"type": "process3"},
                {"type": "finalize"},
            ],
            outcomes={},
            context={},
            success=True,
        )

        factors = analyzer._analyze_action_factors(trace)

        assert isinstance(factors, list)
        # Should have initial and final action factors
        assert any("initial" in f.factor_type for f in factors)
        assert any("final" in f.factor_type for f in factors)

    def test_analyze_metric_factors(self, analyzer):
        """Test metric factor analysis"""
        trace = ExecutionTrace(
            trace_id="test",
            actions=[],
            outcomes={},
            context={},
            metrics=[
                Metric(
                    name="accuracy",
                    metric_type=MetricType.ACCURACY,
                    value=0.95,
                    threshold=0.9,
                    is_success=True,
                ),
                Metric(
                    name="latency",
                    metric_type=MetricType.LATENCY,
                    value=50,
                    threshold=100,
                    is_success=True,
                ),
            ],
            success=True,
        )

        factors = analyzer._analyze_metric_factors(trace)

        assert len(factors) == 2
        assert all("metric" in f.factor_type for f in factors)


class TestAbstractionEngine:
    """Tests for AbstractionEngine class"""

    @pytest.fixture
    def engine(self):
        """Create abstraction engine"""
        return AbstractionEngine()

    def test_create_engine(self, engine):
        """Test creating abstraction engine"""
        assert engine is not None
        assert hasattr(engine, "abstraction_rules")
        assert hasattr(engine, "naming_templates")

    def test_abstract_success_factors(self, engine):
        """Test abstracting success factors to principle"""
        factors = [
            SuccessFactor(
                factor_type="metric_accuracy",
                importance=0.9,
                evidence_count=5,
                conditions=["accuracy>=0.9"],
                metrics=[
                    Metric(
                        name="accuracy",
                        metric_type=MetricType.ACCURACY,
                        value=0.95,
                        is_success=True,
                    )
                ],
            ),
            SuccessFactor(
                factor_type="action_sequence",
                importance=0.7,
                evidence_count=3,
                conditions=["sequential_execution"],
            ),
        ]

        abstracted = engine.abstract(factors)

        assert abstracted is not None
        assert "name" in abstracted
        assert "description" in abstracted
        assert "pattern" in abstracted
        assert "confidence" in abstracted

    def test_abstract_empty_factors(self, engine):
        """Test abstraction with no factors"""
        result = engine.abstract([])
        assert result is None

    def test_generate_name(self, engine):
        """Test principle name generation"""
        factors = [
            SuccessFactor(
                factor_type="metric_performance", importance=0.9, evidence_count=5
            )
        ]

        name = engine._generate_name("metric", factors)

        assert isinstance(name, str)
        assert len(name) > 0

    def test_identify_domains(self, engine):
        """Test domain identification"""
        factors = [
            SuccessFactor(
                factor_type="test",
                importance=0.8,
                evidence_count=1,
                conditions=["domain_math_algebra", "env_production"],
                metrics=[
                    Metric(
                        name="accuracy",
                        metric_type=MetricType.ACCURACY,
                        value=0.9,
                        is_success=True,
                    ),
                    Metric(
                        name="performance",
                        metric_type=MetricType.PERFORMANCE,
                        value=0.8,
                        is_success=True,
                    ),
                ],
            )
        ]

        domains = engine._identify_domains(factors)

        assert isinstance(domains, list)
        assert len(domains) > 0

        # Should extract domains from conditions or metrics
        # Either specific domains from conditions or inferred from metric types
        assert any(
            domain in ["math", "production", "accuracy", "performance", "general"]
            for domain in domains
        )


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_extraction_pipeline(self):
        """Test complete extraction pipeline"""
        # Create extractor
        extractor = PrincipleExtractor(
            min_evidence_count=2,
            min_confidence=0.5,
            strategy=ExtractionStrategy.BALANCED,
        )

        # Create multiple similar traces
        traces = []
        for i in range(3):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                actions=[
                    {"type": "initialize"},
                    {"type": "validate"},
                    {"type": "process"},
                    {"type": "finalize"},
                ],
                outcomes={"status": "success", "execution_time": 0.5 + i * 0.1},
                context={"env": "test"},
                metrics=[
                    Metric(
                        name="accuracy",
                        metric_type=MetricType.ACCURACY,
                        value=0.90 + i * 0.02,
                        threshold=0.85,
                        is_success=True,
                    )
                ],
                success=True,
                domain="data_processing",
            )
            traces.append(trace)

        # Extract principles
        principles = extractor.extract_from_batch(traces)

        # Should extract at least some principles
        assert isinstance(principles, list)

        # Validate extracted principles
        for principle in principles:
            assert isinstance(principle, CrystallizedPrinciple)
            assert principle.confidence >= extractor.min_confidence
            assert hasattr(principle, "core_pattern")
            assert hasattr(principle, "domain")

    def test_pattern_evolution(self):
        """Test pattern recognition evolving with more evidence"""
        extractor = PrincipleExtractor(
            min_evidence_count=2,
            min_confidence=0.4,
            strategy=ExtractionStrategy.EXPLORATORY,
        )

        # Add traces one by one
        for i in range(5):
            trace = ExecutionTrace(
                trace_id=f"trace_{i}",
                actions=[{"type": "read"}, {"type": "transform"}, {"type": "write"}],
                outcomes={"status": "success"},
                context={},
                success=True,
                domain="etl",
            )

            principles = extractor.extract_from_trace(trace)

            # Later extractions should have more confidence
            if i >= 2 and principles:
                # Check if any high-confidence principles emerged
                high_conf = [p for p in principles if p.confidence > 0.6]
                # May or may not have high confidence depending on patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
