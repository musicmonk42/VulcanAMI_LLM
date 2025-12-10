"""
test_contraindication_tracker.py - Comprehensive tests for contraindication tracking
Part of the VULCAN-AGI system test suite
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Import the module components to test
from vulcan.knowledge_crystallizer.contraindication_tracker import (
    CascadeAnalyzer, CascadeImpact, Contraindication, ContraindicationDatabase,
    ContraindicationGraph, FailureMode, Severity, SimpleGraph)

# ============================================================================
# TEST HELPER CLASSES
# ============================================================================


class SimplePrinciple:
    """Simple principle class for testing persistence (must be picklable)"""

    def __init__(self, pid):
        self.id = pid
        self.confidence = 0.8
        self.domain = "optimization"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_contraindication():
    """Create sample contraindication"""
    return Contraindication(
        condition="high_memory_usage",
        failure_mode=FailureMode.RESOURCE.value,
        frequency=5,
        severity=0.7,
        workaround="Reduce batch size",
        domain="optimization",
        confidence=0.85,
    )


@pytest.fixture
def mock_principle():
    """Create mock principle for testing"""
    principle = Mock()
    principle.id = "test_principle_001"
    principle.confidence = 0.8
    principle.domain = "optimization"
    return principle


@pytest.fixture
def contraindication_db(temp_dir):
    """Create contraindication database"""
    db_path = temp_dir / "contraindications.json"
    return ContraindicationDatabase(persistence_path=db_path)


@pytest.fixture
def contraindication_graph():
    """Create contraindication graph"""
    return ContraindicationGraph()


@pytest.fixture
def cascade_analyzer(contraindication_db, contraindication_graph):
    """Create cascade analyzer"""
    return CascadeAnalyzer(contraindication_db, contraindication_graph)


# ============================================================================
# SEVERITY ENUM TESTS
# ============================================================================


class TestSeverity:
    """Tests for Severity enum"""

    def test_from_score_low(self):
        """Test severity conversion for low scores"""
        assert Severity.from_score(0.1) == Severity.LOW
        assert Severity.from_score(0.24) == Severity.LOW

    def test_from_score_medium(self):
        """Test severity conversion for medium scores"""
        assert Severity.from_score(0.25) == Severity.MEDIUM
        assert Severity.from_score(0.49) == Severity.MEDIUM

    def test_from_score_high(self):
        """Test severity conversion for high scores"""
        assert Severity.from_score(0.5) == Severity.HIGH
        assert Severity.from_score(0.74) == Severity.HIGH

    def test_from_score_critical(self):
        """Test severity conversion for critical scores"""
        assert Severity.from_score(0.75) == Severity.CRITICAL
        assert Severity.from_score(1.0) == Severity.CRITICAL

    def test_severity_values(self):
        """Test severity enum values"""
        assert Severity.LOW.value == 1
        assert Severity.MEDIUM.value == 2
        assert Severity.HIGH.value == 3
        assert Severity.CRITICAL.value == 4


# ============================================================================
# CONTRAINDICATION TESTS
# ============================================================================


class TestContraindication:
    """Tests for Contraindication dataclass"""

    def test_creation(self, sample_contraindication):
        """Test contraindication creation"""
        assert sample_contraindication.condition == "high_memory_usage"
        assert sample_contraindication.failure_mode == FailureMode.RESOURCE.value
        assert sample_contraindication.frequency == 5
        assert sample_contraindication.severity == 0.7
        assert sample_contraindication.confidence == 0.85

    def test_get_severity_level(self, sample_contraindication):
        """Test severity level calculation"""
        level = sample_contraindication.get_severity_level()
        assert level == Severity.HIGH

    def test_to_dict(self, sample_contraindication):
        """Test dictionary conversion"""
        data = sample_contraindication.to_dict()

        assert isinstance(data, dict)
        assert data["condition"] == "high_memory_usage"
        assert data["frequency"] == 5
        assert data["severity"] == 0.7
        assert "timestamp" in data

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "condition": "timeout",
            "failure_mode": "timeout",
            "frequency": 3,
            "severity": 0.6,
            "domain": "control",
        }

        contra = Contraindication.from_dict(data)
        assert contra.condition == "timeout"
        assert contra.frequency == 3
        assert contra.severity == 0.6

    def test_update_frequency(self, sample_contraindication):
        """Test frequency update"""
        initial_freq = sample_contraindication.frequency
        initial_severity = sample_contraindication.severity

        sample_contraindication.update_frequency(2)

        assert sample_contraindication.frequency == initial_freq + 2
        assert sample_contraindication.severity > initial_severity
        assert sample_contraindication.severity <= 1.0

    def test_merge_with_same_condition(self):
        """Test merging contraindications"""
        contra1 = Contraindication(
            condition="test_condition",
            failure_mode="performance",
            frequency=5,
            severity=0.6,
        )

        contra2 = Contraindication(
            condition="test_condition",
            failure_mode="performance",
            frequency=3,
            severity=0.8,
        )

        merged = contra1.merge_with(contra2)

        assert merged.frequency == 8
        assert merged.severity == 0.8  # Takes max
        assert merged.condition == "test_condition"

    def test_merge_different_condition_raises(self):
        """Test merging with different condition raises error"""
        contra1 = Contraindication(
            condition="condition1",
            failure_mode="performance",
        )

        contra2 = Contraindication(
            condition="condition2",
            failure_mode="performance",
        )

        with pytest.raises(ValueError, match="same condition"):
            contra1.merge_with(contra2)


# ============================================================================
# CASCADE IMPACT TESTS
# ============================================================================


class TestCascadeImpact:
    """Tests for CascadeImpact dataclass"""

    def test_creation(self):
        """Test cascade impact creation"""
        impact = CascadeImpact()

        assert len(impact.affected_principles) == 0
        assert impact.max_severity == 0.0
        assert impact.total_impact == 0.0
        assert impact.cascade_depth == 0
        assert impact.blast_radius == 0

    def test_add_affected(self, mock_principle):
        """Test adding affected principle"""
        impact = CascadeImpact()

        impact.add_affected(mock_principle, 0.7, "Add monitoring")

        assert len(impact.affected_principles) == 1
        assert impact.max_severity == 0.7
        assert impact.total_impact == 0.7
        assert impact.blast_radius == 1
        assert len(impact.warnings) == 1

    def test_add_multiple_affected(self, mock_principle):
        """Test adding multiple affected principles"""
        impact = CascadeImpact()

        principle2 = Mock()
        principle2.id = "test_principle_002"

        impact.add_affected(mock_principle, 0.6, "Mitigation 1")
        impact.add_affected(principle2, 0.8, "Mitigation 2")

        assert len(impact.affected_principles) == 2
        assert impact.max_severity == 0.8
        assert impact.total_impact == 1.4
        assert impact.blast_radius == 2

    def test_add_warning(self):
        """Test adding warnings"""
        impact = CascadeImpact()

        impact.add_warning("Test warning")
        assert "Test warning" in impact.warnings

    def test_estimate_recovery_time(self):
        """Test recovery time estimation"""
        impact = CascadeImpact()
        impact.max_severity = 0.8
        impact.cascade_depth = 3
        impact.blast_radius = 10

        recovery_time = impact.estimate_recovery_time()

        assert recovery_time > 0
        assert impact.recovery_time_estimate == recovery_time

    def test_get_risk_level_critical(self):
        """Test critical risk level"""
        impact = CascadeImpact()
        impact.max_severity = 0.9

        assert impact.get_risk_level() == "CRITICAL"

    def test_get_risk_level_high(self):
        """Test high risk level"""
        impact = CascadeImpact()
        impact.max_severity = 0.65
        impact.blast_radius = 12

        assert impact.get_risk_level() == "HIGH"

    def test_get_risk_level_medium(self):
        """Test medium risk level"""
        impact = CascadeImpact()
        impact.max_severity = 0.5
        impact.blast_radius = 7

        assert impact.get_risk_level() == "MEDIUM"

    def test_get_risk_level_low(self):
        """Test low risk level"""
        impact = CascadeImpact()
        impact.max_severity = 0.3
        impact.blast_radius = 2

        assert impact.get_risk_level() == "LOW"

    def test_to_dict(self, mock_principle):
        """Test dictionary conversion"""
        impact = CascadeImpact()
        impact.add_affected(mock_principle, 0.7)

        data = impact.to_dict()

        assert isinstance(data, dict)
        assert "affected_principles" in data
        assert "max_severity" in data
        assert "risk_level" in data
        assert data["blast_radius"] == 1


# ============================================================================
# CONTRAINDICATION DATABASE TESTS
# ============================================================================


class TestContraindicationDatabase:
    """Tests for ContraindicationDatabase"""

    def test_initialization(self, contraindication_db):
        """Test database initialization"""
        assert contraindication_db.total_contraindications == 0
        assert len(contraindication_db.contraindications) == 0

    def test_register_contraindication(
        self, contraindication_db, sample_contraindication
    ):
        """Test registering a contraindication"""
        contraindication_db.register("principle_001", sample_contraindication)

        assert contraindication_db.total_contraindications == 1
        assert sample_contraindication.principle_id == "principle_001"

        contras = contraindication_db.get_contraindications("principle_001")
        assert len(contras) == 1
        assert contras[0].condition == "high_memory_usage"

    def test_register_duplicate_updates_frequency(self, contraindication_db):
        """Test registering duplicate contraindication updates frequency"""
        contra = Contraindication(
            condition="test_condition", failure_mode="performance", frequency=1
        )

        contraindication_db.register("principle_001", contra)
        initial_freq = contraindication_db.get_contraindications("principle_001")[
            0
        ].frequency

        # Register duplicate
        contra2 = Contraindication(
            condition="test_condition", failure_mode="performance", frequency=1
        )
        contraindication_db.register("principle_001", contra2)

        # Should update existing, not add new
        contras = contraindication_db.get_contraindications("principle_001")
        assert len(contras) == 1
        assert contras[0].frequency > initial_freq

    def test_batch_register(self, contraindication_db):
        """Test batch registration"""
        contras = [
            (
                "principle_001",
                Contraindication(condition="c1", failure_mode="performance"),
            ),
            (
                "principle_002",
                Contraindication(condition="c2", failure_mode="correctness"),
            ),
            (
                "principle_003",
                Contraindication(condition="c3", failure_mode="stability"),
            ),
        ]

        contraindication_db.batch_register(contras)

        assert contraindication_db.total_contraindications == 3
        assert len(contraindication_db.contraindications) == 3

    def test_get_by_condition(self, contraindication_db):
        """Test getting contraindications by condition"""
        contra1 = Contraindication(condition="timeout", failure_mode="timeout")
        contra2 = Contraindication(condition="timeout", failure_mode="timeout")
        contra3 = Contraindication(condition="memory", failure_mode="resource")

        contraindication_db.register("p1", contra1)
        contraindication_db.register("p2", contra2)
        contraindication_db.register("p3", contra3)

        result = contraindication_db.get_by_condition("timeout")

        assert len(result) == 2
        assert "p1" in result
        assert "p2" in result

    def test_check_domain_compatibility_compatible(self, contraindication_db):
        """Test domain compatibility check - compatible case"""
        contra = Contraindication(
            condition="test",
            failure_mode="performance",
            domain="optimization",
            severity=0.3,  # Low severity
        )

        contraindication_db.register("p1", contra)

        is_compatible, blocking = contraindication_db.check_domain_compatibility(
            "p1", "control"
        )

        assert is_compatible
        assert len(blocking) == 0

    def test_check_domain_compatibility_incompatible(self, contraindication_db):
        """Test domain compatibility check - incompatible case"""
        contra = Contraindication(
            condition="test",
            failure_mode="performance",
            domain="optimization",
            severity=0.9,  # Critical severity
        )

        contraindication_db.register("p1", contra)

        is_compatible, blocking = contraindication_db.check_domain_compatibility(
            "p1", "optimization"
        )

        assert not is_compatible
        assert len(blocking) > 0

    def test_get_domain_contraindicated_principles(self, contraindication_db):
        """Test getting principles contraindicated for domain"""
        contra = Contraindication(
            condition="test", failure_mode="performance", domain="optimization"
        )

        contraindication_db.register("p1", contra)

        principles = contraindication_db.get_domain_contraindicated_principles(
            "optimization"
        )

        assert "p1" in principles

    def test_get_failure_pattern_principles(self, contraindication_db):
        """Test getting principles by failure pattern"""
        contra1 = Contraindication(condition="c1", failure_mode="timeout")
        contra2 = Contraindication(condition="c2", failure_mode="timeout")
        contra3 = Contraindication(condition="c3", failure_mode="memory")

        contraindication_db.register("p1", contra1)
        contraindication_db.register("p2", contra2)
        contraindication_db.register("p3", contra3)

        timeout_principles = contraindication_db.get_failure_pattern_principles(
            "timeout"
        )

        assert len(timeout_principles) == 2
        assert "p1" in timeout_principles
        assert "p2" in timeout_principles

    def test_analyze_failure_patterns(self, contraindication_db):
        """Test failure pattern analysis"""
        contra1 = Contraindication(condition="c1", failure_mode="timeout")
        contra2 = Contraindication(condition="c2", failure_mode="memory")
        contra3 = Contraindication(condition="c3", failure_mode="timeout")

        contraindication_db.register("p1", contra1)
        contraindication_db.register("p1", contra2)  # Multiple on same principle
        contraindication_db.register("p2", contra3)

        analysis = contraindication_db.analyze_failure_patterns()

        assert "most_common_failures" in analysis
        assert "failure_combinations" in analysis
        assert "high_risk_domains" in analysis
        assert "total_patterns" in analysis

    def test_get_statistics(self, contraindication_db, sample_contraindication):
        """Test getting statistics"""
        contraindication_db.register("p1", sample_contraindication)

        stats = contraindication_db.get_statistics()

        assert stats["total_contraindications"] == 1
        assert stats["principles_with_contraindications"] == 1
        assert "severity_distribution" in stats
        assert "pattern_analysis" in stats

    def test_prune_old_contraindications(self, contraindication_db):
        """Test pruning old contraindications"""
        # Add old contraindication
        old_contra = Contraindication(
            condition="old", failure_mode="performance", frequency=1
        )
        old_contra.timestamp = time.time() - (100 * 86400)  # 100 days old

        # Add recent contraindication
        recent_contra = Contraindication(
            condition="recent", failure_mode="performance", frequency=1
        )

        contraindication_db.register("p1", old_contra)
        contraindication_db.register("p2", recent_contra)

        initial_count = contraindication_db.total_contraindications
        pruned = contraindication_db.prune_old_contraindications(age_days=90)

        assert pruned > 0
        assert contraindication_db.total_contraindications < initial_count

    def test_save_and_load(self, temp_dir):
        """Test saving and loading database"""
        db_path = temp_dir / "test_db.json"
        db1 = ContraindicationDatabase(persistence_path=db_path)

        contra = Contraindication(condition="test", failure_mode="performance")
        db1.register("p1", contra)
        db1.save()

        # Load in new database
        db2 = ContraindicationDatabase(persistence_path=db_path)
        db2.load()

        assert db2.total_contraindications == 1
        contras = db2.get_contraindications("p1")
        assert len(contras) == 1
        assert contras[0].condition == "test"

    def test_domains_related(self, contraindication_db):
        """Test domain relationship checking"""
        # Same domain
        assert contraindication_db._domains_related("opt", "opt")

        # Hierarchical
        assert contraindication_db._domains_related("opt_deep", "opt")
        assert contraindication_db._domains_related("opt", "opt_deep")

        # Common parts
        assert contraindication_db._domains_related("ml_opt", "ml_control")

        # Unrelated
        assert not contraindication_db._domains_related("optimization", "control")


# ============================================================================
# SIMPLE GRAPH TESTS
# ============================================================================


class TestSimpleGraph:
    """Tests for SimpleGraph fallback implementation"""

    def test_initialization(self):
        """Test graph initialization"""
        graph = SimpleGraph()

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_add_node(self):
        """Test adding nodes"""
        graph = SimpleGraph()

        graph.add_node("node1", data="test")

        assert graph.number_of_nodes() == 1
        assert "node1" in graph

    def test_add_edge(self):
        """Test adding edges"""
        graph = SimpleGraph()

        graph.add_node("n1")
        graph.add_node("n2")
        graph.add_edge("n1", "n2", weight=0.7)

        assert graph.number_of_edges() == 1
        assert graph.get_edge_weight("n1", "n2") == 0.7

    def test_successors(self):
        """Test getting successors"""
        graph = SimpleGraph()

        graph.add_node("n1")
        graph.add_node("n2")
        graph.add_node("n3")
        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")

        successors = graph.successors("n1")

        assert len(successors) == 2
        assert "n2" in successors
        assert "n3" in successors

    def test_predecessors(self):
        """Test getting predecessors"""
        graph = SimpleGraph()

        graph.add_node("n1")
        graph.add_node("n2")
        graph.add_node("n3")
        graph.add_edge("n1", "n3")
        graph.add_edge("n2", "n3")

        predecessors = graph.predecessors("n3")

        assert len(predecessors) == 2
        assert "n1" in predecessors
        assert "n2" in predecessors

    def test_remove_node(self):
        """Test removing nodes"""
        graph = SimpleGraph()

        graph.add_node("n1")
        graph.add_node("n2")
        graph.add_edge("n1", "n2")

        graph.remove_node("n1")

        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        assert "n1" not in graph

    def test_descendants(self):
        """Test getting all descendants"""
        graph = SimpleGraph()

        # Create chain: n1 -> n2 -> n3 -> n4
        for i in range(1, 5):
            graph.add_node(f"n{i}")

        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        graph.add_edge("n3", "n4")

        descendants = graph.descendants("n1")

        assert len(descendants) == 3
        assert "n2" in descendants
        assert "n3" in descendants
        assert "n4" in descendants

    def test_ancestors(self):
        """Test getting all ancestors"""
        graph = SimpleGraph()

        # Create chain: n1 -> n2 -> n3 -> n4
        for i in range(1, 5):
            graph.add_node(f"n{i}")

        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        graph.add_edge("n3", "n4")

        ancestors = graph.ancestors("n4")

        assert len(ancestors) == 3
        assert "n1" in ancestors
        assert "n2" in ancestors
        assert "n3" in ancestors

    def test_shortest_path(self):
        """Test finding shortest path"""
        graph = SimpleGraph()

        for i in range(1, 5):
            graph.add_node(f"n{i}")

        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        graph.add_edge("n3", "n4")

        path = graph.shortest_path("n1", "n4")

        assert path == ["n1", "n2", "n3", "n4"]

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists"""
        graph = SimpleGraph()

        graph.add_node("n1")
        graph.add_node("n2")
        # No edge between them

        with pytest.raises(ValueError, match="No path"):
            graph.shortest_path("n1", "n2")


# ============================================================================
# CONTRAINDICATION GRAPH TESTS
# ============================================================================


class TestContraindicationGraph:
    """Tests for ContraindicationGraph"""

    def test_initialization(self, contraindication_graph):
        """Test graph initialization"""
        assert len(contraindication_graph.principle_nodes) == 0
        assert len(contraindication_graph.impact_weights) == 0

    def test_add_node(self, contraindication_graph, mock_principle):
        """Test adding principle node"""
        contraindication_graph.add_node(mock_principle)

        assert mock_principle.id in contraindication_graph.principle_nodes

    def test_add_edge(self, contraindication_graph, mock_principle):
        """Test adding dependency edge"""
        principle2 = Mock()
        principle2.id = "test_principle_002"

        contraindication_graph.add_node(mock_principle)
        contraindication_graph.add_node(principle2)

        contraindication_graph.add_edge(mock_principle.id, principle2.id, 0.8)

        assert (
            mock_principle.id,
            principle2.id,
        ) in contraindication_graph.impact_weights
        assert (
            contraindication_graph.impact_weights[(mock_principle.id, principle2.id)]
            == 0.8
        )

    def test_remove_node(self, contraindication_graph, mock_principle):
        """Test removing principle node"""
        contraindication_graph.add_node(mock_principle)
        contraindication_graph.remove_node(mock_principle.id)

        assert mock_principle.id not in contraindication_graph.principle_nodes

    def test_find_cascades(self, contraindication_graph):
        """Test finding cascade paths"""
        # Create cascade: p1 -> p2 -> p3
        for i in range(1, 4):
            principle = Mock()
            principle.id = f"p{i}"
            contraindication_graph.add_node(principle)

        contraindication_graph.add_edge("p1", "p2", 0.8)
        contraindication_graph.add_edge("p2", "p3", 0.7)

        cascades = contraindication_graph.find_cascades("p1", max_depth=3)

        assert len(cascades) > 0
        # Should find paths like ["p1", "p2"] and ["p1", "p2", "p3"]

    def test_calculate_cascade_risk(self, contraindication_graph):
        """Test cascade risk calculation"""
        # Create simple cascade
        p1 = Mock()
        p1.id = "p1"
        p2 = Mock()
        p2.id = "p2"

        contraindication_graph.add_node(p1)
        contraindication_graph.add_node(p2)
        contraindication_graph.add_edge("p1", "p2", 0.9)

        risk = contraindication_graph.calculate_cascade_risk("p1")

        assert 0.0 <= risk <= 1.0
        assert risk > 0  # Should have some risk with edge present

    def test_get_downstream_principles(self, contraindication_graph):
        """Test getting downstream principles"""
        # Create chain
        for i in range(1, 4):
            p = Mock()
            p.id = f"p{i}"
            contraindication_graph.add_node(p)

        contraindication_graph.add_edge("p1", "p2", 0.8)
        contraindication_graph.add_edge("p2", "p3", 0.7)

        downstream = contraindication_graph.get_downstream_principles("p1")

        assert "p2" in downstream
        assert "p3" in downstream

    def test_get_upstream_principles(self, contraindication_graph):
        """Test getting upstream principles"""
        # Create chain
        for i in range(1, 4):
            p = Mock()
            p.id = f"p{i}"
            contraindication_graph.add_node(p)

        contraindication_graph.add_edge("p1", "p2", 0.8)
        contraindication_graph.add_edge("p2", "p3", 0.7)

        upstream = contraindication_graph.get_upstream_principles("p3")

        assert "p1" in upstream
        assert "p2" in upstream

    def test_get_impact_path(self, contraindication_graph):
        """Test getting impact path"""
        # Create chain
        for i in range(1, 4):
            p = Mock()
            p.id = f"p{i}"
            contraindication_graph.add_node(p)

        contraindication_graph.add_edge("p1", "p2", 0.8)
        contraindication_graph.add_edge("p2", "p3", 0.7)

        path, impact = contraindication_graph.get_impact_path("p1", "p3")

        assert len(path) > 0
        assert 0.0 <= impact <= 1.0

    def test_find_critical_nodes(self, contraindication_graph):
        """Test finding critical nodes"""
        # Create high-risk cascade
        p1 = Mock()
        p1.id = "p1"

        contraindication_graph.add_node(p1)

        # Add multiple high-impact edges
        for i in range(2, 6):
            p = Mock()
            p.id = f"p{i}"
            contraindication_graph.add_node(p)
            contraindication_graph.add_edge("p1", f"p{i}", 0.9)

        critical = contraindication_graph.find_critical_nodes(threshold=0.5)

        assert isinstance(critical, list)

    def test_get_statistics(self, contraindication_graph, mock_principle):
        """Test getting graph statistics"""
        contraindication_graph.add_node(mock_principle)

        stats = contraindication_graph.get_statistics()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "critical_nodes" in stats
        assert stats["total_nodes"] >= 1

    def test_save_and_load(self, temp_dir):
        """Test saving and loading graph"""
        graph_path = temp_dir / "test_graph"
        graph1 = ContraindicationGraph(persistence_path=graph_path)

        # Use module-level SimplePrinciple class (defined at top of file)
        p1 = SimplePrinciple("p1")
        graph1.add_node(p1)
        graph1.save()

        # Load in new graph
        graph2 = ContraindicationGraph(persistence_path=graph_path)
        graph2.load()

        assert "p1" in graph2.principle_nodes


# ============================================================================
# CASCADE ANALYZER TESTS
# ============================================================================


class TestCascadeAnalyzer:
    """Tests for CascadeAnalyzer"""

    def test_initialization(self, cascade_analyzer):
        """Test analyzer initialization"""
        assert cascade_analyzer.attenuation_factor == 0.7
        assert cascade_analyzer.min_impact_threshold == 0.1

    def test_analyze_cascade_impact(self, cascade_analyzer, mock_principle):
        """Test cascade impact analysis"""
        cascade_analyzer.graph.add_node(mock_principle)

        impact = cascade_analyzer.analyze_cascade_impact(mock_principle, max_depth=2)

        assert isinstance(impact, CascadeImpact)
        assert impact.cascade_depth == 2

    def test_find_dependent_principles(self, cascade_analyzer, mock_principle):
        """Test finding dependent principles"""
        p2 = Mock()
        p2.id = "p2"

        cascade_analyzer.graph.add_node(mock_principle)
        cascade_analyzer.graph.add_node(p2)
        cascade_analyzer.graph.add_edge(mock_principle.id, p2.id, 0.8)

        dependents = cascade_analyzer.find_dependent_principles(mock_principle)

        assert len(dependents) > 0
        assert dependents[0].id == "p2"

    def test_simulate_failure(self, cascade_analyzer, mock_principle):
        """Test failure simulation"""
        cascade_analyzer.graph.add_node(mock_principle)

        scenario = cascade_analyzer.simulate_failure(
            mock_principle, {"test_context": "value"}
        )

        assert isinstance(scenario, dict)
        assert "principle_id" in scenario
        assert "severity" in scenario
        assert "failure_type" in scenario

    def test_simulate_failure_with_contraindications(
        self, cascade_analyzer, mock_principle
    ):
        """Test failure simulation with contraindications"""
        # Add contraindication
        contra = Contraindication(
            condition="test", failure_mode="critical", severity=0.95
        )
        cascade_analyzer.db.register(mock_principle.id, contra)
        cascade_analyzer.graph.add_node(mock_principle)

        scenario = cascade_analyzer.simulate_failure(mock_principle, {})

        assert scenario["severity"] == 0.95

    def test_suggest_mitigation(self, cascade_analyzer):
        """Test mitigation suggestion"""
        scenario = {
            "severity": 0.85,
            "failure_type": "critical",
            "recovery_possible": False,
            "estimated_recovery_time": 120,
        }

        mitigation = cascade_analyzer.suggest_mitigation(scenario)

        assert mitigation is not None
        assert isinstance(mitigation, str)
        assert len(mitigation) > 0

    def test_calculate_attenuation(self, cascade_analyzer):
        """Test impact attenuation calculation"""
        # Test depth 1 - use pytest.approx for floating point comparison
        attenuated = cascade_analyzer.calculate_attenuation(1, 1.0)
        assert attenuated == pytest.approx(0.7)

        # Test depth 2
        attenuated = cascade_analyzer.calculate_attenuation(2, 1.0)
        assert attenuated == pytest.approx(0.49)

        # Test below threshold
        attenuated = cascade_analyzer.calculate_attenuation(10, 1.0)
        assert attenuated == pytest.approx(0.0)

    def test_predict_cascade_path(self, cascade_analyzer, mock_principle):
        """Test cascade path prediction"""
        cascade_analyzer.graph.add_node(mock_principle)

        # Add some failure patterns
        contra = Contraindication(condition="test", failure_mode="timeout")
        cascade_analyzer.db.register(mock_principle.id, contra)

        path = cascade_analyzer.predict_cascade_path(mock_principle.id, "timeout")

        assert isinstance(path, list)
        assert len(path) > 0

    def test_get_cascade_statistics(self, cascade_analyzer, mock_principle):
        """Test getting cascade statistics"""
        cascade_analyzer.graph.add_node(mock_principle)

        stats = cascade_analyzer.get_cascade_statistics()

        assert "total_principles" in stats
        assert "avg_cascade_risk" in stats
        assert "max_cascade_risk" in stats
        assert stats["total_principles"] >= 1

    def test_cache_cleanup(self, cascade_analyzer, mock_principle):
        """Test simulation cache cleanup"""
        cascade_analyzer.graph.add_node(mock_principle)

        # Add many cached simulations
        for i in range(250):
            cascade_analyzer.simulate_failure(mock_principle, {"iteration": i})

        # Cache should be cleaned to max 200 entries
        assert len(cascade_analyzer.simulation_cache) <= 200


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for the complete system"""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow from registration to analysis"""
        # Setup
        db_path = temp_dir / "integration_db.json"
        graph_path = temp_dir / "integration_graph"

        db = ContraindicationDatabase(persistence_path=db_path)
        graph = ContraindicationGraph(persistence_path=graph_path)
        analyzer = CascadeAnalyzer(db, graph)

        # Use module-level SimplePrinciple class (defined at top of file)
        p1 = SimplePrinciple("integration_p1")
        p2 = SimplePrinciple("integration_p2")
        p3 = SimplePrinciple("integration_p3")

        # Register contraindications
        contra1 = Contraindication(
            condition="high_load",
            failure_mode="performance",
            severity=0.6,
            domain="optimization",
        )
        db.register(p1.id, contra1)

        # Build dependency graph
        graph.add_node(p1)
        graph.add_node(p2)
        graph.add_node(p3)
        graph.add_edge(p1.id, p2.id, 0.8)
        graph.add_edge(p2.id, p3.id, 0.7)

        # Analyze cascade impact
        impact = analyzer.analyze_cascade_impact(p1, max_depth=3)

        # Verify results
        assert isinstance(impact, CascadeImpact)
        assert impact.cascade_depth == 3

        # Test persistence
        db.save()
        graph.save()

        # Load and verify
        db2 = ContraindicationDatabase(persistence_path=db_path)
        db2.load()
        assert db2.total_contraindications > 0

    def test_complex_cascade_scenario(self):
        """Test complex cascade with multiple paths"""
        db = ContraindicationDatabase()
        graph = ContraindicationGraph()
        analyzer = CascadeAnalyzer(db, graph)

        # Create diamond pattern: p1 -> p2, p3 -> p4
        principles = []
        for i in range(1, 5):
            p = Mock()
            p.id = f"cascade_p{i}"
            principles.append(p)
            graph.add_node(p)

        # Add edges forming diamond
        graph.add_edge(principles[0].id, principles[1].id, 0.9)
        graph.add_edge(principles[0].id, principles[2].id, 0.8)
        graph.add_edge(principles[1].id, principles[3].id, 0.7)
        graph.add_edge(principles[2].id, principles[3].id, 0.6)

        # Add contraindications
        for p in principles:
            contra = Contraindication(
                condition=f"condition_{p.id}", failure_mode="cascading", severity=0.5
            )
            db.register(p.id, contra)

        # Analyze
        impact = analyzer.analyze_cascade_impact(principles[0], max_depth=3)

        # Should detect multiple affected principles
        assert len(impact.affected_principles) > 0
        assert impact.max_severity > 0

    def test_performance_with_large_graph(self):
        """Test performance with larger graph"""
        graph = ContraindicationGraph()

        # Create large graph
        num_nodes = 100
        principles = []

        for i in range(num_nodes):
            p = Mock()
            p.id = f"perf_p{i}"
            principles.append(p)
            graph.add_node(p)

        # Add random edges
        np.random.seed(42)
        for i in range(num_nodes - 1):
            # Each node connects to 2-3 others
            targets = np.random.choice(
                range(i + 1, num_nodes), size=min(3, num_nodes - i - 1), replace=False
            )
            for target in targets:
                graph.add_edge(principles[i].id, principles[target].id, 0.7)

        # Test operations
        start_time = time.time()

        # Find cascades
        cascades = graph.find_cascades(principles[0].id, max_depth=3)

        # Calculate risks
        graph.calculate_cascade_risk(principles[0].id)

        # Get statistics
        stats = graph.get_statistics()

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max
        assert stats["total_nodes"] == num_nodes


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
