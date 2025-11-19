"""
Comprehensive test suite for data_augmentor.py
"""

import pytest
import random
from unittest.mock import Mock, MagicMock

from data_augmentor import (
    DataAugmentor,
    GraphValidator,
    SemanticMutator,
    AugmentationMetrics,
    MAX_COMPLEXITY,
    MAX_NODES,
    MAX_EDGES,
    MAX_BATCH_SIZE,
)


@pytest.fixture
def base_graph():
    """Create base test graph."""
    return {
        "graph_id": "test_graph",
        "nodes": [
            {"id": "n1", "label": "Sentiment", "properties": {}},
            {"id": "n2", "label": "Score", "properties": {}},
            {"id": "n3", "label": "Data", "properties": {}}
        ],
        "edges": [
            {"from": "n1", "to": "n2", "weight": 0.5, "properties": {}},
            {"from": "n2", "to": "n3", "weight": 0.7, "properties": {}}
        ],
        "metadata": {}
    }


@pytest.fixture
def augmentor():
    """Create data augmentor with fixed seed."""
    return DataAugmentor(random_seed=42)


class TestGraphValidator:
    """Test GraphValidator class."""
    
    def test_validate_valid_graph(self, base_graph):
        """Test validating valid graph."""
        valid, error = GraphValidator.validate_graph(base_graph)
        
        assert valid is True
        assert error is None
    
    def test_validate_not_dict(self):
        """Test validating non-dict input."""
        valid, error = GraphValidator.validate_graph("not a dict")
        
        assert valid is False
        assert "must be a dictionary" in error
    
    def test_validate_missing_nodes(self):
        """Test validating graph without nodes."""
        graph = {"edges": []}
        
        valid, error = GraphValidator.validate_graph(graph)
        
        assert valid is False
        assert "Missing 'nodes'" in error
    
    def test_validate_nodes_not_list(self):
        """Test validating graph with non-list nodes."""
        graph = {"nodes": "not a list"}
        
        valid, error = GraphValidator.validate_graph(graph)
        
        assert valid is False
        assert "'nodes' must be a list" in error
    
    def test_validate_too_many_nodes(self):
        """Test validating graph with too many nodes."""
        graph = {
            "nodes": [{"id": f"n{i}"} for i in range(MAX_NODES + 1)],
            "edges": []
        }
        
        valid, error = GraphValidator.validate_graph(graph)
        
        assert valid is False
        assert "Too many nodes" in error
    
    def test_validate_duplicate_node_ids(self):
        """Test validating graph with duplicate node IDs."""
        graph = {
            "nodes": [
                {"id": "n1", "label": "Node1"},
                {"id": "n1", "label": "Node2"}
            ],
            "edges": []
        }
        
        valid, error = GraphValidator.validate_graph(graph)
        
        assert valid is False
        assert "Duplicate node id" in error
    
    def test_validate_invalid_edge(self):
        """Test validating graph with invalid edge."""
        graph = {
            "nodes": [{"id": "n1"}],
            "edges": [{"from": "n1", "to": "nonexistent"}]
        }
        
        valid, error = GraphValidator.validate_graph(graph)
        
        assert valid is False
        assert "non-existent" in error
    
    def test_validate_node(self):
        """Test validating individual node."""
        valid, error = GraphValidator.validate_node({"id": "n1", "label": "Test"})
        
        assert valid is True
        assert error is None
    
    def test_validate_node_missing_id(self):
        """Test validating node without ID."""
        valid, error = GraphValidator.validate_node({"label": "Test"})
        
        assert valid is False
        assert "missing 'id'" in error


class TestSemanticMutator:
    """Test SemanticMutator class."""
    
    def test_mutate_node_semantic(self):
        """Test semantic node mutation."""
        node = {"id": "n1", "label": "Sentiment", "properties": {}}
        rng = random.Random(42)
        
        mutated = SemanticMutator.mutate_node_semantic(node, rng)
        
        assert mutated["id"] == "n1"
        assert mutated["label"] != "Sentiment"  # Should be changed
        assert mutated["properties"]["mutation_type"] == "semantic"
        assert mutated["properties"]["original_label"] == "Sentiment"
    
    def test_mutate_unknown_type(self):
        """Test mutating node with unknown type."""
        node = {"id": "n1", "label": "UnknownType", "properties": {}}
        rng = random.Random(42)
        
        mutated = SemanticMutator.mutate_node_semantic(node, rng)
        
        # Should apply generic mutation
        assert mutated["label"] != "UnknownType"
        assert mutated["properties"]["mutation_type"] == "semantic"
    
    def test_create_semantic_edge(self):
        """Test creating semantic edge."""
        rng = random.Random(42)
        
        edge = SemanticMutator.create_semantic_edge("n1", "n2", rng)
        
        assert edge["from"] == "n1"
        assert edge["to"] == "n2"
        assert "weight" in edge
        assert "relation_type" in edge
        assert edge["properties"]["semantic"] is True
    
    def test_create_semantic_edge_specific_relation(self):
        """Test creating edge with specific relation."""
        rng = random.Random(42)
        
        edge = SemanticMutator.create_semantic_edge("n1", "n2", rng, relation_type="depends_on")
        
        assert edge["relation_type"] == "depends_on"


class TestDataAugmentor:
    """Test DataAugmentor class."""
    
    def test_initialization(self):
        """Test augmentor initialization."""
        aug = DataAugmentor(random_seed=42)
        
        assert aug.counter == 0
        assert len(aug.audit_log) == 0
        assert aug.metrics.total_generated == 0
    
    def test_initialization_with_audit_hook(self):
        """Test initialization with audit hook."""
        hook = Mock()
        aug = DataAugmentor(audit_hook=hook)
        
        assert aug.audit_hook == hook
    
    def test_hash_proposal(self, augmentor, base_graph):
        """Test proposal hashing."""
        hash1 = augmentor._hash_proposal(base_graph, "synthetic")
        hash2 = augmentor._hash_proposal(base_graph, "synthetic")
        
        # Same graph and kind should produce same hash
        assert hash1 == hash2
    
    def test_hash_proposal_different_kind(self, augmentor, base_graph):
        """Test hashing with different kinds."""
        hash1 = augmentor._hash_proposal(base_graph, "synthetic")
        hash2 = augmentor._hash_proposal(base_graph, "counterfactual")
        
        # Different kind should produce different hash
        assert hash1 != hash2
    
    def test_check_duplicate(self, augmentor, base_graph):
        """Test duplicate detection."""
        hash1 = augmentor._hash_proposal(base_graph, "synthetic")
        
        is_dup1 = augmentor._check_duplicate(hash1)
        is_dup2 = augmentor._check_duplicate(hash1)
        
        assert is_dup1 is False  # First time
        assert is_dup2 is True   # Second time (duplicate)
        assert augmentor.metrics.duplicates_detected == 1
    
    def test_calculate_quality_score(self, augmentor, base_graph):
        """Test quality score calculation."""
        score = augmentor._calculate_quality_score(base_graph)
        
        assert 0.0 <= score <= 1.0
    
    def test_generate_synthetic_proposal(self, augmentor, base_graph):
        """Test synthetic proposal generation."""
        proposal = augmentor.generate_synthetic_proposal(base_graph, complexity=2)
        
        assert proposal["metadata"]["synthetic"] is True
        assert proposal["metadata"]["complexity"] == 2
        assert "quality_score" in proposal["metadata"]
        assert augmentor.metrics.synthetic_count == 1
    
    def test_generate_synthetic_invalid_graph(self, augmentor):
        """Test synthetic generation with invalid graph."""
        invalid_graph = {"invalid": "graph"}
        
        with pytest.raises(ValueError, match="Invalid base graph"):
            augmentor.generate_synthetic_proposal(invalid_graph)
    
    def test_generate_synthetic_invalid_complexity(self, augmentor, base_graph):
        """Test synthetic generation with invalid complexity."""
        with pytest.raises(ValueError, match="Complexity must be"):
            augmentor.generate_synthetic_proposal(base_graph, complexity=0)
        
        with pytest.raises(ValueError, match="Complexity must be"):
            augmentor.generate_synthetic_proposal(base_graph, complexity=MAX_COMPLEXITY + 1)
    
    def test_counterfactual_proposal(self, augmentor, base_graph):
        """Test counterfactual proposal generation."""
        proposal = augmentor.counterfactual_proposal(base_graph, invert_all=True)
        
        assert proposal["metadata"]["counterfactual"] is True
        assert proposal["metadata"]["invert_all"] is True
        
        # Check node labels changed
        original_label = base_graph["nodes"][0]["label"]
        new_label = proposal["nodes"][0]["label"]
        assert original_label != new_label
        
        # Check edge weights inverted
        original_weight = base_graph["edges"][0]["weight"]
        new_weight = proposal["edges"][0]["weight"]
        assert new_weight == -original_weight
        
        assert augmentor.metrics.counterfactual_count == 1
    
    def test_counterfactual_partial_inversion(self, augmentor, base_graph):
        """Test partial counterfactual inversion."""
        proposal = augmentor.counterfactual_proposal(base_graph, invert_all=False)
        
        # Only first node/edge should be inverted
        assert proposal["nodes"][0]["properties"].get("counterfactual") is True
        assert proposal["edges"][0]["properties"].get("counterfactual") is True
    
    def test_adversarial_proposal(self, augmentor, base_graph):
        """Test adversarial proposal generation."""
        proposal = augmentor.adversarial_proposal(base_graph)
        
        assert proposal["metadata"]["adversarial"] is True
        assert augmentor.metrics.adversarial_count == 1
    
    def test_adversarial_targeted(self, augmentor, base_graph):
        """Test targeted adversarial proposal."""
        proposal = augmentor.adversarial_proposal(base_graph, targeted_node="n2")
        
        assert proposal["metadata"]["targeted_node"] == "n2"
        
        # Check if target was modified
        target_node = next(n for n in proposal["nodes"] if n["id"] == "n2")
        assert target_node["properties"].get("adversarial") is True
    
    def test_adversarial_empty_graph(self, augmentor):
        """Test adversarial with empty graph."""
        empty = {"nodes": [], "edges": []}
        
        with pytest.raises(ValueError, match="empty graph"):
            augmentor.adversarial_proposal(empty)
    
    def test_curriculum_batch(self, augmentor, base_graph):
        """Test curriculum batch generation."""
        batch = augmentor.curriculum_batch(base_graph, n=9)
        
        assert len(batch) == 9
        
        # Check distribution
        synthetic = sum(1 for p in batch if p["metadata"].get("synthetic"))
        counterfactual = sum(1 for p in batch if p["metadata"].get("counterfactual"))
        adversarial = sum(1 for p in batch if p["metadata"].get("adversarial"))
        
        assert synthetic + counterfactual + adversarial == 9
    
    def test_curriculum_batch_invalid_size(self, augmentor, base_graph):
        """Test curriculum batch with invalid size."""
        with pytest.raises(ValueError, match="Batch size"):
            augmentor.curriculum_batch(base_graph, n=0)
        
        with pytest.raises(ValueError, match="Batch size"):
            augmentor.curriculum_batch(base_graph, n=MAX_BATCH_SIZE + 1)
    
    def test_get_metrics(self, augmentor, base_graph):
        """Test getting metrics."""
        # Generate some proposals
        augmentor.generate_synthetic_proposal(base_graph, complexity=1)
        augmentor.counterfactual_proposal(base_graph)
        augmentor.adversarial_proposal(base_graph)
        
        metrics = augmentor.get_metrics()
        
        assert metrics["total_generated"] == 3
        assert metrics["synthetic_count"] == 1
        assert metrics["counterfactual_count"] == 1
        assert metrics["adversarial_count"] == 1
    
    def test_get_diversity_score(self, augmentor, base_graph):
        """Test diversity score calculation."""
        # Generate varied proposals
        augmentor.generate_synthetic_proposal(base_graph, complexity=1)
        augmentor.generate_synthetic_proposal(base_graph, complexity=3)
        
        diversity = augmentor.get_diversity_score()
        
        assert 0.0 <= diversity <= 1.0
    
    def test_get_diversity_score_insufficient_data(self, augmentor):
        """Test diversity score with insufficient data."""
        diversity = augmentor.get_diversity_score()
        
        assert diversity == 0.0
    
    def test_reset_metrics(self, augmentor, base_graph):
        """Test resetting metrics."""
        # Generate some data
        augmentor.generate_synthetic_proposal(base_graph, complexity=1)
        
        augmentor.reset_metrics()
        
        metrics = augmentor.get_metrics()
        assert metrics["total_generated"] == 0
        assert len(augmentor.generated_graphs) == 0


class TestAuditLogging:
    """Test audit logging."""
    
    def test_audit_hook_called(self, base_graph):
        """Test audit hook is called."""
        hook = Mock()
        aug = DataAugmentor(audit_hook=hook, random_seed=42)
        
        aug.generate_synthetic_proposal(base_graph, complexity=1)
        
        assert hook.called
        call_args = hook.call_args[0][0]
        assert "kind" in call_args
        assert "quality_score" in call_args
    
    def test_audit_log_populated(self, augmentor, base_graph):
        """Test audit log is populated."""
        augmentor.generate_synthetic_proposal(base_graph, complexity=1)
        
        assert len(augmentor.audit_log) == 1
        entry = augmentor.audit_log[0]
        
        assert entry["kind"] == "synthetic"
        assert "timestamp" in entry
        assert "quality_score" in entry


class TestThreadSafety:
    """Test thread safety."""
    
    def test_concurrent_generation(self, base_graph):
        """Test concurrent proposal generation."""
        aug = DataAugmentor(random_seed=42)
        
        results = []
        errors = []
        
        def generate():
            try:
                proposal = aug.generate_synthetic_proposal(base_graph, complexity=1)
                results.append(proposal)
            except Exception as e:
                errors.append(e)
        
        import threading
        threads = [threading.Thread(target=generate) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert len(errors) == 0
        assert aug.metrics.total_generated == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])