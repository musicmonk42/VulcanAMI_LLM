"""
Test suite for metacognition module
"""

import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for metacognition tests")

import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from vulcan.config import EMBEDDING_DIM, HIDDEN_DIM
from vulcan.learning.metacognition import (CausalRelation,
                                           CompositionalUnderstanding,
                                           ConfidenceEstimator,
                                           MetaCognitiveMonitor,
                                           ReasoningPhase, ReasoningStep)


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TestReasoningTypes:
    """Test reasoning types and enums"""

    def test_reasoning_phase_enum(self):
        """Test ReasoningPhase enum values"""
        assert ReasoningPhase.PERCEPTION.value == "perception"
        assert ReasoningPhase.PLANNING.value == "planning"
        assert ReasoningPhase.EXECUTION.value == "execution"
        assert ReasoningPhase.LEARNING.value == "learning"
        assert ReasoningPhase.REFLECTION.value == "reflection"

    def test_reasoning_step_creation(self):
        """Test ReasoningStep dataclass"""
        step = ReasoningStep(
            phase=ReasoningPhase.PLANNING,
            content="Test content",
            confidence=0.8,
            timestamp=time.time(),
        )

        assert step.phase == ReasoningPhase.PLANNING
        assert step.content == "Test content"
        assert step.confidence == 0.8
        assert step.metadata == {}

    def test_reasoning_step_with_metadata(self):
        """Test ReasoningStep with metadata"""
        metadata = {"source": "test", "iterations": 5}
        step = ReasoningStep(
            phase=ReasoningPhase.EXECUTION,
            content={"action": "test_action"},
            confidence=0.6,
            timestamp=time.time(),
            metadata=metadata,
        )

        assert step.metadata == metadata
        assert step.metadata["source"] == "test"

    def test_causal_relation_creation(self):
        """Test CausalRelation dataclass"""
        relation = CausalRelation(
            cause="high_learning_rate",
            effect="instability",
            strength=0.8,
            confidence=0.7,
            evidence_count=10,
        )

        assert relation.cause == "high_learning_rate"
        assert relation.effect == "instability"
        assert relation.strength == 0.8
        assert relation.confidence == 0.7
        assert relation.evidence_count == 10


class TestMetaCognitiveMonitor:
    """Test MetaCognitiveMonitor class"""

    @pytest.fixture
    def model(self):
        """Create test model"""
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """Create test optimizer"""
        return optim.Adam(model.parameters(), lr=0.001)

    @pytest.fixture
    def monitor(self):
        """Create MetaCognitiveMonitor instance"""
        return MetaCognitiveMonitor()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.model_ref is None
        assert monitor.optimizer_ref is None
        assert len(monitor.learning_history) == 0
        assert monitor.self_model["learning_style"] == "balanced"
        assert monitor.self_model["confidence_calibration"] == 1.0

    def test_set_model_optimizer(self, monitor, model, optimizer):
        """Test setting model and optimizer references"""
        monitor.set_model_optimizer(model, optimizer)

        assert monitor.model_ref == model
        assert monitor.optimizer_ref == optimizer

    def test_update_self_model(self, monitor):
        """Test updating self model"""
        metrics = {
            "loss": 0.5,
            "modality": "text",
            "predicted_confidence": 0.8,
            "actual_performance": 0.6,
        }

        monitor.update_self_model(metrics)

        assert len(monitor.learning_history) == 1
        assert monitor.learning_history[0] == metrics
        assert len(monitor.confidence_history) == 1

    def test_confidence_calibration(self, monitor):
        """Test confidence calibration updates"""
        # Add multiple confidence predictions
        for i in range(15):
            metrics = {
                "predicted_confidence": 0.9,
                "actual_performance": 0.5,  # Overconfident
                "loss": 0.4,
            }
            monitor.update_self_model(metrics)

        # Should have adjusted calibration
        assert monitor.self_model["confidence_calibration"] < 1.0

    def test_strength_weakness_identification(self, monitor):
        """Test identifying strengths and weaknesses"""
        # Add good performance for text
        for i in range(15):
            monitor.update_self_model(
                {
                    "modality": "text",
                    "loss": 0.2,  # Good performance
                }
            )

        # Add poor performance for vision
        for i in range(15):
            monitor.update_self_model(
                {
                    "modality": "vision",
                    "loss": 0.8,  # Poor performance
                }
            )

        assert "text" in monitor.self_model["strengths"]
        assert "vision" in monitor.self_model["weaknesses"]

    def test_introspect_reasoning_empty(self, monitor):
        """Test introspection with empty reasoning trace"""
        result = monitor.introspect_reasoning([])

        assert result["quality_score"] == 0
        assert "No reasoning trace" in result["analysis"]

    def test_introspect_reasoning_with_steps(self, monitor):
        """Test introspection with reasoning steps"""
        trace = [
            ReasoningStep(
                phase=ReasoningPhase.PERCEPTION,
                content="Perceive input",
                confidence=0.7,
                timestamp=time.time(),
            ),
            ReasoningStep(
                phase=ReasoningPhase.PLANNING,
                content="Plan action",
                confidence=0.8,
                timestamp=time.time(),
            ),
            ReasoningStep(
                phase=ReasoningPhase.EXECUTION,
                content="Execute action",
                confidence=0.9,
                timestamp=time.time(),
            ),
            ReasoningStep(
                phase=ReasoningPhase.LEARNING,
                content="Learn from outcome",
                confidence=0.6,
                timestamp=time.time(),
            ),
        ]

        result = monitor.introspect_reasoning(trace)

        assert result["quality_score"] > 0
        assert "metrics" in result
        assert (
            result["metrics"]["completeness"] == 1.0
        )  # 4/4 phases (all expected phases present)
        assert "suggestions" in result

    def test_introspect_reasoning_with_dicts(self, monitor):
        """Test introspection with dict-based reasoning trace"""
        trace = [
            {"phase": "perception", "content": "test", "confidence": 0.5},
            {"phase": "planning", "content": "test", "confidence": 0.6},
            {"phase": "execution", "content": "test", "confidence": 0.7},
        ]

        result = monitor.introspect_reasoning(trace)

        assert result["quality_score"] > 0
        assert len(result["metrics"]) > 0

    def test_analyze_learning_efficiency_insufficient_data(self, monitor):
        """Test analysis with insufficient data"""
        result = monitor.analyze_learning_efficiency()

        assert result["status"] == "insufficient_data"
        assert result["samples"] == 0

    def test_analyze_learning_efficiency_with_data(self, monitor, model, optimizer):
        """Test learning efficiency analysis"""
        monitor.set_model_optimizer(model, optimizer)

        # Add learning history
        for i in range(20):
            monitor.update_self_model(
                {
                    "loss": 0.5 + i * 0.01,  # Increasing loss (bad)
                    "modality": "text",
                }
            )

        result = monitor.analyze_learning_efficiency()

        assert "avg_loss" in result
        assert "loss_trend" in result
        assert "recommendations" in result
        assert result["loss_trend"] > 0  # Should detect increasing trend

    def test_improvement_strategies(self, monitor, model, optimizer):
        """Test automatic improvement strategies"""
        monitor.set_model_optimizer(model, optimizer)

        # Test learning rate reduction
        old_lr = optimizer.param_groups[0]["lr"]
        result = monitor._strategy_reduce_learning_rate()
        new_lr = optimizer.param_groups[0]["lr"]

        assert new_lr < old_lr
        assert "Reduced learning rate" in result

    def test_strategy_increase_regularization(self, monitor, model):
        """Test regularization increase strategy"""
        monitor.set_model_optimizer(model, None)

        # Get initial dropout rate
        dropout_module = None
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                dropout_module = module
                break

        old_p = dropout_module.p if dropout_module else 0.1
        result = monitor._strategy_increase_regularization()

        if dropout_module:
            assert dropout_module.p > old_p
            assert "dropout rates" in result

    def test_confidence_calibration_strategy(self, monitor):
        """Test confidence calibration strategies"""
        old_calibration = monitor.self_model["confidence_calibration"]

        # Test calibration reduction (for overconfidence)
        result = monitor._strategy_calibrate_confidence()
        assert monitor.self_model["confidence_calibration"] < old_calibration

        # Test confidence boost (for underconfidence)
        result = monitor._strategy_boost_confidence()
        assert monitor.self_model["confidence_calibration"] > 0

    def test_causal_graph_updates(self, monitor):
        """Test causal graph construction"""
        monitor.update_self_model(
            {"cause": "high_lr", "effect": "instability", "correlation": 0.8}
        )

        assert monitor.causal_graph.has_node("high_lr")
        assert monitor.causal_graph.has_node("instability")
        assert monitor.causal_graph.has_edge("high_lr", "instability")

    def test_infer_causal_chain(self, monitor):
        """Test causal chain inference"""
        # Build causal graph
        monitor.update_self_model({"cause": "A", "effect": "B", "correlation": 0.8})
        monitor.update_self_model({"cause": "B", "effect": "C", "correlation": 0.7})

        chain = monitor.infer_causal_chain("A", "C")
        assert chain == ["A", "B", "C"]

        # Test non-existent path
        chain = monitor.infer_causal_chain("C", "A")
        assert chain is None

    def test_compute_trend(self, monitor):
        """Test trend computation"""
        # Increasing values
        trend = monitor._compute_trend([1, 2, 3, 4, 5])
        assert trend > 0

        # Decreasing values
        trend = monitor._compute_trend([5, 4, 3, 2, 1])
        assert trend < 0

        # Constant values
        trend = monitor._compute_trend([3, 3, 3, 3])
        assert abs(trend) < 0.01

    def test_save_and_load_state(self, monitor, temp_dir):
        """Test saving and loading monitor state"""
        monitor.save_path = Path(temp_dir)

        # Add some data
        monitor.update_self_model({"loss": 0.5, "modality": "text"})
        monitor.self_model["strengths"].append("text")

        # Save state
        filepath = monitor.save_state()

        # Create new monitor and load
        new_monitor = MetaCognitiveMonitor()
        new_monitor.load_state(filepath)

        assert len(new_monitor.learning_history) == 1
        assert "text" in new_monitor.self_model["strengths"]

    def test_improvement_audit(self, monitor, model, optimizer):
        """Test improvement audit trail"""
        monitor.set_model_optimizer(model, optimizer)

        # Apply an improvement
        monitor._apply_improvements(
            {
                "recommendations": [
                    {
                        "issue": "high_loss",
                        "auto_fix": True,
                        "priority": "high",
                        "suggestion": "test",
                    }
                ],
                "avg_loss": 0.8,
                "loss_std": 0.1,
            }
        )

        # Check audit trail
        history = monitor.get_improvement_history()
        assert len(history) > 0
        assert history[0]["issue"] == "high_loss"

    def test_thread_safety(self, monitor):
        """Test thread safety of operations"""
        import threading

        def update_task():
            for i in range(10):
                monitor.update_self_model({"loss": np.random.random()})

        threads = [threading.Thread(target=update_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have accumulated updates without errors
        assert len(monitor.learning_history) == 50


class TestConfidenceEstimator:
    """Test ConfidenceEstimator class"""

    @pytest.fixture
    def estimator(self):
        """Create ConfidenceEstimator instance"""
        return ConfidenceEstimator()

    def test_initialization(self, estimator):
        """Test estimator initialization"""
        assert estimator.temperature == 1.0
        assert len(estimator.calibration_history) == 0

    def test_estimate_confidence_softmax(self, estimator):
        """Test softmax confidence estimation"""
        logits = torch.randn(10)
        confidence = estimator.estimate_confidence(logits, method="softmax")

        assert 0 <= confidence <= 1

    def test_estimate_confidence_entropy(self, estimator):
        """Test entropy-based confidence estimation"""
        logits = torch.randn(10)
        confidence = estimator.estimate_confidence(logits, method="entropy")

        assert 0 <= confidence <= 1

    def test_estimate_confidence_margin(self, estimator):
        """Test margin-based confidence estimation"""
        logits = torch.randn(10)
        confidence = estimator.estimate_confidence(logits, method="margin")

        assert 0 <= confidence <= 1

    def test_calibrate_temperature(self, estimator):
        """Test temperature calibration"""
        predictions = [
            (0.9, True),  # High confidence, correct
            (0.8, True),  # High confidence, correct
            (0.9, False),  # High confidence, wrong (overconfident)
            (0.7, True),
            (0.6, False),
            (0.3, False),
            (0.2, False),
            (0.1, False),
            (0.4, True),
            (0.5, True),
        ]

        old_temp = estimator.temperature
        estimator.calibrate_temperature(predictions)

        # Temperature should change
        assert estimator.temperature != old_temp


class TestCompositionalUnderstanding:
    """Test CompositionalUnderstanding class"""

    @pytest.fixture
    def compositor(self):
        """Create CompositionalUnderstanding instance"""
        return CompositionalUnderstanding()

    def test_initialization(self, compositor):
        """Test compositor initialization"""
        assert compositor.embedding_dim == EMBEDDING_DIM
        assert len(compositor.concept_hierarchy) == 0
        assert len(compositor.primitive_concepts) == 0

    def test_compose_concepts_neural(self, compositor):
        """Test neural concept composition"""
        concept1 = torch.randn(EMBEDDING_DIM)
        concept2 = torch.randn(EMBEDDING_DIM)

        composed = compositor.compose_concepts_neural(concept1, concept2)

        assert composed.shape == (EMBEDDING_DIM,)

    def test_decompose_concept_neural(self, compositor):
        """Test neural concept decomposition"""
        concept = torch.randn(EMBEDDING_DIM)

        comp1, comp2 = compositor.decompose_concept_neural(concept)

        assert comp1.shape == (EMBEDDING_DIM,)
        assert comp2.shape == (EMBEDDING_DIM,)

    def test_train_composition(self, compositor):
        """Test training composition network"""
        # Create training data
        concepts = []
        for _ in range(5):
            comp1 = torch.randn(EMBEDDING_DIM)
            comp2 = torch.randn(EMBEDDING_DIM)
            target = torch.randn(EMBEDDING_DIM)
            concepts.append((comp1, comp2, target))

        result = compositor.train_composition(concepts, epochs=2)

        assert "avg_loss" in result
        assert result["avg_loss"] > 0
        assert len(compositor.training_history) > 0

    def test_train_decomposition(self, compositor):
        """Test training decomposition network"""
        # Create training data
        concepts = []
        for _ in range(5):
            composed = torch.randn(EMBEDDING_DIM)
            target1 = torch.randn(EMBEDDING_DIM)
            target2 = torch.randn(EMBEDDING_DIM)
            concepts.append((composed, target1, target2))

        result = compositor.train_decomposition(concepts, epochs=2)

        assert "avg_loss" in result
        assert result["avg_loss"] > 0

    def test_learn_composition(self, compositor):
        """Test learning compositional structure"""
        concept = torch.randn(EMBEDDING_DIM)
        components = [torch.randn(EMBEDDING_DIM) for _ in range(3)]

        result = compositor.learn_composition(concept, components)

        assert "composed" in result
        assert "reconstruction_error" in result
        assert "emergent_properties" in result
        assert "composition_key" in result
        assert result["components_count"] == 3

    def test_hierarchical_composition(self, compositor):
        """Test hierarchical composition"""
        primitives = [torch.randn(EMBEDDING_DIM) for _ in range(4)]

        composition_tree = {
            "op": "compose",
            "left": {"op": "primitive", "index": 0},
            "right": {
                "op": "compose",
                "left": {"op": "primitive", "index": 1},
                "right": {"op": "primitive", "index": 2},
            },
        }

        result = compositor.hierarchical_composition(primitives, composition_tree)

        assert result.shape == (EMBEDDING_DIM,)

    def test_analyze_compositionality(self, compositor):
        """Test compositionality analysis"""
        concept = torch.randn(EMBEDDING_DIM)
        primitives = [torch.randn(EMBEDDING_DIM) for _ in range(5)]

        result = compositor.analyze_compositionality(concept, primitives)

        assert "best_composition" in result
        assert "decomposition" in result
        assert "decomposition_error" in result
        assert "compositionality_score" in result
        assert "is_primitive" in result

    def test_discover_primitives(self, compositor):
        """Test primitive discovery"""
        concepts = [torch.randn(EMBEDDING_DIM) for _ in range(20)]

        primitives = compositor.discover_primitives(concepts, max_primitives=5)

        assert len(primitives) <= 5
        assert len(compositor.primitive_concepts) > 0

        # Check primitives are stored
        for key in compositor.primitive_concepts:
            assert key in compositor.concept_embeddings

    def test_edge_cases(self, compositor):
        """Test edge cases"""
        # Learn composition with single component
        result = compositor.learn_composition(
            torch.randn(EMBEDDING_DIM),
            [torch.randn(EMBEDDING_DIM)],  # Only one component
        )
        assert result["error"] == "Need at least 2 components"

        # Hierarchical composition with invalid op
        with pytest.raises(ValueError):
            compositor.hierarchical_composition(
                [torch.randn(EMBEDDING_DIM)], {"op": "invalid"}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
