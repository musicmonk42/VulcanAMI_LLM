"""
Test suite for continual learning module
"""

from vulcan.learning.learning_types import (FeedbackData, LearningConfig,
                                            TaskInfo)
from vulcan.learning.continual_learning import (ContinualLearner,
                                                ContinualMetrics,
                                                EnhancedContinualLearner,
                                                ProgressiveNeuralNetwork)
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
import time
import threading
import tempfile
import shutil
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip(
    "torch", reason="PyTorch required for continual_learning tests"
)


# Import the modules to test

# Test configuration
TEST_EMBEDDING_DIM = 128
TEST_HIDDEN_DIM = 64
TEST_BATCH_SIZE = 8


class TestContinualLearner:
    """Test basic ContinualLearner for backward compatibility"""

    def test_basic_initialization(self):
        learner = ContinualLearner()
        assert hasattr(learner, "ewc_importance")
        assert hasattr(learner, "task_models")
        assert isinstance(learner.task_models, dict)

    def test_basic_process_experience(self):
        learner = ContinualLearner()
        experience = {"data": "test"}
        result = learner.process_experience(experience)
        assert result["processed"] == True
        assert "loss" in result


class TestProgressiveNeuralNetwork:
    """Test Progressive Neural Network implementation"""

    @pytest.fixture
    def prog_net(self):
        return ProgressiveNeuralNetwork(
            input_dim=TEST_EMBEDDING_DIM,
            hidden_dim=TEST_HIDDEN_DIM,
            output_dim=TEST_EMBEDDING_DIM,
        )

    def test_initialization(self, prog_net):
        assert len(prog_net.columns) == 0
        assert prog_net.input_dim == TEST_EMBEDDING_DIM
        assert prog_net.hidden_dim == TEST_HIDDEN_DIM
        assert prog_net.output_dim == TEST_EMBEDDING_DIM

    def test_add_column(self, prog_net):
        prog_net.add_column("task_0")
        assert len(prog_net.columns) == 1

        prog_net.add_column("task_1")
        assert len(prog_net.columns) == 2
        assert "0_to_1" in prog_net.lateral_connections

    def test_forward_pass(self, prog_net):
        prog_net.add_column("task_0")
        x = torch.randn(TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)

        output = prog_net(x, column_idx=0)
        assert output.shape == (TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)

    def test_lateral_connections(self, prog_net):
        prog_net.add_column("task_0")
        prog_net.add_column("task_1")

        x = torch.randn(TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)

        # Forward through second column should use lateral connections
        output = prog_net(x, column_idx=1)
        assert output.shape == (TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)
        assert "0_to_1" in prog_net.lateral_connections

    def test_invalid_column_index(self, prog_net):
        prog_net.add_column("task_0")
        x = torch.randn(TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)

        with pytest.raises(ValueError):
            prog_net(x, column_idx=5)


class TestEnhancedContinualLearner:
    """Test enhanced continual learner with all features"""

    @pytest.fixture
    def config(self):
        return LearningConfig(
            learning_rate=0.001,
            batch_size=4,
            ewc_lambda=10.0,
            replay_buffer_size=100,
            consolidation_threshold=10,
            rlhf_enabled=False,  # Disable RLHF for basic tests
            checkpoint_frequency=0,  # Disable checkpointing for tests
        )

    @pytest.fixture
    def learner(self, config, tmp_path):
        # FIXED: Removed the overly broad patch for 'Path' which caused
        # the PermissionError in test_save_and_load_state.
        # The learner's own save_path logic is now used, and
        # test_save_and_load_state provides a specific path using tmp_path.
        learner = EnhancedContinualLearner(
            embedding_dim=TEST_EMBEDDING_DIM,
            config=config,
            use_hierarchical=False,  # Disable hierarchical memory
            use_progressive=False,
        )
        # Override save_path to use tmp_path for cleanliness
        learner.save_path = tmp_path

        yield learner
        # Cleanup
        learner.shutdown()

    def test_initialization(self, learner):
        assert learner.embedding_dim == TEST_EMBEDDING_DIM
        assert hasattr(learner, "task_detector")
        assert hasattr(learner, "task_models")
        assert hasattr(learner, "shared_encoder")
        assert hasattr(learner, "general_model")
        assert isinstance(learner.task_models, nn.ModuleDict)

    def test_forward_pass(self, learner):
        x = torch.randn(TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)
        output = learner(x)
        assert output.shape == (TEST_BATCH_SIZE, TEST_EMBEDDING_DIM)

    def test_process_experience(self, learner):
        experience = {
            "embedding": np.random.randn(TEST_EMBEDDING_DIM),
            "reward": 0.5,
            "metadata": {"complexity": 0.3},
        }

        result = learner.process_experience(experience)

        assert "task_id" in result
        assert "loss" in result
        assert "output" in result
        assert result["adapted"] == True
        assert "continual_metrics" in result

    def test_new_task_creation(self, learner):
        # Process experiences with different signatures
        exp1 = {
            "embedding": np.ones(TEST_EMBEDDING_DIM) * 0.1,
            "metadata": {"modality": "vision"},
        }
        exp2 = {
            "embedding": np.ones(TEST_EMBEDDING_DIM) * 0.9,
            "metadata": {"modality": "audio"},
        }

        result1 = learner.process_experience(exp1)
        task1 = result1["task_id"]

        result2 = learner.process_experience(exp2)
        task2 = result2["task_id"]

        # Should create different tasks
        assert task1 in learner.task_models
        if task1 != task2:
            assert task2 in learner.task_models

    def test_ewc_consolidation(self, learner):
        # Process enough experiences to trigger consolidation
        for i in range(12):  # Consolidation threshold is 10
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM),
                "reward": np.random.random(),
            }
            learner.process_experience(exp)

        # Check that Fisher information was computed
        assert len(learner.fisher_information) > 0
        assert len(learner.optimal_params) > 0

    def test_experience_replay(self, learner):
        # Fill replay buffer
        task_id = None
        for i in range(50):
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM),
                "reward": np.random.random(),
            }
            result = learner.process_experience(exp)
            if task_id is None:
                task_id = result.get("task_id")  # Get the *actual* task_id

        # FIXED: Use the dynamically detected task_id instead of hardcoded "task_0"
        assert task_id is not None, "Failed to get a task_id from process_experience"
        result = learner._experience_replay(task_id, n_samples=10)
        assert isinstance(result, float)
        assert result >= 0

    def test_intelligent_replay(self, learner):
        # Create experiences for multiple tasks
        task_id = None
        for i in range(30):
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM) * (i % 3),
                "reward": np.random.random(),
            }
            result = learner.process_experience(exp)
            if task_id is None:
                task_id = result.get("task_id")  # Get the *actual* task_id

        # FIXED: Use the dynamically detected task_id instead of hardcoded "task_0"
        assert task_id is not None, "Failed to get a task_id from process_experience"
        result = learner._intelligent_experience_replay(task_id, n_samples=10)
        assert isinstance(result, float)
        assert result >= 0

    def test_distribution_shift_adaptation(self, learner):
        # Create initial distribution
        for i in range(20):
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM) * 0.1,
                "reward": 0.5,
            }
            learner.process_experience(exp)

        # Create shifted distribution
        new_samples = [np.random.randn(TEST_EMBEDDING_DIM) * 0.9 for _ in range(10)]

        result = learner.adapt_to_distribution_shift(new_samples)
        assert result["adapted"] == True
        assert "shift_magnitude" in result
        assert result["shift_magnitude"] > 0

    def test_continual_metrics_update(self, learner):
        # Process experiences for multiple tasks
        for task_idx in range(3):
            for i in range(15):
                exp = {
                    "embedding": np.random.randn(TEST_EMBEDDING_DIM) * (task_idx + 1),
                    "reward": np.random.random(),
                }
                learner.process_experience(exp)

        # Check metrics
        metrics = learner.continual_metrics
        assert hasattr(metrics, "backward_transfer")
        assert hasattr(metrics, "forward_transfer")
        assert hasattr(metrics, "average_accuracy")
        assert hasattr(metrics, "forgetting_measure")

    def test_thread_safety(self, learner):
        """Test thread-safe operations"""
        results = []
        errors = []

        def process_experiences():
            try:
                for i in range(10):
                    exp = {
                        "embedding": np.random.randn(TEST_EMBEDDING_DIM),
                        "reward": np.random.random(),
                    }
                    result = learner.process_experience(exp)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=process_experiences)
            threads.append(t)
            t.start()

        # Wait for threads to complete
        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0
        assert len(results) == 30  # 3 threads * 10 experiences

    def test_save_and_load_state(self, learner, tmp_path):
        # Process some experiences
        for i in range(20):
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM),
                "reward": np.random.random(),
            }
            learner.process_experience(exp)

        # Save state
        # FIXED: This path is now valid because the overly broad
        # patch in the 'learner' fixture was removed.
        save_path = tmp_path / "test_state.pkl"
        learner.save_state(str(save_path))
        assert save_path.exists()

        # Create new learner and load state
        config = LearningConfig(rlhf_enabled=False)
        new_learner = EnhancedContinualLearner(
            embedding_dim=TEST_EMBEDDING_DIM,
            config=config,
            use_hierarchical=False,
            use_progressive=False,
        )

        new_learner.load_state(str(save_path))

        # Check state was restored
        assert len(new_learner.task_order) == len(learner.task_order)
        assert new_learner.task_order == learner.task_order

    def test_get_statistics(self, learner):
        # Process some experiences
        for i in range(15):
            exp = {
                "embedding": np.random.randn(TEST_EMBEDDING_DIM),
                "reward": np.random.random(),
            }
            learner.process_experience(exp)

        stats = learner.get_statistics()

        assert "num_tasks" in stats
        assert "task_order" in stats
        assert "total_experiences" in stats
        assert "replay_buffer_size" in stats
        assert "continual_metrics" in stats
        assert stats["total_experiences"] == 15

    # FIXED: Moved this test from the (now deleted) TestPackNetCapacity
    # class into TestEnhancedContinualLearner so it can access the 'learner' fixture.
    def test_capacity_allocation(self, learner):
        """Test PackNet-style parameter isolation"""
        # Create multiple tasks
        for task_idx in range(3):
            learner._on_new_task(f"task_{task_idx}")

        # Check masks were created
        assert len(learner.task_masks) >= 2

        # Check capacity is being allocated
        free_capacity = learner.free_capacity.mean().item()
        assert free_capacity < 1.0  # Some capacity should be allocated


class TestProgressiveMode:
    """Test progressive neural network integration"""

    @pytest.fixture
    def prog_learner(self, tmp_path):
        config = LearningConfig(rlhf_enabled=False)
        # FIXED: Removed the broad 'Path' patch here as well
        learner = EnhancedContinualLearner(
            embedding_dim=TEST_EMBEDDING_DIM,
            config=config,
            use_hierarchical=False,
            use_progressive=True,
        )
        learner.save_path = tmp_path
        yield learner
        learner.shutdown()

    def test_progressive_initialization(self, prog_learner):
        assert hasattr(prog_learner, "progressive_network")
        assert isinstance(prog_learner.progressive_network, ProgressiveNeuralNetwork)

    def test_progressive_new_task(self, prog_learner):
        # Process experiences to create tasks
        exp1 = {"embedding": np.ones(TEST_EMBEDDING_DIM) * 0.1}
        exp2 = {"embedding": np.ones(TEST_EMBEDDING_DIM) * 0.9}

        result1 = prog_learner.process_experience(exp1)
        result2 = prog_learner.process_experience(exp2)

        # Check columns were added
        assert len(prog_learner.progressive_network.columns) > 0


# This class is now empty and can be removed, but
# I will leave it to match the original structure.
class TestPackNetCapacity:
    """Test PackNet-style parameter isolation"""

    pass


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def full_learner(self, tmp_path):
        config = LearningConfig(rlhf_enabled=True, checkpoint_frequency=50)
        # FIXED: Removed the broad 'Path' patch here as well
        learner = EnhancedContinualLearner(
            embedding_dim=TEST_EMBEDDING_DIM,
            config=config,
            use_hierarchical=False,
            use_progressive=True,
        )
        learner.save_path = tmp_path
        yield learner
        learner.shutdown()

    def test_full_pipeline(self, full_learner):
        """Test complete learning pipeline"""
        experiences_processed = 0

        # Simulate learning over time
        for epoch in range(3):
            for i in range(20):
                exp = {
                    "embedding": np.random.randn(TEST_EMBEDDING_DIM) * (epoch + 1),
                    "reward": np.random.random(),
                    "feedback": np.random.random(),
                    "metadata": {"epoch": epoch, "step": i},
                }

                result = full_learner.process_experience(exp)
                experiences_processed += 1

                # If the source code fix wasn't applied, this will fail
                assert "task_id" in result, (
                    f"Test failed, process_experience returned: {result}"
                )
                assert "loss" in result
                assert result["adapted"] == True

        # Check final state
        stats = full_learner.get_statistics()
        assert stats["total_experiences"] == experiences_processed
        assert stats["num_tasks"] > 0

        # Check RLHF manager if enabled
        if full_learner.rlhf_manager:
            rlhf_stats = full_learner.rlhf_manager.get_statistics()
            assert rlhf_stats["total_feedback"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
