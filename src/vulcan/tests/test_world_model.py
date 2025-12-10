"""
Test suite for unified world model
"""

import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for world_model tests")

import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from vulcan.learning.world_model import (
    UnifiedWorldModel,
    PlanningAlgorithm,
    WorldState,
    MCTSNode,
    MultiHeadAttention,
    AttentionBlock,
    CuriosityModule,
    StateAbstractor,
)
from vulcan.config import EMBEDDING_DIM, HIDDEN_DIM


class TestWorldModelTypes:
    """Test world model types and enums"""

    def test_planning_algorithm_enum(self):
        """Test PlanningAlgorithm enum values"""
        assert PlanningAlgorithm.GREEDY.value == "greedy"
        assert PlanningAlgorithm.BEAM_SEARCH.value == "beam_search"
        assert PlanningAlgorithm.MCTS.value == "mcts"
        assert PlanningAlgorithm.CEM.value == "cem"
        assert PlanningAlgorithm.MPPI.value == "mppi"

    def test_world_state_creation(self):
        """Test WorldState dataclass"""
        embedding = torch.randn(EMBEDDING_DIM)
        state = WorldState(
            embedding=embedding, uncertainty=0.5, value=10.0, visit_count=5
        )

        assert torch.equal(state.embedding, embedding)
        assert state.uncertainty == 0.5
        assert state.value == 10.0
        assert state.visit_count == 5
        assert state.metadata == {}

    def test_mcts_node_creation(self):
        """Test MCTSNode creation and properties"""
        state = torch.randn(EMBEDDING_DIM)
        node = MCTSNode(state)

        assert torch.equal(node.state, state)
        assert node.parent is None
        assert node.action is None
        assert len(node.children) == 0
        assert node.visit_count == 0
        assert node.value == 0

    def test_mcts_node_ucb_score(self):
        """Test MCTS node UCB score calculation"""
        parent = MCTSNode(torch.randn(EMBEDDING_DIM))
        parent.visit_count = 10

        child = MCTSNode(torch.randn(EMBEDDING_DIM), parent=parent)
        child.visit_count = 2
        child.value_sum = 5
        child.prior = 0.3

        score = child.ucb_score(c_puct=1.4)
        assert score > 0


class TestAttentionModules:
    """Test attention modules"""

    def test_multi_head_attention(self):
        """Test MultiHeadAttention module"""
        attention = MultiHeadAttention(dim=EMBEDDING_DIM, num_heads=4)

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, EMBEDDING_DIM)

        output = attention(x)

        assert output.shape == (batch_size, seq_len, EMBEDDING_DIM)

    def test_attention_block(self):
        """Test AttentionBlock module"""
        block = AttentionBlock(dim=HIDDEN_DIM)

        # Test 1D input
        x1d = torch.randn(HIDDEN_DIM)
        output1d = block(x1d)
        assert output1d.shape == (HIDDEN_DIM,)

        # Test 2D input
        x2d = torch.randn(4, HIDDEN_DIM)
        output2d = block(x2d)
        assert output2d.shape == (4, HIDDEN_DIM)

        # Test 3D input
        x3d = torch.randn(2, 4, HIDDEN_DIM)
        output3d = block(x3d)
        assert output3d.shape == (2, 4, HIDDEN_DIM)


class TestUnifiedWorldModel:
    """Test UnifiedWorldModel class"""

    @pytest.fixture
    def model(self):
        """Create world model instance - fresh instance with train() mode enabled"""
        model = UnifiedWorldModel(
            state_dim=EMBEDDING_DIM, ensemble_size=3, use_attention=True
        )
        model.train()  # Explicitly set to train mode to prevent gradient issues
        return model

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, model):
        """Test model initialization"""
        assert model.state_dim == EMBEDDING_DIM
        assert model.ensemble_size == 3
        assert model.use_attention == True
        assert len(model.dynamics_ensemble) == 3
        assert len(model.reward_ensemble) == 3
        assert model.device in [torch.device("cuda"), torch.device("cpu")]
        # Verify model is in train mode
        assert model.training is True

    def test_forward_single_model(self, model):
        """Test forward pass with single model"""
        state = torch.randn(1, EMBEDDING_DIM)
        action = torch.randn(1, EMBEDDING_DIM)

        next_state, reward, uncertainty = model(state, action, model_idx=0)

        assert next_state.shape == (1, EMBEDDING_DIM)
        assert reward.shape == (1, 1)
        assert uncertainty.shape == (1, 1)

    def test_forward_ensemble(self, model):
        """Test forward pass with ensemble"""
        state = torch.randn(4, EMBEDDING_DIM)
        action = torch.randn(4, EMBEDDING_DIM)

        next_state, reward, uncertainty = model(state, action)

        assert next_state.shape == (4, EMBEDDING_DIM)
        assert reward.shape == (4, 1)
        assert uncertainty.shape == (4, 1)

    def test_predict_value(self, model):
        """Test value prediction"""
        state = torch.randn(2, EMBEDDING_DIM)
        value = model.predict_value(state)

        assert value.shape == (2, 1)

    def test_predict_inverse_dynamics(self, model):
        """Test inverse dynamics prediction"""
        state = torch.randn(3, EMBEDDING_DIM)
        next_state = torch.randn(3, EMBEDDING_DIM)

        action = model.predict_inverse_dynamics(state, next_state)

        assert action.shape == (3, EMBEDDING_DIM)

    def test_compute_curiosity_reward(self, model):
        """Test curiosity reward computation"""
        state = torch.randn(2, EMBEDDING_DIM)
        action = torch.randn(2, EMBEDDING_DIM)
        next_state = torch.randn(2, EMBEDDING_DIM)

        curiosity = model.compute_curiosity_reward(state, action, next_state)

        assert curiosity.shape == (2,)

    def test_abstract_state(self, model):
        """Test state abstraction"""
        state = torch.randn(2, EMBEDDING_DIM)
        abstract = model.abstract_state(state)

        assert abstract.shape[0] == 2
        assert abstract.shape[1] <= EMBEDDING_DIM

    def test_update_state(self, model):
        """Test state update"""
        state = torch.randn(EMBEDDING_DIM)
        action = torch.randn(EMBEDDING_DIM)
        next_state = torch.randn(EMBEDDING_DIM)

        initial_history_size = len(model.state_history)
        initial_buffer_size = len(model.transition_buffer)

        model.update_state(state, action, 0.5, next_state)

        assert len(model.state_history) == initial_history_size + 1
        assert len(model.transition_buffer) == initial_buffer_size + 1
        assert model.training_stats["total_steps"] == 1

    def test_train_step(self, model):
        """Test training step"""
        batch = {
            "state": torch.randn(8, EMBEDDING_DIM),
            "action": torch.randn(8, EMBEDDING_DIM),
            "next_state": torch.randn(8, EMBEDDING_DIM),
            "reward": torch.randn(8),
        }

        losses = model.train_step(batch)

        assert "dynamics" in losses
        assert "reward" in losses
        assert "inverse" in losses
        assert "contrastive" in losses
        assert "value" in losses
        assert "curiosity" in losses

        # Check that losses are scalars
        for key, value in losses.items():
            assert isinstance(value, float)

    def test_contrastive_loss(self, model):
        """Test contrastive loss computation"""
        state = torch.randn(4, EMBEDDING_DIM)
        next_state = torch.randn(4, EMBEDDING_DIM)

        loss = model._contrastive_loss(state, next_state)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_imagine_rollout(self, model):
        """Test imagined rollout"""
        initial_state = torch.randn(1, EMBEDDING_DIM)
        action_sequence = [torch.randn(1, EMBEDDING_DIM) for _ in range(5)]

        rollout = model.imagine_rollout(initial_state, action_sequence, horizon=5)

        assert "states" in rollout
        assert "rewards" in rollout
        assert "uncertainties" in rollout
        assert "values" in rollout
        assert "curiosity_rewards" in rollout
        assert len(rollout["states"]) == 6  # initial + 5 steps
        assert len(rollout["rewards"]) == 5

    def test_plan_greedy(self, model):
        """Test greedy planning"""
        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM) for _ in range(5)]

        best_action, info = model.plan_actions(
            current_state,
            candidate_actions,
            horizon=3,
            algorithm=PlanningAlgorithm.GREEDY,
        )

        assert best_action.shape == (1, EMBEDDING_DIM)
        assert "cumulative_reward" in info

    def test_plan_beam_search(self, model):
        """Test beam search planning"""
        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM) for _ in range(5)]

        best_action, info = model.plan_actions(
            current_state,
            candidate_actions,
            horizon=3,
            algorithm=PlanningAlgorithm.BEAM_SEARCH,
            beam_width=3,
        )

        assert best_action.shape == (1, EMBEDDING_DIM)
        assert "cumulative_reward" in info
        assert "action_sequence" in info

    def test_plan_mcts(self, model):
        """Test MCTS planning"""
        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM) for _ in range(3)]

        best_action, info = model.plan_actions(
            current_state,
            candidate_actions,
            horizon=3,
            algorithm=PlanningAlgorithm.MCTS,
            num_simulations=10,
        )

        assert best_action.shape == (1, EMBEDDING_DIM)
        assert "visit_count" in info
        assert "value" in info

    def test_plan_cem(self, model):
        """Test CEM planning"""
        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM) for _ in range(3)]

        best_action, info = model.plan_actions(
            current_state,
            candidate_actions,
            horizon=3,
            algorithm=PlanningAlgorithm.CEM,
            population_size=10,
            num_iters=2,
        )

        assert best_action.shape == (1, EMBEDDING_DIM)
        assert "cumulative_reward" in info
        assert "action_sequence" in info

    def test_plan_mppi(self, model):
        """Test MPPI planning"""
        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM) for _ in range(3)]

        best_action, info = model.plan_actions(
            current_state,
            candidate_actions,
            horizon=3,
            algorithm=PlanningAlgorithm.MPPI,
            num_samples=10,
        )

        assert best_action.shape == (EMBEDDING_DIM,)
        assert "expected_cost" in info
        assert "temperature" in info

    def test_get_training_stats(self, model):
        """Test getting training statistics"""
        # Add some data
        model.update_state(
            torch.randn(EMBEDDING_DIM),
            torch.randn(EMBEDDING_DIM),
            0.5,
            torch.randn(EMBEDDING_DIM),
        )

        stats = model.get_training_stats()

        assert "total_steps" in stats
        assert "state_history_size" in stats
        assert "transition_buffer_size" in stats
        assert stats["total_steps"] == 1

    def test_save_load_model(self, model, temp_dir):
        """Test saving and loading model"""
        # Train for one step to have some state
        batch = {
            "state": torch.randn(4, EMBEDDING_DIM),
            "action": torch.randn(4, EMBEDDING_DIM),
            "next_state": torch.randn(4, EMBEDDING_DIM),
            "reward": torch.randn(4),
        }
        model.train_step(batch)

        # Save model
        save_path = Path(temp_dir) / "world_model.pt"
        model.save_model(str(save_path))

        assert save_path.exists()

        # Create new model and load
        new_model = UnifiedWorldModel(state_dim=EMBEDDING_DIM, ensemble_size=3)
        new_model.load_model(str(save_path))

        # Check that parameters are loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_shutdown(self, model):
        """Test model shutdown"""
        # Mock param_history to check shutdown is called
        with patch.object(model.param_history, "shutdown") as mock_shutdown:
            model.shutdown()
            mock_shutdown.assert_called_once()

    def test_thread_safety(self, model):
        """Test thread safety of operations"""
        import threading

        def update_task():
            for _ in range(5):
                model.update_state(
                    torch.randn(EMBEDDING_DIM),
                    torch.randn(EMBEDDING_DIM),
                    np.random.random(),
                    torch.randn(EMBEDDING_DIM),
                )

        threads = [threading.Thread(target=update_task) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have accumulated updates without errors
        assert model.training_stats["total_steps"] == 15


class TestCuriosityModule:
    """Test CuriosityModule"""

    def test_curiosity_forward(self):
        """Test curiosity module forward pass"""
        curiosity = CuriosityModule(state_dim=EMBEDDING_DIM)

        state = torch.randn(4, EMBEDDING_DIM)
        action = torch.randn(4, EMBEDDING_DIM)
        next_state = torch.randn(4, EMBEDDING_DIM)

        reward = curiosity(state, action, next_state)

        assert reward.shape == (4,)
        assert torch.all(reward >= 0)


class TestStateAbstractor:
    """Test StateAbstractor"""

    def test_state_abstraction(self):
        """Test state abstraction at different levels"""
        abstractor = StateAbstractor(state_dim=EMBEDDING_DIM, num_levels=3)

        state = torch.randn(2, EMBEDDING_DIM)

        # Test default (most abstract)
        abstract = abstractor(state)
        assert abstract.shape[0] == 2
        assert abstract.shape[1] <= EMBEDDING_DIM

        # Test specific levels
        for level in range(3):
            abstract = abstractor(state, level=level)
            assert abstract.shape[0] == 2

    def test_state_reconstruction(self):
        """Test state reconstruction from abstraction"""
        abstractor = StateAbstractor(state_dim=EMBEDDING_DIM, num_levels=3)

        state = torch.randn(2, EMBEDDING_DIM)
        abstract = abstractor(state, level=1)

        reconstructed = abstractor.reconstruct(abstract, level=1)

        assert reconstructed.shape == state.shape


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_rollout(self):
        """Test rollout with empty action sequence"""
        model = UnifiedWorldModel(state_dim=EMBEDDING_DIM, ensemble_size=2)

        initial_state = torch.randn(1, EMBEDDING_DIM)
        rollout = model.imagine_rollout(initial_state, [], horizon=5)

        assert len(rollout["states"]) == 1
        assert len(rollout["rewards"]) == 0

    def test_single_batch_contrastive_loss(self):
        """Test contrastive loss with batch size 1"""
        model = UnifiedWorldModel(state_dim=EMBEDDING_DIM, ensemble_size=2)

        state = torch.randn(1, EMBEDDING_DIM)
        next_state = torch.randn(1, EMBEDDING_DIM)

        loss = model._contrastive_loss(state, next_state)

        assert loss.item() == 0.0

    def test_planning_with_single_action(self):
        """Test planning with single candidate action"""
        model = UnifiedWorldModel(state_dim=EMBEDDING_DIM, ensemble_size=2)

        current_state = torch.randn(1, EMBEDDING_DIM)
        candidate_actions = [torch.randn(1, EMBEDDING_DIM)]

        for algorithm in PlanningAlgorithm:
            best_action, info = model.plan_actions(
                current_state, candidate_actions, horizon=2, algorithm=algorithm
            )

            assert best_action is not None
            assert info is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
