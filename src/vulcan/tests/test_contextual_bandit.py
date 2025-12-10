"""
Comprehensive Test Suite for Contextual Bandit Learning

Tests all bandit algorithms, exploration strategies, off-policy evaluation,
and numerical stability fixes.
"""

from vulcan.reasoning.contextual_bandit import (AdaptiveBanditOrchestrator,
                                                BanditAction, BanditContext,
                                                BanditFeedback,
                                                ContextualBandit,
                                                ExplorationStrategy,
                                                LinUCBBandit,
                                                NeuralContextualBandit,
                                                OffPolicyEvaluator,
                                                ToolSelectionBandit)
import numpy as np
import tempfile
import shutil
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip(
    "torch", reason="PyTorch required for contextual_bandit tests"
)


# Fixtures
@pytest.fixture
def simple_context():
    """Create a simple bandit context"""
    return BanditContext(
        features=np.array([0.5, 0.3, 0.2, 0.1]),
        problem_type="test",
        constraints={
            "time_budget": 1000.0,
            "energy_budget": 1000.0,
            "min_quality": 0.7,
        },
    )


@pytest.fixture
def bandit_5_actions():
    """Create a basic contextual bandit with 5 actions"""
    return ContextualBandit(
        n_actions=5,
        context_dim=4,
        exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        learning_rate=0.1,
        exploration_rate=0.2,
    )


@pytest.fixture
def linucb_bandit():
    """Create a LinUCB bandit"""
    return LinUCBBandit(n_actions=5, context_dim=4, alpha=1.0)


@pytest.fixture
def neural_bandit():
    """Create a neural contextual bandit"""
    return NeuralContextualBandit(context_dim=4, n_actions=5, hidden_dim=64)


@pytest.fixture
def orchestrator():
    """Create an adaptive bandit orchestrator"""
    return AdaptiveBanditOrchestrator(n_actions=5, context_dim=4)


@pytest.fixture
def tool_selection_bandit():
    """Create a tool selection bandit"""
    return ToolSelectionBandit()


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model saving/loading"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Basic Functionality Tests
class TestBanditContext:
    """Test BanditContext dataclass"""

    def test_context_creation(self):
        context = BanditContext(
            features=np.array([1, 2, 3]),
            problem_type="test",
            constraints={"budget": 100},
        )
        assert context.features.shape == (3,)
        assert context.problem_type == "test"
        assert context.constraints["budget"] == 100

    def test_context_with_metadata(self):
        context = BanditContext(
            features=np.zeros(5),
            problem_type="complex",
            constraints={},
            metadata={"source": "test", "version": 1},
        )
        assert context.metadata["source"] == "test"
        assert context.metadata["version"] == 1


class TestBanditAction:
    """Test BanditAction dataclass"""

    def test_action_creation(self):
        action = BanditAction(
            tool_name="symbolic",
            action_id=0,
            expected_reward=0.75,
            exploration_bonus=0.1,
            probability=0.8,
        )
        assert action.tool_name == "symbolic"
        assert action.action_id == 0
        assert action.expected_reward == 0.75
        assert action.exploration_bonus == 0.1
        assert action.probability == 0.8


class TestBanditFeedback:
    """Test BanditFeedback dataclass"""

    def test_feedback_creation(self, simple_context):
        action = BanditAction("symbolic", 0, 0.5)
        feedback = BanditFeedback(
            context=simple_context,
            action=action,
            reward=0.8,
            execution_time=50.0,
            energy_used=30.0,
            success=True,
        )
        assert feedback.reward == 0.8
        assert feedback.success is True
        assert feedback.execution_time == 50.0


# Contextual Bandit Tests
class TestContextualBandit:
    """Test basic contextual bandit"""

    def test_initialization(self):
        bandit = ContextualBandit(
            n_actions=3, context_dim=5, exploration_strategy=ExplorationStrategy.UCB
        )
        assert bandit.n_actions == 3
        assert bandit.context_dim == 5
        assert bandit.exploration_strategy == ExplorationStrategy.UCB
        assert bandit.total_count == 0

    def test_epsilon_greedy_selection(self, bandit_5_actions, simple_context):
        action = bandit_5_actions.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert 0 <= action.action_id < 5
        assert action.tool_name in [
            "symbolic",
            "probabilistic",
            "causal",
            "analogical",
            "multimodal",
        ]
        assert 0 <= action.probability <= 1

    def test_ucb_selection(self, simple_context):
        bandit = ContextualBandit(
            n_actions=5, context_dim=4, exploration_strategy=ExplorationStrategy.UCB
        )
        action = bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert 0 <= action.action_id < 5

    def test_thompson_sampling_selection(self, simple_context):
        bandit = ContextualBandit(
            n_actions=5,
            context_dim=4,
            exploration_strategy=ExplorationStrategy.THOMPSON_SAMPLING,
        )
        action = bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert 0 <= action.action_id < 5

    def test_update_q_values(self, bandit_5_actions, simple_context):
        action = bandit_5_actions.select_action(simple_context)

        feedback = BanditFeedback(
            context=simple_context,
            action=action,
            reward=0.9,
            execution_time=100.0,
            energy_used=50.0,
            success=True,
        )

        initial_count = bandit_5_actions.total_count
        bandit_5_actions.update(feedback)

        assert bandit_5_actions.total_count == initial_count + 1
        assert bandit_5_actions.action_counts[action.action_id] > 0
        assert bandit_5_actions.stats["total_actions"] > 0

    def test_exploration_exploitation_balance(self, bandit_5_actions, simple_context):
        # Run many iterations
        for _ in range(100):
            action = bandit_5_actions.select_action(simple_context)
            reward = 1.0 if action.action_id == 0 else 0.0
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=reward,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit_5_actions.update(feedback)

        # Check that we have both exploration and exploitation
        assert bandit_5_actions.stats["exploration_actions"] > 0
        assert bandit_5_actions.stats["exploitation_actions"] > 0

    def test_history_tracking(self, bandit_5_actions, simple_context):
        for i in range(5):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=float(i) / 10,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit_5_actions.update(feedback)

        assert len(bandit_5_actions.history) == 5


# Numerical Stability Tests
class TestNumericalStability:
    """Test critical numerical stability fixes"""

    def test_ucb_zero_count_handling(self, simple_context):
        bandit = ContextualBandit(
            n_actions=5, context_dim=4, exploration_strategy=ExplorationStrategy.UCB
        )
        # Should not crash with zero total count
        action = bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)

    def test_ucb_overflow_prevention(self, simple_context):
        bandit = ContextualBandit(
            n_actions=3, context_dim=4, exploration_strategy=ExplorationStrategy.UCB
        )

        # Update one action many times
        for _ in range(1000):
            action = BanditAction("test", 0, 0.5, probability=1.0)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.8,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit.update(feedback)

        # Should not overflow when computing UCB for rarely-selected actions
        action = bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert np.isfinite(action.exploration_bonus)

    def test_thompson_sampling_beta_parameters(self, simple_context):
        bandit = ContextualBandit(
            n_actions=5,
            context_dim=4,
            exploration_strategy=ExplorationStrategy.THOMPSON_SAMPLING,
        )

        # Set extreme q-values
        context_key = bandit._get_context_key(simple_context)
        bandit.q_values[context_key] = np.array([0.0, 1.0, 0.5, 0.001, 0.999])
        bandit.action_counts = np.array([1, 1, 1, 1, 1])

        # Should handle extreme values without error
        action = bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)

    def test_extreme_reward_clamping(self, bandit_5_actions, simple_context):
        action = bandit_5_actions.select_action(simple_context)

        # Test extreme rewards
        extreme_rewards = [-1e10, 1e10, float("inf"), -float("inf")]

        for reward in extreme_rewards:
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=reward,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit_5_actions.update(feedback)

            # Q-values should remain finite
            context_key = bandit_5_actions._get_context_key(simple_context)
            assert np.all(np.isfinite(bandit_5_actions.q_values[context_key]))

    def test_division_by_zero_protection(self, bandit_5_actions):
        # Stats should handle zero actions gracefully
        assert bandit_5_actions.stats["average_reward"] == 0.0


# LinUCB Tests
class TestLinUCBBandit:
    """Test LinUCB bandit algorithm"""

    def test_initialization(self):
        bandit = LinUCBBandit(n_actions=3, context_dim=5, alpha=2.0)
        assert bandit.n_actions == 3
        assert bandit.context_dim == 5
        assert bandit.alpha == 2.0
        assert len(bandit.A) == 3
        assert len(bandit.b) == 3

    def test_action_selection(self, linucb_bandit, simple_context):
        action = linucb_bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert 0 <= action.action_id < 5

    def test_parameter_update(self, linucb_bandit, simple_context):
        action = linucb_bandit.select_action(simple_context)

        A_before = linucb_bandit.A[action.action_id].copy()
        b_before = linucb_bandit.b[action.action_id].copy()

        feedback = BanditFeedback(
            context=simple_context,
            action=action,
            reward=0.85,
            execution_time=50.0,
            energy_used=30.0,
            success=True,
        )
        linucb_bandit.update(feedback)

        # Parameters should have changed
        assert not np.allclose(linucb_bandit.A[action.action_id], A_before)
        assert not np.allclose(linucb_bandit.b[action.action_id], b_before)

    def test_matrix_inversion_stability(self, simple_context):
        bandit = LinUCBBandit(n_actions=3, context_dim=4, alpha=1.0)

        # Make matrix nearly singular by repeated identical updates
        action = BanditAction("test", 0, 0.5, probability=1.0)
        for _ in range(10):
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.5,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit.update(feedback)

        # Should still be able to select action without crash
        new_action = bandit.select_action(simple_context)
        assert isinstance(new_action, BanditAction)

    def test_ridge_regularization(self, linucb_bandit):
        # Check that ridge parameter is being used
        assert linucb_bandit.ridge_param > 0


# Neural Bandit Tests
class TestNeuralContextualBandit:
    """Test neural network-based bandit"""

    def test_initialization(self):
        bandit = NeuralContextualBandit(context_dim=10, n_actions=5, hidden_dim=64)
        assert bandit.context_dim == 10
        assert bandit.n_actions == 5
        assert len(list(bandit.parameters())) > 0

    def test_forward_pass(self, neural_bandit):
        x = torch.randn(2, 4)  # Batch of 2
        means, stds = neural_bandit.forward(x)

        assert means.shape == (2, 5)
        assert stds.shape == (2, 5)
        assert torch.all(stds > 0)  # Standard deviations must be positive

    def test_action_selection(self, neural_bandit, simple_context):
        action = neural_bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert 0 <= action.action_id < 5

    def test_learning(self, neural_bandit, simple_context):
        # Get initial parameters
        initial_params = [p.clone() for p in neural_bandit.parameters()]

        # Perform updates
        for _ in range(10):
            action = neural_bandit.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.9,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            neural_bandit.update(feedback)

        # Parameters should have changed
        final_params = list(neural_bandit.parameters())
        assert any(
            not torch.allclose(p1, p2) for p1, p2 in zip(initial_params, final_params)
        )

    def test_nan_handling(self, neural_bandit, simple_context):
        # Force NaN in forward pass by setting extreme values
        with torch.no_grad():
            for param in neural_bandit.parameters():
                param.fill_(1e10)

        # Should handle gracefully
        action = neural_bandit.select_action(simple_context)
        assert isinstance(action, BanditAction)

    def test_gradient_clipping(self, neural_bandit, simple_context):
        action = neural_bandit.select_action(simple_context)

        # Extreme reward to test gradient clipping
        feedback = BanditFeedback(
            context=simple_context,
            action=action,
            reward=1e6,
            execution_time=10.0,
            energy_used=5.0,
            success=True,
        )
        neural_bandit.update(feedback)

        # Parameters should remain finite
        for param in neural_bandit.parameters():
            assert torch.all(torch.isfinite(param))


# Off-Policy Evaluation Tests
class TestOffPolicyEvaluator:
    """Test off-policy evaluation methods"""

    def test_initialization(self):
        evaluator = OffPolicyEvaluator(context_dim=4, n_actions=5)
        assert "ips" in evaluator.evaluation_methods
        assert "dr" in evaluator.evaluation_methods
        assert "dm" in evaluator.evaluation_methods
        assert "switch" in evaluator.evaluation_methods

    def test_ips_evaluation(self, bandit_5_actions, simple_context):
        # Generate history
        history = []
        for _ in range(50):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.7,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            history.append(feedback)

        # Define new policy
        def new_policy(context):
            return BanditAction("symbolic", 0, 0.8, probability=1.0)

        evaluator = OffPolicyEvaluator(context_dim=4, n_actions=5)
        result = evaluator.evaluate(history, new_policy, method="ips")

        assert "estimated_reward" in result
        assert "effective_sample_size" in result
        assert isinstance(result["estimated_reward"], float)

    def test_doubly_robust_evaluation(self, bandit_5_actions, simple_context):
        history = []
        for _ in range(50):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.7 + 0.1 * action.action_id,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            history.append(feedback)

        def new_policy(context):
            return BanditAction("probabilistic", 1, 0.75, probability=1.0)

        evaluator = OffPolicyEvaluator(context_dim=4, n_actions=5)
        result = evaluator.evaluate(history, new_policy, method="dr")

        assert "estimated_reward" in result
        assert "direct_component" in result
        assert "ips_correction" in result

    def test_direct_method_evaluation(self, bandit_5_actions, simple_context):
        history = []
        for _ in range(30):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.6,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            history.append(feedback)

        def new_policy(context):
            return BanditAction("causal", 2, 0.65, probability=1.0)

        evaluator = OffPolicyEvaluator(context_dim=4, n_actions=5)
        result = evaluator.evaluate(history, new_policy, method="dm")

        assert "estimated_reward" in result
        assert isinstance(result["estimated_reward"], float)

    def test_switch_estimator(self, bandit_5_actions, simple_context):
        history = []
        for _ in range(40):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.65,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            history.append(feedback)

        def new_policy(context):
            return BanditAction("analogical", 3, 0.7, probability=1.0)

        evaluator = OffPolicyEvaluator(context_dim=4, n_actions=5)
        result = evaluator.evaluate(history, new_policy, method="switch")

        assert "estimated_reward" in result
        assert "ips_weight" in result
        assert "ips_component" in result
        assert "dm_component" in result

    def test_importance_weight_capping(self):
        evaluator = OffPolicyEvaluator(context_dim=2, n_actions=3)

        # Create history with very small probabilities
        context = BanditContext(
            features=np.array([0.5, 0.3]), problem_type="test", constraints={}
        )

        history = [
            BanditFeedback(
                context=context,
                action=BanditAction("test", 0, 0.5, probability=0.001),
                reward=1.0,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
        ]

        def new_policy(ctx):
            return BanditAction("test", 0, 0.5, probability=1.0)

        # Should not produce extreme importance weights
        result = evaluator.evaluate(history, new_policy, method="ips")
        assert np.isfinite(result["estimated_reward"])


# Adaptive Orchestrator Tests
class TestAdaptiveBanditOrchestrator:
    """Test adaptive bandit orchestrator"""

    def test_initialization(self):
        orch = AdaptiveBanditOrchestrator(n_actions=5, context_dim=4)
        assert len(orch.bandits) == 5
        assert "epsilon_greedy" in orch.bandits
        assert "linucb" in orch.bandits
        assert "neural" in orch.bandits

    def test_action_selection(self, orchestrator, simple_context):
        action = orchestrator.select_action(simple_context)
        assert isinstance(action, BanditAction)
        assert hasattr(action, "metadata")

    def test_meta_learning(self, orchestrator, simple_context):
        # Run several iterations
        for i in range(20):
            action = orchestrator.select_action(simple_context)

            # Give higher rewards to specific bandit
            reward = (
                0.9
                if action.metadata.get("bandit_algorithm") == "epsilon_greedy"
                else 0.3
            )

            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=reward,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            orchestrator.update(feedback)

        # Should eventually prefer the better-performing bandit
        assert len(orchestrator.performance_window) > 0

    def test_policy_evaluation(self, orchestrator, simple_context):
        # Generate some history
        for _ in range(30):
            action = orchestrator.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.7,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            orchestrator.update(feedback)

        # Test that orchestrator has evaluator
        assert hasattr(orchestrator, "evaluator")
        assert isinstance(orchestrator.evaluator, OffPolicyEvaluator)

    def test_statistics(self, orchestrator, simple_context):
        for _ in range(10):
            action = orchestrator.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.75,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            orchestrator.update(feedback)

        stats = orchestrator.get_statistics()
        assert "active_bandit" in stats
        assert "bandit_performance" in stats

    def test_save_and_load(self, orchestrator, simple_context, temp_model_dir):
        # Train a bit
        for _ in range(10):
            action = orchestrator.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.8,
                execution_time=50.0,
                energy_used=30.0,
                success=True,
            )
            orchestrator.update(feedback)

        # Save
        orchestrator.save_model(temp_model_dir)

        # Create new orchestrator and load
        new_orch = AdaptiveBanditOrchestrator(n_actions=5, context_dim=4)
        new_orch.load_model(temp_model_dir)

        # Should have loaded state
        assert new_orch.active_bandit == orchestrator.active_bandit


# Tool Selection Tests
class TestToolSelectionBandit:
    """Test tool selection bandit"""

    def test_initialization(self):
        tsb = ToolSelectionBandit()
        assert tsb.n_actions == 5
        assert len(tsb.tool_names) == 5
        assert len(tsb.tool_costs) == 5

    def test_tool_selection(self, tool_selection_bandit):
        features = np.random.randn(128)
        constraints = {
            "time_budget": 1000.0,
            "energy_budget": 1000.0,
            "min_quality": 0.7,
        }

        tool = tool_selection_bandit.select_tool(features, constraints)
        assert tool in tool_selection_bandit.tool_names

    def test_execution_update(self, tool_selection_bandit):
        features = np.random.randn(128)
        constraints = {
            "time_budget": 1000.0,
            "energy_budget": 1000.0,
            "min_quality": 0.7,
        }

        tool_selection_bandit.update_from_execution(
            features=features,
            tool_name="symbolic",
            quality=0.85,
            time_ms=50.0,
            energy_mj=30.0,
            constraints=constraints,
        )

        # Should have updated statistics
        stats = tool_selection_bandit.get_statistics()
        assert "bandit_performance" in stats

    def test_reward_computation(self, tool_selection_bandit):
        constraints = {
            "time_budget": 1000.0,
            "energy_budget": 1000.0,
            "quality_weight": 2.0,
            "time_weight": 1.0,
            "energy_weight": 1.0,
        }

        # Perfect performance
        reward1 = tool_selection_bandit._compute_reward(1.0, 0.0, 0.0, constraints)
        assert reward1 == 1.0

        # Poor performance
        reward2 = tool_selection_bandit._compute_reward(
            0.0, 2000.0, 2000.0, constraints
        )
        assert reward2 < 0.5

    def test_constraint_violation_handling(self, tool_selection_bandit):
        features = np.random.randn(128)
        constraints = {"time_budget": 100.0, "energy_budget": 50.0, "min_quality": 0.9}

        # Tool that violates constraints
        tool_selection_bandit.update_from_execution(
            features=features,
            tool_name="multimodal",
            quality=0.5,  # Below min_quality
            time_ms=200.0,  # Over time_budget
            energy_mj=100.0,  # Over energy_budget
            constraints=constraints,
        )

        # Should still work
        stats = tool_selection_bandit.get_statistics()
        assert stats is not None


# Edge Case Tests
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_features(self):
        context = BanditContext(
            features=np.array([]), problem_type="test", constraints={}
        )
        bandit = ContextualBandit(n_actions=3, context_dim=0)

        # Should handle gracefully
        action = bandit.select_action(context)
        assert isinstance(action, BanditAction)

    def test_nan_features(self):
        context = BanditContext(
            features=np.array([np.nan, np.inf, -np.inf, 0.5]),
            problem_type="test",
            constraints={},
        )
        bandit = ContextualBandit(n_actions=3, context_dim=4)

        # Should handle gracefully
        action = bandit.select_action(context)
        assert isinstance(action, BanditAction)

    def test_single_action(self):
        bandit = ContextualBandit(n_actions=1, context_dim=4)
        context = BanditContext(
            features=np.array([0.5, 0.5, 0.5, 0.5]), problem_type="test", constraints={}
        )

        action = bandit.select_action(context)
        assert action.action_id == 0

    def test_very_high_dimensionality(self):
        bandit = ContextualBandit(n_actions=5, context_dim=1000)
        context = BanditContext(
            features=np.random.randn(1000), problem_type="test", constraints={}
        )

        action = bandit.select_action(context)
        assert isinstance(action, BanditAction)

    def test_thread_safety(self, bandit_5_actions, simple_context):
        import threading

        results = []

        def select_and_update():
            for _ in range(10):
                action = bandit_5_actions.select_action(simple_context)
                feedback = BanditFeedback(
                    context=simple_context,
                    action=action,
                    reward=0.7,
                    execution_time=10.0,
                    energy_used=5.0,
                    success=True,
                )
                bandit_5_actions.update(feedback)
                results.append(action.action_id)

        threads = [threading.Thread(target=select_and_update) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(results) == 30


# Integration Tests
class TestIntegration:
    """Integration tests across components"""

    def test_full_learning_cycle(self):
        # Create orchestrator
        orch = AdaptiveBanditOrchestrator(n_actions=5, context_dim=4)

        # Simulate learning over time with different contexts
        for episode in range(50):
            # Generate context
            context = BanditContext(
                features=np.random.randn(4),
                problem_type="integration_test",
                constraints={
                    "time_budget": 1000.0,
                    "energy_budget": 1000.0,
                    "min_quality": 0.7,
                },
            )

            # Select action
            action = orch.select_action(context)

            # Simulate execution with reward based on action
            reward = 0.9 if action.action_id in [0, 2] else 0.4

            feedback = BanditFeedback(
                context=context,
                action=action,
                reward=reward,
                execution_time=50.0,
                energy_used=30.0,
                success=reward > 0.7,
            )

            # Update
            orch.update(feedback)

        # Verify learning occurred
        stats = orch.get_statistics()
        assert stats["bandit_performance"]

    def test_tool_selection_workflow(self):
        tsb = ToolSelectionBandit()

        # Simulate realistic tool selection scenario
        contexts = [
            {"features": np.random.randn(128), "type": "symbolic"},
            {"features": np.random.randn(128), "type": "probabilistic"},
            {"features": np.random.randn(128), "type": "causal"},
        ]

        for ctx_data in contexts * 10:
            constraints = {
                "time_budget": 1000.0,
                "energy_budget": 1000.0,
                "min_quality": 0.7,
            }

            # Select tool
            tool = tsb.select_tool(ctx_data["features"], constraints)

            # Simulate execution
            quality = 0.8 if tool == ctx_data["type"] else 0.6
            time_ms = tsb.tool_costs[tool]["time"]
            energy_mj = tsb.tool_costs[tool]["energy"]

            # Update
            tsb.update_from_execution(
                ctx_data["features"], tool, quality, time_ms, energy_mj, constraints
            )

        # Should have learned something
        stats = tsb.get_statistics()
        assert stats["bandit_performance"]


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    def test_selection_speed(self, bandit_5_actions, simple_context):
        import time

        start = time.time()
        for _ in range(1000):
            bandit_5_actions.select_action(simple_context)
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 5.0  # 1000 selections in under 5 seconds

    def test_update_speed(self, bandit_5_actions, simple_context):
        import time

        feedbacks = []
        for _ in range(1000):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.7,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            feedbacks.append(feedback)

        start = time.time()
        for feedback in feedbacks:
            bandit_5_actions.update(feedback)
        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 5.0

    def test_memory_efficiency(self, bandit_5_actions, simple_context):
        # History should be bounded
        for _ in range(20000):
            action = bandit_5_actions.select_action(simple_context)
            feedback = BanditFeedback(
                context=simple_context,
                action=action,
                reward=0.7,
                execution_time=10.0,
                energy_used=5.0,
                success=True,
            )
            bandit_5_actions.update(feedback)

        # History should be capped at maxlen
        assert len(bandit_5_actions.history) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
