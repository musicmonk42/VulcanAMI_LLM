"""
Comprehensive test suite for auto_ml_nodes.py
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from auto_ml_nodes import (
    MAX_KERNEL_LENGTH,
    MAX_SPACE_DIMENSIONS,
    MAX_TENSOR_SIZE,
    OPTUNA_AVAILABLE,
    HyperParamNode,
    RandomNode,
    SearchNode,
    dispatch_auto_ml_node,
)


@pytest.fixture
def context():
    """Create execution context."""
    return {"audit_log": []}


class TestRandomNode:
    """Test RandomNode."""

    @pytest.mark.asyncio
    async def test_uniform_distribution(self, context):
        """Test uniform distribution."""
        node = RandomNode()

        params = {"distribution": "uniform", "range": [0.0, 1.0]}

        result = await node.execute(params, context)

        assert "value" in result
        assert 0.0 <= result["value"] <= 1.0
        assert result["distribution"] == "uniform"
        assert len(context["audit_log"]) == 1

    @pytest.mark.asyncio
    async def test_normal_distribution(self, context):
        """Test normal distribution."""
        node = RandomNode()

        params = {"distribution": "normal", "range": [0.0, 1.0]}  # mean, std

        result = await node.execute(params, context)

        assert "value" in result
        assert result["distribution"] == "normal"

    @pytest.mark.asyncio
    async def test_discrete_distribution(self, context):
        """Test discrete distribution."""
        node = RandomNode()

        params = {"distribution": "discrete", "range": [1, 2, 3, 4, 5]}

        result = await node.execute(params, context)

        assert "value" in result
        assert result["value"] in [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_invalid_distribution(self, context):
        """Test invalid distribution raises error."""
        node = RandomNode()

        params = {"distribution": "invalid", "range": [0.0, 1.0]}

        with pytest.raises(ValueError, match="Unsupported distribution"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_range(self, context):
        """Test invalid range raises error."""
        node = RandomNode()

        params = {"distribution": "uniform", "range": [0.0]}  # Too short

        with pytest.raises(ValueError, match="Range must be"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_with_tensor(self, context):
        """Test with tensor parameter."""
        node = RandomNode()

        params = {
            "distribution": "uniform",
            "range": [0.0, 1.0],
            "tensor": [[0.1, 0.2], [0.3, 0.4]],
        }

        result = await node.execute(params, context)

        assert "value" in result
        assert "compression_meta" in result

    @pytest.mark.asyncio
    async def test_tensor_too_large(self, context):
        """Test tensor size validation."""
        node = RandomNode()

        large_tensor = np.ones(MAX_TENSOR_SIZE + 1)

        params = {
            "distribution": "uniform",
            "range": [0.0, 1.0],
            "tensor": large_tensor,
        }

        with pytest.raises(ValueError, match="Tensor too large"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_ethical_label(self, context):
        """Test ethical label in result."""
        node = RandomNode()

        params = {
            "distribution": "uniform",
            "range": [0.0, 1.0],
            "ethical_label": "EU2025:Safe",
        }

        result = await node.execute(params, context)

        assert result["ethical_label"] == "EU2025:Safe"
        assert result["audit"]["ethical_label"] == "EU2025:Safe"

    @pytest.mark.asyncio
    async def test_normal_negative_std(self, context):
        """Test normal distribution with negative std."""
        node = RandomNode()

        params = {"distribution": "normal", "range": [0.0, -1.0]}

        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            await node.execute(params, context)


class TestHyperParamNode:
    """Test HyperParamNode."""

    @pytest.mark.asyncio
    async def test_valid_space(self, context):
        """Test valid search space."""
        node = HyperParamNode()

        params = {
            "space": {"learning_rate": [0.001, 0.1], "dropout": [0.1, 0.5]},
            "strategy": "grid",
        }

        result = await node.execute(params, context)

        assert result["dimensions"] == 2
        assert result["strategy"] == "grid"
        assert "search_space" in result
        assert len(context["audit_log"]) == 1

    @pytest.mark.asyncio
    async def test_empty_space(self, context):
        """Test empty search space raises error."""
        node = HyperParamNode()

        params = {"space": {}, "strategy": "grid"}

        with pytest.raises(ValueError, match="Invalid or empty search space"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_space_format(self, context):
        """Test invalid space format."""
        node = HyperParamNode()

        params = {
            "space": {"learning_rate": [0.001]},  # Only one value
            "strategy": "grid",
        }

        with pytest.raises(ValueError, match="Invalid range"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_range(self, context):
        """Test invalid range (min >= max)."""
        node = HyperParamNode()

        params = {
            "space": {"learning_rate": [0.1, 0.001]},  # min > max
            "strategy": "grid",
        }

        with pytest.raises(ValueError, match="min must be < max"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_too_many_dimensions(self, context):
        """Test too many dimensions."""
        node = HyperParamNode()

        space = {f"param{i}": [0, 1] for i in range(MAX_SPACE_DIMENSIONS + 1)}

        params = {"space": space, "strategy": "grid"}

        with pytest.raises(ValueError, match="Too many dimensions"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_strategy(self, context):
        """Test invalid strategy."""
        node = HyperParamNode()

        params = {"space": {"lr": [0.001, 0.1]}, "strategy": "invalid_strategy"}

        with pytest.raises(ValueError, match="Invalid strategy"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_context_storage(self, context):
        """Test search space stored in context."""
        node = HyperParamNode()

        space = {"learning_rate": [0.001, 0.1]}

        params = {"space": space, "strategy": "bayesian"}

        await node.execute(params, context)

        assert context["search_space"] == space
        assert context["search_strategy"] == "bayesian"


class TestSearchNode:
    """Test SearchNode."""

    @pytest.mark.asyncio
    async def test_bayesian_search(self, context):
        """Test Bayesian search."""
        node = SearchNode()

        params = {
            "algorithm": "bayesian",
            "objective": "accuracy",
            "space": {"learning_rate": [0.001, 0.1], "dropout": [0.1, 0.5]},
            "n_trials": 5,
        }

        result = await node.execute(params, context)

        assert "optimal_params" in result
        assert "objective_value" in result
        assert result["algorithm"] == "bayesian"
        assert result["n_trials"] == 5
        assert len(context["audit_log"]) == 1

    @pytest.mark.asyncio
    async def test_no_search_space(self, context):
        """Test search without space raises error."""
        node = SearchNode()

        params = {"algorithm": "bayesian", "objective": "accuracy", "n_trials": 5}

        with pytest.raises(ValueError, match="No search space provided"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_n_trials(self, context):
        """Test invalid n_trials."""
        node = SearchNode()

        params = {"space": {"lr": [0.001, 0.1]}, "n_trials": -1}

        with pytest.raises(ValueError, match="positive integer"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_n_trials_too_large(self, context):
        """Test n_trials too large."""
        node = SearchNode()

        params = {"space": {"lr": [0.001, 0.1]}, "n_trials": 2000}

        with pytest.raises(ValueError, match="too large"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_with_kernel(self, context):
        """Test with kernel parameter."""
        node = SearchNode()

        params = {
            "space": {"lr": [0.001, 0.1]},
            "n_trials": 5,
            "kernel": "def optimize(): return 1.0",
        }

        result = await node.execute(params, context)

        assert "kernel_audit" in result

    @pytest.mark.asyncio
    async def test_kernel_too_long(self, context):
        """Test kernel length validation."""
        node = SearchNode()

        params = {
            "space": {"lr": [0.001, 0.1]},
            "n_trials": 5,
            "kernel": "x" * (MAX_KERNEL_LENGTH + 1),
        }

        with pytest.raises(ValueError, match="Kernel too long"):
            await node.execute(params, context)

    @pytest.mark.asyncio
    async def test_invalid_space_type(self, context):
        """Test invalid space type."""
        node = SearchNode()

        params = {"space": "not a dict", "n_trials": 5}

        with pytest.raises(ValueError, match="must be a dictionary"):
            await node.execute(params, context)

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
    @pytest.mark.asyncio
    async def test_optuna_integration(self, context):
        """Test Optuna integration when available."""
        node = SearchNode()

        params = {"algorithm": "bayesian", "space": {"lr": [0.001, 0.1]}, "n_trials": 3}

        result = await node.execute(params, context)

        assert result["optimal_params"] is not None
        assert "lr" in result["optimal_params"]


class TestDispatchFunction:
    """Test dispatch_auto_ml_node function."""

    @pytest.mark.asyncio
    async def test_dispatch_random_node(self, context):
        """Test dispatching to RandomNode."""
        node = {
            "type": "RandomNode",
            "params": {"distribution": "uniform", "range": [0.0, 1.0]},
        }

        result = await dispatch_auto_ml_node(node, context)

        assert "value" in result

    @pytest.mark.asyncio
    async def test_dispatch_hyperparam_node(self, context):
        """Test dispatching to HyperParamNode."""
        node = {
            "type": "HyperParamNode",
            "params": {"space": {"lr": [0.001, 0.1]}, "strategy": "grid"},
        }

        result = await dispatch_auto_ml_node(node, context)

        assert "search_space" in result

    @pytest.mark.asyncio
    async def test_dispatch_search_node(self, context):
        """Test dispatching to SearchNode."""
        node = {
            "type": "SearchNode",
            "params": {"space": {"lr": [0.001, 0.1]}, "n_trials": 3},
        }

        result = await dispatch_auto_ml_node(node, context)

        assert "optimal_params" in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_node(self, context):
        """Test dispatching to unknown node type."""
        node = {"type": "UnknownNode", "params": {}}

        with pytest.raises(ValueError, match="Unknown AutoML node type"):
            await dispatch_auto_ml_node(node, context)

    @pytest.mark.asyncio
    async def test_dispatch_with_tensor(self, context):
        """Test dispatch with tensor in node."""
        node = {
            "type": "RandomNode",
            "params": {"distribution": "uniform", "range": [0, 1]},
            "tensor": [[0.1, 0.2]],
        }

        result = await dispatch_auto_ml_node(node, context)

        assert "compression_meta" in result

    @pytest.mark.asyncio
    async def test_context_initialization(self):
        """Test context audit log initialization."""
        context = {}

        node = {
            "type": "RandomNode",
            "params": {"distribution": "uniform", "range": [0, 1]},
        }

        await dispatch_auto_ml_node(node, context)

        assert "audit_log" in context
        assert len(context["audit_log"]) > 0


class TestAuditLogging:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_created(self, context):
        """Test audit log entry created."""
        node = RandomNode()

        params = {"distribution": "uniform", "range": [0.0, 1.0]}

        await node.execute(params, context)

        assert len(context["audit_log"]) == 1
        assert context["audit_log"][0]["node_type"] == "RandomNode"
        assert context["audit_log"][0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_audit_on_error(self, context):
        """Test audit log on error."""
        node = RandomNode()

        params = {"distribution": "invalid", "range": [0.0, 1.0]}

        try:
            await node.execute(params, context)
        except:
            pass

        assert len(context["audit_log"]) == 1
        assert context["audit_log"][0]["status"] == "error"
        assert "error" in context["audit_log"][0]

    @pytest.mark.asyncio
    async def test_multiple_nodes_audit(self, context):
        """Test audit log accumulation."""
        node1 = {
            "type": "RandomNode",
            "params": {"distribution": "uniform", "range": [0, 1]},
        }

        node2 = {
            "type": "HyperParamNode",
            "params": {"space": {"lr": [0.001, 0.1]}, "strategy": "grid"},
        }

        await dispatch_auto_ml_node(node1, context)
        await dispatch_auto_ml_node(node2, context)

        assert len(context["audit_log"]) == 2
        assert context["audit_log"][0]["node_type"] == "RandomNode"
        assert context["audit_log"][1]["node_type"] == "HyperParamNode"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
