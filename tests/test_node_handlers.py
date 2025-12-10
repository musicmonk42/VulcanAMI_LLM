"""
Comprehensive pytest suite for node_handlers.py
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional  # <<< --- FIX: Added imports
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

# Import the module to test
try:
    from unified_runtime import node_handlers as nh
    from unified_runtime.node_handlers import AI_ERRORS, NodeContext
except ImportError:
    # Handle path issues if running directly
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'unified_runtime')))
    import node_handlers as nh
    from node_handlers import AI_ERRORS, NodeContext


# Helper function to create a mock context for tests
def create_mock_context(runtime_mock: Any = "DEFAULT", node_map_mock: Optional[Dict] = None, outputs_mock: Optional[Dict] = None, recursion_depth: int = 0):
    """
    Creates a mock context dictionary for testing.
    This returns a dictionary, simulating ExecutionContext.to_dict()

    By default, 'runtime' is a Mock() object unless runtime_mock=None is explicitly passed.
    """
    # Default to a new Mock() if no runtime is specified,
    # but allow 'None' to be passed in explicitly for tests that need it.
    runtime_to_use = Mock() if runtime_mock == "DEFAULT" else runtime_mock

    mock_context_dict = {
        'execution_id': 'test_exec_id',
        'recursion_depth': recursion_depth,
        'metadata': {},
        'runtime': runtime_to_use, # Use the determined runtime
        'graph': {}, # Default graph
        'node_map': node_map_mock if node_map_mock is not None else {},
        'audit_log': [], # Ensure audit_log is a list
        'inputs': {} # Default graph-level inputs
    }
    return mock_context_dict


class TestNodeExecutorError:
    """Test NodeExecutorError exception"""

    def test_exception_creation(self):
        """Test creating exception"""
        error = nh.NodeExecutorError("Test error")
        assert str(error) == "Test error"


class TestNodeContext:
    """Test NodeContext dataclass"""

    def test_context_creation(self):
        """Test creating node context"""
        # Test with the actual dataclass
        context = nh.NodeContext(
            runtime=None,
            graph={},
            node_map={},
            outputs={}
        )

        assert context.recursion_depth == 0
        assert context.audit_log is not None
        assert isinstance(context.audit_log, list)


class TestCoreNodeHandlers:
    """Test core node handlers"""

    @pytest.mark.asyncio
    async def test_const_node(self):
        """Test CONST node"""
        node = {"params": {"value": 42}}
        # Use mock context
        result = await nh.const_node(node, create_mock_context(), {})

        assert result["value"] == 42

    @pytest.mark.asyncio
    async def test_const_node_missing_value(self):
        """Test CONST node with missing value"""
        node = {"params": {}}
        result = await nh.const_node(node, create_mock_context(), {})

        assert "error_code" in result
        assert result["error_code"] == nh.AI_ERRORS.AI_INVALID_REQUEST.value

    @pytest.mark.asyncio
    async def test_add_node(self):
        """Test ADD node"""
        node = {}
        inputs = {"val1": 5, "val2": 3}
        result = await nh.add_node(node, create_mock_context(), inputs)

        # Fix: Check for 'result' key
        assert result["result"] == 8

    @pytest.mark.asyncio
    async def test_add_node_missing_inputs(self):
        """Test ADD node with missing inputs"""
        node = {}
        inputs = {"val1": 5} # val2 is missing

        # Fix: Assert that the correct error is raised
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.add_node(node, create_mock_context(), inputs)
        assert "Missing inputs" in str(e.value)

    @pytest.mark.asyncio
    async def test_add_node_nested_input(self):
        """Test ADD node with nested input structure"""
        node = {}
        # The add_node handler *only* looks for 'val1' and 'val2' at the top level.
        # This input structure *should* fail.
        inputs = {"input": {"val1": 10, "val2": 20}}

        # Fix: Assert that the correct error is raised
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.add_node(node, create_mock_context(), inputs)
        assert "Missing inputs" in str(e.value)

    @pytest.mark.asyncio
    async def test_multiply_node(self):
        """Test MULTIPLY node"""
        node = {}
        inputs = {"val1": 4, "val2": 5}
        result = await nh.multiply_node(node, create_mock_context(), inputs)

        # Fix: Check for 'result' key
        assert result["result"] == 20

    @pytest.mark.asyncio
    async def test_multiply_node_missing_inputs(self):
        """Test MULTIPLY node with missing inputs"""
        node = {}
        inputs = {} # Both val1 and val2 are missing

        # Fix: Assert that the correct error is raised
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.multiply_node(node, create_mock_context(), inputs)
        assert "Missing inputs" in str(e.value)

    @pytest.mark.asyncio
    async def test_branch_node_true(self):
        """Test BRANCH node with true condition"""
        node = {}
        inputs = {"condition": True, "value": 42}
        result = await nh.branch_node(node, create_mock_context(), inputs)

        assert result["on_true"] == 42
        assert result["on_false"] is None

    @pytest.mark.asyncio
    async def test_branch_node_false(self):
        """Test BRANCH node with false condition"""
        node = {}
        inputs = {"condition": False, "value": 42}
        result = await nh.branch_node(node, create_mock_context(), inputs)

        assert result["on_true"] is None
        assert result["on_false"] == 42

    @pytest.mark.asyncio
    async def test_get_property_node(self):
        """Test GET_PROPERTY node"""
        node = {
            "params": {
                "target_node": "n1",
                "property_path": "data.value"
            }
        }
        # Fix: Create a mock context dict with the required 'node_map'
        mock_map = {
            "n1": {
                "params": {
                    "data": {"value": 123}
                }
            }
        }
        context = create_mock_context(node_map_mock=mock_map)

        result = await nh.get_property_node(node, context, {})

        assert result["value"] == 123

    @pytest.mark.asyncio
    async def test_input_node_handler(self):
        """Test InputNode handler"""
        node = {"params": {"value": "input_data"}}
        result = await nh.input_node_handler(node, create_mock_context(), {})

        assert result["output"] == "input_data"

    @pytest.mark.asyncio
    async def test_output_node_handler(self):
        """Test OutputNode handler"""
        node = {}
        inputs = {"input": "output_data"}
        result = await nh.output_node_handler(node, create_mock_context(), inputs)

        assert result["result"] == "output_data"


class TestAINodeHandlers:
    """Test AI and embedding node handlers"""

    @pytest.mark.asyncio
    async def test_embed_node_basic(self):
        """Test EMBED node basic functionality"""
        node = {
            "params": {
                "provider": "test",
                "model": "test-model"
            }
        }
        inputs = {"text": "test text"}

        # Fix: Create a mock runtime and context dict
        mock_ai_runtime = Mock()
        # Mock the sync execute_task method
        mock_ai_runtime.execute_task = Mock(
            return_value=Mock(
                status="SUCCESS",
                is_success=lambda: True, # Add is_success method
                data={"vector": [0.1, 0.2]},
                metadata={}
            )
        )
        mock_runtime = Mock(ai_runtime=mock_ai_runtime)
        context = create_mock_context(runtime_mock=mock_runtime)

        result = await nh.embed_node(node, context, inputs)

        assert "vector" in result
        assert result["model"] == "test-model"
        assert result["vector"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_node_missing_text(self):
        """Test EMBED node without text"""
        node = {"params": {}}
        inputs = {}

        result = await nh.embed_node(node, create_mock_context(), inputs)

        assert "error_code" in result
        assert result["error_code"] == nh.AI_ERRORS.AI_INVALID_REQUEST.value

    @pytest.mark.asyncio
    async def test_generative_node_handler(self):
        """Test GenerativeNode"""
        node = {
            "params": {
                "prompt": "Test prompt",
                "provider": "test",
                "temperature": 0.7
            }
        }

        # Fix: Create mock context dict (default helper provides a Mock runtime)
        context = create_mock_context()
        result = await nh.generative_node_handler(node, context, {})

        assert "text" in result
        assert "tokens" in result

    @pytest.mark.asyncio
    async def test_generative_node_missing_prompt(self):
        """Test GenerativeNode without prompt"""
        node = {"params": {}}
        inputs = {}

        result = await nh.generative_node_handler(node, create_mock_context(), inputs)

        assert "error_code" in result


class TestHardwareNodeHandlers:
    """Test hardware-accelerated node handlers"""

    @pytest.mark.asyncio
    async def test_load_tensor_node_missing_torch(self):
        """Test LOAD_TENSOR without PyTorch"""
        with patch('node_handlers.TORCH_AVAILABLE', False):
            node = {"params": {"filepath": "test.safetensors", "key": "tensor"}}
            result = await nh.load_tensor_node(node, create_mock_context(), {})

            assert "error_code" in result

    @pytest.mark.asyncio
    async def test_memristor_mvm_node(self):
        """Test MEMRISTOR_MVM node"""
        node = {}
        inputs = {
            "tensor1": np.array([[1, 2], [3, 4]]),
            "tensor2": np.array([5, 6])
        }

        # Fix: Create mock context dict
        context = create_mock_context(runtime_mock=Mock(hardware_dispatcher=None)) # Test fallback
        result = await nh.memristor_mvm_node(node, context, inputs)

        # Fix: Check for 'product' key
        assert "product" in result
        assert len(result["product"]) == 2

    @pytest.mark.asyncio
    async def test_memristor_mvm_node_missing_inputs(self):
        """Test MEMRISTOR_MVM without inputs"""
        node = {}
        inputs = {}

        result = await nh.memristor_mvm_node(node, create_mock_context(), inputs)

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_photonic_mvm_node(self):
        """Test PhotonicMVMNode"""
        node = {
            "params": {
                "photonic_params": {
                    "noise_std": 0.01
                }
            }
        }
        inputs = {
            "matrix": np.array([[1.0, 2.0]]),
            "vector": np.array([3.0, 4.0])
        }

        # Fix: Create mock context dict
        context = create_mock_context(runtime_mock=Mock(hardware_dispatcher=None)) # Test fallback
        result = await nh.photonic_mvm_node(node, context, inputs)

        assert "output" in result
        # Check that the calculation is roughly correct (1*3 + 2*4 = 11)
        assert 10.9 < result["output"][0] < 11.1

    @pytest.mark.asyncio
    async def test_photonic_mvm_node_invalid_compression(self):
        """Test PhotonicMVMNode with invalid compression"""
        node = {
            "params": {
                "photonic_params": {
                    "compression": "invalid"
                }
            }
        }
        inputs = {
            "matrix": np.array([[1, 2]]),
            "vector": np.array([3, 4])
        }

        result = await nh.photonic_mvm_node(node, create_mock_context(), inputs)

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_photonic_mvm_node_high_noise(self):
        """Test PhotonicMVMNode with excessive noise"""
        node = {
            "params": {
                "photonic_params": {
                    "noise_std": 0.1  # Too high
                }
            }
        }
        inputs = {
            "matrix": np.array([[1, 2]]),
            "vector": np.array([3, 4])
        }

        result = await nh.photonic_mvm_node(node, create_mock_context(), inputs)

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_sparse_mvm_node(self):
        """Test SPARSE_MVM node"""
        with patch('node_handlers.TORCH_AVAILABLE', False):
            node = {}
            inputs = {"matrix": [1], "vector": [1]} # Need to provide inputs

            result = await nh.sparse_mvm_node(node, create_mock_context(), inputs)

            assert "error_code" in result
            assert "PyTorch" in result["message"]

    @pytest.mark.asyncio
    async def test_fused_kernel_node(self):
        """Test FUSED_KERNEL node"""
        with patch('node_handlers.HIDET_AVAILABLE', False):
            node = {"params": {"subgraph": {"nodes": [], "edges": []}}} # Need valid subgraph

            result = await nh.fused_kernel_node(node, create_mock_context(), {})

            assert "error_code" in result


class TestDistributedNodeHandlers:
    """Test distributed computation node handlers"""

    @pytest.mark.asyncio
    async def test_sharded_computation_node_missing_subgraph(self):
        """Test SHARDED_COMPUTATION without subgraph"""
        node = {"params": {}}

        result = await nh.sharded_computation_node(node, create_mock_context(), {})

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_composite_node_missing_runtime(self):
        """Test COMPOSITE node without runtime"""
        node = {"type": "CustomComposite"}

        # *** FIX: Create a context dict with runtime=None directly ***
        context = create_mock_context(runtime_mock=None)

        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.composite_node(node, context, {})

        assert "Invalid context object" in str(e.value)


class TestMetaNodeHandlers:
    """Test meta and recursive node handlers"""

    @pytest.mark.asyncio
    async def test_meta_graph_node_missing_meta_graph(self):
        """Test MetaGraphNode without meta_graph"""
        node = {"params": {}}

        # <<< --- START FIX --- >>>
        # This test needs a runtime object to exist to get past the first check.
        context = create_mock_context(runtime_mock=Mock())
        # <<< --- END FIX --- >>>
        result = await nh.meta_graph_node(node, context, {})

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_meta_graph_node_missing_runtime(self):
        """Test MetaGraphNode without runtime"""
        node = {"params": {"meta_graph": {}}}

        # Fix: Create a context dict with runtime=None directly
        context = create_mock_context(runtime_mock=None)

        # <<< --- START FIX --- >>>
        # The test should now correctly catch the NodeExecutorError
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.meta_graph_node(node, context, {})

        assert "Invalid context object" in str(e.value)
        # <<< --- END FIX --- >>>

    @pytest.mark.asyncio
    async def test_meta_graph_node_recursion_limit(self):
        """Test MetaGraphNode recursion limit"""
        node = {"params": {"meta_graph": {"nodes": [], "edges": []}}}

        # Fix: Create a mock runtime and a context dict with high recursion_depth
        mock_runtime = Mock()
        # Mock the config attribute on the runtime
        mock_runtime.config = Mock(max_recursion_depth=20)

        context = create_mock_context(
            runtime_mock=mock_runtime,
            recursion_depth=25
        )

        result = await nh.meta_graph_node(node, context, {})

        assert "error_code" in result
        assert "recursion" in result.get("message", "").lower()


class TestAutoMLNodeHandlers:
    """Test AutoML node handlers"""

    @pytest.mark.asyncio
    async def test_random_node_uniform(self):
        """Test RandomNode with uniform distribution"""
        node = {
            "params": {
                "distribution": "uniform",
                "low": 0.0,
                "high": 1.0,
                "shape": [10]
            }
        }

        result = await nh.random_node(node, create_mock_context(), {})

        assert "value" in result
        assert len(result["value"]) == 10

    @pytest.mark.asyncio
    async def test_random_node_normal(self):
        """Test RandomNode with normal distribution"""
        node = {
            "params": {
                "distribution": "normal",
                "loc": 0.0,
                "scale": 1.0,
                "shape": [5]
            }
        }

        result = await nh.random_node(node, create_mock_context(), {})

        assert "value" in result

    @pytest.mark.asyncio
    async def test_random_node_unsupported_distribution(self):
        """Test RandomNode with unsupported distribution"""
        node = {
            "params": {
                "distribution": "unsupported",
                "shape": [5]
            }
        }

        result = await nh.random_node(node, create_mock_context(), {})

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_random_node_too_large(self):
        """Test RandomNode with excessive size"""
        node = {
            "params": {
                "distribution": "uniform",
                "shape": [1000001]  # Too large (limit is 1M)
            }
        }

        result = await nh.random_node(node, create_mock_context(), {})

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_hyperparam_node(self):
        """Test HyperParamNode"""
        node = {"params": {"value": 0.5}}

        result = await nh.hyperparam_node(node, create_mock_context(), {})

        assert result["value"] == 0.5

    @pytest.mark.asyncio
    async def test_search_node_missing_subgraph(self):
        """Test SearchNode without subgraph"""
        node = {"params": {}}

        result = await nh.search_node(node, create_mock_context(), {})

        assert "error_code" in result


class TestGovernanceNodeHandlers:
    """Test governance and audit node handlers"""

    @pytest.mark.asyncio
    async def test_contract_node_missing_aligner(self):
        """Test ContractNode without NSO aligner"""
        node = {}
        inputs = {"proposal": {"content": "test"}}

        # Fix: Create context dict with a runtime that *lacks* 'extensions'
        context = create_mock_context(runtime_mock=Mock(spec=[])) # Mock with no attributes
        result = await nh.contract_node(node, context, inputs)

        # The node should now skip alignment and approve
        assert "error_code" not in result
        assert result["approved"] is True
        assert result["audit_result"] == "skipped_no_aligner"

    @pytest.mark.asyncio
    async def test_proposal_node(self):
        """Test ProposalNode"""
        node = {
            "id": "p1",
            "params": {
                "proposal_content": {"action": "update"}
            }
        }

        result = await nh.proposal_node(node, create_mock_context(), {})

        assert "proposal" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_consensus_node_no_votes(self):
        """Test ConsensusNode with no votes"""
        node = {"params": {"threshold": 0.5}}
        inputs = {}

        result = await nh.consensus_node(node, create_mock_context(), inputs)

        assert result["consensus"] == "no_votes"
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_consensus_node_approved(self):
        """Test ConsensusNode with approval"""
        node = {"params": {"threshold": 0.5}}
        inputs = {
            "votes": [
                {"approve": True},
                {"approve": True},
                {"approve": False}
            ]
        }

        result = await nh.consensus_node(node, create_mock_context(), inputs)

        assert result["approved"] is True
        assert result["approval_rate"] > 0.5

    @pytest.mark.asyncio
    async def test_validation_node_no_graph(self):
        """Test ValidationNode without graph"""
        node = {}
        inputs = {}

        result = await nh.validation_node(node, create_mock_context(), inputs)

        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_validation_node_valid_graph(self):
        """Test ValidationNode with valid graph"""
        node = {}
        inputs = {
            "graph": {
                "nodes": [{"id": "n1", "type": "Test"}],
                "edges": []
            }
        }

        # *** FIX: Create a mock runtime dict that *lacks* 'validate_graph' ***
        # This tests the fallback logic in the handler
        context = create_mock_context(runtime_mock=Mock(spec=[]))
        result = await nh.validation_node(node, context, inputs)

        # Basic validation fallback should pass
        assert result["valid"] is True
        assert "errors" in result
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_audit_node(self):
        """Test AuditNode"""
        node = {
            "id": "a1",
            "params": {"audit_type": "security"}
        }
        inputs = {"data": {"action": "test"}}

        # Fix: Create a mock context dict
        context = create_mock_context()
        result = await nh.audit_node(node, context, inputs)

        assert result["audit"] == "logged"
        assert result["audit_type"] == "security"
        # Check that the log was actually written to the dict
        assert len(context['audit_log']) == 1
        # *** FIX: Check the correct key "type" ***
        assert context['audit_log'][0]["type"] == "security"

    @pytest.mark.asyncio
    async def test_execute_node_disabled(self):
        """Test ExecuteNode is disabled for safety"""
        node = {}

        result = await nh.execute_node(node, create_mock_context(), {})

        assert result["executed"] is False
        assert "safety" in result["reason"].lower()


class TestSchedulerNodeHandlers:
    """Test scheduler node handlers"""

    @pytest.mark.asyncio
    async def test_scheduler_node_missing_subgraph(self):
        """Test SchedulerNode without subgraph"""
        node = {"params": {}}

        result = await nh.scheduler_node(node, create_mock_context(), {})

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_scheduler_node_basic(self):
        """Test SchedulerNode basic functionality"""
        node = {
            "id": "s1",
            "params": {
                "subgraph": {"nodes": [], "edges": []},
                "interval_ms": 10, # Use short interval for testing
                "max_iterations": 1
            }
        }

        # Fix: Mock the runtime and its execute_graph method
        mock_runtime = Mock()
        mock_runtime.execute_graph = AsyncMock(return_value={})
        context = create_mock_context(runtime_mock=mock_runtime)

        result = await nh.scheduler_node(node, context, {})

        assert result["status"] == "scheduled"
        assert result["interval_ms"] == 100 # Check that it was clamped to min

        # Give the background task time to run once
        await asyncio.sleep(0.2)
        mock_runtime.execute_graph.assert_called_once()


class TestUtilityNodeHandlers:
    """Test utility node handlers"""

    @pytest.mark.asyncio
    async def test_normalize_node_minmax(self):
        """Test NormalizeNode with minmax"""
        node = {"params": {"method": "minmax"}}
        inputs = {"data": [1, 2, 3, 4, 5]}

        result = await nh.normalize_node(node, create_mock_context(), inputs)

        assert "output" in result
        assert max(result["output"]) == pytest.approx(1.0)
        assert min(result["output"]) == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_normalize_node_zscore(self):
        """Test NormalizeNode with zscore"""
        node = {"params": {"method": "zscore"}}
        inputs = {"data": np.array([1, 2, 3, 4, 5])}

        result = await nh.normalize_node(node, create_mock_context(), inputs)

        assert "output" in result
        assert np.mean(result["output"]) == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_normalize_node_no_data(self):
        """Test NormalizeNode without data"""
        node = {"params": {}}
        inputs = {}

        result = await nh.normalize_node(node, create_mock_context(), inputs)

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_cnn_node_handler(self):
        """Test CNNNode placeholder"""
        node = {}
        inputs = {"input": "data"}

        result = await nh.cnn_node_handler(node, create_mock_context(), inputs)

        assert "output" in result

    @pytest.mark.asyncio
    async def test_meta_node_handler(self):
        """Test generic MetaNode"""
        node = {}
        inputs = {"input": "test"}

        result = await nh.meta_node_handler(node, create_mock_context(), inputs)

        assert result["output"] == "test"


class TestNodeRegistry:
    """Test node handler registry"""

    def test_get_node_handlers(self):
        """Test getting node handlers registry"""
        handlers = nh.get_node_handlers()

        assert isinstance(handlers, dict)
        assert len(handlers) > 0

    def test_registry_has_core_nodes(self):
        """Test registry contains core nodes"""
        handlers = nh.get_node_handlers()

        assert 'CONST' in handlers
        assert 'ADD' in handlers
        assert 'MUL' in handlers

    def test_registry_has_ai_nodes(self):
        """Test registry contains AI nodes"""
        handlers = nh.get_node_handlers()

        assert 'EMBED' in handlers
        assert 'GenerativeNode' in handlers # Check one of the keys

    def test_registry_has_hardware_nodes(self):
        """Test registry contains hardware nodes"""
        handlers = nh.get_node_handlers()

        assert 'PHOTONIC_MVM' in handlers
        assert 'MEMRISTOR_MVM' in handlers

    def test_registry_handlers_are_callable(self):
        """Test that all handlers are callable"""
        handlers = nh.get_node_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"{name} is not callable"

    def test_validate_node_handler_valid(self):
        """Test validating a valid handler"""
        async def valid_handler(node, context, inputs):
            return {}

        is_valid = nh.validate_node_handler(valid_handler)
        assert is_valid is True

    def test_validate_node_handler_not_async(self):
        """Test validating a non-async handler"""
        def invalid_handler(node, context, inputs):
            return {}

        is_valid = nh.validate_node_handler(invalid_handler)
        assert is_valid is False

    def test_validate_node_handler_wrong_params(self):
        """Test validating handler with wrong number of params"""
        async def invalid_handler(node, context):
            return {}

        is_valid = nh.validate_node_handler(invalid_handler)
        assert is_valid is False


class TestIntegration:
    """Integration tests for node handlers"""

    @pytest.mark.asyncio
    async def test_complete_node_execution_chain(self):
        """Test chaining multiple node executions"""
        mock_context = create_mock_context()

        # Create nodes
        const_result = await nh.const_node({"params": {"value": 10}}, mock_context, {})

        # Use const output in add
        add_result = await nh.add_node(
            {},
            mock_context,
            {"val1": const_result["value"], "val2": 5}
        )

        # Use add output in multiply
        # Fix: Check for 'result' key from add_node
        mul_result = await nh.multiply_node(
            {},
            mock_context,
            {"val1": add_result["result"], "val2": 2}
        )

        # Fix: Check for 'result' key from multiply_node
        assert mul_result["result"] == 30

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors propagate correctly"""
        # Fix: Try to add without inputs and expect an error
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.add_node({}, create_mock_context(), {})

        assert "Missing inputs" in str(e.value)


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_add_node_with_strings(self):
        """Test ADD node with string concatenation"""
        node = {}
        inputs = {"val1": "hello", "val2": "world"}

        # Fix: Assert that the correct error is raised
        with pytest.raises(nh.NodeExecutorError) as e:
            await nh.add_node(node, create_mock_context(), inputs)

        assert "Invalid inputs" in str(e.value)

    @pytest.mark.asyncio
    async def test_multiply_node_with_zero(self):
        """Test MULTIPLY node with zero"""
        node = {}
        inputs = {"val1": 100, "val2": 0}

        result = await nh.multiply_node(node, create_mock_context(), inputs)

        # Fix: Check for 'result' key
        assert result["result"] == 0

    @pytest.mark.asyncio
    async def test_branch_node_with_none_value(self):
        """Test BRANCH node with None value"""
        node = {}
        inputs = {"condition": True, "value": None}

        result = await nh.branch_node(node, create_mock_context(), inputs)

        assert result["on_true"] is None
        assert result["on_false"] is None

    @pytest.mark.asyncio
    async def test_normalize_node_constant_values(self):
        """Test NORMALIZE node with constant values"""
        node = {"params": {"method": "minmax"}}
        inputs = {"data": [5, 5, 5, 5]}

        result = await nh.normalize_node(node, create_mock_context(), inputs)

        # Should handle constant values gracefully (all zeros)
        assert "output" in result
        assert result["output"] == [0.0, 0.0, 0.0, 0.0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
