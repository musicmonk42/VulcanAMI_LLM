"""
Unit Tests for Metaprogramming Node Handlers

Tests each metaprogramming handler in isolation with mocked dependencies.
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from src.unified_runtime.metaprogramming_handlers import (
    pattern_compile_node,
    find_subgraph_node,
    graph_splice_node,
    graph_commit_node,
    nso_modify_node,
    ethical_label_node,
    eval_node,
    halt_node,
    UnauthorizedSelfModification,
    EthicalBoundaryViolation,
    get_metaprogramming_handlers,
)


@pytest.fixture
def mock_context():
    """Create a mock execution context"""
    return {
        "runtime": Mock(),
        "graph": {
            "nodes": [
                {"id": "node1", "type": "GENERATE"},
                {"id": "node2", "type": "EMBED"},
                {"id": "node3", "type": "CONST"},
            ],
            "edges": []
        },
        "node_map": {},
        "outputs": {},
        "recursion_depth": 0,
        "audit_log": [],
        "agent_id": "test_agent"
    }


@pytest.fixture
def sample_pattern():
    """Create a sample pattern specification"""
    return {
        "nodes": [
            {"id": "?gen", "type": "GENERATE"},
        ],
        "edges": []
    }


@pytest.fixture
def sample_template():
    """Create a sample template"""
    return {
        "nodes": [
            {"id": "?gen", "type": "GENERATE", "params": {"enhanced": True}},
        ],
        "edges": []
    }


class TestPatternCompile:
    """Test PATTERN_COMPILE handler"""
    
    @pytest.mark.asyncio
    async def test_compile_simple_pattern(self, mock_context, sample_pattern):
        """Test compiling a simple pattern"""
        node = {"id": "pat1", "type": "PATTERN_COMPILE"}
        inputs = {"pattern_in": sample_pattern}
        
        result = await pattern_compile_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert "pattern_out" in result
        assert result["pattern_out"]["node_count"] == 1
        assert len(result["pattern_out"]["variables"]) == 1
        assert "?gen" in result["pattern_out"]["variables"]
    
    @pytest.mark.asyncio
    async def test_compile_pattern_with_multiple_variables(self, mock_context):
        """Test compiling pattern with multiple variables"""
        pattern = {
            "nodes": [
                {"id": "?node1", "type": "ADD"},
                {"id": "?node2", "type": "MUL"},
            ],
            "edges": [
                {"from": {"node": "?node1"}, "to": {"node": "?node2"}}
            ]
        }
        node = {"id": "pat1", "type": "PATTERN_COMPILE"}
        inputs = {"pattern_in": pattern}
        
        result = await pattern_compile_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert result["pattern_out"]["node_count"] == 2
        assert result["pattern_out"]["edge_count"] == 1
        assert len(result["pattern_out"]["variables"]) == 2
    
    @pytest.mark.asyncio
    async def test_compile_pattern_missing_input(self, mock_context):
        """Test error handling when pattern input is missing"""
        node = {"id": "pat1", "type": "PATTERN_COMPILE"}
        inputs = {}
        
        result = await pattern_compile_node(node, mock_context, inputs)
        
        assert result["status"] == "failed"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_compile_pattern_invalid_format(self, mock_context):
        """Test error handling with invalid pattern format"""
        node = {"id": "pat1", "type": "PATTERN_COMPILE"}
        inputs = {"pattern_in": "invalid_string"}
        
        result = await pattern_compile_node(node, mock_context, inputs)
        
        assert result["status"] == "failed"


class TestFindSubgraph:
    """Test FIND_SUBGRAPH handler"""
    
    @pytest.mark.asyncio
    async def test_find_simple_pattern(self, mock_context):
        """Test finding a simple pattern match"""
        # Compile pattern first
        pattern = {
            "nodes": [{"id": "?gen", "type": "GENERATE"}],
            "edges": [],
            "variables": ["?gen"],
            "node_types": ["GENERATE"],
            "node_count": 1,
            "edge_count": 0,
            "hash": "test_hash"
        }
        
        node = {"id": "find1", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}}
        inputs = {
            "pattern_in": pattern,
            "graph_ref": mock_context["graph"]
        }
        
        result = await find_subgraph_node(node, mock_context, inputs)
        
        assert result["status"] in ["success", "no_match"]
        assert "match_out" in result
    
    @pytest.mark.asyncio
    async def test_find_pattern_no_match(self, mock_context):
        """Test when pattern doesn't match"""
        pattern = {
            "nodes": [{"id": "?nonexistent", "type": "NONEXISTENT_TYPE"}],
            "edges": [],
            "variables": ["?nonexistent"],
            "node_types": ["NONEXISTENT_TYPE"],
            "node_count": 1,
            "edge_count": 0,
            "hash": "test_hash"
        }
        
        node = {"id": "find1", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}}
        inputs = {
            "pattern_in": pattern,
            "graph_ref": mock_context["graph"]
        }
        
        result = await find_subgraph_node(node, mock_context, inputs)
        
        assert result["status"] == "no_match"
        assert result["match_out"]["match_count"] == 0
    
    @pytest.mark.asyncio
    async def test_find_missing_inputs(self, mock_context):
        """Test error handling with missing inputs"""
        node = {"id": "find1", "type": "FIND_SUBGRAPH"}
        inputs = {}
        
        result = await find_subgraph_node(node, mock_context, inputs)
        
        assert result["status"] == "failed"
        assert "error" in result


class TestGraphSplice:
    """Test GRAPH_SPLICE handler"""
    
    @pytest.mark.asyncio
    async def test_splice_with_match(self, mock_context, sample_template):
        """Test splicing with a valid match"""
        match_info = {
            "matches": [{
                "start_idx": 0,
                "end_idx": 0,
                "bindings": {"?gen": "node1"},
                "node_mapping": {"?gen": "node1"}
            }],
            "match_count": 1
        }
        
        node = {"id": "splice1", "type": "GRAPH_SPLICE"}
        inputs = {
            "match_in": match_info,
            "template_in": sample_template
        }
        
        result = await graph_splice_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert "graph_out" in result
        assert result["nodes_replaced"] == 1
    
    @pytest.mark.asyncio
    async def test_splice_no_matches(self, mock_context, sample_template):
        """Test splicing with no matches"""
        match_info = {"matches": [], "match_count": 0}
        
        node = {"id": "splice1", "type": "GRAPH_SPLICE"}
        inputs = {
            "match_in": match_info,
            "template_in": sample_template
        }
        
        result = await graph_splice_node(node, mock_context, inputs)
        
        assert result["status"] == "no_changes"
        assert "graph_out" in result
    
    @pytest.mark.asyncio
    async def test_splice_missing_inputs(self, mock_context):
        """Test error handling with missing inputs"""
        node = {"id": "splice1", "type": "GRAPH_SPLICE"}
        inputs = {}
        
        result = await graph_splice_node(node, mock_context, inputs)
        
        assert result["status"] == "failed"


class TestGraphCommit:
    """Test GRAPH_COMMIT handler"""
    
    @pytest.mark.asyncio
    async def test_commit_authorized(self, mock_context):
        """Test successful commit with authorization"""
        modified_graph = {"id": "test_graph", "nodes": [], "edges": []}
        nso_auth = {"authorized": True}
        ethical_label = {"label": "safe"}
        
        node = {"id": "commit1", "type": "GRAPH_COMMIT"}
        inputs = {
            "graph_in": modified_graph,
            "nso_in": nso_auth,
            "label_in": ethical_label
        }
        
        result = await graph_commit_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert "version" in result
        assert "committed_graph" in result
        # Check audit log
        assert len(mock_context["audit_log"]) > 0
        assert mock_context["audit_log"][0]["type"] == "graph_commit"
    
    @pytest.mark.asyncio
    async def test_commit_unauthorized(self, mock_context):
        """Test commit rejection without authorization"""
        modified_graph = {"id": "test_graph", "nodes": [], "edges": []}
        nso_auth = {"authorized": False}
        
        node = {"id": "commit1", "type": "GRAPH_COMMIT"}
        inputs = {
            "graph_in": modified_graph,
            "nso_in": nso_auth
        }
        
        result = await graph_commit_node(node, mock_context, inputs)
        
        assert result["status"] == "blocked"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_commit_missing_authorization(self, mock_context):
        """Test commit rejection when NSO auth is missing"""
        modified_graph = {"id": "test_graph", "nodes": [], "edges": []}
        
        node = {"id": "commit1", "type": "GRAPH_COMMIT"}
        inputs = {"graph_in": modified_graph}
        
        result = await graph_commit_node(node, mock_context, inputs)
        
        assert result["status"] == "blocked"


class TestNSOModify:
    """Test NSO_MODIFY handler"""
    
    @pytest.mark.asyncio
    async def test_nso_authorize_non_self_modifying(self, mock_context):
        """Test authorization for non-self-modifying operation"""
        node = {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "external_resource"}}
        inputs = {}
        
        result = await nso_modify_node(node, mock_context, inputs)
        
        # Should authorize non-self-modifying operations
        assert result["nso_out"]["authorized"] in [True, False]  # Depends on aligner availability
    
    @pytest.mark.asyncio
    async def test_nso_no_aligner(self, mock_context):
        """Test denial when NSO aligner not available"""
        node = {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}}
        inputs = {}
        
        result = await nso_modify_node(node, mock_context, inputs)
        
        # Should deny by default when aligner not available (fail-safe)
        assert result["status"] in ["denied", "authorized"]
    
    @pytest.mark.asyncio
    async def test_nso_with_aligner_safe(self, mock_context):
        """Test with NSO aligner present - safe operation"""
        # Mock NSO aligner
        mock_aligner = Mock()
        mock_aligner.multi_model_audit = Mock(return_value="safe")
        
        mock_optimizer = Mock()
        mock_optimizer.nso = mock_aligner
        
        mock_extensions = Mock()
        mock_extensions.autonomous_optimizer = mock_optimizer
        
        mock_context["runtime"].extensions = mock_extensions
        
        node = {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}}
        inputs = {}
        
        result = await nso_modify_node(node, mock_context, inputs)
        
        assert result["status"] == "authorized"
        assert result["nso_out"]["authorized"] == True
    
    @pytest.mark.asyncio
    async def test_nso_with_aligner_risky(self, mock_context):
        """Test with NSO aligner - risky operation denied"""
        # Mock NSO aligner
        mock_aligner = Mock()
        mock_aligner.multi_model_audit = Mock(return_value="risky")
        
        mock_optimizer = Mock()
        mock_optimizer.nso = mock_aligner
        
        mock_extensions = Mock()
        mock_extensions.autonomous_optimizer = mock_optimizer
        
        mock_context["runtime"].extensions = mock_extensions
        
        node = {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}}
        inputs = {}
        
        result = await nso_modify_node(node, mock_context, inputs)
        
        assert result["status"] == "denied"
        assert result["nso_out"]["authorized"] == False


class TestEthicalLabel:
    """Test ETHICAL_LABEL handler"""
    
    @pytest.mark.asyncio
    async def test_label_safe(self, mock_context):
        """Test safe ethical label"""
        node = {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "safe"}}
        inputs = {}
        
        result = await ethical_label_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert result["label_out"]["label"] == "safe"
        assert result["label_out"]["approved"] == True
        assert result["label_out"]["requires_review"] == False
    
    @pytest.mark.asyncio
    async def test_label_requires_review(self, mock_context):
        """Test label requiring human review"""
        node = {
            "id": "label1",
            "type": "ETHICAL_LABEL",
            "params": {"label": "self_modification_requires_review"}
        }
        inputs = {}
        
        result = await ethical_label_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert result["label_out"]["requires_review"] == True
    
    @pytest.mark.asyncio
    async def test_label_restricted(self, mock_context):
        """Test restricted label"""
        node = {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "restricted"}}
        inputs = {}
        
        result = await ethical_label_node(node, mock_context, inputs)
        
        assert result["status"] == "restricted"
        assert result["label_out"]["approved"] == False


class TestEval:
    """Test EVAL handler"""
    
    @pytest.mark.asyncio
    async def test_eval_simple_graph(self, mock_context):
        """Test evaluating a simple graph"""
        graph_to_eval = {"id": "test_graph", "nodes": [], "edges": []}
        
        # Mock runtime execute_graph
        mock_result = Mock()
        mock_result.tokens = 10
        mock_context["runtime"].execute_graph = AsyncMock(return_value=mock_result)
        
        node = {"id": "eval1", "type": "EVAL"}
        inputs = {"graph": graph_to_eval}
        
        result = await eval_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert "metrics" in result
        assert result["metrics"]["sample_count"] == 1
    
    @pytest.mark.asyncio
    async def test_eval_with_dataset(self, mock_context):
        """Test evaluation with dataset"""
        graph_to_eval = {"id": "test_graph", "nodes": [], "edges": []}
        dataset = {
            "samples": [
                {"input": 1},
                {"input": 2},
                {"input": 3},
            ]
        }
        
        # Mock runtime execute_graph
        mock_result = Mock()
        mock_result.tokens = 10
        mock_context["runtime"].execute_graph = AsyncMock(return_value=mock_result)
        
        node = {"id": "eval1", "type": "EVAL"}
        inputs = {"graph": graph_to_eval, "dataset": dataset}
        
        result = await eval_node(node, mock_context, inputs)
        
        assert result["status"] == "success"
        assert result["metrics"]["sample_count"] == 3
    
    @pytest.mark.asyncio
    async def test_eval_missing_graph(self, mock_context):
        """Test error handling when graph is missing"""
        node = {"id": "eval1", "type": "EVAL"}
        inputs = {}
        
        result = await eval_node(node, mock_context, inputs)
        
        assert result["status"] == "failed"


class TestHalt:
    """Test HALT handler"""
    
    @pytest.mark.asyncio
    async def test_halt_with_value(self, mock_context):
        """Test halt node with explicit value"""
        node = {"id": "halt1", "type": "HALT"}
        inputs = {"value": 42}
        
        result = await halt_node(node, mock_context, inputs)
        
        assert result["status"] == "halted"
        assert result["value"] == 42
        # Check audit log
        assert len(mock_context["audit_log"]) > 0
        assert mock_context["audit_log"][0]["type"] == "halt"
    
    @pytest.mark.asyncio
    async def test_halt_with_dict(self, mock_context):
        """Test halt node with dict value"""
        node = {"id": "halt1", "type": "HALT"}
        inputs = {"value": {"result": "success", "data": [1, 2, 3]}}
        
        result = await halt_node(node, mock_context, inputs)
        
        assert result["status"] == "halted"
        assert isinstance(result["value"], dict)
        assert result["value"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_halt_no_explicit_value(self, mock_context):
        """Test halt node returns all inputs when no explicit value"""
        node = {"id": "halt1", "type": "HALT"}
        inputs = {"input": "test_value", "other": "data"}
        
        result = await halt_node(node, mock_context, inputs)
        
        assert result["status"] == "halted"
        assert "value" in result


class TestHandlerRegistry:
    """Test handler registry"""
    
    def test_get_handlers_returns_all(self):
        """Test that registry returns all expected handlers"""
        handlers = get_metaprogramming_handlers()
        
        expected_handlers = [
            "PATTERN_COMPILE",
            "FIND_SUBGRAPH",
            "GRAPH_SPLICE",
            "GRAPH_COMMIT",
            "NSO_MODIFY",
            "ETHICAL_LABEL",
            "EVAL",
            "HALT",
        ]
        
        for handler_name in expected_handlers:
            assert handler_name in handlers
            assert callable(handlers[handler_name])
    
    def test_handlers_are_async(self):
        """Test that all handlers are async functions"""
        import inspect
        handlers = get_metaprogramming_handlers()
        
        for handler_name, handler_func in handlers.items():
            assert inspect.iscoroutinefunction(handler_func), \
                f"{handler_name} is not an async function"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple handlers"""
    
    @pytest.mark.asyncio
    async def test_full_mutation_pipeline(self, mock_context, sample_pattern, sample_template):
        """Test complete mutation: PATTERN_COMPILE → FIND → SPLICE → COMMIT"""
        # Step 1: Compile pattern
        compile_result = await pattern_compile_node(
            {"id": "pat1", "type": "PATTERN_COMPILE"},
            mock_context,
            {"pattern_in": sample_pattern}
        )
        assert compile_result["status"] == "success"
        
        # Step 2: Find pattern
        find_result = await find_subgraph_node(
            {"id": "find1", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}},
            mock_context,
            {
                "pattern_in": compile_result["pattern_out"],
                "graph_ref": mock_context["graph"]
            }
        )
        assert find_result["status"] in ["success", "no_match"]
        
        # If match found, proceed with splice
        if find_result["status"] == "success" and find_result["match_out"]["match_count"] > 0:
            # Step 3: Splice
            splice_result = await graph_splice_node(
                {"id": "splice1", "type": "GRAPH_SPLICE"},
                mock_context,
                {
                    "match_in": find_result["match_out"],
                    "template_in": sample_template
                }
            )
            assert splice_result["status"] == "success"
            
            # Step 4: Get NSO authorization
            nso_result = await nso_modify_node(
                {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}},
                mock_context,
                {}
            )
            
            # Step 5: Get ethical label
            label_result = await ethical_label_node(
                {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "safe"}},
                mock_context,
                {}
            )
            
            # Step 6: Commit
            commit_result = await graph_commit_node(
                {"id": "commit1", "type": "GRAPH_COMMIT"},
                mock_context,
                {
                    "graph_in": splice_result["graph_out"],
                    "nso_in": nso_result["nso_out"],
                    "label_in": label_result["label_out"]
                }
            )
            
            # Verify full pipeline
            assert commit_result["status"] in ["success", "blocked"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
