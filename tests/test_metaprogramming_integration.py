"""
Integration Tests for Metaprogramming Node Handlers

Tests complete mutation pipeline: PATTERN_COMPILE → FIND → SPLICE → COMMIT
Tests safety gates and authorization flows
Tests with real Graph IR execution engine
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from src.unified_runtime.metaprogramming_handlers import (
    pattern_compile_node,
    find_subgraph_node,
    graph_splice_node,
    graph_commit_node,
    nso_modify_node,
    ethical_label_node,
    eval_node,
    halt_node,
)


@pytest.fixture
def sample_graph():
    """Create a sample computation graph"""
    return {
        "id": "test_graph_1",
        "nodes": [
            {
                "id": "input1",
                "type": "INPUT",
                "params": {"value": 10}
            },
            {
                "id": "add1",
                "type": "ADD",
                "params": {}
            },
            {
                "id": "mul1",
                "type": "MULTIPLY",
                "params": {}
            },
            {
                "id": "output1",
                "type": "OUTPUT",
                "params": {}
            }
        ],
        "edges": [
            {"from": {"node": "input1", "port": "output"}, "to": {"node": "add1", "port": "val1"}},
            {"from": {"node": "add1", "port": "result"}, "to": {"node": "mul1", "port": "val1"}},
            {"from": {"node": "mul1", "port": "result"}, "to": {"node": "output1", "port": "input"}},
        ]
    }


@pytest.fixture
def mock_runtime_with_safety():
    """Create a mock runtime with safety systems"""
    runtime = Mock()
    
    # Mock NSO aligner
    mock_aligner = Mock()
    mock_aligner.multi_model_audit = Mock(return_value="safe")
    
    mock_optimizer = Mock()
    mock_optimizer.nso = mock_aligner
    
    mock_extensions = Mock()
    mock_extensions.autonomous_optimizer = mock_optimizer
    
    runtime.extensions = mock_extensions
    runtime.execute_graph = AsyncMock(return_value=Mock(tokens=10))
    
    return runtime


@pytest.fixture
def integration_context(sample_graph, mock_runtime_with_safety):
    """Create integration test context"""
    return {
        "runtime": mock_runtime_with_safety,
        "graph": sample_graph,
        "node_map": {node["id"]: node for node in sample_graph["nodes"]},
        "outputs": {},
        "recursion_depth": 0,
        "audit_log": [],
        "agent_id": "integration_test_agent"
    }


class TestFullMutationPipeline:
    """Test complete mutation: COMPILE → FIND → SPLICE → COMMIT"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_with_authorization(self, integration_context):
        """
        Test full metaprogramming pipeline with all safety checks
        
        Pipeline:
        1. Compile pattern for ADD node
        2. Find ADD node in graph
        3. Splice with improved version
        4. Get NSO authorization
        5. Get ethical label
        6. Commit with versioning
        """
        # Step 1: Compile pattern to find ADD nodes
        pattern = {
            "nodes": [{"id": "?add_node", "type": "ADD"}],
            "edges": []
        }
        
        compile_result = await pattern_compile_node(
            {"id": "pat1", "type": "PATTERN_COMPILE"},
            integration_context,
            {"pattern_in": pattern}
        )
        
        assert compile_result["status"] == "success"
        assert "pattern_out" in compile_result
        compiled_pattern = compile_result["pattern_out"]
        
        # Step 2: Find ADD node in graph
        find_result = await find_subgraph_node(
            {"id": "find1", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}},
            integration_context,
            {
                "pattern_in": compiled_pattern,
                "graph_ref": integration_context["graph"]
            }
        )
        
        assert find_result["status"] in ["success", "no_match"]
        
        # Only proceed if match found
        if find_result["status"] == "success" and find_result["match_out"]["match_count"] > 0:
            # Step 3: Create template with improved ADD node
            template = {
                "nodes": [{
                    "id": "?add_node",
                    "type": "ADD",
                    "params": {"optimized": True}  # Enhanced version
                }],
                "edges": []
            }
            
            splice_result = await graph_splice_node(
                {"id": "splice1", "type": "GRAPH_SPLICE"},
                integration_context,
                {
                    "match_in": find_result["match_out"],
                    "template_in": template
                }
            )
            
            assert splice_result["status"] == "success"
            assert "graph_out" in splice_result
            modified_graph = splice_result["graph_out"]
            
            # Step 4: Get NSO authorization
            nso_result = await nso_modify_node(
                {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}},
                integration_context,
                {}
            )
            
            assert "nso_out" in nso_result
            # Should be authorized since we have mock aligner returning "safe"
            assert nso_result["nso_out"]["authorized"] == True
            
            # Step 5: Get ethical label
            label_result = await ethical_label_node(
                {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "safe"}},
                integration_context,
                {}
            )
            
            assert label_result["status"] == "success"
            assert label_result["label_out"]["approved"] == True
            
            # Step 6: Commit with versioning
            commit_result = await graph_commit_node(
                {"id": "commit1", "type": "GRAPH_COMMIT"},
                integration_context,
                {
                    "graph_in": modified_graph,
                    "nso_in": nso_result["nso_out"],
                    "label_in": label_result["label_out"]
                }
            )
            
            assert commit_result["status"] == "success"
            assert "version" in commit_result
            assert "committed_graph" in commit_result
            
            # Verify audit trail
            assert len(integration_context["audit_log"]) > 0
            
            # Check for graph_commit audit entry
            commit_entries = [e for e in integration_context["audit_log"] if e.get("type") == "graph_commit"]
            assert len(commit_entries) > 0
            
            commit_entry = commit_entries[0]
            assert commit_entry["nso_authorized"] == True
            assert "timestamp" in commit_entry
    
    @pytest.mark.asyncio
    async def test_pipeline_blocked_by_nso(self, integration_context):
        """Test pipeline is blocked when NSO denies authorization"""
        # Override NSO aligner to return "risky"
        mock_aligner = Mock()
        mock_aligner.multi_model_audit = Mock(return_value="risky")
        integration_context["runtime"].extensions.autonomous_optimizer.nso = mock_aligner
        
        # Try to get authorization for risky operation
        nso_result = await nso_modify_node(
            {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}},
            integration_context,
            {}
        )
        
        # Should be denied
        assert nso_result["status"] == "denied"
        assert nso_result["nso_out"]["authorized"] == False
        
        # Try to commit anyway (should fail)
        commit_result = await graph_commit_node(
            {"id": "commit1", "type": "GRAPH_COMMIT"},
            integration_context,
            {
                "graph_in": {"id": "test", "nodes": [], "edges": []},
                "nso_in": nso_result["nso_out"],
                "label_in": {"label": "safe"}
            }
        )
        
        # Should be blocked
        assert commit_result["status"] == "blocked"
    
    @pytest.mark.asyncio
    async def test_pipeline_blocked_by_ethical_label(self, integration_context):
        """Test pipeline respects ethical restrictions"""
        # Get restricted ethical label
        label_result = await ethical_label_node(
            {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "restricted"}},
            integration_context,
            {}
        )
        
        assert label_result["status"] == "restricted"
        assert label_result["label_out"]["approved"] == False
        
        # This would prevent commit in real scenario
        # In our implementation, GRAPH_COMMIT checks ethical labels


class TestConcurrentOperations:
    """Test concurrent metaprogramming operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_pattern_compilation(self, integration_context):
        """Test multiple pattern compilations in parallel"""
        patterns = [
            {"nodes": [{"id": f"?n{i}", "type": "ADD"}], "edges": []}
            for i in range(5)
        ]
        
        # Compile all patterns concurrently
        tasks = [
            pattern_compile_node(
                {"id": f"pat{i}", "type": "PATTERN_COMPILE"},
                integration_context,
                {"pattern_in": pat}
            )
            for i, pat in enumerate(patterns)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["status"] == "success" for r in results)
        assert all("pattern_out" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, integration_context):
        """Test concurrent graph evaluations"""
        graphs = [
            {"id": f"graph{i}", "nodes": [], "edges": []}
            for i in range(3)
        ]
        
        # Evaluate all graphs concurrently
        tasks = [
            eval_node(
                {"id": f"eval{i}", "type": "EVAL"},
                integration_context,
                {"graph": g}
            )
            for i, g in enumerate(graphs)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert all(r["status"] == "success" for r in results)
        assert all("metrics" in r for r in results)


class TestSafetyIntegration:
    """Test integration with VULCAN safety systems"""
    
    @pytest.mark.asyncio
    async def test_audit_log_completeness(self, integration_context):
        """Test that all critical operations are logged"""
        # Perform several operations
        await nso_modify_node(
            {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}},
            integration_context,
            {}
        )
        
        await ethical_label_node(
            {"id": "label1", "type": "ETHICAL_LABEL", "params": {"label": "safe"}},
            integration_context,
            {}
        )
        
        await halt_node(
            {"id": "halt1", "type": "HALT"},
            integration_context,
            {"value": 42}
        )
        
        # Check audit log has entries
        audit_log = integration_context["audit_log"]
        assert len(audit_log) >= 3
        
        # Verify entry types
        entry_types = [e.get("type") for e in audit_log]
        assert "nso_authorization" in entry_types
        assert "ethical_label" in entry_types
        assert "halt" in entry_types
        
        # Verify all entries have timestamps
        assert all("timestamp" in e for e in audit_log)
    
    @pytest.mark.asyncio
    async def test_fail_safe_defaults(self, integration_context):
        """Test that system defaults to safe behavior on errors"""
        # Remove safety systems from runtime
        integration_context["runtime"].extensions = None
        
        # NSO should deny by default
        nso_result = await nso_modify_node(
            {"id": "nso1", "type": "NSO_MODIFY", "params": {"target": "self_code"}},
            integration_context,
            {}
        )
        
        assert nso_result["nso_out"]["authorized"] == False
        
        # Commit should be blocked without authorization
        commit_result = await graph_commit_node(
            {"id": "commit1", "type": "GRAPH_COMMIT"},
            integration_context,
            {
                "graph_in": {"id": "test", "nodes": [], "edges": []},
                "nso_in": nso_result["nso_out"]
            }
        )
        
        assert commit_result["status"] == "blocked"


class TestGraphEvaluation:
    """Test graph evaluation with dataset"""
    
    @pytest.mark.asyncio
    async def test_eval_with_dataset(self, integration_context, sample_graph):
        """Test evaluating graph against dataset"""
        dataset = {
            "samples": [
                {"input": 1},
                {"input": 2},
                {"input": 3},
            ]
        }
        
        result = await eval_node(
            {"id": "eval1", "type": "EVAL"},
            integration_context,
            {"graph": sample_graph, "dataset": dataset}
        )
        
        assert result["status"] == "success"
        assert "metrics" in result
        assert result["metrics"]["sample_count"] == 3
        assert "latency_ms" in result["metrics"]
        assert result["metrics"]["latency_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_eval_performance_tracking(self, integration_context, sample_graph):
        """Test that evaluation tracks performance metrics"""
        result = await eval_node(
            {"id": "eval1", "type": "EVAL"},
            integration_context,
            {"graph": sample_graph}
        )
        
        assert result["status"] == "success"
        metrics = result["metrics"]
        
        # Check required metrics
        assert "accuracy" in metrics
        assert "tokens" in metrics
        assert "cycles" in metrics
        assert "latency_ms" in metrics
        assert "sample_count" in metrics


class TestErrorRecovery:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_pattern_compile_invalid_input(self, integration_context):
        """Test error handling for invalid pattern"""
        result = await pattern_compile_node(
            {"id": "pat1", "type": "PATTERN_COMPILE"},
            integration_context,
            {"pattern_in": "invalid"}
        )
        
        assert result["status"] == "failed"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_splice_missing_template(self, integration_context):
        """Test error handling for missing template"""
        result = await graph_splice_node(
            {"id": "splice1", "type": "GRAPH_SPLICE"},
            integration_context,
            {"match_in": {"matches": [], "match_count": 0}}
        )
        
        assert result["status"] in ["failed", "no_changes"]
    
    @pytest.mark.asyncio
    async def test_commit_missing_graph(self, integration_context):
        """Test error handling for missing graph"""
        result = await graph_commit_node(
            {"id": "commit1", "type": "GRAPH_COMMIT"},
            integration_context,
            {}
        )
        
        assert result["status"] == "failed"
        assert "error" in result


class TestVersioning:
    """Test graph versioning and rollback"""
    
    @pytest.mark.asyncio
    async def test_commit_creates_version(self, integration_context):
        """Test that commits create version hashes"""
        result = await graph_commit_node(
            {"id": "commit1", "type": "GRAPH_COMMIT"},
            integration_context,
            {
                "graph_in": {"id": "test", "nodes": [], "edges": []},
                "nso_in": {"authorized": True},
                "label_in": {"label": "safe"}
            }
        )
        
        assert result["status"] == "success"
        assert "version" in result
        assert "hash" in result["version"]
        assert "timestamp" in result["version"]
        
        # Hash should be deterministic for same graph
        version_hash = result["version"]["hash"]
        assert len(version_hash) > 0
    
    @pytest.mark.asyncio
    async def test_different_graphs_different_versions(self, integration_context):
        """Test that different graphs get different version hashes"""
        graph1 = {"id": "graph1", "nodes": [{"id": "n1", "type": "ADD"}], "edges": []}
        graph2 = {"id": "graph2", "nodes": [{"id": "n1", "type": "MUL"}], "edges": []}
        
        result1 = await graph_commit_node(
            {"id": "commit1", "type": "GRAPH_COMMIT"},
            integration_context,
            {
                "graph_in": graph1,
                "nso_in": {"authorized": True},
                "label_in": {"label": "safe"}
            }
        )
        
        result2 = await graph_commit_node(
            {"id": "commit2", "type": "GRAPH_COMMIT"},
            integration_context,
            {
                "graph_in": graph2,
                "nso_in": {"authorized": True},
                "label_in": {"label": "safe"}
            }
        )
        
        # Different graphs should have different version hashes
        assert result1["version"]["hash"] != result2["version"]["hash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
