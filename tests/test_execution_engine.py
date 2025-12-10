"""
Comprehensive pytest suite for execution_engine.py
Fixed to work with the updated API that includes UnifiedRuntime integration

This version properly mocks the runtime's get_node_executor method
to return actual executors that the ExecutionEngine can use.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import the module to test
from src.unified_runtime import execution_engine as ee


# Create a comprehensive mock runtime for testing
class MockRuntime:
    """Mock UnifiedRuntime for testing"""
    def __init__(self):
        self.memory = MagicMock()
        self.reasoning = MagicMock()
        self.world_model = MagicMock()
        self.learning = MagicMock()
        self.safety = MagicMock()
        self.problem_decomposer = MagicMock()
        self.knowledge_crystallizer = MagicMock()
        self.curiosity_engine = MagicMock()
        self.semantic_bridge = MagicMock()
        self.orchestrator = MagicMock()
        self.plan_manager = MagicMock()
        self.processor = MagicMock()
        
        # Create a proper async executor that returns the expected result
        # The signature MUST match what _run_single_node calls: (node, context_dict, inputs_dict)
        async def mock_executor(node: Dict[str, Any], context: Dict[str, Any], inputs: Dict[str, Any]):
            """Mock executor that simulates successful node execution"""
            node_id = node.get('id', 'unknown')
            # Simulate some processing
            await asyncio.sleep(0.001)
            # The handler should return the OUTPUT DATA, not a NodeExecutionResult.
            # The _run_single_node function will wrap this in a NodeExecutionResult.
            return {"result": f"output_{node_id}"}
        
        # Mock get_node_executor to return our mock executor
        def get_node_executor(node_type):
            """Return a mock executor for any node type"""
            # Return the async executor function
            return mock_executor
        
        # Set up the mock to return an executor for any node type
        self.get_node_executor = MagicMock(side_effect=get_node_executor)


class TestExecutionStatus:
    """Test ExecutionStatus enum"""
    
    def test_status_values(self):
        """Test that all status values are strings"""
        assert ee.ExecutionStatus.PENDING.value == "pending"
        assert ee.ExecutionStatus.RUNNING.value == "running"
        assert ee.ExecutionStatus.SUCCESS.value == "success"
        assert ee.ExecutionStatus.FAILED.value == "failed"
        assert ee.ExecutionStatus.CANCELLED.value == "cancelled"
        assert ee.ExecutionStatus.TIMEOUT.value == "timeout"
        assert ee.ExecutionStatus.SKIPPED.value == "skipped"


class TestExecutionContext:
    """Test ExecutionContext dataclass"""
    
    def test_context_creation(self):
        """Test creating execution context"""
        graph = {"nodes": [{"id": "n1", "type": "test"}], "edges": []}
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph=graph,
            node_map={"n1": {"id": "n1", "type": "test"}},
            runtime=runtime
        )
        
        assert context.graph == graph
        assert len(context.node_map) == 1
        assert context.execution_id != ""
        assert len(context.execution_id) == 16
        assert context.runtime == runtime
    
    def test_get_node(self):
        """Test getting node from context"""
        node = {"id": "n1", "type": "test"}
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph={"nodes": [node]},
            node_map={"n1": node},
            runtime=runtime
        )
        
        assert context.get_node("n1") == node
        assert context.get_node("n2") is None
    
    def test_set_get_output(self):
        """Test setting and getting outputs"""
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph={},
            node_map={},
            runtime=runtime
        )
        
        context.set_output("n1", {"result": "test"})
        assert context.get_output("n1") == {"result": "test"}
        assert context.get_output("n2") is None
    
    def test_record_error(self):
        """Test recording errors"""
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph={},
            node_map={},
            runtime=runtime
        )
        
        context.record_error("n1", "Test error")
        assert context.errors["n1"] == "Test error"
    
    def test_add_audit_entry(self):
        """Test adding audit entries"""
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph={},
            node_map={},
            runtime=runtime
        )
        
        context.add_audit_entry({"event": "test"})
        assert len(context.audit_log) == 1
        assert context.audit_log[0]["event"] == "test"
        assert "timestamp" in context.audit_log[0]
        assert context.audit_log[0]["execution_id"] == context.execution_id
    
    def test_create_child_context(self):
        """Test creating child context"""
        runtime = MockRuntime()
        parent = ee.ExecutionContext(
            graph={},
            node_map={},
            runtime=runtime,
            outputs={"p1": "parent_output"}
        )
        
        child_graph = {"nodes": [{"id": "c1"}]}
        child = parent.create_child_context(child_graph)
        
        assert child.parent_context == parent
        assert child.graph == child_graph
        assert child.recursion_depth == parent.recursion_depth + 1
        assert child.runtime == runtime  # Inherits runtime
    
    def test_to_dict(self):
        """Test converting context to dict"""
        runtime = MockRuntime()
        context = ee.ExecutionContext(
            graph={"test": "graph"},
            node_map={"n1": {"id": "n1"}},
            runtime=runtime,
            outputs={"n1": "output"},
            errors={"n2": "error"}
        )
        
        result = context.to_dict()
        assert "graph" in result
        # Check that the context values are accessible
        assert result["graph"] == {"test": "graph"}
        assert context.outputs == {"n1": "output"}
        assert context.errors == {"n2": "error"}


class TestExecutionScheduler:
    """Test ExecutionScheduler class"""
    
    def test_scheduler_creation(self):
        """Test creating scheduler"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"from": "n1", "to": "n2"}]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        assert len(scheduler.node_map) == 2
        assert len(scheduler.edges) == 1
        assert "n1" in scheduler.dependencies["n2"]
        assert "n2" in scheduler.dependents["n1"]
    
    def test_build_dependencies(self):
        """Test dependency building"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n1", "to": "n3"},
                {"from": "n2", "to": "n3"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        assert "n1" in scheduler.dependencies["n2"]
        assert "n1" in scheduler.dependencies["n3"]
        assert "n2" in scheduler.dependencies["n3"]
        assert len(scheduler.dependencies["n1"]) == 0
    
    def test_build_dependents(self):
        """Test dependent building"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"from": "n1", "to": "n2"}]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        assert "n2" in scheduler.dependents["n1"]
        # n2 might not have an entry in dependents if it has no dependents
        assert scheduler.dependents.get("n2", set()) == set()
    
    def test_detect_cycles_no_cycle(self):
        """Test cycle detection with no cycles"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"from": "n1", "to": "n2"}]
        }
        scheduler = ee.ExecutionScheduler(graph)
        assert not scheduler.has_cycles
    
    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with cycles"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n1"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        assert scheduler.has_cycles
    
    def test_get_ready_nodes(self):
        """Test getting ready nodes"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n1", "to": "n3"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        # Initially only n1 is ready (no dependencies)
        ready = scheduler.get_ready_nodes()
        assert ready == ["n1"]
        
        # Mark n1 as executed
        scheduler.mark_executing("n1")
        scheduler.mark_executed("n1")
        
        # Now n2 and n3 should be ready
        ready = scheduler.get_ready_nodes()
        assert set(ready) == {"n2", "n3"}
    
    def test_mark_executed(self):
        """Test marking node as executed"""
        graph = {"nodes": [{"id": "n1"}], "edges": []}
        scheduler = ee.ExecutionScheduler(graph)
        
        scheduler.mark_executing("n1")
        assert "n1" in scheduler.executing
        
        scheduler.mark_executed("n1")
        assert "n1" not in scheduler.executing
        assert "n1" in scheduler.executed
    
    def test_mark_executed_with_failure(self):
        """Test marking node as failed"""
        graph = {"nodes": [{"id": "n1"}], "edges": []}
        scheduler = ee.ExecutionScheduler(graph)
        
        scheduler.mark_executing("n1")
        assert "n1" in scheduler.executing
        
        # Use mark_failed to mark as failed
        scheduler.mark_failed("n1", "Test error")
        assert "n1" not in scheduler.executing
        assert "n1" in scheduler.failed
    
    def test_get_execution_layers(self):
        """Test getting execution layers"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        layers = scheduler.get_execution_layers()
        assert layers == [["n1"], ["n2"], ["n3"]]
    
    def test_get_topological_order(self):
        """Test getting topological order"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        order = scheduler.get_topological_order()
        assert order == ["n1", "n2", "n3"]
    
    def test_get_topological_order_with_cycle(self):
        """Test topological order with cycle returns empty"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n1"}
            ]
        }
        scheduler = ee.ExecutionScheduler(graph)
        
        order = scheduler.get_topological_order()
        assert order == []  # Empty when cycle detected


class TestExecutionEngine:
    """Test ExecutionEngine class"""
    
    @pytest.fixture
    def runtime(self):
        """Create mock runtime"""
        return MockRuntime()
    
    @pytest.fixture
    def engine(self, runtime):
        """Create engine instance with mock runtime"""
        return ee.ExecutionEngine(
            runtime=runtime,
            max_parallel=4,
            timeout_seconds=10
        )
    
    @pytest.fixture
    def simple_graph(self):
        """Simple test graph"""
        return {
            "nodes": [
                {"id": "n1", "type": "test"},
                {"id": "n2", "type": "test"}
            ],
            "edges": [{"from": "n1", "to": "n2"}]
        }
    
    @pytest.mark.asyncio
    async def test_execute_graph_sequential(self, engine, runtime, simple_graph):
        """Test sequential graph execution"""
        context = ee.ExecutionContext(
            graph=simple_graph,
            node_map={n["id"]: n for n in simple_graph["nodes"]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.SEQUENTIAL
        )
        
        assert isinstance(result, ee.GraphExecutionResult)
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.nodes_executed == 2
    
    @pytest.mark.asyncio
    async def test_execute_graph_parallel(self, engine, runtime, simple_graph):
        """Test parallel graph execution"""
        context = ee.ExecutionContext(
            graph=simple_graph,
            node_map={n["id"]: n for n in simple_graph["nodes"]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.PARALLEL
        )
        
        assert isinstance(result, ee.GraphExecutionResult)
        assert result.status == ee.ExecutionStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_execute_graph_with_cycle(self, engine, runtime):
        """Test execution with cyclic graph"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n1"}
            ]
        }
        
        context = ee.ExecutionContext(
            graph=graph,
            node_map={n["id"]: n for n in graph["nodes"]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.SEQUENTIAL
        )
        
        assert result.status == ee.ExecutionStatus.FAILED
        assert "cycle" in result.errors.get("_graph", "").lower()
    
    @pytest.mark.asyncio
    async def test_execute_graph_with_context(self, engine, runtime, simple_graph):
        """Test execution with pre-populated context"""
        context = ee.ExecutionContext(
            graph=simple_graph,
            node_map={n["id"]: n for n in simple_graph["nodes"]},
            runtime=runtime,
            inputs={"test_input": "value"}
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.SEQUENTIAL
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.inputs == {"test_input": "value"}
    
    def test_is_deterministic_node(self, engine):
        """Test deterministic node check"""
        # The current implementation always returns True
        det_node = {"type": "transform", "config": {}}
        assert engine._is_deterministic_node(det_node)
        
        # Even "random" nodes are treated as deterministic in current impl
        non_det_node = {"type": "random", "config": {}}
        assert engine._is_deterministic_node(non_det_node)
    
    def test_get_output_nodes(self, engine, runtime):
        """Test getting output nodes"""
        graph = {
            "nodes": [
                {"id": "n1"},
                {"id": "n2"},
                {"id": "n3"}
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"}
            ]
        }
        
        # _get_output_nodes expects ExecutionContext, not graph dict
        context = ee.ExecutionContext(
            graph=graph,
            node_map={n["id"]: n for n in graph["nodes"]},
            runtime=runtime
        )
        
        outputs = engine._get_output_nodes(context)
        # Since none are marked as OUTPUT type, it returns sink nodes
        assert outputs == ["n3"]
    
    def test_get_metrics(self, engine):
        """Test getting execution metrics"""
        # get_metrics is not async
        metrics = engine.get_metrics()
        
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        # Check for available metrics keys
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ["cache_hits", "cache_misses"])
    
    def test_cleanup(self, engine):
        """Test cleanup"""
        # cleanup is not async, it schedules shutdown
        engine.cleanup()
        assert engine._shutdown_event.is_set()


class TestModuleLevelFunctions:
    """Test module-level functions"""
    
    def test_get_global_engine(self):
        """Test get_global_engine raises appropriate error"""
        with pytest.raises(RuntimeError) as exc_info:
            ee.get_global_engine()
        assert "UnifiedRuntime" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_graph_function(self):
        """Test execute_graph raises appropriate error"""
        graph = {"nodes": [], "edges": []}
        with pytest.raises(NotImplementedError) as exc_info:
            await ee.execute_graph(graph)
        assert "runtime.execute_graph" in str(exc_info.value)


class TestNodeExecutionResult:
    """Test NodeExecutionResult dataclass"""
    
    def test_result_creation(self):
        """Test creating node execution result"""
        result = ee.NodeExecutionResult(
            node_id="n1",
            status=ee.ExecutionStatus.SUCCESS,
            output={"test": "output"},
            error=None,
            duration_ms=100.5
        )
        
        assert result.node_id == "n1"
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.output == {"test": "output"}
        assert result.error is None
        assert result.duration_ms == 100.5


class TestGraphExecutionResult:
    """Test GraphExecutionResult dataclass"""
    
    def test_result_creation(self):
        """Test creating graph execution result"""
        result = ee.GraphExecutionResult(
            status=ee.ExecutionStatus.SUCCESS,
            output={"n1": "output"},
            errors={},
            nodes_executed=5,
            duration_ms=250.0
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.output == {"n1": "output"}
        assert result.nodes_executed == 5
    
    def test_to_dict(self):
        """Test converting result to dict"""
        result = ee.GraphExecutionResult(
            status=ee.ExecutionStatus.SUCCESS,
            output={"n1": "test"},
            errors={"n2": "error"},
            nodes_executed=2
        )
        
        data = result.to_dict()
        assert data["status"] == "success"
        assert data["output"] == {"n1": "test"}
        assert data["errors"] == {"n2": "error"}
        assert data["nodes_executed"] == 2


class TestComplexGraphExecution:
    """Test complex graph execution scenarios"""
    
    @pytest.fixture
    def runtime(self):
        """Create mock runtime"""
        return MockRuntime()
    
    @pytest.fixture
    def engine(self, runtime):
        """Create engine for complex tests"""
        return ee.ExecutionEngine(runtime=runtime)
    
    @pytest.mark.asyncio
    async def test_diamond_graph(self, engine, runtime):
        """Test diamond-shaped dependency graph"""
        graph = {
            "nodes": [
                {"id": "start", "type": "test"},
                {"id": "left", "type": "test"},
                {"id": "right", "type": "test"},
                {"id": "end", "type": "test"}
            ],
            "edges": [
                {"from": "start", "to": "left"},
                {"from": "start", "to": "right"},
                {"from": "left", "to": "end"},
                {"from": "right", "to": "end"}
            ]
        }
        
        context = ee.ExecutionContext(
            graph=graph,
            node_map={n["id"]: n for n in graph["nodes"]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.PARALLEL
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.nodes_executed == 4
    
    @pytest.mark.asyncio
    async def test_parallel_chains(self, engine, runtime):
        """Test multiple parallel chains"""
        graph = {
            "nodes": [
                {"id": "a1", "type": "test"}, {"id": "a2", "type": "test"}, {"id": "a3", "type": "test"},
                {"id": "b1", "type": "test"}, {"id": "b2", "type": "test"}, {"id": "b3", "type": "test"}
            ],
            "edges": [
                {"from": "a1", "to": "a2"},
                {"from": "a2", "to": "a3"},
                {"from": "b1", "to": "b2"},
                {"from": "b2", "to": "b3"}
            ]
        }
        
        context = ee.ExecutionContext(
            graph=graph,
            node_map={n["id"]: n for n in graph["nodes"]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.PARALLEL
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.nodes_executed == 6


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.fixture
    def runtime(self):
        """Create mock runtime"""
        return MockRuntime()
    
    @pytest.fixture
    def engine(self, runtime):
        """Create engine instance"""
        return ee.ExecutionEngine(runtime=runtime)
    
    @pytest.mark.asyncio
    async def test_empty_graph(self, engine, runtime):
        """Test execution with empty graph"""
        graph = {"nodes": [], "edges": []}
        context = ee.ExecutionContext(
            graph=graph,
            node_map={},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.SEQUENTIAL
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.nodes_executed == 0
    
    @pytest.mark.asyncio
    async def test_single_node_graph(self, engine, runtime):
        """Test execution with single node"""
        graph = {
            "nodes": [{"id": "n1", "type": "test"}],
            "edges": []
        }
        
        context = ee.ExecutionContext(
            graph=graph,
            node_map={"n1": graph["nodes"][0]},
            runtime=runtime
        )
        
        result = await engine.run_graph(
            context=context,
            mode=ee.ExecutionMode.SEQUENTIAL
        )
        
        assert result.status == ee.ExecutionStatus.SUCCESS
        assert result.nodes_executed == 1