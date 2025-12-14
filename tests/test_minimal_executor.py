"""
Comprehensive test suite for minimal_executor.py
"""

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

from minimal_executor import (
    DEFAULT_GRAPH_TIMEOUT,
    DEFAULT_NODE_TIMEOUT,
    MAX_EDGE_COUNT,
    MAX_GRAPH_SIZE,
    AuditLogger,
    CycleDetectedError,
    ExecutionError,
    GraphValidator,
    MinimalExecutor,
    ThreadSafeContext,
    TimeoutError,
    ValidationError,
)


@pytest.fixture
def mock_observability():
    """Mock the ObservabilityManager."""
    with patch("minimal_executor.ObservabilityManager") as mock_obs:
        mock_instance = MagicMock()
        mock_instance.log_graph_execution = MagicMock()
        mock_instance.log_node_execution = MagicMock()
        mock_obs.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def executor(tmp_path, mock_observability):
    """Create executor with temp audit log and mocked observability."""
    log_path = tmp_path / "test_audit.jsonl"
    return MinimalExecutor(audit_log_path=str(log_path))


@pytest.fixture
def simple_graph():
    """Create simple valid graph."""
    return {
        "id": "test_graph",
        "nodes": [
            {"id": "in1", "type": "InputNode", "value": "Hello"},
            {"id": "out1", "type": "OutputNode", "in": "in1"},
        ],
        "edges": [{"from": "in1", "to": "out1"}],
    }


class TestThreadSafeContext:
    """Test ThreadSafeContext class."""

    def test_initialization(self):
        """Test context initialization."""
        context = ThreadSafeContext()

        assert context._data == {}

    def test_get_set(self):
        """Test getting and setting values."""
        context = ThreadSafeContext()

        context.set("key1", "value1")

        assert context.get("key1") == "value1"

    def test_get_default(self):
        """Test getting with default."""
        context = ThreadSafeContext()

        result = context.get("nonexistent", "default")

        assert result == "default"

    def test_update(self):
        """Test updating multiple values."""
        context = ThreadSafeContext()

        context.update({"key1": "val1", "key2": "val2"})

        assert context.get("key1") == "val1"
        assert context.get("key2") == "val2"

    def test_to_dict(self):
        """Test converting to dict."""
        context = ThreadSafeContext()
        context.set("key1", "val1")

        result = context.to_dict()

        assert result == {"key1": "val1"}
        assert result is not context._data  # Should be copy

    def test_getitem_setitem(self):
        """Test dict-like access."""
        context = ThreadSafeContext()

        context["key1"] = "val1"

        assert context["key1"] == "val1"


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_initialization(self, tmp_path):
        """Test audit logger initialization."""
        log_path = tmp_path / "audit.jsonl"

        logger = AuditLogger(str(log_path))

        assert logger.log_path.exists()

    def test_log_event(self, tmp_path):
        """Test logging an event."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log("test_event", {"detail": "test"})

        # Check file was written
        assert log_path.exists()
        assert log_path.stat().st_size > 0


class TestGraphValidator:
    """Test GraphValidator class."""

    def test_validate_valid_graph(self, simple_graph):
        """Test validating valid graph."""
        is_valid, error = GraphValidator.validate_graph(simple_graph)

        assert is_valid is True
        assert error is None

    def test_validate_not_dict(self):
        """Test validating non-dict."""
        is_valid, error = GraphValidator.validate_graph("not a dict")

        assert is_valid is False
        assert "must be a dictionary" in error

    def test_validate_missing_nodes(self):
        """Test validating graph without nodes."""
        graph = {"edges": []}

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "missing 'nodes'" in error.lower()

    def test_validate_missing_edges(self):
        """Test validating graph without edges."""
        graph = {"nodes": []}

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "missing 'edges'" in error.lower()

    def test_validate_nodes_not_list(self):
        """Test nodes not a list."""
        graph = {"nodes": "not a list", "edges": []}

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "'nodes' must be a list" in error

    def test_validate_too_many_nodes(self):
        """Test graph with too many nodes."""
        graph = {
            "nodes": [
                {"id": f"n{i}", "type": "InputNode"} for i in range(MAX_GRAPH_SIZE + 1)
            ],
            "edges": [],
        }

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "Too many nodes" in error

    def test_validate_duplicate_node_ids(self):
        """Test duplicate node IDs."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "InputNode"},
                {"id": "n1", "type": "OutputNode"},
            ],
            "edges": [],
        }

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "Duplicate" in error

    def test_validate_edge_references_nonexistent_node(self):
        """Test edge referencing non-existent node."""
        graph = {
            "nodes": [{"id": "n1", "type": "InputNode"}],
            "edges": [{"from": "n1", "to": "nonexistent"}],
        }

        is_valid, error = GraphValidator.validate_graph(graph)

        assert is_valid is False
        assert "non-existent" in error

    def test_detect_cycles_no_cycle(self):
        """Test cycle detection with no cycle."""
        nodes = {
            "n1": {"id": "n1", "type": "InputNode"},
            "n2": {"id": "n2", "type": "OutputNode"},
        }
        edges = [{"from": "n1", "to": "n2"}]

        has_cycle, cycle = GraphValidator.detect_cycles(nodes, edges)

        assert has_cycle is False
        assert cycle is None

    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with cycle."""
        nodes = {
            "n1": {"id": "n1", "type": "InputNode"},
            "n2": {"id": "n2", "type": "GenerativeNode"},
            "n3": {"id": "n3", "type": "OutputNode"},
        }
        edges = [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"},
            {"from": "n3", "to": "n1"},
        ]

        has_cycle, cycle = GraphValidator.detect_cycles(nodes, edges)

        assert has_cycle is True
        assert cycle is not None
        assert len(cycle) > 0


class TestMinimalExecutor:
    """Test MinimalExecutor class."""

    def test_initialization(self, tmp_path, mock_observability):
        """Test executor initialization."""
        log_path = tmp_path / "audit.jsonl"

        executor = MinimalExecutor(
            audit_log_path=str(log_path), node_timeout=10.0, graph_timeout=60.0
        )

        assert executor.node_timeout == 10.0
        assert executor.graph_timeout == 60.0
        assert executor.audit_logger is not None

    @pytest.mark.asyncio
    async def test_execute_input_node(self, executor):
        """Test executing InputNode."""
        node = {"id": "in1", "type": "InputNode", "value": "test data"}
        context = ThreadSafeContext()

        result = await executor._execute_input_node(node, context)

        assert result == "test data"
        assert context["in1"] == "test data"

    @pytest.mark.asyncio
    async def test_execute_output_node(self, executor):
        """Test executing OutputNode."""
        node = {"id": "out1", "type": "OutputNode", "in": "in1"}
        context = ThreadSafeContext()
        context["in1"] = "input data"

        result = await executor._execute_output_node(node, context)

        assert result == "input data"
        assert context["_output"] == "input data"

    @pytest.mark.asyncio
    async def test_execute_generative_node(self, executor):
        """Test executing GenerativeNode."""
        node = {
            "id": "gen1",
            "type": "GenerativeNode",
            "prompt": "Summarize:",
            "in": "in1",
        }
        context = ThreadSafeContext()
        context["in1"] = "test input"

        result = await executor._execute_generative_node(node, context)

        assert "Generated" in result
        assert context["gen1"] == result

    @pytest.mark.asyncio
    async def test_execute_combine_node(self, executor):
        """Test executing CombineNode."""
        node = {"id": "combine1", "type": "CombineNode", "in": ["in1", "in2"]}
        context = ThreadSafeContext()
        context["in1"] = "Hello"
        context["in2"] = "World"

        result = await executor._execute_combine_node(node, context)

        assert "Hello" in result
        assert "World" in result

    @pytest.mark.asyncio
    async def test_execute_combine_node_missing_input(self, executor):
        """Test CombineNode with missing input."""
        node = {"id": "combine1", "type": "CombineNode", "in": ["in1", "nonexistent"]}
        context = ThreadSafeContext()
        context["in1"] = "Hello"

        with pytest.raises(ValidationError):
            await executor._execute_combine_node(node, context)

    @pytest.mark.asyncio
    async def test_execute_transform_node(self, executor):
        """Test executing TransformNode."""
        node = {
            "id": "trans1",
            "type": "TransformNode",
            "in": "in1",
            "transform": "uppercase",
        }
        context = ThreadSafeContext()
        context["in1"] = "hello"

        result = await executor._execute_transform_node(node, context)

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_execute_graph_simple(self, executor, simple_graph):
        """Test executing simple graph."""
        result = await executor.execute_graph(simple_graph)

        assert result["status"] == "completed"
        assert "Hello" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_graph_invalid(self, executor):
        """Test executing invalid graph."""
        invalid_graph = {"invalid": "structure"}

        with pytest.raises(ValidationError):
            await executor.execute_graph(invalid_graph)

    @pytest.mark.asyncio
    async def test_execute_graph_with_cycle(self, executor):
        """Test executing graph with cycle."""
        cycle_graph = {
            "id": "cycle_graph",
            "nodes": [
                {"id": "n1", "type": "InputNode", "value": "A"},
                {"id": "n2", "type": "GenerativeNode", "prompt": "B", "in": "n1"},
                {"id": "n3", "type": "GenerativeNode", "prompt": "C", "in": "n2"},
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n3"},
                {"from": "n3", "to": "n1"},
            ],
        }

        with pytest.raises(CycleDetectedError):
            await executor.execute_graph(cycle_graph)

    @pytest.mark.asyncio
    async def test_execute_graph_parallel(self, executor):
        """Test parallel execution of independent nodes."""
        parallel_graph = {
            "id": "parallel_graph",
            "nodes": [
                {"id": "in1", "type": "InputNode", "value": "A"},
                {"id": "in2", "type": "InputNode", "value": "B"},
                {
                    "id": "gen1",
                    "type": "GenerativeNode",
                    "prompt": "Process",
                    "in": "in1",
                },
                {
                    "id": "gen2",
                    "type": "GenerativeNode",
                    "prompt": "Process",
                    "in": "in2",
                },
                {"id": "combine", "type": "CombineNode", "in": ["gen1", "gen2"]},
                {"id": "out", "type": "OutputNode", "in": "combine"},
            ],
            "edges": [
                {"from": "in1", "to": "gen1"},
                {"from": "in2", "to": "gen2"},
                {"from": "gen1", "to": "combine"},
                {"from": "gen2", "to": "combine"},
                {"from": "combine", "to": "out"},
            ],
        }

        result = await executor.execute_graph(parallel_graph)

        assert result["status"] == "completed"


class TestExceptions:
    """Test custom exceptions."""

    def test_execution_error(self):
        """Test ExecutionError."""
        error = ExecutionError("Test error")

        assert str(error) == "Test error"

    def test_cycle_detected_error(self):
        """Test CycleDetectedError."""
        error = CycleDetectedError("Cycle: n1 -> n2 -> n1")

        assert "Cycle" in str(error)

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Operation timed out")

        assert "timed out" in str(error)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid graph")

        assert "Invalid" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
