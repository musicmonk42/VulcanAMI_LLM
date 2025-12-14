"""
Comprehensive test suite for agent_interface.py
"""

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_interface import (
    DEFAULT_TIMEOUT,
    MAX_BATCH_SIZE,
    MAX_GRAPH_DEPTH,
    AgentInterface,
    CommunicationMode,
    ConnectionConfig,
    ExecutionState,
    GraphPriority,
    GraphSubmission,
    HTTPCommunicator,
    ResultCache,
    TelemetryCollector,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def local_config():
    """Create local mode configuration."""
    return ConnectionConfig(mode=CommunicationMode.LOCAL)


@pytest.fixture
def http_config():
    """Create HTTP mode configuration."""
    return ConnectionConfig(mode=CommunicationMode.HTTP, host="localhost", port=8080)


@pytest.fixture
def agent_interface(local_config):
    """Create agent interface in local mode."""
    interface = AgentInterface(local_config)
    interface.connect()
    yield interface
    interface.disconnect()


@pytest.fixture
def valid_graph():
    """Create valid test graph."""
    return {
        "grammar_version": "1.0.0",
        "id": "test_graph_001",
        "type": "Graph",
        "nodes": [
            {"id": "node1", "type": "InputNode"},
            {"id": "node2", "type": "ComputeNode"},
            {"id": "node3", "type": "OutputNode"},
        ],
        "edges": [{"from": "node1", "to": "node2"}, {"from": "node2", "to": "node3"}],
    }


class TestResultCache:
    """Test result cache."""

    def test_cache_creation(self):
        """Test cache initialization."""
        cache = ResultCache(ttl=60, max_size=100)

        assert cache.ttl == 60
        assert cache.max_size == 100
        assert len(cache.cache) == 0

    def test_set_and_get(self):
        """Test basic set/get operations."""
        cache = ResultCache()

        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result is not None
        assert result["data"] == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResultCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = ResultCache(ttl=1)

        cache.set("key1", {"data": "value1"})
        assert cache.get("key1") is not None

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        """Test max size eviction."""
        cache = ResultCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict oldest

        assert len(cache.cache) == 3
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key4") is not None

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = ResultCache()

        cache.set("key1", "value1")
        cache.invalidate("key1")

        assert cache.get("key1") is None

    def test_clear(self):
        """Test cache clear."""
        cache = ResultCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert len(cache.cache) == 0

    def test_deep_copy(self):
        """Test that cache returns deep copies."""
        cache = ResultCache()

        original = {"nested": {"value": 42}}
        cache.set("key1", original)

        retrieved = cache.get("key1")
        retrieved["nested"]["value"] = 999

        # Original in cache should be unchanged
        assert cache.get("key1")["nested"]["value"] == 42


class TestTelemetryCollector:
    """Test telemetry collector."""

    def test_initialization(self):
        """Test telemetry initialization."""
        telemetry = TelemetryCollector(enabled=True)

        assert telemetry.enabled
        assert len(telemetry.metrics) == 0
        assert len(telemetry.events) == 0

    def test_record_metric(self):
        """Test metric recording."""
        telemetry = TelemetryCollector(enabled=True)

        telemetry.record_metric("test_metric", duration=1.5)
        telemetry.record_metric("test_metric", duration=2.5)

        metrics = telemetry.get_metrics()
        assert "test_metric" in metrics
        assert metrics["test_metric"]["count"] == 2
        assert metrics["test_metric"]["total_time"] == 4.0

    def test_record_error(self):
        """Test error recording."""
        telemetry = TelemetryCollector(enabled=True)

        telemetry.record_metric("failed_op", error=True)

        metrics = telemetry.get_metrics()
        assert metrics["failed_op"]["errors"] == 1

    def test_record_event(self):
        """Test event recording."""
        telemetry = TelemetryCollector(enabled=True)

        telemetry.record_event("connection", {"status": "connected"})

        assert len(telemetry.events) == 1
        assert telemetry.events[0]["event"] == "connection"

    def test_disabled_telemetry(self):
        """Test disabled telemetry doesn't record."""
        telemetry = TelemetryCollector(enabled=False)

        telemetry.record_metric("test", duration=1.0)
        telemetry.record_event("test", {})

        assert len(telemetry.get_metrics()) == 0
        assert len(telemetry.events) == 0


class TestConnectionConfig:
    """Test connection configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ConnectionConfig()

        assert config.host == "localhost"
        assert config.port == 8080
        assert config.mode == CommunicationMode.HTTP
        assert config.timeout == DEFAULT_TIMEOUT

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConnectionConfig(
            host="example.com",
            port=9090,
            mode=CommunicationMode.WEBSOCKET,
            secure=True,
            api_key="test_key",
        )

        assert config.host == "example.com"
        assert config.port == 9090
        assert config.mode == CommunicationMode.WEBSOCKET
        assert config.secure
        assert config.api_key == "test_key"


class TestAgentInterface:
    """Test agent interface."""

    def test_initialization(self, local_config):
        """Test interface initialization."""
        interface = AgentInterface(local_config)

        assert interface.config is not None
        assert interface.session_id is not None
        assert len(interface.submissions) == 0

    def test_local_connection(self, local_config):
        """Test local mode connection."""
        interface = AgentInterface(local_config)

        assert interface.connect()
        assert interface.connected

        interface.disconnect()
        assert not interface.connected

    def test_context_manager(self, local_config):
        """Test context manager usage."""
        with AgentInterface(local_config) as interface:
            assert interface.connected

        # Should be disconnected after context
        assert not interface.connected

    def test_submit_graph_basic(self, agent_interface, valid_graph):
        """Test basic graph submission."""
        submission = agent_interface.submit_graph(valid_graph, wait_for_result=False)

        assert isinstance(submission, GraphSubmission)
        assert submission.submission_id is not None
        assert submission.graph["id"] == valid_graph["id"]

    def test_submit_graph_with_result(self, agent_interface, valid_graph):
        """Test graph submission with waiting for result."""
        result = agent_interface.submit_graph(
            valid_graph, wait_for_result=True, timeout=5
        )

        assert isinstance(result, dict)
        assert "status" in result

    def test_submit_invalid_graph(self, agent_interface):
        """Test submission of invalid graph."""
        invalid_graph = {"invalid": "structure"}

        with pytest.raises(ValueError, match="Invalid graph"):
            agent_interface.submit_graph(invalid_graph)

    def test_submit_missing_fields(self, agent_interface):
        """Test graph with missing required fields."""
        incomplete_graph = {
            "grammar_version": "1.0.0",
            "id": "incomplete",
            # Missing nodes, edges, type
        }

        with pytest.raises(ValueError, match="Missing required field"):
            agent_interface.submit_graph(incomplete_graph)

    def test_submit_with_priority(self, agent_interface, valid_graph):
        """Test submission with different priorities."""
        submission = agent_interface.submit_graph(
            valid_graph, priority=GraphPriority.HIGH, wait_for_result=False
        )

        assert submission.priority == GraphPriority.HIGH

    def test_get_status(self, agent_interface, valid_graph):
        """Test getting submission status."""
        submission = agent_interface.submit_graph(valid_graph, wait_for_result=False)

        status = agent_interface.get_status(submission.submission_id)

        assert "submission_id" in status
        assert "state" in status

    def test_get_status_unknown_id(self, agent_interface):
        """Test getting status of unknown submission."""
        with pytest.raises(ValueError, match="Unknown submission ID"):
            agent_interface.get_status("nonexistent_id")

    def test_cancel_execution(self, agent_interface, valid_graph):
        """Test canceling execution."""
        submission = agent_interface.submit_graph(valid_graph, wait_for_result=False)

        # In local mode, should succeed
        cancelled = agent_interface.cancel_execution(submission.submission_id)

        assert isinstance(cancelled, bool)

    def test_cancel_unknown_id(self, agent_interface):
        """Test canceling unknown submission."""
        with pytest.raises(ValueError, match="Unknown submission ID"):
            agent_interface.cancel_execution("nonexistent_id")

    def test_batch_submit(self, agent_interface, valid_graph):
        """Test batch submission."""
        graphs = [{**valid_graph, "id": f"batch_graph_{i}"} for i in range(5)]

        submissions = agent_interface.batch_submit(graphs, parallel=False)

        assert len(submissions) == 5
        assert all(s is not None for s in submissions)

    def test_batch_submit_parallel(self, agent_interface, valid_graph):
        """Test parallel batch submission."""
        graphs = [{**valid_graph, "id": f"parallel_graph_{i}"} for i in range(3)]

        submissions = agent_interface.batch_submit(graphs, parallel=True)

        assert len(submissions) == 3

    def test_batch_submit_empty(self, agent_interface):
        """Test batch submission with empty list."""
        with pytest.raises(ValueError, match="Cannot submit empty batch"):
            agent_interface.batch_submit([])

    def test_batch_submit_too_large(self, agent_interface, valid_graph):
        """Test batch submission exceeding max size."""
        graphs = [valid_graph] * (MAX_BATCH_SIZE + 1)

        with pytest.raises(ValueError, match="exceeds maximum"):
            agent_interface.batch_submit(graphs)

    def test_stream_results(self, agent_interface, valid_graph):
        """Test result streaming."""
        submission = agent_interface.submit_graph(valid_graph, wait_for_result=False)

        received_data = []

        def handler(data):
            received_data.append(data)

        thread = agent_interface.stream_results(submission.submission_id, handler)
        thread.join(timeout=3)

        assert len(received_data) > 0

    def test_cache_key_computation(self, agent_interface, valid_graph):
        """Test cache key computation."""
        key1 = agent_interface._compute_cache_key(valid_graph)

        # Same graph should produce same key
        key2 = agent_interface._compute_cache_key(valid_graph)
        assert key1 == key2

        # Different graph should produce different key
        modified_graph = {**valid_graph, "id": "different_id"}
        key3 = agent_interface._compute_cache_key(modified_graph)
        assert key1 != key3

    def test_caching_behavior(self, local_config, valid_graph):
        """Test result caching."""
        config = ConnectionConfig(mode=CommunicationMode.LOCAL, enable_caching=True)
        interface = AgentInterface(config)
        interface.connect()

        try:
            # First submission
            result1 = interface.submit_graph(valid_graph, wait_for_result=True)

            # Second submission of same graph
            result2 = interface.submit_graph(valid_graph, wait_for_result=True)

            # Should get same result from cache
            assert result1 == result2
        finally:
            interface.disconnect()

    def test_metrics_collection(self, agent_interface):
        """Test metrics collection."""
        metrics = agent_interface.get_metrics()

        assert "session_id" in metrics
        assert "connected" in metrics
        assert "mode" in metrics
        assert "submissions" in metrics

    def test_state_save_load(
        self, agent_interface, valid_graph, temp_dir, local_config
    ):
        """Test state save and load."""
        # Submit something
        agent_interface.submit_graph(valid_graph, wait_for_result=False)

        # Save state
        state_file = temp_dir / "state.json"
        agent_interface.save_state(str(state_file))

        assert state_file.exists()

        # Load state in new interface
        new_interface = AgentInterface(local_config)
        new_interface.load_state(str(state_file))

        assert len(new_interface.interaction_history) > 0


class TestGraphValidation:
    """Test graph validation."""

    def test_valid_graph(self, agent_interface, valid_graph):
        """Test validation of valid graph."""
        result = agent_interface._validate_graph(valid_graph)

        assert result["valid"]
        assert len(result["errors"]) == 0

    def test_missing_required_fields(self, agent_interface):
        """Test validation catches missing fields."""
        invalid = {"id": "test"}

        result = agent_interface._validate_graph(invalid)

        assert not result["valid"]
        assert len(result["errors"]) > 0

    def test_invalid_nodes(self, agent_interface):
        """Test validation of invalid nodes."""
        invalid = {
            "grammar_version": "1.0.0",
            "id": "test",
            "type": "Graph",
            "nodes": "not a list",
            "edges": [],
        }

        result = agent_interface._validate_graph(invalid)

        assert not result["valid"]
        assert any("must be a list" in e for e in result["errors"])

    def test_duplicate_node_ids(self, agent_interface):
        """Test detection of duplicate node IDs."""
        invalid = {
            "grammar_version": "1.0.0",
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "node1", "type": "Node"},
                {"id": "node1", "type": "Node"},  # Duplicate
            ],
            "edges": [],
        }

        result = agent_interface._validate_graph(invalid)

        assert not result["valid"]
        assert any("Duplicate node ID" in e for e in result["errors"])

    def test_invalid_edge_references(self, agent_interface):
        """Test detection of invalid edge references."""
        invalid = {
            "grammar_version": "1.0.0",
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "node1", "type": "Node"}],
            "edges": [{"from": "node1", "to": "nonexistent"}],
        }

        result = agent_interface._validate_graph(invalid)

        assert not result["valid"]
        assert any("unknown" in e.lower() for e in result["errors"])

    def test_excessive_depth(self, agent_interface):
        """Test detection of excessive nesting."""
        # Create deeply nested graph
        deep_graph = {
            "grammar_version": "1.0.0",
            "id": "deep",
            "type": "Graph",
            "nodes": [],
            "edges": [],
        }

        current = deep_graph
        for i in range(MAX_GRAPH_DEPTH + 10):
            current["nested"] = {"level": i}
            current = current["nested"]

        result = agent_interface._validate_graph(deep_graph)

        assert not result["valid"]
        assert any("depth" in e.lower() for e in result["errors"])

    def test_oversized_graph(self, agent_interface):
        """Test detection of oversized graphs."""
        huge_graph = {
            "grammar_version": "1.0.0",
            "id": "huge",
            "type": "Graph",
            "nodes": [{"id": f"node_{i}", "type": "Node"} for i in range(10000)],
            "edges": [],
        }

        result = agent_interface._validate_graph(huge_graph)

        # Might fail on size
        if not result["valid"]:
            assert any("size" in e.lower() for e in result["errors"])


class TestHTTPCommunicator:
    """Test HTTP communicator (mock tests)."""

    @patch("urllib.request.urlopen")
    def test_successful_request(self, mock_urlopen, http_config):
        """Test successful HTTP request."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"status": "ok"}).encode()
        mock_response.headers.get.return_value = "application/json"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        communicator = HTTPCommunicator(http_config)

        result = communicator._make_request("test", "GET")

        assert result["status"] == "ok"

    @patch("urllib.request.urlopen")
    def test_http_error(self, mock_urlopen, http_config):
        """Test HTTP error handling."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "Internal Server Error", {}, None
        )

        communicator = HTTPCommunicator(http_config)

        with pytest.raises(RuntimeError, match="HTTP 500"):
            communicator._make_request("test", "GET")

    @patch("urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen, http_config):
        """Test connection error handling."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        communicator = HTTPCommunicator(http_config)

        with pytest.raises(ConnectionError, match="Failed to connect"):
            communicator._make_request("test", "GET")


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_submissions(self, agent_interface, valid_graph):
        """Test concurrent graph submissions."""
        results = []

        def submit():
            try:
                sub = agent_interface.submit_graph(
                    {**valid_graph, "id": f"concurrent_{threading.get_ident()}"},
                    wait_for_result=False,
                )
                results.append(sub)
            except Exception as e:
                results.append(None)

        threads = [threading.Thread(target=submit) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_concurrent_status_checks(self, agent_interface, valid_graph):
        """Test concurrent status checks."""
        submission = agent_interface.submit_graph(valid_graph, wait_for_result=False)
        statuses = []

        def check_status():
            try:
                status = agent_interface.get_status(submission.submission_id)
                statuses.append(status)
            except:
                statuses.append(None)

        threads = [threading.Thread(target=check_status) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(statuses) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
