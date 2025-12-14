"""
Comprehensive test suite for listener.py
"""

import json
import socket
import threading
import time
from datetime import datetime, timedelta
from http.client import HTTPConnection
from io import BytesIO
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from listener import (
    MAX_AGENT_ID_LENGTH,
    MAX_CONTENT_LENGTH,
    MAX_REQUESTS_PER_MINUTE,
    MAX_SIGNATURE_LENGTH,
    MIN_CONTENT_LENGTH,
    RATE_LIMIT_WINDOW,
    GraphixListener,
    MockAgentRegistry,
    MockUnifiedRuntime,
    RateLimiter,
    RequestHandler,
)


@pytest.fixture
def rate_limiter():
    """Create rate limiter."""
    return RateLimiter(max_requests=5, window=60)


@pytest.fixture
def valid_graph():
    """Create valid graph."""
    return {
        "nodes": [
            {"id": "node1", "type": "CONST", "params": {"value": 1.0}},
            {"id": "node2", "type": "ADD", "params": {"value": 2.0}},
        ],
        "edges": [{"from": "node1", "to": "node2", "type": "data"}],
    }


@pytest.fixture
def mock_request_handler():
    """Create a mock request handler for testing validation methods."""
    # Create a mock socket object
    mock_socket = MagicMock()
    mock_socket.makefile = MagicMock(return_value=BytesIO())

    # Create mock server
    mock_server = MagicMock()

    # Create handler with mocked setup
    with (
        patch.object(RequestHandler, "setup"),
        patch.object(RequestHandler, "handle"),
        patch.object(RequestHandler, "finish"),
    ):

        handler = RequestHandler(
            request=mock_socket, client_address=("127.0.0.1", 12345), server=mock_server
        )

        # Set required attributes
        handler.rfile = BytesIO()
        handler.wfile = BytesIO()
        handler.registry = MockAgentRegistry()
        handler.runtime = MockUnifiedRuntime()
        handler.rate_limiter = RateLimiter()

        return handler


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, window=30)

        assert limiter.max_requests == 10
        assert limiter.window == 30

    def test_is_allowed_first_request(self, rate_limiter):
        """Test first request is allowed."""
        assert rate_limiter.is_allowed("client1") is True

    def test_is_allowed_within_limit(self, rate_limiter):
        """Test multiple requests within limit."""
        for i in range(4):
            assert rate_limiter.is_allowed("client1") is True

    def test_is_allowed_exceeds_limit(self, rate_limiter):
        """Test exceeding rate limit."""
        # Use up the limit
        for i in range(5):
            rate_limiter.is_allowed("client1")

        # Next request should be denied
        assert rate_limiter.is_allowed("client1") is False

    def test_is_allowed_different_clients(self, rate_limiter):
        """Test different clients have separate limits."""
        for i in range(5):
            rate_limiter.is_allowed("client1")

        # client2 should still be allowed
        assert rate_limiter.is_allowed("client2") is True

    def test_is_allowed_window_expiry(self):
        """Test window expiry."""
        limiter = RateLimiter(max_requests=2, window=1)  # 1 second window

        # Use up limit
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Should be denied
        assert limiter.is_allowed("client1") is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed("client1") is True

    def test_get_stats(self, rate_limiter):
        """Test getting rate limit stats."""
        rate_limiter.is_allowed("client1")
        rate_limiter.is_allowed("client1")

        stats = rate_limiter.get_stats("client1")

        assert stats["client_id"] == "client1"
        assert stats["requests_in_window"] == 2
        assert stats["max_requests"] == 5
        assert stats["remaining"] == 3

    def test_get_stats_unknown_client(self, rate_limiter):
        """Test stats for unknown client."""
        stats = rate_limiter.get_stats("unknown")

        assert stats["requests_in_window"] == 0
        assert stats["remaining"] == 5


class TestMockImplementations:
    """Test mock implementations."""

    def test_mock_registry_verify(self):
        """Test mock registry verification."""
        registry = MockAgentRegistry()

        result = registry.verify_signature("agent1", "message", "signature")

        assert result is True

    def test_mock_runtime_execute(self):
        """Test mock runtime execution."""
        runtime = MockUnifiedRuntime()

        graph = {"nodes": [{"id": "n1"}], "edges": []}

        result = runtime.execute_graph(graph)

        assert result["status"] == "mock_executed"
        assert result["nodes_processed"] == 1


class TestRequestHandler:
    """Test RequestHandler class."""

    def test_validate_graph_valid(self, mock_request_handler, valid_graph):
        """Test validating valid graph."""
        error = mock_request_handler.validate_graph(valid_graph)

        assert error is None

    def test_validate_graph_not_dict(self, mock_request_handler):
        """Test validating non-dict graph."""
        error = mock_request_handler.validate_graph("not a dict")

        assert error is not None
        assert "must be a JSON object" in error

    def test_validate_graph_missing_nodes(self, mock_request_handler):
        """Test validating graph without nodes."""
        graph = {"edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "Missing 'nodes'" in error

    def test_validate_graph_missing_edges(self, mock_request_handler):
        """Test validating graph without edges."""
        graph = {"nodes": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "Missing 'edges'" in error

    def test_validate_graph_nodes_not_list(self, mock_request_handler):
        """Test validating graph with nodes not a list."""
        graph = {"nodes": "not a list", "edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "'nodes' must be an array" in error

    def test_validate_graph_too_many_nodes(self, mock_request_handler):
        """Test validating graph with too many nodes."""
        graph = {"nodes": [{"id": f"n{i}"} for i in range(100001)], "edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "Too many nodes" in error

    def test_validate_graph_edges_not_list(self, mock_request_handler):
        """Test validating graph with edges not a list."""
        graph = {"nodes": [], "edges": "not a list"}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "'edges' must be an array" in error

    def test_validate_graph_too_many_edges(self, mock_request_handler):
        """Test validating graph with too many edges."""
        graph = {
            "nodes": [{"id": "n1"}],
            "edges": [{"from": "n1", "to": "n1"} for _ in range(1000001)],
        }
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "Too many edges" in error

    def test_validate_graph_node_not_dict(self, mock_request_handler):
        """Test validating graph with node not a dict."""
        graph = {"nodes": ["not a dict"], "edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "is not an object" in error

    def test_validate_graph_node_missing_id(self, mock_request_handler):
        """Test validating graph with node missing id."""
        graph = {"nodes": [{"type": "CONST"}], "edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "missing 'id'" in error

    def test_validate_graph_node_non_string_id(self, mock_request_handler):
        """Test validating graph with node having non-string id."""
        graph = {"nodes": [{"id": 123}], "edges": []}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "non-string 'id'" in error

    def test_validate_graph_edge_not_dict(self, mock_request_handler):
        """Test validating graph with edge not a dict."""
        graph = {"nodes": [{"id": "n1"}], "edges": ["not a dict"]}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "is not an object" in error

    def test_validate_graph_edge_missing_from(self, mock_request_handler):
        """Test validating graph with edge missing from."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"to": "n1"}]}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "missing 'from'" in error

    def test_validate_graph_edge_missing_to(self, mock_request_handler):
        """Test validating graph with edge missing to."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"from": "n1"}]}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "missing 'to'" in error

    def test_validate_graph_edge_non_string_from(self, mock_request_handler):
        """Test validating graph with edge having non-string from."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"from": 123, "to": "n1"}]}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "non-string 'from'" in error

    def test_validate_graph_edge_non_string_to(self, mock_request_handler):
        """Test validating graph with edge having non-string to."""
        graph = {"nodes": [{"id": "n1"}], "edges": [{"from": "n1", "to": 123}]}
        error = mock_request_handler.validate_graph(graph)

        assert error is not None
        assert "non-string 'to'" in error


class TestGraphixListener:
    """Test GraphixListener class."""

    def test_initialization(self):
        """Test listener initialization."""
        listener = GraphixListener(host="127.0.0.1", port=8182, use_mock=True)

        assert listener.host == "127.0.0.1"
        assert listener.port == 8182
        assert listener.registry is not None
        assert listener.runtime is not None

    def test_initialization_with_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        listener = GraphixListener(
            host="127.0.0.1", port=8182, use_mock=True, max_requests_per_minute=100
        )

        assert listener.rate_limiter.max_requests == 100

    def test_stop_without_start(self):
        """Test stopping listener that hasn't started."""
        listener = GraphixListener(use_mock=True)

        # Should not raise
        listener.stop()

    def test_stop_twice(self):
        """Test stopping listener twice."""
        listener = GraphixListener(use_mock=True)

        listener.stop()
        listener.stop()  # Should not raise

    def test_initialization_default_host_port(self):
        """Test initialization with default host and port."""
        listener = GraphixListener(use_mock=True)

        assert listener.host == "127.0.0.1"
        assert listener.port == 8181

    def test_shutdown_event_created(self):
        """Test that shutdown event is created."""
        listener = GraphixListener(use_mock=True)

        assert listener.shutdown_event is not None
        assert not listener.shutdown_event.is_set()

    def test_stop_sets_shutdown_event(self):
        """Test that stop sets the shutdown event."""
        listener = GraphixListener(use_mock=True)

        listener.stop()

        assert listener.shutdown_event.is_set()


class TestConstants:
    """Test module constants."""

    def test_max_content_length(self):
        """Test MAX_CONTENT_LENGTH is reasonable."""
        assert MAX_CONTENT_LENGTH == 10 * 1024 * 1024

    def test_min_content_length(self):
        """Test MIN_CONTENT_LENGTH."""
        assert MIN_CONTENT_LENGTH == 1

    def test_rate_limit_constants(self):
        """Test rate limit constants."""
        assert MAX_REQUESTS_PER_MINUTE > 0
        assert RATE_LIMIT_WINDOW > 0

    def test_max_lengths(self):
        """Test maximum length constants."""
        assert MAX_AGENT_ID_LENGTH > 0
        assert MAX_SIGNATURE_LENGTH > 0

    def test_max_agent_id_length_value(self):
        """Test MAX_AGENT_ID_LENGTH has expected value."""
        assert MAX_AGENT_ID_LENGTH == 256

    def test_max_signature_length_value(self):
        """Test MAX_SIGNATURE_LENGTH has expected value."""
        assert MAX_SIGNATURE_LENGTH == 512


class TestThreadSafety:
    """Test thread safety of components."""

    def test_rate_limiter_thread_safe(self):
        """Test rate limiter is thread safe."""
        limiter = RateLimiter(max_requests=100, window=60)
        results = []

        def make_requests():
            for i in range(10):
                result = limiter.is_allowed("client1")
                results.append(result)

        # Create multiple threads
        threads = [threading.Thread(target=make_requests) for _ in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All should be True (50 requests, limit is 100)
        assert len(results) == 50
        assert all(results)

    def test_rate_limiter_multiple_clients_thread_safe(self):
        """Test rate limiter with multiple clients is thread safe."""
        limiter = RateLimiter(max_requests=50, window=60)
        results = {"client1": [], "client2": [], "client3": []}

        def make_requests(client_id):
            for i in range(10):
                result = limiter.is_allowed(client_id)
                results[client_id].append(result)

        # Create threads for different clients
        threads = [
            threading.Thread(target=make_requests, args=(cid,))
            for cid in ["client1", "client2", "client3"]
        ]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Each client should have all True (10 requests each, limit is 50)
        for client_id, client_results in results.items():
            assert len(client_results) == 10
            assert all(client_results), f"{client_id} had rejected requests"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_validate_graph_empty_nodes_and_edges(self, mock_request_handler):
        """Test validating graph with empty nodes and edges."""
        graph = {"nodes": [], "edges": []}
        error = mock_request_handler.validate_graph(graph)

        # Empty graph should be valid
        assert error is None

    def test_validate_graph_with_metadata(self, mock_request_handler):
        """Test validating graph with additional metadata."""
        graph = {"nodes": [{"id": "n1"}], "edges": [], "metadata": {"version": "1.0"}}
        error = mock_request_handler.validate_graph(graph)

        # Should be valid - extra fields are ok
        assert error is None

    def test_rate_limiter_zero_window(self):
        """Test rate limiter with very small window."""
        limiter = RateLimiter(max_requests=5, window=0.1)

        # Should still work
        assert limiter.is_allowed("client1") is True

    def test_rate_limiter_single_request_limit(self):
        """Test rate limiter with single request limit."""
        limiter = RateLimiter(max_requests=1, window=60)

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
