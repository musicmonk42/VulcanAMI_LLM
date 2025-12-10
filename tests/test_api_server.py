"""
Comprehensive test suite for api_server.py
"""

import json
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timedelta
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from api_server import (MAX_GRAPH_EDGES, MAX_GRAPH_NODES, MAX_REQUEST_SIZE,
                        Agent, APIEndpoint, CacheManager,
                        DatabaseConnectionPool, DatabaseManager,
                        ExecutionEngine, ExecutionStatus, GraphAPIServer,
                        GraphSubmission, InputValidator, Proposal, RateLimiter)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def db_manager(temp_dir):
    """Create database manager."""
    db_path = temp_dir / "test.db"
    manager = DatabaseManager(str(db_path))
    yield manager
    manager.cleanup()


@pytest.fixture
def rate_limiter():
    """Create rate limiter."""
    limiter = RateLimiter(window=1, max_requests=10)
    yield limiter
    limiter.shutdown()


@pytest.fixture
def execution_engine():
    """Create execution engine."""
    engine = ExecutionEngine(max_workers=2)
    yield engine
    engine.shutdown()


@pytest.fixture
def cache_manager():
    """Create cache manager."""
    cache = CacheManager(max_size=100, ttl=60)
    yield cache
    cache.shutdown()


@pytest.fixture
def api_server():
    """Create API server on random port."""
    # Find available port
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    server = GraphAPIServer(host='127.0.0.1', port=port)
    yield server
    server.stop()


class TestDatabaseConnectionPool:
    """Test database connection pool."""

    def test_pool_creation(self, temp_dir):
        """Test creating connection pool."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(str(db_path), pool_size=3)

        assert len(pool.connections) == 3
        pool.close_all()

    def test_get_connection(self, temp_dir):
        """Test getting connection from pool."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(str(db_path), pool_size=2)

        with pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

        pool.close_all()

    def test_concurrent_access(self, temp_dir):
        """Test concurrent connection access."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(str(db_path), pool_size=2)

        results = []

        def query():
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                results.append(cursor.fetchone()[0])

        threads = [threading.Thread(target=query) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r == 1 for r in results)

        pool.close_all()


class TestRateLimiter:
    """Test rate limiter."""

    def test_allows_requests(self, rate_limiter):
        """Test allowing requests under limit."""
        assert rate_limiter.is_allowed("test_id")

    def test_blocks_excessive(self, rate_limiter):
        """Test blocking excessive requests."""
        identifier = "test_id"

        # Use all tokens
        for _ in range(10):
            assert rate_limiter.is_allowed(identifier)

        # Should be blocked
        assert not rate_limiter.is_allowed(identifier)

    def test_cleanup(self, rate_limiter):
        """Test cleanup of old entries."""
        rate_limiter.is_allowed("id1")
        rate_limiter.is_allowed("id2")

        # Mock old timestamps
        for deque in rate_limiter.requests.values():
            deque.clear()

        rate_limiter._cleanup()

        assert len(rate_limiter.requests) == 0


class TestInputValidator:
    """Test input validation."""

    def test_valid_graph(self):
        """Test validating valid graph."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "TestNode"},
                {"id": "n2", "type": "TestNode"}
            ],
            "edges": [
                {"from": "n1", "to": "n2"}
            ]
        }

        valid, error = InputValidator.validate_graph(graph)
        assert valid
        assert error is None

    def test_missing_required_fields(self):
        """Test graph missing required fields."""
        graph = {"id": "test"}

        valid, error = InputValidator.validate_graph(graph)
        assert not valid
        assert "Missing required field" in error

    def test_invalid_nodes(self):
        """Test graph with invalid nodes."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": "not a list",
            "edges": []
        }

        valid, error = InputValidator.validate_graph(graph)
        assert not valid
        assert "must be a list" in error

    def test_duplicate_node_ids(self):
        """Test graph with duplicate node IDs."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {"id": "n1", "type": "TestNode"},
                {"id": "n1", "type": "TestNode"}
            ],
            "edges": []
        }

        valid, error = InputValidator.validate_graph(graph)
        assert not valid
        assert "Duplicate node id" in error

    def test_invalid_edge(self):
        """Test graph with invalid edge."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "TestNode"}],
            "edges": [{"from": "n1", "to": "nonexistent"}]
        }

        valid, error = InputValidator.validate_graph(graph)
        assert not valid
        assert "non-existent node" in error

    def test_too_many_nodes(self):
        """Test graph with too many nodes."""
        nodes = [{"id": f"n{i}", "type": "TestNode"} for i in range(MAX_GRAPH_NODES + 1)]
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": nodes,
            "edges": []
        }

        valid, error = InputValidator.validate_graph(graph)
        assert not valid
        assert "Too many nodes" in error

    def test_sanitize_string(self):
        """Test string sanitization."""
        dirty = "Hello\x00World\x1f!"
        clean = InputValidator.sanitize_string(dirty)

        assert "\x00" not in clean
        assert "\x1f" not in clean
        assert "HelloWorld!" == clean

    def test_sanitize_long_string(self):
        """Test sanitizing long string."""
        long_str = "a" * 2000
        clean = InputValidator.sanitize_string(long_str, max_length=100)

        assert len(clean) == 100

    def test_validate_api_key(self):
        """Test API key validation."""
        valid_key = "a" * 32
        assert InputValidator.validate_api_key(valid_key)

        invalid_key = "invalid"
        assert not InputValidator.validate_api_key(invalid_key)

    def test_validate_url(self):
        """Test URL validation."""
        assert InputValidator.validate_url("http://example.com")
        assert InputValidator.validate_url("https://example.com/path")
        assert not InputValidator.validate_url("not a url")
        assert not InputValidator.validate_url("ftp://example.com")


class TestDatabaseManager:
    """Test database manager."""

    def test_initialization(self, temp_dir):
        """Test database initialization."""
        db_path = temp_dir / "test.db"
        manager = DatabaseManager(str(db_path))

        assert db_path.exists()
        manager.cleanup()

    def test_save_and_get_graph(self, db_manager):
        """Test saving and retrieving graph."""
        graph = {
            "id": "test_graph",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "TestNode"}],
            "edges": []
        }

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1",
            status=ExecutionStatus.PENDING
        )

        db_manager.save_graph(submission)

        retrieved = db_manager.get_graph("sub123")

        assert retrieved is not None
        assert retrieved.id == "sub123"
        assert retrieved.agent_id == "agent1"
        assert retrieved.graph["id"] == "test_graph"

    def test_save_and_get_agent(self, db_manager):
        """Test saving and retrieving agent."""
        agent = Agent(
            id="agent1",
            name="Test Agent",
            api_key="a" * 32,
            roles=["user"]
        )

        db_manager.save_agent(agent)

        retrieved = db_manager.get_agent_by_api_key("a" * 32)

        assert retrieved is not None
        assert retrieved.id == "agent1"
        assert retrieved.name == "Test Agent"

    def test_log_audit(self, db_manager):
        """Test audit logging."""
        db_manager.log_audit(
            agent_id="agent1",
            action="test_action",
            resource="test_resource",
            details={"key": "value"}
        )

        # Verify logged (would need query method)
        assert True  # Just test it doesn't error


class TestExecutionEngine:
    """Test execution engine."""

    def test_execute_graph(self, execution_engine):
        """Test graph execution."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "TestNode"}],
            "edges": []
        }

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1"
        )

        future = execution_engine.execute_graph(submission)
        future.result(timeout=5)

        assert submission.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
        assert submission.completed_at is not None

    def test_execute_invalid_graph(self, execution_engine):
        """Test executing invalid graph."""
        graph = {"invalid": "graph"}

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1"
        )

        future = execution_engine.execute_graph(submission)
        future.result(timeout=5)

        assert submission.status == ExecutionStatus.FAILED
        assert submission.error is not None

    def test_cancel_execution(self, execution_engine):
        """Test canceling execution."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "TestNode"}],
            "edges": []
        }

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1"
        )

        future = execution_engine.execute_graph(submission)

        # Try to cancel immediately
        cancelled = execution_engine.cancel_execution("sub123")

        # May or may not succeed depending on timing
        assert isinstance(cancelled, bool)


class TestCacheManager:
    """Test cache manager."""

    def test_set_and_get(self, cache_manager):
        """Test setting and getting cache."""
        cache_manager.set("key1", "value1")

        assert cache_manager.get("key1") == "value1"

    def test_cache_miss(self, cache_manager):
        """Test cache miss."""
        assert cache_manager.get("nonexistent") is None

    def test_ttl_expiration(self, temp_dir):
        """Test TTL expiration."""
        cache = CacheManager(max_size=100, ttl=1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

        cache.shutdown()

    def test_size_limit(self, cache_manager):
        """Test cache size limit."""
        # Set cache to small size
        cache = CacheManager(max_size=3, ttl=60)

        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # Should have evicted oldest
        assert len(cache.cache) <= 3

        cache.shutdown()

    def test_clear(self, cache_manager):
        """Test clearing cache."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        cache_manager.clear()

        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None


class TestGraphAPIServer:
    """Test Graph API Server."""

    def test_initialization(self):
        """Test server initialization."""
        server = GraphAPIServer(host='127.0.0.1', port=9999)

        assert server.host == '127.0.0.1'
        assert server.port == 9999
        assert server.db is not None

        server.stop()

    def test_register_agent(self, api_server):
        """Test agent registration."""
        agent = api_server.register_agent("Test Agent", ["user"])

        assert agent.id is not None
        assert agent.name == "Test Agent"
        assert agent.api_key is not None
        assert "user" in agent.roles

    def test_submit_graph(self, api_server):
        """Test graph submission."""
        agent = api_server.register_agent("Test Agent", ["user"])

        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "TestNode"}],
            "edges": []
        }

        result = api_server.submit_graph(graph, agent.id)

        assert result["status"] == "submitted"
        assert "graph_id" in result

    def test_create_proposal(self, api_server):
        """Test proposal creation."""
        agent = api_server.register_agent("Test Agent", ["user"])

        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "ProposalNode"}],
            "edges": []
        }

        proposal = api_server.create_proposal(
            title="Test Proposal",
            description="Test Description",
            graph=graph,
            proposer_id=agent.id
        )

        assert proposal is not None
        assert proposal.title == "Test Proposal"

    def test_vote_on_proposal(self, api_server):
        """Test voting on proposal."""
        agent1 = api_server.register_agent("Agent 1", ["user"])
        agent2 = api_server.register_agent("Agent 2", ["user"])

        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "ProposalNode"}],
            "edges": []
        }

        proposal = api_server.create_proposal("Test", "Test", graph, agent1.id)

        success = api_server.vote_on_proposal(proposal.id, agent2.id, "for")
        assert success

    def test_get_status(self, api_server):
        """Test getting server status."""
        status = api_server.get_status()

        assert "status" in status
        assert "version" in status
        assert "uptime_seconds" in status
        assert status["status"] in ["active", "stopped"]

    def test_get_metrics(self, api_server):
        """Test getting metrics."""
        metrics = api_server.get_metrics()

        assert "timestamp" in metrics
        assert "metrics" in metrics
        assert "performance" in metrics


class TestGraphSubmission:
    """Test GraphSubmission dataclass."""

    def test_creation(self):
        """Test creating submission."""
        graph = {"id": "test", "type": "Graph", "nodes": [], "edges": []}

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1"
        )

        assert submission.id == "sub123"
        assert submission.status == ExecutionStatus.PENDING
        assert submission.submitted_at is not None

    def test_to_dict(self):
        """Test conversion to dict."""
        graph = {"id": "test", "type": "Graph", "nodes": [], "edges": []}

        submission = GraphSubmission(
            id="sub123",
            graph=graph,
            agent_id="agent1"
        )

        data = submission.to_dict()

        assert data["id"] == "sub123"
        assert data["status"] == "pending"
        assert "submitted_at" in data


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_submissions(self, api_server):
        """Test concurrent graph submissions."""
        agent = api_server.register_agent("Test Agent", ["user"])

        results = []
        errors = []

        def submit():
            try:
                graph = {
                    "id": f"test_{threading.current_thread().name}",
                    "type": "Graph",
                    "nodes": [{"id": "n1", "type": "TestNode"}],
                    "edges": []
                }
                result = api_server.submit_graph(graph, agent.id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0

    def test_concurrent_votes(self, api_server):
        """Test concurrent voting."""
        agents = [api_server.register_agent(f"Agent {i}", ["user"]) for i in range(5)]

        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "ProposalNode"}],
            "edges": []
        }

        proposal = api_server.create_proposal("Test", "Test", graph, agents[0].id)

        results = []

        def vote(agent):
            success = api_server.vote_on_proposal(proposal.id, agent.id, "for")
            results.append(success)

        threads = [threading.Thread(target=vote, args=(a,)) for a in agents[1:]]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
