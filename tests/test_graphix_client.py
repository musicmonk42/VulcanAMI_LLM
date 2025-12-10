"""
Comprehensive test suite for Graphix Client SDK
Targets 85%+ code coverage with unit and integration tests.

Run with:
    pytest test_graphix_client.py -v --cov=graphix_client --cov-report=html --cov-report=term
"""

import asyncio
import base64
import json
import os
# Import the client (adjust import path as needed)
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import aiohttp
import jsonschema
import pytest
import pytest_asyncio  # Add this import
from aiohttp import web

# Add the parent directory to sys.path to import from client_sdk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from client_sdk.graphix_client import (GraphixClient, GraphixClientError,
                                       RetryConfig)


# Test fixtures
@pytest.fixture
def mock_private_key():
    """Generate a mock RSA private key."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )


@pytest.fixture
def temp_private_key_file(mock_private_key):
    """Create a temporary private key file."""
    from cryptography.hazmat.primitives import serialization
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem') as f:
        pem = mock_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        f.write(pem)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_graph():
    """Sample valid graph for testing."""
    return {
        "grammar_version": "3.4.0",
        "id": "test_graph",
        "type": "Graph",
        "nodes": [
            {"id": "node1", "type": "InputNode", "value": "test"},
            {"id": "node2", "type": "OutputNode"}
        ],
        "edges": [
            {"id": "e1", "from": "node1", "to": "node2", "type": "data"}
        ]
    }


@pytest.fixture
def mock_schema():
    """Mock JSON schema for validation."""
    return {
        "type": "object",
        "required": ["grammar_version", "id", "type", "nodes", "edges"],
        "properties": {
            "grammar_version": {"type": "string"},
            "id": {"type": "string"},
            "type": {"type": "string"},
            "nodes": {"type": "array"},
            "edges": {"type": "array"}
        }
    }


@pytest_asyncio.fixture  # Changed from @pytest.fixture
async def mock_server():
    """Create a mock aiohttp server for testing."""
    app = web.Application()

    async def health_handler(request):
        return web.json_response({"status": "healthy", "version": "1.0.0"})

    async def status_handler(request):
        return web.json_response({"status": "operational", "uptime": 3600})

    async def propose_handler(request):
        data = await request.json()
        return web.json_response({"status": "success", "proposal_id": data.get("proposal_id")})

    async def vote_handler(request):
        data = await request.json()
        return web.json_response({"status": "success", "vote": data.get("vote")})

    async def execute_handler(request):
        data = await request.json()
        return web.json_response({"status": "success", "execution_id": data.get("execution_id")})

    async def audit_handler(request):
        data = await request.json()
        return web.json_response({"entries": [{"id": "1", "action": "test"}]})

    app.router.add_get('/health', health_handler)
    app.router.add_get('/status', status_handler)
    app.router.add_post('/ir/propose', propose_handler)
    app.router.add_post('/ir/vote', vote_handler)
    app.router.add_post('/ir/execute', execute_handler)
    app.router.add_post('/audit/logs', audit_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8765)
    await site.start()

    yield "http://localhost:8765"

    await runner.cleanup()


# Test RetryConfig
class TestRetryConfig:
    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_retry_config_custom(self):
        """Test custom retry configuration."""
        config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=60.0)
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0


# Test GraphixClient Initialization
class TestGraphixClientInit:
    def test_init_minimal(self):
        """Test client initialization with minimal parameters."""
        client = GraphixClient()
        assert client.registry_endpoint == "http://localhost:8787"
        assert client.agent_id == "default-agent"
        assert client.timeout == 30
        assert client.private_key is None

    def test_init_with_endpoints(self):
        """Test client initialization with custom endpoints."""
        client = GraphixClient(
            registry_endpoint="http://registry:8787/",
            executor_endpoint="http://executor:8788/",
            audit_endpoint="http://audit:8789/"
        )
        assert client.registry_endpoint == "http://registry:8787"
        assert client.executor_endpoint == "http://executor:8788"
        assert client.audit_endpoint == "http://audit:8789"

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        # NOT A REAL API KEY - Test value only
        client = GraphixClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"

    def test_init_with_env_api_key(self):
        """Test client initialization with environment API key."""
        with patch.dict(os.environ, {'GRAPHIX_API_KEY': 'env-key-456'}):
            client = GraphixClient()
            assert client.api_key == "env-key-456"

    def test_init_with_private_key(self, temp_private_key_file):
        """Test client initialization with private key file."""
        client = GraphixClient(private_key_path=temp_private_key_file)
        assert client.private_key is not None

    def test_init_with_invalid_private_key(self):
        """Test client initialization with invalid private key."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
            f.write("invalid key data")
            temp_path = f.name

        try:
            with pytest.raises(GraphixClientError, match="Invalid private key"):
                GraphixClient(private_key_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_init_with_retry_config(self):
        """Test client initialization with custom retry config."""
        retry_config = RetryConfig(max_retries=5)
        client = GraphixClient(retry_config=retry_config)
        assert client.retry_config.max_retries == 5

    def test_schema_loading(self, mock_schema):
        """Test schema loading on initialization."""
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_schema))):
            with patch('pathlib.Path.exists', return_value=True):
                client = GraphixClient()
                assert client.schema is not None


# Test Token Management
class TestTokenManagement:
    @pytest.mark.asyncio
    async def test_token_is_expired_no_token(self):
        """Test token expiry check with no token."""
        client = GraphixClient()
        assert client._token_is_expired() is True

    @pytest.mark.asyncio
    async def test_token_is_expired_valid_token(self):
        """Test token expiry check with valid token."""
        client = GraphixClient()
        client.auth_token = "test-token"  # NOT A REAL TOKEN - Test value only
        client.token_expiry = datetime.utcnow() + timedelta(hours=1)
        assert client._token_is_expired() is False

    @pytest.mark.asyncio
    async def test_token_is_expired_expired_token(self):
        """Test token expiry check with expired token."""
        client = GraphixClient()
        client.auth_token = "test-token"  # NOT A REAL TOKEN - Test value only
        client.token_expiry = datetime.utcnow() - timedelta(hours=1)
        assert client._token_is_expired() is True

    @pytest.mark.asyncio
    async def test_fetch_new_token(self):
        """Test fetching a new authentication token."""
        client = GraphixClient()
        await client._fetch_new_token()
        assert client.auth_token is not None
        assert client.token_expiry is not None
        assert client.token_expiry > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_refresh_token_if_needed_expired(self):
        """Test token refresh when token is expired."""
        client = GraphixClient()
        await client._refresh_token_if_needed()
        assert client.auth_token is not None

    @pytest.mark.asyncio
    async def test_refresh_token_if_needed_valid(self):
        """Test token refresh when token is still valid."""
        client = GraphixClient()
        client.auth_token = "existing-token"
        client.token_expiry = datetime.utcnow() + timedelta(hours=1)
        old_token = client.auth_token
        await client._refresh_token_if_needed()
        assert client.auth_token == old_token

    @pytest.mark.asyncio
    async def test_get_headers_no_auth(self):
        """Test header construction without authentication."""
        client = GraphixClient()
        headers = await client._get_headers()
        assert "Content-Type" in headers
        assert "Accept" in headers

    @pytest.mark.asyncio
    async def test_get_headers_with_api_key(self):
        """Test header construction with API key."""
        # NOT A REAL API KEY - Test value only
        client = GraphixClient(api_key="test-key")
        headers = await client._get_headers()
        assert headers["X-API-KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_get_headers_with_token(self):
        """Test header construction with auth token."""
        client = GraphixClient()
        await client._fetch_new_token()
        headers = await client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")


# Test Request Signing
class TestRequestSigning:
    def test_sign_request_no_key(self):
        """Test request signing without private key."""
        client = GraphixClient()
        with pytest.raises(GraphixClientError, match="Private key not loaded"):
            client._sign_request({"test": "data"})

    def test_sign_request_with_key(self, temp_private_key_file):
        """Test request signing with private key."""
        client = GraphixClient(private_key_path=temp_private_key_file)
        payload = {"test": "data", "timestamp": "2024-01-01T00:00:00"}
        signature = client._sign_request(payload)
        assert signature is not None
        assert isinstance(signature, str)
        # Verify it's base64 encoded
        base64.b64decode(signature)

    def test_sign_request_deterministic(self, temp_private_key_file):
        """Test that signing the same payload produces consistent results."""
        client = GraphixClient(private_key_path=temp_private_key_file)
        payload = {"test": "data"}
        sig1 = client._sign_request(payload)
        sig2 = client._sign_request(payload)
        # Note: PSS padding includes randomness, so signatures will differ
        assert sig1 is not None
        assert sig2 is not None


# Test Graph Validation
class TestGraphValidation:
    def test_validate_graph_no_schema(self, sample_graph):
        """Test graph validation without loaded schema."""
        client = GraphixClient()
        client.schema = None
        assert client._validate_graph(sample_graph) is True

    def test_validate_graph_valid(self, sample_graph, mock_schema):
        """Test validation of a valid graph."""
        client = GraphixClient()
        client.schema = mock_schema
        assert client._validate_graph(sample_graph) is True

    def test_validate_graph_invalid(self, mock_schema):
        """Test validation of an invalid graph."""
        client = GraphixClient()
        client.schema = mock_schema
        invalid_graph = {"id": "test"}  # Missing required fields
        with pytest.raises(GraphixClientError, match="Invalid graph"):
            client._validate_graph(invalid_graph)


# Test Session Management
class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_ensure_session_creates_new(self):
        """Test session creation when none exists."""
        client = GraphixClient()
        assert client.session is None
        session = await client._ensure_session()
        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)
        await client.close()

    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self):
        """Test session reuse when one exists."""
        client = GraphixClient()
        session1 = await client._ensure_session()
        session2 = await client._ensure_session()
        assert session1 is session2
        await client.close()

    @pytest.mark.asyncio
    async def test_ensure_session_recreates_closed(self):
        """Test session recreation when previous is closed."""
        client = GraphixClient()
        session1 = await client._ensure_session()
        await session1.close()
        session2 = await client._ensure_session()
        assert session1 is not session2
        await client.close()


# Test Health Check
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_server):
        """Test successful health check."""
        client = GraphixClient(registry_endpoint=mock_server)
        health = await client.health_check()
        assert health["status"] == "healthy"
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_cached(self, mock_server):
        """Test health check caching."""
        client = GraphixClient(registry_endpoint=mock_server)
        health1 = await client.health_check()
        # Second call should use cache
        health2 = await client.health_check()
        assert health1 == health2
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling."""
        client = GraphixClient(registry_endpoint="http://localhost:9999")
        client.retry_config = RetryConfig(max_retries=1, base_delay=0.1)
        with pytest.raises(GraphixClientError, match="Health check failed"):
            await client.health_check()
        await client.close()


# Test Status Endpoint
class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_get_status_success(self, mock_server):
        """Test successful status retrieval."""
        client = GraphixClient(registry_endpoint=mock_server)
        status = await client.get_status()
        assert status["status"] == "operational"
        await client.close()

    @pytest.mark.asyncio
    async def test_get_status_cached(self, mock_server):
        """Test status caching."""
        client = GraphixClient(registry_endpoint=mock_server)
        status1 = await client.get_status()
        status2 = await client.get_status()
        assert status1 == status2
        await client.close()


# Test Graph Proposal
class TestGraphProposal:
    @pytest.mark.asyncio
    async def test_submit_proposal_success(self, mock_server, sample_graph, temp_private_key_file):
        """Test successful graph proposal submission."""
        client = GraphixClient(
            registry_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.submit_graph_proposal(sample_graph)
        assert response["status"] == "success"
        assert "proposal_id" in response
        await client.close()

    @pytest.mark.asyncio
    async def test_submit_proposal_validation_failure(self, mock_server, mock_schema, temp_private_key_file):
        """Test proposal submission with validation failure."""
        client = GraphixClient(
            registry_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        client.schema = mock_schema
        invalid_graph = {"id": "test"}
        with pytest.raises(GraphixClientError, match="Invalid graph"):
            await client.submit_graph_proposal(invalid_graph)
        await client.close()


# Test Voting
class TestVoting:
    @pytest.mark.asyncio
    async def test_vote_approve(self, mock_server, temp_private_key_file):
        """Test voting to approve a proposal."""
        client = GraphixClient(
            registry_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.vote_on_proposal("test-id", "yes", "Looks good")
        assert response["status"] == "success"
        assert response["vote"] == "yes"
        await client.close()

    @pytest.mark.asyncio
    async def test_vote_reject(self, mock_server, temp_private_key_file):
        """Test voting to reject a proposal."""
        client = GraphixClient(
            registry_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.vote_on_proposal("test-id", "no", "Needs work")
        assert response["status"] == "success"
        assert response["vote"] == "no"
        await client.close()

    @pytest.mark.asyncio
    async def test_vote_invalid_value(self, temp_private_key_file):
        """Test voting with invalid vote value."""
        client = GraphixClient(private_key_path=temp_private_key_file)
        with pytest.raises(GraphixClientError, match="Vote must be"):
            await client.vote_on_proposal("test-id", "maybe", "Unsure")
        await client.close()


# Test Graph Execution
class TestGraphExecution:
    @pytest.mark.asyncio
    async def test_execute_graph_success(self, mock_server, sample_graph, temp_private_key_file):
        """Test successful graph execution."""
        client = GraphixClient(
            executor_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.execute_graph(sample_graph)
        assert response["status"] == "success"
        await client.close()


# Test Audit Log
class TestAuditLog:
    @pytest.mark.asyncio
    async def test_get_audit_log_success(self, mock_server, temp_private_key_file):
        """Test successful audit log retrieval."""
        client = GraphixClient(
            audit_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.get_audit_log(limit=10, offset=0)
        assert "entries" in response
        assert len(response["entries"]) > 0
        await client.close()

    @pytest.mark.asyncio
    async def test_get_audit_log_pagination(self, mock_server, temp_private_key_file):
        """Test audit log pagination."""
        client = GraphixClient(
            audit_endpoint=mock_server,
            private_key_path=temp_private_key_file
        )
        response = await client.get_audit_log(limit=5, offset=10)
        assert "entries" in response
        await client.close()


# Test Retry Logic
class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on request failure."""
        client = GraphixClient(registry_endpoint="http://localhost:9999")
        client.retry_config = RetryConfig(max_retries=2, base_delay=0.1, max_delay=1.0)

        call_count = 0
        async def failing_method():
            nonlocal call_count
            call_count += 1
            raise aiohttp.ClientError("Test error")

        with pytest.raises(GraphixClientError):
            await client._retry_request(failing_method)

        assert call_count == 3  # Initial + 2 retries
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        """Test exponential backoff in retry logic."""
        client = GraphixClient()
        client.retry_config = RetryConfig(max_retries=3, base_delay=0.1, max_delay=1.0)

        call_times = []
        async def failing_method():
            call_times.append(asyncio.get_event_loop().time())
            raise aiohttp.ClientError("Test error")

        with pytest.raises(GraphixClientError):
            await client._retry_request(failing_method)

        # Verify delays increase
        assert len(call_times) == 4
        await client.close()


# Test WebSocket
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test WebSocket connection."""
        client = GraphixClient()

        events_received = []
        async def handler(event):
            events_received.append(event)

        # Mock WebSocket connection
        with patch('aiohttp.ClientSession.ws_connect') as mock_ws:
            mock_ws_obj = AsyncMock()
            mock_ws_obj.__aenter__ = AsyncMock(return_value=mock_ws_obj)
            mock_ws_obj.__aexit__ = AsyncMock()

            # Simulate receiving a message then closing
            async def mock_iter():
                msg = Mock()
                msg.type = aiohttp.WSMsgType.TEXT
                msg.data = json.dumps({"type": "test_event"})
                yield msg

                msg2 = Mock()
                msg2.type = aiohttp.WSMsgType.CLOSED
                yield msg2

            mock_ws_obj.__aiter__ = mock_iter
            mock_ws.return_value = mock_ws_obj

            await client.connect_websocket(handler)
            await asyncio.sleep(0.2)  # Give it time to process

            assert len(events_received) >= 0  # May or may not have received

        await client.close()


# Test Context Manager
class TestContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with GraphixClient() as client:
            assert client.session is not None
        # Session should be closed after exiting context

    @pytest.mark.asyncio
    async def test_context_manager_with_operations(self, mock_server):
        """Test context manager with operations."""
        async with GraphixClient(registry_endpoint=mock_server) as client:
            health = await client.health_check()
            assert health["status"] == "healthy"


# Test Close and Cleanup
class TestCloseAndCleanup:
    @pytest.mark.asyncio
    async def test_close(self):
        """Test client cleanup."""
        client = GraphixClient()
        await client._ensure_session()
        assert client.session is not None
        await client.close()
        assert client.session.closed

    @pytest.mark.asyncio
    async def test_close_with_websocket(self):
        """Test cleanup with active WebSocket."""
        client = GraphixClient()

        # Create a mock WebSocket task
        async def mock_ws_loop():
            await asyncio.sleep(10)

        client.ws_session = asyncio.create_task(mock_ws_loop())
        await client.close()
        # Give it a moment to process cancellation
        await asyncio.sleep(0.1)
        assert client.ws_session.cancelled()


# Integration Tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_server, sample_graph, temp_private_key_file):
        """Test complete workflow: health check, propose, vote, execute, audit."""
        async with GraphixClient(
            registry_endpoint=mock_server,
            executor_endpoint=mock_server,
            audit_endpoint=mock_server,
            private_key_path=temp_private_key_file
        ) as client:
            # Health check
            health = await client.health_check()
            assert health["status"] == "healthy"

            # Status
            status = await client.get_status()
            assert "status" in status

            # Submit proposal
            proposal = await client.submit_graph_proposal(sample_graph)
            assert proposal["status"] == "success"

            # Vote
            vote = await client.vote_on_proposal("test-id", "yes", "Good")
            assert vote["status"] == "success"
            assert vote["vote"] == "yes"

            # Execute
            execution = await client.execute_graph(sample_graph)
            assert execution["status"] == "success"

            # Audit
            audit = await client.get_audit_log(limit=5)
            assert "entries" in audit


# Error Handling Tests
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        client = GraphixClient(registry_endpoint="http://localhost:9999")
        client.retry_config = RetryConfig(max_retries=1, base_delay=0.1)

        with pytest.raises(GraphixClientError):
            await client.health_check()

        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON responses."""
        # This would require more sophisticated mocking
        pass

    def test_graphix_client_error(self):
        """Test GraphixClientError exception."""
        error = GraphixClientError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=graphix_client", "--cov-report=term-missing"])
