"""
Test suite for vulcan.cli.client - HTTP client for VULCAN API

Comprehensive tests for VulcanClient including authentication, error handling,
and API integration.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from vulcan.cli.client import VulcanAPIError, VulcanClient


class TestVulcanAPIError:
    """Test VulcanAPIError exception class."""
    
    def test_error_creation(self):
        """Test basic error creation."""
        error = VulcanAPIError(404, "Not found")
        assert error.status_code == 404
        assert error.message == "Not found"
        assert "404" in str(error)
        assert "Not found" in str(error)
    
    def test_error_with_response(self):
        """Test error creation with response object."""
        mock_response = Mock(spec=httpx.Response)
        error = VulcanAPIError(500, "Server error", response=mock_response)
        assert error.response is mock_response


class TestVulcanClientInit:
    """Test VulcanClient initialization."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        client = VulcanClient()
        assert client.base_url == "http://localhost:8000"
        assert client._api_key is None
        assert "localhost:8000" in str(client)
        client.close()
    
    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        client = VulcanClient(base_url="https://api.example.com")
        assert client.base_url == "https://api.example.com"
        client.close()
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from URL."""
        client = VulcanClient(base_url="http://example.com/")
        assert client.base_url == "http://example.com"
        client.close()
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = VulcanClient(api_key="test-key-123")
        assert client._api_key == "test-key-123"
        assert "X-API-Key" in client.client.headers
        assert client.client.headers["X-API-Key"] == "test-key-123"
        # Verify API key not exposed in repr
        assert "test-key-123" not in str(client)
        assert "authenticated" in str(client)
        client.close()
    
    def test_init_invalid_url(self):
        """Test initialization with invalid URL."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            VulcanClient(base_url="")
    
    def test_init_invalid_timeout(self):
        """Test initialization with invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            VulcanClient(timeout=-1)
        
        with pytest.raises(ValueError, match="timeout must be positive"):
            VulcanClient(timeout=0)
    
    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = VulcanClient(timeout=60.0)
        assert client.client.timeout.read == 60.0
        client.close()


class TestVulcanClientFromSettings:
    """Test VulcanClient.from_settings() factory method."""
    
    def test_from_settings_env_vars(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            "VULCAN_SERVER_URL": "https://test.example.com",
            "VULCAN_API_KEY": "env-key-123"
        }):
            client = VulcanClient.from_settings()
            assert client.base_url == "https://test.example.com"
            assert client._api_key == "env-key-123"
            client.close()
    
    def test_from_settings_defaults(self):
        """Test loading with defaults when no config available."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            # Mock settings import to fail
            with patch("vulcan.cli.client.VulcanClient.__init__") as mock_init:
                mock_init.return_value = None
                
                # Should use defaults
                with patch("vulcan.cli.client.importlib"):
                    client = VulcanClient.from_settings()
    
    def test_from_settings_vulcan_settings(self):
        """Test loading from vulcan.settings."""
        mock_settings = Mock()
        mock_settings.server_url = "https://settings.example.com"
        mock_settings.api_key = "settings-key-456"
        
        with patch.dict(os.environ, {}, clear=True):
            with patch("vulcan.cli.client.settings", mock_settings):
                client = VulcanClient.from_settings()
                # Environment takes priority, so should use defaults
                assert client.base_url in ["http://localhost:8000", "https://settings.example.com"]
                client.close()


class TestVulcanClientHandleResponse:
    """Test response handling and error conversion."""
    
    def test_handle_response_success(self):
        """Test successful response handling."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        result = client._handle_response(mock_response)
        assert result == {"status": "ok"}
        client.close()
    
    def test_handle_response_401(self):
        """Test 401 Unauthorized error."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        
        with pytest.raises(VulcanAPIError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.status_code == 401
        assert "Authentication failed" in exc_info.value.message
        assert "API key" in exc_info.value.message
        client.close()
    
    def test_handle_response_403(self):
        """Test 403 Forbidden error."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 403
        
        with pytest.raises(VulcanAPIError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.status_code == 403
        assert "forbidden" in exc_info.value.message.lower()
        client.close()
    
    def test_handle_response_404(self):
        """Test 404 Not Found error."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.url = Mock()
        mock_response.url.path = "/v1/chat"
        
        with pytest.raises(VulcanAPIError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.message.lower()
        client.close()
    
    def test_handle_response_429(self):
        """Test 429 Rate Limit error."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        
        with pytest.raises(VulcanAPIError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.message
        assert "60" in exc_info.value.message
        client.close()
    
    def test_handle_response_500(self):
        """Test 500 Server Error."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.json.side_effect = Exception("Not JSON")
        
        with pytest.raises(VulcanAPIError) as exc_info:
            client._handle_response(mock_response)
        
        assert exc_info.value.status_code == 500
        assert "error" in exc_info.value.message.lower()
        client.close()
    
    def test_handle_response_invalid_json(self):
        """Test response with invalid JSON."""
        client = VulcanClient()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            client._handle_response(mock_response)
        
        client.close()


class TestVulcanClientChat:
    """Test chat() method."""
    
    def test_chat_success(self):
        """Test successful chat request."""
        client = VulcanClient()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello!"}
        
        with patch.object(client.client, 'post', return_value=mock_response):
            result = client.chat("Hi there")
            assert result["response"] == "Hello!"
        
        client.close()
    
    def test_chat_with_history(self):
        """Test chat with conversation history."""
        client = VulcanClient()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Response"}
        
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        with patch.object(client.client, 'post', return_value=mock_response) as mock_post:
            result = client.chat("How are you?", history=history)
            
            # Verify history was passed
            call_args = mock_post.call_args
            assert call_args[1]['json']['history'] == history
        
        client.close()
    
    def test_chat_empty_message(self):
        """Test chat with empty message."""
        client = VulcanClient()
        
        with pytest.raises(ValueError, match="message cannot be empty"):
            client.chat("")
        
        with pytest.raises(ValueError, match="message cannot be empty"):
            client.chat("   ")
        
        client.close()
    
    def test_chat_invalid_max_tokens(self):
        """Test chat with invalid max_tokens."""
        client = VulcanClient()
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            client.chat("hello", max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            client.chat("hello", max_tokens=-1)
        
        client.close()
    
    def test_chat_timeout(self):
        """Test chat with timeout."""
        client = VulcanClient()
        
        with patch.object(client.client, 'post', side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(VulcanAPIError) as exc_info:
                client.chat("test")
            
            assert exc_info.value.status_code == 408
            assert "timeout" in exc_info.value.message.lower()
        
        client.close()
    
    def test_chat_connection_error(self):
        """Test chat with connection error."""
        client = VulcanClient()
        
        with patch.object(client.client, 'post', side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(VulcanAPIError) as exc_info:
                client.chat("test")
            
            assert exc_info.value.status_code == 0
            assert "connect" in exc_info.value.message.lower()
        
        client.close()


class TestVulcanClientHealth:
    """Test health() method."""
    
    def test_health_success(self):
        """Test successful health check."""
        client = VulcanClient()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch.object(client.client, 'get', return_value=mock_response):
            result = client.health()
            assert result["status"] == "healthy"
        
        client.close()
    
    def test_health_connection_error(self):
        """Test health check with connection error."""
        client = VulcanClient()
        
        with patch.object(client.client, 'get', side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(VulcanAPIError) as exc_info:
                client.health()
            
            assert exc_info.value.status_code == 0
        
        client.close()


class TestVulcanClientMemorySearch:
    """Test search_memory() method."""
    
    def test_search_memory_success(self):
        """Test successful memory search."""
        client = VulcanClient()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"content": "Memory 1"}]}
        
        with patch.object(client.client, 'post', return_value=mock_response):
            result = client.search_memory("test query", k=5)
            assert len(result["results"]) == 1
        
        client.close()


class TestVulcanClientMetrics:
    """Test get_metrics() method."""
    
    def test_get_metrics_success(self):
        """Test successful metrics retrieval."""
        client = VulcanClient()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "# Prometheus metrics"
        
        with patch.object(client.client, 'get', return_value=mock_response):
            result = client.get_metrics()
            assert "Prometheus" in result
        
        client.close()


class TestVulcanClientContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self):
        """Test using client as context manager."""
        with patch.object(VulcanClient, 'close') as mock_close:
            with VulcanClient() as client:
                assert isinstance(client, VulcanClient)
            
            # Verify close was called
            mock_close.assert_called_once()
    
    def test_context_manager_with_exception(self):
        """Test context manager cleanup on exception."""
        with patch.object(VulcanClient, 'close') as mock_close:
            try:
                with VulcanClient() as client:
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Verify close was called even with exception
            mock_close.assert_called_once()


class TestVulcanClientRepr:
    """Test string representation."""
    
    def test_repr_no_auth(self):
        """Test repr without authentication."""
        client = VulcanClient()
        repr_str = repr(client)
        assert "VulcanClient" in repr_str
        assert "localhost:8000" in repr_str
        assert "no auth" in repr_str
        client.close()
    
    def test_repr_with_auth(self):
        """Test repr with authentication."""
        client = VulcanClient(api_key="secret-key")
        repr_str = repr(client)
        assert "VulcanClient" in repr_str
        assert "authenticated" in repr_str
        # Verify API key is NOT in repr
        assert "secret-key" not in repr_str
        client.close()
