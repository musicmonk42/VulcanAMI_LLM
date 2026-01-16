"""
Vulcan CLI HTTP Client

HTTP client wrapper for VULCAN API with authentication, error handling,
and connection management.

This module provides a robust HTTP client for interacting with the VULCAN API,
with support for API key authentication, comprehensive error handling, and
proper connection lifecycle management.

Security:
    - API keys are never logged or exposed in error messages
    - Timeout protection against hanging requests
    - Secure defaults (HTTPS recommended for production)

Thread Safety:
    - VulcanClient uses httpx.Client which is NOT thread-safe
    - Create separate instances per thread or use locks
    - For async operations, use httpx.AsyncClient instead

Example:
    >>> client = VulcanClient.from_settings()
    >>> try:
    ...     result = client.chat("What is AGI?")
    ...     print(result["response"])
    ... finally:
    ...     client.close()
    
    # Or use as context manager
    >>> with VulcanClient.from_settings() as client:
    ...     result = client.chat("What is AGI?")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class VulcanAPIError(Exception):
    """
    Raised when API returns an error.
    
    Attributes:
        status_code: HTTP status code (0 for connection errors)
        message: Human-readable error message
        response: Optional raw response object
    
    Example:
        >>> try:
        ...     client.chat("hello")
        ... except VulcanAPIError as e:
        ...     if e.status_code == 401:
        ...         print("Authentication failed")
        ...     elif e.status_code == 0:
        ...         print("Connection error")
    """
    
    def __init__(
        self,
        status_code: int,
        message: str,
        response: Optional[httpx.Response] = None
    ) -> None:
        """
        Initialize API error.
        
        Args:
            status_code: HTTP status code (0 for connection errors)
            message: Human-readable error message
            response: Optional raw response object for debugging
        """
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"API Error {status_code}: {message}")


class VulcanClient:
    """
    HTTP client for VULCAN API with authentication.
    
    Supports API key authentication via X-API-Key header.
    Loads configuration from environment variables or vulcan.settings.
    
    Environment Variables:
        VULCAN_SERVER_URL: Server URL (default: http://localhost:8000)
        VULCAN_API_KEY: API key for authentication
    
    Example:
        >>> client = VulcanClient.from_settings()
        >>> result = client.chat("Hello, VULCAN!")
        >>> print(result["response"])
    """
    
    DEFAULT_TIMEOUT = 30.0
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        """
        Initialize VULCAN API client.
        
        Args:
            base_url: Base URL for API server. Should include protocol (http/https).
                     Trailing slash is automatically removed.
            api_key: Optional API key for authentication. If provided, will be sent
                    as X-API-Key header with all requests.
            timeout: Request timeout in seconds. Applied to read operations.
        
        Raises:
            ValueError: If base_url is invalid or timeout is negative
        
        Security Note:
            API key is stored in memory and included in request headers.
            Never log or expose the api_key value.
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        
        self.base_url = base_url.rstrip('/')
        self._api_key = api_key  # Private to avoid accidental logging
        
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True  # Handle redirects automatically
        )
    
    @classmethod
    def from_settings(cls) -> "VulcanClient":
        """
        Create client from vulcan.settings or environment variables.
        
        Priority order:
        1. Environment variables (VULCAN_SERVER_URL, VULCAN_API_KEY)
        2. vulcan.settings (server_url, api_key)
        3. Defaults (http://localhost:8000, no API key)
        
        Returns:
            VulcanClient instance configured from settings
        """
        # First priority: environment variables
        base_url = os.environ.get("VULCAN_SERVER_URL")
        api_key = os.environ.get("VULCAN_API_KEY")
        
        # Second priority: vulcan.settings
        if not base_url or not api_key:
            try:
                from vulcan.settings import settings
                if not base_url:
                    base_url = getattr(settings, "server_url", None)
                if not api_key:
                    api_key = getattr(settings, "api_key", None)
            except ImportError:
                logger.debug("vulcan.settings not available, using environment only")
        
        # Defaults
        if not base_url:
            base_url = "http://localhost:8000"
        
        return cls(base_url=base_url, api_key=api_key)
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Handle API response with proper error messages.
        
        Converts HTTP errors into user-friendly VulcanAPIError exceptions
        with actionable guidance for common issues.
        
        Args:
            response: httpx Response object from API call
            
        Returns:
            Parsed JSON response dictionary
            
        Raises:
            VulcanAPIError: On HTTP errors (4xx, 5xx) with context-specific messages
            ValueError: If response is not valid JSON
        
        Note:
            This method never logs API keys or sensitive request data.
        """
        # Handle specific error codes with helpful messages
        if response.status_code == 401:
            raise VulcanAPIError(
                401,
                "Authentication failed. Check your API key.\n"
                "Set VULCAN_API_KEY environment variable or configure in settings.",
                response
            )
        elif response.status_code == 403:
            raise VulcanAPIError(
                403,
                "Access forbidden. Your API key may lack required permissions.",
                response
            )
        elif response.status_code == 404:
            raise VulcanAPIError(
                404,
                f"Endpoint not found: {response.url.path}\n"
                "Is the server running the correct version?",
                response
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise VulcanAPIError(
                429,
                f"Rate limit exceeded. Retry after: {retry_after}s",
                response
            )
        elif response.status_code >= 500:
            # Try to extract server error details
            error_detail = "Internal server error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_detail = error_data.get("detail", error_detail)
            except Exception:
                # If JSON parsing fails, use status text
                error_detail = response.reason_phrase or error_detail
            
            raise VulcanAPIError(
                response.status_code,
                f"{error_detail}\nCheck server logs for details.",
                response
            )
        
        # Raise for any other error status codes
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise VulcanAPIError(
                response.status_code,
                f"HTTP {response.status_code}: {response.reason_phrase}",
                response
            ) from e
        
        # Parse and validate JSON response
        try:
            return response.json()
        except ValueError as e:
            raise ValueError(
                f"Invalid JSON response from server: {str(e)}"
            ) from e
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        enable_reasoning: bool = True,
        enable_memory: bool = True,
        enable_safety: bool = True
    ) -> Dict[str, Any]:
        """
        Send chat message to /v1/chat endpoint.
        
        Args:
            message: User message to send. Must be non-empty string.
            history: Optional chat history as list of message dicts.
                    Each dict should have 'role' and 'content' keys.
            max_tokens: Maximum tokens in response (default: 2000).
                       Must be positive.
            enable_reasoning: Enable VULCAN reasoning engine (default: True)
            enable_memory: Enable memory retrieval system (default: True)
            enable_safety: Enable safety checks (default: True)
            
        Returns:
            API response dictionary with fields:
                - response (str): The generated response text
                - metadata (dict): Optional processing metadata
            
        Raises:
            VulcanAPIError: On API errors (401, 404, 500, etc.)
            ValueError: If message is empty or max_tokens is invalid
            httpx.TimeoutException: Request timeout (see client timeout setting)
            httpx.ConnectError: Cannot reach server
        
        Example:
            >>> result = client.chat("What is AGI?")
            >>> print(result["response"])
            >>> 
            >>> # With history
            >>> history = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi!"}
            ... ]
            >>> result = client.chat("How are you?", history=history)
        """
        # Input validation
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        try:
            response = self.client.post("/v1/chat", json={
                "message": message,
                "max_tokens": max_tokens,
                "history": history or [],
                "enable_reasoning": enable_reasoning,
                "enable_memory": enable_memory,
                "enable_safety": enable_safety,
            })
            return self._handle_response(response)
        except httpx.TimeoutException:
            raise VulcanAPIError(
                408,
                f"Request timed out after {self.client.timeout.read}s.\n"
                "The server may be overloaded or processing a complex query."
            )
        except httpx.ConnectError as e:
            raise VulcanAPIError(
                0,
                f"Could not connect to server at {self.base_url}.\n"
                f"Details: {str(e)}\n"
                "Is the server running?"
            )
    
    def health(self) -> Dict[str, Any]:
        """
        Get health status from /health endpoint.
        
        Returns:
            Health status information
            
        Raises:
            VulcanAPIError: On API errors
        """
        try:
            response = self.client.get("/health")
            return self._handle_response(response)
        except httpx.ConnectError as e:
            raise VulcanAPIError(
                0,
                f"Could not connect to server at {self.base_url}.\n"
                f"Details: {str(e)}"
            )
    
    def search_memory(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Search memory via /v1/memory/search endpoint.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Search results with memories
            
        Raises:
            VulcanAPIError: On API errors
        """
        try:
            response = self.client.post("/v1/memory/search", json={
                "query": query,
                "k": k
            })
            return self._handle_response(response)
        except httpx.ConnectError as e:
            raise VulcanAPIError(
                0,
                f"Could not connect to server at {self.base_url}.\n"
                f"Details: {str(e)}"
            )
    
    def get_metrics(self) -> str:
        """
        Get Prometheus metrics from /metrics endpoint.
        
        Returns:
            Metrics in Prometheus text format
            
        Raises:
            VulcanAPIError: On API errors
        """
        try:
            response = self.client.get("/metrics")
            if response.status_code != 200:
                raise VulcanAPIError(response.status_code, "Failed to get metrics")
            return response.text
        except httpx.ConnectError as e:
            raise VulcanAPIError(
                0,
                f"Could not connect to server at {self.base_url}.\n"
                f"Details: {str(e)}"
            )
    
    def close(self) -> None:
        """
        Close the HTTP client and release resources.
        
        Should be called when done using the client to prevent resource leaks.
        After calling close(), the client cannot be reused.
        
        Example:
            >>> client = VulcanClient.from_settings()
            >>> try:
            ...     result = client.chat("hello")
            ... finally:
            ...     client.close()
        """
        self.client.close()
    
    def __enter__(self) -> VulcanClient:
        """
        Context manager entry.
        
        Returns:
            self for use in with statement
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit - ensures client is closed.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()
    
    def __repr__(self) -> str:
        """String representation (safe - never exposes API key)."""
        auth_status = "authenticated" if self._api_key else "no auth"
        return f"VulcanClient(base_url='{self.base_url}', {auth_status})"
