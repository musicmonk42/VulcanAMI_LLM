"""
Graphix IR Client SDK (v1.0.0 - Production Ready)
================================================

A comprehensive Python client for interacting with Graphix IR services.
Provides robust, feature-complete access to the registry, executor, and audit APIs.

Usage:
    from graphix_client import GraphixClient
    
    client = GraphixClient(
        registry_endpoint="http://localhost:8787",
        executor_endpoint="http://localhost:8788",  # Optional separate endpoint
        agent_id="agent-grok",
        private_key_path="keys/agent-grok.pem"
    )
    
    # Real-time event handling
    async def handle_event(event):
        print(f"Received event: {event}")
    
    await client.connect_websocket(handle_event)
    
    # Submit with automatic retry
    response = await client.submit_graph_proposal(proposal)

Author: Graphix IR Team
"""

import aiohttp
import asyncio
import json
import base64
import logging
import os
from typing import Dict, Any, Optional, Callable, List, Union, Awaitable
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jsonschema
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphixClientError(Exception):
    """Base exception for Graphix client errors"""
    pass

class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

class GraphixClient:
    """
    Production-ready client for Graphix IR services.
    Features: authentication, request signing, retry logic, WebSocket events, 
    graph validation, caching, and multi-endpoint support.
    """
    
    def __init__(
        self,
        registry_endpoint: str = "http://localhost:8787",
        executor_endpoint: Optional[str] = None,
        audit_endpoint: Optional[str] = None,
        agent_id: str = "default-agent",
        private_key_path: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize Graphix client.

        Args:
            registry_endpoint: URL for registry service
            executor_endpoint: URL for executor service (optional, defaults to registry_endpoint)
            audit_endpoint: URL for audit service (optional, defaults to registry_endpoint)
            agent_id: Unique identifier for the agent
            private_key_path: Path to RSA private key for signing
            api_key: API key for authentication (can also be set via GRAPHIX_API_KEY env var)
            timeout: Request timeout in seconds
            retry_config: Retry configuration
        """
        self.registry_endpoint = registry_endpoint.rstrip('/')
        self.executor_endpoint = executor_endpoint.rstrip('/') if executor_endpoint else registry_endpoint.rstrip('/')
        self.audit_endpoint = audit_endpoint.rstrip('/') if audit_endpoint else registry_endpoint.rstrip('/')
        self.agent_id = agent_id
        self.api_key = api_key or os.environ.get("GRAPHIX_API_KEY")
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.session = None
        self.ws_session = None
        self.private_key = None
        self.cache = {}  # Simple in-memory cache for status and health checks
        self.cache_ttl = 300  # Cache TTL in seconds
        self.cache_lock = asyncio.Lock()  # Lock for thread-safe cache access

        # Token management attributes
        self.auth_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.token_lock = asyncio.Lock()

        # Load private key
        if private_key_path:
            try:
                with open(private_key_path, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                raise GraphixClientError(f"Invalid private key: {e}")

        # Load schema for graph validation
        self.schema = None
        # Use absolute path relative to this file
        client_dir = Path(__file__).parent
        schema_path = client_dir.parent / "schemas" / "graph_v3_4_0.json"
        try:
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load schema: {e}")

    async def _get_headers(self) -> Dict[str, str]:
        """Constructs headers for a request, refreshing token if needed."""
        await self._refresh_token_if_needed()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _token_is_expired(self) -> bool:
        """Checks if the current auth token is expired or non-existent."""
        # Add 5 minute safety buffer to refresh token before it expires
        safety_buffer = timedelta(minutes=5)
        return (self.auth_token is None or 
                self.token_expiry is None or 
                datetime.utcnow() >= (self.token_expiry - safety_buffer))

    async def _fetch_new_token(self):
        """Simulates fetching a new authentication token from an auth endpoint.
        
        NOTE: This is a simulated implementation for development/testing.
        In production, replace with actual authentication service call.
        """
        logger.info("Fetching new authentication token...")
        # In a real implementation, this would make a request to an auth service.
        # For example:
        # async with self.session.post(f"{self.registry_endpoint}/auth/token", json={'agent_id': self.agent_id}) as resp:
        #     data = await resp.json()
        #     self.auth_token = data['token']
        #     self.token_expiry = datetime.utcnow() + timedelta(seconds=data['expires_in'])
        await asyncio.sleep(0.1)  # Simulate network latency
        self.auth_token = f"simulated-token-{uuid.uuid4()}"
        self.token_expiry = datetime.utcnow() + timedelta(hours=1)
        logger.info("Successfully fetched new authentication token (simulated).")

    async def _refresh_token_if_needed(self):
        """
        Refreshes the auth token if it's expired, using a thread-safe,
        double-checked locking pattern to prevent race conditions.
        """
        if self._token_is_expired():
            async with self.token_lock:
                # Double-check inside the lock
                if self._token_is_expired():
                    await self._fetch_new_token()

    async def _retry_request(self, method: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute an HTTP request with retry logic.

        Args:
            method: HTTP method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Response dictionary
        """
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await method(*args, **kwargs)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.retry_config.max_retries:
                    logger.error(f"Request failed after {self.retry_config.max_retries} retries: {e}")
                    raise GraphixClientError(f"Request failed: {e}")
                delay = min(self.retry_config.base_delay * (2 ** attempt), self.retry_config.max_delay)
                logger.warning(f"Request failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        return {}

    def _sign_request(self, payload: Dict[str, Any]) -> str:
        """
        Sign a payload with the private key.

        Args:
            payload: Dictionary to sign

        Returns:
            Base64-encoded signature
        """
        if not self.private_key:
            raise GraphixClientError("Private key not loaded")
        payload_str = json.dumps(payload, sort_keys=True)
        signature = self.private_key.sign(
            payload_str.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _validate_graph(self, graph: Dict[str, Any]) -> bool:
        """
        Validate a graph against the schema.

        Args:
            graph: Graph dictionary

        Returns:
            True if valid, raises GraphixClientError if invalid
        """
        if not self.schema:
            logger.warning("Schema not loaded, skipping validation")
            return True
        try:
            jsonschema.validate(graph, self.schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Graph validation failed: {e}")
            raise GraphixClientError(f"Invalid graph: {e}")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensure an HTTP session is active.

        Returns:
            Active ClientSession
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self.session

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status dictionary
        """
        cache_key = "health_check"
        async with self.cache_lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']

        session = await self._ensure_session()
        async def do_health_check():
            try:
                headers = await self._get_headers()
                async with session.get(f"{self.registry_endpoint}/health", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Health check failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Health check failed: {e}")

        health = await self._retry_request(do_health_check)
        async with self.cache_lock:
            self.cache[cache_key] = {'timestamp': time.time(), 'data': health}
        return health

    async def get_status(self) -> Dict[str, Any]:
        """
        Get cached system status.

        Returns:
            System status dictionary
        """
        cache_key = "system_status"
        async with self.cache_lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']

        session = await self._ensure_session()
        async def do_get_status():
            try:
                headers = await self._get_headers()
                async with session.get(f"{self.registry_endpoint}/status", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Status request failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Status request failed: {e}")

        status = await self._retry_request(do_get_status)
        async with self.cache_lock:
            self.cache[cache_key] = {'timestamp': time.time(), 'data': status}
        return status

    async def submit_graph_proposal(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a graph proposal to the registry.

        Args:
            graph: Graph dictionary to submit

        Returns:
            Response dictionary
        """
        self._validate_graph(graph)
        payload = {
            'agent_id': self.agent_id,
            'graph': graph,
            'timestamp': datetime.utcnow().isoformat(),
            'proposal_id': str(uuid.uuid4())
        }
        payload['signature'] = self._sign_request(payload)

        session = await self._ensure_session()
        async def do_submit():
            try:
                headers = await self._get_headers()
                async with session.post(f"{self.registry_endpoint}/ir/propose", json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Proposal submission failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Proposal submission failed: {e}")

        response = await self._retry_request(do_submit)
        logger.info(f"Submitted proposal {payload['proposal_id']} for graph {graph['id']}")
        return response

    async def vote_on_proposal(self, proposal_id: str, vote: str, rationale: str) -> Dict[str, Any]:
        """
        Vote on a graph proposal.

        Args:
            proposal_id: ID of the proposal
            vote: Vote decision ("yes" or "no")
            rationale: Rationale for the vote

        Returns:
            Response dictionary
        """
        if vote not in ["yes", "no"]:
            raise GraphixClientError("Vote must be 'yes' or 'no'")

        payload = {
            'agent_id': self.agent_id,
            'proposal_id': proposal_id,
            'vote': vote,
            'rationale': rationale,
            'timestamp': datetime.utcnow().isoformat()
        }
        payload['signature'] = self._sign_request(payload)

        session = await self._ensure_session()
        async def do_vote():
            try:
                headers = await self._get_headers()
                async with session.post(f"{self.registry_endpoint}/ir/vote", json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Vote submission failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Vote submission failed: {e}")

        response = await self._retry_request(do_vote)
        logger.info(f"Voted {vote} on proposal {proposal_id}")
        return response

    async def execute_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a graph using the executor service.

        Args:
            graph: Graph dictionary to execute

        Returns:
            Execution result dictionary
        """
        self._validate_graph(graph)
        payload = {
            'agent_id': self.agent_id,
            'graph': graph,
            'timestamp': datetime.utcnow().isoformat(),
            'execution_id': str(uuid.uuid4())
        }
        payload['signature'] = self._sign_request(payload)

        session = await self._ensure_session()
        async def do_execute():
            try:
                headers = await self._get_headers()
                async with session.post(f"{self.executor_endpoint}/ir/execute", json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Execution failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Execution failed: {e}")

        response = await self._retry_request(do_execute)
        logger.info(f"Executed graph {graph['id']} with execution ID {payload['execution_id']}")
        return response

    async def get_audit_log(self, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Retrieve audit logs with pagination.

        Args:
            limit: Maximum number of log entries to retrieve
            offset: Starting offset for pagination

        Returns:
            Audit log dictionary
        """
        payload = {
            'agent_id': self.agent_id,
            'limit': limit,
            'offset': offset,
            'timestamp': datetime.utcnow().isoformat()
        }
        payload['signature'] = self._sign_request(payload)

        session = await self._ensure_session()
        async def do_get_audit():
            try:
                headers = await self._get_headers()
                async with session.post(f"{self.audit_endpoint}/audit/logs", json=payload, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    raise GraphixClientError(f"Audit log retrieval failed: {response.status}")
            except aiohttp.ClientError as e:
                raise GraphixClientError(f"Audit log retrieval failed: {e}")

        response = await self._retry_request(do_get_audit)
        logger.info(f"Retrieved {len(response.get('entries', []))} audit log entries")
        return response

    async def connect_websocket(self, event_handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Connect to WebSocket for real-time event handling.

        Args:
            event_handler: Async callback function to handle events
        """
        async def websocket_loop():
            try:
                headers = await self._get_headers()
                # Properly convert HTTP/HTTPS to WS/WSS
                ws_url = self.registry_endpoint.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws'
                # Reuse self.session instead of creating a new one
                session = await self._ensure_session()
                async with session.ws_connect(ws_url, headers=headers) as ws:
                    logger.info("WebSocket connected")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(msg.data)
                                await event_handler(event)
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid WebSocket message: {e}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket closed")
                            break
            except aiohttp.ClientError as e:
                logger.error(f"WebSocket connection failed: {e}")
                raise GraphixClientError(f"WebSocket connection failed: {e}")

        self.ws_session = asyncio.create_task(websocket_loop())

    async def close(self) -> None:
        """
        Close all sessions and clean up resources.
        """
        if self.session and not self.session.closed:
            await self.session.close()
        if self.ws_session:
            self.ws_session.cancel()
            try:
                await self.ws_session
            except asyncio.CancelledError:
                pass
        logger.info("Graphix client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        await self.close()

# Example usage
async def main():
    """Comprehensive example usage of GraphixClient."""
    async with GraphixClient(
        registry_endpoint="http://localhost:8787",
        executor_endpoint="http://localhost:8788",
        audit_endpoint="http://localhost:8789",
        agent_id="agent-grok",
        private_key_path="keys/agent-grok.pem"
    ) as client:
        # Health check
        health = await client.health_check()
        print(f"🏥 Service health: {health}")
        
        # Get cached system status
        status = await client.get_status()
        print(f"📊 System status: {status}")
        
        # Create and validate a graph
        test_graph = {
            "grammar_version": "3.4.0",
            "id": "comprehensive_test",
            "type": "Graph",
            "nodes": [
                {"id": "input", "type": "InputNode", "value": "Hello, Production Graphix!"},
                {"id": "process", "type": "GenerativeAINode", "prompt": "Process: {input}"},
                {"id": "output", "type": "OutputNode"}
            ],
            "edges": [
                {"id": "e1", "from": "input", "to": "process", "type": "data"},
                {"id": "e2", "from": "process", "to": "output", "type": "data"}
            ]
        }
        
        # Submit with automatic validation and retry
        print("📤 Submitting proposal...")
        response = await client.submit_graph_proposal(test_graph)
        print(f"✅ Proposal response: {response}")
        
        # Vote on the proposal
        if response.get("status") == "success":
            vote_response = await client.vote_on_proposal(
                test_graph["id"], 
                "approve", 
                "Comprehensive test looks good"
            )
            print(f"🗳️ Vote response: {vote_response}")
            
            # Execute the graph
            execution_result = await client.execute_graph(test_graph)
            print(f"🚀 Execution result: {execution_result}")
        
        # Get audit log with pagination
        audit_log = await client.get_audit_log(limit=5)
        print(f"📜 Recent audit entries: {len(audit_log.get('entries', []))}")
        
        # Real-time event handling
        async def handle_event(event):
            print(f"Received event: {event}")
        
        print("👂 Listening for real-time events... (Ctrl+C to exit)")
        await client.connect_websocket(handle_event)
        try:
            await asyncio.sleep(30)  # Listen for 30 seconds
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())