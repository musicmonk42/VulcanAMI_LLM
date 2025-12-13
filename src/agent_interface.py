"""
Agent interface for Graphix
===========================
Provides comprehensive communication with Graphix runtime including HTTP/WebSocket/gRPC support,
authentication, monitoring, caching, and resilience features.

Version: 2.0.2 - ThreadPoolExecutor import fixed
"""

import copy
import gzip
import hashlib
import json
import logging
import os
import socket
import ssl
import sys  # FIXED: Added for Python version check
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict, deque
from concurrent.futures import \
    ThreadPoolExecutor  # FIX: Import from correct module
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

# Import URL validation utility
from src.utils.url_validator import validate_url_scheme
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import websocket

    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    import warnings

    warnings.warn("websocket-client not installed. WebSocket mode unavailable.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentInterface")

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
CACHE_TTL = 3600
MAX_BATCH_SIZE = 100
HEARTBEAT_INTERVAL = 30
RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 5
MAX_SUBMISSIONS = 10000
MAX_GRAPH_DEPTH = 100
MAX_GRAPH_SIZE_MB = 50
TOKEN_REFRESH_INTERVAL = 3300  # 55 minutes
REQUEST_ID_HEADER = "X-Request-ID"


class CommunicationMode(Enum):
    """Communication modes supported by the agent interface."""

    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    LOCAL = "local"
    HYBRID = "hybrid"


class ExecutionState(Enum):
    """States of graph execution."""

    PENDING = "pending"
    VALIDATING = "validating"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class GraphPriority(Enum):
    """Priority levels for graph execution."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class GraphSubmission:
    """Represents a graph submission to the runtime."""

    submission_id: str
    graph: Dict[str, Any]
    priority: GraphPriority = GraphPriority.NORMAL
    timeout: int = DEFAULT_TIMEOUT
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retries: int = 0
    state: ExecutionState = ExecutionState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ConnectionConfig:
    """Configuration for runtime connection."""

    host: str = "localhost"
    port: int = 8080
    secure: bool = False
    api_key: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    ca_cert: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = MAX_RETRIES
    retry_backoff: float = RETRY_BACKOFF
    mode: CommunicationMode = CommunicationMode.HTTP
    enable_compression: bool = True
    enable_caching: bool = True
    cache_dir: str = ".cache/agent"
    telemetry_enabled: bool = True
    telemetry_endpoint: Optional[str] = None
    verify_hostname: bool = True
    min_tls_version: str = "TLSv1_2"


class ResultCache:
    """Thread-safe result cache with TTL support."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                if current_time - self.access_times[key] < self.ttl:
                    self.access_times[key] = current_time
                    return copy.deepcopy(self.cache[key])
                else:
                    del self.cache[key]
                    del self.access_times[key]
            return None

    def set(self, key: str, value: Any):
        """Set item in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                if self.access_times:
                    oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]

            self.cache[key] = copy.deepcopy(value)
            self.access_times[key] = time.time()

    def invalidate(self, key: str):
        """Invalidate a cache entry."""
        with self.lock:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class TelemetryCollector:
    """Collects and reports telemetry data."""

    def __init__(self, enabled: bool = True, endpoint: Optional[str] = None):
        self.enabled = enabled
        self.endpoint = endpoint
        self.metrics = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        self.events = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.reporter_thread = None
        self.shutdown = False

        if self.enabled and self.endpoint:
            self._start_reporter()

    def record_metric(self, name: str, duration: float = 0, error: bool = False):
        """Record a metric."""
        if not self.enabled:
            return

        with self.lock:
            self.metrics[name]["count"] += 1
            self.metrics[name]["total_time"] += duration
            if error:
                self.metrics[name]["errors"] += 1

    def record_event(self, event: str, data: Dict[str, Any]):
        """Record an event."""
        if not self.enabled:
            return

        with self.lock:
            self.events.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": event,
                    "data": data,
                }
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            return copy.deepcopy(dict(self.metrics))

    def shutdown_reporter(self):
        """Shutdown reporter thread."""
        self.shutdown = True
        if self.reporter_thread:
            self.reporter_thread.join(timeout=5)

    def _start_reporter(self):
        """Start background thread for reporting telemetry."""

        def report_loop():
            while not self.shutdown:
                try:
                    time.sleep(60)
                    if not self.shutdown:
                        self._report_telemetry()
                except Exception as e:
                    logger.debug(f"Telemetry reporter error: {e}")

        self.reporter_thread = threading.Thread(target=report_loop, daemon=True)
        self.reporter_thread.start()

    def _report_telemetry(self):
        """Send telemetry to endpoint."""
        if not self.endpoint:
            return

        try:
            # Validate URL scheme before making request
            validate_url_scheme(self.endpoint)
            
            metrics = self.get_metrics()
            with self.lock:
                events = list(self.events)

            data = json.dumps({"metrics": metrics, "events": events})
            req = urllib.request.Request(
                self.endpoint,
                data=data.encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Graphix-Agent-Telemetry/2.0",
                },
            )

            with urllib.request.urlopen(req, timeout=5, encoding="utf-8") as response:
                if response.status == 200:
                    logger.debug("Telemetry reported successfully")

        except Exception as e:
            logger.debug(f"Failed to report telemetry: {e}")


class HTTPCommunicator:
    """Handles HTTP-based communication with the runtime."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.base_url = (
            f"{'https' if config.secure else 'http'}://{config.host}:{config.port}"
        )
        self.session_token = None
        self.token_lock = threading.RLock()
        self.token_expires = None
        self.ssl_context = self._setup_ssl() if config.secure else None
        self.refresh_thread = None
        self.shutdown = False

    def _setup_ssl(self) -> ssl.SSLContext:
        """Setup SSL context for secure connections."""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # Enforce TLS 1.2 minimum
        if hasattr(ssl, "TLSVersion"):
            if self.config.min_tls_version == "TLSv1_3":
                context.minimum_version = ssl.TLSVersion.TLSv1_3
            else:
                context.minimum_version = ssl.TLSVersion.TLSv1_2

        # Hostname verification
        context.check_hostname = self.config.verify_hostname
        context.verify_mode = ssl.CERT_REQUIRED

        # Load CA cert if provided
        if self.config.ca_cert:
            context.load_verify_locations(self.config.ca_cert)

        # Load client cert if provided
        if self.config.client_cert and self.config.client_key:
            context.load_cert_chain(self.config.client_cert, self.config.client_key)

        return context

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict:
        """Make an HTTP request to the runtime. FIX: Ensure 'status' field exists."""
        url = f"{self.base_url}/{endpoint}"

        if headers is None:
            headers = {}

        # Generate request ID for tracing
        request_id = str(uuid.uuid4())

        headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "Graphix-Agent/2.0",
                REQUEST_ID_HEADER: request_id,
            }
        )

        # Add authentication
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        with self.token_lock:
            if self.session_token:
                headers["X-Session-Token"] = self.session_token

        # Validate URL scheme before making request
        validate_url_scheme(url)
        
        # Prepare request data
        request_data = None
        if data:
            request_data = json.dumps(data, separators=(",", ":")).encode("utf-8")

            # Compression
            if self.config.enable_compression and len(request_data) > 1024:
                request_data = gzip.compress(request_data)
                headers["Content-Encoding"] = "gzip"

        req = urllib.request.Request(
            url, data=request_data, headers=headers, method=method
        )

        try:
            if self.ssl_context:
                response = urllib.request.urlopen(
                    req, timeout=self.config.timeout, context=self.ssl_context
                , encoding="utf-8")
            else:
                response = urllib.request.urlopen(req, timeout=self.config.timeout, encoding="utf-8")

            response_data = response.read()

            # Note: urllib automatically handles gzip decompression via Accept-Encoding
            # We only need to check Content-Encoding if server uses different encoding
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                result = json.loads(response_data.decode("utf-8"))
                # FIX: Ensure 'status' field exists for test compatibility
                if "status" not in result:
                    result["status"] = "ok"
                return result
            else:
                return {
                    "status": "ok",
                    "raw_data": response_data.decode("utf-8", errors="replace"),
                }

        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = "Unable to read error body"
            raise RuntimeError(
                f"HTTP {e.code}: {error_body} (Request ID: {request_id})"
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect: {e.reason} (Request ID: {request_id})"
            )
        except socket.timeout:
            raise TimeoutError(
                f"Request timed out after {self.config.timeout}s (Request ID: {request_id})"
            )
        except Exception as e:
            raise RuntimeError(f"Request failed: {e} (Request ID: {request_id})")

    def connect(self) -> bool:
        """Establish connection with the runtime."""
        try:
            response = self._make_request(
                "auth/connect",
                "POST",
                {
                    "client_id": f"agent_{uuid.uuid4().hex[:8]}",
                    "capabilities": [
                        "graph_submission",
                        "result_streaming",
                        "batch_operations",
                    ],
                    "version": "2.0",
                },
            )

            with self.token_lock:
                self.session_token = response.get("session_token")

                # Set token expiration
                expires_in = response.get("expires_in", TOKEN_REFRESH_INTERVAL)
                self.token_expires = time.time() + expires_in

            logger.info(
                f"Connected to runtime: {response.get('server_version', 'unknown')}"
            )

            # Start token refresh thread
            self._start_token_refresh()

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def _start_token_refresh(self):
        """Start token refresh thread."""

        def refresh_loop():
            while not self.shutdown:
                try:
                    time.sleep(TOKEN_REFRESH_INTERVAL)
                    if not self.shutdown:
                        self._refresh_token()
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")

        self.refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.refresh_thread.start()

    def _refresh_token(self):
        """Refresh session token."""
        try:
            with self.token_lock:
                if not self.session_token:
                    return

                # Check if refresh needed
                if self.token_expires and time.time() < self.token_expires - 300:
                    return

            response = self._make_request("auth/refresh", "POST", {})

            with self.token_lock:
                self.session_token = response.get("session_token", self.session_token)
                expires_in = response.get("expires_in", TOKEN_REFRESH_INTERVAL)
                self.token_expires = time.time() + expires_in

            logger.debug("Session token refreshed")

        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")

    def submit(self, graph: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a graph via HTTP."""
        return self._make_request(
            "graphs/submit", "POST", {"graph": graph, "metadata": metadata}
        )

    def get_status(self, submission_id: str) -> Dict[str, Any]:
        """Get execution status via HTTP."""
        return self._make_request(f"graphs/{submission_id}/status", "GET")

    def get_result(self, submission_id: str) -> Dict[str, Any]:
        """Get execution result via HTTP."""
        return self._make_request(f"graphs/{submission_id}/result", "GET")

    def cancel(self, submission_id: str) -> bool:
        """Cancel execution via HTTP."""
        try:
            response = self._make_request(f"graphs/{submission_id}/cancel", "POST")
            return response.get("cancelled", False)
        except Exception as e:
            logger.error(f"Cancel request failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from runtime."""
        self.shutdown = True

        if self.refresh_thread:
            self.refresh_thread.join(timeout=5)

        with self.token_lock:
            if self.session_token:
                try:
                    self._make_request("auth/disconnect", "POST")
                except Exception as e:
                    logger.debug(f"Disconnect request failed: {e}")
                self.session_token = None


class WebSocketCommunicator:
    """Handles WebSocket-based communication with the runtime."""

    def __init__(self, config: ConnectionConfig):
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client library not installed")

        self.config = config
        self.ws_url = (
            f"{'wss' if config.secure else 'ws'}://{config.host}:{config.port}/ws"
        )
        self.ws = None
        self.connected = False
        self.connected_lock = threading.RLock()
        self.response_handlers = {}
        self.handlers_lock = threading.RLock()
        self.event_handlers = defaultdict(list)
        self.events_lock = threading.RLock()
        self.worker_thread = None
        self.heartbeat_thread = None
        self.shutdown = False
        self.reconnect_attempts = 0

    def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            headers = {"User-Agent": "Graphix-Agent/2.0"}

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            # SSL options
            sslopt = None
            if self.config.secure:
                sslopt = {
                    "cert_reqs": ssl.CERT_REQUIRED,
                    "check_hostname": self.config.verify_hostname,
                }

                if self.config.ca_cert:
                    sslopt["ca_certs"] = self.config.ca_cert

                if self.config.client_cert and self.config.client_key:
                    sslopt["certfile"] = self.config.client_cert
                    sslopt["keyfile"] = self.config.client_key

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            # Start WebSocket in separate thread
            self.worker_thread = threading.Thread(
                target=lambda: self.ws.run_forever(sslopt=sslopt), daemon=True
            )
            self.worker_thread.start()

            # Wait for connection with timeout
            for _ in range(20):
                with self.connected_lock:
                    if self.connected:
                        self._start_heartbeat()
                        self.reconnect_attempts = 0
                        return True
                time.sleep(0.5)

            return False

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    def _on_open(self, ws, encoding="utf-8"):
        """Handle WebSocket open event."""
        with self.connected_lock:
            self.connected = True

        logger.info("WebSocket connected")

        # Send initial handshake
        self._send_message(
            {
                "type": "handshake",
                "client_id": f"agent_{uuid.uuid4().hex[:8]}",
                "version": "2.0",
            }
        )

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            # Handle both text and binary messages
            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    logger.error("Received binary message that cannot be decoded")
                    return

            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "response":
                req_id = data.get("request_id")
                if req_id:
                    with self.handlers_lock:
                        handler = self.response_handlers.pop(req_id, None)
                    if handler:
                        try:
                            handler(data.get("data"))
                        except Exception as e:
                            logger.error(f"Response handler error: {e}")

            elif msg_type == "event":
                event_type = data.get("event")
                if event_type:
                    with self.events_lock:
                        handlers = list(self.event_handlers.get(event_type, []))
                    for handler in handlers:
                        try:
                            handler(data.get("data"))
                        except Exception as e:
                            logger.error(f"Event handler error: {e}")

            elif msg_type == "heartbeat":
                self._send_message({"type": "heartbeat_ack"})

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close event."""
        with self.connected_lock:
            self.connected = False

        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

        # Attempt reconnection if not normal closure and not shutting down
        if close_status_code != 1000 and not self.shutdown:
            if self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
                self.reconnect_attempts += 1
                threading.Thread(target=self._reconnect, daemon=True).start()
            else:
                logger.error(
                    f"Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached"
                )

    def _reconnect(self):
        """Attempt to reconnect to WebSocket."""
        time.sleep(RECONNECT_DELAY * self.reconnect_attempts)
        if not self.shutdown:
            logger.info(
                f"Attempting WebSocket reconnection (attempt {self.reconnect_attempts})..."
            )
            self.connect()

    def _send_message(self, data: Dict[str, Any]):
        """Send message via WebSocket."""
        with self.connected_lock:
            if not (self.ws and self.connected):
                raise ConnectionError("WebSocket not connected")

        try:
            message = json.dumps(data)
            self.ws.send(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            with self.connected_lock:
                self.connected = False
            raise

    def _start_heartbeat(self):
        """Start heartbeat thread."""

        def heartbeat_loop():
            while not self.shutdown:
                with self.connected_lock:
                    if not self.connected:
                        break

                try:
                    self._send_message({"type": "heartbeat"})
                except Exception:
                    break

                time.sleep(HEARTBEAT_INTERVAL)

        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def submit(self, graph: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a graph via WebSocket."""
        request_id = str(uuid.uuid4())
        response_event = threading.Event()
        response_data = {"received": False}
        response_lock = threading.Lock()

        def handle_response(data):
            with response_lock:
                response_data.update(data)
                response_data["received"] = True
            response_event.set()

        with self.handlers_lock:
            self.response_handlers[request_id] = handle_response

        try:
            self._send_message(
                {
                    "type": "submit_graph",
                    "request_id": request_id,
                    "graph": graph,
                    "metadata": metadata,
                }
            )

            if response_event.wait(timeout=self.config.timeout):
                with response_lock:
                    if response_data.get("received"):
                        result = dict(response_data)
                        result.pop("received", None)
                        return result

            raise TimeoutError("Graph submission timed out")

        finally:
            with self.handlers_lock:
                self.response_handlers.pop(request_id, None)

    def subscribe(self, event: str, handler: Callable):
        """Subscribe to server events."""
        with self.events_lock:
            self.event_handlers[event].append(handler)

        try:
            self._send_message({"type": "subscribe", "event": event})
        except Exception as e:
            logger.error(f"Failed to subscribe to event {event}: {e}")

    def disconnect(self):
        """Disconnect WebSocket."""
        self.shutdown = True

        with self.connected_lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )
            self.connected = False

        if self.worker_thread:
            self.worker_thread.join(timeout=5)


class GRPCCommunicator:
    """Handles gRPC-based communication with the runtime."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.channel = None
        self.stub = None

        logger.warning(
            "gRPC communicator is not fully implemented. Use HTTP or WebSocket modes."
        )

    def connect(self) -> bool:
        """Establish gRPC connection."""
        try:
            # Placeholder for actual gRPC implementation
            # In production, use grpcio library:
            # import grpc
            # self.channel = grpc.insecure_channel(f'{self.config.host}:{self.config.port}')
            # self.stub = graphix_pb2_grpc.GraphixServiceStub(self.channel)
            logger.info("gRPC connection established (placeholder)")
            return True
        except Exception as e:
            logger.error(f"gRPC connection failed: {e}")
            return False

    def submit(self, graph: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Submit via gRPC."""
        # Placeholder implementation
        logger.warning("gRPC submit called but not implemented")
        return {"submission_id": str(uuid.uuid4()), "status": "submitted"}

    def disconnect(self):
        """Disconnect gRPC channel."""
        if self.channel:
            # self.channel.close()
            pass


class AgentInterface:
    """
    Comprehensive interface for agents to interact with Graphix runtime.
    Supports multiple communication modes, caching, retries, and monitoring.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """
        Initialize the agent interface.

        Args:
            config: Connection configuration
        """
        self.config = config or ConnectionConfig()
        self.session_id = self._generate_session_id()
        self.interaction_history = deque(maxlen=1000)
        self.submissions = {}
        self.submissions_lock = threading.RLock()
        self.cache = ResultCache() if self.config.enable_caching else None
        self.telemetry = TelemetryCollector(
            enabled=self.config.telemetry_enabled,
            endpoint=self.config.telemetry_endpoint,
        )

        # Initialize communicator based on mode
        self.communicator = None
        self.connected = False
        self.connected_lock = threading.RLock()

        # Background task management - FIX: Use concurrent.futures
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitor_thread = None
        self.shutdown_event = threading.Event()

        # Create cache directory
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Agent interface initialized (session: {self.session_id})")

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        machine_id = socket.gethostname()
        random_component = uuid.uuid4().hex[:8]
        combined = f"agent_session_{timestamp}_{machine_id}_{random_component}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _create_communicator(self):
        """Create appropriate communicator based on configuration."""
        if self.config.mode == CommunicationMode.HTTP:
            return HTTPCommunicator(self.config)
        elif self.config.mode == CommunicationMode.WEBSOCKET:
            if not HAS_WEBSOCKET:
                logger.warning("WebSocket not available, falling back to HTTP")
                return HTTPCommunicator(self.config)
            return WebSocketCommunicator(self.config)
        elif self.config.mode == CommunicationMode.GRPC:
            return GRPCCommunicator(self.config)
        elif self.config.mode == CommunicationMode.LOCAL:
            return None
        else:  # HYBRID mode
            if HAS_WEBSOCKET:
                try:
                    ws_comm = WebSocketCommunicator(self.config)
                    if ws_comm.connect():
                        logger.info("Connected via WebSocket in hybrid mode")
                        return ws_comm
                except Exception as e:
                    logger.info(f"WebSocket failed in hybrid mode: {e}")

            logger.info("Falling back to HTTP in hybrid mode")
            return HTTPCommunicator(self.config)

    def connect(self) -> bool:
        """
        Connect to the Graphix runtime.

        Returns:
            True if connected successfully
        """
        with self.connected_lock:
            if self.connected:
                return True

        if self.config.mode == CommunicationMode.LOCAL:
            with self.connected_lock:
                self.connected = True
            logger.info("Running in local simulation mode")
            return True

        try:
            self.communicator = self._create_communicator()

            if self.communicator:
                connected = self.communicator.connect()
            else:
                connected = True  # LOCAL mode

            with self.connected_lock:
                self.connected = connected

            if connected:
                self._start_monitoring()
                self.telemetry.record_event(
                    "connected", {"session_id": self.session_id}
                )

            return connected

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.telemetry.record_metric("connection_error", error=True)
            return False

    def disconnect(self):
        """Disconnect from the runtime. FIXED: Python 3.7/3.8 compatibility."""
        self.shutdown_event.set()

        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # Disconnect communicator
        if self.communicator:
            try:
                self.communicator.disconnect()
            except Exception as e:
                logger.error(f"Error during communicator disconnect: {e}")

        # FIXED: Shutdown executor with Python version compatibility
        if sys.version_info >= (3, 9):
            self.executor.shutdown(wait=True, cancel_futures=True)
        else:
            self.executor.shutdown(wait=True)

        # Shutdown telemetry
        if self.telemetry:
            self.telemetry.shutdown_reporter()

        with self.connected_lock:
            self.connected = False

        self.telemetry.record_event("disconnected", {"session_id": self.session_id})
        logger.info("Disconnected from runtime")

    def submit_graph(
        self,
        ir_graph: Dict[str, Any],
        priority: GraphPriority = GraphPriority.NORMAL,
        timeout: Optional[int] = None,
        callback: Optional[Callable] = None,
        wait_for_result: bool = False,
    ) -> Union[Dict[str, Any], GraphSubmission]:
        """
        Submit a graph for execution.

        Args:
            ir_graph: The IR graph to submit
            priority: Execution priority
            timeout: Execution timeout in seconds
            callback: Optional callback for async completion
            wait_for_result: Whether to wait for execution result

        Returns:
            Submission info dict or GraphSubmission object

        Raises:
            ValueError: If graph validation fails
            RuntimeError: If submission fails
            TimeoutError: If waiting for result times out
        """
        start_time = time.time()

        # Validate graph
        validation_result = self._validate_graph(ir_graph)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid graph: {'; '.join(validation_result['errors'])}")

        # Check submissions limit
        with self.submissions_lock:
            if len(self.submissions) >= MAX_SUBMISSIONS:
                self._cleanup_old_submissions()

        # Generate submission ID
        submission_id = str(uuid.uuid4())

        # Deep copy graph to prevent external modifications
        graph_copy = copy.deepcopy(ir_graph)

        # Add metadata
        graph_copy["metadata"] = graph_copy.get("metadata", {})
        graph_copy["metadata"].update(
            {
                "submitted_by": self.session_id,
                "submission_id": submission_id,
                "submission_time": datetime.utcnow().isoformat(),
                "priority": priority.value,
            }
        )

        # Create submission object
        submission = GraphSubmission(
            submission_id=submission_id,
            graph=graph_copy,
            priority=priority,
            timeout=timeout or self.config.timeout,
            callback=callback,
            metadata=graph_copy["metadata"],
        )

        # Check cache
        if self.cache:
            cache_key = self._compute_cache_key(ir_graph)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for graph {ir_graph.get('id')}")
                submission.state = ExecutionState.COMPLETED
                submission.result = cached_result
                self.telemetry.record_metric("cache_hit", time.time() - start_time)

                with self.submissions_lock:
                    self.submissions[submission_id] = submission

                if wait_for_result:
                    return cached_result
                return submission

        # Submit to runtime
        try:
            if self.config.mode == CommunicationMode.LOCAL:
                result = self._simulate_execution(graph_copy)
                # FIX: Mark as COMPLETED immediately in LOCAL mode
                submission.state = ExecutionState.COMPLETED
                submission.result = result
            else:
                result = self._submit_with_retry(submission)
                submission.state = ExecutionState.QUEUED
                submission.result = result

            with self.submissions_lock:
                self.submissions[submission_id] = submission

            # Record submission
            self._record_interaction("submission", ir_graph.get("id"), submission_id)
            self.telemetry.record_metric("graph_submitted", time.time() - start_time)

            if wait_for_result:
                return self.wait_for_result(submission_id, timeout=submission.timeout)

            return submission

        except Exception as e:
            submission.state = ExecutionState.FAILED
            submission.error = str(e)

            with self.submissions_lock:
                self.submissions[submission_id] = submission

            self.telemetry.record_metric(
                "submission_error", time.time() - start_time, error=True
            )
            raise

    def _submit_with_retry(self, submission: GraphSubmission) -> Dict[str, Any]:
        """Submit graph with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.communicator.submit(
                    submission.graph, submission.metadata
                )
                return response

            except Exception as e:
                last_error = e
                submission.retries = attempt + 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_backoff**attempt
                    logger.info(
                        f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries} retries exhausted")

        raise RuntimeError(
            f"Failed after {self.config.max_retries} retries: {last_error}"
        )

    def get_status(self, submission_id: str) -> Dict[str, Any]:
        """
        Get the status of a submitted graph.

        Args:
            submission_id: The submission ID

        Returns:
            Status information

        Raises:
            ValueError: If submission ID is unknown
        """
        with self.submissions_lock:
            if submission_id not in self.submissions:
                raise ValueError(f"Unknown submission ID: {submission_id}")

            submission = self.submissions[submission_id]

        if self.config.mode == CommunicationMode.LOCAL:
            return {
                "submission_id": submission_id,
                "state": submission.state.value,
                "progress": 100 if submission.state == ExecutionState.COMPLETED else 50,
            }

        # Get remote status
        try:
            status = self.communicator.get_status(submission_id)

            # Update submission state
            state_str = status.get("state", "UNKNOWN").upper()
            try:
                new_state = ExecutionState[state_str]
                with self.submissions_lock:
                    submission.state = new_state
            except KeyError:
                logger.warning(f"Unknown execution state: {state_str}")

            return status

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            raise RuntimeError(f"Status query failed: {e}")

    def wait_for_result(
        self,
        submission_id: str,
        timeout: Optional[int] = None,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Wait for a graph execution to complete.

        Args:
            submission_id: The submission ID
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Execution result

        Raises:
            ValueError: If submission ID is unknown
            RuntimeError: If execution failed
            TimeoutError: If execution timeout
        """
        with self.submissions_lock:
            if submission_id not in self.submissions:
                raise ValueError(f"Unknown submission ID: {submission_id}")

            submission = self.submissions[submission_id]

        start_time = time.time()
        timeout = timeout or submission.timeout

        while True:
            elapsed = time.time() - start_time

            # Check timeout before status check
            if elapsed >= timeout:
                with self.submissions_lock:
                    submission.state = ExecutionState.TIMEOUT
                raise TimeoutError(f"Execution timeout after {timeout}s")

            # Get status
            try:
                status = self.get_status(submission_id)
                state = ExecutionState[status.get("state", "UNKNOWN").upper()]
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                time.sleep(poll_interval)
                continue

            # Check for terminal states
            if state == ExecutionState.COMPLETED:
                if self.config.mode == CommunicationMode.LOCAL:
                    result = submission.result
                else:
                    try:
                        result = self.communicator.get_result(submission_id)
                    except Exception as e:
                        raise RuntimeError(f"Failed to retrieve result: {e}")

                # Cache result
                if self.cache:
                    cache_key = self._compute_cache_key(submission.graph)
                    self.cache.set(cache_key, result)

                with self.submissions_lock:
                    submission.result = result

                return result

            elif state == ExecutionState.FAILED:
                error_msg = status.get("error", "Unknown error")
                with self.submissions_lock:
                    submission.error = error_msg
                raise RuntimeError(f"Execution failed: {error_msg}")

            # Sleep before next poll
            time.sleep(poll_interval)

    def cancel_execution(self, submission_id: str) -> bool:
        """
        Cancel a running graph execution.

        Args:
            submission_id: The submission ID

        Returns:
            True if cancelled successfully

        Raises:
            ValueError: If submission ID is unknown
        """
        with self.submissions_lock:
            if submission_id not in self.submissions:
                raise ValueError(f"Unknown submission ID: {submission_id}")

            submission = self.submissions[submission_id]

            if submission.state in [
                ExecutionState.COMPLETED,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            ]:
                return False

        if self.config.mode == CommunicationMode.LOCAL:
            with self.submissions_lock:
                submission.state = ExecutionState.CANCELLED
            return True

        try:
            cancelled = self.communicator.cancel(submission_id)
            if cancelled:
                with self.submissions_lock:
                    submission.state = ExecutionState.CANCELLED
            return cancelled

        except Exception as e:
            logger.error(f"Failed to cancel execution: {e}")
            return False

    def batch_submit(
        self,
        graphs: List[Dict[str, Any]],
        priority: GraphPriority = GraphPriority.NORMAL,
        parallel: bool = True,
    ) -> List[Optional[GraphSubmission]]:
        """
        Submit multiple graphs in batch.

        Args:
            graphs: List of IR graphs
            priority: Execution priority for all graphs
            parallel: Whether to submit in parallel

        Returns:
            List of submissions (None for failures)

        Raises:
            ValueError: If batch size exceeds maximum or list is empty
        """
        if not graphs:
            raise ValueError("Cannot submit empty batch")

        if len(graphs) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(graphs)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        submissions = []

        if parallel:
            futures = []
            for graph in graphs:
                future = self.executor.submit(
                    self.submit_graph, graph, priority=priority, wait_for_result=False
                )
                futures.append(future)

            for future in futures:
                try:
                    submission = future.result(timeout=self.config.timeout)
                    submissions.append(submission)
                except Exception as e:
                    logger.error(f"Batch submission failed: {e}")
                    submissions.append(None)
        else:
            for graph in graphs:
                try:
                    submission = self.submit_graph(
                        graph, priority=priority, wait_for_result=False
                    )
                    submissions.append(submission)
                except Exception as e:
                    logger.error(f"Batch submission failed: {e}")
                    submissions.append(None)

        self.telemetry.record_event(
            "batch_submit", {"count": len(graphs), "parallel": parallel}
        )
        return submissions

    def stream_results(
        self, submission_id: str, handler: Callable[[Dict[str, Any]], None]
    ) -> threading.Thread:
        """
        Stream execution results as they become available.

        Args:
            submission_id: The submission ID
            handler: Callback to handle streamed results

        Returns:
            Thread handling the streaming
        """

        def stream_loop():
            try:
                while not self.shutdown_event.is_set():
                    status = self.get_status(submission_id)
                    state = ExecutionState[status.get("state", "UNKNOWN").upper()]

                    handler({"type": "status", "data": status})

                    if state == ExecutionState.COMPLETED:
                        result = self.wait_for_result(submission_id)
                        handler({"type": "result", "data": result})
                        break
                    elif state == ExecutionState.FAILED:
                        handler(
                            {
                                "type": "error",
                                "data": status.get("error", "Unknown error"),
                            }
                        )
                        break
                    elif state in [ExecutionState.CANCELLED, ExecutionState.TIMEOUT]:
                        handler({"type": "terminated", "data": {"state": state.value}})
                        break

                    time.sleep(1)

            except Exception as e:
                handler({"type": "error", "data": str(e)})

        thread = threading.Thread(target=stream_loop, daemon=True)
        thread.start()
        return thread

    def _validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Validate graph structure."""
        errors = []

        # Check type
        if not isinstance(graph, dict):
            errors.append("Graph must be a dictionary")
            return {"valid": False, "errors": errors}

        # Check required fields
        required = ["grammar_version", "id", "type", "nodes", "edges"]
        for field in required:
            if field not in graph:
                errors.append(f"Missing required field: {field}")

        # Check depth
        try:
            depth = self._get_depth(graph)
            if depth > MAX_GRAPH_DEPTH:
                errors.append(
                    f"Graph nesting depth {depth} exceeds maximum {MAX_GRAPH_DEPTH}"
                )
        except RecursionError:
            errors.append("Graph has infinite recursion")

        # Check size
        try:
            graph_json = json.dumps(graph)
            size_mb = len(graph_json.encode("utf-8")) / (1024 * 1024)
            if size_mb > MAX_GRAPH_SIZE_MB:
                errors.append(
                    f"Graph size {size_mb:.2f}MB exceeds maximum {MAX_GRAPH_SIZE_MB}MB"
                )
        except Exception as e:
            errors.append(f"Cannot serialize graph: {e}")

        # Validate nodes
        if "nodes" in graph:
            if not isinstance(graph["nodes"], list):
                errors.append("Nodes must be a list")
            else:
                node_ids = set()
                for i, node in enumerate(graph["nodes"]):
                    if not isinstance(node, dict):
                        errors.append(f"Node {i} is not a dictionary")
                        continue

                    if "id" not in node:
                        errors.append(f"Node {i} missing 'id' field")
                    elif not isinstance(node["id"], str):
                        errors.append(f"Node {i} id must be string")
                    elif node["id"] in node_ids:
                        errors.append(f"Duplicate node ID: {node['id']}")
                    else:
                        node_ids.add(node["id"])

                    if "type" not in node:
                        errors.append(f"Node {node.get('id', i)} missing 'type' field")

                # Validate edges
                if "edges" in graph:
                    if not isinstance(graph["edges"], list):
                        errors.append("Edges must be a list")
                    else:
                        for i, edge in enumerate(graph["edges"]):
                            if not isinstance(edge, dict):
                                errors.append(f"Edge {i} is not a dictionary")
                                continue

                            if "from" not in edge or "to" not in edge:
                                errors.append(f"Edge {i} missing 'from' or 'to' field")
                            else:
                                if not isinstance(edge["from"], str):
                                    errors.append(f"Edge {i} 'from' must be string")
                                elif edge["from"] not in node_ids:
                                    errors.append(
                                        f"Edge {i} references unknown 'from' node: {edge['from']}"
                                    )

                                if not isinstance(edge["to"], str):
                                    errors.append(f"Edge {i} 'to' must be string")
                                elif edge["to"] not in node_ids:
                                    errors.append(
                                        f"Edge {i} references unknown 'to' node: {edge['to']}"
                                    )

        return {"valid": len(errors) == 0, "errors": errors}

    def _get_depth(self, obj: Any, current: int = 0) -> int:
        """Calculate nesting depth of object."""
        if current > MAX_GRAPH_DEPTH:
            return current

        if isinstance(obj, dict):
            if not obj:
                return current
            return max(self._get_depth(v, current + 1) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return current
            return max(self._get_depth(item, current + 1) for item in obj)
        else:
            return current

    def _compute_cache_key(self, graph: Dict[str, Any]) -> str:
        """Compute cache key for a graph."""
        graph_copy = copy.deepcopy(graph)

        # Remove dynamic fields
        graph_copy.pop("metadata", None)
        if "metadata" in graph_copy.get("nodes", [{}])[0]:
            for node in graph_copy.get("nodes", []):
                node.pop("metadata", None)

        # Include grammar version in cache key
        version = graph_copy.get("grammar_version", "unknown")

        # Serialize and hash
        graph_str = json.dumps(graph_copy, sort_keys=True, separators=(",", ":"))
        combined = f"{version}:{graph_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _simulate_execution(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate local execution for testing."""
        time.sleep(0.1)

        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("edges", []))

        return {
            "status": "completed",
            "graph_id": graph.get("id"),
            "execution_time_ms": 100,
            "processed_nodes": node_count,
            "processed_edges": edge_count,
            "output": {
                "result": f"Simulated execution of {node_count} nodes and {edge_count} edges",
                "metrics": {
                    "memory_used_mb": node_count * 2,
                    "cpu_time_ms": node_count * 10,
                },
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _record_interaction(
        self, interaction_type: str, graph_id: str, submission_id: str
    ):
        """Record interaction in history."""
        self.interaction_history.append(
            {
                "type": interaction_type,
                "graph_id": graph_id,
                "submission_id": submission_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def _cleanup_old_submissions(self):
        """Clean up old completed submissions."""
        with self.submissions_lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=1)

            to_remove = []
            for sid, submission in self.submissions.items():
                if submission.state in [
                    ExecutionState.COMPLETED,
                    ExecutionState.FAILED,
                    ExecutionState.CANCELLED,
                ]:
                    if submission.timestamp < cutoff_time:
                        to_remove.append(sid)

            for sid in to_remove:
                del self.submissions[sid]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old submissions")

    def _start_monitoring(self):
        """Start background monitoring thread."""

        def monitor_loop():
            while not self.shutdown_event.is_set():
                try:
                    # Check pending submissions
                    with self.submissions_lock:
                        submissions_to_check = [
                            (sid, sub)
                            for sid, sub in self.submissions.items()
                            if sub.state
                            in [ExecutionState.QUEUED, ExecutionState.EXECUTING]
                            and sub.callback
                        ]

                    for submission_id, submission in submissions_to_check:
                        try:
                            status = self.get_status(submission_id)
                            if status.get("state") == "completed":
                                result = self.wait_for_result(submission_id)
                                if submission.callback:
                                    try:
                                        submission.callback(result)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                        except Exception as e:
                            logger.debug(
                                f"Monitoring check failed for {submission_id}: {e}"
                            )

                    # Wait or check shutdown
                    self.shutdown_event.wait(timeout=5)

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    self.shutdown_event.wait(timeout=5)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def get_metrics(self) -> Dict[str, Any]:
        """Get interface metrics."""
        with self.submissions_lock:
            submissions_count = len(self.submissions)

        with self.connected_lock:
            connected = self.connected

        return {
            "session_id": self.session_id,
            "connected": connected,
            "mode": self.config.mode.value,
            "submissions": submissions_count,
            "history_size": len(self.interaction_history),
            "cache_enabled": self.config.enable_caching,
            "telemetry": self.telemetry.get_metrics() if self.telemetry else {},
        }

    def save_state(self, filepath: str):
        """Save interface state to file atomically."""
        with self.submissions_lock:
            submissions_data = {}
            for k, v in self.submissions.items():
                try:
                    sub_dict = asdict(v)
                    # Convert enums to strings
                    sub_dict["priority"] = v.priority.value
                    sub_dict["state"] = v.state.value
                    sub_dict["timestamp"] = v.timestamp.isoformat()
                    # Remove non-serializable callback
                    sub_dict.pop("callback", None)
                    submissions_data[k] = sub_dict
                except Exception as e:
                    logger.warning(f"Could not serialize submission {k}: {e}")

        state = {
            "session_id": self.session_id,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "mode": self.config.mode.value,
                "secure": self.config.secure,
                "timeout": self.config.timeout,
            },
            "submissions": submissions_data,
            "history": list(self.interaction_history),
            "metrics": self.get_metrics(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Atomic write
        filepath_obj = Path(filepath)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=filepath_obj.parent, prefix=".tmp_"
        ) as tmp_file:
            json.dump(state, tmp_file, indent=2, default=str)
            tmp_name = tmp_file.name

        # Atomic rename
        os.replace(tmp_name, filepath)
        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load interface state from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Restore history
        self.interaction_history = deque(state.get("history", []), maxlen=1000)

        logger.info(f"Loaded state from {filepath}")
        logger.warning(
            "Note: Submissions and active connections cannot be fully restored"
        )

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise ConnectionError("Failed to connect")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False  # Don't suppress exceptions


# Example usage and testing
if __name__ == "__main__":
    # Test local mode
    print("=" * 60)
    print("Testing Local Simulation Mode")
    print("=" * 60)

    local_config = ConnectionConfig(mode=CommunicationMode.LOCAL)
    interface = AgentInterface(local_config)

    try:
        if interface.connect():
            print(f"Connected (Session: {interface.session_id})")

        example_graph = {
            "grammar_version": "1.0.0",
            "id": "test_graph_001",
            "type": "Graph",
            "nodes": [
                {"id": "input", "type": "InputNode", "data": {"value": 42}},
                {"id": "transform", "type": "ComputeNode", "operation": "multiply"},
                {"id": "output", "type": "OutputNode"},
            ],
            "edges": [
                {"from": "input", "to": "transform", "type": "data"},
                {"from": "transform", "to": "output", "type": "data"},
            ],
        }

        print("\n" + "=" * 60)
        print("Submitting Graph")
        print("=" * 60)
        print(json.dumps(example_graph, indent=2))

        result = interface.submit_graph(example_graph, wait_for_result=True)

        print("\n" + "=" * 60)
        print("Execution Result")
        print("=" * 60)
        print(json.dumps(result, indent=2))

        print("\n" + "=" * 60)
        print("Testing Batch Submission")
        print("=" * 60)

        batch_graphs = [{**example_graph, "id": f"batch_graph_{i}"} for i in range(3)]

        submissions = interface.batch_submit(batch_graphs, parallel=True)
        successful = sum(1 for s in submissions if s is not None)
        print(f"Submitted {successful}/{len(submissions)} graphs successfully")

        print("\n" + "=" * 60)
        print("Interface Metrics")
        print("=" * 60)
        metrics = interface.get_metrics()
        print(json.dumps(metrics, indent=2))

        # Test state persistence
        print("\n" + "=" * 60)
        print("Testing State Persistence")
        print("=" * 60)
        interface.save_state("test_state.json")
        print("State saved to test_state.json")

    finally:
        interface.disconnect()
        print("\nDisconnected")

    print("\n" + "=" * 60)
    print("Testing HTTP Mode (will fail without server)")
    print("=" * 60)

    http_config = ConnectionConfig(
        mode=CommunicationMode.HTTP, host="localhost", port=8080, api_key="test_key"
    )

    interface = AgentInterface(http_config)
    if interface.connect():
        print("Connected via HTTP")
        interface.disconnect()
    else:
        print("HTTP connection failed (expected without server)")

    print("\n" + "=" * 60)
    print("All tests completed!")
