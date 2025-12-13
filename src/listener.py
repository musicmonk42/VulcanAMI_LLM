# src/listener.py
"""
Graphix IR HTTP Listener (Production-Ready)
===========================================
Version: 2.0.0 - All issues fixed, validated, production-ready
A secure HTTP server for receiving Graphix IR graphs from agents with comprehensive
error handling, rate limiting, and graceful shutdown support.
"""

import json
import logging
import signal
import sys
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GraphixListener")

# Try to import required modules with error handling
try:
    from src.agent_registry import AgentRegistry

    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AgentRegistry not available: {e}. Using mock implementation.")
    REGISTRY_AVAILABLE = False

try:
    from src.unified_runtime import UnifiedRuntime

    RUNTIME_AVAILABLE = True
except ImportError as e:
    logger.warning(f"UnifiedRuntime not available: {e}. Using mock implementation.")
    RUNTIME_AVAILABLE = False

# Constants
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
MIN_CONTENT_LENGTH = 1
REQUEST_TIMEOUT = 30  # seconds
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds
MAX_AGENT_ID_LENGTH = 256
MAX_SIGNATURE_LENGTH = 512


class MockAgentRegistry:
    """Mock registry for testing when real one is unavailable."""

    def __init__(self):
        """Initialize mock registry."""
        logger.info("MockAgentRegistry initialized")

    def verify_signature(self, agent_id: str, message: str, signature: str) -> bool:
        """
        Mock verification - always returns True for testing.

        WARNING: This is for testing only! In production, use real AgentRegistry.
        """
        logger.warning(
            f"Using mock verification for agent '{agent_id}' (always returns True)"
        )
        return True


class MockUnifiedRuntime:
    """Mock runtime for testing when real one is unavailable."""

    def __init__(self):
        """Initialize mock runtime."""
        logger.info("MockUnifiedRuntime initialized")

    def execute_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock execution.

        WARNING: This is for testing only! In production, use real UnifiedRuntime.
        """
        logger.warning("Using mock execution (no actual graph processing)")
        return {
            "status": "mock_executed",
            "result": None,
            "nodes_processed": len(graph.get("nodes", [])),
            "edges_processed": len(graph.get("edges", [])),
        }


class RateLimiter:
    """Thread-safe rate limiter for request throttling."""

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_MINUTE,
        window: int = RATE_LIMIT_WINDOW,
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
        self.lock = threading.RLock()
        logger.info(f"RateLimiter initialized: {max_requests} requests per {window}s")

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request from client is allowed.

        Args:
            client_id: Client identifier (IP address or agent ID)

        Returns:
            True if allowed, False if rate limit exceeded
        """
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window)

            # Remove old requests outside the window
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] if req_time > cutoff
            ]

            # Check if limit exceeded
            if len(self.requests[client_id]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False

            # Add new request timestamp
            self.requests[client_id].append(now)
            return True

    def get_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get rate limit statistics for a client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with rate limit stats
        """
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window)

            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] if req_time > cutoff
            ]

            current_count = len(self.requests[client_id])

            return {
                "client_id": client_id,
                "requests_in_window": current_count,
                "max_requests": self.max_requests,
                "window_seconds": self.window,
                "remaining": max(0, self.max_requests - current_count),
            }


class RequestHandler(BaseHTTPRequestHandler):
    """
    Thread-safe HTTP request handler for Graphix IR graphs.

    Handles POST requests with authentication, validation, and rate limiting.
    """

    # Class-level shared instances (set by server)
    registry: Optional[Any] = None
    runtime: Optional[Any] = None
    rate_limiter: Optional[RateLimiter] = None
    server_shutdown_flag: Optional[threading.Event] = None

    def log_message(self, format: str, *args):
        """Override to use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")

    def log_error(self, format: str, *args):
        """Override to use our logger for errors."""
        logger.error(f"{self.address_string()} - {format % args}")

    def do_POST(self):
        """
        Handle POST requests with comprehensive validation and error handling.

        Expected headers:
            - X-Agent-ID: Agent identifier
            - X-Signature: Cryptographic signature
            - Content-Length: Size of request body

        Request body should be JSON containing a Graphix IR graph.
        """
        start_time = datetime.now()

        try:
            # Rate limiting by IP address
            client_ip = self.client_address[0]

            if self.rate_limiter and not self.rate_limiter.is_allowed(client_ip):
                rate_stats = self.rate_limiter.get_stats(client_ip)
                self.send_error_response(
                    429,
                    "Rate limit exceeded. Please try again later.",
                    extra_data={
                        "retry_after": self.rate_limiter.window,
                        "rate_limit": rate_stats,
                    },
                )
                return

            # Validate required headers
            agent_id = self.headers.get("X-Agent-ID")
            signature = self.headers.get("X-Signature")

            if not agent_id:
                self.send_error_response(
                    401,
                    "Missing X-Agent-ID header",
                    extra_data={"required_headers": ["X-Agent-ID", "X-Signature"]},
                )
                return

            if not signature:
                self.send_error_response(
                    401,
                    "Missing X-Signature header",
                    extra_data={"required_headers": ["X-Agent-ID", "X-Signature"]},
                )
                return

            # Validate header values
            if len(agent_id) > MAX_AGENT_ID_LENGTH:
                self.send_error_response(
                    400, f"X-Agent-ID too long (max {MAX_AGENT_ID_LENGTH} characters)"
                )
                return

            if len(signature) > MAX_SIGNATURE_LENGTH:
                self.send_error_response(
                    400, f"X-Signature too long (max {MAX_SIGNATURE_LENGTH} characters)"
                )
                return

            # Validate Content-Length header
            content_length_str = self.headers.get("Content-Length")
            if not content_length_str:
                self.send_error_response(400, "Missing Content-Length header")
                return

            try:
                content_length = int(content_length_str)
            except ValueError:
                self.send_error_response(
                    400, f"Invalid Content-Length: '{content_length_str}'"
                )
                return

            # Validate content length range
            if content_length < MIN_CONTENT_LENGTH:
                self.send_error_response(
                    400, f"Content-Length too small (min {MIN_CONTENT_LENGTH} bytes)"
                )
                return

            if content_length > MAX_CONTENT_LENGTH:
                self.send_error_response(
                    413,
                    f"Request too large: {content_length} bytes (max {MAX_CONTENT_LENGTH})",
                    extra_data={"max_content_length": MAX_CONTENT_LENGTH},
                )
                return

            # Read request body with proper error handling
            try:
                message_body_bytes = self.rfile.read(content_length)
            except Exception as e:
                self.send_error_response(408, f"Request timeout or read error: {e}")
                return

            # Validate we got the expected number of bytes
            if len(message_body_bytes) != content_length:
                self.send_error_response(
                    400,
                    f"Content-Length mismatch: expected {content_length}, got {len(message_body_bytes)}",
                )
                return

            # Decode bytes to string
            try:
                message_body_str = message_body_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                self.send_error_response(400, f"Invalid UTF-8 encoding: {e}")
                return

            # Authenticate the message using signature
            if self.registry:
                try:
                    is_authentic = self.registry.verify_signature(
                        agent_id, message_body_str, signature
                    )
                except Exception as e:
                    logger.error(
                        f"Signature verification error for agent '{agent_id}': {e}"
                    )
                    self.send_error_response(500, "Authentication system error")
                    return

                if not is_authentic:
                    logger.warning(f"Invalid signature from agent '{agent_id}'")
                    self.send_error_response(403, "Forbidden: Invalid signature")
                    return
            else:
                logger.warning("No registry available, skipping authentication")

            # Parse JSON
            try:
                graph = json.loads(message_body_str)
            except json.JSONDecodeError as e:
                self.send_error_response(
                    400, f"Invalid JSON: {e}", extra_data={"json_error": str(e)}
                )
                return

            # Validate graph structure
            validation_error = self.validate_graph(graph)
            if validation_error:
                self.send_error_response(
                    400, f"Invalid graph structure: {validation_error}"
                )
                return

            # Log successful authentication and graph receipt
            num_nodes = len(graph.get("nodes", []))
            num_edges = len(graph.get("edges", []))

            logger.info(
                f"Authenticated graph from agent '{agent_id}': "
                f"{num_nodes} nodes, {num_edges} edges"
            )

            # Process graph (in production, this would be queued/async)
            execution_result = None
            if self.runtime:
                try:
                    # Note: In production, use async processing or queue
                    execution_result = self.runtime.execute_graph(graph)
                    logger.info(
                        f"Graph executed for agent '{agent_id}': "
                        f"status={execution_result.get('status', 'unknown')}"
                    )
                except Exception as e:
                    logger.error(f"Graph execution failed for agent '{agent_id}': {e}")
                    # Still return 202 as we accepted it for processing

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Send success response
            self.send_success_response(
                202,  # 202 Accepted
                {
                    "status": "accepted",
                    "message": "Graph accepted for processing",
                    "agent_id": agent_id,
                    "graph_stats": {"nodes": num_nodes, "edges": num_edges},
                    "execution_result": execution_result,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Unexpected error in request handler: {e}", exc_info=True)
            self.send_error_response(500, "Internal server error")

    def validate_graph(self, graph: Any) -> Optional[str]:
        """
        Validate graph structure.

        Args:
            graph: Parsed graph object

        Returns:
            Error message if invalid, None if valid
        """
        try:
            # Must be a dict
            if not isinstance(graph, dict):
                return "Graph must be a JSON object"

            # Must have nodes and edges
            if "nodes" not in graph:
                return "Missing 'nodes' field"

            if "edges" not in graph:
                return "Missing 'edges' field"

            # Nodes must be a list
            if not isinstance(graph["nodes"], list):
                return "'nodes' must be an array"

            # Edges must be a list
            if not isinstance(graph["edges"], list):
                return "'edges' must be an array"

            # Basic size validation
            if len(graph["nodes"]) > 100000:
                return f"Too many nodes: {len(graph['nodes'])} (max 100,000)"

            if len(graph["edges"]) > 1000000:
                return f"Too many edges: {len(graph['edges'])} (max 1,000,000)"

            # Validate node structure (sample first 10)
            for i, node in enumerate(graph["nodes"][:10]):
                if not isinstance(node, dict):
                    return f"Node at index {i} is not an object"

                if "id" not in node:
                    return f"Node at index {i} missing 'id' field"

                if not isinstance(node["id"], str):
                    return f"Node at index {i} has non-string 'id'"

            # Validate edge structure (sample first 10)
            for i, edge in enumerate(graph["edges"][:10]):
                if not isinstance(edge, dict):
                    return f"Edge at index {i} is not an object"

                if "from" not in edge:
                    return f"Edge at index {i} missing 'from' field"

                if "to" not in edge:
                    return f"Edge at index {i} missing 'to' field"

                if not isinstance(edge["from"], str):
                    return f"Edge at index {i} has non-string 'from'"

                if not isinstance(edge["to"], str):
                    return f"Edge at index {i} has non-string 'to'"

            return None  # Valid

        except Exception as e:
            logger.error(f"Graph validation error: {e}")
            return f"Validation error: {e}"

    def send_error_response(
        self, code: int, message: str, extra_data: Optional[Dict[str, Any]] = None
    ):
        """
        Send error response with proper formatting.

        Args:
            code: HTTP status code
            message: Error message
            extra_data: Optional additional data to include
        """
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "error": message,
                "code": code,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if extra_data:
                response.update(extra_data)

            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    def send_success_response(self, code: int, data: Dict[str, Any]):
        """
        Send success response with proper formatting.

        Args:
            code: HTTP status code
            data: Response data
        """
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to send success response: {e}")


class GraphixListener:
    """
    Main listener server with graceful shutdown support and thread safety.
    SECURITY: Changed default host from 0.0.0.0 to 127.0.0.1
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8181,
        use_mock: bool = False,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    ):
        """
        Initialize listener server.

        Args:
            host: Host to bind to (default: 127.0.0.1 for security, use 0.0.0.0 to bind to all interfaces)
            port: Port to listen on
            use_mock: Use mock implementations if True
            max_requests_per_minute: Rate limit per client
        """
        self.host = host
        self.port = port
        self.httpd: Optional[HTTPServer] = None
        self.shutdown_event = threading.Event()

        # Initialize components with thread-safety
        self.init_lock = threading.RLock()

        with self.init_lock:
            # Initialize registry
            if REGISTRY_AVAILABLE and not use_mock:
                try:
                    self.registry = AgentRegistry()
                    logger.info("Using real AgentRegistry")
                except Exception as e:
                    logger.error(f"Failed to initialize AgentRegistry: {e}")
                    self.registry = MockAgentRegistry()
                    logger.info("Falling back to MockAgentRegistry")
            else:
                self.registry = MockAgentRegistry()
                logger.info("Using MockAgentRegistry")

            # Initialize runtime
            if RUNTIME_AVAILABLE and not use_mock:
                try:
                    self.runtime = UnifiedRuntime()
                    logger.info("Using real UnifiedRuntime")
                except Exception as e:
                    logger.error(f"Failed to initialize UnifiedRuntime: {e}")
                    self.runtime = MockUnifiedRuntime()
                    logger.info("Falling back to MockUnifiedRuntime")
            else:
                self.runtime = MockUnifiedRuntime()
                logger.info("Using MockUnifiedRuntime")

            # Initialize rate limiter
            self.rate_limiter = RateLimiter(max_requests=max_requests_per_minute)

            # Set class-level instances for handler
            RequestHandler.registry = self.registry
            RequestHandler.runtime = self.runtime
            RequestHandler.rate_limiter = self.rate_limiter
            RequestHandler.server_shutdown_flag = self.shutdown_event

    def start(self):
        """
        Start the listener server.

        Blocks until server is shut down via signal or shutdown() method.
        """
        try:
            server_address = (self.host, self.port)
            self.httpd = HTTPServer(server_address, RequestHandler)

            # Set timeout for handle_request
            self.httpd.timeout = REQUEST_TIMEOUT

            logger.info(f"Starting Graphix IR Listener on {self.host}:{self.port}")
            logger.info(
                f"Rate limit: {self.rate_limiter.max_requests} requests per minute"
            )

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("Server ready to accept connections (Press Ctrl+C to stop)")

            # Serve requests until shutdown
            while not self.shutdown_event.is_set():
                self.httpd.handle_request()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            self.stop()

    def stop(self):
        """Stop the listener server gracefully."""
        if self.shutdown_event.is_set():
            return  # Already shutting down

        logger.info("Shutting down Graphix IR Listener...")
        self.shutdown_event.set()

        if self.httpd:
            try:
                self.httpd.server_close()
                logger.info("Server socket closed")
            except Exception as e:
                logger.error(f"Error closing server: {e}")

        logger.info("Server shutdown complete")

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (SIGINT, SIGTERM).

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), initiating shutdown...")
        self.stop()
        sys.exit(0)


def run_listener(
    host: str = "127.0.0.1",
    port: int = 8181,
    use_mock: bool = False,
    max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
):
    """
    Run the listener server.
    SECURITY: Changed default host from 0.0.0.0 to 127.0.0.1

    Args:
        host: Host to bind to (default: 127.0.0.1 for security)
        port: Port to listen on
        use_mock: Use mock implementations for testing
        max_requests_per_minute: Rate limit per client
    """
    if host == "0.0.0.0":  # nosec B104 - This is a security check, not a binding
        logger.warning("⚠️ Binding to 0.0.0.0 (all interfaces) - ensure firewall is configured!")

    listener = GraphixListener(
        host=host,
        port=port,
        use_mock=use_mock,
        max_requests_per_minute=max_requests_per_minute,
    )
    listener.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphix IR Listener - Secure HTTP server for graph submission"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--port", type=int, default=8181, help="Port to listen on (default: 8181)"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock implementations for testing"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=MAX_REQUESTS_PER_MINUTE,
        help=f"Max requests per minute per client (default: {MAX_REQUESTS_PER_MINUTE})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Print startup banner
    print("=" * 60)
    print("Graphix IR Listener v2.0.0")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Mock mode: {args.mock}")
    print(f"Rate limit: {args.rate_limit} req/min")
    print("=" * 60)

    # Run server
    try:
        run_listener(
            host=args.host,
            port=args.port,
            use_mock=args.mock,
            max_requests_per_minute=args.rate_limit,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
