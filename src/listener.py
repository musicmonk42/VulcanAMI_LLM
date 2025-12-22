# src/listener.py
"""
Graphix IR HTTP Listener (Production-Ready)
===========================================
Version: 3.0.0 - Full production implementation with real integrations
A secure HTTP server for receiving Graphix IR graphs from agents with comprehensive
error handling, rate limiting, cryptographic signature verification, and graceful shutdown.

Security Features:
- HMAC-SHA256 signature verification for all requests
- Rate limiting per client IP
- Request size validation
- Input sanitization and validation
- Comprehensive audit logging
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GraphixListener")

# Try to import the full AgentRegistry, fall back to built-in implementation
try:
    from src.agent_registry import AgentRegistry as FullAgentRegistry
    FULL_REGISTRY_AVAILABLE = True
    logger.info("Full AgentRegistry available")
except ImportError as e:
    logger.info(f"Full AgentRegistry not available: {e}. Using built-in implementation.")
    FULL_REGISTRY_AVAILABLE = False
    FullAgentRegistry = None

# Try to import UnifiedRuntime
try:
    from src.unified_runtime import UnifiedRuntime as FullUnifiedRuntime
    FULL_RUNTIME_AVAILABLE = True
    logger.info("Full UnifiedRuntime available")
except ImportError as e:
    logger.info(f"Full UnifiedRuntime not available: {e}. Using built-in implementation.")
    FULL_RUNTIME_AVAILABLE = False
    FullUnifiedRuntime = None

# Constants
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
MIN_CONTENT_LENGTH = 1
REQUEST_TIMEOUT = 30  # seconds
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # seconds
MAX_AGENT_ID_LENGTH = 256
MAX_SIGNATURE_LENGTH = 512
LISTENER_DB_PATH = os.environ.get("LISTENER_DB_PATH", "listener_registry.db")


class ListenerAgentRegistry:
    """
    Production-grade agent registry for the Listener service.
    
    Provides cryptographic signature verification using HMAC-SHA256,
    agent management, and persistent storage via SQLite.
    
    This is a self-contained implementation that works independently
    when the full AgentRegistry is not available.
    """

    def __init__(self, db_path: str = LISTENER_DB_PATH):
        """
        Initialize the agent registry with persistent storage.
        
        Args:
            db_path: Path to SQLite database for agent storage
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()
        logger.info(f"ListenerAgentRegistry initialized with database: {db_path}")

    def _init_database(self):
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    details TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id)"
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_agent(
        self,
        agent_id: str,
        name: str,
        api_key: str,
        roles: List[str] = None
    ) -> bool:
        """
        Register a new agent with the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            api_key: Secret API key for signature verification
            roles: List of roles assigned to the agent
            
        Returns:
            True if registration successful, False otherwise
        """
        if not agent_id or not api_key:
            return False
            
        # Hash the API key for storage (never store plaintext)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO agents 
                        (agent_id, name, api_key_hash, roles, is_active, created_at, metadata)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                    """, (
                        agent_id,
                        name,
                        api_key_hash,
                        json.dumps(roles or ["agent"]),
                        datetime.utcnow().isoformat(),
                        json.dumps({})
                    ))
                    conn.commit()
                    logger.info(f"Agent '{agent_id}' registered successfully")
                    return True
            except Exception as e:
                logger.error(f"Failed to register agent '{agent_id}': {e}")
                return False

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information by ID.
        
        Args:
            agent_id: The agent identifier
            
        Returns:
            Agent data dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agents WHERE agent_id = ? AND is_active = 1",
                (agent_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "agent_id": row["agent_id"],
                    "name": row["name"],
                    "api_key_hash": row["api_key_hash"],
                    "roles": json.loads(row["roles"]),
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "last_seen": row["last_seen"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            return None

    def verify_signature(
        self,
        agent_id: str,
        message: str,
        signature: str
    ) -> bool:
        """
        Verify a cryptographic signature from an agent.
        
        Signature Scheme:
        -----------------
        The server stores SHA256(api_key) as api_key_hash.
        
        Client Signature Generation:
            derived_key = SHA256(api_key)  # Same as stored api_key_hash
            signature = HMAC-SHA256(derived_key, message)
        
        Server Verification:
            expected = HMAC-SHA256(api_key_hash, message)
            valid = constant_time_compare(expected, signature)
        
        This scheme ensures:
        1. Raw API keys are never stored on the server
        2. HMAC provides message integrity and authentication
        3. Constant-time comparison prevents timing attacks
        
        Args:
            agent_id: The agent making the request
            message: The message that was signed (typically the request body)
            signature: The hex-encoded HMAC-SHA256 signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not agent_id or not message or not signature:
            self._log_audit(agent_id or "unknown", "verify_signature", False, 
                          "Missing required parameters")
            return False

        agent = self.get_agent(agent_id)
        if not agent:
            self._log_audit(agent_id, "verify_signature", False, "Agent not found")
            logger.warning(f"Signature verification failed: Agent '{agent_id}' not found")
            return False

        try:
            # Verify signature using stored api_key_hash as the HMAC key
            # Client computes: HMAC-SHA256(SHA256(api_key), message)
            # Server computes: HMAC-SHA256(api_key_hash, message)
            # These match because api_key_hash = SHA256(api_key)
            expected_sig = hmac.new(
                agent["api_key_hash"].encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(expected_sig, signature.lower())
            
            if is_valid:
                # Update last_seen timestamp
                self._update_last_seen(agent_id)
                self._log_audit(agent_id, "verify_signature", True, "Signature valid")
            else:
                self._log_audit(agent_id, "verify_signature", False, "Invalid signature")
                logger.warning(f"Invalid signature from agent '{agent_id}'")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification error for agent '{agent_id}': {e}")
            self._log_audit(agent_id, "verify_signature", False, f"Error: {e}")
            return False

    def _update_last_seen(self, agent_id: str):
        """Update the last_seen timestamp for an agent."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE agents SET last_seen = ? WHERE agent_id = ?",
                    (datetime.utcnow().isoformat(), agent_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update last_seen for '{agent_id}': {e}")

    def _log_audit(
        self,
        agent_id: str,
        action: str,
        success: bool,
        details: str = None
    ):
        """Log an audit entry."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_log (timestamp, agent_id, action, success, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    agent_id,
                    action,
                    1 if success else 0,
                    details
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")


class ListenerGraphRuntime:
    """
    Production-grade graph execution runtime for the Listener service.
    
    Provides graph validation, execution tracking, and result management.
    This is a self-contained implementation that works independently
    when the full UnifiedRuntime is not available.
    """

    def __init__(self, db_path: str = LISTENER_DB_PATH):
        """
        Initialize the graph runtime.
        
        Args:
            db_path: Path to SQLite database for execution tracking
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()
        self._execution_count = 0
        logger.info("ListenerGraphRuntime initialized")

    def _init_database(self):
        """Initialize the execution tracking database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT UNIQUE NOT NULL,
                    agent_id TEXT NOT NULL,
                    graph_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    nodes_count INTEGER NOT NULL,
                    edges_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_exec_status ON graph_executions(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_exec_agent ON graph_executions(agent_id)"
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute_graph(
        self,
        graph: Dict[str, Any],
        agent_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute a Graphix IR graph.
        
        Performs validation, tracks execution, and returns results.
        
        Args:
            graph: The Graphix IR graph to execute
            agent_id: The agent submitting the graph
            
        Returns:
            Execution result dictionary
        """
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        # Validate graph structure
        validation_result = self._validate_graph(graph)
        if not validation_result["valid"]:
            return {
                "status": "validation_failed",
                "execution_id": execution_id,
                "error": validation_result["error"],
                "nodes_processed": 0,
                "edges_processed": 0,
                "started_at": started_at.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            }

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Calculate graph hash for deduplication/caching
        graph_hash = hashlib.sha256(
            json.dumps(graph, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Record execution start
        self._record_execution(
            execution_id=execution_id,
            agent_id=agent_id,
            graph_hash=graph_hash,
            status="executing",
            nodes_count=len(nodes),
            edges_count=len(edges),
            started_at=started_at
        )
        
        try:
            # Process the graph
            result = self._process_graph(graph, nodes, edges)
            
            # Update execution record
            completed_at = datetime.utcnow()
            self._update_execution(
                execution_id=execution_id,
                status="completed",
                completed_at=completed_at,
                result=result
            )
            
            with self._lock:
                self._execution_count += 1
            
            return {
                "status": "completed",
                "execution_id": execution_id,
                "graph_hash": graph_hash,
                "nodes_processed": len(nodes),
                "edges_processed": len(edges),
                "result": result,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "execution_time_ms": (completed_at - started_at).total_seconds() * 1000,
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Graph execution failed: {error_msg}")
            
            self._update_execution(
                execution_id=execution_id,
                status="failed",
                completed_at=datetime.utcnow(),
                error=error_msg
            )
            
            return {
                "status": "failed",
                "execution_id": execution_id,
                "error": error_msg,
                "nodes_processed": 0,
                "edges_processed": 0,
                "started_at": started_at.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            }

    def _validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate graph structure and content.
        
        Args:
            graph: The graph to validate
            
        Returns:
            Validation result with 'valid' boolean and optional 'error'
        """
        if not isinstance(graph, dict):
            return {"valid": False, "error": "Graph must be a dictionary"}
        
        if "nodes" not in graph:
            return {"valid": False, "error": "Missing 'nodes' field"}
        
        if "edges" not in graph:
            return {"valid": False, "error": "Missing 'edges' field"}
        
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        if not isinstance(nodes, list):
            return {"valid": False, "error": "'nodes' must be a list"}
        
        if not isinstance(edges, list):
            return {"valid": False, "error": "'edges' must be a list"}
        
        # Validate node structure
        node_ids = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                return {"valid": False, "error": f"Node at index {i} is not a dictionary"}
            if "id" not in node:
                return {"valid": False, "error": f"Node at index {i} missing 'id'"}
            if node["id"] in node_ids:
                return {"valid": False, "error": f"Duplicate node id: {node['id']}"}
            node_ids.add(node["id"])
        
        # Validate edge structure
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                return {"valid": False, "error": f"Edge at index {i} is not a dictionary"}
            if "from" not in edge or "to" not in edge:
                return {"valid": False, "error": f"Edge at index {i} missing 'from' or 'to'"}
            if edge["from"] not in node_ids:
                return {"valid": False, "error": f"Edge references unknown node: {edge['from']}"}
            if edge["to"] not in node_ids:
                return {"valid": False, "error": f"Edge references unknown node: {edge['to']}"}
        
        return {"valid": True}

    def _process_graph(
        self,
        graph: Dict[str, Any],
        nodes: List[Dict],
        edges: List[Dict]
    ) -> Dict[str, Any]:
        """
        Process the graph and return results.
        
        Args:
            graph: The full graph
            nodes: List of nodes
            edges: List of edges
            
        Returns:
            Processing result
        """
        # Build adjacency list for traversal
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge["from"]].append(edge["to"])
        
        # Find entry nodes (no incoming edges)
        has_incoming = set(edge["to"] for edge in edges)
        entry_nodes = [n["id"] for n in nodes if n["id"] not in has_incoming]
        
        # Perform topological analysis
        processed_nodes = []
        visited = set()
        
        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for neighbor in adjacency[node_id]:
                visit(neighbor)
            processed_nodes.append(node_id)
        
        for entry in entry_nodes:
            visit(entry)
        
        # Reverse for topological order
        processed_nodes.reverse()
        
        return {
            "graph_type": graph.get("type", "unknown"),
            "grammar_version": graph.get("grammar_version", "1.0.0"),
            "entry_nodes": entry_nodes,
            "processing_order": processed_nodes,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "analysis": {
                "is_dag": len(processed_nodes) == len(nodes),
                "connectivity": len(visited) / len(nodes) if nodes else 0,
            }
        }

    def _record_execution(
        self,
        execution_id: str,
        agent_id: str,
        graph_hash: str,
        status: str,
        nodes_count: int,
        edges_count: int,
        started_at: datetime
    ):
        """Record a new execution."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO graph_executions 
                    (execution_id, agent_id, graph_hash, status, nodes_count, edges_count, started_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution_id, agent_id, graph_hash, status,
                    nodes_count, edges_count, started_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")

    def _update_execution(
        self,
        execution_id: str,
        status: str,
        completed_at: datetime,
        result: Dict = None,
        error: str = None
    ):
        """Update an execution record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE graph_executions 
                    SET status = ?, completed_at = ?, result = ?, error = ?
                    WHERE execution_id = ?
                """, (
                    status,
                    completed_at.isoformat(),
                    json.dumps(result) if result else None,
                    error,
                    execution_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update execution: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM graph_executions 
                GROUP BY status
            """)
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
            
        return {
            "total_executions": self._execution_count,
            "status_counts": status_counts,
        }


# Compatibility aliases
REGISTRY_AVAILABLE = True  # Always available with built-in implementation
RUNTIME_AVAILABLE = True   # Always available with built-in implementation


class MockAgentRegistry:
    """
    Mock agent registry for testing purposes.
    
    Always returns True for signature verification.
    """

    def verify_signature(
        self,
        agent_id: str,
        message: str,
        signature: str
    ) -> bool:
        """
        Mock signature verification - always returns True.
        
        Args:
            agent_id: The agent making the request
            message: The message that was signed
            signature: The signature to verify
            
        Returns:
            Always True for testing
        """
        return True


class MockUnifiedRuntime:
    """
    Mock runtime for testing purposes.
    
    Returns mock execution results.
    """

    def execute_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock graph execution.
        
        Args:
            graph: The graph to execute
            
        Returns:
            Mock execution result
        """
        nodes = graph.get("nodes", [])
        return {
            "status": "mock_executed",
            "nodes_processed": len(nodes),
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
    
    Production-ready HTTP server for receiving Graphix IR graphs with:
    - Cryptographic signature verification
    - Rate limiting
    - Persistent storage
    - Comprehensive logging
    
    SECURITY: Default host is 127.0.0.1 for security. Use 0.0.0.0 for external access.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8181,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        db_path: str = LISTENER_DB_PATH,
        use_mock: bool = False,
    ):
        """
        Initialize listener server.

        Args:
            host: Host to bind to (default: 127.0.0.1 for security, use 0.0.0.0 to bind to all interfaces)
            port: Port to listen on
            max_requests_per_minute: Rate limit per client
            db_path: Path to SQLite database for persistent storage
            use_mock: Use mock implementations for testing (default: False)
        """
        self.host = host
        self.port = port
        self.httpd: Optional[HTTPServer] = None
        self.shutdown_event = threading.Event()

        # Initialize components with thread-safety
        self.init_lock = threading.RLock()

        with self.init_lock:
            # Use mock implementations if requested (for testing)
            if use_mock:
                self.registry = MockAgentRegistry()
                self.runtime = MockUnifiedRuntime()
                logger.info("Using mock implementations for testing")
            else:
                # Initialize registry - prefer full implementation if available
                if FULL_REGISTRY_AVAILABLE:
                    try:
                        self.registry = FullAgentRegistry()
                        logger.info("Using full AgentRegistry from src.agent_registry")
                    except Exception as e:
                        logger.warning(f"Failed to initialize full AgentRegistry: {e}")
                        self.registry = ListenerAgentRegistry(db_path=db_path)
                        logger.info("Using ListenerAgentRegistry (built-in implementation)")
                else:
                    self.registry = ListenerAgentRegistry(db_path=db_path)
                    logger.info("Using ListenerAgentRegistry (built-in implementation)")

                # Initialize runtime - prefer full implementation if available
                if FULL_RUNTIME_AVAILABLE:
                    try:
                        self.runtime = FullUnifiedRuntime()
                        logger.info("Using full UnifiedRuntime from src.unified_runtime")
                    except Exception as e:
                        logger.warning(f"Failed to initialize full UnifiedRuntime: {e}")
                        self.runtime = ListenerGraphRuntime(db_path=db_path)
                        logger.info("Using ListenerGraphRuntime (built-in implementation)")
                else:
                    self.runtime = ListenerGraphRuntime(db_path=db_path)
                    logger.info("Using ListenerGraphRuntime (built-in implementation)")

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
    max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    db_path: str = LISTENER_DB_PATH,
):
    """
    Run the listener server.
    
    Production-ready HTTP server for Graphix IR graph submission.
    SECURITY: Default host is 127.0.0.1 for security.

    Args:
        host: Host to bind to (default: 127.0.0.1 for security)
        port: Port to listen on
        max_requests_per_minute: Rate limit per client
        db_path: Path to SQLite database for persistent storage
    """
    if host == "0.0.0.0":  # nosec B104 - This is a security check, not a binding
        logger.warning(
            "⚠️ Binding to 0.0.0.0 (all interfaces) - ensure firewall is configured!"
        )

    listener = GraphixListener(
        host=host,
        port=port,
        max_requests_per_minute=max_requests_per_minute,
        db_path=db_path,
    )
    listener.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphix IR Listener - Production HTTP server for graph submission"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--port", type=int, default=8181, help="Port to listen on (default: 8181)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=MAX_REQUESTS_PER_MINUTE,
        help=f"Max requests per minute per client (default: {MAX_REQUESTS_PER_MINUTE})",
    )
    parser.add_argument(
        "--db-path",
        default=LISTENER_DB_PATH,
        help=f"Path to SQLite database (default: {LISTENER_DB_PATH})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Print startup banner
    print("=" * 60)
    print("Graphix IR Listener v3.0.0 (Production)")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Rate limit: {args.rate_limit} req/min")
    print(f"Database: {args.db_path}")
    print("=" * 60)

    # Run server
    try:
        run_listener(
            host=args.host,
            port=args.port,
            max_requests_per_minute=args.rate_limit,
            db_path=args.db_path,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
