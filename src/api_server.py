# api_server.py - FULL, HARDENED FILE
"""
Graphix API Server (Production-Ready, Security-Hardened)
========================================================
Version: 2.2.0
Enhancements since 2.1.0:
- Fail-fast if GRAPHIX_JWT_SECRET missing (unless explicitly allowed for dev)
- Expanded JWT claims: iss, aud, nbf, jti + kid header (key rotation ready)
- Optional asymmetric JWT (RS256 / EdDSA) support via env (GRAPHIX_JWT_PRIVATE_KEY / GRAPHIX_JWT_PUBLIC_KEY)
- Token revocation (in-memory + optional Redis) and /auth/logout endpoint
- Adaptive password hashing: support Argon2 if available; dynamic PBKDF2 iteration metadata
- Exponential backoff for login failures (per IP) to deter brute force
- Immediate nonce invalidation on mutual proof attempts (prevents offline cracking via repeated tries)
- Constant-time API key comparison + timing obfuscation (random jitter) to reduce enumeration risk
- RBAC enforcement: role checks for graph submission and proposal creation
- Request correlation: X-Request-ID added to all responses (generated if absent)
- Callback URL domain allowlist (CALLBACK_DOMAIN_ALLOWLIST) to mitigate SSRF risk
- /health extended: memory usage, thread count, revoked token count, optional psutil metrics
- Added auth failure metrics counters
- Added SSL/mTLS optional support if CERT_PATH / KEY_PATH provided
- Added basic revocation persistence (optional Redis; env GRAPHIX_REDIS_URL)
- Added complexity / safety placeholders for future GraphQL execution (still gated)
- Added validation for password hashing iteration threshold (warn if < recommended)
"""

import base64
import gzip
import hashlib
import hmac
import json
import logging
import os
import random
import re
import secrets
import signal
import sqlite3
import ssl
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import defaultdict, deque

# Import URL validation utility
try:
    from src.utils.url_validator import validate_url_scheme
except ImportError:
    from utils.url_validator import validate_url_scheme
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple

# Use cryptographically secure random for security-relevant operations (timing jitter, etc.)
secure_random = secrets.SystemRandom()

# Optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from argon2 import PasswordHasher as Argon2Hasher

    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

# Configure logging EARLY
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GraphAPIServer")

# Vulcan reasoning imports (after logger initialization)
try:
    from src.vulcan.reasoning.unified_reasoning import UnifiedReasoner
    from src.vulcan.reasoning.reasoning_types import ReasoningType, ReasoningResult

    REASONING_AVAILABLE = True
    logger.info("UnifiedReasoner loaded successfully")
except ImportError:
    try:
        # Try without src. prefix
        from vulcan.reasoning.unified_reasoning import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType, ReasoningResult

        REASONING_AVAILABLE = True
        logger.info("UnifiedReasoner loaded successfully")
    except ImportError as e:
        logger.warning(f"UnifiedReasoner not available: {e}")
        REASONING_AVAILABLE = False
        UnifiedReasoner = None
        ReasoningType = None
        ReasoningResult = None

# JWT support
try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available, JWT authentication disabled")

# GraphQL support (placeholder)
try:
    pass

    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False

# ======================================================================
# Security Constants & Configuration
# ======================================================================
RECOMMENDED_PBKDF2_ITERATIONS = 200_000
JWT_SECRET = os.environ.get("GRAPHIX_JWT_SECRET")

# Query preview lengths for logging and audit
QUERY_PREVIEW_LOG_LENGTH = 100  # For logging
QUERY_PREVIEW_AUDIT_LENGTH = 200  # For audit trail

ALLOW_EPHEMERAL_SECRET = (
    os.environ.get("ALLOW_EPHEMERAL_SECRET", "false").lower() == "true"
)

if not JWT_SECRET:
    if ALLOW_EPHEMERAL_SECRET:
        JWT_SECRET = secrets.token_urlsafe(48)
        logger.warning(
            "GRAPHIX_JWT_SECRET missing. Using ephemeral secret (DEVELOPMENT ONLY)."
        )
    else:
        raise RuntimeError(
            "GRAPHIX_JWT_SECRET is required for production. Set ALLOW_EPHEMERAL_SECRET=true to bypass (NOT RECOMMENDED)."
        )

# Optional asymmetric key support
GRAPHIX_JWT_PRIVATE_KEY = os.environ.get("GRAPHIX_JWT_PRIVATE_KEY")
GRAPHIX_JWT_PUBLIC_KEY = os.environ.get("GRAPHIX_JWT_PUBLIC_KEY")
JWT_ISS = os.environ.get("GRAPHIX_JWT_ISS", "graphix-api")
JWT_AUD = os.environ.get("GRAPHIX_JWT_AUD", "graphix-clients")
JWT_ALGORITHM = os.environ.get("GRAPHIX_JWT_ALGO", "HS256").upper().strip()
if GRAPHIX_JWT_PRIVATE_KEY and GRAPHIX_JWT_PUBLIC_KEY:
    if JWT_ALGORITHM not in {"RS256", "EdDSA"}:
        logger.warning(
            "Asymmetric keys provided but JWT_ALGO not RS256/EdDSA; falling back to HS256."
        )
        JWT_ALGORITHM = "HS256"

TOKEN_EXPIRY_HOURS = int(os.environ.get("GRAPHIX_JWT_EXPIRY_HOURS", "24"))
TOKEN_NOT_BEFORE_SECONDS = int(os.environ.get("GRAPHIX_JWT_NBF_OFFSET", "0"))

PBKDF2_ALGO = "sha256"
PBKDF2_ITERATIONS = int(
    os.environ.get("PBKDF2_ITERATIONS", str(RECOMMENDED_PBKDF2_ITERATIONS))
)
PBKDF2_SALT_BYTES = 16
PROOF_ALLOWED_DRIFT_SECONDS = 120  # time-based replay protection window

if PBKDF2_ITERATIONS < RECOMMENDED_PBKDF2_ITERATIONS:
    logger.warning(
        f"PBKDF2_ITERATIONS={PBKDF2_ITERATIONS} below recommended {RECOMMENDED_PBKDF2_ITERATIONS}. "
        "Increase for stronger password hashing."
    )

# Rate limiting & backoff for login
LOGIN_BACKOFF_MAX = 60  # seconds
LOGIN_BACKOFF_BASE = 2
LOGIN_FAILURE_WINDOW_SECONDS = 900

# Callback allowlist
CALLBACK_DOMAIN_ALLOWLIST = [
    d.strip().lower()
    for d in os.environ.get("CALLBACK_DOMAIN_ALLOWLIST", "").split(",")
    if d.strip()
]

# ======================================================================
# Server Constants
# ======================================================================
DEFAULT_PORT = int(os.environ.get("PORT", os.environ.get("GRAPHIX_API_PORT", "8000")))
DEFAULT_HOST = os.environ.get("GRAPHIX_API_HOST", "0.0.0.0")
MAX_REQUEST_SIZE = int(
    os.environ.get("GRAPHIX_MAX_REQUEST_SIZE", str(10 * 1024 * 1024))
)
RATE_LIMIT_WINDOW = int(os.environ.get("GRAPHIX_RATE_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("GRAPHIX_RATE_MAX", "100"))
DATABASE_PATH = os.environ.get("GRAPHIX_DB_PATH", "graphix_api.db")
CACHE_TTL = int(os.environ.get("GRAPHIX_CACHE_TTL", "3600"))
MAX_CACHE_SIZE = int(os.environ.get("GRAPHIX_CACHE_MAX", "1000"))
DB_POOL_SIZE = int(os.environ.get("GRAPHIX_DB_POOL", "5"))
REQUEST_TIMEOUT = int(os.environ.get("GRAPHIX_REQUEST_TIMEOUT", "30"))
MAX_GRAPH_NODES = int(os.environ.get("GRAPHIX_MAX_NODES", "10000"))
MAX_GRAPH_EDGES = int(os.environ.get("GRAPHIX_MAX_EDGES", "50000"))
CLEANUP_INTERVAL = int(os.environ.get("GRAPHIX_CLEANUP_INTERVAL", "300"))

# RBAC roles
REQUIRED_ROLE_GRAPH_SUBMIT = os.environ.get("REQUIRED_ROLE_GRAPH_SUBMIT", "user")
REQUIRED_ROLE_PROPOSAL_CREATE = os.environ.get(
    "REQUIRED_ROLE_PROPOSAL_CREATE", "govern"
)

# Optional Redis for JWT revocation
REDIS_URL = os.environ.get("GRAPHIX_REDIS_URL")
redis_client = None
if REDIS_AVAILABLE and REDIS_URL:
    try:
        redis_client = redis.Redis.from_url(
            REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True
        )
        redis_client.ping()
        logger.info("Connected to Redis for token revocation.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None

REVOCATION_PREFIX = os.environ.get("GRAPHIX_REVOCATION_PREFIX", "graphix:jwt:revoked:")


# ======================================================================
# Data Classes
# ======================================================================
class ExecutionStatus(Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class APIEndpoint(Enum):
    HEALTH = "/health"
    STATUS = "/status"
    AUTH = "/auth"
    GRAPHS = "/graphs"
    PROPOSALS = "/proposals"
    AGENTS = "/agents"
    METRICS = "/metrics"
    ADMIN = "/admin"
    GRAPHQL = "/graphql"
    LOGOUT = "/auth/logout"
    REASON = "/api/reason"


@dataclass
class GraphSubmission:
    id: str
    graph: Dict[str, Any]
    agent_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        for k in ["submitted_at", "started_at", "completed_at"]:
            v = d.get(k)
            if v:
                d[k] = v.isoformat()
        return d


@dataclass
class Proposal:
    id: str
    title: str
    description: str
    proposer_id: str
    graph: Dict[str, Any]
    votes_for: int = 0
    votes_against: int = 0
    status: str = "open"
    created_at: datetime = field(default_factory=datetime.utcnow)
    closes_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    id: str
    name: str
    api_key: str
    roles: List[str]
    trust_level: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    password_hash: Optional[str] = None
    password_salt: Optional[str] = None
    password_algo: Optional[str] = None  # e.g. 'argon2', 'pbkdf2_sha256$200000'


# ======================================================================
# Security Utilities
# ======================================================================
class SecurityUtils:
    @staticmethod
    def generate_salt(n_bytes: int = PBKDF2_SALT_BYTES) -> str:
        return base64.b64encode(os.urandom(n_bytes)).decode("ascii")

    @staticmethod
    def hash_password(
        password: str,
        salt_b64: str,
        iterations: int = PBKDF2_ITERATIONS,
        algo: str = PBKDF2_ALGO,
    ) -> str:
        if not isinstance(password, str) or not password:
            raise ValueError("Password must be a non-empty string")
        salt = base64.b64decode(salt_b64.encode("ascii"))
        dk = hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt, iterations)
        return base64.b64encode(dk).decode("ascii")

    @staticmethod
    def argon2_hash(password: str) -> str:
        if not ARGON2_AVAILABLE:
            raise RuntimeError("Argon2 not available")
        ph = Argon2Hasher()
        return ph.hash(password)

    @staticmethod
    def verify_password(
        password: str,
        salt_b64: Optional[str],
        stored_hash: str,
        algo_spec: Optional[str],
    ) -> bool:
        try:
            if algo_spec and algo_spec.startswith("argon2"):
                if not ARGON2_AVAILABLE:
                    return False
                ph = Argon2Hasher()
                return ph.verify(stored_hash, password)
            # PBKDF2 path
            # algo_spec format: pbkdf2_sha256$<iterations>
            iterations = PBKDF2_ITERATIONS
            if algo_spec and algo_spec.startswith("pbkdf2_sha256"):
                parts = algo_spec.split("$")
                if len(parts) == 2 and parts[1].isdigit():
                    iterations = int(parts[1])
            if not salt_b64:
                return False
            actual = SecurityUtils.hash_password(
                password, salt_b64, iterations, PBKDF2_ALGO
            )
            return hmac.compare_digest(actual, stored_hash)
        except Exception:
            return False

    @staticmethod
    def compute_hmac_signature(secret_hex: str, message: str) -> str:
        key = bytes.fromhex(secret_hex)
        sig = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(sig).decode("ascii")


# ======================================================================
# DB Connection Pool
# ======================================================================
class DatabaseConnectionPool:
    def __init__(self, db_path: str, pool_size: int = DB_POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.available = threading.Semaphore(pool_size)
        self.lock = threading.RLock()
        for _ in range(pool_size):
            conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
                timeout=10.0,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            self.connections.append(conn)

    @contextmanager
    def get_connection(self):
        self.available.acquire()
        try:
            with self.lock:
                conn = self.connections.pop()
            yield conn
        finally:
            with self.lock:
                self.connections.append(conn)
            self.available.release()

    def close_all(self):
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self.connections.clear()


# ======================================================================
# Rate Limiter (simple window) + Backoff
# ======================================================================
class RateLimiter:
    def __init__(
        self,
        window: int = RATE_LIMIT_WINDOW,
        max_requests: int = RATE_LIMIT_MAX_REQUESTS,
    ):
        self.window = window
        self.max_requests = max_requests
        self.requests = defaultdict(deque)
        self.lock = threading.RLock()
        self.shutdown_flag = False
        self._start_cleanup_thread()

    def is_allowed(self, identifier: str) -> bool:
        with self.lock:
            now = time.time()
            while (
                self.requests[identifier]
                and self.requests[identifier][0] < now - self.window
            ):
                self.requests[identifier].popleft()
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            self.requests[identifier].append(now)
            return True

    def _cleanup(self):
        with self.lock:
            now = time.time()
            for identifier, timestamps in list(self.requests.items()):
                while timestamps and timestamps[0] < now - self.window:
                    timestamps.popleft()
                if not timestamps:
                    del self.requests[identifier]

    def _start_cleanup_thread(self):
        def loop():
            while not self.shutdown_flag:
                time.sleep(CLEANUP_INTERVAL)
                if not self.shutdown_flag:
                    self._cleanup()

        t = threading.Thread(target=loop, daemon=True, name="RateLimitCleanup")
        t.start()

    def shutdown(self):
        self.shutdown_flag = True


# ======================================================================
# Input Validation
# ======================================================================
class InputValidator:
    @staticmethod
    def validate_graph(graph: Dict) -> Tuple[bool, Optional[str]]:
        if not isinstance(graph, dict):
            return False, "Graph must be a dictionary"
        required = ["id", "type", "nodes", "edges"]
        for field in required:
            if field not in graph:
                return False, f"Missing required field: {field}"
        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            return False, "Nodes must be a list"
        if len(nodes) > MAX_GRAPH_NODES:
            return False, f"Too many nodes: {len(nodes)} > {MAX_GRAPH_NODES}"
        node_ids = set()
        for node in nodes:
            if not isinstance(node, dict):
                return False, "Each node must be a dictionary"
            if "id" not in node:
                return False, "Node missing id"
            if node["id"] in node_ids:
                return False, f"Duplicate node id: {node['id']}"
            node_ids.add(node["id"])
        edges = graph.get("edges", [])
        if not isinstance(edges, list):
            return False, "Edges must be a list"
        if len(edges) > MAX_GRAPH_EDGES:
            return False, f"Too many edges: {len(edges)} > {MAX_GRAPH_EDGES}"
        for edge in edges:
            if not isinstance(edge, dict):
                return False, "Each edge must be a dictionary"
            if "from" not in edge or "to" not in edge:
                return False, "Edge missing from/to"
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                return False, "Edge references non-existent node"
        return True, None

    @staticmethod
    def sanitize_string(s: str, max_length: int = 1000) -> str:
        if not isinstance(s, str):
            return ""
        s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)
        return s[:max_length]

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        if not isinstance(api_key, str):
            return False
        return bool(re.match(r"^[a-f0-9]{32,128}$", api_key))

    @staticmethod
    def validate_url(url: str) -> bool:
        if not isinstance(url, str):
            return False
        try:
            parsed = urllib.parse.urlparse(url)
            return all([parsed.scheme in ["http", "https"], parsed.netloc])
        except Exception:
            return False

    @staticmethod
    def callback_host_allowed(url: str) -> bool:
        if not CALLBACK_DOMAIN_ALLOWLIST:
            return True
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower()
            for allowed in CALLBACK_DOMAIN_ALLOWLIST:
                if host == allowed or host.endswith("." + allowed):
                    return True
            return False
        except Exception:
            return False


# ======================================================================
# Database Manager
# ======================================================================
class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()
        self.pool = DatabaseConnectionPool(db_path)

    def _add_column_if_missing(
        self, conn: sqlite3.Connection, table: str, column: str, col_type: str
    ):
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS graphs (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                graph_data TEXT NOT NULL,
                status TEXT NOT NULL,
                submitted_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                error TEXT,
                metadata TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_graphs_agent ON graphs(agent_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graphs_status ON graphs(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_graphs_submitted ON graphs(submitted_at DESC)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                proposer_id TEXT NOT NULL,
                graph_data TEXT NOT NULL,
                votes_for INTEGER DEFAULT 0,
                votes_against INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                closes_at TIMESTAMP,
                metadata TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_created ON proposals(created_at DESC)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                roles TEXT NOT NULL,
                trust_level REAL DEFAULT 0.5,
                created_at TIMESTAMP NOT NULL,
                last_seen TIMESTAMP,
                metadata TEXT,
                password_hash TEXT,
                password_salt TEXT,
                password_algo TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_api_key ON agents(api_key)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                agent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                details TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id)"
        )
        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def save_graph(self, submission: GraphSubmission):
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            graph_json = json.dumps(submission.graph)
            graph_b64 = base64.b64encode(
                gzip.compress(graph_json.encode("utf-8"))
            ).decode("ascii")
            result_b64 = None
            if submission.result:
                result_json = json.dumps(submission.result)
                result_b64 = base64.b64encode(
                    gzip.compress(result_json.encode("utf-8"))
                ).decode("ascii")
            cursor.execute(
                """
                INSERT OR REPLACE INTO graphs
                (id, agent_id, graph_data, status, submitted_at, started_at,
                 completed_at, result, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    submission.id,
                    submission.agent_id,
                    graph_b64,
                    submission.status.value,
                    submission.submitted_at,
                    submission.started_at,
                    submission.completed_at,
                    result_b64,
                    submission.error,
                    json.dumps(submission.metadata),
                ),
            )
            conn.commit()

    def get_graph(self, graph_id: str) -> Optional[GraphSubmission]:
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM graphs WHERE id = ?", (graph_id,))
            row = cursor.fetchone()
        if not row:
            return None
        try:
            graph_json = gzip.decompress(base64.b64decode(row[2])).decode("utf-8")
            graph = json.loads(graph_json)
            result = None
            if row[7]:
                result_json = gzip.decompress(base64.b64decode(row[7])).decode("utf-8")
                result = json.loads(result_json)
            return GraphSubmission(
                id=row[0],
                graph=graph,
                agent_id=row[1],
                status=ExecutionStatus(row[3]),
                submitted_at=row[4],
                started_at=row[5],
                completed_at=row[6],
                result=result,
                error=row[8],
                metadata=json.loads(row[9]) if row[9] else {},
            )
        except Exception as e:
            logger.error(f"Error deserializing graph: {e}")
            return None

    def save_agent(self, agent: Agent):
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO agents
                (id, name, api_key, roles, trust_level, created_at, last_seen,
                 metadata, password_hash, password_salt, password_algo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    agent.id,
                    agent.name,
                    agent.api_key,
                    json.dumps(agent.roles),
                    agent.trust_level,
                    agent.created_at,
                    agent.last_seen,
                    json.dumps(agent.metadata),
                    agent.password_hash,
                    agent.password_salt,
                    agent.password_algo,
                ),
            )
            conn.commit()

    def get_agent_by_api_key(self, api_key: str) -> Optional[Agent]:
        if not InputValidator.validate_api_key(api_key):
            return None
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, api_key, roles, trust_level, created_at, last_seen,
                       metadata, password_hash, password_salt, password_algo
                FROM agents WHERE api_key = ?
            """,
                (api_key,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return Agent(
                id=row[0],
                name=row[1],
                api_key=row[2],
                roles=json.loads(row[3]),
                trust_level=row[4],
                created_at=row[5],
                last_seen=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
                password_hash=row[8],
                password_salt=row[9],
                password_algo=row[10],
            )
        except Exception as e:
            logger.error(f"Error deserializing agent: {e}")
            return None

    def log_audit(self, agent_id: str, action: str, resource: str, details: Dict):
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_log (timestamp, agent_id, action, resource, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.utcnow(), agent_id, action, resource, json.dumps(details)),
            )
            conn.commit()

    def cleanup(self):
        self.pool.close_all()


# ======================================================================
# Execution Engine
# ======================================================================
class ExecutionEngine:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="GraphExecutor"
        )
        self.executing: Dict[str, GraphSubmission] = {}
        self.futures: Dict[str, Future] = {}
        self.lock = threading.RLock()

    def execute_graph(
        self, submission: GraphSubmission, callback: Optional[Callable] = None
    ) -> Future:
        def _execute():
            try:
                submission.status = ExecutionStatus.EXECUTING
                submission.started_at = datetime.utcnow()
                # Simulate work
                time.sleep(2)
                valid, error = InputValidator.validate_graph(submission.graph)
                if not valid:
                    raise ValueError(f"Invalid graph: {error}")
                result = {
                    "nodes_processed": len(submission.graph.get("nodes", [])),
                    "edges_processed": len(submission.graph.get("edges", [])),
                    "output": f"Processed graph {submission.id}",
                    "metrics": {"execution_time_ms": 2000, "memory_used_mb": 50},
                }
                submission.status = ExecutionStatus.COMPLETED
                submission.result = result
            except Exception as e:
                submission.status = ExecutionStatus.FAILED
                submission.error = str(e)[:1000]
                logger.error(f"Graph execution failed: {e}")
            finally:
                submission.completed_at = datetime.utcnow()
                with self.lock:
                    self.executing.pop(submission.id, None)
                    self.futures.pop(submission.id, None)
                if callback:
                    try:
                        callback(submission)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        with self.lock:
            self.executing[submission.id] = submission
            future = self.executor.submit(_execute)
            self.futures[submission.id] = future
        return future

    def cancel_execution(self, graph_id: str) -> bool:
        with self.lock:
            if graph_id in self.futures:
                future = self.futures[graph_id]
                cancelled = future.cancel()
                if cancelled or graph_id in self.executing:
                    sub = self.executing.get(graph_id)
                    if sub:
                        sub.status = ExecutionStatus.CANCELLED
                        sub.completed_at = datetime.utcnow()
                    self.executing.pop(graph_id, None)
                    self.futures.pop(graph_id, None)
                    return True
        return False

    def shutdown(self):
        logger.info("Shutting down execution engine...")
        self.executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Execution engine shutdown complete")


# ======================================================================
# Cache Manager
# ======================================================================
class CacheManager:
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl: int = CACHE_TTL):
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        self.shutdown_flag = False
        self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                ts, value = self.cache[key]
                if time.time() - ts < self.ttl:
                    self.access_times[key] = time.time()
                    return value
                else:
                    self.cache.pop(key, None)
                    self.access_times.pop(key, None)
        return None

    def set(self, key: str, value: Any):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            self.cache[key] = (time.time(), value)
            self.access_times[key] = time.time()

    def _evict_lru(self):
        if self.access_times:
            lru_key = min(self.access_times, key=self.access_times.get)
            self.cache.pop(lru_key, None)
            self.access_times.pop(lru_key, None)

    def _cleanup(self):
        with self.lock:
            now = time.time()
            expired = [k for k, (ts, _) in self.cache.items() if now - ts >= self.ttl]
            for k in expired:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)

    def _start_cleanup_thread(self):
        def loop():
            while not self.shutdown_flag:
                time.sleep(CLEANUP_INTERVAL)
                if not self.shutdown_flag:
                    self._cleanup()

        t = threading.Thread(target=loop, daemon=True, name="CacheCleanup")
        t.start()

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def shutdown(self):
        self.shutdown_flag = True


# ======================================================================
# Auth / Revocation
# ======================================================================
revoked_jti_set = set()


def revoke_token(jti: str, exp: Optional[int] = None):
    if redis_client:
        try:
            ttl = 0
            if exp:
                now = int(time.time())
                ttl = max(1, exp - now)
            redis_client.setex(f"{REVOCATION_PREFIX}{jti}", ttl if ttl else 3600, "1")
            return
        except Exception as e:
            logger.error(f"Redis revocation failed: {e}")
    revoked_jti_set.add(jti)


def is_revoked(jti: str) -> bool:
    if redis_client:
        try:
            return redis_client.get(f"{REVOCATION_PREFIX}{jti}") is not None
        except Exception as e:
            logger.debug(f"Operation failed: {e}")
    return jti in revoked_jti_set


# ======================================================================
# Login Failure Backoff
# ======================================================================
login_fail_lock = threading.RLock()
login_fail_counts: Dict[str, int] = {}
login_backoff_expiry: Dict[str, float] = {}


def record_login_failure(ip: str) -> int:
    with login_fail_lock:
        count = login_fail_counts.get(ip, 0) + 1
        login_fail_counts[ip] = count
        backoff = min(LOGIN_BACKOFF_MAX, LOGIN_BACKOFF_BASE ** min(10, count))
        expiry = time.time() + backoff
        login_backoff_expiry[ip] = expiry
        return backoff


def clear_login_failures(ip: str):
    with login_fail_lock:
        login_fail_counts.pop(ip, None)
        login_backoff_expiry.pop(ip, None)


def get_login_backoff_remaining(ip: str) -> float:
    with login_fail_lock:
        expiry = login_backoff_expiry.get(ip)
        if not expiry:
            return 0.0
        return max(0.0, expiry - time.time())


# ======================================================================
# HTTP Request Handler
# ======================================================================
class APIRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, server_instance=None, **kwargs):
        self.server_instance: GraphAPIServer = server_instance
        self._request_id = None
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} ({self._request_id}) - {format % args}")

    def _apply_security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"
        )
        allowed_origin = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Vary", "Origin")
        if self._request_id:
            self.send_header("X-Request-ID", self._request_id)

    def _init_request_id(self):
        incoming = self.headers.get("X-Request-ID")
        if incoming and re.match(r"^[A-Za-z0-9_\-\.]{1,128}$", incoming):
            self._request_id = incoming
        else:
            self._request_id = uuid.uuid4().hex

    def do_GET(self):
        self._init_request_id()
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            if path == APIEndpoint.HEALTH.value:
                self._handle_health()
            elif path == APIEndpoint.STATUS.value:
                self._handle_status()
            elif path.startswith(APIEndpoint.GRAPHS.value):
                self._handle_get_graph(path)
            elif path == APIEndpoint.METRICS.value:
                self._handle_metrics()
            elif path == "/vulcan/insights":
                self._handle_vulcan_insights()
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"GET request error: {e}\n{traceback.format_exc()}")
            self._send_error(500, "Internal Server Error")

    def do_POST(self):
        self._init_request_id()
        try:
            content_length_raw = self.headers.get("Content-Length")
            try:
                content_length = (
                    int(content_length_raw) if content_length_raw is not None else 0
                )
            except ValueError:
                self._send_error(400, "Invalid Content-Length")
                return
            if content_length > MAX_REQUEST_SIZE:
                self._send_error(413, "Request too large")
                return
            if content_length == 0:
                self._send_error(400, "Empty request")
                return
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(400, "Invalid JSON")
                return
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path

            if path == f"{APIEndpoint.AUTH.value}/login":
                ip = self.client_address[0] if self.client_address else "unknown"
                backoff_remaining = get_login_backoff_remaining(ip)
                if backoff_remaining > 0:
                    self._send_error(
                        429,
                        f"Login temporarily blocked. Try in {int(backoff_remaining)}s",
                    )
                    return
                self._handle_login(data, ip)
            elif path == f"{APIEndpoint.AUTH.value}/logout":
                self._handle_logout()
            elif path == f"{APIEndpoint.GRAPHS.value}/submit":
                agent = self._authenticate()
                self._handle_submit_graph(data, agent)
            elif path == APIEndpoint.REASON.value:
                agent = self._authenticate()
                self._handle_reason(data, agent)
            elif path == f"{APIEndpoint.PROPOSALS.value}/create":
                agent = self._authenticate()
                self._handle_create_proposal(data, agent)
            elif path.startswith(f"{APIEndpoint.PROPOSALS.value}/") and path.endswith(
                "/vote"
            ):
                agent = self._authenticate()
                self._handle_vote(path, data, agent)
            elif path == APIEndpoint.GRAPHQL.value and GRAPHQL_AVAILABLE:
                agent = self._authenticate()
                self._handle_graphql(data, agent)
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"POST request error: {e}\n{traceback.format_exc()}")
            self._send_error(500, "Internal Server Error")

    def do_OPTIONS(self):
        self._init_request_id()
        self.send_response(200)
        self._apply_security_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-API-Key, X-Request-ID",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _authenticate(self) -> Optional[Agent]:
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and JWT_AVAILABLE:
            token = auth_header[7:]
            try:
                # Decode without verifying revocation first
                payload = jwt.decode(
                    token,
                    (
                        GRAPHIX_JWT_PUBLIC_KEY
                        if GRAPHIX_JWT_PRIVATE_KEY
                        and GRAPHIX_JWT_PUBLIC_KEY
                        and JWT_ALGORITHM != "HS256"
                        else JWT_SECRET
                    ),
                    algorithms=[JWT_ALGORITHM],
                    audience=JWT_AUD,
                    issuer=JWT_ISS,
                )
                jti = payload.get("jti")
                if jti and is_revoked(jti):
                    logger.warning("Revoked JWT token used")
                    return None
                agent_id = payload.get("agent_id")
                if agent_id and self.server_instance:
                    return self.server_instance.get_agent_by_id(agent_id)
            except jwt.ExpiredSignatureError:
                logger.warning("Expired JWT token")
            except jwt.InvalidTokenError:
                logger.warning("Invalid JWT token")
        api_key = self.headers.get("X-API-Key")
        if api_key and self.server_instance:
            agent = self.server_instance.db.get_agent_by_api_key(api_key)
            if agent:
                agent.last_seen = datetime.utcnow()
                self.server_instance.db.save_agent(agent)
            return agent
        return None

    def _handle_health(self):
        mem_info = {}
        if PSUTIL_AVAILABLE:
            try:
                p = psutil.Process(os.getpid())
                mem_info = {
                    "rss_mb": round(p.memory_info().rss / (1024 * 1024), 2),
                    "threads": p.num_threads(),
                }
            except Exception:
                mem_info = {}
        self._send_json(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.2.0",
                "revoked_tokens": len(revoked_jti_set),
                "memory": mem_info,
            }
        )

    def _handle_status(self):
        if self.server_instance:
            status = self.server_instance.get_status()
            self._send_json(status)
        else:
            self._send_error(500, "Server not initialized")

    def _handle_get_graph(self, path: str):
        parts = path.split("/")
        if len(parts) >= 3 and parts[2]:
            graph_id = InputValidator.sanitize_string(parts[2], 64)
            if self.server_instance:
                submission = self.server_instance.db.get_graph(graph_id)
                if submission:
                    self._send_json(submission.to_dict())
                else:
                    self._send_error(404, "Graph not found")
            else:
                self._send_error(500, "Server not initialized")
        else:
            self._send_error(400, "Invalid graph ID")

    def _require_role(self, agent: Agent, required_role: str) -> bool:
        return required_role in (agent.roles or [])

    def _handle_submit_graph(self, data: Dict, agent: Optional[Agent]):
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        if REQUIRED_ROLE_GRAPH_SUBMIT and not self._require_role(
            agent, REQUIRED_ROLE_GRAPH_SUBMIT
        ):
            self._send_error(
                403, f"Missing required role: {REQUIRED_ROLE_GRAPH_SUBMIT}"
            )
            return
        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_instance.rate_limiter.is_allowed(agent.id):
            self._send_error(429, "Rate limit exceeded")
            return
        graph = data.get("graph")
        priority = data.get("priority", 0)
        timeout = data.get("timeout", REQUEST_TIMEOUT)
        callback = data.get("callback")
        if not graph:
            self._send_error(400, "Missing graph")
            return
        if not isinstance(priority, int) or not (0 <= priority <= 10):
            self._send_error(400, "Invalid priority (must be int between 0 and 10)")
            return
        if not isinstance(timeout, int) or not (0 < timeout <= 300):
            self._send_error(400, "Invalid timeout (must be int between 1 and 300)")
            return
        if callback:
            if not InputValidator.validate_url(callback):
                self._send_error(400, "Invalid callback URL")
                return
            if not InputValidator.callback_host_allowed(callback):
                self._send_error(400, "Callback host not in allowlist")
                return
        valid, error = InputValidator.validate_graph(graph)
        if not valid:
            self._send_error(400, f"Invalid graph: {error}")
            return
        result = self.server_instance.submit_graph(
            graph, agent.id, priority=priority, timeout=timeout, callback=callback
        )
        self._send_json(result)

    def _handle_create_proposal(self, data: Dict, agent: Optional[Agent]):
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        if REQUIRED_ROLE_PROPOSAL_CREATE and not self._require_role(
            agent, REQUIRED_ROLE_PROPOSAL_CREATE
        ):
            self._send_error(
                403, f"Missing required role: {REQUIRED_ROLE_PROPOSAL_CREATE}"
            )
            return
        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return
        title = InputValidator.sanitize_string(data.get("title", ""), 200)
        description = InputValidator.sanitize_string(data.get("description", ""), 2000)
        graph = data.get("graph")
        if not title or not graph:
            self._send_error(400, "Missing title or graph")
            return
        valid, error = InputValidator.validate_graph(graph)
        if not valid:
            self._send_error(400, f"Invalid graph: {error}")
            return
        proposal = self.server_instance.create_proposal(
            title, description, graph, agent.id
        )
        if proposal:
            self._send_json({"id": proposal.id, "status": "created"})
        else:
            self._send_error(400, "Failed to create proposal")

    def _handle_vote(self, path: str, data: Dict, agent: Optional[Agent]):
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return
        parts = path.split("/")
        if len(parts) >= 4:
            proposal_id = InputValidator.sanitize_string(parts[2], 64)
            vote = data.get("vote")
            if vote not in ["for", "against"]:
                self._send_error(400, "Invalid vote (must be 'for' or 'against')")
                return
            success = self.server_instance.vote_on_proposal(proposal_id, agent.id, vote)
            self._send_json({"success": success})
        else:
            self._send_error(400, "Invalid proposal ID")

    def _handle_login(self, data: Dict, ip: str):
        if not JWT_AVAILABLE:
            self._send_error(501, "JWT not available")
            return
        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return
        api_key = data.get("api_key")
        password = data.get("password")
        nonce = data.get("nonce")
        timestamp = data.get("timestamp")
        proof = data.get("proof")
        if not api_key:
            self._send_error(400, "Missing api_key")
            return

        # Uniform start time for timing equalization
        start_time = time.time()

        agent = self.server_instance.db.get_agent_by_api_key(api_key)
        audit_details = {
            "remote_ip": ip,
            "auth_method": (
                "password"
                if password
                else (
                    "mutual_proof" if all([nonce, timestamp, proof]) else "api_key_only"
                )
            ),
        }
        # Require password or mutual proof
        if password is None and not (nonce and timestamp and proof):
            self.server_instance.db.log_audit(
                agent.id if agent else "unknown",
                "login_failed",
                "auth/login",
                {**audit_details, "reason": "missing password or proof"},
            )
            # Timing jitter (using cryptographically secure random)
            time.sleep(secure_random.uniform(0.02, 0.05))
            self._send_error(401, "Invalid credentials")
            return

        if not agent:
            self.server_instance.db.log_audit(
                "unknown",
                "login_failed",
                "auth/login",
                {**audit_details, "reason": "unknown_api_key"},
            )
            time.sleep(secure_random.uniform(0.02, 0.05))
            self._send_error(401, "Invalid credentials")
            return

        agent.last_seen = datetime.utcnow()
        verified = False
        verify_reason = None

        # Password verification
        if password:
            verified = SecurityUtils.verify_password(
                password,
                agent.password_salt,
                agent.password_hash or "",
                agent.password_algo,
            )
            verify_reason = "password_ok" if verified else "password_mismatch"

        # Mutual proof
        if not verified and nonce and timestamp and proof:
            try:
                if isinstance(timestamp, (int, float)):
                    ts = float(timestamp)
                elif isinstance(timestamp, str) and timestamp.isdigit():
                    ts = float(timestamp)
                elif isinstance(timestamp, str):
                    ts = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    ).timestamp()
                else:
                    raise ValueError("Invalid timestamp format")
            except Exception:
                self.server_instance.db.log_audit(
                    agent.id,
                    "login_failed",
                    "auth/login",
                    {**audit_details, "reason": "invalid_timestamp"},
                )
                time.sleep(secure_random.uniform(0.02, 0.05))
                self._send_error(401, "Invalid credentials")
                return

            now_ts = time.time()
            if abs(now_ts - ts) > PROOF_ALLOWED_DRIFT_SECONDS:
                self.server_instance.db.log_audit(
                    agent.id,
                    "login_failed",
                    "auth/login",
                    {**audit_details, "reason": "stale_or_future_timestamp"},
                )
                time.sleep(secure_random.uniform(0.02, 0.05))
                self._send_error(401, "Invalid credentials")
                return

            # Nonce single use (invalidate BEFORE checking proof)
            nonce_cache_key = f"login_nonce:{nonce}"
            if self.server_instance.cache.get(nonce_cache_key) is not None:
                self.server_instance.db.log_audit(
                    agent.id,
                    "login_failed",
                    "auth/login",
                    {**audit_details, "reason": "nonce_reuse"},
                )
                time.sleep(secure_random.uniform(0.02, 0.05))
                self._send_error(401, "Invalid credentials")
                return
            # Mark attempted regardless of success
            self.server_instance.cache.set(nonce_cache_key, True)

            message = f"{agent.id}:{nonce}:{int(ts)}"
            try:
                expected = SecurityUtils.compute_hmac_signature(agent.api_key, message)
                if hmac.compare_digest(expected, proof):
                    verified = True
                    verify_reason = "mutual_proof_ok"
                else:
                    verify_reason = "proof_mismatch"
            except Exception:
                verify_reason = "proof_compute_error"

        if not verified:
            backoff = record_login_failure(ip)
            self.server_instance.db.log_audit(
                agent.id,
                "login_failed",
                "auth/login",
                {
                    **audit_details,
                    "reason": verify_reason or "verification_failed",
                    "backoff": backoff,
                },
            )
            self.server_instance.metrics_lock.acquire()
            self.server_instance.metrics["auth_failures"] += 1
            self.server_instance.metrics_lock.release()
            elapsed = time.time() - start_time
            # Standardize minimum response time
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed + random.uniform(0.005, 0.015))
            self._send_error(401, "Invalid credentials")
            return

        clear_login_failures(ip)

        self.server_instance.db.save_agent(agent)

        jti = uuid.uuid4().hex
        now = datetime.utcnow()
        payload = {
            "agent_id": agent.id,
            "iss": JWT_ISS,
            "aud": JWT_AUD,
            "iat": now,
            "nbf": now + timedelta(seconds=TOKEN_NOT_BEFORE_SECONDS),
            "exp": now + timedelta(hours=TOKEN_EXPIRY_HOURS),
            "jti": jti,
        }
        # kid derived from secret hash or public key fingerprint
        if (
            GRAPHIX_JWT_PRIVATE_KEY
            and GRAPHIX_JWT_PUBLIC_KEY
            and JWT_ALGORITHM != "HS256"
        ):
            kid_source = GRAPHIX_JWT_PUBLIC_KEY
        else:
            kid_source = JWT_SECRET
        kid = hashlib.sha256(kid_source.encode("utf-8")).hexdigest()[:16]
        headers = {"kid": kid}

        signing_key = (
            GRAPHIX_JWT_PRIVATE_KEY
            if GRAPHIX_JWT_PRIVATE_KEY
            and GRAPHIX_JWT_PUBLIC_KEY
            and JWT_ALGORITHM != "HS256"
            else JWT_SECRET
        )
        token = jwt.encode(
            payload, signing_key, algorithm=JWT_ALGORITHM, headers=headers
        )

        self.server_instance.db.log_audit(
            agent.id,
            "login_success",
            "auth/login",
            {**audit_details, "verification": verify_reason, "jti": jti},
        )
        with self.server_instance.metrics_lock:
            self.server_instance.metrics["auth_success"] += 1

        self._send_json(
            {
                "token": token,
                "agent_id": agent.id,
                "expires_in": TOKEN_EXPIRY_HOURS * 3600,
                "issuer": JWT_ISS,
                "audience": JWT_AUD,
                "kid": kid,
            }
        )

    def _handle_logout(self):
        agent = self._authenticate()
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        auth_header = self.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else None
        if not token:
            self._send_error(400, "Missing token")
            return
        try:
            payload = jwt.decode(
                token,
                (
                    GRAPHIX_JWT_PUBLIC_KEY
                    if GRAPHIX_JWT_PRIVATE_KEY
                    and GRAPHIX_JWT_PUBLIC_KEY
                    and JWT_ALGORITHM != "HS256"
                    else JWT_SECRET
                ),
                algorithms=[JWT_ALGORITHM],
                audience=JWT_AUD,
                issuer=JWT_ISS,
            )
            jti = payload.get("jti")
            exp_dt = payload.get("exp")
            exp_ts = None
            if isinstance(exp_dt, (int, float)):
                exp_ts = int(exp_dt)
            elif hasattr(exp_dt, "timestamp"):
                exp_ts = int(exp_dt.timestamp())
            if jti:
                revoke_token(jti, exp_ts)
                self.server_instance.db.log_audit(
                    agent.id, "logout", "auth/logout", {"jti": jti}
                )
                self._send_json({"status": "revoked", "jti": jti})
            else:
                self._send_error(400, "Token missing jti")
        except Exception as e:
            logger.error(f"Logout decode error: {e}")
            self._send_error(400, "Invalid token")

    def _handle_metrics(self):
        if self.server_instance:
            metrics = self.server_instance.get_metrics()
            self._send_json(metrics)
        else:
            self._send_error(500, "Server not initialized")

    def _handle_graphql(self, data: Dict, agent: Optional[Agent]):
        if not GRAPHQL_AVAILABLE:
            self._send_error(501, "GraphQL not available")
            return
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        # Placeholder response (complexity limiting to be added when enabling real queries)
        query = data.get("query", "")
        if len(query) > 10_000:
            self._send_error(400, "Query too large")
            return
        self._send_json(
            {
                "message": "GraphQL endpoint (integration pending)",
                "query_preview": query[:100],
            }
        )

    def _handle_reason(self, data: Dict, agent: Optional[Agent]):
        """
        Handle reasoning endpoint requests.
        Accepts a query and optional reasoning_type, returns reasoning result.
        """
        if not agent:
            self._send_error(401, "Unauthorized")
            return

        if not REASONING_AVAILABLE:
            self._send_error(501, "Reasoning engine not available")
            return

        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return

        # Extract request data
        query = data.get("query")
        reasoning_type_str = data.get("reasoning_type")

        if not query:
            self._send_error(400, "Missing required field: query")
            return

        # Parse reasoning_type if provided
        reasoning_type = None
        if reasoning_type_str:
            try:
                reasoning_type = ReasoningType(reasoning_type_str)
            except (ValueError, AttributeError):
                self._send_error(400, f"Invalid reasoning_type: {reasoning_type_str}")
                return

        try:
            # Thread-safe lazy initialization of reasoner
            with self.server_instance.locks["reasoner"]:
                if self.server_instance._unified_reasoner is None:
                    logger.info("Initializing UnifiedReasoner...")
                    self.server_instance._unified_reasoner = UnifiedReasoner(
                        enable_learning=False,  # Disable learning for API mode
                        enable_safety=True,
                        max_workers=2,  # Limit workers for API server
                    )
                    logger.info("UnifiedReasoner initialized successfully")

            reasoner = self.server_instance._unified_reasoner

            # Prepare input for reasoner
            # UnifiedReasoner expects:
            # - input_data: raw input (can be str, dict, etc.)
            # - query: structured query dict with additional context
            if isinstance(query, str):
                input_data = query
                query_dict = {"query": query}
            else:
                # query is already a dict
                input_data = query.get("query", query)
                query_dict = query

            # Perform reasoning
            logger.info(
                f"Reasoning request from agent {agent.id}: query={str(input_data)[:QUERY_PREVIEW_LOG_LENGTH]}, type={reasoning_type}"
            )
            result = reasoner.reason(
                input_data=input_data, query=query_dict, reasoning_type=reasoning_type
            )

            # Convert result to JSON-serializable format
            response = {
                "conclusion": str(result.conclusion),
                "confidence": float(result.confidence),
                "reasoning_type": (
                    result.reasoning_type.value if result.reasoning_type else "unknown"
                ),
                "explanation": result.explanation,
                "uncertainty": float(result.uncertainty),
                "metadata": result.metadata,
            }

            # Add safety status if available
            if result.safety_status:
                response["safety_status"] = result.safety_status

            # Log audit entry
            self.server_instance.db.log_audit(
                agent.id,
                "reasoning_request",
                "api/reason",
                {
                    "query_preview": str(input_data)[:QUERY_PREVIEW_AUDIT_LENGTH],
                    "reasoning_type": (
                        reasoning_type.value if reasoning_type else "auto"
                    ),
                    "confidence": result.confidence,
                },
            )

            self._send_json(response)

        except Exception as e:
            logger.error(f"Reasoning request failed: {e}\n{traceback.format_exc()}")
            self._send_error(500, f"Reasoning failed: {str(e)}")

    def _handle_vulcan_insights(self):
        agent = self._authenticate()
        if not agent:
            self._send_error(401, "Unauthorized")
            return
        if not self.server_instance:
            self._send_error(500, "Server not initialized")
            return
        try:
            from unified_runtime_core import get_runtime

            runtime = get_runtime()
            if not hasattr(runtime, "vulcan_bridge") or not runtime.vulcan_bridge:
                self._send_json(
                    {
                        "vulcan_enabled": False,
                        "message": "VULCAN integration not active",
                    }
                )
                return
            insights = {
                "vulcan_enabled": True,
                "world_model_active": runtime.vulcan_bridge.world_model is not None,
                "reasoning_enabled": getattr(
                    runtime.config, "enable_vulcan_integration", False
                ),
                "capabilities": [
                    "temporal_reasoning",
                    "safety_validation",
                    "goal_alignment",
                    "proposal_evaluation",
                ],
            }
            self._send_json(insights)
        except Exception as e:
            logger.error(f"VULCAN insights error: {e}")
            self._send_error(500, f"Failed to get VULCAN insights: {e}")

    def _send_json(self, data: Dict[str, Any]):
        response = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self._apply_security_headers()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, code: int, message: str):
        response = json.dumps(
            {
                "error": message,
                "code": code,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": self._request_id,
            },
            indent=2,
        ).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self._apply_security_headers()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)


# ======================================================================
# Threaded Server
# ======================================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, server_instance=None, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)
        self.timeout = REQUEST_TIMEOUT

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(
            request, client_address, self, server_instance=self.server_instance
        )


# ======================================================================
# Main API Server Class
# ======================================================================
class GraphAPIServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("GraphAPIServer")
        self.db = DatabaseManager()
        self.rate_limiter = RateLimiter()
        self.execution_engine = ExecutionEngine()
        self.cache = CacheManager()
        self.submissions: Dict[str, GraphSubmission] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.agents: Dict[str, Agent] = {}
        self.locks = {
            "submissions": threading.RLock(),
            "proposals": threading.RLock(),
            "agents": threading.RLock(),
            "reasoner": threading.RLock(),  # Lock for thread-safe reasoner initialization
        }
        self._unified_reasoner = None  # Initialize to None, created on first use
        self.running = False
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.count_lock = threading.RLock()
        self.http_server: Optional[ThreadedHTTPServer] = None
        self.metrics = defaultdict(int)
        self.metrics_lock = threading.RLock()

    def start(self):
        self.running = True
        try:
            self.http_server = ThreadedHTTPServer(
                (self.host, self.port), APIRequestHandler, server_instance=self
            )
            # Optional SSL wrapping (mTLS)
            cert_path = os.environ.get("CERT_PATH")
            key_path = os.environ.get("KEY_PATH")
            ca_path = os.environ.get("CA_CERT_PATH")
            if cert_path and key_path:
                self.logger.info("Starting server with TLS")
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(certfile=cert_path, keyfile=key_path)
                if ca_path:
                    context.load_verify_locations(cafile=ca_path)
                    context.verify_mode = ssl.CERT_REQUIRED
                self.http_server.socket = context.wrap_socket(
                    self.http_server.socket, server_side=True
                )
        except OSError as e:
            if getattr(e, "errno", None) == 48:
                self.logger.error(f"Port {self.port} already in use")
                raise
            raise

        thread = threading.Thread(
            target=self.http_server.serve_forever, daemon=True, name="HTTPServer"
        )
        thread.start()
        self.logger.info(f"Server started on http://{self.host}:{self.port}")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.logger.info("Stopping server...")
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        self.execution_engine.shutdown()
        self.rate_limiter.shutdown()
        self.cache.shutdown()
        # Clean up reasoner if it was initialized
        if hasattr(self, "_unified_reasoner") and self._unified_reasoner:
            try:
                self.logger.info("Shutting down UnifiedReasoner...")
                self._unified_reasoner.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down reasoner: {e}")
        self.db.cleanup()
        self.logger.info("Server stopped")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def submit_graph(
        self,
        graph: Dict,
        agent_id: str,
        priority: int = 0,
        timeout: int = REQUEST_TIMEOUT,
        callback: Optional[str] = None,
    ) -> Dict:
        graph_id = uuid.uuid4().hex
        submission = GraphSubmission(
            id=graph_id,
            graph=graph,
            agent_id=agent_id,
            metadata={
                "version": graph.get("grammar_version", "1.0.0"),
                "priority": priority,
                "timeout": timeout,
                "callback_url": callback,
            },
        )
        with self.locks["submissions"]:
            self.submissions[graph_id] = submission
            self.db.save_graph(submission)
            self.execution_engine.execute_graph(
                submission, callback=lambda s: self._on_graph_complete(s)
            )
            with self.metrics_lock:
                self.metrics["graphs_submitted"] += 1
            with self.count_lock:
                self.request_count += 1
            self.db.log_audit(
                agent_id,
                "submit_graph",
                graph_id,
                {"nodes": len(graph.get("nodes", []))},
            )
        self.logger.info(f"Graph {graph_id} submitted by {agent_id}")
        return {
            "status": "submitted",
            "graph_id": graph_id,
            "queue_position": len(self.execution_engine.executing),
        }

    def _send_callback(self, url: str, data: Dict):
        try:
            import urllib.request

            # Validate URL scheme before making request
            validate_url_scheme(url)

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "GraphixAPIServer/2.2.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=10, encoding="utf-8"
            ) as response:  # nosec B310 - URL validated at line 1775
                if response.status >= 300:
                    self.logger.error(
                        f"Callback to {url} failed with status {response.status}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to send callback to {url}: {e}")

    def _on_graph_complete(self, submission: GraphSubmission):
        self.db.save_graph(submission)
        with self.metrics_lock:
            if submission.status == ExecutionStatus.COMPLETED:
                self.metrics["graphs_completed"] += 1
            else:
                self.metrics["graphs_failed"] += 1
        with self.count_lock:
            if submission.status != ExecutionStatus.COMPLETED:
                self.error_count += 1
        callback_url = submission.metadata.get("callback_url")
        if callback_url:
            self.logger.info(f"Sending callback for {submission.id} to {callback_url}")
            callback_thread = threading.Thread(
                target=self._send_callback,
                args=(callback_url, submission.to_dict()),
                daemon=True,
            )
            callback_thread.start()
        self.logger.info(
            f"Graph {submission.id} completed with status {submission.status.value}"
        )

    def create_proposal(
        self, title: str, description: str, graph: Dict, proposer_id: str
    ) -> Optional[Proposal]:
        if not all([title, description, graph, proposer_id]):
            return None
        proposal_id = uuid.uuid4().hex[:16]
        proposal = Proposal(
            id=proposal_id,
            title=title,
            description=description,
            graph=graph,
            proposer_id=proposer_id,
            closes_at=datetime.utcnow() + timedelta(days=7),
        )
        with self.locks["proposals"]:
            self.proposals[proposal_id] = proposal
        self.logger.info(f"Proposal {proposal_id} created by {proposer_id}")
        return proposal

    def vote_on_proposal(self, proposal_id: str, agent_id: str, vote: str) -> bool:
        with self.locks["proposals"]:
            if proposal_id not in self.proposals:
                return False
            proposal = self.proposals[proposal_id]
            if proposal.closes_at and datetime.utcnow() > proposal.closes_at:
                return False
            if "voters" not in proposal.metadata:
                proposal.metadata["voters"] = []
            if agent_id in proposal.metadata["voters"]:
                return False
            proposal.metadata["voters"].append(agent_id)
            if vote == "for":
                proposal.votes_for += 1
            else:
                proposal.votes_against += 1
            total_votes = proposal.votes_for + proposal.votes_against
            if total_votes >= 3 and proposal.status == "open":
                if proposal.votes_for > proposal.votes_against:
                    proposal.status = "approved"
                    self._apply_proposal(proposal)
                else:
                    proposal.status = "rejected"
        return True

    def _apply_proposal(self, proposal: Proposal):
        self.logger.info(f"Applying approved proposal {proposal.id}")
        # Future implementation placeholder

    def register_agent(
        self,
        name: str,
        roles: Optional[List[str]] = None,
        password: Optional[str] = None,
    ) -> Agent:
        with self.locks["agents"]:
            # Prevent duplicate names
            for a in self.agents.values():
                if a.name.lower() == name.lower():
                    raise ValueError("Agent name already exists")
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            api_key = secrets.token_hex(32)
            password_salt = None
            password_hash = None
            password_algo = None
            if password:
                if ARGON2_AVAILABLE:
                    password_hash = SecurityUtils.argon2_hash(password)
                    password_algo = "argon2"
                else:
                    password_salt = SecurityUtils.generate_salt()
                    password_hash = SecurityUtils.hash_password(
                        password, password_salt, PBKDF2_ITERATIONS, PBKDF2_ALGO
                    )
                    password_algo = f"pbkdf2_sha256${PBKDF2_ITERATIONS}"
            agent = Agent(
                id=agent_id,
                name=name,
                api_key=api_key,
                roles=roles or ["user"],
                password_hash=password_hash,
                password_salt=password_salt,
                password_algo=password_algo,
            )
            self.agents[agent_id] = agent
            self.db.save_agent(agent)
        self.logger.info(f"Agent {agent_id} registered")
        return agent

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        with self.locks["agents"]:
            if agent_id in self.agents:
                return self.agents[agent_id]
        cached = self.cache.get(f"agent:{agent_id}")
        if cached:
            return cached
        # In full implementation we could query DB here.
        return None

    def get_status(self) -> Dict[str, Any]:
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        with self.metrics_lock:
            graphs_submitted = self.metrics["graphs_submitted"]
            graphs_completed = self.metrics["graphs_completed"]
            graphs_failed = self.metrics["graphs_failed"]
            auth_failures = self.metrics["auth_failures"]
            auth_success = self.metrics["auth_success"]
        with self.count_lock:
            total_requests = self.request_count
            total_errors = self.error_count
        with self.locks["proposals"]:
            total_proposals = len(self.proposals)
            open_proposals = len(
                [p for p in self.proposals.values() if p.status == "open"]
            )
            approved_proposals = len(
                [p for p in self.proposals.values() if p.status == "approved"]
            )
        with self.locks["agents"]:
            total_agents = len(self.agents)
        return {
            "status": "active" if self.running else "stopped",
            "version": "2.2.0",
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat(),
            "graphs": {
                "submitted": graphs_submitted,
                "completed": graphs_completed,
                "failed": graphs_failed,
                "executing": len(self.execution_engine.executing),
            },
            "proposals": {
                "total": total_proposals,
                "open": open_proposals,
                "approved": approved_proposals,
            },
            "agents": {"registered": total_agents},
            "auth": {
                "failures": auth_failures,
                "success": auth_success,
                "revoked_tokens": len(revoked_jti_set),
            },
            "requests": {"total": total_requests, "errors": total_errors},
        }

    def get_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            metrics_copy = dict(self.metrics)
        with self.count_lock:
            total_requests = self.request_count
            total_errors = self.error_count
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_copy,
            "performance": {
                "requests_per_second": total_requests / max(uptime, 1),
                "error_rate": total_errors / max(total_requests, 1),
            },
        }


# ======================================================================
# Main Entrypoint
# ======================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Graphix API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind to"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.host == "0.0.0.0":  # nosec B104 - This is a security check, not a binding
        logger.warning("Binding to 0.0.0.0 - ensure firewall is configured!")
    server = GraphAPIServer(host=args.host, port=args.port)
    print("\n" + "=" * 60)
    print("Graphix API Server v2.2.0")
    print("=" * 60)

    alice_password = secrets.token_urlsafe(16)
    bob_password = secrets.token_urlsafe(16)
    admin_password = secrets.token_urlsafe(20)

    alice = server.register_agent(
        "Alice", ["user", "developer"], password=alice_password
    )
    bob = server.register_agent("Bob", ["user"], password=bob_password)
    admin = server.register_agent("Admin", ["admin", "govern"], password=admin_password)

    # Security: Never print full credentials to logs (they can leak via CI artifacts, log aggregators, etc.)
    def _mask(s: str, visible: int = 4) -> str:
        """Mask a secret, showing only first few chars. Always returns same-length mask to avoid leaking length info."""
        if len(s) <= visible:
            return "*" * 8  # Fixed length mask for short secrets
        return s[:visible] + "..." + ("*" * 4)

    print("\nTest Agents Created (credentials masked for security):")
    print(f"  Alice: API Key: {_mask(alice.api_key)}  Password: ********")
    print(f"  Bob:   API Key: {_mask(bob.api_key)}  Password: ********")
    print(f"  Admin: API Key: {_mask(admin.api_key)}  Password: ********")
    print("  Note: Full credentials are stored internally. Use agent IDs for reference.")

    server.start()

    print(f"\nServer running at:")
    print(f"  REST API: http://{args.host}:{args.port}")
    print(f"  Health:   http://{args.host}:{args.port}/health")
    print(f"  Status:   http://{args.host}:{args.port}/status")
    print("\nSecurity:")
    print(f"  JWT Algorithm: {JWT_ALGORITHM}")
    print(f"  JWT Issuer: {JWT_ISS}")
    print(f"  JWT Audience: {JWT_AUD}")
    print(f"  Revocation Backend: {'Redis' if redis_client else 'In-Memory'}")
    print(
        f"  HTTPS/TLS: {'Enabled' if os.environ.get('CERT_PATH') and os.environ.get('KEY_PATH') else 'Not enabled (use reverse proxy in production)'}"
    )
    print("\nPress Ctrl+C to stop\n")

    test_graph = {
        "grammar_version": "1.0.0",
        "id": "test_graph",
        "type": "Graph",
        "nodes": [
            {"id": "n1", "type": "InputNode"},
            {"id": "n2", "type": "OutputNode"},
        ],
        "edges": [{"from": "n1", "to": "n2"}],
    }
    result = server.submit_graph(test_graph, alice.id)
    print(f"Test submission: {result}\n")

    try:
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
