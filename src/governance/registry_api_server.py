"""
Graphix IR Registry API Server (v3.0.0)
========================================

This module implements a gRPC server that exposes the core functionalities
of the Graphix IR Registry to distributed agents and external systems.

Key Features:
- gRPC server for inter-agent communication
- Authentication and authorization
- Proposal submission and voting
- Validation and deployment workflows
- Audit logging and integrity verification
"""

import hashlib
import json
import logging
import os  # Added for serve function environment variables
import sqlite3
import threading  # Added for locking
import time
from abc import ABC, abstractmethod
from concurrent import futures
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path  # Added for unlink in temp_db fixture
from typing import Any, Dict, List, Optional

# --- gRPC Imports (Optional) ---
try:
    import grpc
    # Import specific protobuf types if available
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.timestamp_pb2 import Timestamp
    from grpc import StatusCode

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False
    logging.warning(
        "gRPC not available. Install 'grpcio' and 'protobuf' for full gRPC server functionality."
    )

    # Create mock StatusCode for when grpc is not available
    class StatusCode:
        PERMISSION_DENIED = "PERMISSION_DENIED"
        FAILED_PRECONDITION = "FAILED_PRECONDITION"
        INTERNAL = "INTERNAL"
        OK = "OK"  # Added OK status for mocks

    # Mock Timestamp
    class Timestamp:
        def FromJsonString(self, ts_str):
            # Basic parsing for testing, assumes ISO format like 'YYYY-MM-DDTHH:MM:SS.ffffffZ'
            try:
                # Attempt to parse, but don't store precisely, just validate format roughly
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                logging.warning(f"Mock Timestamp could not parse: {ts_str}")
                pass  # Ignore parsing errors in mock


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Protobuf Message Definitions (Simplified Python Classes) ---
# These act as stand-ins if actual protobuf generated classes aren't available
class Node:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "")
        self.type = kwargs.get("type", "")
        # Ensure metadata is always a dict
        metadata = kwargs.get("metadata", {})
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.proposed_by = kwargs.get("proposed_by", "")
        self.rationale = kwargs.get("rationale", "")
        # Store content as bytes, assuming it might be serialized proto or JSON bytes
        self.proposal_content = kwargs.get("proposal_content", b"")
        self.target = kwargs.get("target", "")  # Used in ValidationNode
        self.validation_type = kwargs.get(
            "validation_type", ""
        )  # Used in ValidationNode
        self.result = kwargs.get("result", False)  # Used in ValidationNode
        # Ensure votes is always a dict
        votes = kwargs.get("votes", {})  # Used in ConsensusNode
        self.votes = votes if isinstance(votes, dict) else {}
        self.quorum = kwargs.get("quorum", 0.0)  # Used in ConsensusNode
        # Add proposal_id field often used in consensus/validation nodes referencing proposals
        self.proposal_id = kwargs.get(
            "proposal_id", self.id or self.target or ""
        )  # Try id or target if proposal_id missing


class AuditLogEntry:
    def __init__(self, **kwargs):
        # Use a default Timestamp or a string representation if no gRPC
        self.timestamp = kwargs.get(
            "timestamp",
            Timestamp() if HAS_GRPC else datetime.utcnow().isoformat() + "Z",
        )
        self.action = kwargs.get("action", "")
        # Store details as bytes (expected to be JSON bytes)
        details_arg = kwargs.get("details", {})
        if isinstance(details_arg, bytes):
            self.details = details_arg
        elif isinstance(details_arg, dict):
            self.details = json.dumps(details_arg).encode("utf-8")
        else:
            self.details = b"{}"
        self.entity_id = kwargs.get("entity_id", "")
        self.entity_type = kwargs.get("entity_type", "")


# Request/Response Messages (Simplified Python Classes)
class RegisterGraphProposalRequest:
    def __init__(self, agent_id="", signature="", proposal_node=None):
        self.agent_id = agent_id
        self.signature = signature
        self.proposal_node = proposal_node if proposal_node else Node()


class RegisterGraphProposalResponse:
    def __init__(self, status="", message="", proposal_id=""):
        self.status = status
        self.message = message
        self.proposal_id = proposal_id


class SubmitLanguageEvolutionProposalRequest:
    def __init__(self, agent_id="", signature="", proposal_node=None):
        self.agent_id = agent_id
        self.signature = signature
        self.proposal_node = proposal_node if proposal_node else Node()


class SubmitLanguageEvolutionProposalResponse:
    def __init__(self, status="", message="", proposal_id=""):
        self.status = status
        self.message = message
        self.proposal_id = proposal_id


class RecordVoteRequest:
    def __init__(self, agent_id="", signature="", consensus_node=None):
        self.agent_id = agent_id
        self.signature = signature
        # Ensure consensus_node uses 'id' for proposal_id if that's how test passes it
        self.consensus_node = consensus_node if consensus_node else Node()
        if not self.consensus_node.proposal_id and self.consensus_node.id:
            self.consensus_node.proposal_id = self.consensus_node.id


class RecordVoteResponse:
    def __init__(self, status="", message="", consensus_reached=False):
        self.status = status
        self.message = message
        self.consensus_reached = consensus_reached


class RecordValidationRequest:
    def __init__(self, agent_id="", signature="", validation_node=None):
        self.agent_id = agent_id
        self.signature = signature
        self.validation_node = validation_node if validation_node else Node()


class RecordValidationResponse:
    def __init__(self, status="", message="", validation_passed=False):
        self.status = status
        self.message = message
        self.validation_passed = validation_passed


class DeployGrammarVersionRequest:
    def __init__(
        self, agent_id="", signature="", proposal_id="", new_grammar_version=""
    ):
        self.agent_id = agent_id
        self.signature = signature
        self.proposal_id = proposal_id
        self.new_grammar_version = new_grammar_version


class DeployGrammarVersionResponse:
    def __init__(self, status="", message="", deployed=False):
        self.status = status
        self.message = message
        self.deployed = deployed


class QueryProposalsRequest:
    def __init__(
        self, agent_id="", status=None, proposed_by=None, limit=None, offset=0
    ):
        self.agent_id = agent_id
        # Store optional fields correctly
        self._status = status
        self._proposed_by = proposed_by
        self._limit = limit
        self.offset = offset

    # Provide properties to access optional fields like real protos might
    @property
    def status(self):
        return self._status

    @property
    def proposed_by(self):
        return self._proposed_by

    @property
    def limit(self):
        return self._limit

    def HasField(self, field_name):
        # Check if the internal attribute holding the optional value is not None
        if field_name == "status":
            return self._status is not None
        if field_name == "proposed_by":
            return self._proposed_by is not None
        if field_name == "limit":
            return self._limit is not None
        # Fallback for other potential fields (though none are optional here)
        return hasattr(self, field_name) and getattr(self, field_name) is not None


class QueryProposalsResponse:
    def __init__(self, status="", message="", proposals=None):
        self.status = status
        self.message = message
        self.proposals = proposals if proposals else []


class GetFullAuditLogRequest:
    def __init__(self, agent_id="", signature=""):
        self.agent_id = agent_id
        self.signature = signature


class GetFullAuditLogResponse:
    def __init__(self, status="", message="", audit_log=None):
        self.status = status
        self.message = message
        self.audit_log = audit_log if audit_log else []


class VerifyAuditLogIntegrityRequest:
    def __init__(self, agent_id=""):
        self.agent_id = agent_id


class VerifyAuditLogIntegrityResponse:
    def __init__(self, status="", message="", integrity_valid=False):
        self.status = status
        self.message = message
        self.integrity_valid = integrity_valid


# --- Service Base Classes (Abstract) ---
class RegistryServiceServicer(ABC):
    """Abstract base class for registry service implementation."""

    @abstractmethod
    def RegisterGraphProposal(self, request, context):
        pass

    @abstractmethod
    def SubmitLanguageEvolutionProposal(self, request, context):
        pass

    @abstractmethod
    def RecordVote(self, request, context):
        pass

    @abstractmethod
    def RecordValidation(self, request, context):
        pass

    @abstractmethod
    def DeployGrammarVersion(self, request, context):
        pass

    @abstractmethod
    def QueryProposals(self, request, context):
        pass

    @abstractmethod
    def GetFullAuditLog(self, request, context):
        pass

    @abstractmethod
    def VerifyAuditLogIntegrity(self, request, context):
        pass


# --- Persistent Storage ---
DB_PATH = "registry.db"  # Default path
DB_POOL_SIZE = 5


class DatabaseConnectionPool:
    """Thread-safe SQLite connection pool."""

    # FIX: Added timeout parameter
    def __init__(
        self, db_path: str, pool_size: int = DB_POOL_SIZE, timeout: float = 5.0
    ):
        self._db_path = db_path
        self._pool_size = pool_size
        self._timeout = timeout  # Store timeout
        self._connections = list(]
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # Initialize connections - crucial for pool logic
        for _ in range(pool_size):
            try:
                self._connections.append(self._create_connection())
            except sqlite3.Error as e:
                # Log error if initial connection fails, but continue trying for others
                logging.error(f"Failed to create initial DB connection: {e}")
        if not self._connections and pool_size > 0:
            logging.error(
                f"Failed to initialize any database connections for path: {db_path}"
            )
            # Consider raising an error if zero connections are acceptable

    def _create_connection(self):
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # Use a longer internal timeout for SQLite operations than the pool wait timeout
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=max(10.0, self._timeout + 5.0),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")  # Improve concurrency
        return conn

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool, waiting up to the timeout."""
        conn = None
        start_time = time.monotonic()  # Use monotonic clock for timeout calculation
        acquired = False
        with self._condition:
            while not acquired:  # Loop until connection acquired or timeout
                if self._connections:
                    conn = self._connections.pop()
                    acquired = True
                else:
                    # Calculate remaining time accurately
                    elapsed_time = time.monotonic() - start_time
                    remaining_time = self._timeout - elapsed_time
                    if remaining_time <= 0:
                        raise RuntimeError(
                            f"Connection pool exhausted - timeout ({self._timeout}s) waiting for available connection"
                        )
                    # Wait for remaining time or until notified
                    self._condition.wait(timeout=remaining_time)
                    # After waiting, loop will re-check self._connections
        try:
            yield conn
        finally:
            if conn:  # Ensure conn was actually acquired before returning
                with self._condition:
                    # Simple return to pool
                    self._connections.append(conn)
                    # Notify one waiting thread that a connection is available
                    self._condition.notify()


class DatabaseManager:
    """Manages all database interactions with connection pooling and retries."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger("DatabaseManager")  # Init logger first
        try:
            # Ensure parent directory exists before creating pool
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.pool = DatabaseConnectionPool(db_path)
            self._init_database()
        except Exception as e:
            self.logger.exception(
                f"FATAL: DatabaseManager failed to initialize pool or DB at {db_path}: {e}"
            )
            # Depending on application, might re-raise or exit
            raise

    def _init_database(self):
        """Initialize all database tables if they don't exist."""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                # Use IF NOT EXISTS for idempotency
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_proposals (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lang_proposals (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agents (
                        agent_id TEXT PRIMARY KEY,
                        profile_data TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                """)
                # Add index on audit log timestamp for faster ordering
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )
                conn.commit()
                self.logger.info(
                    "Database schema initialized (tables created if not exists)."
                )
        except sqlite3.Error as e:
            self.logger.error(f"Database schema initialization failed: {e}")
        except RuntimeError as e:  # Catch pool exhaustion during init
            self.logger.error(
                f"Database schema initialization failed - connection pool issue: {e}"
            )

    # Security: Allowlist for table and column names to prevent SQL injection
    ALLOWED_TABLES = {
        'graph_proposals': {'id_column': 'id', 'data_column': 'data'},
        'lang_proposals': {'id_column': 'id', 'data_column': 'data'},
        'agents': {'id_column': 'agent_id', 'data_column': 'profile_data'},
        'audit_log': {'id_column': 'id', 'data_column': 'data'}
    }

    def _validate_table_name(self, table_name: str) -> str:
        """Validate table name against allowlist to prevent SQL injection."""
        if table_name not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table_name}. Allowed tables: {list(self.ALLOWED_TABLES.keys())}")
        return table_name

    def _get_column_names(self, table_name: str) -> Dict[str, str]:
        """Get validated column names for a table."""
        validated_table = self._validate_table_name(table_name)
        return self.ALLOWED_TABLES[validated_table]

    def _exec_query(self, query, params=(), fetch_one=False, fetch_all=False):
        """Helper to execute a DB query with retries for busy errors."""
        max_retries = 3
        base_delay = 0.05  # Start with 50ms delay
        for attempt in range(max_retries):
            try:
                with self.pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    if fetch_one:
                        return cursor.fetchone()
                    if fetch_all:
                        return cursor.fetchall()
                    conn.commit()
                    return None  # Indicate success for commit operations
            except sqlite3.OperationalError as e:
                # Check specifically for "database is locked"
                if "database is locked" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        self.logger.warning(
                            f"Database locked, retrying query in {delay:.2f}s (attempt {attempt + 1}/{max_retries}). Query: {query[:100]}..."
                        )
                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"Database locked after {max_retries} attempts. Query: {query[:100]}"
                        )
                        raise  # Reraise if retries exhausted
                else:
                    self.logger.error(
                        f"Database operational error: {e}. Query: {query[:100]}"
                    )
                    raise  # Reraise other operational errors immediately
            except Exception as e:
                self.logger.exception(
                    f"Unexpected database error on query: {query[:100]}. Error: {e}"
                )
                raise  # Reraise other exceptions

    # Generic CRUD methods using _exec_query
    def get_record(self, table_name: str, record_id: str) -> Optional[Dict]:
        """Retrieve a record by ID, parsing its JSON data."""
        # Validate table name to prevent SQL injection
        validated_table = self._validate_table_name(table_name)
        columns = self._get_column_names(validated_table)
        id_column = columns['id_column']
        data_column = columns['data_column']

        try:
            # nosec B608: table/column names validated above via _validate_table_name
            row = self._exec_query(
                f"SELECT {data_column} FROM {validated_table} WHERE {id_column} = ?",  # nosec B608
                (record_id,),
                fetch_one=True,
            )
            if row and len(row) > 0 and row[0] is not None:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Failed to decode JSON data for {validated_table} ID {record_id}: {e}"
                    )
                    return None  # Return None if data is corrupt
            return None  # Record not found
        except Exception as e:
            self.logger.error(
                f"Error getting record {record_id} from {validated_table}: {e}"
            )
            return None  # Return None on DB error

    def save_record(self, table_name: str, record_id: str, data: Dict):
        """Save or update a record, storing data as JSON."""
        # Validate table name to prevent SQL injection
        validated_table = self._validate_table_name(table_name)
        columns = self._get_column_names(validated_table)
        id_column = columns['id_column']
        data_column = columns['data_column']

        try:
            self._exec_query(
                f"INSERT OR REPLACE INTO {validated_table} ({id_column}, {data_column}) VALUES (?, ?)",
                (record_id, json.dumps(data)),
            )
        except Exception as e:
            self.logger.error(f"Error saving record {record_id} to {validated_table}: {e}")
            # Optionally re-raise depending on desired error handling

    def query_records(
        self,
        table_name: str,
        where_clause: str = "",
        params=(),
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict]:
        """Query records with optional filters, limit, and offset, parsing JSON data."""
        # Validate table name to prevent SQL injection
        validated_table = self._validate_table_name(table_name)
        columns = self._get_column_names(validated_table)
        data_column = columns['data_column']
        id_column = columns['id_column']

        # nosec B608: table/column names validated above via _validate_table_name
        query = f"SELECT {data_column} FROM {validated_table}"  # nosec B608
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" ORDER BY {id_column}"  # Add default ordering
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"
        if offset > 0:
            query += f" OFFSET {int(offset)}"

        results = []
        try:
            rows = self._exec_query(query, params, fetch_all=True)
            if rows:
                for row in rows:
                    if row and len(row) > 0 and row[0] is not None:
                        try:
                            results.append(json.loads(row[0]))
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"Failed to decode record data from {validated_table} during query: {e}"
                            )
                            # Skip corrupted records
        except Exception as e:
            self.logger.error(f"Error querying records from {validated_table}: {e}")
            # Return empty list or re-raise depending on desired behavior
        return results

    def log_audit(self, data: Dict):
        """Log an audit entry, storing data as JSON."""
        ts = data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        try:
            self._exec_query(
                "INSERT INTO audit_log (timestamp, data) VALUES (?, ?)",
                (ts, json.dumps(data)),
            )
        except Exception as e:
            self.logger.error(f"Error logging audit data: {e}")

    def get_full_audit_log(self) -> List[Dict]:
        """Retrieve all audit logs, ordered by timestamp, parsing JSON data."""
        query = "SELECT data FROM audit_log ORDER BY timestamp ASC"
        results = []
        try:
            rows = self._exec_query(query, fetch_all=True)
            if rows:
                for row in rows:
                    if row and len(row) > 0 and row[0] is not None:
                        try:
                            results.append(json.loads(row[0]))
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to decode audit log data: {e}")
                            # Skip corrupted records
        except Exception as e:
            self.logger.error(f"Error getting full audit log: {e}")
        return results


# --- Service Components (now using DatabaseManager) ---
class RegistryAPI:
    """Persistent RegistryAPI for graph storage."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("RegistryAPI")

    def submit_proposal(self, proposal_node: Dict) -> str:
        prop_id = proposal_node.get(
            "id",
            f"graph_prop_{hashlib.sha256(json.dumps(proposal_node, sort_keys=True).encode()).hexdigest()[:8]}",
        )
        # Ensure 'id' is in the node itself before saving
        if "id" not in proposal_node:
            proposal_node["id"] = prop_id
        proposal_data = {
            "node": proposal_node,
            "status": "submitted",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.db.save_record("graph_proposals", prop_id, proposal_data)
        self.logger.info(f"Graph proposal {prop_id} submitted.")
        return prop_id

    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        return self.db.get_record("graph_proposals", proposal_id)

    def query_proposals(
        self,
        status: Optional[str] = None,
        proposed_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict]:
        all_proposals_data = self.db.query_records(
            "graph_proposals"
        )  # Fetches full data objects
        results = []
        # Filter in Python
        for pdata in all_proposals_data:
            node_data = pdata.get("node", {})  # Safely get node data
            match_status = status is None or pdata.get("status") == status
            match_proposer = (
                proposed_by is None or node_data.get("proposed_by") == proposed_by
            )
            if match_status and match_proposer:
                results.append(node_data)  # Return the node content

        # Apply limit and offset after filtering
        start = max(0, offset)
        end = start + limit if limit is not None and limit > 0 else None
        return results[start:end]


class LanguageEvolutionRegistry:
    """Persistent LanguageEvolutionRegistry for grammar evolution."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.active_grammar_version = "3.0.0"  # Default initial version
        self.logger = logging.getLogger("LanguageEvolutionRegistry")
        self._proposal_lock = threading.Lock()  # Add this lock

    def submit_proposal(self, proposal_node: Dict) -> str:
        prop_id = proposal_node.get(
            "id",
            f"lang_prop_{hashlib.sha256(json.dumps(proposal_node, sort_keys=True).encode()).hexdigest()[:8]}",
        )
        # Ensure 'id' is in the node itself
        if "id" not in proposal_node:
            proposal_node["id"] = prop_id
        proposal_data = {
            "node": proposal_node,
            "status": "pending",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "votes": {},
            "validations": {},
        }
        self.db.save_record("lang_proposals", prop_id, proposal_data)
        self.logger.info(f"Language proposal {prop_id} submitted.")
        return prop_id

    def record_vote(self, consensus_node: Dict) -> bool:
        prop_id = consensus_node.get("proposal_id")  # Use specific key from dict
        if not prop_id:
            self.logger.error("Missing proposal_id in consensus node dict.")
            return False

        # Use the lock to make read-modify-write atomic for this proposal
        with self._proposal_lock:
            proposal = self.db.get_record("lang_proposals", prop_id)
            if not proposal:
                self.logger.error(f"Proposal {prop_id} not found for voting.")
                return False

            if "votes" not in proposal or not isinstance(proposal["votes"], dict):
                proposal["votes"] = {}
            # Update votes - important: ensure this updates the 'proposal' dict directly
            proposal["votes"].update(consensus_node.get("votes", {}))

            # Example consensus logic (adjust as needed)
            # Requires quorum (e.g., 50%) and majority 'yes' votes
            num_votes = len(proposal["votes"])
            yes_votes = sum(1 for vote in proposal["votes"].values() if vote == "yes")
            no_votes = sum(1 for vote in proposal["votes"].values() if vote == "no")
            # Assume quorum refers to participation threshold (e.g., requires > 50% of potential voters - hard to check here)
            # Simplified: check if yes > no
            approved = yes_votes > no_votes
            rejected = (
                no_votes >= yes_votes and num_votes > 0
            )  # Reject on tie or majority no

            original_status = proposal.get("status")  # Keep track if status changes
            status_changed = False

            if approved and proposal.get("status") != "approved":
                proposal["status"] = "approved"
                self.logger.info(f"Proposal {prop_id} approved by vote.")
                status_changed = True
            elif rejected and proposal.get("status") != "rejected":
                proposal["status"] = "rejected"
                self.logger.info(f"Proposal {prop_id} rejected by vote.")
                status_changed = True

            # Only save if votes were added or status changed to avoid unnecessary writes
            # Check if the votes dict actually changed or if status changed
            # Note: This simple check might not be perfect if the update adds existing votes,
            # but it's better than always writing. A deep comparison could be used if needed.
            if consensus_node.get("votes", {}) or status_changed:
                self.db.save_record("lang_proposals", prop_id, proposal)
            else:
                self.logger.debug(
                    f"No change in votes or status for {prop_id}, skipping save."
                )

        # Return status *after* releasing the lock
        return approved

    def record_validation(self, validation_node: Dict) -> bool:
        prop_id = validation_node.get("target")
        if not prop_id:
            self.logger.error("Missing target proposal_id in validation node.")
            return False

        proposal = self.db.get_record("lang_proposals", prop_id)
        if not proposal:
            self.logger.error(f"Proposal {prop_id} not found for validation.")
            return False

        validation_result = validation_node.get("result", False)

        if "validations" not in proposal or not isinstance()
            proposal["validations"], dict
        ):
            proposal["validations"] = {}
        validator_key = validation_node.get(
            "validator_id", validation_node.get("validation_type", str(time.time()))
        )
        proposal["validations"][validator_key] = validation_result

        # Update status based on this validation (assuming one pass is enough)
        if validation_result is True:
            proposal["status"] = "validated"
            self.logger.info(f"Proposal {prop_id} validated successfully.")
        else:
            proposal["status"] = "validation_failed"
            self.logger.warning(f"Proposal {prop_id} failed validation.")

        self.db.save_record("lang_proposals", prop_id, proposal)
        return validation_result

    def deploy_grammar_version(self, proposal_id: str, new_version: str) -> bool:
        proposal = self.db.get_record("lang_proposals", proposal_id)
        if not proposal or proposal.get("status") != "validated":
            self.logger.error(
                f"Cannot deploy: Proposal {proposal_id} not in 'validated' state. Status is {proposal.get('status') if proposal else 'not found'}."
            )
            return False

        self.active_grammar_version = new_version
        proposal["status"] = "deployed"
        proposal["deployed_version"] = new_version
        self.db.save_record("lang_proposals", proposal_id, proposal)
        self.logger.info(
            f"Grammar version {new_version} deployed for proposal {proposal_id}."
        )
        return True

    def query_proposals(
        self,
        status: Optional[str] = None,
        proposed_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict]:
        all_proposals_data = self.db.query_records("lang_proposals")
        results = []
        for pdata in all_proposals_data:
            node_data = pdata.get("node", {})
            match_status = status is None or pdata.get("status") == status
            match_proposer = (
                proposed_by is None or node_data.get("proposed_by") == proposed_by
            )
            if match_status and match_proposer:
                results.append(node_data)
        start = max(0, offset)
        end = start + limit if limit is not None and limit > 0 else None
        return results[start:end]


class AgentRegistry:
    """Manages agent identities and authentication via database."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("AgentRegistry")

    def register_agent(self, agent_data: Dict[str, Any]):
        """Register a new agent or update existing."""
        agent_id = agent_data.get("id")
        if not agent_id:
            self.logger.error("Attempted to register agent without an 'id'.")
            raise ValueError("Agent data must contain 'id'")
        try:
            profile = {  # Build the profile dict to be stored
                "agent_id": agent_id,
                "name": agent_data.get("name", agent_id),
                "roles": agent_data.get("roles", []),
                "trust_level": agent_data.get("trust_level", 0.5),
                "metadata": agent_data.get("metadata", {}),
                "is_active": agent_data.get(
                    "is_active", True
                ),  # Assume active by default
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            existing_profile = self.get_agent_info(agent_id)  # Check if updating
            profile["created_at"] = (
                existing_profile.get("created_at")
                if existing_profile
                else profile["updated_at"]
            )
            # Save the profile dict as JSON in the 'profile_data' column
            self.db.save_record("agents", agent_id, profile)
            self.logger.info(f"Agent '{agent_id}' registered or updated.")
        except Exception as e:
            self.logger.exception(f"Failed to register agent {agent_id}: {e}")
            raise

    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """Retrieve full profile information for a specific agent."""
        # get_record retrieves the JSON blob and parses it
        return self.db.get_record("agents", agent_id)

    def authenticate_agent(
        self, agent_id: str, message: str, signature_hex: str
    ) -> bool:
        """Authenticate agent based on simplified signature check."""
        self.logger.debug(f"Authenticating agent: {agent_id}")
        agent_info = self.get_agent_info(agent_id)
        if not agent_info:
            self.logger.warning(f"Authentication failed: Agent {agent_id} not found.")
            return False
        # Simplified check (replace with actual crypto verification)
        # Check if signature matches hash of message
        expected_sig = hashlib.sha256(message.encode()).hexdigest()
        is_valid = signature_hex == expected_sig
        if is_valid:
            self.logger.debug(f"Agent {agent_id} authenticated successfully.")
        else:
            self.logger.warning(
                f"Authentication failed for {agent_id}: Invalid signature."
            )
        return is_valid

    def verify_agent_signature(
        self, agent_id: str, message: bytes, signature_hex: str
    ) -> bool:
        """Verify agent signature (replace with actual crypto)."""
        agent_info = self.get_agent_info(agent_id)
        if not agent_info:
            return False
        expected_sig = hashlib.sha256(message).hexdigest()
        return signature_hex == expected_sig

    def query_agents(
        self,
        role: Optional[str] = None,
        min_trust_level: float = 0.0,
        status: Optional[str] = "active",
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Query agents by filtering their profile data."""
        all_agents = self.db.query_records("agents")  # Retrieves list of profile dicts
        results = []
        for agent in all_agents:
            # Apply filters safely using .get() with defaults
            is_active = agent.get(
                "is_active", True
            )  # Default to active if not specified
            trust_level = agent.get("trust_level", 0.0)
            roles = agent.get("roles", [])

            status_match = (
                status is None
                or (status == "active" and is_active)
                or (status == "inactive" and not is_active)
            )
            trust_match = trust_level >= min_trust_level
            role_match = role is None or role in roles

            if status_match and trust_match and role_match:
                results.append(agent)
        # Apply limit after filtering
        return results[:limit] if limit is not None and limit >= 0 else results


class SecurityAuditEngine:
    """Provides security policy enforcement and persistent audit logging."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("SecurityAuditEngine")

    def log_audit(
        self,
        action: str,
        details: Dict,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ):
        """Log an audit event to the database."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "details": details,  # Store details as dict
            "entity_id": entity_id,
            "entity_type": entity_type,
        }
        try:
            self.db.log_audit(log_entry)
            self.logger.info(
                f"AUDIT: {action} logged for {entity_type or 'N/A'}:{entity_id or 'N/A'}"
            )
        except Exception as e:
            self.logger.error(f"Failed to log audit event {action}: {e}")

    def enforce_policies(self, node: Dict) -> bool:
        """Enforce basic security policies on node content."""
        content_to_scan = ""
        # Safely build string from potential content fields
        for key in ["content", "proposal_content", "rationale", "code"]:
            content = node.get(key)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            if isinstance(content, (str, dict, list)):
                content_to_scan += json.dumps(content) + " "
            elif content is not None:
                content_to_scan += str(content) + " "
        content_lower = content_to_scan.lower()
        malicious_patterns = [
            "malicious",
            "exploit",
            "hack",
            "os.system('rm -rf /')",
            "<script>",
            "onerror=",
        ]
        if any(pattern in content_lower for pattern in malicious_patterns):
            self.log_audit(
                "policy_violation",
                {"reason": "malicious_content_detected"},
                node.get("id"),
                node.get("type"),
            )
            self.logger.warning(
                f"Security policy violation: Malicious pattern in node {node.get('id')}"
            )
            return False
        return True

    def validate_trust_policy(self, node: Dict, trust_level: float) -> bool:
        """Validate if agent trust level meets policy requirements."""
        required_trust = 0.3  # Example threshold
        if trust_level < required_trust:
            self.log_audit(
                "trust_violation",
                {"reason": "insufficient_trust"},
                node.get("proposed_by"),
                "agent",
            )
            self.logger.warning(
                f"Trust policy violation: Agent {node.get('proposed_by')} trust {trust_level} < {required_trust}"
            )
            return False
        return True

    def get_full_audit_log(self) -> List[Dict]:
        """Retrieve the entire audit log, ordered by timestamp."""
        return self.db.get_full_audit_log()

    def verify_audit_log_integrity(self) -> bool:
        """Verify the integrity of the audit log (placeholder)."""
        self.logger.info("Verifying audit log integrity (mock implementation).")
        try:
            _ = self.get_full_audit_log()  # Check if retrieval works
            return True
        except Exception as e:
            self.logger.error(f"Failed integrity check while retrieving audit log: {e}")
            return False


# --- gRPC Service Implementation ---
class RegistryServicer(RegistryServiceServicer):
    """Implements the gRPC service methods using persistent components."""

    def __init__(
        self,
        registry_api: RegistryAPI,
        lang_evolution_registry: LanguageEvolutionRegistry,
        agent_registry: AgentRegistry,
        security_audit_engine: SecurityAuditEngine,
    ):
        self.registry_api = registry_api
        self.lang_evolution_registry = lang_evolution_registry
        self.agent_registry = agent_registry
        self.security_audit_engine = security_audit_engine
        self.logger = logging.getLogger("RegistryServicer")

    def _authenticate_request(
        self, agent_id: str, message_content: str, signature: str
    ) -> bool:
        """Authenticate a request using AgentRegistry."""
        return self.agent_registry.authenticate_agent(
            agent_id, message_content, signature
        )

    def _authorize_request(self, agent_id: str, required_roles: List[str]) -> bool:
        """Authorize a request based on agent roles using AgentRegistry."""
        agent_info = self.agent_registry.get_agent_info(agent_id)
        if not agent_info:
            self.logger.warning(f"Authorization failed: Agent {agent_id} not found.")
            return False
        agent_roles = set(agent_info.get("roles", []))
        has_required_role = any(role in agent_roles for role in required_roles)
        if not has_required_role:
            self.logger.warning(
                f"Authorization failed: Agent {agent_id} lacks required roles {required_roles}. Has: {list(agent_roles)}"
            )
            return False
        self.logger.debug(
            f"Authorization successful for {agent_id} (required: {required_roles})"
        )
        return True

    # Helper to safely get metadata dict from Node (handles gRPC MapField or Python dict)
    def _get_metadata_dict(self, node_metadata):
        """Safely convert protobuf metadata (or dict) to dict."""
        # FIX: Check if it's a real protobuf MapField (which has DESCRIPTOR) and HAS_GRPC is True
        if HAS_GRPC and hasattr(node_metadata, "DESCRIPTOR"):
            try:
                # Use MessageToDict for proper conversion of Protobuf MapField
                return MessageToDict(node_metadata)
            except Exception as e:
                self.logger.warning(
                    f"Could not convert proto metadata map to dict: {e}"
                )
                return {}  # Fallback to empty dict
        elif isinstance(node_metadata, dict):
            return node_metadata  # It's already a Python dict
        else:
            self.logger.debug(
                f"Metadata field was not a dict or proto map, received {type(node_metadata)}. Defaulting to empty dict."
            )
            return {}  # Fallback for unexpected types

    def RegisterGraphProposal(self, request, context):
        """Register a new graph proposal."""
        agent_id = request.agent_id
        signature = request.signature
        self.logger.info(
            f"Received RegisterGraphProposal request from agent: {agent_id}"
        )

        try:
            # Safely parse proposal_content (assuming JSON bytes)
            proposal_content_dict = {}
            if request.proposal_node.proposal_content:
                proposal_content_dict = json.loads(
                    request.proposal_node.proposal_content.decode("utf-8")
                )

            # FIX: Safely get metadata using the helper
            metadata_dict = self._get_metadata_dict(request.proposal_node.metadata)

            proposal_node_dict = {
                "id": request.proposal_node.id,
                "type": request.proposal_node.type,
                "metadata": metadata_dict,  # Use converted dict
                "proposed_by": request.proposal_node.proposed_by or agent_id,
                "rationale": request.proposal_node.rationale,
                "proposal_content": proposal_content_dict,
            }
            message_for_auth = json.dumps(proposal_node_dict, sort_keys=True)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Invalid proposal content JSON from {agent_id}: {e}")
            context.set_code(StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Invalid proposal content JSON: {e}")
            return RegisterGraphProposalResponse(
                status="error", message=f"Invalid proposal content JSON: {e}"
            )
        except Exception as e:  # Catch other unexpected errors during prep
            self.logger.error(
                f"Error processing proposal data from {agent_id}: {e}", exc_info=True
            )  # Log stack trace
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Error processing proposal data: {e}")
            return RegisterGraphProposalResponse(
                status="error", message=f"Error processing proposal data: {e}"
            )

        # --- Authentication & Authorization ---
        if not self._authenticate_request(agent_id, message_for_auth, signature):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Authentication failed.")
            return RegisterGraphProposalResponse(
                status="error", message="Authentication failed."
            )
        if not self._authorize_request(agent_id, ["proposer", "governor"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Authorization failed: Missing role.")
            return RegisterGraphProposalResponse(
                status="error", message="Authorization failed."
            )

        # --- Security & Trust Policies ---
        try:
            if not self.security_audit_engine.enforce_policies(proposal_node_dict):
                context.set_code(StatusCode.FAILED_PRECONDITION)
                context.set_details("Violates security policies.")
                return RegisterGraphProposalResponse(
                    status="error", message="Security policy violation."
                )

            agent_info = self.agent_registry.get_agent_info(agent_id)
            trust_level = agent_info.get("trust_level", 0.0) if agent_info else 0.0
            if not self.security_audit_engine.validate_trust_policy(
                proposal_node_dict, trust_level
            ):
                context.set_code(StatusCode.PERMISSION_DENIED)
                context.set_details("Agent trust level too low.")
                return RegisterGraphProposalResponse(
                    status="error", message="Agent trust level too low."
                )

            # --- Submit Proposal ---
            proposal_id = self.registry_api.submit_proposal(proposal_node_dict)
            self.security_audit_engine.log_audit(
                "graph_proposal_registered", {"id": proposal_id}, agent_id, "agent"
            )
            self.logger.info(f"Graph proposal {proposal_id} registered by {agent_id}.")
            return RegisterGraphProposalResponse(
                status="success", proposal_id=proposal_id
            )

        except Exception as e:
            self.logger.exception(
                f"Internal error in RegisterGraphProposal for {agent_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return RegisterGraphProposalResponse(
                status="error", message=f"Internal server error: {e}"
            )

    def SubmitLanguageEvolutionProposal(self, request, context):
        """Submit a language evolution proposal."""
        agent_id = request.agent_id
        signature = request.signature
        self.logger.info(
            f"Received SubmitLanguageEvolutionProposal request from agent: {agent_id}"
        )

        try:
            # Safely parse proposal_content (assuming JSON bytes)
            proposal_content_dict = {}
            if request.proposal_node.proposal_content:
                proposal_content_dict = json.loads(
                    request.proposal_node.proposal_content.decode("utf-8")
                )

            # FIX: Safely get metadata using the helper
            metadata_dict = self._get_metadata_dict(request.proposal_node.metadata)

            proposal_node_dict = {
                "id": request.proposal_node.id,
                "type": request.proposal_node.type,
                "metadata": metadata_dict,  # Use converted dict
                "proposed_by": request.proposal_node.proposed_by or agent_id,
                "rationale": request.proposal_node.rationale,
                "proposal_content": proposal_content_dict,
            }
            message_for_auth = json.dumps(proposal_node_dict, sort_keys=True)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Invalid proposal content JSON from {agent_id}: {e}")
            context.set_code(StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Invalid proposal content JSON: {e}")
            return SubmitLanguageEvolutionProposalResponse(
                status="error", message=f"Invalid proposal content JSON: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Error processing proposal data from {agent_id}: {e}", exc_info=True
            )  # Log stack trace
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Error processing proposal data: {e}")
            return SubmitLanguageEvolutionProposalResponse(
                status="error", message=f"Error processing proposal data: {e}"
            )

        # --- Auth & Security ---
        if not self._authenticate_request(agent_id, message_for_auth, signature):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Authentication failed.")
            return SubmitLanguageEvolutionProposalResponse(
                status="error", message="Authentication failed."
            )
        if not self._authorize_request(agent_id, ["proposer", "governor"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Authorization failed.")
            return SubmitLanguageEvolutionProposalResponse(
                status="error", message="Authorization failed."
            )
        try:
            if not self.security_audit_engine.enforce_policies(proposal_node_dict):
                context.set_code(StatusCode.FAILED_PRECONDITION)
                context.set_details("Violates security policies.")
                return SubmitLanguageEvolutionProposalResponse(
                    status="error", message="Security policy violation."
                )
            agent_info = self.agent_registry.get_agent_info(agent_id)
            trust_level = agent_info.get("trust_level", 0.0) if agent_info else 0.0
            if not self.security_audit_engine.validate_trust_policy(
                proposal_node_dict, trust_level
            ):
                context.set_code(StatusCode.PERMISSION_DENIED)
                context.set_details("Agent trust level too low.")
                return SubmitLanguageEvolutionProposalResponse(
                    status="error", message="Agent trust level too low."
                )

            # --- Submit ---
            proposal_id = self.lang_evolution_registry.submit_proposal(
                proposal_node_dict
            )
            self.security_audit_engine.log_audit(
                "language_proposal_submitted", {"id": proposal_id}, agent_id, "agent"
            )
            self.logger.info(
                f"Language proposal {proposal_id} submitted by {agent_id}."
            )
            return SubmitLanguageEvolutionProposalResponse(
                status="success", proposal_id=proposal_id
            )

        except Exception as e:
            self.logger.exception(
                f"Internal error in SubmitLanguageEvolutionProposal for {agent_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return SubmitLanguageEvolutionProposalResponse(
                status="error", message=f"Internal server error: {e}"
            )

    def RecordVote(self, request, context):
        agent_id = request.agent_id
        signature = request.signature
        self.logger.info(f"Received RecordVote request from agent: {agent_id}")
        try:
            # Convert votes map proxy to a standard dict for JSON serialization
            votes_dict = (
                dict(request.consensus_node.votes)
                if request.consensus_node.votes
                else {}
            )
            consensus_node_dict = {
                "proposal_id": request.consensus_node.id
                or request.consensus_node.proposal_id,  # Allow id or proposal_id
                "votes": votes_dict,
                "quorum": request.consensus_node.quorum,
            }
            if not consensus_node_dict["proposal_id"]:
                raise ValueError("Missing proposal_id in consensus node")
            message_for_auth = json.dumps(consensus_node_dict, sort_keys=True)
            proposal_id = consensus_node_dict["proposal_id"]
        except Exception as e:
            self.logger.error(f"Invalid vote data from {agent_id}: {e}")
            context.set_code(StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Invalid vote data: {e}")
            return RecordVoteResponse(status="error", message=f"Invalid vote data: {e}")

        if not self._authenticate_request(
            agent_id, message_for_auth, signature
        ) or not self._authorize_request(agent_id, ["voter", "governor"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Auth failed.")
            return RecordVoteResponse(status="error", message="Auth failed.")
        try:
            consensus_reached = self.lang_evolution_registry.record_vote(
                consensus_node_dict
            )
            self.security_audit_engine.log_audit(
                "vote_recorded",
                {
                    "proposal_id": proposal_id,
                    "voter": agent_id,
                    "vote": votes_dict.get(agent_id),
                    "consensus": consensus_reached,
                },
                proposal_id,
                "language_proposal",
            )
            return RecordVoteResponse(
                status="success", consensus_reached=consensus_reached
            )
        except Exception as e:
            self.logger.exception(
                f"Internal error in RecordVote for {agent_id} on {proposal_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return RecordVoteResponse(status="error", message=f"Internal error: {e}")

    def RecordValidation(self, request, context):
        agent_id = request.agent_id
        signature = request.signature
        self.logger.info(f"Received RecordValidation request from agent: {agent_id}")
        try:
            validation_node_dict = {
                "target": request.validation_node.target,
                "validation_type": request.validation_node.validation_type,
                "result": request.validation_node.result,
                "validator_id": agent_id,
            }
            if not validation_node_dict["target"]:
                raise ValueError("Missing target proposal_id")
            message_for_auth = json.dumps(validation_node_dict, sort_keys=True)
            proposal_id = validation_node_dict["target"]
        except Exception as e:
            self.logger.error(f"Invalid validation data from {agent_id}: {e}")
            context.set_code(StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Invalid validation data: {e}")
            return RecordValidationResponse(
                status="error", message=f"Invalid validation data: {e}"
            )

        if not self._authenticate_request(
            agent_id, message_for_auth, signature
        ) or not self._authorize_request(agent_id, ["validator", "governor"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Auth failed.")
            return RecordValidationResponse(status="error", message="Auth failed.")
        try:
            validation_passed = self.lang_evolution_registry.record_validation(
                validation_node_dict
            )
            self.security_audit_engine.log_audit(
                "validation_recorded",
                {
                    "proposal_id": proposal_id,
                    "validator": agent_id,
                    "passed": validation_passed,
                },
                proposal_id,
                "language_proposal",
            )
            return RecordValidationResponse(
                status="success", validation_passed=validation_passed
            )
        except Exception as e:
            self.logger.exception(
                f"Internal error in RecordValidation for {agent_id} on {proposal_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return RecordValidationResponse(
                status="error", message=f"Internal error: {e}"
            )

    def DeployGrammarVersion(self, request, context):
        agent_id = request.agent_id
        signature = request.signature
        proposal_id = request.proposal_id
        new_grammar_version = request.new_grammar_version
        self.logger.info(
            f"Received DeployGrammarVersion request: {agent_id} for {proposal_id} -> {new_grammar_version}"
        )
        try:
            request_dict = {
                "agent_id": agent_id,
                "proposal_id": proposal_id,
                "new_grammar_version": new_grammar_version,
            }
            message_for_auth = json.dumps(request_dict, sort_keys=True)
        except Exception as e:  # Should not happen with simple dict
            self.logger.error(f"Error creating message for auth: {e}")
            context.set_code(StatusCode.INTERNAL)
            context.set_details("Internal error.")
            return DeployGrammarVersionResponse(
                status="error", message="Internal error."
            )

        if not self._authenticate_request(
            agent_id, message_for_auth, signature
        ) or not self._authorize_request(agent_id, ["governor", "deployer"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Auth failed.")
            return DeployGrammarVersionResponse(status="error", message="Auth failed.")
        try:
            deployed = self.lang_evolution_registry.deploy_grammar_version(
                proposal_id, new_grammar_version
            )
            if deployed:
                self.security_audit_engine.log_audit(
                    "grammar_deployed",
                    {"proposal_id": proposal_id, "version": new_grammar_version},
                    agent_id,
                    "agent",
                )
                return DeployGrammarVersionResponse(status="success", deployed=True)
            else:
                context.set_code(StatusCode.FAILED_PRECONDITION)
                context.set_details("Deployment failed (proposal not validated?).")
                return DeployGrammarVersionResponse(
                    status="error",
                    message="Deployment failed (proposal not validated?).",
                    deployed=False,
                )
        except Exception as e:
            self.logger.exception(
                f"Internal error in DeployGrammarVersion for {agent_id} on {proposal_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return DeployGrammarVersionResponse(
                status="error", message=f"Internal error: {e}"
            )

    def QueryProposals(self, request, context):
        agent_id = request.agent_id
        agent_info = self.agent_registry.get_agent_info(agent_id)
        if not agent_info:
            self.logger.warning(f"Query attempt by unregistered agent: {agent_id}")
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Agent not registered.")
            return QueryProposalsResponse(
                status="error", message="Agent not registered."
            )

        self.logger.info(f"Received QueryProposals request from {agent_id}")
        filters = {
            "status": request.status if request.HasField("status") else None,
            "proposed_by": request.proposed_by
            if request.HasField("proposed_by")
            else None,
        }
        self.security_audit_engine.log_audit(
            "proposals_queried",
            {"querier": agent_id, "filters": filters},
            agent_id,
            "agent_query",
        )
        try:
            proposals_data = self.lang_evolution_registry.query_proposals(
                status=request.status if request.HasField("status") else None,
                proposed_by=request.proposed_by
                if request.HasField("proposed_by")
                else None,
                limit=request.limit if request.HasField("limit") else None,
                offset=request.offset if request.HasField("offset") else 0,
            )
            pb_proposals = []
            for prop_dict in proposals_data:
                content = prop_dict.get("proposal_content", {})
                content_bytes = json.dumps(
                    content if isinstance(content, dict) else {}
                ).encode("utf-8")
                pb_prop = Node(
                    id=prop_dict.get("id", ""),
                    type=prop_dict.get("type", "ProposalNode"),
                    metadata=prop_dict.get("metadata", {}),
                    proposed_by=prop_dict.get("proposed_by", ""),
                    rationale=prop_dict.get("rationale", ""),
                    proposal_content=content_bytes,
                )
                pb_proposals.append(pb_prop)
            return QueryProposalsResponse(status="success", proposals=pb_proposals)
        except Exception as e:
            self.logger.exception(
                f"Internal error during QueryProposals for {agent_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {e}")
            return QueryProposalsResponse(
                status="error", message=f"Internal server error: {e}"
            )

    def GetFullAuditLog(self, request, context):
        agent_id = request.agent_id
        signature = request.signature
        self.logger.info(f"Received GetFullAuditLog request from {agent_id}")
        request_dict = {"agent_id": agent_id, "action": "get_audit_log"}
        message_for_auth = json.dumps(request_dict, sort_keys=True)

        if not self._authenticate_request(
            agent_id, message_for_auth, signature
        ) or not self._authorize_request(agent_id, ["auditor", "governor"]):
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Auth failed.")
            return GetFullAuditLogResponse(status="error", message="Auth failed.")
        try:
            audit_log_data = self.security_audit_engine.get_full_audit_log()
            pb_audit_log = []
            for entry_dict in audit_log_data:
                details = entry_dict.get("details", {})
                details_bytes = json.dumps(
                    details if isinstance(details, dict) else {}
                ).encode("utf-8")
                pb_entry = AuditLogEntry(
                    action=entry_dict.get("action", ""),
                    details=details_bytes,
                    entity_id=entry_dict.get("entity_id", ""),
                    entity_type=entry_dict.get("entity_type", ""),
                )
                ts_str = entry_dict.get("timestamp")
                if ts_str:
                    try:
                        if HAS_GRPC:
                            ts = Timestamp()
                            ts.FromJsonString(ts_str)
                            pb_entry.timestamp = ts
                        else:
                            pb_entry.timestamp = (
                                ts_str  # Store string in mock attribute
                            )
                    except Exception as ts_err:
                        self.logger.warning(
                            f"Could not parse timestamp '{ts_str}' for audit log: {ts_err}"
                        )
                        # Keep default timestamp
                pb_audit_log.append(pb_entry)
            return GetFullAuditLogResponse(status="success", audit_log=pb_audit_log)
        except Exception as e:
            self.logger.exception(
                f"Internal error during GetFullAuditLog for {agent_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return GetFullAuditLogResponse(
                status="error", message=f"Internal error: {e}"
            )

    def VerifyAuditLogIntegrity(self, request, context):
        agent_id = request.agent_id
        agent_info = self.agent_registry.get_agent_info(agent_id)
        if not agent_info:
            self.logger.warning(f"Integrity check by unregistered agent: {agent_id}")
            context.set_code(StatusCode.PERMISSION_DENIED)
            context.set_details("Agent not registered.")
            return VerifyAuditLogIntegrityResponse(
                status="error", message="Agent not registered."
            )

        self.logger.info(f"Received VerifyAuditLogIntegrity request from {agent_id}")
        self.security_audit_engine.log_audit(
            "integrity_check_requested",
            {"requester": agent_id},
            agent_id,
            "agent_action",
        )
        try:
            is_valid = self.security_audit_engine.verify_audit_log_integrity()
            self.logger.info(f"Audit log integrity check result: {is_valid}")
            return VerifyAuditLogIntegrityResponse(
                status="success", integrity_valid=is_valid
            )
        except Exception as e:
            self.logger.exception(
                f"Internal error during VerifyAuditLogIntegrity for {agent_id}: {e}"
            )
            context.set_code(StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return VerifyAuditLogIntegrityResponse(
                status="error", message=f"Internal error: {e}"
            )


def serve(port: int = 50051, db_path: str = DB_PATH):
    """Start the gRPC server (or mock if gRPC not installed)."""
    if not HAS_GRPC:
        logging.error("gRPC packages not found. Cannot start full gRPC server.")
        logging.error("Install with: pip install grpcio protobuf")
        return

    # Initialize components with persistent storage
    db_manager = DatabaseManager(db_path=db_path)
    registry_api = RegistryAPI(db_manager)
    lang_evolution_registry = LanguageEvolutionRegistry(db_manager)
    agent_registry = AgentRegistry(db_manager)
    security_audit_engine = SecurityAuditEngine(db_manager)

    # Example: Register agents if needed (idempotent)
    example_agents = [
        (
            "agent-alice",
            ["proposer", "voter", "validator", "governor", "auditor", "deployer"],
            0.9,
        ),
        ("agent-bob", ["voter", "proposer"], 0.6),
        ("agent-charlie", ["observer"], 0.3),
    ]
    for agent_id, roles, trust_level in example_agents:
        if not agent_registry.get_agent_info(agent_id):  # Check before registering
            try:
                agent_registry.register_agent(
                    {"id": agent_id, "roles": roles, "trust_level": trust_level}
                )
            except Exception as e:
                logging.error(f"Failed to register example agent {agent_id}: {e}")

    # Create and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # --- This part requires the generated pb2_grpc file ---
    try:
        # Attempt to import the generated gRPC bindings
        # This assumes the test setup or project structure makes this importable
        # e.g., from src.governance import registry_pb2_grpc
        # Adjust relative import based on actual file structure
        # For this example, let's assume it's in the same package 'src.governance'
        from . import \
            registry_pb2_grpc  # Requires __init__.py and compiled protos

        registry_pb2_grpc.add_RegistryServiceServicer_to_server(
            RegistryServicer(
                registry_api=registry_api,
                lang_evolution_registry=lang_evolution_registry,
                agent_registry=agent_registry,
                security_audit_engine=security_audit_engine,
            ),
            server,
        )
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        logging.info(
            f"Registry API Server started on port {port} using database {db_path}"
        )
        server.wait_for_termination()  # Keep server running until interrupted

    except ImportError:
        logging.error(
            "Could not import 'registry_pb2_grpc'. Ensure protobuf files are compiled."
        )
        logging.error("Example compile command (run from root):")
        logging.error(
            "python -m grpc_tools.protoc -I./src/governance/protos --python_out=./src/governance --grpc_python_out=./src/governance ./src/governance/protos/registry.proto"
        )
        # Server cannot start without the generated code
        server.stop(0)  # Stop the server if it was started but servicer failed
    except Exception as e:
        logging.exception(f"Failed to start gRPC server: {e}")
    finally:
        # Ensure server stops gracefully
        if "server" in locals():
            server.stop(0)
            logging.info("Server stopped.")


if __name__ == "__main__":
    db_file = os.environ.get("REGISTRY_DB_PATH", DB_PATH)
    server_port = int(os.environ.get("REGISTRY_PORT", 50051))
    serve(port=server_port, db_path=db_file)
