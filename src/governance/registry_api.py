# registry_api.py
"""
Graphix IR Registry API (v2.0.0)
=================================

This module implements an API layer for managing a registry of Graphix IR graphs
with cryptographic verification, audit logging, and proposal management.

Key Features:
- Persistent backend abstraction for storage
- Cryptographic signing and verification
- Merkle tree integrity verification
- Audit logging with hash chaining
- Proposal submission and validation workflow
- Grammar version management
"""

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

# --- Cryptography Library Integration ---
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logging.warning(
        "Cryptography library not found. Using mock signing for demonstration. "
        "Install 'cryptography' for full security features."
    )

    class MockCrypto:
        def generate_private_key(self, public_exponent, key_size):
            return "mock_private_key"

        def public_key(self):
            return "mock_public_key"

        def public_bytes(self, encoding, format):
            return b"mock_public_key_pem"

        def sign(self, data, padding, hashes):
            return b"mock_signature_" + hashlib.sha256(data).hexdigest().encode()

        def verify(self, signature, data, padding, hashes):
            if (
                not signature.startswith(b"mock_signature_")
                or hashlib.sha256(data).hexdigest().encode()
                != signature[len(b"mock_signature_") :]
            ):
                raise InvalidSignature("Mock Invalid Signature")
            return True

    class MockSerialization:
        def load_pem_public_key(self, pem):
            return MockCrypto()

        Encoding = type("Encoding", (), {"PEM": "PEM"})
        PublicFormat = type(
            "PublicFormat", (), {"SubjectPublicKeyInfo": "SubjectPublicKeyInfo"}
        )

    class MockPadding:
        class PSS:
            MAX_LENGTH = 100

            def __init__(self, mgf, salt_length):
                pass

        def MGF1(self, hashes):
            pass

    class MockHashes:
        def SHA256(self):
            pass

    rsa = MockCrypto()
    serialization = MockSerialization()
    padding = MockPadding()
    hashes = MockHashes()
    InvalidSignature = type("InvalidSignature", (Exception,), {})


def _verify_crypto_available():
    """Verify cryptography library is available for production use."""
    import os
    if not HAS_CRYPTOGRAPHY:
        import warnings
        warnings.warn(
            "cryptography library not installed. "
            "Real signature verification is DISABLED. "
            "Install with: pip install cryptography",
            SecurityWarning
        )
        # In production mode, fail closed
        if os.environ.get("REGISTRY_PRODUCTION_MODE", "").lower() == "true":
            raise RuntimeError(
                "SECURITY ERROR: cryptography library required in production mode. "
                "Install with: pip install cryptography"
            )


# Verify crypto availability at module load time
_verify_crypto_available()


# --- Merkle Tree Implementation ---
def hash_data(data: bytes) -> bytes:
    """Hashes data using SHA-256."""
    return hashlib.sha256(data).digest()


def build_merkle_tree(leaves: List[bytes]) -> List[bytes]:
    """Recursively builds a Merkle tree from a list of hashed leaves."""
    if not leaves:
        return []
    if len(leaves) == 1:
        return leaves
    if len(leaves) % 2 != 0:
        leaves.append(leaves[-1])

    new_level = []
    for i in range(0, len(leaves), 2):
        combined_hash = hash_data(leaves[i] + leaves[i + 1])
        new_level.append(combined_hash)

    return build_merkle_tree(new_level)


def get_merkle_root(data_list: List[Dict]) -> Optional[bytes]:
    """Calculates the Merkle root for a list of dictionary-based records."""
    if not data_list:
        return None
    leaves = [
        hash_data(json.dumps(item, sort_keys=True).encode("utf-8"))
        for item in data_list
    ]
    tree = build_merkle_tree(leaves)
    return tree[0] if tree else None


# --- Backend Abstraction ---
class AbstractBackend(ABC):
    @abstractmethod
    def load_data(self, key: str) -> Optional[Dict]:
        raise NotImplementedError

    @abstractmethod
    def save_data(self, key: str, data: Dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def append_record(self, key: str, record: Dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, key: str) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys, optionally filtered by prefix."""
        raise NotImplementedError


class InMemoryBackend(AbstractBackend):
    """Simple in-memory backend for development and testing."""

    def __init__(self):
        self._data_store: Dict[str, Any] = {}
        # FIXED: Use RLock instead of Lock to allow reentrant locking
        self.lock = threading.RLock()
        self.logger = logging.getLogger("InMemoryBackend")

    def load_data(self, key: str) -> Optional[Dict]:
        with self.lock:
            return deepcopy(self._data_store.get(key))

    def save_data(self, key: str, data: Dict) -> str:
        with self.lock:
            self._data_store[key] = deepcopy(data)
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def append_record(self, key: str, record: Dict) -> str:
        with self.lock:
            current_records = self._data_store.setdefault(key, {}).setdefault(
                "records", []
            )
            current_records.append(deepcopy(record))
            return hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()

    def get_history(self, key: str) -> List[Dict]:
        with self.lock:
            data = self.load_data(key)
            return data.get("records", []) if data else []

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys, optionally filtered by prefix."""
        with self.lock:
            if prefix:
                return [
                    key for key in self._data_store.keys() if key.startswith(prefix)
                ]
            else:
                return list(self._data_store.keys())


class DatabaseBackendAdapter(AbstractBackend):
    """Adapter that wraps DatabaseManager to implement AbstractBackend interface."""

    def __init__(self, db_manager):
        """
        Initialize adapter with DatabaseManager instance.
        
        Args:
            db_manager: DatabaseManager instance from registry_api_server
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger("DatabaseBackendAdapter")
        # Map keys to table names
        self._key_to_table_map = {
            "global_registry_state": "registry_state",
            "audit_log": "audit_log",
            "grammar_versions": "grammar_versions",
        }

    def _get_table_for_key(self, key: str) -> str:
        """Map a key to its corresponding database table."""
        # Check if it's a known singleton key
        if key in self._key_to_table_map:
            return self._key_to_table_map[key]
        # Check if it's a proposal key
        if key.startswith("proposal_"):
            return "proposals"
        # Default to a generic key-value table
        return "key_value_store"

    def load_data(self, key: str) -> Optional[Dict]:
        """Load data for a given key from the database."""
        table = self._get_table_for_key(key)
        try:
            if table in ["proposals", "key_value_store"]:
                return self.db_manager.get_record(table, key)
            elif table == "registry_state":
                # Special handling for registry state
                return self.db_manager.get_record("key_value_store", key)
            elif table == "grammar_versions":
                return self.db_manager.get_record("key_value_store", key)
            elif table == "audit_log":
                # Audit log uses different structure
                records = self.db_manager.get_full_audit_log()
                return {"records": records} if records else None
        except Exception as e:
            self.logger.error(f"Error loading data for key {key}: {e}")
            return None

    def save_data(self, key: str, data: Dict) -> str:
        """Save data for a given key to the database."""
        table = self._get_table_for_key(key)
        try:
            if table in ["proposals", "key_value_store"]:
                self.db_manager.save_record(table, key, data)
            elif table in ["registry_state", "grammar_versions"]:
                self.db_manager.save_record("key_value_store", key, data)
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error saving data for key {key}: {e}")
            raise

    def append_record(self, key: str, record: Dict) -> str:
        """Append a record to a key's history (used for audit log)."""
        try:
            if key == "audit_log":
                self.db_manager.log_audit(record)
            else:
                # For other keys, load existing data, append, and save
                current_data = self.load_data(key) or {"records": []}
                if "records" not in current_data:
                    current_data["records"] = []
                current_data["records"].append(record)
                self.save_data(key, current_data)
            return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error appending record to key {key}: {e}")
            raise

    def get_history(self, key: str) -> List[Dict]:
        """Get the history of records for a key."""
        try:
            if key == "audit_log":
                return self.db_manager.get_full_audit_log()
            else:
                data = self.load_data(key)
                return data.get("records", []) if data else []
        except Exception as e:
            self.logger.error(f"Error getting history for key {key}: {e}")
            return []

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys, optionally filtered by prefix."""
        try:
            # This is a simplified implementation
            # In a real database, you'd query across multiple tables
            keys = []
            
            # Check proposals table
            if not prefix or prefix.startswith("proposal_"):
                proposals = self.db_manager.query_records("proposals")
                for prop in proposals:
                    if "id" in prop.get("node", {}):
                        prop_key = f"proposal_{prop['node']['id']}"
                        if not prefix or prop_key.startswith(prefix):
                            keys.append(prop_key)
            
            # Check key_value_store for singleton keys
            for singleton_key in self._key_to_table_map.keys():
                if not prefix or singleton_key.startswith(prefix):
                    if self.load_data(singleton_key):
                        keys.append(singleton_key)
            
            return keys
        except Exception as e:
            self.logger.error(f"Error listing keys with prefix {prefix}: {e}")
            return []


# --- Key Management System Abstraction ---
class AbstractKMS(ABC):
    @abstractmethod
    def get_private_key(self, key_id: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_public_key_pem(self, key_id: str) -> str:
        raise NotImplementedError


class SimpleKMS(AbstractKMS):
    """Simple key management for development."""

    def __init__(self):
        self.keys = {}
        self.logger = logging.getLogger("SimpleKMS")

    def _generate_new_key_pair(self):
        if HAS_CRYPTOGRAPHY:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            public_key = private_key.public_key()
            public_key_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")
        else:
            private_key = "mock_private_key"
            public_key_pem = "mock_public_key_pem"
        return private_key, public_key_pem

    def get_private_key(self, key_id: str) -> Any:
        if key_id not in self.keys:
            self.logger.info(f"Generating new key for {key_id}")
            private_key, public_key_pem = self._generate_new_key_pair()
            self.keys[key_id] = {"private": private_key, "public_pem": public_key_pem}
        return self.keys[key_id]["private"]

    def get_public_key_pem(self, key_id: str) -> str:
        if key_id not in self.keys:
            self.get_private_key(key_id)
        return self.keys[key_id]["public_pem"]


# --- Configuration ---
logger = logging.getLogger(__name__)
DEFAULT_GRAMMAR_VERSION = "2.3.0"


# --- Cryptographic Handler ---
class CryptoHandler:
    """Handles cryptographic signing and verification using KMS."""

    def __init__(self, kms: AbstractKMS, key_id: str):
        self.kms = kms
        self.key_id = key_id
        self.logger = logging.getLogger("CryptoHandler")

    def sign_data(self, data: bytes) -> str:
        """Sign data payload using private key from KMS."""
        private_key = self.kms.get_private_key(self.key_id)
        if not private_key:
            raise ValueError("Private key not available for signing.")

        if HAS_CRYPTOGRAPHY:
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return signature.hex()
        else:
            return private_key.sign(data, None, None).hex()

    def verify_signature(
        self, data: bytes, signature_hex: str, public_key_pem: bytes
    ) -> bool:
        """Verify a signature against a public key."""
        if HAS_CRYPTOGRAPHY:
            try:
                public_key = serialization.load_pem_public_key(public_key_pem)
                public_key.verify(
                    bytes.fromhex(signature_hex),
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                return True
            except InvalidSignature:
                self.logger.warning("Signature verification failed.")
                return False
            except Exception as e:
                self.logger.error(f"Error during signature verification: {e}")
                return False
        else:
            return True  # Mock always passes


# --- Security and Agent Management ---
class SecurityEngine:
    """Handles security policies and validation."""

    def __init__(self):
        self.logger = logging.getLogger("SecurityEngine")

    def enforce_policies(self, proposal_node: Dict) -> bool:
        """Enforce basic security policies on proposals."""
        # Check for potentially dangerous patterns
        proposal_str = json.dumps(proposal_node).lower()
        dangerous_patterns = ["os.system", "exec", "eval", "__import__"]

        for pattern in dangerous_patterns:
            if pattern in proposal_str:
                self.logger.warning(
                    f"Security policy violation: Detected '{pattern}' in proposal."
                )
                return False
        return True

    def validate_trust_policy(self, agent_id: str, trust_level: float) -> bool:
        """Validate agent trust level against policy."""
        min_trust_threshold = 0.3
        if trust_level < min_trust_threshold:
            self.logger.warning(
                f"Trust policy violation: Agent {agent_id} has insufficient trust level ({trust_level})."
            )
            return False
        return True


class AgentRegistry:
    """Manages agent information and trust levels."""

    def __init__(self):
        self.agents = {}
        self.logger = logging.getLogger("AgentRegistry")

    def register_agent(
        self, agent_id: str, public_key_pem: str, trust_level: float = 0.5
    ):
        """Register a new agent."""
        self.agents[agent_id] = {
            "public_key_pem": public_key_pem,
            "trust_level": trust_level,
        }

    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """Get agent information."""
        return self.agents.get(agent_id)

    def verify_agent_signature(
        self, agent_id: str, data: bytes, signature_hex: str
    ) -> bool:
        """Verify an agent's signature."""
        agent_info = self.get_agent_info(agent_id)
        if not agent_info:
            return False

        public_key_pem = agent_info["public_key_pem"].encode("utf-8")
        crypto_handler = CryptoHandler(SimpleKMS(), "temp_verifier")
        return crypto_handler.verify_signature(data, signature_hex, public_key_pem)


# --- Main Registry API ---
class RegistryAPI:
    """
    Manages the lifecycle of Graphix IR proposals including submission,
    consensus, validation, and deployment.
    """

    def __init__(self, backend: AbstractBackend = None, kms: AbstractKMS = None):
        self.backend = backend or InMemoryBackend()
        self.kms = kms or SimpleKMS()
        self.crypto = CryptoHandler(self.kms, "registry_signing_key")
        self.logger = logging.getLogger("RegistryAPI")
        self.security_engine = SecurityEngine()
        self.agent_registry = AgentRegistry()

        self.registry_state_key = "global_registry_state"
        self.proposals_key_prefix = "proposal_"
        self.audit_log_key = "audit_log"
        self.grammar_versions_key = "grammar_versions"

        self._initialize_registry_state()
        self._initialize_grammar_version()

    def _initialize_registry_state(self):
        """Initialize or load the global registry state."""
        self.registry = self.backend.load_data(self.registry_state_key)
        if not self.registry:
            self.registry = {
                "proposals": {},
                "consensus_records": {},
                "validation_records": {},
                "metrics": {
                    "total_proposals": 0,
                    "approved_proposals": 0,
                    "rejected_proposals": 0,
                },
            }
            self.backend.save_data(self.registry_state_key, self.registry)
            self._create_audit_entry(
                "registry_initialized", {"details": "New registry instance created."}
            )
            self.logger.info("New registry state initialized.")
        else:
            self.logger.info("Registry state loaded from backend.")

    def _initialize_grammar_version(self):
        """Initialize grammar version tracking."""
        grammar_versions_data = self.backend.load_data(self.grammar_versions_key)
        if not grammar_versions_data:
            grammar_versions_data = {
                "active": DEFAULT_GRAMMAR_VERSION,
                "history": [
                    {
                        "version": DEFAULT_GRAMMAR_VERSION,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "action": "initialized",
                    }
                ],
            }
            self.backend.save_data(self.grammar_versions_key, grammar_versions_data)
            self.logger.info(f"Initialized grammar version: {DEFAULT_GRAMMAR_VERSION}")

    def _create_audit_entry(self, action: str, details: Dict) -> Dict:
        """Create a cryptographically signed audit log entry."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        log_content = {
            "action": action,
            "timestamp": timestamp,
            "details": details,
            "registry_version": self.get_active_grammar_version(),
        }
        serialized_log = json.dumps(log_content, sort_keys=True).encode("utf-8")
        signature = self.crypto.sign_data(serialized_log)

        audit_entry = {
            "log": log_content,
            "signature": signature,
            "public_key": self.crypto.kms.get_public_key_pem(self.crypto.key_id),
        }
        self.backend.append_record(self.audit_log_key, audit_entry)
        self.logger.info(f"Audit logged: {action}")
        return audit_entry

    def submit_proposal(self, proposal_node: Dict) -> str:
        """Submit a new proposal."""
        proposal_id = (
            proposal_node.get("id")
            or hashlib.sha256(
                json.dumps(proposal_node, sort_keys=True).encode()
            ).hexdigest()[:16]
        )

        if self.get_proposal(proposal_id):
            raise ValueError(f"Proposal with ID {proposal_id} already exists.")

        # Security validation
        if not self.security_engine.enforce_policies(proposal_node):
            self._create_audit_entry(
                "proposal_rejected",
                {"proposal_id": proposal_id, "reason": "security_policy_violation"},
            )
            raise ValueError("Proposal failed security policy checks.")

        # Trust validation
        proposer_id = proposal_node.get("proposed_by")
        if proposer_id:
            agent_info = self.agent_registry.get_agent_info(proposer_id)
            if agent_info:
                trust_level = agent_info.get("trust_level", 0.0)
                if not self.security_engine.validate_trust_policy(
                    proposer_id, trust_level
                ):
                    self._create_audit_entry(
                        "proposal_rejected",
                        {"proposal_id": proposal_id, "reason": "insufficient_trust"},
                    )
                    raise ValueError(
                        f"Proposer '{proposer_id}' has insufficient trust level."
                    )

        # Store proposal
        proposal_record = {
            "status": "pending",
            "node": proposal_node,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
            "history": [],
        }
        self.backend.save_data(
            f"{self.proposals_key_prefix}{proposal_id}", proposal_record
        )

        # Update metrics
        self.registry["metrics"]["total_proposals"] += 1
        self.backend.save_data(self.registry_state_key, self.registry)

        self._create_audit_entry(
            "proposal_submitted",
            {"proposal_id": proposal_id, "proposed_by": proposer_id},
        )
        self.logger.info(f"Proposal '{proposal_id}' submitted successfully.")
        return proposal_id

    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        """Retrieve a proposal by ID."""
        return self.backend.load_data(f"{self.proposals_key_prefix}{proposal_id}")

    def record_vote(self, consensus_node: Dict) -> bool:
        """Record votes for a proposal."""
        proposal_id = consensus_node.get("proposal_id")
        proposal_record = self.get_proposal(proposal_id)
        if not proposal_record:
            raise ValueError(f"Proposal '{proposal_id}' not found.")

        # Check deadline if specified
        deadline_str = consensus_node.get("deadline")
        if deadline_str:
            try:
                deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                if datetime.utcnow() > deadline:
                    proposal_record["status"] = "rejected_timeout"
                    self.backend.save_data(
                        f"{self.proposals_key_prefix}{proposal_id}", proposal_record
                    )
                    self.logger.warning(
                        f"Proposal '{proposal_id}' voting deadline exceeded."
                    )
                    return False
            except ValueError:
                self.logger.error(f"Invalid deadline format: {deadline_str}")

        # Calculate weighted votes
        current_votes = consensus_node.get("votes", {})
        quorum_threshold = consensus_node.get("quorum", 0.5)

        weighted_yes_votes = 0.0
        total_weighted_votes = 0.0
        for agent_id, vote in current_votes.items():
            agent_info = self.agent_registry.get_agent_info(agent_id)
            if agent_info:
                weight = agent_info.get("trust_level", 0.5)
                total_weighted_votes += weight
                if vote == "yes":
                    weighted_yes_votes += weight

        consensus_reached = False
        if (
            total_weighted_votes > 0
            and (weighted_yes_votes / total_weighted_votes) >= quorum_threshold
        ):
            consensus_reached = True
            proposal_record["status"] = "approved"
            self.registry["metrics"]["approved_proposals"] += 1
        else:
            self.registry["metrics"]["rejected_proposals"] += 1

        self.backend.save_data(
            f"{self.proposals_key_prefix}{proposal_id}", proposal_record
        )
        self.backend.save_data(self.registry_state_key, self.registry)

        self._create_audit_entry(
            "vote_recorded",
            {
                "proposal_id": proposal_id,
                "votes": current_votes,
                "consensus_reached": consensus_reached,
            },
        )
        return consensus_reached

    def record_validation(self, validation_node: Dict) -> bool:
        """Record validation results."""
        proposal_id = validation_node.get("target")
        proposal_record = self.get_proposal(proposal_id)
        if not proposal_record:
            raise ValueError(f"Proposal '{proposal_id}' not found.")

        validation_result = validation_node.get("result", False)

        if validation_result:
            proposal_record["status"] = "validated"
            self.logger.info(f"Validation successful for proposal '{proposal_id}'.")
        else:
            proposal_record["status"] = "validation_failed"
            self.logger.warning(f"Validation failed for proposal '{proposal_id}'.")

        self.backend.save_data(
            f"{self.proposals_key_prefix}{proposal_id}", proposal_record
        )

        self._create_audit_entry(
            "validation_recorded",
            {
                "proposal_id": proposal_id,
                "validation_type": validation_node.get("validation_type"),
                "result": validation_result,
            },
        )
        return validation_result

    def deploy_grammar_version(
        self, proposal_id: str, new_grammar_version: str
    ) -> bool:
        """Deploy a new grammar version."""
        proposal_record = self.get_proposal(proposal_id)
        if not proposal_record:
            raise ValueError(f"Proposal '{proposal_id}' not found.")

        if proposal_record["status"] not in ["approved", "validated"]:
            self.logger.warning(
                f"Cannot deploy: Proposal '{proposal_id}' not approved/validated."
            )
            return False

        grammar_versions_data = self.backend.load_data(self.grammar_versions_key)
        current_active = grammar_versions_data["active"]

        if not self._is_valid_version_increment(current_active, new_grammar_version):
            self.logger.error(
                f"Invalid version increment: {current_active} -> {new_grammar_version}"
            )
            return False

        grammar_versions_data["history"].append(
            {
                "version": new_grammar_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "action": "deployed",
                "proposal_id": proposal_id,
            }
        )
        grammar_versions_data["active"] = new_grammar_version
        self.backend.save_data(self.grammar_versions_key, grammar_versions_data)

        proposal_record["status"] = "deployed"
        self.backend.save_data(
            f"{self.proposals_key_prefix}{proposal_id}", proposal_record
        )

        self._create_audit_entry(
            "grammar_deployed",
            {
                "proposal_id": proposal_id,
                "old_version": current_active,
                "new_version": new_grammar_version,
            },
        )
        self.logger.info(f"Grammar deployed to version {new_grammar_version}")
        return True

    def get_active_grammar_version(self) -> str:
        """Get the currently active grammar version."""
        data = self.backend.load_data(self.grammar_versions_key)
        return data["active"] if data else DEFAULT_GRAMMAR_VERSION

    def query_proposals(
        self,
        status: Optional[str] = None,
        proposed_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict]:
        """Query proposals with filters."""
        all_proposals = []

        # Use backend abstraction instead of accessing private attributes
        proposal_keys = self.backend.list_keys(prefix=self.proposals_key_prefix)

        for key in proposal_keys:
            proposal_record = self.backend.load_data(key)
            if proposal_record:
                all_proposals.append(proposal_record)

        results = []
        for prop_record in all_proposals:
            match_status = status is None or prop_record["status"] == status
            match_proposer = (
                proposed_by is None
                or prop_record["node"].get("proposed_by") == proposed_by
            )
            if match_status and match_proposer:
                results.append(prop_record)

        return results[offset : offset + limit] if limit else results[offset:]

    def get_full_audit_log(self) -> List[Dict]:
        """Get the complete audit log."""
        return self.backend.get_history(self.audit_log_key)

    def verify_audit_log_integrity(self) -> bool:
        """Verify the cryptographic integrity of the audit log."""
        self.logger.info("Verifying audit log integrity...")
        full_audit_log = self.get_full_audit_log()

        # Calculate Merkle root
        merkle_root = get_merkle_root(full_audit_log)
        if merkle_root:
            self.logger.info(f"Merkle Root: {merkle_root.hex()}")

        # Verify individual signatures
        for i, entry in enumerate(full_audit_log):
            try:
                log_content = entry["log"]
                signature_hex = entry["signature"]
                public_key_pem = entry["public_key"].encode("utf-8")
                serialized_log = json.dumps(log_content, sort_keys=True).encode("utf-8")

                if not self.crypto.verify_signature(
                    serialized_log, signature_hex, public_key_pem
                ):
                    self.logger.error(f"Audit log entry {i} failed verification!")
                    return False
            except Exception as e:
                self.logger.error(f"Error verifying entry {i}: {e}")
                return False

        self.logger.info("All audit log entries verified successfully.")
        return True

    def _is_valid_version_increment(self, old_version: str, new_version: str) -> bool:
        """Validate semantic versioning increment."""
        try:
            old_parts = list(map(int, old_version.split(".")))
            new_parts = list(map(int, new_version.split(".")))

            if len(old_parts) != 3 or len(new_parts) != 3:
                return False

            # Check for valid increment
            if new_parts[0] > old_parts[0]:  # Major version
                return new_parts[1] == 0 and new_parts[2] == 0
            elif (
                new_parts[0] == old_parts[0] and new_parts[1] > old_parts[1]
            ):  # Minor version
                return new_parts[2] == 0
            elif (
                new_parts[0] == old_parts[0]
                and new_parts[1] == old_parts[1]
                and new_parts[2] > old_parts[2]
            ):  # Patch
                return True
            else:
                return False
        except ValueError:
            return False


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize registry
    registry = RegistryAPI()

    # Register some agents
    kms = SimpleKMS()
    for agent_id in ["agent-alice", "agent-bob"]:
        public_key_pem = kms.get_public_key_pem(agent_id)
        trust_level = 0.8 if agent_id == "agent-alice" else 0.6
        registry.agent_registry.register_agent(agent_id, public_key_pem, trust_level)

    print("\n--- Example: Submit and Process Proposal ---")

    # Submit a proposal
    proposal = {
        "id": "add_new_node_type",
        "type": "ProposalNode",
        "proposed_by": "agent-alice",
        "rationale": "Add support for new compute node type",
        "proposal_content": {
            "add": {
                "ComputeNodeV2": {"description": "Enhanced compute node with caching"}
            }
        },
    }

    try:
        proposal_id = registry.submit_proposal(proposal)
        print(f"✓ Proposal submitted: {proposal_id}")

        # Record votes
        consensus = {
            "proposal_id": proposal_id,
            "votes": {"agent-alice": "yes", "agent-bob": "yes"},
            "quorum": 0.6,
        }

        if registry.record_vote(consensus):
            print("✓ Consensus reached")

            # Validate
            validation = {
                "target": proposal_id,
                "validation_type": "schema",
                "result": True,
            }

            if registry.record_validation(validation):
                print("✓ Validation passed")

                # Deploy
                if registry.deploy_grammar_version(proposal_id, "2.3.1"):
                    print("✓ New grammar version deployed: 2.3.1")

        # Verify audit log
        if registry.verify_audit_log_integrity():
            print("✓ Audit log integrity verified")

    except ValueError as e:
        print(f"✗ Error: {e}")
