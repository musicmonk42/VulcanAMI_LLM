"""
Comprehensive test suite for registry_api_server.py
"""

import hashlib
import json
import sqlite3
import tempfile
import threading
import time
import uuid  # Added for generating unique IDs in tests
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Assuming registry_api_server.py is in src/governance/
# Adjust the import path if your structure is different
from src.governance.registry_api_server import \
    DB_POOL_SIZE  # <--- FIXED: Added missing import
from src.governance.registry_api_server import HAS_GRPC  # FIX: Import HAS_GRPC
from src.governance.registry_api_server import \
    StatusCode  # Import mocked StatusCode
from src.governance.registry_api_server import \
    Timestamp  # Import mocked Timestamp
from src.governance.registry_api_server import (
    AgentRegistry, AuditLogEntry, DatabaseConnectionPool, DatabaseManager,
    DeployGrammarVersionRequest, DeployGrammarVersionResponse,
    GetFullAuditLogRequest, GetFullAuditLogResponse, LanguageEvolutionRegistry,
    Node, QueryProposalsRequest, QueryProposalsResponse,
    RecordValidationRequest, RecordValidationResponse, RecordVoteRequest,
    RecordVoteResponse, RegisterGraphProposalRequest,
    RegisterGraphProposalResponse, RegistryAPI, RegistryServicer,
    SecurityAuditEngine, SubmitLanguageEvolutionProposalRequest,
    SubmitLanguageEvolutionProposalResponse, VerifyAuditLogIntegrityRequest,
    VerifyAuditLogIntegrityResponse)


@pytest.fixture
def temp_db():
    """Create temporary database file within a directory."""
    # Use TemporaryDirectory for better cleanup context management
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_registry.db"
        yield str(db_path)
        # Directory and file are automatically cleaned up
        # Add small delay *before* implicit cleanup for Windows file release
        time.sleep(0.05)


@pytest.fixture
def db_manager(temp_db):
    """Create DatabaseManager instance."""
    # Ensure the manager uses the specific temp_db path for this test run
    manager = DatabaseManager(temp_db)
    yield manager
    # FIX: Explicitly close all connections in the pool during teardown
    if hasattr(manager, 'pool') and hasattr(manager.pool, '_connections'):
        with manager.pool._condition:
            while manager.pool._connections:
                conn = manager.pool._connections.pop(0) # Use pop(0) if LIFO matters
                try:
                    # Check if connection is still valid before closing
                    conn.execute("SELECT 1")
                    conn.close()
                except Exception:
                    pass # Ignore errors on already closed/invalid connections
    del manager


@pytest.fixture
def registry_api(db_manager):
    """Create RegistryAPI instance."""
    return RegistryAPI(db_manager)


@pytest.fixture
def lang_evolution_registry(db_manager):
    """Create LanguageEvolutionRegistry instance."""
    return LanguageEvolutionRegistry(db_manager)


@pytest.fixture
def agent_registry(db_manager):
    """Create AgentRegistry instance."""
    return AgentRegistry(db_manager)


@pytest.fixture
def security_audit_engine(db_manager):
    """Create SecurityAuditEngine instance."""
    return SecurityAuditEngine(db_manager)


@pytest.fixture
def registry_servicer(registry_api, lang_evolution_registry, agent_registry, security_audit_engine):
    """Create RegistryServicer instance."""
    return RegistryServicer(
        registry_api=registry_api,
        lang_evolution_registry=lang_evolution_registry,
        agent_registry=agent_registry,
        security_audit_engine=security_audit_engine
    )


class TestProtobufMessages:
    """Test protobuf message classes."""

    def test_node_creation(self):
        """Test creating Node."""
        node = Node(id="test_id", type="ProposalNode", metadata={"key": "value"})
        assert node.id == "test_id"
        assert node.type == "ProposalNode"
        assert node.metadata == {"key": "value"}

    def test_node_default_values(self):
        """Test Node default values."""
        node = Node()
        assert node.id == ''
        assert node.type == ''
        assert node.metadata == {}

    def test_audit_log_entry_creation(self):
        """Test creating AuditLogEntry."""
        # FIX: Use HAS_GRPC correctly
        ts = Timestamp() if HAS_GRPC else datetime.utcnow().isoformat()+'Z'
        entry = AuditLogEntry(action="test_action", entity_id="entity_1", timestamp=ts)
        assert entry.action == "test_action"
        assert entry.entity_id == "entity_1"
        assert entry.timestamp is not None


class TestRequestResponseMessages:
    """Test request/response message classes."""

    def test_register_graph_proposal_request(self):
        node = Node(id="test")
        request = RegisterGraphProposalRequest(agent_id="agent1", signature="sig", proposal_node=node)
        assert request.agent_id == "agent1"
        assert request.signature == "sig"
        assert request.proposal_node == node

    def test_register_graph_proposal_response(self):
        response = RegisterGraphProposalResponse(status="success", message="Created", proposal_id="prop123")
        assert response.status == "success"
        assert response.proposal_id == "prop123"

    def test_query_proposals_request_has_field(self):
        request = QueryProposalsRequest(agent_id="agent1", status="pending")
        assert request.HasField("status") is True
        assert request.HasField("proposed_by") is False
        assert request.HasField("limit") is False


class TestDatabaseConnectionPool:
    """Test DatabaseConnectionPool."""

    def test_initialization(self, temp_db):
        pool = DatabaseConnectionPool(temp_db, pool_size=3)
        assert pool._pool_size == 3
        # Check initial connections were attempted
        assert len(pool._connections) <= 3
        # Cleanup
        with pool._condition:
            while pool._connections:
                conn = pool._connections.pop(); conn.close()

    def test_get_connection(self, temp_db):
        pool = DatabaseConnectionPool(temp_db, pool_size=2)
        conn_list = [] # Keep track of connections to close
        try:
            with pool.get_connection() as conn:
                conn_list.append(conn)
                assert conn is not None
                assert isinstance(conn, sqlite3.Connection)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1
        finally:
            # Ensure connections are closed
            with pool._condition:
                 while pool._connections:
                     c = pool._connections.pop(); c.close()
            for c in conn_list:
                try: c.close()
                except: pass


    def test_connection_returned_to_pool(self, temp_db):
        pool = DatabaseConnectionPool(temp_db, pool_size=2)
        initial_count = len(pool._connections)
        try:
            with pool.get_connection() as conn:
                during_count = len(pool._connections)
            after_count = len(pool._connections)
            assert during_count == initial_count - 1
            assert after_count == initial_count
        finally:
             with pool._condition:
                 while pool._connections:
                     c = pool._connections.pop(); c.close()

    def test_connection_pool_exhaustion_timeout(self, temp_db):
        # Use shorter timeout in pool for faster test
        pool = DatabaseConnectionPool(temp_db, pool_size=1, timeout=0.2)
        timed_out_event = threading.Event()
        error_occurred = threading.Event()
        connection_acquired_unexpectedly = threading.Event()
        ctx_manager = None # Define outside try block
        conn1 = None

        try:
            # Acquire the only connection using the context manager
            ctx_manager = pool.get_connection()
            conn1 = ctx_manager.__enter__() # Manually enter context

            def try_get_connection():
                try:
                    # Attempt to get connection, should time out
                    with pool.get_connection():
                        connection_acquired_unexpectedly.set()
                except RuntimeError as e:
                    if "Connection pool exhausted" in str(e):
                        timed_out_event.set()
                except Exception as e:
                    logging.exception("Unexpected error in thread") # Log full traceback
                    error_occurred.set()

            thread = threading.Thread(target=try_get_connection)
            thread.start()
            thread.join(timeout=1.0) # Wait longer than pool timeout

            # Explicitly release the first connection by exiting context manager
            if ctx_manager:
                ctx_manager.__exit__(None, None, None)
                conn1 = None # Mark as released

            assert not connection_acquired_unexpectedly.is_set(), "Connection acquired unexpectedly"
            assert not error_occurred.is_set(), "An unexpected error occurred in thread"
            assert timed_out_event.is_set(), "Expected RuntimeError (pool exhausted) was not raised"

        finally:
             # Ensure conn1 is attempted to be released if context manager exit failed
             if ctx_manager and conn1:
                 try: ctx_manager.__exit__(None, None, None)
                 except: pass # Ignore errors during cleanup __exit__
             # Clean up pool connections
             with pool._condition:
                 while pool._connections:
                     c = pool._connections.pop()
                     try: c.close()
                     except: pass


class TestDatabaseManager:
    """Test DatabaseManager."""

    def test_initialization(self, db_manager): # Use fixture
        """Test database manager initialization."""
        assert db_manager.db_path is not None # Check path exists
        assert db_manager.pool is not None
        # Check if pool was initialized
        assert db_manager.pool._pool_size == DB_POOL_SIZE
        # Check if connections were created (or at least attempted)
        assert len(db_manager.pool._connections) <= DB_POOL_SIZE


    def test_tables_created(self, db_manager): # Use fixture
        """Test that all tables are created."""
        with db_manager.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            assert {"graph_proposals", "lang_proposals", "agents", "audit_log"}.issubset(tables)


    def test_save_and_get_record(self, db_manager):
        data = {"key": "value", "number": 123}
        record_id = f"test_id_{uuid.uuid4()}"
        db_manager.save_record("graph_proposals", record_id, data)
        retrieved = db_manager.get_record("graph_proposals", record_id)
        assert retrieved == data

    def test_get_nonexistent_record(self, db_manager):
        result = db_manager.get_record("graph_proposals", f"nonexistent_{uuid.uuid4()}")
        assert result is None

    def test_query_records_all(self, db_manager):
        prefix = f"all_{uuid.uuid4()}"
        expected_nums = set()
        for i in range(3):
            record_id = f"{prefix}_{i}"
            expected_nums.add(i)
            # Store data that query_records returns (the full dict)
            db_manager.save_record("graph_proposals", record_id, {"num": i, "id": record_id})

        results = db_manager.query_records("graph_proposals", where_clause="id LIKE ?", params=(f"{prefix}_%",))
        retrieved_nums = {r.get('num') for r in results if isinstance(r, dict)}
        assert retrieved_nums == expected_nums
        assert len(results) == 3


    def test_query_records_with_limit(self, db_manager):
        prefix = f"limit_{uuid.uuid4()}"
        for i in range(5):
            db_manager.save_record("graph_proposals", f"{prefix}_{i}", {"num": i})
        results = db_manager.query_records("graph_proposals", where_clause="id LIKE ?", params=(f"{prefix}_%",), limit=2)
        assert len(results) == 2

    def test_query_records_with_offset(self, db_manager):
        prefix = f"offset_{uuid.uuid4()}"
        for i in range(5):
             db_manager.save_record("graph_proposals", f"{prefix}_{i}", {"num": i})
        all_relevant = db_manager.query_records("graph_proposals", where_clause="id LIKE ?", params=(f"{prefix}_%",))
        assert len(all_relevant) == 5 # Ensure all were saved

        results = db_manager.query_records("graph_proposals", where_clause="id LIKE ?", params=(f"{prefix}_%",), limit=2, offset=2)
        assert len(results) == 2
        nums = sorted([r.get('num') for r in results if isinstance(r, dict)])
        assert nums == [2, 3] # Assumes default order by id

    def test_log_audit(self, db_manager):
        action = f"test_action_{uuid.uuid4()}"
        audit_data = {"timestamp": datetime.utcnow().isoformat() + 'Z', "action": action, "details": {"key": "value"}}
        db_manager.log_audit(audit_data)
        logs = db_manager.get_full_audit_log()
        assert len(logs) > 0
        assert any(log.get("action") == action for log in logs)

    def test_get_full_audit_log_ordered(self, db_manager):
        timestamps_written = []
        prefix = f"ordered_log_{uuid.uuid4()}"
        for i in range(3):
            time.sleep(0.01)
            ts = datetime.utcnow().isoformat() + 'Z'
            timestamps_written.append(ts)
            audit_data = {"timestamp": ts, "action": f"{prefix}_action_{i}"}
            db_manager.log_audit(audit_data)
        logs = db_manager.get_full_audit_log()
        timestamps_read = [log["timestamp"] for log in logs if log.get("action", "").startswith(prefix)]
        assert timestamps_read == timestamps_written


class TestRegistryAPI:
    """Test RegistryAPI with database persistence."""

    def test_initialization(self, registry_api):
        assert registry_api is not None
        assert registry_api.db is not None

    def test_submit_proposal(self, registry_api):
        prop_id = f"test_proposal_{uuid.uuid4()}"
        proposal_node = {"id": prop_id, "type": "ProposalNode", "proposed_by": "agent1"}
        submitted_id = registry_api.submit_proposal(proposal_node)
        assert submitted_id == prop_id

    def test_get_proposal(self, registry_api):
        prop_id = f"get_proposal_{uuid.uuid4()}"
        proposal_node = {"id": prop_id, "type": "ProposalNode", "data": "test_data_api"}
        submitted_id = registry_api.submit_proposal(proposal_node)
        retrieved_data = registry_api.get_proposal(submitted_id) # Retrieves full {node:..., status:...} dict
        assert retrieved_data is not None
        assert retrieved_data.get("node", {}).get("id") == prop_id
        assert retrieved_data.get("node", {}).get("data") == "test_data_api"

    def test_query_proposals_empty(self, registry_api):
        results = registry_api.query_proposals(proposed_by=f"nonexistent_agent_{uuid.uuid4()}")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_query_proposals_with_filter(self, registry_api):
        prefix = f"filter_test_api_{uuid.uuid4()}"
        proposer1 = f"agent1_{prefix}"
        proposer2 = f"agent2_{prefix}"
        ids_by_proposer1 = set()
        for i in range(3):
            prop_id = f"{prefix}_prop_{i}"
            proposer = proposer1 if i % 2 == 0 else proposer2
            proposal = {"id": prop_id, "proposed_by": proposer}
            registry_api.submit_proposal(proposal)
            if proposer == proposer1: ids_by_proposer1.add(prop_id)
        results = registry_api.query_proposals(proposed_by=proposer1) # Returns list of node dicts
        retrieved_ids = {p.get("id") for p in results}
        assert len(results) >= 1
        assert retrieved_ids.issubset(ids_by_proposer1)
        assert all(p.get("proposed_by") == proposer1 for p in results)


class TestLanguageEvolutionRegistry:
    """Test LanguageEvolutionRegistry."""

    def test_initialization(self, lang_evolution_registry):
        assert lang_evolution_registry is not None
        assert lang_evolution_registry.active_grammar_version == "3.0.0"

    def test_submit_proposal(self, lang_evolution_registry):
        prop_id = f"lang_prop_submit_{uuid.uuid4()}"
        proposal_node = {"id": prop_id, "type": "ProposalNode"}
        submitted_id = lang_evolution_registry.submit_proposal(proposal_node)
        assert submitted_id == prop_id

    def test_record_vote_approve(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_approve_vote_{uuid.uuid4()}"})
        consensus_node = {"proposal_id": proposal_id, "votes": {"agent1": "yes"}}
        result = lang_evolution_registry.record_vote(consensus_node)
        assert result is True
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "approved"

    def test_record_vote_reject(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_reject_vote_{uuid.uuid4()}"})
        consensus_node = {"proposal_id": proposal_id, "votes": {"agent1": "no"}}
        result = lang_evolution_registry.record_vote(consensus_node)
        assert result is False
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "rejected"

    def test_record_vote_nonexistent(self, lang_evolution_registry):
        consensus_node = {"proposal_id": f"nonexistent_vote_{uuid.uuid4()}", "votes": {"agent1": "yes"}}
        result = lang_evolution_registry.record_vote(consensus_node)
        assert result is False

    def test_record_validation_pass(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_valid_pass_{uuid.uuid4()}"})
        validation_node = {"target": proposal_id, "result": True, "validator_id": "val1"}
        result = lang_evolution_registry.record_validation(validation_node)
        assert result is True
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "validated"
        assert proposal_data.get("validations", {}).get("val1") is True

    def test_record_validation_fail(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_valid_fail_{uuid.uuid4()}"})
        validation_node = {"target": proposal_id, "result": False, "validator_id": "val1"}
        result = lang_evolution_registry.record_validation(validation_node)
        assert result is False
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "validation_failed"
        assert proposal_data.get("validations", {}).get("val1") is False

    def test_deploy_grammar_version_validated(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_deploy_valid_{uuid.uuid4()}"})
        lang_evolution_registry.record_validation({"target": proposal_id, "result": True})
        new_version = f"3.1.{uuid.uuid4().int % 1000}"
        result = lang_evolution_registry.deploy_grammar_version(proposal_id, new_version)
        assert result is True
        assert lang_evolution_registry.active_grammar_version == new_version
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "deployed"
        assert proposal_data.get("deployed_version") == new_version

    def test_deploy_grammar_version_not_validated(self, lang_evolution_registry):
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"lang_prop_deploy_invalid_{uuid.uuid4()}"})
        initial_version = lang_evolution_registry.active_grammar_version
        result = lang_evolution_registry.deploy_grammar_version(proposal_id, f"3.2.{uuid.uuid4().int % 1000}")
        assert result is False
        assert lang_evolution_registry.active_grammar_version == initial_version
        proposal_data = lang_evolution_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") != "deployed"


class TestAgentRegistry:
    """Test AgentRegistry with database persistence."""

    def test_initialization(self, agent_registry):
        assert agent_registry is not None

    def test_register_agent(self, agent_registry):
        agent_id = f"agent_reg_{uuid.uuid4()}"
        agent_data = {"id": agent_id, "roles": ["proposer"], "trust_level": 0.8}
        agent_registry.register_agent(agent_data)
        info = agent_registry.get_agent_info(agent_id)
        assert info is not None
        assert info.get("agent_id") == agent_id # Check correct key
        assert info.get("roles") == ["proposer"]

    def test_get_agent_info_nonexistent(self, agent_registry):
        info = agent_registry.get_agent_info(f"nonexistent_{uuid.uuid4()}")
        assert info is None

    def test_authenticate_agent_valid(self, agent_registry):
        agent_id = f"agent_auth_valid_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        message = f"test_message_auth_valid_{uuid.uuid4()}"
        signature = hashlib.sha256(message.encode()).hexdigest()
        result = agent_registry.authenticate_agent(agent_id, message, signature)
        assert result is True

    def test_authenticate_agent_invalid_signature(self, agent_registry):
        agent_id = f"agent_auth_invalid_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        result = agent_registry.authenticate_agent(agent_id, "message", "wrong_signature_hash")
        assert result is False

    def test_authenticate_agent_not_found(self, agent_registry):
        result = agent_registry.authenticate_agent(f"nonexistent_auth_{uuid.uuid4()}", "message", "sig")
        assert result is False

    def test_verify_agent_signature(self, agent_registry):
        agent_id = f"agent_verify_sig_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        message = f"test_message_verify_{uuid.uuid4()}".encode()
        signature = hashlib.sha256(message).hexdigest()
        result = agent_registry.verify_agent_signature(agent_id, message, signature)
        assert result is True

    def test_query_agents_by_role(self, agent_registry):
        prefix = f"role_test_{uuid.uuid4()}"
        agent1_id, agent2_id = f"{prefix}_agent1", f"{prefix}_agent2"
        agent_registry.register_agent({"id": agent1_id, "roles": ["proposer", "voter"]})
        agent_registry.register_agent({"id": agent2_id, "roles": ["observer"]})
        results = agent_registry.query_agents(role="proposer")
        assert len(results) >= 1
        retrieved_ids = {a.get("agent_id") for a in results}
        assert agent1_id in retrieved_ids
        assert agent2_id not in retrieved_ids

    def test_query_agents_by_trust_level(self, agent_registry):
        prefix = f"trust_test_{uuid.uuid4()}"
        agent1_id, agent2_id = f"{prefix}_agent1", f"{prefix}_agent2"
        agent_registry.register_agent({"id": agent1_id, "trust_level": 0.9})
        agent_registry.register_agent({"id": agent2_id, "trust_level": 0.3})
        results = agent_registry.query_agents(min_trust_level=0.5)
        assert len(results) >= 1
        retrieved_ids = {a.get("agent_id") for a in results}
        assert all(a.get("trust_level", 0) >= 0.5 for a in results)
        assert agent1_id in retrieved_ids
        assert agent2_id not in retrieved_ids


class TestSecurityAuditEngine:
    """Test SecurityAuditEngine."""

    def test_initialization(self, security_audit_engine):
        assert security_audit_engine is not None

    def test_log_audit(self, security_audit_engine):
        action = f"test_audit_log_{uuid.uuid4()}"
        security_audit_engine.log_audit(action=action, details={"key": "value"}, entity_id="entity1", entity_type="agent")
        logs = security_audit_engine.get_full_audit_log()
        assert len(logs) > 0
        assert any(log.get("action") == action for log in logs)

    def test_enforce_policies_clean(self, security_audit_engine):
        node = {"type": "ProposalNode", "content": "safe data"}
        result = security_audit_engine.enforce_policies(node)
        assert result is True

    def test_enforce_policies_malicious(self, security_audit_engine):
        node = {"type": "ProposalNode", "content": "contains malicious keyword"}
        result = security_audit_engine.enforce_policies(node)
        assert result is False

    def test_validate_trust_policy_sufficient(self, security_audit_engine):
        node = {"proposed_by": "trusted_agent"}
        result = security_audit_engine.validate_trust_policy(node, 0.8)
        assert result is True

    def test_validate_trust_policy_insufficient(self, security_audit_engine):
        node = {"proposed_by": "untrusted_agent"}
        result = security_audit_engine.validate_trust_policy(node, 0.2)
        assert result is False

    def test_verify_audit_log_integrity(self, security_audit_engine):
        result = security_audit_engine.verify_audit_log_integrity()
        assert result is True # Mock implementation


class TestRegistryServicer:
    """Test RegistryServicer gRPC implementation."""

    def test_initialization(self, registry_servicer):
        assert registry_servicer is not None
        assert registry_servicer.registry_api is not None

    def test_register_graph_proposal_success(self, registry_servicer, agent_registry):
        agent_id = f"agent_reg_success_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["proposer"], "trust_level": 0.8})
        prop_id = f"test_prop_success_{uuid.uuid4()}"
        proposal_content_dict = {"test": "data"}
        node = Node(id=prop_id, type="ProposalNode", proposed_by=agent_id,
                    proposal_content=json.dumps(proposal_content_dict).encode('utf-8'),
                    metadata={"meta_key": "meta_val"}) # Add metadata
        # FIX: Ensure message for auth matches exactly what the server constructs
        message_dict = {"id": node.id, "type": node.type, "metadata": node.metadata, # Use node.metadata (which is a dict)
                        "proposed_by": node.proposed_by, "rationale": node.rationale,
                        "proposal_content": proposal_content_dict}
        message_str = json.dumps(message_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = RegisterGraphProposalRequest(agent_id=agent_id, signature=signature, proposal_node=node)
        context = MagicMock()
        response = registry_servicer.RegisterGraphProposal(request, context)
        assert response.status == "success"
        assert response.proposal_id == prop_id
        context.set_code.assert_not_called()

    def test_register_graph_proposal_auth_failed(self, registry_servicer):
        agent_id = f"unknown_agent_{uuid.uuid4()}" # Use unique ID
        node = Node(id=f"test_prop_auth_fail_{uuid.uuid4()}", proposed_by=agent_id, metadata={"a": "b"})
        # Signature is intentionally wrong
        request = RegisterGraphProposalRequest(agent_id=agent_id, signature="invalid_short_sig", proposal_node=node)
        context = MagicMock()
        response = registry_servicer.RegisterGraphProposal(request, context)
        # It should fail authentication *before* trying to parse content
        assert response.status == "error"
        assert "Authentication failed" in response.message # Check for the correct error
        context.set_code.assert_called_with(StatusCode.PERMISSION_DENIED)
        context.set_details.assert_called_with("Authentication failed.")

    def test_submit_language_evolution_proposal(self, registry_servicer, agent_registry):
        agent_id = f"agent_lang_evol_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["proposer", "governor"], "trust_level": 0.8})
        prop_id = f"lang_prop_{uuid.uuid4()}"
        node = Node(id=prop_id, type="ProposalNode", proposed_by=agent_id, metadata={"lang": "en"}) # Add metadata
        # FIX: Ensure message for auth matches exactly what the server constructs
        message_dict = {"id": node.id, "type": node.type, "metadata": node.metadata, # Use node.metadata (dict)
                        "proposed_by": node.proposed_by, "rationale": node.rationale, "proposal_content": {}}
        message_str = json.dumps(message_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = SubmitLanguageEvolutionProposalRequest(agent_id=agent_id, signature=signature, proposal_node=node)
        context = MagicMock()
        response = registry_servicer.SubmitLanguageEvolutionProposal(request, context)
        assert response.status == "success"
        assert response.proposal_id == prop_id
        context.set_code.assert_not_called()

    def test_record_vote(self, registry_servicer, agent_registry, lang_evolution_registry):
        agent_id = f"agent_vote_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["voter"], "trust_level": 0.8})
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"prop_vote_{uuid.uuid4()}"})
        consensus_node_dict = {"proposal_id": proposal_id, "votes": {agent_id: "yes"}, "quorum": 0.5}
        consensus_node_pb = Node(id=proposal_id, votes={agent_id: "yes"}, quorum=0.5)
        message_str = json.dumps(consensus_node_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = RecordVoteRequest(agent_id=agent_id, signature=signature, consensus_node=consensus_node_pb)
        context = MagicMock()
        response = registry_servicer.RecordVote(request, context)
        assert response.status == "success"
        assert response.consensus_reached is True
        context.set_code.assert_not_called()

    def test_record_validation(self, registry_servicer, agent_registry, lang_evolution_registry):
        agent_id = f"agent_validate_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["validator"], "trust_level": 0.8})
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"prop_validate_{uuid.uuid4()}"})
        validation_node_dict = {"target": proposal_id, "validation_type": "schema", "result": True, "validator_id": agent_id}
        validation_node_pb = Node(target=proposal_id, validation_type="schema", result=True)
        message_str = json.dumps(validation_node_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = RecordValidationRequest(agent_id=agent_id, signature=signature, validation_node=validation_node_pb)
        context = MagicMock()
        response = registry_servicer.RecordValidation(request, context)
        assert response.status == "success"
        assert response.validation_passed is True
        context.set_code.assert_not_called()

    def test_deploy_grammar_version(self, registry_servicer, agent_registry, lang_evolution_registry):
        agent_id = f"agent_deploy_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["governor", "deployer"], "trust_level": 0.8})
        proposal_id = lang_evolution_registry.submit_proposal({"id": f"prop_deploy_{uuid.uuid4()}"})
        lang_evolution_registry.record_validation({"target": proposal_id, "result": True})
        new_version = f"3.1.{uuid.uuid4().int % 1000}"
        request_dict = {"agent_id": agent_id, "proposal_id": proposal_id, "new_grammar_version": new_version}
        message_str = json.dumps(request_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = DeployGrammarVersionRequest(agent_id=agent_id, signature=signature, proposal_id=proposal_id, new_grammar_version=new_version)
        context = MagicMock()
        response = registry_servicer.DeployGrammarVersion(request, context)
        assert response.status == "success"
        assert response.deployed is True
        context.set_code.assert_not_called()

    def test_query_proposals(self, registry_servicer, agent_registry, lang_evolution_registry):
        agent_id = f"agent_query_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["voter"]})
        prop1_id = lang_evolution_registry.submit_proposal({"id": f"prop_q1_{uuid.uuid4()}", "proposed_by": agent_id})
        prop2_id = lang_evolution_registry.submit_proposal({"id": f"prop_q2_{uuid.uuid4()}", "proposed_by": "other_agent"})
        request = QueryProposalsRequest(agent_id=agent_id, proposed_by=agent_id)
        context = MagicMock()
        response = registry_servicer.QueryProposals(request, context)
        assert response.status == "success"
        assert isinstance(response.proposals, list)
        assert len(response.proposals) >= 1
        assert any(p.id == prop1_id for p in response.proposals)
        assert not any(p.id == prop2_id for p in response.proposals)
        context.set_code.assert_not_called()

    def test_get_full_audit_log(self, registry_servicer, agent_registry):
        agent_id = f"agent_audit_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["auditor", "governor"], "trust_level": 0.9})
        request_dict = {"agent_id": agent_id, "action": "get_audit_log"}
        message_str = json.dumps(request_dict, sort_keys=True)
        signature = hashlib.sha256(message_str.encode()).hexdigest()
        request = GetFullAuditLogRequest(agent_id=agent_id, signature=signature)
        context = MagicMock()
        response = registry_servicer.GetFullAuditLog(request, context)
        assert response.status == "success"
        assert isinstance(response.audit_log, list)
        context.set_code.assert_not_called()

    def test_verify_audit_log_integrity(self, registry_servicer, agent_registry):
        agent_id = f"agent_verify_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        request = VerifyAuditLogIntegrityRequest(agent_id=agent_id)
        context = MagicMock()
        response = registry_servicer.VerifyAuditLogIntegrity(request, context)
        assert response.status == "success"
        assert response.integrity_valid is True
        context.set_code.assert_not_called()


class TestAuthenticationAuthorization:
    """Test authentication and authorization helpers directly."""

    def test_authenticate_request_valid(self, registry_servicer, agent_registry):
        agent_id = f"agent_auth_valid_aa_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        message = f"test_message_auth_aa_{uuid.uuid4()}"
        signature = hashlib.sha256(message.encode()).hexdigest()
        result = registry_servicer._authenticate_request(agent_id, message, signature)
        assert result is True

    def test_authenticate_request_invalid(self, registry_servicer, agent_registry):
        agent_id = f"agent_auth_invalid_aa_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id})
        result = registry_servicer._authenticate_request(agent_id, "message", "wrong_sig_format_different")
        assert result is False

    def test_authenticate_request_unknown_agent(self, registry_servicer):
        result = registry_servicer._authenticate_request(f"unknown_aa_{uuid.uuid4()}", "message", "sig_hash_placeholder")
        assert result is False

    def test_authorize_request_valid(self, registry_servicer, agent_registry):
        agent_id = f"agent_authz_valid_aa_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["proposer", "voter"]})
        result = registry_servicer._authorize_request(agent_id, ["proposer"])
        assert result is True

    def test_authorize_request_missing_role(self, registry_servicer, agent_registry):
        agent_id = f"agent_authz_missing_aa_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["observer"]})
        result = registry_servicer._authorize_request(agent_id, ["proposer"])
        assert result is False

    def test_authorize_request_unknown_agent(self, registry_servicer):
        result = registry_servicer._authorize_request(f"unknown_authz_{uuid.uuid4()}", ["proposer"])
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_proposal_content(self, registry_api):
        """Test proposal with empty content can be saved and retrieved."""
        prop_id = f"empty_prop_{uuid.uuid4()}"
        # Proposal node itself might be minimal, just ID and proposer
        proposal = {"id": prop_id, "proposed_by": "agent_empty"} # No proposal_content key
        proposal_id = registry_api.submit_proposal(proposal)
        assert proposal_id == prop_id
        retrieved = registry_api.get_proposal(proposal_id)
        assert retrieved is not None
        assert retrieved.get("node", {}).get("id") == prop_id
        # Check that proposal_content defaults correctly if accessed later
        assert "proposal_content" not in retrieved.get("node", {}) # It shouldn't exist if not provided


    def test_concurrent_database_access(self, db_manager):
        """Test concurrent database writes using the connection pool."""
        num_threads = 10
        errors = []
        prefix = f"concurrent_prop_{uuid.uuid4()}"
        # FIX: Remove the barrier that causes deadlock
        # barrier = threading.Barrier(num_threads)

        def write_data(index):
            try:
                # barrier.wait() # <--- REMOVED DEADLOCKING BARRIER
                # save_record handles its own connection pooling
                db_manager.save_record("graph_proposals", f"{prefix}_{index}", {"num": index})
            except Exception as e:
                # Log error and traceback for debugging
                logging.error(f"Error in thread {index}: {e}", exc_info=True)
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=write_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join() # Wait for all threads to complete

        assert not errors, f"Concurrent access errors occurred: {errors}"

        # Verify all records were saved correctly
        results = db_manager.query_records("graph_proposals", where_clause="id LIKE ?", params=(f"{prefix}_%",))
        assert len(results) == num_threads
        saved_nums = {r.get('num') for r in results if isinstance(r, dict)}
        assert saved_nums == set(range(num_threads))


    def test_malformed_json_in_proposal(self, registry_servicer, agent_registry):
        """Test handling malformed JSON in proposal content during gRPC call."""
        agent_id = f"agent_malformed_{uuid.uuid4()}"
        agent_registry.register_agent({"id": agent_id, "roles": ["proposer"], "trust_level": 0.8})

        node = Node(
            id=f"test_malformed_{uuid.uuid4()}",
            proposed_by=agent_id,
            proposal_content=b'{"key": "value", invalid json}' # Malformed JSON bytes
        )

        # Signature based on what the server *would* parse if content was empty
        message_dict = {"id": node.id, "type": node.type, "metadata": node.metadata,
                        "proposed_by": node.proposed_by, "rationale": node.rationale,
                        "proposal_content": {}} # Assume empty dict as it fails parsing
        message_str = json.dumps(message_dict, sort_keys=True)
        # Use a "valid" signature for the *wrong* content (server rejects on parse anyway)
        signature = hashlib.sha256(message_str.encode()).hexdigest()

        request = RegisterGraphProposalRequest(agent_id=agent_id, signature=signature, proposal_node=node)
        context = MagicMock()
        response = registry_servicer.RegisterGraphProposal(request, context)

        assert response.status == "error"
        assert "Invalid proposal content JSON" in response.message
        context.set_code.assert_called_with(StatusCode.FAILED_PRECONDITION)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
