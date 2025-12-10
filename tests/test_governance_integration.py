"""
Integration test suite for the governance module
Tests that registry_api.py and registry_api_server.py work together properly
"""

import hashlib
import json
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional  # Ensure Dict is imported
from unittest.mock import MagicMock

import pytest

# Import from registry_api
from src.governance.registry_api import AgentRegistry as APIAgentRegistry
from src.governance.registry_api import CryptoHandler, InMemoryBackend
from src.governance.registry_api import RegistryAPI as APIRegistryAPI
from src.governance.registry_api import SecurityEngine, SimpleKMS
# Import from registry_api_server
from src.governance.registry_api_server import \
    AgentRegistry as ServerAgentRegistry
from src.governance.registry_api_server import (AuditLogEntry, DatabaseManager,
                                                DeployGrammarVersionRequest,
                                                DeployGrammarVersionResponse,
                                                GetFullAuditLogRequest,
                                                GetFullAuditLogResponse,
                                                LanguageEvolutionRegistry,
                                                Node, QueryProposalsRequest,
                                                QueryProposalsResponse,
                                                RecordValidationRequest,
                                                RecordValidationResponse,
                                                RecordVoteRequest,
                                                RecordVoteResponse,
                                                RegisterGraphProposalRequest,
                                                RegisterGraphProposalResponse)
from src.governance.registry_api_server import \
    RegistryAPI as ServerRegistryAPI  # <--- Added StatusCode import
from src.governance.registry_api_server import (
    RegistryServicer, SecurityAuditEngine, StatusCode,
    SubmitLanguageEvolutionProposalRequest,
    SubmitLanguageEvolutionProposalResponse, VerifyAuditLogIntegrityRequest,
    VerifyAuditLogIntegrityResponse)


@pytest.fixture
def temp_db():
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        Path(db_path).unlink(missing_ok=True) # Use missing_ok=True
    except Exception as e:
        print(f"Warning: Failed to delete temp db {db_path}: {e}")


@pytest.fixture
def integrated_system(temp_db):
    """Create fully integrated governance system."""
    # Database backend
    db_manager = DatabaseManager(temp_db)

    # Server components
    server_registry_api = ServerRegistryAPI(db_manager)
    lang_evolution_registry = LanguageEvolutionRegistry(db_manager)
    server_agent_registry = ServerAgentRegistry(db_manager)
    security_audit_engine = SecurityAuditEngine(db_manager)

    # API components (for comparison/verification)
    in_memory_backend = InMemoryBackend()
    kms = SimpleKMS()
    api_registry = APIRegistryAPI(backend=in_memory_backend, kms=kms)

    # gRPC Servicer
    servicer = RegistryServicer(
        registry_api=server_registry_api,
        lang_evolution_registry=lang_evolution_registry,
        agent_registry=server_agent_registry,
        security_audit_engine=security_audit_engine
    )

    # Register test agents
    for agent_id, roles, trust_level in [
        ("agent-alice", ["proposer", "voter", "validator", "governor", "deployer", "auditor"], 0.9),
        ("agent-bob", ["proposer", "voter", "validator"], 0.7),
        ("agent-charlie", ["voter"], 0.5),
        ("agent-observer", ["observer"], 0.3),
    ]:
        server_agent_registry.register_agent({
            "id": agent_id,
            "roles": roles,
            "trust_level": trust_level
        })

        # Also register in API agent registry
        public_key_pem = kms.get_public_key_pem(agent_id)
        api_registry.agent_registry.register_agent(agent_id, public_key_pem, trust_level)

    return {
        "db_manager": db_manager,
        "server_registry_api": server_registry_api,
        "lang_evolution_registry": lang_evolution_registry,
        "server_agent_registry": server_agent_registry,
        "security_audit_engine": security_audit_engine,
        "servicer": servicer,
        "api_registry": api_registry,
        "kms": kms
    }


# Helper to create the dict the server uses for auth message
def create_node_dict_for_auth(node: Node, agent_id: str) -> Dict:
    """Creates the dictionary structure used by the server for proposal auth."""
    proposal_content_dict = {}
    if node.proposal_content:
        try:
            proposal_content_dict = json.loads(node.proposal_content.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass # Keep empty if invalid

    # Metadata handling needs to be robust for tests (can be None or dict)
    metadata_dict = node.metadata if isinstance(node.metadata, dict) else {}

    return {
        "id": node.id,
        "type": node.type,
        "metadata": metadata_dict,
        "proposed_by": node.proposed_by or agent_id,
        "rationale": node.rationale,
        "proposal_content": proposal_content_dict
    }

# Helper to calculate signature based on the server's method
def calculate_signature(data_dict: Dict) -> str:
    """Calculates the SHA256 signature the server expects."""
    message_for_auth = json.dumps(data_dict, sort_keys=True)
    return hashlib.sha256(message_for_auth.encode()).hexdigest()


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_system_initialization(self, integrated_system):
        """Test that all components initialize correctly."""
        assert integrated_system["db_manager"] is not None
        assert integrated_system["server_registry_api"] is not None
        assert integrated_system["lang_evolution_registry"] is not None
        assert integrated_system["server_agent_registry"] is not None
        assert integrated_system["security_audit_engine"] is not None
        assert integrated_system["servicer"] is not None
        assert integrated_system["api_registry"] is not None

    def test_agents_registered(self, integrated_system):
        """Test that agents are registered in both systems."""
        server_agent_registry = integrated_system["server_agent_registry"]
        api_registry = integrated_system["api_registry"]

        # Check server registry
        alice_server = server_agent_registry.get_agent_info("agent-alice")
        assert alice_server is not None
        assert alice_server["trust_level"] == 0.9

        # Check API registry
        alice_api = api_registry.agent_registry.get_agent_info("agent-alice")
        assert alice_api is not None
        assert alice_api["trust_level"] == 0.9

    def test_database_persistence(self, integrated_system):
        """Test that database persists data correctly."""
        db_manager = integrated_system["db_manager"]

        # Save data
        test_data = {"test": "value", "number": 123}
        db_manager.save_record("graph_proposals", "test_id", test_data)

        # Retrieve data
        retrieved = db_manager.get_record("graph_proposals", "test_id")

        assert retrieved == test_data


class TestEndToEndGraphProposalWorkflow:
    """Test complete workflow for graph proposals."""

    def test_graph_proposal_submission(self, integrated_system):
        """Test submitting graph proposal through server."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-alice"

        # Create proposal content
        proposal_content = {"add": {"NewNodeType": {"description": "New type"}}}
        proposal_content_bytes = json.dumps(proposal_content).encode('utf-8')

        # Create the Node object
        node = Node(
            id="graph_proposal_1",
            type="ProposalNode",
            proposed_by=agent_id,
            rationale="Add new graph node type",
            proposal_content=proposal_content_bytes,
            metadata={"version": "1.0"} # Example metadata
        )

        # **FIX: Calculate correct signature**
        node_dict_for_auth = create_node_dict_for_auth(node, agent_id)
        correct_signature = calculate_signature(node_dict_for_auth)

        request = RegisterGraphProposalRequest(
            agent_id=agent_id,
            signature=correct_signature, # Use correct signature
            proposal_node=node
        )

        context = MagicMock()

        # Submit proposal
        response = servicer.RegisterGraphProposal(request, context)

        # --- Assertions ---
        assert response.status == "success", f"Expected 'success', got '{response.status}' with message: {response.message}"
        assert len(response.proposal_id) > 0

        # Verify proposal was stored
        server_registry = integrated_system["server_registry_api"]
        proposal = server_registry.get_proposal(response.proposal_id)

        assert proposal is not None
        assert proposal["node"]["id"] == "graph_proposal_1"
        assert proposal["node"]["proposal_content"] == proposal_content # Verify content stored correctly

    def test_graph_proposal_query(self, integrated_system):
        """Test querying graph proposals."""
        servicer = integrated_system["servicer"]
        server_registry = integrated_system["server_registry_api"]

        # Submit multiple proposals
        for i in range(3):
            proposal = {
                "id": f"graph_prop_{i}",
                "type": "ProposalNode",
                "proposed_by": "agent-alice" if i % 2 == 0 else "agent-bob"
            }
            # Note: Directly calling submit_proposal doesn't involve auth/server layer
            server_registry.submit_proposal(proposal)

        # Query all
        results = server_registry.query_proposals()
        assert len(results) >= 3

        # Query by proposer
        alice_results = server_registry.query_proposals(proposed_by="agent-alice")
        assert all(p.get("proposed_by") == "agent-alice" for p in alice_results)


class TestEndToEndLanguageEvolutionWorkflow:
    """Test complete workflow for language evolution proposals."""

    def test_complete_language_evolution_workflow(self, integrated_system):
        """Test full workflow: submit -> vote -> validate -> deploy."""
        servicer = integrated_system["servicer"]
        lang_registry = integrated_system["lang_evolution_registry"]
        agent_id = "agent-alice" # Governor agent

        # Step 1: Submit proposal
        proposal_content = {"add": {"NewFeature": {"syntax": "new_syntax"}}}
        proposal_content_bytes = json.dumps(proposal_content).encode('utf-8')
        proposal_node = Node(
            id="lang_evolution_1",
            type="ProposalNode",
            proposed_by=agent_id,
            rationale="Add new language feature",
            proposal_content=proposal_content_bytes
        )

        # **FIX: Calculate correct signature for submit**
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, agent_id)
        submit_signature = calculate_signature(node_dict_for_auth)

        submit_request = SubmitLanguageEvolutionProposalRequest(
            agent_id=agent_id,
            signature=submit_signature,
            proposal_node=proposal_node
        )

        context = MagicMock()
        submit_response = servicer.SubmitLanguageEvolutionProposal(submit_request, context)

        assert submit_response.status == "success", f"Submit failed: {submit_response.message}"
        proposal_id = submit_response.proposal_id

        # Step 2: Record votes
        votes_dict = {"agent-alice": "yes", "agent-bob": "yes", "agent-charlie": "yes"}
        consensus_node = Node(
            id=proposal_id, # Can use id here as server checks both id/proposal_id
            proposal_id=proposal_id, # Explicitly add proposal_id too
            votes=votes_dict,
            quorum=0.5
        )

        # **FIX: Calculate correct signature for vote**
        # The server uses a dict with 'proposal_id', 'votes', 'quorum' for vote auth
        vote_auth_dict = {
            "proposal_id": proposal_id,
            "votes": votes_dict,
            "quorum": 0.5
        }
        vote_signature = calculate_signature(vote_auth_dict)

        vote_request = RecordVoteRequest(
            agent_id=agent_id, # Alice votes (has governor role)
            signature=vote_signature,
            consensus_node=consensus_node
        )

        vote_response = servicer.RecordVote(vote_request, context)

        assert vote_response.status == "success", f"Vote failed: {vote_response.message}"
        assert vote_response.consensus_reached is True

        # Step 3: Record validation
        validation_node = Node(
            target=proposal_id,
            validation_type="schema",
            result=True
        )

        # **FIX: Calculate correct signature for validation**
        # Server uses 'target', 'validation_type', 'result', 'validator_id'
        validation_auth_dict = {
            "target": proposal_id,
            "validation_type": "schema",
            "result": True,
            "validator_id": agent_id # Agent performing validation
        }
        validation_signature = calculate_signature(validation_auth_dict)

        validation_request = RecordValidationRequest(
            agent_id=agent_id, # Alice validates (has governor role)
            signature=validation_signature,
            validation_node=validation_node
        )

        validation_response = servicer.RecordValidation(validation_request, context)

        assert validation_response.status == "success", f"Validation failed: {validation_response.message}"
        assert validation_response.validation_passed is True

        # Step 4: Deploy grammar version
        new_version = "3.1.0"
        # **FIX: Calculate correct signature for deploy**
        # Server uses 'agent_id', 'proposal_id', 'new_grammar_version'
        deploy_auth_dict = {
            "agent_id": agent_id,
            "proposal_id": proposal_id,
            "new_grammar_version": new_version
        }
        deploy_signature = calculate_signature(deploy_auth_dict)

        deploy_request = DeployGrammarVersionRequest(
            agent_id=agent_id, # Alice deploys (has governor role)
            signature=deploy_signature,
            proposal_id=proposal_id,
            new_grammar_version=new_version
        )

        deploy_response = servicer.DeployGrammarVersion(deploy_request, context)

        assert deploy_response.status == "success", f"Deploy failed: {deploy_response.message}"
        assert deploy_response.deployed is True

        # Verify grammar version updated
        assert lang_registry.active_grammar_version == new_version

    def test_proposal_rejection_workflow(self, integrated_system):
        """Test workflow where proposal is rejected."""
        servicer = integrated_system["servicer"]
        lang_registry = integrated_system["lang_evolution_registry"]
        proposer_agent_id = "agent-bob"
        voter_agent_id = "agent-alice" # Alice votes no

        # Submit proposal
        proposal_node = Node(
            id="rejected_proposal",
            type="ProposalNode",
            proposed_by=proposer_agent_id
        )

        # **FIX: Signature for submit**
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, proposer_agent_id)
        submit_signature = calculate_signature(node_dict_for_auth)

        submit_request = SubmitLanguageEvolutionProposalRequest(
            agent_id=proposer_agent_id,
            signature=submit_signature,
            proposal_node=proposal_node
        )

        context = MagicMock()
        submit_response = servicer.SubmitLanguageEvolutionProposal(submit_request, context)
        assert submit_response.status == "success"
        proposal_id = submit_response.proposal_id

        # Record negative votes
        votes_dict = {"agent-alice": "no", "agent-bob": "no", "agent-charlie": "no"}
        consensus_node = Node(
            id=proposal_id,
            proposal_id=proposal_id,
            votes=votes_dict,
            quorum=0.5
        )

        # **FIX: Signature for vote**
        vote_auth_dict = {
            "proposal_id": proposal_id,
            "votes": votes_dict,
            "quorum": 0.5
        }
        vote_signature = calculate_signature(vote_auth_dict)

        vote_request = RecordVoteRequest(
            agent_id=voter_agent_id, # Alice votes
            signature=vote_signature,
            consensus_node=consensus_node
        )

        vote_response = servicer.RecordVote(vote_request, context)

        assert vote_response.status == "success"
        # Consensus should NOT be reached with 'no' votes
        proposal_data = lang_registry.db.get_record("lang_proposals", proposal_id)
        assert proposal_data.get("status") == "rejected" # Check final status
        assert vote_response.consensus_reached is False # API returns False if not approved

        # Verify cannot deploy (attempt by governor Alice)
        new_version = "3.1.0"
        deploy_auth_dict = {
            "agent_id": "agent-alice",
            "proposal_id": proposal_id,
            "new_grammar_version": new_version
        }
        deploy_signature = calculate_signature(deploy_auth_dict)

        deploy_request = DeployGrammarVersionRequest(
            agent_id="agent-alice",
            signature=deploy_signature,
            proposal_id=proposal_id,
            new_grammar_version=new_version
        )

        deploy_response = servicer.DeployGrammarVersion(deploy_request, context)

        assert deploy_response.deployed is False
        assert deploy_response.status == "error" # Server should return error status


class TestSecurityAndAuditIntegration:
    """Test security and audit logging integration."""

    def test_security_policy_enforcement(self, integrated_system):
        """Test that security policies are enforced across the system."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-alice"

        # Try to submit malicious proposal
        malicious_content = {"exploit": "os.system('rm -rf /')"}
        malicious_content_bytes = json.dumps(malicious_content).encode('utf-8')
        malicious_node = Node(
            id="malicious_proposal",
            type="ProposalNode",
            proposed_by=agent_id,
            proposal_content=malicious_content_bytes
        )

        # **FIX: Calculate correct signature**
        node_dict_for_auth = create_node_dict_for_auth(malicious_node, agent_id)
        correct_signature = calculate_signature(node_dict_for_auth)

        request = RegisterGraphProposalRequest(
            agent_id=agent_id,
            signature=correct_signature, # Use correct signature
            proposal_node=malicious_node
        )

        context = MagicMock()
        response = servicer.RegisterGraphProposal(request, context)

        # Should be rejected
        assert response.status == "error"
        assert "Security policy violation" in response.message
        context.set_code.assert_called()

    def test_audit_logging_integration(self, integrated_system):
        """Test that audit logging works across operations."""
        servicer = integrated_system["servicer"]
        security_audit = integrated_system["security_audit_engine"]
        agent_id = "agent-alice" # Auditor

        # Perform an operation to generate logs
        proposal_node = Node(id="audit_test", type="ProposalNode", proposed_by=agent_id)
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, agent_id)
        submit_signature = calculate_signature(node_dict_for_auth)
        submit_request = SubmitLanguageEvolutionProposalRequest(
            agent_id=agent_id,
            signature=submit_signature,
            proposal_node=proposal_node
        )
        context = MagicMock()
        servicer.SubmitLanguageEvolutionProposal(submit_request, context)

        # Get audit log
        # **FIX: Calculate correct signature for get audit log**
        audit_auth_dict = {"agent_id": agent_id, "action": "get_audit_log"}
        audit_signature = calculate_signature(audit_auth_dict)

        audit_request = GetFullAuditLogRequest(
            agent_id=agent_id,
            signature=audit_signature # Use correct signature
        )

        audit_response = servicer.GetFullAuditLog(audit_request, context)

        assert audit_response.status == "success", f"GetAuditLog failed: {audit_response.message}"
        # **FIX: Adjust assertion to expect >= 1 log entry from test action**
        assert len(audit_response.audit_log) >= 1 # Expect at least the proposal submission log

        # Verify integrity (Doesn't require signature in mock, but log action does)
        integrity_request = VerifyAuditLogIntegrityRequest(agent_id=agent_id)
        integrity_response = servicer.VerifyAuditLogIntegrity(integrity_request, context)

        assert integrity_response.integrity_valid is True

        # Check that integrity check itself was logged
        final_log = security_audit.get_full_audit_log()
        assert any(entry.get("action") == "integrity_check_requested" for entry in final_log)

    def test_trust_level_enforcement(self, integrated_system):
        """Test that trust levels are enforced."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-observer" # Low trust agent

        # Low trust agent tries to submit proposal
        proposal_node = Node(
            id="low_trust_proposal",
            type="ProposalNode",
            proposed_by=agent_id
        )

        # **FIX: Calculate correct signature**
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, agent_id)
        correct_signature = calculate_signature(node_dict_for_auth)

        request = RegisterGraphProposalRequest(
            agent_id=agent_id,
            signature=correct_signature,
            proposal_node=proposal_node
        )

        context = MagicMock()
        response = servicer.RegisterGraphProposal(request, context)

        # Should be rejected (primarily due to missing role in this setup)
        assert response.status == "error"
        # Check if the error message reflects authorization or trust issue
        assert "Authorization failed" in response.message or "trust level too low" in response.message


class TestAuthenticationAuthorizationIntegration:
    """Test authentication and authorization across the system."""

    def test_authentication_required(self, integrated_system):
        """Test that authentication is required for operations."""
        servicer = integrated_system["servicer"]

        # Try with unknown agent and invalid signature
        request = SubmitLanguageEvolutionProposalRequest(
            agent_id="unknown_agent",
            signature="invalid_signature", # Incorrect signature
            proposal_node=Node(id="test")
        )

        context = MagicMock()
        response = servicer.SubmitLanguageEvolutionProposal(request, context)

        assert response.status == "error"
        assert "Authentication failed" in response.message
        context.set_code.assert_called()

    def test_role_based_authorization(self, integrated_system):
        """Test role-based authorization."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-observer" # Lacks proposer role

        # Observer tries to submit proposal
        proposal_node=Node(id="test", type="ProposalNode", proposed_by=agent_id)

        # **FIX: Calculate correct signature**
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, agent_id)
        correct_signature = calculate_signature(node_dict_for_auth)

        request = SubmitLanguageEvolutionProposalRequest(
            agent_id=agent_id,
            signature=correct_signature, # Correct signature, but wrong role
            proposal_node=proposal_node
        )

        context = MagicMock()
        response = servicer.SubmitLanguageEvolutionProposal(request, context)

        # Should be denied due to lack of proposer/governor role
        assert response.status == "error"
        assert "Authorization failed" in response.message

    def test_correct_roles_authorized(self, integrated_system):
        """Test that correct roles are authorized."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-alice" # Has proposer/governor roles

        # Alice submits proposal
        proposal_node=Node(
            id="authorized_proposal",
            type="ProposalNode",
            proposed_by=agent_id
        )

        # **FIX: Calculate correct signature**
        node_dict_for_auth = create_node_dict_for_auth(proposal_node, agent_id)
        correct_signature = calculate_signature(node_dict_for_auth)

        request = SubmitLanguageEvolutionProposalRequest(
            agent_id=agent_id,
            signature=correct_signature, # Correct signature and role
            proposal_node=proposal_node
        )

        context = MagicMock()
        response = servicer.SubmitLanguageEvolutionProposal(request, context)

        assert response.status == "success", f"Auth test failed: {response.message}"


class TestDatabasePersistenceIntegration:
    """Test database persistence across operations."""

    def test_proposals_persist(self, integrated_system):
        """Test that proposals persist in database."""
        lang_registry = integrated_system["lang_evolution_registry"]
        db_manager = integrated_system["db_manager"]

        # Submit proposal directly to registry component (bypasses server auth)
        proposal_id = lang_registry.submit_proposal({
            "id": "persist_test",
            "type": "ProposalNode"
        })

        # Retrieve directly from database manager
        db_proposal = db_manager.get_record("lang_proposals", proposal_id)

        assert db_proposal is not None
        assert db_proposal["node"]["id"] == "persist_test"

    def test_audit_log_persists(self, integrated_system):
        """Test that audit log persists."""
        security_audit = integrated_system["security_audit_engine"]
        db_manager = integrated_system["db_manager"]

        # Log audit entry directly (bypasses server auth)
        security_audit.log_audit(
            action="test_action",
            details={"test": "data"},
            entity_id="entity1",
            entity_type="agent"
        )

        # Retrieve from database manager
        audit_log = db_manager.get_full_audit_log()

        assert len(audit_log) > 0 # Should have initialization logs + test_action
        assert any(entry.get("action") == "test_action" for entry in audit_log)

    def test_state_survives_restart(self, temp_db):
        """Test that state survives system restart."""
        # First session
        db_manager1 = DatabaseManager(temp_db)
        lang_registry1 = LanguageEvolutionRegistry(db_manager1)

        proposal_id = lang_registry1.submit_proposal({
            "id": "restart_test",
            "type": "ProposalNode"
        })

        # Simulate restart - create new instances using the *same db file*
        db_manager2 = DatabaseManager(temp_db)
        lang_registry2 = LanguageEvolutionRegistry(db_manager2)

        # Retrieve proposal using the second instance
        proposals = lang_registry2.query_proposals() # Queries DB directly

        assert any(p.get("id") == "restart_test" for p in proposals)


class TestQueryingIntegration:
    """Test querying across the integrated system."""

    def test_query_proposals_with_filters(self, integrated_system):
        """Test querying with various filters via server."""
        servicer = integrated_system["servicer"]
        lang_registry = integrated_system["lang_evolution_registry"] # Use this to seed data

        # Submit proposals with different attributes directly to registry
        for i in range(5):
            proposal = {
                "id": f"query_test_{i}",
                "proposed_by": "agent-alice" if i < 3 else "agent-bob",
                "type": "ProposalNode"
            }
            lang_registry.submit_proposal(proposal)

        # Query all via server (requires valid agent)
        request = QueryProposalsRequest(agent_id="agent-alice")
        context = MagicMock()
        response = servicer.QueryProposals(request, context)
        assert response.status == "success"
        assert len(response.proposals) >= 5

        # Query by proposer via server
        request_filtered = QueryProposalsRequest(
            agent_id="agent-alice",
            proposed_by="agent-alice"
        )
        response_filtered = servicer.QueryProposals(request_filtered, context)
        assert response_filtered.status == "success"
        # Check the actual returned proposal data
        assert len(response_filtered.proposals) >= 3
        assert all(p.proposed_by == "agent-alice" for p in response_filtered.proposals)

    def test_agent_querying(self, integrated_system):
        """Test querying agents directly via registry component."""
        agent_registry = integrated_system["server_agent_registry"]

        # Query by role
        proposers = agent_registry.query_agents(role="proposer")
        assert len(proposers) >= 2  # alice and bob should have this role

        # Query by trust level
        high_trust = agent_registry.query_agents(min_trust_level=0.8)
        assert len(high_trust) >= 1 # Only alice has >= 0.8
        assert all(a.get("trust_level", 0) >= 0.8 for a in high_trust)


class TestConcurrencyIntegration:
    """Test concurrent operations (using direct registry access for simplicity)."""

    def test_concurrent_proposal_submissions(self, integrated_system):
        """Test concurrent proposal submissions directly to registry."""
        import threading

        lang_registry = integrated_system["lang_evolution_registry"]
        results = {}
        lock = threading.Lock()

        def submit_proposal(index):
            try:
                proposal_id = lang_registry.submit_proposal({
                    "id": f"concurrent_{index}",
                    "type": "ProposalNode"
                })
                with lock:
                    results[index] = proposal_id
            except Exception as e:
                 with lock:
                    results[index] = f"Error: {e}"

        threads = []
        for i in range(10):
            t = threading.Thread(target=submit_proposal, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All submissions should succeed and have unique IDs
        successful_ids = [v for v in results.values() if not isinstance(v, str) or not v.startswith("Error")]
        print(f"Concurrent submission results: {results}")
        assert len(successful_ids) == 10, f"Expected 10 successful submissions, got {len(successful_ids)}"
        assert len(set(successful_ids)) == 10, "Proposal IDs were not unique"

    def test_concurrent_voting(self, integrated_system):
        """Test concurrent voting on same proposal directly via registry."""
        import threading

        lang_registry = integrated_system["lang_evolution_registry"]

        # Submit proposal
        proposal_id = lang_registry.submit_proposal({
            "id": "vote_concurrent",
            "type": "ProposalNode"
        })

        errors = []
        lock = threading.Lock()

        def record_vote(agent_id, vote):
            try:
                consensus_node = {
                    "proposal_id": proposal_id,
                    "votes": {agent_id: vote} # Each thread adds its own vote
                }
                lang_registry.record_vote(consensus_node)
            except Exception as e:
                with lock:
                    errors.append(f"Error voting for {agent_id}: {e}")

        threads = []
        agent_votes = {"agent-alice": "yes", "agent-bob": "yes", "agent-charlie": "no"}
        for agent_id, vote in agent_votes.items():
            t = threading.Thread(target=record_vote, args=(agent_id, vote))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check for errors during voting
        assert not errors, f"Errors occurred during concurrent voting: {errors}"

        # Verify final state of votes in the proposal
        final_proposal = lang_registry.db.get_record("lang_proposals", proposal_id)
        assert final_proposal is not None, "Proposal not found after concurrent voting"
        assert "votes" in final_proposal, "Votes field missing after concurrent voting"
        # Check if all votes were recorded (exact count depends on race conditions if not careful,
        # but all keys should be present if save_record is atomic enough)
        print(f"Final votes: {final_proposal['votes']}")
        assert len(final_proposal["votes"]) == len(agent_votes), "Not all concurrent votes were recorded"
        assert final_proposal["votes"]["agent-alice"] == "yes"
        assert final_proposal["votes"]["agent-bob"] == "yes"
        assert final_proposal["votes"]["agent-charlie"] == "no"


class TestErrorHandlingIntegration:
    """Test error handling across the system."""

    def test_invalid_proposal_id(self, integrated_system):
        """Test handling invalid proposal IDs directly via registry."""
        lang_registry = integrated_system["lang_evolution_registry"]

        # Try to vote on nonexistent proposal
        # Expecting it to handle gracefully and return False
        result = lang_registry.record_vote({
            "proposal_id": "nonexistent",
            "votes": {"agent-alice": "yes"}
        })

        assert result is False

    def test_malformed_data_handling(self, integrated_system):
        """Test handling malformed data via server."""
        servicer = integrated_system["servicer"]
        agent_id = "agent-alice"

        # Submit proposal with node having non-JSON bytes content
        malformed_node = Node(
             id="malformed_data_test",
             type="ProposalNode",
             proposed_by=agent_id,
             proposal_content=b'\x80abc' # Invalid UTF-8
        )

        # **FIX: Calculate signature based on how server *would* parse (even if it fails later)**
        # Server tries to decode, fails, uses {}. Auth message uses this {}.
        node_dict_for_auth = {
            "id": malformed_node.id,
            "type": malformed_node.type,
            "metadata": {}, # Assume empty if not provided or invalid
            "proposed_by": malformed_node.proposed_by or agent_id,
            "rationale": malformed_node.rationale,
            "proposal_content": {} # Represents the failed parse result
        }
        correct_signature = calculate_signature(node_dict_for_auth)


        request = RegisterGraphProposalRequest(
            agent_id=agent_id,
            signature=correct_signature, # Signature based on expected failure state
            proposal_node=malformed_node
        )

        context = MagicMock()
        response = servicer.RegisterGraphProposal(request, context)

        # Should fail gracefully due to invalid content JSON during processing
        assert response.status == "error"
        assert "Invalid proposal content JSON" in response.message
        # **FIX: Use actual StatusCode enum/class for assertion**
        context.set_code.assert_called_with(StatusCode.FAILED_PRECONDITION)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])