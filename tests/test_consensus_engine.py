"""
Comprehensive test suite for consensus_engine.py
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from consensus_engine import (
    DEFAULT_APPROVAL_THRESHOLD,
    DEFAULT_QUORUM,
    MAX_PROPOSAL_SIZE,
    MAX_RATIONALE_LENGTH,
    MAX_TRUST_LEVEL,
    MIN_TRUST_LEVEL,
    Agent,
    ConsensusEngine,
    Proposal,
    ProposalStatus,
    Vote,
    VoteType,
)


@pytest.fixture
def engine():
    """Create consensus engine."""
    eng = ConsensusEngine(
        quorum=0.51, approval_threshold=0.66, proposal_duration_days=7
    )
    yield eng
    eng.shutdown()


@pytest.fixture
def registered_agents(engine):
    """Register test agents."""
    agents = []
    for i, trust in enumerate([0.9, 0.8, 0.7, 0.6]):
        engine.register_agent(f"agent{i}", trust_level=trust)
        agents.append(f"agent{i}")
    return agents


class TestAgentRegistration:
    """Test agent registration."""

    def test_register_agent(self, engine):
        """Test registering new agent."""
        result = engine.register_agent("agent1", trust_level=0.8)

        assert result is True
        assert "agent1" in engine.agents
        assert engine.agents["agent1"].trust_level == 0.8

    def test_register_duplicate_agent(self, engine):
        """Test registering duplicate agent fails."""
        engine.register_agent("agent1", trust_level=0.8)

        with pytest.raises(ValueError, match="already registered"):
            engine.register_agent("agent1", trust_level=0.7)

    def test_invalid_agent_id(self, engine):
        """Test invalid agent ID."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            engine.register_agent("", trust_level=0.8)

    def test_invalid_trust_level(self, engine):
        """Test invalid trust level."""
        with pytest.raises(ValueError, match="Trust level must be"):
            engine.register_agent("agent1", trust_level=1.5)

        with pytest.raises(ValueError, match="Trust level must be"):
            engine.register_agent("agent2", trust_level=-0.1)

    def test_trust_level_bounds(self, engine):
        """Test trust level boundary values."""
        engine.register_agent("agent1", trust_level=MIN_TRUST_LEVEL)
        engine.register_agent("agent2", trust_level=MAX_TRUST_LEVEL)

        assert engine.agents["agent1"].trust_level == MIN_TRUST_LEVEL
        assert engine.agents["agent2"].trust_level == MAX_TRUST_LEVEL


class TestProposalCreation:
    """Test proposal creation."""

    def test_create_proposal(self, engine, registered_agents):
        """Test creating valid proposal."""
        proposal_graph = {
            "id": "test_proposal",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": {"NewNode": {}}},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        assert proposal_id is not None
        assert proposal_id in engine.proposals
        assert engine.proposals[proposal_id].proposer_id == registered_agents[0]

    def test_propose_unregistered_agent(self, engine):
        """Test proposing with unregistered agent."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        with pytest.raises(ValueError, match="not registered"):
            engine.propose(proposal_graph, "unknown_agent")

    def test_invalid_proposal_structure(self, engine, registered_agents):
        """Test invalid proposal structure."""
        invalid_graph = {"id": "test", "type": "InvalidType", "nodes": []}

        with pytest.raises(ValueError, match="Invalid proposal"):
            engine.propose(invalid_graph, registered_agents[0])

    def test_proposal_missing_required_fields(self, engine, registered_agents):
        """Test proposal missing required fields."""
        invalid_graph = {
            "id": "test"
            # Missing type and nodes
        }

        with pytest.raises(ValueError, match="Missing required field"):
            engine.propose(invalid_graph, registered_agents[0])

    def test_proposal_without_proposal_node(self, engine, registered_agents):
        """Test proposal without ProposalNode."""
        graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [{"id": "n1", "type": "MathNode"}],
        }

        with pytest.raises(ValueError, match="must contain a ProposalNode"):
            engine.propose(graph, registered_agents[0])

    def test_proposal_too_large(self, engine, registered_agents):
        """Test proposal size limit."""
        # Create very large proposal with valid structure
        large_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {
                        "add": "x" * MAX_PROPOSAL_SIZE  # Valid structure but too large
                    },
                }
            ],
        }

        with pytest.raises(ValueError, match="Proposal too large"):
            engine.propose(large_graph, registered_agents[0])

    def test_custom_duration(self, engine, registered_agents):
        """Test custom proposal duration."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(
            proposal_graph, registered_agents[0], duration_days=14
        )

        proposal = engine.proposals[proposal_id]
        duration = (proposal.closes_at - proposal.created_at).days

        assert duration == 14


class TestVoting:
    """Test voting functionality."""

    def test_vote_approve(self, engine, registered_agents):
        """Test approve vote."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])
        result = engine.vote(proposal_id, registered_agents[1], "approve")

        assert result is True
        assert registered_agents[1] in engine.proposals[proposal_id].votes

    def test_vote_reject(self, engine, registered_agents):
        """Test reject vote."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])
        result = engine.vote(proposal_id, registered_agents[1], "reject")

        assert result is True
        vote = engine.proposals[proposal_id].votes[registered_agents[1]]
        assert vote.vote == VoteType.REJECT

    def test_vote_abstain(self, engine, registered_agents):
        """Test abstain vote."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])
        result = engine.vote(proposal_id, registered_agents[1], "abstain")

        assert result is True
        vote = engine.proposals[proposal_id].votes[registered_agents[1]]
        assert vote.vote == VoteType.ABSTAIN

    def test_invalid_vote_type(self, engine, registered_agents):
        """Test invalid vote type."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        with pytest.raises(ValueError, match="Invalid vote type"):
            engine.vote(proposal_id, registered_agents[1], "invalid")

    def test_vote_on_nonexistent_proposal(self, engine, registered_agents):
        """Test voting on non-existent proposal."""
        with pytest.raises(ValueError, match="not found"):
            engine.vote("nonexistent", registered_agents[0], "approve")

    def test_vote_unregistered_agent(self, engine, registered_agents):
        """Test voting with unregistered agent."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        with pytest.raises(ValueError, match="not registered"):
            engine.vote(proposal_id, "unknown_agent", "approve")

    def test_vote_change(self, engine, registered_agents):
        """Test changing vote."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        engine.vote(proposal_id, registered_agents[1], "approve")
        engine.vote(proposal_id, registered_agents[1], "reject")

        vote = engine.proposals[proposal_id].votes[registered_agents[1]]
        assert vote.vote == VoteType.REJECT

    def test_vote_with_rationale(self, engine, registered_agents):
        """Test vote with rationale."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])
        rationale = "This is my reasoning"

        engine.vote(proposal_id, registered_agents[1], "approve", rationale=rationale)

        vote = engine.proposals[proposal_id].votes[registered_agents[1]]
        assert vote.rationale == rationale

    def test_rationale_too_long(self, engine, registered_agents):
        """Test rationale length limit."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])
        long_rationale = "x" * (MAX_RATIONALE_LENGTH + 1)

        with pytest.raises(ValueError, match="Rationale too long"):
            engine.vote(
                proposal_id, registered_agents[1], "approve", rationale=long_rationale
            )


class TestConsensusEvaluation:
    """Test consensus evaluation."""

    def test_consensus_approved(self, engine, registered_agents):
        """Test proposal approved by consensus."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": {"NewNode": {}}},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # Vote with 3/4 approval (75% > 66% threshold)
        engine.vote(proposal_id, registered_agents[0], "approve")
        engine.vote(proposal_id, registered_agents[1], "approve")
        engine.vote(proposal_id, registered_agents[2], "approve")
        engine.vote(proposal_id, registered_agents[3], "reject")

        # Close proposal
        engine.proposals[proposal_id].status = ProposalStatus.CLOSED

        result = engine.evaluate_consensus(proposal_id)

        assert result["status"] == "approved"
        assert result["quorum_met"] is True

    def test_consensus_rejected(self, engine, registered_agents):
        """Test proposal rejected by consensus."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # Vote with majority reject
        engine.vote(proposal_id, registered_agents[0], "reject")
        engine.vote(proposal_id, registered_agents[1], "reject")
        engine.vote(proposal_id, registered_agents[2], "reject")
        engine.vote(proposal_id, registered_agents[3], "approve")

        engine.proposals[proposal_id].status = ProposalStatus.CLOSED

        result = engine.evaluate_consensus(proposal_id)

        assert result["status"] == "rejected"

    def test_quorum_not_met(self, engine, registered_agents):
        """Test quorum not met."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # Only one vote (25% < 51% quorum)
        engine.vote(proposal_id, registered_agents[0], "approve")

        result = engine.evaluate_consensus(proposal_id)

        assert result["status"] == "pending"
        assert result["quorum_met"] is False

    def test_trust_weighted_voting(self, engine, registered_agents):
        """Test trust-weighted vote calculation."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # All agents vote
        for agent in registered_agents:
            engine.vote(proposal_id, agent, "approve")

        result = engine.evaluate_consensus(proposal_id)

        assert "trust_weighted" in result
        assert result["trust_weighted"]["total_weight"] > 0

    def test_abstain_votes_excluded(self, engine, registered_agents):
        """Test abstain votes excluded from approval calculation."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        engine.vote(proposal_id, registered_agents[0], "approve")
        engine.vote(proposal_id, registered_agents[1], "approve")
        engine.vote(proposal_id, registered_agents[2], "abstain")
        engine.vote(proposal_id, registered_agents[3], "reject")

        result = engine.evaluate_consensus(proposal_id)

        # Approval ratio should be 2/3 (abstain excluded)
        assert result["vote_breakdown"]["abstain"] == 1


class TestProposalApplication:
    """Test applying approved proposals."""

    def test_apply_approved_proposal(self, engine, registered_agents):
        """Test applying approved proposal."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {
                        "add": {"AnalyticsNode": {"description": "New node"}}
                    },
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # Vote to approve
        for agent in registered_agents:
            engine.vote(proposal_id, agent, "approve")

        engine.proposals[proposal_id].status = ProposalStatus.CLOSED

        result = engine.apply_approved_proposal(proposal_id)

        assert result["success"] is True
        assert "AnalyticsNode" in engine.allowed_node_types

    def test_apply_unapproved_proposal(self, engine, registered_agents):
        """Test applying unapproved proposal fails."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": {"NewNode": {}}},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        # Vote to reject
        for agent in registered_agents:
            engine.vote(proposal_id, agent, "reject")

        engine.proposals[proposal_id].status = ProposalStatus.CLOSED

        result = engine.apply_approved_proposal(proposal_id)

        assert result["success"] is False

    def test_apply_already_applied(self, engine, registered_agents):
        """Test applying already applied proposal."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": {"NewNode": {}}},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        for agent in registered_agents:
            engine.vote(proposal_id, agent, "approve")

        engine.proposals[proposal_id].status = ProposalStatus.CLOSED

        # Apply once
        engine.apply_approved_proposal(proposal_id)

        # Try to apply again
        result = engine.apply_approved_proposal(proposal_id)

        assert result["success"] is False
        assert "already applied" in result["reason"]


class TestStatisticsAndQueries:
    """Test statistics and query methods."""

    def test_get_proposal(self, engine, registered_agents):
        """Test getting proposal by ID."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        proposal = engine.get_proposal(proposal_id)

        assert proposal is not None
        assert proposal["proposal_id"] == proposal_id

    def test_get_nonexistent_proposal(self, engine):
        """Test getting non-existent proposal."""
        proposal = engine.get_proposal("nonexistent")

        assert proposal is None

    def test_get_agent(self, engine, registered_agents):
        """Test getting agent by ID."""
        agent = engine.get_agent(registered_agents[0])

        assert agent is not None
        assert agent["agent_id"] == registered_agents[0]

    def test_get_all_proposals(self, engine, registered_agents):
        """Test getting all proposals."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        engine.propose(proposal_graph, registered_agents[0])
        engine.propose(proposal_graph, registered_agents[1])

        proposals = engine.get_all_proposals()

        assert len(proposals) == 2

    def test_get_allowed_node_types(self, engine):
        """Test getting allowed node types."""
        node_types = engine.get_allowed_node_types()

        assert isinstance(node_types, list)
        assert "ProposalNode" in node_types

    def test_get_statistics(self, engine, registered_agents):
        """Test getting statistics."""
        stats = engine.get_statistics()

        assert "total_agents" in stats
        assert "total_proposals" in stats
        assert stats["total_agents"] == len(registered_agents)


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_registration(self, engine):
        """Test concurrent agent registration."""
        results = []
        errors = []

        def register(agent_id):
            try:
                result = engine.register_agent(agent_id, trust_level=0.8)
                results.append(result)
            except Exception as e:
                errors.append(e)

        import threading

        threads = [
            threading.Thread(target=register, args=(f"agent{i}",)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0

    def test_concurrent_voting(self, engine, registered_agents):
        """Test concurrent voting."""
        proposal_graph = {
            "id": "test",
            "type": "Graph",
            "nodes": [
                {
                    "id": "p1",
                    "type": "ProposalNode",
                    "proposal_content": {"add": "test_change"},
                }
            ],
        }

        proposal_id = engine.propose(proposal_graph, registered_agents[0])

        results = []

        def vote(agent):
            result = engine.vote(proposal_id, agent, "approve")
            results.append(result)

        import threading

        threads = [
            threading.Thread(target=vote, args=(agent,))
            for agent in registered_agents[1:]
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
