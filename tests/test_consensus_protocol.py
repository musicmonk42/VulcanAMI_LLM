"""Tests for canonical ConsensusProtocol."""
import pytest
from src.protocols.consensus import ConsensusEngine, ProposalStatus


class TestConsensusEngine:
    def setup_method(self):
        self.engine = ConsensusEngine(quorum_ratio=0.5)

    def test_register_and_propose(self):
        self.engine.register_agent("a1", 1.0)
        pid = self.engine.propose({"action": "test"})
        assert pid is not None
        prop = self.engine.get_proposal(pid)
        assert prop.status == ProposalStatus.OPEN

    def test_vote_and_approve(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.register_agent("a2", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a1", "approve")
        self.engine.vote(pid, "a2", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "approved"
        assert result["confidence"] == 1.0

    def test_vote_and_reject(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.register_agent("a2", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a1", "reject")
        self.engine.vote(pid, "a2", "reject")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "rejected"

    def test_trust_weighted_voting(self):
        self.engine.register_agent("a1", 10.0)
        self.engine.register_agent("a2", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a1", "approve")
        self.engine.vote(pid, "a2", "reject")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "approved"

    def test_elect_leader(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.register_agent("a2", 5.0)
        leader = self.engine.elect_leader()
        assert leader == "a2"

    def test_shutdown_clears_state(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.shutdown()
        assert self.engine.elect_leader() is None
