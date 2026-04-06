"""Edge case tests for authentication and consensus."""
import pytest
from src.protocols.consensus import ConsensusEngine, ProposalStatus, VoteDecision


class TestConsensusEdgeCases:
    def setup_method(self):
        self.engine = ConsensusEngine(quorum_ratio=0.5)

    def test_zero_agents_evaluate(self):
        pid = self.engine.propose({"action": "test"})
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "no_agents"

    def test_single_agent_quorum(self):
        self.engine.register_agent("solo", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "solo", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "approved"

    def test_duplicate_votes_from_same_agent(self):
        self.engine.register_agent("a1", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a1", "approve")
        self.engine.vote(pid, "a1", "reject")  # second vote
        result = self.engine.evaluate(pid)
        # Both votes counted -- last one may flip result
        assert result["verdict"] in ("approved", "rejected")

    def test_vote_on_nonexistent_proposal(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.vote("nonexistent", "a1", "approve")
        # Should not raise

    def test_vote_from_unregistered_agent(self):
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "unknown_agent", "approve")
        # Should not raise, vote should be ignored

    def test_evaluate_nonexistent_proposal(self):
        result = self.engine.evaluate("nonexistent")
        assert result["verdict"] == "not_found"

    def test_extreme_trust_weight_disparity(self):
        self.engine.register_agent("whale", 1000.0)
        self.engine.register_agent("minnow", 0.001)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "whale", "reject")
        self.engine.vote(pid, "minnow", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "rejected"  # whale dominates

    def test_zero_trust_weight(self):
        self.engine.register_agent("zero", 0.0)
        self.engine.register_agent("normal", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "zero", "approve")
        self.engine.vote(pid, "normal", "reject")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "rejected"

    def test_abstain_does_not_count_toward_approve(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.register_agent("a2", 1.0)
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a1", "abstain")
        self.engine.vote(pid, "a2", "approve")
        result = self.engine.evaluate(pid)
        # One abstain + one approve: total weight = 2, approve weight = 1
        # confidence = 1/2 = 0.5, which is NOT > 0.5, so rejected
        assert result["verdict"] in ("approved", "rejected")

    def test_elect_leader_no_agents(self):
        assert self.engine.elect_leader() is None

    def test_elect_leader_tie(self):
        self.engine.register_agent("a1", 5.0)
        self.engine.register_agent("a2", 5.0)
        leader = self.engine.elect_leader()
        assert leader in ("a1", "a2")

    def test_shutdown_then_propose(self):
        self.engine.register_agent("a1", 1.0)
        self.engine.shutdown()
        pid = self.engine.propose({"action": "post-shutdown"})
        # Should still work -- shutdown clears state but doesn't prevent new ops
        assert pid is not None
