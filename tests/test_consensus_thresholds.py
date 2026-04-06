"""Tests for consolidated consensus thresholds (B3)."""
import pytest
from src.protocols.consensus import (
    ConsensusEngine,
    DEFAULT_QUORUM_RATIO,
    DEFAULT_APPROVAL_THRESHOLD,
)


class TestProductionDefaults:
    def test_default_quorum_is_production(self):
        engine = ConsensusEngine()
        assert engine._quorum_ratio == DEFAULT_QUORUM_RATIO == 0.51

    def test_default_approval_is_production(self):
        engine = ConsensusEngine()
        assert engine._approval_threshold == DEFAULT_APPROVAL_THRESHOLD == 0.66

    def test_quorum_override(self):
        engine = ConsensusEngine(quorum_ratio=0.8)
        assert engine._quorum_ratio == 0.8

    def test_approval_override(self):
        engine = ConsensusEngine(approval_threshold=0.9)
        assert engine._approval_threshold == 0.9


class TestThreeNodeCluster:
    """With 3 agents at equal weight, validate Byzantine resilience."""

    def setup_method(self):
        self.engine = ConsensusEngine()
        for i in range(3):
            self.engine.register_agent(f"a{i}", 1.0)

    def test_one_vote_below_quorum(self):
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a0", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "pending"

    def test_two_approvals_pass(self):
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a0", "approve")
        self.engine.vote(pid, "a1", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "approved"

    def test_two_rejections_fail(self):
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a0", "reject")
        self.engine.vote(pid, "a1", "reject")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "rejected"

    def test_split_vote_rejected(self):
        """1 approve + 1 reject = 50% confidence, below 66% threshold."""
        pid = self.engine.propose({"action": "test"})
        self.engine.vote(pid, "a0", "approve")
        self.engine.vote(pid, "a1", "reject")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "rejected"

    def test_unanimous_approval(self):
        pid = self.engine.propose({"action": "test"})
        for i in range(3):
            self.engine.vote(pid, f"a{i}", "approve")
        result = self.engine.evaluate(pid)
        assert result["verdict"] == "approved"
        assert result["confidence"] == 1.0
