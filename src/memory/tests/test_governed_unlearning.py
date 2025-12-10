"""Comprehensive test suite for governed_unlearning.py"""

import sys
import threading
import time
from unittest.mock import Mock

import pytest

sys.path.insert(0, "/mnt/user-data/uploads")

from governed_unlearning import (GovernanceResult, GovernedUnlearning,
                                 IRProposal, ProposalStatus,
                                 UnlearningAuditLogger, UnlearningMethod,
                                 UnlearningMetrics, UnlearningTask,
                                 UrgencyLevel)

# ============================================================
# TEST DATA CLASSES
# ============================================================


class TestIRProposal:
    def test_initialization(self):
        proposal = IRProposal(
            proposal_id="prop-123", ir_content={"pattern": "test"}, proposer_id="user-1"
        )
        assert proposal.proposal_id == "prop-123"
        assert proposal.urgency == UrgencyLevel.NORMAL
        assert isinstance(proposal.timestamp, float)

    def test_to_dict(self):
        proposal = IRProposal("prop-123", {"data": "test"}, "user-1")
        result = proposal.to_dict()
        assert result["proposal_id"] == "prop-123"
        assert result["urgency"] == "normal"


class TestGovernanceResult:
    def test_initialization(self):
        result = GovernanceResult(
            proposal_id="prop-123",
            status=ProposalStatus.APPROVED,
            details={"reason": "Valid"},
        )
        assert result.proposal_id == "prop-123"
        assert result.status == ProposalStatus.APPROVED


class TestUnlearningTask:
    def test_initialization(self):
        proposal = IRProposal("prop-1", {}, "user-1")
        task = UnlearningTask(
            task_id="task-1",
            proposal=proposal,
            method=UnlearningMethod.GRADIENT_SURGERY,
            pattern="sensitive_data",
            affected_packs=["pack1"],
        )
        assert task.task_id == "task-1"
        assert task.status == ProposalStatus.PENDING

    def test_get_duration(self):
        proposal = IRProposal("prop-1", {}, "user-1")
        task = UnlearningTask(
            "task-1", proposal, UnlearningMethod.EXACT_REMOVAL, "test", []
        )
        task.started_at = time.time()
        task.completed_at = task.started_at + 5.0
        duration = task.get_duration()
        assert 4.9 <= duration <= 5.1


class TestUnlearningMetrics:
    def test_get_success_rate(self):
        metrics = UnlearningMetrics(completed_tasks=8, failed_tasks=2)
        assert metrics.get_success_rate() == 0.8


# ============================================================
# TEST AUDIT LOGGER
# ============================================================


class TestUnlearningAuditLogger:
    def test_initialization(self):
        logger = UnlearningAuditLogger()
        assert len(logger.audit_trail) == 0

    def test_log_proposal(self):
        logger = UnlearningAuditLogger()
        proposal = IRProposal("prop-1", {"test": "data"}, "user-1")
        result = GovernanceResult("prop-1", ProposalStatus.APPROVED, {})
        logger.log_proposal(proposal, result)
        assert len(logger.audit_trail) == 1


# ============================================================
# TEST GOVERNED UNLEARNING
# ============================================================


class TestGovernedUnlearning:
    @pytest.fixture
    def mock_memory(self):
        memory = Mock()
        memory.unlearning_engine = Mock()
        memory.unlearning_engine.gradient_surgery = Mock()
        memory.zk_prover = Mock()
        memory.zk_prover.generate_unlearning_proof = Mock(
            return_value={"proof": "test"}
        )
        return memory

    def test_initialization(self, mock_memory):
        system = GovernedUnlearning(mock_memory)
        assert system.memory == mock_memory
        assert isinstance(system.metrics, UnlearningMetrics)
        system.shutdown()

    def test_propose_unlearning(self, mock_memory):
        system = GovernedUnlearning(mock_memory)
        proposal_id = system.submit_ir_proposal(
            ir_content={"pattern": "sensitive_data"}, proposer_id="user-1"
        )
        assert isinstance(proposal_id, str)
        assert len(proposal_id) > 0
        system.shutdown()

    def test_get_metrics(self, mock_memory):
        system = GovernedUnlearning(mock_memory)
        metrics = system.get_unlearning_metrics()
        assert isinstance(metrics, UnlearningMetrics)
        system.shutdown()

    def test_multiple_proposals(self, mock_memory):
        system = GovernedUnlearning(mock_memory)

        for i in range(5):
            system.submit_ir_proposal(
                ir_content={"pattern": f"data_{i}"}, proposer_id=f"user-{i}"
            )

        metrics = system.get_unlearning_metrics()
        assert metrics.total_requests == 5
        system.shutdown()

    def test_thread_safety(self, mock_memory):
        system = GovernedUnlearning(mock_memory)

        def propose_requests():
            for i in range(3):
                system.submit_ir_proposal(
                    ir_content={"pattern": f"data_{i}"}, proposer_id="user-1"
                )

        threads = [threading.Thread(target=propose_requests) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = system.get_unlearning_metrics()
        assert metrics.total_requests == 9
        system.shutdown()


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    def test_complete_workflow(self):
        memory = Mock()
        memory.unlearning_engine = Mock()
        memory.unlearning_engine.gradient_surgery = Mock()

        system = GovernedUnlearning(memory)
        proposal_id = system.submit_ir_proposal(
            ir_content={"pattern": "sensitive_data"}, proposer_id="user-1"
        )

        assert isinstance(proposal_id, str)
        assert len(proposal_id) > 0

        metrics = system.get_unlearning_metrics()
        assert metrics.total_requests > 0

        system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
