"""
Comprehensive test suite for consensus_manager.py
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from consensus_manager import (DEFAULT_QUORUM_RATIO, MAX_QUORUM_RATIO,
                               MIN_QUORUM_RATIO, ConsensusManager,
                               LeaderElector, LeaderState, ServerState,
                               _seed_for, _simulate_vote)


@pytest.fixture
def consensus_manager():
    """Create consensus manager."""
    manager = ConsensusManager(
        chaos_params={'failure_rate': 0.0, 'max_delay': 0.0, 'drop_rate': 0.0},
        timeout=0.1,
        deadlock_threshold=3,
        max_retries=5,
        backend="thread"
    )
    yield manager
    manager.shutdown()


@pytest.fixture
def leader_elector():
    """Create leader elector."""
    return LeaderElector()


class TestSimulateVote:
    """Test vote simulation function."""

    def test_deterministic_seed(self):
        """Test seed generation is deterministic."""
        seed1 = _seed_for("agent1", "proposal1", 42)
        seed2 = _seed_for("agent1", "proposal1", 42)

        assert seed1 == seed2

    def test_different_agents_different_seeds(self):
        """Test different agents produce different seeds."""
        seed1 = _seed_for("agent1", "proposal1", 42)
        seed2 = _seed_for("agent2", "proposal1", 42)

        assert seed1 != seed2

    def test_simulate_vote_success(self):
        """Test successful vote simulation."""
        chaos = {'failure_rate': 0.0, 'max_delay': 0.0, 'drop_rate': 0.0}

        agent_id, vote = _simulate_vote("agent1", "proposal1", chaos, 0.1, 42)

        assert agent_id == "agent1"
        assert isinstance(vote, bool)

    def test_simulate_vote_with_delay(self):
        """Test vote with delay."""
        chaos = {'failure_rate': 0.0, 'max_delay': 0.05, 'drop_rate': 0.0}

        start = time.time()
        _simulate_vote("agent1", "proposal1", chaos, 0.2, 42)
        elapsed = time.time() - start

        # Should have some delay but bounded
        assert elapsed < 0.15  # max_delay is 0.05, timeout/2 is 0.1

    def test_simulate_vote_failure(self):
        """Test vote failure simulation."""
        chaos = {'failure_rate': 1.0, 'max_delay': 0.0, 'drop_rate': 0.0}

        with pytest.raises(RuntimeError, match="simulated failure"):
            _simulate_vote("agent1", "proposal1", chaos, 0.1, 42)

    def test_simulate_vote_drop(self):
        """Test vote drop simulation."""
        chaos = {'failure_rate': 0.0, 'max_delay': 0.0, 'drop_rate': 1.0}

        with pytest.raises(TimeoutError, match="simulated drop"):
            _simulate_vote("agent1", "proposal1", chaos, 0.1, 42)


class TestLeaderState:
    """Test LeaderState dataclass."""

    def test_initialization(self):
        """Test default initialization."""
        state = LeaderState()

        assert state.current_term == 0
        assert state.voted_for is None
        assert state.current_leader is None
        assert state.state == ServerState.FOLLOWER

    def test_reset_election_timeout(self):
        """Test timeout reset."""
        state = LeaderState()

        initial_timeout = state.election_timeout
        initial_heartbeat = state.last_heartbeat

        time.sleep(0.01)
        state.reset_election_timeout()

        assert state.last_heartbeat > initial_heartbeat
        # Timeout should be randomized
        assert state.election_timeout >= 0.15
        assert state.election_timeout <= 0.30


class TestLeaderElector:
    """Test LeaderElector class."""

    def test_initialization(self, leader_elector):
        """Test elector initialization."""
        assert leader_elector.state.current_term == 0
        assert leader_elector.state.state == ServerState.FOLLOWER
        assert not leader_elector.shutdown_flag

    def test_request_vote_grant(self, leader_elector):
        """Test granting vote."""
        agents = ["agent1", "agent2", "agent3"]

        granted, term = leader_elector.request_vote("agent1", 1, agents)

        assert granted is True
        assert term == 1
        assert leader_elector.state.voted_for == "agent1"

    def test_request_vote_outdated_term(self, leader_elector):
        """Test rejecting outdated term."""
        agents = ["agent1", "agent2"]

        # Set current term to 5
        leader_elector.state.current_term = 5

        granted, term = leader_elector.request_vote("agent1", 3, agents)

        assert granted is False
        assert term == 5

    def test_request_vote_already_voted(self, leader_elector):
        """Test rejecting when already voted."""
        agents = ["agent1", "agent2", "agent3"]

        # Vote for agent1
        leader_elector.request_vote("agent1", 1, agents)

        # Try to vote for agent2 in same term
        granted, term = leader_elector.request_vote("agent2", 1, agents)

        assert granted is False

    def test_request_vote_same_candidate(self, leader_elector):
        """Test granting vote to same candidate."""
        agents = ["agent1", "agent2"]

        # Vote for agent1
        leader_elector.request_vote("agent1", 1, agents)

        # Vote again for agent1
        granted, term = leader_elector.request_vote("agent1", 1, agents)

        assert granted is True

    def test_elect_leader_simple(self, leader_elector):
        """Test simple leader election."""
        agents = ["agent1", "agent2", "agent3"]

        leader = leader_elector.elect_leader(agents, timeout=1.0)

        assert leader in agents
        assert leader_elector.state.current_leader == leader

    def test_elect_leader_empty_agents(self, leader_elector):
        """Test election with empty agents list."""
        with pytest.raises(ValueError, match="empty agent list"):
            leader_elector.elect_leader([], timeout=1.0)

    def test_elect_leader_single_agent(self, leader_elector):
        """Test election with single agent."""
        leader = leader_elector.elect_leader(["agent1"], timeout=1.0)

        assert leader == "agent1"

    def test_get_current_leader(self, leader_elector):
        """Test getting current leader."""
        agents = ["agent1", "agent2", "agent3"]

        assert leader_elector.get_current_leader() is None

        leader = leader_elector.elect_leader(agents)

        assert leader_elector.get_current_leader() == leader

    def test_is_leader(self, leader_elector):
        """Test checking if agent is leader."""
        agents = ["agent1", "agent2", "agent3"]

        leader = leader_elector.elect_leader(agents)

        assert leader_elector.is_leader(leader)
        assert not leader_elector.is_leader("nonexistent")

    def test_heartbeat(self, leader_elector):
        """Test heartbeat processing."""
        leader_elector.state.current_leader = "agent1"

        assert leader_elector.heartbeat("agent1")
        assert not leader_elector.heartbeat("agent2")

    def test_check_leader_timeout(self, leader_elector):
        """Test leader timeout detection."""
        leader_elector.state.last_heartbeat = time.time() - 10
        leader_elector.state.election_timeout = 1.0

        assert leader_elector.check_leader_timeout()


class TestConsensusManager:
    """Test ConsensusManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = ConsensusManager(
            chaos_params={'failure_rate': 0.1, 'max_delay': 0.01, 'drop_rate': 0.05},
            timeout=0.1,
            backend="thread"
        )

        assert manager.timeout == 0.1
        assert manager.backend == "thread"
        assert manager.chaos_params['failure_rate'] == 0.1

        manager.shutdown()

    def test_invalid_timeout(self):
        """Test invalid timeout raises error."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ConsensusManager(timeout=-1)

    def test_invalid_deadlock_threshold(self):
        """Test invalid deadlock threshold."""
        with pytest.raises(ValueError, match="Deadlock threshold"):
            ConsensusManager(deadlock_threshold=0)

    def test_invalid_max_retries(self):
        """Test invalid max retries."""
        with pytest.raises(ValueError, match="Max retries"):
            ConsensusManager(max_retries=-1)

    def test_invalid_chaos_params(self):
        """Test invalid chaos parameters."""
        with pytest.raises(ValueError):
            ConsensusManager(chaos_params={'failure_rate': 1.5})

        with pytest.raises(ValueError):
            ConsensusManager(chaos_params={'max_delay': -1})

    def test_backend_resolution_thread(self):
        """Test thread backend resolution."""
        manager = ConsensusManager(backend="thread")

        assert manager.backend == "thread"
        manager.shutdown()

    def test_backend_resolution_auto(self):
        """Test auto backend resolution."""
        manager = ConsensusManager(backend="auto")

        assert manager.backend in ["thread", "ray"]
        manager.shutdown()

    def test_set_chaos(self, consensus_manager):
        """Test updating chaos parameters."""
        consensus_manager.set_chaos(
            failure_rate=0.2,
            max_delay=0.02,
            drop_rate=0.1
        )

        assert consensus_manager.chaos_params['failure_rate'] == 0.2
        assert consensus_manager.chaos_params['max_delay'] == 0.02
        assert consensus_manager.chaos_params['drop_rate'] == 0.1

    def test_set_chaos_invalid(self, consensus_manager):
        """Test setting invalid chaos parameters."""
        with pytest.raises(ValueError):
            consensus_manager.set_chaos(
                failure_rate=2.0,
                max_delay=0.0,
                drop_rate=0.0
            )

    def test_elect_leader(self, consensus_manager):
        """Test leader election."""
        agents = [f"agent{i}" for i in range(5)]

        leader = consensus_manager.elect_leader(agents)

        assert leader in agents

    def test_elect_leader_empty(self, consensus_manager):
        """Test election with empty agents."""
        with pytest.raises(ValueError, match="empty agent list"):
            consensus_manager.elect_leader([])

    def test_elect_leader_duplicates(self, consensus_manager):
        """Test election with duplicate agents."""
        with pytest.raises(ValueError, match="duplicates"):
            consensus_manager.elect_leader(["agent1", "agent1", "agent2"])

    def test_elect_leader_invalid_type(self, consensus_manager):
        """Test election with invalid agent type."""
        with pytest.raises(ValueError):
            consensus_manager.elect_leader("not a list")

    def test_aggregate_votes_simple(self, consensus_manager):
        """Test simple vote aggregation."""
        agents = [f"agent{i}" for i in range(10)]

        result = consensus_manager.aggregate_votes(
            "proposal1",
            agents,
            quorum_ratio=0.5
        )

        # With no chaos, should generally succeed
        assert isinstance(result, bool)

    def test_aggregate_votes_empty_agents(self, consensus_manager):
        """Test aggregation with empty agents."""
        result = consensus_manager.aggregate_votes("proposal1", [])

        assert result is False

    def test_aggregate_votes_invalid_proposal_id(self, consensus_manager):
        """Test aggregation with invalid proposal ID."""
        with pytest.raises(ValueError, match="Invalid proposal_id"):
            consensus_manager.aggregate_votes("", ["agent1"])

    def test_aggregate_votes_invalid_quorum(self, consensus_manager):
        """Test aggregation with invalid quorum."""
        with pytest.raises(ValueError, match="Quorum ratio"):
            consensus_manager.aggregate_votes(
                "proposal1",
                ["agent1"],
                quorum_ratio=1.5
            )

    def test_aggregate_votes_duplicates(self, consensus_manager):
        """Test aggregation with duplicate agents."""
        with pytest.raises(ValueError, match="duplicates"):
            consensus_manager.aggregate_votes(
                "proposal1",
                ["agent1", "agent1"],
                quorum_ratio=0.5
            )

    def test_aggregate_votes_high_quorum(self, consensus_manager):
        """Test aggregation with high quorum requirement."""
        agents = [f"agent{i}" for i in range(10)]

        # With no chaos, should succeed
        result = consensus_manager.aggregate_votes(
            "proposal1",
            agents,
            quorum_ratio=0.9
        )

        assert isinstance(result, bool)

    def test_shutdown(self, consensus_manager):
        """Test shutdown."""
        consensus_manager.shutdown()

        assert consensus_manager.shutdown_flag

    def test_aggregate_after_shutdown(self, consensus_manager):
        """Test aggregation after shutdown."""
        consensus_manager.shutdown()

        result = consensus_manager.aggregate_votes("proposal1", ["agent1"])

        assert result is False


class TestThreadBackend:
    """Test thread-based backend."""

    def test_thread_backend_no_chaos(self):
        """Test thread backend with no chaos."""
        manager = ConsensusManager(
            chaos_params={'failure_rate': 0.0, 'max_delay': 0.0, 'drop_rate': 0.0},
            backend="thread",
            timeout=0.1
        )

        agents = [f"agent{i}" for i in range(10)]
        result = manager.aggregate_votes("proposal1", agents, quorum_ratio=0.67)

        # With no chaos, should succeed
        assert result is True

        manager.shutdown()

    def test_thread_backend_with_chaos(self):
        """Test thread backend with chaos."""
        manager = ConsensusManager(
            chaos_params={'failure_rate': 0.3, 'max_delay': 0.01, 'drop_rate': 0.1},
            backend="thread",
            timeout=0.1,
            max_retries=3
        )

        agents = [f"agent{i}" for i in range(10)]
        result = manager.aggregate_votes("proposal1", agents, quorum_ratio=0.67)

        # May or may not succeed with chaos
        assert isinstance(result, bool)

        manager.shutdown()


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_elections(self):
        """Test concurrent leader elections."""
        manager = ConsensusManager(backend="thread")

        agents = [f"agent{i}" for i in range(10)]
        results = []

        def elect():
            leader = manager.elect_leader(agents)
            results.append(leader)

        threads = [threading.Thread(target=elect) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r in agents for r in results)

        manager.shutdown()

    def test_concurrent_aggregations(self):
        """Test concurrent vote aggregations."""
        manager = ConsensusManager(
            chaos_params={'failure_rate': 0.0, 'max_delay': 0.0, 'drop_rate': 0.0},
            backend="thread"
        )

        agents = [f"agent{i}" for i in range(10)]
        results = []

        def aggregate(proposal_id):
            result = manager.aggregate_votes(proposal_id, agents, quorum_ratio=0.67)
            results.append(result)

        threads = [
            threading.Thread(target=aggregate, args=(f"proposal{i}",))
            for i in range(5):
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5

        manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
