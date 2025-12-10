"""
Tests for token_consensus_adapter.py
"""

from token_consensus_adapter import (ConsensusAdapterConfig, ConsensusProposal,
                                     TokenConsensusAdapter)
import asyncio
import sys

import pytest

sys.path.insert(0, "/mnt/user-data/uploads")


class MockEngine:
    """Mock consensus engine for testing."""

    def __init__(self, should_approve=True, is_async=True, delay=0.0):
        self.should_approve = should_approve
        self.is_async = is_async
        self.delay = delay
        self.calls = 0

    async def approve(self, proposal):
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.calls += 1
        return self.should_approve


class TestConsensusAdapterConfig:
    """Test ConsensusAdapterConfig."""

    def test_default_config(self):
        config = ConsensusAdapterConfig()
        assert config.fail_closed is False
        assert config.timeout_seconds == 2.0
        assert config.max_retries == 3

    def test_custom_config(self):
        config = ConsensusAdapterConfig(
            fail_closed=True, timeout_seconds=5.0, max_retries=5
        )
        assert config.fail_closed is True
        assert config.timeout_seconds == 5.0
        assert config.max_retries == 5


class TestConsensusProposal:
    """Test ConsensusProposal dataclass."""

    def test_basic_proposal(self):
        proposal = ConsensusProposal(type="token_emission", token="test", position=0)
        assert proposal.type == "token_emission"
        assert proposal.token == "test"
        assert proposal.position == 0


class TestTokenConsensusAdapter:
    """Test TokenConsensusAdapter functionality."""

    def test_initialization(self):
        adapter = TokenConsensusAdapter()
        assert adapter.config is not None
        assert adapter.max_retries == 3
        assert adapter.fail_closed is False

    def test_initialization_with_config(self):
        config = ConsensusAdapterConfig(fail_closed=True, max_retries=5)
        adapter = TokenConsensusAdapter(config=config)
        assert adapter.fail_closed is True
        assert adapter.max_retries == 5

    @pytest.mark.asyncio
    async def test_approve_success(self):
        engine = MockEngine(should_approve=True)
        adapter = TokenConsensusAdapter(engine=engine)

        proposal = {"type": "token_emission", "token": "test", "position": 0}

        result = await adapter.approve(proposal)

        assert result is True
        assert engine.calls == 1
        assert adapter._calls == 1
        assert adapter._successes == 1

    @pytest.mark.asyncio
    async def test_approve_rejection(self):
        engine = MockEngine(should_approve=False)
        adapter = TokenConsensusAdapter(engine=engine)

        proposal = {"type": "token_emission", "token": "test", "position": 0}

        result = await adapter.approve(proposal)

        assert result is False
        assert engine.calls == 1
        assert adapter._calls == 1
        assert adapter._successes == 0

    @pytest.mark.asyncio
    async def test_approve_no_engine_fail_open(self, encoding="utf-8"):
        adapter = TokenConsensusAdapter(
            engine=None, config=ConsensusAdapterConfig(fail_closed=False)
        )

        proposal = {"type": "token_emission", "token": "test", "position": 0}

        result = await adapter.approve(proposal)
        assert result is True  # Fail open

    @pytest.mark.asyncio
    async def test_approve_no_engine_fail_closed(self):
        adapter = TokenConsensusAdapter(
            engine=None, config=ConsensusAdapterConfig(fail_closed=True)
        )

        proposal = {"type": "token_emission", "token": "test", "position": 0}

        result = await adapter.approve(proposal)
        assert result is False  # Fail closed

    @pytest.mark.asyncio
    async def test_validation_error_missing_field(self):
        adapter = TokenConsensusAdapter(config=ConsensusAdapterConfig(fail_closed=True))

        # Missing 'position' field
        proposal = {"type": "token_emission", "token": "test"}

        result = await adapter.approve(proposal)
        assert result is False  # Fail closed on validation error

    @pytest.mark.asyncio
    async def test_validation_error_empty_token(self):
        adapter = TokenConsensusAdapter(config=ConsensusAdapterConfig(fail_closed=True))

        proposal = {
            "type": "token_emission",
            "token": "",  # Empty token
            "position": 0,
        }

        result = await adapter.approve(proposal)
        assert result is False  # Validation should fail

    @pytest.mark.asyncio
    async def test_validation_error_negative_position(self):
        adapter = TokenConsensusAdapter(config=ConsensusAdapterConfig(fail_closed=True))

        proposal = {
            "type": "token_emission",
            "token": "test",
            "position": -1,  # Negative position
        }

        result = await adapter.approve(proposal)
        assert result is False  # Validation should fail

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        # Engine that always times out
        engine = MockEngine(delay=10.0)  # Longer than timeout
        adapter = TokenConsensusAdapter(
            engine=engine,
            config=ConsensusAdapterConfig(
                timeout_seconds=0.01, max_retries=2, fail_closed=False
            ),
        )

        proposal = {"type": "token_emission", "token": "test", "position": 0}

        result = await adapter.approve(proposal)

        # Should fail open after retries
        assert result is True
        assert adapter._calls == 1

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        engine = MockEngine(should_approve=True)
        adapter = TokenConsensusAdapter(engine=engine)

        proposals = [
            {"type": "token_emission", "token": f"test{i}", "position": i}
            for i in range(5):
        ]

        for proposal in proposals:
            await adapter.approve(proposal)

        assert adapter._calls == 5
        assert adapter._successes == 5

    @pytest.mark.asyncio
    async def test_chosen_index_validation(self):
        adapter = TokenConsensusAdapter()

        proposal = {
            "type": "token_emission",
            "token": "test",
            "position": 0,
            "chosen_index": -1,  # Invalid negative index
        }

        await adapter.approve(proposal)
        # Should fail validation
        assert adapter._calls == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
