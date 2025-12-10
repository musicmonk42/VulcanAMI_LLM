"""
Tests for speculative_helpers.py
"""

from speculative_helpers import (KL_THRESHOLD, LowRankDraftTransformer,
                                 SpeculativeStats,
                                 speculative_sampling_and_verify,
                                 speculative_sampling_and_verify_async)
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/user-data/uploads")


class MockTransformer(nn.Module):
    """Mock transformer for testing."""

    def __init__(self, vocab_size=1000, hidden_size=768):
        super().__init__()
        self.config = type(
            "obj", (object,), {"vocab_size": vocab_size, "hidden_size": hidden_size}
        )()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, output_hidden_states=False):
        embeds = self.embedding(input_ids)
        logits = self.lm_head(embeds)

        if output_hidden_states:
            return type(
                "obj", (object,), {"logits": logits, "hidden_states": (embeds,)}
            )()
        return type("obj", (object,), {"logits": logits})()


class TestSpeculativeStats:
    """Test SpeculativeStats dataclass."""

    def test_initialization(self):
        stats = SpeculativeStats()
        assert stats.drafted == 0
        assert stats.accepted == 0
        assert stats.total_kl == 0.0

    def test_efficiency_gain(self):
        stats = SpeculativeStats(drafted=100, accepted=75)
        assert stats.efficiency_gain == 0.75

    def test_efficiency_gain_zero_drafted(self):
        stats = SpeculativeStats(drafted=0, accepted=0)
        assert stats.efficiency_gain == 0.0

    def test_acceptance_rate(self):
        stats = SpeculativeStats(drafted=100, accepted=80)
        assert stats.acceptance_rate == 0.80

    def test_avg_kl(self):
        stats = SpeculativeStats(total_kl=10.0, kl_divergences=[1.0, 2.0, 3.0, 4.0])
        assert stats.avg_kl == 2.5


class TestLowRankDraftTransformer:
    """Test LowRankDraftTransformer."""

    def test_initialization(self):
        parent = MockTransformer(vocab_size=1000, hidden_size=768)
        draft = LowRankDraftTransformer(parent, rank=16, shrink_factor=0.5)

        assert draft.rank == 16
        assert draft.shrink == 0.5
        assert draft.vocab_size == 1000

    def test_initialization_without_config(self):
        parent = nn.Linear(10, 10)  # No config attribute

        with pytest.raises(ValueError, match="Parent model must have a 'config'"):
            LowRankDraftTransformer(parent)

    def test_encode(self):
        parent = MockTransformer(vocab_size=1000, hidden_size=768)
        draft = LowRankDraftTransformer(parent, rank=16)

        input_ids = torch.randint(0, 1000, (1, 10))
        compressed = draft.encode(input_ids)

        assert compressed.shape == (1, 16)

    def test_get_logits(self):
        parent = MockTransformer(vocab_size=1000, hidden_size=768)
        draft = LowRankDraftTransformer(parent, rank=16)

        low_rank_hidden = torch.randn(1, 16)
        input_ids = torch.randint(0, 1000, (1, 10))
        logits = draft.get_logits(low_rank_hidden, input_ids)

        assert logits.shape == (1, 1000)


class TestSpeculativeSampling:
    """Test speculative sampling and verification."""

    def test_speculative_sampling_basic(self):
        torch.manual_seed(42)

        parent = MockTransformer(vocab_size=100, hidden_size=64)
        draft = LowRankDraftTransformer(parent, rank=8, shrink_factor=0.5)

        input_ids = torch.randint(0, 100, (2, 5))  # Batch size 2, seq len 5
        stats = SpeculativeStats()

        result, stats = speculative_sampling_and_verify(
            parent,
            draft,
            input_ids,
            stats,
            max_lookahead=3,
            kl_threshold=KL_THRESHOLD,
            temperature=1.0,
        )

        # Check result shape (should have added at least 1 token)
        assert result.shape[0] == 2  # Batch size preserved
        assert result.shape[1] > input_ids.shape[1]  # Sequence length increased

        # Check stats were updated
        assert stats.drafted > 0
        assert stats.total_steps > 0

    def test_speculative_sampling_low_entropy_fallback(self):
        torch.manual_seed(42)

        parent = MockTransformer(vocab_size=100, hidden_size=64)
        draft = LowRankDraftTransformer(
            parent, rank=8, entropy_threshold=100.0
        )  # Very high threshold

        input_ids = torch.randint(0, 100, (1, 5))
        stats = SpeculativeStats()

        result, stats = speculative_sampling_and_verify(
            parent,
            draft,
            input_ids,
            stats,
            max_lookahead=3,
            entropy_threshold=100.0,  # Trigger low entropy fallback
        )

        # Should still generate tokens
        assert result.shape[1] > input_ids.shape[1]
        assert stats.rejection_reason == "LowEntropy"

    @pytest.mark.asyncio
    async def test_async_speculative_sampling(self):
        torch.manual_seed(42)

        parent = MockTransformer(vocab_size=100, hidden_size=64)
        draft = LowRankDraftTransformer(parent, rank=8)

        input_ids = torch.randint(0, 100, (1, 5))
        stats = SpeculativeStats()

        result, stats = await speculative_sampling_and_verify_async(
            parent, draft, input_ids, stats, max_lookahead=2
        )

        assert result.shape[1] > input_ids.shape[1]
        assert stats.drafted > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
