"""
Tests for parallel_candidate_scorer.py
"""

import sys

import pytest
import torch

sys.path.insert(0, "/mnt/user-data/uploads")

from parallel_candidate_scorer import (CacheConfig, DeviceConfig, DeviceType,
                                       EmbeddingArchitecture, EmbeddingConfig,
                                       PenaltyConfig, PerformanceConfig,
                                       ScoringConfig, ScoringStrategy,
                                       VulcanCandidateScorer,
                                       VulcanScorerConfig, get_global_scorer,
                                       score_candidate_sync)


class TestDeviceConfig:
    """Test DeviceConfig."""

    def test_default_config(self):
        config = DeviceConfig()
        assert config.device_type == DeviceType.CPU

    def test_get_device_cpu(self):
        config = DeviceConfig(device_type=DeviceType.CPU)
        device = config.get_device()
        assert device.type == "cpu"

    def test_get_all_devices(self):
        config = DeviceConfig()
        devices = config.get_all_devices()
        assert len(devices) == 1


class TestEmbeddingConfig:
    """Test EmbeddingConfig."""

    def test_default_config(self):
        config = EmbeddingConfig()
        assert config.architecture == EmbeddingArchitecture.HYBRID_LSTM_ATTENTION
        assert config.embedding_dim == 384
        assert config.hidden_size == 256

    def test_custom_config(self):
        config = EmbeddingConfig(
            architecture=EmbeddingArchitecture.LSTM, embedding_dim=512, hidden_size=256
        )
        assert config.architecture == EmbeddingArchitecture.LSTM
        assert config.embedding_dim == 512


class TestScoringConfig:
    """Test ScoringConfig."""

    def test_default_config(self):
        config = ScoringConfig()
        assert config.strategy == ScoringStrategy.HYBRID


class TestPenaltyConfig:
    """Test PenaltyConfig."""

    def test_default_config(self):
        config = PenaltyConfig()
        assert config.length_penalty_factor == 0.01
        assert config.diversity_penalty_factor == 0.5
        assert config.repetition_penalty_factor == 0.3


class TestCacheConfig:
    """Test CacheConfig."""

    def test_default_config(self):
        config = CacheConfig()
        assert config.enable_cache is True
        assert config.max_cache_size == 10000
        assert config.cache_ttl_seconds == 3600.0


class TestPerformanceConfig:
    """Test PerformanceConfig."""

    def test_default_config(self):
        config = PerformanceConfig()
        assert config.max_workers == 4
        assert config.batch_size == 32


class TestVulcanScorerConfig:
    """Test VulcanScorerConfig."""

    def test_default_config(self):
        config = VulcanScorerConfig()
        assert config.device is not None
        assert config.embedding is not None
        assert config.scoring is not None
        assert config.penalty is not None
        assert config.cache is not None
        assert config.performance is not None


class TestVulcanCandidateScorer:
    """Test VulcanCandidateScorer."""

    def test_initialization(self):
        scorer = VulcanCandidateScorer()
        assert scorer.config is not None
        assert scorer.device is not None
        assert scorer.embedder is not None

    def test_initialization_with_config(self):
        config = VulcanScorerConfig()
        config.embedding.embedding_dim = 256
        scorer = VulcanCandidateScorer(config)
        assert scorer.config.embedding.embedding_dim == 256

    def test_score_candidate_basic(self):
        scorer = VulcanCandidateScorer()

        context = {
            "prompt_text": "The quick brown fox",
            "prompt_tokens": [1, 2, 3, 4],
            "vocab": set(range(1000)),
        }

        score, metadata = scorer.score_candidate(None, "jumps over", context)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(metadata, dict)

    def test_score_candidate_with_list_tokens(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test prompt", "vocab": set(range(1000))}

        score, metadata = scorer.score_candidate(None, [10, 20, 30], context)

        assert isinstance(score, float)
        assert isinstance(metadata, dict)

    def test_score_candidate_caching(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test prompt", "vocab": set(range(1000))}

        # First call - no cache
        score1, meta1 = scorer.score_candidate(None, "test candidate", context)

        # Second call - should use cache
        score2, meta2 = scorer.score_candidate(None, "test candidate", context)

        # Scores should be identical due to caching
        assert score1 == score2

    def test_get_metrics(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test", "vocab": set(range(1000))}
        scorer.score_candidate(None, "test", context)

        metrics = scorer.get_metrics()

        assert "total_scores" in metrics
        assert "total_time" in metrics
        assert metrics["total_scores"] > 0

    def test_reset_metrics(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test", "vocab": set(range(1000))}
        scorer.score_candidate(None, "test", context)

        scorer.reset_metrics()

        metrics = scorer.get_metrics()
        assert metrics["total_scores"] == 0
        assert metrics["total_time"] == 0.0

    def test_clear_cache(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test", "vocab": set(range(1000))}
        scorer.score_candidate(None, "test", context)

        # Clear cache
        scorer.clear_cache()

        # Cache should be empty
        cache_stats = scorer.cache.get_stats()
        assert cache_stats["size"] == 0

    @pytest.mark.asyncio
    async def test_score_candidates_async(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test prompt", "vocab": set(range(1000))}

        candidates = ["candidate1", "candidate2", "candidate3"]
        scores = await scorer.score_candidates_async(None, candidates, context)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_score_candidates_batch(self):
        scorer = VulcanCandidateScorer()

        context = {"prompt_text": "Test prompt", "vocab": set(range(1000))}

        candidates = ["cand1", "cand2", "cand3", "cand4", "cand5"]
        results = scorer.score_candidates_batch(None, candidates, context)

        assert len(results) == 5

    def test_length_penalty(self):
        scorer = VulcanCandidateScorer()

        # Short candidate
        penalty_short = scorer._apply_length_penalty("short", {})

        # Long candidate
        long_text = "very " * 100 + "long"
        penalty_long = scorer._apply_length_penalty(long_text, {})

        # Longer text should have higher penalty
        assert penalty_long >= penalty_short

    def test_diversity_penalty(self):
        scorer = VulcanCandidateScorer()

        context = {
            "previous_candidates": ["test candidate", "another test", "more tests"]
        }

        # Similar candidate
        penalty_similar = scorer._apply_diversity_penalty(
            "test candidate again", context
        )

        # Dissimilar candidate
        penalty_different = scorer._apply_diversity_penalty(
            "completely different words", context
        )

        # Similar should have higher penalty
        assert penalty_similar > penalty_different

    def test_oov_penalty(self):
        scorer = VulcanCandidateScorer()

        context = {"vocab": {"test", "known", "words"}}

        # All known words
        penalty_known = scorer._apply_oov_penalty("test known words", context)

        # Has OOV words
        penalty_oov = scorer._apply_oov_penalty("test unknown strange", context)

        # OOV should have higher penalty
        assert penalty_oov > penalty_known

    def test_repetition_penalty(self):
        scorer = VulcanCandidateScorer()

        # Repetitive text
        penalty_rep = scorer._apply_repetition_penalty("test test test test", {})

        # Non-repetitive
        penalty_norep = scorer._apply_repetition_penalty(
            "different words each time", {}
        )

        # Repetitive should have higher penalty
        assert penalty_rep > penalty_norep


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_global_scorer(self):
        scorer1 = get_global_scorer()
        scorer2 = get_global_scorer()

        # Should return the same instance
        assert scorer1 is scorer2

    def test_score_candidate_sync(self):
        context = {"prompt_text": "Test", "vocab": set(range(1000))}

        score, metadata = score_candidate_sync(None, "test candidate", context)

        assert isinstance(score, float)
        assert isinstance(metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
