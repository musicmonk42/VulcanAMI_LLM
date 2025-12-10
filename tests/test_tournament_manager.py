"""
Comprehensive test suite for tournament_manager.py
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from tournament_manager import (TournamentError, TournamentManager,
                                ValidationError, trace_id)


@pytest.fixture
def tournament_manager():
    """Create TournamentManager instance."""
    return TournamentManager(
        similarity_threshold=0.8,
        diversity_penalty=0.2,
        winner_percentage=0.1,
        adaptive=False  # Disable for predictable testing
    )


@pytest.fixture
def sample_proposals():
    """Create sample proposals."""
    return [
        {"id": i, "data": f"proposal_{i}"}
        for i in range(20):
    ]


@pytest.fixture
def sample_fitness():
    """Create sample fitness values."""
    return np.random.uniform(0.5, 1.0, 20).tolist()


@pytest.fixture
def embedding_func():
    """Create embedding function."""
    def embed(proposal):
        # Simple embedding based on id
        emb = np.zeros(16)
        emb[proposal["id"] % 16] = 1.0
        return emb + np.random.normal(0, 0.1, 16)
    return embed


class TestTournamentManagerInitialization:
    """Test TournamentManager initialization."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        tm = TournamentManager()

        assert tm.similarity_threshold == 0.8
        assert tm.current_diversity_penalty == 0.2
        assert tm.winner_percentage == 0.1

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        tm = TournamentManager(
            similarity_threshold=0.7,
            diversity_penalty=0.3,
            winner_percentage=0.15,
            min_winners=2,
            max_winners=10
        )

        assert tm.similarity_threshold == 0.7
        assert tm.current_diversity_penalty == 0.3
        assert tm.winner_percentage == 0.15
        assert tm.min_winners == 2
        assert tm.max_winners == 10

    def test_initialization_invalid_similarity_threshold(self):
        """Test initialization with invalid similarity threshold."""
        with pytest.raises(ValidationError, match="similarity_threshold"):
            TournamentManager(similarity_threshold=1.5)

        with pytest.raises(ValidationError, match="similarity_threshold"):
            TournamentManager(similarity_threshold=-0.1)

    def test_initialization_invalid_diversity_penalty(self):
        """Test initialization with invalid diversity penalty."""
        with pytest.raises(ValidationError, match="diversity_penalty"):
            TournamentManager(diversity_penalty=0)

        with pytest.raises(ValidationError, match="diversity_penalty"):
            TournamentManager(diversity_penalty=1.5)

    def test_initialization_invalid_winner_percentage(self):
        """Test initialization with invalid winner percentage."""
        with pytest.raises(ValidationError, match="winner_percentage"):
            TournamentManager(winner_percentage=0)

        with pytest.raises(ValidationError, match="winner_percentage"):
            TournamentManager(winner_percentage=1.5)

    def test_initialization_invalid_min_winners(self):
        """Test initialization with invalid min_winners."""
        with pytest.raises(ValidationError, match="min_winners"):
            TournamentManager(min_winners=0)

    def test_initialization_invalid_max_winners(self):
        """Test initialization with invalid max_winners."""
        with pytest.raises(ValidationError, match="max_winners"):
            TournamentManager(min_winners=5, max_winners=3)


class TestInputValidation:
    """Test input validation."""

    def test_validate_empty_proposals(self, tournament_manager, embedding_func):
        """Test validation with empty proposals."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            tournament_manager.run_adaptive_tournament([], [], embedding_func)

    def test_validate_length_mismatch(self, tournament_manager, sample_proposals, embedding_func):
        """Test validation with length mismatch."""
        with pytest.raises(ValidationError, match="length mismatch"):
            tournament_manager.run_adaptive_tournament(
                sample_proposals,
                [0.5] * 10,  # Wrong length
                embedding_func
            )

    def test_validate_invalid_fitness(self, tournament_manager, sample_proposals, embedding_func):
        """Test validation with invalid fitness values."""
        invalid_fitness = [0.5] * 19 + [float('nan')]

        with pytest.raises(ValidationError, match="Invalid fitness"):
            tournament_manager.run_adaptive_tournament(
                sample_proposals,
                invalid_fitness,
                embedding_func
            )


class TestEmbeddingValidation:
    """Test embedding validation."""

    def test_validate_embeddings_wrong_type(self, tournament_manager):
        """Test embedding validation with wrong type."""
        with pytest.raises(ValidationError, match="must be numpy array"):
            tournament_manager._validate_embeddings([1, 2, 3], 3, "test")

    def test_validate_embeddings_wrong_dimensions(self, tournament_manager):
        """Test embedding validation with wrong dimensions."""
        embeddings = np.array([1, 2, 3])  # 1D instead of 2D

        with pytest.raises(ValidationError, match="must be 2D"):
            tournament_manager._validate_embeddings(embeddings, 3, "test")

    def test_validate_embeddings_wrong_count(self, tournament_manager):
        """Test embedding validation with wrong count."""
        embeddings = np.random.rand(5, 10)

        with pytest.raises(ValidationError, match="Expected 3 embeddings"):
            tournament_manager._validate_embeddings(embeddings, 3, "test")

    def test_validate_embeddings_non_finite(self, tournament_manager):
        """Test embedding validation with non-finite values."""
        embeddings = np.random.rand(3, 10)
        embeddings[0, 0] = np.nan

        with pytest.raises(ValidationError, match="non-finite"):
            tournament_manager._validate_embeddings(embeddings, 3, "test")


class TestTournamentExecution:
    """Test tournament execution."""

    def test_run_adaptive_tournament_basic(
        self,
        tournament_manager,
        sample_proposals,
        sample_fitness,
        embedding_func
    ):
        """Test basic tournament execution."""
        winners = tournament_manager.run_adaptive_tournament(
            sample_proposals,
            sample_fitness,
            embedding_func
        )

        assert len(winners) > 0
        assert len(winners) <= len(sample_proposals)
        assert all(isinstance(w, int) for w in winners)

    def test_run_adaptive_tournament_with_meta(
        self,
        tournament_manager,
        sample_proposals,
        sample_fitness,
        embedding_func
    ):
        """Test tournament with metadata collection."""
        meta = {}

        winners = tournament_manager.run_adaptive_tournament(
            sample_proposals,
            sample_fitness,
            embedding_func,
            meta=meta
        )

        assert "trace_id" in meta
        assert "num_proposals" in meta
        assert "num_winners" in meta
        assert "innovation_score" in meta
        assert "diversity" in meta
        assert meta["num_proposals"] == len(sample_proposals)

    def test_run_adaptive_tournament_min_winners(self, sample_proposals, sample_fitness, embedding_func):
        """Test that min_winners is respected."""
        tm = TournamentManager(min_winners=5, winner_percentage=0.01)

        winners = tm.run_adaptive_tournament(
            sample_proposals,
            sample_fitness,
            embedding_func
        )

        assert len(winners) >= 5

    def test_run_adaptive_tournament_max_winners(self, sample_proposals, sample_fitness, embedding_func):
        """Test that max_winners is respected."""
        tm = TournamentManager(max_winners=3, winner_percentage=0.5)

        winners = tm.run_adaptive_tournament(
            sample_proposals,
            sample_fitness,
            embedding_func
        )

        assert len(winners) <= 3

    def test_run_adaptive_tournament_winner_percentage(self, sample_proposals, sample_fitness, embedding_func):
        """Test winner percentage selection."""
        tm = TournamentManager(winner_percentage=0.2)

        winners = tm.run_adaptive_tournament(
            sample_proposals,
            sample_fitness,
            embedding_func
        )

        # Should select approximately 20% (4 out of 20)
        expected_winners = int(np.ceil(0.2 * len(sample_proposals)))
        assert len(winners) == expected_winners


class TestDiversityScoring:
    """Test diversity scoring."""

    def test_diversity_score_identical(self, tournament_manager):
        """Test diversity score with identical items."""
        # All ones - very similar
        sim_matrix = np.ones((5, 5))

        diversity = tournament_manager._diversity_score(sim_matrix)

        # Should be low (close to 0)
        assert diversity < 0.1

    def test_diversity_score_diverse(self, tournament_manager):
        """Test diversity score with diverse items."""
        # Identity matrix - very diverse
        sim_matrix = np.eye(5)

        diversity = tournament_manager._diversity_score(sim_matrix)

        # Should be high (close to 1)
        assert diversity > 0.9

    def test_diversity_score_single_item(self, tournament_manager):
        """Test diversity score with single item."""
        sim_matrix = np.array([[1.0]])

        diversity = tournament_manager._diversity_score(sim_matrix)

        assert diversity == 1.0


class TestCoherenceScoring:
    """Test coherence scoring."""

    def test_coherence_score_identical(self, tournament_manager):
        """Test coherence score with identical items."""
        sim_matrix = np.ones((5, 5))

        coherence = tournament_manager._coherence_score(sim_matrix)

        # Should be high (close to 1)
        assert coherence > 0.9

    def test_coherence_score_diverse(self, tournament_manager):
        """Test coherence score with diverse items."""
        sim_matrix = np.eye(5)

        coherence = tournament_manager._coherence_score(sim_matrix)

        # Should be low (close to 0)
        assert coherence < 0.1


class TestAdaptivePenalty:
    """Test adaptive penalty mechanism."""

    def test_adapt_penalty_below_target(self):
        """Test penalty adaptation when below target."""
        tm = TournamentManager(
            diversity_penalty=0.2,
            adaptive=True,
            target_innovation=0.7
        )

        # Set last innovation below target
        tm.last_innovation_score = 0.5

        initial_penalty = tm.current_diversity_penalty
        tm._adapt_penalty("test")

        # Penalty should increase
        assert tm.current_diversity_penalty > initial_penalty

    def test_adapt_penalty_above_target(self):
        """Test penalty adaptation when above target."""
        tm = TournamentManager(
            diversity_penalty=0.2,
            adaptive=True,
            target_innovation=0.5
        )

        # Set last innovation above target
        tm.last_innovation_score = 0.7

        initial_penalty = tm.current_diversity_penalty
        tm._adapt_penalty("test")

        # Penalty should decrease
        assert tm.current_diversity_penalty < initial_penalty

    def test_adapt_penalty_clamping(self):
        """Test that penalty is clamped to reasonable range."""
        tm = TournamentManager(
            diversity_penalty=0.2,
            adaptive=True,
            target_innovation=0.9
        )

        # Force extreme adaptation
        tm.last_innovation_score = 0.0

        for _ in range(10):
            tm._adapt_penalty("test")

        # Should be clamped to max 0.5
        assert tm.current_diversity_penalty <= 0.5
        assert tm.current_diversity_penalty >= 0.05


class TestUtilityFunctions:
    """Test utility functions."""

    def test_normalize(self, tournament_manager):
        """Test embedding normalization."""
        embeddings = np.random.rand(5, 10)

        normalized = tournament_manager._normalize(embeddings)

        # Check unit length
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

    def test_pairwise_cosine(self, tournament_manager):
        """Test pairwise cosine similarity."""
        embeddings = np.random.rand(5, 10)

        sim_matrix = tournament_manager._pairwise_cosine(embeddings)

        # Check shape
        assert sim_matrix.shape == (5, 5)

        # Check diagonal is 1 (self-similarity)
        assert np.allclose(np.diag(sim_matrix), 1.0)

        # Check symmetry
        assert np.allclose(sim_matrix, sim_matrix.T)

    def test_trace_id(self):
        """Test trace ID generation."""
        id1 = trace_id()
        id2 = trace_id()

        assert len(id1) == 8
        assert id1 != id2  # Should be unique


class TestStatistics:
    """Test statistics."""

    def test_get_stats(self, tournament_manager):
        """Test getting statistics."""
        stats = tournament_manager.get_stats()

        assert "similarity_threshold" in stats
        assert "current_diversity_penalty" in stats
        assert "target_innovation_score" in stats
        assert "winner_percentage" in stats
        assert "adaptive" in stats


class TestDistributedSharder:
    """Test distributed sharder integration."""

    @patch('tournament_manager.DistributedSharder')
    def test_with_sharder_available(self, mock_sharder_class):
        """Test with sharder available."""
        mock_sharder = MagicMock()
        mock_sharder_class.return_value = mock_sharder

        tm = TournamentManager(sharder_threshold=10)

        assert tm.sharder is not None

    @patch('tournament_manager.DistributedSharder', None)
    def test_without_sharder(self):
        """Test without sharder."""
        tm = TournamentManager()

        assert tm.sharder is None

    @patch('tournament_manager.DistributedSharder')
    def test_sharder_usage_threshold(
        self,
        mock_sharder_class,
        sample_proposals,
        sample_fitness,
        embedding_func
    ):
        """Test that sharder is used above threshold."""
        mock_sharder = MagicMock()
        mock_sharder.map = Mock(return_value=[embedding_func(p) for p in sample_proposals])
        mock_sharder_class.return_value = mock_sharder

        tm = TournamentManager(sharder_threshold=10)
        tm.sharder = mock_sharder

        # Should use sharder (20 proposals > threshold of 10)
        tm.run_adaptive_tournament(sample_proposals, sample_fitness, embedding_func)

        assert mock_sharder.map.called


class TestExceptions:
    """Test custom exceptions."""

    def test_tournament_error(self):
        """Test TournamentError."""
        error = TournamentError("test error")

        assert str(error) == "test error"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("validation failed")

        assert str(error) == "validation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
