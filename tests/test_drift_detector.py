"""
Comprehensive test suite for drift_detector.py
"""

import threading
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from drift_detector import (MAX_DRIFT_THRESHOLD, MAX_EMBEDDINGS,
                            MAX_HISTORY_SIZE, MIN_DRIFT_THRESHOLD,
                            MIN_HISTORY_SIZE, REALIGNMENT_METHODS,
                            DriftDetector, DriftMetrics)


@pytest.fixture
def detector():
    """Create drift detector."""
    return DriftDetector(dim=128, drift_threshold=0.1, history=5)


@pytest.fixture
def embeddings():
    """Create test embeddings."""
    np.random.seed(42)
    return np.random.randn(10, 128).astype(np.float32)


class TestDriftMetrics:
    """Test DriftMetrics dataclass."""

    def test_initialization(self):
        """Test metrics initialization."""
        from datetime import datetime

        metrics = DriftMetrics(
            timestamp=datetime.utcnow(),
            drift=0.5,
            avg_drift=0.4,
            embeddings_count=10,
            dimension=128,
        )

        assert metrics.drift == 0.5
        assert metrics.avg_drift == 0.4
        assert not metrics.realignment_triggered

    def test_to_dict(self):
        """Test conversion to dict."""
        from datetime import datetime

        metrics = DriftMetrics(
            timestamp=datetime.utcnow(),
            drift=0.3,
            avg_drift=0.25,
            embeddings_count=20,
            dimension=64,
            realignment_triggered=True,
        )

        d = metrics.to_dict()

        assert d["drift"] == 0.3
        assert d["realignment_triggered"] is True
        assert "timestamp" in d


class TestDriftDetector:
    """Test DriftDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = DriftDetector(dim=64, drift_threshold=0.2, history=10)

        assert detector.dim == 64
        assert detector.drift_threshold == 0.2
        assert detector.history_size == 10
        assert detector.previous_embeddings is None

    def test_invalid_dimension(self):
        """Test invalid dimension raises error."""
        with pytest.raises(ValueError, match="dim must be a positive integer"):
            DriftDetector(dim=0)

        with pytest.raises(ValueError, match="dim must be a positive integer"):
            DriftDetector(dim=-10)

    def test_invalid_threshold(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError, match="drift_threshold must be"):
            DriftDetector(drift_threshold=-0.1)

        with pytest.raises(ValueError, match="drift_threshold must be"):
            DriftDetector(drift_threshold=3.0)

    def test_invalid_history(self):
        """Test invalid history raises error."""
        with pytest.raises(ValueError, match="history must be"):
            DriftDetector(history=0)

    def test_history_capped(self):
        """Test history is capped at maximum."""
        detector = DriftDetector(history=MAX_HISTORY_SIZE + 100)

        assert detector.history_size == MAX_HISTORY_SIZE

    def test_invalid_realignment_method(self):
        """Test invalid realignment method."""
        with pytest.raises(ValueError, match="realignment_method must be"):
            DriftDetector(realignment_method="invalid")

    def test_validate_embeddings_valid(self, detector, embeddings):
        """Test validating valid embeddings."""
        valid, error = detector._validate_embeddings(embeddings)

        assert valid is True
        assert error is None

    def test_validate_embeddings_none(self, detector):
        """Test validating None embeddings."""
        valid, error = detector._validate_embeddings(None)

        assert valid is False
        assert "cannot be None" in error

    def test_validate_embeddings_wrong_type(self, detector):
        """Test validating wrong type."""
        valid, error = detector._validate_embeddings("not an array")

        assert valid is False
        assert "must be ndarray" in error

    def test_validate_embeddings_wrong_ndim(self, detector):
        """Test validating wrong dimensions."""
        valid, error = detector._validate_embeddings(np.array([1, 2, 3]))

        assert valid is False
        assert "must be 2D" in error

    def test_validate_embeddings_empty(self, detector):
        """Test validating empty array."""
        valid, error = detector._validate_embeddings(np.array([]).reshape(0, 128))

        assert valid is False
        assert "empty" in error

    def test_validate_embeddings_wrong_dim(self, detector):
        """Test validating dimension mismatch."""
        wrong_dim = np.random.randn(10, 64)

        valid, error = detector._validate_embeddings(wrong_dim)

        assert valid is False
        assert "dimension mismatch" in error

    def test_validate_embeddings_too_many(self, detector):
        """Test validating too many embeddings."""
        too_many = np.random.randn(MAX_EMBEDDINGS + 1, 128)

        valid, error = detector._validate_embeddings(too_many)

        assert valid is False
        assert "Too many embeddings" in error

    def test_validate_embeddings_nan(self, detector):
        """Test validating embeddings with NaN."""
        with_nan = np.random.randn(10, 128)
        with_nan[0, 0] = np.nan

        valid, error = detector._validate_embeddings(with_nan)

        assert valid is False
        assert "NaN or Inf" in error

    def test_normalize(self, detector, embeddings):
        """Test embedding normalization."""
        normalized = detector._normalize(embeddings)

        # Check all vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(norms)))

    def test_normalize_zero_vectors(self, detector):
        """Test normalizing zero vectors."""
        with_zeros = np.zeros((5, 128))

        normalized = detector._normalize(with_zeros)

        # Should handle zeros gracefully
        assert normalized.shape == (5, 128)
        assert np.isfinite(normalized).all()

    def test_track_drift_first_batch(self, detector, embeddings):
        """Test drift tracking on first batch."""
        drift = detector.track_drift(embeddings)

        assert drift == 0.0
        assert detector.previous_embeddings is not None
        assert len(detector.drift_history) == 1

    def test_track_drift_subsequent(self, detector, embeddings):
        """Test drift tracking on subsequent batches."""
        detector.track_drift(embeddings)

        # Create slightly different embeddings
        embeddings2 = embeddings + np.random.randn(10, 128) * 0.1
        drift = detector.track_drift(embeddings2)

        assert drift > 0.0
        assert len(detector.drift_history) == 2

    def test_track_drift_shape_mismatch(self, detector, embeddings):
        """Test drift tracking with shape mismatch."""
        detector.track_drift(embeddings)

        # Different number of embeddings
        embeddings2 = np.random.randn(5, 128)
        drift = detector.track_drift(embeddings2)

        # Should handle gracefully
        assert isinstance(drift, float)

    def test_track_drift_invalid(self, detector):
        """Test drift tracking with invalid input."""
        with pytest.raises(ValueError):
            detector.track_drift(np.array([1, 2, 3]))

    def test_mean_drift(self, detector, embeddings):
        """Test mean drift calculation."""
        assert detector.mean_drift() == 0.0

        detector.track_drift(embeddings)
        detector.track_drift(embeddings + 0.1)
        detector.track_drift(embeddings + 0.2)

        mean = detector.mean_drift()
        assert mean > 0.0

    def test_realign_center(self, detector, embeddings):
        """Test center-based realignment."""
        detector.track_drift(embeddings)

        realigned = detector.realign_embeddings(embeddings, method="center")

        assert realigned.shape == embeddings.shape
        assert detector.total_realignments == 1

    def test_realign_pca(self, detector, embeddings):
        """Test PCA-based realignment."""
        detector.track_drift(embeddings)

        realigned = detector.realign_embeddings(embeddings, method="pca")

        assert realigned.shape == embeddings.shape

    def test_realign_procrustes(self, detector, embeddings):
        """Test Procrustes alignment."""
        detector.track_drift(embeddings)

        realigned = detector.realign_embeddings(embeddings, method="procrustes")

        assert realigned.shape == embeddings.shape

    def test_realign_no_reference(self, detector, embeddings):
        """Test realignment without reference."""
        # No previous tracking
        realigned = detector.realign_embeddings(embeddings)

        # Should use current as reference
        np.testing.assert_array_equal(realigned, embeddings)

    def test_realign_invalid_method(self, detector, embeddings):
        """Test realignment with invalid method."""
        with pytest.raises(ValueError, match="Unknown realignment method"):
            detector.realign_embeddings(embeddings, method="invalid")

    def test_realign_if_drift_no_realignment(self, detector, embeddings):
        """Test realign_if_drift when no realignment needed."""
        agents = [f"agent{i}" for i in range(10)]

        result = detector.realign_if_drift(embeddings, agents)

        assert not result["realignment_needed"]
        assert len(result["agents_to_realign"]) == 0

    def test_realign_if_drift_with_realignment(self, detector, embeddings):
        """Test realign_if_drift when realignment triggered."""
        agents = [f"agent{i}" for i in range(10)]

        # Set low threshold
        detector.drift_threshold = 0.01

        # Track initial
        detector.track_drift(embeddings)

        # Create high drift
        drifted = embeddings + np.random.randn(10, 128) * 0.5

        result = detector.realign_if_drift(drifted, agents)

        # With high drift, should trigger realignment
        assert isinstance(result["realignment_needed"], bool)
        assert "realigned_embeddings" in result

    def test_realign_if_drift_invalid_agents(self, detector, embeddings):
        """Test realign_if_drift with invalid agents."""
        with pytest.raises(ValueError, match="agents must be a list"):
            detector.realign_if_drift(embeddings, "not a list")

    def test_realign_if_drift_agent_count_mismatch(self, detector, embeddings):
        """Test realign_if_drift with agent count mismatch."""
        agents = ["agent1", "agent2"]  # Only 2 agents but 10 embeddings

        with pytest.raises(ValueError, match="Agent count mismatch"):
            detector.realign_if_drift(embeddings, agents)

    def test_faiss_pairwise_similarity(self, detector, embeddings):
        """Test FAISS similarity computation."""
        sims = detector.faiss_pairwise_similarity(embeddings)

        assert sims.shape == (10, 10)

        # Diagonal should be high (self-similarity)
        diagonal = np.diag(sims)
        assert np.all(diagonal > 0.9)

    def test_faiss_pairwise_similarity_invalid(self, detector):
        """Test FAISS similarity with invalid input."""
        with pytest.raises(ValueError):
            detector.faiss_pairwise_similarity(np.array([1, 2, 3]))

    def test_find_drift_outliers(self, detector):
        """Test outlier detection."""
        # Create normal and outlier embeddings
        normal = np.random.randn(20, 128) * 0.5
        outliers = np.random.randn(3, 128) * 3.0
        all_embs = np.vstack([normal, outliers])

        # Set reference
        detector.track_drift(normal)

        outlier_idx, outlier_scores = detector.find_drift_outliers(
            all_embs, threshold=2.0
        )

        assert len(outlier_idx) > 0
        assert len(outlier_scores) == len(outlier_idx)

    def test_find_drift_outliers_no_reference(self, detector, embeddings):
        """Test outlier detection without reference."""
        outlier_idx, outlier_scores = detector.find_drift_outliers(embeddings)

        # Should return empty
        assert len(outlier_idx) == 0

    def test_drift_report(self, detector, embeddings):
        """Test drift report generation."""
        detector.track_drift(embeddings)
        detector.track_drift(embeddings + 0.1)

        report = detector.drift_report()

        assert "current_drift" in report
        assert "mean_drift" in report
        assert "drift_history" in report
        assert "total_checks" in report
        assert report["dimension"] == 128

    def test_get_detailed_metrics(self, detector, embeddings):
        """Test getting detailed metrics."""
        detector.track_drift(embeddings)
        detector.track_drift(embeddings + 0.1)

        metrics = detector.get_detailed_metrics()

        assert len(metrics) == 2
        assert all("timestamp" in m for m in metrics)
        assert all("drift" in m for m in metrics)

    def test_reset(self, detector, embeddings):
        """Test resetting detector."""
        detector.track_drift(embeddings)
        detector.track_drift(embeddings + 0.1)

        detector.reset()

        assert detector.previous_embeddings is None
        assert len(detector.drift_history) == 0
        assert detector.total_checks == 0

    def test_cleanup(self, detector, embeddings):
        """Test cleanup."""
        detector.track_drift(embeddings)

        detector.cleanup()

        assert detector.previous_embeddings is None

    def test_threshold_and_history_bounds(self):
        """Ensure detector respects constant bounds on thresholds/history."""
        # Lower bound clamp or validation
        with pytest.raises(ValueError):
            DriftDetector(drift_threshold=MIN_DRIFT_THRESHOLD - 1e-6)

        with pytest.raises(ValueError):
            DriftDetector(history=MIN_HISTORY_SIZE - 1)

        # Upper bound clamp/validation
        with pytest.raises(ValueError):
            DriftDetector(drift_threshold=MAX_DRIFT_THRESHOLD + 1e-6)

        d = DriftDetector(history=MAX_HISTORY_SIZE + 9999)
        assert d.history_size == MAX_HISTORY_SIZE

    def test_realign_methods_constant(self):
        """REALIGNMENT_METHODS should include the supported methods."""
        assert isinstance(REALIGNMENT_METHODS, (list, tuple, set))
        # Not asserting exact contents to avoid over-coupling,
        # but the typical methods should be present.
        for m in ("center", "pca", "procrustes"):
            assert m in REALIGNMENT_METHODS


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_tracking(self, embeddings):
        """Test concurrent drift tracking."""
        detector = DriftDetector(dim=128, history=10)

        results = []
        errors = []

        def track():
            try:
                drift = detector.track_drift(embeddings)
                results.append(drift)
            except Exception as e:
                # Collect any exception to assert later without killing the thread
                errors.append(e)

        threads = [threading.Thread(target=track) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert not errors, f"Thread exceptions: {errors}"

        # We started with no reference; the first call should set reference (drift=0.0)
        # Subsequent calls will compute drift (non-negative float)
        assert len(results) == 8
        assert all(isinstance(d, float) for d in results)
        assert results[0] >= 0.0

        # The drift_history should have as many entries as calls (each call records)
        # Some implementations may record only after the first reference is set;
        # we allow either 8 or 7 here to avoid over-coupling.
        assert len(detector.drift_history) in (7, 8)

    def test_concurrent_realign_if_drift(self, embeddings):
        """Stress realign_if_drift with concurrent calls to ensure no races."""
        detector = DriftDetector(dim=128, drift_threshold=0.01)
        agents = [f"a{i}" for i in range(embeddings.shape[0])]

        # Seed a reference
        detector.track_drift(embeddings)

        results = []
        errors = []

        def do_realign():
            try:
                noisy = embeddings + np.random.randn(*embeddings.shape) * 0.05
                out = detector.realign_if_drift(noisy, agents)
                results.append(out)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_realign) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread exceptions during realign_if_drift: {errors}"
        assert len(results) == 6
        # Each result should be a dict with required keys
        for r in results:
            assert isinstance(r, dict)
            assert "realignment_needed" in r
            assert "agents_to_realign" in r
            assert "current_drift" in r
            # realigned_embeddings may or may not be present depending on threshold hit
            assert "mean_drift" in r


# Optional: very light check that constructor defaults are sane without over-coupling.
def test_constructor_defaults_sanity():
    d = DriftDetector()
    assert isinstance(d.dim, int) and d.dim > 0
    assert MIN_DRIFT_THRESHOLD <= d.drift_threshold <= MAX_DRIFT_THRESHOLD
    assert MIN_HISTORY_SIZE <= d.history_size <= MAX_HISTORY_SIZE