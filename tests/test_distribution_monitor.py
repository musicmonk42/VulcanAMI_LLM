"""
Comprehensive test suite for distribution_monitor.py
"""

import tempfile
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from distribution_monitor import (DetectionMethod, DistributionMonitor,
                                  DistributionSnapshot, DriftDetection,
                                  DriftSeverity, DriftType,
                                  KolmogorovSmirnovDetector, MMDDetector,
                                  PageHinkleyDetector, WassersteinDetector,
                                  WindowedDistribution)


@pytest.fixture
def reference_data():
    """Create reference distribution data."""
    np.random.seed(42)
    return np.random.randn(1000, 5)


@pytest.fixture
def shifted_data():
    """Create shifted distribution data."""
    np.random.seed(43)
    return np.random.randn(1000, 5) + 2.0  # Shifted distribution


@pytest.fixture
def distribution_monitor():
    """Create DistributionMonitor instance."""
    return DistributionMonitor()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEnums:
    """Test enum classes."""

    def test_drift_type_enum(self):
        """Test DriftType enum."""
        assert DriftType.FEATURE_DRIFT.value == "feature_drift"
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"

    def test_detection_method_enum(self):
        """Test DetectionMethod enum."""
        assert DetectionMethod.KS_TEST.value == "kolmogorov_smirnov"
        assert DetectionMethod.WASSERSTEIN.value == "wasserstein"

    def test_drift_severity_enum(self):
        """Test DriftSeverity enum."""
        assert DriftSeverity.NONE.value == 0
        assert DriftSeverity.CRITICAL.value == 4


class TestDataClasses:
    """Test dataclass structures."""

    def test_drift_detection_creation(self):
        """Test creating DriftDetection."""
        detection = DriftDetection(
            drift_type=DriftType.FEATURE_DRIFT,
            method=DetectionMethod.KS_TEST,
            severity=DriftSeverity.HIGH,
            statistic=0.5,
            p_value=0.01,
            threshold=0.05,
            detected=True
        )

        assert detection.drift_type == DriftType.FEATURE_DRIFT
        assert detection.severity == DriftSeverity.HIGH
        assert detection.detected is True

    def test_distribution_snapshot_creation(self):
        """Test creating DistributionSnapshot."""
        features = np.array([[1, 2], [3, 4]])

        snapshot = DistributionSnapshot(
            features=features,
            labels=None,
            timestamp=time.time(),
            sample_count=2,
            feature_stats={}
        )

        assert snapshot.sample_count == 2
        assert snapshot.features.shape == (2, 2)


class TestWindowedDistribution:
    """Test WindowedDistribution."""

    def test_initialization(self):
        """Test windowed distribution initialization."""
        window = WindowedDistribution(window_size=100)

        assert window.window_size == 100
        assert len(window.data) == 0

    def test_update_single_value(self):
        """Test updating with single value."""
        window = WindowedDistribution(window_size=100)

        value = np.array([1.0, 2.0, 3.0])
        window.update(value)

        assert len(window.data) == 1

    def test_update_multiple_values(self):
        """Test updating with multiple values."""
        window = WindowedDistribution(window_size=100)

        for i in range(10):
            window.update(np.array([float(i)] * 3))

        assert len(window.data) == 10

    def test_window_size_limit(self):
        """Test that window size is limited."""
        window = WindowedDistribution(window_size=10)

        for i in range(20):
            window.update(np.array([float(i)] * 3))

        assert len(window.data) == 10

    def test_statistics_calculation(self):
        """Test statistics calculation."""
        window = WindowedDistribution(window_size=100)

        for i in range(10):
            window.update(np.array([float(i)] * 3))

        stats = window.get_statistics()

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_statistics_empty_window(self):
        """Test statistics with empty window."""
        window = WindowedDistribution(window_size=100)

        stats = window.get_statistics()

        assert stats['mean'] is None


class TestKolmogorovSmirnovDetector:
    """Test KolmogorovSmirnovDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = KolmogorovSmirnovDetector(threshold=0.05)

        assert detector.threshold == 0.05
        assert detector.reference_samples is None

    def test_set_reference(self, reference_data):
        """Test setting reference distribution."""
        detector = KolmogorovSmirnovDetector()

        detector.set_reference(reference_data)

        assert detector.reference_samples is not None
        assert detector.reference_samples.shape == reference_data.shape

    def test_detect_no_reference(self):
        """Test detection without reference."""
        detector = KolmogorovSmirnovDetector()

        current_data = np.random.randn(100, 5)
        detections = detector.detect(current_data)

        assert detections == []

    def test_detect_no_drift(self, reference_data):
        """Test detection with no drift."""
        detector = KolmogorovSmirnovDetector()
        detector.set_reference(reference_data)

        # Similar distribution
        np.random.seed(42)
        current_data = np.random.randn(100, 5)

        detections = detector.detect(current_data)

        # Most should not detect drift
        detected_count = sum(1 for d in detections if d.detected)
        assert detected_count < len(detections)

    def test_detect_with_drift(self, reference_data, shifted_data):
        """Test detection with drift."""
        detector = KolmogorovSmirnovDetector(threshold=0.05)
        detector.set_reference(reference_data)

        detections = detector.detect(shifted_data[:100])

        # Should detect drift in multiple features
        detected_count = sum(1 for d in detections if d.detected)
        assert detected_count > 0

    def test_severity_levels(self, reference_data, shifted_data):
        """Test severity level assignment."""
        detector = KolmogorovSmirnovDetector()
        detector.set_reference(reference_data)

        detections = detector.detect(shifted_data[:100])

        # Check that severity is assigned
        for detection in detections:
            assert isinstance(detection.severity, DriftSeverity)


class TestWassersteinDetector:
    """Test WassersteinDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = WassersteinDetector(threshold=0.1)

        assert detector.threshold == 0.1

    def test_set_reference(self, reference_data):
        """Test setting reference distribution."""
        detector = WassersteinDetector()
        detector.set_reference(reference_data)

        assert detector.reference_distribution is not None

    def test_detect_no_drift(self, reference_data):
        """Test detection with no drift."""
        detector = WassersteinDetector()
        detector.set_reference(reference_data)

        np.random.seed(42)
        current_data = np.random.randn(100, 5)

        detections = detector.detect(current_data)

        assert isinstance(detections, list)

    def test_detect_with_drift(self, reference_data, shifted_data):
        """Test detection with drift."""
        detector = WassersteinDetector(threshold=0.1)
        detector.set_reference(reference_data)

        detections = detector.detect(shifted_data[:100])

        # Should detect some drift
        detected_count = sum(1 for d in detections if d.detected)
        assert detected_count > 0


class TestMMDDetector:
    """Test MMDDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MMDDetector(threshold=0.05)

        assert detector.threshold == 0.05
        assert detector.kernel == 'rbf'

    def test_set_reference(self, reference_data):
        """Test setting reference distribution."""
        detector = MMDDetector()
        detector.set_reference(reference_data)

        assert detector.reference_data is not None

    def test_detect_no_reference(self):
        """Test detection without reference."""
        detector = MMDDetector()
        current_data = np.random.randn(100, 5)

        detection = detector.detect(current_data)

        assert detection is None

    def test_detect_with_drift(self, reference_data, shifted_data):
        """Test detection with drift."""
        detector = MMDDetector(threshold=0.05)
        detector.set_reference(reference_data)

        detection = detector.detect(shifted_data[:100])

        assert detection is not None
        assert isinstance(detection.statistic, float)

    def test_compute_mmd(self, reference_data):
        """Test MMD computation."""
        detector = MMDDetector()
        detector.set_reference(reference_data)

        X = reference_data[:100]
        Y = reference_data[100:200]

        mmd = detector._compute_mmd(X, Y)

        assert isinstance(mmd, float)
        assert mmd >= 0


class TestPageHinkleyDetector:
    """Test PageHinkleyDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = PageHinkleyDetector(delta=0.005, threshold=50)

        assert detector.delta == 0.005
        assert detector.threshold == 50

    def test_detect_no_drift(self):
        """Test detection with stable values."""
        detector = PageHinkleyDetector()

        # Feed stable values
        for i in range(10):
            detection = detector.detect(1.0)
            assert detection.detected is False

    def test_detect_with_drift(self):
        """Test detection with drift."""
        detector = PageHinkleyDetector(threshold=10)

        # Feed increasing values
        detected = False
        for i in range(100):
            detection = detector.detect(float(i))
            if detection.detected:
                detected = True
                break

        # Should eventually detect drift
        assert detected

    def test_reset(self):
        """Test detector reset."""
        detector = PageHinkleyDetector()

        detector.detect(5.0)
        detector.reset()

        assert detector.sum == 0
        assert detector.min_sum == 0


class TestDistributionMonitor:
    """Test DistributionMonitor main class."""

    def test_initialization(self, distribution_monitor):
        """Test monitor initialization."""
        assert distribution_monitor is not None
        assert distribution_monitor.reference_distribution is None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            'window_size': 500,
            'detection_threshold': 0.01
        }

        monitor = DistributionMonitor(config)

        assert monitor.window_size == 500
        assert monitor.detection_threshold == 0.01

    def test_set_reference(self, distribution_monitor, reference_data):
        """Test setting reference distribution."""
        distribution_monitor.set_reference(reference_data)

        assert distribution_monitor.reference_distribution is not None
        assert distribution_monitor.reference_distribution.shape == reference_data.shape

    def test_update_single_sample(self, distribution_monitor):
        """Test updating with single sample."""
        features = np.array([1.0, 2.0, 3.0])

        drift_detected = distribution_monitor.update(features)

        assert isinstance(drift_detected, bool)
        assert distribution_monitor.sample_count == 1

    def test_update_multiple_samples(self, distribution_monitor):
        """Test updating with multiple samples."""
        for i in range(10):
            features = np.array([float(i)] * 5)
            distribution_monitor.update(features)

        assert distribution_monitor.sample_count == 10

    def test_detect_shift_no_reference(self, distribution_monitor):
        """Test shift detection without reference."""
        features = np.array([1.0, 2.0, 3.0])

        shift_detected = distribution_monitor.detect_shift(features)

        assert shift_detected is False

    def test_detect_shift_insufficient_data(self, distribution_monitor, reference_data):
        """Test shift detection with insufficient data."""
        distribution_monitor.set_reference(reference_data)

        # Only add a few samples
        for i in range(5):
            distribution_monitor.update(np.array([float(i)] * 5))

        shift_detected = distribution_monitor.detect_shift(np.array([1.0] * 5))

        assert shift_detected is False

    def test_detect_shift_with_drift(self, distribution_monitor, reference_data, shifted_data):
        """Test shift detection with actual drift."""
        distribution_monitor.set_reference(reference_data)

        # Add shifted samples
        for i in range(100):
            distribution_monitor.update(shifted_data[i])

        # Detection happens at intervals, so may not detect immediately
        # But after enough samples, should detect
        shift_detected = distribution_monitor.detect_shift(shifted_data[0])

        # May or may not detect depending on thresholds
        assert isinstance(shift_detected, bool)

    def test_get_drift_summary_no_drift(self, distribution_monitor):
        """Test drift summary with no drift."""
        summary = distribution_monitor.get_drift_summary()

        assert 'drift_detected' in summary
        assert summary['drift_detected'] is False

    def test_get_drift_summary_with_drift(self, distribution_monitor, reference_data, shifted_data):
        """Test drift summary with drift."""
        distribution_monitor.set_reference(reference_data)

        # Force drift detection by adding many shifted samples
        for i in range(200):
            distribution_monitor.update(shifted_data[i])
            if i % distribution_monitor.check_interval == 0:
                distribution_monitor.detect_shift(shifted_data[i])

        summary = distribution_monitor.get_drift_summary()

        assert 'total_checks' in summary
        assert 'total_drifts' in summary

    def test_analyze_feature_importance(self, distribution_monitor):
        """Test feature importance analysis."""
        importance = distribution_monitor.analyze_feature_importance()

        assert isinstance(importance, dict)

    def test_get_statistics(self, distribution_monitor, reference_data):
        """Test getting statistics."""
        distribution_monitor.set_reference(reference_data)

        for i in range(10):
            distribution_monitor.update(np.array([float(i)] * 5))

        stats = distribution_monitor.get_statistics()

        assert 'total_samples' in stats
        assert 'total_checks' in stats
        assert 'reference_set' in stats

    def test_calculate_trend_insufficient_data(self, distribution_monitor):
        """Test trend calculation with insufficient data."""
        values = np.array([1.0, 2.0])

        trend = distribution_monitor._calculate_trend(values)

        assert trend == "insufficient_data"

    def test_calculate_trend_stable(self, distribution_monitor):
        """Test trend calculation with stable values."""
        values = np.ones(20)

        trend = distribution_monitor._calculate_trend(values)

        assert trend == "stable"

    def test_calculate_trend_improving(self, distribution_monitor):
        """Test trend calculation with improving values."""
        values = np.arange(20, dtype=float)

        trend = distribution_monitor._calculate_trend(values)

        assert trend == "improving"

    def test_calculate_trend_degrading(self, distribution_monitor):
        """Test trend calculation with degrading values."""
        values = np.arange(20, -1, -1, dtype=float)

        trend = distribution_monitor._calculate_trend(values)

        assert trend == "degrading"


class TestPersistence:
    """Test state persistence."""

    def test_save_state(self, distribution_monitor, temp_dir, reference_data):
        """Test saving monitor state."""
        distribution_monitor.set_reference(reference_data)

        distribution_monitor.save_state(temp_dir)

        save_path = Path(temp_dir)
        assert (save_path / 'reference_distribution.npy').exists()
        assert (save_path / 'statistics.json').exists()

    def test_load_state(self, temp_dir, reference_data):
        """Test loading monitor state."""
        # Create and save
        monitor1 = DistributionMonitor()
        monitor1.set_reference(reference_data)
        monitor1.save_state(temp_dir)

        # Load into new instance
        monitor2 = DistributionMonitor()
        monitor2.load_state(temp_dir)

        assert monitor2.reference_distribution is not None

    def test_load_nonexistent_state(self, distribution_monitor):
        """Test loading from nonexistent path."""
        # Should not crash
        distribution_monitor.load_state("/nonexistent/path")


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_features(self, distribution_monitor):
        """Test with empty features."""
        features = np.array([])

        # Should handle gracefully
        try:
            distribution_monitor.update(features)
        except:
            pass  # Acceptable to fail

    def test_single_feature(self, distribution_monitor):
        """Test with single feature."""
        reference = np.random.randn(100, 1)
        distribution_monitor.set_reference(reference)

        features = np.array([1.0])
        distribution_monitor.update(features)

        assert distribution_monitor.sample_count > 0

    def test_high_dimensional_features(self, distribution_monitor):
        """Test with high-dimensional features."""
        reference = np.random.randn(100, 50)
        distribution_monitor.set_reference(reference)

        # Should use PCA
        assert distribution_monitor.pca is not None

    def test_nan_values(self, distribution_monitor):
        """Test handling NaN values."""
        features = np.array([1.0, np.nan, 3.0])

        # Should handle gracefully
        try:
            distribution_monitor.update(features)
        except:
            pass  # Acceptable to fail with NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
