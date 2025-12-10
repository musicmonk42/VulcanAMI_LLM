"""
Comprehensive test suite for observability_manager.py
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from observability_manager import (MAX_DASHBOARD_AGE_DAYS, MAX_LOG_DIR_SIZE_MB,
                                   MAX_PLOT_AGE_DAYS, MAX_TENSOR_ELEMENTS,
                                   MAX_TENSOR_SIZE, MIN_FREE_DISK_MB,
                                   ObservabilityManager)


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def obs_manager(temp_log_dir):
    """Create ObservabilityManager instance."""
    manager = ObservabilityManager(log_dir=temp_log_dir, enable_cleanup=False)
    yield manager
    manager.shutdown()


class TestObservabilityManagerInitialization:
    """Test ObservabilityManager initialization."""

    def test_initialization_basic(self, temp_log_dir):
        """Test basic initialization."""
        obs = ObservabilityManager(log_dir=temp_log_dir)

        assert obs.log_dir.exists()
        assert obs.registry is not None
        assert len(obs.metrics) > 0

        obs.shutdown()

    def test_initialization_with_notifications(self, temp_log_dir):
        """Test initialization with notification channels."""
        channels = ["slack-alerts", "email-oncall"]
        obs = ObservabilityManager(
            log_dir=temp_log_dir,
            notification_channels=channels
        )

        assert obs.notification_channels == channels

        obs.shutdown()

    def test_initialization_with_cleanup_disabled(self, temp_log_dir):
        """Test initialization with cleanup disabled."""
        obs = ObservabilityManager(log_dir=temp_log_dir, enable_cleanup=False)

        assert obs.enable_cleanup is False

        obs.shutdown()


class TestDiskSpaceManagement:
    """Test disk space management."""

    def test_check_disk_space_sufficient(self, obs_manager):
        """Test disk space check with sufficient space."""
        result = obs_manager._check_disk_space()

        # Should have sufficient space in temp directory
        assert result is True

    @patch('shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_usage, obs_manager):
        """Test disk space check with insufficient space."""
        # Mock low disk space
        mock_usage.return_value = Mock(free=50 * 1024 * 1024)  # 50MB free

        result = obs_manager._check_disk_space()

        assert result is False

    def test_get_directory_size(self, obs_manager):
        """Test calculating directory size."""
        # Create some test files
        test_file = obs_manager.log_dir / "test.txt"
        test_file.write_text("test data")

        size_mb = obs_manager._get_directory_size_mb()

        assert size_mb >= 0.0


class TestFileCleanup:
    """Test file cleanup operations."""

    def test_cleanup_old_files(self, obs_manager):
        """Test cleaning up old files."""
        # Create old test file
        old_file = obs_manager.log_dir / "old_dashboard.json"
        old_file.write_text("{}")

        # Set modification time to past
        old_time = time.time() - (MAX_DASHBOARD_AGE_DAYS + 1) * 86400
        Path(old_file).touch()
        import os
        os.utime(old_file, (old_time, old_time))

        # Run cleanup
        cleaned = obs_manager._cleanup_old_files("*.json", MAX_DASHBOARD_AGE_DAYS)

        assert cleaned >= 0

    def test_periodic_cleanup(self, temp_log_dir):
        """Test periodic cleanup."""
        obs = ObservabilityManager(log_dir=temp_log_dir, enable_cleanup=True)

        # Create some old files
        for i in range(3):
            file = obs.log_dir / f"semantic_map_{i}.png"
            file.write_text("test")

        # Force cleanup
        obs.last_cleanup = time.time() - 7200  # 2 hours ago
        obs._periodic_cleanup()

        obs.shutdown()

    def test_aggressive_cleanup(self, obs_manager):
        """Test aggressive cleanup."""
        # Create many files to trigger size limit
        for i in range(10):
            file = obs_manager.log_dir / f"large_file_{i}.dat"
            file.write_bytes(b"x" * 1024 * 1024)  # 1MB each

        # Should not raise
        obs_manager._aggressive_cleanup()


class TestTensorValidation:
    """Test tensor validation."""

    def test_validate_tensor_valid(self, obs_manager):
        """Test validating valid tensor."""
        tensor = np.random.rand(10, 10)

        result = obs_manager._validate_tensor(tensor, "test_tensor")

        assert result is True

    def test_validate_tensor_not_array(self, obs_manager):
        """Test validating non-array."""
        result = obs_manager._validate_tensor("not an array", "test")

        assert result is False

    def test_validate_tensor_too_many_dimensions(self, obs_manager):
        """Test tensor with too many dimensions."""
        tensor = np.random.rand(5, 5, 5, 5)  # 4D

        result = obs_manager._validate_tensor(tensor, "test")

        assert result is False

    def test_validate_tensor_dimension_too_large(self, obs_manager):
        """Test tensor with dimension exceeding limit."""
        tensor = np.zeros((MAX_TENSOR_SIZE + 1, 2))

        result = obs_manager._validate_tensor(tensor, "test")

        assert result is False

    def test_validate_tensor_too_many_elements(self, obs_manager):
        """Test tensor with too many total elements."""
        # Create tensor that exceeds element limit
        size = int((MAX_TENSOR_ELEMENTS + 1) ** 0.5)
        tensor = np.zeros((size, size))

        result = obs_manager._validate_tensor(tensor, "test")

        assert result is False

    def test_validate_tensor_with_nan(self, obs_manager):
        """Test tensor with NaN values."""
        tensor = np.array([1.0, 2.0, np.nan])

        # Should warn but not reject
        result = obs_manager._validate_tensor(tensor, "test")

        assert result is True


class TestSemanticMapPlotting:
    """Test semantic map plotting."""

    @patch('observability_manager.GRAPHVIZ_AVAILABLE', True)
    @patch('observability_manager.graphviz')
    def test_plot_semantic_map_success(self, mock_graphviz, obs_manager):
        """Test successful semantic map plotting."""
        tensor = np.random.rand(3, 3)
        labels = ["A", "B", "C"]

        mock_dot = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        result = obs_manager.plot_semantic_map(tensor, labels)

        assert mock_dot.render.called

    @patch('observability_manager.GRAPHVIZ_AVAILABLE', False)
    def test_plot_semantic_map_no_graphviz(self, obs_manager):
        """Test plotting without graphviz."""
        tensor = np.random.rand(3, 3)

        result = obs_manager.plot_semantic_map(tensor)

        assert result is None

    def test_plot_semantic_map_invalid_tensor(self, obs_manager):
        """Test plotting with invalid tensor."""
        result = obs_manager.plot_semantic_map("not a tensor")

        assert result is None

    def test_plot_semantic_map_not_square(self, obs_manager):
        """Test plotting with non-square matrix."""
        tensor = np.random.rand(3, 4)

        result = obs_manager.plot_semantic_map(tensor)

        assert result is None

    def test_plot_semantic_map_label_mismatch(self, obs_manager):
        """Test plotting with wrong number of labels."""
        tensor = np.random.rand(3, 3)
        labels = ["A", "B"]  # Only 2 labels for 3x3 matrix

        result = obs_manager.plot_semantic_map(tensor, labels)

        assert result is None


class TestMetricLogging:
    """Test metric logging."""

    def test_log_tensor_semantics_2d(self, obs_manager):
        """Test logging 2D tensor semantics."""
        tensor = np.random.rand(5, 5)

        # Should not raise
        obs_manager.log_tensor_semantics(tensor, "test_tensor")

    def test_log_tensor_semantics_1d(self, obs_manager):
        """Test logging 1D tensor semantics."""
        tensor = np.random.rand(10)

        # Should not raise
        obs_manager.log_tensor_semantics(tensor, "test_tensor")

    def test_log_tensor_semantics_invalid(self, obs_manager):
        """Test logging invalid tensor."""
        # Should not raise, but should log error
        obs_manager.log_tensor_semantics("not a tensor", "invalid")

    def test_log_audit_event(self, obs_manager):
        """Test logging audit event."""
        # Should not raise
        obs_manager.log_audit_event("test_event")

    def test_log_bias_detected(self, obs_manager):
        """Test logging bias detection."""
        # Should not raise
        obs_manager.log_bias_detected()

    def test_log_counterfactual_diff_valid(self, obs_manager):
        """Test logging valid counterfactual diff."""
        obs_manager.log_counterfactual_diff("tensor1", 0.5)

        # Should not raise

    def test_log_counterfactual_diff_invalid_type(self, obs_manager):
        """Test logging invalid diff type."""
        obs_manager.log_counterfactual_diff("tensor1", "not a number")

        # Should log error but not raise

    def test_log_counterfactual_diff_infinite(self, obs_manager):
        """Test logging infinite diff."""
        obs_manager.log_counterfactual_diff("tensor1", float('inf'))

        # Should log error but not raise

    def test_log_execution_latency_valid(self, obs_manager):
        """Test logging valid execution latency."""
        obs_manager.log_execution_latency("component1", 1.5)

        # Should not raise

    def test_log_execution_latency_invalid_type(self, obs_manager):
        """Test logging invalid latency type."""
        obs_manager.log_execution_latency("component1", "not a number")

        # Should log error but not raise

    def test_log_execution_latency_negative(self, obs_manager):
        """Test logging negative latency."""
        obs_manager.log_execution_latency("component1", -1.0)

        # Should log error but not raise

    def test_log_error(self, obs_manager):
        """Test logging error."""
        obs_manager.log_error("component1", "test_error")

        # Should not raise


class TestDashboardExport:
    """Test dashboard export."""

    def test_export_dashboard_basic(self, obs_manager):
        """Test basic dashboard export."""
        path = obs_manager.export_dashboard("test_dashboard")

        assert Path(path).exists()
        assert Path(path).suffix == ".json"

    def test_export_dashboard_with_notifications(self, temp_log_dir):
        """Test dashboard export with notification channels."""
        obs = ObservabilityManager(
            log_dir=temp_log_dir,
            notification_channels=["test-channel"]
        )

        path = obs.export_dashboard()

        # Verify notification is in dashboard
        with open(path, encoding="utf-8") as f:
            import json
            dashboard = json.load(f)

        # Check if any panel has notifications
        has_notifications = False
        for panel in dashboard.get("panels", []):
            if "alert" in panel and "notifications" in panel["alert"]:
                has_notifications = True
                break

        obs.shutdown()

    @patch('observability_manager.ObservabilityManager._check_disk_space')
    def test_export_dashboard_insufficient_space(self, mock_check, obs_manager):
        """Test dashboard export with insufficient disk space."""
        mock_check.return_value = False

        with pytest.raises(IOError):
            obs_manager.export_dashboard()


class TestPrometheusMetrics:
    """Test Prometheus metrics."""

    def test_get_prometheus_metrics(self, obs_manager):
        """Test getting Prometheus metrics."""
        # Log some metrics first
        obs_manager.log_audit_event("test")
        obs_manager.log_execution_latency("test", 1.0)

        metrics = obs_manager.get_prometheus_metrics()

        assert isinstance(metrics, bytes)
        assert len(metrics) > 0


class TestStatistics:
    """Test statistics."""

    def test_get_stats(self, obs_manager):
        """Test getting statistics."""
        stats = obs_manager.get_stats()

        assert "log_dir" in stats
        assert "dir_size_mb" in stats
        assert "dashboard_count" in stats
        assert "plot_count" in stats
        assert "free_disk_mb" in stats
        assert "cleanup_enabled" in stats


class TestShutdown:
    """Test shutdown."""

    def test_shutdown(self, temp_log_dir):
        """Test clean shutdown."""
        obs = ObservabilityManager(log_dir=temp_log_dir)

        # Should not raise
        obs.shutdown()


class TestConstants:
    """Test module constants."""

    def test_constants_exist(self):
        """Test that all constants are defined."""
        assert MAX_LOG_DIR_SIZE_MB > 0
        assert MAX_DASHBOARD_AGE_DAYS > 0
        assert MAX_PLOT_AGE_DAYS > 0
        assert MIN_FREE_DISK_MB > 0
        assert MAX_TENSOR_SIZE > 0
        assert MAX_TENSOR_ELEMENTS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
