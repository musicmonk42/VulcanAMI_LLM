"""
Comprehensive test suite for generate_transparency_report.py
"""

import json
import os
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from generate_transparency_report import (
    ANOMALY_THRESHOLD_STDDEV,
    ARCHIVE_PATH,
    MAX_ARCHIVE_COUNT,
    MAX_REPORT_SIZE,
    REPORT_PATH,
    FileLock,
    MetricHistory,
    TransparencyReportValidator,
    append_report,
    check_for_anomalies,
    cleanup_old_archives,
    ensure_archive,
    fetch_audit_metrics,
    fetch_bias_taxonomy,
    fetch_interpretability_metrics,
    generate_audit_table,
    generate_interpretability_table,
    get_previous_report_metrics,
)


@pytest.fixture
def temp_report_file():
    """Create temporary report file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_archive_dir():
    """Create temporary archive directory."""
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def validator():
    """Create validator."""
    return TransparencyReportValidator()


class TestMetricHistory:
    """Test MetricHistory dataclass."""

    def test_initialization(self):
        """Test history initialization."""
        history = MetricHistory()

        assert len(history.values) == 0
        assert len(history.timestamps) == 0

    def test_add_value(self):
        """Test adding values."""
        import datetime

        history = MetricHistory()

        now = datetime.datetime.utcnow()
        history.add(0.5, now)

        assert len(history.values) == 1
        assert history.values[0] == 0.5

    def test_has_enough_samples(self):
        """Test sample count check."""
        import datetime

        history = MetricHistory()

        assert not history.has_enough_samples()

        for i in range(10):
            history.add(float(i), datetime.datetime.utcnow())

        assert history.has_enough_samples()

    def test_get_mean(self):
        """Test mean calculation."""
        import datetime

        history = MetricHistory()

        assert history.get_mean() is None

        for i in range(5):
            history.add(float(i), datetime.datetime.utcnow())

        mean = history.get_mean()
        assert mean == 2.0

    def test_get_stddev(self):
        """Test standard deviation."""
        import datetime

        history = MetricHistory()

        assert history.get_stddev() is None

        history.add(1.0, datetime.datetime.utcnow())
        assert history.get_stddev() is None  # Need at least 2

        for i in range(2, 6):
            history.add(float(i), datetime.datetime.utcnow())

        stddev = history.get_stddev()
        assert stddev is not None
        assert stddev > 0


class TestTransparencyReportValidator:
    """Test TransparencyReportValidator class."""

    def test_validate_metrics_valid(self, validator):
        """Test validating valid metrics."""
        metrics = {"metric1": 0.5, "metric2": 100, "metric3": "value"}

        valid, error = validator.validate_metrics(metrics)

        assert valid is True
        assert error is None

    def test_validate_metrics_not_dict(self, validator):
        """Test validating non-dict metrics."""
        valid, error = validator.validate_metrics([1, 2, 3])

        assert valid is False
        assert "must be dict" in error

    def test_validate_metrics_invalid_key(self, validator):
        """Test validating metrics with invalid key."""
        metrics = {123: "value"}

        valid, error = validator.validate_metrics(metrics)

        assert valid is False
        assert "key must be string" in error

    def test_validate_metrics_invalid_value(self, validator):
        """Test validating metrics with invalid value."""
        metrics = {"key": {"nested": "dict"}}

        valid, error = validator.validate_metrics(metrics)

        assert valid is False
        assert "must be numeric or string" in error

    def test_validate_audit_record(self, validator):
        """Test validating audit record."""
        record = {"field1": "value1", "field2": 123}

        valid, error = validator.validate_audit_record(record)

        assert valid is True

    def test_validate_json_line_valid(self, validator):
        """Test validating valid JSON line."""
        line = '{"key": "value", "number": 123}'

        valid, error, data = validator.validate_json_line(line)

        assert valid is True
        assert error is None
        assert data == {"key": "value", "number": 123}

    def test_validate_json_line_invalid(self, validator):
        """Test validating invalid JSON."""
        line = "{invalid json}"

        valid, error, data = validator.validate_json_line(line)

        assert valid is False
        assert "Invalid JSON" in error
        assert data is None


class TestFileLock:
    """Test FileLock class."""

    def test_file_lock_context_manager(self, temp_report_file):
        """Test file lock as context manager."""
        with FileLock(temp_report_file):
            # Lock is acquired
            assert temp_report_file.parent.exists()

        # Lock is released


class TestArchiveManagement:
    """Test archive management functions."""

    @patch("generate_transparency_report.ARCHIVE_PATH")
    def test_ensure_archive(self, mock_path, temp_archive_dir):
        """Test ensure archive directory."""
        mock_path.return_value = temp_archive_dir / "test_archive"
        mock_path.mkdir = Mock()

        ensure_archive()

        # Should attempt to create directory
        mock_path.mkdir.assert_called()

    @patch("generate_transparency_report.ARCHIVE_PATH")
    def test_cleanup_old_archives(self, mock_path, temp_archive_dir):
        """Test cleaning up old archives."""
        mock_path.return_value = temp_archive_dir
        mock_path.exists = Mock(return_value=True)
        mock_path.glob = Mock(return_value=[])

        cleanup_old_archives()

        # Should check if path exists
        mock_path.exists.assert_called()


class TestMetricFetching:
    """Test metric fetching functions."""

    @patch("generate_transparency_report.REPORT_PATH")
    def test_get_previous_report_metrics(self, mock_path, temp_report_file):
        """Test getting previous report metrics."""
        # Write sample report
        with open(temp_report_file, "w", encoding="utf-8") as f:
            f.write("| Metric | Value |\n")
            f.write("|---|---|\n")
            f.write("| Test Metric | 0.5 |\n")

        mock_path.return_value = temp_report_file
        mock_path.exists = Mock(return_value=True)

        # Mock open to return our temp file
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = open(
                temp_report_file, "r", encoding="utf-8"
            )

            metrics = get_previous_report_metrics()

        assert isinstance(metrics, dict)

    @patch("generate_transparency_report.OBSERVABILITY_AVAILABLE", True)
    @patch("generate_transparency_report.get_prometheus_metrics")
    def test_fetch_interpretability_metrics(self, mock_get_metrics):
        """Test fetching interpretability metrics."""
        mock_get_metrics.return_value = {"shap_coverage": 0.8}

        metrics = fetch_interpretability_metrics()

        assert isinstance(metrics, dict)

    @patch("generate_transparency_report.AUDIT_LOG_PATH")
    def test_fetch_audit_metrics(self, mock_path, temp_report_file):
        """Test fetching audit metrics."""
        # Write sample audit log
        with open(temp_report_file, "w", encoding="utf-8") as f:
            f.write('{"proposal_id": "test", "bias_detected": true}\n')
            f.write('{"proposal_id": "test2", "bias_detected": false}\n')

        mock_path.return_value = str(temp_report_file)

        with patch("os.path.exists", return_value=True):
            with patch("os.access", return_value=True):
                proposal_count, bias_count, recent = fetch_audit_metrics()

        assert proposal_count >= 0
        assert bias_count >= 0
        assert isinstance(recent, list)

    @patch("generate_transparency_report.NSO_ALIGNER_AVAILABLE", True)
    @patch("generate_transparency_report.get_bias_taxonomy")
    def test_fetch_bias_taxonomy(self, mock_get_taxonomy):
        """Test fetching bias taxonomy."""
        mock_get_taxonomy.return_value = {"bias_type_1": 5}

        taxonomy = fetch_bias_taxonomy()

        assert isinstance(taxonomy, dict)


class TestTableGeneration:
    """Test table generation functions."""

    def test_generate_interpretability_table(self):
        """Test generating interpretability table."""
        metrics = {"metric1": 0.5, "metric2": 0.8}
        prev = {"Metric1": 0.4}

        table = generate_interpretability_table(metrics, prev)

        assert isinstance(table, str)
        assert "metric1" in table.lower() or "Metric1" in table

    def test_generate_interpretability_table_empty(self):
        """Test generating table with no metrics."""
        table = generate_interpretability_table({}, {})

        assert "No interpretability metric" in table

    def test_generate_audit_table(self):
        """Test generating audit table."""
        table = generate_audit_table(100, 10, {})

        assert isinstance(table, str)
        assert "100" in table
        assert "10" in table


class TestAnomalyDetection:
    """Test anomaly detection."""

    def test_check_for_anomalies_no_history(self):
        """Test anomaly detection without history."""
        metrics = {"metric1": 0.5}
        prev = {}

        anomalies = check_for_anomalies(metrics, prev)

        # Should handle gracefully
        assert isinstance(anomalies, list)

    def test_check_for_anomalies_statistical(self):
        """Test statistical anomaly detection."""
        import datetime

        # Build up history
        history = {"metric1": MetricHistory()}

        for i in range(10):
            history["metric1"].add(1.0, datetime.datetime.utcnow())

        # Now add anomalous value
        metrics = {"metric1": 10.0}  # Much higher than history
        prev = {}

        anomalies = check_for_anomalies(metrics, prev, history)

        # Should detect anomaly
        assert len(anomalies) > 0

    def test_check_for_anomalies_insufficient_history(self):
        """Test anomaly detection with insufficient history."""
        import datetime

        history = {"metric1": MetricHistory()}

        # Add only 2 values (not enough for stddev)
        history["metric1"].add(1.0, datetime.datetime.utcnow())
        history["metric1"].add(1.1, datetime.datetime.utcnow())

        metrics = {"metric1": 5.0}
        prev = {"metric1": 1.0}

        anomalies = check_for_anomalies(metrics, prev, history)

        # Should use fallback method
        assert isinstance(anomalies, list)


class TestReportAppending:
    """Test report appending function."""

    def test_append_report_new_file(self, temp_report_file):
        """Test appending to new file."""
        # Remove temp file to simulate new file
        if temp_report_file.exists():
            temp_report_file.unlink()

        with patch("generate_transparency_report.REPORT_PATH", temp_report_file):
            with patch("generate_transparency_report.ensure_archive"):
                append_report("Test content")

        # File should be created
        assert temp_report_file.exists()

        # Content should be written
        content = temp_report_file.read_text()
        assert "Test content" in content

    def test_append_report_size_limit(self, temp_report_file):
        """Test size limit handling."""
        # Write large file
        with open(temp_report_file, "w", encoding="utf-8") as f:
            f.write("x" * (MAX_REPORT_SIZE + 1))

        # Mock archive_current_report to track if it was called
        with patch("generate_transparency_report.REPORT_PATH", temp_report_file):
            with patch(
                "generate_transparency_report.archive_current_report"
            ) as mock_archive:
                with patch("generate_transparency_report.ensure_archive"):
                    append_report("New content")

        # Should trigger archive due to size
        mock_archive.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
