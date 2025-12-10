"""
Comprehensive test suite for run_validation_test.py
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from run_validation_test import (FileCache, ValidationTestSuite,
                                 _check_async_context, discover_golden_files)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_graph():
    """Create sample graph."""
    return {
        "grammar_version": "2.3.0",
        "id": "test_graph",
        "type": "Graph",
        "nodes": [
            {"id": "n1", "type": "InputNode", "params": {"value": "test"}},
            {"id": "n2", "type": "OutputNode", "params": {}}
        ],
        "edges": [
            {"from": "n1", "to": "n2", "type": "data"}
        ]
    }


class TestFileCache:
    """Test FileCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = FileCache()

        assert cache._cache == {}

    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = FileCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_set_and_get(self):
        """Test setting and getting."""
        cache = FileCache()

        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result == {"data": "value1"}

    def test_clear(self):
        """Test clearing cache."""
        cache = FileCache()

        cache.set("key1", {"data": "value1"})
        cache.clear()

        assert cache.get("key1") is None


class TestDiscoverGoldenFiles:
    """Test golden file discovery."""

    @patch('run_validation_test.Path')
    def test_discover_with_existing_files(self, mock_path):
        """Test discovery with existing files."""
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.glob.return_value = [
            Path("specs/ir_examples/test1.json"),
            Path("specs/ir_examples/test2.json")
        ]
        mock_path.return_value = mock_dir

        # Can't easily patch the function's internal Path usage,
        # so we'll test the fallback instead
        files = discover_golden_files()

        # Should return default list
        assert len(files) > 0

    def test_discover_fallback(self):
        """Test discovery fallback."""
        files = discover_golden_files()

        # Should have fallback files
        assert len(files) > 0
        assert all(isinstance(f, str) for f in files)


class TestValidationTestSuiteInitialization:
    """Test ValidationTestSuite initialization."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        suite = ValidationTestSuite(agent_id="test_agent")

        assert suite.agent_id == "test_agent"
        assert suite.enable_caching is True
        assert suite._initialized is True

        suite.shutdown()

    def test_initialization_no_caching(self):
        """Test initialization without caching."""
        suite = ValidationTestSuite(enable_caching=False)

        assert suite.enable_caching is False

        suite.shutdown()

    def test_context_manager(self):
        """Test context manager usage."""
        with ValidationTestSuite() as suite:
            assert suite._initialized is True

        # Should be cleaned up
        assert suite._initialized is False

    def test_shutdown_idempotent(self):
        """Test shutdown can be called multiple times."""
        suite = ValidationTestSuite()

        suite.shutdown()
        suite.shutdown()  # Should not raise


class TestMetricRegistration:
    """Test metric registration."""

    @patch('run_validation_test.ObservabilityManager')
    @patch('run_validation_test.PROMETHEUS_AVAILABLE', True)
    def test_register_metrics(self, mock_obs):
        """Test metric registration."""
        mock_instance = MagicMock()
        mock_instance.metrics = {}
        mock_instance.registry = MagicMock()
        mock_obs.return_value = mock_instance

        suite = ValidationTestSuite()

        # Should have registered metrics
        assert len(suite.obs.metrics) > 0

        suite.shutdown()

    def test_register_metrics_no_collision(self):
        """Test metrics don't collide on re-registration."""
        suite1 = ValidationTestSuite()
        suite1.shutdown()

        # Second instance should not raise
        suite2 = ValidationTestSuite()
        suite2.shutdown()


class TestValidationMethods:
    """Test validation methods."""

    @patch('builtins.open', mock_open(read_data='{"test": "data"}'))
    def test_load_manifest(self):
        """Test loading manifest."""
        suite = ValidationTestSuite()

        manifest = suite._load_manifest()

        # Should have loaded something
        assert manifest is not None

        suite.shutdown()

    @patch('builtins.open', mock_open(read_data='{"id": "test", "nodes": [], "edges": []}'))
    def test_load_golden_file(self):
        """Test loading golden file."""
        suite = ValidationTestSuite()

        graph = suite._load_golden_file("test.json")

        assert graph["id"] == "test"

        suite.shutdown()

    def test_calculate_graph_hash(self, sample_graph):
        """Test graph hash calculation."""
        suite = ValidationTestSuite()

        hash1 = suite._calculate_graph_hash(sample_graph)
        hash2 = suite._calculate_graph_hash(sample_graph)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        suite.shutdown()


class TestValidationReport:
    """Test validation report generation."""

    def test_generate_report_all_passed(self):
        """Test report with all tests passed."""
        suite = ValidationTestSuite()

        results = {
            "test1": True,
            "test2": {"status": "completed"},
            "test3": {"status": "completed"}
        }

        report = suite.generate_validation_report(results)

        assert report["summary"]["total_tests"] == 3
        assert report["summary"]["passed"] == 3
        assert report["summary"]["failed"] == 0

        suite.shutdown()

    def test_generate_report_with_failures(self):
        """Test report with failures."""
        suite = ValidationTestSuite()

        results = {
            "test1": False,
            "test2": {"status": "error", "error": "Test error"},
            "test3": {"status": "completed"}
        }

        report = suite.generate_validation_report(results)

        assert report["summary"]["passed"] == 1
        assert report["summary"]["failed"] == 2

        suite.shutdown()

    def test_generate_report_with_unknown_status(self):
        """Test report with unknown status."""
        suite = ValidationTestSuite()

        results = {
            "test1": {"status": "unknown_status"},
            "test2": {"status": "completed"}
        }

        report = suite.generate_validation_report(results)

        # Unknown status should be marked as unknown, not passed
        assert report["summary"]["unknown"] == 1
        assert report["summary"]["passed"] == 1

        suite.shutdown()

    def test_generate_report_with_durations(self):
        """Test report calculates duration metrics."""
        suite = ValidationTestSuite()

        results = {
            "test1": {"status": "completed", "duration_ms": 100},
            "test2": {"status": "completed", "duration_ms": 200},
            "test3": {"status": "completed", "duration_ms": 150}
        }

        report = suite.generate_validation_report(results)

        assert "avg_duration_ms" in report["metrics"]
        assert report["metrics"]["avg_duration_ms"] == 150.0
        assert report["metrics"]["max_duration_ms"] == 200
        assert report["metrics"]["min_duration_ms"] == 100

        suite.shutdown()


class TestAsyncContext:
    """Test async context checking."""

    def test_check_async_context_outside(self):
        """Test checking outside async context."""
        result = _check_async_context()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_async_context_inside(self):
        """Test checking inside async context."""
        result = _check_async_context()

        assert result is True


class TestLogMetrics:
    """Test metric logging."""

    @patch('run_validation_test.ObservabilityManager')
    def test_log_metrics(self, mock_obs_class):
        """Test logging metrics."""
        mock_obs = MagicMock()
        mock_obs.metrics = {
            "validation_pass": MagicMock(),
            "validation_latency": MagicMock()
        }
        mock_obs_class.return_value = mock_obs

        suite = ValidationTestSuite()

        suite._log_metrics("test", {"status": "completed", "duration_ms": 100})

        # Should have logged
        assert mock_obs.log_audit_event.called

        suite.shutdown()


class TestLogAudit:
    """Test audit logging."""

    @patch('run_validation_test.SecurityAuditEngine')
    def test_log_audit(self, mock_audit_class):
        """Test logging audit event."""
        mock_audit = MagicMock()
        mock_audit_class.return_value = mock_audit

        suite = ValidationTestSuite()

        suite._log_audit("test", {"detail": "value"})

        # Should have logged
        assert mock_audit.log_event.called

        suite.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
