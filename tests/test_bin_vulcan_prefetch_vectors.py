"""
Tests for vulcan-prefetch-vectors Python script

This version includes Windows compatibility fixes.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
VULCAN_PREFETCH = os.path.join(BIN_DIR, "vulcan-prefetch-vectors")


def run_vulcan_prefetch(args, **kwargs):
    """
    Helper function to run vulcan-prefetch-vectors with proper platform-specific handling.

    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    """
    if platform.system() == "Windows":
        command = [sys.executable, VULCAN_PREFETCH] + args
    else:
        command = [VULCAN_PREFETCH] + args

    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)

    return subprocess.run(command, **kwargs)


class TestVulcanPrefetchVectors:
    """Test suite for vulcan-prefetch-vectors"""

    def test_prefetch_exists(self):
        """Test that vulcan-prefetch-vectors exists and is executable"""
        assert os.path.exists(VULCAN_PREFETCH)
        if platform.system() != "Windows":
            assert os.access(VULCAN_PREFETCH, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_prefetch(["--help"])
        assert result.returncode == 0
        assert "VulcanAMI Vector Prefetch" in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_prefetch(["--help"])
        assert "4.6.0" in result.stdout

    def test_prefetch_basic(self):
        """Test basic prefetching"""
        result = run_vulcan_prefetch(["query-123"], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Prefetch" in output or "query" in output.lower() or "SUCCESS" in output

    def test_prefetch_hot_tier(self):
        """Test prefetching from hot tier"""
        result = run_vulcan_prefetch(["query-123", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_warm_tier(self):
        """Test prefetching from warm tier"""
        result = run_vulcan_prefetch(["query-123", "--tier", "warm"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_cold_tier(self):
        """Test prefetching from cold tier"""
        result = run_vulcan_prefetch(["query-123", "--tier", "cold"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_with_top_k(self):
        """Test custom top-k parameter"""
        for k in [10, 50, 100, 500]:
            result = run_vulcan_prefetch(["query-123", "--top-k", str(k)], timeout=30)
            assert result.returncode == 0

    def test_prefetch_ml_predicted_strategy(self):
        """Test ML predicted strategy"""
        result = run_vulcan_prefetch(
            ["query-123", "--strategy", "ml_predicted"], timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_popularity_strategy(self):
        """Test popularity strategy"""
        result = run_vulcan_prefetch(
            ["query-123", "--strategy", "popularity"], timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_recent_strategy(self):
        """Test recent strategy"""
        result = run_vulcan_prefetch(["query-123", "--strategy", "recent"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_prefetch(["query-123", "--verbose"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_prefetch(["query-123", "--quiet"], timeout=30)
        assert result.returncode == 0

    def test_prefetch_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "prefetch.json")

            result = run_vulcan_prefetch(
                ["query-123", "--json", json_output], timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

            # Verify JSON structure
            with open(json_output, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "query_id" in data
                assert "tier" in data
                assert "vectors_prefetched" in data
                assert "success" in data

    def test_prefetch_displays_summary(self):
        """Test that prefetch displays summary"""
        result = run_vulcan_prefetch(["query-123"], timeout=30)

        output = result.stdout + result.stderr
        assert "Query" in output or "Vectors" in output or "Summary" in output

    def test_prefetch_invalid_tier(self):
        """Test invalid tier is rejected"""
        result = run_vulcan_prefetch(["query-123", "--tier", "invalid"], timeout=30)
        assert result.returncode != 0

    def test_prefetch_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = run_vulcan_prefetch(["query-123", "--strategy", "invalid"], timeout=30)
        assert result.returncode != 0

    def test_prefetch_no_query_shows_help(self):
        """Test running without query ID shows help"""
        result = run_vulcan_prefetch([], timeout=30)
        assert result.returncode != 0

    def test_prefetch_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "result.json")

            result = run_vulcan_prefetch(
                [
                    "query-123",
                    "--tier",
                    "hot",
                    "--top-k",
                    "200",
                    "--strategy",
                    "ml_predicted",
                    "--verbose",
                    "--json",
                    json_output,
                ],
                timeout=30,
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_prefetch_multiple_queries(self):
        """Test prefetching for multiple queries"""
        queries = ["query-001", "query-002", "query-003"]
        for query in queries:
            result = run_vulcan_prefetch([query, "--quiet"], timeout=30)
            assert result.returncode == 0
