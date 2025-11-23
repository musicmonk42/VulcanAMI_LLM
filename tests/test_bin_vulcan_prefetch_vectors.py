"""
Tests for vulcan-prefetch-vectors Python script
"""
import subprocess
import pytest
import os
import tempfile
import json


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PREFETCH = os.path.join(BIN_DIR, 'vulcan-prefetch-vectors')


class TestVulcanPrefetchVectors:
    """Test suite for vulcan-prefetch-vectors"""

    def test_prefetch_exists(self):
        """Test that vulcan-prefetch-vectors exists and is executable"""
        assert os.path.exists(VULCAN_PREFETCH)
        assert os.access(VULCAN_PREFETCH, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_PREFETCH, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Vector Prefetch' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_PREFETCH, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_prefetch_basic(self):
        """Test basic prefetching"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Prefetch' in output or 'query' in output.lower() or 'SUCCESS' in output

    def test_prefetch_hot_tier(self):
        """Test prefetching from hot tier"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_warm_tier(self):
        """Test prefetching from warm tier"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--tier', 'warm'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_cold_tier(self):
        """Test prefetching from cold tier"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--tier', 'cold'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_with_top_k(self):
        """Test custom top-k parameter"""
        for k in [10, 50, 100, 500]:
            result = subprocess.run(
                [VULCAN_PREFETCH, 'query-123', '--top-k', str(k)],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_prefetch_ml_predicted_strategy(self):
        """Test ML predicted strategy"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--strategy', 'ml_predicted'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_popularity_strategy(self):
        """Test popularity strategy"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--strategy', 'popularity'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_recent_strategy(self):
        """Test recent strategy"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--strategy', 'recent'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_verbose_mode(self):
        """Test verbose mode"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--verbose'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_quiet_mode(self):
        """Test quiet mode"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_prefetch_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'prefetch.json')
            
            result = subprocess.run(
                [VULCAN_PREFETCH, 'query-123', '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)
            
            # Verify JSON structure
            with open(json_output, 'r') as f:
                data = json.load(f)
                assert 'query_id' in data
                assert 'tier' in data
                assert 'vectors_prefetched' in data
                assert 'success' in data

    def test_prefetch_displays_summary(self):
        """Test that prefetch displays summary"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'Query' in output or 'Vectors' in output or 'Summary' in output

    def test_prefetch_invalid_tier(self):
        """Test invalid tier is rejected"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--tier', 'invalid'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_prefetch_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = subprocess.run(
            [VULCAN_PREFETCH, 'query-123', '--strategy', 'invalid'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_prefetch_no_query_shows_help(self):
        """Test running without query ID shows help"""
        result = subprocess.run(
            [VULCAN_PREFETCH],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_prefetch_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'result.json')
            
            result = subprocess.run(
                [VULCAN_PREFETCH, 'query-123',
                 '--tier', 'hot',
                 '--top-k', '200',
                 '--strategy', 'ml_predicted',
                 '--verbose',
                 '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_prefetch_multiple_queries(self):
        """Test prefetching for multiple queries"""
        queries = ['query-001', 'query-002', 'query-003']
        for query in queries:
            result = subprocess.run(
                [VULCAN_PREFETCH, query, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
