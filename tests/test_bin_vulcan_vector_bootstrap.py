"""
Tests for vulcan-vector-bootstrap Python script

This version includes Windows compatibility fixes.
"""
import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_BOOTSTRAP = os.path.join(BIN_DIR, 'vulcan-vector-bootstrap')


def run_vulcan_bootstrap(args, **kwargs):
    """
    Helper function to run vulcan-vector-bootstrap with proper platform-specific handling.

    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    """
    if platform.system() == 'Windows':
        command = [sys.executable, VULCAN_BOOTSTRAP] + args
    else:
        command = [VULCAN_BOOTSTRAP] + args

    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)

    return subprocess.run(command, **kwargs)


class TestVulcanVectorBootstrap:
    """Test suite for vulcan-vector-bootstrap"""

    def test_bootstrap_exists(self):
        """Test that vulcan-vector-bootstrap exists and is executable"""
        assert os.path.exists(VULCAN_BOOTSTRAP)
        if platform.system() != 'Windows':
            assert os.access(VULCAN_BOOTSTRAP, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_bootstrap(['--help'])
        assert result.returncode == 0
        assert 'VulcanAMI Vector Bootstrap' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_bootstrap(['--help'])
        assert '4.6.0' in result.stdout

    def test_bootstrap_default(self):
        """Test default bootstrap (all tiers)"""
        result = run_vulcan_bootstrap([], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Bootstrap' in output or 'collection' in output.lower() or 'SUCCESS' in output

    def test_bootstrap_all_tiers(self):
        """Test bootstrapping all tiers"""
        result = run_vulcan_bootstrap(['--tier', 'all'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_hot_tier(self):
        """Test bootstrapping hot tier"""
        result = run_vulcan_bootstrap(['--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_warm_tier(self):
        """Test bootstrapping warm tier"""
        result = run_vulcan_bootstrap(['--tier', 'warm'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_cold_tier(self):
        """Test bootstrapping cold tier"""
        result = run_vulcan_bootstrap(['--tier', 'cold'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_dimension(self):
        """Test custom dimension parameter"""
        for dim in [64, 128, 256, 512]:
            result = run_vulcan_bootstrap(
                ['--dimension', str(dim), '--tier', 'hot'],
                timeout=30
            )
            assert result.returncode == 0

    def test_bootstrap_with_l2_metric(self):
        """Test L2 distance metric"""
        result = run_vulcan_bootstrap(['--metric', 'L2', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_ip_metric(self):
        """Test IP (inner product) metric"""
        result = run_vulcan_bootstrap(['--metric', 'IP', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_cosine_metric(self):
        """Test COSINE metric"""
        result = run_vulcan_bootstrap(['--metric', 'COSINE', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_flat_index(self):
        """Test FLAT index type"""
        result = run_vulcan_bootstrap(['--index-type', 'FLAT', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_ivf_flat_index(self):
        """Test IVF_FLAT index type"""
        result = run_vulcan_bootstrap(['--index-type', 'IVF_FLAT', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_ivf_sq8_index(self):
        """Test IVF_SQ8 index type"""
        result = run_vulcan_bootstrap(['--index-type', 'IVF_SQ8', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_hnsw_index(self):
        """Test HNSW index type"""
        result = run_vulcan_bootstrap(['--index-type', 'HNSW', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_drop_existing(self):
        """Test --drop-existing flag"""
        result = run_vulcan_bootstrap(['--drop-existing', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_bootstrap(['--verbose', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_bootstrap(['--quiet', '--tier', 'hot'], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'bootstrap.json')

            result = run_vulcan_bootstrap(
                ['--tier', 'hot', '--json', json_output],
                timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

            # Verify JSON structure
            with open(json_output, 'r') as f:
                data = json.load(f)
                assert 'collections' in data
                assert 'bootstrap_time' in data
                assert 'success' in data

    def test_bootstrap_displays_summary(self):
        """Test that bootstrap displays summary"""
        result = run_vulcan_bootstrap(['--tier', 'hot'], timeout=30)

        output = result.stdout + result.stderr
        assert 'Bootstrap' in output or 'Collection' in output or 'Summary' in output

    def test_bootstrap_invalid_tier(self):
        """Test invalid tier is rejected"""
        result = run_vulcan_bootstrap(['--tier', 'invalid'], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_invalid_metric(self):
        """Test invalid metric is rejected"""
        result = run_vulcan_bootstrap(['--metric', 'INVALID'], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_invalid_index_type(self):
        """Test invalid index type is rejected"""
        result = run_vulcan_bootstrap(['--index-type', 'INVALID'], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'result.json')

            result = run_vulcan_bootstrap(
                ['--tier', 'hot', '--dimension', '256', '--metric', 'COSINE',
                 '--index-type', 'HNSW', '--verbose', '--json', json_output],
                timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_bootstrap_multiple_tiers_sequentially(self):
        """Test bootstrapping multiple tiers"""
        tiers = ['hot', 'warm', 'cold']
        for tier in tiers:
            result = run_vulcan_bootstrap(['--tier', tier, '--quiet'], timeout=30)
            assert result.returncode == 0
