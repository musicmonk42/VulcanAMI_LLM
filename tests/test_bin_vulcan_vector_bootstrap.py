"""
Tests for vulcan-vector-bootstrap Python script
"""
import subprocess
import pytest
import os
import tempfile
import json


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_BOOTSTRAP = os.path.join(BIN_DIR, 'vulcan-vector-bootstrap')


class TestVulcanVectorBootstrap:
    """Test suite for vulcan-vector-bootstrap"""

    def test_bootstrap_exists(self):
        """Test that vulcan-vector-bootstrap exists and is executable"""
        assert os.path.exists(VULCAN_BOOTSTRAP)
        assert os.access(VULCAN_BOOTSTRAP, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Vector Bootstrap' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_bootstrap_default(self):
        """Test default bootstrap (all tiers)"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Bootstrap' in output or 'collection' in output.lower() or 'SUCCESS' in output

    def test_bootstrap_all_tiers(self):
        """Test bootstrapping all tiers"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'all'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_hot_tier(self):
        """Test bootstrapping hot tier"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_warm_tier(self):
        """Test bootstrapping warm tier"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'warm'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_cold_tier(self):
        """Test bootstrapping cold tier"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'cold'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_dimension(self):
        """Test custom dimension parameter"""
        for dim in [64, 128, 256, 512]:
            result = subprocess.run(
                [VULCAN_BOOTSTRAP, '--dimension', str(dim), '--tier', 'hot'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_bootstrap_with_l2_metric(self):
        """Test L2 distance metric"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--metric', 'L2', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_ip_metric(self):
        """Test IP (inner product) metric"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--metric', 'IP', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_cosine_metric(self):
        """Test COSINE metric"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--metric', 'COSINE', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_flat_index(self):
        """Test FLAT index type"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--index-type', 'FLAT', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_ivf_flat_index(self):
        """Test IVF_FLAT index type"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--index-type', 'IVF_FLAT', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_ivf_sq8_index(self):
        """Test IVF_SQ8 index type"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--index-type', 'IVF_SQ8', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_hnsw_index(self):
        """Test HNSW index type"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--index-type', 'HNSW', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_drop_existing(self):
        """Test --drop-existing flag"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--drop-existing', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_verbose_mode(self):
        """Test verbose mode"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--verbose', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_quiet_mode(self):
        """Test quiet mode"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--quiet', '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'bootstrap.json')
            
            result = subprocess.run(
                [VULCAN_BOOTSTRAP, '--tier', 'hot', '--json', json_output],
                capture_output=True,
                text=True,
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
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'hot'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'Bootstrap' in output or 'Collection' in output or 'Summary' in output

    def test_bootstrap_invalid_tier(self):
        """Test invalid tier is rejected"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--tier', 'invalid'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_bootstrap_invalid_metric(self):
        """Test invalid metric is rejected"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--metric', 'INVALID'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_bootstrap_invalid_index_type(self):
        """Test invalid index type is rejected"""
        result = subprocess.run(
            [VULCAN_BOOTSTRAP, '--index-type', 'INVALID'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_bootstrap_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'result.json')
            
            result = subprocess.run(
                [VULCAN_BOOTSTRAP,
                 '--tier', 'hot',
                 '--dimension', '256',
                 '--metric', 'COSINE',
                 '--index-type', 'HNSW',
                 '--verbose',
                 '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_bootstrap_multiple_tiers_sequentially(self):
        """Test bootstrapping multiple tiers"""
        tiers = ['hot', 'warm', 'cold']
        for tier in tiers:
            result = subprocess.run(
                [VULCAN_BOOTSTRAP, '--tier', tier, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
