"""
Tests for vulcan-repack Python script
"""
import subprocess
import pytest
import os
import tempfile
import json


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_REPACK = os.path.join(BIN_DIR, 'vulcan-repack')


class TestVulcanRepack:
    """Test suite for vulcan-repack"""

    def test_repack_exists(self):
        """Test that vulcan-repack exists and is executable"""
        assert os.path.exists(VULCAN_REPACK)
        assert os.access(VULCAN_REPACK, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_REPACK, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Repack' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_REPACK, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_repack_basic(self):
        """Test basic repacking"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack-001'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Repack' in output or 'SUCCESS' in output

    def test_repack_adaptive_strategy(self):
        """Test adaptive repacking strategy"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--strategy', 'adaptive'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_aggressive_strategy(self):
        """Test aggressive repacking strategy"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--strategy', 'aggressive'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_conservative_strategy(self):
        """Test conservative repacking strategy"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--strategy', 'conservative'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_with_compression_level(self):
        """Test custom compression level"""
        for level in [1, 6, 12]:
            result = subprocess.run(
                [VULCAN_REPACK, 'test-pack', '--compression', str(level)],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_repack_with_output_path(self):
        """Test custom output path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'repacked.pack')
            result = subprocess.run(
                [VULCAN_REPACK, 'test-pack', '--output', output_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_repack_dry_run_mode(self):
        """Test dry run mode"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--dry-run'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_verbose_mode(self):
        """Test verbose mode"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--verbose'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_quiet_mode(self):
        """Test quiet mode"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_repack_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'repack.json')
            
            result = subprocess.run(
                [VULCAN_REPACK, 'test-pack', '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)
            
            # Verify JSON structure
            with open(json_output, 'r') as f:
                data = json.load(f)
                assert 'pack_id' in data
                assert 'strategy' in data
                assert 'success' in data

    def test_repack_displays_summary(self):
        """Test that repack displays summary"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'Pack' in output or 'Summary' in output or 'Size' in output

    def test_repack_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = subprocess.run(
            [VULCAN_REPACK, 'test-pack', '--strategy', 'invalid'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_repack_no_pack_id_shows_help(self):
        """Test running without pack ID shows help"""
        result = subprocess.run(
            [VULCAN_REPACK],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_repack_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'out.pack')
            json_output = os.path.join(tmpdir, 'stats.json')
            
            result = subprocess.run(
                [VULCAN_REPACK, 'test-pack',
                 '--strategy', 'aggressive',
                 '--compression', '9',
                 '--output', output,
                 '--verbose',
                 '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_repack_multiple_packs(self):
        """Test repacking multiple packs"""
        pack_ids = ['pack-001', 'pack-002', 'pack-003']
        for pack_id in pack_ids:
            result = subprocess.run(
                [VULCAN_REPACK, pack_id, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
