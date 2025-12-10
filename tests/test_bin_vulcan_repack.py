"""
Tests for vulcan-repack Python script

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
VULCAN_REPACK = os.path.join(BIN_DIR, 'vulcan-repack')


def run_vulcan_repack(args, **kwargs):
    """
    Helper function to run vulcan-repack with proper platform-specific handling.
    
    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    """
    if platform.system() == 'Windows':
        command = [sys.executable, VULCAN_REPACK] + args
    else:
        command = [VULCAN_REPACK] + args
    
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    
    return subprocess.run(command, **kwargs)


class TestVulcanRepack:
    """Test suite for vulcan-repack"""

    def test_repack_exists(self):
        """Test that vulcan-repack exists and is executable"""
        assert os.path.exists(VULCAN_REPACK)
        if platform.system() != 'Windows':
            assert os.access(VULCAN_REPACK, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_repack(['--help'])
        assert result.returncode == 0
        assert 'VulcanAMI Repack' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_repack(['--help'])
        assert '4.6.0' in result.stdout

    def test_repack_basic(self):
        """Test basic repacking"""
        result = run_vulcan_repack(['test-pack-001'], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Repack' in output or 'SUCCESS' in output

    def test_repack_adaptive_strategy(self):
        """Test adaptive repacking strategy"""
        result = run_vulcan_repack(['test-pack', '--strategy', 'adaptive'], timeout=30)
        assert result.returncode == 0

    def test_repack_aggressive_strategy(self):
        """Test aggressive repacking strategy"""
        result = run_vulcan_repack(['test-pack', '--strategy', 'aggressive'], timeout=30)
        assert result.returncode == 0

    def test_repack_conservative_strategy(self):
        """Test conservative repacking strategy"""
        result = run_vulcan_repack(['test-pack', '--strategy', 'conservative'], timeout=30)
        assert result.returncode == 0

    def test_repack_with_compression_level(self):
        """Test custom compression level"""
        for level in [1, 6, 12]:
            result = run_vulcan_repack(
                ['test-pack', '--compression', str(level)],
                timeout=30
            )
            assert result.returncode == 0

    def test_repack_with_output_path(self):
        """Test custom output path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'repacked.pack')
            result = run_vulcan_repack(
                ['test-pack', '--output', output_path],
                timeout=30
            )
            assert result.returncode == 0

    def test_repack_dry_run_mode(self):
        """Test dry run mode"""
        result = run_vulcan_repack(['test-pack', '--dry-run'], timeout=30)
        assert result.returncode == 0

    def test_repack_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_repack(['test-pack', '--verbose'], timeout=30)
        assert result.returncode == 0

    def test_repack_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_repack(['test-pack', '--quiet'], timeout=30)
        assert result.returncode == 0

    def test_repack_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'repack.json')
            
            result = run_vulcan_repack(
                ['test-pack', '--json', json_output],
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
        result = run_vulcan_repack(['test-pack'], timeout=30)
        
        output = result.stdout + result.stderr
        assert 'Pack' in output or 'Summary' in output or 'Size' in output

    def test_repack_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = run_vulcan_repack(['test-pack', '--strategy', 'invalid'], timeout=30)
        assert result.returncode != 0

    def test_repack_no_pack_id_shows_help(self):
        """Test running without pack ID shows help"""
        result = run_vulcan_repack([], timeout=30)
        assert result.returncode != 0

    def test_repack_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, 'out.pack')
            json_output = os.path.join(tmpdir, 'stats.json')
            
            result = run_vulcan_repack(
                ['test-pack', '--strategy', 'aggressive', '--compression', '9',
                 '--output', output, '--verbose', '--json', json_output],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_repack_multiple_packs(self):
        """Test repacking multiple packs"""
        pack_ids = ['pack-001', 'pack-002', 'pack-003']
        for pack_id in pack_ids:
            result = run_vulcan_repack([pack_id, '--quiet'], timeout=30)
            assert result.returncode == 0
