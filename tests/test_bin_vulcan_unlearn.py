"""
Tests for vulcan-unlearn Python script

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
VULCAN_UNLEARN = os.path.join(BIN_DIR, 'vulcan-unlearn')


def run_vulcan_unlearn(args, **kwargs):
    """
    Helper function to run vulcan-unlearn with proper platform-specific handling.

    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    """
    if platform.system() == 'Windows':
        command = [sys.executable, VULCAN_UNLEARN] + args
    else:
        command = [VULCAN_UNLEARN] + args

    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)

    return subprocess.run(command, **kwargs)


class TestVulcanUnlearn:
    """Test suite for vulcan-unlearn"""

    def test_unlearn_exists(self):
        """Test that vulcan-unlearn exists and is executable"""
        assert os.path.exists(VULCAN_UNLEARN)
        if platform.system() != 'Windows':
            assert os.access(VULCAN_UNLEARN, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_unlearn(['--help'])
        assert result.returncode == 0
        assert 'VulcanAMI Unlearning Engine' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_unlearn(['--help'])
        assert '4.6.0' in result.stdout

    def test_unlearn_pattern_basic(self):
        """Test basic unlearning with pattern"""
        result = run_vulcan_unlearn(['test_pattern'], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Unlearning' in output or 'SUCCESS' in output

    def test_unlearn_with_packfile(self):
        """Test unlearning with specific packfile"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--packfile', 'test.pack'],
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_gradient_surgery_strategy(self):
        """Test gradient surgery strategy"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--strategy', 'gradient_surgery'],
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'gradient' in output.lower() or 'surgery' in output.lower() or 'SUCCESS' in output

    def test_unlearn_deletion_strategy(self):
        """Test deletion strategy"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--strategy', 'deletion'],
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_perturbation_strategy(self):
        """Test perturbation strategy"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--strategy', 'perturbation'],
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_fast_lane_mode(self):
        """Test fast lane mode"""
        result = run_vulcan_unlearn(['test_pattern', '--fast-lane'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_no_proof_mode(self):
        """Test skipping proof generation"""
        result = run_vulcan_unlearn(['test_pattern', '--no-proof'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_with_verification(self):
        """Test unlearning with verification"""
        result = run_vulcan_unlearn(['test_pattern', '--verify'], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'verif' in output.lower() or 'SUCCESS' in output

    def test_unlearn_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_unlearn(['test_pattern', '--verbose'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_unlearn(['test_pattern', '--quiet'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'unlearn.json')

            result = run_vulcan_unlearn(
                ['test_pattern', '--json', json_output],
                timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

            # Verify JSON structure
            with open(json_output, 'r', encoding="utf-8") as f:
                data = json.load(f)
                assert 'success' in data
                assert 'pattern' in data
                assert 'records_affected' in data
                assert 'execution_time' in data

    def test_unlearn_displays_summary(self):
        """Test that unlearn displays summary"""
        result = run_vulcan_unlearn(['test_pattern'], timeout=30)

        output = result.stdout + result.stderr
        assert 'Pattern' in output or 'Records' in output or 'Summary' in output

    def test_unlearn_multiple_patterns(self):
        """Test unlearning multiple patterns"""
        patterns = ['pattern1', 'pattern2', 'pattern3']
        for pattern in patterns:
            result = run_vulcan_unlearn(
                [pattern, '--fast-lane', '--no-proof'],
                timeout=30
            )
            assert result.returncode == 0

    def test_unlearn_complex_pattern(self):
        """Test unlearning with complex pattern"""
        result = run_vulcan_unlearn(['user_id:12345'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_email_pattern(self):
        """Test unlearning email pattern"""
        result = run_vulcan_unlearn(['email:test@example.com'], timeout=30)
        assert result.returncode == 0

    def test_unlearn_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--strategy', 'invalid_strategy'],
            timeout=30
        )
        assert result.returncode != 0

    def test_unlearn_no_pattern_shows_help(self):
        """Test running without pattern shows help"""
        result = run_vulcan_unlearn([], timeout=30)
        assert result.returncode != 0

    def test_unlearn_generates_audit_log(self):
        """Test that audit log is generated"""
        result = run_vulcan_unlearn(['test_pattern'], timeout=30)

        output = result.stdout + result.stderr
        assert 'Audit' in output or 'audit' in output or 'SUCCESS' in output

    def test_unlearn_with_all_flags(self):
        """Test combining multiple flags"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'result.json')

            result = run_vulcan_unlearn(
                ['test_pattern', '--strategy', 'deletion', '--fast-lane',
                 '--no-proof', '--verbose', '--json', json_output],
                timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_unlearn_secure_erase_flag(self):
        """Test secure erase flag"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--secure-erase', '--no-proof'],
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'secure' in output.lower() or 'erase' in output.lower() or 'SUCCESS' in output

    def test_unlearn_secure_erase_with_verification(self):
        """Test secure erase with verification"""
        result = run_vulcan_unlearn(
            ['test_pattern', '--secure-erase', '--verify'],
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'verif' in output.lower() or 'SUCCESS' in output

    def test_unlearn_shows_warning_about_simplified_zk(self):
        """Test that ZK implementation status is shown"""
        result = run_vulcan_unlearn(['test_pattern', '--verbose'], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'groth16' in output.lower() or 'zk' in output.lower() or 'proof' in output.lower()

    def test_unlearn_secure_erase_in_help(self):
        """Test that secure erase is documented in help"""
        result = run_vulcan_unlearn(['--help'])
        assert result.returncode == 0
        assert '--secure-erase' in result.stdout or 'secure erase' in result.stdout.lower()
