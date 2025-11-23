"""
Tests for vulcan-unlearn Python script
"""
import subprocess
import pytest
import os
import tempfile
import json


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_UNLEARN = os.path.join(BIN_DIR, 'vulcan-unlearn')


class TestVulcanUnlearn:
    """Test suite for vulcan-unlearn"""

    def test_unlearn_exists(self):
        """Test that vulcan-unlearn exists and is executable"""
        assert os.path.exists(VULCAN_UNLEARN)
        assert os.access(VULCAN_UNLEARN, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_UNLEARN, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Unlearning Engine' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_UNLEARN, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_unlearn_pattern_basic(self):
        """Test basic unlearning with pattern"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Unlearning' in output or 'SUCCESS' in output

    def test_unlearn_with_packfile(self):
        """Test unlearning with specific packfile"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--packfile', 'test.pack'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_gradient_surgery_strategy(self):
        """Test gradient surgery strategy"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--strategy', 'gradient_surgery'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'gradient' in output.lower() or 'surgery' in output.lower() or 'SUCCESS' in output

    def test_unlearn_deletion_strategy(self):
        """Test deletion strategy"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--strategy', 'deletion'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_perturbation_strategy(self):
        """Test perturbation strategy"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--strategy', 'perturbation'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_fast_lane_mode(self):
        """Test fast lane mode"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--fast-lane'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_no_proof_mode(self):
        """Test skipping proof generation"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--no-proof'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_with_verification(self):
        """Test unlearning with verification"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--verify'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'verif' in output.lower() or 'SUCCESS' in output

    def test_unlearn_verbose_mode(self):
        """Test verbose mode"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--verbose'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_quiet_mode(self):
        """Test quiet mode"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'unlearn.json')
            
            result = subprocess.run(
                [VULCAN_UNLEARN, 'test_pattern', '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)
            
            # Verify JSON structure
            with open(json_output, 'r') as f:
                data = json.load(f)
                assert 'success' in data
                assert 'pattern' in data
                assert 'records_affected' in data
                assert 'execution_time' in data

    def test_unlearn_displays_summary(self):
        """Test that unlearn displays summary"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'Pattern' in output or 'Records' in output or 'Summary' in output

    def test_unlearn_multiple_patterns(self):
        """Test unlearning multiple patterns"""
        patterns = ['pattern1', 'pattern2', 'pattern3']
        for pattern in patterns:
            result = subprocess.run(
                [VULCAN_UNLEARN, pattern, '--fast-lane', '--no-proof'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_unlearn_complex_pattern(self):
        """Test unlearning with complex pattern"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'user_id:12345'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_email_pattern(self):
        """Test unlearning email pattern"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'email:test@example.com'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_unlearn_invalid_strategy(self):
        """Test invalid strategy is rejected"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern', '--strategy', 'invalid_strategy'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_unlearn_no_pattern_shows_help(self):
        """Test running without pattern shows help"""
        result = subprocess.run(
            [VULCAN_UNLEARN],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_unlearn_generates_audit_log(self):
        """Test that audit log is generated"""
        result = subprocess.run(
            [VULCAN_UNLEARN, 'test_pattern'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        # Should mention audit log
        assert 'Audit' in output or 'audit' in output or 'SUCCESS' in output

    def test_unlearn_with_all_flags(self):
        """Test combining multiple flags"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'result.json')
            
            result = subprocess.run(
                [VULCAN_UNLEARN, 'test_pattern',
                 '--strategy', 'deletion',
                 '--fast-lane',
                 '--no-proof',
                 '--verbose',
                 '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)
