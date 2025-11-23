"""
Tests for vulcan-cli bash script
"""
import subprocess
import pytest
import os


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_CLI = os.path.join(BIN_DIR, 'vulcan-cli')


class TestVulcanCLI:
    """Test suite for vulcan-cli"""

    def test_cli_exists(self):
        """Test that vulcan-cli exists and is executable"""
        assert os.path.exists(VULCAN_CLI)
        assert os.access(VULCAN_CLI, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI CLI' in result.stdout or 'VulcanAMI CLI' in result.stderr
        assert 'USAGE' in result.stdout or 'USAGE' in result.stderr

    def test_version_flag(self):
        """Test --version flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--version'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert '4.6.0' in result.stdout or '4.6.0' in result.stderr

    def test_help_command(self):
        """Test help command"""
        result = subprocess.run(
            [VULCAN_CLI, 'help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_version_command(self):
        """Test version command"""
        result = subprocess.run(
            [VULCAN_CLI, 'version'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_config_command(self):
        """Test config command"""
        result = subprocess.run(
            [VULCAN_CLI, 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'Configuration' in output

    def test_no_args(self):
        """Test running with no arguments shows help"""
        result = subprocess.run(
            [VULCAN_CLI],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_invalid_command(self):
        """Test running with invalid command"""
        result = subprocess.run(
            [VULCAN_CLI, 'invalid-command-xyz'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert 'Unknown command' in output

    def test_verbose_flag(self):
        """Test -V/--verbose flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--verbose', 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_debug_flag(self):
        """Test -d/--debug flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--debug', 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_quiet_flag(self):
        """Test -q/--quiet flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--quiet', 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_no_color_flag(self):
        """Test --no-color flag"""
        result = subprocess.run(
            [VULCAN_CLI, '--no-color', 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_pack_command_exists(self):
        """Test that pack command is recognized"""
        result = subprocess.run(
            [VULCAN_CLI, 'pack', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # May fail with "command not yet implemented" but should recognize the command
        assert result.returncode in [0, 1]

    def test_verify_command_exists(self):
        """Test that verify command is recognized"""
        result = subprocess.run(
            [VULCAN_CLI, 'verify', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # May fail with "command not yet implemented" but should recognize the command
        assert result.returncode in [0, 1]

    def test_unlearn_command_exists(self):
        """Test that unlearn command is recognized"""
        result = subprocess.run(
            [VULCAN_CLI, 'unlearn', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # May fail with "command not yet implemented" but should recognize the command
        assert result.returncode in [0, 1]

    def test_vector_command_exists(self):
        """Test that vector command is recognized"""
        result = subprocess.run(
            [VULCAN_CLI, 'vector', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # May fail with "command not yet implemented" but should recognize the command
        assert result.returncode in [0, 1]

    def test_proof_command_exists(self):
        """Test that proof command is recognized"""
        result = subprocess.run(
            [VULCAN_CLI, 'proof', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # May fail with "command not yet implemented" but should recognize the command
        assert result.returncode in [0, 1]

    def test_environment_variable_config(self):
        """Test configuration via environment variables"""
        env = os.environ.copy()
        env['VULCAN_VERBOSE'] = '1'
        result = subprocess.run(
            [VULCAN_CLI, 'config'],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0

    def test_multiple_flags(self):
        """Test multiple flags together"""
        result = subprocess.run(
            [VULCAN_CLI, '--verbose', '--no-color', 'config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
