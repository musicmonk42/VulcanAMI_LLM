"""
Tests for vulcan-cli bash script

This version explicitly uses Git Bash on Windows, avoiding broken WSL installations.
Fixed: Handles None values in stdout/stderr concatenation.
"""
import subprocess
import pytest
import os
import sys
import platform
import shutil


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_CLI = os.path.join(BIN_DIR, 'vulcan-cli')


def find_git_bash():
    """
    Find Git Bash executable on Windows, avoiding WSL.
    
    Returns:
        str: Full path to bash.exe from Git installation
        None: If Git Bash not found
    """
    if platform.system() != 'Windows':
        return 'bash'  # Use system bash on Unix/Linux
    
    # Common Git Bash installation paths on Windows
    common_paths = [
        r'C:\Program Files\Git\bin\bash.exe',
        r'C:\Program Files (x86)\Git\bin\bash.exe',
        r'C:\Git\bin\bash.exe',
    ]
    
    # Check common paths first
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Try to find bash.exe using 'where' command (Windows equivalent of 'which')
    try:
        result = subprocess.run(
            ['where', 'bash'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # 'where' returns all matches, one per line
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                # Skip WSL bash (contains 'System32' or 'wsl')
                if line and 'System32' not in line and 'wsl' not in line.lower():
                    # Verify it's actually Git Bash by checking if it works
                    try:
                        test = subprocess.run(
                            [line, '--version'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if test.returncode == 0 and 'GNU bash' in test.stdout:
                            return line
                    except:
                        continue
    except:
        pass
    
    # Try using Git's installation directory from PATH
    git_path = shutil.which('git')
    if git_path:
        # git.exe is usually in C:\Program Files\Git\cmd\git.exe
        # bash.exe is in C:\Program Files\Git\bin\bash.exe
        git_dir = os.path.dirname(os.path.dirname(git_path))
        bash_path = os.path.join(git_dir, 'bin', 'bash.exe')
        if os.path.exists(bash_path):
            return bash_path
    
    return None


def get_command_prefix():
    """
    Get the command prefix needed to execute bash scripts.
    
    Returns:
        list: ['<path-to-git-bash>'] on Windows
              [] on Unix/Linux
    """
    if platform.system() == 'Windows':
        bash_path = find_git_bash()
        if bash_path:
            return [bash_path]
        else:
            pytest.fail("Git Bash not found. Please install Git for Windows from https://git-scm.com/download/win")
    else:
        # On Unix/Linux, execute script directly (shebang handles it)
        return []


def run_vulcan_cli(args, **kwargs):
    """
    Helper function to run vulcan-cli with proper platform-specific handling.
    
    Args:
        args: List of arguments to pass to vulcan-cli
        **kwargs: Additional arguments to pass to subprocess.run()
    
    Returns:
        subprocess.CompletedProcess object
    """
    prefix = get_command_prefix()
    command = prefix + [VULCAN_CLI] + args
    
    # Set default values for common parameters
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    
    return subprocess.run(command, **kwargs)


def get_output(result):
    """
    Safely concatenate stdout and stderr, handling None values.
    
    Args:
        result: subprocess.CompletedProcess object
    
    Returns:
        str: Combined stdout and stderr output
    """
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return stdout + stderr


class TestVulcanCLI:
    """Test suite for vulcan-cli"""

    def test_cli_exists(self):
        """Test that vulcan-cli exists and is readable"""
        assert os.path.exists(VULCAN_CLI), f"vulcan-cli not found at {VULCAN_CLI}"
        assert os.path.isfile(VULCAN_CLI), f"vulcan-cli is not a file: {VULCAN_CLI}"
        
        # On Unix/Linux, also check if it's executable
        if platform.system() != 'Windows':
            assert os.access(VULCAN_CLI, os.X_OK), f"vulcan-cli is not executable: {VULCAN_CLI}"

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_cli(['--help'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'VulcanAMI CLI' in output, "Expected 'VulcanAMI CLI' in output"
        assert 'USAGE' in output, "Expected 'USAGE' in output"

    def test_version_flag(self):
        """Test --version flag"""
        result = run_vulcan_cli(['--version'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert '4.6.0' in output, f"Expected version '4.6.0' in output, got: {output}"

    def test_help_command(self):
        """Test help command"""
        result = run_vulcan_cli(['help'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'VulcanAMI CLI' in output or 'USAGE' in output

    def test_version_command(self):
        """Test version command"""
        result = run_vulcan_cli(['version'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert '4.6.0' in output

    def test_config_command(self):
        """Test config command"""
        result = run_vulcan_cli(['config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'Configuration' in output, f"Expected 'Configuration' in output, got: {output}"

    def test_no_args(self):
        """Test running with no arguments shows help"""
        result = run_vulcan_cli([])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'VulcanAMI CLI' in output or 'USAGE' in output

    def test_invalid_command(self):
        """Test running with invalid command"""
        result = run_vulcan_cli(['invalid-command-xyz'])
        assert result.returncode == 1, f"Expected return code 1, got {result.returncode}"
        output = get_output(result)
        assert 'Unknown command' in output, f"Expected 'Unknown command' in output, got: {output}"

    def test_verbose_flag(self):
        """Test -V/--verbose flag"""
        result = run_vulcan_cli(['--verbose', 'config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    def test_debug_flag(self):
        """Test -d/--debug flag"""
        result = run_vulcan_cli(['--debug', 'config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    def test_quiet_flag(self):
        """Test -q/--quiet flag"""
        result = run_vulcan_cli(['--quiet', 'config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    def test_no_color_flag(self):
        """Test --no-color flag"""
        result = run_vulcan_cli(['--no-color', 'config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    def test_pack_command_exists(self):
        """Test that pack command is recognized"""
        result = run_vulcan_cli(['pack', '--help'], timeout=10)
        # May fail with "command not yet implemented" but should recognize the command
        # Return code 0 = success, 1 = recognized but not implemented
        assert result.returncode in [0, 1], f"Unexpected return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        # Should not say "Unknown command"
        assert 'Unknown command' not in output or result.returncode == 0

    def test_verify_command_exists(self):
        """Test that verify command is recognized"""
        result = run_vulcan_cli(['verify', '--help'], timeout=10)
        assert result.returncode in [0, 1], f"Unexpected return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'Unknown command' not in output or result.returncode == 0

    def test_unlearn_command_exists(self):
        """Test that unlearn command is recognized"""
        result = run_vulcan_cli(['unlearn', '--help'], timeout=10)
        assert result.returncode in [0, 1], f"Unexpected return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'Unknown command' not in output or result.returncode == 0

    def test_vector_command_exists(self):
        """Test that vector command is recognized"""
        result = run_vulcan_cli(['vector', '--help'], timeout=10)
        assert result.returncode in [0, 1], f"Unexpected return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'Unknown command' not in output or result.returncode == 0

    def test_proof_command_exists(self):
        """Test that proof command is recognized"""
        result = run_vulcan_cli(['proof', '--help'], timeout=10)
        assert result.returncode in [0, 1], f"Unexpected return code {result.returncode}\nStderr: {result.stderr}"
        output = get_output(result)
        assert 'Unknown command' not in output or result.returncode == 0

    def test_environment_variable_config(self):
        """Test configuration via environment variables"""
        env = os.environ.copy()
        env['VULCAN_VERBOSE'] = '1'
        result = run_vulcan_cli(['config'], env=env)
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    def test_multiple_flags(self):
        """Test multiple flags together"""
        result = run_vulcan_cli(['--verbose', '--no-color', 'config'])
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStderr: {result.stderr}"

    @pytest.mark.skipif(platform.system() != 'Windows', reason="Windows-specific test")
    def test_git_bash_available_on_windows(self):
        """Test that Git Bash is available on Windows (not broken WSL)"""
        bash_path = find_git_bash()
        assert bash_path is not None, (
            "Git Bash not found. Please install Git for Windows from https://git-scm.com/download/win\n"
            "WSL bash is not supported for these tests due to path translation issues."
        )
        
        # Verify it works
        result = subprocess.run([bash_path, '--version'], capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, f"Git Bash found at {bash_path} but doesn't work"
        assert 'GNU bash' in result.stdout, f"Expected GNU bash, got: {result.stdout}"

    def test_platform_detection(self):
        """Test that platform detection works correctly"""
        prefix = get_command_prefix()
        if platform.system() == 'Windows':
            assert len(prefix) == 1, "On Windows, should have bash path"
            assert 'bash' in prefix[0].lower(), f"Should contain 'bash', got: {prefix[0]}"
            # Should NOT be WSL bash
            assert 'System32' not in prefix[0], "Should not use WSL bash"
        else:
            assert prefix == [], "On Unix/Linux, should not use prefix"
