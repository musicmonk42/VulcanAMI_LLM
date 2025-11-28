"""
Tests for vulcan-pack Python script

This version includes Windows compatibility fixes.
"""
import subprocess
import pytest
import os
import sys
import tempfile
import json
import struct
import platform


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PACK = os.path.join(BIN_DIR, 'vulcan-pack')


def run_vulcan_pack(args, **kwargs):
    """
    Helper function to run vulcan-pack with proper platform-specific handling.
    
    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    
    Args:
        args: List of arguments to pass to vulcan-pack
        **kwargs: Additional arguments to pass to subprocess.run()
    
    Returns:
        subprocess.CompletedProcess object
    """
    if platform.system() == 'Windows':
        # On Windows, explicitly use Python to run the script
        command = [sys.executable, VULCAN_PACK] + args
    else:
        # On Unix/Linux, the shebang handles it
        command = [VULCAN_PACK] + args
    
    # Set default values for common parameters
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    
    return subprocess.run(command, **kwargs)


class TestVulcanPack:
    """Test suite for vulcan-pack"""

    def test_pack_exists(self):
        """Test that vulcan-pack exists and is readable"""
        assert os.path.exists(VULCAN_PACK), f"vulcan-pack not found at {VULCAN_PACK}"
        assert os.path.isfile(VULCAN_PACK), f"vulcan-pack is not a file"
        
        # On Unix/Linux, also check if it's executable
        if platform.system() != 'Windows':
            assert os.access(VULCAN_PACK, os.X_OK), f"vulcan-pack is not executable"

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_pack(['--help'])
        assert result.returncode == 0
        assert 'VulcanAMI Pack Builder' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_pack(['--help'])
        assert '4.6.0' in result.stdout

    def test_no_args_requires_output(self):
        """Test that running without input provides help or error"""
        result = run_vulcan_pack([], input="")
        # May return error or help
        assert result.returncode in [0, 1, 2]

    def test_build_pack_from_json(self):
        """Test building pack from JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input JSON
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data', 'value': 123}, f)
            
            # Create output pack
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            # Check result
            assert result.returncode == 0
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0

    def test_build_pack_from_json_array(self):
        """Test building pack from JSON array"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input JSON array
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump([{'id': 1, 'data': 'test1'}, {'id': 2, 'data': 'test2'}], f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)

    def test_build_pack_with_compression_levels(self):
        """Test different compression levels"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data' * 100}, f)
            
            for level in [1, 3, 9]:
                output_file = os.path.join(tmpdir, f'output_{level}.pack')
                result = run_vulcan_pack(
                    ['-i', input_file, '-o', output_file, 
                     '--compression', str(level), '--no-dqs'],
                    timeout=30
                )
                assert result.returncode == 0
                assert os.path.exists(output_file)

    def test_build_pack_with_stats_output(self):
        """Test statistics output to JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            stats_file = os.path.join(tmpdir, 'stats.json')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file,
                 '--stats', stats_file, '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(stats_file)
            
            # Verify stats file is valid JSON
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                assert 'chunk_count' in stats
                assert 'total_size' in stats
                assert 'compressed_size' in stats

    def test_build_pack_verbose_mode(self):
        """Test verbose mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file, 
                 '--verbose', '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0

    def test_build_pack_quiet_mode(self):
        """Test quiet mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file,
                 '--quiet', '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0

    def test_build_pack_from_directory(self):
        """Test building pack from directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files in directory
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            
            for i in range(3):
                with open(os.path.join(data_dir, f'file{i}.txt'), 'w') as f:
                    f.write(f'test data {i}\n')
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-d', data_dir, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)

    def test_build_pack_from_directory_recursive(self):
        """Test building pack from directory recursively"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            data_dir = os.path.join(tmpdir, 'data')
            sub_dir = os.path.join(data_dir, 'subdir')
            os.makedirs(sub_dir)
            
            with open(os.path.join(data_dir, 'file1.txt'), 'w') as f:
                f.write('top level\n')
            with open(os.path.join(sub_dir, 'file2.txt'), 'w') as f:
                f.write('sub level\n')
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-d', data_dir, '-o', output_file,
                 '--recursive', '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)

    def test_pack_header_structure(self):
        """Test that pack header has correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            
            # Verify pack header
            with open(output_file, 'rb') as f:
                magic = f.read(4)
                assert magic == b'GPK2', "Pack file should start with GPK2 magic"
                
                version = struct.unpack('>I', f.read(4))[0]
                assert version == 2, "Pack version should be 2"

    def test_invalid_compression_level(self):
        """Test invalid compression level is rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file,
                 '--compression', '99', '--no-dqs'],
                timeout=30
            )
            
            # Should fail with invalid compression level
            assert result.returncode != 0

    def test_dqs_threshold_parameter(self):
        """Test DQS threshold parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            # Test with custom threshold (will use mock DQS since service not available)
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file,
                 '--dqs-threshold', '0.90', '--no-dqs'],
                timeout=30
            )
            
            # With --no-dqs, should still work
            assert result.returncode == 0

    def test_bloom_filter_size_parameter(self):
        """Test bloom filter size parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file,
                 '--bloom-size', '256', '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0

    def test_empty_input_handling(self):
        """Test handling of empty input"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty directory
            data_dir = os.path.join(tmpdir, 'empty')
            os.makedirs(data_dir)
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-d', data_dir, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            # Should handle empty input gracefully (may error or create empty pack)
            assert result.returncode in [0, 1]

    def test_file_list_input(self):
        """Test building pack from file list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some data files
            for i in range(3):
                with open(os.path.join(tmpdir, f'data{i}.txt'), 'w') as f:
                    f.write(f'data {i}\n')
            
            # Create file list
            file_list = os.path.join(tmpdir, 'files.txt')
            with open(file_list, 'w') as f:
                for i in range(3):
                    f.write(os.path.join(tmpdir, f'data{i}.txt') + '\n')
            
            output_file = os.path.join(tmpdir, 'output.pack')
            
            result = run_vulcan_pack(
                ['-f', file_list, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)

    def test_missing_input_file(self):
        """Test error handling for missing input file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'out.pack')
            result = run_vulcan_pack(
                ['-i', '/nonexistent/file.json', '-o', output_file],
                timeout=30
            )
            
            # Should fail with error
            assert result.returncode != 0

    def test_missing_output_directory(self):
        """Test error handling when output directory doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            # Try to write to non-existent directory
            output_file = '/nonexistent/directory/output.pack'
            
            result = run_vulcan_pack(
                ['-i', input_file, '-o', output_file, '--no-dqs'],
                timeout=30
            )
            
            # Should fail
            assert result.returncode != 0

    def test_python_executable_available(self):
        """Test that Python executable is available"""
        result = subprocess.run(
            [sys.executable, '--version'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'Python' in result.stdout or 'Python' in result.stderr
