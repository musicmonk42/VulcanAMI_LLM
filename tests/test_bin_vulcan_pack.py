"""
Tests for vulcan-pack Python script
"""
import subprocess
import pytest
import os
import tempfile
import json
import struct


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PACK = os.path.join(BIN_DIR, 'vulcan-pack')


class TestVulcanPack:
    """Test suite for vulcan-pack"""

    def test_pack_exists(self):
        """Test that vulcan-pack exists and is executable"""
        assert os.path.exists(VULCAN_PACK)
        assert os.access(VULCAN_PACK, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_PACK, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Pack Builder' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_PACK, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_no_args_requires_output(self):
        """Test that running without input provides help or error"""
        result = subprocess.run(
            [VULCAN_PACK],
            capture_output=True,
            text=True,
            input=""
        )
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
                result = subprocess.run(
                    [VULCAN_PACK, '-i', input_file, '-o', output_file, 
                     '--compression', str(level), '--no-dqs'],
                    capture_output=True,
                    text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file,
                 '--stats', stats_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file, 
                 '--verbose', '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file,
                 '--quiet', '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-d', data_dir, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-d', data_dir, '-o', output_file,
                 '--recursive', '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file,
                 '--compression', '99', '--no-dqs'],
                capture_output=True,
                text=True,
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
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file,
                 '--dqs-threshold', '0.90', '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file,
                 '--bloom-size', '256', '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-d', data_dir, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-f', file_list, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)

    def test_missing_input_file(self):
        """Test error handling for missing input file"""
        result = subprocess.run(
            [VULCAN_PACK, '-i', '/nonexistent/file.json', '-o', '/tmp/out.pack'],
            capture_output=True,
            text=True,
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
            
            result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', output_file, '--no-dqs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should fail
            assert result.returncode != 0
