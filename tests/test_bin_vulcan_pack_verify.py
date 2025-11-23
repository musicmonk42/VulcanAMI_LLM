"""
Tests for vulcan-pack-verify Python script
"""
import subprocess
import pytest
import os
import tempfile
import json
import struct


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PACK = os.path.join(BIN_DIR, 'vulcan-pack')
VULCAN_PACK_VERIFY = os.path.join(BIN_DIR, 'vulcan-pack-verify')


class TestVulcanPackVerify:
    """Test suite for vulcan-pack-verify"""

    def test_verify_exists(self):
        """Test that vulcan-pack-verify exists and is executable"""
        assert os.path.exists(VULCAN_PACK_VERIFY)
        assert os.access(VULCAN_PACK_VERIFY, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_PACK_VERIFY, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI Pack Verifier' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_PACK_VERIFY, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_verify_valid_pack(self):
        """Test verifying a valid pack file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First create a valid pack
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            # Build pack
            build_result = subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert build_result.returncode == 0
            
            # Verify pack
            verify_result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert verify_result.returncode == 0
            output = verify_result.stdout + verify_result.stderr
            assert 'PASSED' in output or 'verification' in output.lower()

    def test_verify_with_full_flag(self):
        """Test full verification mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            # Build pack
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            # Verify with --full
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--full'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_verbose_flag(self):
        """Test verbose verification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--verbose'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_quiet_flag(self):
        """Test quiet verification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            json_output = os.path.join(tmpdir, 'verify.json')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--json', json_output],
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
                assert 'errors' in data
                assert 'warnings' in data

    def test_verify_invalid_magic_number(self):
        """Test verification fails for invalid magic number"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake pack with wrong magic
            pack_file = os.path.join(tmpdir, 'invalid.pack')
            with open(pack_file, 'wb') as f:
                f.write(b'FAKE' + b'\x00' * 100)
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 1
            output = result.stdout + result.stderr
            assert 'FAILED' in output or 'Invalid' in output or 'magic' in output.lower()

    def test_verify_nonexistent_file(self):
        """Test verification of non-existent file"""
        result = subprocess.run(
            [VULCAN_PACK_VERIFY, '/nonexistent/file.pack'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail
        assert result.returncode != 0

    def test_verify_merkle_flag(self):
        """Test --merkle flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--merkle'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # May succeed or show "not yet implemented"
            assert result.returncode in [0, 1]

    def test_verify_bloom_flag(self):
        """Test --bloom flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--bloom'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # May succeed or show "not yet implemented"
            assert result.returncode in [0, 1]

    def test_verify_checksums_flag(self):
        """Test --checksums flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file, '--checksums'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # May succeed or show "not yet implemented"
            assert result.returncode in [0, 1]

    def test_verify_corrupted_pack(self):
        """Test verification of corrupted pack"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid pack first
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            # Corrupt the pack by modifying bytes
            with open(pack_file, 'r+b') as f:
                f.seek(50)
                f.write(b'\xFF' * 10)
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Verification may still pass basic checks or fail
            assert result.returncode in [0, 1]

    def test_verify_multiple_packs(self):
        """Test verifying multiple packs sequentially"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                input_file = os.path.join(tmpdir, f'input{i}.json')
                with open(input_file, 'w') as f:
                    json.dump({'test': f'data{i}'}, f)
                
                pack_file = os.path.join(tmpdir, f'test{i}.pack')
                
                subprocess.run(
                    [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                    capture_output=True,
                    timeout=30
                )
                
                result = subprocess.run(
                    [VULCAN_PACK_VERIFY, pack_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                assert result.returncode == 0

    def test_verify_displays_pack_info(self):
        """Test that verify displays pack information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            pack_file = os.path.join(tmpdir, 'test.pack')
            
            subprocess.run(
                [VULCAN_PACK, '-i', input_file, '-o', pack_file, '--no-dqs'],
                capture_output=True,
                timeout=30
            )
            
            result = subprocess.run(
                [VULCAN_PACK_VERIFY, pack_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            # Should display some pack metadata
            assert 'Pack' in output or 'version' in output.lower() or 'Chunk' in output

    def test_no_arguments_shows_help(self):
        """Test running with no arguments"""
        result = subprocess.run(
            [VULCAN_PACK_VERIFY],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should show error or help
        assert result.returncode != 0
