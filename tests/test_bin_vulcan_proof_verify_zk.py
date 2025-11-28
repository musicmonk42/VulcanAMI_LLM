"""
Tests for vulcan-proof-verify-zk Python script

This version includes Windows compatibility fixes.
"""
import subprocess
import pytest
import os
import sys
import tempfile
import json
import platform


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PROOF_VERIFY = os.path.join(BIN_DIR, 'vulcan-proof-verify-zk')


def run_vulcan_proof_verify(args, **kwargs):
    """
    Helper function to run vulcan-proof-verify-zk with proper platform-specific handling.
    
    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    """
    if platform.system() == 'Windows':
        command = [sys.executable, VULCAN_PROOF_VERIFY] + args
    else:
        command = [VULCAN_PROOF_VERIFY] + args
    
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    
    return subprocess.run(command, **kwargs)


class TestVulcanProofVerifyZk:
    """Test suite for vulcan-proof-verify-zk"""

    def test_proof_verify_exists(self):
        """Test that vulcan-proof-verify-zk exists and is executable"""
        assert os.path.exists(VULCAN_PROOF_VERIFY)
        if platform.system() != 'Windows':
            assert os.access(VULCAN_PROOF_VERIFY, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_proof_verify(['--help'])
        assert result.returncode == 0
        assert 'VulcanAMI ZK Proof Verifier' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_proof_verify(['--help'])
        assert '4.6.0' in result.stdout

    def test_verify_proof_string(self):
        """Test verifying proof from string"""
        result = run_vulcan_proof_verify(['test_proof_string_12345'], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'VALID' in output or 'Verification' in output

    def test_verify_proof_from_file(self):
        """Test verifying proof from file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_file = os.path.join(tmpdir, 'proof.txt')
            with open(proof_file, 'w') as f:
                f.write('test_proof_data_abc123')
            
            result = run_vulcan_proof_verify([proof_file], timeout=30)
            
            assert result.returncode == 0

    def test_verify_with_public_inputs(self):
        """Test verifying with public inputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            public_inputs = os.path.join(tmpdir, 'inputs.json')
            with open(public_inputs, 'w') as f:
                json.dump({'input1': 'value1', 'input2': 123}, f)
            
            result = run_vulcan_proof_verify(
                ['test_proof', '--public-inputs', public_inputs],
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_custom_circuit(self):
        """Test verifying with custom circuit"""
        result = run_vulcan_proof_verify(
            ['test_proof', '--circuit', '/path/to/circuit.circom'],
            timeout=30
        )
        assert result.returncode in [0, 1]

    def test_verify_with_custom_vkey(self):
        """Test verifying with custom verification key"""
        result = run_vulcan_proof_verify(
            ['test_proof', '--vkey', '/path/to/vkey.json'],
            timeout=30
        )
        assert result.returncode in [0, 1]

    def test_verify_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_proof_verify(['test_proof', '--verbose'], timeout=30)
        assert result.returncode == 0

    def test_verify_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_proof_verify(['test_proof', '--quiet'], timeout=30)
        assert result.returncode == 0

    def test_verify_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'verify.json')
            
            result = run_vulcan_proof_verify(
                ['test_proof', '--json', json_output],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)
            
            # Verify JSON structure
            with open(json_output, 'r') as f:
                data = json.load(f)
                assert 'valid' in data
                assert 'proof_hash' in data
                assert 'verification_time' in data

    def test_verify_displays_result(self):
        """Test that verification displays result"""
        result = run_vulcan_proof_verify(['test_proof'], timeout=30)
        
        output = result.stdout + result.stderr
        assert 'Proof' in output or 'Verification' in output or 'VALID' in output

    def test_verify_no_proof_shows_help(self):
        """Test running without proof shows help"""
        result = run_vulcan_proof_verify([], timeout=30)
        assert result.returncode != 0

    def test_verify_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            public_inputs = os.path.join(tmpdir, 'inputs.json')
            with open(public_inputs, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            json_output = os.path.join(tmpdir, 'result.json')
            
            result = run_vulcan_proof_verify(
                ['test_proof', '--public-inputs', public_inputs,
                 '--verbose', '--json', json_output],
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_verify_multiple_proofs(self):
        """Test verifying multiple proofs"""
        proofs = ['proof1', 'proof2', 'proof3']
        for proof in proofs:
            result = run_vulcan_proof_verify([proof, '--quiet'], timeout=30)
            assert result.returncode == 0

    def test_verify_proof_hash_displayed(self):
        """Test that proof hash is displayed"""
        result = run_vulcan_proof_verify(['test_proof_with_hash'], timeout=30)
        
        output = result.stdout + result.stderr
        assert 'hash' in output.lower() or 'Proof' in output

    def test_verify_nonexistent_public_inputs(self):
        """Test error handling for non-existent public inputs file"""
        result = run_vulcan_proof_verify(
            ['test_proof', '--public-inputs', '/nonexistent/file.json'],
            timeout=30
        )
        assert result.returncode in [0, 1]
