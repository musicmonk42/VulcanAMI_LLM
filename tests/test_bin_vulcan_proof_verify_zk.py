"""
Tests for vulcan-proof-verify-zk Python script
"""
import subprocess
import pytest
import os
import tempfile
import json


BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
VULCAN_PROOF_VERIFY = os.path.join(BIN_DIR, 'vulcan-proof-verify-zk')


class TestVulcanProofVerifyZk:
    """Test suite for vulcan-proof-verify-zk"""

    def test_proof_verify_exists(self):
        """Test that vulcan-proof-verify-zk exists and is executable"""
        assert os.path.exists(VULCAN_PROOF_VERIFY)
        assert os.access(VULCAN_PROOF_VERIFY, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'VulcanAMI ZK Proof Verifier' in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, '--help'],
            capture_output=True,
            text=True
        )
        assert '4.6.0' in result.stdout

    def test_verify_proof_string(self):
        """Test verifying proof from string"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof_string_12345'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert 'VALID' in output or 'Verification' in output

    def test_verify_proof_from_file(self):
        """Test verifying proof from file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_file = os.path.join(tmpdir, 'proof.txt')
            with open(proof_file, 'w') as f:
                f.write('test_proof_data_abc123')
            
            result = subprocess.run(
                [VULCAN_PROOF_VERIFY, proof_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_public_inputs(self):
        """Test verifying with public inputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            public_inputs = os.path.join(tmpdir, 'inputs.json')
            with open(public_inputs, 'w') as f:
                json.dump({'input1': 'value1', 'input2': 123}, f)
            
            result = subprocess.run(
                [VULCAN_PROOF_VERIFY, 'test_proof', '--public-inputs', public_inputs],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0

    def test_verify_with_custom_circuit(self):
        """Test verifying with custom circuit"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof', '--circuit', '/path/to/circuit.circom'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # May fail if circuit doesn't exist, but should accept the parameter
        assert result.returncode in [0, 1]

    def test_verify_with_custom_vkey(self):
        """Test verifying with custom verification key"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof', '--vkey', '/path/to/vkey.json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # May fail if vkey doesn't exist, but should accept the parameter
        assert result.returncode in [0, 1]

    def test_verify_verbose_mode(self):
        """Test verbose mode"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof', '--verbose'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_verify_quiet_mode(self):
        """Test quiet mode"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof', '--quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_verify_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, 'verify.json')
            
            result = subprocess.run(
                [VULCAN_PROOF_VERIFY, 'test_proof', '--json', json_output],
                capture_output=True,
                text=True,
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
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'Proof' in output or 'Verification' in output or 'VALID' in output

    def test_verify_no_proof_shows_help(self):
        """Test running without proof shows help"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_verify_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            public_inputs = os.path.join(tmpdir, 'inputs.json')
            with open(public_inputs, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            json_output = os.path.join(tmpdir, 'result.json')
            
            result = subprocess.run(
                [VULCAN_PROOF_VERIFY, 'test_proof',
                 '--public-inputs', public_inputs,
                 '--verbose',
                 '--json', json_output],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_verify_multiple_proofs(self):
        """Test verifying multiple proofs"""
        proofs = ['proof1', 'proof2', 'proof3']
        for proof in proofs:
            result = subprocess.run(
                [VULCAN_PROOF_VERIFY, proof, '--quiet'],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0

    def test_verify_proof_hash_displayed(self):
        """Test that proof hash is displayed"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof_with_hash'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        assert 'hash' in output.lower() or 'Proof' in output

    def test_verify_nonexistent_public_inputs(self):
        """Test error handling for non-existent public inputs file"""
        result = subprocess.run(
            [VULCAN_PROOF_VERIFY, 'test_proof', '--public-inputs', '/nonexistent/file.json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should handle gracefully (may skip or error)
        assert result.returncode in [0, 1]
