"""Tests for secure_pickle utility module."""

import pytest
import pickle
import os
import tempfile
from pathlib import Path

from src.utils.secure_pickle import (
    SecurePickle,
    RestrictedUnpickler,
    SecurePickleError,
    SignatureVerificationError,
    RestrictedTypeError,
    restricted_loads,
    restricted_load,
)


class TestSecurePickle:
    """Tests for SecurePickle class."""
    
    @pytest.fixture
    def test_key(self):
        """Test secret key (DO NOT use in production)."""
        return b"test-key-for-testing-only-" + b"0" * 32
    
    @pytest.fixture
    def secure_pickle(self, test_key):
        """SecurePickle instance for testing."""
        return SecurePickle(secret_key=test_key)
    
    def test_dumps_and_loads(self, secure_pickle):
        """Test basic serialization and deserialization."""
        test_data = {'key': 'value', 'numbers': [1, 2, 3]}
        signed = secure_pickle.dumps(test_data)
        restored = secure_pickle.loads(signed)
        assert restored == test_data
    
    def test_signature_verification_passes(self, secure_pickle):
        """Test that valid signature is accepted."""
        data = {'test': True}
        signed = secure_pickle.dumps(data)
        # Should not raise
        result = secure_pickle.loads(signed)
        assert result == data
    
    def test_tampered_data_rejected(self, secure_pickle):
        """Test that tampered data is rejected."""
        data = {'test': True}
        signed = secure_pickle.dumps(data)
        
        # Tamper with data
        tampered = signed[:32] + b'X' + signed[33:]
        
        with pytest.raises(SignatureVerificationError):
            secure_pickle.loads(tampered)
    
    def test_too_short_data_rejected(self, secure_pickle):
        """Test that data shorter than signature is rejected."""
        with pytest.raises(SignatureVerificationError, match="too short"):
            secure_pickle.loads(b"short")
    
    def test_dump_and_load_file(self, secure_pickle):
        """Test file-based serialization."""
        data = {'file': 'test', 'nested': {'value': 123}}
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            temp_path = f.name
            secure_pickle.dump(data, f)
        
        try:
            with open(temp_path, 'rb') as f:
                restored = secure_pickle.load(f)
            assert restored == data
        finally:
            os.unlink(temp_path)
    
    def test_different_keys_fail(self, test_key):
        """Test that different keys produce different signatures."""
        sp1 = SecurePickle(test_key)
        sp2 = SecurePickle(b"different-key-" + b"X" * 32)
        
        data = {'key': 'value'}
        signed = sp1.dumps(data)
        
        with pytest.raises(SignatureVerificationError):
            sp2.loads(signed)
    
    def test_requires_secret_key(self):
        """Test that secret key is required."""
        # Clear environment variable if present
        old_value = os.environ.pop('PICKLE_SECRET_KEY', None)
        try:
            with pytest.raises(ValueError, match="PICKLE_SECRET_KEY"):
                SecurePickle()
        finally:
            if old_value:
                os.environ['PICKLE_SECRET_KEY'] = old_value
    
    def test_key_minimum_length(self):
        """Test that key must be at least 32 bytes."""
        with pytest.raises(ValueError, match="at least 32 bytes"):
            SecurePickle(b"short_key")


class TestRestrictedUnpickler:
    """Tests for RestrictedUnpickler class."""
    
    def test_safe_types_allowed(self):
        """Test that safe built-in types are allowed."""
        safe_data = {
            'list': [1, 2, 3],
            'dict': {'nested': True},
            'tuple': (4, 5),
            'set': {6, 7},
            'str': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
        }
        
        pickled = pickle.dumps(safe_data)
        restored = restricted_loads(pickled)
        
        assert restored['list'] == [1, 2, 3]
        assert restored['dict'] == {'nested': True}
        assert restored['str'] == 'hello'
        assert restored['int'] == 42
    
    def test_forbidden_class_rejected(self):
        """Test that forbidden classes are rejected."""
        
        class ForbiddenClass:
            """Test class that should be rejected."""
            pass
        
        obj = ForbiddenClass()
        pickled = pickle.dumps(obj)
        
        with pytest.raises(RestrictedTypeError, match="Forbidden class"):
            restricted_loads(pickled)
    
    def test_malicious_pickle_rejected(self):
        """Test that malicious pickle attempting code execution is rejected."""
        
        class MaliciousObject:
            """Malicious object that tries to execute code."""
            def __reduce__(self):
                import os
                return (os.system, ('echo pwned',))
        
        malicious = pickle.dumps(MaliciousObject())
        
        with pytest.raises(RestrictedTypeError):
            restricted_loads(malicious)
    
    def test_datetime_types_allowed(self):
        """Test that datetime types are in allowlist."""
        from datetime import datetime, date, timedelta
        
        data = {
            'datetime': datetime(2025, 11, 20, 12, 0, 0),
            'date': date(2025, 11, 20),
            'timedelta': timedelta(days=1, hours=2),
        }
        
        pickled = pickle.dumps(data)
        restored = restricted_loads(pickled)
        
        assert restored['datetime'] == data['datetime']
        assert restored['date'] == data['date']
        assert restored['timedelta'] == data['timedelta']
    
    def test_collections_types_allowed(self):
        """Test that collections types are in allowlist."""
        from collections import OrderedDict, defaultdict, Counter
        
        data = {
            'ordered': OrderedDict([('a', 1), ('b', 2)]),
            'counter': Counter(['a', 'a', 'b']),
        }
        
        pickled = pickle.dumps(data)
        restored = restricted_loads(pickled)
        
        assert list(restored['ordered'].items()) == [('a', 1), ('b', 2)]
        assert restored['counter']['a'] == 2
    
    def test_custom_safe_modules(self):
        """Test that custom safe modules can be provided."""
        
        # Custom allowlist that includes our test class
        class SafeCustomClass:
            def __init__(self, value):
                self.value = value
        
        custom_safe = {
            'builtins': {'dict', 'list', 'str', 'int'},
            __name__: {'SafeCustomClass'},
        }
        
        obj = SafeCustomClass(42)
        pickled = pickle.dumps(obj)
        
        # Should work with custom allowlist
        restored = restricted_loads(pickled, safe_modules=custom_safe)
        assert restored.value == 42
        
        # Should fail with default allowlist
        with pytest.raises(RestrictedTypeError):
            restricted_loads(pickled)
    
    def test_file_based_restricted_load(self):
        """Test file-based restricted loading."""
        data = {'safe': 'data', 'numbers': [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            temp_path = f.name
            pickle.dump(data, f)
        
        try:
            with open(temp_path, 'rb') as f:
                restored = restricted_load(f)
            assert restored == data
        finally:
            os.unlink(temp_path)


class TestSecurityScenarios:
    """Tests for real-world security scenarios."""
    
    def test_prevents_os_command_execution(self):
        """Test that OS command execution via pickle is prevented."""
        
        # Create a pickle that would execute 'id' command
        class CommandExecution:
            def __reduce__(self):
                import subprocess
                return (subprocess.check_output, (['id'],))
        
        dangerous_pickle = pickle.dumps(CommandExecution())
        
        # Should be rejected by RestrictedUnpickler
        with pytest.raises(RestrictedTypeError):
            restricted_loads(dangerous_pickle)
    
    def test_prevents_file_operations(self):
        """Test that file operations via pickle are prevented."""
        
        class FileOperation:
            def __reduce__(self):
                return (open, ('/etc/passwd', 'r'))
        
        dangerous_pickle = pickle.dumps(FileOperation())
        
        with pytest.raises(RestrictedTypeError):
            restricted_loads(dangerous_pickle)
    
    def test_prevents_network_operations(self):
        """Test that network operations via pickle are prevented."""
        
        class NetworkOperation:
            def __reduce__(self):
                import socket
                s = socket.socket()
                return (s.connect, (('evil.com', 80),))
        
        dangerous_pickle = pickle.dumps(NetworkOperation())
        
        with pytest.raises(RestrictedTypeError):
            restricted_loads(dangerous_pickle)


class TestIntegration:
    """Integration tests for secure pickle workflows."""
    
    def test_secure_pickle_workflow(self):
        """Test complete workflow with SecurePickle."""
        # Simulate saving trusted data
        trusted_data = {
            'model_config': {'layers': 12, 'hidden_size': 768},
            'metadata': {'version': '1.0', 'author': 'test'},
        }
        
        test_key = b"integration-test-key-" + b"X" * 32
        sp = SecurePickle(secret_key=test_key)
        
        # Save
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            temp_path = f.name
            sp.dump(trusted_data, f)
        
        try:
            # Load and verify
            with open(temp_path, 'rb') as f:
                restored = sp.load(f)
            
            assert restored == trusted_data
        finally:
            os.unlink(temp_path)
    
    def test_restricted_unpickler_workflow(self):
        """Test complete workflow with RestrictedUnpickler."""
        # Simulate receiving data from untrusted source
        untrusted_data = {
            'user_input': 'some data',
            'counts': [1, 2, 3, 4, 5],
            'config': {'enabled': True},
        }
        
        # Serialize (simulating untrusted source)
        pickled = pickle.dumps(untrusted_data)
        
        # Safely deserialize
        restored = restricted_loads(pickled)
        
        assert restored == untrusted_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
