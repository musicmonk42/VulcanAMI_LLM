"""
Comprehensive Test Suite for Unified KeyManager
==============================================

Tests to verify that the unified KeyManager implementation works correctly
for all three use cases and maintains backward compatibility.

Test Coverage:
1. ECC-only mode (persistence.py style)
2. Multi-algorithm mode (agent_registry.py style)
3. Agent-based mode (security_nodes.py style)
4. Backward compatibility wrappers
5. Thread safety
6. Error handling
"""

import os
import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.key_manager import (
    KeyManager,
    KeyAlgorithm,
    KeyManagementError,
    KeyGenerationError,
    KeyStorageError,
    SignatureError,
    create_persistence_key_manager,
    create_registry_key_manager,
    create_agent_key_manager,
)


@pytest.fixture
def temp_keystore():
    """Create a temporary keystore directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestKeyManagerECCMode:
    """Test ECC-only mode (persistence.py compatibility)."""

    def test_auto_generate_keys(self, temp_keystore):
        """Test that keys are auto-generated on initialization."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Keys should exist
        assert km.private_key_path.exists()
        assert km.public_key_path.exists()

        # Cached keys should be loaded
        assert km._cached_private_key is not None
        assert km._cached_public_key is not None

    def test_sign_and_verify(self, temp_keystore):
        """Test sign_data and verify_signature methods."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        data = b"test message"
        signature = km.sign_data(data)

        # Signature should be hex string
        assert isinstance(signature, str)
        assert len(signature) > 0

        # Verification should succeed
        assert km.verify_signature(data, signature) is True

        # Verification should fail with wrong data
        assert km.verify_signature(b"wrong data", signature) is False

        # Verification should fail with wrong signature
        assert km.verify_signature(data, "deadbeef" * 16) is False

    def test_keys_persist_across_instances(self, temp_keystore):
        """Test that keys persist across KeyManager instances."""
        # Create first instance
        km1 = KeyManager(key_store_dir=temp_keystore, auto_generate=True)
        data = b"test message"
        signature = km1.sign_data(data)

        # Create second instance (should load existing keys)
        km2 = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Should be able to verify with second instance
        assert km2.verify_signature(data, signature) is True

    def test_file_permissions(self, temp_keystore):
        """Test that private key has restrictive permissions."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Check private key permissions (should be 0o600)
        private_key_stat = os.stat(km.private_key_path)
        permissions = oct(private_key_stat.st_mode)[-3:]
        assert permissions == "600", f"Private key permissions are {permissions}, expected 600"


class TestKeyManagerMultiAlgorithm:
    """Test multi-algorithm mode (agent_registry.py compatibility)."""

    def test_generate_rsa_2048(self, temp_keystore):
        """Test RSA-2048 key generation."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)

        assert b"BEGIN PUBLIC KEY" in public_pem
        assert b"BEGIN PRIVATE KEY" in private_pem

    def test_generate_ed25519(self, temp_keystore):
        """Test Ed25519 key generation."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.ED25519)

        assert b"BEGIN PUBLIC KEY" in public_pem
        assert b"BEGIN PRIVATE KEY" in private_pem

    def test_generate_ecdsa_p384(self, temp_keystore):
        """Test ECDSA P-384 key generation."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.ECDSA_P384)

        assert b"BEGIN PUBLIC KEY" in public_pem
        assert b"BEGIN PRIVATE KEY" in private_pem

    def test_sign_and_verify_message_rsa(self, temp_keystore):
        """Test message signing with RSA."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)

        message = b"test message"
        signature = km.sign_message(message, private_pem, KeyAlgorithm.RSA_2048)

        # Signature should be bytes
        assert isinstance(signature, bytes)
        assert len(signature) > 0

        # Verification should succeed
        assert (
            km.verify_message_signature(
                message, signature, public_pem, KeyAlgorithm.RSA_2048
            )
            is True
        )

        # Verification should fail with wrong message
        assert (
            km.verify_message_signature(
                b"wrong message", signature, public_pem, KeyAlgorithm.RSA_2048
            )
            is False
        )

    def test_sign_and_verify_message_ed25519(self, temp_keystore):
        """Test message signing with Ed25519."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.ED25519)

        message = b"test message"
        signature = km.sign_message(message, private_pem, KeyAlgorithm.ED25519)

        # Verification should succeed
        assert (
            km.verify_message_signature(
                message, signature, public_pem, KeyAlgorithm.ED25519
            )
            is True
        )


class TestKeyManagerAgentMode:
    """Test agent-based mode (security_nodes.py compatibility)."""

    def test_agent_scoped_directory(self, temp_keystore):
        """Test that agent_id creates scoped subdirectory."""
        agent_id = "test_agent_001"
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id=agent_id, auto_generate=False
        )

        # Directory should include agent_id
        assert agent_id in str(km.key_store_dir)
        assert km.key_store_dir.exists()

    def test_store_and_retrieve_key(self, temp_keystore):
        """Test key storage and retrieval."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        # Store a key
        km.store_key("api_key", b"secret_token_123")

        # Retrieve the key
        retrieved = km.get_key("api_key")
        assert retrieved == b"secret_token_123"

    def test_get_nonexistent_key(self, temp_keystore):
        """Test retrieving non-existent key returns None."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        assert km.get_key("nonexistent") is None

    def test_list_keys(self, temp_keystore):
        """Test listing stored keys."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        # Store multiple keys
        km.store_key("key1", "value1")
        km.store_key("key2", "value2")
        km.store_key("key3", "value3")

        # List should contain all keys
        keys = km.list_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_delete_key(self, temp_keystore):
        """Test key deletion."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        # Store and delete key
        km.store_key("temp_key", "temp_value")
        assert km.delete_key("temp_key") is True

        # Key should no longer exist
        assert km.get_key("temp_key") is None

        # Deleting non-existent key should return False
        assert km.delete_key("nonexistent") is False


class TestBackwardCompatibility:
    """Test backward compatibility wrappers."""

    def test_persistence_wrapper(self, temp_keystore):
        """Test persistence.py style wrapper."""
        from src.persistence import KeyManager as PersistenceKeyManager

        km = PersistenceKeyManager(keys_dir=temp_keystore)

        # Test sign_data and verify_signature
        data = b"test data"
        signature = km.sign_data(data)
        assert km.verify_signature(data, signature) is True

    def test_agent_registry_wrapper(self, temp_keystore):
        """Test agent_registry.py style wrapper."""
        from src.agent_registry import KeyManager as RegistryKeyManager, KeyAlgorithm

        km = RegistryKeyManager(key_store_dir=str(temp_keystore))

        # Test generate_key_pair
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)
        assert public_pem is not None
        assert private_pem is not None

        # Test sign_message and verify_signature
        message = b"test message"
        signature = km.sign_message(message, private_pem, KeyAlgorithm.RSA_2048)
        assert (
            km.verify_signature(message, signature, public_pem, KeyAlgorithm.RSA_2048)
            is True
        )


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_persistence_key_manager(self, temp_keystore):
        """Test persistence factory function."""
        km = create_persistence_key_manager(temp_keystore)

        # Should auto-generate keys
        assert km._cached_private_key is not None
        assert km._cached_public_key is not None

        # Should use ECDSA P256
        assert km.algorithm == KeyAlgorithm.ECDSA_P256

    def test_create_registry_key_manager(self, temp_keystore):
        """Test registry factory function."""
        km = create_registry_key_manager(str(temp_keystore))

        # Should NOT auto-generate
        assert km._cached_private_key is None

        # Should support multi-algorithm
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_4096)
        assert public_pem is not None

    def test_create_agent_key_manager(self, temp_keystore):
        """Test agent factory function."""
        km = create_agent_key_manager("test_agent_123")

        # Should have agent_id
        assert km.agent_id == "test_agent_123"

        # Should support key storage
        km.store_key("test", "value")
        assert km.get_key("test") == "value"


class TestThreadSafety:
    """Test thread safety of KeyManager."""

    def test_concurrent_sign_operations(self, temp_keystore):
        """Test that concurrent signing operations are safe."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        results = []
        lock = threading.Lock()

        def sign_data():
            data = b"test message " + os.urandom(16)
            signature = km.sign_data(data)
            with lock:
                results.append((data, signature))

        # Run multiple threads
        threads = [threading.Thread(target=sign_data) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 10

        # All signatures should be valid
        for data, signature in results:
            assert km.verify_signature(data, signature) is True

    def test_concurrent_key_storage(self, temp_keystore):
        """Test concurrent key storage operations."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        def store_keys(start_idx):
            for i in range(start_idx, start_idx + 10):
                km.store_key(f"key_{i}", f"value_{i}")

        # Run multiple threads
        threads = [threading.Thread(target=store_keys, args=(i * 10,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All keys should be stored
        keys = km.list_keys()
        assert len(keys) == 50


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_signature_format(self, temp_keystore):
        """Test handling of invalid signature format."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Invalid hex string should return False, not raise
        assert km.verify_signature(b"data", "not_a_valid_hex") is False

    def test_key_generation_error_handling(self, temp_keystore):
        """Test that key generation errors are properly handled."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=False)

        # This should work (valid algorithm)
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.ECDSA_P256)
        assert public_pem is not None

    def test_corrupt_key_handling(self, temp_keystore):
        """Test handling of corrupted key files."""
        # Create KeyManager with valid keys
        km1 = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Corrupt the private key file
        with open(km1.private_key_path, "w") as f:
            f.write("CORRUPT KEY DATA")

        # Creating new instance should raise error
        with pytest.raises(KeyManagementError):
            km2 = KeyManager(key_store_dir=temp_keystore, auto_generate=True)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_data_signing(self, temp_keystore):
        """Test signing empty data."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Should handle empty data
        signature = km.sign_data(b"")
        assert km.verify_signature(b"", signature) is True

    def test_large_data_signing(self, temp_keystore):
        """Test signing large amounts of data."""
        km = KeyManager(key_store_dir=temp_keystore, auto_generate=True)

        # Large data (1MB)
        large_data = os.urandom(1024 * 1024)
        signature = km.sign_data(large_data)
        assert km.verify_signature(large_data, signature) is True

    def test_unicode_key_names(self, temp_keystore):
        """Test agent key storage with unicode names."""
        km = KeyManager(
            key_store_dir=temp_keystore, agent_id="test_agent", auto_generate=False
        )

        # Unicode key names should work
        km.store_key("key_✓", "value")
        km.store_key("key_日本語", "value")
        assert km.get_key("key_✓") == "value"
        assert km.get_key("key_日本語") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
