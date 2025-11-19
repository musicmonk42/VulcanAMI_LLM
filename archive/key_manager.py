# src/key_manager.py
"""
Graphix Key Manager (Production-Ready)
======================================
Version: 2.0.0 - All issues fixed, validated, production-ready
Basic key management for agents using standard library hashes.
"""

import json
import hashlib
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
MAX_AGENT_ID_LENGTH = 256
MIN_AGENT_ID_LENGTH = 1
MAX_MESSAGE_LENGTH = 10_000_000  # 10MB
HASH_ALGORITHM = 'sha256'
ENTROPY_BYTES = 32


class KeyManager:
    """
    Production-ready hash-based key manager for single agent.
    
    Features:
    - Secure key generation with entropy
    - Atomic file writes
    - Comprehensive validation
    - Message signing and verification
    - Persistent key storage
    """

    def __init__(self, agent_id: str, keystore_dir: str = "keystore"):
        """
        Initialize KeyManager for a single agent.
        
        Args:
            agent_id: Unique identifier for the agent
            keystore_dir: Directory for storing keys
        
        Raises:
            ValueError: If agent_id is invalid
            TypeError: If arguments have wrong types
        """
        # Validate agent_id type
        if not isinstance(agent_id, str):
            raise TypeError(f"agent_id must be string, got {type(agent_id)}")
        
        # Validate agent_id content
        if not agent_id or len(agent_id) < MIN_AGENT_ID_LENGTH:
            raise ValueError("agent_id must be a non-empty string")
        
        if len(agent_id) > MAX_AGENT_ID_LENGTH:
            raise ValueError(f"agent_id too long (max {MAX_AGENT_ID_LENGTH} characters)")
        
        # Validate agent_id characters (alphanumeric, underscore, hyphen)
        if not all(c.isalnum() or c in ('_', '-') for c in agent_id):
            raise ValueError("agent_id must contain only alphanumeric characters, underscores, and hyphens")
        
        # Validate keystore_dir type
        if not isinstance(keystore_dir, str):
            raise TypeError(f"keystore_dir must be string, got {type(keystore_dir)}")
        
        self.agent_id = agent_id
        self.keystore_path = Path(keystore_dir) / f"{agent_id}_key.json"
        self.logger = logging.getLogger(f"KeyManager-{agent_id}")
        
        # Single key (not a dict) since this manages one agent
        self.key_hash: Optional[str] = None
        
        # Load existing key if available
        self._load_key_from_file()
        
        self.logger.info(f"KeyManager initialized for agent: {agent_id}")

    def _load_key_from_file(self):
        """
        Load key from file if it exists.
        
        This method attempts to load an existing key from persistent storage.
        If the file doesn't exist or is corrupted, it logs appropriately but
        doesn't fail - a new key can be generated later.
        """
        if not self.keystore_path.exists():
            self.logger.info("No existing key file found")
            return
        
        try:
            with open(self.keystore_path, "r") as f:
                data = json.load(f)
                
                # Validate loaded data
                if not isinstance(data, dict):
                    self.logger.warning("Key file has invalid format")
                    return
                
                self.key_hash = data.get("key_hash")
                
                if self.key_hash:
                    # Validate key_hash format (should be hex string)
                    if not isinstance(self.key_hash, str):
                        self.logger.warning("Invalid key_hash type in file")
                        self.key_hash = None
                        return
                    
                    # SHA-256 produces 64 hex characters
                    if len(self.key_hash) != 64:
                        self.logger.warning("Invalid key_hash length in file")
                        self.key_hash = None
                        return
                    
                    created_at = data.get("created_at", "unknown")
                    self.logger.info(f"Loaded existing key (created: {created_at})")
                else:
                    self.logger.warning("Key file exists but no key_hash found")
        
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON in key file: {e}, will generate new key")
        except Exception as e:
            self.logger.error(f"Failed to load key: {e}")

    def _save_key_to_file(self, key_hash: str):
        """
        Save key to file with atomic write.
        
        Args:
            key_hash: The key hash to save
        
        Raises:
            ValueError: If key_hash is invalid
            IOError: If save operation fails
        """
        # Validate key_hash
        if not isinstance(key_hash, str):
            raise TypeError(f"key_hash must be string, got {type(key_hash)}")
        
        if len(key_hash) != 64:
            raise ValueError(f"Invalid key_hash length: {len(key_hash)}")
        
        try:
            # Create parent directory if needed
            self.keystore_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write using temp file
            temp_path = self.keystore_path.with_suffix('.tmp')
            
            # Prepare data to save
            save_data = {
                "key_hash": key_hash,
                "agent_id": self.agent_id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0.0"
            }
            
            # Write to temp file with proper flushing
            with open(temp_path, "w") as f:
                json.dump(save_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename (on POSIX systems)
            temp_path.rename(self.keystore_path)
            
            self.logger.info(f"Saved key to {self.keystore_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save key: {e}")
            # Clean up temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            raise IOError(f"Failed to save key to {self.keystore_path}: {e}")

    def generate_key(self) -> str:
        """
        Generate and save a cryptographically strong hash-based key.
        
        Uses system entropy (os.urandom) combined with timestamp and agent_id
        to generate a unique key hash.
        
        Returns:
            The generated key hash (hex string)
        
        Raises:
            IOError: If key cannot be saved
        """
        # Generate key with strong entropy
        timestamp = datetime.utcnow().isoformat()
        random_bytes = os.urandom(ENTROPY_BYTES)
        
        # Combine components for key generation
        key_data = f"{self.agent_id}_{timestamp}_{random_bytes.hex()}"
        
        # Generate hash
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Store in memory
        self.key_hash = key_hash
        
        # Persist to disk
        self._save_key_to_file(key_hash)
        
        self.logger.info(f"Key generated for {self.agent_id}")
        
        return key_hash

    def sign(self, message: str) -> str:
        """
        Sign a message with the agent's key.
        
        Creates a cryptographic signature by hashing the message combined
        with the agent's key.
        
        Args:
            message: The message to sign
        
        Returns:
            Signature as hex string
        
        Raises:
            TypeError: If message is not a string
            ValueError: If no key is available
        """
        # Validate message type
        if not isinstance(message, str):
            raise TypeError(f"message must be string, got {type(message)}")
        
        # Validate message length
        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message too long: {len(message)} > {MAX_MESSAGE_LENGTH}")
        
        # Check key availability
        if not self.key_hash:
            raise ValueError(
                "No key for agent. Please generate one first using generate_key()"
            )
        
        try:
            # Create signature
            signature_data = message + self.key_hash
            signed = hashlib.sha256(signature_data.encode()).hexdigest()
            
            self.logger.debug(f"Signed message of length {len(message)}")
            
            return signed
        
        except Exception as e:
            self.logger.error(f"Signing failed: {e}")
            raise

    def verify(self, message: str, signature: str) -> bool:
        """
        Verify a message signature.
        
        Args:
            message: The original message
            signature: The signature to verify
        
        Returns:
            True if signature is valid, False otherwise
        """
        # Validate inputs
        if not isinstance(message, str):
            self.logger.warning(f"Invalid message type: {type(message)}")
            return False
        
        if not isinstance(signature, str):
            self.logger.warning(f"Invalid signature type: {type(signature)}")
            return False
        
        # Validate message length
        if len(message) > MAX_MESSAGE_LENGTH:
            self.logger.warning(f"Message too long: {len(message)}")
            return False
        
        # Validate signature format (should be 64 hex chars)
        if len(signature) != 64:
            self.logger.warning(f"Invalid signature length: {len(signature)}")
            return False
        
        # Check key availability
        if not self.key_hash:
            self.logger.warning("No key available for verification")
            return False
        
        try:
            # Compute expected signature
            signature_data = message + self.key_hash
            expected = hashlib.sha256(signature_data.encode()).hexdigest()
            
            # Constant-time comparison (helps prevent timing attacks)
            is_valid = hmac_compare_digest(signature, expected)
            
            if is_valid:
                self.logger.debug("Signature verified successfully")
            else:
                self.logger.warning("Signature verification failed")
            
            return is_valid
        
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return False

    def has_key(self) -> bool:
        """
        Check if a key is currently loaded.
        
        Returns:
            True if key is available, False otherwise
        """
        return self.key_hash is not None

    def get_agent_id(self) -> str:
        """
        Get the agent ID this KeyManager is managing.
        
        Returns:
            Agent ID string
        """
        return self.agent_id

    def rotate_key(self) -> str:
        """
        Generate a new key, replacing the old one.
        
        This is useful for periodic key rotation security practices.
        
        Returns:
            The new key hash
        
        Raises:
            IOError: If new key cannot be saved
        """
        old_key = self.key_hash
        new_key = self.generate_key()
        
        if old_key:
            self.logger.info(f"Key rotated for {self.agent_id}")
        else:
            self.logger.info(f"Initial key generated for {self.agent_id}")
        
        return new_key

    def delete_key(self):
        """
        Delete the key from memory and disk.
        
        Warning: This operation cannot be undone. Use with caution.
        """
        try:
            # Clear from memory
            self.key_hash = None
            
            # Delete from disk if exists
            if self.keystore_path.exists():
                self.keystore_path.unlink()
                self.logger.info(f"Key deleted for {self.agent_id}")
            else:
                self.logger.info(f"No key file to delete for {self.agent_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to delete key: {e}")
            raise IOError(f"Failed to delete key: {e}")


def hmac_compare_digest(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        True if strings are equal, False otherwise
    """
    # Use hmac.compare_digest if available (Python 3.3+)
    try:
        import hmac
        return hmac.compare_digest(a, b)
    except (ImportError, AttributeError):
        # Fallback for older Python versions
        if len(a) != len(b):
            return False
        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)
        return result == 0


# Demo and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Key Manager - Production Demo")
    print("=" * 60)
    
    # Test 1: Initialize KeyManager
    print("\n1. Initialize KeyManager:")
    km = KeyManager("agent_001", keystore_dir="test_keystore")
    print(f"   Agent ID: {km.get_agent_id()}")
    print(f"   Has key: {km.has_key()}")
    
    # Test 2: Generate key
    print("\n2. Generate Key:")
    key_hash = km.generate_key()
    print(f"   Key hash: {key_hash[:16]}...{key_hash[-16:]}")
    print(f"   Has key: {km.has_key()}")
    
    # Test 3: Sign message
    print("\n3. Sign Message:")
    message = "Hello, this is a test message!"
    signature = km.sign(message)
    print(f"   Message: {message}")
    print(f"   Signature: {signature[:16]}...{signature[-16:]}")
    
    # Test 4: Verify signature
    print("\n4. Verify Signature:")
    is_valid = km.verify(message, signature)
    print(f"   Valid signature: {is_valid}")
    
    # Test with wrong message
    wrong_valid = km.verify("Different message", signature)
    print(f"   Wrong message valid: {wrong_valid}")
    
    # Test 5: Load existing key
    print("\n5. Load Existing Key:")
    km2 = KeyManager("agent_001", keystore_dir="test_keystore")
    print(f"   Has key: {km2.has_key()}")
    
    # Verify with loaded key
    is_valid_loaded = km2.verify(message, signature)
    print(f"   Verify with loaded key: {is_valid_loaded}")
    
    # Test 6: Key rotation
    print("\n6. Key Rotation:")
    old_hash = km.key_hash
    new_hash = km.rotate_key()
    print(f"   Old key: {old_hash[:16] if old_hash else 'None'}...")
    print(f"   New key: {new_hash[:16]}...")
    print(f"   Keys different: {old_hash != new_hash}")
    
    # Test 7: Input validation
    print("\n7. Input Validation:")
    try:
        KeyManager("", keystore_dir="test_keystore")
        print("   ERROR: Should have raised ValueError for empty agent_id")
    except ValueError as e:
        print(f"   Correctly rejected empty agent_id: {str(e)[:50]}...")
    
    try:
        KeyManager("a" * 300, keystore_dir="test_keystore")
        print("   ERROR: Should have raised ValueError for long agent_id")
    except ValueError as e:
        print(f"   Correctly rejected long agent_id: {str(e)[:50]}...")
    
    # Test 8: Delete key
    print("\n8. Delete Key:")
    km.delete_key()
    print(f"   Has key after deletion: {km.has_key()}")
    
    # Cleanup
    print("\n9. Cleanup:")
    try:
        import shutil
        shutil.rmtree("test_keystore")
        print("   Test keystore cleaned up")
    except Exception as e:
        print(f"   Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)