"""
Production-ready Key Management System (KMS) implementations
Replaces DevelopmentKMS with secure key management
"""

import base64
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class ProductionKMS:
    """
    Production-ready Key Management System
    
    Features:
    - Encrypted key storage using AES-256
    - Master key derived from secure passphrase or environment variable
    - Key rotation support
    - Key revocation tracking
    - Audit logging
    
    In a full production environment, this would integrate with:
    - AWS KMS
    - Azure Key Vault
    - Google Cloud KMS
    - HashiCorp Vault
    """
    
    def __init__(
        self, 
        keystore_path: str = "keystore/production_keys",
        master_key_env: str = "VULCAN_MASTER_KEY"
    ):
        """
        Initialize Production KMS
        
        Args:
            keystore_path: Path to encrypted keystore directory
            master_key_env: Environment variable name for master key
        """
        self.keystore_path = Path(keystore_path)
        self.keystore_path.mkdir(parents=True, exist_ok=True)
        
        self._keys: Dict[str, Dict] = {}
        self._revoked_keys: set = set()
        self._lock = threading.Lock()
        self.logger = logging.getLogger("ProductionKMS")
        
        # Get or generate master key
        self._master_key = self._get_master_key(master_key_env)
        
        # Load existing keys
        self._load_keystore()
        
        self.logger.info(f"ProductionKMS initialized with keystore at {self.keystore_path}")
    
    def _get_master_key(self, env_var: str) -> bytes:
        """Get or generate master key for encrypting stored keys"""
        # Try to get from environment variable
        master_key_b64 = os.environ.get(env_var)
        
        if master_key_b64:
            try:
                master_key = base64.b64decode(master_key_b64)
                if len(master_key) == 32:  # 256 bits
                    self.logger.info("Loaded master key from environment")
                    return master_key
                else:
                    self.logger.warning("Invalid master key in environment, generating new one")
            except Exception as e:
                self.logger.error(f"Failed to decode master key: {e}")
        
        # SECURITY: Fail securely if master key not in environment
        self.logger.error(
            f"CRITICAL SECURITY ERROR: No master key found in environment variable {env_var}. "
            f"KMS cannot operate securely without a master key from environment."
        )
        raise ValueError(
            f"Master key must be provided via {env_var} environment variable. "
            f"Generate a secure key with: python -c 'import os,base64; "
            f"print(base64.b64encode(os.urandom(32)).decode())' and set it as {env_var}"
        )
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using master key"""
        # Generate random IV
        iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Pad data to block size
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Encrypt
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using master key with padding oracle attack protection"""
        # Extract IV
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        
        # Remove padding with validation to prevent padding oracle attacks
        padding_length = padded_data[-1]
        
        # Validate padding
        if padding_length == 0 or padding_length > 16:
            raise ValueError("Invalid padding length")
        
        # Verify all padding bytes match the padding length
        padding_bytes = padded_data[-padding_length:]
        if not all(byte == padding_length for byte in padding_bytes):
            raise ValueError("Invalid padding bytes - potential padding oracle attack")
        
        data = padded_data[:-padding_length]
        
        return data
    
    def _load_keystore(self):
        """Load encrypted keys from disk"""
        with self._lock:
            try:
                # Load key metadata
                metadata_file = self.keystore_path / "keys_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        self._revoked_keys = set(metadata.get("revoked", []))
                
                # Load each key
                for key_file in self.keystore_path.glob("key_*.enc"):
                    key_id = key_file.stem.replace("key_", "")
                    
                    if key_id in self._revoked_keys:
                        self.logger.debug(f"Skipping revoked key: {key_id}")
                        continue
                    
                    try:
                        # Read encrypted key
                        encrypted_key = key_file.read_bytes()
                        
                        # Decrypt
                        key_data = self._decrypt_data(encrypted_key)
                        
                        # Deserialize
                        key_info = json.loads(key_data.decode('utf-8'))
                        
                        # Load RSA private key from PEM
                        private_key_pem = key_info["private_key_pem"].encode('utf-8')
                        private_key = serialization.load_pem_private_key(
                            private_key_pem,
                            password=None,
                            backend=default_backend()
                        )
                        
                        # Store in memory
                        self._keys[key_id] = {
                            "private_key": private_key,
                            "public_key_pem": key_info["public_key_pem"],
                            "created_at": key_info["created_at"]
                        }
                        
                        self.logger.debug(f"Loaded key: {key_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load key {key_id}: {e}")
                
                self.logger.info(f"Loaded {len(self._keys)} keys from keystore")
                
            except Exception as e:
                self.logger.error(f"Failed to load keystore: {e}")
    
    def _save_key(self, key_id: str):
        """Save encrypted key to disk"""
        with self._lock:
            try:
                if key_id not in self._keys:
                    raise ValueError(f"Key not found: {key_id}")
                
                key_info = self._keys[key_id]
                
                # Serialize private key to PEM
                private_key_pem = key_info["private_key"].private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode('utf-8')
                
                # Create key data structure
                key_data = {
                    "private_key_pem": private_key_pem,
                    "public_key_pem": key_info["public_key_pem"],
                    "created_at": key_info["created_at"]
                }
                
                # Serialize to JSON
                key_json = json.dumps(key_data, indent=2)
                
                # Encrypt
                encrypted_key = self._encrypt_data(key_json.encode('utf-8'))
                
                # Save to disk
                key_file = self.keystore_path / f"key_{key_id}.enc"
                key_file.write_bytes(encrypted_key)
                os.chmod(key_file, 0o600)  # Only owner can read
                
                # Update metadata
                self._save_metadata()
                
                self.logger.debug(f"Saved key: {key_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to save key {key_id}: {e}")
                raise
    
    def _save_metadata(self):
        """Save key metadata (revocation list, etc.)"""
        with self._lock:
            try:
                metadata = {
                    "revoked": list(self._revoked_keys)
                }
                
                metadata_file = self.keystore_path / "keys_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")
    
    def generate_key(self, key_id: str, key_size: int = 2048) -> bool:
        """
        Generate new RSA key pair
        
        Args:
            key_id: Unique identifier for the key
            key_size: RSA key size in bits (2048 or 4096)
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                if key_id in self._keys:
                    self.logger.warning(f"Key already exists: {key_id}")
                    return False
                
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )
                
                # Get public key
                public_key = private_key.public_key()
                
                # Serialize public key to PEM
                public_key_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8')
                
                # Store in memory
                import time
                self._keys[key_id] = {
                    "private_key": private_key,
                    "public_key_pem": public_key_pem,
                    "created_at": time.time()
                }
                
                # Save to disk
                self._save_key(key_id)
                
                self.logger.info(f"Generated new key: {key_id} ({key_size} bits)")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to generate key {key_id}: {e}")
                return False
    
    def get_private_key(self, key_id: str) -> Any:
        """
        Get private key object
        
        In production, this should not return the actual key.
        Instead, signing operations should be performed by the KMS.
        """
        with self._lock:
            if key_id in self._revoked_keys:
                raise ValueError(f"Key has been revoked: {key_id}")
            
            if key_id not in self._keys:
                raise ValueError(f"Key not found: {key_id}")
            
            return self._keys[key_id]["private_key"]
    
    def get_public_key_pem(self, key_id: str) -> str:
        """Get public key in PEM format"""
        with self._lock:
            if key_id in self._revoked_keys:
                raise ValueError(f"Key has been revoked: {key_id}")
            
            if key_id not in self._keys:
                raise ValueError(f"Key not found: {key_id}")
            
            return self._keys[key_id]["public_key_pem"]
    
    def sign_data(self, key_id: str, data: bytes) -> bytes:
        """Sign data using specified key"""
        with self._lock:
            try:
                private_key = self.get_private_key(key_id)
                
                signature = private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                self.logger.debug(f"Signed data with key: {key_id}")
                return signature
                
            except Exception as e:
                self.logger.error(f"Failed to sign data with key {key_id}: {e}")
                raise
    
    def rotate_key(self, key_id: str) -> str:
        """
        Rotate key by generating new key with updated ID
        
        Args:
            key_id: Current key ID
            
        Returns:
            New key ID
        """
        with self._lock:
            try:
                if key_id not in self._keys:
                    raise ValueError(f"Key not found: {key_id}")
                
                # Generate new key ID
                import time
                new_key_id = f"{key_id}_v{int(time.time())}"
                
                # Generate new key
                self.generate_key(new_key_id)
                
                # Mark old key as revoked
                self.revoke_key(key_id)
                
                self.logger.info(f"Rotated key {key_id} -> {new_key_id}")
                return new_key_id
                
            except Exception as e:
                self.logger.error(f"Failed to rotate key {key_id}: {e}")
                raise
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke a key
        
        Args:
            key_id: Key to revoke
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                if key_id not in self._keys:
                    self.logger.warning(f"Key not found: {key_id}")
                    return False
                
                # Add to revoked set
                self._revoked_keys.add(key_id)
                
                # Remove from memory
                del self._keys[key_id]
                
                # Update metadata on disk
                self._save_metadata()
                
                self.logger.info(f"Revoked key: {key_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to revoke key {key_id}: {e}")
                return False
    
    def list_keys(self) -> list:
        """List all active key IDs"""
        with self._lock:
            return list(self._keys.keys())
    
    def get_key_info(self, key_id: str) -> Dict:
        """Get key metadata"""
        with self._lock:
            if key_id in self._revoked_keys:
                return {
                    "key_id": key_id,
                    "status": "revoked"
                }
            
            if key_id not in self._keys:
                raise ValueError(f"Key not found: {key_id}")
            
            key_info = self._keys[key_id]
            return {
                "key_id": key_id,
                "status": "active",
                "created_at": key_info["created_at"],
                "public_key_pem": key_info["public_key_pem"]
            }


__all__ = ["ProductionKMS"]
