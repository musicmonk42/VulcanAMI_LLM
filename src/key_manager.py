"""
Unified KeyManager Implementation
==================================

A comprehensive, production-grade cryptographic key management system that consolidates
three previously incompatible KeyManager implementations into a single, unified API.

This module provides:
- Support for multiple key algorithms (RSA, Ed25519, ECDSA)
- Secure key generation, storage, and retrieval
- Digital signature operations (sign/verify)
- Thread-safe operations with proper locking
- Flexible initialization for different use cases
- Backward compatibility with existing code

Design Philosophy:
-----------------
This implementation follows the Single Responsibility Principle and provides a clean
separation between key storage backends and cryptographic operations. It supports
three primary use cases that were previously handled by separate implementations:

1. ECC-only use case (from persistence.py)
   - SECP256R1 elliptic curve
   - Sign/verify with ECDSA
   - Simple file-based persistence

2. Multi-algorithm use case (from agent_registry.py)
   - RSA-2048, RSA-4096
   - Ed25519
   - ECDSA (P256, P384, P521)
   - Advanced key management

3. Agent-based use case (from security_nodes.py)
   - Dictionary-based key storage
   - Agent-scoped keys
   - Simple get/store operations

Industry Standards:
------------------
- FIPS 186-4 compliant key generation
- Proper file permissions (0o600 for private keys)
- Thread-safe with RLock for reentrant operations
- Comprehensive error handling
- Type hints throughout
- Extensive documentation

Author: VulcanAMI Team
Version: 1.0.0
License: MIT
"""

import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

# Configure logging
logger = logging.getLogger("KeyManager")


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================


class KeyAlgorithm(Enum):
    """Supported cryptographic algorithms."""

    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa_p256"  # SECP256R1
    ECDSA_P384 = "ecdsa_p384"  # SECP384R1
    ECDSA_P521 = "ecdsa_p521"  # SECP521R1


# Default algorithm for backward compatibility with persistence.py
DEFAULT_ALGORITHM = KeyAlgorithm.ECDSA_P256

# File permissions for private keys
PRIVATE_KEY_PERMS = 0o600
PUBLIC_KEY_PERMS = 0o644


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================


class KeyManagementError(Exception):
    """Base exception for key management errors."""


class KeyGenerationError(KeyManagementError):
    """Raised when key generation fails."""


class KeyStorageError(KeyManagementError):
    """Raised when key storage operations fail."""


class KeyRetrievalError(KeyManagementError):
    """Raised when key retrieval fails."""


class SignatureError(KeyManagementError):
    """Raised when signature operations fail."""


# ============================================================
# UNIFIED KEY MANAGER
# ============================================================


class KeyManager:
    """
    Unified cryptographic key manager supporting multiple algorithms and use cases.

    This class consolidates three previously separate KeyManager implementations
    into a single, comprehensive API that maintains backward compatibility.

    Features:
        - Multiple key algorithms (RSA, Ed25519, ECDSA)
        - Secure file-based key storage
        - Thread-safe operations
        - Agent-scoped key management
        - Signature generation and verification
        - Automatic key generation on first use

    Thread Safety:
        All methods are thread-safe using RLock to support reentrant calls.

    Example Usage:
        # Use case 1: ECC-only (persistence.py style)
        km = KeyManager(key_store_dir=Path("keys"))
        signature = km.sign_data(b"message")
        is_valid = km.verify_signature(b"message", signature)

        # Use case 2: Multi-algorithm (agent_registry.py style)
        km = KeyManager(key_store_dir="keys", algorithm=KeyAlgorithm.RSA_2048)
        public_pem, private_pem = km.generate_key_pair()
        signature = km.sign_message(b"message", private_pem)

        # Use case 3: Agent-based (security_nodes.py style)
        km = KeyManager(agent_id="agent_001")
        km.store_key("key1", b"key_data")
        key_data = km.get_key("key1")
    """

    def __init__(
        self,
        key_store_dir: Optional[Union[str, Path]] = None,
        agent_id: Optional[str] = None,
        algorithm: KeyAlgorithm = DEFAULT_ALGORITHM,
        auto_generate: bool = True,
    ):
        """
        Initialize the KeyManager.

        Args:
            key_store_dir: Directory for key storage. If None, uses "keystore" or
                          agent-specific directory.
            agent_id: Agent identifier for agent-scoped keys. If provided, keys
                     are stored in an agent-specific subdirectory.
            algorithm: Default algorithm for key generation.
            auto_generate: If True, automatically generate keys on first access.

        Raises:
            KeyManagementError: If initialization fails.
        """
        self.algorithm = algorithm
        self.auto_generate = auto_generate
        self.agent_id = agent_id

        # Setup key storage directory
        if key_store_dir is None:
            if agent_id:
                self.key_store_dir = Path("keystore") / agent_id
            else:
                self.key_store_dir = Path("keystore")
        else:
            self.key_store_dir = Path(key_store_dir)
            if agent_id:
                self.key_store_dir = self.key_store_dir / agent_id

        # Create directory with secure permissions
        try:
            self.key_store_dir.mkdir(parents=True, exist_ok=True)
            # Set directory permissions to 0o700 (owner only)
            os.chmod(self.key_store_dir, 0o700)
        except Exception as e:
            raise KeyManagementError(f"Failed to create key store directory: {e}")

        # Thread safety
        self.lock = threading.RLock()

        # Cryptography backend
        self.backend = default_backend()

        # Default key paths (for ECC-only mode compatibility)
        self.private_key_path = self.key_store_dir / "private_key.pem"
        self.public_key_path = self.key_store_dir / "public_key.pem"

        # Cached keys (for ECC-only mode)
        self._cached_private_key = None
        self._cached_public_key = None

        # Agent-specific key storage (dict-based for agent mode)
        self.keys: Dict[str, Any] = {}

        # Load or generate default keys if in ECC-only mode and auto_generate is True
        if auto_generate and not agent_id:
            self._ensure_default_keys()

        logger.debug(
            f"KeyManager initialized: dir={self.key_store_dir}, "
            f"algorithm={algorithm.value}, agent_id={agent_id}"
        )

    # ============================================================
    # KEY GENERATION
    # ============================================================

    def generate_key_pair(
        self, algorithm: Optional[KeyAlgorithm] = None
    ) -> Tuple[bytes, bytes]:
        """
        Generate a new cryptographic key pair.

        This method generates a public/private key pair using the specified algorithm.
        The keys are returned as PEM-encoded bytes but are NOT automatically stored.
        Use save_key_pair() to persist the keys to disk.

        Args:
            algorithm: Algorithm to use. If None, uses the manager's default.

        Returns:
            Tuple of (public_pem, private_pem) as bytes.

        Raises:
            KeyGenerationError: If key generation fails.

        Example:
            public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)
            # Keys are in memory but not saved yet
        """
        algo = algorithm or self.algorithm

        with self.lock:
            try:
                # Generate private key based on algorithm
                if algo == KeyAlgorithm.RSA_2048:
                    private_key = rsa.generate_private_key(
                        public_exponent=65537, key_size=2048, backend=self.backend
                    )
                elif algo == KeyAlgorithm.RSA_4096:
                    private_key = rsa.generate_private_key(
                        public_exponent=65537, key_size=4096, backend=self.backend
                    )
                elif algo == KeyAlgorithm.ED25519:
                    private_key = ed25519.Ed25519PrivateKey.generate()
                elif algo == KeyAlgorithm.ECDSA_P256:
                    private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
                elif algo == KeyAlgorithm.ECDSA_P384:
                    private_key = ec.generate_private_key(ec.SECP384R1(), self.backend)
                elif algo == KeyAlgorithm.ECDSA_P521:
                    private_key = ec.generate_private_key(ec.SECP521R1(), self.backend)
                else:
                    raise KeyGenerationError(f"Unsupported algorithm: {algo}")

                # Serialize private key
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                    # NOTE: In production, consider using BestAvailableEncryption(password)
                )

                # Extract and serialize public key
                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )

                logger.debug(f"Generated key pair using algorithm: {algo.value}")
                return public_pem, private_pem

            except Exception as e:
                raise KeyGenerationError(f"Failed to generate key pair: {e}") from e

    def _ensure_default_keys(self):
        """
        Ensure default ECC keys exist (for persistence.py compatibility).

        This is called automatically during initialization if auto_generate=True.
        It loads existing keys or generates new ones if they don't exist.

        Raises:
            KeyManagementError: If key loading/generation fails.
        """
        with self.lock:
            if self._cached_private_key is not None:
                return  # Keys already loaded

            if self.private_key_path.exists() and self.public_key_path.exists():
                try:
                    # Load existing keys
                    with open(self.private_key_path, "rb") as f:
                        self._cached_private_key = serialization.load_pem_private_key(
                            f.read(), password=None, backend=self.backend
                        )

                    with open(self.public_key_path, "rb") as f:
                        self._cached_public_key = serialization.load_pem_public_key(
                            f.read(), backend=self.backend
                        )

                    logger.info(f"Loaded existing keys from {self.key_store_dir}")
                    return

                except Exception as e:
                    logger.error(f"Failed to load existing keys: {e}")
                    raise KeyManagementError(f"Failed to load keys: {e}") from e

            # Generate new keys
            logger.info(f"Generating new {self.algorithm.value} keys")
            try:
                public_pem, private_pem = self.generate_key_pair()

                # Load into memory
                self._cached_private_key = serialization.load_pem_private_key(
                    private_pem, password=None, backend=self.backend
                )
                self._cached_public_key = serialization.load_pem_public_key(
                    public_pem, backend=self.backend
                )

                # Save to disk
                self._save_default_keys(private_pem, public_pem)

                logger.info(f"Generated and saved new keys to {self.key_store_dir}")

            except Exception as e:
                raise KeyManagementError(f"Failed to generate/save keys: {e}") from e

    def _save_default_keys(self, private_pem: bytes, public_pem: bytes):
        """
        Save default keys to disk with proper permissions.

        Args:
            private_pem: PEM-encoded private key
            public_pem: PEM-encoded public key

        Raises:
            KeyStorageError: If saving fails.
        """
        try:
            # Save private key with restrictive permissions
            with open(self.private_key_path, "wb") as f:
                f.write(private_pem)
            os.chmod(self.private_key_path, PRIVATE_KEY_PERMS)

            # Save public key
            with open(self.public_key_path, "wb") as f:
                f.write(public_pem)
            os.chmod(self.public_key_path, PUBLIC_KEY_PERMS)

            logger.debug(f"Saved keys to {self.key_store_dir}")

        except Exception as e:
            raise KeyStorageError(f"Failed to save keys: {e}") from e

    # ============================================================
    # SIGNATURE OPERATIONS (persistence.py style)
    # ============================================================

    def sign_data(self, data: bytes) -> str:
        """
        Sign data using the default ECDSA key and return hex signature.

        This is the persistence.py-compatible API. It uses the cached default
        key to sign the data.

        Args:
            data: Data to sign

        Returns:
            Hex-encoded signature string

        Raises:
            SignatureError: If signing fails

        Example:
            signature = km.sign_data(b"my message")
            # signature is a hex string
        """
        with self.lock:
            try:
                self._ensure_default_keys()

                if not isinstance(self._cached_private_key, ec.EllipticCurvePrivateKey):
                    raise SignatureError(
                        "Default key is not ECDSA. Use sign_message() for other algorithms."
                    )

                signature_bytes = self._cached_private_key.sign(
                    data, ec.ECDSA(hashes.SHA256())
                )
                return signature_bytes.hex()

            except Exception as e:
                raise SignatureError(f"Failed to sign data: {e}") from e

    def verify_signature(self, data: bytes, signature: str) -> bool:
        """
        Verify a signature using the default ECDSA public key.

        This is the persistence.py-compatible API. It uses the cached default
        public key to verify the signature.

        Args:
            data: Original data that was signed
            signature: Hex-encoded signature string

        Returns:
            True if signature is valid, False otherwise

        Example:
            is_valid = km.verify_signature(b"my message", signature_hex)
        """
        with self.lock:
            try:
                self._ensure_default_keys()

                if not isinstance(self._cached_public_key, ec.EllipticCurvePublicKey):
                    raise SignatureError(
                        "Default key is not ECDSA. Use verify_message_signature() for other algorithms."
                    )

                signature_bytes = bytes.fromhex(signature)
                self._cached_public_key.verify(
                    signature_bytes, data, ec.ECDSA(hashes.SHA256())
                )
                return True

            except (InvalidSignature, ValueError) as e:
                logger.debug(f"Signature verification failed: {e}")
                return False
            except Exception as e:
                raise SignatureError(f"Error during verification: {e}") from e

    # ============================================================
    # SIGNATURE OPERATIONS (agent_registry.py style)
    # ============================================================

    def sign_message(
        self,
        message: bytes,
        private_key_pem: bytes,
        algorithm: Optional[KeyAlgorithm] = None,
    ) -> bytes:
        """
        Sign a message with a specific private key.

        This is the agent_registry.py-compatible API. It accepts a PEM-encoded
        private key and signs the message using the appropriate algorithm.

        Args:
            message: Message to sign
            private_key_pem: PEM-encoded private key
            algorithm: Algorithm to use. If None, uses manager's default.

        Returns:
            Raw signature bytes

        Raises:
            SignatureError: If signing fails

        Example:
            public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)
            signature = km.sign_message(b"message", private_pem, KeyAlgorithm.RSA_2048)
        """
        algo = algorithm or self.algorithm

        with self.lock:
            try:
                # Load private key
                private_key = serialization.load_pem_private_key(
                    private_key_pem, password=None, backend=self.backend
                )

                # Sign based on algorithm
                if algo in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
                    signature = private_key.sign(
                        message,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH,
                        ),
                        hashes.SHA256(),
                    )
                elif algo == KeyAlgorithm.ED25519:
                    signature = private_key.sign(message)
                elif algo in [
                    KeyAlgorithm.ECDSA_P256,
                    KeyAlgorithm.ECDSA_P384,
                    KeyAlgorithm.ECDSA_P521,
                ]:
                    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
                else:
                    raise SignatureError(f"Unsupported algorithm: {algo}")

                return signature

            except Exception as e:
                raise SignatureError(f"Failed to sign message: {e}") from e

    def verify_message_signature(
        self,
        message: bytes,
        signature: bytes,
        public_key_pem: bytes,
        algorithm: Optional[KeyAlgorithm] = None,
    ) -> bool:
        """
        Verify a message signature with a specific public key.

        This is the agent_registry.py-compatible API. It accepts PEM-encoded keys
        and verifies using the appropriate algorithm.

        Args:
            message: Original message
            signature: Signature bytes
            public_key_pem: PEM-encoded public key
            algorithm: Algorithm to use. If None, uses manager's default.

        Returns:
            True if signature is valid, False otherwise

        Example:
            is_valid = km.verify_message_signature(
                b"message", signature, public_pem, KeyAlgorithm.RSA_2048
            )
        """
        algo = algorithm or self.algorithm

        with self.lock:
            try:
                # Load public key
                public_key = serialization.load_pem_public_key(
                    public_key_pem, backend=self.backend
                )

                # Verify based on algorithm
                if algo in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
                    public_key.verify(
                        signature,
                        message,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH,
                        ),
                        hashes.SHA256(),
                    )
                elif algo == KeyAlgorithm.ED25519:
                    public_key.verify(signature, message)
                elif algo in [
                    KeyAlgorithm.ECDSA_P256,
                    KeyAlgorithm.ECDSA_P384,
                    KeyAlgorithm.ECDSA_P521,
                ]:
                    public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
                else:
                    raise SignatureError(f"Unsupported algorithm: {algo}")

                return True

            except InvalidSignature:
                logger.debug("Signature verification failed")
                return False
            except Exception as e:
                logger.error(f"Error during signature verification: {e}")
                return False

    # ============================================================
    # AGENT-BASED KEY STORAGE (security_nodes.py style)
    # ============================================================

    def store_key(self, key_id: str, key_data: Any):
        """
        Store a key in the agent-specific key store.

        This is the security_nodes.py-compatible API for storing arbitrary
        key data associated with a key identifier.

        Args:
            key_id: Unique identifier for the key
            key_data: Key data (can be bytes, string, or any serializable object)

        Example:
            km = KeyManager(agent_id="agent_001")
            km.store_key("fernet_key", b"...")
            km.store_key("api_token", "abc123")
        """
        with self.lock:
            self.keys[key_id] = key_data
            logger.debug(f"Stored key: {key_id} (agent: {self.agent_id})")

    def get_key(self, key_id: str) -> Optional[Any]:
        """
        Retrieve a key from the agent-specific key store.

        This is the security_nodes.py-compatible API for retrieving stored keys.

        Args:
            key_id: Unique identifier for the key

        Returns:
            The key data, or None if not found

        Example:
            key_data = km.get_key("fernet_key")
            if key_data is None:
                print("Key not found")
        """
        with self.lock:
            return self.keys.get(key_id)

    def list_keys(self) -> list:
        """
        List all stored key identifiers.

        Returns:
            List of key identifiers

        Example:
            key_ids = km.list_keys()
            print(f"Stored keys: {key_ids}")
        """
        with self.lock:
            return list(self.keys.keys())

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key from the agent-specific key store.

        Args:
            key_id: Unique identifier for the key

        Returns:
            True if key was deleted, False if not found

        Example:
            if km.delete_key("old_key"):
                print("Key deleted")
        """
        with self.lock:
            if key_id in self.keys:
                del self.keys[key_id]
                logger.debug(f"Deleted key: {key_id} (agent: {self.agent_id})")
                return True
            return False

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def get_public_key_pem(self) -> bytes:
        """
        Get the default public key as PEM bytes.

        Returns:
            PEM-encoded public key

        Raises:
            KeyRetrievalError: If key retrieval fails
        """
        with self.lock:
            try:
                self._ensure_default_keys()
                return self._cached_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            except Exception as e:
                raise KeyRetrievalError(f"Failed to get public key: {e}") from e

    def get_private_key_pem(self) -> bytes:
        """
        Get the default private key as PEM bytes.

        **WARNING**: This exposes the private key. Use with caution.

        Returns:
            PEM-encoded private key

        Raises:
            KeyRetrievalError: If key retrieval fails
        """
        with self.lock:
            try:
                self._ensure_default_keys()
                return self._cached_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            except Exception as e:
                raise KeyRetrievalError(f"Failed to get private key: {e}") from e


# ============================================================
# FACTORY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================


def create_persistence_key_manager(keys_dir: Union[str, Path]) -> KeyManager:
    """
    Create a KeyManager for persistence.py use case (ECC-only).

    This factory function creates a KeyManager configured for the persistence.py
    use case with ECDSA P256 keys and automatic key generation.

    Args:
        keys_dir: Directory for key storage

    Returns:
        Configured KeyManager instance

    Example:
        km = create_persistence_key_manager(Path("keys"))
        signature = km.sign_data(b"data")
    """
    return KeyManager(
        key_store_dir=keys_dir, algorithm=KeyAlgorithm.ECDSA_P256, auto_generate=True
    )


def create_registry_key_manager(key_store_dir: str = "keys") -> KeyManager:
    """
    Create a KeyManager for agent_registry.py use case (multi-algorithm).

    This factory function creates a KeyManager configured for the agent_registry.py
    use case supporting multiple algorithms.

    Args:
        key_store_dir: Directory for key storage

    Returns:
        Configured KeyManager instance

    Example:
        km = create_registry_key_manager("keys")
        public_pem, private_pem = km.generate_key_pair(KeyAlgorithm.RSA_2048)
    """
    return KeyManager(
        key_store_dir=key_store_dir,
        algorithm=KeyAlgorithm.ECDSA_P256,
        auto_generate=False,
    )


def create_agent_key_manager(agent_id: str) -> KeyManager:
    """
    Create a KeyManager for security_nodes.py use case (agent-based).

    This factory function creates a KeyManager configured for the security_nodes.py
    use case with agent-scoped key storage.

    Args:
        agent_id: Agent identifier

    Returns:
        Configured KeyManager instance

    Example:
        km = create_agent_key_manager("agent_001")
        km.store_key("api_key", "secret")
        key = km.get_key("api_key")
    """
    return KeyManager(agent_id=agent_id, auto_generate=False)


# ============================================================
# MODULE METADATA
# ============================================================

__all__ = [
    "KeyManager",
    "KeyAlgorithm",
    "KeyManagementError",
    "KeyGenerationError",
    "KeyStorageError",
    "KeyRetrievalError",
    "SignatureError",
    "create_persistence_key_manager",
    "create_registry_key_manager",
    "create_agent_key_manager",
]

__version__ = "1.0.0"
__author__ = "VulcanAMI Team"
