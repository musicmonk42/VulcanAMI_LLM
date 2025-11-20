"""
Secure Pickle Utilities
=======================

This module provides secure alternatives to Python pickle module to prevent
deserialization attacks (CWE-502). Use these utilities instead of direct pickle
usage when dealing with untrusted data.

Security Features:
- HMAC signature verification for integrity
- Restricted unpickler to allow only safe types
- Comprehensive error handling and logging

Author: Security Team
Version: 1.0.0
"""

import pickle
import hmac
import hashlib
import os
import logging
import io
from typing import Any, BinaryIO
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurePickleError(Exception):
    """Base exception for secure pickle operations."""
    pass

class SignatureVerificationError(SecurePickleError):
    """Raised when HMAC signature verification fails."""
    pass

class RestrictedTypeError(SecurePickleError):
    """Raised when attempting to unpickle forbidden type."""
    pass


# Safe modules and types for RestrictedUnpickler
SAFE_BUILTINS = {
    'dict', 'list', 'tuple', 'set', 'frozenset',
    'int', 'float', 'str', 'bool', 'bytes', 'bytearray',
    'complex', 'NoneType', 'type',
}

SAFE_MODULES = {
    'builtins': SAFE_BUILTINS,
    'collections': {'OrderedDict', 'defaultdict', 'Counter', 'deque'},
    'datetime': {'datetime', 'date', 'time', 'timedelta'},
    'numpy': {'ndarray', 'dtype'},
    'torch': {'Tensor', 'FloatTensor', 'Size'},
}


class SecurePickle:
    """
    Secure pickle wrapper with HMAC signature verification.
    
    Prevents tampering with pickled data via HMAC signatures.
    Note: Does NOT prevent code execution from malicious pickles!
    For untrusted data, use RestrictedUnpickler instead.
    """
    
    SIGNATURE_SIZE = 32  # SHA256 digest size
    
    def __init__(self, secret_key: bytes = None):
        """Initialize with secret key for HMAC."""
        if secret_key is None:
            secret_key_str = os.environ.get('PICKLE_SECRET_KEY', '')
            if not secret_key_str:
                raise ValueError(
                    "PICKLE_SECRET_KEY environment variable must be set. "
                    "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            secret_key = secret_key_str.encode('utf-8')
        
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 bytes")
        
        self.secret_key = secret_key
    
    def _compute_signature(self, data: bytes) -> bytes:
        """Compute HMAC-SHA256 signature."""
        return hmac.new(self.secret_key, data, hashlib.sha256).digest()
    
    def _verify_signature(self, signature: bytes, data: bytes) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = self._compute_signature(data)
        return hmac.compare_digest(signature, expected)
    
    def dumps(self, obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
        """Serialize object with HMAC signature."""
        pickled_data = pickle.dumps(obj, protocol=protocol)
        signature = self._compute_signature(pickled_data)
        return signature + pickled_data
    
    def loads(self, data: bytes) -> Any:
        """Deserialize object after verifying HMAC signature."""
        if len(data) < self.SIGNATURE_SIZE:
            raise SignatureVerificationError("Invalid pickle data: too short")
        
        signature = data[:self.SIGNATURE_SIZE]
        pickled_data = data[self.SIGNATURE_SIZE:]
        
        if not self._verify_signature(signature, pickled_data):
            raise SignatureVerificationError("Invalid HMAC signature")
        
        return pickle.loads(pickled_data)
    
    def dump(self, obj: Any, file: BinaryIO, protocol: int = pickle.HIGHEST_PROTOCOL):
        """Serialize object to file with HMAC signature."""
        file.write(self.dumps(obj, protocol=protocol))
    
    def load(self, file: BinaryIO) -> Any:
        """Deserialize object from file after verifying HMAC signature."""
        return self.loads(file.read())


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe types.
    
    Prevents code execution by restricting which classes can be unpickled.
    This is the RECOMMENDED approach for loading untrusted pickle data.
    """
    
    def __init__(self, file, *, fix_imports=True, encoding="ASCII",
                 errors="strict", buffers=None, safe_modules=None):
        """Initialize restricted unpickler."""
        super().__init__(file, fix_imports=fix_imports, encoding=encoding,
                        errors=errors, buffers=buffers)
        self.safe_modules = safe_modules if safe_modules is not None else SAFE_MODULES
    
    def find_class(self, module: str, name: str):
        """Override find_class to restrict allowed types."""
        if module in self.safe_modules:
            if name in self.safe_modules[module]:
                return super().find_class(module, name)
        
        logger.error(f"Attempted to unpickle forbidden class: {module}.{name}")
        raise RestrictedTypeError(
            f"Forbidden class: {module}.{name} is not in allowlist"
        )


def restricted_loads(data: bytes, safe_modules: dict = None) -> Any:
    """Safely load pickle data with restricted types."""
    return RestrictedUnpickler(io.BytesIO(data), safe_modules=safe_modules).load()


def restricted_load(file: BinaryIO, safe_modules: dict = None) -> Any:
    """Safely load pickle file with restricted types."""
    return RestrictedUnpickler(file, safe_modules=safe_modules).load()
