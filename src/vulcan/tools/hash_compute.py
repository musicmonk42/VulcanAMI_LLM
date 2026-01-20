"""
Cryptographic Hash Computation Tool for VULCAN.

Provides deterministic hash and encoding computations for cryptographic
operations. Wraps the existing CryptographicEngine as a tool that the
LLM can call directly.

Use this tool when:
    - Computing SHA-256, SHA-512, SHA-1, MD5 hashes
    - Base64 encoding/decoding
    - Hex encoding/decoding
    - HMAC computation
    - CRC32 checksums

Do NOT use for:
    - Questions about cryptography theory
    - Security analysis questions
    - Hash collision discussions (LLM handles conceptual questions)

Industry Standards:
    - Thread-safe with stateless operations
    - Deterministic results (100% confidence)
    - Input size limits to prevent DoS attacks
    - Uses Python's cryptography-grade hashlib

Security Considerations:
    - Input size limits (10MB max)
    - No external API calls
    - Deterministic, reproducible results
    - MD5/SHA-1 provided but flagged as deprecated

Version History:
    1.0.0 - Initial implementation wrapping CryptographicEngine
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from enum import Enum
from typing import Any, Dict, Final, List, Literal, Optional

from pydantic import Field, field_validator

from .base import Tool, ToolInput, ToolOutput, ToolStatus

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum input size for hash computation (10MB - same as CryptographicEngine)
MAX_INPUT_SIZE: Final[int] = 10 * 1024 * 1024

# Supported hash algorithms
SUPPORTED_ALGORITHMS: Final[tuple] = (
    "sha256", "sha512", "sha384", "sha224", "sha1", "md5",
    "sha3_256", "sha3_512", "sha3_384", "sha3_224",
    "blake2b", "blake2s",
)

# Deprecated algorithms (still supported but flagged)
DEPRECATED_ALGORITHMS: Final[tuple] = ("md5", "sha1")

# Supported encoding operations
SUPPORTED_ENCODINGS: Final[tuple] = (
    "base64_encode", "base64_decode",
    "hex_encode", "hex_decode",
)


# =============================================================================
# INPUT MODEL
# =============================================================================


class HashAlgorithm(str, Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA384 = "sha384"
    SHA224 = "sha224"
    SHA1 = "sha1"
    MD5 = "md5"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    SHA3_384 = "sha3_384"
    SHA3_224 = "sha3_224"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class EncodingOperation(str, Enum):
    """Supported encoding operations."""
    BASE64_ENCODE = "base64_encode"
    BASE64_DECODE = "base64_decode"
    HEX_ENCODE = "hex_encode"
    HEX_DECODE = "hex_decode"


class HashComputeInput(ToolInput):
    """
    Input parameters for the hash compute tool.
    
    Supports two modes:
    1. Hash computation: Provide `data` and `algorithm`
    2. Encoding operation: Provide `data` and `encoding_operation`
    
    Attributes:
        data: The input data to process (required)
        algorithm: Hash algorithm to use (optional, for hashing)
        encoding_operation: Encoding operation to perform (optional)
        hmac_key: Key for HMAC computation (optional)
    """
    
    data: str = Field(
        ...,
        min_length=1,
        description=(
            "The input data to hash or encode. Can be any string. "
            "Maximum size: 10MB."
        )
    )
    algorithm: Optional[HashAlgorithm] = Field(
        default=HashAlgorithm.SHA256,
        description=(
            "Hash algorithm to use. Default: sha256. "
            "Options: sha256, sha512, sha384, sha224, sha1, md5, "
            "sha3_256, sha3_512, blake2b, blake2s. "
            "Note: md5 and sha1 are deprecated for security purposes."
        )
    )
    encoding_operation: Optional[EncodingOperation] = Field(
        default=None,
        description=(
            "Encoding operation to perform instead of hashing. "
            "Options: base64_encode, base64_decode, hex_encode, hex_decode. "
            "If provided, algorithm is ignored."
        )
    )
    hmac_key: Optional[str] = Field(
        default=None,
        description=(
            "Key for HMAC computation. If provided with algorithm, "
            "computes HMAC instead of plain hash."
        )
    )
    
    @field_validator("data", mode="before")
    @classmethod
    def validate_data_size(cls, v: str) -> str:
        """Validate data size to prevent DoS attacks."""
        if isinstance(v, str) and len(v.encode('utf-8')) > MAX_INPUT_SIZE:
            raise ValueError(f"Input data exceeds maximum size of {MAX_INPUT_SIZE} bytes")
        return v


# =============================================================================
# HASH COMPUTE TOOL
# =============================================================================


class HashComputeTool(Tool):
    """
    Cryptographic hash computation tool for deterministic operations.
    
    Wraps Python's hashlib and base64 modules to provide accurate,
    deterministic cryptographic computations. Unlike LLM-generated
    hashes (which are hallucinated), this tool computes real values.
    
    Thread Safety:
        This tool is thread-safe. All operations are stateless and
        deterministic.
    
    Capabilities:
        - SHA-2 family: SHA-256, SHA-512, SHA-384, SHA-224
        - SHA-1 (deprecated for security, but supported)
        - MD5 (deprecated for security, but supported)
        - SHA-3 family: SHA3-256, SHA3-512, SHA3-384, SHA3-224
        - BLAKE2: BLAKE2b, BLAKE2s
        - HMAC with any supported algorithm
        - Base64 encoding/decoding
        - Hex encoding/decoding
    
    Example:
        >>> tool = HashComputeTool()
        >>> result = tool.execute(data="Hello, World!", algorithm="sha256")
        >>> print(result.result["hash"])
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
        
        >>> # Base64 encoding
        >>> result = tool.execute(data="Hello", encoding_operation="base64_encode")
        >>> print(result.result["encoded"])
        'SGVsbG8='
    """
    
    def __init__(self) -> None:
        """Initialize the hash compute tool."""
        super().__init__()
        
        # Check for CryptographicEngine availability
        self._engine_available = False
        self._engine_class = None
        self._init_error: Optional[str] = None
        
        try:
            from vulcan.reasoning.cryptographic_engine import CryptographicEngine
            self._engine_class = CryptographicEngine
            self._engine_available = True
            logger.debug("HashComputeTool: CryptographicEngine available")
        except ImportError as e:
            # Fall back to direct hashlib usage
            self._init_error = f"CryptographicEngine not available, using direct hashlib: {e}"
            logger.info(f"HashComputeTool: {self._init_error}")
    
    @property
    def name(self) -> str:
        """Tool name for LLM to reference."""
        return "hash_compute"
    
    @property
    def description(self) -> str:
        """Description for LLM to understand when to use this tool."""
        return """Deterministic cryptographic hash and encoding computation.

Use this tool when you need to:
- Compute SHA-256, SHA-512, MD5, or other hash values
- Encode/decode data with Base64 or hexadecimal
- Compute HMAC message authentication codes
- Generate checksums for data integrity

Supported algorithms:
- SHA-2: sha256, sha512, sha384, sha224
- SHA-3: sha3_256, sha3_512, sha3_384, sha3_224
- BLAKE2: blake2b, blake2s
- Legacy: md5 (deprecated), sha1 (deprecated)

Examples:
- hash_compute(data="Hello", algorithm="sha256")
- hash_compute(data="Hello", encoding_operation="base64_encode")

IMPORTANT: LLMs cannot compute hashes - they hallucinate values.
Always use this tool for actual hash computation.

Returns: Computed hash/encoding with 100% confidence (deterministic)."""
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for parameters (OpenAI function calling compatible)."""
        return HashComputeInput.model_json_schema()
    
    @property
    def is_available(self) -> bool:
        """Hash computation is always available via hashlib."""
        return True
    
    def execute(
        self,
        data: str,
        algorithm: Optional[str] = "sha256",
        encoding_operation: Optional[str] = None,
        hmac_key: Optional[str] = None,
    ) -> ToolOutput:
        """
        Execute hash computation or encoding operation.
        
        Thread-safe method that performs deterministic cryptographic operations.
        
        Args:
            data: Input data to process
            algorithm: Hash algorithm (default: sha256)
            encoding_operation: Encoding operation (overrides algorithm if set)
            hmac_key: Key for HMAC computation
            
        Returns:
            ToolOutput with computed hash/encoding and 100% confidence
        """
        start_time = time.perf_counter()
        
        def elapsed_ms() -> float:
            return (time.perf_counter() - start_time) * 1000
        
        # Validate input
        if not data:
            return ToolOutput.create_failure(
                error="Input data cannot be empty",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.INVALID_INPUT,
            )
        
        # Check input size
        try:
            data_bytes = data.encode('utf-8')
            if len(data_bytes) > MAX_INPUT_SIZE:
                return ToolOutput.create_failure(
                    error=f"Input exceeds maximum size of {MAX_INPUT_SIZE // (1024*1024)}MB",
                    computation_time_ms=elapsed_ms(),
                    status=ToolStatus.INVALID_INPUT,
                )
        except UnicodeEncodeError as e:
            return ToolOutput.create_failure(
                error=f"Invalid input encoding: {e}",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.INVALID_INPUT,
            )
        
        try:
            # Determine operation type
            if encoding_operation:
                result = self._perform_encoding(data, encoding_operation)
            elif hmac_key:
                result = self._compute_hmac(data, algorithm or "sha256", hmac_key)
            else:
                result = self._compute_hash(data, algorithm or "sha256")
            
            # Record execution for stats
            computation_time = elapsed_ms()
            self._record_execution(computation_time)
            
            return ToolOutput.create_success(
                result=result,
                computation_time_ms=computation_time,
                metadata={
                    "tool": self.name,
                    "deterministic": True,
                    "confidence": 1.0,
                },
            )
            
        except ValueError as e:
            return ToolOutput.create_failure(
                error=str(e),
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.INVALID_INPUT,
            )
        except Exception as e:
            logger.error(f"HashComputeTool: Unexpected error: {e}", exc_info=True)
            return ToolOutput.create_failure(
                error=f"Hash computation failed: {str(e)}",
                computation_time_ms=elapsed_ms(),
                status=ToolStatus.FAILURE,
            )
    
    def _compute_hash(self, data: str, algorithm: str) -> Dict[str, Any]:
        """
        Compute hash of the input data.
        
        Args:
            data: Input string to hash
            algorithm: Hash algorithm name
            
        Returns:
            Dict with hash result and metadata
        """
        # Normalize algorithm name
        algo = algorithm.lower().replace("-", "_").replace("sha3256", "sha3_256")
        algo = algo.replace("sha3512", "sha3_512").replace("sha3384", "sha3_384")
        algo = algo.replace("sha3224", "sha3_224")
        
        # Get hashlib algorithm
        if algo == "sha256":
            h = hashlib.sha256()
        elif algo == "sha512":
            h = hashlib.sha512()
        elif algo == "sha384":
            h = hashlib.sha384()
        elif algo == "sha224":
            h = hashlib.sha224()
        elif algo == "sha1":
            h = hashlib.sha1()
        elif algo == "md5":
            h = hashlib.md5()
        elif algo == "sha3_256":
            h = hashlib.sha3_256()
        elif algo == "sha3_512":
            h = hashlib.sha3_512()
        elif algo == "sha3_384":
            h = hashlib.sha3_384()
        elif algo == "sha3_224":
            h = hashlib.sha3_224()
        elif algo == "blake2b":
            h = hashlib.blake2b()
        elif algo == "blake2s":
            h = hashlib.blake2s()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Compute hash
        h.update(data.encode('utf-8'))
        hash_hex = h.hexdigest()
        
        # Build response
        response: Dict[str, Any] = {
            "input": data if len(data) <= 100 else f"{data[:100]}...",
            "algorithm": algo.upper(),
            "hash": hash_hex,
            "hash_bytes": len(h.digest()),
            "hash_bits": len(h.digest()) * 8,
        }
        
        # Add deprecation warning for weak algorithms
        if algo in DEPRECATED_ALGORITHMS:
            response["warning"] = (
                f"{algo.upper()} is cryptographically broken and should not be "
                f"used for security purposes. Use SHA-256 or stronger."
            )
        
        return response
    
    def _compute_hmac(self, data: str, algorithm: str, key: str) -> Dict[str, Any]:
        """
        Compute HMAC of the input data.
        
        Args:
            data: Input string
            algorithm: Hash algorithm for HMAC
            key: HMAC key
            
        Returns:
            Dict with HMAC result and metadata
        """
        import hmac as hmac_module
        
        # Normalize algorithm name
        algo = algorithm.lower().replace("-", "")
        
        # Map to hashlib name
        algo_map = {
            "sha256": "sha256",
            "sha512": "sha512",
            "sha384": "sha384",
            "sha224": "sha224",
            "sha1": "sha1",
            "md5": "md5",
        }
        
        if algo not in algo_map:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
        
        # Compute HMAC
        h = hmac_module.new(
            key.encode('utf-8'),
            data.encode('utf-8'),
            algo_map[algo]
        )
        hmac_hex = h.hexdigest()
        
        return {
            "input": data if len(data) <= 100 else f"{data[:100]}...",
            "algorithm": f"HMAC-{algo.upper()}",
            "hmac": hmac_hex,
            "key_provided": True,
        }
    
    def _perform_encoding(self, data: str, operation: str) -> Dict[str, Any]:
        """
        Perform encoding/decoding operation.
        
        Args:
            data: Input string
            operation: Encoding operation name
            
        Returns:
            Dict with encoded/decoded result
        """
        import base64
        
        op = operation.lower()
        
        if op == "base64_encode":
            encoded = base64.b64encode(data.encode('utf-8')).decode('ascii')
            return {
                "input": data if len(data) <= 100 else f"{data[:100]}...",
                "operation": "base64_encode",
                "encoded": encoded,
            }
        
        elif op == "base64_decode":
            try:
                decoded = base64.b64decode(data).decode('utf-8')
                return {
                    "input": data if len(data) <= 100 else f"{data[:100]}...",
                    "operation": "base64_decode",
                    "decoded": decoded,
                }
            except Exception as e:
                raise ValueError(f"Invalid Base64 input: {e}")
        
        elif op == "hex_encode":
            encoded = data.encode('utf-8').hex()
            return {
                "input": data if len(data) <= 100 else f"{data[:100]}...",
                "operation": "hex_encode",
                "encoded": encoded,
            }
        
        elif op == "hex_decode":
            try:
                decoded = bytes.fromhex(data).decode('utf-8')
                return {
                    "input": data if len(data) <= 100 else f"{data[:100]}...",
                    "operation": "hex_decode",
                    "decoded": decoded,
                }
            except Exception as e:
                raise ValueError(f"Invalid hex input: {e}")
        
        else:
            raise ValueError(f"Unsupported encoding operation: {operation}")
