"""
Cryptographic Engine for deterministic hash and encoding computations.

Note: The system was falling back to OpenAI for cryptographic
computations (SHA-256, MD5, etc.), which resulted in hallucinated 
(incorrect) hash values. This engine provides accurate, deterministic
cryptographic operations.

Features:
    - SHA-256, SHA-1, SHA-512, MD5 hash computation
    - Base64 encoding/decoding
    - Hex encoding/decoding
    - HMAC computation
    - CRC32 checksum computation

Example:
    >>> engine = CryptographicEngine()
    >>> result = engine.compute("Calculate SHA-256 of 'Hello, World!'")
    >>> print(result['result'])
    'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'

Industry Standards:
    - Uses Python's hashlib for cryptographic functions
    - Deterministic results guaranteed
    - No external API calls (prevents hallucination)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import threading
import urllib.parse
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Final, FrozenSet, Optional, Pattern, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Maximum input size to prevent DoS attacks (10MB)
MAX_INPUT_SIZE: Final[int] = 10 * 1024 * 1024

# Supported hash algorithms
SUPPORTED_HASH_ALGORITHMS: Final[FrozenSet[str]] = frozenset({
    'sha256', 'sha1', 'sha512', 'md5', 'sha384', 'sha224'
})

# Keywords that indicate cryptographic operations
CRYPTO_KEYWORDS: Final[FrozenSet[str]] = frozenset({
    # SHA-2 family
    'sha-256', 'sha256', 'sha-1', 'sha1', 'sha-512', 'sha512',
    'sha-384', 'sha384', 'sha-224', 'sha224',
    # SHA-3 family
    'sha3-256', 'sha3256', 'sha3-512', 'sha3512',
    'sha3-384', 'sha3384', 'sha3-224', 'sha3224',
    # BLAKE2 family
    'blake2b', 'blake2s', 'blake2',
    # Other hashes
    'md5', 'ripemd160', 'ripemd-160',
    # Generic terms
    'hash', 'checksum', 'digest',
    # Encoding
    'base64', 'b64', 'hex', 'hexadecimal',
    'url encode', 'url decode', 'urlencode', 'urldecode',
    'percent encode', 'percent decode',
    # HMAC and checksums
    'hmac', 'crc32', 'crc-32', 'encode', 'decode'
})

# Keywords that indicate computation requests
COMPUTE_KEYWORDS: Final[FrozenSet[str]] = frozenset({
    'calculate', 'compute', 'generate', 'find', 'get', 
    'what is', 'determine', 'produce', 'create'
})

# Keywords that indicate theoretical/educational questions about crypto
# NOT actual requests to compute hashes
# Note: Prevents "I'm a researcher testing AI capabilities" from being hashed
# ROOT CAUSE FIX: Narrowed list to avoid blocking legitimate computation queries
THEORETICAL_CRYPTO_KEYWORDS: Final[FrozenSet[str]] = frozenset({
    # Security concepts (only block when combined with questions)
    'collision resistance', 'collision attack',
    'preimage attack', 'preimage resistance',
    'second preimage', 'birthday attack', 'birthday paradox',
    'vulnerable', 'weakness', 'broken',
    # Educational questions (these indicate NOT computation)
    'why is', 'why does', 'why do', 'how does', 'how do',
    'explain', 'describe', 'definition', 'what does',
    # Theoretical topics
    'proof', 'prove', 'theorem', 'reduction',
    'cryptograph', 'claims', 'demonstrates',
    # AI capability testing (from bug report)
    'ai capabilities', 'ai system', 'testing ai',
})

# ROOT CAUSE FIX: Strong indicators that query is actually asking for computation
# Even if theoretical keywords are present, these override and indicate computation
COMPUTATION_OVERRIDE_KEYWORDS: Final[FrozenSet[str]] = frozenset({
    'hash of', 'sha256 of', 'sha-256 of', 'md5 of', 'sha1 of', 'sha-1 of',
    'sha512 of', 'sha-512 of', 'crc32 of', 'crc-32 of',
    'base64 encode', 'base64 decode', 'hex encode', 'hex decode',
    'compute the hash', 'calculate the hash', 'generate the hash',
    'give me the hash', 'return the hash', 'output the hash',
    'hash this', 'hash the', 'hash value',
})

# CODE REVIEW FIX: Pre-compiled regex pattern for quoted data detection
# Match quoted strings that are likely data (not contractions like don't, can't)
# Single-quoted with 2+ chars (excludes 't, 's, etc.) OR double-quoted with 2+ chars
QUOTED_DATA_PATTERN: Pattern = re.compile(
    r"'[^']{2,}'|"  # Single-quoted with 2+ chars
    r'"[^"]{2,}"'   # Double-quoted with 2+ chars
)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class CryptoOperation(Enum):
    """
    Supported cryptographic operations.
    
    Each operation represents a specific cryptographic function that
    the engine can perform deterministically.
    
    Attributes:
        # SHA-2 Family (FIPS 180-4)
        SHA256: SHA-256 hash (256-bit)
        SHA1: SHA-1 hash (160-bit, deprecated for security)
        SHA512: SHA-512 hash (512-bit)
        SHA384: SHA-384 hash (384-bit)
        SHA224: SHA-224 hash (224-bit)
        
        # SHA-3 Family (FIPS 202)
        SHA3_256: SHA3-256 hash (256-bit)
        SHA3_512: SHA3-512 hash (512-bit)
        SHA3_384: SHA3-384 hash (384-bit)
        SHA3_224: SHA3-224 hash (224-bit)
        
        # BLAKE2 Family (RFC 7693)
        BLAKE2B: BLAKE2b hash (512-bit, fast)
        BLAKE2S: BLAKE2s hash (256-bit, optimized for 32-bit)
        
        # Legacy Hashes
        MD5: MD5 hash (128-bit, deprecated for security)
        RIPEMD160: RIPEMD-160 hash (160-bit)
        
        # Encoding
        BASE64_ENCODE: Base64 encoding (RFC 4648)
        BASE64_DECODE: Base64 decoding (RFC 4648)
        HEX_ENCODE: Hexadecimal encoding
        HEX_DECODE: Hexadecimal decoding
        URL_ENCODE: URL/Percent encoding (RFC 3986)
        URL_DECODE: URL/Percent decoding (RFC 3986)
        
        # HMAC
        HMAC_SHA256: HMAC with SHA-256 (RFC 2104)
        HMAC_SHA512: HMAC with SHA-512 (RFC 2104)
        
        # Checksums
        CRC32: CRC-32 checksum (ISO 3309)
        
        UNKNOWN: Unrecognized operation
    """
    # SHA-2 Family
    SHA256 = "sha256"
    SHA1 = "sha1"
    SHA512 = "sha512"
    SHA384 = "sha384"
    SHA224 = "sha224"
    
    # SHA-3 Family
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    SHA3_384 = "sha3_384"
    SHA3_224 = "sha3_224"
    
    # BLAKE2 Family
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    
    # Legacy
    MD5 = "md5"
    RIPEMD160 = "ripemd160"
    
    # Encoding
    BASE64_ENCODE = "base64_encode"
    BASE64_DECODE = "base64_decode"
    HEX_ENCODE = "hex_encode"
    HEX_DECODE = "hex_decode"
    URL_ENCODE = "url_encode"
    URL_DECODE = "url_decode"
    
    # HMAC
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"
    
    # Checksums
    CRC32 = "crc32"
    
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CryptoResult:
    """
    Immutable result of a cryptographic operation.
    
    This dataclass provides a structured, type-safe way to return
    results from cryptographic operations.
    
    Attributes:
        success: Whether the operation completed successfully
        operation: The cryptographic operation that was performed
        input_value: The original input value
        result: The computed result (hash, encoded string, etc.)
        error: Error message if the operation failed
        
    Example:
        >>> result = CryptoResult(
        ...     success=True,
        ...     operation=CryptoOperation.SHA256,
        ...     input_value="Hello",
        ...     result="185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
        ... )
    """
    success: bool
    operation: CryptoOperation
    input_value: str
    result: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Main Engine Class
# =============================================================================

class CryptographicEngine:
    """
    Engine for deterministic cryptographic computations.
    
    Note: This engine provides accurate hash computations instead
    of relying on LLM fallback which hallucinates incorrect values.
    
    Security Considerations:
        - All operations are deterministic and reproducible
        - No external API calls (prevents hallucination)
        - Input size limited to MAX_INPUT_SIZE to prevent DoS
        - Uses Python's cryptography-grade hashlib implementation
        - MD5 and SHA-1 are provided but deprecated for security purposes
    
    Thread Safety:
        This class is thread-safe. All operations are stateless and
        pattern compilation is done once during initialization.
    
    Architecture:
        The engine uses a pipeline approach:
        1. Query detection (is this a crypto query?)
        2. Operation detection (which crypto operation?)
        3. Input extraction (what data to process?)
        4. Computation (deterministic cryptographic operation)
        5. Result formatting (structured response)
    
    Supported Operations:
        - Hash functions: SHA-256, SHA-1, SHA-512, SHA-384, SHA-224, MD5
        - Encoding: Base64, Hexadecimal
        - Authentication: HMAC-SHA256, HMAC-SHA512
        - Checksums: CRC32
    
    Example:
        >>> engine = CryptographicEngine()
        >>> result = engine.compute("Calculate SHA-256 of 'Hello, World!'")
        >>> assert result['success'] == True
        >>> assert result['result'] == 'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
    
    Attributes:
        _patterns: Compiled regex patterns for operation detection (immutable after init)
    """
    
    __slots__ = ('_patterns',)  # Memory optimization
    
    def __init__(self) -> None:
        """
        Initialize the cryptographic engine.
        
        Compiles all regex patterns once for efficient repeated use.
        """
        self._patterns: Dict[CryptoOperation, Pattern[str]] = self._compile_patterns()
        logger.debug("[CryptoEngine] Note: Cryptographic engine initialized")
    
    def _compile_patterns(self) -> Dict[CryptoOperation, Pattern[str]]:
        """
        Compile regex patterns for detecting crypto operations.
        
        Returns:
            Dictionary mapping operations to compiled regex patterns
        """
        return {
            # SHA-2 Family
            CryptoOperation.SHA256: re.compile(
                r'(?:sha-?256|sha256)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA1: re.compile(
                r'(?:sha-?1|sha1)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA512: re.compile(
                r'(?:sha-?512|sha512)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA384: re.compile(
                r'(?:sha-?384|sha384)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA224: re.compile(
                r'(?:sha-?224|sha224)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # SHA-3 Family
            CryptoOperation.SHA3_256: re.compile(
                r'(?:sha3-?256|sha3256)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA3_512: re.compile(
                r'(?:sha3-?512|sha3512)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA3_384: re.compile(
                r'(?:sha3-?384|sha3384)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.SHA3_224: re.compile(
                r'(?:sha3-?224|sha3224)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # BLAKE2 Family
            CryptoOperation.BLAKE2B: re.compile(
                r'(?:blake2b|blake2-b)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.BLAKE2S: re.compile(
                r'(?:blake2s|blake2-s)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # Legacy Hashes
            CryptoOperation.MD5: re.compile(
                r'(?:md5)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.RIPEMD160: re.compile(
                r'(?:ripemd-?160|ripemd160)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # Encoding - Base64
            CryptoOperation.BASE64_ENCODE: re.compile(
                r'(?:base64|b64)\s+(?:encode|encoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.BASE64_DECODE: re.compile(
                r'(?:base64|b64)\s+(?:decode|decoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # Encoding - Hex
            CryptoOperation.HEX_ENCODE: re.compile(
                r'(?:hex|hexadecimal)\s+(?:encode|encoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.HEX_DECODE: re.compile(
                r'(?:hex|hexadecimal)\s+(?:decode|decoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # Encoding - URL/Percent
            CryptoOperation.URL_ENCODE: re.compile(
                r'(?:url|percent)\s+(?:encode|encoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.URL_DECODE: re.compile(
                r'(?:url|percent)\s+(?:decode|decoding)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # HMAC
            CryptoOperation.HMAC_SHA256: re.compile(
                r'hmac[-_]?sha-?256\s+(?:of\s+)?["\'](.+?)["\']\s+(?:with\s+)?(?:key\s+)?["\'](.+?)["\']',
                re.I
            ),
            CryptoOperation.HMAC_SHA512: re.compile(
                r'hmac[-_]?sha-?512\s+(?:of\s+)?["\'](.+?)["\']\s+(?:with\s+)?(?:key\s+)?["\'](.+?)["\']',
                re.I
            ),
            
            # Checksums
            CryptoOperation.CRC32: re.compile(
                r'(?:crc32|crc-32)\s+(?:of\s+)?["\'](.+?)["\']',
                re.I
            ),
        }
    
    def is_crypto_query(self, query: str) -> bool:
        """
        Check if a query is about cryptographic computation.
        
        Note: Used to route queries to this engine instead of LLM.
        
        ROOT CAUSE FIX: Previous implementation was too restrictive - required
        quoted data which missed valid computation requests like:
        - "What is the SHA-256 hash of Hello World?"
        - "Compute MD5 for test"
        
        Now uses a more flexible approach:
        1. Check for computation override keywords (highest priority)
        2. Check for crypto keywords
        3. Check for compute patterns
        4. Check for theoretical questions (only if not overridden)
        5. Accept queries with quoted data OR with "of X" pattern
        
        Args:
            query: The input query string
            
        Returns:
            True if query involves cryptographic computation with actual data
        """
        if not isinstance(query, str):
            return False
        
        query_lower = query.lower()
        
        # Step 1: Check for crypto keywords
        has_crypto_keyword = any(kw in query_lower for kw in CRYPTO_KEYWORDS)
        if not has_crypto_keyword:
            return False
        
        # Step 2: Check for compute patterns
        has_compute_pattern = any(pat in query_lower for pat in COMPUTE_KEYWORDS)
        if not has_compute_pattern:
            return False
        
        # ROOT CAUSE FIX: Check for computation override keywords
        # These indicate a definite computation request regardless of other filters
        has_computation_override = any(kw in query_lower for kw in COMPUTATION_OVERRIDE_KEYWORDS)
        
        if has_computation_override:
            logger.debug(
                f"[CryptoEngine] Computation override detected, accepting query: "
                f"'{query[:50]}...'"
            )
            return True
        
        # Step 3: Check for theoretical/educational questions
        # These should NOT trigger crypto computation (unless overridden above)
        has_theoretical = any(kw in query_lower for kw in THEORETICAL_CRYPTO_KEYWORDS)
        if has_theoretical:
            logger.debug(
                f"[CryptoEngine] Query detected as theoretical/educational, not computation: "
                f"'{query[:50]}...'"
            )
            return False
        
        # Step 4: Check for data indicators
        # Accept if we have quoted data OR an "of X" pattern indicating input
        has_quoted_data = bool(QUOTED_DATA_PATTERN.search(query))
        
        # ROOT CAUSE FIX: Also accept "of X" pattern without quotes
        # Examples: "SHA-256 of Hello World", "hash of test string"
        has_of_pattern = bool(re.search(
            r'(?:sha-?256|sha-?1|sha-?512|md5|hash|base64|hex|crc32)\s+(?:of|for)\s+\w',
            query_lower
        ))
        
        if has_quoted_data or has_of_pattern:
            return True
        
        logger.debug(
            f"[CryptoEngine] Query has no data indicator, not a computation request: "
            f"'{query[:50]}...'"
        )
        return False
    
    def _validate_input(self, input_value: str) -> Optional[str]:
        """
        Validate input data before processing.
        
        Security measure to prevent DoS attacks and invalid input.
        
        Args:
            input_value: The input value to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        if not input_value:
            return "Input value cannot be empty"
        
        if len(input_value) > MAX_INPUT_SIZE:
            return f"Input exceeds maximum size ({MAX_INPUT_SIZE} bytes)"
        
        return None
    
    def compute(self, query: str) -> Dict[str, Any]:
        """
        Compute cryptographic operation based on query.
        
        Note: Main entry point for deterministic crypto computation.
        
        This method parses the natural language query, detects the intended
        cryptographic operation, extracts the input data, and performs the
        computation deterministically.
        
        Args:
            query: Natural language query describing the operation
            
        Returns:
            Dict with operation result containing:
            - success (bool): Whether the operation succeeded
            - operation (str): The operation performed
            - input (str): The input value
            - result (str | None): The computed result
            - error (str | None): Error message if failed
            - confidence (float): Always 1.0 for deterministic operations
            - metadata (dict): Additional information
        
        Raises:
            No exceptions raised; all errors returned in result dict
        
        Example:
            >>> engine = CryptographicEngine()
            >>> result = engine.compute("Calculate SHA-256 of 'Hello'")
            >>> result['success']
            True
            >>> result['result']
            '185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969'
        """
        # Input validation
        if not query:
            return self._error_result("Empty query", CryptoOperation.UNKNOWN, "")
        
        if not isinstance(query, str):
            return self._error_result(
                f"Query must be string, got {type(query).__name__}",
                CryptoOperation.UNKNOWN,
                str(query)
            )
        
        # Detect operation and extract input
        operation, input_value, extra = self._detect_operation(query)
        
        if operation == CryptoOperation.UNKNOWN:
            return self._error_result(
                "Could not detect cryptographic operation from query",
                CryptoOperation.UNKNOWN,
                query
            )
        
        # Validate input
        validation_error = self._validate_input(input_value)
        if validation_error:
            return self._error_result(validation_error, operation, input_value)
        
        # Perform the operation
        try:
            result = self._perform_operation(operation, input_value, extra)
            
            logger.info(
                f"[CryptoEngine] Note: Computed {operation.value} of "
                f"'{input_value[:30]}...' -> '{result[:30]}...'"
            )
            
            return {
                'success': True,
                'operation': operation.value,
                'input': input_value,
                'result': result,
                'error': None,
                'confidence': 1.0,  # Deterministic computation = 100% confidence
                'metadata': {
                    'engine': 'CryptographicEngine',
                    'bug_fix': 'BUG#14',
                    'deterministic': True,
                }
            }
            
        except Exception as e:
            logger.error(f"[CryptoEngine] Operation failed: {e}")
            return self._error_result(str(e), operation, input_value)
    
    def _detect_operation(self, query: str) -> Tuple[CryptoOperation, str, Optional[str]]:
        """
        Detect the operation type and extract input from query.
        
        ROOT CAUSE FIX: Now handles queries without quotes using "of X" pattern.
        
        Args:
            query: The input query
            
        Returns:
            Tuple of (operation, input_value, extra_param)
        """
        for operation, pattern in self._patterns.items():
            match = pattern.search(query)
            if match:
                groups = match.groups()
                input_value = groups[0]
                extra = groups[1] if len(groups) > 1 else None
                return operation, input_value, extra
        
        # Fallback 1: Try to extract quoted string for hash operations
        query_lower = query.lower()
        quoted_match = re.search(r'["\'](.+?)["\']', query)
        
        if quoted_match:
            input_value = quoted_match.group(1)
            return self._detect_algorithm_from_keywords(query_lower, input_value)
        
        # ROOT CAUSE FIX: Fallback 2 - Extract unquoted data using "of X" pattern
        # Matches patterns like "SHA-256 of Hello World", "hash of test"
        of_pattern_match = re.search(
            r'(?:sha-?256|sha-?1|sha-?512|sha-?384|sha-?224|md5|hash|base64|hex|crc-?32|'
            r'sha3-?256|sha3-?512|sha3-?384|sha3-?224|blake2b|blake2s|ripemd-?160)'
            r'\s+(?:of|for)\s+(.+?)(?:\?|$|\.)',
            query_lower,
            re.I
        )
        
        if of_pattern_match:
            input_value = of_pattern_match.group(1).strip()
            # Remove trailing punctuation or common phrases
            input_value = re.sub(r'\s*(?:please|thanks|thank you).*$', '', input_value, flags=re.I)
            return self._detect_algorithm_from_keywords(query_lower, input_value)
        
        return CryptoOperation.UNKNOWN, "", None
    
    def _detect_algorithm_from_keywords(
        self, 
        query_lower: str, 
        input_value: str
    ) -> Tuple[CryptoOperation, str, Optional[str]]:
        """
        ROOT CAUSE FIX: Helper to detect algorithm from keywords in query.
        
        Extracted to avoid code duplication between quoted and unquoted fallbacks.
        
        Args:
            query_lower: Lowercased query string
            input_value: The extracted input data
            
        Returns:
            Tuple of (operation, input_value, extra_param)
        """
        # SHA-2 Family (check specific lengths before generic)
        if 'sha-256' in query_lower or 'sha256' in query_lower:
            return CryptoOperation.SHA256, input_value, None
        elif 'sha-512' in query_lower or 'sha512' in query_lower:
            return CryptoOperation.SHA512, input_value, None
        elif 'sha-384' in query_lower or 'sha384' in query_lower:
            return CryptoOperation.SHA384, input_value, None
        elif 'sha-224' in query_lower or 'sha224' in query_lower:
            return CryptoOperation.SHA224, input_value, None
        elif 'sha-1' in query_lower or 'sha1' in query_lower:
            return CryptoOperation.SHA1, input_value, None
        
        # SHA-3 Family
        elif 'sha3-256' in query_lower or 'sha3256' in query_lower:
            return CryptoOperation.SHA3_256, input_value, None
        elif 'sha3-512' in query_lower or 'sha3512' in query_lower:
            return CryptoOperation.SHA3_512, input_value, None
        elif 'sha3-384' in query_lower or 'sha3384' in query_lower:
            return CryptoOperation.SHA3_384, input_value, None
        elif 'sha3-224' in query_lower or 'sha3224' in query_lower:
            return CryptoOperation.SHA3_224, input_value, None
        
        # BLAKE2 Family
        elif 'blake2b' in query_lower:
            return CryptoOperation.BLAKE2B, input_value, None
        elif 'blake2s' in query_lower:
            return CryptoOperation.BLAKE2S, input_value, None
        
        # Legacy
        elif 'md5' in query_lower:
            return CryptoOperation.MD5, input_value, None
        elif 'ripemd160' in query_lower or 'ripemd-160' in query_lower:
            return CryptoOperation.RIPEMD160, input_value, None
        
        # Checksums
        elif 'crc32' in query_lower or 'crc-32' in query_lower:
            return CryptoOperation.CRC32, input_value, None
        
        # Encoding - Base64
        elif 'base64' in query_lower or 'b64' in query_lower:
            if 'decode' in query_lower:
                return CryptoOperation.BASE64_DECODE, input_value, None
            return CryptoOperation.BASE64_ENCODE, input_value, None
        
        # Encoding - Hex
        elif 'hex' in query_lower:
            if 'decode' in query_lower:
                return CryptoOperation.HEX_DECODE, input_value, None
            return CryptoOperation.HEX_ENCODE, input_value, None
        
        # Encoding - URL
        elif 'url' in query_lower or 'percent' in query_lower:
            if 'decode' in query_lower:
                return CryptoOperation.URL_DECODE, input_value, None
            return CryptoOperation.URL_ENCODE, input_value, None
        
        # Generic "hash" keyword - default to SHA-256
        elif 'hash' in query_lower:
            logger.debug(
                f"[CryptoEngine] Generic 'hash' detected, defaulting to SHA-256"
            )
            return CryptoOperation.SHA256, input_value, None
        
        return CryptoOperation.UNKNOWN, input_value, None
    
    def _perform_operation(
        self, 
        operation: CryptoOperation, 
        input_value: str, 
        extra: Optional[str] = None
    ) -> str:
        """
        Perform the specified cryptographic operation.
        
        This method contains the core cryptographic implementations using
        Python's standard library hashlib, base64, and zlib modules.
        
        Args:
            operation: The operation to perform
            input_value: The input value (will be UTF-8 encoded)
            extra: Optional extra parameter (e.g., key for HMAC)
            
        Returns:
            The computed result as a hexadecimal or encoded string
            
        Raises:
            ValueError: If operation is not supported
        """
        input_bytes = input_value.encode('utf-8')
        
        # =====================================================================
        # SHA-2 Family Hash Operations
        # =====================================================================
        if operation == CryptoOperation.SHA256:
            return hashlib.sha256(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA1:
            return hashlib.sha1(input_bytes, usedforsecurity=False).hexdigest()
        
        elif operation == CryptoOperation.SHA512:
            return hashlib.sha512(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA384:
            return hashlib.sha384(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA224:
            return hashlib.sha224(input_bytes).hexdigest()
        
        # =====================================================================
        # SHA-3 Family Hash Operations (FIPS 202)
        # =====================================================================
        elif operation == CryptoOperation.SHA3_256:
            return hashlib.sha3_256(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA3_512:
            return hashlib.sha3_512(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA3_384:
            return hashlib.sha3_384(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.SHA3_224:
            return hashlib.sha3_224(input_bytes).hexdigest()
        
        # =====================================================================
        # BLAKE2 Family Hash Operations (RFC 7693)
        # =====================================================================
        elif operation == CryptoOperation.BLAKE2B:
            return hashlib.blake2b(input_bytes).hexdigest()
        
        elif operation == CryptoOperation.BLAKE2S:
            return hashlib.blake2s(input_bytes).hexdigest()
        
        # =====================================================================
        # Legacy Hash Operations
        # =====================================================================
        elif operation == CryptoOperation.MD5:
            return hashlib.md5(input_bytes, usedforsecurity=False).hexdigest()
        
        elif operation == CryptoOperation.RIPEMD160:
            # RIPEMD-160 may not be available in all Python installations
            try:
                return hashlib.new('ripemd160', input_bytes).hexdigest()
            except ValueError:
                raise ValueError(
                    "RIPEMD-160 is not available in this Python installation. "
                    "It may require OpenSSL support."
                )
        
        # =====================================================================
        # Encoding Operations
        # =====================================================================
        elif operation == CryptoOperation.BASE64_ENCODE:
            return base64.b64encode(input_bytes).decode('utf-8')
        
        elif operation == CryptoOperation.BASE64_DECODE:
            try:
                return base64.b64decode(input_bytes).decode('utf-8')
            except UnicodeDecodeError:
                # Return raw bytes as hex if not valid UTF-8
                return base64.b64decode(input_bytes).hex()
            except Exception as e:
                raise ValueError(f"Invalid base64 input: {e}")
        
        elif operation == CryptoOperation.HEX_ENCODE:
            return input_bytes.hex()
        
        elif operation == CryptoOperation.HEX_DECODE:
            try:
                return bytes.fromhex(input_value).decode('utf-8')
            except UnicodeDecodeError:
                # Return raw bytes if not valid UTF-8
                return bytes.fromhex(input_value).hex()
            except Exception as e:
                raise ValueError(f"Invalid hex input: {e}")
        
        elif operation == CryptoOperation.URL_ENCODE:
            return urllib.parse.quote(input_value, safe='')
        
        elif operation == CryptoOperation.URL_DECODE:
            return urllib.parse.unquote(input_value)
        
        # =====================================================================
        # HMAC Operations
        # =====================================================================
        elif operation == CryptoOperation.HMAC_SHA256:
            key_bytes = (extra or "").encode('utf-8')
            return hmac.new(key_bytes, input_bytes, hashlib.sha256).hexdigest()
        
        elif operation == CryptoOperation.HMAC_SHA512:
            key_bytes = (extra or "").encode('utf-8')
            return hmac.new(key_bytes, input_bytes, hashlib.sha512).hexdigest()
        
        # =====================================================================
        # Checksum Operations
        # =====================================================================
        elif operation == CryptoOperation.CRC32:
            return format(zlib.crc32(input_bytes) & 0xffffffff, '08x')
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _error_result(
        self, 
        error: str, 
        operation: CryptoOperation, 
        input_value: str
    ) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {
            'success': False,
            'operation': operation.value,
            'input': input_value,
            'result': None,
            'error': error,
            'confidence': 0.0,
            'metadata': {
                'engine': 'CryptographicEngine',
                'bug_fix': 'BUG#14',
            }
        }
    
    # =========================================================================
    # Convenience Methods for Direct Computation
    # =========================================================================
    
    # SHA-2 Family
    def sha256(self, data: Union[str, bytes]) -> str:
        """Compute SHA-256 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def sha1(self, data: Union[str, bytes]) -> str:
        """Compute SHA-1 hash (CRYPTOGRAPHICALLY BROKEN - DO NOT USE FOR SECURITY).
        
        ⚠️  SECURITY WARNING: SHA-1 is cryptographically broken (NIST deprecated since 2011)
        
        SHA-1 MUST NOT be used for:
        - Password hashing (use Argon2id, bcrypt, or scrypt)
        - Digital signatures (use RSA-PSS with SHA-256+, or Ed25519)
        - Security certificates (use SHA-256 or SHA-384)
        - HMAC authentication (use HMAC-SHA256 or HMAC-SHA512)
        - Any security-critical operations
        
        SHA-1 may ONLY be used for:
        - Non-cryptographic checksums (consider CRC32 for better performance)
        - Legacy system compatibility (e.g., Git commit hashing)
        - Cache keys for non-sensitive data
        
        Known attacks:
        - Collision attacks demonstrated in 2017 (SHAttered attack)
        - Practical collision attacks possible with moderate resources
        
        Recommended alternatives:
        - General purpose: SHA-256, SHA3-256, or BLAKE2b
        - Password hashing: Argon2id (OWASP recommended)
        - Fast checksums: CRC32, xxHash
        
        References:
        - NIST SP 800-131A: Deprecated SHA-1 for digital signatures (2011)
        - RFC 6194: Security Considerations for SHA-0 and SHA-1
        - SHAttered attack: https://shattered.io/
        - OWASP: https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
        
        Args:
            data: Input data (string or bytes). Strings are encoded as UTF-8.
        
        Returns:
            Hexadecimal digest string (40 characters)
        """
        import warnings
        warnings.warn(
            "SHA-1 is cryptographically broken (collision attacks demonstrated 2017). "
            "Use SHA-256, SHA-3, or BLAKE2 for security purposes. "
            "Use CRC32 or xxHash for fast non-cryptographic checksums.",
            DeprecationWarning,
            stacklevel=2
        )
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha1(data, usedforsecurity=False).hexdigest()
    
    def sha512(self, data: Union[str, bytes]) -> str:
        """Compute SHA-512 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha512(data).hexdigest()
    
    def sha384(self, data: Union[str, bytes]) -> str:
        """Compute SHA-384 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha384(data).hexdigest()
    
    def sha224(self, data: Union[str, bytes]) -> str:
        """Compute SHA-224 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha224(data).hexdigest()
    
    # SHA-3 Family
    def sha3_256(self, data: Union[str, bytes]) -> str:
        """Compute SHA3-256 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha3_256(data).hexdigest()
    
    def sha3_512(self, data: Union[str, bytes]) -> str:
        """Compute SHA3-512 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha3_512(data).hexdigest()
    
    def sha3_384(self, data: Union[str, bytes]) -> str:
        """Compute SHA3-384 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha3_384(data).hexdigest()
    
    def sha3_224(self, data: Union[str, bytes]) -> str:
        """Compute SHA3-224 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha3_224(data).hexdigest()
    
    # BLAKE2 Family
    def blake2b(self, data: Union[str, bytes]) -> str:
        """Compute BLAKE2b hash (512-bit)."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.blake2b(data).hexdigest()
    
    def blake2s(self, data: Union[str, bytes]) -> str:
        """Compute BLAKE2s hash (256-bit)."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.blake2s(data).hexdigest()
    
    # Legacy Hashes
    def md5(self, data: Union[str, bytes]) -> str:
        """Compute MD5 hash (CRYPTOGRAPHICALLY BROKEN - DO NOT USE FOR SECURITY).
        
        ⚠️  SECURITY WARNING: MD5 is cryptographically broken (NIST deprecated since 2011)
        
        MD5 MUST NOT be used for:
        - Password hashing (use Argon2id, bcrypt, or scrypt)
        - Digital signatures (use RSA-PSS with SHA-256+, or Ed25519)
        - Security certificates (use SHA-256 or SHA-384)
        - HMAC authentication (use HMAC-SHA256 or HMAC-SHA512)
        - Any security-critical operations
        
        MD5 may ONLY be used for:
        - Non-cryptographic checksums (consider CRC32 for better performance)
        - Legacy system compatibility (where security is not a concern)
        - Cache keys for non-sensitive data
        
        Recommended alternatives:
        - General purpose: SHA-256, SHA3-256, or BLAKE2b
        - Password hashing: Argon2id (OWASP recommended)
        - Fast checksums: CRC32, xxHash
        
        References:
        - NIST SP 800-131A: Deprecated MD5 for all security applications
        - RFC 6151: Updated Security Considerations for MD5
        - OWASP: https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html
        
        Args:
            data: Input data (string or bytes). Strings are encoded as UTF-8.
        
        Returns:
            Hexadecimal digest string (32 characters)
        """
        import warnings
        warnings.warn(
            "MD5 is cryptographically broken (NIST deprecated 2011). "
            "Use SHA-256, SHA-3, or BLAKE2 for security purposes. "
            "Use CRC32 or xxHash for fast non-cryptographic checksums.",
            DeprecationWarning,
            stacklevel=2
        )
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    
    def ripemd160(self, data: Union[str, bytes]) -> str:
        """Compute RIPEMD-160 hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.new('ripemd160', data).hexdigest()
    
    # Encoding - Base64
    def base64_encode(self, data: Union[str, bytes]) -> str:
        """Encode data to base64."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8')
    
    def base64_decode(self, data: Union[str, bytes]) -> bytes:
        """Decode base64 data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64decode(data)
    
    # Encoding - Hex
    def hex_encode(self, data: Union[str, bytes]) -> str:
        """Encode data to hexadecimal."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return data.hex()
    
    def hex_decode(self, data: str) -> bytes:
        """Decode hexadecimal data."""
        return bytes.fromhex(data)
    
    # Encoding - URL/Percent
    def url_encode(self, data: str) -> str:
        """URL/Percent encode data (RFC 3986)."""
        return urllib.parse.quote(data, safe='')
    
    def url_decode(self, data: str) -> str:
        """URL/Percent decode data (RFC 3986)."""
        return urllib.parse.unquote(data)
    
    # HMAC
    def hmac_sha256(self, data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """Compute HMAC-SHA256."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def hmac_sha512(self, data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """Compute HMAC-SHA512."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        return hmac.new(key, data, hashlib.sha512).hexdigest()
    
    # Checksums
    def crc32(self, data: Union[str, bytes]) -> str:
        """Compute CRC32 checksum."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return format(zlib.crc32(data) & 0xffffffff, '08x')


# Singleton instance for easy access (thread-safe)
_crypto_engine: Optional[CryptographicEngine] = None
_crypto_engine_lock: threading.Lock = threading.Lock()


def get_crypto_engine() -> CryptographicEngine:
    """
    Get the singleton cryptographic engine instance (thread-safe).
    
    Uses double-checked locking pattern for thread safety without
    unnecessary lock acquisition on subsequent calls.
    """
    global _crypto_engine
    if _crypto_engine is None:
        with _crypto_engine_lock:
            # Double-check after acquiring lock
            if _crypto_engine is None:
                _crypto_engine = CryptographicEngine()
    return _crypto_engine


# Convenience function
def compute_crypto(query: str) -> Dict[str, Any]:
    """
    Compute cryptographic operation from natural language query.
    
    Note: Convenience function for deterministic crypto computation.
    
    Args:
        query: Natural language query describing the operation
        
    Returns:
        Dict with operation result
        
    Example:
        >>> result = compute_crypto("Calculate SHA-256 of 'Hello, World!'")
        >>> result['result']
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
    """
    return get_crypto_engine().compute(query)


__all__ = [
    'CryptographicEngine',
    'CryptoOperation',
    'CryptoResult',
    'get_crypto_engine',
    'compute_crypto',
]
