"""
Cryptographic Tool Wrapper - Adapts CryptographicEngine to the common reason() interface.

This wrapper:
1. Detects if the query involves cryptographic computation
2. Routes to the CryptographicEngine for deterministic computation
3. Returns accurate, verifiable results (100% confidence)

Supported Operations:
- Hash functions: SHA-256, SHA-1, SHA-512, SHA-384, SHA-224, MD5
- Encoding: Base64, Hexadecimal
- Authentication: HMAC-SHA256, HMAC-SHA512
- Checksums: CRC32

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CryptographicToolWrapper:
    """
    Wrapper for CryptographicEngine that exposes reason() method.

    Note: The system was falling back to OpenAI for cryptographic
    computations (SHA-256, MD5, etc.), which resulted in hallucinated
    (incorrect) hash values.

    This wrapper provides deterministic, accurate cryptographic results
    instead of relying on LLM fallback which hallucinates.
    """

    # Keywords for detecting cryptographic queries
    _CRYPTO_KEYWORDS = frozenset({
        'sha-256', 'sha256', 'sha-1', 'sha1', 'sha-512', 'sha512',
        'sha-384', 'sha384', 'sha-224', 'sha224',
        'md5', 'hash', 'checksum', 'digest',
        'base64', 'b64', 'hex', 'hexadecimal',
        'hmac', 'crc32', 'crc-32', 'encode', 'decode'
    })

    _COMPUTE_KEYWORDS = frozenset({
        'calculate', 'compute', 'generate', 'find', 'get',
        'what is', 'determine', 'produce', 'create'
    })

    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with a CryptographicEngine instance.

        Args:
            engine: CryptographicEngine instance
            config: Optional configuration dict (for warm pool compatibility).
        """
        # BUG FIX #1: Defensive Programming - Validate engine type at initialization
        # Industry Standard: Comprehensive input validation with clear error messages
        if isinstance(engine, str):
            raise TypeError(
                f"CryptographicToolWrapper received string '{engine}' instead of CryptographicEngine instance. "
                f"Expected: CryptographicEngine instance, Got: {type(engine).__name__}"
            )
        if engine is None:
            raise ValueError(
                "CryptographicToolWrapper received None as engine. "
                "Cannot initialize wrapper with null engine."
            )
        if not (hasattr(engine, 'compute') or hasattr(engine, 'hash') or hasattr(engine, 'reason')):
            raise AttributeError(
                f"CryptographicToolWrapper engine must have 'compute()', 'hash()', or 'reason()' method. "
                f"Got type: {type(engine).__name__}"
            )
        self.engine = engine
        self.name = "cryptographic"
        self.config = config or {}
        logger.debug(f"[CryptographicToolWrapper] Initialized with engine type: {type(engine).__name__}")

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Execute cryptographic computation on the problem.

        Note: Provides deterministic, accurate cryptographic results
        instead of relying on LLM fallback which hallucinates.

        Args:
            problem: Dict with query, or string query

        Returns:
            Dict with computation result and confidence
        """
        start_time = time.time()

        try:
            # Extract query string from problem
            query_str = self._extract_query_text(problem)

            if not query_str:
                return self._not_applicable_result(
                    "No query text provided",
                    start_time
                )

            # Gate check - is this actually a cryptographic query?
            if not self._is_crypto_query(query_str):
                return self._not_applicable_result(
                    "Query does not involve cryptographic computation",
                    start_time
                )

            # Execute the cryptographic computation
            result = self.engine.compute(query_str)

            if result['success']:
                return {
                    "tool": self.name,
                    "applicable": True,
                    "result": result['result'],
                    "operation": result['operation'],
                    "input": result['input'],
                    "confidence": 1.0,  # Deterministic = 100% confidence
                    "engine": "CryptographicEngine",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "deterministic": True,
                        "bug_fix": "BUG#14",
                    }
                }
            else:
                return {
                    "tool": self.name,
                    "applicable": False,
                    "reason": result.get('error', 'Unknown error'),
                    "confidence": 0.0,
                    "engine": "CryptographicEngine",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                }

        except Exception as e:
            logger.error(f"[CryptographicToolWrapper] Error: {e}")
            return {
                "tool": self.name,
                "applicable": False,
                "reason": f"Computation failed: {str(e)}",
                "confidence": 0.0,
                "engine": "CryptographicEngine",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def _extract_query_text(self, problem: Any) -> str:
        """Extract query text from problem input."""
        if isinstance(problem, str):
            return problem
        elif isinstance(problem, dict):
            # Try common keys
            for key in ['query', 'problem', 'question', 'input', 'text']:
                if key in problem and problem[key]:
                    return str(problem[key])
            # Fall back to string representation
            return str(problem)
        else:
            return str(problem)

    def _is_crypto_query(self, query: str) -> bool:
        """
        Check if query involves cryptographic computation.

        Args:
            query: The query string

        Returns:
            True if query involves cryptographic operations
        """
        query_lower = query.lower()

        has_crypto_keyword = any(kw in query_lower for kw in self._CRYPTO_KEYWORDS)
        has_compute_keyword = any(kw in query_lower for kw in self._COMPUTE_KEYWORDS)

        return has_crypto_keyword and has_compute_keyword

    def _not_applicable_result(self, reason: str, start_time: float) -> Dict[str, Any]:
        """Return a not-applicable result."""
        return {
            "tool": self.name,
            "applicable": False,
            "reason": reason,
            "confidence": 0.0,
            "engine": "CryptographicEngine",
            "execution_time_ms": (time.time() - start_time) * 1000,
        }
