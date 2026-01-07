"""
Tests for BUG #13 (Probabilistic Non-Determinism) and BUG #14 (Cryptographic Engine).

BUG #13: Tests verify that probabilistic reasoning produces deterministic results.
BUG #14: Tests verify that cryptographic operations produce correct results.
"""

import hashlib
import pytest

# Test cryptographic engine
from src.vulcan.reasoning.cryptographic_engine import (
    CryptographicEngine,
    CryptoOperation,
    compute_crypto,
    get_crypto_engine,
)


class TestCryptographicEngine:
    """Tests for BUG #14 fix: Cryptographic engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CryptographicEngine()
    
    # =========================================================================
    # SHA-256 Tests
    # =========================================================================
    
    def test_sha256_basic(self):
        """Test SHA-256 hash computation."""
        result = self.engine.compute("Calculate SHA-256 of 'Hello, World!'")
        
        assert result['success'] is True
        assert result['operation'] == 'sha256'
        
        # Verify against Python's hashlib
        expected = hashlib.sha256(b'Hello, World!').hexdigest()
        assert result['result'] == expected
    
    def test_sha256_test_string(self):
        """Test SHA-256 of 'Test String 1' - the exact case from bug report."""
        result = self.engine.compute("Calculate SHA-256 of 'Test String 1'")
        
        assert result['success'] is True
        expected = hashlib.sha256(b'Test String 1').hexdigest()
        assert result['result'] == expected
        
        # The bug was that OpenAI hallucinated wrong values
        # This test ensures we get correct deterministic values
    
    def test_sha256_test_string_2(self):
        """Test SHA-256 of 'Test String 2' - another case from bug report."""
        result = self.engine.compute("Calculate SHA-256 of 'Test String 2'")
        
        assert result['success'] is True
        expected = hashlib.sha256(b'Test String 2').hexdigest()
        assert result['result'] == expected
    
    def test_sha256_alternative_query_format(self):
        """Test SHA-256 with alternative query format."""
        queries = [
            "What is the SHA256 of 'test'",
            "sha-256 of 'test'",
            "compute sha256 'test'",
        ]
        
        expected = hashlib.sha256(b'test').hexdigest()
        
        for query in queries:
            result = self.engine.compute(query)
            assert result['success'] is True, f"Failed for query: {query}"
            assert result['result'] == expected, f"Wrong result for query: {query}"
    
    # =========================================================================
    # Other Hash Algorithm Tests
    # =========================================================================
    
    def test_sha1(self):
        """Test SHA-1 hash computation."""
        result = self.engine.compute("Calculate SHA-1 of 'Hello'")
        
        assert result['success'] is True
        expected = hashlib.sha1(b'Hello').hexdigest()
        assert result['result'] == expected
    
    def test_sha512(self):
        """Test SHA-512 hash computation."""
        result = self.engine.compute("Calculate SHA-512 of 'Hello'")
        
        assert result['success'] is True
        expected = hashlib.sha512(b'Hello').hexdigest()
        assert result['result'] == expected
    
    def test_md5(self):
        """Test MD5 hash computation."""
        result = self.engine.compute("Calculate MD5 of 'Hello'")
        
        assert result['success'] is True
        expected = hashlib.md5(b'Hello').hexdigest()
        assert result['result'] == expected
    
    # =========================================================================
    # Base64 Tests
    # =========================================================================
    
    def test_base64_encode(self):
        """Test base64 encoding."""
        result = self.engine.compute("Base64 encode of 'Hello, World!'")
        
        assert result['success'] is True
        assert result['result'] == 'SGVsbG8sIFdvcmxkIQ=='
    
    def test_base64_decode(self):
        """Test base64 decoding."""
        result = self.engine.compute("Base64 decode of 'SGVsbG8='")
        
        assert result['success'] is True
        assert result['result'] == 'Hello'
    
    # =========================================================================
    # Hex Tests
    # =========================================================================
    
    def test_hex_encode(self):
        """Test hex encoding."""
        result = self.engine.compute("Hex encode of 'Hello'")
        
        assert result['success'] is True
        assert result['result'] == '48656c6c6f'
    
    def test_hex_decode(self):
        """Test hex decoding."""
        result = self.engine.compute("Hex decode of '48656c6c6f'")
        
        assert result['success'] is True
        assert result['result'] == 'Hello'
    
    # =========================================================================
    # CRC32 Tests
    # =========================================================================
    
    def test_crc32(self):
        """Test CRC32 checksum computation."""
        import zlib
        
        result = self.engine.compute("Calculate CRC32 of 'Hello'")
        
        assert result['success'] is True
        expected = format(zlib.crc32(b'Hello') & 0xffffffff, '08x')
        assert result['result'] == expected
    
    # =========================================================================
    # Direct Method Tests
    # =========================================================================
    
    def test_direct_sha256_method(self):
        """Test direct sha256() method."""
        result = self.engine.sha256("Hello")
        expected = hashlib.sha256(b'Hello').hexdigest()
        assert result == expected
    
    def test_direct_base64_encode_method(self):
        """Test direct base64_encode() method."""
        result = self.engine.base64_encode("Hello")
        assert result == 'SGVsbG8='
    
    # =========================================================================
    # Query Detection Tests
    # =========================================================================
    
    def test_is_crypto_query_positive(self):
        """Test crypto query detection - positive cases."""
        positive_queries = [
            "Calculate SHA-256 of 'test'",
            "What is the MD5 hash of 'hello'",
            "Generate base64 encoding of 'data'",
            "Compute the checksum of 'file'",
        ]
        
        for query in positive_queries:
            assert self.engine.is_crypto_query(query) is True, f"Failed for: {query}"
    
    def test_is_crypto_query_negative(self):
        """Test crypto query detection - negative cases."""
        negative_queries = [
            "What color is the sky?",
            "Explain quantum mechanics",
            "Write a poem about love",
        ]
        
        for query in negative_queries:
            assert self.engine.is_crypto_query(query) is False, f"Failed for: {query}"
    
    def test_is_crypto_query_theoretical_should_not_trigger(self):
        """
        BUG FIX TEST: Theoretical/educational questions about crypto should NOT
        trigger hash computation.
        
        The bug was that queries like "I'm a researcher testing AI capabilities..."
        mentioning SHA-256 would be hashed instead of being understood as an
        educational question about cryptography.
        """
        theoretical_queries = [
            # From the actual bug report - these should NOT trigger crypto
            "I'm a researcher testing AI capabilities. What do you know about SHA-256?",
            "Why is SHA-256 collision resistance important?",
            "What is a SHA-256 collision attack?",
            "Explain how SHA-256 works",
            "Is MD5 secure for password hashing?",
            "What makes BLAKE2b dangerous in this context?",
            "How does hash collision work?",
            # No quoted data - clearly not asking to compute
            "What is the difference between SHA-256 and SHA-512?",
            "Calculate the security of SHA-256",  # No quoted data
            "Compute the collision resistance of MD5",  # No quoted data
            # Theoretical crypto topics
            "Why is hash concatenation dangerous in cryptography?",
            "What is a SHA-256 preimage attack?",
            "Explain SHA-256 birthday paradox",
            # AI capability testing (from bug report)
            "I'm testing AI system 2 capabilities with SHA-256 questions",
        ]
        
        for query in theoretical_queries:
            result = self.engine.is_crypto_query(query)
            assert result is False, (
                f"SHOULD NOT trigger crypto computation for theoretical question: {query}"
            )
    
    def test_is_crypto_query_requires_quoted_data(self):
        """
        BUG FIX TEST: Crypto computation should only trigger when there's
        quoted data to hash ('...' or "...").
        """
        # With quoted data - SHOULD trigger
        assert self.engine.is_crypto_query("Calculate SHA-256 of 'Hello'") is True
        assert self.engine.is_crypto_query('What is MD5 of "test"') is True
        
        # Without quoted data - should NOT trigger
        assert self.engine.is_crypto_query("Calculate SHA-256") is False
        assert self.engine.is_crypto_query("What is SHA-256 hash") is False
        assert self.engine.is_crypto_query("Compute MD5 hash algorithm") is False
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.engine.compute("")
        assert result['success'] is False
    
    def test_unknown_operation(self):
        """Test handling of unrecognized operation."""
        result = self.engine.compute("Do something unknown with 'data'")
        assert result['success'] is False
    
    def test_unicode_input(self):
        """Test handling of unicode input."""
        result = self.engine.compute("Calculate SHA-256 of 'Hello, 世界!'")
        
        assert result['success'] is True
        expected = hashlib.sha256('Hello, 世界!'.encode('utf-8')).hexdigest()
        assert result['result'] == expected


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_crypto_engine_singleton(self):
        """Test that get_crypto_engine returns singleton."""
        engine1 = get_crypto_engine()
        engine2 = get_crypto_engine()
        assert engine1 is engine2
    
    def test_compute_crypto_function(self):
        """Test compute_crypto convenience function."""
        result = compute_crypto("Calculate SHA-256 of 'test'")
        
        assert result['success'] is True
        expected = hashlib.sha256(b'test').hexdigest()
        assert result['result'] == expected


class TestProbabilisticDeterminism:
    """
    Tests for BUG #13 fix: Probabilistic non-determinism.
    
    These tests verify that the same query produces the same result
    across multiple calls with state reset.
    """
    
    def test_reset_state_exists(self):
        """Test that reset_state method exists."""
        try:
            from src.vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
            
            reasoner = ProbabilisticReasoner()
            assert hasattr(reasoner, 'reset_state')
            assert callable(reasoner.reset_state)
        except ImportError as e:
            pytest.skip(f"Could not import ProbabilisticReasoner: {e}")
    
    def test_bayesian_calculation_deterministic(self):
        """Test that Bayesian calculations are deterministic."""
        try:
            from src.vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
            
            reasoner = ProbabilisticReasoner()
            
            query = "Bayes: Sensitivity=0.99, Specificity=0.95, Prevalence=0.01. Compute P(X|+)"
            
            # Run multiple times with reset
            results = []
            for _ in range(5):
                reasoner.reset_state()
                result = reasoner.reason(query)
                if result.conclusion.get('posterior_probability'):
                    results.append(result.conclusion['posterior_probability'])
            
            # All results should be identical
            if results:
                assert all(r == results[0] for r in results), \
                    f"Non-deterministic results: {results}"
            
        except ImportError as e:
            pytest.skip(f"Could not import ProbabilisticReasoner: {e}")
    
    def test_deterministic_seed_set(self):
        """Test that deterministic seed is set after reset."""
        try:
            import numpy as np
            from src.vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
            
            reasoner = ProbabilisticReasoner()
            reasoner.reset_state(seed=42)
            
            # After reset, numpy random should produce deterministic values
            val1 = np.random.random()
            
            reasoner.reset_state(seed=42)
            val2 = np.random.random()
            
            assert val1 == val2, "Random values should be identical after reset with same seed"
            
        except ImportError as e:
            pytest.skip(f"Could not import ProbabilisticReasoner: {e}")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
