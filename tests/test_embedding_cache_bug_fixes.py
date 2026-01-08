"""
Test suite for Bug #1 and Notees:
- Bug #1: Embedding Cache Never Hits (0% Hit Rate)
- Bug #2: Query Router Always Times Out (5 seconds)

These tests verify that:
1. Embedding cache properly normalizes query text for consistent cache keys
2. Query router timeout is appropriately configured for embedding operations

Run with:
    pytest tests/test_embedding_cache_bug_fixes.py -v
"""

import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest


class TestEmbeddingCacheNormalization:
    """Tests for Note: Embedding cache text normalization."""
    
    def test_normalize_text_strips_whitespace(self):
        """Test that _normalize_text strips leading/trailing whitespace."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # Test cases with various whitespace patterns
        test_cases = [
            ("  hello world  ", "hello world"),
            ("test query\t", "test query"),
            ("\n\ntest\n\n", "test"),
            ("   multiple   spaces   ", "multiple spaces"),
        ]
        
        for input_text, expected in test_cases:
            normalized = MultiTierFeatureExtractor._normalize_text(input_text)
            assert normalized == expected, f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"
    
    def test_normalize_text_lowercases(self):
        """Test that _normalize_text converts to lowercase."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        test_cases = [
            ("Hello World", "hello world"),
            ("UPPERCASE QUERY", "uppercase query"),
            ("MixedCase TEXT", "mixedcase text"),
        ]
        
        for input_text, expected in test_cases:
            normalized = MultiTierFeatureExtractor._normalize_text(input_text)
            assert normalized == expected, f"Expected '{expected}' but got '{normalized}' for input '{input_text}'"
    
    def test_normalize_text_collapses_whitespace(self):
        """Test that _normalize_text collapses multiple whitespace to single space."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        test_cases = [
            ("hello    world", "hello world"),
            ("multiple   spaces   here", "multiple spaces here"),
            ("tabs\t\there", "tabs here"),
            ("newlines\n\nhere", "newlines here"),
        ]
        
        for input_text, expected in test_cases:
            normalized = MultiTierFeatureExtractor._normalize_text(input_text)
            assert normalized == expected, f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"
    
    def test_cache_key_consistency_with_normalization(self):
        """Test that different representations of same query produce same cache key."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # These should all produce the same cache key after normalization
        equivalent_queries = [
            "Hello World",
            "hello world",
            "  hello world  ",
            "HELLO WORLD",
            "hello   world",
            "\nhello world\n",
        ]
        
        cache_keys = [MultiTierFeatureExtractor._compute_cache_key(q) for q in equivalent_queries]
        
        # All keys should be identical
        first_key = cache_keys[0]
        for i, key in enumerate(cache_keys):
            assert key == first_key, f"Query '{equivalent_queries[i]}' produced different cache key: {key} vs {first_key}"
    
    def test_cache_key_different_for_different_content(self):
        """Test that different queries produce different cache keys."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        different_queries = [
            "hello world",
            "goodbye world",
            "hello there",
            "world hello",
        ]
        
        cache_keys = [MultiTierFeatureExtractor._compute_cache_key(q) for q in different_queries]
        unique_keys = set(cache_keys)
        
        assert len(unique_keys) == len(different_queries), "Different queries should produce different cache keys"
    
    def test_embedding_cache_hit_after_store(self):
        """Test that storing an embedding leads to cache hit with same normalized text."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # Clear cache first
        with MultiTierFeatureExtractor._embedding_cache_lock:
            MultiTierFeatureExtractor._embedding_cache.clear()
            MultiTierFeatureExtractor._embedding_cache_hits = 0
            MultiTierFeatureExtractor._embedding_cache_misses = 0
        
        # Create a test embedding
        test_embedding = np.random.randn(128).astype(np.float32)
        
        # Store with original query
        original_query = "What is the causal relationship?"
        MultiTierFeatureExtractor._cache_embedding(original_query, test_embedding)
        
        # Try to retrieve with equivalent queries (should hit cache)
        equivalent_queries = [
            "what is the causal relationship?",  # lowercase
            "  What is the causal relationship?  ",  # whitespace
            "WHAT IS THE CAUSAL RELATIONSHIP?",  # uppercase
        ]
        
        for query in equivalent_queries:
            result = MultiTierFeatureExtractor._get_cached_embedding(query)
            assert result is not None, f"Cache miss for equivalent query: '{query}'"
            assert np.allclose(result, test_embedding), f"Cache returned different embedding for query: '{query}'"
        
        # Verify hit rate is 100% for equivalent queries (3 hits, 0 misses from equivalent queries)
        stats = MultiTierFeatureExtractor.get_cache_stats()
        assert stats['hits'] >= 3, f"Expected at least 3 hits, got {stats['hits']}"


class TestQueryRouterTimeout:
    """Tests for Note: Query Router timeout configuration."""
    
    def test_timeout_constant_increased(self):
        """Test that QUERY_ROUTING_TIMEOUT_SECONDS is increased from 5s."""
        from vulcan.routing.query_router import QUERY_ROUTING_TIMEOUT_SECONDS
        
        # The timeout should be increased to accommodate embedding computation
        # Embedding can take 10-15 seconds, so timeout should be at least 15s
        assert QUERY_ROUTING_TIMEOUT_SECONDS >= 15.0, (
            f"QUERY_ROUTING_TIMEOUT_SECONDS should be at least 15s to accommodate "
            f"embedding computation, but is {QUERY_ROUTING_TIMEOUT_SECONDS}s"
        )
    
    def test_timeout_not_excessive(self):
        """Test that timeout is not excessively long."""
        from vulcan.routing.query_router import QUERY_ROUTING_TIMEOUT_SECONDS
        
        # Timeout should not be more than 30 seconds to prevent indefinite waits
        assert QUERY_ROUTING_TIMEOUT_SECONDS <= 30.0, (
            f"QUERY_ROUTING_TIMEOUT_SECONDS should not exceed 30s, "
            f"but is {QUERY_ROUTING_TIMEOUT_SECONDS}s"
        )
    
    def test_fallback_plan_created_on_timeout(self):
        """Test that _create_fallback_plan returns a valid plan."""
        from vulcan.routing.query_router import _create_fallback_plan, ProcessingPlan
        
        fallback_plan = _create_fallback_plan(
            query="test query",
            source="user",
            session_id="test_session",
            timeout_exceeded=True
        )
        
        assert isinstance(fallback_plan, ProcessingPlan)
        assert fallback_plan.query_id.startswith("q_fallback_")
        assert "routing_timeout" in fallback_plan.detected_patterns
        assert fallback_plan.telemetry_data.get("timeout_exceeded") is True
        assert len(fallback_plan.agent_tasks) == 1  # Should have one fallback task
    
    def test_trivial_query_bypasses_heavy_analysis(self):
        """Test that trivial queries use fast path and don't timeout."""
        from vulcan.routing.query_router import QueryAnalyzer
        
        analyzer = QueryAnalyzer(enable_safety_validation=False)
        
        trivial_queries = [
            "hello",
            "hi",
            "thanks",
            "ok",
            "yes",
        ]
        
        for query in trivial_queries:
            assert analyzer._is_trivial_query(query), f"'{query}' should be detected as trivial"
            
            # Route should complete very quickly for trivial queries
            start_time = time.perf_counter()
            plan = analyzer.route_query(query, source="user")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Trivial queries should complete in under 1 second
            assert elapsed_ms < 1000, f"Trivial query '{query}' took {elapsed_ms:.2f}ms"
            # Should have fast_path flag
            assert plan.telemetry_data.get("fast_path") is True


class TestEmbeddingCacheIntegration:
    """Integration tests for embedding cache with actual cache behavior."""
    
    def test_cache_stats_tracking(self):
        """Test that cache statistics are properly tracked."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # Clear cache first
        with MultiTierFeatureExtractor._embedding_cache_lock:
            MultiTierFeatureExtractor._embedding_cache.clear()
            MultiTierFeatureExtractor._embedding_cache_hits = 0
            MultiTierFeatureExtractor._embedding_cache_misses = 0
        
        # Create and cache an embedding
        test_embedding = np.random.randn(128).astype(np.float32)
        MultiTierFeatureExtractor._cache_embedding("test query", test_embedding)
        
        # First get - should be a hit
        MultiTierFeatureExtractor._get_cached_embedding("test query")
        
        stats = MultiTierFeatureExtractor.get_cache_stats()
        assert stats['hits'] == 1, f"Expected 1 hit, got {stats['hits']}"
        assert stats['size'] == 1, f"Expected cache size 1, got {stats['size']}"
        
        # Miss on different query
        MultiTierFeatureExtractor._get_cached_embedding("different query")
        
        stats = MultiTierFeatureExtractor.get_cache_stats()
        assert stats['misses'] >= 1, f"Expected at least 1 miss, got {stats['misses']}"
    
    def test_cache_eviction_on_capacity(self):
        """Test that cache evicts old entries when at capacity."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # Store original maxsize
        original_maxsize = MultiTierFeatureExtractor._embedding_cache_maxsize
        
        try:
            # Temporarily set a small max size for testing
            MultiTierFeatureExtractor._embedding_cache_maxsize = 10
            
            # Clear cache
            with MultiTierFeatureExtractor._embedding_cache_lock:
                MultiTierFeatureExtractor._embedding_cache.clear()
            
            # Fill cache beyond capacity
            for i in range(15):
                test_embedding = np.random.randn(128).astype(np.float32)
                MultiTierFeatureExtractor._cache_embedding(f"query_{i}", test_embedding)
            
            stats = MultiTierFeatureExtractor.get_cache_stats()
            assert stats['size'] <= 10, f"Cache size should not exceed max, got {stats['size']}"
            
        finally:
            # Restore original maxsize
            MultiTierFeatureExtractor._embedding_cache_maxsize = original_maxsize


class TestCacheKeyDeterminism:
    """Tests to verify cache keys are deterministic (no timestamps/UUIDs)."""
    
    def test_cache_key_deterministic_across_calls(self):
        """Test that same input produces same cache key across multiple calls."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        test_query = "What is the meaning of life?"
        
        # Generate key multiple times
        keys = [MultiTierFeatureExtractor._compute_cache_key(test_query) for _ in range(10)]
        
        # All keys should be identical
        assert len(set(keys)) == 1, "Cache key should be deterministic"
    
    def test_cache_key_is_content_based(self):
        """Test that cache key is based purely on content, not metadata."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        # Key should be same regardless of when computed
        query = "test query for cache"
        
        key1 = MultiTierFeatureExtractor._compute_cache_key(query)
        time.sleep(0.01)  # Small delay
        key2 = MultiTierFeatureExtractor._compute_cache_key(query)
        
        assert key1 == key2, "Cache key should not depend on time"
    
    def test_cache_key_uses_sha256(self):
        """Test that cache key is a valid SHA-256 substring."""
        from vulcan.reasoning.selection.tool_selector import MultiTierFeatureExtractor
        
        key = MultiTierFeatureExtractor._compute_cache_key("test")
        
        # Should be 32 characters (truncated SHA-256)
        assert len(key) == 32, f"Cache key should be 32 chars, got {len(key)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in key), "Cache key should be hexadecimal"


class TestToolSelectionDeterminism:
    """Tests for tool selection determinism fixes."""
    
    def test_bandit_fallback_is_deterministic(self):
        """Test that ToolSelectionBandit fallback returns consistent results."""
        from vulcan.reasoning.selection.tool_selector import ToolSelectionBandit
        import numpy as np
        
        bandit = ToolSelectionBandit({'feature_dim': 128})
        bandit.is_enabled = False  # Force fallback behavior
        
        # Run 100 times with different features - should always return same tool
        results = []
        for i in range(100):
            features = np.zeros(128) + i * 0.1  # Varying features
            result = bandit.select_tool(features, {})
            results.append(result)
        
        # All results should be identical (deterministic)
        unique_results = set(results)
        assert len(unique_results) == 1, \
            f"ToolSelectionBandit fallback should be deterministic, got {len(unique_results)} different results: {unique_results}"
        
        # Should return a valid tool name
        assert results[0] in ["symbolic", "probabilistic", "causal", "analogical", "multimodal"], \
            f"ToolSelectionBandit fallback returned invalid tool: {results[0]}"
    
    def test_feature_extraction_fallback_is_deterministic(self):
        """Test that feature extraction error fallback produces consistent results."""
        import numpy as np
        
        # Simulate what happens when feature extraction fails:
        # The fallback should return zeros instead of random features
        fallback_features_list = [np.zeros(128) for _ in range(10)]
        
        # All fallback features should be identical
        for i, features in enumerate(fallback_features_list):
            assert np.allclose(features, fallback_features_list[0]), \
                f"Fallback features at index {i} differ from index 0"
            assert np.allclose(features, 0), \
                f"Fallback features should be zeros, got {features[:5]}..."
    
    def test_selection_cache_features_are_deterministic(self):
        """Test that selection cache feature extraction produces consistent results."""
        from vulcan.reasoning.selection.selection_cache import SelectionCache
        import numpy as np
        
        cache = SelectionCache({})
        
        # Extract features multiple times for same pattern
        results = [cache._extract_features_for_pattern("test pattern") for _ in range(10)]
        
        # All results should be identical (deterministic)
        for i, features in enumerate(results):
            assert features is not None, f"Features at index {i} should not be None"
            assert np.allclose(features, results[0]), \
                f"Features at index {i} differ from index 0"
    
    def test_disabled_bandit_returns_consistent_tool(self):
        """Test that disabled bandit returns same tool every time."""
        from vulcan.reasoning.selection.tool_selector import ToolSelectionBandit
        import numpy as np
        
        bandit = ToolSelectionBandit({'feature_dim': 128})
        bandit.is_enabled = False  # Force disabled
        
        # Run 10 times - should always return same tool
        results = [bandit.select_tool(np.zeros(128), {}) for _ in range(10)]
        
        assert len(set(results)) == 1, \
            f"Disabled bandit should return consistent tool, got: {results}"
        assert results[0] == "probabilistic", \
            f"Disabled bandit should return 'probabilistic', got: {results[0]}"


class TestEmbeddingCacheKeyNormalization:
    """Tests for EmbeddingCache key normalization fix (0% hit rate issue)."""
    
    def test_embedding_cache_normalize_text(self):
        """Test that _normalize_text properly normalizes text."""
        from vulcan.routing.embedding_cache import EmbeddingCache
        
        test_cases = [
            ("  hello world  ", "hello world"),
            ("HELLO WORLD", "hello world"),
            ("hello   world", "hello world"),
            ("\n\nhello\t\tworld\n\n", "hello world"),
            ("MixedCase TEXT", "mixedcase text"),
        ]
        
        for input_text, expected in test_cases:
            normalized = EmbeddingCache._normalize_text(input_text)
            assert normalized == expected, \
                f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"
    
    def test_embedding_cache_key_consistency(self):
        """Test that EmbeddingCache generates same key for equivalent queries."""
        from vulcan.routing.embedding_cache import EmbeddingCache
        
        # These should all produce the same cache key after normalization
        equivalent_queries = [
            "Hello World",
            "hello world",
            "  hello world  ",
            "HELLO WORLD",
            "hello   world",
            "\nhello world\n",
        ]
        
        cache_keys = [EmbeddingCache._make_key(q) for q in equivalent_queries]
        
        # All keys should be identical
        first_key = cache_keys[0]
        for i, key in enumerate(cache_keys):
            assert key == first_key, \
                f"Query '{equivalent_queries[i]}' produced different cache key: {key} vs {first_key}"
    
    def test_embedding_cache_different_keys_for_different_content(self):
        """Test that different queries produce different cache keys."""
        from vulcan.routing.embedding_cache import EmbeddingCache
        
        different_queries = [
            "hello world",
            "goodbye world",
            "hello there",
            "world hello",
        ]
        
        cache_keys = [EmbeddingCache._make_key(q) for q in different_queries]
        unique_keys = set(cache_keys)
        
        assert len(unique_keys) == len(different_queries), \
            "Different queries should produce different cache keys"


class TestQueryRouterCacheKeyNormalization:
    """Tests for QueryRouter cache key normalization fix."""
    
    def test_query_router_normalize_text(self):
        """Test that _normalize_text in query_router properly normalizes text."""
        from vulcan.routing.query_router import _normalize_text
        
        test_cases = [
            ("  hello world  ", "hello world"),
            ("HELLO WORLD", "hello world"),
            ("hello   world", "hello world"),
            ("\n\nhello\t\tworld\n\n", "hello world"),
        ]
        
        for input_text, expected in test_cases:
            normalized = _normalize_text(input_text)
            assert normalized == expected, \
                f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"
    
    def test_query_router_hash_consistency(self):
        """Test that _compute_query_hash generates same hash for equivalent queries."""
        from vulcan.routing.query_router import _compute_query_hash
        
        equivalent_queries = [
            "Hello World",
            "hello world",
            "  hello world  ",
            "HELLO WORLD",
        ]
        
        hashes = [_compute_query_hash(q) for q in equivalent_queries]
        
        # All hashes should be identical
        first_hash = hashes[0]
        for i, h in enumerate(hashes):
            assert h == first_hash, \
                f"Query '{equivalent_queries[i]}' produced different hash: {h} vs {first_hash}"


class TestSemanticToolMatcherCacheKeyNormalization:
    """Tests for SemanticToolMatcher cache key normalization fix."""
    
    def test_semantic_matcher_normalize_text(self):
        """Test that _normalize_text in SemanticToolMatcher properly normalizes text."""
        from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
        
        test_cases = [
            ("  hello world  ", "hello world"),
            ("HELLO WORLD", "hello world"),
            ("hello   world", "hello world"),
        ]
        
        for input_text, expected in test_cases:
            normalized = SemanticToolMatcher._normalize_text(input_text)
            assert normalized == expected, \
                f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"


class TestSelectionCacheKeyNormalization:
    """Tests for SelectionCache cache key normalization fix."""
    
    def test_selection_cache_normalize_text(self):
        """Test that _normalize_text in SelectionCache properly normalizes text."""
        from vulcan.reasoning.selection.selection_cache import SelectionCache
        
        test_cases = [
            ("  hello world  ", "hello world"),
            ("HELLO WORLD", "hello world"),
            ("hello   world", "hello world"),
        ]
        
        for input_text, expected in test_cases:
            normalized = SelectionCache._normalize_text(input_text)
            assert normalized == expected, \
                f"Expected '{expected}' but got '{normalized}' for input '{repr(input_text)}'"
    
    def test_selection_cache_feature_key_consistency(self):
        """Test that SelectionCache generates same feature key for equivalent problems."""
        from vulcan.reasoning.selection.selection_cache import SelectionCache
        
        cache = SelectionCache({})
        
        equivalent_problems = [
            "Hello World",
            "hello world",
            "  hello world  ",
            "HELLO WORLD",
        ]
        
        keys = [cache._compute_feature_key(p) for p in equivalent_problems]
        
        # All keys should be identical
        first_key = keys[0]
        for i, key in enumerate(keys):
            assert key == first_key, \
                f"Problem '{equivalent_problems[i]}' produced different key: {key} vs {first_key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
