"""
Tests for cognitive_loop.py
"""

import sys

import pytest

# Import from src.integration module path
from src.integration.cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopResult,
    LoopRuntimeConfig,
    LoopSamplingConfig,
    apply_top_k,
    apply_top_p,
    choose_token,
    penalize_repetition,
    softmax,
)


class MockBridge:
    """Mock bridge for testing."""

    def __init__(self):
        self.world_model = MockWorldModel()
        self.reasoning = None

    async def before_execution(self, prompt):
        return {"prompt": prompt, "tokens": prompt.split()}

    async def reason_next_token(self, hidden, context):
        return [0, 1, 2]

    async def validate_token(self, token, context, hidden):
        return token, None

    async def consensus_approve_token(self, token, position, chosen_index=None):
        return True


class MockWorldModel:
    """Mock world model."""

    async def intervene_before_emit(self, token, context, hidden):
        return None


class MockTransformer:
    """Mock transformer (renamed from MockModel for clarity)."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, tokens):
        # Return mock logits
        pass

        1 if isinstance(tokens, list) else len(tokens)
        seq_len = len(tokens) if isinstance(tokens, list) else 1
        return type(
            "obj",
            (object,),
            {"logits": [[1.0] * self.vocab_size for _ in range(seq_len)]},
        )()


class MockSafety:
    """Mock safety monitor for testing."""

    def __init__(self):
        pass

    async def validate_token(self, token, context):
        """Mock token validation - always approve for testing."""
        return True

    async def validate_sequence(self, tokens, context, world_model):
        """Mock sequence validation - always approve for testing."""
        return True


class TestUtilityFunctions:
    """Test utility functions."""

    def test_softmax(self):
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)

        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-5
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_empty(self):
        result = softmax([])
        assert result == []

    def test_apply_top_k(self):
        logits = [1.0, 5.0, 3.0, 2.0, 4.0]
        filtered = apply_top_k(logits, 3)

        # Should keep top 3: 5.0, 4.0, 3.0
        assert filtered[1] == 5.0  # Kept
        assert filtered[4] == 4.0  # Kept
        assert filtered[2] == 3.0  # Kept
        assert filtered[0] == float("-inf")  # Filtered

    def test_apply_top_p(self):
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        filtered = apply_top_p(logits, 0.9)

        # Should filter some low-probability tokens
        assert any(l == float("-inf") for l in filtered)

    def test_penalize_repetition(self):
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        generated = [1, 2, 1, 2]  # Repeated tokens

        penalized = penalize_repetition(logits, generated, penalty=1.5, window=10)

        # Tokens 1 and 2 should be penalized
        assert penalized[1] < logits[1]
        assert penalized[2] < logits[2]

    def test_choose_token_greedy(self):
        logits = [1.0, 5.0, 3.0, 2.0]
        token = choose_token(logits, temperature=0.0)

        # Should choose argmax (index 1)
        assert token == 1

    def test_choose_token_sampling(self):
        logits = [1.0, 5.0, 3.0, 2.0]
        token = choose_token(logits, temperature=1.0)

        # Should return a valid index
        assert 0 <= token < len(logits)


class TestLoopSamplingConfig:
    """Test LoopSamplingConfig."""

    def test_default_config(self):
        config = LoopSamplingConfig()
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.max_tokens == 128

    def test_custom_config(self):
        config = LoopSamplingConfig(temperature=0.9, top_k=100, max_tokens=256)
        assert config.temperature == 0.9
        assert config.top_k == 100
        assert config.max_tokens == 256


class TestLoopRuntimeConfig:
    """Test LoopRuntimeConfig."""

    def test_default_config(self):
        config = LoopRuntimeConfig()
        assert config.enable_stream is True
        assert config.enable_audit is True
        assert config.safety_per_token is True

    def test_custom_config(self):
        config = LoopRuntimeConfig(enable_stream=False, safety_per_token=False)
        assert config.enable_stream is False
        assert config.safety_per_token is False


class TestCognitiveLoop:
    """Test CognitiveLoop functionality."""

    def test_initialization(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()

        loop = CognitiveLoop(bridge=bridge, transformer=transformer, safety=safety)

        assert loop.bridge is bridge
        assert loop.transformer is transformer
        assert loop.safety is safety
        assert loop.sampling is not None
        assert loop.runtime is not None

    def test_initialization_with_configs(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        sampling = LoopSamplingConfig(temperature=0.5)
        runtime = LoopRuntimeConfig(enable_stream=False)

        loop = CognitiveLoop(
            bridge=bridge,
            transformer=transformer,
            safety=safety,
            sampling_config=sampling,
            runtime_config=runtime,
        )

        assert loop.sampling.temperature == 0.5
        assert loop.runtime.enable_stream is False

    @pytest.mark.asyncio
    async def test_tokenize(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        tokens = await loop._tokenize("hello world")
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_decode(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        text = await loop._decode(["hello", "world"])
        assert "hello" in text
        assert "world" in text

    @pytest.mark.asyncio
    async def test_decode_mixed_token_types(self):
        """Test that decode handles mixed int and str token types."""
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        # Test with mixed int and str tokens (simulates real-world scenario
        # where some tokens can't be converted to strings by vocab)
        mixed_tokens = ["hello", 42, "world", 100]
        text = await loop._decode(mixed_tokens)
        assert "hello" in text
        assert "world" in text
        # Integer tokens should be converted to strings
        assert "42" in text
        assert "100" in text

    @pytest.mark.asyncio
    async def test_decode_all_int_tokens(self):
        """Test decode with all integer tokens (fallback behavior)."""
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        # Test with all integer tokens
        int_tokens = [1, 2, 3, 4]
        text = await loop._decode(int_tokens)
        # Without a proper tokenizer/vocab, integers should be converted to strings
        assert "1" in text
        assert "2" in text
        assert "3" in text
        assert "4" in text

    @pytest.mark.asyncio
    async def test_per_token_consensus(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        # No consensus engine, should auto-approve
        approved = await loop._per_token_consensus("token", [], 0, {})
        assert approved is True

    @pytest.mark.asyncio
    async def test_validate_token(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)

        result = await loop._validate_token("test", {}, None)
        # _validate_token returns Tuple[Token, Optional[Dict[str, Any]]]
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        final_token, notes = result
        assert final_token == "test"

    @pytest.mark.asyncio
    async def test_build_token_rationale(self):
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        runtime = LoopRuntimeConfig(attach_token_rationale=True)
        loop = CognitiveLoop(
            bridge=bridge,
            transformer=transformer,
            safety=safety,
            runtime_config=runtime,
        )

        rationale = await loop._build_token_rationale(
            "token", "hidden", ["cand1", "cand2"], {}
        )

        assert "token" in rationale
        assert "candidate_set_size" in rationale
        assert rationale["candidate_set_size"] == 2


class TestCognitiveLoopResult:
    """Test CognitiveLoopResult dataclass."""

    def test_result_creation(self):
        result = CognitiveLoopResult(
            tokens=["hello", "world"],
            text="hello world",
            reasoning_trace=[],
            safety_events=[],
            audit_records=[],
            beam_metadata=None,
            speculative_stats=None,
            metrics={"num_generated": 2},
            completed=True,
            stopped_reason="max_tokens",
            duration_seconds=1.5,
        )

        assert result.tokens == ["hello", "world"]
        assert result.text == "hello world"
        assert result.completed is True
        assert result.stopped_reason == "max_tokens"
        assert result.duration_seconds == 1.5


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    def test_encoding_cache_initialization(self):
        """Test that encoding cache is properly initialized."""
        from src.integration.cognitive_loop import EncodingCache
        
        cache = EncodingCache(max_size=100, ttl_seconds=30.0)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 30.0
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_encoding_cache_put_get(self):
        """Test encoding cache put and get operations."""
        from src.integration.cognitive_loop import EncodingCache
        
        cache = EncodingCache(max_size=100, ttl_seconds=30.0)
        tokens = ["hello", "world"]
        value = {"hidden": [1.0, 2.0, 3.0]}
        
        cache.put(tokens, value)
        result = cache.get(tokens)
        
        assert result == value
        
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1

    def test_encoding_cache_miss(self):
        """Test encoding cache miss."""
        from src.integration.cognitive_loop import EncodingCache
        
        cache = EncodingCache(max_size=100, ttl_seconds=30.0)
        result = cache.get(["nonexistent"])
        
        assert result is None
        
        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_logits_cache_operations(self):
        """Test logits cache put and get operations."""
        from src.integration.cognitive_loop import LogitsCache
        
        cache = LogitsCache(max_size=50, ttl_seconds=15.0)
        tokens = [1, 2, 3]
        logits = [0.1, 0.5, 0.4]
        
        cache.put(tokens, logits)
        result = cache.get(tokens)
        
        assert result == logits

    def test_cognitive_loop_performance_stats(self):
        """Test that performance stats are properly exposed."""
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)
        
        stats = loop.get_performance_stats()
        
        assert "encoding_cache" in stats
        assert "perf_metrics" in stats
        assert "context_cache_step" in stats

    def test_cognitive_loop_clear_caches(self):
        """Test cache clearing functionality."""
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)
        
        # Clear caches should not raise
        loop.clear_caches()
        
        # Verify caches are cleared
        assert loop._cached_context is None
        assert loop._context_cache_step == -1

    def test_optimized_sampling_with_cache(self):
        """Test optimized sampling method."""
        bridge = MockBridge()
        transformer = MockTransformer()
        safety = MockSafety()
        loop = CognitiveLoop(bridge, transformer, safety)
        
        logits = [0.1, 0.5, 0.3, 0.1]
        generated = [1, 2]
        
        chosen_idx, filtered = loop._sample_optimized(
            logits=logits,
            generated_tokens=generated,
            temperature=0.7,
            top_k=4,
            top_p=0.9,
        )
        
        assert 0 <= chosen_idx < len(logits)
        assert len(filtered) == len(logits)

    def test_vectorized_softmax(self):
        """Test vectorized softmax with numpy."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        probs = softmax(logits)
        
        assert len(probs) == len(logits)
        assert abs(sum(probs) - 1.0) < 1e-5
        # Probabilities should be monotonically increasing with logits
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_vectorized_apply_top_k(self):
        """Test vectorized top-k filtering."""
        logits = [1.0, 5.0, 2.0, 4.0, 3.0]
        filtered = apply_top_k(logits, 3)
        
        # Check that we kept top 3 values
        non_inf_count = sum(1 for x in filtered if x > float("-inf"))
        assert non_inf_count == 3
        
        # Check that top values are kept
        assert filtered[1] == 5.0  # Highest
        assert filtered[3] == 4.0  # Second highest

    def test_vectorized_apply_top_p(self):
        """Test vectorized nucleus sampling."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        filtered = apply_top_p(logits, 0.5)
        
        # Should filter out some low-probability tokens
        inf_count = sum(1 for x in filtered if x == float("-inf"))
        assert inf_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
