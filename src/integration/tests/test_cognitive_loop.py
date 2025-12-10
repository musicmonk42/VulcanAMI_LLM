"""
Tests for cognitive_loop.py
"""

import sys

import pytest

# Import from current directory instead of uploads
sys.path.insert(0, "/mnt/user-data/outputs")

from cognitive_loop import (CognitiveLoop, CognitiveLoopResult,
                            LoopRuntimeConfig, LoopSamplingConfig, apply_top_k,
                            apply_top_p, choose_token, penalize_repetition,
                            softmax)


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
        import torch

        batch_size = 1 if isinstance(tokens, list) else len(tokens)
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
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "final_token" in result
        assert result["final_token"] == "test"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
