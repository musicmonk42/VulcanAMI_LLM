"""
Comprehensive tests for graphix_vulcan_bridge.py
"""

import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest
import torch

from src.integration.graphix_vulcan_bridge import (
    BridgeConfig,
    GraphixVulcanBridge,
    HierarchicalMemory,
    WorldModelCore,
)


class TestBridgeConfig:
    """Test BridgeConfig dataclass."""

    def test_default_config(self):
        config = BridgeConfig()
        assert config.async_timeout == 2.0
        assert config.embedding_dim == 256
        assert config.memory_capacity == 100
        assert config.kl_guard_threshold == 0.05

    def test_custom_config(self):
        config = BridgeConfig(async_timeout=5.0, embedding_dim=512, memory_capacity=200)
        assert config.async_timeout == 5.0
        assert config.embedding_dim == 512
        assert config.memory_capacity == 200


class TestWorldModelCore:
    """Test WorldModelCore functionality."""

    @pytest.mark.asyncio
    async def test_update(self):
        wm = WorldModelCore()
        await wm.update("test observation")
        assert wm._state["last_obs"] == "test observation"
        assert "timestamp" in wm._state

    @pytest.mark.asyncio
    async def test_update_from_text(self):
        wm = WorldModelCore()
        tokens = ["hello", "world", "test"]
        await wm.update_from_text(tokens)
        assert wm._state["last_tokens"] == tokens
        assert "hello" in wm._concept_registry
        assert "world" in wm._concept_registry

    @pytest.mark.asyncio
    async def test_validate_generation_repetition(self):
        wm = WorldModelCore()
        context = {"prompt_tokens": ["test", "word", "another", "test", "word"]}

        # Should fail due to repetition
        result = await wm.validate_generation("word", context)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_generation_valid(self):
        wm = WorldModelCore()
        context = {"prompt_tokens": ["test", "word"]}

        result = await wm.validate_generation("newtoken", context)
        assert result is True

    @pytest.mark.asyncio
    async def test_suggest_correction(self):
        wm = WorldModelCore()
        # Add some concepts
        wm._concept_registry = {"testing": 5, "example": 3}

        correction = await wm.suggest_correction("temp", {"strategy": "test"})
        # Should suggest "testing" if it starts with 't'
        assert correction == "TESTING"

    @pytest.mark.asyncio
    async def test_intervene_before_emit(self):
        wm = WorldModelCore()
        intervention = await wm.intervene_before_emit("UPPER", {}, None)
        assert intervention is not None
        assert intervention["modified_token"] == "upper"

    def test_explain(self):
        wm = WorldModelCore()
        wm._concept_registry = {"test": 1, "example": 2}
        explanation = wm.explain("concept")
        assert "concept" in explanation
        assert "2" in explanation


class TestHierarchicalMemory:
    """Test HierarchicalMemory functionality."""

    def test_initialization(self):
        config = BridgeConfig(embedding_dim=128, vocab_size=1000)
        memory = HierarchicalMemory(config)
        assert memory.config.embedding_dim == 128
        assert len(memory.episodic) == 0

    def test_embed_text(self):
        config = BridgeConfig()
        memory = HierarchicalMemory(config)

        embedding = memory._embed_text("test text")
        assert embedding.shape == (config.embedding_dim,)
        assert torch.allclose(torch.norm(embedding), torch.tensor(1.0), atol=1e-5)

    @pytest.mark.asyncio
    async def test_astore_generation(self):
        config = BridgeConfig()
        memory = HierarchicalMemory(config)

        await memory.astore_generation("prompt text", "generated text", {})
        assert len(memory.episodic) == 1
        assert memory._embedding_tensor is not None

    @pytest.mark.asyncio
    async def test_aretrieve_context(self):
        config = BridgeConfig()
        memory = HierarchicalMemory(config)

        # Store some data
        await memory.astore_generation("test prompt", "test generation", {})
        await memory.astore_generation("another prompt", "another generation", {})

        # Retrieve
        context, cache_hit = await memory.aretrieve_context("test", top_k=2)
        assert "episodic" in context
        assert len(context["episodic"]) > 0
        assert cache_hit is False


class TestGraphixVulcanBridge:
    """Test GraphixVulcanBridge main functionality."""

    def test_initialization(self):
        bridge = GraphixVulcanBridge()
        assert bridge.config is not None
        assert bridge.memory is not None
        assert bridge.world_model is not None
        assert bridge.reasoning is not None

    @pytest.mark.asyncio
    async def test_before_execution(self):
        bridge = GraphixVulcanBridge()
        context = await bridge.before_execution("test prompt")

        assert isinstance(context, dict)
        assert "raw_input" in context
        assert "memory" in context
        assert "world_state" in context

    @pytest.mark.asyncio
    async def test_after_execution(self):
        bridge = GraphixVulcanBridge()
        result = {
            "prompt": "test prompt",
            "tokens": "generated text",
            "reasoning_trace": {},
        }

        # Should not raise
        await bridge.after_execution(result)

        # Check memory was updated
        assert len(bridge.memory.episodic) > 0


class TestPerformanceFixes:
    """
    Test performance fixes for progressive degradation issue.
    
    These tests verify that data structures are properly bounded
    and that the system does not degrade over extended operation.
    """

    @pytest.mark.asyncio
    async def test_concept_registry_bounded_growth(self):
        """Verify that concept registry doesn't grow unboundedly."""
        wm = WorldModelCore()
        
        # Generate many unique tokens to force concept registry growth
        for i in range(2000):
            tokens = [f"uniquetoken{i}abc"]  # Words > 3 chars get added
            await wm.update_from_text(tokens)
        
        # Registry should be bounded to max_concept_registry_size / 2 after trimming
        assert len(wm._concept_registry) <= wm._max_concept_registry_size
    
    @pytest.mark.asyncio
    async def test_memory_cache_bounded(self):
        """Verify that memory cache doesn't grow unboundedly."""
        config = BridgeConfig(cache_ttl_seconds=1.0)  # Short TTL for testing
        memory = HierarchicalMemory(config)
        
        # Store some data first
        await memory.astore_generation("prompt", "generated", {})
        
        # Make many queries to fill cache
        for i in range(20):
            await memory.aretrieve_context(f"query_{i}", top_k=2)
        
        # Cache should be bounded to _cache_capacity (10)
        assert len(memory._cache) <= memory._cache_capacity
    
    @pytest.mark.asyncio
    async def test_episodic_memory_bounded(self):
        """Verify that episodic memory doesn't grow unboundedly."""
        config = BridgeConfig(memory_capacity=10)  # Small capacity for testing
        memory = HierarchicalMemory(config)
        
        # Store many entries
        for i in range(50):
            await memory.astore_generation(f"prompt_{i}", f"generated_{i}", {})
        
        # Episodic memory should be bounded to memory_capacity
        assert len(memory.episodic) <= config.memory_capacity
        assert memory._embedding_count <= config.memory_capacity
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Verify that cache entries expire and are cleaned up."""
        import time
        
        config = BridgeConfig(cache_ttl_seconds=0.1)  # Very short TTL
        memory = HierarchicalMemory(config)
        memory._cache_cleanup_interval = 0.05  # Cleanup frequently
        
        # Store some data first
        await memory.astore_generation("prompt", "generated", {})
        
        # Make a query to populate cache
        _, cache_hit = await memory.aretrieve_context("test_query", top_k=2)
        assert cache_hit is False
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Trigger another retrieval which should clean up expired entries
        _, cache_hit = await memory.aretrieve_context("test_query", top_k=2)
        # Cache entry should have expired and been cleaned up
        assert cache_hit is False  # Entry expired, so no cache hit
    
    @pytest.mark.asyncio
    async def test_embedding_tensor_preallocation(self):
        """Verify that embedding tensor uses pre-allocation instead of torch.cat()."""
        config = BridgeConfig(memory_capacity=10, embedding_dim=128)
        memory = HierarchicalMemory(config)
        
        # Store first entry - should pre-allocate tensor
        await memory.astore_generation("prompt_0", "generated_0", {})
        
        # Tensor should be pre-allocated to full capacity
        assert memory._embedding_tensor is not None
        assert memory._embedding_tensor.shape[0] == config.memory_capacity
        assert memory._embedding_count == 1
        
        # Store more entries - should use in-place assignment
        for i in range(1, 15):
            await memory.astore_generation(f"prompt_{i}", f"generated_{i}", {})
        
        # Count should be capped at memory_capacity
        assert memory._embedding_count == config.memory_capacity
        # Tensor shape should remain unchanged (no fragmentation from torch.cat)
        assert memory._embedding_tensor.shape[0] == config.memory_capacity
    
    def test_bridge_cleanup(self):
        """Verify that bridge cleanup method works correctly."""
        bridge = GraphixVulcanBridge()
        
        # Get initial stats
        stats_before = bridge.get_memory_stats()
        assert isinstance(stats_before, dict)
        
        # Add some data
        bridge.world_model._concept_registry = {f"concept_{i}": i for i in range(500)}
        bridge.memory._cache = {f"query_{i}": ({}, 0.0) for i in range(20)}
        
        # Run cleanup
        bridge.cleanup(graceful=True)
        
        # Get stats after cleanup
        stats_after = bridge.get_memory_stats()
        
        # Concept registry should be trimmed
        assert stats_after["concept_registry_size"] <= 100
        # Cache should be cleared
        assert stats_after["cache_size"] == 0
    
    @pytest.mark.asyncio
    async def test_reduced_async_sleep_latency(self):
        """Verify that async sleep times are reduced for better performance."""
        import time
        
        wm = WorldModelCore()
        config = BridgeConfig()
        memory = HierarchicalMemory(config)
        
        # Store some data
        await memory.astore_generation("prompt", "generated", {})
        
        # Time multiple operations to verify they're fast
        start = time.time()
        for _ in range(10):
            await wm.update("observation")
            await wm.update_from_text(["test", "tokens"])
            await wm.validate_generation("token", {})
            await memory.aretrieve_context("query", top_k=2)
        elapsed = time.time() - start
        
        # 10 iterations with reduced sleep times should complete quickly
        # Old times (0.01s + 0.01s + 0.005s + 0.05s) * 10 = 0.75s
        # New times (0.001s + 0.001s + 0.0005s + 0.001s) * 10 = 0.035s
        # Allow some margin for actual computation
        assert elapsed < 0.5, f"Operations too slow: {elapsed:.3f}s (expected < 0.5s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
