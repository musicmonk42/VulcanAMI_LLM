<<<<<<< HEAD
"""
Comprehensive tests for graphix_vulcan_bridge.py
"""
import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from graphix_vulcan_bridge import (
    GraphixVulcanBridge,
    BridgeConfig,
    WorldModelCore,
    HierarchicalMemory,
    UnifiedReasoning,
    BridgeContext
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
        config = BridgeConfig(
            async_timeout=5.0,
            embedding_dim=512,
            memory_capacity=200
        )
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
            "reasoning_trace": {}
        }
        
        # Should not raise
        await bridge.after_execution(result)
        
        # Check memory was updated
        assert len(bridge.memory.episodic) > 0


if __name__ == "__main__":
=======
"""
Comprehensive tests for graphix_vulcan_bridge.py
"""
import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from graphix_vulcan_bridge import (
    GraphixVulcanBridge,
    BridgeConfig,
    WorldModelCore,
    HierarchicalMemory,
    UnifiedReasoning,
    BridgeContext
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
        config = BridgeConfig(
            async_timeout=5.0,
            embedding_dim=512,
            memory_capacity=200
        )
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
            "reasoning_trace": {}
        }
        
        # Should not raise
        await bridge.after_execution(result)
        
        # Check memory was updated
        assert len(bridge.memory.episodic) > 0


if __name__ == "__main__":
>>>>>>> ea7a1e4 (LLM training)
    pytest.main([__file__, "-v", "--tb=short"])