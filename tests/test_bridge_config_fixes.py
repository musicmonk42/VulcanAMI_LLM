"""
Test fixes for BridgeConfig validation and timeout handling.
Tests the improvements made during the deep code audit.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock


def test_bridge_config_validation_positive():
    """Test that valid configurations pass validation."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    # Should not raise any errors
    config = BridgeConfig(
        async_timeout=2.0,
        embedding_dim=256,
        memory_capacity=100,
        kl_guard_threshold=0.05,
        max_retries=3,
        vocab_size=5000,
        cache_ttl_seconds=60.0,
        consensus_timeout_seconds=2.0
    )
    
    assert config.async_timeout == 2.0
    assert config.embedding_dim == 256


def test_bridge_config_validation_zero_allowed():
    """Test that fields that allow zero work correctly."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    # kl_guard_threshold can be zero
    config = BridgeConfig(kl_guard_threshold=0.0)
    assert config.kl_guard_threshold == 0.0
    
    # max_retries can be zero
    config = BridgeConfig(max_retries=0)
    assert config.max_retries == 0


def test_bridge_config_validation_negative_timeout():
    """Test that negative timeout raises ValueError."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    with pytest.raises(ValueError, match="async_timeout must be non-negative"):
        BridgeConfig(async_timeout=-1.0)


def test_bridge_config_validation_zero_timeout():
    """Test that zero timeout raises ValueError (must be positive)."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    # Zero is non-negative but not positive, so it should raise
    with pytest.raises(ValueError, match="async_timeout must be positive"):
        BridgeConfig(async_timeout=0.0)


def test_bridge_config_validation_negative_embedding_dim():
    """Test that negative embedding_dim raises ValueError."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    with pytest.raises(ValueError, match="embedding_dim must be non-negative"):
        BridgeConfig(embedding_dim=-100)


def test_bridge_config_validation_negative_kl_threshold():
    """Test that negative kl_guard_threshold raises ValueError."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    with pytest.raises(ValueError, match="kl_guard_threshold must be non-negative"):
        BridgeConfig(kl_guard_threshold=-0.1)


@pytest.mark.asyncio
async def test_safe_call_async_uses_custom_timeout():
    """Test that custom timeout parameter is respected."""
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge, BridgeConfig
    
    # Create bridge with default timeout of 2.0 seconds
    config = BridgeConfig(async_timeout=2.0)
    bridge = GraphixVulcanBridge(config=config)
    
    # Create a mock that takes longer than custom timeout
    slow_mock = AsyncMock()
    
    async def slow_func():
        await asyncio.sleep(0.5)  # 0.5 seconds
        return "completed"
    
    slow_mock.side_effect = slow_func
    
    # Call with shorter custom timeout (should timeout)
    result = await bridge._safe_call_async(
        slow_mock,
        args=(),
        default="default_value",
        timeout=0.1,  # 0.1 second timeout
        max_retries=0  # No retries
    )
    
    # Should return default due to timeout
    assert result == "default_value"


@pytest.mark.asyncio
async def test_safe_call_async_uses_config_default_timeout():
    """Test that config default timeout is used when no custom timeout provided."""
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge, BridgeConfig
    
    # Create bridge with very short default timeout
    config = BridgeConfig(async_timeout=0.1)
    bridge = GraphixVulcanBridge(config=config)
    
    # Create a mock that takes longer than config timeout
    slow_mock = AsyncMock()
    
    async def slow_func():
        await asyncio.sleep(0.5)
        return "completed"
    
    slow_mock.side_effect = slow_func
    
    # Call without custom timeout (should use config default)
    result = await bridge._safe_call_async(
        slow_mock,
        args=(),
        default="default_value",
        max_retries=0  # No retries
    )
    
    # Should return default due to timeout
    assert result == "default_value"


@pytest.mark.asyncio
async def test_safe_call_async_respects_custom_max_retries():
    """Test that custom max_retries parameter is respected."""
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge, BridgeConfig
    
    config = BridgeConfig(max_retries=5)  # Default is 5
    bridge = GraphixVulcanBridge(config=config)
    
    # Create a mock that always times out
    failing_mock = AsyncMock()
    call_count = 0
    
    async def failing_func():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(10)  # Long sleep to trigger timeout
        return "completed"
    
    failing_mock.side_effect = failing_func
    
    # Call with custom max_retries=1 (should try 2 times total: initial + 1 retry)
    result = await bridge._safe_call_async(
        failing_mock,
        args=(),
        default="default_value",
        timeout=0.05,  # Very short timeout
        max_retries=1  # Only 1 retry
    )
    
    # Should have tried twice (initial + 1 retry)
    assert call_count == 2
    assert result == "default_value"


@pytest.mark.asyncio
async def test_consensus_timeout_uses_config_value():
    """Test that consensus_approve_token uses configured consensus timeout."""
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge, BridgeConfig
    
    # Create bridge with custom consensus timeout
    config = BridgeConfig(consensus_timeout_seconds=5.0)
    bridge = GraphixVulcanBridge(config=config)
    
    # Create a mock consensus engine
    mock_consensus = MagicMock()
    mock_consensus.approve = AsyncMock(return_value=True)
    bridge.set_consensus_engine(mock_consensus)
    
    # Call consensus_approve_token
    result = await bridge.consensus_approve_token("test_token", position=0)
    
    # Verify it was called
    assert result is True
    mock_consensus.approve.assert_called_once()


def test_validation_refactoring_maintains_behavior():
    """Test that refactored validation maintains original behavior."""
    from src.integration.graphix_vulcan_bridge import BridgeConfig
    
    # All positive values should work
    config = BridgeConfig(
        async_timeout=1.0,
        embedding_dim=128,
        memory_capacity=50,
        kl_guard_threshold=0.1,
        max_retries=2,
        vocab_size=1000,
        cache_ttl_seconds=30.0,
        consensus_timeout_seconds=3.0
    )
    assert config is not None
    
    # Each field should validate independently
    with pytest.raises(ValueError):
        BridgeConfig(async_timeout=-1.0)
    
    with pytest.raises(ValueError):
        BridgeConfig(embedding_dim=0)
    
    with pytest.raises(ValueError):
        BridgeConfig(memory_capacity=-5)
    
    with pytest.raises(ValueError):
        BridgeConfig(vocab_size=0)
    
    with pytest.raises(ValueError):
        BridgeConfig(cache_ttl_seconds=-10.0)
    
    with pytest.raises(ValueError):
        BridgeConfig(consensus_timeout_seconds=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
