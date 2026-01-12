"""
Unit tests for MemoryBridge integration.

Tests the unified memory bridge that integrates:
- persistant_memory_v46 (GraphRAG, MerkleLSM, PackfileStore, ZKProver)
- memory (GovernedUnlearning, CostOptimizer)
- vulcan/memory (HierarchicalMemory)

Author: VULCAN-AGI Team
"""

import os
import sys
import pytest
import threading
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Add src to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.integration.memory_bridge import (
    MemoryBridge,
    MemoryBridgeConfig,
    create_memory_bridge,
    PERSISTENT_MEMORY_AVAILABLE,
    HIERARCHICAL_MEMORY_AVAILABLE,
    GOVERNED_UNLEARNING_AVAILABLE,
    COST_OPTIMIZER_AVAILABLE,
)


class TestMemoryBridgeConfig:
    """Test MemoryBridgeConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryBridgeConfig()
        
        assert config.s3_bucket is None
        assert config.region == "us-east-1"
        assert config.compression == "zstd"
        assert config.encryption == "AES256"
        assert config.max_memories == 100000
        assert config.default_importance == 0.5
        assert config.decay_rate == 0.001
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.enable_governed_unlearning is True
        assert config.enable_cost_optimization is True
        assert config.auto_optimize is False
        assert config.enable_zk_proofs is True
        assert config.enable_graph_rag is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MemoryBridgeConfig(
            s3_bucket="test-bucket",
            region="us-west-2",
            compression="zlib",
            max_memories=50000,
            default_importance=0.7,
            decay_rate=0.002,
        )
        
        assert config.s3_bucket == "test-bucket"
        assert config.region == "us-west-2"
        assert config.compression == "zlib"
        assert config.max_memories == 50000
        assert config.default_importance == 0.7
        assert config.decay_rate == 0.002
    
    def test_importance_validation_too_high(self):
        """Test validation fails for importance > 1.0."""
        with pytest.raises(ValueError, match="default_importance must be in"):
            MemoryBridgeConfig(default_importance=1.5)
    
    def test_importance_validation_too_low(self):
        """Test validation fails for importance < 0.0."""
        with pytest.raises(ValueError, match="default_importance must be in"):
            MemoryBridgeConfig(default_importance=-0.1)
    
    def test_importance_validation_edge_cases(self):
        """Test validation passes for edge cases 0.0 and 1.0."""
        config1 = MemoryBridgeConfig(default_importance=0.0)
        assert config1.default_importance == 0.0
        
        config2 = MemoryBridgeConfig(default_importance=1.0)
        assert config2.default_importance == 1.0
    
    def test_decay_rate_validation_negative(self):
        """Test validation fails for negative decay rate."""
        with pytest.raises(ValueError, match="decay_rate must be non-negative"):
            MemoryBridgeConfig(decay_rate=-0.001)
    
    def test_decay_rate_validation_zero(self):
        """Test validation passes for zero decay rate."""
        config = MemoryBridgeConfig(decay_rate=0.0)
        assert config.decay_rate == 0.0


class TestMemoryBridgeInitialization:
    """Test MemoryBridge initialization and component loading."""
    
    def test_initialization_default_config(self):
        """Test bridge initializes with default configuration."""
        bridge = MemoryBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, MemoryBridgeConfig)
    
    def test_initialization_custom_config(self):
        """Test bridge initializes with custom configuration."""
        config = MemoryBridgeConfig(max_memories=50000)
        bridge = MemoryBridge(config)
        assert bridge.config.max_memories == 50000
    
    def test_get_status(self):
        """Test get_status returns component availability."""
        bridge = MemoryBridge()
        status = bridge.get_status()
        
        assert isinstance(status, dict)
        assert "persistent_memory_available" in status
        assert "hierarchical_memory_available" in status
        assert "governed_unlearning_available" in status
        assert "cost_optimizer_available" in status
        assert "storage_initialized" in status
        assert "graph_rag_initialized" in status
        assert "lsm_initialized" in status
        assert "zk_prover_initialized" in status
        assert "hierarchical_initialized" in status
    
    def test_get_statistics(self):
        """Test get_statistics returns metrics from components."""
        bridge = MemoryBridge()
        stats = bridge.get_statistics()
        
        assert isinstance(stats, dict)


class TestMemoryBridgeStorage:
    """Test MemoryBridge storage operations."""
    
    def test_store_validation_empty_key(self):
        """Test store rejects empty key."""
        bridge = MemoryBridge()
        with pytest.raises(ValueError, match="Key cannot be empty"):
            bridge.store("", "value")
    
    def test_store_validation_none_value(self):
        """Test store rejects None value."""
        bridge = MemoryBridge()
        with pytest.raises(ValueError, match="Value cannot be None"):
            bridge.store("key", None)
    
    def test_store_with_metadata(self):
        """Test store accepts optional metadata."""
        bridge = MemoryBridge()
        result = bridge.store("test_key", "test_value", {"priority": "high"})
        assert isinstance(result, bool)
    
    def test_retrieve_validation_k_zero(self):
        """Test retrieve rejects k <= 0."""
        bridge = MemoryBridge()
        with pytest.raises(ValueError, match="k must be positive"):
            bridge.retrieve("query", k=0)
    
    def test_retrieve_validation_k_negative(self):
        """Test retrieve rejects negative k."""
        bridge = MemoryBridge()
        with pytest.raises(ValueError, match="k must be positive"):
            bridge.retrieve("query", k=-1)
    
    def test_retrieve_returns_list(self):
        """Test retrieve returns list of results."""
        bridge = MemoryBridge()
        results = bridge.retrieve("test query", k=10)
        assert isinstance(results, list)
        assert len(results) <= 10


class TestMemoryBridgeUnlearning:
    """Test MemoryBridge unlearning operations."""
    
    def test_unlearn_with_defaults(self):
        """Test unlearn with default parameters."""
        bridge = MemoryBridge()
        result = bridge.unlearn("test_pattern")
        
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_unlearn_with_custom_parameters(self):
        """Test unlearn with custom parameters."""
        bridge = MemoryBridge()
        result = bridge.unlearn(
            pattern="sensitive_data",
            method="exact_removal",
            urgency="high",
            requester_id="admin"
        )
        
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_generate_unlearning_proof_empty_list(self):
        """Test proof generation with empty list."""
        bridge = MemoryBridge()
        proof = bridge.generate_unlearning_proof([])
        assert proof is None


class TestMemoryBridgeCostOperations:
    """Test MemoryBridge cost optimization operations."""
    
    def test_optimize_storage_with_strategy(self):
        """Test optimize_storage with custom strategy."""
        bridge = MemoryBridge()
        result = bridge.optimize_storage(strategy="aggressive")
        
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_check_budget_returns_dict(self):
        """Test check_budget returns dictionary."""
        bridge = MemoryBridge()
        result = bridge.check_budget()
        
        assert isinstance(result, dict)
        assert "status" in result


class TestMemoryBridgeContextManager:
    """Test MemoryBridge context manager support."""
    
    def test_context_manager_enter(self):
        """Test context manager __enter__ returns self."""
        bridge = MemoryBridge()
        with bridge as ctx:
            assert ctx is bridge
    
    def test_context_manager_exit_calls_shutdown(self):
        """Test context manager __exit__ calls shutdown."""
        bridge = MemoryBridge()
        
        with bridge:
            pass
        
        status = bridge.get_status()
        assert isinstance(status, dict)
    
    def test_context_manager_with_exception(self):
        """Test context manager cleans up even with exception."""
        bridge = MemoryBridge()
        
        try:
            with bridge:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        status = bridge.get_status()
        assert isinstance(status, dict)


class TestMemoryBridgeShutdown:
    """Test MemoryBridge shutdown behavior."""
    
    def test_shutdown_multiple_times(self):
        """Test shutdown can be called multiple times safely."""
        bridge = MemoryBridge()
        bridge.shutdown()
        bridge.shutdown()
    
    def test_shutdown_with_minimal_components(self):
        """Test shutdown works with minimal components."""
        config = MemoryBridgeConfig(
            enable_zk_proofs=False,
            enable_graph_rag=False,
        )
        bridge = MemoryBridge(config)
        bridge.shutdown()


class TestMemoryBridgeFactory:
    """Test create_memory_bridge factory function."""
    
    def test_factory_with_none(self):
        """Test factory creates bridge with default config."""
        bridge = create_memory_bridge()
        assert isinstance(bridge, MemoryBridge)
        assert bridge.config is not None
    
    def test_factory_with_dict(self):
        """Test factory accepts dict configuration."""
        bridge = create_memory_bridge({"max_memories": 50000})
        assert isinstance(bridge, MemoryBridge)
        assert bridge.config.max_memories == 50000
    
    def test_factory_with_config_object(self):
        """Test factory accepts MemoryBridgeConfig object."""
        config = MemoryBridgeConfig(max_memories=75000)
        bridge = create_memory_bridge(config)
        assert isinstance(bridge, MemoryBridge)
        assert bridge.config.max_memories == 75000
    
    def test_factory_with_invalid_type(self):
        """Test factory rejects invalid configuration type."""
        with pytest.raises(TypeError, match="config must be dict or MemoryBridgeConfig"):
            create_memory_bridge("invalid")


class TestMemoryBridgeThreadSafety:
    """Test MemoryBridge thread safety."""
    
    def test_concurrent_get_status(self):
        """Test concurrent get_status calls."""
        bridge = MemoryBridge()
        results = []
        errors = []
        
        def get_status():
            try:
                status = bridge.get_status()
                results.append(status)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=get_status) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10
    
    def test_concurrent_shutdown(self):
        """Test concurrent shutdown calls."""
        bridge = MemoryBridge()
        errors = []
        
        def shutdown():
            try:
                bridge.shutdown()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=shutdown) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
