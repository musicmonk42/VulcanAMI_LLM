"""
Comprehensive tests for unlearning.py module.

Tests cover:
- GradientSurgeryUnlearner functionality
- UnlearningEngine with all methods
- Verification mechanisms
- Edge cases and error handling
- Async operations
"""

import pytest
import asyncio
import numpy as np
import hashlib
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from unlearning import (
    GradientSurgeryUnlearner,
    UnlearningEngine
)


class TestGradientSurgeryUnlearner:
    """Test suite for GradientSurgeryUnlearner class."""
    
    def test_initialization(self):
        """Test GradientSurgeryUnlearner initialization."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        assert unlearner.merkle_graph is merkle_graph
        assert unlearner.unlearning_history == []
    
    def test_unlearn_batch_basic(self):
        """Test basic unlearn_batch functionality."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        forget_set = [b"data1", b"data2", b"data3"]
        retain_set = [b"data4", b"data5", b"data6", b"data7"]
        
        result = unlearner.unlearn_batch(
            forget_set=forget_set,
            retain_set=retain_set,
            learning_rate=0.01,
            iterations=10,
            regularization=0.001
        )
        
        # Check result structure
        assert result["method"] == "gradient_surgery"
        assert result["forget_count"] == 3
        assert result["retain_count"] == 4
        assert result["iterations"] == 10
        assert "metrics" in result
        assert "elapsed_time" in result
        assert "timestamp" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "initial_forget_loss" in metrics
        assert "final_forget_loss" in metrics
        assert "forget_loss_reduction" in metrics
        assert "initial_retain_loss" in metrics
        assert "final_retain_loss" in metrics
        
        # History should be recorded
        assert len(unlearner.unlearning_history) == 1
    
    def test_unlearn_batch_empty_sets(self):
        """Test unlearn_batch with empty sets."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        result = unlearner.unlearn_batch(
            forget_set=[],
            retain_set=[],
            iterations=5
        )
        
        assert result["forget_count"] == 0
        assert result["retain_count"] == 0
        assert result["metrics"]["initial_forget_loss"] == 0.0
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        data = [b"test1", b"test2"]
        gradients = unlearner._compute_gradients(data)
        
        assert isinstance(gradients, np.ndarray)
        assert len(gradients) == 1000  # Default grad_dim
    
    def test_gradient_surgery(self):
        """Test gradient surgery operation."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        forget_grads = np.random.randn(100)
        retain_grads = np.random.randn(100)
        
        surgical_grads = unlearner._gradient_surgery(
            forget_grads,
            retain_grads,
            regularization=0.001
        )
        
        assert isinstance(surgical_grads, np.ndarray)
        assert len(surgical_grads) == 100
    
    def test_gradient_surgery_edge_cases(self):
        """Test gradient surgery with edge cases."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        # Test with zero gradients
        forget_grads = np.zeros(100)
        retain_grads = np.random.randn(100)
        
        result = unlearner._gradient_surgery(forget_grads, retain_grads, 0.001)
        assert isinstance(result, np.ndarray)
        
        # Test with both zero
        result = unlearner._gradient_surgery(np.zeros(100), np.zeros(100), 0.001)
        assert isinstance(result, np.ndarray)
    
    def test_loss_computation(self):
        """Test loss computation."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        data = [b"item1", b"item2", b"item3"]
        loss = unlearner._compute_loss(data)
        
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 1.0
    
    def test_multiple_unlearn_operations(self):
        """Test multiple unlearning operations."""
        merkle_graph = Mock()
        unlearner = GradientSurgeryUnlearner(merkle_graph)
        
        for i in range(3):
            forget_set = [f"forget{i}".encode()]
            retain_set = [f"retain{i}".encode()]
            
            unlearner.unlearn_batch(forget_set, retain_set, iterations=5)
        
        assert len(unlearner.unlearning_history) == 3


class TestUnlearningEngine:
    """Test suite for UnlearningEngine class."""
    
    def test_initialization(self):
        """Test UnlearningEngine initialization."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="gradient_surgery"
        )
        
        assert engine.merkle_graph is merkle_graph
        assert engine.method == "gradient_surgery"
        assert engine.impl is not None
        assert engine.unlearning_log == []
        assert engine.verified_removals == set()
    
    def test_unlearn_gradient_surgery(self):
        """Test unlearn with gradient surgery method."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="gradient_surgery"
        )
        
        forget = [b"item1", b"item2"]
        retain = [b"item3", b"item4", b"item5"]
        
        result = engine.unlearn(data_to_data_to_forget=forget, data_to_data_to_retain=retain)
        
        assert "method" in result
        assert "elapsed_time" in result
        assert "verification" in result
        assert len(engine.unlearning_log) == 1
    
    def test_unlearn_sisa(self):
        """Test unlearn with SISA method."""
        merkle_graph = Mock()
        merkle_graph.nodes = {}
        
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="sisa",
            shard_count=5
        )
        
        forget = [b"item1"]
        retain = [b"item2", b"item3"]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result["method"] == "sisa"
        assert "affected_shards" in result
        assert "retrained_shards" in result
    
    def test_unlearn_influence(self):
        """Test unlearn with influence functions."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="influence"
        )
        
        forget = [b"item1"]
        retain = [b"item2", b"item3"]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result["method"] == "influence"
        assert "avg_influence" in result
    
    def test_unlearn_amnesiac(self):
        """Test unlearn with amnesiac method."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="amnesiac"
        )
        
        forget = [b"item1"]
        retain = [b"item2"]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result["method"] == "amnesiac"
        assert "noise_magnitude" in result
    
    def test_unlearn_certified(self):
        """Test unlearn with certified removal."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="certified"
        )
        
        forget = [b"item1", b"item2"]
        retain = [b"item3"]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result["method"] == "certified"
        assert "certificate" in result
        assert "epsilon" in result
    
    def test_unlearn_pattern(self):
        """Test pattern-based unlearning."""
        merkle_graph = Mock()
        merkle_graph.nodes = {
            "node1": Mock(pack_ids=["pack1", "pack2"]),
            "node2": Mock(pack_ids=["pack3"])
        }
        
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        result = engine.unlearn_pattern(pattern="sensitive_*")
        
        assert "pattern" in result
        assert "affected_packfiles" in result
    
    def test_verification(self):
        """Test verification mechanism."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            enable_verification=True
        )
        
        forget = [b"item1"]
        retain = [b"item2", b"item3"]
        
        verification = engine._verify_unlearning(forget, retain)
        
        assert "passed" in verification
        assert "avg_forget_score" in verification
        assert "avg_retain_score" in verification
    
    def test_get_affected_shards(self):
        """Test shard determination."""
        merkle_graph = Mock()
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            shard_count=10
        )
        
        forget = [b"item1", b"item2", b"item3"]
        affected = engine._get_affected_shards(forget)
        
        assert isinstance(affected, list)
        assert all(0 <= shard < 10 for shard in affected)
    
    def test_compute_influence(self):
        """Test influence computation."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        item = b"test_item"
        retain = [b"retain1", b"retain2"]
        
        influence = engine._compute_influence(item, retain)
        
        assert isinstance(influence, float)
        assert 0.0 <= influence <= 1.0
    
    def test_generate_removal_certificate(self):
        """Test certificate generation."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        forget = [b"item1", b"item2", b"item3"]
        certificate = engine._generate_removal_certificate(forget)
        
        assert isinstance(certificate, str)
        assert len(certificate) > 0
    
    def test_verify_certified_removal(self):
        """Test certificate verification."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        items = [b"item1", b"item2"]
        certificate = engine._generate_removal_certificate(items)
        
        # Valid certificate
        assert engine.verify_certified_removal(certificate, items) is True
        
        # Invalid certificate
        assert engine.verify_certified_removal("invalid", items) is False
    
    def test_measure_memorization(self):
        """Test memorization measurement."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        item = b"test_item"
        score = engine._measure_memorization(item)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_to_bytes_conversion(self):
        """Test item to bytes conversion."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        # Test bytes
        assert engine._to_bytes(b"test") == b"test"
        
        # Test string
        assert engine._to_bytes("test") == b"test"
        
        # Test other types
        assert engine._to_bytes(123) == b"123"
    
    @pytest.mark.asyncio
    async def test_unlearn_async(self):
        """Test async unlearning."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        forget = [b"item1"]
        retain = [b"item2"]
        
        result = await engine.unlearn_async(data_to_forget=forget, data_to_retain=retain)
        
        assert "method" in result
        assert "elapsed_time" in result
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        # Perform some operations
        engine.unlearn(data_to_forget=[b"item1"], data_to_retain=[b"item2"])
        
        stats = engine.get_statistics()
        
        assert "method" in stats
        assert "total_unlearning_operations" in stats
        assert "verified_removals" in stats
        assert "shard_count" in stats
        assert stats["total_unlearning_operations"] == 1
    
    def test_export_log(self, tmp_path):
        """Test log export."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        # Perform operation
        engine.unlearn(data_to_forget=[b"item1"], data_to_retain=[b"item2"])
        
        # Export log
        log_file = tmp_path / "unlearning_log.json"
        engine.export_log(str(log_file))
        
        assert log_file.exists()
    
    def test_close(self):
        """Test resource cleanup."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        # Should not raise
        engine.close()
    
    def test_empty_forget_set(self):
        """Test unlearning with empty forget set."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        result = engine.unlearn(data_to_forget=[], data_to_retain=[b"item1"])
        
        assert result is not None
    
    def test_empty_retain_set(self):
        """Test unlearning with empty retain set."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        result = engine.unlearn(data_to_forget=[b"item1"], data_to_retain=[])
        
        assert result is not None
    
    def test_both_empty_sets(self):
        """Test unlearning with both sets empty."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        result = engine.unlearn(data_to_forget=[], data_to_retain=[])
        
        assert result is not None


class TestUnlearningIntegration:
    """Integration tests for unlearning module."""
    
    def test_full_unlearning_workflow(self):
        """Test complete unlearning workflow."""
        merkle_graph = Mock()
        merkle_graph.nodes = {}
        
        engine = UnlearningEngine(
            merkle_graph=merkle_graph,
            method="gradient_surgery",
            enable_verification=True
        )
        
        # Create data
        forget_data = [f"sensitive_{i}".encode() for i in range(10)]
        retain_data = [f"normal_{i}".encode() for i in range(20)]
        
        # Perform unlearning
        result = engine.unlearn(
            data_to_forget=forget_data,
            data_to_retain=retain_data
        )
        
        # Verify results
        assert result["method"] == "gradient_surgery"
        assert result["verification"]["passed"] in [True, False]
        assert len(engine.unlearning_log) == 1
        
        # Get statistics
        stats = engine.get_statistics()
        assert stats["total_unlearning_operations"] == 1
    
    def test_multiple_methods_comparison(self):
        """Test different unlearning methods."""
        merkle_graph = Mock()
        merkle_graph.nodes = {}
        
        methods = ["gradient_surgery", "sisa", "influence", "amnesiac", "certified"]
        results = {}
        
        for method in methods:
            engine = UnlearningEngine(
                merkle_graph=merkle_graph,
                method=method
            )
            
            result = engine.unlearn(
                data_to_forget=[b"item1"],
                data_to_retain=[b"item2", b"item3"]
            )
            
            results[method] = result
        
        # All methods should return results
        assert len(results) == len(methods)
        for method in methods:
            assert results[method]["method"] == method


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_dataset(self):
        """Test with large dataset."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        # Create large dataset
        forget = [f"forget_{i}".encode() for i in range(1000)]
        retain = [f"retain_{i}".encode() for i in range(5000)]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result is not None
        assert result["forget_count"] == 1000
        assert result["retain_count"] == 5000
    
    def test_duplicate_items(self):
        """Test with duplicate items."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        forget = [b"item1", b"item1", b"item2"]
        retain = [b"item3", b"item3", b"item3"]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result is not None
    
    def test_unicode_data(self):
        """Test with unicode data."""
        merkle_graph = Mock()
        engine = UnlearningEngine(merkle_graph=merkle_graph)
        
        forget = ["你好".encode(), "مرحبا".encode()]
        retain = ["привет".encode(), "שלום".encode()]
        
        result = engine.unlearn(data_to_forget=forget, data_to_retain=retain)
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])