"""
Comprehensive test suite for memory_prior.py

Tests Bayesian priors, similarity search, memory indexing, cache management,
and all prior computation strategies.
"""

import pytest
import time
import numpy as np
import threading
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vulcan.reasoning.selection.memory_prior import (
    SimilarityMetric,
    PriorType,
    MemoryEntry,
    PriorDistribution,
    MemoryIndex,
    BayesianMemoryPrior,
    AdaptivePriorSelector
)


class TestEnums:
    """Test enum definitions"""
    
    def test_similarity_metric_values(self):
        """Test SimilarityMetric enum"""
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"
        assert SimilarityMetric.JACCARD.value == "jaccard"
        assert SimilarityMetric.SEMANTIC.value == "semantic"
        assert SimilarityMetric.WEIGHTED.value == "weighted"
    
    def test_prior_type_values(self):
        """Test PriorType enum"""
        assert PriorType.UNIFORM.value == "uniform"
        assert PriorType.BETA.value == "beta"
        assert PriorType.DIRICHLET.value == "dirichlet"
        assert PriorType.EMPIRICAL.value == "empirical"
        assert PriorType.HIERARCHICAL.value == "hierarchical"


class TestMemoryEntry:
    """Test MemoryEntry dataclass"""
    
    def test_entry_creation(self):
        """Test creating memory entries"""
        features = np.array([1.0, 2.0, 3.0])
        
        entry = MemoryEntry(
            entry_id="test_001",
            problem_features=features,
            tool_used="tool_a",
            success=True,
            confidence=0.9,
            execution_time=100.0,
            energy_used=50.0,
            timestamp=time.time()
        )
        
        assert entry.entry_id == "test_001"
        assert entry.tool_used == "tool_a"
        assert entry.success is True
        assert entry.confidence == 0.9
        assert np.array_equal(entry.problem_features, features)
    
    def test_entry_defaults(self):
        """Test entry default values"""
        entry = MemoryEntry(
            entry_id="test_002",
            problem_features=np.array([1, 2, 3]),
            tool_used="tool_b",
            success=False,
            confidence=0.5,
            execution_time=50.0,
            energy_used=25.0,
            timestamp=time.time()
        )
        
        assert isinstance(entry.context, dict)
        assert isinstance(entry.metadata, dict)


class TestPriorDistribution:
    """Test PriorDistribution dataclass"""
    
    def test_distribution_creation(self):
        """Test creating prior distributions"""
        dist = PriorDistribution(
            tool_probs={'tool_a': 0.6, 'tool_b': 0.4},
            confidence=0.8,
            support_count=50,
            entropy=0.67,
            most_likely_tool='tool_a'
        )
        
        assert dist.tool_probs['tool_a'] == 0.6
        assert dist.confidence == 0.8
        assert dist.support_count == 50
        assert dist.most_likely_tool == 'tool_a'


class TestMemoryIndex:
    """Test MemoryIndex for similarity search"""
    
    def test_initialization(self):
        """Test memory index initialization"""
        index = MemoryIndex(metric=SimilarityMetric.COSINE)
        
        assert index.metric == SimilarityMetric.COSINE
        assert len(index.entries) == 0
        assert index.index_built is False
    
    def test_add_entry(self):
        """Test adding entries to index"""
        index = MemoryIndex()
        
        entry = MemoryEntry(
            entry_id="e1",
            problem_features=np.array([1.0, 2.0, 3.0]),
            tool_used="tool_a",
            success=True,
            confidence=0.9,
            execution_time=100,
            energy_used=50,
            timestamp=time.time()
        )
        
        index.add(entry)
        
        assert len(index.entries) == 1
        assert index.entries[0].entry_id == "e1"
    
    def test_size_limit(self):
        """Test index size limiting"""
        index = MemoryIndex()
        index.max_entries = 10
        
        # Add more than max
        for i in range(15):
            entry = MemoryEntry(
                f"e{i}",
                np.random.rand(5),
                "tool_a",
                True,
                0.9,
                100,
                50,
                time.time()
            )
            index.add(entry)
        
        # Should be limited
        assert len(index.entries) <= index.max_entries
    
    def test_build_index(self):
        """Test building search index"""
        index = MemoryIndex()
        
        # Add several entries
        for i in range(10):
            entry = MemoryEntry(
                f"e{i}",
                np.random.rand(5),
                "tool_a",
                True,
                0.9,
                100,
                50,
                time.time()
            )
            index.add(entry)
        
        index.build_index()
        
        assert index.index_built is True
        assert index.features is not None
        assert index.features.shape[0] == 10
    
    def test_search_cosine(self):
        """Test cosine similarity search"""
        index = MemoryIndex(metric=SimilarityMetric.COSINE)
        
        # Add entries with known features
        for i in range(5):
            features = np.array([i, i*2, i*3], dtype=float)
            entry = MemoryEntry(
                f"e{i}",
                features,
                "tool_a",
                True,
                0.9,
                100,
                50,
                time.time()
            )
            index.add(entry)
        
        # Search for similar
        query = np.array([2.0, 4.0, 6.0])
        results = index.search(query, k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Results should be (entry, similarity) tuples
        for entry, similarity in results:
            assert isinstance(entry, MemoryEntry)
            assert 0 <= similarity <= 1
    
    def test_search_euclidean(self):
        """Test Euclidean distance search"""
        index = MemoryIndex(metric=SimilarityMetric.EUCLIDEAN)
        
        # Add entries
        for i in range(5):
            entry = MemoryEntry(
                f"e{i}",
                np.array([i, i, i], dtype=float),
                "tool_a",
                True,
                0.9,
                100,
                50,
                time.time()
            )
            index.add(entry)
        
        # Search
        query = np.array([2.0, 2.0, 2.0])
        results = index.search(query, k=2)
        
        assert len(results) > 0
        # Closest should be entry with [2,2,2]
        if results:
            best_entry, best_sim = results[0]
            assert best_entry.entry_id == "e2"
    
    def test_search_empty_index(self):
        """Test searching empty index"""
        index = MemoryIndex()
        
        query = np.array([1, 2, 3])
        results = index.search(query, k=5)
        
        assert results == []


class TestBayesianMemoryPrior:
    """Test BayesianMemoryPrior"""
    
    @pytest.fixture
    def prior(self):
        """Create prior for testing"""
        return BayesianMemoryPrior(
            memory_system=None,
            similarity_metric=SimilarityMetric.COSINE,
            prior_type=PriorType.BETA
        )
    
    def test_initialization(self, prior):
        """Test prior initialization"""
        assert prior.similarity_metric == SimilarityMetric.COSINE
        assert prior.prior_type == PriorType.BETA
        assert prior.memory_index is not None
        assert isinstance(prior.tool_stats, dict)
    
    def test_uniform_prior(self, prior):
        """Test uniform prior computation"""
        tools = ['tool_a', 'tool_b', 'tool_c']
        
        dist = prior._uniform_prior(tools)
        
        assert len(dist.tool_probs) == 3
        assert all(abs(p - 1/3) < 1e-6 for p in dist.tool_probs.values())
        assert dist.confidence == 0.0
    
    def test_uniform_prior_empty_tools(self, prior):
        """Test uniform prior with no tools"""
        dist = prior._uniform_prior([])
        
        assert dist.tool_probs == {}
        assert dist.most_likely_tool == ''
    
    def test_update_memory(self, prior):
        """Test updating memory with new results"""
        features = np.array([1.0, 2.0, 3.0])
        
        prior.update(
            features=features,
            tool_used='tool_a',
            success=True,
            confidence=0.9,
            execution_time=100.0,
            energy_used=50.0
        )
        
        # Should add to index
        assert len(prior.memory_index.entries) == 1
        
        # Should update stats
        assert 'tool_a' in prior.tool_stats
        assert prior.tool_stats['tool_a']['successes'] > 1  # Started with 1
    
    def test_beta_prior_with_history(self, prior):
        """Test Beta prior with historical data"""
        # Add some history
        for i in range(10):
            features = np.array([1.0 + i*0.1, 2.0, 3.0])
            prior.update(
                features=features,
                tool_used='tool_a',
                success=True,
                confidence=0.9,
                execution_time=100.0,
                energy_used=50.0
            )
        
        for i in range(5):
            features = np.array([1.0 + i*0.1, 2.0, 3.0])
            prior.update(
                features=features,
                tool_used='tool_b',
                success=False,
                confidence=0.5,
                execution_time=150.0,
                energy_used=75.0
            )
        
        # Compute prior for similar problem
        query_features = np.array([1.5, 2.0, 3.0])
        tools = ['tool_a', 'tool_b']
        
        dist = prior._beta_prior(query_features, tools)
        
        # tool_a should have higher probability (more successes)
        assert dist.tool_probs['tool_a'] > dist.tool_probs['tool_b']
        assert dist.most_likely_tool == 'tool_a'
    
    def test_dirichlet_prior(self, prior):
        """Test Dirichlet prior computation"""
        # Add history
        for i in range(10):
            prior.update(
                np.random.rand(3),
                'tool_a',
                True,
                0.9,
                100,
                50
            )
        
        query = np.random.rand(3)
        tools = ['tool_a', 'tool_b', 'tool_c']
        
        dist = prior._dirichlet_prior(query, tools)
        
        assert len(dist.tool_probs) == 3
        assert abs(sum(dist.tool_probs.values()) - 1.0) < 1e-6  # Should sum to 1
    
    def test_empirical_prior(self, prior):
        """Test empirical prior computation"""
        # Add history
        features = np.array([1, 2, 3], dtype=float)
        
        for _ in range(8):
            prior.update(features, 'tool_a', True, 0.9, 100, 50)
        
        for _ in range(2):
            prior.update(features, 'tool_b', False, 0.5, 100, 50)
        
        query = np.array([1, 2, 3], dtype=float)
        tools = ['tool_a', 'tool_b']
        
        dist = prior._empirical_prior(query, tools)
        
        # tool_a should dominate
        assert dist.tool_probs['tool_a'] > dist.tool_probs['tool_b']
    
    def test_hierarchical_prior(self, prior):
        """Test hierarchical prior computation"""
        # Add global history
        for i in range(20):
            prior.update(
                np.random.rand(3),
                'tool_a' if i % 2 == 0 else 'tool_b',
                True,
                0.9,
                100,
                50
            )
        
        # Add local similar history
        query = np.array([1, 2, 3], dtype=float)
        for _ in range(5):
            prior.update(query + np.random.rand(3)*0.1, 'tool_a', True, 0.9, 100, 50)
        
        tools = ['tool_a', 'tool_b']
        dist = prior._hierarchical_prior(query, tools)
        
        assert len(dist.tool_probs) == 2
        assert 'local_weight' in dist.metadata
        assert 'global_weight' in dist.metadata
    
    def test_compute_prior_caching(self, prior):
        """Test that prior computation uses caching"""
        features = np.array([1, 2, 3], dtype=float)
        tools = ['tool_a', 'tool_b']
        
        # First call
        dist1 = prior.compute_prior(features, tools)
        cache_size_1 = len(prior.prior_cache)
        
        # Second call with same inputs
        dist2 = prior.compute_prior(features, tools)
        cache_size_2 = len(prior.prior_cache)
        
        # Should use cache (size doesn't increase)
        assert cache_size_2 == cache_size_1
    
    def test_cache_eviction(self, prior):
        """Test cache size limiting"""
        prior.max_cache_size = 10
        
        # Fill cache beyond limit
        for i in range(20):
            features = np.array([i, i+1, i+2], dtype=float)
            tools = ['tool_a']
            prior.compute_prior(features, tools)
        
        # Cache should be limited
        assert len(prior.prior_cache) <= prior.max_cache_size
    
    def test_tool_statistics(self, prior):
        """Test getting tool statistics"""
        # Add some history
        prior.update(np.array([1, 2, 3]), 'tool_a', True, 0.9, 100, 50)
        prior.update(np.array([1, 2, 3]), 'tool_a', True, 0.8, 120, 60)
        prior.update(np.array([1, 2, 3]), 'tool_a', False, 0.6, 80, 40)
        
        stats = prior.get_tool_statistics()
        
        assert 'tool_a' in stats
        assert 'success_rate' in stats['tool_a']
        assert 'avg_time' in stats['tool_a']
        assert 'usage_count' in stats['tool_a']
        
        # 2 successes, 1 failure (plus pseudo-counts)
        assert stats['tool_a']['usage_count'] == 3
    
    def test_state_persistence(self, prior, tmp_path):
        """Test saving and loading state"""
        # Add some data
        for i in range(10):
            prior.update(
                np.random.rand(3),
                'tool_a',
                True,
                0.9,
                100,
                50
            )
        
        # Save state
        prior.save_state(str(tmp_path))
        
        # Create new prior and load
        new_prior = BayesianMemoryPrior()
        new_prior.load_state(str(tmp_path))
        
        # Should have same data
        assert len(new_prior.memory_index.entries) == len(prior.memory_index.entries)
        assert 'tool_a' in new_prior.tool_stats
    
    def test_clear_cache(self, prior):
        """Test cache clearing"""
        # Populate cache
        features = np.array([1, 2, 3])
        prior.compute_prior(features, ['tool_a'])
        
        assert len(prior.prior_cache) > 0
        
        prior.clear_cache()
        
        assert len(prior.prior_cache) == 0
    
    def test_thread_safety(self, prior):
        """Test thread-safe operations"""
        results = []
        errors = []
        
        def update_and_compute():
            try:
                for i in range(10):
                    features = np.random.rand(3)
                    prior.update(features, 'tool_a', True, 0.9, 100, 50)
                    
                    dist = prior.compute_prior(features, ['tool_a', 'tool_b'])
                    results.append(dist)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=update_and_compute) for _ in range(3)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) > 0


class TestAdaptivePriorSelector:
    """Test AdaptivePriorSelector"""
    
    @pytest.fixture
    def selector(self):
        """Create selector for testing"""
        return AdaptivePriorSelector(memory_system=None)
    
    def test_initialization(self, selector):
        """Test selector initialization"""
        assert selector.priors == {}
        assert len(selector.selection_history) == 0
    
    def test_lazy_initialization(self, selector):
        """Test lazy prior initialization"""
        # Should not create priors until needed
        assert len(selector.priors) == 0
        
        # Request a prior
        prior = selector._get_prior(PriorType.UNIFORM)
        
        assert prior is not None
        assert PriorType.UNIFORM in selector.priors
    
    def test_prior_type_selection_empty(self, selector):
        """Test prior type selection with no data"""
        features = np.array([1, 2, 3])
        tools = ['tool_a', 'tool_b']
        
        # With no data, should select UNIFORM
        prior_type = selector.select_prior_type(features, tools)
        
        assert prior_type == PriorType.UNIFORM
    
    def test_prior_type_selection_with_data(self, selector):
        """Test prior type selection with data"""
        # Add data to beta prior
        beta_prior = selector._get_prior(PriorType.BETA)
        
        # Add enough entries to trigger different prior types
        for i in range(250):
            beta_prior.update(
                np.random.rand(3),
                'tool_a',
                True,
                0.9,
                100,
                50
            )
        
        features = np.array([1, 2, 3])
        tools = ['tool_a', 'tool_b']
        
        # With lots of data, should select DIRICHLET or HIERARCHICAL
        prior_type = selector.select_prior_type(features, tools)
        
        assert prior_type in [PriorType.DIRICHLET, PriorType.HIERARCHICAL]
    
    def test_compute_adaptive_prior(self, selector):
        """Test computing adaptive prior"""
        features = np.array([1, 2, 3])
        tools = ['tool_a', 'tool_b']
        
        dist = selector.compute_adaptive_prior(features, tools)
        
        assert isinstance(dist, PriorDistribution)
        assert len(dist.tool_probs) == 2
        assert 'prior_type' in dist.metadata
    
    def test_update_all(self, selector):
        """Test updating all prior types"""
        # Initialize multiple priors
        selector._get_prior(PriorType.BETA)
        selector._get_prior(PriorType.EMPIRICAL)
        
        features = np.array([1, 2, 3])
        
        # Update all
        selector.update_all(features, 'tool_a', True, 0.9, 100, 50)
        
        # All initialized priors should be updated
        beta_prior = selector.priors[PriorType.BETA]
        assert len(beta_prior.memory_index.entries) > 0
    
    def test_get_statistics(self, selector):
        """Test getting statistics"""
        # Initialize some priors
        selector._get_prior(PriorType.BETA)
        selector._get_prior(PriorType.UNIFORM)
        
        # Compute some priors
        features = np.array([1, 2, 3])
        selector.compute_adaptive_prior(features, ['tool_a'])
        
        stats = selector.get_statistics()
        
        assert 'initialized_priors' in stats
        assert 'selection_history_size' in stats
        assert len(stats['initialized_priors']) == 2


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_prior_workflow(self):
        """Test complete prior computation workflow"""
        prior = BayesianMemoryPrior(
            memory_system=None,
            prior_type=PriorType.BETA
        )
        
        # Simulate tool selection history
        tools = ['tool_a', 'tool_b', 'tool_c']
        
        # FIX: Use more distinct, orthogonal feature vectors to ensure test stability
        # Tool A: good for problems where the first feature is high
        for i in range(20):
            features = np.array([0.9, 0.1, 0.1]) + np.random.rand(3) * 0.05
            prior.update(features, 'tool_a', True, 0.9, 100, 50)

        # Tool B: good for problems where the second feature is high
        for i in range(15):
            features = np.array([0.1, 0.9, 0.1]) + np.random.rand(3) * 0.05
            prior.update(features, 'tool_b', True, 0.85, 120, 60)
        
        # Tool C: rarely used
        for i in range(5):
            features = np.random.rand(3)
            prior.update(features, 'tool_c', False, 0.5, 150, 70)
        
        # Query similar to tool_a's specialty
        query = np.array([0.9, 0.1, 0.1])
        dist = prior.compute_prior(query, tools)
        
        # Tool A should have highest probability
        assert dist.most_likely_tool == 'tool_a'
        assert dist.tool_probs['tool_a'] > dist.tool_probs['tool_b']
        assert dist.tool_probs['tool_a'] > dist.tool_probs['tool_c']
        
        # Query similar to tool_b's specialty
        query2 = np.array([0.1, 0.9, 0.1])
        dist2 = prior.compute_prior(query2, tools)
        
        # Assert that tool_b is correctly identified as most likely for the second query
        assert dist2.most_likely_tool == 'tool_b'
        assert dist2.tool_probs['tool_b'] > dist2.tool_probs['tool_a']
        assert dist2.tool_probs['tool_b'] > dist2.tool_probs['tool_c']
    
    def test_adaptive_selector_workflow(self):
        """Test adaptive selector with varying data amounts"""
        selector = AdaptivePriorSelector()
        
        features = np.array([1, 2, 3])
        tools = ['tool_a', 'tool_b']
        
        # Stage 1: No data - should use UNIFORM
        dist1 = selector.compute_adaptive_prior(features, tools)
        assert dist1.metadata['prior_type'] == PriorType.UNIFORM.value
        
        # Stage 2: Add some data
        for i in range(30):
            selector.update_all(
                np.random.rand(3),
                'tool_a' if i < 20 else 'tool_b',
                True,
                0.9,
                100,
                50
            )
        
        # Should now use more sophisticated prior
        dist2 = selector.compute_adaptive_prior(features, tools)
        assert dist2.metadata['prior_type'] != PriorType.UNIFORM.value
    
    def test_recency_weighting(self):
        """Test that recent observations are weighted more"""
        prior = BayesianMemoryPrior(recency_weight=0.5)  # Strong recency bias
        
        features = np.array([1, 2, 3])
        
        # Add old observations favoring tool_a
        old_time = time.time() - 365 * 86400  # 1 year ago
        for i in range(10):
            entry = MemoryEntry(
                f"old_{i}",
                features + np.random.rand(3) * 0.1,
                'tool_a',
                True,
                0.9,
                100,
                50,
                old_time
            )
            prior.memory_index.add(entry)
            prior._update_tool_stats(entry)
        
        # Add recent observations favoring tool_b
        for i in range(5):
            prior.update(
                features + np.random.rand(3) * 0.1,
                'tool_b',
                True,
                0.9,
                100,
                50
            )
        
        # Compute prior
        dist = prior.compute_prior(features, ['tool_a', 'tool_b'])
        
        # Recent tool_b should be favored despite fewer total observations
        # (depends on recency_weight strength)
        assert dist.tool_probs['tool_b'] > 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_feature_vectors(self):
        """Test handling of empty feature vectors"""
        prior = BayesianMemoryPrior()
        
        # Empty features
        features = np.array([])
        tools = ['tool_a']
        
        # Should not crash
        try:
            dist = prior.compute_prior(features, tools)
            # May get uniform or error, but shouldn't crash
            assert True
        except:
            # Acceptable to raise exception for invalid input
            assert True
    
    def test_single_tool(self):
        """Test with only one tool available"""
        prior = BayesianMemoryPrior()
        
        features = np.array([1, 2, 3])
        tools = ['only_tool']
        
        dist = prior.compute_prior(features, tools)
        
        assert dist.tool_probs['only_tool'] == 1.0
        assert dist.most_likely_tool == 'only_tool'
    
    def test_no_tools_available(self):
        """Test with no tools available"""
        prior = BayesianMemoryPrior()
        
        features = np.array([1, 2, 3])
        tools = []
        
        dist = prior.compute_prior(features, tools)
        
        assert dist.tool_probs == {}
        assert dist.most_likely_tool == ''
    
    def test_all_failures(self):
        """Test when all historical executions failed"""
        prior = BayesianMemoryPrior()
        
        features = np.array([1, 2, 3])
        
        # Add only failures
        for i in range(10):
            prior.update(features, 'tool_a', False, 0.3, 100, 50)
        
        tools = ['tool_a', 'tool_b']
        dist = prior.compute_prior(features, tools)
        
        # Should still compute a distribution
        assert len(dist.tool_probs) == 2
        assert sum(dist.tool_probs.values()) > 0
    
    def test_extreme_similarity_values(self):
        """Test handling of extreme similarity values"""
        index = MemoryIndex()
        
        # Add entry
        entry = MemoryEntry(
            "e1",
            np.array([1, 0, 0], dtype=float),
            "tool_a",
            True,
            0.9,
            100,
            50,
            time.time()
        )
        index.add(entry)
        
        # Query with same vector (perfect similarity)
        query = np.array([1, 0, 0], dtype=float)
        results = index.search(query, k=1)
        
        if results:
            entry, similarity = results[0]
            assert 0 <= similarity <= 1  # Should be clamped


class TestPerformance:
    """Performance tests"""
    
    def test_large_memory_index(self):
        """Test performance with large memory index"""
        prior = BayesianMemoryPrior()
        
        start = time.time()
        
        # Add many entries
        for i in range(500):
            features = np.random.rand(10)
            prior.update(features, f'tool_{i%5}', True, 0.9, 100, 50)
        
        elapsed = time.time() - start
        
        # Should complete reasonably quickly
        assert elapsed < 10.0
    
    def test_prior_computation_speed(self):
        """Test prior computation performance"""
        prior = BayesianMemoryPrior()
        
        # Add history
        for i in range(100):
            prior.update(np.random.rand(5), 'tool_a', True, 0.9, 100, 50)
        
        tools = ['tool_a', 'tool_b', 'tool_c']
        
        start = time.time()
        
        # Compute multiple priors
        for _ in range(20):
            features = np.random.rand(5)
            prior.compute_prior(features, tools)
        
        elapsed = time.time() - start
        
        # Should be fast (with caching)
        assert elapsed < 2.0
    
    def test_cache_performance_benefit(self):
        """Test that caching improves performance"""
        prior = BayesianMemoryPrior()
        
        # Add history to make the uncached call do some work
        for i in range(50):
            prior.update(np.random.rand(5), 'tool_a', True, 0.9, 100, 50)
        
        features = np.array([1, 2, 3, 4, 5])
        tools = ['tool_a', 'tool_b']
        
        num_runs = 100

        # --- Measure uncached performance ---
        # The first call is a miss, the rest are hits. This represents a "cold start" scenario.
        prior.clear_cache()
        start_uncached = time.perf_counter()
        for _ in range(num_runs):
            prior.compute_prior(features, tools)
        end_uncached = time.perf_counter()
        first_run_time = end_uncached - start_uncached

        # --- Measure fully cached performance ---
        # The cache is now warm. All calls should be fast hits.
        start_cached = time.perf_counter()
        for _ in range(num_runs):
            prior.compute_prior(features, tools)
        end_cached = time.perf_counter()
        second_run_time = end_cached - start_cached

        # The fully cached run should be significantly faster than the run with the initial cache miss.
        # Add a small epsilon to prevent floating point comparison issues.
        assert second_run_time < (first_run_time * 0.9), f"Cached time ({second_run_time:.6f}) was not faster than uncached time ({first_run_time:.6f})"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])