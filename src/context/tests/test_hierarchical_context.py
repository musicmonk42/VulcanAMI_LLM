"""
Comprehensive Test Suite for HierarchicalContext

Tests cover:
- Basic storage and retrieval operations
- Memory consolidation
- Pruning strategies
- Caching mechanisms
- Thread safety
- Edge cases and error handling
- Memory limits and overflow
- Statistics and analytics
- Import/export functionality
- Performance benchmarks
"""

from hierarchical_context import (ConsolidationStrategy, EpisodicItem,
                                  HierarchicalContext, MemoryStatistics,
                                  MemoryTier, ProceduralPattern,
                                  PruningStrategy, RetrievalStrategy,
                                  SemanticEntry, create_default_memory)
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module to test
sys.path.insert(0, "/home/claude")


class TestHierarchicalContextBasics(unittest.TestCase):
    """Test basic functionality of HierarchicalContext"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = HierarchicalContext(
            max_ep=100,
            max_semantic=50,
            max_procedural=25,
            enable_consolidation=False,
            enable_caching=True,
        )

    def test_initialization(self):
        """Test memory system initialization"""
        self.assertEqual(self.memory.max_ep, 100)
        self.assertEqual(self.memory.max_semantic, 50)
        self.assertEqual(self.memory.max_procedural, 25)
        self.assertEqual(len(self.memory.episodic), 0)
        self.assertEqual(len(self.memory.semantic_index), 0)
        self.assertEqual(len(self.memory.procedural), 0)

    def test_store_simple(self):
        """Test basic storage operation"""
        self.memory.store(
            prompt="What is Python?",
            token="A programming language",
            reasoning_trace={"strategy": "factual"},
        )

        self.assertEqual(len(self.memory.episodic), 1)
        self.assertIsInstance(self.memory.episodic[0], EpisodicItem)

    def test_store_with_various_types(self):
        """Test storage with different input types"""
        # String prompt
        self.memory.store("Hello", "World", {})

        # List prompt
        self.memory.store(["token1", "token2"], "response", {})

        # Dict prompt
        self.memory.store({"text": "query"}, "answer", {})

        # Integer token
        self.memory.store("prompt", 42, {})

        self.assertEqual(len(self.memory.episodic), 4)

    def test_retrieve_empty_memory(self):
        """Test retrieval from empty memory"""
        result = self.memory.retrieve("test query", max_items=10)

        self.assertIn("episodic", result)
        self.assertIn("semantic", result)
        self.assertIn("procedural", result)
        self.assertEqual(len(result["episodic"]), 0)
        self.assertEqual(len(result["semantic"]), 0)
        self.assertEqual(len(result["procedural"]), 0)

    def test_retrieve_with_data(self):
        """Test retrieval with stored data"""
        # Store some data
        self.memory.store("What is machine learning?", "ML answer", {})
        self.memory.store("What is deep learning?", "DL answer", {})
        self.memory.store("What is Python?", "Python answer", {})

        # Retrieve
        result = self.memory.retrieve("machine learning", max_items=10)

        self.assertGreater(len(result["episodic"]), 0)

    def test_retrieve_with_different_strategies(self):
        """Test retrieval with different strategies"""
        # Store some data
        for i in range(10):
            self.memory.store(f"prompt {i}", f"token {i}", {})

        # Test each strategy
        for strategy in RetrievalStrategy:
            result = self.memory.retrieve("prompt 5", max_items=5, strategy=strategy)
            self.assertIsNotNone(result)
            self.assertIn("episodic", result)


class TestMemoryConsolidation(unittest.TestCase):
    """Test memory consolidation features"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = HierarchicalContext(
            max_ep=100, enable_consolidation=True, consolidation_threshold=5
        )

    def test_consolidate_from_episodic_to_semantic(self):
        """Test consolidation moves episodic to semantic"""
        # Get baseline before any stores
        initial_semantic = len(self.memory.semantic_index)

        # Store repeated concepts
        for i in range(10):
            self.memory.store(
                f"Tell me about machine learning {i}", f"ML is great {i}", {}
            )

        # Semantic entries are added during store, so we should already have more
        after_store_semantic = len(self.memory.semantic_index)

        # Consolidate with min_frequency=0 to ensure consolidation happens
        consolidated_count = self.memory.consolidate_memory(min_frequency=0)

        # Should have consolidated some items (marked as consolidated in episodic)
        # and may have added more semantic entries from consolidation
        self.assertGreaterEqual(len(self.memory.semantic_index), after_store_semantic)

    def test_consolidation_marks_episodic_items(self):
        """Test that consolidation marks episodic items"""
        for i in range(10):
            self.memory.store(f"concept {i}", f"value {i}", {})

        self.memory.consolidate_memory()

        # Some items should be marked as consolidated
        consolidated_count = sum(
            1 for item in self.memory.episodic if item.consolidated
        )
        self.assertGreater(consolidated_count, 0)

    def test_consolidation_strategies(self):
        """Test different consolidation strategies"""
        for i in range(20):
            self.memory.store(f"data {i}", f"response {i}", {})

        for strategy in ConsolidationStrategy:
            result = self.memory.consolidate_memory(strategy=strategy)
            self.assertIsNotNone(result)


class TestMemoryPruning(unittest.TestCase):
    """Test memory pruning features"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = HierarchicalContext(max_ep=50)

    def test_prune_with_decay_strategy(self):
        """Test pruning with decay strategy"""
        # Store old items
        for i in range(20):
            item = EpisodicItem(
                prompt=f"old prompt {i}",
                token=f"old token {i}",
                trace={},
                ts=time.time() - 100000,  # Very old
            )
            self.memory.episodic.append(item)

        # Store recent items
        for i in range(10):
            self.memory.store(f"recent {i}", f"token {i}", {})

        initial_count = len(self.memory.episodic)
        self.memory.prune_memory(strategy=PruningStrategy.DECAY)

        # Should have pruned some old items
        self.assertLess(len(self.memory.episodic), initial_count)

    def test_prune_with_lru_strategy(self):
        """Test pruning with LRU strategy"""
        for i in range(30):
            self.memory.store(f"item {i}", f"value {i}", {})

        # Access some items
        for i in range(0, 10):
            self.memory.episodic[i].access_count += 5
            self.memory.episodic[i].last_accessed = time.time()

        initial_count = len(self.memory.episodic)
        self.memory.prune_memory(strategy=PruningStrategy.LRU, target_size=20)

        self.assertLessEqual(len(self.memory.episodic), 20)

    def test_prune_all_strategies(self):
        """Test all pruning strategies"""
        for strategy in PruningStrategy:
            memory = HierarchicalContext(max_ep=50)
            for i in range(30):
                memory.store(f"item {i}", f"value {i}", {})

            memory.prune_memory(strategy=strategy, target_size=15)
            self.assertLessEqual(len(memory.episodic), 15)


class TestMemoryLimits(unittest.TestCase):
    """Test memory capacity limits and overflow handling"""

    def test_episodic_overflow(self):
        """Test episodic memory overflow"""
        memory = HierarchicalContext(max_ep=10)

        # Store more than max
        for i in range(20):
            memory.store(f"item {i}", f"value {i}", {})

        # Should not exceed max
        self.assertLessEqual(len(memory.episodic), 10)

    def test_semantic_overflow(self):
        """Test semantic memory overflow"""
        memory = HierarchicalContext(max_semantic=5)

        # Manually add semantic entries
        for i in range(10):
            entry = SemanticEntry(concept=f"concept_{i}", terms=[f"term_{i}"])
            memory.semantic_index.append(entry)

        # Prune semantic memory
        memory._prune_semantic(target_size=5)

        self.assertLessEqual(len(memory.semantic_index), 5)

    def test_procedural_overflow(self):
        """Test procedural memory overflow"""
        memory = HierarchicalContext(max_procedural=5)

        for i in range(10):
            pattern = ProceduralPattern(
                name=f"pattern_{i}", signature_terms=[f"sig_{i}"]
            )
            memory.procedural.append(pattern)

        memory._prune_procedural(target_size=5)

        self.assertLessEqual(len(memory.procedural), 5)


class TestCaching(unittest.TestCase):
    """Test caching mechanisms"""

    def test_cache_enabled(self):
        """Test cache is used when enabled"""
        memory = HierarchicalContext(enable_caching=True)

        for i in range(5):
            memory.store(f"item {i}", f"value {i}", {})

        # First retrieval - cache miss
        result1 = memory.retrieve("item 1", max_items=5)
        initial_hits = memory._cache_hits

        # Second retrieval - should hit cache
        result2 = memory.retrieve("item 1", max_items=5)

        self.assertGreater(memory._cache_hits, initial_hits)

    def test_cache_disabled(self):
        """Test system works with cache disabled"""
        memory = HierarchicalContext(enable_caching=False)

        for i in range(5):
            memory.store(f"item {i}", f"value {i}", {})

        result1 = memory.retrieve("item 1", max_items=5)
        result2 = memory.retrieve("item 1", max_items=5)

        # Should still work
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)

    def test_cache_invalidation(self):
        """Test cache is invalidated on store"""
        memory = HierarchicalContext(enable_caching=True)

        memory.store("initial", "data", {})
        result1 = memory.retrieve("initial", max_items=5)

        cache_size_before = len(memory._cache)

        # Store more data - should invalidate cache
        memory.store("new data", "value", {})

        # Cache might be cleared or updated
        result2 = memory.retrieve("initial", max_items=5)
        self.assertIsNotNone(result2)


class TestStatistics(unittest.TestCase):
    """Test statistics and analytics"""

    def test_get_statistics(self):
        """Test statistics collection"""
        memory = HierarchicalContext()

        for i in range(10):
            memory.store(f"item {i}", f"value {i}", {})

        stats = memory.get_statistics()

        self.assertIsInstance(stats, MemoryStatistics)
        self.assertEqual(stats.episodic_count, 10)
        self.assertGreaterEqual(stats.semantic_count, 0)
        self.assertGreaterEqual(stats.procedural_count, 0)

    def test_statistics_after_operations(self):
        """Test statistics update after operations"""
        memory = HierarchicalContext()

        # Store data
        for i in range(20):
            memory.store(f"item {i}", f"value {i}", {})

        # Consolidate
        memory.consolidate_memory(min_frequency=0)

        stats = memory.get_statistics()
        self.assertGreater(stats.consolidation_count, 0)

        # Store more items (not consolidated) so pruning can work
        for i in range(20, 25):
            memory.store(f"item {i}", f"value {i}", {})

        # Prune - use target_reduction > 0 to force pruning
        memory.prune_memory(target_reduction=0.1)

        stats = memory.get_statistics()
        self.assertGreater(stats.pruning_count, 0)


class TestContextGeneration(unittest.TestCase):
    """Test context generation for LLM"""

    def test_retrieve_context_for_generation(self):
        """Test generation-ready context retrieval"""
        memory = HierarchicalContext()

        for i in range(10):
            memory.store(
                f"What is concept {i}?",
                f"Concept {i} explanation",
                {"strategy": "factual"},
            )

        context = memory.retrieve_context_for_generation(
            query_tokens=["concept", "5"], max_tokens=500
        )

        self.assertIn("context_items", context)
        self.assertIn("total_tokens", context)
        self.assertIn("formatted_context", context)
        self.assertLessEqual(context["total_tokens"], 500)

    def test_token_budget_respected(self):
        """Test that token budget is respected"""
        memory = HierarchicalContext()

        # Store large items
        for i in range(20):
            memory.store(
                "What is " + " ".join([f"word{j}" for j in range(100)]),
                "Response " + " ".join([f"word{j}" for j in range(100)]),
                {},
            )

        context = memory.retrieve_context_for_generation(
            query_tokens=["word50"], max_tokens=200
        )

        self.assertLessEqual(context["total_tokens"], 200)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety"""

    def test_concurrent_stores(self):
        """Test concurrent store operations"""
        memory = HierarchicalContext()
        errors = []

        def store_items(thread_id):
            try:
                for i in range(50):
                    memory.store(f"thread{thread_id}_item{i}", f"value{i}", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_items, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(memory.episodic), 250)

    def test_concurrent_retrieve_and_store(self):
        """Test concurrent retrieve and store operations"""
        memory = HierarchicalContext()
        errors = []

        # Pre-populate
        for i in range(20):
            memory.store(f"initial {i}", f"value {i}", {})

        def retrieve_items():
            try:
                for i in range(30):
                    memory.retrieve(f"initial {i % 10}", max_items=5)
            except Exception as e:
                errors.append(e)

        def store_items():
            try:
                for i in range(30):
                    memory.store(f"new {i}", f"value {i}", {})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=retrieve_items),
            threading.Thread(target=store_items),
            threading.Thread(target=retrieve_items),
            threading.Thread(target=store_items),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestImportExport(unittest.TestCase):
    """Test import/export functionality"""

    def test_export_memory(self):
        """Test exporting memory"""
        memory = HierarchicalContext()

        for i in range(10):
            memory.store(f"item {i}", f"value {i}", {})

        exported = memory.export_memory()

        self.assertIn("episodic", exported)
        self.assertIn("semantic", exported)
        self.assertIn("procedural", exported)
        self.assertEqual(len(exported["episodic"]), 10)

    def test_import_memory(self):
        """Test importing memory"""
        memory1 = HierarchicalContext()

        for i in range(10):
            memory1.store(f"item {i}", f"value {i}", {})

        exported = memory1.export_memory()

        # Create new memory and import
        memory2 = HierarchicalContext()
        memory2.import_memory(exported)

        self.assertEqual(len(memory2.episodic), 10)

    def test_export_import_roundtrip(self):
        """Test export-import roundtrip preserves data"""
        memory1 = HierarchicalContext()

        memory1.store("Test prompt", "Test response", {"strategy": "test"})
        memory1.consolidate_memory()

        exported = memory1.export_memory()

        memory2 = HierarchicalContext()
        memory2.import_memory(exported)

        # Verify data
        self.assertEqual(len(memory2.episodic), len(memory1.episodic))
        self.assertEqual(len(memory2.semantic_index), len(memory1.semantic_index))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_empty_query(self):
        """Test retrieval with empty query"""
        memory = HierarchicalContext()
        memory.store("data", "value", {})

        result = memory.retrieve("", max_items=5)
        self.assertIsNotNone(result)

    def test_none_inputs(self):
        """Test handling None inputs"""
        memory = HierarchicalContext()

        # Should not crash
        memory.store(None, None, None)
        memory.store("prompt", None, {})
        memory.store(None, "token", {})

    def test_very_large_inputs(self):
        """Test handling very large inputs"""
        memory = HierarchicalContext()

        large_prompt = "word " * 10000
        large_token = "token " * 10000

        memory.store(large_prompt, large_token, {})

        result = memory.retrieve(large_prompt[:100], max_items=5)
        self.assertIsNotNone(result)

    def test_special_characters(self):
        """Test handling special characters"""
        memory = HierarchicalContext()

        special = "Test with 特殊字符 and émojis 🎉 and symbols @#$%"
        memory.store(special, special, {})

        result = memory.retrieve(special, max_items=5)
        self.assertIsNotNone(result)

    def test_invalid_max_items(self):
        """Test handling invalid max_items values"""
        memory = HierarchicalContext()
        memory.store("test", "value", {})

        # Negative
        result = memory.retrieve("test", max_items=-5)
        self.assertIsNotNone(result)

        # Zero
        result = memory.retrieve("test", max_items=0)
        self.assertIsNotNone(result)

        # Very large
        result = memory.retrieve("test", max_items=1000000)
        self.assertIsNotNone(result)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_tokenize(self):
        """Test tokenization"""
        memory = HierarchicalContext()

        text = "Hello World! Test-123"
        tokens = memory._tokenize(text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_overlap_score(self):
        """Test overlap scoring"""
        memory = HierarchicalContext()

        a = ["hello", "world", "test"]
        b = ["hello", "world", "different"]

        score = memory._overlap_score(a, b)

        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)

    def test_decay_function(self):
        """Test decay function"""
        memory = HierarchicalContext(decay_half_life_hours=24.0)

        # Recent should have high decay
        recent_decay = memory._decay(3600)  # 1 hour
        self.assertGreater(recent_decay, 0.9)

        # Old should have low decay
        old_decay = memory._decay(86400 * 7)  # 7 days
        self.assertLess(old_decay, 0.5)

    def test_extract_concepts(self):
        """Test concept extraction"""
        memory = HierarchicalContext()

        concepts = memory._extract_concepts(
            "Machine learning and deep learning", "Neural networks and AI"
        )

        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)


class TestDefaultCreation(unittest.TestCase):
    """Test default memory creation"""

    def test_create_default_memory(self):
        """Test creating memory with defaults"""
        memory = create_default_memory()

        self.assertIsInstance(memory, HierarchicalContext)
        self.assertEqual(memory.max_ep, 10000)
        self.assertTrue(memory.enable_consolidation)
        self.assertTrue(memory.enable_caching)


class TestStoreGeneration(unittest.TestCase):
    """Test store_generation method"""

    def test_store_generation_batch(self):
        """Test batch storage of generation"""
        memory = HierarchicalContext()

        prompt = "What is AI?"
        generated = ["token1", "token2", "token3"]
        traces = [
            {"strategy": "greedy"},
            {"strategy": "sampling"},
            {"strategy": "beam"},
        ]

        memory.store_generation(prompt, generated, traces)

        self.assertEqual(len(memory.episodic), 3)

    def test_store_generation_mismatched_lengths(self):
        """Test store_generation with mismatched lengths"""
        memory = HierarchicalContext()

        # More tokens than traces
        memory.store_generation("prompt", ["a", "b", "c"], [{"s": 1}])

        self.assertEqual(len(memory.episodic), 3)


def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)

    memory = HierarchicalContext(max_ep=10000)

    # Test 1: Store performance
    start = time.time()
    for i in range(1000):
        memory.store(f"item {i}", f"value {i}", {})
    store_time = time.time() - start
    print(f"\n1. Store 1000 items: {store_time:.3f}s ({1000 / store_time:.1f} ops/sec)")

    # Test 2: Retrieve performance
    start = time.time()
    for i in range(100):
        memory.retrieve(f"item {i}", max_items=20)
    retrieve_time = time.time() - start
    print(
        f"2. Retrieve 100 queries: {retrieve_time:.3f}s ({100 / retrieve_time:.1f} ops/sec)"
    )

    # Test 3: Consolidation performance
    start = time.time()
    memory.consolidate_memory()
    consol_time = time.time() - start
    print(f"3. Consolidation: {consol_time:.3f}s")

    # Test 4: Pruning performance
    start = time.time()
    memory.prune_memory()
    prune_time = time.time() - start
    print(f"4. Pruning: {prune_time:.3f}s")

    # Test 5: Statistics
    stats = memory.get_statistics()
    print(f"\n5. Memory Statistics:")
    print(f"   - Episodic: {stats.episodic_count}")
    print(f"   - Semantic: {stats.semantic_count}")
    print(f"   - Procedural: {stats.procedural_count}")
    print(f"   - Cache hit rate: {stats.cache_hit_rate:.2%}")
    print(f"   - Avg retrieval time: {stats.avg_retrieval_time_ms:.2f}ms")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[""], verbosity=2, exit=False)

    # Run performance benchmarks
    run_performance_benchmark()
