"""
Comprehensive Test Suite for CausalContext

Tests cover:
- Causal graph traversal
- Temporal reasoning
- Intervention tracking
- Counterfactual analysis
- Confounding detection
- Path finding algorithms
- Decay functions
- Caching mechanisms
- Edge cases and error handling
- Integration with world models
- Performance benchmarks
"""

from causal_context import (CausalContext, CounterfactualScenario, TemporalDecayFunction)
import sys
import time
import unittest

# Import the module to test
sys.path.insert(0, "/home/claude")


class MockWorldModel:
    """Mock world model for testing"""

    def __init__(self, with_causal_graph=True):
        self.with_causal_graph = with_causal_graph

        if with_causal_graph:
            self.causal_graph = {
                "nodes": ["A", "B", "C", "D", "E"],
                "edges": [
                    {"source": "A", "target": "B", "weight": 0.8},
                    {"source": "B", "target": "C", "weight": 0.7},
                    {"source": "A", "target": "C", "weight": 0.5},
                    {"source": "C", "target": "D", "weight": 0.9},
                    {"source": "D", "target": "E", "weight": 0.6},
                ],
            }

    def extract_concepts(self, text):
        """Extract concepts from text"""
        words = text.lower().split()
        return [w for w in words if len(w) > 2][:10]

    def get_related_concepts(self, concepts):
        """Get related concepts"""
        related = []
        for c in concepts:
            related.extend([f"{c}_related_1", f"{c}_related_2"])
        return related[:20]

    def get_causal_parents(self, node):
        """Get causal parents of a node"""
        if not self.with_causal_graph:
            return []

        parents = []
        for edge in self.causal_graph["edges"]:
            if edge["target"] == node:
                parents.append(edge["source"])
        return parents

    def get_causal_children(self, node):
        """Get causal children of a node"""
        if not self.with_causal_graph:
            return []

        children = []
        for edge in self.causal_graph["edges"]:
            if edge["source"] == node:
                children.append(edge["target"])
        return children

    def compute_intervention_effect(self, intervention, outcome):
        """Compute intervention effect"""
        return 0.75


class TestCausalContextBasics(unittest.TestCase):
    """Test basic functionality of CausalContext"""

    def setUp(self):
        """Set up test fixtures"""
        self.causal_ctx = CausalContext(
            causal_depth=3, temporal_window=86400.0, enable_caching=True
        )
        self.world_model = MockWorldModel()

    def test_initialization(self):
        """Test causal context initialization"""
        self.assertEqual(self.causal_ctx.causal_depth, 3)
        self.assertEqual(self.causal_ctx.temporal_window, 86400.0)
        self.assertTrue(self.causal_ctx.enable_caching)

    def test_select_with_string_query(self):
        """Test select with string query"""
        result = self.causal_ctx.select(
            world_model=self.world_model, query="What causes A?"
        )

        self.assertIn("causal_context", result)
        self.assertIn("concepts", result)
        self.assertIn("statistics", result)

    def test_select_with_dict_query(self):
        """Test select with dict query"""
        query = {"text": "What causes B?", "limit": 10, "causal_depth": 2}

        result = self.causal_ctx.select(world_model=self.world_model, query=query)

        self.assertIsNotNone(result)
        self.assertIn("causal_context", result)

    def test_select_with_memory(self):
        """Test select with memory data"""
        memory_data = {
            "episodic": [
                {"prompt": "test", "token": "value", "trace": {}, "ts": time.time()}
            ],
            "semantic": [
                {"concept": "machine_learning", "terms": ["ml", "ai"], "freq": 5}
            ],
            "procedural": [
                {"name": "strategy:greedy", "signature_terms": ["greedy"], "freq": 3}
            ],
        }

        query = {"text": "machine learning", "memory": memory_data, "limit": 10}

        result = self.causal_ctx.select(world_model=self.world_model, query=query)

        self.assertGreater(len(result["causal_context"]), 0)

    def test_select_without_world_model(self):
        """Test select without world model"""
        result = self.causal_ctx.select(world_model=None, query="test query")

        self.assertIsNotNone(result)
        self.assertIn("causal_context", result)


class TestCausalGraphTraversal(unittest.TestCase):
    """Test causal graph traversal functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.causal_ctx = CausalContext(causal_depth=3)
        self.world_model = MockWorldModel()

    def test_get_causal_graph(self):
        """Test causal graph extraction"""
        concepts = ["A", "B"]
        graph = self.causal_ctx._get_causal_graph(self.world_model, concepts)

        self.assertIsNotNone(graph)
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)

    def test_find_path_direct(self):
        """Test finding direct path"""
        adjacency = {"A": ["B"], "B": ["C"], "C": ["D"]}

        path = self.causal_ctx._find_path("A", "C", adjacency)

        self.assertIsNotNone(path)
        self.assertIn("A", path)
        self.assertIn("C", path)

    def test_find_path_no_connection(self):
        """Test finding path when no connection exists"""
        adjacency = {"A": ["B"], "C": ["D"]}

        path = self.causal_ctx._find_path("A", "D", adjacency)

        # Should return None or empty when no path exists
        self.assertIn(path, [None, []])

    def test_traverse_causal_graph(self):
        """Test causal graph traversal"""
        concepts = ["A", "B"]
        graph = self.causal_ctx._get_causal_graph(self.world_model, concepts)

        result = self.causal_ctx._traverse_causal_graph(concepts, graph, max_depth=2)

        self.assertIsInstance(result, dict)
        self.assertIn("paths", result)
        self.assertIn("nodes_visited", result)

    def test_identify_confounders(self):
        """Test confounding identification"""
        concepts = ["B", "D"]
        graph = self.causal_ctx._get_causal_graph(self.world_model, concepts)

        confounders = self.causal_ctx._identify_confounders(concepts, graph)

        self.assertIsInstance(confounders, list)

    def test_identify_mediators(self):
        """Test mediator identification"""
        concepts = ["A", "C"]
        graph = self.causal_ctx._get_causal_graph(self.world_model, concepts)

        mediators = self.causal_ctx._identify_mediators(concepts, graph)

        self.assertIsInstance(mediators, list)


class TestTemporalReasoning(unittest.TestCase):
    """Test temporal causal reasoning"""

    def test_temporal_decay_exponential(self):
        """Test exponential decay"""
        ctx = CausalContext(
            decay_function=TemporalDecayFunction.EXPONENTIAL, decay_half_life_hours=24.0
        )

        # Recent should have high decay
        recent = ctx._time_decay(3600)  # 1 hour
        self.assertGreater(recent, 0.9)

        # Old should have low decay
        old = ctx._time_decay(86400)  # 24 hours
        self.assertAlmostEqual(old, 0.5, places=1)

    def test_temporal_decay_hyperbolic(self):
        """Test hyperbolic decay"""
        ctx = CausalContext(decay_function=TemporalDecayFunction.HYPERBOLIC)

        decay = ctx._time_decay(7200)  # 2 hours
        self.assertGreater(decay, 0)
        self.assertLess(decay, 1)

    def test_temporal_decay_power_law(self):
        """Test power law decay"""
        ctx = CausalContext(decay_function=TemporalDecayFunction.POWER_LAW)

        decay = ctx._time_decay(3600)
        self.assertGreater(decay, 0)
        self.assertLess(decay, 1)

    def test_temporal_decay_linear(self):
        """Test linear decay"""
        ctx = CausalContext(
            decay_function=TemporalDecayFunction.LINEAR, decay_half_life_hours=24.0
        )

        decay = ctx._time_decay(43200)  # 12 hours (half of 24)
        self.assertGreater(decay, 0.4)
        self.assertLess(decay, 0.6)

    def test_temporal_window_filtering(self):
        """Test temporal window filtering"""
        ctx = CausalContext(temporal_window=3600.0)  # 1 hour
        world_model = MockWorldModel()

        now = time.time()
        memory_data = {
            "episodic": [
                {
                    "prompt": "recent",
                    "token": "val",
                    "trace": {},
                    "ts": now - 1800,
                },  # 30 min ago
                {
                    "prompt": "old",
                    "token": "val",
                    "trace": {},
                    "ts": now - 7200,
                },  # 2 hours ago
            ]
        }

        result = ctx.select(
            world_model=world_model, query={"text": "test", "memory": memory_data}
        )

        # Should prefer recent items
        self.assertGreater(len(result["causal_context"]), 0)


class TestInterventions(unittest.TestCase):
    """Test intervention tracking and analysis"""

    def test_track_intervention(self):
        """Test intervention tracking"""
        ctx = CausalContext()

        ctx.track_intervention(variable="X", value=10, effect_on={"Y": 0.8, "Z": 0.3})

        self.assertEqual(len(ctx._interventions), 1)

    def test_multiple_interventions(self):
        """Test tracking multiple interventions"""
        ctx = CausalContext()

        for i in range(10):
            ctx.track_intervention(variable=f"X{i}", value=i, effect_on={f"Y{i}": 0.5})

        self.assertEqual(len(ctx._interventions), 10)

    def test_intervention_overflow(self):
        """Test intervention deque overflow"""
        ctx = CausalContext()

        # Add more than max (1000)
        for i in range(1500):
            ctx.track_intervention(variable=f"X{i}", value=i)

        # Should keep only last 1000
        self.assertEqual(len(ctx._interventions), 1000)

    def test_get_recent_interventions(self):
        """Test getting recent interventions"""
        ctx = CausalContext()

        for i in range(20):
            ctx.track_intervention(variable=f"X{i}", value=i)
            time.sleep(0.01)

        recent = ctx.get_recent_interventions(limit=5)

        self.assertEqual(len(recent), 5)


class TestCounterfactualReasoning(unittest.TestCase):
    """Test counterfactual reasoning"""

    def test_compute_counterfactual_basic(self):
        """Test basic counterfactual computation"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        scenario = ctx.compute_counterfactual(
            world_model=world_model,
            intervention={"variable": "A", "value": 100},
            outcome="E",
        )

        self.assertIsInstance(scenario, CounterfactualScenario)
        self.assertIsNotNone(scenario.outcome_difference)

    def test_counterfactual_without_causal_graph(self):
        """Test counterfactual without causal graph"""
        ctx = CausalContext()
        world_model = MockWorldModel(with_causal_graph=False)

        scenario = ctx.compute_counterfactual(
            world_model=world_model,
            intervention={"variable": "X", "value": 5},
            outcome="Y",
        )

        # Should still return a scenario with limited info
        self.assertIsInstance(scenario, CounterfactualScenario)


class TestCausalStrength(unittest.TestCase):
    """Test causal strength computation"""

    def test_compute_path_strength(self):
        """Test path strength computation"""
        ctx = CausalContext()

        path = ["A", "B", "C"]
        causal_graph = {
            "edges": [
                {"source": "A", "target": "B", "weight": 0.8},
                {"source": "B", "target": "C", "weight": 0.7},
            ]
        }

        strength = ctx._compute_causal_strength(path, causal_graph)

        self.assertGreater(strength, 0)
        self.assertLessEqual(strength, 1)

    def test_compute_strength_empty_path(self):
        """Test strength computation with empty path"""
        ctx = CausalContext()

        strength = ctx._compute_causal_strength([], {})

        self.assertEqual(strength, 0.0)

    def test_compute_strength_no_graph(self):
        """Test strength computation without graph"""
        ctx = CausalContext()

        strength = ctx._compute_causal_strength(["A", "B"], None)

        self.assertEqual(strength, 0.0)


class TestScoring(unittest.TestCase):
    """Test context scoring mechanisms"""

    def setUp(self):
        """Set up test fixtures"""
        self.ctx = CausalContext()
        self.world_model = MockWorldModel()

    def test_score_episodic_item(self):
        """Test episodic item scoring"""
        item = {
            "prompt": "machine learning test",
            "token": "response",
            "trace": {},
            "ts": time.time(),
            "importance": 1.0,
        }

        qterms = ["machine", "learning"]
        concepts = ["machine", "learning", "ai"]

        score = self.ctx._score_episodic_item(item, qterms, concepts, self.world_model)

        self.assertGreater(score, 0)

    def test_score_semantic_entry(self):
        """Test semantic entry scoring"""
        entry = {
            "concept": "neural_network",
            "terms": ["neural", "network", "deep"],
            "freq": 10,
            "last_seen": time.time(),
            "importance": 1.5,
        }

        qterms = ["neural", "network"]
        concepts = ["neural", "network"]

        score = self.ctx._score_semantic_entry(
            entry, qterms, concepts, self.world_model
        )

        self.assertGreater(score, 0)

    def test_score_procedural_pattern(self):
        """Test procedural pattern scoring"""
        pattern = {
            "name": "strategy:greedy",
            "signature_terms": ["greedy", "search"],
            "freq": 5,
            "last_seen": time.time(),
            "importance": 1.0,
            "success_rate": 0.85,
        }

        qterms = ["greedy", "strategy"]
        concepts = ["greedy"]

        score = self.ctx._score_procedural_pattern(
            pattern, qterms, concepts, self.world_model
        )

        self.assertGreater(score, 0)


class TestCaching(unittest.TestCase):
    """Test caching mechanisms"""

    def test_cache_enabled(self):
        """Test that caching works when enabled"""
        ctx = CausalContext(enable_caching=True)
        world_model = MockWorldModel()

        # First call - cache miss
        ctx.select(world_model, "test query")
        cache_size_1 = len(ctx._cache)

        # Second call - should hit cache
        ctx.select(world_model, "test query")

        self.assertEqual(len(ctx._cache), cache_size_1)

    def test_cache_disabled(self):
        """Test system works with cache disabled"""
        ctx = CausalContext(enable_caching=False)
        world_model = MockWorldModel()

        result1 = ctx.select(world_model, "test query")
        result2 = ctx.select(world_model, "test query")

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(len(ctx._cache), 0)

    def test_cache_eviction(self):
        """Test cache LRU eviction"""
        ctx = CausalContext(enable_caching=True, cache_size=5)
        world_model = MockWorldModel()

        # Fill cache beyond capacity
        for i in range(10):
            ctx.select(world_model, f"query {i}")

        # Cache should not exceed max size
        self.assertLessEqual(len(ctx._cache), 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_empty_query(self):
        """Test with empty query"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        result = ctx.select(world_model, "")

        self.assertIsNotNone(result)
        self.assertIn("causal_context", result)

    def test_none_world_model(self):
        """Test with None world model"""
        ctx = CausalContext()

        result = ctx.select(None, "test query")

        self.assertIsNotNone(result)

    def test_malformed_memory_data(self):
        """Test with malformed memory data"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        query = {
            "text": "test",
            "memory": {
                "episodic": [{}],  # Missing fields
                "semantic": None,
                "procedural": "invalid",
            },
        }

        # Should not crash
        result = ctx.select(world_model, query)
        self.assertIsNotNone(result)

    def test_invalid_causal_depth(self):
        """Test with invalid causal depth"""
        # Should coerce to valid value
        ctx = CausalContext(causal_depth=-5)
        self.assertGreaterEqual(ctx.causal_depth, 1)

        ctx = CausalContext(causal_depth=0)
        self.assertGreaterEqual(ctx.causal_depth, 1)

    def test_negative_temporal_window(self):
        """Test with negative temporal window"""
        ctx = CausalContext(temporal_window=-100)
        world_model = MockWorldModel()

        # Should still work
        result = ctx.select(world_model, "test")
        self.assertIsNotNone(result)

    def test_very_large_memory(self):
        """Test with very large memory data"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        large_memory = {
            "episodic": [
                {"prompt": f"p{i}", "token": f"t{i}", "trace": {}, "ts": time.time()}
                for i in range(1000):
            ],
            "semantic": [
                {"concept": f"c{i}", "terms": [f"t{i}"], "freq": 1} for i in range(500)
            ],
            "procedural": [
                {"name": f"proc{i}", "signature_terms": [f"s{i}"], "freq": 1}
                for i in range(100):
            ],
        }

        query = {"text": "test", "memory": large_memory, "limit": 10}

        result = ctx.select(world_model, query)

        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result["causal_context"]), 10)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""

    def test_normalize_query_string(self):
        """Test query normalization with string"""
        ctx = CausalContext()

        qtext, qterms = ctx._normalize_query("Hello World")

        self.assertEqual(qtext, "Hello World")
        self.assertIsInstance(qterms, list)

    def test_normalize_query_dict(self):
        """Test query normalization with dict"""
        ctx = CausalContext()

        qtext, qterms = ctx._normalize_query({"text": "Test query"})

        self.assertEqual(qtext, "Test query")

    def test_normalize_query_tokens(self):
        """Test query normalization with tokens"""
        ctx = CausalContext()

        qtext, qterms = ctx._normalize_query({"tokens": ["hello", "world"]})

        self.assertIn("hello", qtext)
        self.assertIn("world", qtext)

    def test_tokenize(self):
        """Test tokenization"""
        ctx = CausalContext()

        tokens = ctx._tokenize("Hello, World! Test-123")

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_overlap(self):
        """Test overlap computation"""
        ctx = CausalContext()

        overlap = ctx._overlap(
            ["hello", "world", "test"], ["hello", "world", "different"]
        )

        self.assertGreater(overlap, 0)
        self.assertLessEqual(overlap, 1)

    def test_overlap_empty(self):
        """Test overlap with empty lists"""
        ctx = CausalContext()

        overlap = ctx._overlap([], ["hello"])
        self.assertEqual(overlap, 0.0)

        overlap = ctx._overlap(["hello"], [])
        self.assertEqual(overlap, 0.0)


class TestStatistics(unittest.TestCase):
    """Test statistics collection"""

    def test_statistics_in_result(self):
        """Test that statistics are included in results"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        result = ctx.select(world_model, "test query")

        self.assertIn("statistics", result)
        stats = result["statistics"]
        self.assertIn("computation_time_ms", stats)

    def test_statistics_accuracy(self):
        """Test statistics accuracy"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        memory = {
            "episodic": [{"prompt": "p", "token": "t", "trace": {}, "ts": time.time()}],
            "semantic": [{"concept": "c", "terms": ["t"], "freq": 1}],
            "procedural": [],
        }

        result = ctx.select(world_model, {"text": "test", "memory": memory})

        stats = result["statistics"]
        self.assertGreater(stats["computation_time_ms"], 0)


class TestWorldModelIntegration(unittest.TestCase):
    """Test integration with different world model types"""

    def test_with_minimal_world_model(self):
        """Test with minimal world model (no methods)"""
        ctx = CausalContext()
        world_model = object()  # No methods

        result = ctx.select(world_model, "test")

        self.assertIsNotNone(result)

    def test_with_partial_world_model(self):
        """Test with partial world model implementation"""
        ctx = CausalContext()

        class PartialWorldModel:
            def extract_concepts(self, text):
                return text.split()

        world_model = PartialWorldModel()
        result = ctx.select(world_model, "test query")

        self.assertIsNotNone(result)

    def test_world_model_exception_handling(self):
        """Test handling of world model exceptions"""
        ctx = CausalContext()

        class BrokenWorldModel:
            def extract_concepts(self, text):
                raise Exception("Broken!")

            causal_graph = {"nodes": [], "edges": []}

        world_model = BrokenWorldModel()

        # Should not crash
        result = ctx.select(world_model, "test")
        self.assertIsNotNone(result)


class TestCausalPathFinding(unittest.TestCase):
    """Test causal path finding algorithms"""

    def test_build_causal_path(self):
        """Test building causal path from terms"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        terms = ["A", "C", "E"]
        graph = world_model.causal_graph

        path = ctx._build_causal_path(terms, graph)

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

    def test_path_contains_all_terms(self):
        """Test that path contains all requested terms"""
        ctx = CausalContext()
        world_model = MockWorldModel()

        terms = ["A", "B"]
        graph = world_model.causal_graph

        path = ctx._build_causal_path(terms, graph)

        for term in terms:
            self.assertIn(term, path)


class TestExplanationGeneration(unittest.TestCase):
    """Test causal explanation generation"""

    def test_generate_explanations(self):
        """Test explanation generation"""
        ctx = CausalContext()

        scored_items = [
            {
                "source": "episodic",
                "score": 0.95,
                "causal_path": ["A", "B", "C"],
                "causal_strength": 0.8,
                "temporal_relevance": 0.9,
            }
        ]

        concepts = ["A", "B"]
        graph = {"nodes": ["A", "B", "C"], "edges": []}

        explanations = ctx._generate_causal_explanations(scored_items, concepts, graph)

        self.assertIsInstance(explanations, list)
        self.assertGreater(len(explanations), 0)


def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "=" * 70)
    print("CAUSAL CONTEXT PERFORMANCE BENCHMARKS")
    print("=" * 70)

    ctx = CausalContext()
    world_model = MockWorldModel()

    # Prepare large memory
    large_memory = {
        "episodic": [
            {
                "prompt": f"prompt {i}",
                "token": f"token {i}",
                "trace": {},
                "ts": time.time(),
            }
            for i in range(500):
        ],
        "semantic": [
            {"concept": f"concept_{i}", "terms": [f"term_{i}"], "freq": i}
            for i in range(200):
        ],
        "procedural": [
            {"name": f"proc_{i}", "signature_terms": [f"sig_{i}"], "freq": i}
            for i in range(50):
        ],
    }

    # Test 1: Basic selection
    start = time.time()
    for i in range(50):
        ctx.select(world_model, f"query {i}")
    basic_time = time.time() - start
    print(
        f"\n1. Basic selection (50 queries): {basic_time:.3f}s ({50 / basic_time:.1f} ops/sec)"
    )

    # Test 2: Selection with large memory
    start = time.time()
    for i in range(20):
        ctx.select(
            world_model, {"text": f"test {i}", "memory": large_memory, "limit": 10}
        )
    memory_time = time.time() - start
    print(
        f"2. With large memory (20 queries): {memory_time:.3f}s ({20 / memory_time:.1f} ops/sec)"
    )

    # Test 3: Deep causal traversal
    deep_ctx = CausalContext(causal_depth=5)
    start = time.time()
    for i in range(20):
        deep_ctx.select(world_model, f"deep query {i}")
    deep_time = time.time() - start
    print(
        f"3. Deep traversal (20 queries, depth=5): {deep_time:.3f}s ({20 / deep_time:.1f} ops/sec)"
    )

    # Test 4: Intervention tracking
    start = time.time()
    for i in range(1000):
        ctx.track_intervention(f"X{i}", i, {f"Y{i}": 0.5})
    intervention_time = time.time() - start
    print(f"4. Intervention tracking (1000 interventions): {intervention_time:.3f}s")

    # Test 5: Counterfactual computation
    start = time.time()
    for i in range(20):
        ctx.compute_counterfactual(world_model, {"variable": "A", "value": i}, "E")
    counterfactual_time = time.time() - start
    print(f"5. Counterfactual analysis (20 scenarios): {counterfactual_time:.3f}s")

    # Test 6: Cache performance
    cached_ctx = CausalContext(enable_caching=True)
    start = time.time()
    # First run - populate cache
    for i in range(10):
        cached_ctx.select(world_model, f"cached query {i}")
    # Second run - hit cache
    for i in range(10):
        cached_ctx.select(world_model, f"cached query {i}")
    cache_time = time.time() - start
    print(f"6. With caching (20 queries, 10 unique): {cache_time:.3f}s")

    print(f"\n7. Final cache size: {len(ctx._cache)}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[""], verbosity=2, exit=False)

    # Run performance benchmarks
    run_performance_benchmark()
