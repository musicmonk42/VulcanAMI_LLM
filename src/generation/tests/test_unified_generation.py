"""
Comprehensive test suite for unified_generation.py

Tests cover:
- Basic candidate generation
- Multiple fusion strategies
- Normalization methods
- Dynamic weight adaptation
- Cross-module interaction
- Caching functionality
- Fallback behavior
- Edge cases and error handling
- Performance metrics
"""

import unittest
import math
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from collections import deque

# Add the parent directory to the path to import the module
sys.path.insert(0, '/mnt/user-data/uploads')

from unified_generation import (
    UnifiedGeneration,
    UnifiedGenConfig,
    FusionStrategy,
    NormalizationMethod,
    CandidateMetadata
)


class MockModule:
    """Mock reasoning module for testing"""
    
    def __init__(self, name: str, tokens: list, scores: list = None, method: str = "propose_candidates"):
        self.name = name
        self.tokens = tokens
        self.scores = scores or [1.0] * len(tokens)
        self.method = method
        self.confidence = 0.8
        
    def propose_candidates(self, hidden_state, context):
        """Return candidates as dicts with token and score"""
        return [
            {"token": t, "score": s, "logit": math.log(max(s, 1e-9))}
            for t, s in zip(self.tokens, self.scores)
        ]
    
    def generate_candidates(self, hidden_state, context):
        """Alternative method name"""
        return self.propose_candidates(hidden_state, context)
    
    def select_next_token(self, hidden_state, context):
        """Return best single token"""
        if not self.tokens:
            return None
        best_idx = self.scores.index(max(self.scores))
        return self.tokens[best_idx]
    
    def generate(self, hidden_state, context=None):
        """Another alternative method"""
        return self.select_next_token(hidden_state, context)
    
    def score_candidates(self, candidates, context):
        """Score a list of candidates"""
        return [c.get("score", 0.5) for c in candidates]
    
    def get_confidence(self):
        """Return module confidence"""
        return self.confidence


class TestUnifiedGenerationBasics(unittest.TestCase):
    """Test basic functionality of UnifiedGeneration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig(max_candidates=5)
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_initialization(self):
        """Test that generator initializes correctly"""
        self.assertIsNotNone(self.gen)
        self.assertEqual(self.gen.cfg.max_candidates, 5)
        self.assertIsInstance(self.gen._cache, dict)
        self.assertEqual(self.gen._cache_hits, 0)
        
    def test_empty_modules(self):
        """Test behavior with no modules"""
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={}
        )
        # Should return fallback candidates
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all("token" in c for c in candidates))
        
    def test_single_module(self):
        """Test with a single reasoning module"""
        module = MockModule("test", tokens=[1, 2, 3], scores=[0.5, 0.3, 0.2])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        # Should have metadata
        self.assertTrue(all("token" in c for c in candidates))
        self.assertTrue(all("prob" in c for c in candidates))
        self.assertTrue(all("score" in c for c in candidates))
        self.assertTrue(all("rank" in c for c in candidates))
        self.assertTrue(all("provenance" in c for c in candidates))
        
    def test_multiple_modules(self):
        """Test with multiple reasoning modules"""
        module1 = MockModule("symbolic", tokens=[1, 2, 3], scores=[0.6, 0.3, 0.1])
        module2 = MockModule("causal", tokens=[2, 3, 4], scores=[0.5, 0.3, 0.2])
        module3 = MockModule("language", tokens=[1, 3, 5], scores=[0.4, 0.4, 0.2])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={
                "symbolic": module1,
                "causal": module2,
                "language": module3
            }
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Check that module agreement is tracked
        for c in candidates:
            self.assertIn("module_agreement", c)
            self.assertGreaterEqual(c["module_agreement"], 1)
            
    def test_deduplication(self):
        """Test that duplicate tokens are deduplicated"""
        module1 = MockModule("mod1", tokens=[1, 2, 3], scores=[0.5, 0.3, 0.2])
        module2 = MockModule("mod2", tokens=[1, 2, 4], scores=[0.4, 0.4, 0.2])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": module1, "mod2": module2}
        )
        
        # Extract unique tokens
        tokens = [c["token"] for c in candidates]
        unique_tokens = set(tokens)
        
        # Should have deduplication by default
        self.assertEqual(len(tokens), len(unique_tokens))


class TestFusionStrategies(unittest.TestCase):
    """Test different fusion strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hidden_state = [0.1, 0.2, 0.3]
        self.module1 = MockModule("mod1", tokens=[1, 2, 3], scores=[0.6, 0.3, 0.1])
        self.module2 = MockModule("mod2", tokens=[1, 2, 4], scores=[0.5, 0.3, 0.2])
        
    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion strategy"""
        config = UnifiedGenConfig(
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": self.module1, "mod2": self.module2}
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        # Scores should be weighted sums
        self.assertTrue(all(c["score"] > 0 for c in candidates))
        
    def test_product_fusion(self):
        """Test product fusion strategy"""
        config = UnifiedGenConfig(
            fusion_strategy=FusionStrategy.PRODUCT,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": self.module1, "mod2": self.module2}
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
    def test_max_fusion(self):
        """Test max fusion strategy"""
        config = UnifiedGenConfig(
            fusion_strategy=FusionStrategy.MAX,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": self.module1, "mod2": self.module2}
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
    def test_rank_fusion(self):
        """Test rank fusion strategy"""
        config = UnifiedGenConfig(
            fusion_strategy=FusionStrategy.RANK_FUSION,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": self.module1, "mod2": self.module2}
        )
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)


class TestNormalizationMethods(unittest.TestCase):
    """Test different normalization methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hidden_state = [0.1, 0.2, 0.3]
        self.module = MockModule("test", tokens=[1, 2, 3], scores=[0.6, 0.3, 0.1])
        
    def test_softmax_normalization(self):
        """Test softmax normalization"""
        config = UnifiedGenConfig(
            normalization_method=NormalizationMethod.SOFTMAX,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": self.module}
        )
        
        # Probabilities should sum to ~1.0
        total_prob = sum(c["prob"] for c in candidates)
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
    def test_min_max_normalization(self):
        """Test min-max normalization"""
        config = UnifiedGenConfig(
            normalization_method=NormalizationMethod.MIN_MAX,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": self.module}
        )
        
        # Should normalize probabilities
        self.assertTrue(all(0 <= c["prob"] <= 1 for c in candidates))
        
    def test_z_score_normalization(self):
        """Test z-score normalization"""
        config = UnifiedGenConfig(
            normalization_method=NormalizationMethod.Z_SCORE,
            max_candidates=5
        )
        gen = UnifiedGeneration(config)
        
        candidates = gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": self.module}
        )
        
        self.assertTrue(all(c["prob"] >= 0 for c in candidates))


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced features like caching and dynamic weights"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig(
            enable_caching=True,
            enable_dynamic_weights=True,
            cache_size=100
        )
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_caching(self):
        """Test that caching works correctly"""
        module = MockModule("test", tokens=[1, 2, 3], scores=[0.6, 0.3, 0.1])
        
        # First call - should miss cache
        candidates1 = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        self.assertEqual(self.gen._cache_misses, 1)
        
        # Second call with same inputs - should hit cache
        candidates2 = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        self.assertEqual(self.gen._cache_hits, 1)
        
        # Results should be identical
        self.assertEqual(len(candidates1), len(candidates2))
        
    def test_weight_override(self):
        """Test module weight override"""
        module1 = MockModule("mod1", tokens=[1, 2], scores=[0.5, 0.5])
        module2 = MockModule("mod2", tokens=[1, 2], scores=[0.5, 0.5])
        
        # Override weight for mod1
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={
                "mod1": module1,
                "mod2": module2,
                "weights": {"mod1": 2.0, "mod2": 0.5}
            }
        )
        
        # Mod1 should have higher influence
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
    def test_temperature_scaling(self):
        """Test temperature scaling"""
        module = MockModule("test", tokens=[1, 2, 3], scores=[0.9, 0.05, 0.05])
        
        # High temperature (more uniform)
        candidates_high_temp = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module, "temperature": 2.0}
        )
        
        # Low temperature (more peaked)
        candidates_low_temp = self.gen.generate_candidates(
            hidden_state=[0.2, 0.3, 0.4],  # Different state to avoid cache
            reasoning_modules={"test": module, "temperature": 0.5}
        )
        
        # Both should return candidates
        self.assertGreater(len(candidates_high_temp), 0)
        self.assertGreater(len(candidates_low_temp), 0)
        
    def test_max_candidates_override(self):
        """Test max_candidates override"""
        module = MockModule("test", tokens=list(range(20)), scores=[0.1]*20)
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module, "max_candidates": 3}
        )
        
        # Should respect override
        self.assertLessEqual(len(candidates), 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig()
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_module_with_no_candidates(self):
        """Test module that returns empty candidates"""
        module = MockModule("empty", tokens=[], scores=[])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"empty": module}
        )
        
        # Should return fallback
        self.assertIsInstance(candidates, list)
        
    def test_module_with_invalid_method(self):
        """Test module without standard methods"""
        module = Mock()
        module.some_other_method = Mock(return_value=[])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"invalid": module}
        )
        
        # Should handle gracefully with fallback
        self.assertIsInstance(candidates, list)
        
    def test_module_exception(self):
        """Test module that raises exception"""
        module = Mock()
        module.propose_candidates = Mock(side_effect=Exception("Test error"))
        module.generate_candidates = Mock(side_effect=Exception("Test error"))
        module.select_next_token = Mock(side_effect=Exception("Test error"))
        module.generate = Mock(side_effect=Exception("Test error"))
        
        # Should handle exception and continue
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"error": module}
        )
        
        self.assertIsInstance(candidates, list)
        
    def test_invalid_hidden_state(self):
        """Test with invalid hidden state"""
        module = MockModule("test", tokens=[1, 2], scores=[0.5, 0.5])
        
        # Should handle None
        candidates = self.gen.generate_candidates(
            hidden_state=None,
            reasoning_modules={"test": module}
        )
        self.assertIsInstance(candidates, list)
        
        # Should handle string
        candidates = self.gen.generate_candidates(
            hidden_state="invalid",
            reasoning_modules={"test": module}
        )
        self.assertIsInstance(candidates, list)
        
    def test_zero_scores(self):
        """Test with all zero scores"""
        module = MockModule("test", tokens=[1, 2, 3], scores=[0.0, 0.0, 0.0])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        
        # Should handle with uniform distribution
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
    def test_negative_scores(self):
        """Test with negative scores"""
        module = MockModule("test", tokens=[1, 2, 3], scores=[-0.5, -0.3, -0.2])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        
        # Should handle gracefully
        self.assertIsInstance(candidates, list)


class TestProvenance(unittest.TestCase):
    """Test provenance tracking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig()
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_provenance_single_module(self):
        """Test provenance with single module"""
        module = MockModule("test", tokens=[1, 2], scores=[0.7, 0.3])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        
        # Check provenance exists
        for c in candidates:
            self.assertIn("provenance", c)
            self.assertIsInstance(c["provenance"], list)
            self.assertGreater(len(c["provenance"]), 0)
            # Should have module name
            self.assertEqual(c["provenance"][0]["module"], "test")
            
    def test_provenance_multiple_modules(self):
        """Test provenance with multiple modules"""
        module1 = MockModule("symbolic", tokens=[1, 2], scores=[0.6, 0.4])
        module2 = MockModule("causal", tokens=[1, 3], scores=[0.5, 0.5])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"symbolic": module1, "causal": module2}
        )
        
        # Find token 1 which should have provenance from both modules
        token_1_candidates = [c for c in candidates if c["token"] == 1]
        if token_1_candidates:
            c = token_1_candidates[0]
            module_names = [p["module"] for p in c["provenance"]]
            # Should have multiple modules in provenance
            self.assertIn("symbolic", module_names)
            self.assertIn("causal", module_names)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance tracking and metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig()
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_generation_history(self):
        """Test that generation history is tracked"""
        module = MockModule("test", tokens=[1, 2], scores=[0.6, 0.4])
        
        initial_history_len = len(self.gen._generation_history)
        
        self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        
        # History should be updated
        self.assertGreater(len(self.gen._generation_history), initial_history_len)
        
    def test_cache_statistics(self):
        """Test cache hit/miss statistics"""
        module = MockModule("test", tokens=[1, 2], scores=[0.6, 0.4])
        
        # Enable caching
        self.gen.cfg.enable_caching = True
        
        # First call - cache miss
        initial_misses = self.gen._cache_misses
        self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        self.assertEqual(self.gen._cache_misses, initial_misses + 1)
        
        # Second call - cache hit
        initial_hits = self.gen._cache_hits
        self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"test": module}
        )
        self.assertEqual(self.gen._cache_hits, initial_hits + 1)


class TestModuleInteraction(unittest.TestCase):
    """Test cross-module interaction features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UnifiedGenConfig(enable_cross_module_interaction=True)
        self.gen = UnifiedGeneration(self.config)
        self.hidden_state = [0.1, 0.2, 0.3]
        
    def test_module_agreement_boost(self):
        """Test that module agreement boosts scores"""
        # Both modules propose token 1
        module1 = MockModule("mod1", tokens=[1, 2], scores=[0.5, 0.5])
        module2 = MockModule("mod2", tokens=[1, 3], scores=[0.5, 0.5])
        
        candidates = self.gen.generate_candidates(
            hidden_state=self.hidden_state,
            reasoning_modules={"mod1": module1, "mod2": module2}
        )
        
        # Find token 1
        token_1 = next((c for c in candidates if c["token"] == 1), None)
        self.assertIsNotNone(token_1)
        
        # Should have agreement from 2 modules
        self.assertEqual(token_1.get("module_agreement", 0), 2)


class TestHelperMethods(unittest.TestCase):
    """Test helper and utility methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gen = UnifiedGeneration()
        
    def test_softmax(self):
        """Test softmax normalization"""
        scores = [1.0, 2.0, 3.0]
        probs = self.gen._softmax(scores)
        
        # Should sum to 1
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        # Should be monotonic
        self.assertTrue(probs[0] < probs[1] < probs[2])
        
    def test_softmax_empty(self):
        """Test softmax with empty list"""
        probs = self.gen._softmax([])
        self.assertEqual(probs, [])
        
    def test_normalize(self):
        """Test simple normalization"""
        scores = [2.0, 4.0, 4.0]
        probs = self.gen._normalize(scores)
        
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        
    def test_min_max_normalize(self):
        """Test min-max normalization"""
        scores = [1.0, 5.0, 9.0]
        normalized = self.gen._min_max_normalize(scores)
        
        self.assertAlmostEqual(sum(normalized), 1.0, places=6)
        
    def test_uniform(self):
        """Test uniform distribution"""
        dist = self.gen._uniform(5)
        
        self.assertEqual(len(dist), 5)
        self.assertTrue(all(abs(d - 0.2) < 1e-6 for d in dist))
        
    def test_merge_weights(self):
        """Test weight merging"""
        overrides = {"symbolic": 2.0, "causal": 1.5}
        merged = self.gen._merge_weights(overrides)
        
        self.assertEqual(merged["symbolic"], 2.0)
        self.assertEqual(merged["causal"], 1.5)
        # Should keep defaults for others
        self.assertIn("language", merged)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedGenerationBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizationMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestProvenance))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperMethods))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)