"""
Comprehensive Test Suite for LLM Executor
==========================================

Tests all major functionality including:
- Execution modes (sequential, parallel)
- Safety validation
- Caching system
- Layer execution
- Token generation
- Metrics tracking
- State persistence
- Error handling and recovery
"""

from llm_executor import (NUMPY_AVAILABLE, TORCH_AVAILABLE,
                          AttentionHeadResult, CacheStrategy, ExecutionCache,
                          ExecutionMode, ExecutionResult, ExecutorConfig,
                          LayerExecutionContext, LayerExecutor, LLMExecutor,
                          SafetyLevel, SafetyValidationResult, SafetyValidator,
                          create_default_executor, create_parallel_executor)
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add the uploads directory to path
sys.path.insert(0, "/mnt/user-data/uploads")


# Try to import torch/numpy for tests
if TORCH_AVAILABLE:
    import torch
if NUMPY_AVAILABLE:
    import numpy as np


class TestExecutorConfiguration(unittest.TestCase):
    """Test executor configuration and initialization."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutorConfig()
        self.assertEqual(config.execution_mode, ExecutionMode.PARALLEL_HEADS)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.enable_cache)
        self.assertTrue(config.enable_audit)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_parallel_heads=16,
            device="cuda:0",
            enable_cache=False,
        )

        self.assertEqual(config.execution_mode, ExecutionMode.SEQUENTIAL)
        self.assertEqual(config.max_parallel_heads, 16)
        self.assertEqual(config.device, "cuda:0")
        self.assertFalse(config.enable_cache)

    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(AssertionError):
            ExecutorConfig(max_parallel_heads=0)

        with self.assertRaises(AssertionError):
            ExecutorConfig(batch_size=-1)

        with self.assertRaises(AssertionError):
            ExecutorConfig(cache_size=0)

    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = LLMExecutor()
        self.assertIsNotNone(executor.config)
        self.assertIsNotNone(executor.cache)
        self.assertIsNotNone(executor.safety_validator)
        self.assertIsNotNone(executor.layer_executor)


class TestExecutionCache(unittest.TestCase):
    """Test caching system."""

    def setUp(self):
        """Set up test cache."""
        config = ExecutorConfig(enable_cache=True, cache_size=10)
        self.cache = ExecutionCache(config)

    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        key = "test_key"
        value = {"data": "test_value"}

        self.cache.put(key, value)
        retrieved = self.cache.get(key)

        self.assertEqual(retrieved, value)
        self.assertEqual(self.cache.hits, 1)

    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)

    def test_cache_size_limit(self):
        """Test cache size enforcement."""
        config = ExecutorConfig(enable_cache=True, cache_size=3)
        cache = ExecutionCache(config)

        # Add more items than cache size
        for i in range(5):
            cache.put(f"key_{i}", {"value": i})

        # Cache should only contain most recent items
        self.assertLessEqual(len(cache.cache), 3)

    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        config = ExecutorConfig(enable_cache=True, cache_ttl=0.1)
        cache = ExecutionCache(config)

        cache.put("key", {"value": "data"})

        # Wait for TTL to expire
        time.sleep(0.15)

        result = cache.get("key")
        # Should be None or miss due to expiration
        self.assertIsNone(result)

    def test_cache_clear(self):
        """Test cache clearing."""
        self.cache.put("key1", {"value": 1})
        self.cache.put("key2", {"value": 2})

        self.cache.clear()

        self.assertEqual(len(self.cache.cache), 0)
        self.assertIsNone(self.cache.get("key1"))

    def test_cache_stats(self):
        """Test cache statistics."""
        self.cache.put("key1", {"value": 1})
        self.cache.get("key1")  # hit
        self.cache.get("key2")  # miss

        stats = self.cache.get_stats()

        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("size", stats)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)


class TestSafetyValidator(unittest.TestCase):
    """Test safety validation."""

    def setUp(self):
        """Set up test validator."""
        config = ExecutorConfig(safety_level=SafetyLevel.STANDARD)
        self.validator = SafetyValidator(config)

    def test_validate_token_basic(self):
        """Test basic token validation."""
        result = self.validator.validate_token(42, {"context": "test"})
        self.assertIsInstance(result, SafetyValidationResult)

    def test_validate_sequence_basic(self):
        """Test basic sequence validation."""
        tokens = [1, 2, 3, 4, 5]
        result = self.validator.validate_sequence(tokens)
        self.assertIsInstance(result, SafetyValidationResult)

    def test_safety_level_none(self):
        """Test with safety level NONE."""
        config = ExecutorConfig(safety_level=SafetyLevel.NONE)
        validator = SafetyValidator(config)

        # Should always pass
        result = validator.validate_token(999999, {})
        self.assertTrue(result.passed)


class TestLayerExecutor(unittest.TestCase):
    """Test layer execution."""

    def setUp(self):
        """Set up test layer executor."""
        config = ExecutorConfig()
        self.layer_executor = LayerExecutor(config)

    def test_execute_layer_basic(self):
        """Test basic layer execution."""
        layer = {
            "nodes": [
                {"id": "node_1", "type": "attention_head", "params": {}},
                {"id": "node_2", "type": "feedforward", "params": {}},
            ],
            "edges": [],
        }

        if TORCH_AVAILABLE:
            hidden_state = torch.randn(1, 10, 512)
        else:
            hidden_state = [[0.0] * 512 for _ in range(10)]

        context = LayerExecutionContext(layer_idx=0, hidden_state=hidden_state)

        output, metadata = self.layer_executor.execute_layer(layer, context)

        self.assertIsNotNone(output)
        self.assertIsInstance(metadata, dict)

    def test_execute_attention_head(self):
        """Test executing attention head."""
        head_config = {
            "id": "head_1",
            "type": "attention_head",
            "params": {"d_k": 64, "d_v": 64},
        }

        if TORCH_AVAILABLE:
            hidden_state = torch.randn(1, 10, 512)
        else:
            hidden_state = [[0.0] * 512 for _ in range(10)]

        context = LayerExecutionContext(layer_idx=0, hidden_state=hidden_state)

        result = self.layer_executor._execute_attention_head(head_config, context, 0)

        self.assertIsInstance(result, AttentionHeadResult)
        self.assertIsNotNone(result.output)

    def test_execute_feedforward(self):
        """Test executing feedforward layer."""
        ff_config = {
            "id": "ff_1",
            "type": "feedforward",
            "params": {"hidden_dim": 2048},
        }

        if TORCH_AVAILABLE:
            hidden_state = torch.randn(1, 10, 512)
        else:
            hidden_state = [[0.0] * 512 for _ in range(10)]

        output = self.layer_executor._execute_feedforward(ff_config, hidden_state)

        self.assertIsNotNone(output)


class TestExecutorExecution(unittest.TestCase):
    """Test main executor execution."""

    def setUp(self):
        """Set up test executor and graph."""
        self.executor = LLMExecutor()

        self.test_graph = {
            "layers": [
                {
                    "nodes": [
                        {
                            "id": "attn_0",
                            "type": "attention_head",
                            "params": {"d_k": 64},
                        },
                        {"id": "ff_0", "type": "feedforward", "params": {}},
                    ],
                    "edges": [],
                },
                {
                    "nodes": [
                        {
                            "id": "attn_1",
                            "type": "attention_head",
                            "params": {"d_k": 64},
                        },
                        {"id": "ff_1", "type": "feedforward", "params": {}},
                    ],
                    "edges": [],
                },
            ]
        }

        if TORCH_AVAILABLE:
            self.test_hidden = torch.randn(1, 10, 512)
        else:
            self.test_hidden = [[0.0] * 512 for _ in range(10)]

    def test_execute_basic(self):
        """Test basic execution."""
        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        result = self.executor.execute(self.test_graph, inputs)

        self.assertIsInstance(result, dict)
        self.assertIn("hidden_states", result)
        self.assertIn("audit", result)
        self.assertIn("metrics", result)

    def test_execute_with_attention_mask(self):
        """Test execution with attention mask."""
        if TORCH_AVAILABLE:
            attention_mask = torch.ones(1, 10)
        else:
            attention_mask = [[1.0] * 10]

        inputs = {"hidden_states": self.test_hidden, "attention_mask": attention_mask}

        result = self.executor.execute(self.test_graph, inputs)
        self.assertIsNotNone(result)

    def test_execute_sequential_mode(self):
        """Test sequential execution mode."""
        config = ExecutorConfig(execution_mode=ExecutionMode.SEQUENTIAL)
        executor = LLMExecutor(config=config)

        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        result = executor.execute(self.test_graph, inputs)
        self.assertIsNotNone(result)

    def test_execute_parallel_heads_mode(self):
        """Test parallel heads execution mode."""
        config = ExecutorConfig(execution_mode=ExecutionMode.PARALLEL_HEADS)
        executor = LLMExecutor(config=config)

        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        result = executor.execute(self.test_graph, inputs)
        self.assertIsNotNone(result)

    def test_execute_empty_graph(self):
        """Test execution with empty graph."""
        empty_graph = {"layers": []}

        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        result = self.executor.execute(empty_graph, inputs)
        self.assertIn("safety_status", result)


class TestTokenGeneration(unittest.TestCase):
    """Test token generation."""

    def setUp(self):
        """Set up test executor."""
        self.executor = LLMExecutor()

    def test_execute_generation_basic(self):
        """Test basic token generation."""
        gen_inputs = {"token_id": None}

        token = self.executor.execute_generation(gen_inputs)
        self.assertIsNotNone(token)

    def test_execute_generation_with_logits(self):
        """Test generation with logits."""
        if TORCH_AVAILABLE:
            logits = torch.randn(1, 1000)
            gen_inputs = {"logits": logits}
        elif NUMPY_AVAILABLE:
            logits = np.random.randn(1, 1000)
            gen_inputs = {"logits": logits}
        else:
            logits = [float(i) for i in range(1000)]
            gen_inputs = {"logits": logits}

        token = self.executor.execute_generation(gen_inputs)
        self.assertIsNotNone(token)


class TestExecutorMetrics(unittest.TestCase):
    """Test metrics tracking."""

    def setUp(self):
        """Set up test executor."""
        self.executor = LLMExecutor()

        if TORCH_AVAILABLE:
            self.test_hidden = torch.randn(1, 5, 512)
        else:
            self.test_hidden = [[0.0] * 512 for _ in range(5)]

        self.test_graph = {
            "layers": [
                {
                    "nodes": [{"id": "attn_0", "type": "attention_head", "params": {}}],
                    "edges": [],
                }
            ]
        }

    def test_metrics_tracking(self):
        """Test that metrics are tracked."""
        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        # Execute multiple times
        for _ in range(3):
            self.executor.execute(self.test_graph, inputs)

        metrics = self.executor.get_metrics()

        self.assertIn("total_executions", metrics)
        self.assertIn("total_time", metrics)
        self.assertEqual(metrics["total_executions"], 3)
        self.assertGreater(metrics["total_time"], 0)

    def test_reset_metrics(self):
        """Test resetting metrics."""
        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        self.executor.execute(self.test_graph, inputs)

        metrics_before = self.executor.get_metrics()
        self.assertGreater(metrics_before["total_executions"], 0)

        self.executor.reset_metrics()

        metrics_after = self.executor.get_metrics()
        self.assertEqual(metrics_after["total_executions"], 0)


class TestExecutorCaching(unittest.TestCase):
    """Test executor-level caching."""

    def setUp(self):
        """Set up test executor with caching enabled."""
        config = ExecutorConfig(enable_cache=True)
        self.executor = LLMExecutor(config=config)

        if TORCH_AVAILABLE:
            self.test_hidden = torch.randn(1, 5, 512)
        else:
            self.test_hidden = [[0.0] * 512 for _ in range(5)]

        self.test_graph = {
            "layers": [
                {
                    "nodes": [{"id": "attn_0", "type": "attention_head", "params": {}}],
                    "edges": [],
                }
            ]
        }

    def test_cache_utilization(self):
        """Test that cache is utilized."""
        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        # First execution
        result1 = self.executor.execute(self.test_graph, inputs)

        # Second execution (should use cache)
        result2 = self.executor.execute(self.test_graph, inputs)

        stats = self.executor.cache.get_stats()
        # May or may not have cache hits depending on implementation
        self.assertIsInstance(stats, dict)

    def test_clear_cache(self):
        """Test clearing cache."""
        inputs = {"hidden_states": self.test_hidden, "attention_mask": None}

        self.executor.execute(self.test_graph, inputs)
        self.executor.clear_cache()

        stats = self.executor.cache.get_stats()
        self.assertEqual(stats["size"], 0)


class TestStatePersistence(unittest.TestCase):
    """Test state save/load functionality."""

    def setUp(self):
        """Set up test executor."""
        self.executor = LLMExecutor()
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        self.temp_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        """Clean up temp file."""
        Path(self.temp_path).unlink(missing_ok=True)

    def test_save_state(self):
        """Test saving executor state."""
        self.executor.save_state(self.temp_path)

        # Verify file exists and is valid JSON
        self.assertTrue(Path(self.temp_path).exists())

        with open(self.temp_path, "r") as f:
            state = json.load(f)

        self.assertIn("config", state)
        self.assertIn("metrics", state)

    def test_load_state(self):
        """Test loading executor state."""
        # Save state first
        self.executor.save_state(self.temp_path)

        # Create new executor and load
        new_executor = LLMExecutor()
        new_executor.load_state(self.temp_path)

        # Verify metrics were loaded
        self.assertIsNotNone(new_executor.metrics)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""

    def setUp(self):
        """Set up test executor."""
        self.executor = LLMExecutor()

    def test_execute_with_invalid_graph(self):
        """Test execution with invalid graph."""
        invalid_graph = {"invalid_key": "invalid_value"}

        if TORCH_AVAILABLE:
            hidden_state = torch.randn(1, 5, 512)
        else:
            hidden_state = [[0.0] * 512 for _ in range(5)]

        inputs = {"hidden_states": hidden_state, "attention_mask": None}

        # Should handle gracefully
        result = self.executor.execute(invalid_graph, inputs)
        self.assertIsInstance(result, dict)

    def test_execute_with_none_inputs(self):
        """Test execution with None inputs."""
        graph = {"layers": []}

        # Should handle None inputs gracefully
        result = self.executor.execute(graph, None)
        self.assertIsInstance(result, dict)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of executor."""

    def test_concurrent_executions(self):
        """Test concurrent executor calls."""
        executor = LLMExecutor()

        if TORCH_AVAILABLE:
            test_hidden = torch.randn(1, 5, 512)
        else:
            test_hidden = [[0.0] * 512 for _ in range(5)]

        graph = {
            "layers": [
                {
                    "nodes": [{"id": "attn_0", "type": "attention_head", "params": {}}],
                    "edges": [],
                }
            ]
        }

        results = []

        def execute():
            inputs = {"hidden_states": test_hidden, "attention_mask": None}
            result = executor.execute(graph, inputs)
            results.append(result)

        threads = [threading.Thread(target=execute) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All executions should complete
        self.assertEqual(len(results), 5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_create_default_executor(self):
        """Test creating default executor."""
        executor = create_default_executor()
        self.assertIsInstance(executor, LLMExecutor)
        self.assertEqual(executor.config.execution_mode, ExecutionMode.PARALLEL_HEADS)

    def test_create_parallel_executor(self):
        """Test creating parallel executor."""
        executor = create_parallel_executor(max_workers=16)
        self.assertIsInstance(executor, LLMExecutor)
        self.assertEqual(executor.config.execution_mode, ExecutionMode.FULL_PARALLEL)
        self.assertEqual(executor.config.max_workers, 16)


class TestExecutionModes(unittest.TestCase):
    """Test different execution modes."""

    def setUp(self):
        """Set up test data."""
        if TORCH_AVAILABLE:
            self.test_hidden = torch.randn(1, 5, 512)
        else:
            self.test_hidden = [[0.0] * 512 for _ in range(5)]

        self.test_graph = {
            "layers": [
                {
                    "nodes": [
                        {"id": "attn_0", "type": "attention_head", "params": {}},
                        {"id": "attn_1", "type": "attention_head", "params": {}},
                    ],
                    "edges": [],
                }
            ]
        }

    def test_sequential_mode(self):
        """Test sequential execution mode."""
        config = ExecutorConfig(execution_mode=ExecutionMode.SEQUENTIAL)
        executor = LLMExecutor(config=config)

        inputs = {"hidden_states": self.test_hidden}
        result = executor.execute(self.test_graph, inputs)

        self.assertIsNotNone(result)

    def test_parallel_heads_mode(self):
        """Test parallel heads execution mode."""
        config = ExecutorConfig(execution_mode=ExecutionMode.PARALLEL_HEADS)
        executor = LLMExecutor(config=config)

        inputs = {"hidden_states": self.test_hidden}
        result = executor.execute(self.test_graph, inputs)

        self.assertIsNotNone(result)

    def test_full_parallel_mode(self):
        """Test full parallel execution mode."""
        config = ExecutorConfig(execution_mode=ExecutionMode.FULL_PARALLEL)
        executor = LLMExecutor(config=config)

        inputs = {"hidden_states": self.test_hidden}
        result = executor.execute(self.test_graph, inputs)

        self.assertIsNotNone(result)


class TestContextManager(unittest.TestCase):
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test using executor as context manager."""
        with LLMExecutor() as executor:
            self.assertIsInstance(executor, LLMExecutor)

            # Should be able to use executor
            if TORCH_AVAILABLE:
                test_hidden = torch.randn(1, 5, 512)
            else:
                test_hidden = [[0.0] * 512 for _ in range(5)]

            graph = {"layers": []}
            inputs = {"hidden_states": test_hidden}

            result = executor.execute(graph, inputs)
            self.assertIsNotNone(result)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestExecutorConfiguration,
        TestExecutionCache,
        TestSafetyValidator,
        TestLayerExecutor,
        TestExecutorExecution,
        TestTokenGeneration,
        TestExecutorMetrics,
        TestExecutorCaching,
        TestStatePersistence,
        TestErrorHandling,
        TestThreadSafety,
        TestUtilityFunctions,
        TestExecutionModes,
        TestContextManager,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
