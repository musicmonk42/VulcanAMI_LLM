"""
Comprehensive test suite for self_optimizer.py
"""

import asyncio
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
from self_optimizer import (OptimizationStrategy, PerformanceMetrics,
                            SelfOptimizer)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing."""
    with patch('self_optimizer.psutil') as mock:
        # Mock Process
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=1024*1024*100)  # 100MB
        mock_process.cpu_percent.return_value = 50.0
        
        mock.Process.return_value = mock_process
        mock.cpu_count.return_value = 8
        
        yield mock


@pytest.fixture
def optimizer(mock_psutil):
    """Create SelfOptimizer instance with mocked psutil."""
    return SelfOptimizer(
        target_latency_ms=100,
        target_memory_mb=1000,
        optimization_interval_s=1,
        enable_auto_tune=True
    )


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=200.0,
            memory_mb=500.0,
            cpu_percent=30.0,
            gpu_percent=20.0
        )
        
        assert metrics.latency_ms == 50.0
        assert metrics.throughput_ops == 200.0
        assert metrics.memory_mb == 500.0
        assert metrics.cpu_percent == 30.0
    
    def test_metrics_score_calculation(self):
        """Test performance score calculation."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        score = metrics.score()
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_metrics_score_lower_better(self):
        """Test that lower latency gives better score."""
        metrics_low = PerformanceMetrics(
            latency_ms=10.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        metrics_high = PerformanceMetrics(
            latency_ms=100.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        assert metrics_low.score() > metrics_high.score()
    
    def test_metrics_timestamp(self):
        """Test that timestamp is set."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        assert metrics.timestamp > 0


class TestOptimizationStrategy:
    """Test OptimizationStrategy dataclass."""
    
    def test_strategy_creation(self):
        """Test creating OptimizationStrategy."""
        strategy = OptimizationStrategy(
            name="test_strategy",
            enabled=True,
            priority=5,
            parameters={"param1": 10}
        )
        
        assert strategy.name == "test_strategy"
        assert strategy.enabled is True
        assert strategy.priority == 5
    
    def test_strategy_effectiveness_no_data(self):
        """Test effectiveness with no data."""
        strategy = OptimizationStrategy(name="test")
        
        effectiveness = strategy.effectiveness()
        
        assert effectiveness == 0.5  # Unknown effectiveness
    
    def test_strategy_effectiveness_with_data(self):
        """Test effectiveness calculation."""
        strategy = OptimizationStrategy(name="test")
        strategy.success_count = 7
        strategy.failure_count = 3
        
        effectiveness = strategy.effectiveness()
        
        assert effectiveness == 0.7


class TestSelfOptimizerInitialization:
    """Test SelfOptimizer initialization."""
    
    def test_initialization_basic(self, mock_psutil):
        """Test basic initialization."""
        optimizer = SelfOptimizer()
        
        assert optimizer.target_latency_ms > 0
        assert optimizer.enable_auto_tune is True
        assert len(optimizer.strategies) > 0
    
    def test_initialization_custom_params(self, mock_psutil):
        """Test initialization with custom parameters."""
        optimizer = SelfOptimizer(
            target_latency_ms=200,
            target_memory_mb=2000,
            optimization_interval_s=30,
            enable_auto_tune=False
        )
        
        assert optimizer.target_latency_ms == 200
        assert optimizer.target_memory_mb == 2000
        assert optimizer.optimization_interval_s == 30
        assert optimizer.enable_auto_tune is False
    
    def test_initialization_parameter_bounds(self, mock_psutil):
        """Test that parameters are bounded."""
        optimizer = SelfOptimizer(
            target_latency_ms=-100,  # Invalid
            target_memory_mb=999999,  # Too high
            optimization_interval_s=0  # Invalid
        )
        
        # Should be clamped to valid ranges
        assert optimizer.target_latency_ms >= 1
        assert optimizer.target_memory_mb <= SelfOptimizer.MAX_MEMORY_MB
        assert optimizer.optimization_interval_s >= 1
    
    def test_strategies_initialized(self, optimizer):
        """Test that all strategies are initialized."""
        expected_strategies = [
            'caching', 'batching', 'parallelization',
            'pruning', 'quantization', 'compilation'
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in optimizer.strategies
            assert isinstance(optimizer.strategies[strategy_name], OptimizationStrategy)
    
    def test_tunable_parameters_initialized(self, optimizer):
        """Test that tunable parameters are initialized."""
        assert 'batch_size' in optimizer.tunable_parameters
        assert 'num_workers' in optimizer.tunable_parameters
        assert 'cache_size' in optimizer.tunable_parameters
        assert 'learning_rate' in optimizer.tunable_parameters


class TestMetricsCollection:
    """Test metrics collection."""
    
    def test_collect_metrics(self, optimizer):
        """Test collecting metrics."""
        metrics = optimizer._collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.throughput_ops > 0
    
    def test_collect_metrics_error_handling(self, optimizer):
        """Test metrics collection with errors."""
        with patch('self_optimizer.psutil.Process', side_effect=Exception("Error")):
            metrics = optimizer._collect_metrics()
        
        # Should return safe defaults
        assert isinstance(metrics, PerformanceMetrics)
    
    def test_measure_latency_no_history(self, optimizer):
        """Test latency measurement with no history."""
        latency = optimizer._measure_latency()
        
        assert latency == 100.0  # Default
    
    def test_measure_latency_with_history(self, optimizer):
        """Test latency measurement with history."""
        # Add some metrics to history
        for i in range(5):
            metrics = PerformanceMetrics(
                latency_ms=50.0 + i*10,
                throughput_ops=100.0,
                memory_mb=500.0,
                cpu_percent=30.0
            )
            optimizer.metrics_history.append(metrics)
        
        latency = optimizer._measure_latency()
        
        assert 50.0 <= latency <= 90.0
    
    def test_measure_throughput(self, optimizer):
        """Test throughput measurement."""
        throughput = optimizer._measure_throughput()
        
        assert throughput > 0
    
    def test_get_gpu_usage_no_gpu(self, optimizer):
        """Test GPU usage when GPU not available."""
        gpu_usage = optimizer._get_gpu_usage()
        
        # Should return 0 when GPU not available
        assert gpu_usage == 0.0


class TestOptimizationDecisions:
    """Test optimization decision making."""
    
    def test_should_optimize_no_baseline(self, optimizer):
        """Test should_optimize with no baseline."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        result = optimizer._should_optimize(metrics)
        
        assert result is False
    
    def test_should_optimize_high_latency(self, optimizer):
        """Test optimization needed due to high latency."""
        optimizer.baseline_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        high_latency_metrics = PerformanceMetrics(
            latency_ms=200.0,  # Much higher than target
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        result = optimizer._should_optimize(high_latency_metrics)
        
        assert result is True
    
    def test_should_optimize_high_memory(self, optimizer):
        """Test optimization needed due to high memory."""
        optimizer.baseline_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        high_memory_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=2000.0,  # Much higher than target
            cpu_percent=30.0
        )
        
        result = optimizer._should_optimize(high_memory_metrics)
        
        assert result is True


class TestOptimizationStrategies:
    """Test optimization strategy implementations."""
    
    def test_optimize_caching_low_hit_rate(self, optimizer):
        """Test caching optimization with low hit rate."""
        params = {'cache_size': 1000}
        initial_size = params['cache_size']
        
        # Mock low hit rate
        optimizer.cache_stats['hits'] = 10
        optimizer.cache_stats['misses'] = 90
        
        result = optimizer._optimize_caching(params)
        
        assert result is True
        # Should increase cache size
        assert params['cache_size'] >= initial_size
    
    def test_optimize_caching_high_hit_rate(self, optimizer):
        """Test caching optimization with high hit rate."""
        params = {'cache_size': 5000}
        initial_size = params['cache_size']
        
        # Mock high hit rate
        optimizer.cache_stats['hits'] = 95
        optimizer.cache_stats['misses'] = 5
        
        result = optimizer._optimize_caching(params)
        
        assert result is True
        # Should decrease cache size
        assert params['cache_size'] <= initial_size
    
    def test_optimize_batching(self, optimizer):
        """Test batching optimization."""
        params = {'batch_size': 32}
        
        result = optimizer._optimize_batching(params)
        
        assert result is True
        assert params['batch_size'] >= SelfOptimizer.MIN_BATCH_SIZE
        assert params['batch_size'] <= SelfOptimizer.MAX_BATCH_SIZE
    
    def test_optimize_parallelization_low_cpu(self, optimizer):
        """Test parallelization with low CPU usage."""
        params = {'num_workers': 4}
        
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0  # Low CPU
        )
        
        initial_workers = params['num_workers']
        result = optimizer._optimize_parallelization(params)
        
        assert result is True
        # Should potentially increase workers
        assert params['num_workers'] >= initial_workers
    
    def test_optimize_parallelization_high_cpu(self, optimizer):
        """Test parallelization with high CPU usage."""
        params = {'num_workers': 8}
        
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=85.0  # High CPU
        )
        
        initial_workers = params['num_workers']
        result = optimizer._optimize_parallelization(params)
        
        assert result is True
        # Should decrease workers
        assert params['num_workers'] < initial_workers
    
    def test_optimize_pruning(self, optimizer):
        """Test pruning optimization."""
        params = {'sparsity': 0.1}
        
        result = optimizer._optimize_pruning(params)
        
        assert result is True
        assert 0.0 <= params['sparsity'] <= 0.9
    
    def test_optimize_quantization(self, optimizer):
        """Test quantization optimization."""
        params = {'bits': 8}
        
        result = optimizer._optimize_quantization(params)
        
        assert result is True
        assert 4 <= params['bits'] <= 32
    
    def test_optimize_compilation(self, optimizer):
        """Test compilation optimization."""
        params = {}
        
        result = optimizer._optimize_compilation(params)
        
        assert result is True
        assert params['optimize'] is True
        assert params['backend'] == 'jit'


class TestParameterEvaluation:
    """Test parameter evaluation functions."""
    
    def test_evaluate_batch_size_optimal(self, optimizer):
        """Test batch size evaluation at optimal value."""
        score_optimal = optimizer._evaluate_batch_size(32)
        score_suboptimal = optimizer._evaluate_batch_size(256)
        
        assert score_optimal > score_suboptimal
    
    def test_evaluate_batch_size_extreme(self, optimizer):
        """Test batch size evaluation at extremes."""
        score_very_small = optimizer._evaluate_batch_size(1)
        score_normal = optimizer._evaluate_batch_size(32)
        
        assert score_normal > score_very_small
    
    def test_evaluate_num_workers(self, optimizer):
        """Test number of workers evaluation."""
        score = optimizer._evaluate_num_workers(4)
        
        assert 0 <= score <= 100
    
    def test_evaluate_cache_size(self, optimizer):
        """Test cache size evaluation."""
        # Set some cache stats
        optimizer.cache_stats['hits'] = 70
        optimizer.cache_stats['misses'] = 30
        
        score = optimizer._evaluate_cache_size(1000)
        
        assert 0 <= score <= 100
    
    def test_evaluate_learning_rate_optimal(self, optimizer):
        """Test learning rate evaluation."""
        score_optimal = optimizer._evaluate_learning_rate(0.001)
        score_too_high = optimizer._evaluate_learning_rate(1.0)
        
        assert score_optimal > score_too_high
    
    def test_evaluate_parameter_safe_unknown(self, optimizer):
        """Test safe evaluation with unknown parameter."""
        score = optimizer._evaluate_parameter_safe('unknown_param', 0.5)
        
        # Should return default score
        assert score == 0.0


class TestAutoTuning:
    """Test auto-tuning functionality."""
    
    def test_auto_tune_parameters(self, optimizer):
        """Test parameter auto-tuning."""
        # Set initial metrics
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        initial_batch_size = optimizer.tunable_parameters['batch_size']['current']
        
        # Run auto-tuning
        optimizer._auto_tune_parameters()
        
        # Parameters should be updated
        new_batch_size = optimizer.tunable_parameters['batch_size']['current']
        
        # Should stay within bounds
        assert (optimizer.tunable_parameters['batch_size']['min'] <= 
                new_batch_size <= 
                optimizer.tunable_parameters['batch_size']['max'])
    
    def test_auto_tune_respects_bounds(self, optimizer):
        """Test that auto-tuning respects parameter bounds."""
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        # Run multiple times
        for _ in range(10):
            optimizer._auto_tune_parameters()
        
        # All parameters should be within bounds
        for param_name, param_info in optimizer.tunable_parameters.items():
            current = param_info['current']
            assert param_info['min'] <= current <= param_info['max']


class TestCacheManagement:
    """Test cache management functionality."""
    
    def test_cache_get_miss(self, optimizer):
        """Test cache get with miss."""
        result = optimizer.cache_get("nonexistent_key")
        
        assert result is None
        assert optimizer.cache_stats['misses'] > 0
    
    def test_cache_set_and_get(self, optimizer):
        """Test cache set and get."""
        optimizer.cache_set("test_key", "test_value")
        result = optimizer.cache_get("test_key")
        
        assert result == "test_value"
        assert optimizer.cache_stats['hits'] > 0
    
    def test_cache_lru_eviction(self, optimizer):
        """Test LRU cache eviction."""
        # Fill cache beyond limit
        for i in range(150):
            optimizer.cache_set(f"key_{i}", f"value_{i}")
        
        # Evict to smaller size
        optimizer._evict_cache_lru(50)
        
        assert len(optimizer.cache) <= 50
    
    def test_cache_hit_rate(self, optimizer):
        """Test cache hit rate calculation."""
        optimizer.cache_stats['hits'] = 70
        optimizer.cache_stats['misses'] = 30
        
        hit_rate = optimizer._get_cache_hit_rate()
        
        assert hit_rate == 0.7
    
    def test_cache_hit_rate_no_accesses(self, optimizer):
        """Test hit rate with no accesses."""
        hit_rate = optimizer._get_cache_hit_rate()
        
        assert hit_rate == 0.0


class TestOptimizationCycle:
    """Test optimization cycle execution."""
    
    def test_run_optimization_cycle(self, optimizer):
        """Test running optimization cycle."""
        # Set up metrics
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        optimizer._run_optimization_cycle()
        
        # Should have attempted optimization
        assert not optimizer.is_optimizing  # Should be False after completion
    
    def test_apply_strategy(self, optimizer):
        """Test applying optimization strategy."""
        strategy = optimizer.strategies['caching']
        
        result = optimizer._apply_strategy(strategy)
        
        assert isinstance(result, bool)
    
    def test_update_strategy_priorities(self, optimizer):
        """Test updating strategy priorities."""
        strategies = list(optimizer.strategies.values())
        improvements = [(strategies[0], 10.0), (strategies[1], 5.0)]
        
        optimizer._update_strategy_priorities(improvements)
        
        # Higher improvement should have higher priority
        assert strategies[0].priority > strategies[1].priority


class TestThreading:
    """Test threading functionality."""
    
    def test_start_stop(self, optimizer):
        """Test starting and stopping optimizer."""
        optimizer.start()
        
        assert optimizer.optimization_thread is not None
        assert optimizer.optimization_thread.is_alive()
        
        time.sleep(0.1)  # Let it run briefly
        
        optimizer.stop()
        
        # Should stop cleanly
        assert optimizer.stop_event.is_set()
    
    @pytest.mark.asyncio
    async def test_async_start_stop(self, optimizer):
        """Test async start and stop."""
        await optimizer.start_async()
        
        assert hasattr(optimizer, '_async_task')
        
        await asyncio.sleep(0.1)
        
        await optimizer.stop_async()
        
        assert optimizer.stop_event.is_set()


class TestResourceCleanup:
    """Test resource cleanup."""
    
    def test_cleanup_resources_metrics_history(self, optimizer):
        """Test cleanup of metrics history."""
        # Fill history beyond threshold
        for i in range(1000):
            metrics = PerformanceMetrics(
                latency_ms=50.0,
                throughput_ops=100.0,
                memory_mb=500.0,
                cpu_percent=30.0
            )
            optimizer.metrics_history.append(metrics)
        
        optimizer._cleanup_resources()
        
        # Should have cleaned up old metrics
        assert len(optimizer.metrics_history) <= 1000
    
    def test_cleanup_resources_cache_stats(self, optimizer):
        """Test cleanup of cache stats."""
        # Set high counters
        optimizer.cache_stats['hits'] = 80000
        optimizer.cache_stats['misses'] = 20000
        
        optimizer._cleanup_resources()
        
        # Should have reset counters
        total = optimizer.cache_stats['hits'] + optimizer.cache_stats['misses']
        assert total < 100000


class TestPersistence:
    """Test state persistence."""
    
    def test_save_state(self, optimizer, temp_dir):
        """Test saving optimizer state."""
        filepath = Path(temp_dir) / "optimizer_state.pkl"
        
        # Add some data
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        optimizer.save_state(str(filepath))
        
        assert filepath.exists()
    
    def test_load_state(self, optimizer, temp_dir):
        """Test loading optimizer state."""
        filepath = Path(temp_dir) / "optimizer_state.pkl"
        
        # Save state
        optimizer.strategies['caching'].success_count = 10
        optimizer.save_state(str(filepath))
        
        # Create new optimizer and load
        new_optimizer = SelfOptimizer()
        new_optimizer.load_state(str(filepath))
        
        # Should have loaded state
        assert new_optimizer.strategies['caching'].success_count == 10
    
    def test_load_state_nonexistent(self, optimizer):
        """Test loading from nonexistent file."""
        # Should not crash
        optimizer.load_state("nonexistent_file.pkl")
    
    def test_save_load_roundtrip(self, optimizer, temp_dir):
        """Test save/load roundtrip."""
        filepath = Path(temp_dir) / "state.pkl"
        
        # Set some state
        optimizer.tunable_parameters['batch_size']['current'] = 64
        optimizer.cache_stats['hits'] = 100
        
        # Save
        optimizer.save_state(str(filepath))
        
        # Load into new instance
        new_optimizer = SelfOptimizer()
        new_optimizer.load_state(str(filepath))
        
        # Should match
        assert new_optimizer.tunable_parameters['batch_size']['current'] == 64
        assert new_optimizer.cache_stats['hits'] == 100


class TestReporting:
    """Test reporting functionality."""
    
    def test_get_optimization_report(self, optimizer):
        """Test getting optimization report."""
        report = optimizer.get_optimization_report()
        
        assert 'is_optimizing' in report
        assert 'strategies' in report
        assert 'tunable_parameters' in report
        assert 'cache_hit_rate' in report
    
    def test_report_includes_all_strategies(self, optimizer):
        """Test that report includes all strategies."""
        report = optimizer.get_optimization_report()
        
        for strategy_name in optimizer.strategies.keys():
            assert strategy_name in report['strategies']
    
    def test_report_includes_metrics(self, optimizer):
        """Test that report includes current metrics."""
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        
        report = optimizer.get_optimization_report()
        
        assert report['current_metrics'] is not None
        assert 'latency_ms' in report['current_metrics']
        assert 'score' in report['current_metrics']


class TestReset:
    """Test reset functionality."""
    
    def test_reset(self, optimizer):
        """Test resetting optimizer."""
        # Add some data
        optimizer.current_metrics = PerformanceMetrics(
            latency_ms=50.0,
            throughput_ops=100.0,
            memory_mb=500.0,
            cpu_percent=30.0
        )
        optimizer.cache_set("key", "value")
        optimizer.strategies['caching'].success_count = 10
        
        # Reset
        optimizer.reset()
        
        # Should be cleared
        assert len(optimizer.metrics_history) == 0
        assert optimizer.current_metrics is None
        assert len(optimizer.cache) == 0
        assert optimizer.strategies['caching'].success_count == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extreme_parameter_values(self, mock_psutil):
        """Test with extreme parameter values."""
        optimizer = SelfOptimizer(
            target_latency_ms=999999,
            target_memory_mb=-1000,
            optimization_interval_s=999999
        )
        
        # Should be bounded
        assert optimizer.target_latency_ms <= 10000
        assert optimizer.target_memory_mb >= SelfOptimizer.MIN_MEMORY_MB
        assert optimizer.optimization_interval_s <= 3600
    
    def test_cache_with_non_string_key(self, optimizer):
        """Test cache with non-string key."""
        optimizer.cache_set(123, "value")
        result = optimizer.cache_get(123)
        
        assert result == "value"
    
    def test_parameter_bounds_enforcement(self, optimizer):
        """Test that parameter bounds are enforced."""
        # Try to set invalid values
        params = optimizer.tunable_parameters['batch_size']
        
        # Set to extreme value
        params['current'] = 99999
        
        # Auto-tune should bring it back to valid range
        optimizer._auto_tune_parameters()
        
        assert params['min'] <= params['current'] <= params['max']


class TestConstants:
    """Test class constants."""
    
    def test_constants_defined(self):
        """Test that all constants are defined."""
        assert SelfOptimizer.MAX_MEMORY_MB > 0
        assert SelfOptimizer.MIN_MEMORY_MB > 0
        assert SelfOptimizer.MAX_CPU_PERCENT > 0
        assert SelfOptimizer.MAX_WORKERS > 0
        assert SelfOptimizer.MAX_CACHE_SIZE > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])