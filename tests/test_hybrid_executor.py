"""
Comprehensive test suite for hybrid_executor.py
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.compiler.hybrid_executor import (
    CompiledBinaryCache,
    ExecutionMetrics,
    ExecutionMode,
    GraphProfile,
    HybridExecutor,
    OptimizationLevel,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def binary_cache(temp_cache_dir):
    """Create CompiledBinaryCache instance."""
    return CompiledBinaryCache(temp_cache_dir)


@pytest.fixture
def mock_runtime():
    """Create mock runtime."""
    runtime = MagicMock()
    runtime.execute_graph = AsyncMock(return_value={"status": "success"})
    return runtime


@pytest.fixture
def mock_compiler():
    """Create mock compiler."""
    compiler = MagicMock()
    compiler.can_compile = Mock(return_value=True)
    compiler.compile_graph = Mock(return_value=b"compiled_binary")
    return compiler


@pytest.fixture
def executor(mock_runtime, temp_cache_dir, mock_compiler):
    """Create HybridExecutor instance with mocked compiler."""
    # Patch the GraphCompiler import at the correct location
    with patch("src.compiler.graph_compiler.GraphCompiler", return_value=mock_compiler):
        executor = HybridExecutor(
            runtime=mock_runtime,
            cache_dir=temp_cache_dir,
            enable_compilation=False,  # Disable for most tests
        )
        executor.compiler = mock_compiler  # Ensure compiler is set
        return executor


@pytest.fixture
def simple_graph():
    """Create simple graph for testing."""
    return {
        "id": "test_graph",
        "nodes": [
            {"id": "input1", "type": "InputNode", "params": {"value": 1.0}},
            {"id": "add1", "type": "ADD", "params": {}},
            {"id": "output1", "type": "OutputNode", "params": {"size": 10}},
        ],
        "edges": [{"from": "input1", "to": "add1"}, {"from": "add1", "to": "output1"}],
    }


class TestExecutionMode:
    """Test ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test execution mode values."""
        assert ExecutionMode.INTERPRETED.value == "interpreted"
        assert ExecutionMode.COMPILED.value == "compiled"
        assert ExecutionMode.HYBRID.value == "hybrid"
        assert ExecutionMode.AUTO.value == "auto"


class TestOptimizationLevel:
    """Test OptimizationLevel enum."""

    def test_optimization_level_values(self):
        """Test optimization level values."""
        assert OptimizationLevel.O0.value == 0
        assert OptimizationLevel.O2.value == 2
        assert OptimizationLevel.O3.value == 3


class TestExecutionMetrics:
    """Test ExecutionMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating ExecutionMetrics."""
        metrics = ExecutionMetrics(
            mode=ExecutionMode.INTERPRETED,
            duration_ms=10.5,
            memory_mb=100.0,
            cpu_percent=50.0,
            cache_hits=5,
            cache_misses=2,
        )

        assert metrics.mode == ExecutionMode.INTERPRETED
        assert metrics.duration_ms == 10.5

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ExecutionMetrics(
            mode=ExecutionMode.COMPILED,
            duration_ms=5.0,
            memory_mb=50.0,
            cpu_percent=30.0,
            cache_hits=10,
            cache_misses=0,
        )

        dict_form = metrics.to_dict()

        assert dict_form["mode"] == "compiled"
        assert dict_form["duration_ms"] == 5.0


class TestGraphProfile:
    """Test GraphProfile dataclass."""

    def test_profile_creation(self):
        """Test creating GraphProfile."""
        profile = GraphProfile(graph_id="test", graph_hash="abc123")

        assert profile.graph_id == "test"
        assert profile.graph_hash == "abc123"
        assert profile.best_mode == ExecutionMode.INTERPRETED

    def test_update_best_mode_compiled_faster(self):
        """Test updating best mode when compiled is faster."""
        profile = GraphProfile(graph_id="test", graph_hash="abc")

        # Add interpreted metrics (slower)
        for _ in range(5):
            profile.interpreted_metrics.append(
                ExecutionMetrics(
                    mode=ExecutionMode.INTERPRETED,
                    duration_ms=100.0,
                    memory_mb=50.0,
                    cpu_percent=30.0,
                    cache_hits=0,
                    cache_misses=0,
                )
            )

        # Add compiled metrics (faster)
        for _ in range(5):
            profile.compiled_metrics.append(
                ExecutionMetrics(
                    mode=ExecutionMode.COMPILED,
                    duration_ms=50.0,
                    memory_mb=40.0,
                    cpu_percent=25.0,
                    cache_hits=0,
                    cache_misses=0,
                )
            )

        profile.update_best_mode()

        assert profile.best_mode == ExecutionMode.COMPILED
        assert profile.speedup > 1.2


class TestCompiledBinaryCache:
    """Test CompiledBinaryCache."""

    def test_cache_initialization(self, binary_cache):
        """Test cache initialization."""
        assert binary_cache.cache_dir.exists()

    def test_cache_put_and_get(self, binary_cache):
        """Test storing and retrieving from cache."""
        graph_hash = "test_hash_123"
        binary = b"compiled_code_here"

        binary_cache.put(graph_hash, binary)
        retrieved = binary_cache.get(graph_hash)

        assert retrieved == binary

    def test_cache_get_nonexistent(self, binary_cache):
        """Test retrieving non-existent entry."""
        result = binary_cache.get("nonexistent")

        assert result is None

    def test_cache_cleanup(self, binary_cache):
        """Test cache cleanup."""
        # Add some entries with old timestamps
        current_time = time.time()

        # Add an old entry (10 days old)
        binary_cache.put("hash1", b"code1")
        binary_cache.cache_index["hash1"]["timestamp"] = current_time - (10 * 24 * 3600)

        # Add a recent entry (1 day old)
        binary_cache.put("hash2", b"code2")
        binary_cache.cache_index["hash2"]["timestamp"] = current_time - (1 * 24 * 3600)

        # Save index with modified timestamps
        binary_cache._save_index()

        # Cleanup entries older than 7 days
        binary_cache.cleanup(max_age_days=7)

        # Only the recent entry should remain
        assert "hash1" not in binary_cache.cache_index
        assert "hash2" in binary_cache.cache_index


class TestHybridExecutorInitialization:
    """Test HybridExecutor initialization."""

    def test_initialization_basic(self, mock_runtime, temp_cache_dir):
        """Test basic initialization."""
        with patch("src.compiler.graph_compiler.GraphCompiler"):
            executor = HybridExecutor(runtime=mock_runtime, cache_dir=temp_cache_dir)

        assert executor.runtime == mock_runtime
        assert executor.optimization_level == OptimizationLevel.O2

    def test_initialization_custom_params(self, mock_runtime, temp_cache_dir):
        """Test initialization with custom parameters."""
        with patch("src.compiler.graph_compiler.GraphCompiler"):
            executor = HybridExecutor(
                runtime=mock_runtime,
                cache_dir=temp_cache_dir,
                optimization_level=OptimizationLevel.O3,
                profile_window=20,
                enable_profiling=False,
            )

        assert executor.optimization_level == OptimizationLevel.O3
        assert executor.profile_window == 20
        assert executor.enable_profiling is False


class TestGraphHashing:
    """Test graph hashing."""

    def test_compute_graph_hash(self, executor, simple_graph):
        """Test computing graph hash."""
        hash1 = executor._compute_graph_hash(simple_graph)
        hash2 = executor._compute_graph_hash(simple_graph)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_compute_graph_hash_removes_metadata(self, executor):
        """Test that metadata is removed from hash."""
        graph1 = {
            "nodes": [{"id": "n1", "type": "ADD"}],
            "edges": [],
            "timestamp": "2025-01-01",
            "metadata": {"extra": "data"},
        }

        graph2 = {
            "nodes": [{"id": "n1", "type": "ADD"}],
            "edges": [],
            "timestamp": "2025-01-02",
            "metadata": {"different": "data"},
        }

        hash1 = executor._compute_graph_hash(graph1)
        hash2 = executor._compute_graph_hash(graph2)

        # Hashes should be same despite different metadata
        assert hash1 == hash2


class TestInterpretedExecution:
    """Test interpreted execution."""

    @pytest.mark.asyncio
    async def test_execute_interpreted(self, executor, simple_graph):
        """Test interpreted execution."""
        result, metrics = await executor._execute_interpreted(simple_graph)

        assert result["status"] == "success"
        assert metrics.mode == ExecutionMode.INTERPRETED
        assert metrics.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_interpreted_with_context(self, executor, simple_graph):
        """Test interpreted execution with context."""
        context = {"key": "value"}

        result, metrics = await executor._execute_interpreted(simple_graph, context)

        assert result is not None
        assert metrics.mode == ExecutionMode.INTERPRETED


class TestCompilation:
    """Test compilation functionality."""

    def test_compile_graph_disabled(self, executor, simple_graph):
        """Test compilation when disabled."""
        executor.enable_compilation = False
        graph_hash = executor._compute_graph_hash(simple_graph)

        result = executor._compile_graph(simple_graph, graph_hash)

        assert result is None

    def test_compile_graph_not_compilable(self, executor, simple_graph):
        """Test compilation when graph is not compilable."""
        executor.enable_compilation = True
        executor.compiler.can_compile = Mock(return_value=False)

        graph_hash = executor._compute_graph_hash(simple_graph)
        result = executor._compile_graph(simple_graph, graph_hash)

        assert result is None


class TestCompiledExecution:
    """Test compiled execution."""

    def test_prepare_inputs(self, executor, simple_graph):
        """Test preparing inputs for compiled execution."""
        inputs = executor._prepare_inputs(simple_graph)

        assert inputs is not None
        # Should have the input node value
        assert hasattr(inputs, "input1") or hasattr(inputs, "dummy")

    def test_allocate_outputs(self, executor, simple_graph):
        """Test allocating outputs for compiled execution."""
        outputs = executor._allocate_outputs(simple_graph)

        assert outputs is not None
        # Should have output buffer
        assert hasattr(outputs, "output1") or hasattr(outputs, "default")


class TestExecutionWithProfiling:
    """Test execution with profiling."""

    @pytest.mark.asyncio
    async def test_execute_with_profiling_interpreted(self, executor, simple_graph):
        """Test profiling execution in interpreted mode."""
        result = await executor.execute_with_profiling(
            simple_graph, force_mode=ExecutionMode.INTERPRETED
        )

        assert "execution_metrics" in result
        assert "execution_profile" in result
        assert result["execution_metrics"]["mode"] == "interpreted"

    @pytest.mark.asyncio
    async def test_execute_with_profiling_creates_profile(self, executor, simple_graph):
        """Test that profiling creates graph profile."""
        await executor.execute_with_profiling(simple_graph)

        graph_hash = executor._compute_graph_hash(simple_graph)

        assert graph_hash in executor.profiles
        assert executor.profiles[graph_hash].execution_count > 0

    @pytest.mark.asyncio
    async def test_execute_with_profiling_updates_history(self, executor, simple_graph):
        """Test that execution updates history."""
        initial_len = len(executor.execution_history)

        await executor.execute_with_profiling(simple_graph)

        assert len(executor.execution_history) > initial_len


class TestBenchmarking:
    """Test benchmarking functionality."""

    @pytest.mark.asyncio
    async def test_benchmark_graph(self, executor, simple_graph):
        """Test graph benchmarking."""
        # Mock compilation to avoid actual compilation
        with patch.object(executor, "_compile_graph", return_value=None):
            results = await executor.benchmark_graph(simple_graph, iterations=2)

        assert "graph_id" in results
        assert "iterations" in results
        assert "interpreted" in results
        assert len(results["interpreted"]) == 2


class TestStatistics:
    """Test statistics collection."""

    def test_get_statistics(self, executor):
        """Test getting statistics."""
        stats = executor.get_statistics()

        assert "total_executions" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "profiles_count" in stats

    @pytest.mark.asyncio
    async def test_statistics_updated_after_execution(self, executor, simple_graph):
        """Test that statistics are updated after execution."""
        initial_executions = executor.total_executions

        await executor.execute_with_profiling(simple_graph)

        assert executor.total_executions > initial_executions


class TestResourceMeasurement:
    """Test resource measurement."""

    def test_measure_resources(self, executor):
        """Test measuring resources."""
        memory, cpu = executor._measure_resources()

        assert isinstance(memory, float)
        assert isinstance(cpu, float)
        assert memory >= 0
        assert cpu >= 0


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup(self, executor):
        """Test cleanup method."""
        # Should not raise
        executor.cleanup()

        # Ensure we can call it multiple times (idempotent)
        executor.cleanup()

    def test_cleanup_in_destructor(self, mock_runtime, temp_cache_dir, mock_compiler):
        """Test cleanup in destructor."""
        with patch(
            "src.compiler.graph_compiler.GraphCompiler", return_value=mock_compiler
        ):
            executor = HybridExecutor(runtime=mock_runtime, cache_dir=temp_cache_dir)

            # Should not raise
            del executor


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
