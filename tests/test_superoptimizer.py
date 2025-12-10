"""
Comprehensive test suite for superoptimizer.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from superoptimizer import (KernelGenerationError, Superoptimizer,
                            SuperoptimizerError, ValidationError)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def optimizer(temp_cache_dir):
    """Create Superoptimizer instance."""
    return Superoptimizer(
        cache_dir=temp_cache_dir,
        use_llm=False,
        use_emulator=False
    )


@pytest.fixture
def sample_subgraph():
    """Create sample subgraph."""
    return {
        "nodes": [
            {"id": "n1", "type": "MVM", "params": {"shape": [4, 4]}},
            {"id": "n2", "type": "ADD", "params": {}},
            {"id": "n3", "type": "MUL", "params": {}}
        ],
        "edges": [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"}
        ]
    }


class TestSuperoptimizerInitialization:
    """Test Superoptimizer initialization."""

    def test_initialization_basic(self, temp_cache_dir):
        """Test basic initialization."""
        optimizer = Superoptimizer(cache_dir=temp_cache_dir)

        assert optimizer.cache_dir.exists()
        assert optimizer.max_retries == 3
        assert optimizer.timeout_seconds == 30

    def test_initialization_with_llm(self, temp_cache_dir):
        """Test initialization with LLM."""
        with patch('superoptimizer.LLM_AVAILABLE', True):
            with patch('superoptimizer.LLMClient'):
                optimizer = Superoptimizer(
                    cache_dir=temp_cache_dir,
                    use_llm=True
                )

                # LLM usage depends on availability
                assert isinstance(optimizer.use_llm, bool)

    def test_initialization_creates_cache_dir(self, temp_cache_dir):
        """Test that cache directory is created."""
        cache_path = Path(temp_cache_dir) / "new_cache"
        optimizer = Superoptimizer(cache_dir=str(cache_path))

        assert cache_path.exists()

    def test_supported_backends(self, optimizer):
        """Test that supported backends are defined."""
        assert "cuda" in optimizer.supported_backends
        assert "llvm" in optimizer.supported_backends
        assert "photonic" in optimizer.supported_backends
        assert "cpu" in optimizer.supported_backends


class TestKernelGeneration:
    """Test kernel generation."""

    def test_generate_kernel_basic(self, optimizer, sample_subgraph):
        """Test basic kernel generation."""
        result = optimizer.generate_kernel(sample_subgraph, "cpu")

        assert result is not None
        assert "kernel_code" in result
        assert "backend" in result
        assert result["backend"] == "cpu"

    def test_generate_kernel_empty_subgraph(self, optimizer):
        """Test generation with empty subgraph."""
        with pytest.raises(ValueError, match="cannot be empty"):
            optimizer.generate_kernel({}, "cpu")

    def test_generate_kernel_unsupported_backend(self, optimizer, sample_subgraph):
        """Test generation with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            optimizer.generate_kernel(sample_subgraph, "invalid_backend")

    def test_generate_kernel_cuda(self, optimizer, sample_subgraph):
        """Test CUDA kernel generation."""
        result = optimizer.generate_kernel(sample_subgraph, "cuda")

        assert result["kernel_code"] is not None
        assert result["backend"] == "cuda"
        assert result["metadata"]["fallback_used"] is True

    def test_generate_kernel_llvm(self, optimizer, sample_subgraph):
        """Test LLVM IR generation."""
        result = optimizer.generate_kernel(sample_subgraph, "llvm")

        assert result["kernel_code"] is not None
        assert result["backend"] == "llvm"
        assert "define" in result["kernel_code"]

    def test_generate_kernel_photonic(self, optimizer, sample_subgraph):
        """Test photonic kernel generation."""
        result = optimizer.generate_kernel(sample_subgraph, "photonic")

        assert result["kernel_code"] is not None
        assert result["backend"] == "photonic"

    def test_generate_kernel_with_optimization_level(self, optimizer, sample_subgraph):
        """Test kernel generation with optimization level."""
        result = optimizer.generate_kernel(
            sample_subgraph,
            "cpu",
            optimization_level=3
        )

        assert result["metadata"]["optimization_level"] == 3


class TestCaching:
    """Test kernel caching."""

    def test_cache_hit(self, optimizer, sample_subgraph):
        """Test cache hit."""
        # First generation - cache miss
        result1 = optimizer.generate_kernel(sample_subgraph, "cpu", use_cache=True)

        # Second generation - cache hit
        result2 = optimizer.generate_kernel(sample_subgraph, "cpu", use_cache=True)

        assert result1["kernel_code"] == result2["kernel_code"]
        assert optimizer.get_cache_stats()["hits"] >= 1

    def test_cache_disabled(self, optimizer, sample_subgraph):
        """Test with cache disabled."""
        result1 = optimizer.generate_kernel(sample_subgraph, "cpu", use_cache=False)
        result2 = optimizer.generate_kernel(sample_subgraph, "cpu", use_cache=False)

        # Both should be fresh generations
        assert optimizer.get_cache_stats()["hits"] == 0

    def test_cache_key_generation(self, optimizer, sample_subgraph):
        """Test cache key generation."""
        key1 = optimizer._make_cache_key(sample_subgraph, "cpu")
        key2 = optimizer._make_cache_key(sample_subgraph, "cpu")

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    def test_cache_different_backends(self, optimizer, sample_subgraph):
        """Test that different backends have different cache keys."""
        key_cpu = optimizer._make_cache_key(sample_subgraph, "cpu")
        key_cuda = optimizer._make_cache_key(sample_subgraph, "cuda")

        assert key_cpu != key_cuda

    def test_clear_cache(self, optimizer, sample_subgraph):
        """Test cache clearing."""
        # Generate some cached kernels
        optimizer.generate_kernel(sample_subgraph, "cpu")
        optimizer.generate_kernel(sample_subgraph, "cuda")

        count = optimizer.clear_cache()

        assert count >= 0


class TestFallbackGeneration:
    """Test fallback kernel generation."""

    def test_generate_fallback_cuda(self, optimizer, sample_subgraph):
        """Test CUDA fallback generation."""
        kernel = optimizer._generate_fallback_cuda(sample_subgraph)

        assert kernel is not None
        assert "__global__" in kernel
        assert "kernel" in kernel

    def test_generate_fallback_photonic(self, optimizer, sample_subgraph):
        """Test photonic fallback generation."""
        kernel = optimizer._generate_fallback_photonic(sample_subgraph)

        assert kernel is not None
        assert "photonic" in kernel.lower()

    def test_generate_fallback_cpu(self, optimizer, sample_subgraph):
        """Test CPU fallback generation."""
        kernel = optimizer._generate_fallback_cpu(sample_subgraph)

        assert kernel is not None
        assert "void" in kernel

    def test_generate_fallback_llvm(self, optimizer, sample_subgraph):
        """Test LLVM fallback generation."""
        kernel = optimizer._generate_fallback_llvm(sample_subgraph, "cpu")

        assert kernel is not None
        assert "define" in kernel
        assert "%" in kernel  # LLVM IR uses % for variables


class TestValidation:
    """Test kernel validation."""

    def test_validate_cuda_kernel(self, optimizer):
        """Test CUDA kernel validation."""
        valid_cuda = "__global__ void kernel() {}"
        invalid_cuda = "void kernel() {}"

        assert optimizer._validate_cuda_kernel(valid_cuda) is True
        assert optimizer._validate_cuda_kernel(invalid_cuda) is False

    def test_validate_llvm_ir(self, optimizer):
        """Test LLVM IR validation."""
        valid_llvm = """define void @func() {
            %1 = alloca i32
            store i32 0, i32* %1
            %2 = load i32, i32* %1
            ret void
        }"""

        invalid_llvm = "not llvm ir"

        assert optimizer._validate_llvm_ir(valid_llvm) is True
        assert optimizer._validate_llvm_ir(invalid_llvm) is False

    def test_validate_kernel_generic(self, optimizer):
        """Test generic kernel validation."""
        valid_code = "void kernel() { for (int i = 0; i < 10; i++) {} }"
        invalid_code = "x"

        assert optimizer._validate_kernel(valid_code, "generic") is True
        assert optimizer._validate_kernel(invalid_code, "generic") is False

    def test_validate_empty_kernel(self, optimizer):
        """Test validation of empty kernel."""
        assert optimizer._validate_kernel("", "cpu") is False
        assert optimizer._validate_kernel(None, "cpu") is False


class TestTensorSizeInference:
    """Test tensor size inference."""

    def test_infer_tensor_size_from_shape(self, optimizer):
        """Test inference from shape hint."""
        subgraph = {
            "nodes": [
                {"id": "n1", "type": "MVM", "params": {"shape": [8, 8]}}
            ],
            "edges": []
        }

        size = optimizer._infer_tensor_size(subgraph)

        assert size >= 4

    def test_infer_tensor_size_many_nodes(self, optimizer):
        """Test inference with many nodes."""
        subgraph = {
            "nodes": [{"id": f"n{i}", "type": "ADD", "params": {}} for i in range(15)],
            "edges": []
        }

        size = optimizer._infer_tensor_size(subgraph)

        assert size >= 6

    def test_infer_tensor_size_few_nodes(self, optimizer):
        """Test inference with few nodes."""
        subgraph = {
            "nodes": [{"id": "n1", "type": "ADD", "params": {}}],
            "edges": []
        }

        size = optimizer._infer_tensor_size(subgraph)

        assert size >= 4


class TestTensorOperations:
    """Test tensor operation extraction."""

    def test_extract_tensor_ops(self, optimizer):
        """Test extracting tensor operations."""
        subgraph = {
            "nodes": [
                {"id": "n1", "type": "MVM", "params": {}},
                {"id": "n2", "type": "ADD", "params": {}},
                {"id": "n3", "type": "Input", "params": {}}
            ],
            "edges": []
        }

        ops = optimizer._extract_tensor_ops(subgraph)

        assert len(ops) == 2  # MVM and ADD
        assert all(op["type"] in ["MVM", "ADD"] for op in ops)

    def test_extract_tensor_ops_empty(self, optimizer):
        """Test extracting from graph with no tensor ops."""
        subgraph = {
            "nodes": [
                {"id": "n1", "type": "Input", "params": {}},
                {"id": "n2", "type": "Output", "params": {}}
            ],
            "edges": []
        }

        ops = optimizer._extract_tensor_ops(subgraph)

        assert len(ops) == 0


class TestEmulatorIntegration:
    """Test hardware emulator integration."""

    @patch('superoptimizer.EMULATOR_AVAILABLE', True)
    @patch('superoptimizer.analog_photonic_emulator')
    def test_test_kernel_with_emulator_photonic(self, mock_emulator, optimizer, sample_subgraph):
        """Test photonic kernel with emulator."""
        mock_emulator.emulate_photonic_mvm = Mock(return_value=MagicMock(shape=(4, 1)))

        kernel_code = "void photonic_mvm() {}"
        result = optimizer.test_kernel_with_emulator(kernel_code, sample_subgraph, "photonic")

        assert result["tested"] is True

    @patch('superoptimizer.EMULATOR_AVAILABLE', False)
    def test_test_kernel_without_emulator(self, optimizer, sample_subgraph):
        """Test kernel testing without emulator for photonic backend."""
        kernel_code = "void photonic_mvm() {}"
        result = optimizer.test_kernel_with_emulator(kernel_code, sample_subgraph, "photonic")

        assert result["tested"] is False
        assert "not available" in result["errors"][0].lower()

    def test_test_kernel_cuda_simulation(self, optimizer, sample_subgraph):
        """Test CUDA kernel simulation."""
        kernel_code = "__global__ void kernel() {}"
        result = optimizer.test_kernel_with_emulator(kernel_code, sample_subgraph, "cuda")

        assert result["tested"] is True
        assert "performance" in result


class TestLLMIntegration:
    """Test LLM integration."""

    @patch('superoptimizer.LLM_AVAILABLE', True)
    def test_generate_with_llm_success(self, optimizer):
        """Test successful LLM generation."""
        mock_client = MagicMock()
        mock_client.generate = Mock(return_value="```cuda\n__global__ void kernel() {}\n```")
        optimizer.llm_client = mock_client
        optimizer.use_llm = True

        result = optimizer._generate_with_llm({"nodes": [], "edges": []}, "cuda")

        assert result is not None
        assert "__global__" in result

    @patch('superoptimizer.LLM_AVAILABLE', True)
    def test_generate_with_llm_failure(self, optimizer):
        """Test LLM generation failure."""
        mock_client = MagicMock()
        mock_client.generate = Mock(side_effect=Exception("LLM error"))
        optimizer.llm_client = mock_client
        optimizer.use_llm = True

        result = optimizer._generate_with_llm({"nodes": [], "edges": []}, "cuda")

        assert result is None

    def test_extract_code_from_response_with_markers(self, optimizer):
        """Test extracting code from response with markers."""
        response = "Here is the code:\n```cuda\ncode here\n```\nDone!"

        code = optimizer._extract_code_from_response(response, "cuda")

        assert code == "code here"

    def test_extract_code_from_response_without_markers(self, optimizer):
        """Test extracting code without markers."""
        response = "define void @func() { ret void }"

        code = optimizer._extract_code_from_response(response, "llvm")

        assert "define" in code

    def test_extract_code_from_empty_response(self, optimizer):
        """Test extracting from empty response."""
        code = optimizer._extract_code_from_response("", "cuda")

        assert code is None


class TestPromptGeneration:
    """Test prompt generation."""

    def test_make_llvm_prompt(self, optimizer, sample_subgraph):
        """Test LLVM prompt generation."""
        prompt = optimizer._make_llvm_prompt(sample_subgraph, "cpu")

        assert "LLVM IR" in prompt
        assert "cpu" in prompt.lower()
        assert len(sample_subgraph["nodes"]) > 0

    def test_make_prompt_generic(self, optimizer, sample_subgraph):
        """Test generic prompt generation."""
        prompt = optimizer._make_prompt(sample_subgraph, "cuda")

        assert "cuda" in prompt.lower()
        assert "kernel" in prompt.lower()

    def test_get_target_triple(self, optimizer):
        """Test target triple generation."""
        assert "x86_64" in optimizer._get_target_triple("cpu")
        assert "nvptx" in optimizer._get_target_triple("gpu")
        assert "arm" in optimizer._get_target_triple("arm")


class TestCacheStatistics:
    """Test cache statistics."""

    def test_get_cache_stats(self, optimizer):
        """Test getting cache statistics."""
        stats = optimizer.get_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "generations" in stats
        assert "failures" in stats

    def test_cache_stats_update(self, optimizer, sample_subgraph):
        """Test that cache stats are updated."""
        initial_stats = optimizer.get_cache_stats()

        # Generate kernel
        optimizer.generate_kernel(sample_subgraph, "cpu")

        updated_stats = optimizer.get_cache_stats()

        assert updated_stats["generations"] > initial_stats["generations"]


class TestExceptions:
    """Test custom exceptions."""

    def test_superoptimizer_error(self):
        """Test SuperoptimizerError."""
        error = SuperoptimizerError("test error")

        assert str(error) == "test error"

    def test_kernel_generation_error(self):
        """Test KernelGenerationError."""
        error = KernelGenerationError("generation failed")

        assert str(error) == "generation failed"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("validation failed")

        assert str(error) == "validation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
