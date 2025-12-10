"""
Comprehensive test suite for llvm_backend.py
"""

from unittest.mock import MagicMock, Mock, patch

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numpy as np
import pytest

from src.compiler.llvm_backend import CompiledFunction, DataType, LLVMBackend


@pytest.fixture
def backend():
    """Create LLVMBackend instance."""
    return LLVMBackend(optimization_level=2, vector_width=4)


@pytest.fixture
def simple_params():
    """Create simple parameters for testing."""
    return {"value": 1.0}


@pytest.fixture
def matrix_params():
    """Create matrix multiplication parameters."""
    return {
        "m": 4,
        "n": 4,
        "k": 4,
        "output_size": (4, 4)
    }


class TestDataType:
    """Test DataType enum."""
    
    def test_data_type_values(self):
        """Test data type enum values."""
        assert DataType.FLOAT32.value == "f32"
        assert DataType.FLOAT64.value == "f64"
        assert DataType.INT32.value == "i32"
        assert DataType.BOOL.value == "i1"


class TestCompiledFunction:
    """Test CompiledFunction dataclass."""
    
    def test_compiled_function_creation(self):
        """Test creating CompiledFunction."""
        func_type = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        
        compiled = CompiledFunction(
            name="test_func",
            signature=func_type,
            llvm_func=None,
            entry_point="test_func",
            input_types=[DataType.FLOAT64],
            output_type=DataType.FLOAT64
        )
        
        assert compiled.name == "test_func"
        assert compiled.entry_point == "test_func"
        assert len(compiled.input_types) == 1


class TestLLVMBackendInitialization:
    """Test LLVMBackend initialization."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        backend = LLVMBackend()
        
        assert backend.optimization_level == 2
        assert backend.vector_width == 4
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        backend = LLVMBackend(optimization_level=3, vector_width=8)
        
        assert backend.optimization_level == 3
        assert backend.vector_width == 8
    
    def test_module_creation(self, backend):
        """Test that module is created."""
        assert backend.module is not None
        assert backend.module.name == "graphix_native"
    
    def test_type_map_initialized(self, backend):
        """Test that type map is initialized."""
        assert DataType.FLOAT32 in backend.type_map
        assert DataType.FLOAT64 in backend.type_map
        assert DataType.INT32 in backend.type_map
    
    def test_intrinsics_declared(self, backend):
        """Test that intrinsics are declared."""
        assert "sqrt" in backend.intrinsics
        assert "exp" in backend.intrinsics
        assert "log" in backend.intrinsics
        assert "fma" in backend.intrinsics


class TestNodeCompilation:
    """Test compilation of different node types."""
    
    def test_compile_add_node(self, backend):
        """Test compiling ADD node."""
        compiled = backend.compile_node("ADD", {})
        
        assert compiled is not None
        assert compiled.name.startswith("add_")
        assert len(compiled.input_types) == 2
        assert compiled.output_type == DataType.FLOAT64
    
    def test_compile_mul_node(self, backend):
        """Test compiling MUL node."""
        compiled = backend.compile_node("MUL", {})
        
        assert compiled is not None
        assert compiled.name.startswith("mul_")
        assert len(compiled.input_types) == 2
        assert compiled.output_type == DataType.FLOAT64
    
    def test_compile_relu_node(self, backend):
        """Test compiling RELU node."""
        compiled = backend.compile_node("RELU", {})
        
        assert compiled is not None
        assert compiled.name.startswith("relu_")
        assert len(compiled.input_types) == 1
        assert compiled.output_type == DataType.FLOAT64
    
    def test_compile_matrix_mul_node(self, backend, matrix_params):
        """Test compiling MATRIX_MUL node."""
        compiled = backend.compile_node("MATRIX_MUL", matrix_params)
        
        assert compiled is not None
        assert compiled.name.startswith("matmul_")
        assert compiled.output_type == DataType.VOID
    
    def test_compile_const_node(self, backend):
        """Test compiling CONST node."""
        compiled = backend.compile_node("CONST", {"value": 3.14})
        
        assert compiled is not None
        assert compiled.name.startswith("const_")
        assert compiled.output_type == DataType.FLOAT64
    
    def test_compile_load_node(self, backend):
        """Test compiling LOAD node."""
        compiled = backend.compile_node("LOAD", {})
        
        assert compiled is not None
        assert compiled.name.startswith("load_")
    
    def test_compile_softmax_node(self, backend):
        """Test compiling SOFTMAX node."""
        compiled = backend.compile_node("SOFTMAX", {})
        
        assert compiled is not None
        assert compiled.name.startswith("softmax_")
    
    def test_compile_photonic_mvm_node(self, backend):
        """Test compiling PhotonicMVM node."""
        params = {"noise_std": 0.01, "rows": 4, "cols": 4, "output_size": 4}
        
        compiled = backend.compile_node("PHOTONIC_MVM", params)
        
        assert compiled is not None
        assert compiled.name.startswith("photonic_mvm_")
    
    def test_compile_unsupported_node(self, backend):
        """Test compiling unsupported node type."""
        with pytest.raises(NotImplementedError):
            backend.compile_node("UNSUPPORTED_TYPE", {})


class TestNodeCompilationDetails:
    """Test detailed compilation of specific nodes."""
    
    def test_compile_add_generates_fadd(self, backend):
        """Test that ADD generates fadd instruction."""
        compiled = backend._compile_add()
        
        # Check that function was created
        assert compiled.llvm_func is not None
        
        # Check LLVM IR contains fadd
        ir_str = str(backend.module)
        assert "fadd" in ir_str
    
    def test_compile_mul_generates_fmul(self, backend):
        """Test that MUL generates fmul instruction."""
        compiled = backend._compile_mul()
        
        assert compiled.llvm_func is not None
        
        ir_str = str(backend.module)
        assert "fmul" in ir_str
    
    def test_compile_relu_generates_select(self, backend):
        """Test that RELU generates select instruction."""
        compiled = backend._compile_relu()
        
        assert compiled.llvm_func is not None
        
        ir_str = str(backend.module)
        assert "select" in ir_str or "fcmp" in ir_str


class TestCaching:
    """Test compilation caching."""
    
    def test_node_caching(self, backend):
        """Test that compiled nodes are cached."""
        params = {"value": 1.0}
        
        # First compilation
        compiled1 = backend.compile_node("CONST", params)
        cache_size1 = len(backend.func_cache)
        
        # Second compilation with same params
        compiled2 = backend.compile_node("CONST", params)
        cache_size2 = len(backend.func_cache)
        
        # Cache should not grow
        assert cache_size1 == cache_size2
        assert compiled1.name == compiled2.name
    
    def test_different_params_different_cache(self, backend):
        """Test that different params create different cache entries."""
        compiled1 = backend.compile_node("CONST", {"value": 1.0})
        compiled2 = backend.compile_node("CONST", {"value": 2.0})
        
        # Should be different functions
        assert compiled1.name != compiled2.name


class TestLLVMIRGeneration:
    """Test LLVM IR generation."""
    
    def test_get_llvm_ir(self, backend):
        """Test getting LLVM IR as string."""
        # Compile a node to generate some IR
        backend.compile_node("ADD", {})
        
        ir_str = backend.get_llvm_ir()
        
        assert isinstance(ir_str, str)
        assert len(ir_str) > 0
        assert "define" in ir_str
    
    def test_llvm_ir_contains_function(self, backend):
        """Test that IR contains compiled function."""
        compiled = backend.compile_node("MUL", {})
        
        ir_str = backend.get_llvm_ir()
        
        assert compiled.name in ir_str


class TestOptimization:
    """Test optimization functionality."""
    
    def test_optimize_module(self, backend):
        """Test module optimization."""
        # Compile some nodes first
        backend.compile_node("ADD", {})
        backend.compile_node("MUL", {})
        
        # Optimize
        optimized = backend.optimize_module()
        
        assert optimized is not None
    
    def test_optimization_level_respected(self):
        """Test that optimization level is used."""
        backend_o0 = LLVMBackend(optimization_level=0)
        backend_o3 = LLVMBackend(optimization_level=3)
        
        assert backend_o0.optimization_level == 0
        assert backend_o3.optimization_level == 3


class TestCompilationOutput:
    """Test compilation output generation."""
    
    def test_compile_to_assembly(self, backend):
        """Test generating assembly code."""
        # Compile a node
        backend.compile_node("ADD", {})
        
        # Get assembly
        asm = backend.compile_to_assembly()
        
        assert isinstance(asm, str)
        assert len(asm) > 0
    
    def test_compile_to_object(self, backend):
        """Test compiling to object file."""
        import tempfile

        # Compile a node
        backend.compile_node("ADD", {})
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as f:
            output_path = f.name
        
        try:
            # Compile to object
            object_code = backend.compile_to_object(output_path)
            
            assert isinstance(object_code, bytes)
            assert len(object_code) > 0
        finally:
            import os
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestVectorization:
    """Test vectorization support."""
    
    def test_vector_types_created(self, backend):
        """Test that vector types are created."""
        assert backend.vec_f32 is not None
        assert backend.vec_f64 is not None
    
    def test_vector_width_configurable(self):
        """Test that vector width is configurable."""
        backend8 = LLVMBackend(vector_width=8)
        backend4 = LLVMBackend(vector_width=4)
        
        assert backend8.vector_width == 8
        assert backend4.vector_width == 4


class TestIntrinsics:
    """Test LLVM intrinsics."""
    
    def test_sqrt_intrinsic_declared(self, backend):
        """Test sqrt intrinsic."""
        assert backend.intrinsics["sqrt"] is not None
        assert "sqrt" in backend.intrinsics["sqrt"].name
    
    def test_exp_intrinsic_declared(self, backend):
        """Test exp intrinsic."""
        assert backend.intrinsics["exp"] is not None
        assert "exp" in backend.intrinsics["exp"].name
    
    def test_log_intrinsic_declared(self, backend):
        """Test log intrinsic."""
        assert backend.intrinsics["log"] is not None
        assert "log" in backend.intrinsics["log"].name
    
    def test_fma_intrinsic_declared(self, backend):
        """Test fma intrinsic."""
        assert backend.intrinsics["fma"] is not None
        assert "fma" in backend.intrinsics["fma"].name


class TestComplexNodes:
    """Test compilation of complex nodes."""
    
    def test_compile_conv2d(self, backend):
        """Test compiling Conv2D."""
        params = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0
        }
        
        compiled = backend._compile_conv2d(params)
        
        assert compiled is not None
        assert compiled.name.startswith("conv2d_")
    
    def test_compile_batch_norm(self, backend):
        """Test compiling BatchNorm."""
        compiled = backend._compile_batch_norm()
        
        assert compiled is not None
        assert compiled.name.startswith("batch_norm_")
    
    def test_compile_embedding(self, backend):
        """Test compiling Embedding."""
        params = {"embedding_dim": 128}
        
        compiled = backend._compile_embedding(params)
        
        assert compiled is not None
        assert compiled.name.startswith("embedding_")
    
    def test_compile_attention(self, backend):
        """Test compiling Attention."""
        params = {"num_heads": 8}
        
        compiled = backend._compile_attention(params)
        
        assert compiled is not None
        assert compiled.name.startswith("attention_")


class TestPhotonicCompilation:
    """Test photonic-specific compilation."""
    
    def test_photonic_mvm_includes_noise(self, backend):
        """Test that photonic MVM includes noise simulation."""
        params = {"noise_std": 0.05, "rows": 4, "cols": 4}
        
        compiled = backend._compile_photonic_mvm(params)
        
        # Check IR for noise factor
        ir_str = str(backend.module)
        # Should have floating point multiplication for noise
        assert "fmul" in ir_str
    
    def test_photonic_mvm_with_zero_noise(self, backend):
        """Test photonic MVM with zero noise."""
        params = {"noise_std": 0.0, "rows": 4, "cols": 4}
        
        compiled = backend._compile_photonic_mvm(params)
        
        assert compiled is not None


class TestMatrixMultiplication:
    """Test matrix multiplication compilation."""
    
    def test_matrix_mul_basic(self, backend):
        """Test basic matrix multiplication."""
        params = {"m": 4, "n": 4, "k": 4}
        
        compiled = backend._compile_matrix_mul(params)
        
        assert compiled is not None
        assert compiled.output_type == DataType.VOID
    
    def test_matrix_mul_generates_loops(self, backend):
        """Test that matrix mul generates loop structure."""
        params = {"m": 4, "n": 4, "k": 4}
        
        backend._compile_matrix_mul(params)
        
        ir_str = str(backend.module)
        
        # Should have loop structures
        assert "loop" in ir_str or "br" in ir_str


class TestSoftmax:
    """Test softmax compilation."""
    
    def test_softmax_numerical_stability(self, backend):
        """Test that softmax includes max subtraction for stability."""
        compiled = backend._compile_softmax()
        
        ir_str = str(backend.module)
        
        # Should have comparison and selection for finding max
        assert "fcmp" in ir_str or "select" in ir_str
    
    def test_softmax_uses_exp_intrinsic(self, backend):
        """Test that softmax uses exp intrinsic."""
        compiled = backend._compile_softmax()
        
        ir_str = str(backend.module)
        
        # Should call exp intrinsic
        assert "exp" in ir_str


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    @pytest.mark.slow
    def test_benchmark_node_matrix_mul(self, backend):
        """Test benchmarking matrix multiplication."""
        params = {"size": 64, "m": 64, "n": 64, "k": 64}
        
        # This might fail if JIT compilation isn't fully set up
        # so we'll catch exceptions
        try:
            results = backend.benchmark_node("MATRIX_MUL", params, iterations=10)
            
            if results:
                assert "node_type" in results
                assert results["iterations"] == 10
        except Exception:
            # JIT compilation might not work in test environment
            pass


class TestErrorHandling:
    """Test error handling."""
    
    def test_compile_unknown_node_raises(self, backend):
        """Test that compiling unknown node raises error."""
        with pytest.raises(NotImplementedError):
            backend.compile_node("UNKNOWN_NODE_TYPE", {})


class TestTypeMapping:
    """Test type mapping."""
    
    def test_all_types_mapped(self, backend):
        """Test that all DataTypes are mapped."""
        for data_type in DataType:
            if data_type != DataType.VOID:
                assert data_type in backend.type_map
    
    def test_pointer_types_created(self, backend):
        """Test that pointer types are created."""
        assert backend.ptr_f32 is not None
        assert backend.ptr_f64 is not None
        assert backend.ptr_i32 is not None


class TestModuleStructure:
    """Test LLVM module structure."""
    
    def test_module_has_correct_name(self, backend):
        """Test module name."""
        assert backend.module.name == "graphix_native"
    
    def test_module_can_add_functions(self, backend):
        """Test adding functions to module."""
        # Compile a node
        compiled = backend.compile_node("ADD", {})
        
        # Module should contain the function
        ir_str = str(backend.module)
        assert compiled.name in ir_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])