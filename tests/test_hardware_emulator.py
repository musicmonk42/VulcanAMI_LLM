"""
Comprehensive test suite for hardware_emulator.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from hardware_emulator import (
    HardwareEmulator,
    emulate_memristor,
    batch_emulate_memristor,
    emulate_rdkit_analog,
    emulate_pyscf_analog,
    MAX_NOISE_STD,
    MIN_NOISE_STD,
    DEFAULT_MAX_NORM,
    MAX_BATCH_SIZE,
)


@pytest.fixture
def emulator():
    """Create basic emulator."""
    return HardwareEmulator(noise_std=0.01, noise_type="gaussian")


@pytest.fixture
def matrix():
    """Create test matrix."""
    np.random.seed(42)
    return np.random.randn(4, 4).astype(np.float32)


@pytest.fixture
def vector():
    """Create test vector."""
    np.random.seed(42)
    return np.random.randn(4).astype(np.float32)


class TestHardwareEmulator:
    """Test HardwareEmulator class."""
    
    def test_initialization(self):
        """Test emulator initialization."""
        emu = HardwareEmulator(noise_std=0.05, noise_type="uniform", debug=True)
        
        assert emu.noise_std == 0.05
        assert emu.noise_type == "uniform"
        assert emu.debug is True
        assert len(emu.registry) > 0
    
    def test_invalid_noise_std_type(self):
        """Test invalid noise_std type."""
        with pytest.raises(TypeError, match="noise_std must be numeric"):
            HardwareEmulator(noise_std="invalid")
    
    def test_invalid_noise_std_range_low(self):
        """Test noise_std below minimum."""
        with pytest.raises(ValueError, match="noise_std must be in"):
            HardwareEmulator(noise_std=-0.1)
    
    def test_invalid_noise_std_range_high(self):
        """Test noise_std above maximum."""
        with pytest.raises(ValueError, match="noise_std must be in"):
            HardwareEmulator(noise_std=1.5)
    
    def test_invalid_noise_type(self):
        """Test invalid noise type."""
        with pytest.raises(ValueError, match="noise_type must be"):
            HardwareEmulator(noise_type="invalid")
    
    def test_register_op(self, emulator):
        """Test operation registration."""
        def custom_op(x):
            return x * 2
        
        emulator.register_op("custom", custom_op)
        
        assert "custom" in emulator.registry
        assert emulator.registry["custom"] == custom_op
    
    def test_register_op_invalid_name(self, emulator):
        """Test registering with invalid name."""
        with pytest.raises(ValueError, match="non-empty string"):
            emulator.register_op("", lambda x: x)
    
    def test_register_op_not_callable(self, emulator):
        """Test registering non-callable."""
        with pytest.raises(TypeError, match="must be callable"):
            emulator.register_op("test", "not callable")
    
    def test_inject_noise_gaussian(self, emulator):
        """Test Gaussian noise injection."""
        arr = np.ones((10, 10))
        noisy = emulator._inject_noise(arr, scale=1.0)
        
        assert noisy.shape == arr.shape
        assert not np.array_equal(noisy, arr)  # Should have noise
    
    def test_inject_noise_uniform(self):
        """Test uniform noise injection."""
        emu = HardwareEmulator(noise_std=0.1, noise_type="uniform")
        
        arr = np.ones((10, 10))
        noisy = emu._inject_noise(arr, scale=1.0)
        
        assert noisy.shape == arr.shape
        assert not np.array_equal(noisy, arr)
    
    def test_inject_noise_invalid_array(self, emulator):
        """Test noise injection with invalid array."""
        with pytest.raises(TypeError, match="must be numpy array"):
            emulator._inject_noise("not an array")
    
    def test_inject_noise_invalid_scale(self, emulator):
        """Test noise injection with invalid scale."""
        arr = np.ones((5, 5))
        
        with pytest.raises(TypeError, match="scale must be numeric"):
            emulator._inject_noise(arr, scale="invalid")
    
    def test_mvm_basic(self, emulator, matrix, vector):
        """Test basic matrix-vector multiplication."""
        result = emulator.mvm(matrix, vector)
        
        assert result.shape == (4,)
        assert isinstance(result, np.ndarray)
    
    def test_mvm_invalid_tensor_type(self, emulator, vector):
        """Test MVM with invalid tensor type."""
        with pytest.raises(TypeError, match="tensor must be numpy array"):
            emulator.mvm("not an array", vector)
    
    def test_mvm_invalid_vector_type(self, emulator, matrix):
        """Test MVM with invalid vector type."""
        with pytest.raises(TypeError, match="vector must be numpy array"):
            emulator.mvm(matrix, "not an array")
    
    def test_mvm_wrong_tensor_shape(self, emulator, vector):
        """Test MVM with wrong tensor dimensions."""
        tensor_1d = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="tensor must be 2D"):
            emulator.mvm(tensor_1d, vector)
    
    def test_mvm_wrong_vector_shape(self, emulator, matrix):
        """Test MVM with wrong vector dimensions."""
        vector_2d = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="vector must be 1D"):
            emulator.mvm(matrix, vector_2d)
    
    def test_mvm_shape_mismatch(self, emulator):
        """Test MVM with incompatible shapes."""
        matrix = np.random.randn(4, 5)
        vector = np.random.randn(3)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            emulator.mvm(matrix, vector)
    
    def test_mvm_with_custom_max_norm(self, emulator, matrix, vector):
        """Test MVM with custom max_norm."""
        result = emulator.mvm(matrix, vector, max_norm=5.0)
        
        assert result.shape == (4,)
    
    def test_matmul_basic(self, emulator):
        """Test matrix-matrix multiplication."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        
        result = emulator.matmul(a, b)
        
        assert result.shape == (3, 5)
    
    def test_matmul_invalid_a_type(self, emulator):
        """Test matmul with invalid first matrix type."""
        b = np.random.randn(4, 5)
        
        with pytest.raises(TypeError, match="a must be numpy array"):
            emulator.matmul("not an array", b)
    
    def test_matmul_invalid_b_type(self, emulator):
        """Test matmul with invalid second matrix type."""
        a = np.random.randn(3, 4)
        
        with pytest.raises(TypeError, match="b must be numpy array"):
            emulator.matmul(a, "not an array")
    
    def test_matmul_wrong_dimensions(self, emulator):
        """Test matmul with wrong dimensions."""
        a = np.array([1, 2, 3])
        b = np.random.randn(3, 4)
        
        with pytest.raises(ValueError, match="must be at least 2D"):
            emulator.matmul(a, b)
    
    def test_matmul_shape_mismatch(self, emulator):
        """Test matmul with incompatible shapes."""
        a = np.random.randn(3, 4)
        b = np.random.randn(5, 6)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            emulator.matmul(a, b)
    
    def test_emulate_mvm(self, emulator, matrix, vector):
        """Test emulate with mvm operation."""
        result = emulator.emulate("mvm", matrix, vector)
        
        assert result.shape == (4,)
    
    def test_emulate_matmul(self, emulator):
        """Test emulate with matmul operation."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        
        result = emulator.emulate("matmul", a, b)
        
        assert result.shape == (3, 5)
    
    def test_emulate_invalid_op_type_type(self, emulator, matrix):
        """Test emulate with invalid op_type type."""
        with pytest.raises(TypeError, match="op_type must be string"):
            emulator.emulate(123, matrix)
    
    def test_emulate_unknown_operation(self, emulator, matrix):
        """Test emulate with unknown operation."""
        with pytest.raises(ValueError, match="Unknown op_type"):
            emulator.emulate("nonexistent_op", matrix)
    
    def test_batch_emulate_mvm(self, emulator):
        """Test batch MVM emulation."""
        batch_matrices = [np.random.randn(4, 4) for _ in range(3)]
        batch_vectors = [np.random.randn(4) for _ in range(3)]
        
        results = emulator.batch_emulate("mvm", batch_matrices, batch_vectors)
        
        assert len(results) == 3
        assert all(r.shape == (4,) for r in results)
    
    def test_batch_emulate_mvm_optimized(self, emulator):
        """Test optimized batch MVM with einsum."""
        batch_matrices = np.random.randn(5, 4, 4)
        batch_vectors = np.random.randn(5, 4)
        
        result = emulator.batch_emulate("mvm", batch_matrices, batch_vectors)
        
        assert result.shape == (5, 4)
    
    def test_batch_emulate_empty_tensors(self, emulator):
        """Test batch emulate with empty tensors."""
        with pytest.raises(ValueError, match="tensors cannot be empty"):
            emulator.batch_emulate("mvm", [])
    
    def test_batch_emulate_too_large(self, emulator):
        """Test batch emulate exceeding max batch size."""
        large_batch = [np.random.randn(2, 2) for _ in range(MAX_BATCH_SIZE + 1)]
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            emulator.batch_emulate("mvm", large_batch)
    
    def test_batch_emulate_mvm_no_vectors(self, emulator):
        """Test batch MVM without vectors."""
        batch_matrices = [np.random.randn(4, 4) for _ in range(3)]
        
        with pytest.raises(ValueError, match="vectors required"):
            emulator.batch_emulate("mvm", batch_matrices)
    
    def test_batch_emulate_mvm_length_mismatch(self, emulator):
        """Test batch MVM with mismatched lengths."""
        batch_matrices = [np.random.randn(4, 4) for _ in range(3)]
        batch_vectors = [np.random.randn(4) for _ in range(2)]
        
        with pytest.raises(ValueError, match="Length mismatch"):
            emulator.batch_emulate("mvm", batch_matrices, batch_vectors)
    
    def test_batch_emulate_mvm_shape_mismatch(self, emulator):
        """Test batch MVM with shape mismatch."""
        batch_matrices = np.random.randn(3, 4, 5)
        batch_vectors = np.random.randn(3, 4)  # Should be 5
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            emulator.batch_emulate("mvm", batch_matrices, batch_vectors)
    
    def test_emulate_rdkit_analog_without_rdkit(self, emulator):
        """Test RDKit analog without RDKit installed."""
        result = emulator.emulate_rdkit_analog("CCO", resistance=1.0)
        
        # Should return fallback value
        assert isinstance(result, (int, float))
    
    def test_emulate_rdkit_analog_invalid_smiles_type(self, emulator):
        """Test RDKit with invalid SMILES type."""
        with pytest.raises(TypeError, match="smiles must be string"):
            emulator.emulate_rdkit_analog(123)
    
    def test_emulate_rdkit_analog_invalid_resistance_type(self, emulator):
        """Test RDKit with invalid resistance type."""
        with pytest.raises(TypeError, match="resistance must be numeric"):
            emulator.emulate_rdkit_analog("CCO", resistance="invalid")
    
    def test_emulate_rdkit_analog_with_tensor(self, emulator):
        """Test RDKit analog with tensor."""
        tensor = np.random.randn(3, 3)
        result = emulator.emulate_rdkit_analog("CCO", tensor=tensor)
        
        # Should return array with same shape
        assert hasattr(result, 'shape') or isinstance(result, (int, float))
    
    def test_emulate_pyscf_analog_without_pyscf(self, emulator):
        """Test PySCF analog without PySCF installed."""
        result = emulator.emulate_pyscf_analog()
        
        # Should return fallback value
        assert isinstance(result, (int, float))
    
    def test_emulate_pyscf_analog_invalid_atom_config_type(self, emulator):
        """Test PySCF with invalid atom_config type."""
        with pytest.raises(TypeError, match="atom_config must be string"):
            emulator.emulate_pyscf_analog(atom_config=123)
    
    def test_emulate_pyscf_analog_invalid_basis_type(self, emulator):
        """Test PySCF with invalid basis type."""
        with pytest.raises(TypeError, match="basis must be string"):
            emulator.emulate_pyscf_analog(basis=123)


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    def test_emulate_memristor_mvm(self, matrix, vector):
        """Test memristor MVM."""
        result = emulate_memristor(matrix, op_type="mvm", vector=vector)
        
        assert result.shape == (4,)
    
    def test_emulate_memristor_mvm_no_vector(self, matrix):
        """Test memristor MVM without vector."""
        with pytest.raises(ValueError, match="vector required"):
            emulate_memristor(matrix, op_type="mvm")
    
    def test_emulate_memristor_matmul(self):
        """Test memristor matmul."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        
        result = emulate_memristor((a, b), op_type="matmul")
        
        assert result.shape == (3, 5)
    
    def test_emulate_memristor_matmul_wrong_length(self):
        """Test memristor matmul with wrong tuple length."""
        with pytest.raises(ValueError, match="requires exactly 2 arrays"):
            emulate_memristor((np.array([1, 2, 3]),), op_type="matmul")
    
    def test_emulate_memristor_matmul_not_tuple(self, matrix):
        """Test memristor matmul without tuple."""
        with pytest.raises(ValueError, match="requires tuple"):
            emulate_memristor(matrix, op_type="matmul")
    
    def test_emulate_memristor_invalid_op_with_tuple(self):
        """Test memristor with unsupported op and tuple."""
        with pytest.raises(ValueError, match="Unsupported operation"):
            emulate_memristor((np.array([1]), np.array([2])), op_type="unknown")
    
    def test_batch_emulate_memristor_mvm(self):
        """Test batch memristor MVM."""
        batch_matrices = [np.random.randn(4, 4) for _ in range(3)]
        batch_vectors = [np.random.randn(4) for _ in range(3)]
        
        results = batch_emulate_memristor(
            batch_matrices,
            op_type="mvm",
            vectors=batch_vectors
        )
        
        assert len(results) == 3
    
    def test_batch_emulate_memristor_mvm_no_vectors(self):
        """Test batch memristor MVM without vectors."""
        batch_matrices = [np.random.randn(4, 4) for _ in range(3)]
        
        with pytest.raises(ValueError, match="vectors required"):
            batch_emulate_memristor(batch_matrices, op_type="mvm")
    
    def test_emulate_rdkit_analog_module(self):
        """Test module-level RDKit function."""
        result = emulate_rdkit_analog("CCO", resistance=2.0)
        
        assert isinstance(result, (int, float, np.ndarray))
    
    def test_emulate_pyscf_analog_module(self):
        """Test module-level PySCF function."""
        result = emulate_pyscf_analog()
        
        assert isinstance(result, (int, float, np.ndarray))


class TestNoiseTypes:
    """Test different noise types."""
    
    def test_gaussian_noise_distribution(self):
        """Test Gaussian noise properties."""
        emu = HardwareEmulator(noise_std=0.1, noise_type="gaussian")
        
        arr = np.zeros((1000, 1000))
        noisy = emu._inject_noise(arr, scale=1.0)
        
        noise = noisy - arr
        
        # Check that noise has approximately zero mean
        assert abs(np.mean(noise)) < 0.01
        
        # Check that noise std is approximately correct
        assert 0.08 < np.std(noise) < 0.12
    
    def test_uniform_noise_distribution(self):
        """Test uniform noise properties."""
        emu = HardwareEmulator(noise_std=0.1, noise_type="uniform")
        
        arr = np.zeros((1000, 1000))
        noisy = emu._inject_noise(arr, scale=1.0)
        
        noise = noisy - arr
        
        # Check that noise is roughly uniform
        assert abs(np.mean(noise)) < 0.01


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_noise(self):
        """Test with zero noise."""
        emu = HardwareEmulator(noise_std=0.0)
        
        matrix = np.ones((3, 3))
        vector = np.ones(3)
        
        result = emu.mvm(matrix, vector)
        
        # With zero noise, should be close to exact result
        expected = np.dot(matrix, vector)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_large_matrix(self):
        """Test with large matrix."""
        emu = HardwareEmulator(noise_std=0.01)
        
        matrix = np.random.randn(1000, 1000)
        vector = np.random.randn(1000)
        
        result = emu.mvm(matrix, vector)
        
        assert result.shape == (1000,)
    
    def test_single_element(self):
        """Test with single element."""
        emu = HardwareEmulator(noise_std=0.01)
        
        matrix = np.array([[5.0]])
        vector = np.array([2.0])
        
        result = emu.mvm(matrix, vector)
        
        assert result.shape == (1,)
        assert abs(result[0] - 10.0) < 1.0  # Should be close to 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])