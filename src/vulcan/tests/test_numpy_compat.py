"""
Tests for numpy_compat module - centralized numpy fallback implementation.

Tests that the FakeNumpy implementation provides consistent behavior and
raises appropriate errors on invalid inputs.
"""

import pytest

# Test with FakeNumpy explicitly
import sys
from unittest.mock import patch

# Force using FakeNumpy for testing
with patch.dict(sys.modules, {'numpy': None}):
    # Reimport to get FakeNumpy version
    import importlib
    import vulcan.world_model.meta_reasoning.numpy_compat as numpy_compat_module
    importlib.reload(numpy_compat_module)
    from vulcan.world_model.meta_reasoning.numpy_compat import np as fake_np


class TestFakeNumpyBasicOperations:
    """Test basic array operations"""
    
    def test_array_conversion(self):
        """Test array() converts data to list"""
        result = fake_np.array([1, 2, 3])
        assert result == [1, 2, 3]
        
        result = fake_np.array((4, 5, 6))
        assert result == [4, 5, 6]
    
    def test_zeros(self):
        """Test zeros() creates arrays of zeros"""
        result = fake_np.zeros(3)
        assert result == [0.0, 0.0, 0.0]
        
        result = fake_np.zeros((2, 3))
        assert result == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    def test_ones(self):
        """Test ones() creates arrays of ones"""
        result = fake_np.ones(3)
        assert result == [1.0, 1.0, 1.0]
        
        result = fake_np.ones((2, 2))
        assert result == [[1.0, 1.0], [1.0, 1.0]]
    
    def test_arange(self):
        """Test arange() creates sequences"""
        result = fake_np.arange(5)
        assert result == [0, 1, 2, 3, 4]
        
        result = fake_np.arange(2, 7)
        assert result == [2, 3, 4, 5, 6]
        
        result = fake_np.arange(0, 10, 2)
        assert result == [0, 2, 4, 6, 8]
    
    def test_linspace(self):
        """Test linspace() creates evenly spaced sequences"""
        result = fake_np.linspace(0, 10, 5)
        expected = [0.0, 2.5, 5.0, 7.5, 10.0]
        assert len(result) == 5
        for r, e in zip(result, expected):
            assert abs(r - e) < 0.0001


class TestFakeNumpyStatistics:
    """Test statistical operations"""
    
    def test_mean(self):
        """Test mean() calculation"""
        result = fake_np.mean([1, 2, 3, 4, 5])
        assert result == 3.0
        
        # Test with empty list should raise error
        with pytest.raises(ValueError, match="empty"):
            fake_np.mean([])
    
    def test_average_weighted(self):
        """Test average() with weights"""
        result = fake_np.average([1, 2, 3], weights=[1, 2, 1])
        expected = (1*1 + 2*2 + 3*1) / (1 + 2 + 1)
        assert abs(result - expected) < 0.0001
    
    def test_sum(self):
        """Test sum() calculation"""
        result = fake_np.sum([1, 2, 3, 4])
        assert result == 10
        
        result = fake_np.sum([])
        assert result == 0.0
    
    def test_std_and_var(self):
        """Test standard deviation and variance"""
        data = [1, 2, 3, 4, 5]
        mean = sum(data) / len(data)
        
        var_result = fake_np.var(data)
        expected_var = sum((x - mean) ** 2 for x in data) / len(data)
        assert abs(var_result - expected_var) < 0.0001
        
        std_result = fake_np.std(data)
        assert abs(std_result ** 2 - var_result) < 0.0001
    
    def test_max_min(self):
        """Test max() and min()"""
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        
        assert fake_np.max(data) == 9
        assert fake_np.min(data) == 1
        
        # Test with empty should raise error
        with pytest.raises(ValueError):
            fake_np.max([])
        
        with pytest.raises(ValueError):
            fake_np.min([])
    
    def test_argmin(self):
        """Test argmin() returns index of minimum"""
        data = [3, 1, 4, 0, 5]
        assert fake_np.argmin(data) == 3
    
    def test_clip(self):
        """Test clip() bounds values"""
        result = fake_np.clip(5, 0, 10)
        assert result == 5
        
        result = fake_np.clip(-5, 0, 10)
        assert result == 0
        
        result = fake_np.clip(15, 0, 10)
        assert result == 10
        
        result = fake_np.clip([1, 5, 10, 15], 3, 12)
        assert result == [3, 5, 10, 12]


class TestFakeNumpyMath:
    """Test mathematical operations"""
    
    def test_sqrt(self):
        """Test sqrt() calculation"""
        result = fake_np.sqrt(4)
        assert result == 2.0
        
        # Test with negative should raise error
        with pytest.raises(ValueError, match="negative"):
            fake_np.sqrt(-1)
    
    def test_log(self):
        """Test log() calculation"""
        import math
        result = fake_np.log(math.e)
        assert abs(result - 1.0) < 0.0001
        
        # Test with non-positive should raise error
        with pytest.raises(ValueError):
            fake_np.log(0)
        
        with pytest.raises(ValueError):
            fake_np.log(-1)
    
    def test_dot_product_valid(self):
        """Test dot() with valid inputs"""
        result = fake_np.dot([1, 2, 3], [4, 5, 6])
        expected = 1*4 + 2*5 + 3*6
        assert result == expected
        
        result = fake_np.dot([1, 0], [0, 1])
        assert result == 0
    
    def test_dot_product_invalid_dimensions(self):
        """Test dot() raises error on incompatible dimensions"""
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            fake_np.dot([1, 2, 3], [4, 5])
    
    def test_diff(self):
        """Test diff() calculates differences"""
        result = fake_np.diff([1, 3, 6, 10])
        assert result == [2, 3, 4]
        
        result = fake_np.diff([1])
        assert result == []
    
    def test_corrcoef(self):
        """Test corrcoef() calculates correlation"""
        # Perfect correlation
        result = fake_np.corrcoef([1, 2, 3], [2, 4, 6])
        assert abs(result[0][1] - 1.0) < 0.0001
        assert abs(result[1][0] - 1.0) < 0.0001
        
        # No correlation
        result = fake_np.corrcoef([1, 2, 3], [3, 2, 1])
        assert abs(result[0][1] - (-1.0)) < 0.0001


class TestFakeNumpyRandom:
    """Test random number generation"""
    
    def test_random_scalar(self):
        """Test random() generates scalar"""
        result = fake_np.random.random()
        assert 0 <= result < 1
    
    def test_random_array(self):
        """Test random() generates array"""
        result = fake_np.random.random(5)
        assert len(result) == 5
        assert all(0 <= x < 1 for x in result)
    
    def test_randn_scalar(self):
        """Test randn() generates scalar"""
        result = fake_np.random.randn()
        assert isinstance(result, float)
    
    def test_randn_1d(self):
        """Test randn() generates 1D array"""
        result = fake_np.random.randn(5)
        assert len(result) == 5
    
    def test_randn_2d(self):
        """Test randn() generates 2D array"""
        result = fake_np.random.randn(3, 4)
        assert len(result) == 3
        assert len(result[0]) == 4
    
    def test_randn_invalid_dimensions(self):
        """Test randn() raises error on invalid dimensions"""
        with pytest.raises(ValueError, match="Invalid dimension"):
            fake_np.random.randn(-1)
        
        with pytest.raises(ValueError, match="at most 2 dimensions"):
            fake_np.random.randn(2, 3, 4)
    
    def test_normal(self):
        """Test normal() generates from normal distribution"""
        result = fake_np.random.normal(10, 2, 100)
        assert len(result) == 100
        # Rough sanity check - mean should be close to 10
        mean = sum(result) / len(result)
        assert 8 < mean < 12  # Allow reasonable variance
    
    def test_choice(self):
        """Test choice() selects from array"""
        arr = [1, 2, 3, 4, 5]
        result = fake_np.random.choice(arr)
        assert result in arr
        
        result = fake_np.random.choice(arr, size=3)
        assert len(result) == 3
        assert all(x in arr for x in result)
        
        # Test without replacement
        result = fake_np.random.choice(arr, size=3, replace=False)
        assert len(result) == 3
        assert len(set(result)) == 3  # All unique


class TestFakeNumpyLinearAlgebra:
    """Test linear algebra operations"""
    
    def test_linalg_norm(self):
        """Test linalg.norm() calculates L2 norm"""
        result = fake_np.linalg.norm([3, 4])
        assert result == 5.0
        
        result = fake_np.linalg.norm([1, 0, 0])
        assert result == 1.0
    
    def test_lstsq_returns_correct_structure(self):
        """Test lstsq() returns correct tuple structure"""
        A = [[1, 0], [0, 1], [1, 1]]
        b = [1, 2, 3]
        
        result = fake_np.lstsq(A, b)
        
        # Should return (solution, residuals, rank, singular_values)
        assert len(result) == 4
        solution, residuals, rank, singular_values = result
        
        assert isinstance(solution, list)
        assert len(solution) == 2  # Number of columns in A
        
        # Note: FakeNumpy's lstsq is a placeholder
        # Real implementation would calculate actual least squares


class TestFakeNumpyEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_array_errors(self):
        """Test operations on empty arrays raise appropriate errors"""
        with pytest.raises(ValueError):
            fake_np.mean([])
        
        with pytest.raises(ValueError):
            fake_np.std([])
        
        with pytest.raises(ValueError):
            fake_np.var([])
    
    def test_dimension_mismatch_errors(self):
        """Test dimension mismatches raise appropriate errors"""
        with pytest.raises(ValueError):
            fake_np.dot([1, 2], [1, 2, 3])
        
        with pytest.raises(ValueError):
            fake_np.average([1, 2, 3], weights=[1, 2])
    
    def test_invalid_input_errors(self):
        """Test invalid inputs raise appropriate errors"""
        with pytest.raises(ValueError):
            fake_np.sqrt(-1)
        
        with pytest.raises(ValueError):
            fake_np.log(0)
        
        with pytest.raises(ValueError):
            fake_np.random.choice([])


class TestNumpyCompatModule:
    """Test module-level behavior"""
    
    def test_numpy_available_flag(self):
        """Test NUMPY_AVAILABLE flag is set correctly"""
        # When mocked, it should be False
        assert not numpy_compat_module.NUMPY_AVAILABLE
    
    def test_exports(self):
        """Test module exports np and NUMPY_AVAILABLE"""
        assert hasattr(numpy_compat_module, 'np')
        assert hasattr(numpy_compat_module, 'NUMPY_AVAILABLE')
        assert 'np' in numpy_compat_module.__all__
        assert 'NUMPY_AVAILABLE' in numpy_compat_module.__all__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
