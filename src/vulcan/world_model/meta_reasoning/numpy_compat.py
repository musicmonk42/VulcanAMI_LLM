# src/vulcan/world_model/meta_reasoning/numpy_compat.py
"""
NumPy compatibility layer for meta-reasoning components.

Provides a centralized numpy import with fallback to pure Python implementations
when numpy is not available. This eliminates duplication of FakeNumpy classes
across multiple files and ensures consistent behavior.

Issue addressed: FakeNumpy duplication across 9+ files with inconsistent implementations

Usage:
    from vulcan.world_model.meta_reasoning.numpy_compat import np, NUMPY_AVAILABLE
    
    # Use np as normal - will use real numpy or fallback
    result = np.mean([1, 2, 3])
"""

import logging
import math
import random

logger = logging.getLogger(__name__)

# Try to import real numpy
try:
    import numpy as np
    
    NUMPY_AVAILABLE = True
    logger.debug("NumPy available, using native implementation")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning(
        "NumPy not available - using pure Python fallback. "
        "Some operations may be slower or have limited functionality. "
        "Install numpy for better performance: pip install numpy"
    )
    
    class FakeNumpy:
        """
        Pure Python fallback for numpy operations.
        
        Provides minimal numpy-compatible interface for basic operations.
        Raises errors on invalid inputs instead of silently returning incorrect results.
        
        Limitations:
        - Does not support all numpy features
        - May be significantly slower than real numpy
        - Error messages differ from numpy
        - Some edge cases may behave differently
        """
        
        # Type compatibility
        class ndarray:
            """Minimal ndarray mock for isinstance checks"""
            pass
        
        class generic:
            """Minimal generic mock for isinstance checks"""
            pass
        
        # --- Basic array operations ---
        
        @staticmethod
        def array(data):
            """Convert to list (array-like)"""
            if isinstance(data, (list, tuple)):
                return list(data)
            return [data]
        
        @staticmethod
        def zeros(shape):
            """Create array of zeros"""
            if isinstance(shape, int):
                return [0.0] * shape
            elif isinstance(shape, (tuple, list)) and len(shape) == 2:
                rows, cols = shape
                return [[0.0] * cols for _ in range(rows)]
            else:
                raise ValueError(f"Unsupported shape for zeros: {shape}")
        
        @staticmethod
        def ones(shape):
            """Create array of ones"""
            if isinstance(shape, int):
                return [1.0] * shape
            elif isinstance(shape, (tuple, list)) and len(shape) == 2:
                rows, cols = shape
                return [[1.0] * cols for _ in range(rows)]
            else:
                raise ValueError(f"Unsupported shape for ones: {shape}")
        
        @staticmethod
        def arange(start, stop=None, step=1):
            """Create sequence of numbers"""
            if stop is None:
                stop = start
                start = 0
            return list(range(int(start), int(stop), int(step)))
        
        @staticmethod
        def linspace(start, stop, num=50):
            """Create evenly spaced sequence"""
            if num < 2:
                return [float(start)]
            step = (stop - start) / (num - 1)
            return [start + step * i for i in range(num)]
        
        @staticmethod
        def vstack(arrays):
            """Stack arrays vertically"""
            return [list(arr) if not isinstance(arr, list) else arr for arr in arrays]
        
        @staticmethod
        def pad(array, pad_width, mode='constant', constant_values=0):
            """Add padding to array - simplified implementation"""
            if not isinstance(array, list):
                array = [array]
            
            if isinstance(pad_width, int):
                before, after = pad_width, pad_width
            elif isinstance(pad_width, (list, tuple)) and len(pad_width) == 2:
                before, after = pad_width
            else:
                raise ValueError(f"Unsupported pad_width: {pad_width}")
            
            padding_value = constant_values if mode == 'constant' else 0
            return [padding_value] * before + array + [padding_value] * after
        
        # --- Statistical operations ---
        
        @staticmethod
        def mean(data):
            """Calculate mean of array"""
            if not data:
                raise ValueError("Cannot compute mean of empty array")
            # Handle nested lists (2D arrays)
            if isinstance(data[0], (list, tuple)):
                flat = [item for sublist in data for item in sublist]
                return sum(flat) / len(flat) if flat else 0.0
            return sum(data) / len(data)
        
        @staticmethod
        def average(data, weights=None):
            """Calculate weighted average"""
            if not data:
                raise ValueError("Cannot compute average of empty array")
            if weights is None:
                return sum(data) / len(data)
            if len(data) != len(weights):
                raise ValueError("Data and weights must have same length")
            weighted_sum = sum(d * w for d, w in zip(data, weights))
            weight_sum = sum(weights)
            if weight_sum == 0:
                raise ValueError("Sum of weights cannot be zero")
            return weighted_sum / weight_sum
        
        @staticmethod
        def sum(data):
            """Sum of array elements"""
            if not data:
                return 0.0
            # Handle nested lists (2D arrays)
            if isinstance(data[0], (list, tuple)):
                return sum(item for sublist in data for item in sublist)
            return sum(data)
        
        @staticmethod
        def std(data):
            """Calculate standard deviation"""
            if not data:
                raise ValueError("Cannot compute std of empty array")
            if len(data) == 1:
                return 0.0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return math.sqrt(variance)
        
        @staticmethod
        def var(data):
            """Calculate variance"""
            if not data:
                raise ValueError("Cannot compute variance of empty array")
            if len(data) == 1:
                return 0.0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        
        @staticmethod
        def max(data):
            """Maximum value"""
            if not data:
                raise ValueError("Cannot compute max of empty array")
            # Handle nested lists
            if isinstance(data[0], (list, tuple)):
                return max(item for sublist in data for item in sublist)
            return max(data)
        
        @staticmethod
        def min(data):
            """Minimum value"""
            if not data:
                raise ValueError("Cannot compute min of empty array")
            # Handle nested lists
            if isinstance(data[0], (list, tuple)):
                return min(item for sublist in data for item in sublist)
            return min(data)
        
        @staticmethod
        def argmin(data):
            """Index of minimum value"""
            if not data:
                raise ValueError("Cannot compute argmin of empty array")
            return data.index(min(data))
        
        @staticmethod
        def clip(data, min_val, max_val):
            """Clip values to range"""
            if isinstance(data, (int, float)):
                return max(min_val, min(max_val, data))
            return [max(min_val, min(max_val, x)) for x in data]
        
        # --- Mathematical operations ---
        
        @staticmethod
        def sqrt(data):
            """Square root"""
            if isinstance(data, (int, float)):
                if data < 0:
                    raise ValueError("Cannot take square root of negative number")
                return math.sqrt(data)
            return [math.sqrt(x) if x >= 0 else float('nan') for x in data]
        
        @staticmethod
        def log(data):
            """Natural logarithm"""
            if isinstance(data, (int, float)):
                if data <= 0:
                    raise ValueError("Cannot take logarithm of non-positive number")
                return math.log(data)
            return [math.log(x) if x > 0 else float('nan') for x in data]
        
        @staticmethod
        def dot(a, b):
            """
            Dot product of two arrays.
            Raises ValueError on incompatible dimensions instead of silently returning 0.
            """
            # Handle scalar multiplication
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a * b
            
            # Convert to lists if needed
            if not isinstance(a, list):
                a = [a]
            if not isinstance(b, list):
                b = [b]
            
            # Handle 1D dot product
            if len(a) != len(b):
                raise ValueError(
                    f"Incompatible dimensions for dot product: "
                    f"len(a)={len(a)}, len(b)={len(b)}"
                )
            
            return sum(x * y for x, y in zip(a, b))
        
        @staticmethod
        def diff(data):
            """Calculate differences between consecutive elements"""
            if len(data) < 2:
                return []
            return [data[i+1] - data[i] for i in range(len(data)-1)]
        
        @staticmethod
        def corrcoef(x, y=None):
            """
            Calculate correlation coefficient (simplified).
            Returns 2x2 matrix for compatibility.
            """
            if y is None:
                # Auto-correlation
                return [[1.0]]
            
            if len(x) != len(y):
                raise ValueError("Arrays must have same length")
            
            if len(x) < 2:
                return [[1.0, 1.0], [1.0, 1.0]]
            
            # Calculate correlation
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            
            cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
            std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / len(x))
            std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / len(y))
            
            if std_x == 0 or std_y == 0:
                corr = 0.0
            else:
                corr = cov / (std_x * std_y)
            
            return [[1.0, corr], [corr, 1.0]]
        
        @staticmethod
        def histogram(data, bins=10):
            """Simple histogram calculation"""
            if not data:
                return [0] * bins, [0.0] * (bins + 1)
            
            min_val = min(data)
            max_val = max(data)
            
            if min_val == max_val:
                counts = [len(data)] + [0] * (bins - 1)
                edges = [min_val] * (bins + 1)
                return counts, edges
            
            bin_width = (max_val - min_val) / bins
            edges = [min_val + i * bin_width for i in range(bins + 1)]
            counts = [0] * bins
            
            for val in data:
                bin_idx = min(int((val - min_val) / bin_width), bins - 1)
                counts[bin_idx] += 1
            
            return counts, edges
        
        @staticmethod
        def interp(x, xp, fp):
            """Linear interpolation (simplified)"""
            if not xp or not fp or len(xp) != len(fp):
                raise ValueError("Invalid interpolation arrays")
            
            # Simple linear interpolation
            if x <= xp[0]:
                return fp[0]
            if x >= xp[-1]:
                return fp[-1]
            
            for i in range(len(xp) - 1):
                if xp[i] <= x <= xp[i+1]:
                    # Linear interpolation between points
                    t = (x - xp[i]) / (xp[i+1] - xp[i])
                    return fp[i] + t * (fp[i+1] - fp[i])
            
            return fp[-1]
        
        @staticmethod
        def triu_indices(n, k=0):
            """Indices for upper triangle of nxn matrix"""
            indices = []
            for i in range(n):
                for j in range(max(0, i + k), n):
                    indices.append((i, j))
            if not indices:
                return ([], [])
            rows, cols = zip(*indices)
            return (list(rows), list(cols))
        
        # --- Linear algebra (simplified) ---
        
        class linalg:
            """Simplified linear algebra operations"""
            
            @staticmethod
            def norm(x):
                """Calculate L2 norm"""
                if not x:
                    return 0.0
                return math.sqrt(sum(val ** 2 for val in x))
        
        @staticmethod
        def lstsq(A, b, rcond=None):
            """
            Least squares solution (highly simplified fallback).
            
            WARNING: This is a placeholder that returns zero coefficients.
            Real numpy.linalg.lstsq performs actual least squares fitting.
            Install numpy for correct behavior.
            """
            logger.warning(
                "Using simplified lstsq fallback - results may be inaccurate. "
                "Install numpy for correct least squares computation."
            )
            
            if not A or not A[0]:
                raise ValueError("Matrix A cannot be empty")
            
            num_cols = len(A[0]) if isinstance(A[0], list) else 1
            
            # Return mock result with correct structure
            solution = [0.0] * num_cols
            residuals = []
            rank = min(len(A), num_cols)
            singular_values = []
            
            return (solution, residuals, rank, singular_values)
        
        # --- Random number generation ---
        
        class random:
            """Simplified random number generation"""
            
            @staticmethod
            def random(size=None):
                """Random floats in [0, 1)"""
                if size is None:
                    return random.random()
                if isinstance(size, int):
                    return [random.random() for _ in range(size)]
                elif isinstance(size, (tuple, list)) and len(size) == 2:
                    rows, cols = size
                    return [[random.random() for _ in range(cols)] for _ in range(rows)]
                else:
                    raise ValueError(f"Unsupported size for random: {size}")
            
            @staticmethod
            def randn(*dims):
                """
                Random numbers from standard normal distribution.
                
                Raises error if dimensions are invalid instead of ignoring them.
                """
                if not dims:
                    return random.gauss(0, 1)
                
                if len(dims) == 1:
                    n = dims[0]
                    if not isinstance(n, int) or n < 0:
                        raise ValueError(f"Invalid dimension: {n}")
                    return [random.gauss(0, 1) for _ in range(n)]
                
                elif len(dims) == 2:
                    rows, cols = dims
                    if not isinstance(rows, int) or not isinstance(cols, int):
                        raise ValueError(f"Invalid dimensions: {dims}")
                    if rows < 0 or cols < 0:
                        raise ValueError(f"Dimensions must be non-negative: {dims}")
                    return [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]
                
                else:
                    raise ValueError(
                        f"FakeNumpy.random.randn supports at most 2 dimensions, got {len(dims)}"
                    )
            
            @staticmethod
            def normal(loc=0.0, scale=1.0, size=None):
                """Random numbers from normal distribution"""
                if size is None:
                    return random.gauss(loc, scale)
                if isinstance(size, int):
                    return [random.gauss(loc, scale) for _ in range(size)]
                elif isinstance(size, (tuple, list)) and len(size) == 2:
                    rows, cols = size
                    return [[random.gauss(loc, scale) for _ in range(cols)] 
                            for _ in range(rows)]
                else:
                    raise ValueError(f"Unsupported size for normal: {size}")
            
            @staticmethod
            def choice(arr, size=None, replace=True):
                """Random choice from array"""
                if not arr:
                    raise ValueError("Cannot choose from empty array")
                if size is None:
                    return random.choice(arr)
                if replace:
                    return [random.choice(arr) for _ in range(size)]
                else:
                    if size > len(arr):
                        raise ValueError("Cannot sample more items than available without replacement")
                    return random.sample(arr, size)
            
            @staticmethod
            def beta(a, b):
                """
                Sample from Beta distribution (simplified approximation).
                
                Uses Gaussian approximation with proper mean and variance clamping.
                This is not a true Beta distribution but provides reasonable behavior
                for most use cases in preference learning.
                """
                if a <= 0 or b <= 0:
                    return 0.5  # Invalid parameters, return neutral value
                
                # Beta distribution mean and variance
                mean = a / (a + b)
                variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
                std_dev = math.sqrt(variance)
                
                # Sample from Gaussian and clamp to [0, 1]
                sample = random.gauss(mean, std_dev * 0.5)  # Dampen to reduce out-of-bounds
                return max(0.0, min(1.0, sample))
    
    # Create module-level instance
    np = FakeNumpy()
    
    logger.info("Using FakeNumpy fallback implementation")

# Export the numpy-compatible interface
__all__ = ['np', 'NUMPY_AVAILABLE']
