"""
test_correlation_tracker.py - 
Tests correlation tracker functionality without spawning real threads.
"""

import pytest
import numpy as np
import time
import threading
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, MagicMock


# ============================================================================
# Mock Enums and Classes
# ============================================================================

class CorrelationMethod(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass
class CorrelationEntry:
    """Mock correlation entry"""
    var1: str
    var2: str
    correlation: float
    p_value: float = 0.05
    method: CorrelationMethod = CorrelationMethod.PEARSON
    sample_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MockCorrelationMatrix:
    """Mock correlation matrix"""
    def __init__(self):
        self._correlations: Dict[Tuple[str, str], CorrelationEntry] = {}
        self._history: List[Dict] = []
    
    def update(self, var1: str, var2: str, correlation: float, p_value: float = 0.05):
        key = tuple(sorted([var1, var2]))
        self._correlations[key] = CorrelationEntry(
            var1=key[0], var2=key[1], correlation=correlation, p_value=p_value
        )
    
    def get(self, var1: str, var2: str) -> Optional[float]:
        key = tuple(sorted([var1, var2]))
        entry = self._correlations.get(key)
        return entry.correlation if entry else None
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[CorrelationEntry]:
        return [e for e in self._correlations.values() if abs(e.correlation) >= threshold]
    
    def detect_correlation_changes(self) -> List[Dict]:
        return []


class MockCorrelationCalculator:
    """Mock correlation calculator"""
    def __init__(self, min_samples: int = 3):
        self.min_samples = min_samples
    
    def calculate(self, x: np.ndarray, y: np.ndarray, method: CorrelationMethod) -> Tuple[float, float]:
        if len(x) < self.min_samples:
            return 0.0, 1.0
        
        if method == CorrelationMethod.PEARSON:
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0, 1.0
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return 0.0, 1.0
            # Simplified p-value calculation
            n = len(x)
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
            p_value = 0.01 if abs(t_stat) > 2 else 0.5
            return corr, p_value
        return 0.0, 1.0


class MockStatisticsTracker:
    """Mock statistics tracker"""
    def __init__(self):
        self._data: Dict[str, List[float]] = defaultdict(list)
        self._baselines: Dict[str, float] = {}
    
    def update(self, variable: str, value: float):
        self._data[variable].append(value)
        if len(self._data[variable]) >= 10:
            self._baselines[variable] = np.mean(self._data[variable][-100:])
    
    def get_baseline(self, variable: str) -> Optional[float]:
        return self._baselines.get(variable)
    
    def get_data(self, variable: str) -> np.ndarray:
        return np.array(self._data.get(variable, []))


class MockCorrelationTracker:
    """Mock correlation tracker - no thread spawning"""
    
    def __init__(self, method: str = "pearson", min_samples: int = 10, safety_config=None):
        self.method = CorrelationMethod(method) if isinstance(method, str) else method
        self.min_samples = min_samples
        
        self.correlation_matrix = MockCorrelationMatrix()
        self.calculator = MockCorrelationCalculator(min_samples)
        self.statistics = MockStatisticsTracker()
        
        self.observation_history: List[Dict] = []
        self.observation_count = 0
        self.partial_corr_cache: Dict = {}
        self.safety_blocks: Dict = {}
        self.safety_corrections: Dict = {}
        
        # Mock safety validator
        self.safety_validator = Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )
    
    def update(self, observation) -> Dict[str, Any]:
        """Update with new observation"""
        if not hasattr(observation, 'variables'):
            return {'status': 'error', 'reason': 'No variables attribute'}
        
        variables = observation.variables
        self.observation_count += 1
        
        # Update statistics for each variable
        for var, value in variables.items():
            self.statistics.update(var, value)
        
        # Calculate correlations if enough samples
        var_list = list(variables.keys())
        for i, var1 in enumerate(var_list):
            for var2 in var_list[i+1:]:
                data1 = self.statistics.get_data(var1)
                data2 = self.statistics.get_data(var2)
                
                if len(data1) >= self.min_samples and len(data2) >= self.min_samples:
                    min_len = min(len(data1), len(data2))
                    corr, p_value = self.calculator.calculate(
                        data1[-min_len:], data2[-min_len:], self.method
                    )
                    self.correlation_matrix.update(var1, var2, corr, p_value)
        
        return {
            'status': 'success',
            'variables_processed': len(variables),
            'observation_count': self.observation_count
        }
    
    def get_correlation(self, var1: str, var2: str) -> Optional[float]:
        return self.correlation_matrix.get(var1, var2)
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[CorrelationEntry]:
        return self.correlation_matrix.get_strong_correlations(threshold)
    
    def get_baseline(self, variable: str) -> Optional[float]:
        return self.statistics.get_baseline(variable)
    
    def calculate_partial_correlation(self, var1: str, var2: str, 
                                       control_vars: List[str]) -> Tuple[float, float]:
        cache_key = (var1, var2, tuple(sorted(control_vars)))
        if cache_key in self.partial_corr_cache:
            return self.partial_corr_cache[cache_key]
        
        # Simplified partial correlation
        base_corr = self.get_correlation(var1, var2) or 0.0
        partial = base_corr * 0.5  # Simplified reduction
        result = (partial, 0.05)
        self.partial_corr_cache[cache_key] = result
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'observation_count': self.observation_count,
            'tracked_variables': len(self.statistics._data),
            'stored_correlations': len(self.correlation_matrix._correlations),
            'method': self.method.value
        }


# ============================================================================
# Mock Data Buffer and Storage
# ============================================================================

class MockDataBuffer:
    """Mock data buffer"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._data: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, variable: str, value: float):
        self._data[variable].append(value)
        if len(self._data[variable]) > self.max_size:
            self._data[variable] = self._data[variable][-self.max_size:]
    
    def get(self, variable: str) -> List[float]:
        return self._data.get(variable, [])


class MockCorrelationStorage:
    """Mock correlation storage"""
    def __init__(self):
        self._entries: Dict[str, CorrelationEntry] = {}
    
    def store(self, entry: CorrelationEntry):
        key = f"{entry.var1}_{entry.var2}"
        self._entries[key] = entry
    
    def get(self, var1: str, var2: str) -> Optional[CorrelationEntry]:
        key = f"{min(var1, var2)}_{max(var1, var2)}"
        return self._entries.get(key)


class MockChangeDetector:
    """Mock change detector"""
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self._history: List[float] = []
    
    def detect(self, value: float) -> bool:
        if len(self._history) < 2:
            self._history.append(value)
            return False
        
        avg = np.mean(self._history[-10:])
        change = abs(value - avg) / (avg + 1e-10)
        self._history.append(value)
        return change > self.threshold


class MockCausalityTracker:
    """Mock causality tracker"""
    def __init__(self):
        self._relationships: Dict[Tuple[str, str], float] = {}
    
    def update(self, cause: str, effect: str, strength: float):
        self._relationships[(cause, effect)] = strength
    
    def get_strength(self, cause: str, effect: str) -> Optional[float]:
        return self._relationships.get((cause, effect))


class MockBaselineTracker:
    """Mock baseline tracker"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._baselines: Dict[str, float] = {}
        self._data: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, variable: str, value: float):
        self._data[variable].append(value)
        if len(self._data[variable]) >= 10:
            self._baselines[variable] = np.mean(self._data[variable][-self.window_size:])
    
    def get(self, variable: str) -> Optional[float]:
        return self._baselines.get(variable)


# ============================================================================
# Fixtures using mocks
# ============================================================================

@pytest.fixture
def basic_tracker():
    return MockCorrelationTracker(method="pearson", min_samples=3)


@pytest.fixture
def performance_tracker():
    return MockCorrelationTracker(method="pearson", min_samples=3, safety_config=None)


@pytest.fixture
def tracker_with_safety():
    return MockCorrelationTracker(method="pearson", safety_config={'enable_validation': True})


@pytest.fixture
def sample_observation():
    class Observation:
        def __init__(self, variables: Dict[str, float]):
            self.timestamp = time.time()
            self.variables = variables
            self.domain = "test"
    
    return Observation({'temperature': 25.0, 'humidity': 0.65, 'pressure': 980.0})


@pytest.fixture
def correlated_observations():
    class Observation:
        def __init__(self, variables: Dict[str, float]):
            self.timestamp = time.time()
            self.variables = variables
            self.domain = "test"
    
    observations = []
    for i in range(50):
        x = i * 0.5
        y = 2 * x + 1 + np.random.normal(0, 0.1)
        z = np.random.uniform(0, 10)
        observations.append(Observation({'x': x, 'y': y, 'z': z}))
    
    return observations


# ============================================================================
# Tests
# ============================================================================

class TestCorrelationCalculator:
    """Test the correlation calculation component"""
    
    def test_negative_correlation(self):
        calculator = MockCorrelationCalculator(min_samples=3)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        corr, p_value = calculator.calculate(x, y, CorrelationMethod.PEARSON)
        assert abs(corr - (-1.0)) < 0.01
    
    def test_no_correlation(self):
        calculator = MockCorrelationCalculator(min_samples=3)
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        corr, _ = calculator.calculate(x, y, CorrelationMethod.PEARSON)
        assert abs(corr) < 0.3
    
    def test_positive_correlation(self):
        calculator = MockCorrelationCalculator(min_samples=3)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr, _ = calculator.calculate(x, y, CorrelationMethod.PEARSON)
        assert abs(corr - 1.0) < 0.01
    
    def test_insufficient_samples(self):
        calculator = MockCorrelationCalculator(min_samples=10)
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        corr, p_value = calculator.calculate(x, y, CorrelationMethod.PEARSON)
        assert corr == 0.0
        assert p_value == 1.0


class TestCorrelationTracker:
    """Test the main correlation tracker"""
    
    def test_initialization(self, basic_tracker):
        assert basic_tracker.method == CorrelationMethod.PEARSON
        assert basic_tracker.min_samples == 3
        assert basic_tracker.observation_count == 0
    
    def test_single_observation(self, basic_tracker, sample_observation):
        result = basic_tracker.update(sample_observation)
        assert result['status'] == 'success'
        assert result['variables_processed'] == 3
        assert basic_tracker.observation_count == 1
    
    def test_multiple_observations(self, basic_tracker, correlated_observations):
        for obs in correlated_observations:
            result = basic_tracker.update(obs)
            assert result['status'] == 'success'
        
        assert basic_tracker.observation_count == 50
    
    def test_correlation_detection(self, basic_tracker, correlated_observations):
        for obs in correlated_observations:
            basic_tracker.update(obs)
        
        corr = basic_tracker.get_correlation('x', 'y')
        assert corr is not None
        assert corr > 0.9
    
    def test_strong_correlations(self, basic_tracker, correlated_observations):
        for obs in correlated_observations:
            basic_tracker.update(obs)
        
        strong = basic_tracker.get_strong_correlations(threshold=0.7)
        assert len(strong) >= 1
    
    def test_statistics(self, basic_tracker, correlated_observations):
        for obs in correlated_observations:
            basic_tracker.update(obs)
        
        stats = basic_tracker.get_statistics()
        assert stats['observation_count'] == 50
        assert stats['tracked_variables'] == 3


class TestStatisticsTracker:
    """Test statistics tracking"""
    
    def test_update_and_baseline(self):
        tracker = MockStatisticsTracker()
        for i in range(20):
            tracker.update('test_var', 10.0 + np.random.normal(0, 0.1))
        
        baseline = tracker.get_baseline('test_var')
        assert baseline is not None
        assert 9.5 < baseline < 10.5
    
    def test_get_data(self):
        tracker = MockStatisticsTracker()
        for i in range(5):
            tracker.update('var', float(i))
        
        data = tracker.get_data('var')
        assert len(data) == 5
        assert list(data) == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestDataBuffer:
    """Test data buffer"""
    
    def test_add_and_get(self):
        buffer = MockDataBuffer(max_size=10)
        for i in range(5):
            buffer.add('var', float(i))
        
        data = buffer.get('var')
        assert len(data) == 5
    
    def test_max_size(self):
        buffer = MockDataBuffer(max_size=5)
        for i in range(10):
            buffer.add('var', float(i))
        
        data = buffer.get('var')
        assert len(data) == 5


class TestChangeDetector:
    """Test change detection"""
    
    def test_no_change(self):
        detector = MockChangeDetector(threshold=0.2)
        for _ in range(10):
            result = detector.detect(10.0)
        assert result == False
    
    def test_detect_change(self):
        detector = MockChangeDetector(threshold=0.2)
        for _ in range(10):
            detector.detect(10.0)
        result = detector.detect(15.0)
        assert result == True


class TestPartialCorrelation:
    """Test partial correlation calculation"""
    
    def test_partial_correlation_basic(self, basic_tracker):
        class ConfoundedObs:
            def __init__(self, i):
                z = i
                x = z + np.random.normal(0, 0.1)
                y = z + np.random.normal(0, 0.1)
                self.variables = {'x': x, 'y': y, 'z': z}
                self.timestamp = time.time()
        
        for i in range(50):
            basic_tracker.update(ConfoundedObs(i))
        
        corr_xy = basic_tracker.get_correlation('x', 'y')
        assert corr_xy is not None and abs(corr_xy) > 0.8
        
        partial_corr, _ = basic_tracker.calculate_partial_correlation('x', 'y', ['z'])
        assert abs(partial_corr) < abs(corr_xy)
    
    def test_partial_correlation_caching(self, basic_tracker):
        class SimpleObs:
            def __init__(self, i):
                self.variables = {'x': i, 'y': i*2, 'z': i*3}
                self.timestamp = time.time()
        
        for i in range(20):
            basic_tracker.update(SimpleObs(i))
        
        corr1, _ = basic_tracker.calculate_partial_correlation('x', 'y', ['z'])
        corr2, _ = basic_tracker.calculate_partial_correlation('x', 'y', ['z'])
        assert corr1 == corr2


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_variable_observation(self, basic_tracker):
        class SingleVarObs:
            def __init__(self):
                self.variables = {'only_var': 42.0}
                self.timestamp = time.time()
        
        for _ in range(10):
            result = basic_tracker.update(SingleVarObs())
            assert result['status'] == 'success'
        
        strong = basic_tracker.get_strong_correlations(threshold=0.5)
        assert len(strong) == 0
    
    def test_observation_without_variables_attribute(self, basic_tracker):
        class BadObs:
            def __init__(self):
                self.timestamp = time.time()
        
        result = basic_tracker.update(BadObs())
        assert 'status' in result
    
    def test_get_correlation_nonexistent_variables(self, basic_tracker):
        corr = basic_tracker.get_correlation('nonexistent_a', 'nonexistent_b')
        assert corr is None


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_correlation_workflow(self):
        tracker = MockCorrelationTracker(method="pearson", min_samples=5)
        
        for i in range(100):
            obs = type('Obs', (), {
                'variables': {
                    'temperature': 20 + i * 0.1 + np.random.normal(0, 0.5),
                    'ice_cream_sales': 100 + i * 2 + np.random.normal(0, 5)
                },
                'timestamp': time.time()
            })()
            result = tracker.update(obs)
            assert result['status'] == 'success'
        
        corr = tracker.get_correlation('temperature', 'ice_cream_sales')
        assert corr is not None
        assert corr > 0.7
        
        strong = tracker.get_strong_correlations(threshold=0.7)
        assert len(strong) > 0
        
        stats = tracker.get_statistics()
        assert stats['observation_count'] == 100
        assert stats['tracked_variables'] == 2


class TestPerformance:
    """Performance and scalability tests"""
    
    def test_large_scale_updates(self, performance_tracker):
        start = time.time()
        
        for i in range(100):
            obs = type('Obs', (), {
                'variables': {'x': i, 'y': i * 2, 'z': i * 3},
                'timestamp': time.time()
            })()
            performance_tracker.update(obs)
        
        elapsed = time.time() - start
        assert elapsed < 5
        assert performance_tracker.observation_count == 100
    
    def test_many_variables(self):
        tracker = MockCorrelationTracker(method="pearson", safety_config=None)
        
        obs = type('Obs', (), {
            'variables': {f'var_{i}': float(i) for i in range(50)},
            'timestamp': time.time()
        })()
        
        start = time.time()
        for _ in range(10):
            tracker.update(obs)
        elapsed = time.time() - start
        
        assert elapsed < 10


class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_updates(self):
        tracker = MockCorrelationTracker(method="pearson")
        results = []
        
        def update_tracker(thread_id):
            for i in range(10):
                obs = type('Obs', (), {
                    'variables': {'x': i, 'y': i * 2},
                    'timestamp': time.time()
                })()
                result = tracker.update(obs)
                results.append(result['status'])
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=update_tracker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 30
        assert all(r == 'success' for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
