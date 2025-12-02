"""
test_prediction_engine.py - OPTIMIZED VERSION
Comprehensive test suite for PredictionEngine

OPTIMIZED: Uses module-scoped fixtures to avoid re-initializing expensive objects.
"""

import pytest
import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple

from vulcan.world_model.prediction_engine import (
    CombinationMethod,
    Path,
    PathCluster,
    Prediction,
    PathAnalyzer,
    PathEffectCalculator,
    PathTracer,
    PathClusterer,
    MonteCarloSampler,
    PredictionCombiner,
    EnsemblePredictor
)


# ============================================================
# MODULE-SCOPED FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def simple_path():
    """Create a simple path"""
    return Path(
        nodes=['A', 'B', 'C'],
        edges=[('A', 'B', 0.8), ('B', 'C', 0.7)],
        total_strength=0.56,
        confidence=0.9
    )


@pytest.fixture(scope="module")
def complex_path():
    """Create a more complex path"""
    return Path(
        nodes=['X', 'Y', 'Z', 'W'],
        edges=[('X', 'Y', 0.9), ('Y', 'Z', 0.85), ('Z', 'W', 0.75)],
        total_strength=0.57,
        confidence=0.85,
        evidence_types=['correlation', 'intervention']
    )


@pytest.fixture(scope="module")
def correlated_paths():
    """Create correlated paths for clustering"""
    return [
        Path(
            nodes=['A', 'B', 'C'],
            edges=[('A', 'B', 0.8), ('B', 'C', 0.7)],
            total_strength=0.56
        ),
        Path(
            nodes=['A', 'B', 'D'],
            edges=[('A', 'B', 0.8), ('B', 'D', 0.65)],
            total_strength=0.52
        ),
        Path(
            nodes=['A', 'E', 'C'],
            edges=[('A', 'E', 0.75), ('E', 'C', 0.72)],
            total_strength=0.54
        )
    ]


@pytest.fixture(scope="module")
def sample_predictions():
    """Create sample predictions"""
    return [
        Prediction(expected=5.0, lower_bound=4.0, upper_bound=6.0, confidence=0.9, method="test"),
        Prediction(expected=5.5, lower_bound=4.5, upper_bound=6.5, confidence=0.85, method="test"),
        Prediction(expected=4.8, lower_bound=3.8, upper_bound=5.8, confidence=0.88, method="test")
    ]


@pytest.fixture(scope="module")
def basic_context():
    """Create basic prediction context"""
    return {'initial_values': {'A': 1.0}, 'add_noise': False}


# Module-scoped component fixtures
@pytest.fixture(scope="module")
def shared_path_analyzer():
    """Module-scoped path analyzer"""
    return PathAnalyzer()


@pytest.fixture(scope="module")
def shared_effect_calculator():
    """Module-scoped effect calculator"""
    return PathEffectCalculator()


@pytest.fixture(scope="module")
def shared_path_tracer():
    """Module-scoped path tracer"""
    return PathTracer(min_path_strength=0.1, max_path_length=5)


@pytest.fixture(scope="module")
def shared_clusterer(shared_path_analyzer):
    """Module-scoped path clusterer"""
    return PathClusterer(shared_path_analyzer)


@pytest.fixture(scope="module")
def shared_sampler():
    """Module-scoped Monte Carlo sampler"""
    return MonteCarloSampler()


@pytest.fixture(scope="module")
def shared_combiner():
    """Module-scoped prediction combiner"""
    return PredictionCombiner()


@pytest.fixture(scope="module")
def shared_ensemble_predictor():
    """Module-scoped ensemble predictor"""
    return EnsemblePredictor(default_method="weighted_quantile")


# Function-scoped fixtures for tests that modify state
@pytest.fixture
def path_analyzer():
    return PathAnalyzer()


@pytest.fixture
def effect_calculator():
    return PathEffectCalculator()


@pytest.fixture
def path_tracer():
    return PathTracer(min_path_strength=0.1, max_path_length=5)


@pytest.fixture
def clusterer(path_analyzer):
    return PathClusterer(path_analyzer)


@pytest.fixture
def sampler():
    return MonteCarloSampler()


@pytest.fixture
def combiner():
    return PredictionCombiner()


@pytest.fixture
def ensemble_predictor():
    return EnsemblePredictor(default_method="weighted_quantile")


# ============================================================
# PATH DATACLASS TESTS
# ============================================================

class TestPath:
    """Test Path dataclass"""
    
    def test_path_creation(self, simple_path):
        """Test basic path creation"""
        assert len(simple_path.nodes) == 3
        assert len(simple_path.edges) == 2
        assert simple_path.total_strength == 0.56
        assert simple_path.confidence == 0.9
    
    def test_path_length(self, simple_path):
        """Test path length calculation"""
        assert len(simple_path) == 2
    
    def test_contains_node(self, simple_path):
        """Test checking if path contains node"""
        assert simple_path.contains_node('A') == True
        assert simple_path.contains_node('B') == True
        assert simple_path.contains_node('D') == False
    
    def test_get_edge_strength(self, simple_path):
        """Test getting edge strength"""
        assert simple_path.get_edge_strength('A', 'B') == 0.8
        assert simple_path.get_edge_strength('B', 'C') == 0.7
        assert simple_path.get_edge_strength('A', 'C') is None
    
    def test_get_strengths(self, simple_path):
        """Test getting all edge strengths"""
        strengths = simple_path.get_strengths()
        assert len(strengths) == 2
        assert strengths[0] == 0.8
        assert strengths[1] == 0.7
    
    def test_strengths_property(self, simple_path):
        """Test strengths property accessor"""
        assert simple_path.strengths == [0.8, 0.7]
    
    def test_path_with_metadata(self):
        """Test path with metadata"""
        path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', 0.9)],
            total_strength=0.9,
            metadata={'source': 'test', 'domain': 'example'}
        )
        
        assert path.metadata['source'] == 'test'


# ============================================================
# PATH CLUSTER TESTS
# ============================================================

class TestPathCluster:
    """Test PathCluster dataclass"""
    
    def test_cluster_creation(self, correlated_paths):
        """Test basic cluster creation"""
        correlation_matrix = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0]
        ])
        
        cluster = PathCluster(
            paths=correlated_paths,
            correlation_matrix=correlation_matrix,
            representative_path=correlated_paths[0],
            cluster_confidence=0.85
        )
        
        assert cluster.size == 3
        assert cluster.representative_path == correlated_paths[0]
        assert cluster.cluster_confidence == 0.85
    
    def test_cluster_size_property(self, correlated_paths):
        """Test cluster size property"""
        cluster = PathCluster(
            paths=correlated_paths,
            correlation_matrix=np.eye(3),
            representative_path=correlated_paths[0],
            cluster_confidence=0.8
        )
        
        assert cluster.size == 3


# ============================================================
# PREDICTION DATACLASS TESTS
# ============================================================

class TestPrediction:
    """Test Prediction dataclass"""
    
    def test_prediction_creation(self, sample_predictions):
        """Test basic prediction creation"""
        pred = sample_predictions[0]
        
        assert pred.expected == 5.0
        assert pred.lower_bound == 4.0
        assert pred.upper_bound == 6.0
        assert pred.confidence == 0.9
    
    def test_prediction_uncertainty(self, sample_predictions):
        """Test prediction uncertainty calculation"""
        pred = sample_predictions[0]
        
        uncertainty = pred.uncertainty
        
        assert uncertainty >= 0.0
        assert uncertainty == (pred.upper_bound - pred.lower_bound) / 2
    
    def test_prediction_interval_width(self, sample_predictions):
        """Test prediction interval width"""
        pred = sample_predictions[0]
        
        width = pred.interval_width
        
        assert width == 2.0


# ============================================================
# PATH ANALYZER TESTS
# ============================================================

class TestPathAnalyzer:
    """Test PathAnalyzer class"""
    
    def test_calculate_path_confidence(self, shared_path_analyzer, simple_path):
        """Test path confidence calculation"""
        confidence = shared_path_analyzer.calculate_path_confidence(simple_path)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_path_correlation(self, shared_path_analyzer, correlated_paths):
        """Test path correlation calculation"""
        corr = shared_path_analyzer.calculate_path_correlation(
            correlated_paths[0], correlated_paths[1]
        )
        
        assert -1.0 <= corr <= 1.0
    
    def test_calculate_correlation_matrix(self, shared_path_analyzer, correlated_paths):
        """Test correlation matrix calculation"""
        matrix = shared_path_analyzer.calculate_correlation_matrix(correlated_paths)
        
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0)


# ============================================================
# PATH EFFECT CALCULATOR TESTS
# ============================================================

class TestPathEffectCalculator:
    """Test PathEffectCalculator class"""
    
    def test_calculate_effect_basic(self, shared_effect_calculator, simple_path):
        """Test basic effect calculation"""
        effect = shared_effect_calculator.calculate_effect(simple_path, 1.0, {})
        
        assert isinstance(effect, float)
    
    def test_calculate_effect_with_context(self, shared_effect_calculator, simple_path):
        """Test effect calculation with context"""
        context = {'initial_values': {'A': 2.0}}
        
        effect = shared_effect_calculator.calculate_effect(simple_path, 1.0, context)
        
        assert isinstance(effect, float)


# ============================================================
# PATH TRACER TESTS
# ============================================================

class TestPathTracer:
    """Test PathTracer class"""
    
    def test_trace_path_basic(self, shared_path_tracer, simple_path, basic_context):
        """Test basic path tracing"""
        effect = shared_path_tracer.trace_path(simple_path, 1.0, basic_context)
        
        assert isinstance(effect, float)
    
    def test_trace_path_caching(self, path_tracer, simple_path, basic_context):
        """Test that tracing is cached"""
        effect1 = path_tracer.trace_path(simple_path, 1.0, basic_context)
        effect2 = path_tracer.trace_path(simple_path, 1.0, basic_context)
        
        assert effect1 == effect2
    
    def test_clear_cache(self, path_tracer, simple_path, basic_context):
        """Test clearing cache"""
        path_tracer.trace_path(simple_path, 1.0, basic_context)
        
        initial_size = len(path_tracer.cache)
        path_tracer.clear_cache()
        
        assert len(path_tracer.cache) == 0


# ============================================================
# PATH CLUSTERER TESTS
# ============================================================

class TestPathClusterer:
    """Test PathClusterer class"""
    
    def test_cluster_paths_basic(self, shared_clusterer, correlated_paths):
        """Test basic path clustering"""
        clusters = shared_clusterer.cluster_paths(correlated_paths, n_clusters=2)
        
        assert len(clusters) <= 2
        for cluster in clusters:
            assert isinstance(cluster, PathCluster)
    
    def test_find_representative_path(self, shared_clusterer, correlated_paths):
        """Test finding representative path"""
        representative = shared_clusterer.find_representative_path(correlated_paths)
        
        assert representative in correlated_paths


# ============================================================
# MONTE CARLO SAMPLER TESTS
# ============================================================

class TestMonteCarloSampler:
    """Test MonteCarloSampler class"""
    
    def test_sample_paths(self, shared_sampler, correlated_paths, basic_context):
        """Test sampling paths"""
        samples = shared_sampler.sample_paths(
            correlated_paths, 1.0, basic_context, n_samples=10
        )
        
        assert len(samples) == 10
        assert all(isinstance(s, float) for s in samples)
    
    def test_generate_prediction(self, shared_sampler, correlated_paths, basic_context):
        """Test generating prediction from samples"""
        samples = shared_sampler.sample_paths(
            correlated_paths, 1.0, basic_context, n_samples=50
        )
        
        prediction = shared_sampler.generate_prediction(samples, correlated_paths)
        
        assert isinstance(prediction, Prediction)
        assert prediction.lower_bound <= prediction.expected <= prediction.upper_bound


# ============================================================
# PREDICTION COMBINER TESTS
# ============================================================

class TestPredictionCombiner:
    """Test PredictionCombiner class"""
    
    def test_combine_mean(self, shared_combiner, sample_predictions):
        """Test combining predictions with mean"""
        combined = shared_combiner.combine(
            sample_predictions, CombinationMethod.MEAN
        )
        
        assert isinstance(combined, Prediction)
        expected_mean = np.mean([p.expected for p in sample_predictions])
        assert abs(combined.expected - expected_mean) < 0.01
    
    def test_combine_weighted(self, shared_combiner, sample_predictions):
        """Test combining predictions with weighting"""
        combined = shared_combiner.combine(
            sample_predictions, CombinationMethod.WEIGHTED
        )
        
        assert isinstance(combined, Prediction)
    
    def test_combine_quantile(self, shared_combiner, sample_predictions):
        """Test combining with quantile method"""
        combined = shared_combiner.combine(
            sample_predictions, CombinationMethod.QUANTILE
        )
        
        assert isinstance(combined, Prediction)


# ============================================================
# ENSEMBLE PREDICTOR TESTS
# ============================================================

class TestEnsemblePredictor:
    """Test EnsemblePredictor class"""
    
    def test_predict_basic(self, shared_ensemble_predictor, simple_path, basic_context):
        """Test basic prediction"""
        prediction = shared_ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, [simple_path]
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.expected != 0.0
    
    def test_predict_multiple_paths(self, shared_ensemble_predictor, correlated_paths, basic_context):
        """Test prediction with multiple paths"""
        prediction = shared_ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, correlated_paths
        )
        
        assert isinstance(prediction, Prediction)
        assert len(prediction.supporting_paths) > 0
    
    def test_combine_predictions(self, shared_ensemble_predictor, sample_predictions):
        """Test combining predictions"""
        combined = shared_ensemble_predictor.combine_predictions(sample_predictions)
        
        assert isinstance(combined, Prediction)
    
    def test_prediction_validation(self, shared_ensemble_predictor, simple_path, basic_context):
        """Test prediction with safety validation"""
        prediction = shared_ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, [simple_path]
        )
        
        assert isinstance(prediction, Prediction)
        assert 0.0 <= prediction.confidence <= 1.0


# ============================================================
# THREAD SAFETY TESTS
# ============================================================

class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_path_tracing(self, path_tracer, simple_path, basic_context):
        """Test concurrent path tracing"""
        results = []
        
        def trace():
            effect = path_tracer.trace_path(simple_path, 1.0, basic_context)
            results.append(effect)
        
        threads = [threading.Thread(target=trace) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert len(set(results)) == 1
    
    def test_concurrent_predictions(self, ensemble_predictor, simple_path, basic_context):
        """Test concurrent predictions"""
        results = []
        
        def predict():
            prediction = ensemble_predictor.predict_with_path_ensemble(
                1.0, basic_context, [simple_path]
            )
            results.append(prediction.expected)
        
        threads = [threading.Thread(target=predict) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_prediction_workflow(self, correlated_paths):
        """Test complete prediction workflow"""
        ensemble = EnsemblePredictor(default_method="weighted_quantile")
        context = {'initial_values': {'A': 1.0}, 'n_samples': 50}
        
        prediction = ensemble.predict_with_path_ensemble(
            action=2.0,
            context=context,
            paths=correlated_paths
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.lower_bound <= prediction.expected <= prediction.upper_bound
        assert 0.0 <= prediction.confidence <= 1.0
    
    def test_multiple_predictions_with_combination(self):
        """Test making multiple predictions and combining them"""
        ensemble = EnsemblePredictor()
        
        paths1 = [
            Path(['A', 'B'], [('A', 'B', 0.8)], 0.8),
            Path(['A', 'C'], [('A', 'C', 0.7)], 0.7)
        ]
        
        paths2 = [
            Path(['A', 'D'], [('A', 'D', 0.9)], 0.9),
            Path(['A', 'E'], [('A', 'E', 0.75)], 0.75)
        ]
        
        pred1 = ensemble.predict_with_path_ensemble(1.0, {}, paths1)
        pred2 = ensemble.predict_with_path_ensemble(1.0, {}, paths2)
        
        combined = ensemble.combine_predictions([pred1, pred2])
        
        assert isinstance(combined, Prediction)


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_paths_list(self, shared_ensemble_predictor, basic_context):
        """Test prediction with empty paths list"""
        prediction = shared_ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, []
        )
        
        assert isinstance(prediction, Prediction)
    
    def test_single_path(self, shared_ensemble_predictor, simple_path, basic_context):
        """Test prediction with single path"""
        prediction = shared_ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, [simple_path]
        )
        
        assert isinstance(prediction, Prediction)
    
    def test_extreme_path_strength(self, path_tracer, basic_context):
        """Test with extreme path strength"""
        extreme_path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', 1e6)],
            total_strength=1e6
        )
        
        try:
            effect = path_tracer.trace_path(extreme_path, 1.0, basic_context)
            assert isinstance(effect, float)
        except (ValueError, OverflowError):
            pass


# ============================================================
# PERFORMANCE TESTS
# ============================================================

class TestPerformance:
    """Performance and scalability tests"""
    
    def test_large_path_ensemble(self):
        """Test prediction with large number of paths"""
        paths = []
        for i in range(100):
            path = Path(
                nodes=[f'node_{i}', f'node_{i+1}'],
                edges=[(f'node_{i}', f'node_{i+1}', np.random.uniform(0.5, 1.0))],
                total_strength=np.random.uniform(0.5, 1.0)
            )
            paths.append(path)
        
        ensemble = EnsemblePredictor()
        context = {'n_samples': 20}
        
        start = time.time()
        prediction = ensemble.predict_with_path_ensemble(1.0, context, paths)
        elapsed = time.time() - start
        
        assert elapsed < 10, f"Prediction took {elapsed}s for 100 paths"
        assert isinstance(prediction, Prediction)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
