"""
test_prediction_engine.py - Comprehensive test suite for PredictionEngine
Part of the VULCAN-AGI system

Tests cover:
- Path dataclass and operations
- PathCluster operations
- Prediction dataclass and uncertainty metrics
- PathAnalyzer for path analysis
- PathEffectCalculator for effect computation
- PathTracer with type checking and caching
- PathClusterer for path clustering
- MonteCarloSampler for sampling
- PredictionCombiner for combining predictions
- EnsemblePredictor with safety validation
- Thread safety
- Edge cases and error handling
"""

import pytest
import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple

# Import from prediction_engine
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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_path():
    """Create a simple path"""
    return Path(
        nodes=['A', 'B', 'C'],
        edges=[('A', 'B', 0.8), ('B', 'C', 0.7)],
        total_strength=0.56,
        confidence=0.9
    )


@pytest.fixture
def complex_path():
    """Create a more complex path"""
    return Path(
        nodes=['X', 'Y', 'Z', 'W'],
        edges=[('X', 'Y', 0.9), ('Y', 'Z', 0.85), ('Z', 'W', 0.75)],
        total_strength=0.57,
        confidence=0.85,
        evidence_types=['correlation', 'intervention']
    )


@pytest.fixture
def correlated_paths():
    """Create correlated paths for clustering"""
    paths = [
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
    return paths


@pytest.fixture
def sample_predictions():
    """Create sample predictions"""
    return [
        Prediction(
            expected=5.0,
            lower_bound=4.0,
            upper_bound=6.0,
            confidence=0.9,
            method="test"
        ),
        Prediction(
            expected=5.5,
            lower_bound=4.5,
            upper_bound=6.5,
            confidence=0.85,
            method="test"
        ),
        Prediction(
            expected=4.8,
            lower_bound=3.8,
            upper_bound=5.8,
            confidence=0.88,
            method="test"
        )
    ]


@pytest.fixture
def path_analyzer():
    """Create path analyzer"""
    return PathAnalyzer()


@pytest.fixture
def effect_calculator():
    """Create effect calculator"""
    return PathEffectCalculator()


@pytest.fixture
def path_tracer():
    """Create path tracer"""
    return PathTracer(min_path_strength=0.1, max_path_length=5)


@pytest.fixture
def clusterer(path_analyzer):
    """Create path clusterer"""
    return PathClusterer(path_analyzer)


@pytest.fixture
def sampler():
    """Create Monte Carlo sampler"""
    return MonteCarloSampler()


@pytest.fixture
def combiner():
    """Create prediction combiner"""
    return PredictionCombiner()


@pytest.fixture
def ensemble_predictor():
    """Create ensemble predictor"""
    return EnsemblePredictor(default_method="weighted_quantile")


@pytest.fixture
def basic_context():
    """Create basic prediction context"""
    return {
        'initial_values': {'A': 1.0},
        'add_noise': False
    }


# ============================================================================
# Test Path Dataclass
# ============================================================================

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
        assert len(simple_path) == 2  # Number of edges
    
    def test_contains_node(self, simple_path):
        """Test checking if path contains node"""
        assert simple_path.contains_node('A') == True
        assert simple_path.contains_node('B') == True
        assert simple_path.contains_node('C') == True
        assert simple_path.contains_node('D') == False
    
    def test_get_edge_strength(self, simple_path):
        """Test getting edge strength"""
        strength = simple_path.get_edge_strength('A', 'B')
        assert strength == 0.8
        
        strength2 = simple_path.get_edge_strength('B', 'C')
        assert strength2 == 0.7
        
        # Non-existent edge
        strength3 = simple_path.get_edge_strength('A', 'C')
        assert strength3 is None
    
    def test_get_strengths(self, simple_path):
        """Test getting all edge strengths"""
        strengths = simple_path.get_strengths()
        
        assert len(strengths) == 2
        assert strengths[0] == 0.8
        assert strengths[1] == 0.7
    
    def test_strengths_property(self, simple_path):
        """Test strengths property accessor"""
        strengths = simple_path.strengths
        
        assert len(strengths) == 2
        assert strengths == [0.8, 0.7]
    
    def test_path_with_metadata(self):
        """Test path with metadata"""
        path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', 0.9)],
            total_strength=0.9,
            metadata={'source': 'test', 'domain': 'example'}
        )
        
        assert path.metadata['source'] == 'test'
        assert path.metadata['domain'] == 'example'


# ============================================================================
# Test PathCluster Dataclass
# ============================================================================

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
            correlation_matrix=np.eye(len(correlated_paths)),
            representative_path=correlated_paths[0]
        )
        
        assert cluster.size == len(correlated_paths)


# ============================================================================
# Test Prediction Dataclass
# ============================================================================

class TestPrediction:
    """Test Prediction dataclass"""
    
    def test_prediction_creation(self):
        """Test basic prediction creation"""
        pred = Prediction(
            expected=5.0,
            lower_bound=4.0,
            upper_bound=6.0,
            confidence=0.9,
            method="test"
        )
        
        assert pred.expected == 5.0
        assert pred.lower_bound == 4.0
        assert pred.upper_bound == 6.0
        assert pred.confidence == 0.9
        assert pred.method == "test"
    
    def test_uncertainty_range(self):
        """Test uncertainty range calculation"""
        pred = Prediction(
            expected=5.0,
            lower_bound=3.0,
            upper_bound=7.0,
            confidence=0.8,
            method="test"
        )
        
        assert pred.uncertainty_range() == 4.0
    
    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation"""
        pred = Prediction(
            expected=10.0,
            lower_bound=8.0,
            upper_bound=12.0,
            confidence=0.8,
            method="test"
        )
        
        assert pred.relative_uncertainty() == 0.4  # 4.0 / 10.0
    
    def test_relative_uncertainty_zero_expected(self):
        """Test relative uncertainty with zero expected value"""
        pred = Prediction(
            expected=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            confidence=0.5,
            method="test"
        )
        
        # Should return 1.0 when expected is near zero
        assert pred.relative_uncertainty() == 1.0
    
    def test_to_dict(self):
        """Test converting prediction to dictionary"""
        pred = Prediction(
            expected=5.0,
            lower_bound=4.0,
            upper_bound=6.0,
            confidence=0.9,
            method="test"
        )
        
        pred_dict = pred.to_dict()
        
        assert pred_dict['expected'] == 5.0
        assert pred_dict['lower_bound'] == 4.0
        assert pred_dict['upper_bound'] == 6.0
        assert pred_dict['confidence'] == 0.9
        assert pred_dict['method'] == "test"
        assert 'uncertainty_range' in pred_dict
        assert 'relative_uncertainty' in pred_dict
    
    def test_prediction_with_supporting_paths(self, simple_path):
        """Test prediction with supporting paths"""
        pred = Prediction(
            expected=5.0,
            lower_bound=4.0,
            upper_bound=6.0,
            confidence=0.9,
            method="test",
            supporting_paths=[simple_path]
        )
        
        assert len(pred.supporting_paths) == 1
        assert pred.supporting_paths[0] == simple_path


# ============================================================================
# Test PathAnalyzer
# ============================================================================

class TestPathAnalyzer:
    """Test PathAnalyzer component"""
    
    def test_calculate_path_confidence(self, path_analyzer, simple_path):
        """Test calculating path confidence"""
        confidence = path_analyzer.calculate_path_confidence(simple_path)
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_confidence_caching(self, path_analyzer, simple_path):
        """Test that confidence calculation is cached"""
        # First call
        conf1 = path_analyzer.calculate_path_confidence(simple_path)
        
        # Second call should use cache
        conf2 = path_analyzer.calculate_path_confidence(simple_path)
        
        assert conf1 == conf2
    
    def test_calculate_path_correlation(self, path_analyzer, simple_path, complex_path):
        """Test calculating correlation between paths"""
        correlation = path_analyzer.calculate_path_correlation(simple_path, complex_path)
        
        assert 0.0 <= correlation <= 1.0
        assert isinstance(correlation, float)
    
    def test_correlation_symmetry(self, path_analyzer, simple_path, complex_path):
        """Test that correlation is symmetric"""
        corr1 = path_analyzer.calculate_path_correlation(simple_path, complex_path)
        corr2 = path_analyzer.calculate_path_correlation(complex_path, simple_path)
        
        assert corr1 == corr2
    
    def test_self_correlation(self, path_analyzer, simple_path):
        """Test that self-correlation is high"""
        correlation = path_analyzer.calculate_path_correlation(simple_path, simple_path)
        
        # Self-correlation should be 1.0 or very close
        assert correlation >= 0.9
    
    def test_correlation_caching(self, path_analyzer, simple_path, complex_path):
        """Test that correlation calculation is cached"""
        # First call
        corr1 = path_analyzer.calculate_path_correlation(simple_path, complex_path)
        
        # Second call should use cache
        corr2 = path_analyzer.calculate_path_correlation(simple_path, complex_path)
        
        assert corr1 == corr2


# ============================================================================
# Test PathEffectCalculator
# ============================================================================

class TestPathEffectCalculator:
    """Test PathEffectCalculator component"""
    
    def test_calculate_path_effect(self, effect_calculator, simple_path, basic_context):
        """Test calculating effect along a path"""
        effect = effect_calculator.calculate_path_effect(simple_path, 1.0, basic_context)
        
        assert isinstance(effect, float)
        # Effect should be approximately 0.56 (0.8 * 0.7)
        assert 0.4 < effect < 0.7
    
    def test_calculate_chain_effects(self, effect_calculator, simple_path, basic_context):
        """Test calculating effects along chain"""
        effects = effect_calculator.calculate_chain_effects('A', simple_path, basic_context)
        
        assert isinstance(effects, list)
        assert len(effects) > 0
        # First effect should be initial value
        assert effects[0] == 1.0
    
    def test_effect_with_noise(self, effect_calculator, simple_path):
        """Test effect calculation with noise"""
        # Use a specific seed that ensures variation is detected
        np.random.seed(42)
        
        context = {
            'add_noise': True,
            'noise_level': 0.1
        }
        
        effects = []
        for _ in range(100):  # Increase samples for more reliable variation detection
            effect = effect_calculator.calculate_path_effect(simple_path, 1.0, context)
            effects.append(effect)
        
        # Effects should vary due to noise - check with standard deviation
        assert np.std(effects) > 0.01, "Effects should show variation when noise is added"
    
    def test_effect_with_moderators(self, effect_calculator, simple_path):
        """Test effect calculation with moderators"""
        context = {
            'moderators': {
                'A->B': 1.5,
                'B->C': 0.8
            }
        }
        
        effect = effect_calculator.calculate_path_effect(simple_path, 1.0, context)
        
        assert isinstance(effect, float)
    
    def test_path_length_decay(self, effect_calculator):
        """Test that longer paths have decay applied"""
        short_path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', 0.8)],
            total_strength=0.8
        )
        
        long_path = Path(
            nodes=['A', 'B', 'C', 'D', 'E'],
            edges=[('A', 'B', 0.8), ('B', 'C', 0.8), ('C', 'D', 0.8), ('D', 'E', 0.8)],
            total_strength=0.4096
        )
        
        context = {}
        short_effect = effect_calculator.calculate_path_effect(short_path, 1.0, context)
        long_effect = effect_calculator.calculate_path_effect(long_path, 1.0, context)
        
        # Longer path should have more decay
        # Even though base strengths are same, decay should reduce long_effect more
        assert long_effect < short_effect


# ============================================================================
# Test PathTracer
# ============================================================================

class TestPathTracer:
    """Test PathTracer component"""
    
    def test_initialization(self, path_tracer):
        """Test path tracer initialization"""
        assert path_tracer.min_path_strength == 0.1
        assert path_tracer.max_path_length == 5
    
    def test_trace_path(self, path_tracer, simple_path, basic_context):
        """Test tracing a path"""
        effect = path_tracer.trace_path(simple_path, 1.0, basic_context)
        
        assert isinstance(effect, float)
    
    def test_trace_path_type_validation(self, path_tracer, basic_context):
        """Test that trace_path validates path type"""
        # Pass non-Path object
        with pytest.raises(TypeError, match="Expected Path object"):
            path_tracer.trace_path("not a path", 1.0, basic_context)
    
    def test_trace_path_missing_attributes(self, path_tracer, basic_context):
        """Test that trace_path validates path type with fake object - FIXED"""
        # Create object without required Path attributes
        class FakePath:
            pass
        
        fake_path = FakePath()
        
        # FIXED: Should raise TypeError first due to type checking before attribute validation
        with pytest.raises(TypeError, match="Expected Path object"):
            path_tracer.trace_path(fake_path, 1.0, basic_context)
    
    def test_trace_path_caching(self, path_tracer, simple_path, basic_context):
        """Test that path tracing is cached"""
        # First trace
        effect1 = path_tracer.trace_path(simple_path, 1.0, basic_context)
        initial_hits = path_tracer.trace_stats['cache_hits']
        
        # Second trace with same parameters
        effect2 = path_tracer.trace_path(simple_path, 1.0, basic_context)
        final_hits = path_tracer.trace_stats['cache_hits']
        
        assert effect1 == effect2
        assert final_hits > initial_hits
    
    def test_trace_different_actions(self, path_tracer, simple_path, basic_context):
        """Test tracing with different actions"""
        effect1 = path_tracer.trace_path(simple_path, 1.0, basic_context)
        effect2 = path_tracer.trace_path(simple_path, 2.0, basic_context)
        
        # Different actions should give different effects
        assert effect1 != effect2
        assert effect2 > effect1  # Larger action should give larger effect
    
    def test_trace_causal_chain(self, path_tracer, simple_path, basic_context):
        """Test tracing causal chain"""
        effects = path_tracer.trace_causal_chain('A', simple_path, basic_context)
        
        assert isinstance(effects, list)
        assert len(effects) > 0
    
    def test_calculate_path_confidence(self, path_tracer, simple_path):
        """Test calculating path confidence via tracer"""
        confidence = path_tracer.calculate_path_confidence(simple_path)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_path_correlation(self, path_tracer, simple_path, complex_path):
        """Test calculating path correlation via tracer"""
        correlation = path_tracer.calculate_path_correlation(simple_path, complex_path)
        
        assert 0.0 <= correlation <= 1.0


# ============================================================================
# Test PathClusterer
# ============================================================================

class TestPathClusterer:
    """Test PathClusterer component"""
    
    def test_cluster_single_path(self, clusterer, simple_path):
        """Test clustering with single path"""
        clusters = clusterer.cluster_paths([simple_path])
        
        assert len(clusters) == 1
        assert clusters[0].size == 1
    
    def test_cluster_multiple_paths(self, clusterer, correlated_paths):
        """Test clustering with multiple paths"""
        clusters = clusterer.cluster_paths(correlated_paths)
        
        assert len(clusters) > 0
        # Total paths should be preserved
        total_paths = sum(c.size for c in clusters)
        assert total_paths == len(correlated_paths)
    
    def test_cluster_empty_list(self, clusterer):
        """Test clustering with empty list"""
        clusters = clusterer.cluster_paths([])
        
        assert len(clusters) == 0
    
    def test_cluster_representative_path(self, clusterer, correlated_paths):
        """Test that each cluster has representative path"""
        clusters = clusterer.cluster_paths(correlated_paths)
        
        for cluster in clusters:
            assert cluster.representative_path is not None
            assert cluster.representative_path in cluster.paths


# ============================================================================
# Test MonteCarloSampler
# ============================================================================

class TestMonteCarloSampler:
    """Test MonteCarloSampler component"""
    
    def test_sample_from_single_path_cluster(self, sampler, simple_path):
        """Test sampling from cluster with single path"""
        cluster = PathCluster(
            paths=[simple_path],
            correlation_matrix=np.array([[1.0]]),
            representative_path=simple_path
        )
        
        samples = sampler.sample_from_cluster(cluster, n_samples=10)
        
        assert len(samples) == 10
        assert all(isinstance(s, Path) for s in samples)
    
    def test_sample_from_multiple_path_cluster(self, sampler, correlated_paths):
        """Test sampling from cluster with multiple paths"""
        correlation_matrix = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0]
        ])
        
        cluster = PathCluster(
            paths=correlated_paths,
            correlation_matrix=correlation_matrix,
            representative_path=correlated_paths[0]
        )
        
        samples = sampler.sample_from_cluster(cluster, n_samples=20)
        
        assert len(samples) == 20
        assert all(isinstance(s, Path) for s in samples)
    
    def test_sample_variations(self, sampler, simple_path):
        """Test that samples have variations"""
        # Use a specific seed that ensures variation is detected
        np.random.seed(42)
        
        cluster = PathCluster(
            paths=[simple_path],
            correlation_matrix=np.array([[1.0]]),
            representative_path=simple_path
        )
        
        samples = sampler.sample_from_cluster(cluster, n_samples=100)  # Increase samples
        
        # Check that edge strengths vary using standard deviation
        strengths = [s.edges[0][2] for s in samples]
        assert np.std(strengths) > 0.001, "Sampled edge strengths should show variation"
    
    def test_sample_from_empty_cluster(self, sampler):
        """Test sampling from empty cluster"""
        cluster = PathCluster(
            paths=[],
            correlation_matrix=np.array([[]]),
            representative_path=None
        )
        
        samples = sampler.sample_from_cluster(cluster, n_samples=10)
        
        assert len(samples) == 0


# ============================================================================
# Test PredictionCombiner
# ============================================================================

class TestPredictionCombiner:
    """Test PredictionCombiner component"""
    
    def test_combine_weighted_mean(self, combiner, sample_predictions):
        """Test weighted mean combination"""
        combined = combiner.combine(sample_predictions, CombinationMethod.WEIGHTED_MEAN)
        
        assert isinstance(combined, Prediction)
        assert combined.method == "weighted_mean"
        assert combined.confidence > 0
    
    def test_combine_weighted_median(self, combiner, sample_predictions):
        """Test weighted median combination"""
        combined = combiner.combine(sample_predictions, CombinationMethod.WEIGHTED_MEDIAN)
        
        assert isinstance(combined, Prediction)
        assert combined.method == "weighted_median"
    
    def test_combine_weighted_quantile(self, combiner, sample_predictions):
        """Test weighted quantile combination"""
        combined = combiner.combine(sample_predictions, CombinationMethod.WEIGHTED_QUANTILE)
        
        assert isinstance(combined, Prediction)
        assert combined.method == "weighted_quantile"
        assert combined.lower_bound <= combined.expected <= combined.upper_bound
    
    def test_combine_bootstrap(self, combiner, sample_predictions):
        """Test bootstrap combination"""
        combined = combiner.combine(sample_predictions, CombinationMethod.BOOTSTRAP)
        
        assert isinstance(combined, Prediction)
        assert combined.method == "bootstrap"
    
    def test_combine_with_tuples(self, combiner):
        """Test combining tuple predictions"""
        predictions = [
            (5.0, 0.9),
            (5.5, 0.85),
            (4.8, 0.88)
        ]
        
        combined = combiner.combine(predictions, "weighted_mean")
        
        assert isinstance(combined, Prediction)
    
    def test_combine_empty_list(self, combiner):
        """Test combining empty list"""
        combined = combiner.combine([], "weighted_mean")
        
        assert combined.expected == 0.0
        assert combined.confidence == 0.0
    
    def test_combine_single_prediction(self, combiner):
        """Test combining single prediction"""
        pred = Prediction(
            expected=5.0,
            lower_bound=4.0,
            upper_bound=6.0,
            confidence=0.9,
            method="test"
        )
        
        combined = combiner.combine([pred], "weighted_mean")
        
        assert isinstance(combined, Prediction)


# ============================================================================
# Test EnsemblePredictor
# ============================================================================

class TestEnsemblePredictor:
    """Test EnsemblePredictor class"""
    
    def test_initialization(self, ensemble_predictor):
        """Test ensemble predictor initialization"""
        assert ensemble_predictor.default_method == "weighted_quantile"
        assert ensemble_predictor.path_tracer is not None
    
    def test_predict_with_path_ensemble(self, ensemble_predictor, correlated_paths, basic_context):
        """Test prediction with path ensemble"""
        prediction = ensemble_predictor.predict_with_path_ensemble(
            action=1.0,
            context=basic_context,
            paths=correlated_paths
        )
        
        assert isinstance(prediction, Prediction)
        assert len(prediction.supporting_paths) > 0
    
    def test_predict_with_empty_paths(self, ensemble_predictor, basic_context):
        """Test prediction with empty path list"""
        prediction = ensemble_predictor.predict_with_path_ensemble(
            action=1.0,
            context=basic_context,
            paths=[]
        )
        
        # --- FIX: Update assertions to match the new "neutral prediction" logic ---
        assert prediction.confidence == 0.3  # Was 0.0
        assert prediction.method == "no_paths"
        assert prediction.expected == 0.5    # Add check for neutral expectation
        # --- END FIX ---
    
    def test_predict_paths_type_validation(self, ensemble_predictor, basic_context):
        """Test that predict validates paths parameter type"""
        # Pass non-list
        with pytest.raises(TypeError, match="paths must be list"):
            ensemble_predictor.predict_with_path_ensemble(
                action=1.0,
                context=basic_context,
                paths="not a list"
            )
    
    def test_predict_path_object_validation(self, ensemble_predictor, basic_context):
        """Test that predict validates each path object"""
        # Pass list with non-Path object
        with pytest.raises(TypeError, match="must be Path object"):
            ensemble_predictor.predict_with_path_ensemble(
                action=1.0,
                context=basic_context,
                paths=["not a path", "also not a path"]
            )
    
    def test_predict_with_single_path(self, ensemble_predictor, simple_path, basic_context):
        """Test prediction with single path"""
        prediction = ensemble_predictor.predict_with_path_ensemble(
            action=1.0,
            context=basic_context,
            paths=[simple_path]
        )
        
        assert isinstance(prediction, Prediction)
    
    def test_combine_predictions(self, ensemble_predictor, sample_predictions):
        """Test combining predictions via public interface"""
        combined = ensemble_predictor.combine_predictions(
            sample_predictions,
            method="weighted_mean"
        )
        
        assert isinstance(combined, Prediction)
    
    def test_get_statistics(self, ensemble_predictor, correlated_paths, basic_context):
        """Test getting ensemble predictor statistics"""
        # Make a prediction first
        ensemble_predictor.predict_with_path_ensemble(1.0, basic_context, correlated_paths)
        
        stats = ensemble_predictor.get_statistics()
        
        assert 'prediction_history_size' in stats
        assert 'path_tracer_stats' in stats
        assert 'default_method' in stats
    
    def test_prediction_history_tracking(self, ensemble_predictor, simple_path, basic_context):
        """Test that predictions are tracked in history"""
        initial_size = len(ensemble_predictor.prediction_history)
        
        ensemble_predictor.predict_with_path_ensemble(1.0, basic_context, [simple_path])
        
        assert len(ensemble_predictor.prediction_history) == initial_size + 1


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_path_with_zero_edges(self):
        """Test path with no edges"""
        path = Path(
            nodes=['A'],
            edges=[],
            total_strength=1.0
        )
        
        assert len(path) == 0
        assert path.get_strengths() == []
    
    def test_path_with_negative_strength(self):
        """Test path with negative edge strength"""
        path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', -0.5)],
            total_strength=-0.5
        )
        
        strengths = path.get_strengths()
        assert strengths[0] == -0.5
    
    def test_prediction_with_inverted_bounds(self):
        """Test prediction with lower bound > upper bound"""
        pred = Prediction(
            expected=5.0,
            lower_bound=7.0,  # Inverted
            upper_bound=3.0,
            confidence=0.5,
            method="test"
        )
        
        # Should still calculate range
        range_val = pred.uncertainty_range()
        assert range_val == -4.0  # Will be negative
    
    def test_combiner_with_single_value(self, combiner):
        """Test combiner with single value"""
        predictions = [
            Prediction(5.0, 5.0, 5.0, 1.0, "test")
        ]
        
        combined = combiner.combine(predictions, "weighted_mean")
        
        assert combined.expected == 5.0
    
    def test_combiner_with_zero_weights(self, combiner):
        """Test combiner with all zero weights"""
        predictions = [
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0)
        ]
        
        combined = combiner.combine(predictions, "weighted_mean")
        
        # Should handle gracefully
        assert isinstance(combined, Prediction)
    
    def test_path_tracer_with_extreme_values(self, path_tracer, basic_context):
        """Test path tracer with extreme values"""
        extreme_path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', 1e6)],  # Extreme strength
            total_strength=1e6
        )
        
        # Should handle without crashing
        try:
            effect = path_tracer.trace_path(extreme_path, 1.0, basic_context)
            assert isinstance(effect, float)
        except (ValueError, OverflowError):
            # Acceptable to raise error for extreme values
            pass
    
    def test_ensemble_with_nan_in_path(self, ensemble_predictor, basic_context):
        """Test ensemble predictor with NaN in path"""
        nan_path = Path(
            nodes=['A', 'B'],
            edges=[('A', 'B', np.nan)],
            total_strength=np.nan
        )
        
        # Should be filtered by safety validator or handled gracefully
        prediction = ensemble_predictor.predict_with_path_ensemble(
            1.0, basic_context, [nan_path]
        )
        
        assert isinstance(prediction, Prediction)


# ============================================================================
# Test Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test thread-safe operations"""
    
    def test_concurrent_path_tracing(self, path_tracer, simple_path, basic_context):
        """Test concurrent path tracing"""
        results = []
        
        def trace():
            effect = path_tracer.trace_path(simple_path, 1.0, basic_context)
            results.append(effect)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=trace)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        # All should get same result due to caching
        assert len(set(results)) == 1
    
    def test_concurrent_predictions(self, ensemble_predictor, simple_path, basic_context):
        """Test concurrent predictions"""
        results = []
        
        def predict():
            prediction = ensemble_predictor.predict_with_path_ensemble(
                1.0, basic_context, [simple_path]
            )
            results.append(prediction.expected)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=predict)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)
    
    def test_concurrent_path_confidence_calculation(self, path_analyzer, simple_path):
        """Test concurrent confidence calculations"""
        results = []
        
        def calc_confidence():
            conf = path_analyzer.calculate_path_confidence(simple_path)
            results.append(conf)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=calc_confidence)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        # All should get same result due to caching
        assert len(set(results)) == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_prediction_workflow(self, correlated_paths):
        """Test complete prediction workflow"""
        # Create ensemble predictor
        ensemble = EnsemblePredictor(default_method="weighted_quantile")
        
        # Create context
        context = {
            'initial_values': {'A': 1.0},
            'n_samples': 50
        }
        
        # Make prediction
        prediction = ensemble.predict_with_path_ensemble(
            action=2.0,
            context=context,
            paths=correlated_paths
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.expected != 0.0
        assert prediction.lower_bound <= prediction.expected <= prediction.upper_bound
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.supporting_paths) > 0
    
    def test_multiple_predictions_with_combination(self):
        """Test making multiple predictions and combining them"""
        ensemble = EnsemblePredictor()
        
        # Create different path sets
        paths1 = [
            Path(['A', 'B'], [('A', 'B', 0.8)], 0.8),
            Path(['A', 'C'], [('A', 'C', 0.7)], 0.7)
        ]
        
        paths2 = [
            Path(['A', 'D'], [('A', 'D', 0.9)], 0.9),
            Path(['A', 'E'], [('A', 'E', 0.75)], 0.75)
        ]
        
        context = {}
        
        # Make predictions
        pred1 = ensemble.predict_with_path_ensemble(1.0, context, paths1)
        pred2 = ensemble.predict_with_path_ensemble(1.0, context, paths2)
        
        # Combine predictions
        combined = ensemble.combine_predictions([pred1, pred2])
        
        assert isinstance(combined, Prediction)
    
    def test_iterative_prediction_refinement(self):
        """Test iterative prediction refinement"""
        ensemble = EnsemblePredictor()
        
        # Initial prediction
        initial_paths = [
            Path(['A', 'B', 'C'], [('A', 'B', 0.8), ('B', 'C', 0.7)], 0.56)
        ]
        
        context = {}
        pred1 = ensemble.predict_with_path_ensemble(1.0, context, initial_paths)
        
        # Add more paths (simulating learning)
        refined_paths = initial_paths + [
            Path(['A', 'D', 'C'], [('A', 'D', 0.75), ('D', 'C', 0.72)], 0.54),
            Path(['A', 'E', 'C'], [('A', 'E', 0.78), ('E', 'C', 0.68)], 0.53)
        ]
        
        pred2 = ensemble.predict_with_path_ensemble(1.0, context, refined_paths)
        
        # Refined prediction should have more supporting paths
        assert len(pred2.supporting_paths) > len(pred1.supporting_paths)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scalability tests"""
    
    def test_large_path_ensemble(self):
        """Test prediction with large number of paths"""
        # Create many paths
        paths = []
        for i in range(100):
            path = Path(
                nodes=[f'node_{i}', f'node_{i+1}'],
                edges=[(f'node_{i}', f'node_{i+1}', np.random.uniform(0.5, 1.0))],
                total_strength=np.random.uniform(0.5, 1.0)
            )
            paths.append(path)
        
        ensemble = EnsemblePredictor()
        context = {'n_samples': 20}  # Reduce samples for performance
        
        import time as time_module
        start = time_module.time()
        
        prediction = ensemble.predict_with_path_ensemble(1.0, context, paths)
        
        elapsed = time_module.time() - start
        
        assert elapsed < 10, f"Prediction took {elapsed}s for 100 paths"
        assert isinstance(prediction, Prediction)
    
    def test_many_predictions(self, simple_path):
        """Test making many predictions"""
        ensemble = EnsemblePredictor()
        context = {}
        
        import time as time_module
        start = time_module.time()
        
        for _ in range(100):
            ensemble.predict_with_path_ensemble(1.0, context, [simple_path])
        
        elapsed = time_module.time() - start
        
        assert elapsed < 5, f"100 predictions took {elapsed}s"
    
    def test_path_correlation_matrix_computation(self, path_analyzer):
        """Test correlation matrix computation with many paths"""
        # Create many paths
        paths = []
        for i in range(50):
            path = Path(
                nodes=[f'A_{i}', f'B_{i}'],
                edges=[(f'A_{i}', f'B_{i}', 0.8)],
                total_strength=0.8
            )
            paths.append(path)
        
        import time as time_module
        start = time_module.time()
        
        # Calculate all pairwise correlations
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                path_analyzer.calculate_path_correlation(paths[i], paths[j])
        
        elapsed = time_module.time() - start
        
        # Should complete in reasonable time with caching
        assert elapsed < 10, f"Correlation computation took {elapsed}s"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
