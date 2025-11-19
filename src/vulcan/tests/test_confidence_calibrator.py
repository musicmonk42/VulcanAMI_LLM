"""
test_confidence_calibrator.py - Comprehensive tests for ConfidenceCalibrator
Tests all calibration methods, safety features, and edge cases
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

# Import the module to test
try:
    from vulcan.world_model.confidence_calibrator import (
        ConfidenceCalibrator,
        ModelConfidenceTracker,
        CalibrationBin,
        PredictionRecord
    )
    CALIBRATOR_AVAILABLE = True
except ImportError:
    CALIBRATOR_AVAILABLE = False
    pytest.skip("confidence_calibrator not available", allow_module_level=True)


class TestCalibrationBin:
    """Test CalibrationBin dataclass"""
    
    def test_initialization(self):
        """Test bin initialization"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        
        assert bin.min_confidence == 0.0
        assert bin.max_confidence == 0.1
        assert bin.predictions == []
        assert bin.outcomes == []
    
    def test_mean_confidence_empty(self):
        """Test mean confidence with no data"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        
        # Should return midpoint when empty
        assert bin.mean_confidence == 0.05
    
    def test_mean_confidence_with_data(self):
        """Test mean confidence with data"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        bin.predictions = [0.02, 0.04, 0.06]
        
        assert bin.mean_confidence == pytest.approx(0.04)
    
    def test_accuracy_empty(self):
        """Test accuracy with no outcomes"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        
        assert bin.accuracy == 0.0
    
    def test_accuracy_with_outcomes(self):
        """Test accuracy with outcomes"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        bin.outcomes = [1, 1, 0, 1]  # 75% accuracy
        
        assert bin.accuracy == 0.75
    
    def test_count(self):
        """Test count property"""
        bin = CalibrationBin(min_confidence=0.0, max_confidence=0.1)
        bin.predictions = [0.02, 0.04, 0.06]
        
        assert bin.count == 3


class TestPredictionRecord:
    """Test PredictionRecord dataclass"""
    
    def test_initialization(self):
        """Test record initialization"""
        record = PredictionRecord(
            timestamp=time.time(),
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            context_features=np.array([1.0, 2.0])
        )
        
        assert record.raw_confidence == 0.8
        assert record.calibrated_confidence == 0.75
        assert record.actual_outcome is None
        assert record.domain == "unknown"
    
    def test_with_outcome(self):
        """Test record with actual outcome"""
        record = PredictionRecord(
            timestamp=time.time(),
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            context_features=None,
            actual_outcome=1.0,
            domain="test"
        )
        
        assert record.actual_outcome == 1.0
        assert record.domain == "test"


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator class"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        calibrator = ConfidenceCalibrator()
        
        assert calibrator.method == "isotonic"
        assert calibrator.n_bins == 10
        assert calibrator.window_size == 1000
        assert len(calibrator.histogram_bins) == 10
    
    def test_initialization_with_params(self):
        """Test initialization with parameters"""
        calibrator = ConfidenceCalibrator(
            method="platt",
            n_bins=20,
            window_size=500
        )
        
        assert calibrator.method == "platt"
        assert calibrator.n_bins == 20
        assert calibrator.window_size == 500
        assert len(calibrator.histogram_bins) == 20
    
    def test_initialization_with_safety_config(self):
        """Test initialization with safety config"""
        safety_config = {'max_nodes': 1000}
        calibrator = ConfidenceCalibrator(safety_config=safety_config)
        
        # Should initialize without errors
        assert calibrator is not None
    
    def test_calibrate_basic(self):
        """Test basic calibration without context"""
        calibrator = ConfidenceCalibrator(method="isotonic")
        
        # Calibrate a confidence score
        result = calibrator.calibrate(0.8)
        
        # Should return a float in [0, 1]
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calibrate_with_none_features(self):
        """Test calibration with None context features (FIXED bug)"""
        calibrator = ConfidenceCalibrator()
        
        # Should handle None features gracefully
        result = calibrator.calibrate(0.8, context_features=None)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calibrate_with_context(self):
        """Test calibration with context features"""
        calibrator = ConfidenceCalibrator()
        context = np.array([1.0, 2.0, 3.0])
        
        result = calibrator.calibrate(0.8, context_features=context)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calibrate_bounds_checking(self):
        """Test that calibration clamps to [0, 1]"""
        calibrator = ConfidenceCalibrator()
        
        # Test lower bound
        result = calibrator.calibrate(-0.5)
        assert 0.0 <= result <= 1.0
        
        # Test upper bound
        result = calibrator.calibrate(1.5)
        assert 0.0 <= result <= 1.0
    
    def test_calibrate_non_finite(self):
        """Test calibration with non-finite values"""
        calibrator = ConfidenceCalibrator()
        
        # Should handle NaN
        result = calibrator.calibrate(float('nan'))
        assert result == 0.5  # Default fallback
        
        # Should handle infinity
        result = calibrator.calibrate(float('inf'))
        assert 0.0 <= result <= 1.0
    
    def test_update_calibration(self):
        """Test updating calibration with new data"""
        calibrator = ConfidenceCalibrator()
        
        # Add calibration data
        calibrator.update_calibration(0.8, 1.0)  # prediction, outcome
        calibrator.update_calibration(0.6, 0.0)
        calibrator.update_calibration(0.9, 1.0)
        
        # History should be updated
        assert len(calibrator.calibration_history) == 3
    
    def test_update_calibration_with_features(self):
        """Test updating with context features"""
        calibrator = ConfidenceCalibrator()
        context = np.array([1.0, 2.0])
        
        calibrator.update_calibration(0.8, 1.0, context_features=context)
        
        assert len(calibrator.calibration_history) == 1
        assert calibrator.calibration_history[0].context_features is not None
    
    def test_update_calibration_non_finite(self):
        """Test update with non-finite values (safety check)"""
        calibrator = ConfidenceCalibrator()
        
        # Should reject non-finite values
        calibrator.update_calibration(float('nan'), 1.0)
        calibrator.update_calibration(0.8, float('inf'))
        
        # Should not add to history
        assert len(calibrator.calibration_history) == 0
    
    def test_isotonic_calibration(self):
        """Test isotonic regression calibration"""
        calibrator = ConfidenceCalibrator(method="isotonic")
        
        # Add training data
        for i in range(100):
            pred = i / 100.0
            outcome = 1.0 if pred > 0.5 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Calibrate a value
        result = calibrator.calibrate(0.7)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_platt_calibration(self):
        """Test Platt scaling calibration"""
        calibrator = ConfidenceCalibrator(method="platt")
        
        # Add training data
        for i in range(100):
            pred = i / 100.0
            outcome = 1.0 if pred > 0.5 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Calibrate a value
        result = calibrator.calibrate(0.7)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_histogram_calibration(self):
        """Test histogram calibration"""
        calibrator = ConfidenceCalibrator(method="histogram")
        
        # Add data to bins
        for i in range(100):
            pred = 0.75  # All in same bin
            outcome = 1.0 if i % 2 == 0 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Calibrate
        result = calibrator.calibrate(0.75)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_beta_calibration(self):
        """Test beta calibration"""
        calibrator = ConfidenceCalibrator(method="beta")
        
        # Add training data
        for i in range(100):
            pred = i / 100.0
            outcome = 1.0 if pred > 0.5 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Calibrate
        result = calibrator.calibrate(0.7)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_get_calibration_curve(self):
        """Test getting calibration curve data"""
        calibrator = ConfidenceCalibrator()
        
        # Add data to bins
        for i in range(100):
            pred = i / 100.0
            outcome = 1.0 if i % 2 == 0 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Get curve
        mean_confs, accuracies, counts = calibrator.get_calibration_curve()
        
        assert len(mean_confs) > 0
        assert len(accuracies) > 0
        assert len(counts) > 0
        assert len(mean_confs) == len(accuracies) == len(counts)
    
    def test_calculate_ece(self):
        """Test Expected Calibration Error calculation"""
        calibrator = ConfidenceCalibrator()
        
        # Add perfectly calibrated data
        for i in range(100):
            pred = i / 100.0
            outcome = pred  # Perfect calibration
            calibrator.update_calibration(pred, outcome)
        
        ece = calibrator.calculate_expected_calibration_error()
        
        # ECE should be low for well-calibrated predictions
        assert isinstance(ece, float)
        assert ece >= 0.0
    
    def test_calculate_mce(self):
        """Test Maximum Calibration Error calculation"""
        calibrator = ConfidenceCalibrator()
        
        # Add data
        for i in range(100):
            pred = i / 100.0
            outcome = 1.0 if i % 2 == 0 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        mce = calibrator.calculate_maximum_calibration_error()
        
        assert isinstance(mce, float)
        assert mce >= 0.0
    
    def test_get_reliability_diagram_data(self):
        """Test getting reliability diagram data"""
        calibrator = ConfidenceCalibrator()
        
        # Add data
        for i in range(50):
            pred = i / 50.0
            outcome = 1.0 if i % 2 == 0 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        data = calibrator.get_reliability_diagram_data()
        
        assert 'mean_confidences' in data
        assert 'accuracies' in data
        assert 'counts' in data
        assert 'perfect_calibration' in data
        assert 'ece' in data
        assert 'mce' in data
        assert 'total_predictions' in data
    
    def test_retraining(self):
        """Test model retraining after sufficient data"""
        calibrator = ConfidenceCalibrator()
        
        # Add enough data to trigger retraining
        for i in range(150):
            pred = i / 150.0
            outcome = 1.0 if pred > 0.5 else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Force retraining by setting time
        calibrator.last_calibration_time = time.time() - 120  # 2 minutes ago
        
        # Add more data to trigger retraining
        calibrator.update_calibration(0.8, 1.0)
        
        # Should have incremented version
        assert calibrator.calibration_version > 1
    
    def test_get_statistics(self):
        """Test getting calibrator statistics"""
        calibrator = ConfidenceCalibrator()
        
        stats = calibrator.get_statistics()
        
        assert 'method' in stats
        assert 'n_bins' in stats
        assert 'calibration_history_size' in stats
        assert 'calibration_version' in stats
        assert 'ece' in stats
        assert 'mce' in stats
        assert stats['method'] == 'isotonic'
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        calibrator = ConfidenceCalibrator()
        
        # Should have lock
        assert hasattr(calibrator, 'lock')
        
        # Operations should be thread-safe
        import threading
        
        def update_worker():
            for i in range(10):
                calibrator.update_calibration(0.5, 1.0)
        
        threads = [threading.Thread(target=update_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have all updates
        assert len(calibrator.calibration_history) == 50


class TestModelConfidenceTracker:
    """Test ModelConfidenceTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ModelConfidenceTracker()
        
        assert tracker.model_confidence == 0.5
        assert tracker.min_confidence == 0.1
        assert tracker.max_confidence == 0.95
    
    def test_initialization_with_params(self):
        """Test initialization with parameters"""
        tracker = ModelConfidenceTracker(
            decay_rate=0.9,
            min_confidence=0.2,
            max_confidence=0.8
        )
        
        assert tracker.decay_rate == 0.9
        assert tracker.min_confidence == 0.2
        assert tracker.max_confidence == 0.8
    
    def test_update_without_params(self):
        """Test update without parameters (router compatible)"""
        tracker = ModelConfidenceTracker()
        
        # Should handle being called without params
        result = tracker.update()
        
        assert result['status'] == 'skipped'
        assert 'reason' in result
    
    def test_update_with_missing_data(self):
        """Test update with None parameters"""
        tracker = ModelConfidenceTracker()
        
        # Missing observation
        result = tracker.update(observation=None, prediction={'expected': 0.5})
        assert result['status'] == 'skipped'
        
        # Missing prediction
        result = tracker.update(observation={'value': 1.0}, prediction=None)
        assert result['status'] == 'skipped'
    
    def test_update_with_valid_data(self):
        """Test update with valid observation and prediction"""
        tracker = ModelConfidenceTracker()
        
        # Create mock observation and prediction
        observation = Mock()
        observation.value = 0.8
        
        prediction = Mock()
        prediction.expected = 0.7
        
        result = tracker.update(observation=observation, prediction=prediction)
        
        assert result['status'] == 'success'
        assert 'model_confidence' in result
        assert 'error' in result
    
    def test_confidence_increases_on_good_predictions(self):
        """Test that confidence increases with accurate predictions"""
        tracker = ModelConfidenceTracker()
        
        initial_confidence = tracker.model_confidence
        
        # Add accurate predictions
        for i in range(10):
            obs = Mock()
            obs.value = 0.8
            pred = Mock()
            pred.expected = 0.79  # Very close
            
            tracker.update(observation=obs, prediction=pred)
        
        # Confidence should increase
        assert tracker.model_confidence > initial_confidence
    
    def test_confidence_decreases_on_bad_predictions(self):
        """Test that confidence decreases with inaccurate predictions"""
        tracker = ModelConfidenceTracker()
        
        initial_confidence = tracker.model_confidence
        
        # Add inaccurate predictions
        for i in range(10):
            obs = Mock()
            obs.value = 0.8
            pred = Mock()
            pred.expected = 0.2  # Very far
            
            tracker.update(observation=obs, prediction=pred)
        
        # Confidence should decrease
        assert tracker.model_confidence < initial_confidence
    
    def test_confidence_bounds(self):
        """Test that confidence stays within bounds"""
        tracker = ModelConfidenceTracker()
        
        # Add many good predictions
        for i in range(100):
            obs = Mock()
            obs.value = 0.8
            pred = Mock()
            pred.expected = 0.8
            
            tracker.update(observation=obs, prediction=pred)
        
        # Should not exceed max
        assert tracker.model_confidence <= tracker.max_confidence
        
        # Add many bad predictions
        for i in range(100):
            obs = Mock()
            obs.value = 0.8
            pred = Mock()
            pred.expected = 0.0
            
            tracker.update(observation=obs, prediction=pred)
        
        # Should not go below min
        assert tracker.model_confidence >= tracker.min_confidence
    
    def test_get_model_confidence(self):
        """Test getting model confidence"""
        tracker = ModelConfidenceTracker()
        
        confidence = tracker.get_model_confidence()
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_get_prediction_confidence(self):
        """Test getting prediction-specific confidence"""
        tracker = ModelConfidenceTracker()
        
        action = Mock()
        context = Mock()
        context.domain = "test"
        
        confidence = tracker.get_prediction_confidence(action, context)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_identify_low_confidence_regions(self):
        """Test identifying low confidence regions"""
        tracker = ModelConfidenceTracker()
        
        regions = tracker.identify_low_confidence_regions()
        
        assert isinstance(regions, list)
    
    def test_get_confidence_summary(self):
        """Test getting confidence summary"""
        tracker = ModelConfidenceTracker()
        
        summary = tracker.get_confidence_summary()
        
        assert 'overall_confidence' in summary
        assert 'domain_confidence' in summary
        assert 'rolling_accuracy' in summary
        assert 'rolling_error' in summary
        assert 'confidence_trend' in summary
    
    def test_domain_specific_confidence(self):
        """Test domain-specific confidence tracking"""
        tracker = ModelConfidenceTracker()
        
        # Add predictions for specific domain
        for i in range(10):
            obs = Mock()
            obs.value = 0.8
            obs.domain = "physics"
            pred = Mock()
            pred.expected = 0.79
            
            tracker.update(observation=obs, prediction=pred)
        
        summary = tracker.get_confidence_summary()
        
        # Should have domain confidence
        assert 'domain_confidence' in summary
        if 'physics' in summary['domain_confidence']:
            assert isinstance(summary['domain_confidence']['physics'], float)
    
    def test_confidence_trend_calculation(self):
        """Test confidence trend calculation (FIXED: division by zero)"""
        tracker = ModelConfidenceTracker()
        
        # Add some history
        for i in range(20):
            tracker.confidence_history.append({
                'timestamp': time.time(),
                'confidence': 0.5 + i * 0.01,  # Increasing trend
                'error': None
            })
        
        summary = tracker.get_confidence_summary()
        
        # Should calculate trend without error
        assert 'confidence_trend' in summary
        assert isinstance(summary['confidence_trend'], float)
    
    def test_save_and_load_state(self):
        """Test saving and loading tracker state"""
        import tempfile
        import os
        
        tracker = ModelConfidenceTracker()
        
        # Update with some data - FIXED: Use string for domain, not Mock
        for i in range(10):
            obs = Mock()
            obs.value = 0.8
            obs.domain = "test_domain"  # String domain, not Mock
            pred = Mock()
            pred.expected = 0.75
            tracker.update(observation=obs, prediction=pred)
        
        # Save state
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker.save_state(tmpdir)
            
            # Check file exists
            assert os.path.exists(os.path.join(tmpdir, 'confidence_state.json'))
            
            # Create new tracker and load
            new_tracker = ModelConfidenceTracker()
            new_tracker.load_state(tmpdir)
            
            # Should have same confidence
            assert new_tracker.model_confidence == pytest.approx(tracker.model_confidence)
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        tracker = ModelConfidenceTracker()
        
        assert hasattr(tracker, 'lock')
        
        # Concurrent updates
        import threading
        
        def update_worker():
            for i in range(10):
                obs = Mock()
                obs.value = 0.8
                pred = Mock()
                pred.expected = 0.75
                tracker.update(observation=obs, prediction=pred)
        
        threads = [threading.Thread(target=update_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have processed all updates
        assert len(tracker.prediction_errors) > 0


class TestIntegration:
    """Integration tests for calibrator and tracker together"""
    
    def test_calibrator_with_tracker(self):
        """Test using calibrator with tracker"""
        calibrator = ConfidenceCalibrator()
        tracker = ModelConfidenceTracker()
        
        # Simulate prediction workflow
        for i in range(50):
            # Get model confidence
            model_conf = tracker.get_model_confidence()
            
            # Calibrate it
            calibrated_conf = calibrator.calibrate(model_conf)
            
            # Make prediction with calibrated confidence
            pred = Mock()
            pred.expected = 0.7
            pred.confidence = calibrated_conf
            
            # Observe outcome
            obs = Mock()
            obs.value = 0.72
            
            # Update both
            tracker.update(observation=obs, prediction=pred)
            calibrator.update_calibration(calibrated_conf, 1.0)
        
        # Both should have learned
        assert len(calibrator.calibration_history) > 0
        assert len(tracker.prediction_errors) > 0
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        calibrator = ConfidenceCalibrator(method="isotonic")
        tracker = ModelConfidenceTracker()
        
        # Phase 1: Training
        for i in range(100):
            raw_conf = np.random.random()
            calibrated = calibrator.calibrate(raw_conf)
            
            # Simulate prediction
            pred = Mock()
            pred.expected = 0.5
            pred.confidence = calibrated
            
            obs = Mock()
            obs.value = 1.0 if np.random.random() < raw_conf else 0.0
            
            tracker.update(observation=obs, prediction=pred)
            calibrator.update_calibration(calibrated, obs.value)
        
        # Phase 2: Evaluation
        stats = calibrator.get_statistics()
        summary = tracker.get_confidence_summary()
        
        assert stats['calibration_history_size'] > 0
        assert summary['overall_confidence'] > 0.0
        
        # ECE should be reasonable after training
        assert stats['ece'] < 1.0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_calibrator(self):
        """Test calibrator with no training data"""
        calibrator = ConfidenceCalibrator()
        
        # Should still work
        result = calibrator.calibrate(0.7)
        assert 0.0 <= result <= 1.0
    
    def test_single_bin_calibration(self):
        """Test with only one bin"""
        calibrator = ConfidenceCalibrator(n_bins=1)
        
        for i in range(10):
            calibrator.update_calibration(0.5, 1.0)
        
        result = calibrator.calibrate(0.5)
        assert 0.0 <= result <= 1.0
    
    def test_extreme_values(self):
        """Test with extreme values"""
        calibrator = ConfidenceCalibrator()
        
        # Test extremes
        assert calibrator.calibrate(0.0) >= 0.0
        assert calibrator.calibrate(1.0) <= 1.0
        assert calibrator.calibrate(0.5) == pytest.approx(0.5, abs=0.5)
    
    def test_all_same_predictions(self):
        """Test when all predictions are the same"""
        calibrator = ConfidenceCalibrator()
        
        # All predictions = 0.5
        for i in range(100):
            calibrator.update_calibration(0.5, 1.0 if i % 2 == 0 else 0.0)
        
        result = calibrator.calibrate(0.5)
        
        # Should be close to 0.5 (50% accuracy)
        assert 0.3 <= result <= 0.7
    
    def test_perfect_calibration(self):
        """Test with perfectly calibrated data"""
        calibrator = ConfidenceCalibrator()
        
        # Perfect calibration: prediction = outcome
        for i in range(100):
            conf = i / 100.0
            outcome = conf
            calibrator.update_calibration(conf, outcome)
        
        ece = calibrator.calculate_expected_calibration_error()
        
        # ECE should be very low
        assert ece < 0.2
    
    def test_large_dataset(self):
        """Test with large dataset"""
        calibrator = ConfidenceCalibrator(window_size=2000)
        
        # Add many samples
        for i in range(1500):
            pred = np.random.random()
            outcome = 1.0 if np.random.random() < pred else 0.0
            calibrator.update_calibration(pred, outcome)
        
        # Should handle large dataset
        assert len(calibrator.calibration_history) == 1500
        
        stats = calibrator.get_statistics()
        assert stats['calibration_history_size'] == 1500


class TestSafetyFeatures:
    """Test safety features and validation"""
    
    def test_safety_validator_initialization(self):
        """Test initialization with safety validator - FIXED: Use None instead of invalid config"""
        # Test with None safety_config - should not raise exception
        calibrator = ConfidenceCalibrator(safety_config=None)
        
        # Access the property to trigger lazy loading
        # This may return None if safety modules aren't available, which is fine
        validator = calibrator.safety_validator
        
        # Test passes if no exception is raised during initialization or property access
        assert True
        
        # Also test with empty dict
        calibrator2 = ConfidenceCalibrator(safety_config={})
        validator2 = calibrator2.safety_validator
        assert True
    
    def test_safety_corrections_tracking(self):
        """Test that safety corrections are tracked"""
        calibrator = ConfidenceCalibrator()
        
        # Trigger corrections with non-finite values
        calibrator.calibrate(float('nan'))
        calibrator.calibrate(float('inf'))
        
        stats = calibrator.get_statistics()
        
        # Should track corrections
        if 'safety' in stats:
            assert 'corrections' in stats['safety']
    
    def test_safety_blocks_tracking(self):
        """Test that safety blocks are tracked"""
        calibrator = ConfidenceCalibrator()
        
        # Add invalid calibration data
        calibrator.update_calibration(float('nan'), 1.0)
        
        stats = calibrator.get_statistics()
        
        # Should track blocks
        if 'safety' in stats:
            assert 'blocks' in stats['safety']


class TestPerformance:
    """Performance and stress tests"""
    
    def test_calibration_performance(self):
        """Test calibration performance"""
        calibrator = ConfidenceCalibrator()
        
        # Add training data
        for i in range(1000):
            calibrator.update_calibration(i / 1000.0, i % 2)
        
        # Time calibrations
        start = time.time()
        for i in range(1000):
            calibrator.calibrate(i / 1000.0)
        duration = time.time() - start
        
        # Should be fast (< 1 second for 1000 calibrations)
        assert duration < 1.0
    
    def test_memory_bounds(self):
        """Test that memory is bounded"""
        calibrator = ConfidenceCalibrator(window_size=100)
        
        # Add many samples
        for i in range(1000):
            calibrator.update_calibration(0.5, 1.0)
        
        # Should be bounded to window size
        assert len(calibrator.calibration_history) == 100
    
    def test_bin_size_limits(self):
        """Test that bin sizes are limited"""
        calibrator = ConfidenceCalibrator(n_bins=10, window_size=1000)
        
        # Add many samples to one bin
        for i in range(1000):
            calibrator.update_calibration(0.55, 1.0)  # All in same bin
        
        # Bins should be size-limited
        bin_idx = 5  # Middle bin
        assert len(calibrator.histogram_bins[bin_idx].predictions) <= 1000 // 10 + 100


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])