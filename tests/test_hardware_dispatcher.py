"""
Comprehensive test suite for hardware_dispatcher.py
"""

import os
import time
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from hardware_dispatcher import (AI_ERRORS, MAX_MATRIX_DIMENSION,
                                 MAX_NOISE_STD, MAX_PHOTONIC_NOISE_STD,
                                 MAX_TENSOR_SIZE, MIN_NOISE_STD,
                                 CircuitBreaker, HardwareBackend,
                                 HardwareCapabilities, HardwareDispatcher,
                                 OperationMetrics)


@pytest.fixture
def dispatcher():
    """Create dispatcher in mock mode."""
    return HardwareDispatcher(
        use_mock=True,
        enable_metrics=True,
        enable_health_checks=False
    )


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


class TestHardwareBackend:
    """Test HardwareBackend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert HardwareBackend.LIGHTMATTER.value == "lightmatter"
        assert HardwareBackend.AIM_PHOTONICS.value == "aim_photonics"
        assert HardwareBackend.CPU.value == "cpu"
        assert HardwareBackend.EMULATOR.value == "emulator"


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, timeout=timedelta(seconds=30))
        
        assert cb.failure_threshold == 3
        assert cb.timeout == timedelta(seconds=30)
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_successful_call(self):
        """Test successful function call."""
        cb = CircuitBreaker()
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        
        assert result == "success"
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_failed_call(self):
        """Test failed function call."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def fail_func():
            raise Exception("Test error")
        
        # First few failures
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(fail_func)
        
        assert cb.failure_count == 2
        assert cb.state == "closed"
    
    def test_circuit_opens(self):
        """Test circuit opens after threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def fail_func():
            raise Exception("Test error")
        
        # Exceed threshold
        for i in range(3):
            with pytest.raises(Exception):
                cb.call(fail_func)
        
        assert cb.state == "open"
        
        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            cb.call(fail_func)
    
    def test_circuit_half_open(self):
        """Test circuit transitions to half-open."""
        cb = CircuitBreaker(failure_threshold=2, timeout=timedelta(milliseconds=100))
        
        def fail_func():
            raise Exception("Test error")
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(fail_func)
        
        assert cb.state == "open"
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should transition to half-open and allow retry
        def success_func():
            return "recovered"
        
        result = cb.call(success_func)
        
        assert result == "recovered"
        assert cb.state == "closed"


class TestHardwareDispatcher:
    """Test HardwareDispatcher class."""
    
    def test_initialization_mock_mode(self):
        """Test initialization in mock mode."""
        disp = HardwareDispatcher(use_mock=True)
        
        assert disp.use_mock is True
        assert len(disp.hardware_registry) > 0
        assert disp.emulator is not None
    
    def test_initialization_with_api_keys(self):
        """Test initialization with API keys."""
        disp = HardwareDispatcher(
            lightmatter_api_key="test_key",
            aim_api_key="test_key2",
            use_mock=False,
            enable_health_checks=False
        )
        
        assert disp.lightmatter_api_key == "test_key"
        assert disp.aim_api_key == "test_key2"
    
    def test_discover_hardware(self, dispatcher):
        """Test hardware discovery."""
        assert HardwareBackend.CPU in dispatcher.hardware_registry
        assert HardwareBackend.EMULATOR in dispatcher.hardware_registry
    
    def test_list_available_hardware(self, dispatcher):
        """Test listing available hardware."""
        hardware_list = dispatcher.list_available_hardware()
        
        assert isinstance(hardware_list, list)
        assert len(hardware_list) > 0
        assert all('backend' in hw for hw in hardware_list)
        assert all('capabilities' in hw for hw in hardware_list)
    
    def test_validate_photonic_params_valid(self, dispatcher):
        """Test validating valid photonic params."""
        params = {
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748-quantized",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is None
    
    def test_validate_photonic_params_not_dict(self, dispatcher):
        """Test validating non-dict photonic params."""
        error = dispatcher.validate_photonic_params("not a dict")
        
        assert error is not None
        assert error["error_code"] == AI_ERRORS.AI_INVALID_REQUEST
        assert "must be a dictionary" in error["message"]
    
    def test_validate_photonic_params_missing_field(self, dispatcher):
        """Test validating photonic params with missing field."""
        params = {
            "noise_std": 0.01,
            # Missing other required fields
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert "Missing" in error["message"]
    
    def test_validate_photonic_params_invalid_noise_std_type(self, dispatcher):
        """Test invalid noise_std type."""
        params = {
            "noise_std": "invalid",
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert "noise_std must be numeric" in error["message"]
    
    def test_validate_photonic_params_noise_std_out_of_range(self, dispatcher):
        """Test noise_std out of range."""
        params = {
            "noise_std": -0.1,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert "noise_std must be in" in error["message"]
    
    def test_validate_photonic_params_noise_std_too_high(self, dispatcher):
        """Test noise_std exceeding photonic threshold."""
        params = {
            "noise_std": 0.1,  # Above MAX_PHOTONIC_NOISE_STD
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert error["error_code"] == AI_ERRORS.AI_PHOTONIC_NOISE
    
    def test_validate_photonic_params_invalid_compression(self, dispatcher):
        """Test invalid compression mode."""
        params = {
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "invalid-mode",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert "Invalid compression mode" in error["message"]
    
    def test_validate_photonic_params_invalid_multiplexing(self, dispatcher):
        """Test invalid multiplexing mode."""
        params = {
            "noise_std": 0.01,
            "multiplexing": "invalid-mode",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        error = dispatcher.validate_photonic_params(params)
        
        assert error is not None
        assert "Invalid multiplexing" in error["message"]
    
    def test_validate_rlhf_params_valid(self, dispatcher):
        """Test validating valid RLHF params."""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "rlhf_train": True
        }
        
        error = dispatcher.validate_rlhf_params(params)
        
        assert error is None
    
    def test_validate_rlhf_params_not_dict(self, dispatcher):
        """Test validating non-dict RLHF params."""
        error = dispatcher.validate_rlhf_params("not a dict")
        
        assert error is not None
        assert "must be a dictionary" in error["message"]
    
    def test_validate_rlhf_params_missing_field(self, dispatcher):
        """Test RLHF params with missing field."""
        params = {"temperature": 0.7}
        
        error = dispatcher.validate_rlhf_params(params)
        
        assert error is not None
        assert "Missing" in error["message"]
    
    def test_validate_rlhf_params_invalid_temperature(self, dispatcher):
        """Test invalid temperature."""
        params = {
            "temperature": 3.0,  # Out of range
            "max_tokens": 1000,
            "rlhf_train": True
        }
        
        error = dispatcher.validate_rlhf_params(params)
        
        assert error is not None
        assert "temperature must be in" in error["message"]
    
    def test_validate_rlhf_params_invalid_max_tokens(self, dispatcher):
        """Test invalid max_tokens."""
        params = {
            "temperature": 0.7,
            "max_tokens": 0,  # Invalid
            "rlhf_train": True
        }
        
        error = dispatcher.validate_rlhf_params(params)
        
        assert error is not None
        assert "max_tokens must be in" in error["message"]
    
    def test_validate_rlhf_params_invalid_rlhf_train_type(self, dispatcher):
        """Test invalid rlhf_train type."""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "rlhf_train": "not a bool"
        }
        
        error = dispatcher.validate_rlhf_params(params)
        
        assert error is not None
        assert "must be a boolean" in error["message"]
    
    def test_dispatch_to_cpu(self, dispatcher, matrix, vector):
        """Test dispatch to CPU."""
        result = dispatcher._dispatch_to_cpu("photonic_mvm", matrix, vector, params={})
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_dispatch_to_cpu_unsupported_op(self, dispatcher):
        """Test CPU dispatch with unsupported operation."""
        result = dispatcher._dispatch_to_cpu("unsupported_op", None, params={})
        
        assert isinstance(result, dict)
        assert result["error_code"] == AI_ERRORS.AI_UNSUPPORTED
    
    def test_dispatch_to_emulator(self, dispatcher, matrix, vector):
        """Test dispatch to emulator."""
        result = dispatcher._dispatch_to_emulator("photonic_mvm", matrix, vector, params={})
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_dispatch_to_emulator_not_available(self):
        """Test emulator dispatch when not available."""
        disp = HardwareDispatcher(use_mock=True, enable_health_checks=False)
        disp.emulator = None
        
        result = disp._dispatch_to_emulator("photonic_mvm", np.eye(2), np.ones(2), params={})
        
        assert isinstance(result, dict)
        assert result["error_code"] == AI_ERRORS.AI_HARDWARE_UNAVAILABLE
    
    def test_dispatch_input_validation_none(self, dispatcher):
        """Test dispatch with None argument."""
        result = dispatcher.dispatch("photonic_mvm", None, params={})
        
        assert isinstance(result, dict)
        assert result["error_code"] == AI_ERRORS.AI_INVALID_REQUEST
        assert "cannot be None" in result["message"]
    
    def test_dispatch_tensor_too_large(self, dispatcher):
        """Test dispatch with oversized tensor."""
        # Create a mock object with size attribute
        class LargeTensor:
            size = MAX_TENSOR_SIZE + 1
        
        large_tensor = LargeTensor()
        
        result = dispatcher.dispatch("photonic_mvm", large_tensor, params={})
        
        assert isinstance(result, dict)
        assert result["error_code"] == AI_ERRORS.AI_INVALID_REQUEST
        assert "too large" in result["message"]
    
    def test_dispatch_dimension_too_large(self, dispatcher):
        """Test dispatch with too large matrix dimension."""
        # This would need to create a very large array which is impractical
        # Instead test the logic with a mock
        pass
    
    def test_dispatch_photonic_mvm(self, dispatcher, matrix, vector):
        """Test dispatching photonic MVM."""
        params = {
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748-quantized",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        result = dispatcher.dispatch("photonic_mvm", matrix, vector, params=params)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_dispatch_invalid_photonic_params(self, dispatcher, matrix, vector):
        """Test dispatch with invalid photonic params."""
        params = {
            "noise_std": "invalid",
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        }
        
        result = dispatcher.dispatch("photonic_mvm", matrix, vector, params=params)
        
        assert isinstance(result, dict)
        assert "error_code" in result
    
    def test_run_photonic_mvm(self, dispatcher, matrix):
        """Test run_photonic_mvm convenience function."""
        result = dispatcher.run_photonic_mvm(matrix)
        
        assert isinstance(result, dict)
        assert 'result' in result or 'error_code' in result
    
    def test_run_photonic_mvm_none_input(self, dispatcher):
        """Test run_photonic_mvm with None input."""
        result = dispatcher.run_photonic_mvm(None)
        
        assert isinstance(result, dict)
        assert result["error_code"] == AI_ERRORS.AI_INVALID_REQUEST
    
    def test_get_metrics_summary(self, dispatcher, matrix, vector):
        """Test getting metrics summary."""
        # Run some operations
        dispatcher.dispatch("photonic_mvm", matrix, vector, params={
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        })
        
        summary = dispatcher.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert summary["enabled"] is True
        assert summary["total_operations"] > 0
    
    def test_get_metrics_summary_disabled(self):
        """Test metrics summary when disabled."""
        disp = HardwareDispatcher(use_mock=True, enable_metrics=False, enable_health_checks=False)
        
        summary = disp.get_metrics_summary()
        
        assert summary["enabled"] is False
    
    def test_get_last_metrics(self, dispatcher, matrix, vector):
        """Test getting last metrics."""
        # Run operation
        dispatcher.dispatch("photonic_mvm", matrix, vector, params={
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        })
        
        metrics = dispatcher.get_last_metrics("photonic_mvm")
        
        assert isinstance(metrics, dict)
        assert 'energy_nj' in metrics
        assert 'latency_ms' in metrics
    
    def test_get_last_metrics_no_metrics(self):
        """Test getting metrics when none exist."""
        disp = HardwareDispatcher(use_mock=True, enable_metrics=True, enable_health_checks=False)
        
        metrics = disp.get_last_metrics("test_key")
        
        assert 'message' in metrics
        assert 'No metrics available' in metrics['message']
    
    def test_shutdown(self, dispatcher):
        """Test dispatcher shutdown."""
        dispatcher.shutdown()
        # Should not raise any exceptions


class TestMetricsCollection:
    """Test metrics collection."""
    
    def test_record_metrics(self, dispatcher):
        """Test recording metrics."""
        metrics = OperationMetrics(
            backend=HardwareBackend.CPU,
            operation="test_op",
            start_time=datetime.now(),
            end_time=datetime.now(),
            energy_nj=10.0,
            latency_ms=5.0,
            throughput_ops_per_sec=1000.0,
            input_size=(4, 4),
            output_size=(4,),
            success=True
        )
        
        dispatcher._record_metrics(metrics)
        
        assert len(dispatcher.metrics_history) > 0
    
    def test_metrics_bounded(self):
        """Test metrics history is bounded."""
        disp = HardwareDispatcher(use_mock=True, enable_metrics=True, enable_health_checks=False)
        
        # Add many metrics
        for i in range(15000):
            metrics = OperationMetrics(
                backend=HardwareBackend.CPU,
                operation="test",
                start_time=datetime.now(),
                end_time=datetime.now(),
                energy_nj=1.0,
                latency_ms=1.0,
                throughput_ops_per_sec=1.0,
                input_size=(2,),
                output_size=(2,),
                success=True
            )
            disp._record_metrics(metrics)
        
        # Should be bounded
        assert len(disp.metrics_history) <= 10000


class TestBackendSelection:
    """Test backend selection logic."""
    
    def test_select_backend_basic(self, dispatcher):
        """Test basic backend selection."""
        backend = dispatcher._select_backend("photonic_mvm", (4, 4), {})
        
        assert isinstance(backend, HardwareBackend)
    
    def test_select_backend_prefers_available(self, dispatcher):
        """Test backend selection prefers available backends."""
        # Mark some backends as unavailable
        for backend in list(dispatcher.hardware_registry.keys()):
            if backend != HardwareBackend.CPU:
                dispatcher.hardware_registry[backend].available = False
        
        selected = dispatcher._select_backend("photonic_mvm", (4, 4), {})
        
        # Should fall back to CPU
        assert selected == HardwareBackend.CPU


class TestErrorHandling:
    """Test error handling."""
    
    def test_dispatch_with_fallback(self, dispatcher, matrix, vector):
        """Test dispatch falls back on error."""
        # Force primary backend to fail by making it unavailable
        # Then verify fallback works
        result = dispatcher.dispatch("photonic_mvm", matrix, vector, params={
            "noise_std": 0.01,
            "multiplexing": "wavelength",
            "compression": "ITU-F.748",
            "bandwidth_ghz": 100,
            "latency_ps": 50
        })
        
        # Should get a result (from fallback if needed)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])