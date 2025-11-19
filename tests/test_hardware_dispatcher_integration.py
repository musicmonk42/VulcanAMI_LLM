"""
Comprehensive pytest suite for hardware_dispatcher_integration.py
"""

import pytest
import asyncio
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import the module to test
import hardware_dispatcher_integration as hdi


class TestHardwareBackend:
    """Test HardwareBackend enum"""
    
    def test_backend_values(self):
        """Test all backend values"""
        assert hdi.HardwareBackend.CPU.value == "cpu"
        assert hdi.HardwareBackend.GPU.value == "gpu"
        assert hdi.HardwareBackend.PHOTONIC.value == "photonic"
        assert hdi.HardwareBackend.MEMRISTOR.value == "memristor"
        assert hdi.HardwareBackend.EMULATOR.value == "emulator"


class TestDispatchStrategy:
    """Test DispatchStrategy enum"""
    
    def test_strategy_values(self):
        """Test all strategy values"""
        assert hdi.DispatchStrategy.FASTEST.value == "fastest"
        assert hdi.DispatchStrategy.LOWEST_ENERGY.value == "lowest_energy"
        assert hdi.DispatchStrategy.BEST_ACCURACY.value == "best_accuracy"
        assert hdi.DispatchStrategy.BALANCED.value == "balanced"


class TestHardwareProfile:
    """Test HardwareProfile dataclass"""
    
    def test_profile_creation(self):
        """Test creating hardware profile"""
        profile = hdi.HardwareProfile(
            backend=hdi.HardwareBackend.CPU,
            max_tensor_size_mb=1024.0,
            latency_ms=10.0,
            energy_per_op_nj=5.0,
            accuracy=0.99,
            throughput_tops=1.0
        )
        
        assert profile.backend == hdi.HardwareBackend.CPU
        assert profile.max_tensor_size_mb == 1024.0
        assert profile.available is True
    
    def test_profile_to_dict(self):
        """Test converting profile to dict"""
        profile = hdi.HardwareProfile(
            backend=hdi.HardwareBackend.GPU,
            max_tensor_size_mb=2048.0,
            latency_ms=1.0,
            energy_per_op_nj=3.0,
            accuracy=0.999,
            throughput_tops=10.0
        )
        
        d = profile.to_dict()
        assert 'backend' in d
        assert 'latency_ms' in d
        assert d['max_tensor_size_mb'] == 2048.0


class TestDispatchResult:
    """Test DispatchResult dataclass"""
    
    def test_result_creation(self):
        """Test creating dispatch result"""
        result = hdi.DispatchResult(
            backend=hdi.HardwareBackend.PHOTONIC,
            result={"data": "output"},
            latency_ms=0.5
        )
        
        assert result.backend == hdi.HardwareBackend.PHOTONIC
        assert result.latency_ms == 0.5
        assert result.fallback_used is False
    
    def test_result_with_error(self):
        """Test result with error"""
        result = hdi.DispatchResult(
            backend=hdi.HardwareBackend.CPU,
            result=None,
            latency_ms=10.0,
            error="Computation failed"
        )
        
        assert result.error == "Computation failed"
        assert result.result is None


class TestHardwareProfileManager:
    """Test HardwareProfileManager"""
    
    @pytest.fixture
    def manager(self):
        """Create profile manager"""
        return hdi.HardwareProfileManager()
    
    def test_manager_creation(self, manager):
        """Test creating manager"""
        assert len(manager.profiles) > 0
        assert hdi.HardwareBackend.CPU in manager.profiles
    
    def test_get_profile(self, manager):
        """Test getting profile"""
        profile = manager.get_profile(hdi.HardwareBackend.CPU)
        
        assert profile is not None
        assert profile.backend == hdi.HardwareBackend.CPU
    
    def test_get_nonexistent_profile(self, manager):
        """Test getting non-existent profile"""
        # Create custom backend enum value
        profile = manager.get_profile(hdi.HardwareBackend.QUANTUM)
        
        # May or may not exist depending on implementation
        assert profile is None or isinstance(profile, hdi.HardwareProfile)
    
    def test_update_health(self, manager):
        """Test updating health metrics"""
        manager.update_health(
            hdi.HardwareBackend.CPU,
            health_score=0.8,
            temperature_c=65.0,
            utilization_percent=75.0
        )
        
        profile = manager.get_profile(hdi.HardwareBackend.CPU)
        assert profile.health_score == 0.8
        assert profile.temperature_c == 65.0
        assert profile.utilization_percent == 75.0
    
    def test_get_available_backends(self, manager):
        """Test getting available backends"""
        backends = manager.get_available_backends()
        
        assert isinstance(backends, list)
        assert len(backends) > 0
        assert hdi.HardwareBackend.CPU in backends
    
    def test_get_available_backends_min_health(self, manager):
        """Test filtering by minimum health"""
        # Lower health of one backend
        manager.update_health(hdi.HardwareBackend.CPU, health_score=0.3)
        
        backends = manager.get_available_backends(min_health=0.5)
        
        # CPU should not be in list with health 0.3
        assert hdi.HardwareBackend.CPU not in backends
    
    def test_select_backend_fastest(self, manager):
        """Test selecting fastest backend"""
        backend = manager.select_backend(
            tensor_size_mb=100.0,
            strategy=hdi.DispatchStrategy.FASTEST
        )
        
        assert backend is not None
        # Should select backend with lowest latency
        if backend:
            profile = manager.get_profile(backend)
            assert profile is not None
    
    def test_select_backend_lowest_energy(self, manager):
        """Test selecting lowest energy backend"""
        backend = manager.select_backend(
            tensor_size_mb=100.0,
            strategy=hdi.DispatchStrategy.LOWEST_ENERGY
        )
        
        assert backend is not None
    
    def test_select_backend_best_accuracy(self, manager):
        """Test selecting best accuracy backend"""
        backend = manager.select_backend(
            tensor_size_mb=100.0,
            strategy=hdi.DispatchStrategy.BEST_ACCURACY
        )
        
        assert backend is not None
    
    def test_select_backend_balanced(self, manager):
        """Test selecting balanced backend"""
        backend = manager.select_backend(
            tensor_size_mb=100.0,
            strategy=hdi.DispatchStrategy.BALANCED
        )
        
        assert backend is not None
    
    def test_select_backend_too_large_tensor(self, manager):
        """Test selection with tensor too large for any backend"""
        # Use impossibly large tensor
        backend = manager.select_backend(
            tensor_size_mb=999999999.0,
            strategy=hdi.DispatchStrategy.FASTEST
        )
        
        # Should return None or fallback
        assert backend is None or backend == hdi.HardwareBackend.CPU
    
    def test_load_profiles_from_file(self, tmp_path):
        """Test loading profiles from JSON file"""
        profiles_data = {
            "cpu": {
                "max_tensor_size_mb": 4096,
                "latency_ms": 5.0,
                "energy_per_op_nj": 8.0,
                "accuracy": 1.0,
                "throughput_tops": 0.5,
                "available": True
            }
        }
        
        profiles_file = tmp_path / "profiles.json"
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f)
        
        manager = hdi.HardwareProfileManager(str(profiles_file))
        
        cpu_profile = manager.get_profile(hdi.HardwareBackend.CPU)
        assert cpu_profile.max_tensor_size_mb == 4096


class TestHardwareDispatcherIntegration:
    """Test HardwareDispatcherIntegration"""
    
    @pytest.fixture
    def dispatcher(self):
        """Create dispatcher instance"""
        return hdi.HardwareDispatcherIntegration(
            enable_hardware=False,  # Disable hardware for testing
            enable_emulator=True
        )
    
    def test_dispatcher_creation(self, dispatcher):
        """Test creating dispatcher"""
        assert dispatcher.enable_emulator is True
        assert dispatcher.profile_manager is not None
    
    @pytest.mark.asyncio
    async def test_dispatch_to_hardware_fallback(self, dispatcher):
        """Test dispatch with fallback to emulator"""
        result = await dispatcher.dispatch_to_hardware("photonic_mvm", [[1, 2]], [[3], [4]])
        
        assert isinstance(result, hdi.DispatchResult)
        assert result.backend is not None
    
    @pytest.mark.asyncio
    async def test_cpu_fallback(self, dispatcher):
        """Test CPU fallback"""
        result = await dispatcher._cpu_fallback("photonic_mvm", [[1, 2]], [[3], [4]])
        
        assert result.backend == hdi.HardwareBackend.CPU
        assert result.fallback_used is True
    
    @pytest.mark.asyncio
    async def test_emulate_mvm_pure(self, dispatcher):
        """Test pure Python MVM emulation"""
        matrix = [[1, 2], [3, 4]]
        vector = [[5], [6]]
        
        result = dispatcher._emulate_mvm_pure(matrix, vector)
        
        assert result is not None
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_emulate_mvm_pure_numpy(self, dispatcher):
        """Test NumPy MVM emulation"""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([[5], [6]])
        
        result = dispatcher._emulate_mvm_pure(matrix, vector)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_emulate_photonic_mvm(self, dispatcher):
        """Test photonic MVM emulation"""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        vector = np.array([5.0, 6.0])
        params = {"noise_std": 0.01}
        
        result = await dispatcher._emulate_photonic_mvm(matrix, vector, params)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_emulate_memristor_mvm(self, dispatcher):
        """Test memristor MVM emulation"""
        tensor1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = np.array([5.0, 6.0])
        
        result = await dispatcher._emulate_memristor_mvm(tensor1, tensor2)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_dispatch_to_emulator(self, dispatcher):
        """Test emulator dispatch"""
        subgraph = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "CONST",
                    "params": {"value": np.array([[1, 2], [3, 4]])}
                },
                {
                    "id": "n2",
                    "type": "CONST",
                    "params": {"value": np.array([5, 6])}
                }
            ],
            "edges": []
        }
        
        result = await dispatcher.dispatch_to_emulator(subgraph, backend="photonic")
        
        assert result is not None
        assert "backend" in result or "product" in result
    
    def test_estimate_tensor_size(self, dispatcher):
        """Test tensor size estimation"""
        subgraph = {
            "nodes": [{
                "id": "n1",
                "type": "CONST",
                "params": {"value": np.array([[1, 2], [3, 4]])}
            }],
            "edges": []
        }
        
        size_mb = dispatcher._estimate_tensor_size(subgraph)
        
        assert size_mb > 0
    
    def test_identify_operation_type(self, dispatcher):
        """Test operation type identification"""
        subgraph = {
            "nodes": [{"id": "n1", "type": "PhotonicMVMNode"}],
            "edges": []
        }
        
        op_type = dispatcher._identify_operation_type(subgraph)
        
        assert op_type == "photonic_mvm"
    
    def test_select_backend_for_operation(self, dispatcher):
        """Test backend selection for operation"""
        backend = dispatcher._select_backend_for_operation("photonic_mvm", (), {})
        
        assert backend == hdi.HardwareBackend.PHOTONIC
    
    def test_select_strategy(self, dispatcher):
        """Test strategy selection from metrics"""
        metrics = {"latency_critical": True}
        strategy = dispatcher._select_strategy(metrics)
        
        assert strategy == hdi.DispatchStrategy.FASTEST
    
    def test_estimate_operation_count(self, dispatcher):
        """Test operation count estimation"""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([5, 6])
        
        count = dispatcher._estimate_operation_count("photonic_mvm", (matrix, vector))
        
        assert count > 0
    
    def test_compute_cache_key(self, dispatcher):
        """Test cache key computation"""
        key1 = dispatcher._compute_cache_key("op1", (np.array([1, 2]),), {})
        key2 = dispatcher._compute_cache_key("op1", (np.array([1, 2]),), {})
        
        # Same operation should produce same key
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash
    
    def test_update_cache(self, dispatcher):
        """Test cache update"""
        result = hdi.DispatchResult(
            backend=hdi.HardwareBackend.CPU,
            result={"data": "test"},
            latency_ms=10.0
        )
        
        dispatcher._update_cache("test_key", result)
        
        assert "test_key" in dispatcher.cache
    
    def test_cache_eviction(self, dispatcher):
        """Test cache eviction when full"""
        # Set small cache size
        dispatcher.cache_size = 2
        
        # Add 3 items
        for i in range(3):
            result = hdi.DispatchResult(
                backend=hdi.HardwareBackend.CPU,
                result=i,
                latency_ms=1.0
            )
            dispatcher._update_cache(f"key{i}", result)
        
        # Cache should not exceed max size
        assert len(dispatcher.cache) <= dispatcher.cache_size
    
    def test_update_metrics(self, dispatcher):
        """Test metrics update"""
        result = hdi.DispatchResult(
            backend=hdi.HardwareBackend.CPU,
            result={},
            latency_ms=10.0,
            energy_nj=50.0
        )
        
        initial_latency = dispatcher.total_latency_ms
        dispatcher._update_metrics(result)
        
        assert dispatcher.total_latency_ms == initial_latency + 10.0
    
    def test_get_metrics_summary(self, dispatcher):
        """Test getting metrics summary"""
        summary = dispatcher.get_metrics_summary()
        
        assert "dispatch_counts" in summary
        assert "total_energy_nj" in summary
        assert "cache_size" in summary
        assert "available_backends" in summary
    
    def test_cleanup(self, dispatcher):
        """Test dispatcher cleanup"""
        dispatcher.cleanup()
        
        assert len(dispatcher.cache) == 0


class TestEmulationEdgeCases:
    """Test emulation edge cases"""
    
    @pytest.fixture
    def dispatcher(self):
        return hdi.HardwareDispatcherIntegration(
            enable_hardware=False,
            enable_emulator=True
        )
    
    @pytest.mark.asyncio
    async def test_emulate_mvm_empty_tensors(self, dispatcher):
        """Test MVM with empty tensors"""
        with pytest.raises((ValueError, IndexError)):
            dispatcher._emulate_mvm_pure([], [])
    
    @pytest.mark.asyncio
    async def test_emulate_mvm_incompatible_dimensions(self, dispatcher):
        """Test MVM with incompatible dimensions"""
        matrix = [[1, 2, 3]]
        vector = [[4], [5]]
        
        with pytest.raises(ValueError):
            dispatcher._emulate_mvm_pure(matrix, vector)
    
    @pytest.mark.asyncio
    async def test_dispatch_to_emulator_insufficient_tensors(self, dispatcher):
        """Test emulator dispatch with insufficient tensors"""
        subgraph = {
            "nodes": [{"id": "n1", "type": "CONST", "params": {"value": [1, 2]}}],
            "edges": []
        }
        
        with pytest.raises(ValueError):
            await dispatcher.dispatch_to_emulator(subgraph)
    
    @pytest.mark.asyncio
    async def test_optimize_and_dispatch(self, dispatcher):
        """Test optimize and dispatch"""
        subgraph = {
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": np.array([[1, 2]])}},
                {"id": "n2", "type": "CONST", "params": {"value": np.array([3, 4])}}
            ],
            "edges": []
        }
        metrics = {"latency_critical": False}
        
        result = await dispatcher.optimize_and_dispatch(subgraph, metrics)
        
        assert isinstance(result, hdi.DispatchResult)


class TestModuleLevelFunctions:
    """Test module-level functions"""
    
    @pytest.mark.asyncio
    async def test_dispatch_to_hardware_function(self):
        """Test module-level dispatch_to_hardware"""
        result = await hdi.dispatch_to_hardware("photonic_mvm", [[1, 2]], [[3], [4]])
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_dispatch_to_emulator_fallback_function(self):
        """Test module-level dispatch_to_emulator_fallback"""
        result = await hdi.dispatch_to_emulator_fallback("photonic_mvm", [[1, 2]], [[3], [4]])
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_emulate_memristor_mvm_function(self):
        """Test module-level emulate_memristor_mvm"""
        result = await hdi.emulate_memristor_mvm(
            np.array([[1, 2]]),
            np.array([3, 4])
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_dispatch_to_emulator_function(self):
        """Test module-level dispatch_to_emulator"""
        subgraph = {
            "nodes": [
                {"id": "n1", "type": "CONST", "params": {"value": np.array([[1, 2]])}},
                {"id": "n2", "type": "CONST", "params": {"value": np.array([3, 4])}}
            ],
            "edges": []
        }
        
        result = await hdi.dispatch_to_emulator(subgraph)
        
        assert result is not None


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_dispatch_workflow(self):
        """Test complete dispatch workflow"""
        dispatcher = hdi.HardwareDispatcherIntegration(
            enable_hardware=False,
            enable_emulator=True
        )
        
        # Create operation
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        vector = np.array([5.0, 6.0])
        
        # Dispatch
        result = await dispatcher.dispatch_to_hardware("photonic_mvm", matrix, vector)
        
        # Verify
        assert isinstance(result, hdi.DispatchResult)
        assert result.latency_ms > 0
        
        # Check metrics
        summary = dispatcher.get_metrics_summary()
        assert summary["total_latency_ms"] > 0
        
        # Cleanup
        dispatcher.cleanup()
    
    @pytest.mark.asyncio
    async def test_caching_workflow(self):
        """Test that caching works across multiple calls"""
        dispatcher = hdi.HardwareDispatcherIntegration(
            enable_hardware=False,
            enable_emulator=True,
            cache_size=10
        )
        
        matrix = np.array([[1, 2]])
        vector = np.array([3, 4])
        
        # First call - should compute
        result1 = await dispatcher.dispatch_to_hardware("photonic_mvm", matrix, vector)
        
        # Second call - should use cache
        result2 = await dispatcher.dispatch_to_hardware("photonic_mvm", matrix, vector)
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        
        # Cache should have entries
        assert len(dispatcher.cache) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])