#!/usr/bin/env python3
"""
Comprehensive test suite for demo_graphix.py
Target: 85%+ code coverage

Run with:
    pytest test_demo_graphix.py -v --cov=demo_graphix --cov-report=term-missing
    pytest test_demo_graphix.py -v --cov=demo_graphix --cov-report=html
"""

import pytest
import asyncio
import json
import tempfile
import pickle
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime
import numpy as np

# Add demo directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
demo_dir = project_root / "demo"
src_dir = project_root / "src"

# Add paths in order of priority
for path in [str(project_root), str(demo_dir), str(src_dir), str(test_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Store modules to mock - DON'T mock them at module level!
# They'll be mocked by the mock_demo_modules fixture
_MODULES_TO_MOCK = [
    'src.graphix_client',
    'src.tournament_manager',
    'src.unified_runtime',
    'src.nso_aligner',
    'src.observability_manager',
    'src.stdio_policy',
    'src.hardware_dispatcher',
    'src.security_audit_engine',
]

# Temporarily mock for import of demo_graphix, then restore
_temp_mocks = {}
for mod_name in _MODULES_TO_MOCK:
    _temp_mocks[mod_name] = sys.modules.get(mod_name)
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Import the module under test
import demo_graphix
from demo_graphix import (
    EnhancedGraphixDemo,
    DemoConfig,
    StepResult,
    DemoPhase,
    PersistentResultCache,
    setup_logging
)

# IMMEDIATELY restore original modules after import to prevent pollution
for mod_name, original_mod in _temp_mocks.items():
    if original_mod is None:
        # Remove the temporary mock
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    else:
        # Restore original
        sys.modules[mod_name] = original_mod


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def mock_demo_modules():
    """
    Mock external modules for each test in this module.
    This ensures mocks are active during test execution.
    """
    mocks = {}
    original_modules = {}
    
    for mod_name in _MODULES_TO_MOCK:
        original_modules[mod_name] = sys.modules.get(mod_name)
        mocks[mod_name] = MagicMock()
        sys.modules[mod_name] = mocks[mod_name]
    
    yield mocks
    
    # Restore after each test
    for mod_name, original_mod in original_modules.items():
        if original_mod is None:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        else:
            sys.modules[mod_name] = original_mod


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def demo_config(temp_output_dir):
    """Create test configuration."""
    return DemoConfig(
        graph_type="sentiment_3d",
        photonic=False,
        output_dir=temp_output_dir,
        verbose=True,
        max_retries=2,
        parallel=False,
        interactive=False,
        cache_enabled=True,
        timeout_seconds=10,
        registry_endpoint="http://localhost:5000",
        agent_endpoint="http://127.0.0.1:8000",
        verify_ssl=False,
        trusted_hosts=["localhost", "127.0.0.1"]
    )


@pytest.fixture
def mock_graph():
    """Create mock graph data."""
    return {
        "id": "test_graph_123",
        "nodes": [{"id": "n1"}, {"id": "n2"}],
        "edges": [{"from": "n1", "to": "n2"}],
        "metadata": {
            "goal": "test",
            "ethical_label": "EU2025:Safe"
        }
    }


@pytest.fixture
def mock_components():
    """Mock all external components."""
    with patch('demo_graphix.GraphixClient') as mock_client, \
         patch('demo_graphix.TournamentManager') as mock_tournament, \
         patch('demo_graphix.UnifiedRuntime') as mock_runtime, \
         patch('demo_graphix.NSOAligner') as mock_nso, \
         patch('demo_graphix.ObservabilityManager') as mock_obs, \
         patch('demo_graphix.HardwareDispatcher') as mock_hardware, \
         patch('demo_graphix.SecurityAuditEngine') as mock_audit:
        
        # Setup return values
        mock_client_instance = Mock()
        mock_client_instance.submit_graph_proposal = AsyncMock(return_value={"id": "generated"})
        mock_client.return_value = mock_client_instance
        
        mock_tournament_instance = Mock()
        mock_tournament_instance.run_adaptive_tournament = Mock(return_value=[0, 1, 2])
        mock_tournament.return_value = mock_tournament_instance
        
        mock_runtime_instance = Mock()
        mock_runtime_instance.execute_graph = AsyncMock(return_value={"status": "completed"})
        mock_runtime.return_value = mock_runtime_instance
        
        mock_nso_instance = Mock()
        mock_nso_instance.multi_model_audit = Mock(return_value="safe")
        mock_nso.return_value = mock_nso_instance
        
        mock_obs_instance = Mock()
        mock_obs_instance.export_dashboard = Mock(return_value={"dashboard": "data"})
        mock_obs_instance.plot_semantic_map = Mock(return_value=Path("map.png"))
        mock_obs.return_value = mock_obs_instance
        
        mock_hardware_instance = Mock()
        mock_hardware_instance.get_photonic_params = AsyncMock(return_value={
            "energy_nj": 0.4,
            "latency_ps": 50,
            "source": "hardware"
        })
        mock_hardware.return_value = mock_hardware_instance
        
        mock_audit_instance = Mock()
        mock_audit_instance.log_event = Mock()
        mock_audit.return_value = mock_audit_instance
        
        yield {
            "client": mock_client_instance,
            "tournament": mock_tournament_instance,
            "runtime": mock_runtime_instance,
            "nso": mock_nso_instance,
            "obs": mock_obs_instance,
            "hardware": mock_hardware_instance,
            "audit": mock_audit_instance
        }


# ============================================================================
# Test PersistentResultCache
# ============================================================================

class TestPersistentResultCache:
    """Test cache functionality."""
    
    def test_cache_init_enabled(self, temp_output_dir):
        """Test cache initialization when enabled."""
        cache = PersistentResultCache(enabled=True, cache_file=temp_output_dir / "cache.pkl")
        assert cache.enabled is True
        assert cache._cache == {}
    
    def test_cache_init_disabled(self):
        """Test cache initialization when disabled."""
        cache = PersistentResultCache(enabled=False)
        assert cache.enabled is False
        assert cache._cache == {}
    
    def test_cache_get_set(self, temp_output_dir):
        """Test cache get and set operations."""
        cache = PersistentResultCache(cache_file=temp_output_dir / "cache.pkl")
        
        config = {"type": "test", "version": 1}
        result = {"data": "test_result"}
        
        # Set and get
        cache.set("phase1", config, result)
        retrieved = cache.get("phase1", config)
        
        assert retrieved == result
        assert cache._dirty is True
    
    def test_cache_get_miss(self, temp_output_dir):
        """Test cache miss."""
        cache = PersistentResultCache(cache_file=temp_output_dir / "cache.pkl")
        result = cache.get("nonexistent", {"config": "data"})
        assert result is None
    
    def test_cache_key_generation(self, temp_output_dir):
        """Test cache key generation."""
        cache = PersistentResultCache(cache_file=temp_output_dir / "cache.pkl")
        
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same data, different order
        
        key1 = cache.get_key("phase", config1)
        key2 = cache.get_key("phase", config2)
        
        # Keys should be identical (dict sorted)
        assert key1 == key2
    
    def test_cache_save_load(self, temp_output_dir):
        """Test cache persistence."""
        cache_file = temp_output_dir / "cache.pkl"
        
        # Create cache and save data
        cache1 = PersistentResultCache(cache_file=cache_file)
        cache1.set("phase1", {"test": 1}, {"result": "data"})
        cache1.save()
        
        # Load cache in new instance
        cache2 = PersistentResultCache(cache_file=cache_file)
        result = cache2.get("phase1", {"test": 1})
        
        assert result == {"result": "data"}
    
    def test_cache_clear(self, temp_output_dir):
        """Test cache clearing."""
        cache = PersistentResultCache(cache_file=temp_output_dir / "cache.pkl")
        cache.set("phase1", {"test": 1}, {"data": "test"})
        
        cache.clear()
        
        assert cache._cache == {}
        # Note: clear() calls save() which sets _dirty = False
        assert cache._dirty is False
    
    def test_cache_disabled_operations(self):
        """Test that disabled cache doesn't store data."""
        cache = PersistentResultCache(enabled=False)
        cache.set("phase1", {"test": 1}, {"data": "test"})
        result = cache.get("phase1", {"test": 1})
        
        assert result is None


# ============================================================================
# Test DemoConfig and StepResult
# ============================================================================

class TestDataClasses:
    """Test data classes."""
    
    def test_demo_config_defaults(self, temp_output_dir):
        """Test DemoConfig default values."""
        config = DemoConfig(
            graph_type="test",
            photonic=True,
            output_dir=temp_output_dir,
            verbose=False
        )
        
        assert config.max_retries == 3
        assert config.parallel is True
        assert config.cache_enabled is True
        assert config.timeout_seconds == 300
    
    def test_step_result_creation(self):
        """Test StepResult creation."""
        result = StepResult(
            step_name="test_step",
            status="success",
            duration_ms=123.45,
            data={"key": "value"},
            retries=2,
            data_source="real"
        )
        
        assert result.step_name == "test_step"
        assert result.status == "success"
        assert result.duration_ms == 123.45
        assert result.data == {"key": "value"}
        assert result.retries == 2
        assert result.data_source == "real"
    
    def test_step_result_optional_fields(self):
        """Test StepResult with optional fields."""
        result = StepResult(
            step_name="test",
            status="failure",
            duration_ms=100.0,
            error="Test error"
        )
        
        assert result.error == "Test error"
        assert result.data is None


# ============================================================================
# Test EnhancedGraphixDemo Initialization
# ============================================================================

class TestEnhancedGraphixDemoInit:
    """Test demo initialization."""
    
    def test_init_success(self, demo_config, mock_components):
        """Test successful initialization."""
        demo = EnhancedGraphixDemo(demo_config)
        
        assert demo.config == demo_config
        assert demo.cache is not None
        assert demo.executor is not None
        assert demo.results == {}
    
    def test_init_creates_output_dir(self, demo_config, mock_components):
        """Test output directory creation."""
        demo = EnhancedGraphixDemo(demo_config)
        assert demo.config.output_dir.exists()
    
    def test_validate_endpoints_trusted(self, demo_config, mock_components):
        """Test endpoint validation for trusted hosts."""
        demo = EnhancedGraphixDemo(demo_config)
        # Should not raise any exceptions
        assert demo.client is not None
    
    def test_validate_endpoints_untrusted_warning(self, demo_config, mock_components, capsys):
        """Test warning for untrusted endpoints."""
        demo_config.registry_endpoint = "http://untrusted.example.com:5000"
        demo_config.trusted_hosts = ["localhost"]
        
        demo = EnhancedGraphixDemo(demo_config)
        
        # Check that warning was logged in stdout
        captured = capsys.readouterr()
        assert "not in trusted hosts" in captured.out
    
    def test_component_init_failure_handling(self, demo_config):
        """Test graceful handling of component initialization failures."""
        with patch('demo_graphix.GraphixClient', side_effect=Exception("Init failed")):
            demo = EnhancedGraphixDemo(demo_config)
            assert demo.client is None


# ============================================================================
# Test Retry Logic
# ============================================================================

class TestRetryLogic:
    """Test retry mechanisms."""
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self, demo_config, mock_components):
        """Test async retry succeeds on first attempt."""
        demo = EnhancedGraphixDemo(demo_config)
        
        mock_func = AsyncMock(return_value="success")
        result, retries = await demo._retry_async(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        assert retries == 0
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self, demo_config, mock_components):
        """Test async retry succeeds after failures."""
        demo = EnhancedGraphixDemo(demo_config)
        demo.config.max_retries = 3
        
        mock_func = AsyncMock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "success"
        ])
        
        result, retries = await demo._retry_async(mock_func)
        
        assert result == "success"
        assert retries == 2
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_all_fail(self, demo_config, mock_components):
        """Test async retry exhausts all attempts."""
        demo = EnhancedGraphixDemo(demo_config)
        demo.config.max_retries = 2
        
        mock_func = AsyncMock(side_effect=Exception("Always fails"))
        
        with pytest.raises(Exception, match="Always fails"):
            await demo._retry_async(mock_func)
        
        assert mock_func.call_count == 2
    
    def test_retry_sync_success(self, demo_config, mock_components):
        """Test sync retry succeeds."""
        demo = EnhancedGraphixDemo(demo_config)
        
        mock_func = Mock(return_value="success")
        result, retries = demo._retry_sync(mock_func)
        
        assert result == "success"
        assert retries == 0
    
    def test_retry_sync_after_failures(self, demo_config, mock_components):
        """Test sync retry with failures."""
        demo = EnhancedGraphixDemo(demo_config)
        demo.config.max_retries = 3
        
        mock_func = Mock(side_effect=[
            Exception("Fail 1"),
            Exception("Fail 2"),
            "success"
        ])
        
        result, retries = demo._retry_sync(mock_func)
        
        assert result == "success"
        assert retries == 2


# ============================================================================
# Test Demo Phases
# ============================================================================

class TestGenerateGraph:
    """Test graph generation phase."""
    
    @pytest.mark.asyncio
    async def test_generate_graph_success(self, demo_config, mock_components, mock_graph):
        """Test successful graph generation."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.return_value = mock_graph
        
        result = await demo.generate_graph()
        
        assert result.status == "success"
        assert result.data == mock_graph
        assert result.data_source == "real"
        assert (demo.config.output_dir / "generated_graph.json").exists()
    
    @pytest.mark.asyncio
    async def test_generate_graph_cached(self, demo_config, mock_components, mock_graph):
        """Test graph generation with cache hit."""
        demo = EnhancedGraphixDemo(demo_config)
        
        # Pre-populate cache
        cache_key = {"type": demo_config.graph_type, "photonic": demo_config.photonic}
        demo.cache.set("generation", cache_key, mock_graph)
        
        result = await demo.generate_graph()
        
        assert result.status == "success"
        assert result.data_source == "cached"
        assert demo.metrics["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_graph_failure(self, demo_config, mock_components):
        """Test graph generation failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.side_effect = Exception("Generation failed")
        
        result = await demo.generate_graph()
        
        assert result.status == "failure"
        assert "Generation failed" in result.error
        assert result.data_source == "error"
    
    @pytest.mark.asyncio
    async def test_generate_graph_no_client(self, demo_config):
        """Test graph generation without client."""
        with patch('demo_graphix.GraphixClient', None):
            demo = EnhancedGraphixDemo(demo_config)
            demo.client = None
            
            result = await demo.generate_graph()
            
            assert result.status == "failure"
            assert "not available" in result.error


class TestEvolveGraph:
    """Test graph evolution phase."""
    
    @pytest.mark.asyncio
    async def test_evolve_graph_success(self, demo_config, mock_components, mock_graph):
        """Test successful graph evolution."""
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.evolve_graph(mock_graph)
        
        assert result.status == "success"
        assert result.data is not None
        assert "metadata" in result.data
        assert result.data["metadata"]["evolution_winner"] is True
        assert (demo.config.output_dir / "evolved_graph.json").exists()
        assert (demo.config.output_dir / "evolution_metadata.json").exists()
    
    @pytest.mark.asyncio
    async def test_evolve_graph_failure(self, demo_config, mock_components, mock_graph):
        """Test graph evolution failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["tournament"].run_adaptive_tournament.side_effect = Exception("Evolution failed")
        
        result = await demo.evolve_graph(mock_graph)
        
        assert result.status == "failure"
        assert "Evolution failed" in result.error
    
    @pytest.mark.asyncio
    async def test_evolve_graph_no_tournament(self, demo_config, mock_graph):
        """Test evolution without tournament manager."""
        with patch('demo_graphix.TournamentManager', None):
            demo = EnhancedGraphixDemo(demo_config)
            demo.tournament = None
            
            result = await demo.evolve_graph(mock_graph)
            
            assert result.status == "failure"


class TestExecuteGraph:
    """Test graph execution phase."""
    
    @pytest.mark.asyncio
    async def test_execute_graph_success(self, demo_config, mock_components, mock_graph):
        """Test successful graph execution."""
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.execute_graph(mock_graph)
        
        assert result.status == "success"
        assert "photonic_meta" in result.data
        assert result.data["photonic_meta"]["data_source"] == "simulation"
        assert (demo.config.output_dir / "execution_result.json").exists()
    
    @pytest.mark.asyncio
    async def test_execute_graph_with_photonic_hardware(self, demo_config, mock_components, mock_graph):
        """Test execution with photonic hardware."""
        demo_config.photonic = True
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.execute_graph(mock_graph)
        
        assert result.status == "success"
        assert result.data["photonic_meta"]["data_source"] == "hardware"
    
    @pytest.mark.asyncio
    async def test_execute_graph_hardware_fallback(self, demo_config, mock_components, mock_graph):
        """Test fallback to simulation when hardware fails."""
        demo_config.photonic = True
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["hardware"].get_photonic_params.side_effect = Exception("Hardware unavailable")
        
        result = await demo.execute_graph(mock_graph)
        
        assert result.status == "success"
        assert result.data["photonic_meta"]["data_source"] == "simulation_fallback"
    
    @pytest.mark.asyncio
    async def test_execute_graph_timeout(self, demo_config, mock_components, mock_graph):
        """Test execution timeout."""
        demo = EnhancedGraphixDemo(demo_config)
        demo.config.timeout_seconds = 0.1
        
        async def slow_execute(*args):
            await asyncio.sleep(1)
            return {"status": "completed"}
        
        mock_components["runtime"].execute_graph = slow_execute
        
        result = await demo.execute_graph(mock_graph)
        
        assert result.status == "failure"
        assert "timeout" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_graph_failure(self, demo_config, mock_components, mock_graph):
        """Test execution failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["runtime"].execute_graph.side_effect = Exception("Execution failed")
        
        result = await demo.execute_graph(mock_graph)
        
        assert result.status == "failure"


class TestValidateEthics:
    """Test ethics validation phase."""
    
    @pytest.mark.asyncio
    async def test_validate_ethics_success(self, demo_config, mock_components, mock_graph):
        """Test successful ethics validation."""
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.validate_ethics(mock_graph)
        
        assert result.status == "success"
        assert result.data["result"] == "safe"
        assert result.data["compliance"]["EU2025"] is True
        assert (demo.config.output_dir / "ethics_report.json").exists()
    
    @pytest.mark.asyncio
    async def test_validate_ethics_unsafe(self, demo_config, mock_components, mock_graph):
        """Test unsafe ethics result."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["nso"].multi_model_audit.return_value = "unsafe"
        
        result = await demo.validate_ethics(mock_graph)
        
        assert result.status == "success"
        assert result.data["result"] == "unsafe"
        assert result.data["compliance"]["EU2025"] is False
        assert result.data["risk_level"] == "medium"
    
    @pytest.mark.asyncio
    async def test_validate_ethics_with_audit(self, demo_config, mock_components, mock_graph):
        """Test ethics validation with audit logging."""
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.validate_ethics(mock_graph)
        
        # Verify audit engine was called
        assert mock_components["audit"].log_event.called
    
    @pytest.mark.asyncio
    async def test_validate_ethics_failure(self, demo_config, mock_components, mock_graph):
        """Test ethics validation failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["nso"].multi_model_audit.side_effect = Exception("Audit failed")
        
        result = await demo.validate_ethics(mock_graph)
        
        assert result.status == "failure"


class TestGenerateVisualizations:
    """Test visualization generation phase."""
    
    @pytest.mark.asyncio
    async def test_generate_visualizations_success(self, demo_config, mock_components, mock_graph):
        """Test successful visualization generation."""
        demo = EnhancedGraphixDemo(demo_config)
        
        result = await demo.generate_visualizations(mock_graph)
        
        assert result.status == "success"
        assert "dashboard" in result.data
        assert "metrics" in result.data
        assert (demo.config.output_dir / "visualization_metadata.json").exists()
    
    @pytest.mark.asyncio
    async def test_generate_visualizations_with_results(self, demo_config, mock_components, mock_graph):
        """Test visualization with existing results."""
        demo = EnhancedGraphixDemo(demo_config)
        
        # Add some results
        demo.results["test_phase"] = StepResult(
            step_name="test",
            status="success",
            duration_ms=100.0,
            data_source="real"
        )
        
        result = await demo.generate_visualizations(mock_graph)
        
        assert result.status == "success"
        assert (demo.config.output_dir / "performance_metrics.json").exists()
    
    @pytest.mark.asyncio
    async def test_generate_visualizations_failure(self, demo_config, mock_components, mock_graph):
        """Test visualization generation failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["obs"].export_dashboard.side_effect = Exception("Viz failed")
        
        result = await demo.generate_visualizations(mock_graph)
        
        assert result.status == "failure"


# ============================================================================
# Test Parallel Execution
# ============================================================================

class TestParallelExecution:
    """Test parallel step execution."""
    
    @pytest.mark.asyncio
    async def test_run_parallel_steps_success(self, demo_config, mock_components, mock_graph):
        """Test successful parallel execution."""
        demo = EnhancedGraphixDemo(demo_config)
        
        results = await demo.run_parallel_steps(mock_graph)
        
        assert "execution" in results
        assert "ethics" in results
        assert "visualization" in results
        assert all(r.status == "success" for r in results.values())
    
    @pytest.mark.asyncio
    async def test_run_parallel_steps_with_failure(self, demo_config, mock_components, mock_graph):
        """Test parallel execution with one failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["runtime"].execute_graph.side_effect = Exception("Exec failed")
        
        results = await demo.run_parallel_steps(mock_graph)
        
        assert results["execution"].status == "failure"
        # Other steps should still complete
        assert "ethics" in results
        assert "visualization" in results


# ============================================================================
# Test Full Demo Run
# ============================================================================

class TestFullDemoRun:
    """Test complete demo execution."""
    
    @pytest.mark.asyncio
    async def test_run_sequential_success(self, demo_config, mock_components, mock_graph):
        """Test successful sequential demo run."""
        demo_config.parallel = False
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.return_value = mock_graph
        
        report = await demo.run()
        
        assert report["summary"]["total_steps"] > 0
        assert report["summary"]["successful"] > 0
        assert (demo.config.output_dir / "demo_report.json").exists()
    
    @pytest.mark.asyncio
    async def test_run_parallel_success(self, demo_config, mock_components, mock_graph):
        """Test successful parallel demo run."""
        demo_config.parallel = True
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.return_value = mock_graph
        
        report = await demo.run()
        
        assert report["summary"]["successful"] > 0
        assert DemoPhase.EXECUTION.value in demo.results
        assert DemoPhase.ETHICS.value in demo.results
    
    @pytest.mark.asyncio
    async def test_run_generation_failure_aborts(self, demo_config, mock_components):
        """Test that generation failure aborts demo."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.side_effect = Exception("Gen failed")
        
        report = await demo.run()
        
        assert DemoPhase.GENERATION.value in demo.results
        assert demo.results[DemoPhase.GENERATION.value].status == "failure"
        # Should not have other phases
        assert DemoPhase.EXECUTION.value not in demo.results
    
    @pytest.mark.asyncio
    async def test_run_evolution_failure_continues(self, demo_config, mock_components, mock_graph):
        """Test that evolution failure allows demo to continue."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["client"].submit_graph_proposal.return_value = mock_graph
        mock_components["tournament"].run_adaptive_tournament.side_effect = Exception("Evo failed")
        
        report = await demo.run()
        
        assert DemoPhase.EVOLUTION.value in demo.results
        assert DemoPhase.EXECUTION.value in demo.results
        # Should use generated graph instead of evolved


# ============================================================================
# Test Resource Management
# ============================================================================

class TestResourceManagement:
    """Test resource cleanup and context management."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self, demo_config, mock_components):
        """Test async context manager."""
        async with EnhancedGraphixDemo(demo_config) as demo:
            assert demo.executor is not None
        
        # After exit, executor should be shut down
        # (We can't directly test this without implementation details)
    
    @pytest.mark.asyncio
    async def test_close_method(self, demo_config, mock_components):
        """Test explicit close."""
        demo = EnhancedGraphixDemo(demo_config)
        demo.cache.set("test", {}, {"data": "test"})
        
        await demo.close()
        
        # Cache should be saved
        assert not demo.cache._dirty


# ============================================================================
# Test Reporting
# ============================================================================

class TestReporting:
    """Test report generation and summary functions."""
    
    def test_generate_summary_report(self, demo_config, mock_components):
        """Test summary report generation."""
        demo = EnhancedGraphixDemo(demo_config)
        
        # Add some results
        demo.results["phase1"] = StepResult(
            step_name="phase1",
            status="success",
            duration_ms=100.0,
            retries=1,
            data_source="real"
        )
        demo.results["phase2"] = StepResult(
            step_name="phase2",
            status="failure",
            duration_ms=50.0,
            error="Test error",
            data_source="error"
        )
        demo.metrics["total_duration_ms"] = 150.0
        
        report = demo.generate_summary_report()
        
        assert report["summary"]["total_steps"] == 2
        assert report["summary"]["successful"] == 1
        assert report["summary"]["failed"] == 1
        assert report["summary"]["total_duration_ms"] == 150.0
        assert "phase1" in report["results"]
        assert "phase2" in report["results"]
    
    def test_print_summary(self, demo_config, mock_components):
        """Test summary printing."""
        with patch('demo_graphix.safe_print') as mock_safe_print:
            demo = EnhancedGraphixDemo(demo_config)
            
            demo.results["test"] = StepResult(
                step_name="test",
                status="success",
                duration_ms=123.45,
                retries=2,
                data_source="real"
            )
            demo.metrics["total_duration_ms"] = 123.45
            demo.metrics["cache_hits"] = 1
            demo.metrics["cache_misses"] = 2
            
            demo.print_summary()
            
            # Verify safe_print was called with expected content
            assert mock_safe_print.called
            # Check that SUMMARY appears in one of the calls
            calls_text = ' '.join(str(call) for call in mock_safe_print.call_args_list)
            assert "SUMMARY" in calls_text


# ============================================================================
# Test Simulation Functions
# ============================================================================

class TestSimulation:
    """Test simulation and helper functions."""
    
    def test_simulate_photonic_metadata(self, demo_config, mock_components):
        """Test photonic metadata simulation."""
        demo = EnhancedGraphixDemo(demo_config)
        
        metadata = demo._simulate_photonic_metadata()
        
        assert "energy_nj" in metadata
        assert "latency_ps" in metadata
        assert metadata["source"] == "simulation"
        assert metadata["emulated"] is True
        assert "timestamp" in metadata


# ============================================================================
# Test Main and CLI
# ============================================================================

class TestMain:
    """Test main function and CLI."""
    
    @patch('demo_graphix.asyncio.run')
    @patch('sys.argv', ['demo_graphix.py', '--graph-type', 'sentiment_3d'])
    def test_main_basic(self, mock_asyncio_run):
        """Test main with basic arguments."""
        demo_graphix.main()
        
        assert mock_asyncio_run.called
    
    @patch('demo_graphix.asyncio.run')
    @patch('sys.argv', ['demo_graphix.py', '--photonic', '--parallel', '--verbose'])
    def test_main_with_options(self, mock_asyncio_run):
        """Test main with multiple options."""
        demo_graphix.main()
        
        assert mock_asyncio_run.called
    
    @patch('demo_graphix.asyncio.run', side_effect=KeyboardInterrupt)
    @patch('sys.argv', ['demo_graphix.py'])
    def test_main_keyboard_interrupt(self, mock_asyncio_run):
        """Test main with keyboard interrupt."""
        with pytest.raises(SystemExit) as exc_info:
            demo_graphix.main()
        
        assert exc_info.value.code == 130
    
    @patch('demo_graphix.asyncio.run', side_effect=Exception("Fatal error"))
    @patch('sys.argv', ['demo_graphix.py'])
    def test_main_exception(self, mock_asyncio_run):
        """Test main with exception."""
        with pytest.raises(SystemExit) as exc_info:
            demo_graphix.main()
        
        assert exc_info.value.code == 1


# ============================================================================
# Test Logging Setup
# ============================================================================

class TestLogging:
    """Test logging configuration."""
    
    def test_setup_logging_basic(self, temp_output_dir):
        """Test basic logging setup."""
        log_file = temp_output_dir / "test.log"
        logger = setup_logging(verbose=False, log_file=str(log_file))
        
        assert logger is not None
        assert log_file.exists()
    
    def test_setup_logging_verbose(self, temp_output_dir):
        """Test verbose logging setup."""
        import logging as log_module
        log_file = temp_output_dir / "test_verbose.log"
        logger = setup_logging(verbose=True, log_file=str(log_file))
        
        # Check the root logger level, not the named logger
        assert log_module.getLogger().level == log_module.DEBUG


# ============================================================================
# Additional Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_get_photonic_metadata_failure(self, demo_config, mock_components, mock_graph):
        """Test photonic metadata retrieval failure."""
        demo = EnhancedGraphixDemo(demo_config)
        mock_components["hardware"].get_photonic_params.side_effect = Exception("Hardware error")
        
        with pytest.raises(Exception):
            await demo._get_photonic_metadata(mock_graph)
    
    def test_cache_save_error_handling(self, temp_output_dir, caplog):
        """Test cache save with permission error."""
        cache = PersistentResultCache(cache_file=temp_output_dir / "cache.pkl")
        cache.set("test", {}, {"data": "test"})
        
        # Mock open to raise exception
        with patch('builtins.open', side_effect=PermissionError("No permission")):
            cache.save()
        
        assert "Failed to save cache" in caplog.text
    
    def test_cache_load_error_handling(self, temp_output_dir, caplog):
        """Test cache load with corrupted file."""
        cache_file = temp_output_dir / "corrupt.pkl"
        
        # Create corrupted cache file
        with open(cache_file, 'wb') as f:
            f.write(b'corrupted data')
        
        cache = PersistentResultCache(cache_file=cache_file)
        
        assert cache._cache == {}
        assert "Failed to load cache" in caplog.text


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=demo_graphix", "--cov-report=term-missing"])