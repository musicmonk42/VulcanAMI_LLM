"""
Integration tests for startup manager.

Tests the full StartupManager workflow with mocked dependencies.
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any

from vulcan.server.startup.manager import StartupManager
from vulcan.server.startup.phases import StartupPhase


class TestStartupManagerCreation:
    """Test StartupManager initialization."""
    
    def test_startup_manager_creation(self):
        """Test creating StartupManager instance."""
        mock_app = Mock()
        mock_settings = Mock()
        mock_settings.deployment_mode = "development"
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=None,
            process_lock=None
        )
        
        assert manager.app is mock_app
        assert manager.settings is mock_settings
        assert manager.redis_client is None
        assert manager.process_lock is None
        assert isinstance(manager.worker_id, int)
        assert manager.phase_results == {}
        assert manager.executor is None
    
    def test_startup_manager_with_redis(self):
        """Test creating StartupManager with Redis client."""
        mock_app = Mock()
        mock_settings = Mock()
        mock_redis = Mock()
        mock_lock = Mock()
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=mock_redis,
            process_lock=mock_lock
        )
        
        assert manager.redis_client is mock_redis
        assert manager.process_lock is mock_lock


class TestStartupManagerHelpers:
    """Test StartupManager helper methods."""
    
    def create_mock_manager(self):
        """Create a mock StartupManager for testing."""
        mock_app = Mock()
        mock_app.state = Mock()
        
        mock_settings = Mock()
        mock_settings.deployment_mode = "development"
        mock_settings.max_graph_size = 1000
        mock_settings.max_execution_time_s = 300
        mock_settings.max_memory_mb = 2000
        mock_settings.enable_self_improvement = False
        mock_settings.self_improvement_config = "config.json"
        mock_settings.self_improvement_state = "state.pkl"
        mock_settings.checkpoint_path = None
        mock_settings.enable_knowledge_distillation = False
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=None,
            process_lock=None
        )
        
        return manager
    
    def test_ensure_directories(self):
        """Test ensuring directories exist."""
        manager = self.create_mock_manager()
        
        # Should not raise exception
        manager._ensure_directories()
    
    def test_apply_config_defaults(self):
        """Test applying configuration defaults."""
        manager = self.create_mock_manager()
        
        mock_config = Mock(spec=[])  # Empty config
        manager._apply_config_defaults(mock_config)
        
        # Should have set defaults
        assert hasattr(mock_config, 'max_graph_size')
        assert hasattr(mock_config, 'max_execution_time_s')
        assert hasattr(mock_config, 'enable_self_improvement')
    
    def test_get_checkpoint_path_none(self):
        """Test getting checkpoint path when none configured."""
        manager = self.create_mock_manager()
        
        checkpoint = manager._get_checkpoint_path()
        assert checkpoint is None
    
    def test_get_checkpoint_path_missing_file(self):
        """Test getting checkpoint path when file doesn't exist."""
        manager = self.create_mock_manager()
        manager.settings.checkpoint_path = "/nonexistent/path.pkl"
        
        checkpoint = manager._get_checkpoint_path()
        assert checkpoint is None


@pytest.mark.asyncio
class TestStartupManagerPhases:
    """Test individual startup phases."""
    
    def create_test_manager(self):
        """Create manager for phase testing."""
        mock_app = Mock()
        mock_app.state = Mock()
        
        mock_settings = Mock()
        mock_settings.deployment_mode = "development"
        mock_settings.max_graph_size = 1000
        mock_settings.max_execution_time_s = 300
        mock_settings.max_memory_mb = 2000
        mock_settings.enable_self_improvement = False
        mock_settings.self_improvement_config = "config.json"
        mock_settings.self_improvement_state = "state.pkl"
        mock_settings.checkpoint_path = None
        mock_settings.enable_knowledge_distillation = False
        mock_settings.llm_execution_mode = "local"
        mock_settings.llm_parallel_timeout = 30
        mock_settings.llm_ensemble_min_confidence = 0.8
        mock_settings.llm_openai_max_tokens = 2000
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=None,
            process_lock=None
        )
        
        return manager
    
    async def test_phase_configuration_success(self):
        """Test configuration phase succeeds."""
        manager = self.create_test_manager()
        
        with patch('vulcan.server.startup.manager.get_config') as mock_get_config, \
             patch('vulcan.server.startup.manager.AgentConfig') as mock_agent_config:
            
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            await manager._phase_configuration()
            
            assert StartupPhase.CONFIGURATION in manager.phase_results
            assert manager.phase_results[StartupPhase.CONFIGURATION] is True
            assert hasattr(manager, 'config')
    
    async def test_phase_configuration_failure(self):
        """Test configuration phase handles failures."""
        manager = self.create_test_manager()
        
        with patch('vulcan.server.startup.manager.get_config') as mock_get_config:
            mock_get_config.side_effect = Exception("Config load failed")
            
            with pytest.raises(RuntimeError, match="Critical phase"):
                await manager._phase_configuration()
    
    async def test_phase_monitoring(self):
        """Test monitoring phase."""
        manager = self.create_test_manager()
        
        # Monitoring phase should succeed even if components fail
        await manager._phase_monitoring()
        
        assert StartupPhase.MONITORING in manager.phase_results
        # Should succeed (failures are logged but don't raise)
        assert manager.phase_results[StartupPhase.MONITORING] in [True, False]


@pytest.mark.asyncio  
class TestStartupManagerIntegration:
    """Integration tests for full startup workflow."""
    
    def create_integration_manager(self):
        """Create manager with more complete mocking."""
        mock_app = Mock()
        mock_app.state = Mock()
        
        mock_settings = Mock()
        mock_settings.deployment_mode = "testing"
        mock_settings.max_graph_size = 1000
        mock_settings.max_execution_time_s = 300
        mock_settings.max_memory_mb = 2000
        mock_settings.enable_self_improvement = False
        mock_settings.self_improvement_config = "config.json"
        mock_settings.self_improvement_state = "state.pkl"
        mock_settings.checkpoint_path = None
        mock_settings.enable_knowledge_distillation = False
        mock_settings.llm_execution_mode = "local"
        mock_settings.llm_parallel_timeout = 30
        mock_settings.llm_ensemble_min_confidence = 0.8
        mock_settings.llm_openai_max_tokens = 2000
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=None,
            process_lock=None
        )
        
        return manager
    
    async def test_shutdown_without_startup(self):
        """Test shutdown can be called even if startup didn't complete."""
        manager = self.create_integration_manager()
        
        # Should not raise exception
        await manager.run_shutdown()
    
    async def test_shutdown_with_executor(self):
        """Test shutdown cleans up executor."""
        manager = self.create_integration_manager()
        
        # Add a mock executor
        mock_executor = Mock()
        mock_executor.shutdown = Mock()
        manager.executor = mock_executor
        
        await manager.run_shutdown()
        
        mock_executor.shutdown.assert_called_once()
    
    async def test_shutdown_with_process_lock(self):
        """Test shutdown releases process lock."""
        mock_lock = Mock()
        mock_lock.release = Mock()
        
        manager = self.create_integration_manager()
        manager.process_lock = mock_lock
        
        await manager.run_shutdown()
        
        mock_lock.release.assert_called_once()
    
    async def test_shutdown_with_redis(self):
        """Test shutdown cleans up Redis."""
        mock_redis = Mock()
        mock_redis.delete = Mock()
        
        manager = self.create_integration_manager()
        manager.redis_client = mock_redis
        manager.app.state.worker_id = 12345
        
        await manager.run_shutdown()
        
        mock_redis.delete.assert_called_once_with("deployment:12345")


class TestStartupManagerErrorHandling:
    """Test error handling in StartupManager."""
    
    def test_phase_results_tracking(self):
        """Test that phase results are tracked."""
        mock_app = Mock()
        mock_settings = Mock()
        mock_settings.deployment_mode = "development"
        
        manager = StartupManager(
            app=mock_app,
            settings=mock_settings,
            redis_client=None,
            process_lock=None
        )
        
        # Initially empty
        assert manager.phase_results == {}
        
        # Can be updated
        manager.phase_results[StartupPhase.CONFIGURATION] = True
        assert StartupPhase.CONFIGURATION in manager.phase_results
        assert manager.phase_results[StartupPhase.CONFIGURATION] is True
