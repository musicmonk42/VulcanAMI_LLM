"""
Unit tests for startup subsystems manager.

Tests the SubsystemManager and SubsystemConfig classes.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any

from vulcan.server.startup.subsystems import (
    SubsystemManager,
    SubsystemConfig,
)


class TestSubsystemConfig:
    """Test suite for SubsystemConfig dataclass."""
    
    def test_subsystem_config_creation(self):
        """Test creating SubsystemConfig instances."""
        config = SubsystemConfig(
            attr_name="symbolic",
            display_name="Symbolic Reasoning"
        )
        
        assert config.attr_name == "symbolic"
        assert config.display_name == "Symbolic Reasoning"
        assert config.critical is False  # Default
        assert config.needs_init is False  # Default
        assert config.pre_check is None  # Default
    
    def test_subsystem_config_with_all_parameters(self):
        """Test SubsystemConfig with all parameters."""
        def pre_check(deps):
            return True
        
        config = SubsystemConfig(
            attr_name="test_subsystem",
            display_name="Test Subsystem",
            critical=True,
            needs_init=True,
            pre_check=pre_check
        )
        
        assert config.attr_name == "test_subsystem"
        assert config.display_name == "Test Subsystem"
        assert config.critical is True
        assert config.needs_init is True
        assert config.pre_check is pre_check


class TestSubsystemManager:
    """Test suite for SubsystemManager class."""
    
    def create_mock_deployment(self):
        """Create a mock deployment with deps."""
        mock_deployment = Mock()
        mock_deployment.collective = Mock()
        mock_deployment.collective.deps = Mock()
        return mock_deployment
    
    def test_subsystem_manager_creation(self):
        """Test creating SubsystemManager instance."""
        mock_deployment = self.create_mock_deployment()
        manager = SubsystemManager(mock_deployment)
        
        assert manager.deployment is mock_deployment
        assert manager.activated == []
        assert manager.failed == []
    
    def test_activate_single_subsystem_success(self):
        """Test activating a single subsystem successfully."""
        mock_deployment = self.create_mock_deployment()
        mock_subsystem = Mock()
        mock_deployment.collective.deps.symbolic = mock_subsystem
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("symbolic", "Symbolic Reasoning")
        
        result = manager._activate_single(config)
        
        assert result is True
        assert "Symbolic Reasoning" in manager.activated
        assert len(manager.failed) == 0
    
    def test_activate_single_subsystem_missing(self):
        """Test activating a subsystem that doesn't exist."""
        mock_deployment = self.create_mock_deployment()
        # Don't add the attribute
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("missing", "Missing Subsystem")
        
        result = manager._activate_single(config)
        
        assert result is False
        assert "Missing Subsystem" not in manager.activated
        assert len(manager.failed) == 0  # Non-critical
    
    def test_activate_single_critical_subsystem_missing(self):
        """Test activating a critical subsystem that doesn't exist."""
        mock_deployment = self.create_mock_deployment()
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("missing", "Missing Critical", critical=True)
        
        result = manager._activate_single(config)
        
        assert result is False
        assert "Missing Critical" not in manager.activated
        assert len(manager.failed) == 1
        assert manager.failed[0]["name"] == "Missing Critical"
    
    def test_activate_single_subsystem_none(self):
        """Test activating a subsystem that is None."""
        mock_deployment = self.create_mock_deployment()
        mock_deployment.collective.deps.symbolic = None
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("symbolic", "Symbolic Reasoning")
        
        result = manager._activate_single(config)
        
        assert result is False
        assert "Symbolic Reasoning" not in manager.activated
    
    def test_activate_single_with_initialization(self):
        """Test activating a subsystem that needs initialization."""
        mock_deployment = self.create_mock_deployment()
        mock_subsystem = Mock()
        mock_subsystem.initialize = Mock()
        mock_deployment.collective.deps.symbolic = mock_subsystem
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("symbolic", "Symbolic Reasoning", needs_init=True)
        
        result = manager._activate_single(config)
        
        assert result is True
        assert "Symbolic Reasoning" in manager.activated
        mock_subsystem.initialize.assert_called_once()
    
    def test_activate_single_with_pre_check_pass(self):
        """Test activating a subsystem with pre-check that passes."""
        mock_deployment = self.create_mock_deployment()
        mock_subsystem = Mock()
        mock_deployment.collective.deps.symbolic = mock_subsystem
        
        def pre_check(deps):
            return True
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig(
            "symbolic", 
            "Symbolic Reasoning",
            pre_check=pre_check
        )
        
        result = manager._activate_single(config)
        
        assert result is True
        assert "Symbolic Reasoning" in manager.activated
    
    def test_activate_single_with_pre_check_fail(self):
        """Test activating a subsystem with pre-check that fails."""
        mock_deployment = self.create_mock_deployment()
        mock_subsystem = Mock()
        mock_deployment.collective.deps.symbolic = mock_subsystem
        
        def pre_check(deps):
            return False
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig(
            "symbolic", 
            "Symbolic Reasoning",
            pre_check=pre_check
        )
        
        result = manager._activate_single(config)
        
        assert result is False
        assert "Symbolic Reasoning" not in manager.activated
    
    def test_activate_single_exception_non_critical(self):
        """Test handling exception in non-critical subsystem."""
        mock_deployment = self.create_mock_deployment()
        mock_subsystem = Mock()
        mock_deployment.collective.deps.symbolic = mock_subsystem
        
        # Make accessing the subsystem raise an exception
        type(mock_deployment.collective.deps).symbolic = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )
        
        manager = SubsystemManager(mock_deployment)
        config = SubsystemConfig("symbolic", "Symbolic Reasoning", critical=False)
        
        result = manager._activate_single(config)
        
        assert result is False
        assert len(manager.failed) == 0  # Non-critical errors not tracked in failed
    
    def test_activate_reasoning_subsystems(self):
        """Test activating reasoning subsystems."""
        mock_deployment = self.create_mock_deployment()
        
        # Create mock subsystems
        mock_deployment.collective.deps.symbolic = Mock()
        mock_deployment.collective.deps.probabilistic = Mock()
        mock_deployment.collective.deps.causal = Mock()
        
        manager = SubsystemManager(mock_deployment)
        count = manager.activate_reasoning_subsystems()
        
        # Should activate at least the ones we created
        assert count >= 3
        assert len(manager.activated) >= 3
    
    def test_activate_memory_subsystems(self):
        """Test activating memory subsystems."""
        mock_deployment = self.create_mock_deployment()
        
        # Create mock memory subsystems
        mock_deployment.collective.deps.ltm = Mock()
        mock_deployment.collective.deps.am = Mock()
        
        manager = SubsystemManager(mock_deployment)
        count = manager.activate_memory_subsystems()
        
        assert count >= 2
        assert len(manager.activated) >= 2
    
    def test_activate_agent_pool_success(self):
        """Test activating agent pool successfully."""
        mock_deployment = self.create_mock_deployment()
        
        mock_pool = Mock()
        mock_pool.get_pool_status.return_value = {
            "total_agents": 5,
            "state_distribution": {"idle": 3}
        }
        mock_pool.min_agents = 3
        mock_deployment.collective.agent_pool = mock_pool
        
        manager = SubsystemManager(mock_deployment)
        result = manager.activate_agent_pool()
        
        assert result is True
        assert "Agent Pool" in manager.activated
    
    def test_activate_agent_pool_scaling(self):
        """Test agent pool scaling when below minimum."""
        mock_deployment = self.create_mock_deployment()
        
        mock_pool = Mock()
        mock_pool.get_pool_status.return_value = {
            "total_agents": 2,
            "state_distribution": {"idle": 2}
        }
        mock_pool.min_agents = 5
        mock_pool.spawn_agent = Mock()
        mock_deployment.collective.agent_pool = mock_pool
        
        # Mock the second call after spawning
        mock_pool.get_pool_status.side_effect = [
            {"total_agents": 2, "state_distribution": {"idle": 2}},
            {"total_agents": 5, "state_distribution": {"idle": 5}}
        ]
        
        with patch('vulcan.orchestrator.agent_lifecycle.AgentCapability'):
            manager = SubsystemManager(mock_deployment)
            result = manager.activate_agent_pool()
        
        assert result is True
        assert mock_pool.spawn_agent.call_count == 3  # 5 - 2 = 3
    
    def test_activate_agent_pool_missing(self):
        """Test activating agent pool when not available."""
        mock_deployment = self.create_mock_deployment()
        # Don't add agent_pool
        
        manager = SubsystemManager(mock_deployment)
        result = manager.activate_agent_pool()
        
        assert result is False
        assert "Agent Pool" not in manager.activated
    
    def test_activate_world_model_success(self):
        """Test activating world model successfully."""
        mock_deployment = self.create_mock_deployment()
        mock_app_state = Mock()
        
        mock_world_model = Mock()
        mock_world_model.motivational_introspection = Mock()
        mock_world_model.self_improvement_drive = Mock()
        mock_deployment.collective.deps.world_model = mock_world_model
        
        manager = SubsystemManager(mock_deployment)
        result = manager.activate_world_model(mock_app_state)
        
        assert result is True
        assert "World Model" in manager.activated
    
    def test_activate_world_model_with_system_observer(self):
        """Test activating world model with SystemObserver."""
        mock_deployment = self.create_mock_deployment()
        mock_app_state = Mock()
        
        mock_world_model = Mock()
        mock_deployment.collective.deps.world_model = mock_world_model
        
        with patch('vulcan.world_model.system_observer.initialize_system_observer') as mock_init:
            mock_observer = Mock()
            mock_init.return_value = mock_observer
            
            manager = SubsystemManager(mock_deployment)
            result = manager.activate_world_model(mock_app_state)
        
        assert result is True
        assert mock_app_state.system_observer == mock_observer
    
    def test_activate_safety_subsystems_with_constraints(self):
        """Test activating safety subsystems with constraints."""
        mock_deployment = self.create_mock_deployment()
        
        mock_validator = Mock()
        mock_validator.activate_all_constraints = Mock()
        mock_deployment.collective.deps.safety_validator = mock_validator
        mock_deployment.collective.deps.governance = Mock()
        
        manager = SubsystemManager(mock_deployment)
        count = manager.activate_safety_subsystems()
        
        assert count >= 1
        mock_validator.activate_all_constraints.assert_called_once()
    
    def test_get_summary(self):
        """Test getting activation summary."""
        mock_deployment = self.create_mock_deployment()
        mock_deployment.collective.deps.get_status.return_value = {
            "available_count": 10,
            "total_dependencies": 15
        }
        
        manager = SubsystemManager(mock_deployment)
        manager.activated = ["System1", "System2", "System3"]
        manager.failed = [{"name": "System4", "error": "Failed"}]
        
        summary = manager.get_summary()
        
        assert summary["activated"] == 3
        assert summary["failed"] == 1
        assert summary["available"] == 10
        assert summary["total"] == 15
        assert len(summary["activated_list"]) == 3
        assert len(summary["failed_list"]) == 1
    
    def test_log_summary(self, caplog):
        """Test logging activation summary."""
        import logging
        caplog.set_level(logging.INFO)
        
        mock_deployment = self.create_mock_deployment()
        mock_deployment.collective.deps.get_status.return_value = {
            "available_count": 10,
            "total_dependencies": 12
        }
        
        manager = SubsystemManager(mock_deployment)
        manager.activated = ["System1", "System2"]
        manager.failed = []
        
        manager.log_summary()
        
        assert "10/12" in caplog.text
        assert "0 failed" in caplog.text
    
    def test_log_summary_with_failures(self, caplog):
        """Test logging summary with failures."""
        import logging
        caplog.set_level(logging.WARNING)
        
        mock_deployment = self.create_mock_deployment()
        mock_deployment.collective.deps.get_status.return_value = {
            "available_count": 8,
            "total_dependencies": 10
        }
        
        manager = SubsystemManager(mock_deployment)
        manager.activated = ["System1"]
        manager.failed = [
            {"name": "System2", "error": "Connection failed"}
        ]
        
        manager.log_summary()
        
        assert "8/10" in caplog.text
        assert "1 failed" in caplog.text
        assert "System2" in caplog.text


class TestSubsystemActivationGroups:
    """Test activating different groups of subsystems."""
    
    def create_full_mock_deployment(self):
        """Create a mock deployment with all subsystems."""
        mock_deployment = Mock()
        mock_deployment.collective = Mock()
        mock_deployment.collective.deps = Mock()
        
        # Add all subsystem types
        for attr in ['symbolic', 'probabilistic', 'causal', 'abstract', 'cross_modal',
                     'ltm', 'am', 'compressed_memory',
                     'multimodal',
                     'continual', 'meta_cognitive', 'compositional',
                     'goal_system', 'resource_compute',
                     'experiment_generator', 'problem_executor',
                     'self_improvement_drive', 'motivational_introspection',
                     'objective_hierarchy', 'objective_negotiator', 'goal_conflict_detector']:
            setattr(mock_deployment.collective.deps, attr, Mock())
        
        return mock_deployment
    
    def test_activate_all_subsystem_groups(self):
        """Test activating all subsystem groups."""
        mock_deployment = self.create_full_mock_deployment()
        mock_app_state = Mock()
        
        manager = SubsystemManager(mock_deployment)
        
        # Activate all groups
        reasoning_count = manager.activate_reasoning_subsystems()
        memory_count = manager.activate_memory_subsystems()
        processing_count = manager.activate_processing_subsystems()
        learning_count = manager.activate_learning_subsystems()
        planning_count = manager.activate_planning_subsystems()
        curiosity_count = manager.activate_curiosity_subsystems()
        meta_count = manager.activate_meta_reasoning_subsystems()
        
        # Should have activated subsystems in each category
        assert reasoning_count > 0
        assert memory_count > 0
        assert processing_count > 0
        assert learning_count > 0
        assert planning_count > 0
        assert curiosity_count > 0
        assert meta_count > 0
        
        # Total activated should be sum of all
        total = (reasoning_count + memory_count + processing_count + 
                learning_count + planning_count + curiosity_count + meta_count)
        assert len(manager.activated) >= total
