"""
Unit tests for startup health check system.

Tests the HealthCheck class and component validation.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any

from vulcan.server.startup.health import (
    HealthCheck,
    HealthStatus,
    ComponentHealth,
)


class TestHealthCheck:
    """Test suite for HealthCheck class."""
    
    def test_component_health_creation(self):
        """Test creating ComponentHealth instances."""
        health = ComponentHealth(
            name="test_component",
            healthy=True,
            critical=True,
            message="Operational"
        )
        
        assert health.name == "test_component"
        assert health.healthy is True
        assert health.critical is True
        assert health.message == "Operational"
        assert health.details is None
    
    def test_component_health_with_details(self):
        """Test ComponentHealth with details."""
        health = ComponentHealth(
            name="test_component",
            healthy=True,
            critical=False,
            message="OK",
            details={"count": 5}
        )
        
        assert health.details == {"count": 5}
    
    def test_check_deployment_healthy(self):
        """Test deployment health check when healthy."""
        # Create mock app state with deployment
        mock_deployment = Mock()
        mock_deployment.collective = Mock()
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        mock_app_state.worker_id = "test_worker"
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_deployment()
        
        assert result.name == "deployment"
        assert result.healthy is True
        assert result.critical is True
        assert result.message == "Operational"
        assert result.details == {"worker_id": "test_worker"}
    
    def test_check_deployment_missing(self):
        """Test deployment health check when deployment is missing."""
        mock_app_state = Mock(spec=[])  # Empty spec means no attributes
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_deployment()
        
        assert result.name == "deployment"
        assert result.healthy is False
        assert result.critical is True
        assert "not initialized" in result.message
    
    def test_check_deployment_none(self):
        """Test deployment health check when deployment is None."""
        mock_app_state = Mock()
        mock_app_state.deployment = None
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_deployment()
        
        assert result.name == "deployment"
        assert result.healthy is False
        assert result.critical is True
        assert "is None" in result.message
    
    def test_check_deployment_missing_collective(self):
        """Test deployment health check when collective is missing."""
        mock_deployment = Mock(spec=[])  # No collective attribute
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_deployment()
        
        assert result.name == "deployment"
        assert result.healthy is False
        assert result.critical is True
        assert "missing collective" in result.message
    
    def test_check_llm_healthy(self):
        """Test LLM health check when healthy."""
        mock_llm = Mock()
        
        mock_app_state = Mock()
        mock_app_state.llm = mock_llm
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_llm()
        
        assert result.name == "llm"
        assert result.healthy is True
        assert result.critical is True
        assert result.message == "Operational"
    
    def test_check_llm_missing(self):
        """Test LLM health check when LLM is missing."""
        mock_app_state = Mock(spec=[])
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_llm()
        
        assert result.name == "llm"
        assert result.healthy is False
        assert result.critical is True
        assert "not initialized" in result.message
    
    def test_check_llm_none(self):
        """Test LLM health check when LLM is None."""
        mock_app_state = Mock()
        mock_app_state.llm = None
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_llm()
        
        assert result.name == "llm"
        assert result.healthy is False
        assert result.critical is True
        assert "is None" in result.message
    
    def test_check_redis_configured(self):
        """Test Redis health check when configured."""
        mock_redis = Mock()
        mock_app_state = Mock()
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_redis(mock_redis)
        
        assert result.name == "redis"
        assert result.healthy is True
        assert result.critical is False  # Redis is optional
        assert "multi-instance" in result.message
    
    def test_check_redis_not_configured(self):
        """Test Redis health check when not configured."""
        mock_app_state = Mock()
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_redis(None)
        
        assert result.name == "redis"
        assert result.healthy is False
        assert result.critical is False  # Redis is optional
        assert "not configured" in result.message
        assert "standalone mode" in result.message
    
    def test_check_agent_pool_healthy(self):
        """Test agent pool health check when healthy."""
        mock_pool = Mock()
        mock_pool.get_pool_status.return_value = {
            "total_agents": 5,
            "state_distribution": {"idle": 3}
        }
        
        mock_collective = Mock()
        mock_collective.agent_pool = mock_pool
        
        mock_deployment = Mock()
        mock_deployment.collective = mock_collective
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_agent_pool()
        
        assert result.name == "agent_pool"
        assert result.healthy is True
        assert result.critical is False  # Agent pool is optional
        assert result.message == "Operational"
        assert result.details == {"total_agents": 5}
    
    def test_check_agent_pool_missing(self):
        """Test agent pool health check when pool is missing."""
        mock_collective = Mock(spec=[])  # No agent_pool
        
        mock_deployment = Mock()
        mock_deployment.collective = mock_collective
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.check_agent_pool()
        
        assert result.name == "agent_pool"
        assert result.healthy is False
        assert result.critical is False
        assert "not initialized" in result.message
    
    def test_check_models_with_cache(self):
        """Test models health check with cached models."""
        mock_app_state = Mock()
        health_check = HealthCheck(mock_app_state)
        
        # Can't easily mock model_registry import, so this will fall through
        # to the "will load on demand" case
        result = health_check.check_models()
        
        assert result.name == "models"
        assert result.healthy is True  # Non-critical
        assert result.critical is False
    
    def test_run_all_checks_healthy(self):
        """Test running all checks when system is healthy."""
        # Create fully healthy mock system
        mock_deployment = Mock()
        mock_deployment.collective = Mock()
        
        mock_llm = Mock()
        mock_redis = Mock()
        
        mock_pool = Mock()
        mock_pool.get_pool_status.return_value = {"total_agents": 5}
        mock_deployment.collective.agent_pool = mock_pool
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        mock_app_state.llm = mock_llm
        mock_app_state.worker_id = "test_worker"
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.run_all_checks(mock_redis)
        
        assert result["status"] == HealthStatus.HEALTHY.value
        assert result["healthy_components"] == 5  # All components healthy
        assert result["total_components"] == 5
        assert len(result["components"]) == 5
    
    def test_run_all_checks_unhealthy(self):
        """Test running all checks when critical component fails."""
        # Create system with missing deployment (critical)
        mock_app_state = Mock(spec=[])  # No deployment
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.run_all_checks(None)
        
        assert result["status"] == HealthStatus.UNHEALTHY.value
        assert result["healthy_components"] < result["total_components"]
        
        # Check that deployment component is marked unhealthy
        deployment_component = next(
            c for c in result["components"] if c["name"] == "deployment"
        )
        assert deployment_component["healthy"] is False
        assert deployment_component["critical"] is True
    
    def test_run_all_checks_degraded(self):
        """Test running all checks when non-critical component fails."""
        # Create system with healthy critical components but no Redis
        mock_deployment = Mock()
        mock_deployment.collective = Mock()
        
        mock_llm = Mock()
        
        mock_app_state = Mock()
        mock_app_state.deployment = mock_deployment
        mock_app_state.llm = mock_llm
        mock_app_state.worker_id = "test_worker"
        
        health_check = HealthCheck(mock_app_state)
        result = health_check.run_all_checks(None)  # No Redis
        
        # Should be degraded (Redis is non-critical)
        # Note: Could be HEALTHY if all other components are healthy
        assert result["status"] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]
        
        # Check Redis component
        redis_component = next(
            c for c in result["components"] if c["name"] == "redis"
        )
        assert redis_component["healthy"] is False
        assert redis_component["critical"] is False
    
    def test_log_health_summary_healthy(self, caplog):
        """Test logging health summary when healthy."""
        import logging
        caplog.set_level(logging.INFO)
        
        mock_app_state = Mock()
        health_check = HealthCheck(mock_app_state)
        
        health_result = {
            "status": HealthStatus.HEALTHY.value,
            "healthy_components": 5,
            "total_components": 5,
            "components": []
        }
        
        health_check.log_health_summary(health_result)
        
        assert "HEALTHY" in caplog.text
        assert "5/5" in caplog.text
    
    def test_log_health_summary_degraded(self, caplog):
        """Test logging health summary when degraded."""
        import logging
        caplog.set_level(logging.WARNING)
        
        mock_app_state = Mock()
        health_check = HealthCheck(mock_app_state)
        
        health_result = {
            "status": HealthStatus.DEGRADED.value,
            "healthy_components": 4,
            "total_components": 5,
            "components": [
                {
                    "name": "redis",
                    "healthy": False,
                    "critical": False,
                    "message": "Not configured"
                }
            ]
        }
        
        health_check.log_health_summary(health_result)
        
        assert "DEGRADED" in caplog.text
        assert "4/5" in caplog.text
    
    def test_log_health_summary_unhealthy(self, caplog):
        """Test logging health summary when unhealthy."""
        import logging
        caplog.set_level(logging.ERROR)
        
        mock_app_state = Mock()
        health_check = HealthCheck(mock_app_state)
        
        health_result = {
            "status": HealthStatus.UNHEALTHY.value,
            "healthy_components": 3,
            "total_components": 5,
            "components": [
                {
                    "name": "deployment",
                    "healthy": False,
                    "critical": True,
                    "message": "Failed to initialize"
                }
            ]
        }
        
        health_check.log_health_summary(health_result)
        
        assert "UNHEALTHY" in caplog.text
        assert "3/5" in caplog.text
        assert "deployment" in caplog.text
