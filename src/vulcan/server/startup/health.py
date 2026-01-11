"""
Startup Health Validation

Validates critical system components after startup to ensure
the server is ready to handle requests.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """
    Health status for a single component.
    
    Attributes:
        name: Component name
        healthy: True if component is operational
        critical: True if component is critical for operation
        message: Status message or error description
        details: Additional diagnostic information
    """
    name: str
    healthy: bool
    critical: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthCheck:
    """
    Validates system health after startup.
    
    Performs checks on critical components and determines overall
    system health status based on individual component results.
    """
    
    def __init__(self, app_state: Any):
        """
        Initialize health checker.
        
        Args:
            app_state: FastAPI application state object
        """
        self.app_state = app_state
        self.component_results: List[ComponentHealth] = []
    
    def check_deployment(self) -> ComponentHealth:
        """
        Check deployment component health.
        
        Returns:
            ComponentHealth result for deployment
        """
        try:
            if not hasattr(self.app_state, "deployment"):
                return ComponentHealth(
                    name="deployment",
                    healthy=False,
                    critical=True,
                    message="Deployment not initialized"
                )
            
            deployment = self.app_state.deployment
            if deployment is None:
                return ComponentHealth(
                    name="deployment",
                    healthy=False,
                    critical=True,
                    message="Deployment is None"
                )
            
            # Check if deployment has required attributes
            if not hasattr(deployment, "collective"):
                return ComponentHealth(
                    name="deployment",
                    healthy=False,
                    critical=True,
                    message="Deployment missing collective"
                )
            
            return ComponentHealth(
                name="deployment",
                healthy=True,
                critical=True,
                message="Operational",
                details={"worker_id": getattr(self.app_state, "worker_id", "unknown")}
            )
            
        except Exception as e:
            logger.error(f"Deployment health check failed: {e}", exc_info=True)
            return ComponentHealth(
                name="deployment",
                healthy=False,
                critical=True,
                message=f"Health check error: {str(e)}"
            )
    
    def check_llm(self) -> ComponentHealth:
        """
        Check LLM component health.
        
        Returns:
            ComponentHealth result for LLM
        """
        try:
            if not hasattr(self.app_state, "llm"):
                return ComponentHealth(
                    name="llm",
                    healthy=False,
                    critical=True,
                    message="LLM not initialized"
                )
            
            llm = self.app_state.llm
            if llm is None:
                return ComponentHealth(
                    name="llm",
                    healthy=False,
                    critical=True,
                    message="LLM is None"
                )
            
            return ComponentHealth(
                name="llm",
                healthy=True,
                critical=True,
                message="Operational"
            )
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}", exc_info=True)
            return ComponentHealth(
                name="llm",
                healthy=False,
                critical=True,
                message=f"Health check error: {str(e)}"
            )
    
    def check_redis(self, redis_client: Any) -> ComponentHealth:
        """
        Check Redis connection health.
        
        Args:
            redis_client: Redis client instance (may be None)
            
        Returns:
            ComponentHealth result for Redis
        """
        try:
            if redis_client is None:
                return ComponentHealth(
                    name="redis",
                    healthy=False,
                    critical=False,  # Redis is optional
                    message="Redis not configured (running in standalone mode)"
                )
            
            # Redis is available and configured
            return ComponentHealth(
                name="redis",
                healthy=True,
                critical=False,
                message="Operational (multi-instance mode)"
            )
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return ComponentHealth(
                name="redis",
                healthy=False,
                critical=False,
                message=f"Health check error: {str(e)}"
            )
    
    def check_agent_pool(self) -> ComponentHealth:
        """
        Check agent pool health.
        
        P1 Fix: Issue #8 - Exception handling for status checks during shutdown.
        
        Returns:
            ComponentHealth result for agent pool
        """
        try:
            if not hasattr(self.app_state, "deployment"):
                return ComponentHealth(
                    name="agent_pool",
                    healthy=False,
                    critical=False,
                    message="Deployment not available"
                )
            
            deployment = self.app_state.deployment
            if not hasattr(deployment, "collective"):
                return ComponentHealth(
                    name="agent_pool",
                    healthy=False,
                    critical=False,
                    message="Collective not available"
                )
            
            if not hasattr(deployment.collective, "agent_pool"):
                return ComponentHealth(
                    name="agent_pool",
                    healthy=False,
                    critical=False,
                    message="Agent pool not initialized"
                )
            
            agent_pool = deployment.collective.agent_pool
            if agent_pool is None:
                return ComponentHealth(
                    name="agent_pool",
                    healthy=False,
                    critical=False,
                    message="Agent pool is None"
                )
            
            # Get pool status with exception handling (P1 Fix: Issue #8)
            try:
                pool_status = agent_pool.get_pool_status()
                total_agents = pool_status.get("total_agents", 0)
            except Exception as status_error:
                logger.warning(f"Could not get agent pool status: {status_error}")
                return ComponentHealth(
                    name="agent_pool",
                    healthy=False,
                    critical=False,
                    message=f"Agent pool status unavailable: {str(status_error)}"
                )
            
            return ComponentHealth(
                name="agent_pool",
                healthy=True,
                critical=False,
                message="Operational",
                details={"total_agents": total_agents}
            )
            
        except Exception as e:
            logger.warning(f"Agent pool health check failed: {e}")
            return ComponentHealth(
                name="agent_pool",
                healthy=False,
                critical=False,
                message=f"Health check error: {str(e)}"
            )
    
    def check_models(self) -> ComponentHealth:
        """
        Check model preloading status.
        
        Returns:
            ComponentHealth result for models
        """
        try:
            # Check if model registry has cached models
            models_cached = 0
            try:
                from vulcan.models import model_registry
                stats = model_registry.get_cache_stats()
                models_cached = stats.get("models_cached", 0)
            except (ImportError, Exception):
                pass
            
            if models_cached > 0:
                return ComponentHealth(
                    name="models",
                    healthy=True,
                    critical=False,
                    message="Models preloaded",
                    details={"models_cached": models_cached}
                )
            else:
                return ComponentHealth(
                    name="models",
                    healthy=True,  # Not critical, will load lazily
                    critical=False,
                    message="Models will load on demand"
                )
                
        except Exception as e:
            logger.debug(f"Model health check failed: {e}")
            return ComponentHealth(
                name="models",
                healthy=True,  # Non-critical
                critical=False,
                message="Models will load on demand"
            )
    
    def run_all_checks(self, redis_client: Any = None) -> Dict[str, Any]:
        """
        Run all health checks and determine overall status.
        
        Args:
            redis_client: Optional Redis client instance
            
        Returns:
            Dictionary with overall status and component results
        """
        self.component_results = []
        
        # Run all checks
        self.component_results.append(self.check_deployment())
        self.component_results.append(self.check_llm())
        self.component_results.append(self.check_redis(redis_client))
        self.component_results.append(self.check_agent_pool())
        self.component_results.append(self.check_models())
        
        # Determine overall status
        critical_failed = any(
            not result.healthy and result.critical
            for result in self.component_results
        )
        
        if critical_failed:
            overall_status = HealthStatus.UNHEALTHY
        elif any(not result.healthy for result in self.component_results):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Build summary
        healthy_count = sum(1 for r in self.component_results if r.healthy)
        total_count = len(self.component_results)
        
        return {
            "status": overall_status.value,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": [
                {
                    "name": r.name,
                    "healthy": r.healthy,
                    "critical": r.critical,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.component_results
            ]
        }
    
    def log_health_summary(self, health_result: Dict[str, Any]) -> None:
        """
        Log health check summary.
        
        Args:
            health_result: Result from run_all_checks()
        """
        status = health_result["status"]
        healthy = health_result["healthy_components"]
        total = health_result["total_components"]
        
        if status == HealthStatus.HEALTHY.value:
            logger.info(f"✅ Health Check: HEALTHY ({healthy}/{total} components operational)")
        elif status == HealthStatus.DEGRADED.value:
            logger.warning(f"⚠️  Health Check: DEGRADED ({healthy}/{total} components operational)")
        else:
            logger.error(f"❌ Health Check: UNHEALTHY ({healthy}/{total} components operational)")
        
        # Log any failed components
        for component in health_result["components"]:
            if not component["healthy"] and component["critical"]:
                logger.error(f"  ❌ {component['name']}: {component['message']}")
            elif not component["healthy"]:
                logger.warning(f"  ⚠️  {component['name']}: {component['message']}")
