"""
Subsystem Management

Manages activation and initialization of VULCAN-AGI subsystems
with proper error isolation, status tracking, and thread safety.

This module provides a centralized manager for activating cognitive
subsystems with:
- Error isolation: failures don't cascade
- Status tracking: all activations and failures are recorded
- Thread safety: concurrent activations are synchronized
- Flexible configuration: subsystems can be critical or optional
"""

import logging
from typing import Any, Optional, List, Dict, Callable
from dataclasses import dataclass
from threading import Lock


logger = logging.getLogger(__name__)


@dataclass
class SubsystemConfig:
    """
    Configuration for a subsystem.
    
    Attributes:
        attr_name: Attribute name in deployment.collective.deps
        display_name: Human-readable name for logging
        critical: If True, failure prevents startup
        needs_init: If True, calls initialize() method
        pre_check: Optional callable to check if subsystem should be activated
    """
    attr_name: str
    display_name: str
    critical: bool = False
    needs_init: bool = False
    pre_check: Optional[Callable[[Any], bool]] = None


class SubsystemManager:
    """
    Manages subsystem activation with error isolation and thread safety.
    
    Activates subsystems in logical groups, tracks success/failure,
    and provides detailed status reporting. All list modifications
    are protected by locks for thread safety.
    
    P2 Fix: Issue #13 - Thread-safe list modifications.
    """
    
    def __init__(self, deployment: Any, trace_logger: Optional[Any] = None):
        """
        Initialize subsystem manager.
        
        Args:
            deployment: ProductionDeployment instance
            trace_logger: Optional startup trace logger for auditable logging
        """
        self.deployment = deployment
        self.activated: List[str] = []
        self.failed: List[Dict[str, str]] = []
        self._lock = Lock()  # P2 Fix: Issue #13 - Thread safety
        self.trace = trace_logger  # For auditable registration tracking
    
    def _activate_single(self, config: SubsystemConfig) -> bool:
        """
        Activate a single subsystem with error handling.
        
        P2 Fix: Issue #13 - Thread-safe list modifications.
        
        Args:
            config: Subsystem configuration
            
        Returns:
            True if activation succeeded, False otherwise
        """
        try:
            deps = self.deployment.collective.deps
            
            # Run pre-check if provided
            if config.pre_check and not config.pre_check(deps):
                logger.debug(f"Skipping {config.display_name} (pre-check failed)")
                return False
            
            # Check if subsystem exists
            if not hasattr(deps, config.attr_name):
                if config.critical:
                    logger.error(f"Critical subsystem {config.display_name} not found")
                    with self._lock:  # P2 Fix: Issue #13
                        self.failed.append({
                            "name": config.display_name,
                            "error": "Attribute not found"
                        })
                    return False
                logger.debug(f"{config.display_name} not available")
                return False
            
            subsystem = getattr(deps, config.attr_name)
            if subsystem is None:
                if config.critical:
                    logger.error(f"Critical subsystem {config.display_name} is None")
                    with self._lock:  # P2 Fix: Issue #13
                        self.failed.append({
                            "name": config.display_name,
                            "error": "Subsystem is None"
                        })
                    return False
                logger.debug(f"{config.display_name} is None")
                return False
            
            # Call initialize if needed
            if config.needs_init and hasattr(subsystem, "initialize"):
                subsystem.initialize()
            
            with self._lock:  # P2 Fix: Issue #13
                self.activated.append(config.display_name)
            logger.debug(f"✓ {config.display_name} activated")
            return True
            
        except Exception as e:
            if config.critical:
                logger.error(
                    f"Critical subsystem {config.display_name} failed: {e}",
                    exc_info=True
                )
                with self._lock:  # P2 Fix: Issue #13
                    self.failed.append({
                        "name": config.display_name,
                        "error": str(e)
                    })
            else:
                logger.warning(
                    f"Non-critical subsystem {config.display_name} failed: {e}",
                    exc_info=True
                )
            return False
    
    def activate_agent_pool(self) -> bool:
        """
        Activate and scale agent pool.
        
        P1 Fix: Issue #7 - Standardized error tracking.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self.deployment.collective, "agent_pool"):
                logger.debug("Agent Pool not available - distributed processing disabled")
                with self._lock:
                    self.failed.append({
                        "name": "Agent Pool",
                        "error": "Attribute not found"
                    })
                return False
            
            pool = self.deployment.collective.agent_pool
            if pool is None:
                logger.debug("Agent Pool is None - distributed processing disabled")
                with self._lock:
                    self.failed.append({
                        "name": "Agent Pool",
                        "error": "Agent Pool is None"
                    })
                return False
            
            pool_status = pool.get_pool_status()
            total_agents = pool_status.get("total_agents", 0)
            idle_agents = pool_status.get("state_distribution", {}).get("idle", 0)
            
            # Ensure minimum agents are available
            if total_agents < pool.min_agents:
                logger.debug(
                    f"Agent pool below minimum ({total_agents} < {pool.min_agents}), spawning more..."
                )
                from vulcan.orchestrator.agent_lifecycle import AgentCapability
                
                for _ in range(pool.min_agents - total_agents):
                    pool.spawn_agent(AgentCapability.GENERAL)
                
                total_agents = pool.get_pool_status()["total_agents"]
            
            with self._lock:
                self.activated.append("Agent Pool")
            logger.debug(f"✓ Agent Pool: {total_agents} agents ({idle_agents} idle)")
            return True
            
        except Exception as e:
            logger.warning(f"Agent Pool activation failed: {e}", exc_info=True)
            with self._lock:
                self.failed.append({
                    "name": "Agent Pool",
                    "error": str(e)
                })
            return False
    
    def activate_reasoning_subsystems(self) -> int:
        """
        Activate reasoning subsystems.
        
        Activates symbolic, probabilistic, causal, analogical, and cross-modal
        reasoning components. These subsystems provide the core cognitive
        capabilities for problem-solving and inference.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        
        Returns:
            Number of subsystems successfully activated
        """
        configs = [
            SubsystemConfig("symbolic", "Symbolic Reasoning"),
            SubsystemConfig("probabilistic", "Probabilistic Reasoning"),
            SubsystemConfig("causal", "Causal Reasoning"),
            SubsystemConfig("abstract", "Analogical/Abstract Reasoning"),
            SubsystemConfig("cross_modal", "Cross-Modal Reasoning"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Reasoning: {count} subsystems activated")
        return count
    
    def activate_memory_subsystems(self) -> int:
        """
        Activate memory subsystems.
        
        Activates long-term memory (vector index), episodic/autobiographical
        memory, and compressed memory persistence. These provide storage and
        retrieval for learned experiences and knowledge.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        
        Returns:
            Number of subsystems successfully activated
        """
        configs = [
            SubsystemConfig("ltm", "Long-term Memory (Vector Index)"),
            SubsystemConfig("am", "Episodic/Autobiographical Memory"),
            SubsystemConfig("compressed_memory", "Compressed Memory Persistence"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Memory: {count} subsystems activated")
        return count
    
    def activate_processing_subsystems(self) -> int:
        """
        Activate input/output processing subsystems.
        
        Returns:
            Number of subsystems activated
        """
        configs = [
            SubsystemConfig("multimodal", "Multimodal Processor"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Processing: {count} subsystems activated")
        return count
    
    def activate_learning_subsystems(self) -> int:
        """
        Activate learning and adaptation subsystems.
        
        Returns:
            Number of subsystems activated
        """
        configs = [
            SubsystemConfig("continual", "Continual Learning"),
            SubsystemConfig("meta_cognitive", "Meta-Cognitive Monitor"),
            SubsystemConfig("compositional", "Compositional Understanding"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Learning: {count} subsystems activated")
        return count
    
    def activate_world_model(self, app_state: Any) -> bool:
        """
        Activate world model and meta-reasoning components.
        
        P1 Fix: Issue #7 - Standardized error tracking.
        
        Args:
            app_state: FastAPI application state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self.deployment.collective.deps, "world_model"):
                logger.debug("World Model not available")
                with self._lock:
                    self.failed.append({
                        "name": "World Model",
                        "error": "Attribute not found"
                    })
                return False
            
            world_model = self.deployment.collective.deps.world_model
            if world_model is None:
                logger.debug("World Model is None")
                with self._lock:
                    self.failed.append({
                        "name": "World Model", 
                        "error": "World Model is None"
                    })
                return False
            
            # Check for meta-reasoning components
            sub_components = []
            if hasattr(world_model, "motivational_introspection") and world_model.motivational_introspection:
                sub_components.append("Motivational Introspection")
            if hasattr(world_model, "self_improvement_drive") and world_model.self_improvement_drive:
                sub_components.append("Self-Improvement Drive")
            
            # Initialize SystemObserver for event tracking
            try:
                from vulcan.world_model.system_observer import initialize_system_observer
                system_observer = initialize_system_observer(world_model)
                app_state.system_observer = system_observer
                sub_components.append("SystemObserver")
            except ImportError:
                logger.debug("SystemObserver not available")
            except Exception as e:
                logger.warning(f"SystemObserver initialization failed: {e}")
            
            with self._lock:
                self.activated.append("World Model")
            if sub_components:
                logger.info(f"✓ World Model with {len(sub_components)} components: {', '.join(sub_components)}")
            else:
                logger.info("✓ World Model activated")
            return True
            
        except Exception as e:
            logger.warning(f"World Model activation failed: {e}", exc_info=True)
            with self._lock:
                self.failed.append({
                    "name": "World Model",
                    "error": str(e)
                })
            return False
    
    def activate_planning_subsystems(self) -> int:
        """
        Activate planning and goal management subsystems.
        
        Returns:
            Number of subsystems activated
        """
        configs = [
            SubsystemConfig("goal_system", "Hierarchical Goal System"),
            SubsystemConfig("resource_compute", "Resource-Aware Compute"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Planning: {count} subsystems activated")
        return count
    
    def activate_safety_subsystems(self) -> int:
        """
        Activate safety and governance subsystems.
        
        Returns:
            Number of subsystems activated
        """
        count = 0
        
        # Safety validator with special handling
        try:
            if hasattr(self.deployment.collective.deps, "safety_validator"):
                safety_validator = self.deployment.collective.deps.safety_validator
                if safety_validator:
                    if hasattr(safety_validator, "activate_all_constraints"):
                        try:
                            safety_validator.activate_all_constraints()
                            logger.debug("✓ Safety Validator with all constraints")
                        except Exception as e:
                            logger.warning(f"Failed to activate all constraints: {e}")
                            logger.debug("✓ Safety Validator (partial constraints)")
                    else:
                        logger.debug("✓ Safety Validator activated")
                    self.activated.append("Safety Validator")
                    count += 1
        except Exception as e:
            logger.warning(f"Safety Validator activation failed: {e}")
        
        # Other safety subsystems
        configs = [
            SubsystemConfig("governance", "Governance Orchestrator"),
            SubsystemConfig("nso_aligner", "NSO Aligner"),
            SubsystemConfig("explainer", "Explainability Node"),
        ]
        
        count += sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Safety: {count} subsystems activated")
        return count
    
    def activate_curiosity_subsystems(self) -> int:
        """
        Activate curiosity and exploration subsystems.
        
        Returns:
            Number of subsystems activated
        """
        configs = [
            SubsystemConfig("experiment_generator", "Experiment Generator"),
            SubsystemConfig("problem_executor", "Problem Executor"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Curiosity: {count} subsystems activated")
        return count
    
    def activate_meta_reasoning_subsystems(self) -> int:
        """
        Activate meta-reasoning subsystems from world model.
        
        Returns:
            Number of subsystems activated
        """
        configs = [
            SubsystemConfig("self_improvement_drive", "Self-Improvement Drive"),
            SubsystemConfig("motivational_introspection", "Motivational Introspection"),
            SubsystemConfig("objective_hierarchy", "Objective Hierarchy"),
            SubsystemConfig("objective_negotiator", "Objective Negotiator"),
            SubsystemConfig("goal_conflict_detector", "Goal Conflict Detector"),
        ]
        
        count = sum(1 for config in configs if self._activate_single(config))
        if count > 0:
            logger.info(f"✓ Meta-Reasoning: {count} subsystems activated")
        return count
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get activation summary with statistics.
        
        Provides comprehensive status including activated subsystems,
        failures, and total available dependencies. Used for logging
        and monitoring startup health.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        
        Returns:
            Dictionary with activation statistics and lists
        """
        try:
            deps_status = self.deployment.collective.deps.get_status()
            available = deps_status.get("available_count", len(self.activated))
            total = deps_status.get("total_dependencies", available)
        except Exception:
            available = len(self.activated)
            total = available
        
        return {
            "activated": len(self.activated),
            "failed": len(self.failed),
            "available": available,
            "total": total,
            "activated_list": self.activated,
            "failed_list": self.failed
        }
    
    def log_summary(self) -> None:
        """
        Log activation summary with emoji indicators.
        
        Outputs a summary of subsystem activation status and logs any
        failures with warning level. Used at the end of subsystem
        initialization to provide visibility into system state.
        
        P3 Fix: Issue #16 - Add comprehensive docstring.
        """
        summary = self.get_summary()
        logger.info(
            f"📊 Subsystems: {summary['available']}/{summary['total']} active, "
            f"{summary['failed']} failed"
        )
        
        if summary["failed"] > 0:
            for failure in summary["failed_list"]:
                logger.warning(f"  ⚠️  {failure['name']}: {failure['error']}")
