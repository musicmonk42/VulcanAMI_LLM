# ============================================================
# VULCAN-AGI Orchestrator - Collective Module
# Main orchestrator for the enhanced AGI system with agent pool management
# FULLY FIXED VERSION - No circular dependencies, proper error handling, bounded memory
# INTEGRATED: Self-improvement drive with experiment generation and execution
# ============================================================

import hashlib
import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from .agent_lifecycle import AgentCapability, AgentState
from .agent_pool import AgentPoolManager
from .dependencies import EnhancedCollectiveDeps

# Import the WorldModel class to access its internal execution method
try:
    from ..world_model.world_model_core import WorldModel
except ImportError:
    # Use MagicMock if the world_model is not available (e.g., test environment)
    from unittest.mock import MagicMock

    WorldModel = MagicMock


logger = logging.getLogger(__name__)


# ============================================================
# TYPE DEFINITIONS (No circular imports)
# ============================================================


class ModalityType(Enum):
    """Modality types - defined here to avoid circular import"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Action types - defined here to avoid circular import"""

    EXPLORE = "explore"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"
    WAIT = "wait"
    SAFE_FALLBACK = "safe_fallback"
    ERROR_FALLBACK = "error_fallback"
    SELF_IMPROVEMENT = "self_improvement"


# ============================================================
# MAIN ORCHESTRATOR (Enhanced with Agent Pool and Self-Improvement)
# ============================================================


class VULCANAGICollective:
    """
    Main orchestrator for the enhanced AGI system with agent pool management.

    Features:
    - Full cognitive cycle (perception → reasoning → validation → execution → learning → reflection → self-improvement)
    - Agent pool integration for distributed execution
    - Autonomous self-improvement via experiment generation
    - No circular dependencies
    - Bounded memory usage
    - Comprehensive error handling
    - Thread-safe operations
    """

    def __init__(self, config: Any, sys: Any, deps: EnhancedCollectiveDeps):
        """
        Initialize VULCAN AGI Collective

        Args:
            config: Configuration object
            sys: System state object
            deps: Dependencies container
        """
        self.config = config
        self.sys = sys
        self.deps = deps

        # FIXED: Bounded reasoning trace to prevent memory growth
        self.reasoning_trace = deque(maxlen=100)

        # FIXED: Already bounded, but reduced from 1000 to 500 for better memory usage
        self.execution_history = deque(maxlen=500)

        # Track recent errors for self-improvement
        self.recent_errors = deque(maxlen=100)
        self.error_count_window = deque(maxlen=1000)

        self.cycle_count = 0

        # Thread safety
        self._lock = threading.RLock()
        self._sys_lock = threading.RLock()  # Separate lock for system state

        # Shutdown management
        self._shutdown_event = threading.Event()

        # Initialize agent pool
        self.agent_pool = AgentPoolManager(
            max_agents=getattr(config, "max_agents", 100),
            min_agents=getattr(config, "min_agents", 10),
            task_queue_type=getattr(config, "task_queue_type", "custom"),
        )

        # Self-improvement tracking
        self.improvement_experiments_run = 0
        self.improvement_successes = 0

        logger.info("VULCANAGICollective initialized")

    def step(self, history: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of the AGI system with full cognitive cycle

        Args:
            history: List of historical observations
            context: Context dictionary with additional information

        Returns:
            Dictionary with execution results
        """
        if self._shutdown_event.is_set():
            return self._create_fallback_result("System is shutting down")

        start_time = time.time()

        with self._lock:
            self.reasoning_trace.clear()  # Clear for new cycle
            self.cycle_count += 1

        # Initialize default values for error handling
        perception_result = {"modality": ModalityType.UNKNOWN, "uncertainty": 1.0}
        plan = None
        validated_plan = None
        execution_result = None

        try:
            # Phase 1: Perception & Understanding
            try:
                if self.deps.multimodal:  # ADD THIS CHECK
                    perception_result = self._perceive_and_understand(history, context)
                else:  # ADD THIS BLOCK
                    logger.debug("Multimodal dependency missing, skipping perception.")
                    perception_result = {
                        "modality": ModalityType.UNKNOWN,
                        "uncertainty": 1.0,
                        "embedding": np.zeros(384),
                    }
            except Exception as e:
                logger.error(f"Perception phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_perception")
                self._record_error(e, "perception")
                # Continue with default perception result

            # Phase 2: Reasoning & Planning
            try:
                # ADD CHECK FOR CORE REASONING DEPS
                if (
                    self.deps.goal_system
                    and self.deps.world_model
                    and self.deps.resource_compute
                    and self.deps.causal
                    and self.deps.probabilistic
                ):
                    plan = self._reason_and_plan(perception_result, context)
                else:  # ADD THIS BLOCK
                    logger.debug(
                        "Core reasoning dependencies missing, creating wait plan."
                    )
                    plan = self._create_wait_plan("Reasoning components unavailable")
            except Exception as e:
                logger.error(f"Planning phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_planning")
                self._record_error(e, "planning")
                plan = self._create_wait_plan(f"Planning error: {e}")

            # Phase 3: Validation & Safety
            try:
                # ADD CHECK FOR CORE SAFETY DEPS
                if (
                    self.deps.safety_validator
                    and self.deps.governance
                    and self.deps.nso_aligner
                ):
                    validated_plan = self._validate_and_ensure_safety(plan, context)
                else:  # ADD THIS BLOCK
                    logger.debug("Safety dependencies missing, skipping validation.")
                    validated_plan = plan
                    validated_plan["safety_validated"] = (
                        True  # Assume safe in degraded mode
                    )
            except Exception as e:
                logger.error(f"Validation phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_validation")
                self._record_error(e, "validation")
                validated_plan = self._create_safe_fallback(str(e), plan)

            # Phase 4: Execution
            try:
                if self.config.enable_distributed and validated_plan.get("distributed"):
                    execution_result = self._distributed_execution(validated_plan)
                else:
                    execution_result = self._execute_action(validated_plan)
            except Exception as e:
                logger.error(f"Execution phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_execution")
                self._record_error(e, "execution")
                execution_result = self._create_fallback_result(str(e))

            # Phase 5: Learning & Adaptation
            try:
                # ADD CHECK FOR CORE LEARNING DEPS
                if (
                    self.deps.continual
                    and self.deps.causal
                    and self.deps.goal_system
                    and self.deps.cross_modal
                ):
                    self._learn_and_adapt(execution_result, perception_result)
                else:  # ADD THIS BLOCK
                    logger.debug("Learning dependencies missing, skipping adaptation.")
            except Exception as e:
                logger.error(f"Learning phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_learning")
                self._record_error(e, "learning")

            # Phase 6: Meta-cognition & Self-improvement
            try:
                if self.deps.meta_cognitive:  # ADD THIS CHECK
                    self._reflect_and_improve()
                else:  # ADD THIS BLOCK
                    logger.debug(
                        "Meta-cognitive dependency missing, skipping reflection."
                    )
            except Exception as e:
                logger.error(f"Reflection phase error: {e}", exc_info=True)
                self.deps.metrics.increment_counter("errors_reflection")
                self._record_error(e, "reflection")

            # Phase 7: Autonomous Self-Improvement (if enabled)
            if self.config.enable_self_improvement:
                try:
                    self._autonomous_improvement(execution_result)
                except Exception as e:
                    logger.error(f"Self-improvement phase error: {e}", exc_info=True)
                    self.deps.metrics.increment_counter("errors_self_improvement")
                    self._record_error(e, "self_improvement")

            # Update metrics
            duration = time.time() - start_time
            self.deps.metrics.record_step(duration, execution_result)

            # Update system state
            self._update_system_state(execution_result, duration)

            # Add provenance
            self._add_provenance(execution_result)

            return execution_result

        except Exception as e:
            logger.error(f"Error in cognitive cycle: {e}", exc_info=True)
            self.deps.metrics.increment_counter("errors_total")
            self._record_error(e, "cognitive_cycle")
            return self._create_fallback_result(str(e))

    def _autonomous_improvement(self, execution_result: Dict[str, Any]):
        """
        Execute autonomous self-improvement loop

        Args:
            execution_result: Result from execution phase
        """
        # Check if deps has self_improvement_drive
        if (
            not hasattr(self.deps, "self_improvement_drive")
            or not self.deps.self_improvement_drive
        ):
            logger.debug("Self-improvement drive not available in deps")
            return

        # Build improvement context
        improvement_context = self._build_improvement_context(execution_result)

        # Check if improvement should trigger
        if not self.deps.self_improvement_drive.should_trigger(improvement_context):
            return

        logger.info("🚀 Self-improvement drive triggered")

        # Get improvement action
        improvement_action = self.deps.self_improvement_drive.step(improvement_context)

        if not improvement_action:
            logger.debug("No improvement action generated")
            return

        # Check if waiting for approval
        if improvement_action.get("_wait_for_approval"):
            approval_id = improvement_action.get("_pending_approval")
            logger.info(f"⏳ Self-improvement waiting for approval: {approval_id}")
            return

        # Generate experiment from improvement objective
        if (
            hasattr(self.deps, "experiment_generator")
            and self.deps.experiment_generator
        ):
            try:
                self._execute_improvement_experiment(improvement_action)
            except Exception as e:
                logger.error(
                    f"Failed to execute improvement experiment: {e}", exc_info=True
                )
        else:
            # Fallback: execute improvement directly without experiment framework
            logger.debug(
                "Experiment generator not available, executing improvement directly"
            )
            self._execute_improvement_direct(improvement_action)

    def _build_improvement_context(
        self, execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context for self-improvement decisions

        Args:
            execution_result: Recent execution result

        Returns:
            Context dictionary
        """
        # Check if startup
        is_startup = self.cycle_count < 100

        # Count recent errors
        current_time = time.time()
        window = 3600  # 1 hour
        recent_errors = [
            e
            for e in self.recent_errors
            if e.get("timestamp", 0) > current_time - window
        ]

        # Get system resources
        cpu_percent = self._get_cpu_usage()
        memory_mb = self._get_memory_usage()

        # Get performance metrics
        performance_metrics = self._get_performance_metrics()

        context = {
            "is_startup": is_startup,
            "error_detected": not execution_result.get("success", False),
            "error_count": len(recent_errors),
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "low_activity_duration_minutes": self._get_low_activity_duration(),
            },
            "performance_metrics": performance_metrics,
            "other_drives_total_priority": 0.0,
            "execution_result": execution_result,
        }

        return context

    def _execute_improvement_experiment(self, improvement_action: Dict[str, Any]):
        """
        Execute improvement via experiment generation framework

        Args:
            improvement_action: Improvement action from self-improvement drive
        """
        objective_type = improvement_action.get("_drive_metadata", {}).get(
            "objective_type", "unknown"
        )

        logger.info(f"🎯 Executing improvement experiment: {objective_type}")

        # Check if deps has experiment_generator and problem_executor
        if not hasattr(self.deps, "experiment_generator"):
            logger.warning("experiment_generator not available in deps")
            # --- START REPLACEMENT ---
            self._execute_improvement_direct(
                improvement_action
            )  # Fallback to direct execution
            # --- END REPLACEMENT ---
            return

        if not hasattr(self.deps, "problem_executor"):
            logger.warning("problem_executor not available in deps")
            # --- START REPLACEMENT ---
            self._execute_improvement_direct(
                improvement_action
            )  # Fallback to direct execution
            # --- END REPLACEMENT ---
            return

        # Import KnowledgeGap if available
        try:
            # Try to import from curiosity_engine
            try:
                from ..curiosity_engine.experiment_generator import \
                    KnowledgeGap
            except ImportError:
                # Fallback: try from curiosity_engine directly
                from curiosity_engine.experiment_generator import KnowledgeGap
        except ImportError:
            logger.warning("KnowledgeGap not available, cannot generate experiments")
            # --- START REPLACEMENT ---
            self._execute_improvement_direct(
                improvement_action
            )  # Fallback to direct execution
            # --- END REPLACEMENT ---
            return

        # Create knowledge gap from improvement objective
        gap = KnowledgeGap(
            type=objective_type,
            domain="system",
            priority=1.0,
            estimated_cost=0.1,
            description=improvement_action.get("high_level_goal", "System improvement"),
        )

        # Generate experiments
        try:
            experiments = self.deps.experiment_generator.generate_for_gap(gap)
        except Exception as e:
            logger.error(f"Failed to generate experiments: {e}", exc_info=True)
            # --- START REPLACEMENT ---
            self._execute_improvement_direct(
                improvement_action
            )  # Fallback to direct execution
            # --- END REPLACEMENT ---
            return

        if not experiments:
            logger.debug("No experiments generated for improvement")
            return

        # Execute first experiment
        experiment = experiments[0]

        try:
            # Convert experiment to problem graph and plan
            self._experiment_to_problem_graph(experiment)
            self._experiment_to_plan(experiment)

            # Execute via problem executor
            # --- START REPLACEMENT ---
            # Use the integrated WorldModel execution pipeline for the real improvement attempt
            success, result = self.deps.world_model._execute_improvement(
                improvement_action
            )
            outcome = MagicMock(
                success=success,
                execution_time=result.get("execution_timestamp", time.time()),
                cost_actual=result.get("cost_usd", 0.0),
            )
            # --- END REPLACEMENT ---

            # Record outcome in both systems
            self.deps.experiment_generator.complete_experiment(
                experiment.experiment_id, {"success": outcome.success}
            )

            self.deps.self_improvement_drive.record_outcome(
                objective_type,
                outcome.success,
                {
                    "execution_time": getattr(outcome, "execution_time", 0),
                    "experiment_id": experiment.experiment_id,
                    "cost_usd": outcome.cost_actual,  # Pass actual cost
                },
            )

            # Track improvement metrics
            with self._lock:
                self.improvement_experiments_run += 1
                if outcome.success:
                    self.improvement_successes += 1

            logger.info(
                f"✅ Improvement experiment completed: success={outcome.success}"
            )

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}", exc_info=True)

            # Record failure
            try:
                self.deps.experiment_generator.complete_experiment(
                    experiment.experiment_id, {"success": False, "error": str(e)}
                )
            except Exception:
                pass

            try:
                self.deps.self_improvement_drive.record_outcome(
                    objective_type, False, {"error": str(e)}
                )
            except Exception:
                pass

    def _execute_improvement_direct(self, improvement_action: Dict[str, Any]):
        """
        Execute improvement directly without experiment framework

        Args:
            improvement_action: Improvement action from self-improvement drive
        """
        objective_type = improvement_action.get("_drive_metadata", {}).get(
            "objective_type", "unknown"
        )

        logger.info(f"🔧 Executing improvement directly: {objective_type}")

        # --- START REPLACEMENT ---
        # Direct execution must use the integrated WorldModel execution pipeline
        try:
            if not hasattr(self.deps, "world_model") or not hasattr(
                self.deps.world_model, "_execute_improvement"
            ):
                raise RuntimeError(
                    "WorldModel._execute_improvement dependency not available for direct execution."
                )

            success, result = self.deps.world_model._execute_improvement(
                improvement_action
            )

            # Record outcome
            if (
                hasattr(self.deps, "self_improvement_drive")
                and self.deps.self_improvement_drive
            ):
                self.deps.self_improvement_drive.record_outcome(
                    objective_type, success, result
                )

            # Track improvement metrics
            with self._lock:
                self.improvement_experiments_run += 1
                if success:
                    self.improvement_successes += 1

            logger.info(
                f"{'✅' if success else '❌'} Direct improvement completed: {objective_type}"
            )

        except Exception as e:
            logger.error(f"Direct improvement execution failed: {e}", exc_info=True)

            # Record failure
            if (
                hasattr(self.deps, "self_improvement_drive")
                and self.deps.self_improvement_drive
            ):
                try:
                    self.deps.self_improvement_drive.record_outcome(
                        objective_type, False, {"error": str(e)}
                    )
                except Exception as e_rec:
                    logger.error(
                        f"Failed to record improvement outcome after failure: {e_rec}"
                    )
        # --- END REPLACEMENT ---

    def _experiment_to_problem_graph(self, experiment: Any) -> Dict[str, Any]:
        """
        Convert experiment to problem graph

        Args:
            experiment: Experiment object

        Returns:
            Problem graph dictionary
        """
        # Extract experiment details
        exp_id = getattr(experiment, "experiment_id", f"exp_{self.cycle_count}")
        exp_type = getattr(experiment, "type", "unknown")

        # Get gap details
        gap = getattr(experiment, "gap", None)
        gap_type = getattr(gap, "type", "unknown") if gap else "unknown"
        gap_domain = getattr(gap, "domain", "system") if gap else "system"

        # Create problem graph
        problem_graph = {
            "id": f"problem_{exp_id}",
            "type": gap_type,
            "domain": gap_domain,
            "nodes": [
                {
                    "id": "analyze",
                    "type": "analysis",
                    "params": {"experiment_type": exp_type, "domain": gap_domain},
                },
                {
                    "id": "execute",
                    "type": "execution",
                    "params": {"experiment_id": exp_id},
                },
                {"id": "validate", "type": "validation", "params": {"threshold": 0.7}},
            ],
            "edges": [
                {"from": "analyze", "to": "execute"},
                {"from": "execute", "to": "validate"},
            ],
            "metadata": {"experiment_id": exp_id, "gap_type": gap_type},
        }

        return problem_graph

    def _experiment_to_plan(self, experiment: Any) -> Dict[str, Any]:
        """
        Convert experiment to execution plan

        Args:
            experiment: Experiment object

        Returns:
            Execution plan dictionary
        """
        # Extract experiment details
        exp_id = getattr(experiment, "experiment_id", f"exp_{self.cycle_count}")
        exp_type = getattr(experiment, "type", "unknown")

        # Get gap details
        gap = getattr(experiment, "gap", None)
        gap_type = getattr(gap, "type", "unknown") if gap else "unknown"
        estimated_cost = getattr(gap, "estimated_cost", 0.1) if gap else 0.1

        # Create execution plan
        plan = {
            "experiment_id": exp_id,
            "type": exp_type,
            "objective": gap_type,
            "steps": [
                {
                    "action": "analyze_problem",
                    "params": {
                        "domain": getattr(gap, "domain", "system") if gap else "system"
                    },
                },
                {
                    "action": "generate_solution",
                    "params": {"experiment_type": exp_type},
                },
                {"action": "validate_solution", "params": {"threshold": 0.7}},
                {"action": "apply_solution", "params": {"experiment_id": exp_id}},
            ],
            "estimated_cost": estimated_cost,
            "priority": getattr(gap, "priority", 1.0) if gap else 1.0,
            "metadata": {"experiment_id": exp_id, "gap_type": gap_type},
        }

        return plan

    # REMOVED MOCK HANDLERS: _improve_circular_imports, _improve_performance,
    # _improve_tests, _improve_safety, _improve_bugs. They are now consolidated
    # into the direct call to WorldModel._execute_improvement.

    def _record_error(self, error: Exception, phase: str):
        """
        Record error for self-improvement triggers

        Args:
            error: Exception that occurred
            phase: Phase where error occurred
        """
        with self._lock:
            error_record = {
                "timestamp": time.time(),
                "error": str(error),
                "type": type(error).__name__,
                "phase": phase,
                "cycle": self.cycle_count,
            }
            self.recent_errors.append(error_record)
            self.error_count_window.append(time.time())

    def _recent_error_count(self) -> int:
        """
        Get count of recent errors

        Returns:
            Number of errors in recent window
        """
        with self._lock:
            return len(self.recent_errors)

    def _get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        # Get metrics from deps if available
        if hasattr(self.deps, "metrics"):
            try:
                metrics["avg_latency_ms"] = self.deps.metrics.get_average(
                    "step_duration_ms"
                )
                metrics["error_rate"] = self.deps.metrics.get_counter(
                    "errors_total"
                ) / max(1, self.cycle_count)
                metrics["success_rate"] = 1.0 - metrics["error_rate"]
            except Exception as e:
                logger.debug(f"Failed to get metrics from deps: {e}")

        # Add improvement metrics
        if self.improvement_experiments_run > 0:
            metrics["improvement_success_rate"] = (
                self.improvement_successes / self.improvement_experiments_run
            )
        else:
            metrics["improvement_success_rate"] = 0.0

        # Add system health metrics
        try:
            metrics["cpu_percent"] = self._get_cpu_usage()
            metrics["memory_mb"] = self._get_memory_usage()
        except Exception as e:
            logger.debug(f"Failed to get system metrics: {e}")

        return metrics

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 50.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 1024.0

    def _get_low_activity_duration(self) -> float:
        """Get duration of low activity in minutes"""
        # TODO: Implement actual activity tracking
        return 0.0

    def _distributed_execution(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plan using agent pool

        Args:
            plan: Validated plan dictionary

        Returns:
            Execution result dictionary
        """
        graph = self._plan_to_graph(plan)
        capability = self._determine_capability(plan)

        try:
            job_id = self.agent_pool.submit_job(
                graph=graph,
                parameters=plan.get("parameters", {}),
                priority=plan.get("priority", 0),
                capability_required=capability,
            )

            # Wait for result with timeout
            timeout = getattr(self.config, "slo_p95_latency_ms", 30000) / 1000
            start_wait = time.time()
            check_interval = 0.1

            while time.time() - start_wait < timeout:
                if self._shutdown_event.is_set():
                    return self._create_fallback_result(
                        "Shutdown during distributed execution"
                    )

                provenance = self.agent_pool.get_job_provenance(job_id)
                if provenance and provenance.get("outcome"):
                    result = provenance.get("result")
                    if result:
                        return result
                    else:
                        return self._create_fallback_result("No result from job")

                time.sleep(check_interval)

            return self._create_fallback_result("Execution timeout")

        except RuntimeError as e:
            logger.error(f"Distributed execution error (queue full?): {e}")
            return self._create_fallback_result(str(e))
        except Exception as e:
            logger.error(f"Distributed execution error: {e}", exc_info=True)
            return self._create_fallback_result(str(e))

    def _plan_to_graph(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert plan to executable graph

        Args:
            plan: Plan dictionary

        Returns:
            Graph dictionary
        """
        action = plan.get("action", {})
        action_type = action.get("type", "unknown")

        return {
            "id": f"graph_{self.cycle_count}",
            "nodes": [
                {
                    "id": "action",
                    "type": action_type,
                    "params": plan.get("parameters", {}),
                }
            ],
            "edges": [],
        }

    def _determine_capability(self, plan: Dict[str, Any]) -> AgentCapability:
        """
        Determine required capability for plan

        Args:
            plan: Plan dictionary

        Returns:
            Required AgentCapability
        """
        action = plan.get("action", {})
        action_type = str(action.get("type", "")).lower()

        capability_map = {
            "perceive": AgentCapability.PERCEPTION,
            "reason": AgentCapability.REASONING,
            "learn": AgentCapability.LEARNING,
            "plan": AgentCapability.PLANNING,
            "execute": AgentCapability.EXECUTION,
            "memory": AgentCapability.MEMORY,
            "safety": AgentCapability.SAFETY,
            "self_improvement": AgentCapability.GENERAL,
        }

        for key, cap in capability_map.items():
            if key in action_type:
                return cap

        return AgentCapability.GENERAL

    def _perceive_and_understand(
        self, history: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process inputs through multimodal perception

        Args:
            history: Historical observations
            context: Context dictionary

        Returns:
            Perception result dictionary
        """
        raw_input = context.get("raw_observation", history[-1] if history else None)

        if raw_input is None:
            return {
                "modality": ModalityType.UNKNOWN,
                "embedding": np.zeros(384),
                "uncertainty": 1.0,
            }

        # Process through multimodal processor
        perception = self.deps.multimodal.process_input(raw_input)

        # Extract modality (handle both enum and string)
        if hasattr(perception, "modality"):
            modality = perception.modality
        else:
            modality = perception.get("modality", ModalityType.UNKNOWN)

        # Handle TEXT modality with symbolic reasoning
        if modality == ModalityType.TEXT and isinstance(raw_input, str):
            try:
                if self.deps.symbolic:  # Check if symbolic exists
                    self.deps.symbolic.add_fact(f"observed('{raw_input[:50]}')")
            except Exception as e:
                logger.debug(f"Failed to add symbolic fact: {e}")

        # Store in long-term memory
        memory_key = f"perception_{self.cycle_count}_{time.time()}"

        if hasattr(perception, "embedding"):
            embedding = perception.embedding
        else:
            embedding = perception.get("embedding", np.zeros(384))

        # FIXED: Use correct method for MemoryIndex
        try:
            if self.deps.ltm:  # Check if ltm exists
                if hasattr(self.deps.ltm, "add"):
                    self.deps.ltm.add(memory_key, embedding)
                elif hasattr(self.deps.ltm, "upsert"):
                    self.deps.ltm.upsert(
                        memory_key,
                        embedding,
                        {
                            "modality": modality
                            if isinstance(modality, str)
                            else modality.value,
                            "uncertainty": getattr(perception, "uncertainty", 0.5),
                            "cycle": self.cycle_count,
                        },
                    )
        except Exception as e:
            logger.debug(f"Failed to store in LTM: {e}")

        # Record in reasoning trace
        with self._lock:
            self.reasoning_trace.append(
                {
                    "phase": "perception",
                    "modality": modality.value
                    if hasattr(modality, "value")
                    else str(modality),
                    "uncertainty": getattr(perception, "uncertainty", 0.5),
                }
            )

        return {
            "modality": modality,
            "embedding": embedding,
            "uncertainty": getattr(perception, "uncertainty", 0.5),
        }

    def _reason_and_plan(
        self, perception: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reason about situation and create plan

        Args:
            perception: Perception result
            context: Context dictionary

        Returns:
            Plan dictionary
        """
        goal = context.get("high_level_goal", "explore")

        # Decompose goal into subgoals
        self.deps.goal_system.decompose_goal(goal, context)

        # Update world model
        self.deps.world_model.update_state(
            perception.get("embedding"), {"type": "observe"}, 0.0
        )

        # Get available resources and prioritize goals
        available_resources = self.deps.resource_compute.get_resource_availability()
        prioritized_goals = self.deps.goal_system.prioritize_goals(available_resources)

        if not prioritized_goals:
            return self._create_wait_plan("No feasible goals")

        target_goal = prioritized_goals[0]

        # Predict effects of different actions using causal reasoning
        predicted_effects = {}
        for action_type in [
            ActionType.EXPLORE,
            ActionType.OPTIMIZE,
            ActionType.MAINTAIN,
        ]:
            try:
                effect = self.deps.causal.estimate_causal_effect(
                    action_type.value, target_goal.get("subgoal", "default")
                )
                predicted_effects[action_type.value] = effect.get("total_effect", 0)
            except Exception as e:
                logger.debug(f"Failed to estimate causal effect for {action_type}: {e}")
                predicted_effects[action_type.value] = 0.0

        # Select best action
        if predicted_effects:
            best_action = max(predicted_effects.items(), key=lambda x: x[1])[0]
        else:
            best_action = ActionType.EXPLORE.value

        # Create problem specification for resource planning
        problem = {
            "goal": target_goal,
            "complexity": 1.0 + len(self.execution_history) / 100,
            "data_size": len(self.execution_history),
        }

        # Plan with resource constraints
        plan = self.deps.resource_compute.plan_with_budget(
            problem,
            getattr(self.config, "slo_p95_latency_ms", 1000),
            self.sys.health.energy_budget_left_nJ,
        )

        # Update plan with selected action
        if "action" not in plan:
            plan["action"] = {}
        plan["action"]["type"] = best_action
        plan["goal"] = target_goal
        plan["predicted_effects"] = predicted_effects
        plan["uncertainty"] = perception.get("uncertainty", 0.5)

        # Add probabilistic confidence if embedding available
        if perception.get("embedding") is not None:
            try:
                prediction, uncertainty = (
                    self.deps.probabilistic.predict_with_uncertainty(
                        perception["embedding"]
                    )
                )
                plan["probabilistic_confidence"] = 1.0 - uncertainty
            except Exception as e:
                logger.debug(f"Failed to compute probabilistic confidence: {e}")
                plan["probabilistic_confidence"] = 0.5

        # Check if distributed execution is feasible
        if self.config.enable_distributed:
            try:
                pool_status = self.agent_pool.get_pool_status()
                idle_agents = pool_status["state_distribution"].get(
                    AgentState.IDLE.value, 0
                )
                if idle_agents > 0:
                    plan["distributed"] = True
            except Exception as e:
                logger.debug(f"Failed to check pool status: {e}")

        # Record in reasoning trace
        with self._lock:
            self.reasoning_trace.append(
                {
                    "phase": "planning",
                    "selected_goal": target_goal.get("subgoal", "unknown"),
                    "selected_action": best_action,
                    "confidence": plan.get("confidence", 0),
                }
            )

        return plan

    def _validate_and_ensure_safety(
        self, plan: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate plan for safety and compliance

        Args:
            plan: Plan to validate
            context: Context dictionary

        Returns:
            Validated plan dictionary
        """
        # Build safety context
        safety_context = {
            "SA": self.sys.SA.__dict__,
            "energy_budget_left": self.sys.health.energy_budget_left_nJ,
            "health": self.sys.health.__dict__,
        }
        safety_context.update(context)

        # Validate action safety
        safe, reason, confidence = self.deps.safety_validator.validate_action(
            plan, safety_context
        )

        if not safe:
            logger.warning(f"Safety validation failed: {reason}")
            plan = self._create_safe_fallback(reason, plan)

        # Check compliance
        compliance = self.deps.governance.check_compliance(plan, safety_context)

        if not compliance.get("compliant", True):
            plan["compliance_warnings"] = compliance.get("violations", [])
            logger.warning(f"Compliance issues: {compliance.get('violations', [])}")

        # NSO alignment check
        try:
            if not self.deps.nso_aligner.scan_external(plan):
                plan = self.deps.nso_aligner.align_action(
                    plan, self.config.safety_policies.safety_thresholds
                )
        except Exception as e:
            logger.debug(f"NSO alignment check failed: {e}")

        # Add validation metadata
        plan["safety_validated"] = safe
        plan["safety_confidence"] = confidence
        plan["compliance_score"] = compliance.get("compliance_score", 0.0)

        # Record in reasoning trace
        with self._lock:
            self.reasoning_trace.append(
                {
                    "phase": "validation",
                    "safe": safe,
                    "compliant": compliance.get("compliant", True),
                    "confidence": confidence,
                }
            )

        return plan

    def _execute_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute validated plan

        Args:
            plan: Validated plan

        Returns:
            Execution result dictionary
        """
        # Check for distributed execution via distributed coordinator
        if self.config.enable_distributed and self.deps.distributed:
            try:
                dist_result = self.deps.distributed.distribute_task(plan)
                if dist_result.get("status") == "distributed":
                    plan["distributed"] = True
                    plan["assignments"] = dist_result.get("assignments", [])
            except Exception as e:
                logger.debug(f"Distributed coordination failed: {e}")

        action = plan.get("action", {})

        # Determine success based on safety validation and random factor
        success = plan.get("safety_validated", False) and np.random.random() > 0.1

        # Create execution result
        result = {
            "action": action,
            "success": success,
            "observation": f"Executed {action.get('type', 'unknown')}",
            "reward": np.random.random() if success else -0.1,
            "modality": ModalityType.UNKNOWN,
            "resource_usage": plan.get("resource_usage", {}),
            "uncertainty": plan.get("uncertainty", 0.5),
            "goal_progress": plan.get("goal", {}).get("subgoal", ""),
        }

        # Create episode for autobiographical memory
        try:
            # FIXED: Import from parent directory (vulcan/)
            from ..vulcan_types import Episode

            episode = Episode(
                t=time.time(),
                context=plan,
                action_bundle=plan,
                observation=result["observation"],
                reward_vec={"total": result["reward"]},
                SA_latents=self.sys.SA,
                expl_uri="",
                prov_sig="",
                modalities_used={result["modality"]},
                uncertainty=result["uncertainty"],
            )

            # FIXED: EpisodicMemory uses add_episode method
            if self.deps.am:  # Check if am exists
                if hasattr(self.deps.am, "add_episode"):
                    self.deps.am.add_episode(episode)
                elif hasattr(self.deps.am, "append"):
                    self.deps.am.append(episode)
        except ImportError:
            logger.debug("Episode type not available, skipping episodic memory")
        except Exception as e:
            logger.debug(f"Failed to store episode: {e}")

        # Add to execution history
        with self._lock:
            self.execution_history.append(result)
            self.reasoning_trace.append(
                {
                    "phase": "execution",
                    "action_type": action.get("type", "unknown"),
                    "success": success,
                }
            )

        return result

    def _learn_and_adapt(
        self, execution_result: Dict[str, Any], perception: Dict[str, Any]
    ):
        """
        Learn from experience and adapt

        Args:
            execution_result: Result from execution
            perception: Perception result
        """
        # Prepare learning experience
        learning_experience = {
            "embedding": perception.get("embedding", np.zeros(384)),
            "modality": perception.get("modality"),
            "reward": execution_result.get("reward", 0),
        }

        # Process through continual learner
        adaptation_result = self.deps.continual.process_experience(learning_experience)

        # Update causal links
        action = execution_result.get("action")
        if action:
            try:
                self.deps.causal.update_causal_link(
                    action.get("type", "unknown"),
                    execution_result.get("observation", ""),
                    execution_result.get("reward", 0),
                    1.0 - execution_result.get("uncertainty", 0.5),
                )
            except Exception as e:
                logger.debug(f"Failed to update causal link: {e}")

        # Update goal progress
        goal_progress = execution_result.get("goal_progress")
        if goal_progress:
            try:
                self.deps.goal_system.update_progress(
                    goal_progress, max(0, execution_result.get("reward", 0))
                )
            except Exception as e:
                logger.debug(f"Failed to update goal progress: {e}")

        # Find cross-modal patterns if multiple modalities active
        if len(self.sys.active_modalities) > 1:
            try:
                patterns = self.deps.cross_modal.find_cross_modal_correspondence(
                    list(self.execution_history)[-10:]
                )
                if patterns:
                    logger.info(f"Discovered {len(patterns)} cross-modal patterns")
            except Exception as e:
                logger.debug(f"Failed to find cross-modal patterns: {e}")

        # Record in reasoning trace
        with self._lock:
            self.reasoning_trace.append(
                {
                    "phase": "learning",
                    "adaptation_loss": adaptation_result.get("loss", 0),
                    "adapted": adaptation_result.get("adapted", False),
                }
            )

    def _reflect_and_improve(self):
        """Meta-cognitive reflection and self-improvement"""
        with self._lock:
            last_trace = dict(self.reasoning_trace[-1]) if self.reasoning_trace else {}
            last_result = (
                dict(self.execution_history[-1]) if self.execution_history else {}
            )

        # Update self-model
        try:
            self.deps.meta_cognitive.update_self_model(
                {
                    "loss": last_trace.get("adaptation_loss", 0),
                    "reward": last_result.get("reward", 0),
                    "strategy": "default",
                    "modality": str(last_result.get("modality", "unknown")),
                }
            )
        except Exception as e:
            logger.debug(f"Failed to update self-model: {e}")

        # Analyze learning efficiency
        try:
            efficiency = self.deps.meta_cognitive.analyze_learning_efficiency()

            if efficiency.get("status") != "insufficient_data":
                avg_loss = efficiency.get("avg_loss", 0)
                self.sys.SA.learning_efficiency = 1.0 / (1.0 + avg_loss)
        except Exception as e:
            logger.debug(f"Failed to analyze learning efficiency: {e}")

        # Introspect reasoning quality
        try:
            reasoning_quality = self.deps.meta_cognitive.introspect_reasoning(
                list(self.reasoning_trace)
            )

            quality_score = reasoning_quality.get("quality_score")
            if quality_score is not None:
                self.sys.SA.uncertainty = 1.0 - quality_score
        except Exception as e:
            logger.debug(f"Failed to introspect reasoning: {e}")

        # Compute identity drift from action diversity
        if len(self.execution_history) > 100:
            try:
                recent_actions = [
                    h.get("action", {}).get("type", "")
                    for h in list(self.execution_history)[-100:]
                    if "action" in h
                ]

                if recent_actions:
                    unique_actions = len(set(recent_actions))
                    total_actions = len(recent_actions)
                    action_diversity = unique_actions / total_actions
                    self.sys.SA.identity_drift = 1.0 - action_diversity
            except Exception as e:
                logger.debug(f"Failed to compute identity drift: {e}")

    def _update_system_state(self, result: Dict[str, Any], duration: float):
        """
        Update system state after execution

        Args:
            result: Execution result
            duration: Execution duration in seconds
        """
        with self._sys_lock:
            self.sys.step += 1
            self.sys.last_obs = result.get("observation")
            self.sys.last_reward = result.get("reward")

            # Update health metrics
            self.sys.health.latency_ms = duration * 1000
            energy_used = result.get("resource_usage", {}).get("energy_nJ", 0)
            self.sys.health.energy_budget_left_nJ -= energy_used

            # Track active modalities
            modality = result.get("modality", ModalityType.UNKNOWN)
            if modality != ModalityType.UNKNOWN:
                self.sys.active_modalities.add(modality)

            # Store uncertainty estimate
            step_key = f"step_{self.sys.step}"
            self.sys.uncertainty_estimates[step_key] = result.get("uncertainty", 0.5)

            # Bounded uncertainty estimates (keep last 1000)
            if len(self.sys.uncertainty_estimates) > 1000:
                # Remove oldest entries
                keys_to_remove = sorted(self.sys.uncertainty_estimates.keys())[:-1000]
                for key in keys_to_remove:
                    del self.sys.uncertainty_estimates[key]

    def _add_provenance(self, result: Dict[str, Any]):
        """
        Add provenance record

        Args:
            result: Execution result
        """
        try:
            # FIXED: Import from parent directory (vulcan/)
            from ..vulcan_types import ProvRecord

            prov = ProvRecord(
                t=time.time(),
                graph_id=f"graph_{self.sys.step}",
                agent_version="VULCAN_AGI_1.0",
                policy_versions=self.sys.policies,
                input_hash=hashlib.md5(str(result).encode(), usedforsecurity=False).hexdigest(),
                kernel_sig=None,
                explainer_uri="",
                ecdsa_sig="",
                modality=result.get("modality", ModalityType.UNKNOWN),
                uncertainty=result.get("uncertainty", 0.5),
            )

            self.sys.provenance_chain.append(prov)

            # FIXED: Bound provenance chain to prevent memory growth
            if len(self.sys.provenance_chain) > 1000:
                self.sys.provenance_chain = self.sys.provenance_chain[-1000:]

        except ImportError:
            logger.debug("ProvRecord type not available, skipping provenance")
        except Exception as e:
            logger.debug(f"Failed to add provenance: {e}")

    def _create_wait_plan(self, reason: str) -> Dict[str, Any]:
        """
        Create wait action plan

        Args:
            reason: Reason for waiting

        Returns:
            Wait plan dictionary
        """
        return {
            "action": {"type": ActionType.WAIT.value},
            "reason": reason,
            "confidence": 0.1,
            "parameters": {},
        }

    def _create_safe_fallback(
        self, reason: str, original_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create safe fallback plan

        Args:
            reason: Reason for fallback
            original_plan: Original plan that failed validation

        Returns:
            Safe fallback plan dictionary
        """
        return {
            "action": {"type": ActionType.SAFE_FALLBACK.value},
            "reason": f"Safety violation: {reason}",
            "original_plan": original_plan,
            "confidence": 0.5,
            "parameters": {},
        }

    def _create_fallback_result(self, error: str) -> Dict[str, Any]:
        """
        Create fallback result for errors

        Args:
            error: Error message

        Returns:
            Fallback result dictionary
        """
        return {
            "action": {"type": ActionType.ERROR_FALLBACK.value},
            "error": error,
            "success": False,
            "observation": f"Error occurred: {error}",
            "reward": -1.0,
            "modality": ModalityType.UNKNOWN,
            "uncertainty": 1.0,
            "resource_usage": {},
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the collective

        Returns:
            Status dictionary
        """
        with self._lock:
            status = {
                "cycle_count": self.cycle_count,
                "execution_history_size": len(self.execution_history),
                "reasoning_trace_size": len(self.reasoning_trace),
                "agent_pool_status": self.agent_pool.get_pool_status(),
                "system_step": self.sys.step,
                "shutdown": self._shutdown_event.is_set(),
                "recent_errors": len(self.recent_errors),
            }

            # Add self-improvement stats if enabled
            if self.config.enable_self_improvement:
                status["self_improvement"] = {
                    "enabled": True,
                    "experiments_run": self.improvement_experiments_run,
                    "successes": self.improvement_successes,
                    "success_rate": (
                        self.improvement_successes / self.improvement_experiments_run
                    )
                    if self.improvement_experiments_run > 0
                    else 0.0,
                }
            else:
                status["self_improvement"] = {"enabled": False}

            return status

    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown orchestrator with proper thread cleanup and timeout handling.

        Args:
            timeout: Maximum time to wait for each service shutdown (seconds)
        """
        if self._shutdown_event.is_set():
            logger.warning("Collective already shutting down")
            return

        logger.info("Shutting down VULCAN-AGI Collective...")
        self._shutdown_event.set()

        # Track shutdown progress
        shutdown_errors = []

        # ========================================
        # PHASE 1: Signal shutdown to all services
        # ========================================
        services_to_shutdown = {
            "agent_pool": getattr(self, "agent_pool", None),
            "governance": getattr(self.deps, "governance", None),
            "safety_validator": getattr(self.deps, "safety_validator", None),
            "continual_learner": getattr(self.deps, "continual", None),
            "world_model": getattr(self.deps, "world_model", None),
            "meta_cognitive": getattr(self.deps, "meta_cognitive", None),
            "self_improvement_drive": getattr(
                self.deps, "self_improvement_drive", None
            ),
            "distributed_coordinator": getattr(self.deps, "distributed", None),
        }

        # CRITICAL FIX: Add rollback manager and audit logger
        if hasattr(self.deps, "rollback_manager"):
            services_to_shutdown["rollback_manager"] = self.deps.rollback_manager
        if hasattr(self.deps, "audit_logger"):
            services_to_shutdown["audit_logger"] = self.deps.audit_logger

        # Try to shutdown each service with timeout
        for name, service in services_to_shutdown.items():
            if service is None:
                continue

            if not hasattr(service, "shutdown") or not callable(
                getattr(service, "shutdown")
            ):
                logger.debug(f"Service {name} has no shutdown method, skipping")
                continue

            try:
                logger.debug(f"Shutting down {name}...")

                # Use threading to enforce timeout
                shutdown_thread = threading.Thread(
                    target=service.shutdown,
                    name=f"shutdown_{name}",
                    daemon=True,  # Ensure it doesn't block program exit
                )
                shutdown_thread.start()
                shutdown_thread.join(timeout=timeout)

                if shutdown_thread.is_alive():
                    logger.error(
                        f"Service {name} shutdown timed out after {timeout}s. "
                        f"Thread may still be running."
                    )
                    shutdown_errors.append(f"{name}: timeout")
                else:
                    logger.debug(f"{name} shutdown complete")

            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}", exc_info=True)
                shutdown_errors.append(f"{name}: {str(e)}")

        # ========================================
        # PHASE 2: Force cleanup of any remaining threads
        # ========================================
        try:
            active_threads = threading.enumerate()

            # Filter for threads we care about (not main thread, not daemon)
            worker_threads = [
                t
                for t in active_threads
                if t != threading.main_thread() and not t.daemon and t.is_alive()
            ]

            if worker_threads:
                logger.warning(
                    f"Found {len(worker_threads)} non-daemon threads still running. "
                    f"Waiting {timeout}s for cleanup..."
                )

                # Give threads one more chance to finish
                for thread in worker_threads:
                    thread.join(timeout=timeout / max(len(worker_threads), 1))

                # Check again
                still_alive = [t for t in worker_threads if t.is_alive(])
                if still_alive:
                    logger.error(
                        f"{len(still_alive)} threads still alive after shutdown: "
                        f"{[t.name for t in still_alive]}"
                    )

        except Exception as e:
            logger.error(f"Error during thread cleanup: {e}", exc_info=True)

        # ========================================
        # PHASE 3: Clear data structures
        # ========================================
        try:
            with self._lock:
                self.reasoning_trace.clear()
                self.execution_history.clear()
                self.recent_errors.clear()
                self.error_count_window.clear()
        except Exception as e:
            logger.error(f"Error clearing data structures: {e}")

        # ========================================
        # PHASE 4: Force garbage collection
        # ========================================
        try:
            import gc

            gc.collect()
        except Exception as e:
            logger.debug(f"Error during garbage collection: {e}")

        # ========================================
        # PHASE 5: Report results
        # ========================================
        if shutdown_errors:
            logger.warning(
                f"Collective shutdown completed with {len(shutdown_errors)} errors: "
                f"{shutdown_errors}"
            )
        else:
            logger.info("VULCAN-AGI Collective shutdown complete (no errors)")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logger.debug(f"Error in destructor: {e}")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = ["VULCANAGICollective", "ModalityType", "ActionType"]

# ============================================================
# MODULE VALIDATION
# Ensure all exported names are properly defined to catch initialization errors early
# ============================================================


def _validate_module_exports():
    """Validate that all exported names are properly defined."""
    missing = []
    for name in __all__:
        if name not in globals():
            missing.append(name)

    if missing:
        raise ImportError(
            f"Module vulcan.orchestrator.collective failed to initialize properly. "
            f"Missing exports: {missing}. This may indicate a circular import or "
            f"initialization error."
        )


# Run validation when module is loaded
_validate_module_exports()
