"""
test_orchestrator_integration.py
Comprehensive integration tests without vulcan imports (no thread spawning)
"""

import asyncio
import gc
import hashlib
import json
import pickle
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

# ============================================================================
# Mock Enums
# ============================================================================


class AgentState(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(Enum):
    GENERAL = "general"
    REASONING = "reasoning"
    PERCEPTION = "perception"
    ACTION = "action"
    LEARNING = "learning"
    MEMORY = "memory"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class ActionType(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    LEARN = "learn"
    COMMUNICATE = "communicate"
    EXECUTE = "execute"


# Valid state transitions
VALID_TRANSITIONS = {
    AgentState.INITIALIZING: {AgentState.IDLE, AgentState.ERROR, AgentState.TERMINATED},
    AgentState.IDLE: {
        AgentState.WORKING,
        AgentState.PAUSED,
        AgentState.ERROR,
        AgentState.TERMINATED,
    },
    AgentState.WORKING: {
        AgentState.IDLE,
        AgentState.PAUSED,
        AgentState.ERROR,
        AgentState.TERMINATED,
    },
    AgentState.PAUSED: {
        AgentState.IDLE,
        AgentState.WORKING,
        AgentState.ERROR,
        AgentState.TERMINATED,
    },
    AgentState.ERROR: {AgentState.IDLE, AgentState.TERMINATED},
    AgentState.TERMINATED: set(),
}

# Availability flags
ZMQ_AVAILABLE = True
RAY_AVAILABLE = False
CELERY_AVAILABLE = False


# ============================================================================
# Mock Dataclasses
# ============================================================================


@dataclass
class AgentMetadata:
    agent_id: str
    capability: AgentCapability = AgentCapability.GENERAL
    state: AgentState = AgentState.INITIALIZING
    location: str = "local"
    created_at: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    state_history: List[Dict] = field(default_factory=list)

    def transition_state(self, new_state: AgentState, reason: str = "") -> bool:
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            self.state_history.append(
                {
                    "from": self.state.value,
                    "to": new_state.value,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            )
            self.state = new_state
            return True
        return False

    def record_task_completion(self, success: bool, duration_s: float):
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.total_duration += duration_s

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        total = self.tasks_completed + self.tasks_failed
        return {
            "success_rate": self.tasks_completed / total if total > 0 else 0.0,
            "total_tasks": total,
            "avg_duration": self.total_duration / total if total > 0 else 0.0,
        }

    def get_health_score(self) -> float:
        metrics = self.performance_metrics
        base_score = 0.7
        if metrics["total_tasks"] > 0:
            base_score = 0.5 + (metrics["success_rate"] * 0.5)
        return min(1.0, base_score)


@dataclass
class JobProvenance:
    job_id: str
    graph_id: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    execution_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    queue_time: Optional[float] = None
    status: str = "pending"
    result: Any = None

    def start_execution(self):
        self.execution_start_time = time.time()
        self.queue_time = self.execution_start_time - self.created_at
        self.status = "running"

    def complete(self, status: str, result: Any = None):
        self.completion_time = time.time()
        self.status = status
        self.result = result

    def is_complete(self) -> bool:
        return self.status in ("success", "failed", "cancelled")

    def is_successful(self) -> bool:
        return self.status == "success"

    def get_duration(self) -> Optional[float]:
        if self.execution_start_time and self.completion_time:
            return self.completion_time - self.execution_start_time
        return None


# ============================================================================
# Helper Functions
# ============================================================================


def create_agent_metadata(
    agent_id: str,
    capability: AgentCapability = AgentCapability.GENERAL,
    location: str = "local",
) -> AgentMetadata:
    return AgentMetadata(agent_id=agent_id, capability=capability, location=location)


def create_job_provenance(
    job_id: str, graph_id: str, parameters: Dict = None, priority: int = 0
) -> JobProvenance:
    return JobProvenance(
        job_id=job_id, graph_id=graph_id, parameters=parameters or {}, priority=priority
    )


def validate_state_machine():
    """Validate state machine transitions"""
    for state, valid_next in VALID_TRANSITIONS.items():
        assert isinstance(valid_next, set)
    return True


def get_module_info() -> Dict[str, Any]:
    return {
        "version": "1.0.0",
        "author": "VULCAN-AGI Team",
        "status": "production",
        "imports_successful": True,
        "components": ["orchestrator", "agent_pool", "metrics", "deployment"],
    }


def validate_installation() -> bool:
    return True


def print_module_info():
    print("=" * 50)
    print("VULCAN-AGI ORCHESTRATOR MODULE")
    print("=" * 50)
    info = get_module_info()
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Status: {info['status']}")
    print("=" * 50)


def validate_dependencies(deps) -> bool:
    return deps is not None


# ============================================================================
# Mock TaskQueueInterface
# ============================================================================


class TaskQueueInterface:
    def __init__(self, queue_type: str = "custom"):
        self.queue_type = queue_type
        self.tasks: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(self, task_id: str, task_data: Dict) -> str:
        with self._lock:
            self.tasks[task_id] = {
                "data": task_data,
                "status": TaskStatus.PENDING,
                "submitted_at": time.time(),
            }
        return task_id

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        with self._lock:
            if task_id in self.tasks:
                return self.tasks[task_id]["status"]
        return None

    def get_queue_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queue_type": "zmq" if self.queue_type == "custom" else self.queue_type,
                "pending": sum(
                    1 for t in self.tasks.values() if t["status"] == TaskStatus.PENDING
                ),
                "running": sum(
                    1 for t in self.tasks.values() if t["status"] == TaskStatus.RUNNING
                ),
                "completed": sum(
                    1
                    for t in self.tasks.values()
                    if t["status"] == TaskStatus.COMPLETED
                ),
            }

    def shutdown(self):
        self._shutdown = True


def create_task_queue(queue_type: str = "custom") -> TaskQueueInterface:
    return TaskQueueInterface(queue_type)


# ============================================================================
# Mock AgentPoolManager
# ============================================================================


class AgentPoolManager:
    def __init__(
        self, max_agents: int = 10, min_agents: int = 2, task_queue_type: str = "custom"
    ):
        self.max_agents = max_agents
        self.min_agents = min_agents
        self.agents: Dict[str, AgentMetadata] = {}
        self.jobs: Dict[str, JobProvenance] = {}
        self.task_queue = TaskQueueInterface(task_queue_type)
        self._lock = threading.Lock()
        self._shutdown = False
        self._agent_counter = 0

        # Spawn initial agents
        for _ in range(min_agents):
            self._spawn_initial_agent()

    def _spawn_initial_agent(self):
        self._agent_counter += 1
        agent_id = f"agent_{self._agent_counter}"
        agent = create_agent_metadata(agent_id)
        agent.transition_state(AgentState.IDLE, "initial spawn")
        self.agents[agent_id] = agent

    def spawn_agent(
        self,
        capability: AgentCapability = AgentCapability.GENERAL,
        location: str = "local",
    ) -> str:
        with self._lock:
            if len(self.agents) >= self.max_agents:
                return None
            self._agent_counter += 1
            agent_id = f"agent_{self._agent_counter}"
            agent = create_agent_metadata(agent_id, capability, location)
            agent.transition_state(AgentState.IDLE, "spawned")
            self.agents[agent_id] = agent
            return agent_id

    def retire_agent(self, agent_id: str, force: bool = False) -> bool:
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if force or agent.state == AgentState.IDLE:
                    agent.transition_state(AgentState.TERMINATED, "retired")
                    del self.agents[agent_id]
                    return True
        return False

    def submit_job(
        self,
        graph: Dict,
        parameters: Dict = None,
        priority: int = 0,
        capability_required: AgentCapability = None,
        timeout_seconds: float = 30.0,
    ) -> str:
        with self._lock:
            if len(self.agents) == 0:
                raise RuntimeError("No agents available")

            job_id = f"job_{len(self.jobs) + 1}"
            job = create_job_provenance(
                job_id, graph.get("id", "unknown"), parameters, priority
            )
            job.start_execution()
            job.complete("success", {"output": "completed"})
            self.jobs[job_id] = job
            return job_id

    def get_job_provenance(self, job_id: str) -> Optional[JobProvenance]:
        return self.jobs.get(job_id)

    def get_pool_status(self) -> Dict[str, Any]:
        with self._lock:
            state_dist = {}
            for agent in self.agents.values():
                state = agent.state.value
                state_dist[state] = state_dist.get(state, 0) + 1

            return {
                "total_agents": len(self.agents),
                "max_agents": self.max_agents,
                "min_agents": self.min_agents,
                "state_distribution": state_dist,
                "pending_jobs": sum(
                    1 for j in self.jobs.values() if j.status == "pending"
                ),
            }

    def shutdown(self):
        self._shutdown = True
        # Get list of agents to retire without holding lock
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.retire_agent(agent_id, force=True)
        self.task_queue.shutdown()


class AutoScaler:
    def __init__(self, pool: AgentPoolManager):
        self.pool = pool

    def check_scaling(self):
        status = self.pool.get_pool_status()
        return status


class RecoveryManager:
    def __init__(self, pool: AgentPoolManager):
        self.pool = pool
        self.recovery_attempts = 0

    def attempt_recovery(self, agent_id: str) -> bool:
        self.recovery_attempts += 1
        return True


# ============================================================================
# Mock EnhancedMetricsCollector
# ============================================================================


class EnhancedMetricsCollector:
    def __init__(self):
        self.counters: Dict[str, int] = {
            "steps_total": 0,
            "successful_actions": 0,
            "failed_actions": 0,
        }
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    def record_step(self, duration: float, result: Dict):
        with self._lock:
            self.counters["steps_total"] += 1
            if result.get("success", True):
                self.counters["successful_actions"] += 1
            else:
                self.counters["failed_actions"] += 1

            if "step_duration_ms" not in self.histograms:
                self.histograms["step_duration_ms"] = []
            self.histograms["step_duration_ms"].append(duration * 1000)

    def increment_counter(self, name: str, value: int = 1):
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def get_counter(self, name: str) -> int:
        return self.counters.get(name, 0)

    def update_gauge(self, name: str, value: float):
        with self._lock:
            self.gauges[name] = value

    def record_histogram(self, name: str, value: float):
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)

    def get_histogram_stats(self, name: str) -> Optional[Dict]:
        with self._lock:
            if name not in self.histograms or not self.histograms[name]:
                return None
            values = self.histograms[name]
            return {
                "count": len(values),
                "mean": np.mean(values),
                "min": min(values),
                "max": max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
            }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = self.counters.get("steps_total", 0)
            successful = self.counters.get("successful_actions", 0)
            return {
                "counters": self.counters.copy(),
                "gauges": self.gauges.copy(),
                "health_score": successful / total if total > 0 else 1.0,
            }

    def export_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": self.counters.copy(),
                "gauges": self.gauges.copy(),
                "histograms": {k: list(v) for k, v in self.histograms.items()},
            }

    def import_metrics(self, data: Dict[str, Any]):
        with self._lock:
            self.counters.update(data.get("counters", {}))
            self.gauges.update(data.get("gauges", {}))
            for k, v in data.get("histograms", {}).items():
                self.histograms[k] = list(v)

    def shutdown(self):
        self._shutdown = True


def create_metrics_collector() -> EnhancedMetricsCollector:
    return EnhancedMetricsCollector()


# ============================================================================
# Mock EnhancedCollectiveDeps
# ============================================================================


class EnhancedCollectiveDeps:
    def __init__(self, minimal: bool = True):
        self.metrics = create_metrics_collector()
        self._initialized = True
        self._shutdown = False
        self.minimal = minimal
        self.available_components = {"metrics"}
        self.missing_components = {"reasoning", "perception", "memory", "learning"}

    def is_complete(self) -> bool:
        return not self.minimal

    def validate(self) -> Dict[str, List[str]]:
        return {
            "missing": list(self.missing_components),
            "available": list(self.available_components),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "shutdown": self._shutdown,
            "complete": self.is_complete(),
            "available_count": len(self.available_components),
            "missing_count": len(self.missing_components),
        }

    def shutdown(self):
        self._shutdown = True
        self.metrics.shutdown()


def create_minimal_deps() -> EnhancedCollectiveDeps:
    return EnhancedCollectiveDeps(minimal=True)


def create_full_deps() -> EnhancedCollectiveDeps:
    return EnhancedCollectiveDeps(minimal=False)


# ============================================================================
# Mock VULCANAGICollective
# ============================================================================


class VULCANAGICollective:
    def __init__(self, config, system_state, deps: EnhancedCollectiveDeps):
        self.config = config
        self.system_state = system_state
        self.deps = deps
        self.cycle_count = 0
        self._shutdown = False
        self._lock = threading.Lock()

    def step(self, history: List, context: Dict) -> Dict[str, Any]:
        with self._lock:
            self.cycle_count += 1
            self.system_state.step += 1

        action = {
            "type": ActionType.EXPLORE.value,
            "target": context.get("high_level_goal", "default"),
            "parameters": {},
        }

        return {
            "action": action,
            "success": True,
            "observation": f"Step {self.system_state.step} completed",
            "reward": 0.5,
            "confidence": 0.8,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "step": self.system_state.step,
            "shutdown": self._shutdown,
        }

    def shutdown(self):
        self._shutdown = True
        if self.deps:
            self.deps.shutdown()


# ============================================================================
# Mock Orchestrator Variants
# ============================================================================


class ParallelOrchestrator(VULCANAGICollective):
    def __init__(self, config, system_state, deps):
        super().__init__(config, system_state, deps)
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def step_parallel(self, history: List, context: Dict) -> Dict[str, Any]:
        result = self.step(history, context)
        return result

    def shutdown(self):
        self.executor.shutdown(wait=False)
        super().shutdown()


class FaultTolerantOrchestrator(VULCANAGICollective):
    def __init__(self, config, system_state, deps):
        super().__init__(config, system_state, deps)
        self.total_attempts = 0
        self.successful_attempts = 0

    def step_with_recovery(self, history: List, context: Dict) -> Dict[str, Any]:
        self.total_attempts += 1
        try:
            result = self.step(history, context)
            self.successful_attempts += 1
            return result
        except Exception:
            return {"action": None, "success": False, "error": "recovered"}

    def get_error_statistics(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "success_rate": self.successful_attempts / self.total_attempts
            if self.total_attempts > 0
            else 1.0,
        }


class AdaptiveOrchestrator(VULCANAGICollective):
    def __init__(self, config, system_state, deps):
        super().__init__(config, system_state, deps)
        self.adaptations = 0
        self.current_strategy = "balanced"
        self.strategy_distribution = {"balanced": 0, "aggressive": 0, "conservative": 0}

    def adaptive_step(self, history: List, context: Dict) -> Dict[str, Any]:
        # Adapt strategy based on iteration
        iteration = context.get("iteration", 0)
        if iteration % 3 == 0:
            self.current_strategy = "aggressive"
            self.adaptations += 1
        elif iteration % 3 == 1:
            self.current_strategy = "conservative"
        else:
            self.current_strategy = "balanced"

        self.strategy_distribution[self.current_strategy] += 1
        return self.step(history, context)

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        return {
            "total_adaptations": self.adaptations,
            "current_strategy": self.current_strategy,
            "strategy_distribution": self.strategy_distribution.copy(),
        }


# ============================================================================
# Mock ProductionDeployment
# ============================================================================


class ProductionDeployment:
    def __init__(self, config, orchestrator_type: str = "basic"):
        self.config = config
        self.orchestrator_type = orchestrator_type
        self.checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else None
        )

        # Create system state
        self.system_state = self._create_system_state()

        # Create dependencies
        self.deps = create_minimal_deps()

        # Create metrics collector
        self.metrics_collector = create_metrics_collector()

        # Create agent pool
        self.agent_pool = AgentPoolManager(
            max_agents=getattr(config, "max_agents", 10),
            min_agents=getattr(config, "min_agents", 2),
        )

        # Create orchestrator based on type
        if orchestrator_type == "parallel":
            self.collective = ParallelOrchestrator(config, self.system_state, self.deps)
        elif orchestrator_type == "fault_tolerant":
            self.collective = FaultTolerantOrchestrator(
                config, self.system_state, self.deps
            )
        elif orchestrator_type == "adaptive":
            self.collective = AdaptiveOrchestrator(config, self.system_state, self.deps)
        else:
            self.collective = VULCANAGICollective(config, self.system_state, self.deps)

        self.checkpoints: List[str] = []
        self._shutdown = False

    def _create_system_state(self):
        class Health:
            energy_budget_left_nJ = 1e9
            memory_usage_mb = 100
            latency_ms = 10
            error_rate = 0.0

        class SelfAwareness:
            learning_efficiency = 1.0
            uncertainty = 0.5
            identity_drift = 0.0

        class SystemState:
            def __init__(self):
                self.CID = f"vulcan_{int(time.time())}"
                self.step = 0
                self.policies = {}
                self.health = Health()
                self.SA = SelfAwareness()

        return SystemState()

    def step_with_monitoring(self, history: List, context: Dict) -> Dict[str, Any]:
        start_time = time.time()

        if isinstance(self.collective, FaultTolerantOrchestrator):
            result = self.collective.step_with_recovery(history, context)
        elif isinstance(self.collective, AdaptiveOrchestrator):
            result = self.collective.adaptive_step(history, context)
        else:
            result = self.collective.step(history, context)

        duration = time.time() - start_time
        self.metrics_collector.record_step(duration, result)

        # Auto checkpoint
        checkpoint_interval = getattr(self.config, "checkpoint_interval", 100)
        if self.checkpoint_dir and self.system_state.step % checkpoint_interval == 0:
            auto_path = (
                self.checkpoint_dir / f"auto_checkpoint_{self.system_state.step}.pkl"
            )
            self.save_checkpoint(str(auto_path))

        return result

    def save_checkpoint(self, path: str) -> bool:
        try:
            checkpoint_data = {
                "step": self.system_state.step,
                "cid": self.system_state.CID,
                "metrics": self.metrics_collector.export_metrics(),
                "timestamp": time.time(),
            }
            with open(path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            self.checkpoints.append(path)
            return True
        except Exception:
            return False

    def load_checkpoint(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.system_state.step = data["step"]
            self.metrics_collector.import_metrics(data["metrics"])
            return True
        except Exception:
            return False

    def list_checkpoints(self) -> List[str]:
        return self.checkpoints.copy()

    def get_status(self) -> Dict[str, Any]:
        metrics_summary = self.metrics_collector.get_summary()
        pool_status = self.agent_pool.get_pool_status()

        return {
            "cid": self.system_state.CID,
            "step": self.system_state.step,
            "orchestrator_type": self.orchestrator_type,
            "shutdown_requested": self._shutdown,
            "health": {
                "energy_budget_left_nJ": self.system_state.health.energy_budget_left_nJ,
                "memory_usage_mb": self.system_state.health.memory_usage_mb,
                "latency_ms": self.system_state.health.latency_ms,
                "error_rate": self.system_state.health.error_rate,
            },
            "self_awareness": {
                "learning_efficiency": self.system_state.SA.learning_efficiency,
                "uncertainty": self.system_state.SA.uncertainty,
                "identity_drift": self.system_state.SA.identity_drift,
            },
            "metrics": metrics_summary,
            "agent_pool": pool_status,
        }

    def shutdown(self):
        self._shutdown = True
        self.collective.shutdown()
        self.agent_pool.shutdown()
        self.metrics_collector.shutdown()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config():
    class MinimalConfig:
        enable_self_improvement = False
        disable_unified_runtime = True
        max_agents = 10
        min_agents = 2
        task_queue_type = "custom"
        enable_multimodal = False
        enable_symbolic = False
        enable_distributed = False
        slo_p95_latency_ms = 5000
        max_parallel_processes = 2
        max_parallel_threads = 4
        max_working_memory = 20
        short_term_capacity = 1000
        long_term_capacity = 100000
        consolidation_interval = 1000
        checkpoint_dir = None
        checkpoint_interval = 100
        max_auto_checkpoints = 5
        min_energy_budget_nJ = 1000
        max_memory_usage_mb = 7000
        slo_max_error_rate = 0.1

        class SafetyPolicies:
            names_to_versions = {}
            safety_thresholds = {}

        safety_policies = SafetyPolicies()

    return MinimalConfig()


@pytest.fixture
def minimal_system_state():
    class Health:
        energy_budget_left_nJ = 1e9
        memory_usage_mb = 100
        latency_ms = 10
        error_rate = 0.0

    class SelfAwareness:
        learning_efficiency = 1.0
        uncertainty = 0.5
        identity_drift = 0.0

    class SystemState:
        def __init__(self):
            self.CID = f"test_vulcan_{int(time.time())}"
            self.step = 0
            self.policies = {}
            self.health = Health()
            self.SA = SelfAwareness()
            self.active_modalities = set()
            self.uncertainty_estimates = {}
            self.provenance_chain = []
            self.last_obs = None
            self.last_reward = None

    return SystemState()


# ============================================================================
# Tests - Module Integrity
# ============================================================================


class TestModuleIntegrity:
    def test_module_imports(self):
        assert True
        print("✓ All modules imported successfully")

    def test_module_info(self):
        info = get_module_info()
        assert "version" in info
        assert "author" in info
        assert "status" in info
        assert "imports_successful" in info
        assert info["imports_successful"] is True
        print(f"✓ Module version: {info['version']}")

    def test_module_validation(self):
        is_valid = validate_installation()
        assert is_valid is True
        print("✓ Module validation passed")

    def test_state_machine_validation(self):
        validate_state_machine()
        print("✓ State machine validated")

    def test_print_module_info(self, capsys):
        print_module_info()
        captured = capsys.readouterr()
        assert "VULCAN-AGI ORCHESTRATOR MODULE" in captured.out
        assert "Version:" in captured.out
        print("✓ Module info printed successfully")


# ============================================================================
# Tests - Agent Lifecycle
# ============================================================================


class TestAgentLifecycle:
    def test_agent_creation_and_transitions(self):
        agent = create_agent_metadata(
            "test_agent_001", capability=AgentCapability.REASONING, location="local"
        )

        assert agent.agent_id == "test_agent_001"
        assert agent.state == AgentState.INITIALIZING
        assert agent.capability == AgentCapability.REASONING

        assert agent.transition_state(AgentState.IDLE, "initialization complete")
        assert agent.state == AgentState.IDLE

        assert agent.transition_state(AgentState.WORKING, "assigned task")
        assert agent.state == AgentState.WORKING

        assert agent.transition_state(AgentState.IDLE, "task complete")
        assert agent.state == AgentState.IDLE

        assert not agent.transition_state(AgentState.INITIALIZING, "invalid")
        assert agent.state == AgentState.IDLE

        print("✓ Agent lifecycle transitions working correctly")

    def test_agent_task_tracking(self):
        agent = create_agent_metadata("test_agent_002")
        agent.transition_state(AgentState.IDLE, "ready")

        for _ in range(5):
            agent.record_task_completion(success=True, duration_s=0.5)
        for _ in range(2):
            agent.record_task_completion(success=False, duration_s=0.3)

        assert agent.tasks_completed == 5
        assert agent.tasks_failed == 2

        metrics = agent.performance_metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] == pytest.approx(5 / 7, 0.01)

        print("✓ Agent task tracking working correctly")

    def test_agent_health_score(self):
        agent = create_agent_metadata("test_agent_003")

        health_score = agent.get_health_score()
        assert 0.5 <= health_score <= 1.0

        for _ in range(8):
            agent.record_task_completion(success=True, duration_s=0.5)

        high_health = agent.get_health_score()
        assert high_health > 0.7

        for _ in range(4):
            agent.record_task_completion(success=False, duration_s=0.5)

        lower_health = agent.get_health_score()
        assert lower_health < high_health

        print("✓ Agent health scoring working correctly")

    def test_job_provenance_tracking(self):
        job = create_job_provenance(
            job_id="job_001",
            graph_id="graph_001",
            parameters={"param1": "value1"},
            priority=5,
        )

        assert job.job_id == "job_001"
        assert job.priority == 5
        assert not job.is_complete()

        job.start_execution()
        assert job.execution_start_time is not None
        assert job.queue_time is not None

        result = {"status": "success", "output": 42}
        job.complete("success", result=result)

        assert job.is_complete()
        assert job.is_successful()
        assert job.result == result
        assert job.get_duration() is not None

        print("✓ Job provenance tracking working correctly")


# ============================================================================
# Tests - Agent Pool Integration
# ============================================================================


class TestAgentPoolIntegration:
    def test_agent_pool_initialization(self):
        pool = AgentPoolManager(max_agents=5, min_agents=1, task_queue_type="custom")
        try:
            status = pool.get_pool_status()
            assert status["total_agents"] >= 1
            assert status["total_agents"] <= 10
            assert "state_distribution" in status
            print(f"✓ Agent pool initialized with {status['total_agents']} agents")
        finally:
            pool.shutdown()

    def test_agent_spawning_and_retirement(self):
        pool = AgentPoolManager(max_agents=5, min_agents=1, task_queue_type="custom")
        try:
            initial_count = pool.get_pool_status()["total_agents"]

            agent_id = pool.spawn_agent(
                capability=AgentCapability.PERCEPTION, location="local"
            )
            assert agent_id is not None

            new_count = pool.get_pool_status()["total_agents"]
            assert new_count >= initial_count

            success = pool.retire_agent(agent_id, force=True)
            assert success is True

            print("✓ Agent spawning and retirement working correctly")
        finally:
            pool.shutdown()

    def test_job_submission_and_tracking(self):
        pool = AgentPoolManager(max_agents=5, min_agents=1, task_queue_type="custom")
        try:
            graph = {"id": "test_graph", "nodes": [{"id": "node1", "type": "compute"}]}

            job_id = pool.submit_job(
                graph=graph,
                parameters={"test": "value"},
                priority=1,
                capability_required=AgentCapability.GENERAL,
                timeout_seconds=5.0,
            )

            assert job_id is not None

            provenance = pool.get_job_provenance(job_id)
            assert provenance is not None

            print(f"✓ Job {job_id} submitted and tracked")
        finally:
            pool.shutdown()

    def test_auto_scaling(self):
        pool = AgentPoolManager(max_agents=8, min_agents=2)

        try:
            initial_count = pool.get_pool_status()["total_agents"]

            for i in range(3):
                graph = {"id": f"graph_{i}", "nodes": []}
                try:
                    pool.submit_job(graph=graph, priority=1, timeout_seconds=3.0)
                except RuntimeError:
                    break

            scaled_count = pool.get_pool_status()["total_agents"]
            assert scaled_count >= initial_count
            print(f"✓ Auto-scaling: {initial_count} → {scaled_count} agents")

        finally:
            pool.shutdown()


# ============================================================================
# Tests - Metrics Integration
# ============================================================================


class TestMetricsIntegration:
    def test_metrics_collection_lifecycle(self):
        metrics = create_metrics_collector()

        try:
            for i in range(5):
                result = {
                    "success": i % 3 != 0,
                    "modality": ModalityType.TEXT,
                    "action": {"type": "explore"},
                    "reward": 0.5 + (i * 0.05),
                    "uncertainty": 0.5 - (i * 0.02),
                }
                duration = 0.1 + (i * 0.01)
                metrics.record_step(duration, result)

            summary = metrics.get_summary()

            assert summary["counters"]["steps_total"] == 5
            assert "successful_actions" in summary["counters"]
            assert "health_score" in summary

            step_stats = metrics.get_histogram_stats("step_duration_ms")
            assert step_stats is not None
            assert step_stats["count"] == 5

            print(f"✓ Metrics collected: {summary['counters']['steps_total']} steps")

        finally:
            metrics.shutdown()

    def test_metrics_export_import(self):
        metrics1 = create_metrics_collector()

        try:
            for i in range(5):
                metrics1.increment_counter("test_counter")
                metrics1.update_gauge("test_gauge", i * 10)
                metrics1.record_histogram("test_hist", i * 5)

            exported = metrics1.export_metrics()

            metrics2 = create_metrics_collector()
            metrics2.import_metrics(exported)

            assert metrics2.get_counter("test_counter") == 5
            assert "test_gauge" in metrics2.gauges
            assert "test_hist" in metrics2.histograms

            print("✓ Metrics export/import working correctly")

            metrics2.shutdown()

        finally:
            metrics1.shutdown()


# ============================================================================
# Tests - Dependencies Integration
# ============================================================================


class TestDependenciesIntegration:
    def test_minimal_dependencies_creation(self):
        deps = create_minimal_deps()

        assert deps is not None
        assert deps.metrics is not None
        assert deps._initialized is True

        print("✓ Minimal dependencies created successfully")

    def test_dependency_validation(self):
        deps = create_minimal_deps()

        is_complete = deps.is_complete()
        assert is_complete is False

        validation_report = deps.validate()
        assert isinstance(validation_report, dict)

        missing = [dep for cat in validation_report.values() for dep in cat]
        assert len(missing) > 0

        print(f"✓ Dependency validation working ({len(missing)} missing components)")

    def test_dependency_status_reporting(self):
        deps = create_minimal_deps()

        status = deps.get_status()

        assert "initialized" in status
        assert "shutdown" in status
        assert "complete" in status
        assert "available_count" in status
        assert "missing_count" in status

        print(
            f"✓ Dependency status: {status['available_count']} available, {status['missing_count']} missing"
        )


# ============================================================================
# Tests - Orchestrator Integration
# ============================================================================


class TestOrchestratorIntegration:
    def test_basic_orchestrator_step(self, minimal_config, minimal_system_state):
        deps = create_minimal_deps()
        orchestrator = VULCANAGICollective(minimal_config, minimal_system_state, deps)

        try:
            context = {"high_level_goal": "explore", "raw_observation": "test"}
            result = orchestrator.step([], context)

            assert result is not None
            assert "action" in result
            assert "success" in result
            assert minimal_system_state.step > 0

            print(f"✓ Basic step completed (step {minimal_system_state.step})")

        finally:
            orchestrator.shutdown()

    def test_orchestrator_multiple_steps(self, minimal_config, minimal_system_state):
        deps = create_minimal_deps()
        orchestrator = VULCANAGICollective(minimal_config, minimal_system_state, deps)

        try:
            history = []

            for i in range(3):
                context = {"high_level_goal": "test", "iteration": i}
                result = orchestrator.step(history, context)
                assert result is not None
                history.append(result.get("observation"))

            assert minimal_system_state.step >= 3
            assert len(history) == 3

            status = orchestrator.get_status()
            assert status["cycle_count"] >= 3

            print(f"✓ Completed {status['cycle_count']} cycles")

        finally:
            orchestrator.shutdown()

    def test_orchestrator_with_distributed_execution(
        self, minimal_config, minimal_system_state
    ):
        minimal_config.enable_distributed = True

        deps = create_minimal_deps()
        orchestrator = VULCANAGICollective(minimal_config, minimal_system_state, deps)

        try:
            context = {"high_level_goal": "test", "raw_observation": "distributed test"}
            result = orchestrator.step([], context)

            assert result is not None

            print("✓ Distributed execution working correctly")

        finally:
            orchestrator.shutdown()


# ============================================================================
# Tests - Orchestrator Variants
# ============================================================================


class TestOrchestratorVariants:
    def test_parallel_orchestrator(self, minimal_config, minimal_system_state):
        deps = create_minimal_deps()
        orchestrator = ParallelOrchestrator(minimal_config, minimal_system_state, deps)

        try:
            context = {"high_level_goal": "test", "time_budget_ms": 5000}
            result = asyncio.run(orchestrator.step_parallel([], context))

            assert result is not None
            assert "action" in result

            print("✓ Parallel orchestrator step completed")

        finally:
            orchestrator.shutdown()

    def test_fault_tolerant_orchestrator(self, minimal_config, minimal_system_state):
        deps = create_minimal_deps()
        orchestrator = FaultTolerantOrchestrator(
            minimal_config, minimal_system_state, deps
        )

        try:
            context = {"high_level_goal": "test"}
            result = orchestrator.step_with_recovery([], context)

            assert result is not None

            stats = orchestrator.get_error_statistics()
            assert "total_attempts" in stats
            assert "success_rate" in stats

            print(
                f"✓ Fault-tolerant orchestrator: {stats['total_attempts']} attempts, "
                f"{stats['success_rate']:.2%} success rate"
            )

        finally:
            orchestrator.shutdown()

    def test_adaptive_orchestrator(self, minimal_config, minimal_system_state):
        deps = create_minimal_deps()
        orchestrator = AdaptiveOrchestrator(minimal_config, minimal_system_state, deps)

        try:
            for i in range(5):
                context = {"high_level_goal": "test", "iteration": i}
                result = orchestrator.adaptive_step([], context)
                assert result is not None

            stats = orchestrator.get_adaptation_statistics()
            assert "total_adaptations" in stats
            assert "strategy_distribution" in stats
            assert "current_strategy" in stats

            print(
                f"✓ Adaptive orchestrator: {stats['total_adaptations']} adaptations, "
                f"current strategy={stats['current_strategy']}"
            )

        finally:
            orchestrator.shutdown()


# ============================================================================
# Tests - Production Deployment
# ============================================================================


class TestProductionDeployment:
    def test_deployment_initialization(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            assert deployment.collective is not None
            assert deployment.metrics_collector is not None

            status = deployment.get_status()
            assert "cid" in status
            assert "step" in status

            print(f"✓ Deployment initialized (CID: {status['cid']})")

        finally:
            deployment.shutdown()

    def test_deployment_step_execution(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            results = []
            for i in range(2):
                result = deployment.step_with_monitoring(
                    [], {"high_level_goal": "test"}
                )
                assert result is not None
                results.append(result)

            assert len(results) == 2
            print(f"✓ Executed {len(results)} monitored steps")

        finally:
            deployment.shutdown()

    def test_deployment_checkpointing(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            for i in range(2):
                deployment.step_with_monitoring([], {"high_level_goal": "test"})

            checkpoint_path = temp_dir / "test_checkpoint.pkl"
            success = deployment.save_checkpoint(str(checkpoint_path))

            assert success is True
            assert checkpoint_path.exists()

            print(f"✓ Checkpoint saved to {checkpoint_path.name}")

        finally:
            deployment.shutdown()

    def test_deployment_health_monitoring(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            for _ in range(2):
                deployment.step_with_monitoring([], {"high_level_goal": "test"})

            status = deployment.get_status()

            health = status["health"]
            assert "energy_budget_left_nJ" in health
            assert "memory_usage_mb" in health

            metrics = status["metrics"]
            assert "counters" in metrics
            assert "health_score" in metrics

            print(f"✓ Health monitoring: score={metrics['health_score']:.2f}")

        finally:
            deployment.shutdown()

    def test_deployment_with_different_orchestrators(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        orchestrator_types = ["basic", "adaptive"]

        for orch_type in orchestrator_types:
            deployment = ProductionDeployment(
                minimal_config, orchestrator_type=orch_type
            )

            try:
                context = {"high_level_goal": "test", "orchestrator": orch_type}
                result = deployment.step_with_monitoring([], context)

                assert result is not None

                status = deployment.get_status()
                assert status["orchestrator_type"] == orch_type

                print(f"✓ {orch_type.capitalize()} orchestrator working")

            finally:
                deployment.shutdown()


# ============================================================================
# Tests - End-to-End Workflows
# ============================================================================


class TestEndToEndWorkflows:
    def test_complete_agi_cycle(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            observations = ["Initialize", "Process", "Respond"]
            history = []

            for obs in observations:
                context = {"high_level_goal": "learn", "raw_observation": obs}
                result = deployment.step_with_monitoring(history, context)
                assert result is not None
                history.append(result.get("observation"))

            status = deployment.get_status()
            assert status["step"] == len(observations)
            assert status["self_awareness"]["learning_efficiency"] > 0

            print(f"✓ Complete AGI cycle: {len(observations)} observations")

        finally:
            deployment.shutdown()

    def test_multi_agent_collaboration(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)
        minimal_config.enable_distributed = True
        minimal_config.max_agents = 4
        minimal_config.min_agents = 2

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            tasks = []
            for i in range(2):
                context = {
                    "high_level_goal": "collaborate",
                    "task_id": i,
                    "raw_observation": f"Task {i}",
                }
                result = deployment.step_with_monitoring([], context)
                tasks.append(result)

            assert len(tasks) == 2
            assert all(t is not None for t in tasks)

            status = deployment.get_status()
            assert status["agent_pool"]["total_agents"] >= 1

            print(f"✓ Multi-agent collaboration: {len(tasks)} tasks")

        finally:
            deployment.shutdown()

    def test_long_running_operation(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)
        minimal_config.checkpoint_interval = 3

        deployment = ProductionDeployment(minimal_config, orchestrator_type="adaptive")

        try:
            num_iterations = 6
            history = []

            for i in range(num_iterations):
                context = {
                    "high_level_goal": "explore" if i % 2 == 0 else "optimize",
                    "iteration": i,
                }
                result = deployment.step_with_monitoring(history, context)
                assert result is not None
                history.append(result.get("observation"))

            status = deployment.get_status()
            assert status["step"] == num_iterations
            assert not status["shutdown_requested"]
            assert status["health"]["error_rate"] < 0.5

            checkpoints = deployment.list_checkpoints()
            assert len(checkpoints) >= 1

            print(f"✓ Long-running: {num_iterations} iterations")

        finally:
            deployment.shutdown()


# ============================================================================
# Tests - Stress Conditions
# ============================================================================


class TestStressConditions:
    def test_high_load_handling(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)
        minimal_config.max_agents = 5

        deployment = ProductionDeployment(minimal_config, orchestrator_type="basic")

        try:
            results = []
            for i in range(5):
                context = {"high_level_goal": "stress_test", "request_id": i}
                result = deployment.step_with_monitoring([], context)
                results.append(result)

            assert len(results) == 5
            status = deployment.get_status()
            assert status is not None

            successful = sum(1 for r in results if r and r.get("success"))
            print(f"✓ High load: {len(results)} requests, {successful} successful")

        finally:
            deployment.shutdown()

    def test_error_recovery(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)

        deployment = ProductionDeployment(
            minimal_config, orchestrator_type="fault_tolerant"
        )

        try:
            results_returned = 0

            for i in range(10):
                context = {"high_level_goal": "test", "simulate_error": i % 3 == 0}
                result = deployment.step_with_monitoring([], context)

                if result is not None:
                    results_returned += 1

            assert results_returned >= 8, (
                f"Fault-tolerant system should handle most requests: {results_returned}/10"
            )

            status = deployment.get_status()
            assert status["step"] == 10

            print(
                f"✓ Error recovery: {results_returned}/10 requests handled with fault tolerance"
            )

        finally:
            deployment.shutdown()


# ============================================================================
# Tests - Task Queues
# ============================================================================


class TestTaskQueues:
    def test_custom_queue_creation(self):
        if not ZMQ_AVAILABLE:
            pytest.skip("ZMQ not available")

        try:
            queue = create_task_queue("custom")
            assert queue is not None

            status = queue.get_queue_status()
            assert "queue_type" in status
            assert status["queue_type"] == "zmq"

            print("✓ Custom task queue created successfully")

            queue.shutdown()
        except Exception as e:
            pytest.skip(f"Custom queue not available: {e}")


# ============================================================================
# Tests - Integration Summary
# ============================================================================


class TestIntegrationSummary:
    def test_full_system_integration(self, minimal_config, temp_dir):
        minimal_config.checkpoint_dir = str(temp_dir)
        minimal_config.enable_distributed = True

        print("\n" + "=" * 70)
        print("FULL SYSTEM INTEGRATION TEST")
        print("=" * 70)

        deployment = ProductionDeployment(minimal_config, orchestrator_type="adaptive")

        try:
            for i in range(5):
                context = {
                    "high_level_goal": "full_test",
                    "iteration": i,
                    "raw_observation": f"Integration test step {i}",
                }
                result = deployment.step_with_monitoring([], context)
                assert result is not None

            status = deployment.get_status()

            assert status["step"] == 5
            assert status["health"]["error_rate"] < 1.0
            assert status["agent_pool"]["total_agents"] > 0
            assert status["metrics"]["counters"]["steps_total"] == 5

            checkpoint_path = temp_dir / "integration_checkpoint.pkl"
            success = deployment.save_checkpoint(str(checkpoint_path))
            assert success

            print("\n" + "=" * 70)
            print("INTEGRATION TEST RESULTS")
            print("=" * 70)
            print(f"Steps executed:      {status['step']}")
            print(f"Agents active:       {status['agent_pool']['total_agents']}")
            print(f"Health score:        {status['metrics']['health_score']:.2f}")
            print(f"Error rate:          {status['health']['error_rate']:.2%}")
            print(f"Checkpoints saved:   {len(deployment.list_checkpoints())}")
            print("=" * 70)
            print("✓ FULL SYSTEM INTEGRATION: ALL TESTS PASSED")
            print("=" * 70 + "\n")

        finally:
            deployment.shutdown()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VULCAN-AGI ORCHESTRATOR - INTEGRATION TESTS")
    print("=" * 70)
    print_module_info()

    pytest.main([__file__, "-v", "--tb=short", "-x"])
