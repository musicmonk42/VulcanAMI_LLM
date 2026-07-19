"""Dependency-light canonical runtime owner for learning lifecycle.

This module intentionally imports only Python standard-library modules so runtime
ownership can be inspected without loading neural or web dependencies.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


class LearningCapabilityStatus(Enum):
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    OBSERVE_ONLY = "observe_only"
    EXPERIMENTAL = "experimental"
    SHADOW = "shadow"
    VERIFIED_INACTIVE = "verified_inactive"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


class LearningOwnerState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    UNHEALTHY = "unhealthy"


class LearningOwnerClosedError(RuntimeError):
    """Raised when a closed owner receives new work."""


class LearningOwnerBackpressureError(RuntimeError):
    """Raised immediately when a bounded owner queue is full."""


@dataclass(frozen=True)
class LearningCapabilitySnapshot:
    capability_id: str
    status: LearningCapabilityStatus
    implementation_digest: str
    required_dependencies: tuple[str, ...]
    proof_evaluation_id: str | None
    active_policy_revision: str | None
    unavailability_reason: str
    readiness_state: LearningOwnerState


@dataclass(frozen=True)
class QueueHealthSnapshot:
    capacity: int
    pending: int
    in_flight: int


@dataclass(frozen=True)
class LearningOwnerStatusSnapshot:
    owner_id: str
    state: LearningOwnerState
    capability: LearningCapabilityStatus
    observation_queue: QueueHealthSnapshot
    work_queue: QueueHealthSnapshot
    resources: Mapping[str, str] = field(default_factory=dict)
    capabilities: tuple[LearningCapabilitySnapshot, ...] = field(default_factory=tuple)


class LearningOwner:
    """Canonical runtime-owned learning lifecycle and queue authority."""

    def __init__(
        self,
        *,
        owner_id: str | None = None,
        observation_capacity: int = 128,
        work_capacity: int = 32,
        capability: LearningCapabilityStatus = LearningCapabilityStatus.OBSERVE_ONLY,
        resources: Mapping[str, Any] | None = None,
        shadow_bandit: Any | None = None,
        isolated_test_owner: bool = False,
    ) -> None:
        if not isolated_test_owner and owner_id is not None:
            raise RuntimeError("production learning owner identity is runtime-owned")
        if observation_capacity <= 0 or work_capacity <= 0:
            raise ValueError("learning owner queue capacities must be positive")
        if capability is LearningCapabilityStatus.ACTIVE:
            raise RuntimeError("active learning capability is disabled until verification gates pass")
        self._owner_id = owner_id or f"learning-{uuid4()}"
        self._state = LearningOwnerState.OPEN
        self._capability = capability
        self._observation_capacity = int(observation_capacity)
        self._work_capacity = int(work_capacity)
        self._observations: deque[Any] = deque()
        self._work: deque[Any] = deque()
        self._observation_in_flight = 0
        self._work_in_flight = 0
        self._worker_failed = False
        self._shadow_bandit = shadow_bandit
        owned_resources = dict(resources or {})
        if shadow_bandit is not None:
            owned_resources["shadow_tool_bandit"] = shadow_bandit
        self._resources: tuple[tuple[str, Any], ...] = tuple(owned_resources.items())
        self._closed_resource_ids: set[int] = set()

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def state(self) -> LearningOwnerState:
        return self._state

    @property
    def capability(self) -> LearningCapabilityStatus:
        if self._state is LearningOwnerState.CLOSED:
            return LearningCapabilityStatus.CLOSED
        if self._state is LearningOwnerState.UNHEALTHY or self._worker_failed:
            return LearningCapabilityStatus.UNHEALTHY
        return self._capability

    def readiness(self) -> bool:
        if self._state is LearningOwnerState.CLOSED:
            raise LearningOwnerClosedError("learning owner is closed")
        if self._worker_failed or self._state is LearningOwnerState.UNHEALTHY:
            raise RuntimeError("learning owner worker is unhealthy")
        return True

    def submit_observation(self, item: Any) -> str:
        from vulcan.learning_observation import LearningObservation, validate_observation

        if not isinstance(item, LearningObservation):
            raise ValueError("learning owner accepts only LearningObservation")
        validate_observation(item)
        return self._submit(self._observations, self._observation_capacity, item)

    def submit_work(self, item: Any) -> str:
        return self._submit(self._work, self._work_capacity, item)

    def record_shadow_tool_selection(self, observation: Any, candidate_set: Any = None) -> Any:
        """Record active/candidate tool distributions without influencing routing."""
        if self._shadow_bandit is None:
            raise RuntimeError("shadow tool bandit is not composed")
        self.readiness()
        return self._shadow_bandit.select_shadow(observation, candidate_set)

    def apply_committed_observation(self, observation: Any) -> Any:
        """Apply exactly one committed observation to the shadow candidate policy."""
        if self._shadow_bandit is None:
            raise RuntimeError("shadow tool bandit is not composed")
        self.readiness()
        return self._shadow_bandit.update_from_observation(observation)

    def _submit(self, queue: deque[Any], capacity: int, item: Any) -> str:
        if self._state is not LearningOwnerState.OPEN:
            raise LearningOwnerClosedError("learning owner is closed")
        if len(queue) >= capacity:
            raise LearningOwnerBackpressureError("learning owner queue is full")
        queue.append(item)
        return self._owner_id

    def mark_worker_failed(self) -> None:
        self._worker_failed = True
        if self._state is LearningOwnerState.OPEN:
            self._state = LearningOwnerState.UNHEALTHY

    def capability_matrix(self) -> tuple[LearningCapabilitySnapshot, ...]:
        readiness = self._state
        unhealthy = self.capability in (LearningCapabilityStatus.CLOSED, LearningCapabilityStatus.UNHEALTHY)
        specs = (
            ("online-learning-api", LearningCapabilityStatus.UNAVAILABLE, (), None, "legacy /learn endpoint fails closed"),
            ("canonical-observation-outbox", LearningCapabilityStatus.SHADOW, ("sqlite3",), "tests/test_learning_outbox.py", "shadow observation persistence only"),
            ("tool-selection-bandit", LearningCapabilityStatus.SHADOW, (), "tests/test_learning_shadow_bandit.py", "candidate cannot affect live routing"),
            ("governed-policy-activation", LearningCapabilityStatus.SHADOW, (), "tests/test_learning_governance.py", "requires external alignment/CSIU approval"),
            ("metacognition", LearningCapabilityStatus.OBSERVE_ONLY, (), "tests/test_learning_containment.py", "mutation application is unavailable"),
            ("progressive-learning", LearningCapabilityStatus.EXPERIMENTAL, ("torch",), "tests/test_progressive_research.py", "research-only; production activation rejected"),
            ("fomaml", LearningCapabilityStatus.EXPERIMENTAL, ("torch",), "tests/test_fomaml_research.py", "research-only; not connected to runtime"),
            ("maml", LearningCapabilityStatus.UNAVAILABLE, ("torch",), None, "not separately proven"),
            ("proto", LearningCapabilityStatus.UNAVAILABLE, ("torch",), None, "not separately proven"),
            ("rlhf-shadow-reward", LearningCapabilityStatus.EXPERIMENTAL, ("torch",), "tests/test_rlhf_shadow_reward.py", "shadow candidate only; PPO disabled"),
            ("ppo", LearningCapabilityStatus.UNAVAILABLE, ("torch",), None, "policy updates disabled"),
            ("packnet", LearningCapabilityStatus.UNAVAILABLE, ("torch",), None, "not proven or active"),
            ("world-model-planning", LearningCapabilityStatus.EXPERIMENTAL, ("torch",), "tests/test_world_model_research.py", "isolated research planner; production disabled"),
            ("federated-learning", LearningCapabilityStatus.UNAVAILABLE, (), None, "no verified implementation"),
            ("transfer-learning", LearningCapabilityStatus.UNAVAILABLE, (), None, "no verified implementation"),
            ("supervised-learning", LearningCapabilityStatus.UNAVAILABLE, (), None, "no production learning update path"),
            ("autonomous-self-improvement", LearningCapabilityStatus.UNAVAILABLE, (), None, "runtime learning mutation not authorized"),
            ("hallucination-prevention", LearningCapabilityStatus.UNAVAILABLE, (), None, "not a guaranteed learning capability"),
        )
        snapshots: list[LearningCapabilitySnapshot] = []
        for capability_id, status, deps, proof, reason in specs:
            final_status = LearningCapabilityStatus.UNHEALTHY if unhealthy else status
            final_reason = "learning owner is not ready" if unhealthy else reason
            digest_payload = {"capability_id": capability_id, "status": final_status.value, "proof": proof or "", "version": "learning-capability-matrix/1"}
            impl_digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            snapshots.append(LearningCapabilitySnapshot(capability_id, final_status, impl_digest, tuple(deps), proof, None, final_reason[:160], readiness))
        return tuple(snapshots)

    def status_snapshot(self) -> LearningOwnerStatusSnapshot:
        return LearningOwnerStatusSnapshot(
            owner_id=self._owner_id,
            state=self._state,
            capability=self.capability,
            observation_queue=QueueHealthSnapshot(
                self._observation_capacity, len(self._observations), self._observation_in_flight
            ),
            work_queue=QueueHealthSnapshot(self._work_capacity, len(self._work), self._work_in_flight),
            resources=MappingProxyType({name: type(resource).__name__ for name, resource in self._resources}),
            capabilities=self.capability_matrix(),
        )

    async def close(self) -> None:
        if self._state is LearningOwnerState.CLOSED:
            return
        self._state = LearningOwnerState.CLOSING
        first_error: BaseException | None = None
        for _name, resource in self._resources:
            if id(resource) in self._closed_resource_ids:
                continue
            self._closed_resource_ids.add(id(resource))
            close = getattr(resource, "close", None) or getattr(resource, "shutdown", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._observations.clear()
        self._work.clear()
        self._state = LearningOwnerState.CLOSED
        if first_error is not None:
            raise first_error
