"""Runtime health state machine and bounded probe contracts."""
from __future__ import annotations

import asyncio
import inspect
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable


class ProcessState(str, Enum):
    STARTING = "starting"
    ADMITTED = "admitted"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    DRAINING = "draining"
    CLOSED = "closed"


class HealthFailureCategory(str, Enum):
    TRANSIENT_READINESS = "transient_readiness"
    CORRUPTION = "corruption"
    MISSING_OWNER = "missing_owner"
    CLOSED = "closed"
    DEPENDENCY = "dependency"
    DISK_SPACE = "disk_space"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntegrityResult:
    ok: bool
    category: HealthFailureCategory | None
    checked_at: datetime
    last_success_at: datetime | None


@dataclass(frozen=True)
class HealthSnapshot:
    state: ProcessState
    last_integrity: IntegrityResult | None


_TERMINAL = {ProcessState.FAILED, ProcessState.CLOSED}


class HealthStateMachine:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state = ProcessState.STARTING
        self._last_integrity: IntegrityResult | None = None

    @property
    def state(self) -> ProcessState:
        return self._state

    def snapshot(self) -> HealthSnapshot:
        return HealthSnapshot(self._state, self._last_integrity)

    def admit(self) -> None:
        self._transition({ProcessState.STARTING}, ProcessState.ADMITTED)

    def ready(self) -> None:
        self._transition({ProcessState.ADMITTED, ProcessState.DEGRADED, ProcessState.READY}, ProcessState.READY)

    def degrade(self, category: HealthFailureCategory = HealthFailureCategory.TRANSIENT_READINESS) -> None:
        if self._state in _TERMINAL or self._state is ProcessState.DRAINING:
            return
        self._state = ProcessState.DEGRADED

    def fail(self, category: HealthFailureCategory = HealthFailureCategory.CORRUPTION) -> None:
        if self._state is not ProcessState.CLOSED:
            self._state = ProcessState.FAILED

    def drain(self) -> None:
        if self._state not in _TERMINAL:
            self._state = ProcessState.DRAINING

    def close(self) -> None:
        self._state = ProcessState.CLOSED

    def record_integrity(self, *, ok: bool, category: HealthFailureCategory | None = None) -> IntegrityResult:
        now = self._clock()
        last_success = now if ok else (self._last_integrity.last_success_at if self._last_integrity else None)
        result = IntegrityResult(ok=ok, category=None if ok else (category or HealthFailureCategory.UNKNOWN), checked_at=now, last_success_at=last_success)
        self._last_integrity = result
        if ok and self._state in {ProcessState.ADMITTED, ProcessState.DEGRADED}:
            self.ready()
        elif not ok and result.category is HealthFailureCategory.CORRUPTION:
            self.fail(result.category)
        return result

    def _transition(self, allowed: set[ProcessState], target: ProcessState) -> None:
        if self._state not in allowed:
            raise RuntimeError(f"invalid health transition {self._state.value}->{target.value}")
        self._state = target


def categorize_failure(exc: BaseException) -> HealthFailureCategory:
    text = str(exc).lower()
    if "corruption" in text or "hash mismatch" in text or "integrity" in text:
        return HealthFailureCategory.CORRUPTION
    if "unavailable" in text or "missing" in text:
        return HealthFailureCategory.MISSING_OWNER
    if "disk" in text or "space" in text:
        return HealthFailureCategory.DISK_SPACE
    if "closed" in text:
        return HealthFailureCategory.CLOSED
    return HealthFailureCategory.TRANSIENT_READINESS


async def maybe_await_call(fn):
    value = fn()
    if inspect.isawaitable(value):
        return await value
    return value


async def bounded_disk_check(root: Path, *, minimum_free_bytes: int = 1_048_576) -> None:
    usage = await asyncio.to_thread(shutil.disk_usage, root)
    if usage.free < minimum_free_bytes:
        raise RuntimeError("runtime disk-space check failed")
