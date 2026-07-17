"""Typed owner for the one production cognitive object graph."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any
from vulcan.memory.governed import GovernedMemoryPort, compose_governed_memory
from uuid import uuid4

from .kernel import CognitiveKernel
from .finalization import SafetyResponseFinalizer


@dataclass
class RuntimeContainer:
    runtime_id: str
    deployment: Any
    world_state: Any
    kernel: CognitiveKernel
    safety: Any
    memory: GovernedMemoryPort
    closed: bool = False

    async def close(self) -> None:
        """Release owned resources once, in reverse construction order."""
        if self.closed:
            return
        self.closed = True
        self.memory.close()
        shutdown = getattr(self.deployment, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    @classmethod
    def new(cls, *, deployment: Any) -> "RuntimeContainer":
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            raise RuntimeError("required canonical World State is unavailable")
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            raise RuntimeError("required safety finalization service is unavailable")
        memory = compose_governed_memory()
        memory.readiness()
        kernel = CognitiveKernel(state_authority=world_state, finalizer=SafetyResponseFinalizer(safety), memory=memory)
        return cls(runtime_id=str(uuid4()), deployment=deployment, world_state=world_state,
                   kernel=kernel, safety=safety, memory=memory)
