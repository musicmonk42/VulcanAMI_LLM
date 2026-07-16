"""Typed owner for the one production cognitive object graph."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .kernel import CognitiveKernel


@dataclass
class RuntimeContainer:
    runtime_id: str
    deployment: Any
    world_state: Any
    kernel: CognitiveKernel
    safety: Any
    memory: Any | None
    closed: bool = False

    async def close(self) -> None:
        """Release owned resources once, in reverse construction order."""
        if self.closed:
            return
        self.closed = True
        shutdown = getattr(self.deployment, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    @classmethod
    def new(cls, *, deployment: Any, executor: Any) -> "RuntimeContainer":
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            raise RuntimeError("required canonical World State is unavailable")
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            raise RuntimeError("required safety finalization service is unavailable")
        kernel = CognitiveKernel(state_authority=world_state, executor=executor)
        return cls(runtime_id=str(uuid4()), deployment=deployment, world_state=world_state,
                   kernel=kernel, safety=safety, memory=getattr(deps, "memory", None))
