"""Typed owner for the one production cognitive object graph."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from vulcan.memory.governed import GovernedMemoryPort, compose_governed_memory

from .finalization import SafetyResponseFinalizer
from .kernel import CognitiveKernel
from .output import DeterministicLanguageOutput, LanguageOutputPort
from .semantic import DeterministicLanguageInput, LanguageInputPort

LanguageMode = Literal["disabled", "deterministic_only"]


@dataclass(frozen=True)
class LanguageRuntimeConfig:
    """Closed deployment selection; local/provider modes are not yet selectable."""
    mode: LanguageMode = "deterministic_only"

    def validated(self) -> "LanguageRuntimeConfig":
        if self.mode not in {"disabled", "deterministic_only"}:
            raise RuntimeError("unapproved language-interface mode")
        return self


@dataclass
class RuntimeContainer:
    runtime_id: str
    deployment: Any
    world_state: Any
    kernel: CognitiveKernel
    safety: Any
    memory: GovernedMemoryPort
    language_input: LanguageInputPort
    language_output: LanguageOutputPort
    language_config: LanguageRuntimeConfig
    closed: bool = False

    async def close(self) -> None:
        """Release every owned resource once, preserving the first failure.

        Shutdown is deliberately best-effort across the complete composed graph:
        one broken legacy dependency must not leak the remaining owned handles.
        The first error is re-raised *after* every close hook has been offered a
        chance to run.
        """
        if self.closed:
            return
        self.closed = True
        first_error: BaseException | None = None
        seen: set[int] = set()
        for resource in (
            self.language_output,
            self.language_input,
            self.memory,
            self.kernel,
            self.safety,
            self.world_state,
            self.deployment,
        ):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            shutdown = getattr(resource, "close", None) or getattr(resource, "shutdown", None)
            if shutdown is not None:
                try:
                    result = shutdown()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as exc:  # continue closing all remaining owners
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error

    async def readiness(self) -> None:
        """Verify the actual object graph rather than a route-local flag."""
        if self.closed:
            raise RuntimeError("canonical runtime is closed")
        required = {
            "deployment": self.deployment,
            "world_state": self.world_state,
            "kernel": self.kernel,
            "safety": self.safety,
            "memory": self.memory,
            "language_input": self.language_input,
            "language_output": self.language_output,
        }
        for name, owner in required.items():
            if owner is None:
                raise RuntimeError(f"required canonical {name} is unavailable")
            check = getattr(owner, "readiness", None) or getattr(owner, "healthcheck", None)
            if check is not None:
                result = check()
                if inspect.isawaitable(result):
                    result = await result
                if result is False:
                    raise RuntimeError(f"required canonical {name} is unhealthy")

    def capabilities(self) -> tuple[str, ...]:
        """Return only capabilities supplied by the composed kernel object."""
        advertised = getattr(self.kernel, "capabilities", None)
        if not callable(advertised):
            raise RuntimeError("canonical kernel does not expose its capabilities")
        result = advertised()
        if not isinstance(result, tuple) or not all(isinstance(value, str) for value in result):
            raise RuntimeError("canonical kernel returned an invalid capability list")
        return result

    @classmethod
    def new(cls, *, deployment: Any, language_config: LanguageRuntimeConfig | None = None) -> "RuntimeContainer":
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            raise RuntimeError("required canonical World State is unavailable")
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            raise RuntimeError("required safety finalization service is unavailable")
        config = (language_config or LanguageRuntimeConfig()).validated()
        # Both supported modes intentionally construct only deterministic, no-model ports.
        language_input: LanguageInputPort = DeterministicLanguageInput()
        language_output: LanguageOutputPort = DeterministicLanguageOutput()
        memory = compose_governed_memory()
        try:
            memory.readiness()
            kernel = CognitiveKernel(state_authority=world_state, finalizer=SafetyResponseFinalizer(safety),
                                     language_input=language_input, language_output=language_output, memory=memory)
            return cls(str(uuid4()), deployment, world_state, kernel, safety, memory,
                       language_input, language_output, config)
        except Exception:
            memory.close()
            language_output.close()
            language_input.close()
            raise
