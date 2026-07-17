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
        """Release each owned resource once, in reverse construction order."""
        if self.closed:
            return
        self.closed = True
        for resource in (self.language_output, self.language_input, self.memory, self.deployment):
            shutdown = getattr(resource, "close", None) or getattr(resource, "shutdown", None)
            if shutdown is not None:
                result = shutdown()
                if inspect.isawaitable(result):
                    await result

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
