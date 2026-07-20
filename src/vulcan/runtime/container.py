"""Typed owner for the one production cognitive object graph."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4
from pathlib import Path

from vulcan.memory.governed import GovernedMemoryPort, MemoryRuntimeConfig, compose_governed_memory
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner
from vulcan.learning_bandit import ShadowLinUCBToolBandit

from .alignment import AlignmentRegistry
from .audit import CanonicalAudit
from .domain_registry import PersistentDomainRegistry
from .finalization import SafetyResponseFinalizer
from .kernel import CognitiveKernel
from .output import DeterministicLanguageOutput, LanguageOutputPort
from .semantic import DeterministicLanguageInput, LanguageInputPort
from .self_improvement import SelfImprovementRuntime, compose_self_improvement_runtime
from .settings import RuntimeSettings


LanguageMode = Literal["disabled", "deterministic_only", "transformer_proposal"]


@dataclass(frozen=True)
class LanguageRuntimeConfig:
    """Closed deployment selection; local/provider modes are not yet selectable."""
    mode: LanguageMode = "deterministic_only"
    release_path: str | None = None
    provider_factory: Any = None

    def validated(self) -> "LanguageRuntimeConfig":
        if self.mode not in {"disabled", "deterministic_only", "transformer_proposal"}:
            raise RuntimeError("unapproved language-interface mode")
        if self.mode == "transformer_proposal":
            from pathlib import Path
            if not self.release_path or not Path(self.release_path).is_absolute():
                raise RuntimeError("transformer mode requires an absolute approved release path")
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
    audit: CanonicalAudit | None = None
    alignment: AlignmentRegistry | None = None
    domain_registry: PersistentDomainRegistry | None = None
    durable_root: Path | None = None
    self_improvement: SelfImprovementRuntime | None = None
    learning_owner: LearningOwner | None = None
    settings: RuntimeSettings | None = None
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
            self.alignment,
            self.audit,
            self.self_improvement,
            self.learning_owner,
            self.domain_registry,
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
            "audit": self.audit,
            "alignment": self.alignment,
            "domain_registry": self.domain_registry,
            "durable_root": self.durable_root,
            "self_improvement": self.self_improvement,
            "learning_owner": self.learning_owner,
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
        if self.language_config.mode == "transformer_proposal":
            try:
                meta = self.language_input.readiness()
                abi = meta["runtime_abi"]
                result = tuple(dict.fromkeys((*result, "verified-transformer-span", f"language-abi:{abi}")))
            except Exception:
                pass
        if self.self_improvement is not None:
            result = tuple(dict.fromkeys((*result, *self.self_improvement.capabilities())))
        if self.learning_owner is None:
            raise RuntimeError("canonical learning owner is unavailable")
        learning_status = self.learning_owner.capability.value
        result = tuple(dict.fromkeys((*result, f"learning:{learning_status}")))
        if not isinstance(result, tuple) or not all(isinstance(value, str) for value in result):
            raise RuntimeError("canonical kernel returned an invalid capability list")
        return result

    @classmethod
    def new(cls, *, deployment: Any, settings: RuntimeSettings, language_config: LanguageRuntimeConfig | None = None) -> "RuntimeContainer":
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            raise RuntimeError("required canonical World State is unavailable")
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            raise RuntimeError("required safety finalization service is unavailable")
        config = (language_config or LanguageRuntimeConfig(settings.language_mode.value, str(settings.language_release_path) if settings.language_release_path else None)).validated()
        # Deterministic remains default/fallback; transformer mode is admitted only after strict release verification.
        language_input: LanguageInputPort = DeterministicLanguageInput()
        if config.mode == "transformer_proposal":
            if config.provider_factory is None:
                raise RuntimeError("verified transformer release present but no safe provider factory is configured")
            from vulcan.local_language import build_verified_adapter
            language_input = build_verified_adapter(release_root=config.release_path or "", provider_factory=config.provider_factory)
        language_output: LanguageOutputPort = DeterministicLanguageOutput()
        memory = compose_governed_memory(MemoryRuntimeConfig(settings.memory_enabled, settings.memory_sqlite_path, settings.durable_root, settings.replicas, settings.memory_backend.value))
        root = str(settings.durable_root)
        Path(root).mkdir(parents=True, exist_ok=True)
        audit = alignment = domain_registry = self_improvement = None
        try:
            memory.readiness()
            audit = CanonicalAudit(f"{root}/audit/events.jsonl")
            alignment = AlignmentRegistry(f"{root}/alignment/active.json", audit=audit)
            domain_registry = PersistentDomainRegistry(f"{root}/domains", audit=audit)
            self_improvement = compose_self_improvement_runtime(durable_root=Path(root), audit=audit, alignment=alignment, world_model=world_state, approval_hmac_secret=settings.approval_hmac_secret.reveal() if settings.approval_hmac_secret else None)
            shadow_bandit = ShadowLinUCBToolBandit()
            learning_owner = LearningOwner(
                capability=LearningCapabilityStatus.SHADOW,
                resources={"deployment_continual": getattr(deps, "continual", None)},
                shadow_bandit=shadow_bandit,
            )
            learning_owner.readiness()
            setattr(deps, "learning_owner", learning_owner)
            setattr(deps, "learning_system", learning_owner)
            setattr(deployment, "learning_owner", learning_owner)
            setattr(deployment, "learning_system", learning_owner)
            setattr(world_state, "domain", domain_registry)
            setattr(world_state, "self_improvement_runtime", self_improvement)
            setattr(world_state, "self_improvement_drive", self_improvement.drive)
            kernel = CognitiveKernel(state_authority=world_state, finalizer=SafetyResponseFinalizer(safety),
                                     language_input=language_input, language_output=language_output, memory=memory, audit=audit, alignment=alignment)
            return cls(str(uuid4()), deployment, world_state, kernel, safety, memory,
                       language_input, language_output, config, audit, alignment, domain_registry, Path(root), self_improvement, learning_owner, settings)
        except Exception:
            for r in (locals().get("learning_owner"), self_improvement, domain_registry, alignment, audit, memory, language_output, language_input):
                if r is not None:
                    close=getattr(r,"close",None)
                    if close: close()
            raise
