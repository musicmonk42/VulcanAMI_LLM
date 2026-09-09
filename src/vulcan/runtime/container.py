"""Typed owner for the one production cognitive object graph."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4
from pathlib import Path

from vulcan.memory.composition import GovernedMemoryPort, MemoryRuntimeConfig, compose_governed_memory
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner
from vulcan.learning_bandit import ShadowLinUCBToolBandit

from .alignment import AlignmentRegistry
from .audit import CanonicalAudit
from .domain_registry import PersistentDomainRegistry
from vulcan.safety.response_adapter import EnhancedSafetyResponseAdapter
from .finalization import SafetyResponseFinalizer
from .kernel import CognitiveKernel
from .output import DeterministicLanguageOutput, LanguageOutputPort
from .semantic import DeterministicLanguageInput, LanguageInputPort
from vulcan.improvement.proposal import ImprovementProposalStore
from .settings import RuntimeSettings
from .health import HealthFailureCategory, HealthStateMachine, ProcessState, bounded_disk_check, categorize_failure
from vulcan.microkernel.snapshots import MAX_EPISODE_LIFETIME, SnapshotBundle, construct_snapshot_bundle
from .state_authorities import ContentBoundStateAuthority, DisabledCSIUPolicyAuthority, StateAuthoritySet, disabled_authority
from .capabilities import CapabilityManifestAuthority, LiveOwnerFact, composed_runtime_ports, load_capability_registry
from vulcan.constitution.primitives import Digest, canonical_json


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
    improvement_proposals: ImprovementProposalStore | None = None
    learning_owner: LearningOwner | None = None
    settings: RuntimeSettings | None = None
    closed: bool = False
    health: HealthStateMachine | None = None
    max_episode_lifetime_seconds: int = int(MAX_EPISODE_LIFETIME.total_seconds())
    episode_store: Any = None
    epistemic_store: Any = None
    state_authorities: StateAuthoritySet | None = None
    capability_authority: CapabilityManifestAuthority | None = None

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
        if self.health is not None:
            self.health.close()
        first_error: BaseException | None = None
        seen: set[int] = set()
        for resource in (
            self.language_output,
            self.language_input,
            self.memory,
            self.alignment,
            self.audit,
            self.learning_owner,
            self.domain_registry,
            self.episode_store,
            self.epistemic_store,
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

    def _required_owners(self) -> dict[str, Any]:
        return {
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
            "learning_owner": self.learning_owner,
        }

    async def admission(self) -> None:
        """Traffic gate: admitted runtime exists and is not draining/closing."""
        if self.closed or (self.health is not None and self.health.state in {ProcessState.DRAINING, ProcessState.CLOSED, ProcessState.FAILED}):
            raise RuntimeError("canonical runtime is closed")

    async def shallow_readiness(self) -> None:
        """Fast readiness: verify mandatory owners are present without deep I/O."""
        await self.admission()
        try:
            for name, owner in self._required_owners().items():
                if owner is None:
                    raise RuntimeError(f"required canonical {name} is unavailable")
            if isinstance(self.durable_root, (str, Path)):
                await bounded_disk_check(Path(self.durable_root))
            if self.health is not None:
                self.health.ready()
        except Exception as exc:
            if self.health is not None:
                self.health.degrade(categorize_failure(exc))
            raise

    async def deep_integrity(self) -> None:
        """Deep integrity: execute every owner-provided readiness/health check."""
        await self.admission()
        for name, owner in self._required_owners().items():
            if owner is None:
                if self.health is not None: self.health.record_integrity(ok=False, category=HealthFailureCategory.MISSING_OWNER)
                raise RuntimeError(f"required canonical {name} is unavailable")
        if isinstance(self.durable_root, (str, Path)):
            await bounded_disk_check(Path(self.durable_root))
        required = self._required_owners()
        try:
            for name, owner in required.items():
                check = getattr(owner, "readiness", None) or getattr(owner, "healthcheck", None)
                if check is not None:
                    result = await asyncio.to_thread(check)
                    if inspect.isawaitable(result):
                        result = await result
                    if result is False:
                        raise RuntimeError(f"required canonical {name} is unhealthy")
            if self.health is not None:
                self.health.record_integrity(ok=True)
        except Exception as exc:
            category = categorize_failure(exc)
            if self.health is not None:
                self.health.record_integrity(ok=False, category=category)
            raise

    async def readiness(self) -> None:
        """Backward-compatible deep readiness adapter; not used for liveness."""
        await self.deep_integrity()

    def admit_snapshot_bundle(self, episode_id: str) -> SnapshotBundle:
        """Pin every mutable state authority for one bounded episode lifetime."""
        if self.closed:
            raise RuntimeError("canonical runtime is closed")
        if self.state_authorities is None:
            raise RuntimeError("required faithful state authorities are unavailable")
        self.state_authorities.validate()
        from datetime import timedelta
        return construct_snapshot_bundle(episode_id=episode_id, providers=self.state_authorities.providers(), lifetime=timedelta(seconds=self.max_episode_lifetime_seconds))

    def capabilities(self) -> tuple[str, ...]:
        """Compatibility projection of canonical public attestation IDs."""
        if self.capability_authority is None:
            raise RuntimeError("canonical capability authority is unavailable")
        return tuple(item.capability_id for item in self.capability_authority.public_capabilities())

    @classmethod
    def new(cls, *, deployment: Any, settings: RuntimeSettings, language_config: LanguageRuntimeConfig | None = None) -> "RuntimeContainer":
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            from .errors import StartupErrorCategory, StartupFailure
            raise StartupFailure(StartupErrorCategory.WORLD_MISSING, "required canonical World State is unavailable")
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            from .errors import StartupErrorCategory, StartupFailure
            raise StartupFailure(StartupErrorCategory.SAFETY_MISSING, "required safety finalization service is unavailable")
        config = (language_config or LanguageRuntimeConfig(settings.language_mode.value, str(settings.language_release_path) if settings.language_release_path else None)).validated()
        # Deterministic remains default/fallback; transformer mode is admitted only after strict release verification.
        language_input: LanguageInputPort = DeterministicLanguageInput()
        if config.mode == "transformer_proposal":
            if config.provider_factory is None:
                raise RuntimeError("verified transformer release present but no safe provider factory is configured")
            from vulcan.local_language import build_verified_adapter
            language_input = build_verified_adapter(release_root=config.release_path or "", provider_factory=config.provider_factory)
        language_output: LanguageOutputPort = DeterministicLanguageOutput()
        root = str(settings.durable_root)
        Path(root).mkdir(parents=True, exist_ok=True)
        memory = audit = alignment = domain_registry = None
        try:
            audit = CanonicalAudit(f"{root}/audit/events.jsonl")
            memory = compose_governed_memory(MemoryRuntimeConfig(settings.memory_enabled, settings.memory_sqlite_path, settings.durable_root, settings.replicas, settings.memory_backend.value), audit=audit)
            memory.readiness()
            alignment = AlignmentRegistry(f"{root}/alignment/active.json", audit=audit)
            domain_registry = PersistentDomainRegistry(f"{root}/domains", audit=audit)
            improvement_proposals = ImprovementProposalStore(Path(root) / "improvement-proposals")
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
            response_safety = EnhancedSafetyResponseAdapter(safety)
            response_safety.readiness()
            kernel = CognitiveKernel(state_authority=world_state, finalizer=SafetyResponseFinalizer(response_safety),
                                     language_input=language_input, language_output=language_output, memory=memory, audit=audit, alignment=alignment)
            container = cls(str(uuid4()), deployment, world_state, kernel, safety, memory,
                       language_input, language_output, config, audit, alignment, domain_registry, Path(root), improvement_proposals, learning_owner, settings, False, HealthStateMachine(), int(MAX_EPISODE_LIFETIME.total_seconds()))
            def domain_state():
                lease = domain_registry.lease()
                return lease.domain_snapshot_id, {"snapshot_id": lease.domain_snapshot_id}, lease
            def alignment_state():
                lease = alignment.lease()
                return str(lease.revision), {"policy_digest": lease.policy_digest}, lease
            def memory_state():
                revision, state = memory.snapshot_state()
                return revision, state, None
            capability_registry = load_capability_registry()
            arithmetic = capability_registry.records["cap.bounded_arithmetic"]
            def live_capability_facts():
                kernel_caps = tuple(kernel.capabilities())
                ready = isinstance(kernel, CognitiveKernel) and "bounded-arithmetic" in kernel_caps
                policy_digest = hashlib.sha256(
                    (Path(__file__).resolve().parents[3] / "docs" / "architecture" / "ami-invariants.yaml").read_bytes()
                ).hexdigest()
                ports = composed_runtime_ports()
                return {"cap.bounded_arithmetic": LiveOwnerFact(
                    owner=kernel.CAPABILITY_OWNER,
                    release_digest=kernel.CAPABILITY_RELEASE_DIGEST,
                    canonical_reachable=all(port in ports for port in arithmetic.port_reachability),
                    mode=config.mode,
                    state_digest=Digest.of_bytes(canonical_json({
                        "kernel_type": f"{type(kernel).__module__}.{type(kernel).__qualname__}",
                        "kernel_capabilities": kernel_caps,
                        "language_mode": config.mode,
                    })).hex,
                    ready=ready,
                    constitutionally_permitted=policy_digest == arithmetic.active_policy_digest,
                )}
            capability_authority = CapabilityManifestAuthority(registry=capability_registry, live_facts=live_capability_facts)
            container.capability_authority = capability_authority
            container.state_authorities = StateAuthoritySet(
                world=disabled_authority("world", reason="legacy world model has no faithful constitutional state export"),
                self_state=disabled_authority("self", reason="canonical self-state authority is not implemented"),
                social=disabled_authority("social", reason="canonical social-state authority is not implemented"),
                normative=disabled_authority("normative", reason="normative decisions are owned by alignment, not legacy world state"),
                domain=ContentBoundStateAuthority(kind="domain", owner="domain-registry", schema="vulcan-domain-snapshot.v1", release="constitutional-v1", read=domain_state),
                memory=ContentBoundStateAuthority(kind="memory", owner="governed-memory", schema="vulcan-memory-snapshot.v1", release="constitutional-v1", read=memory_state),
                capability=capability_authority,
                csiu=DisabledCSIUPolicyAuthority(reason="no serving-process CSIU policy owner is authorized"),
                alignment=ContentBoundStateAuthority(kind="alignment", owner="alignment-registry", schema="vulcan-alignment-snapshot.v1", release="constitutional-v1", read=alignment_state),
            )
            container.state_authorities.validate()
            # Exercise and release every reader before accepting traffic.
            startup_probe = container.admit_snapshot_bundle("startup-faithfulness-probe")
            startup_probe.close()
            container.health.admit()
            return container
        except Exception:
            for r in (locals().get("learning_owner"), domain_registry, alignment, audit, memory, language_output, language_input):
                if r is not None:
                    close=getattr(r,"close",None)
                    if close: close()
            raise
