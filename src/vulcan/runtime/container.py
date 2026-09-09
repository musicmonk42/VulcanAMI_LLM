"""Typed owner for the one production cognitive object graph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from vulcan.constitution.primitives import Digest, canonical_json
from vulcan.graphix.runtime import DeterministicLanguageInput, LanguageInputPort
from vulcan.improvement.proposal import ImprovementProposalStore
from vulcan.learning_bandit import ShadowLinUCBToolBandit
from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner
from vulcan.memory.composition import (
    GovernedMemoryPort,
    MemoryRuntimeConfig,
    compose_governed_memory,
)
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.lineage import LineageStore, LineageTransactionService
from vulcan.microkernel.principals import Principal, PrincipalKind
from vulcan.microkernel.snapshots import (
    MAX_EPISODE_LIFETIME,
    SnapshotBundle,
    construct_snapshot_bundle,
)
from vulcan.microkernel.transactions import ConstitutionalTransactionService
from vulcan.safety.response_adapter import EnhancedSafetyResponseAdapter

from .alignment import AlignmentRegistry
from .audit import CanonicalAudit
from .capabilities import (
    CapabilityManifestAuthority,
    LiveOwnerFact,
    composed_runtime_ports,
    load_capability_registry,
)
from .constitutional_kernel import ConstitutionalCognitiveKernel
from .domain_registry import PersistentDomainRegistry
from .finalization import SafetyResponseFinalizer
from .health import (
    HealthFailureCategory,
    HealthStateMachine,
    ProcessState,
    bounded_disk_check,
    categorize_failure,
)
from .kernel import CognitiveKernel
from .output import DeterministicLanguageOutput, LanguageOutputPort
from .settings import RuntimeSettings
from .state_authorities import (
    ContentBoundStateAuthority,
    DisabledCSIUPolicyAuthority,
    StateAuthoritySet,
    disabled_authority,
)

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
                raise RuntimeError(
                    "transformer mode requires an absolute approved release path"
                )
        return self


@dataclass(frozen=True)
class RuntimeOwnerInputs:
    """Already-constructed, typed edge owners consumed by the composition root.

    ``legacy_root`` is accepted only by :meth:`RuntimeContainer.new`, the named
    test/migration adapter.  Canonical composition always leaves it ``None``.
    """

    world_proposal: Any
    safety_validator: Any
    legacy_continual_proposal: Any = None
    legacy_root: Any = None
    audit_factory: Any = CanonicalAudit
    memory_factory: Any = compose_governed_memory
    alignment_factory: Any = AlignmentRegistry
    domain_factory: Any = PersistentDomainRegistry
    learning_factory: Any = LearningOwner
    safety_port_factory: Any = EnhancedSafetyResponseAdapter
    finalizer_factory: Any = SafetyResponseFinalizer
    state_authority_factory: Any = StateAuthoritySet
    capability_manifest_factory: Any = CapabilityManifestAuthority
    language_proposal_factory: Any = DeterministicLanguageInput
    language_output_factory: Any = DeterministicLanguageOutput
    episode_store_factory: Any = EpisodeStore
    epistemic_store_factory: Any = EpistemicStore
    lineage_store_factory: Any = LineageStore
    transaction_service_factory: Any = ConstitutionalTransactionService


@dataclass
class RuntimeContainer:
    runtime_id: str
    deployment: Any | None
    world_state: Any
    kernel: CognitiveKernel | ConstitutionalCognitiveKernel
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
    lineage_store: Any = None
    state_authorities: StateAuthoritySet | None = None
    capability_authority: CapabilityManifestAuthority | None = None
    transaction_service: ConstitutionalTransactionService | None = None
    ownership_close_order: tuple[str, ...] = ()
    response_safety: Any = None
    safety_finalizer: Any = None

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
        resources = {
            "language_output": self.language_output,
            "language_input": self.language_input,
            "memory": self.memory,
            "alignment": self.alignment,
            "audit": self.audit,
            "learning": self.learning_owner,
            "improvement_proposals": self.improvement_proposals,
            "domain_lookup": self.domain_registry,
            "episode_store": self.episode_store,
            "epistemic_store": self.epistemic_store,
            "lineage_store": self.lineage_store,
            "kernel": self.kernel,
            "transaction_service": self.transaction_service,
            "state_authorities": self.state_authorities,
            "capability_authority": self.capability_authority,
            "safety_finalizer": self.safety_finalizer,
            "response_safety": self.response_safety,
            "safety": self.safety,
            "world_proposal": self.world_state,
            "legacy_root": self.deployment,
        }
        order = self.ownership_close_order or tuple(resources)
        if self.deployment is not None and "legacy_root" not in order:
            order = (*order, "legacy_root")
        for resource in (resources[name] for name in order):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            shutdown = getattr(resource, "close", None) or getattr(
                resource, "shutdown", None
            )
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
        required = {
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
        if self.ownership_close_order:
            required.update(
                episode_store=self.episode_store,
                epistemic_store=self.epistemic_store,
                state_authorities=self.state_authorities,
                capability_authority=self.capability_authority,
                transaction_service=self.transaction_service,
                safety_finalizer=self.safety_finalizer,
                response_safety=self.response_safety,
                improvement_proposals=self.improvement_proposals,
            )
        elif self.deployment is not None:
            required = {"deployment": self.deployment, **required}
        return required

    async def admission(self) -> None:
        """Traffic gate: admitted runtime exists and is not draining/closing."""
        if self.closed or (
            self.health is not None
            and self.health.state
            in {ProcessState.DRAINING, ProcessState.CLOSED, ProcessState.FAILED}
        ):
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
                if self.health is not None:
                    self.health.record_integrity(
                        ok=False, category=HealthFailureCategory.MISSING_OWNER
                    )
                raise RuntimeError(f"required canonical {name} is unavailable")
        if isinstance(self.durable_root, (str, Path)):
            await bounded_disk_check(Path(self.durable_root))
        required = self._required_owners()
        try:
            for name, owner in required.items():
                check = getattr(owner, "readiness", None) or getattr(
                    owner, "healthcheck", None
                )
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

        return construct_snapshot_bundle(
            episode_id=episode_id,
            providers=self.state_authorities.providers(),
            lifetime=timedelta(seconds=self.max_episode_lifetime_seconds),
        )

    def capabilities(self) -> tuple[str, ...]:
        """Compatibility projection of canonical public attestation IDs."""
        if self.capability_authority is None:
            raise RuntimeError("canonical capability authority is unavailable")
        return tuple(
            item.capability_id
            for item in self.capability_authority.public_capabilities()
        )

    @classmethod
    def new(
        cls,
        *,
        deployment: Any,
        settings: RuntimeSettings,
        language_config: LanguageRuntimeConfig | None = None,
    ) -> "RuntimeContainer":
        """Compatibility adapter for tests and migrations using legacy shells.

        Remove when downstream callers construct :class:`RuntimeOwnerInputs`.
        The production composition root never calls this method.
        """
        deps = getattr(getattr(deployment, "collective", None), "deps", None)
        world_state = getattr(deps, "world_model", None)
        if world_state is None:
            from .errors import StartupErrorCategory, StartupFailure

            raise StartupFailure(
                StartupErrorCategory.WORLD_MISSING,
                "required canonical World State is unavailable",
            )
        safety = getattr(deps, "safety_validator", None)
        if safety is None:
            from .errors import StartupErrorCategory, StartupFailure

            raise StartupFailure(
                StartupErrorCategory.SAFETY_MISSING,
                "required safety finalization service is unavailable",
            )
        return cls.from_owner_inputs(
            inputs=RuntimeOwnerInputs(
                world_proposal=world_state,
                safety_validator=safety,
                legacy_continual_proposal=getattr(deps, "continual", None),
                legacy_root=deployment,
                audit_factory=CanonicalAudit,
                memory_factory=compose_governed_memory,
                alignment_factory=AlignmentRegistry,
                domain_factory=PersistentDomainRegistry,
                learning_factory=LearningOwner,
                safety_port_factory=EnhancedSafetyResponseAdapter,
                finalizer_factory=SafetyResponseFinalizer,
                state_authority_factory=StateAuthoritySet,
                capability_manifest_factory=CapabilityManifestAuthority,
                language_proposal_factory=DeterministicLanguageInput,
                language_output_factory=DeterministicLanguageOutput,
            ),
            settings=settings,
            language_config=language_config,
        )

    @classmethod
    def from_owner_inputs(
        cls,
        *,
        inputs: RuntimeOwnerInputs,
        settings: RuntimeSettings,
        language_config: LanguageRuntimeConfig | None = None,
    ) -> "RuntimeContainer":
        """Build and validate the complete graph before publishing a container."""
        world_state = inputs.world_proposal
        safety = inputs.safety_validator
        if world_state is None:
            from .errors import StartupErrorCategory, StartupFailure

            raise StartupFailure(
                StartupErrorCategory.WORLD_MISSING,
                "required world proposal port is unavailable",
            )
        if safety is None:
            from .errors import StartupErrorCategory, StartupFailure

            raise StartupFailure(
                StartupErrorCategory.SAFETY_MISSING,
                "required safety finalization service is unavailable",
            )

        config = (
            language_config
            or LanguageRuntimeConfig(
                settings.language_mode.value,
                (
                    str(settings.language_release_path)
                    if settings.language_release_path
                    else None
                ),
            )
        ).validated()
        root = Path(settings.durable_root)
        root.mkdir(parents=True, exist_ok=True)
        constructed: list[tuple[str, Any]] = []
        try:
            _claim_owner(constructed, "world_proposal", world_state)
            _claim_owner(constructed, "safety", safety)
            if config.mode == "transformer_proposal":
                if config.provider_factory is None:
                    raise RuntimeError(
                        "verified transformer release present but no safe provider factory is configured"
                    )
                from vulcan.local_language import build_verified_adapter

                language_input = build_verified_adapter(
                    release_root=config.release_path or "",
                    provider_factory=config.provider_factory,
                )
            else:
                language_input = inputs.language_proposal_factory()
            _claim_owner(constructed, "language_input", language_input)
            language_output: LanguageOutputPort = inputs.language_output_factory()
            _claim_owner(constructed, "language_output", language_output)
            audit = inputs.audit_factory(root / "audit" / "events.jsonl")
            _claim_owner(constructed, "audit", audit)
            memory = inputs.memory_factory(
                MemoryRuntimeConfig(
                    settings.memory_enabled,
                    settings.memory_sqlite_path,
                    settings.durable_root,
                    settings.replicas,
                    settings.memory_backend.value,
                ),
                audit=audit,
            )
            _claim_owner(constructed, "memory", memory)
            memory.readiness()
            alignment = inputs.alignment_factory(
                root / "alignment" / "active.json", audit=audit
            )
            _claim_owner(constructed, "alignment", alignment)
            domain_registry = inputs.domain_factory(root / "domains", audit=audit)
            _claim_owner(constructed, "domain_lookup", domain_registry)
            improvement_proposals = ImprovementProposalStore(
                root / "improvement-proposals"
            )
            _claim_owner(constructed, "improvement_proposals", improvement_proposals)
            learning_owner = inputs.learning_factory(
                capability=LearningCapabilityStatus.SHADOW,
                resources={
                    "legacy_continual_proposal": inputs.legacy_continual_proposal
                },
                shadow_bandit=ShadowLinUCBToolBandit(),
            )
            _claim_owner(constructed, "learning", learning_owner)
            learning_owner.readiness()
            response_safety = inputs.safety_port_factory(safety)
            _claim_owner(constructed, "response_safety", response_safety)
            response_safety.readiness()
            safety_finalizer = inputs.finalizer_factory(response_safety)
            _claim_owner(constructed, "safety_finalizer", safety_finalizer)
            delegate = CognitiveKernel(
                state_authority=world_state,
                finalizer=safety_finalizer,
                language_input=language_input,
                language_output=language_output,
                memory=memory,
                audit=audit,
                alignment=alignment,
                domain_lookup=domain_registry,
            )

            def domain_state():
                lease = domain_registry.lease()
                return (
                    lease.domain_snapshot_id,
                    {"snapshot_id": lease.domain_snapshot_id},
                    lease,
                )

            def alignment_state():
                lease = alignment.lease()
                return (
                    str(lease.revision),
                    {"policy_digest": lease.policy_digest},
                    lease,
                )

            def memory_state():
                revision, state = memory.snapshot_state()
                return revision, state, None

            capability_registry = load_capability_registry()
            arithmetic = capability_registry.records["cap.bounded_arithmetic"]

            def live_capability_facts():
                kernel_caps = tuple(delegate.capabilities())
                from .capabilities import release_evidence_root

                policy_digest = hashlib.sha256(
                    (
                        release_evidence_root()
                        / "docs"
                        / "architecture"
                        / "ami-invariants.yaml"
                    ).read_bytes()
                ).hexdigest()
                ports = composed_runtime_ports()
                return {
                    "cap.bounded_arithmetic": LiveOwnerFact(
                        owner=delegate.CAPABILITY_OWNER,
                        release_digest=delegate.CAPABILITY_RELEASE_DIGEST,
                        canonical_reachable=all(
                            port in ports for port in arithmetic.port_reachability
                        ),
                        mode=config.mode,
                        state_digest=Digest.of_bytes(
                            canonical_json(
                                {
                                    "kernel_type": f"{type(delegate).__module__}.{type(delegate).__qualname__}",
                                    "kernel_capabilities": kernel_caps,
                                    "language_mode": config.mode,
                                }
                            )
                        ).hex,
                        ready="bounded-arithmetic" in kernel_caps,
                        constitutionally_permitted=policy_digest
                        == arithmetic.active_policy_digest,
                    )
                }

            capability_authority = inputs.capability_manifest_factory(
                registry=capability_registry, live_facts=live_capability_facts
            )
            _claim_owner(constructed, "capability_authority", capability_authority)
            state_authorities = inputs.state_authority_factory(
                world=disabled_authority(
                    "world",
                    reason="legacy world model has no faithful constitutional state export",
                ),
                self_state=disabled_authority(
                    "self", reason="canonical self-state authority is not implemented"
                ),
                social=disabled_authority(
                    "social",
                    reason="canonical social-state authority is not implemented",
                ),
                normative=disabled_authority(
                    "normative",
                    reason="normative decisions are owned by alignment, not legacy world state",
                ),
                domain=ContentBoundStateAuthority(
                    kind="domain",
                    owner="domain-registry",
                    schema="vulcan-domain-snapshot.v1",
                    release="constitutional-v1",
                    read=domain_state,
                ),
                memory=ContentBoundStateAuthority(
                    kind="memory",
                    owner="governed-memory",
                    schema="vulcan-memory-snapshot.v1",
                    release="constitutional-v1",
                    read=memory_state,
                ),
                capability=capability_authority,
                csiu=DisabledCSIUPolicyAuthority(
                    reason="no serving-process CSIU policy owner is authorized"
                ),
                alignment=ContentBoundStateAuthority(
                    kind="alignment",
                    owner="alignment-registry",
                    schema="vulcan-alignment-snapshot.v1",
                    release="constitutional-v1",
                    read=alignment_state,
                ),
            )
            _claim_owner(constructed, "state_authorities", state_authorities)
            state_authorities.validate()

            def admit_snapshot(episode_id: str) -> SnapshotBundle:
                state_authorities.validate()
                from datetime import timedelta

                return construct_snapshot_bundle(
                    episode_id=episode_id,
                    providers=state_authorities.providers(),
                    lifetime=timedelta(
                        seconds=int(MAX_EPISODE_LIFETIME.total_seconds())
                    ),
                )

            startup_probe = admit_snapshot("startup-faithfulness-probe")
            startup_probe.close()
            episode_store = inputs.episode_store_factory(
                root / "episodes" / "episodes.sqlite3",
                outbox_sink=audit.append_episode_transition,
            )
            _claim_owner(constructed, "episode_store", episode_store)
            epistemic_store = inputs.epistemic_store_factory(
                root / "epistemic" / "epistemic.sqlite3",
                outbox_sink=audit.append_epistemic_commit,
            )
            _claim_owner(constructed, "epistemic_store", epistemic_store)
            lineage_store = inputs.lineage_store_factory(
                root / "episodes" / "episodes.sqlite3"
            )
            _claim_owner(constructed, "lineage_store", lineage_store)
            lineage_principal = Principal(
                PrincipalKind.SYSTEM_KERNEL,
                "constitutional-cognitive-kernel",
                hashlib.sha256(b"vulcan-constitutional-kernel-v1").hexdigest(),
            )
            lineage = LineageTransactionService(lineage_store, lineage_principal)
            branch_id = "branch-primary"
            instance_id = f"instance-{uuid4().hex}"
            try:
                existing_lineage = lineage_store.load(branch_id)
            except Exception as exc:
                from vulcan.microkernel.lineage import LineageError

                if (
                    not isinstance(exc, LineageError)
                    or str(exc) != "unknown lineage branch"
                ):
                    raise
                lineage.genesis("lineage-primary", branch_id, instance_id)
            else:
                if existing_lineage.suspended:
                    lineage.resume(branch_id, existing_lineage.digest, instance_id)
                else:
                    lineage.restart(branch_id, existing_lineage.digest, instance_id)
            delegate.disable_legacy_case_audit()
            kernel = ConstitutionalCognitiveKernel.from_kernel(
                delegate,
                snapshot_admitter=admit_snapshot,
                episode_store=episode_store,
                epistemic_store=epistemic_store,
                transaction_service_factory=inputs.transaction_service_factory,
                lineage=lineage,
                lineage_branch_id=branch_id,
            )
            _claim_owner(constructed, "transaction_service", kernel.transaction_service)
            _claim_owner(constructed, "kernel", kernel)
            close_order = tuple(name for name, _ in reversed(constructed))
            health = HealthStateMachine()
            health.admit()
            return cls(
                runtime_id=str(uuid4()),
                deployment=inputs.legacy_root,
                world_state=world_state,
                kernel=kernel,
                safety=safety,
                memory=memory,
                language_input=language_input,
                language_output=language_output,
                language_config=config,
                audit=audit,
                alignment=alignment,
                domain_registry=domain_registry,
                durable_root=root,
                improvement_proposals=improvement_proposals,
                learning_owner=learning_owner,
                settings=settings,
                health=health,
                episode_store=episode_store,
                epistemic_store=epistemic_store,
                lineage_store=lineage_store,
                state_authorities=state_authorities,
                capability_authority=capability_authority,
                transaction_service=kernel.transaction_service,
                ownership_close_order=close_order,
                response_safety=response_safety,
                safety_finalizer=safety_finalizer,
            )
        except BaseException as exc:
            _close_constructed(constructed, exc)
            raise


def _claim_owner(constructed: list[tuple[str, Any]], name: str, owner: Any) -> None:
    if owner is None:
        raise RuntimeError(f"required canonical {name} owner is unavailable")
    for prior_name, prior_owner in constructed:
        if prior_owner is owner:
            raise RuntimeError(f"duplicate ownership: {prior_name} and {name}")
    constructed.append((name, owner))


def _close_constructed(
    constructed: list[tuple[str, Any]], primary: BaseException
) -> None:
    """Close a partial graph completely without replacing its startup failure."""
    for name, owner in reversed(constructed):
        close = getattr(owner, "close", None) or getattr(owner, "shutdown", None)
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                _run_cleanup_awaitable(result)
        except BaseException as cleanup_error:
            primary.add_note(f"cleanup failed for {name}: {cleanup_error}")


def _run_cleanup_awaitable(awaitable: Any) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
        return
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(awaitable)
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
