"""Reduced Phase-A serving graph for deterministic constitutional arithmetic."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from vulcan.constitution.primitives import Digest, canonical_json
from vulcan.graphix.runtime import DeterministicLanguageInput
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.lineage import (
    LineageError,
    LineageStore,
    LineageTransactionService,
)
from vulcan.microkernel.principals import Principal, PrincipalKind
from vulcan.microkernel.snapshots import MAX_EPISODE_LIFETIME, construct_snapshot_bundle
from vulcan.safety.response_adapter import EnhancedSafetyResponseAdapter

from .audit import CanonicalAudit
from .capabilities import (
    CapabilityManifestAuthority,
    LiveOwnerFact,
    composed_runtime_ports,
    load_capability_registry,
    release_evidence_root,
)
from .constitutional_kernel import ConstitutionalCognitiveKernel
from .finalization import SafetyResponseFinalizer
from .kernel import CognitiveKernel
from .output import DeterministicLanguageOutput
from .response_safety import CanonicalResponseSafetyValidator
from .settings import RuntimeSettings
from .state_authorities import (
    DisabledCSIUPolicyAuthority,
    StateAuthoritySet,
    disabled_authority,
)


@dataclass
class PhaseARuntime:
    runtime_id: str
    kernel: ConstitutionalCognitiveKernel
    audit: CanonicalAudit
    capability_authority: CapabilityManifestAuthority
    episode_store: EpisodeStore
    epistemic_store: EpistemicStore
    lineage_store: LineageStore
    closed: bool = False

    async def admission(self) -> None:
        if self.closed:
            raise RuntimeError("Phase-A runtime is closed")

    async def shallow_readiness(self) -> None:
        await self.admission()
        self.audit.readiness()

    async def deep_integrity(self) -> None:
        await self.admission()
        self.audit.deep_verify()
        self.episode_store.verify_all()
        self.epistemic_store.reconcile()
        self.lineage_store.verify_all()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for owner in (self.epistemic_store, self.episode_store, self.audit):
            owner.close()


def compose_phase_a_runtime(settings: RuntimeSettings) -> PhaseARuntime:
    reject_forbidden_settings(settings)
    with ExitStack() as cleanup:
        root = Path(settings.durable_root)
        audit = CanonicalAudit(root / "audit" / "events.jsonl")
        cleanup.callback(audit.close)
        episode_store = EpisodeStore(
            root / "episodes" / "episodes.sqlite3",
            outbox_sink=audit.append_episode_transition,
        )
        cleanup.callback(episode_store.close)
        epistemic_store = EpistemicStore(
            root / "epistemic" / "epistemic.sqlite3",
            outbox_sink=audit.append_epistemic_commit,
        )
        cleanup.callback(epistemic_store.close)
        lineage_store = LineageStore(root / "episodes" / "episodes.sqlite3")

        validator = CanonicalResponseSafetyValidator()
        response_safety = EnhancedSafetyResponseAdapter(validator)
        delegate = CognitiveKernel(
            state_authority=object(),
            finalizer=SafetyResponseFinalizer(response_safety),
            language_input=DeterministicLanguageInput(),
            language_output=DeterministicLanguageOutput(),
        )
        registry = load_capability_registry()
        arithmetic = registry.records["cap.bounded_arithmetic"]

        def live_facts():
            policy_digest = hashlib.sha256(
                (
                    release_evidence_root() / "docs/architecture/ami-invariants.yaml"
                ).read_bytes()
            ).hexdigest()
            return {
                "cap.bounded_arithmetic": LiveOwnerFact(
                    owner=delegate.CAPABILITY_OWNER,
                    release_digest=delegate.CAPABILITY_RELEASE_DIGEST,
                    canonical_reachable=all(
                        port in composed_runtime_ports()
                        for port in arithmetic.port_reachability
                    ),
                    mode="deterministic_only",
                    state_digest=Digest.of_bytes(
                        canonical_json({"kernel_capabilities": delegate.capabilities()})
                    ).hex,
                    ready="bounded-arithmetic" in delegate.capabilities(),
                    constitutionally_permitted=policy_digest
                    == arithmetic.active_policy_digest,
                )
            }

        capability = CapabilityManifestAuthority(
            registry=registry, live_facts=live_facts
        )
        disabled = lambda kind: disabled_authority(
            kind, reason="not admitted in the Phase-A serving graph"
        )
        authorities = StateAuthoritySet(
            world=disabled("world"),
            self_state=disabled("self"),
            social=disabled("social"),
            normative=disabled("normative"),
            domain=disabled("domain"),
            memory=disabled("memory"),
            capability=capability,
            csiu=DisabledCSIUPolicyAuthority(reason="not admitted in Phase A"),
            alignment=disabled("alignment"),
        )

        def admit_snapshot(episode_id: str):
            return construct_snapshot_bundle(
                episode_id=episode_id,
                providers=authorities.providers(),
                lifetime=timedelta(seconds=int(MAX_EPISODE_LIFETIME.total_seconds())),
            )

        principal = Principal(
            PrincipalKind.SYSTEM_KERNEL,
            "constitutional-cognitive-kernel",
            hashlib.sha256(b"vulcan-constitutional-kernel-v1").hexdigest(),
        )
        lineage = LineageTransactionService(lineage_store, principal)
        branch_id = "branch-primary"
        try:
            head = lineage_store.load(branch_id)
        except LineageError:
            lineage.genesis("lineage-primary", branch_id, f"instance-{uuid4().hex}")
        else:
            lineage.restart(branch_id, head.digest, f"instance-{uuid4().hex}")
        kernel = ConstitutionalCognitiveKernel.from_kernel(
            delegate,
            snapshot_admitter=admit_snapshot,
            episode_store=episode_store,
            epistemic_store=epistemic_store,
            lineage=lineage,
            lineage_branch_id=branch_id,
        )
        cleanup.pop_all()
        return PhaseARuntime(
            str(uuid4()),
            kernel,
            audit,
            capability,
            episode_store,
            epistemic_store,
            lineage_store,
        )


def reject_forbidden_settings(settings: RuntimeSettings) -> None:
    rejected = {
        "memory": settings.memory_enabled,
        "learning": settings.learning_enabled,
        "csiu": settings.csiu_enabled,
        "self_improvement": settings.self_improvement_enabled,
        "openai": settings.openai_enabled,
        "anthropic": settings.anthropic_enabled,
        "transformer": settings.language_mode.value != "deterministic_only",
    }
    enabled = sorted(name for name, value in rejected.items() if value)
    if enabled:
        raise ValueError(
            f"Phase-A profile rejects configured owners: {', '.join(enabled)}"
        )
