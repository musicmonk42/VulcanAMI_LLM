"""Live, constitutional capability-manifest authority.

Static evidence is an input to this owner, never an advertisement by itself.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Callable, Mapping

from vulcan.assurance.capabilities import CapabilityRegistry, CapabilityStatus
from vulcan.constitution.primitives import Digest, canonical_json
from vulcan.microkernel.snapshots import SnapshotMaterialRef

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PACKAGED_EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "_release_evidence"


def release_evidence_root() -> Path:
    """Locate immutable release evidence without assuming a source checkout."""
    configured = os.environ.get("VULCAN_RELEASE_EVIDENCE_ROOT")
    if configured:
        root = Path(configured)
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("invalid release evidence root")
        return root.resolve(strict=True)
    if _PACKAGED_EVIDENCE_ROOT.is_dir():
        return _PACKAGED_EVIDENCE_ROOT
    return _SOURCE_ROOT


class LiveCapabilityStatus(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    RESEARCH_ONLY = "research-only"
    UNAVAILABLE = "unavailable"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class LiveOwnerFact:
    owner: str
    release_digest: str
    canonical_reachable: bool
    mode: str
    state_digest: str
    ready: bool
    constitutionally_permitted: bool


@dataclass(frozen=True, slots=True)
class CapabilityAttestation:
    capability_id: str
    status: str
    static_status: str
    evidence_digest: str
    implementation_digest: str
    canonical_reachability: bool
    owner: str | None
    owner_release_digest: str | None
    configured_mode: str | None
    state_digest: str | None
    ready: bool
    constitutionally_permitted: bool
    limitations: tuple[str, ...]
    attestation_digest: str

    def public_projection(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "status": self.status,
            "owner": self.owner,
            "release_digest": self.owner_release_digest,
            "state_digest": self.state_digest,
            "attestation_digest": self.attestation_digest,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """One atomic observation used by both snapshots and HTTP projections."""

    digest: str
    attestations: tuple[CapabilityAttestation, ...]

    def public_capabilities(self) -> tuple[CapabilityAttestation, ...]:
        return tuple(
            item
            for item in self.attestations
            if item.status == LiveCapabilityStatus.CANONICAL.value
        )


def composed_runtime_ports() -> set[str]:
    from vulcan.runtime.route_manifest import (
        PHASE_A_ROUTE_REGISTRY,
        generate_route_manifest,
    )

    return {
        f"{item['method']} {item['path']}"
        for item in generate_route_manifest(registry=PHASE_A_ROUTE_REGISTRY)
    }


def load_capability_registry(now: datetime | None = None) -> CapabilityRegistry:
    root = release_evidence_root()
    return CapabilityRegistry.from_json_text(
        (root / "config" / "capabilities.yaml").read_text(encoding="utf-8"),
        root=root,
        now=now or datetime.now(timezone.utc),
        composed_ports=composed_runtime_ports(),
    )


class CapabilityManifestAuthority:
    """Sole owner of digest-bound live capability truth and its snapshot port."""

    kind = "capability"
    schema = "vulcan-capability-attestation.v1"
    owner = "CapabilityManifestAuthority"
    release = "constitutional-v1"

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        live_facts: Callable[[], Mapping[str, LiveOwnerFact]],
    ):
        self._registry, self._live_facts, self._lock = registry, live_facts, RLock()

    @staticmethod
    def _status(
        static: CapabilityStatus,
        fact: LiveOwnerFact | None,
        expected_owner: str,
        expected_release: str,
    ) -> LiveCapabilityStatus:
        if static in {
            CapabilityStatus.ABSENT,
            CapabilityStatus.RETIRED,
            CapabilityStatus.SUSPENDED,
        }:
            return LiveCapabilityStatus.DISABLED
        if static is CapabilityStatus.SHADOW:
            return LiveCapabilityStatus.SHADOW
        if static in {
            CapabilityStatus.RESEARCH,
            CapabilityStatus.EVALUATED,
            CapabilityStatus.ADMITTED,
        }:
            return LiveCapabilityStatus.RESEARCH_ONLY
        if (
            fact is None
            or not fact.ready
            or fact.owner != expected_owner
            or fact.release_digest != expected_release
        ):
            return LiveCapabilityStatus.UNAVAILABLE
        if (
            not fact.canonical_reachable
            or fact.mode == "disabled"
            or not fact.constitutionally_permitted
        ):
            return LiveCapabilityStatus.DISABLED
        return LiveCapabilityStatus.CANONICAL

    @staticmethod
    def _validate_fact(capability_id: str, fact: LiveOwnerFact) -> None:
        if not fact.owner or not fact.mode:
            raise ValueError(f"invalid live capability identity: {capability_id}")
        for name, value in (
            ("release_digest", fact.release_digest),
            ("state_digest", fact.state_digest),
        ):
            try:
                Digest.from_legacy_hex(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid live capability {name}: {capability_id}"
                ) from exc
        if not all(
            isinstance(value, bool)
            for value in (
                fact.canonical_reachable,
                fact.ready,
                fact.constitutionally_permitted,
            )
        ):
            raise TypeError(f"invalid live capability booleans: {capability_id}")

    def snapshot(self) -> CapabilitySnapshot:
        with self._lock:
            facts = dict(self._live_facts())
            unknown = set(facts) - set(self._registry.records)
            if unknown:
                raise ValueError(f"unknown live capability facts: {sorted(unknown)!r}")
            for capability_id, fact in facts.items():
                if not isinstance(fact, LiveOwnerFact):
                    raise TypeError(f"invalid live capability fact: {capability_id}")
                self._validate_fact(capability_id, fact)
            result = []
            for capability_id, record in sorted(self._registry.records.items()):
                fact = facts.get(capability_id)
                status = self._status(
                    self._registry.effective_statuses[capability_id],
                    fact,
                    record.owner,
                    record.release_digest,
                )
                static_evidence = {
                    "capability_id": capability_id,
                    "owner": record.owner,
                    "release_digest": record.release_digest,
                    "implementation_digest": record.implementation_digest,
                    "active_policy_digest": record.active_policy_digest,
                    "routes": record.route_reachability,
                    "ports": record.port_reachability,
                    "evaluation": asdict(record.evaluation_artifact),
                    "safety": [asdict(item) for item in record.safety_artifacts],
                    "impact": [asdict(item) for item in record.impact_artifacts],
                }
                evidence_digest = Digest.of_bytes(canonical_json(static_evidence)).hex
                body = {
                    "capability_id": capability_id,
                    "status": status.value,
                    "static_status": self._registry.effective_statuses[
                        capability_id
                    ].value,
                    "evidence_digest": evidence_digest,
                    "implementation_digest": record.implementation_digest,
                    "canonical_reachability": bool(fact and fact.canonical_reachable),
                    "owner": None if fact is None else fact.owner,
                    "owner_release_digest": (
                        None if fact is None else fact.release_digest
                    ),
                    "configured_mode": None if fact is None else fact.mode,
                    "state_digest": None if fact is None else fact.state_digest,
                    "ready": bool(fact and fact.ready),
                    "constitutionally_permitted": bool(
                        fact and fact.constitutionally_permitted
                    ),
                    "limitations": record.limitations,
                }
                result.append(
                    CapabilityAttestation(
                        **body,
                        attestation_digest=Digest.of_bytes(canonical_json(body)).hex,
                    )
                )
            attestations = tuple(result)
            return CapabilitySnapshot(
                Digest.of_bytes(
                    canonical_json([asdict(item) for item in attestations])
                ).hex,
                attestations,
            )

    def attestations(self) -> tuple[CapabilityAttestation, ...]:
        return self.snapshot().attestations

    def public_capabilities(self) -> tuple[CapabilityAttestation, ...]:
        return self.snapshot().public_capabilities()

    def state_digest(self) -> str:
        return self.snapshot().digest

    def lease_snapshot(
        self, *, kind: str, episode_id: str, acquired_at: datetime, expires_at: datetime
    ):
        if kind != self.kind:
            raise RuntimeError("capability authority cannot serve another state kind")
        snapshot = self.snapshot()
        digest = snapshot.digest
        return (
            SnapshotMaterialRef(
                kind,
                digest,
                self.schema,
                self.owner,
                digest,
                acquired_at,
                acquired_at,
                expires_at,
                f"capability-lease:{digest[:32]}:{episode_id[-32:]}",
                canonical_json([asdict(item) for item in snapshot.attestations]),
            ),
            None,
        )


def public_capability_response(
    authority: CapabilityManifestAuthority,
) -> dict[str, object]:
    snapshot = authority.snapshot()
    return {
        "snapshot_digest": snapshot.digest,
        "capabilities": [
            item.public_projection() for item in snapshot.public_capabilities()
        ],
    }
