"""Evidence-driven capability maturity registry contracts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CapabilityStatus(str, Enum):
    ABSENT = "ABSENT"
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    EVALUATED = "EVALUATED"
    ADMITTED = "ADMITTED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


PUBLIC_STATUSES = frozenset({CapabilityStatus.ACTIVE, CapabilityStatus.DEGRADED})
DEPENDENCY_READY_STATUSES = frozenset({CapabilityStatus.ACTIVE, CapabilityStatus.DEGRADED, CapabilityStatus.ADMITTED})


@dataclass(frozen=True, slots=True)
class EvidenceArtifactRef:
    evidence_type: str
    artifact_uri: str
    artifact_sha256: str
    schema: str

    def validate(self, root: Path) -> None:
        if _SHA256_RE.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase full SHA-256 digest")
        if self.artifact_uri.startswith("file://"):
            rel = self.artifact_uri.removeprefix("file://")
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError("file evidence URI must be repository-relative")
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(f"evidence artifact missing: {rel}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != self.artifact_sha256:
                raise ValueError(f"digest mismatch for {rel}")
        elif not self.artifact_uri.startswith(("git://", "https://")):
            raise ValueError("artifact_uri must use file://, git://, or https://")


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    capability_id: str
    status: CapabilityStatus
    owner: str
    implementation_digest: str
    release_digest: str
    route_reachability: tuple[str, ...]
    port_reachability: tuple[str, ...]
    evaluation_artifact: EvidenceArtifactRef
    safety_artifacts: tuple[EvidenceArtifactRef, ...]
    impact_artifacts: tuple[EvidenceArtifactRef, ...]
    active_policy_digest: str
    rollback_method: str
    limitations: tuple[str, ...]
    review_date: date
    expires_at: datetime
    dependencies: tuple[str, ...]

    def validate_static(self) -> None:
        if not self.capability_id.startswith("cap."):
            raise ValueError("capability_id must start with cap.")
        if not self.owner:
            raise ValueError("owner is required")
        for field_name, digest in (("implementation_digest", self.implementation_digest), ("release_digest", self.release_digest), ("active_policy_digest", self.active_policy_digest)):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{field_name} must be a lowercase full SHA-256 digest")
        if self.expires_at.tzinfo != timezone.utc:
            raise ValueError("expires_at must be timezone-aware UTC")
        if not self.rollback_method:
            raise ValueError("rollback_method is required")
        if self.status in PUBLIC_STATUSES and not (self.route_reachability or self.port_reachability):
            raise ValueError("public capability requires route or port reachability")

    def validate_evidence(self, root: Path, now: datetime, composed_ports: set[str]) -> None:
        self.validate_static()
        if self.expires_at <= now:
            raise ValueError(f"capability evidence expired: {self.capability_id}")
        for port in self.port_reachability:
            if port not in composed_ports:
                raise ValueError(f"composed port is not present: {port}")
        self.evaluation_artifact.validate(root)
        for artifact in (*self.safety_artifacts, *self.impact_artifacts):
            artifact.validate(root)

    def public_projection(self) -> Mapping[str, object]:
        return MappingProxyType({
            "capability_id": self.capability_id,
            "status": self.status.value,
            "owner": self.owner,
            "limitations": list(self.limitations),
            "review_date": self.review_date.isoformat(),
            "route_reachability": list(self.route_reachability),
        })


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _artifact(data: Mapping[str, object]) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(str(data["evidence_type"]), str(data["artifact_uri"]), str(data["artifact_sha256"]), str(data["schema"]))


def _record(data: Mapping[str, object]) -> CapabilityRecord:
    return CapabilityRecord(
        capability_id=str(data["capability_id"]),
        status=CapabilityStatus(str(data["status"])),
        owner=str(data["owner"]),
        implementation_digest=str(data["implementation_digest"]),
        release_digest=str(data["release_digest"]),
        route_reachability=tuple(str(item) for item in data.get("route_reachability", [])),
        port_reachability=tuple(str(item) for item in data.get("port_reachability", [])),
        evaluation_artifact=_artifact(data["evaluation_artifact"]),
        safety_artifacts=tuple(_artifact(item) for item in data.get("safety_artifacts", [])),
        impact_artifacts=tuple(_artifact(item) for item in data.get("impact_artifacts", [])),
        active_policy_digest=str(data["active_policy_digest"]),
        rollback_method=str(data["rollback_method"]),
        limitations=tuple(str(item) for item in data.get("limitations", [])),
        review_date=date.fromisoformat(str(data["review_date"])),
        expires_at=datetime.fromisoformat(str(data["expires_at"]).replace("Z", "+00:00")),
        dependencies=tuple(str(item) for item in data.get("dependencies", [])),
    )


class CapabilityRegistry:
    def __init__(self, records: Sequence[CapabilityRecord], *, root: Path, now: datetime, composed_ports: set[str]) -> None:
        self.root = root
        self.now = now
        self.composed_ports = set(composed_ports)
        self.records = {record.capability_id: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate capability_id")
        for record in records:
            record.validate_evidence(root, now, self.composed_ports)
            for dep in record.dependencies:
                if dep not in self.records:
                    raise ValueError(f"missing dependency {dep}")
        self.effective_statuses = self._effective_statuses()

    @classmethod
    def from_json_text(cls, text: str, *, root: Path, now: datetime, composed_ports: set[str]) -> "CapabilityRegistry":
        data = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        records = [_record(item) for item in data["capabilities"]]
        return cls(records, root=root, now=now, composed_ports=composed_ports)

    def _effective_statuses(self) -> Mapping[str, CapabilityStatus]:
        resolved: dict[str, CapabilityStatus] = {}
        visiting: set[str] = set()

        def visit(capability_id: str) -> CapabilityStatus:
            if capability_id in resolved:
                return resolved[capability_id]
            if capability_id in visiting:
                raise ValueError("circular capability dependency")
            visiting.add(capability_id)
            record = self.records[capability_id]
            status = record.status
            for dep in record.dependencies:
                dep_status = visit(dep)
                if status in PUBLIC_STATUSES and dep_status not in DEPENDENCY_READY_STATUSES:
                    status = CapabilityStatus.SUSPENDED
                if dep_status is CapabilityStatus.SUSPENDED and status not in {CapabilityStatus.RETIRED, CapabilityStatus.ABSENT}:
                    status = CapabilityStatus.SUSPENDED
            visiting.remove(capability_id)
            resolved[capability_id] = status
            return status

        for capability_id in self.records:
            visit(capability_id)
        return MappingProxyType(resolved)

    def public_capabilities(self) -> list[Mapping[str, object]]:
        projected = []
        for capability_id in sorted(self.records):
            record = self.records[capability_id]
            effective = self.effective_statuses[capability_id]
            if effective in PUBLIC_STATUSES:
                view = dict(record.public_projection())
                view["status"] = effective.value
                projected.append(MappingProxyType(view))
        return projected

    def operator_capabilities(self) -> list[Mapping[str, object]]:
        return [MappingProxyType({"capability_id": cid, "status": rec.status.value, "effective_status": self.effective_statuses[cid].value, "owner": rec.owner, "dependencies": list(rec.dependencies)}) for cid, rec in sorted(self.records.items())]
