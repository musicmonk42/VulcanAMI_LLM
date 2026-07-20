"""Immutable assurance evidence records.

These contracts describe repository-owned assurance evidence only. They do not
claim certification, legal compliance, safety, erasure, or production readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_FIELD_RE = re.compile(r"(secret|token|password|credential|private[_-]?key|prompt)", re.IGNORECASE)
_ARTIFACT_URI_RE = re.compile(r"^(file|git|https)://[^\s]+$")
_SCHEMA_RE = re.compile(r"^ami-assurance-evidence/v[0-9]+$|^cyclonedx/[^\s]+$|^spdx/[^\s]+$|^slsa/[^\s]+$")


class EvidenceType(str, Enum):
    SOURCE_DIGEST = "source_digest"
    TEST_RESULT = "test_result"
    EVALUATION_REPORT = "evaluation_report"
    THREAT_MODEL_RESULT = "threat_model_result"
    IMPACT_ASSESSMENT = "impact_assessment"
    SIGNED_APPROVAL = "signed_approval"
    SBOM = "sbom"
    PROVENANCE_ATTESTATION = "provenance_attestation"
    MODEL_DATA_CARD = "model_data_card"
    INCIDENT_RECORD = "incident_record"
    ROLLBACK_PROOF = "rollback_proof"
    OPERATOR_REVIEW = "operator_review"


def sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parse_utc_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("timestamp must use UTC Z suffix")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError("timestamp must be UTC")
    return parsed


def _validate_sha256(value: str, field: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full SHA-256 digest")


def _validate_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for key, value in metadata.items():
        if _SECRET_FIELD_RE.search(key):
            raise ValueError(f"metadata field {key!r} may carry secrets")
        if not isinstance(value, str):
            raise TypeError("metadata values must be strings")
        if len(value) > 512:
            raise ValueError(f"metadata field {key!r} exceeds bounded length")
        copied[str(key)] = value
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: EvidenceType
    control_id: str
    artifact_uri: str
    artifact_sha256: str
    produced_at: datetime
    producer: str
    schema: str
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.evidence_type is not EvidenceType.TEST_RESULT and self.artifact_uri.startswith("file://tests/"):
            raise ValueError("test paths require a test_result evidence artifact")
        if not self.evidence_id.startswith("ev-"):
            raise ValueError("evidence_id must start with ev-")
        if not self.control_id.startswith("AMI-"):
            raise ValueError("control_id must start with AMI-")
        if _ARTIFACT_URI_RE.fullmatch(self.artifact_uri) is None:
            raise ValueError("artifact_uri must be an explicit file://, git://, or https:// URI")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if self.produced_at.tzinfo != timezone.utc:
            raise ValueError("produced_at must be timezone-aware UTC")
        if _SCHEMA_RE.fullmatch(self.schema) is None:
            raise ValueError("schema must be an exact assurance, CycloneDX, SPDX, or SLSA schema identifier")
        object.__setattr__(self, "metadata", _validate_metadata(self.metadata))

    @classmethod
    def from_json(cls, payload: str) -> "EvidenceRecord":
        pairs = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
        return cls(
            evidence_id=pairs["evidence_id"],
            evidence_type=EvidenceType(pairs["evidence_type"]),
            control_id=pairs["control_id"],
            artifact_uri=pairs["artifact_uri"],
            artifact_sha256=pairs["artifact_sha256"],
            produced_at=_parse_utc_timestamp(pairs["produced_at"]),
            producer=pairs["producer"],
            schema=pairs["schema"],
            metadata=pairs.get("metadata", {}),
        )

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_uri": self.artifact_uri,
            "control_id": self.control_id,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "metadata": dict(sorted(self.metadata.items())),
            "produced_at": self.produced_at.isoformat().replace("+00:00", "Z"),
            "producer": self.producer,
            "schema": self.schema,
        }

    def to_canonical_json(self) -> bytes:
        return canonical_json(self.to_canonical_dict())

    def record_sha256(self) -> str:
        return sha256_hexdigest(self.to_canonical_json())


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result
