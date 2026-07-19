"""Canonical evidence-bound learning observation contract.

Dependency-light, immutable schema for observations accepted by LearningOwner.
No training, persistence, or outbox delivery is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping
from uuid import uuid4

SCHEMA_VERSION = "vulcan-learning-observation/1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^obs-[0-9a-f]{32}$")
_CASE_RE = re.compile(r"^case-[A-Za-z0-9_.:-]{1,120}$")
_MAX_SKEW = timedelta(seconds=5)
_MAX_AGE = timedelta(minutes=15)
_TOOL_REGISTRY = frozenset({"graphix_arithmetic", "graphix_retrieval", "graphix_symbolic"})
_SECRET_RE = re.compile(r"(?i)(bearer\s+[a-z0-9._~+/=-]+|api[_-]?key|secret|password|prompt|answer)")


class TerminalStatus(Enum):
    VALIDATED_SUCCESS = "validated_success"
    VALIDATED_FAILURE = "validated_failure"
    SAFETY_FILTERED = "safety_filtered"
    LEDGER_REJECTED = "ledger_rejected"


class ProvenanceType(Enum):
    DERIVATION = "derivation"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
    SYSTEM_AUDIT = "system_audit"


class EligibilityStatus(Enum):
    ELIGIBLE_POSITIVE = "eligible_positive"
    NOT_ACCEPTED = "not_accepted"


@dataclass(frozen=True)
class EligibilityResult:
    status: EligibilityStatus
    reason: str


@dataclass(frozen=True)
class ObservationContext:
    case_id: str
    case_digest: str
    request_digest: str
    tenant_digest: str
    alignment_revision: int
    alignment_digest: str
    csiu_policy_digest: str
    csiu_snapshot_digest: str
    domain_snapshot_digest: str | None
    runtime_owner_id: str
    acquisition_time: datetime


@dataclass(frozen=True)
class LearningObservation:
    schema_version: str
    observation_id: str
    case_id: str
    case_digest: str
    request_digest: str
    tenant_digest: str
    selected_plan_digest: str
    selected_tool_id: str
    selection_distribution_digest: str
    action_propensity: float
    terminal_status: TerminalStatus
    ledger_digest: str
    evidence_digest: str
    alignment_revision: int
    alignment_digest: str
    csiu_policy_digest: str
    csiu_snapshot_digest: str
    domain_snapshot_digest: str | None
    acquisition_time_utc: str
    provenance_type: ProvenanceType
    runtime_owner_id: str
    schema_version_digest: str
    canonical_observation_digest: str

    def canonical_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "case_id": self.case_id,
            "case_digest": self.case_digest,
            "request_digest": self.request_digest,
            "tenant_digest": self.tenant_digest,
            "selected_plan_digest": self.selected_plan_digest,
            "selected_tool_id": self.selected_tool_id,
            "selection_distribution_digest": self.selection_distribution_digest,
            "action_propensity": self.action_propensity,
            "terminal_status": self.terminal_status.value,
            "ledger_digest": self.ledger_digest,
            "evidence_digest": self.evidence_digest,
            "alignment_revision": self.alignment_revision,
            "alignment_digest": self.alignment_digest,
            "csiu_policy_digest": self.csiu_policy_digest,
            "csiu_snapshot_digest": self.csiu_snapshot_digest,
            "domain_snapshot_digest": self.domain_snapshot_digest,
            "acquisition_time_utc": self.acquisition_time_utc,
            "provenance_type": self.provenance_type.value,
            "runtime_owner_id": self.runtime_owner_id,
            "schema_version_digest": self.schema_version_digest,
        }
        if include_digest:
            payload["canonical_observation_digest"] = self.canonical_observation_digest
        return payload

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_payload())


SCHEMA_VERSION_DIGEST = hashlib.sha256(SCHEMA_VERSION.encode("utf-8")).hexdigest()
_ALLOWED_FIELDS = frozenset(LearningObservation.__dataclass_fields__)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _require_hex(name: str, value: str | None, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be lowercase 64-hex")


def _require_clean_string(name: str, value: str, max_len: int = 160) -> None:
    if not isinstance(value, str) or not (1 <= len(value) <= max_len):
        raise ValueError(f"{name} is out of bounds")
    if _SECRET_RE.search(value):
        raise ValueError(f"{name} contains forbidden raw content")


def _validate_propensity(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("action_propensity must be numeric")
    f = float(value)
    if not math.isfinite(f) or f <= 0.0 or f > 1.0:
        raise ValueError("action_propensity must be in (0, 1]")
    if f == 0.0 or (f == 0.0 and math.copysign(1.0, f) < 0):
        raise ValueError("negative zero propensity is not canonical")


def _format_utc(dt: datetime) -> str:
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError("acquisition timestamp must be UTC")
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z") or "." not in value:
        raise ValueError("timestamp must be UTC with microsecond precision")
    dt = datetime.fromisoformat(value[:-1] + "+00:00")
    if dt.microsecond < 0:
        raise ValueError("timestamp must include microseconds")
    return dt


def construct_observation(
    *,
    context: ObservationContext,
    selected_plan_digest: str,
    selected_tool_id: str,
    selection_distribution_digest: str,
    action_propensity: float,
    terminal_status: TerminalStatus,
    ledger_digest: str,
    evidence_digest: str,
    provenance_type: ProvenanceType,
    terminal_case_validated: bool,
    ledger_validated: bool,
    evidence_integrity_validated: bool,
    bindings_match: bool,
    alignment_matches_lease: bool,
    csiu_bindings_valid: bool,
    clock: Callable[[], datetime] | None = None,
) -> tuple[LearningObservation, EligibilityResult]:
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    _validate_timestamp(context.acquisition_time, now)
    if context.acquisition_time > now + _MAX_SKEW:
        raise ValueError("acquisition timestamp is in the future")
    for name in ("case_digest", "request_digest", "tenant_digest", "alignment_digest", "csiu_policy_digest", "csiu_snapshot_digest"):
        _require_hex(name, getattr(context, name))
    _require_hex("domain_snapshot_digest", context.domain_snapshot_digest, allow_none=True)
    _require_clean_string("case_id", context.case_id)
    if not _CASE_RE.fullmatch(context.case_id):
        raise ValueError("case_id is not canonical")
    _require_clean_string("runtime_owner_id", context.runtime_owner_id)
    _require_hex("selected_plan_digest", selected_plan_digest)
    _require_hex("selection_distribution_digest", selection_distribution_digest)
    _require_hex("ledger_digest", ledger_digest)
    _require_hex("evidence_digest", evidence_digest)
    if selected_tool_id not in _TOOL_REGISTRY:
        raise ValueError("unknown selected tool")
    _validate_propensity(action_propensity)
    if not isinstance(terminal_status, TerminalStatus) or not isinstance(provenance_type, ProvenanceType):
        raise ValueError("closed enum value required")

    base = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": f"obs-{uuid4().hex}",
        "case_id": context.case_id,
        "case_digest": context.case_digest,
        "request_digest": context.request_digest,
        "tenant_digest": context.tenant_digest,
        "selected_plan_digest": selected_plan_digest,
        "selected_tool_id": selected_tool_id,
        "selection_distribution_digest": selection_distribution_digest,
        "action_propensity": float(action_propensity),
        "terminal_status": terminal_status.value,
        "ledger_digest": ledger_digest,
        "evidence_digest": evidence_digest,
        "alignment_revision": int(context.alignment_revision),
        "alignment_digest": context.alignment_digest,
        "csiu_policy_digest": context.csiu_policy_digest,
        "csiu_snapshot_digest": context.csiu_snapshot_digest,
        "domain_snapshot_digest": context.domain_snapshot_digest,
        "acquisition_time_utc": _format_utc(context.acquisition_time),
        "provenance_type": provenance_type.value,
        "runtime_owner_id": context.runtime_owner_id,
        "schema_version_digest": SCHEMA_VERSION_DIGEST,
    }
    canonical_digest = digest_json(base)
    record = dict(base)
    record["terminal_status"] = terminal_status
    record["provenance_type"] = provenance_type
    obs = LearningObservation(canonical_observation_digest=canonical_digest, **record)
    return obs, evaluate_positive_eligibility(
        obs,
        terminal_case_validated=terminal_case_validated,
        ledger_validated=ledger_validated,
        evidence_integrity_validated=evidence_integrity_validated,
        bindings_match=bindings_match,
        alignment_matches_lease=alignment_matches_lease,
        csiu_bindings_valid=csiu_bindings_valid,
        clock=lambda: now,
    )


def evaluate_positive_eligibility(
    observation: LearningObservation,
    *,
    terminal_case_validated: bool,
    ledger_validated: bool,
    evidence_integrity_validated: bool,
    bindings_match: bool,
    alignment_matches_lease: bool,
    csiu_bindings_valid: bool,
    seen_observation_ids: set[str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> EligibilityResult:
    try:
        validate_observation(observation, seen_observation_ids=seen_observation_ids, clock=clock)
    except ValueError as exc:
        return EligibilityResult(EligibilityStatus.NOT_ACCEPTED, str(exc))
    checks = [
        (terminal_case_validated, "terminal case validation failed"),
        (observation.terminal_status is TerminalStatus.VALIDATED_SUCCESS, "terminal status is not eligible"),
        (ledger_validated, "ledger validation failed"),
        (evidence_integrity_validated, "evidence integrity failed"),
        (bindings_match, "case/request/tenant binding mismatch"),
        (alignment_matches_lease, "alignment snapshot mismatch"),
        (csiu_bindings_valid, "csiu bindings invalid"),
    ]
    for ok, reason in checks:
        if not ok:
            return EligibilityResult(EligibilityStatus.NOT_ACCEPTED, reason)
    return EligibilityResult(EligibilityStatus.ELIGIBLE_POSITIVE, "eligible")


def validate_observation(
    observation: LearningObservation,
    *,
    seen_observation_ids: set[str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> None:
    if not isinstance(observation, LearningObservation):
        raise ValueError("learning owner accepts only LearningObservation")
    if observation.schema_version != SCHEMA_VERSION or observation.schema_version_digest != SCHEMA_VERSION_DIGEST:
        raise ValueError("unknown schema version")
    if not _ID_RE.fullmatch(observation.observation_id):
        raise ValueError("invalid observation id")
    if seen_observation_ids is not None:
        if observation.observation_id in seen_observation_ids:
            raise ValueError("duplicate observation id")
        seen_observation_ids.add(observation.observation_id)
    for name in ("case_digest", "request_digest", "tenant_digest", "selected_plan_digest", "selection_distribution_digest", "ledger_digest", "evidence_digest", "alignment_digest", "csiu_policy_digest", "csiu_snapshot_digest"):
        _require_hex(name, getattr(observation, name))
    _require_hex("domain_snapshot_digest", observation.domain_snapshot_digest, allow_none=True)
    if observation.selected_tool_id not in _TOOL_REGISTRY:
        raise ValueError("unknown selected tool")
    _validate_propensity(observation.action_propensity)
    if not isinstance(observation.terminal_status, TerminalStatus) or not isinstance(observation.provenance_type, ProvenanceType):
        raise ValueError("invalid enum")
    _validate_timestamp(_parse_utc(observation.acquisition_time_utc), (clock or (lambda: datetime.now(timezone.utc)))())
    expected = digest_json(observation.canonical_payload(include_digest=False))
    if observation.canonical_observation_digest != expected:
        raise ValueError("canonical observation digest mismatch")


def _validate_timestamp(acquired: datetime, now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("clock must return UTC")
    if acquired.tzinfo is None or acquired.utcoffset() != timedelta(0):
        raise ValueError("acquisition timestamp must be UTC")
    if acquired > now + _MAX_SKEW:
        raise ValueError("future observation")
    if now - acquired > _MAX_AGE:
        raise ValueError("stale observation")


def observation_from_canonical_json(raw: bytes | str, *, clock: Callable[[], datetime] | None = None) -> LearningObservation:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    data = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    if set(data) != _ALLOWED_FIELDS:
        raise ValueError("unknown or missing observation field")
    enum_data = dict(data)
    enum_data["terminal_status"] = TerminalStatus(enum_data["terminal_status"])
    enum_data["provenance_type"] = ProvenanceType(enum_data["provenance_type"])
    obs = LearningObservation(**enum_data)
    if obs.canonical_json() != text:
        raise ValueError("noncanonical JSON")
    validate_observation(obs, clock=clock)
    return obs
