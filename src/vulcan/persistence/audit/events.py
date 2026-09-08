"""Typed immutable audit event contracts for canonical audit v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

Digest = str
EventFamily = Literal["audit","episode","case","capability","domain","alignment","memory","csiu","learning","improvement","release","consent","relationship","runtime"]

HEX64 = re.compile(r"[0-9a-f]{64}")
SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
FORBIDDEN_DATA_KEYS = frozenset({"raw_prompt","prompt","authorization","jwt","token","secret","password","stack","exception_text","hidden_prompt","chain_of_thought"})

EVENT_FAMILY_BY_TYPE = {
    "audit.migration_boundary": "audit",
    "runtime.ready": "runtime",
    "episode.transitioned": "episode",
    "case.started": "case", "case.interpreted": "case", "case.plan_compiled": "case", "case.ledger_committed": "case", "case.alignment_decided": "case", "case.finalized": "case", "case.completed": "case", "case.abstained": "case", "case.blocked": "case", "case.finalization_error": "case", "case.cancelled": "case", "case.failed": "case",
    "capability.activation_prepared": "capability", "capability.activation_committed": "capability", "capability.activation_aborted": "capability",
    "domain.activation_prepared": "domain", "domain.activation_committed": "domain", "domain.activation_aborted": "domain",
    "alignment.activation_prepared": "alignment", "alignment.activation_committed": "alignment", "alignment.activation_aborted": "alignment",
    "memory.write_prepared": "memory", "memory.write_committed": "memory", "memory.write_aborted": "memory",
    "csiu.snapshot_validated": "csiu", "csiu.snapshot_rejected": "csiu", "csiu.decision_prepared": "csiu", "csiu.influence_applied": "csiu", "csiu.influence_blocked": "csiu", "csiu.decision_aborted": "csiu", "csiu.weight_proposed": "csiu", "csiu.alignment_proposed": "csiu", "csiu.kill_switch_changed": "csiu",
    "learning.update_prepared": "learning", "learning.update_aborted": "learning", "learning.update_committed": "learning", "learning.update_published": "learning", "learning.manual_recovery_required": "learning", "learning.policy_activation_prepared": "learning", "learning.policy_activation_committed": "learning", "learning.policy_activation_aborted": "learning",
    "improvement.proposed": "improvement", "improvement.approved": "improvement", "improvement.apply_prepared": "improvement", "improvement.candidate_installed": "improvement", "improvement.gate_completed": "improvement", "improvement.applied": "improvement", "improvement.aborted": "improvement", "improvement.rollback_completed": "improvement", "improvement.manual_recovery_required": "improvement",
    "release.prepared": "release", "release.committed": "release", "release.aborted": "release",
    "consent.granted": "consent", "consent.revoked": "consent",
    "relationship.created": "relationship", "relationship.updated": "relationship", "relationship.ended": "relationship",
}
TRANSACTION_FAMILIES = frozenset({"capability","domain","alignment","memory","csiu","learning","improvement","release"})
PREPARED_EVENTS = frozenset(t for t,f in EVENT_FAMILY_BY_TYPE.items() if f in TRANSACTION_FAMILIES and (t.endswith("_prepared") or t.endswith(".prepared")))
COMMIT_EVENTS = frozenset(t for t,f in EVENT_FAMILY_BY_TYPE.items() if f in TRANSACTION_FAMILIES and (t.endswith("_committed") or t.endswith(".committed") or t in {"csiu.influence_applied","improvement.applied","learning.update_published"}))
ABORT_EVENTS = frozenset(t for t,f in EVENT_FAMILY_BY_TYPE.items() if f in TRANSACTION_FAMILIES and (t.endswith("_aborted") or t.endswith(".aborted") or t.endswith(".manual_recovery_required")))
TERMINAL_CASE_EVENTS = frozenset({"case.completed","case.abstained","case.blocked","case.finalization_error","case.cancelled","case.failed"})
NONRECOVERABLE_CASE_START = "case.started"

@dataclass(frozen=True, slots=True)
class AuditEventSchema:
    family: EventFamily
    required: frozenset[str]
    digest_fields: frozenset[str] = frozenset()

SCHEMAS: dict[EventFamily, AuditEventSchema] = {
    "audit": AuditEventSchema("audit", frozenset({"legacy_schema_version","legacy_source_digest","legacy_events"}), frozenset({"legacy_source_digest"})),
    "runtime": AuditEventSchema("runtime", frozenset()),
    "episode": AuditEventSchema(
        "episode",
        frozenset(
            {
                "episode_id", "transition_digest", "transition", "from_state", "to_state",
                "prior_episode_digest", "resulting_episode_digest", "episode_digest",
                "snapshot_bundle_digest", "authority_references",
                "policy_references", "evidence_references", "transaction_id",
            }
        ),
        frozenset(
            {
                "transition_digest", "prior_episode_digest",
                "resulting_episode_digest", "episode_digest", "snapshot_bundle_digest",
            }
        ),
    ),
    "case": AuditEventSchema("case", frozenset({"case_id","request_digest"}), frozenset({"request_digest","response_ir_digest","rendered_text_digest","actor_digest"})),
    "capability": AuditEventSchema("capability", frozenset({"transaction_id","capability","actor_digest"}), frozenset({"actor_digest"})),
    "domain": AuditEventSchema("domain", frozenset({"transaction_id","domain","actor_digest"}), frozenset({"actor_digest"})),
    "alignment": AuditEventSchema("alignment", frozenset({"transaction_id","policy_id","actor_digest"}), frozenset({"actor_digest","policy_digest"})),
    "memory": AuditEventSchema("memory", frozenset({"transaction_id","record_id","actor_digest"}), frozenset({"actor_digest","record_digest"})),
    "csiu": AuditEventSchema("csiu", frozenset({"transaction_id","incident_id","actor_digest"}), frozenset({"actor_digest","decision_digest"})),
    "learning": AuditEventSchema("learning", frozenset({"transaction_id","policy_id","actor_digest"}), frozenset({"actor_digest","model_digest","policy_digest"})),
    "improvement": AuditEventSchema("improvement", frozenset({"transaction_id","proposal_digest","actor_digest"}), frozenset({"actor_digest","proposal_digest"})),
    "release": AuditEventSchema("release", frozenset({"transaction_id","release_id","actor_digest"}), frozenset({"actor_digest","release_digest"})),
    "consent": AuditEventSchema("consent", frozenset({"consent_id","actor_digest"}), frozenset({"actor_digest","subject_digest"})),
    "relationship": AuditEventSchema("relationship", frozenset({"relationship_id","actor_digest"}), frozenset({"actor_digest","subject_digest"})),
}

def require_digest(value: object, field: str) -> None:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"invalid {field}")

def require_safe_id(value: object, field: str) -> None:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {field}")

def validate_event_data(event_type: str, data: dict[str, object]) -> None:
    if any(k in data for k in FORBIDDEN_DATA_KEYS):
        raise ValueError("forbidden audit data")
    family = EVENT_FAMILY_BY_TYPE.get(event_type)
    if family is None:
        raise ValueError("invalid event type")
    schema = SCHEMAS[family]  # type: ignore[index]
    missing = schema.required.difference(data)
    if missing:
        raise ValueError(f"missing audit fields: {sorted(missing)}")
    for field in schema.required:
        if family == "audit" and field == "legacy_events":
            if type(data[field]) is not int or data[field] < 0:
                raise ValueError("invalid legacy_events")
            continue
        if family == "audit" and field == "legacy_schema_version":
            if data[field] != "vulcan-audit/1":
                raise ValueError("invalid legacy_schema_version")
            continue
        if family == "episode" and (field.endswith("references") or field == "transition"):
            continue
        if field.endswith("digest"):
            require_digest(data[field], field)
        else:
            require_safe_id(data[field], field)
    for field in schema.digest_fields.intersection(data):
        require_digest(data[field], field)
    if family == "episode":
        require_safe_id(data["episode_id"], "episode_id")
        require_safe_id(data["transaction_id"], "transaction_id")
        for field in ("from_state", "to_state"):
            require_safe_id(data[field], field)
        authorities = data["authority_references"]
        if not isinstance(authorities, list) or not all(
            isinstance(item, str) and SAFE_ID.fullmatch(item) for item in authorities
        ):
            raise ValueError("invalid authority_references")
        for field in ("policy_references", "evidence_references"):
            references = data[field]
            if not isinstance(references, list) or not all(
                isinstance(item, dict)
                and set(item) == {"artifact_id", "digest", "kind"}
                and isinstance(item["artifact_id"], str)
                and SAFE_ID.fullmatch(item["artifact_id"])
                and isinstance(item["kind"], str)
                and SAFE_ID.fullmatch(item["kind"])
                and isinstance(item["digest"], str)
                and HEX64.fullmatch(item["digest"])
                for item in references
            ):
                raise ValueError(f"invalid {field}")
    if "response_digest" in data or "rendered_response_digest" in data:
        raise ValueError("use response_ir_digest and rendered_text_digest")
