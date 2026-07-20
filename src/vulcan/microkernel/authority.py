"""Authority lattice, operations, and audited authorization decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .principals import Principal, digest


class AuthorityLevel(str, Enum):
    UNTRUSTED_PROPOSAL = "untrusted_proposal"
    VALIDATED_CANDIDATE = "validated_candidate"
    COMMITTED_BELIEF = "committed_belief"
    AUTHORIZED_PLAN = "authorized_plan"
    EXECUTED_EFFECT = "executed_effect"


_AUTHORITY_ORDER = {
    AuthorityLevel.UNTRUSTED_PROPOSAL: 0,
    AuthorityLevel.VALIDATED_CANDIDATE: 1,
    AuthorityLevel.COMMITTED_BELIEF: 2,
    AuthorityLevel.AUTHORIZED_PLAN: 3,
    AuthorityLevel.EXECUTED_EFFECT: 4,
}


class Operation(str, Enum):
    PROPOSE = "proposal.create"
    READ = "authority.read"
    COMMIT_BELIEF = "belief.commit"
    AUTHORIZE_PLAN = "plan.authorize"
    EXECUTE_EFFECT = "effect.execute"
    MUTATE_MEMORY = "memory.mutate"
    ACTIVATE_POLICY = "policy.activate"


_OPERATION_MINIMUM = {
    Operation.PROPOSE: AuthorityLevel.UNTRUSTED_PROPOSAL,
    Operation.READ: AuthorityLevel.VALIDATED_CANDIDATE,
    Operation.COMMIT_BELIEF: AuthorityLevel.COMMITTED_BELIEF,
    Operation.AUTHORIZE_PLAN: AuthorityLevel.AUTHORIZED_PLAN,
    Operation.EXECUTE_EFFECT: AuthorityLevel.EXECUTED_EFFECT,
    Operation.MUTATE_MEMORY: AuthorityLevel.COMMITTED_BELIEF,
    Operation.ACTIVATE_POLICY: AuthorityLevel.AUTHORIZED_PLAN,
}

_HIGH_RISK = frozenset({Operation.COMMIT_BELIEF, Operation.AUTHORIZE_PLAN, Operation.EXECUTE_EFFECT, Operation.MUTATE_MEMORY, Operation.ACTIVATE_POLICY})


class AuthorityError(PermissionError):
    """Raised when authority validation fails closed."""


@dataclass(frozen=True)
class EvidenceRecord:
    validator_principal_digest: str
    validation_digest: str
    policy_digest: str
    observed_at: datetime
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("validator_principal_digest", "validation_digest", "policy_digest"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase sha256 hex digest")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_json(self) -> dict[str, object]:
        return {"evidence_refs": list(self.evidence_refs), "observed_at": self.observed_at.astimezone(timezone.utc).isoformat(), "policy_digest": self.policy_digest, "validation_digest": self.validation_digest, "validator_principal_digest": self.validator_principal_digest}


@dataclass(frozen=True)
class AuthorityGrant:
    principal_digest: str
    level: AuthorityLevel
    evidence_digest: str


@dataclass(frozen=True)
class AuditEvent:
    decision: str
    principal_digest: str
    operation: str
    authority_level: str
    episode_id: str
    resource_digest: str
    reason: str
    at: datetime

    def to_json(self) -> dict[str, str]:
        return {"at": self.at.astimezone(timezone.utc).isoformat(), "authority_level": self.authority_level, "decision": self.decision, "episode_id": self.episode_id, "operation": self.operation, "principal_digest": self.principal_digest, "reason": self.reason, "resource_digest": self.resource_digest}


@dataclass
class AuditSink:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def promote_authority(*, current: AuthorityLevel, target: AuthorityLevel, principal: Principal, evidence: EvidenceRecord) -> AuthorityGrant:
    if not principal.is_kernel:
        raise AuthorityError("only SYSTEM_KERNEL may promote authority")
    if _AUTHORITY_ORDER[target] < _AUTHORITY_ORDER[current]:
        raise AuthorityError("authority promotion must be monotonic")
    return AuthorityGrant(principal.identity_digest, target, digest(evidence.to_json()))


def operation_from_value(value: object) -> Operation:
    if isinstance(value, Operation):
        return value
    if isinstance(value, str):
        try:
            return Operation(value)
        except ValueError as exc:
            raise AuthorityError("unknown operation denied") from exc
    raise AuthorityError("operation must be an Operation")


def require_authority(*, principal: Principal, grant: AuthorityGrant, operation: Operation | str, episode_id: str, resource_digest: str, audit: AuditSink, clock) -> None:
    op = operation_from_value(operation)
    granted = grant.principal_digest == principal.identity_digest and _AUTHORITY_ORDER[grant.level] >= _AUTHORITY_ORDER[_OPERATION_MINIMUM[op]]
    if op in _HIGH_RISK:
        audit.record(AuditEvent("granted" if granted else "denied", principal.identity_digest, op.value, grant.level.value, episode_id, resource_digest, "capability_authorization", clock()))
    if not granted:
        raise AuthorityError("capability denied")
