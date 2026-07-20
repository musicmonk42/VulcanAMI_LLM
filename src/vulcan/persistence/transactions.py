"""Authoritative transaction protocol and reusable journal primitives.

This module defines a shared state machine without centralizing subsystem data
stores.  Each subsystem remains its durable authority while using these typed
contracts for prepare/persist/audit/publish/recover lifecycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Callable, Iterable, NewType, Protocol, Sequence

from vulcan.core.canonical import canonical_digest, canonical_json

TransactionId = NewType("TransactionId", str)
PrincipalDigest = NewType("PrincipalDigest", str)
TargetIdentity = NewType("TargetIdentity", str)
Revision = NewType("Revision", str)
Digest = NewType("Digest", str)
SubsystemName = NewType("SubsystemName", str)


class TransactionState(str, Enum):
    PREPARED = "prepared"
    PERSISTED = "persisted"
    AUDIT_COMMITTED = "audit_committed"
    PUBLISHED = "published"
    ABORTED = "aborted"
    STALE_CAS = "stale_cas"
    MANUAL_RECOVERY = "manual_recovery"


TERMINAL_STATES = frozenset({
    TransactionState.PUBLISHED,
    TransactionState.ABORTED,
    TransactionState.STALE_CAS,
    TransactionState.MANUAL_RECOVERY,
})

_ALLOWED_TRANSITIONS = {
    TransactionState.PREPARED: frozenset({TransactionState.PERSISTED, TransactionState.ABORTED, TransactionState.STALE_CAS, TransactionState.MANUAL_RECOVERY}),
    TransactionState.PERSISTED: frozenset({TransactionState.AUDIT_COMMITTED, TransactionState.MANUAL_RECOVERY}),
    TransactionState.AUDIT_COMMITTED: frozenset({TransactionState.PUBLISHED, TransactionState.MANUAL_RECOVERY}),
    TransactionState.PUBLISHED: frozenset(),
    TransactionState.ABORTED: frozenset(),
    TransactionState.STALE_CAS: frozenset(),
    TransactionState.MANUAL_RECOVERY: frozenset(),
}


class ResultCategory(str, Enum):
    OK = "ok"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    STALE_CAS = "stale_cas"
    ABORTED = "aborted"
    AMBIGUOUS_EFFECT = "ambiguous_effect"
    INVALID_TRANSITION = "invalid_transition"
    AUTHORITY_REJECTED = "authority_rejected"


class RecoveryAction(str, Enum):
    NONE = "none"
    ABORT_PREPARED = "abort_prepared"
    COMPLETE_AUDIT = "complete_audit"
    PUBLISH_COMMITTED = "publish_committed"
    MANUAL_RECOVERY = "manual_recovery"


class TransactionError(ValueError):
    def __init__(self, category: ResultCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class TransactionEvent:
    transaction_id: TransactionId
    subsystem: SubsystemName
    state: TransactionState
    actor_principal_digest: PrincipalDigest
    target_identity: TargetIdentity
    prior_revision: Revision | None
    prior_digest: Digest | None
    proposed_digest: Digest
    result_category: ResultCategory
    occurred_at: datetime

    def to_record(self) -> dict[str, object]:
        return {
            "transaction_id": str(self.transaction_id),
            "subsystem": str(self.subsystem),
            "state": self.state.value,
            "actor_principal_digest": str(self.actor_principal_digest),
            "target_identity": str(self.target_identity),
            "prior_revision": None if self.prior_revision is None else str(self.prior_revision),
            "prior_digest": None if self.prior_digest is None else str(self.prior_digest),
            "proposed_digest": str(self.proposed_digest),
            "result_category": self.result_category.value,
            "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_record())


@dataclass(frozen=True, slots=True)
class TransactionRecord:
    transaction_id: TransactionId
    subsystem: SubsystemName
    actor_principal_digest: PrincipalDigest
    target_identity: TargetIdentity
    prior_revision: Revision | None
    prior_digest: Digest | None
    proposed_digest: Digest
    state: TransactionState = TransactionState.PREPARED
    result_category: ResultCategory = ResultCategory.OK
    event_digests: tuple[str, ...] = field(default_factory=tuple)

    def transition(self, state: TransactionState, category: ResultCategory, event_digest: str) -> "TransactionRecord":
        if state not in _ALLOWED_TRANSITIONS[self.state]:
            raise TransactionError(ResultCategory.INVALID_TRANSITION, f"cannot transition {self.state.value} to {state.value}")
        return TransactionRecord(self.transaction_id, self.subsystem, self.actor_principal_digest, self.target_identity, self.prior_revision, self.prior_digest, self.proposed_digest, state, category, self.event_digests + (event_digest,))

    def recovery_action(self) -> RecoveryAction:
        if self.state is TransactionState.PREPARED:
            return RecoveryAction.ABORT_PREPARED
        if self.state is TransactionState.PERSISTED:
            return RecoveryAction.COMPLETE_AUDIT
        if self.state is TransactionState.AUDIT_COMMITTED:
            return RecoveryAction.PUBLISH_COMMITTED
        if self.state is TransactionState.MANUAL_RECOVERY:
            return RecoveryAction.MANUAL_RECOVERY
        return RecoveryAction.NONE


class Reconciler(Protocol):
    subsystem: SubsystemName
    def reconcile(self, records: Sequence[TransactionRecord]) -> Sequence[RecoveryAction]: ...


class ReconciliationRegistry:
    def __init__(self) -> None:
        self._items: dict[SubsystemName, Reconciler] = {}
    def register(self, reconciler: Reconciler) -> None:
        if reconciler.subsystem in self._items:
            raise TransactionError(ResultCategory.AUTHORITY_REJECTED, f"duplicate reconciler for {reconciler.subsystem}")
        self._items[reconciler.subsystem] = reconciler
    def reconcile(self, records: Iterable[TransactionRecord]) -> dict[SubsystemName, Sequence[RecoveryAction]]:
        grouped: dict[SubsystemName, list[TransactionRecord]] = {}
        for record in records:
            if record.recovery_action() is not RecoveryAction.NONE:
                grouped.setdefault(record.subsystem, []).append(record)
        return {name: self._items[name].reconcile(tuple(group)) for name, group in grouped.items() if name in self._items}


class JsonlTransactionJournal:
    """Append-only deterministic journal for transaction events."""

    def __init__(self, path: Path, *, clock: Clock = utc_now) -> None:
        self.path = path
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TransactionEvent) -> str:
        payload = event.to_record()
        encoded = canonical_json(payload).decode("utf-8")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return canonical_digest(payload)

    def load_events(self) -> tuple[TransactionEvent, ...]:
        if not self.path.exists():
            return ()
        events: list[TransactionEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                events.append(TransactionEvent(
                    transaction_id=TransactionId(raw["transaction_id"]), subsystem=SubsystemName(raw["subsystem"]), state=TransactionState(raw["state"]), actor_principal_digest=PrincipalDigest(raw["actor_principal_digest"]), target_identity=TargetIdentity(raw["target_identity"]), prior_revision=None if raw["prior_revision"] is None else Revision(raw["prior_revision"]), prior_digest=None if raw["prior_digest"] is None else Digest(raw["prior_digest"]), proposed_digest=Digest(raw["proposed_digest"]), result_category=ResultCategory(raw["result_category"]), occurred_at=datetime.fromisoformat(raw["occurred_at"].replace("Z", "+00:00"))))
        return tuple(events)


def validate_idempotent_replay(existing: TransactionRecord, proposed_digest: Digest) -> ResultCategory:
    if existing.proposed_digest != proposed_digest:
        raise TransactionError(ResultCategory.AUTHORITY_REJECTED, "transaction id replay has different proposed digest")
    return ResultCategory.IDEMPOTENT_REPLAY


def make_event(record: TransactionRecord, state: TransactionState, category: ResultCategory, clock: Clock = utc_now) -> TransactionEvent:
    return TransactionEvent(record.transaction_id, record.subsystem, state, record.actor_principal_digest, record.target_identity, record.prior_revision, record.prior_digest, record.proposed_digest, category, clock())
