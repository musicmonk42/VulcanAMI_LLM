from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vulcan.core.ownership import ResourceHandle, ResourceOwnership
from vulcan.persistence.transactions import (
    Digest,
    JsonlTransactionJournal,
    PrincipalDigest,
    ReconciliationRegistry,
    RecoveryAction,
    ResultCategory,
    Revision,
    SubsystemName,
    TargetIdentity,
    TransactionError,
    TransactionId,
    TransactionRecord,
    TransactionState,
    make_event,
    validate_idempotent_replay,
)


def fixed_clock():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def record(tx="tx-1", proposed="p1"):
    return TransactionRecord(
        transaction_id=TransactionId(tx),
        subsystem=SubsystemName("file+audit"),
        actor_principal_digest=PrincipalDigest("a" * 64),
        target_identity=TargetIdentity("memory:item:1"),
        prior_revision=Revision("7"),
        prior_digest=Digest("b" * 64),
        proposed_digest=Digest(proposed),
    )


def advance(rec, state, category=ResultCategory.OK):
    return rec.transition(state, category, f"digest:{state.value}")


def test_state_machine_allows_only_authoritative_sequence():
    rec = record()
    persisted = advance(rec, TransactionState.PERSISTED)
    audited = advance(persisted, TransactionState.AUDIT_COMMITTED)
    published = advance(audited, TransactionState.PUBLISHED)
    assert published.state is TransactionState.PUBLISHED
    with pytest.raises(TransactionError) as err:
        advance(published, TransactionState.ABORTED)
    assert err.value.category is ResultCategory.INVALID_TRANSITION


@pytest.mark.parametrize("terminal", [TransactionState.ABORTED, TransactionState.STALE_CAS, TransactionState.MANUAL_RECOVERY])
def test_prepared_terminal_outcomes_are_explicit(terminal):
    category = ResultCategory.STALE_CAS if terminal is TransactionState.STALE_CAS else ResultCategory.ABORTED
    if terminal is TransactionState.MANUAL_RECOVERY:
        category = ResultCategory.AMBIGUOUS_EFFECT
    rec = advance(record(), terminal, category)
    assert rec.state is terminal
    assert rec.recovery_action() in {RecoveryAction.NONE, RecoveryAction.MANUAL_RECOVERY}


def test_stale_cas_is_normal_terminal_outcome_not_manual_recovery():
    stale = advance(record(), TransactionState.STALE_CAS, ResultCategory.STALE_CAS)
    assert stale.result_category is ResultCategory.STALE_CAS
    assert stale.recovery_action() is RecoveryAction.NONE


def test_event_contains_required_boundary_fields_and_canonical_digest():
    rec = record()
    event = make_event(rec, TransactionState.PREPARED, ResultCategory.OK, fixed_clock)
    payload = event.to_record()
    assert set(payload) == {
        "transaction_id", "subsystem", "state", "actor_principal_digest", "target_identity",
        "prior_revision", "prior_digest", "proposed_digest", "result_category", "occurred_at",
    }
    assert len(event.digest) == 64


def test_jsonl_journal_replays_idempotently(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = JsonlTransactionJournal(path, clock=fixed_clock)
    event = make_event(record(), TransactionState.PREPARED, ResultCategory.OK, fixed_clock)
    first = journal.append(event)
    second = journal.append(event)
    assert first == second
    assert [loaded.digest for loaded in journal.load_events()] == [first, second]


def test_idempotency_rejects_digest_escalation():
    assert validate_idempotent_replay(record(proposed="same"), Digest("same")) is ResultCategory.IDEMPOTENT_REPLAY
    with pytest.raises(TransactionError) as err:
        validate_idempotent_replay(record(proposed="same"), Digest("evil"))
    assert err.value.category is ResultCategory.AUTHORITY_REJECTED


def test_reconciliation_registry_routes_ambiguous_and_prepared_work():
    class FileAuditReconciler:
        subsystem = SubsystemName("file+audit")
        def reconcile(self, records):
            return tuple(item.recovery_action() for item in records)

    registry = ReconciliationRegistry()
    registry.register(FileAuditReconciler())
    persisted = advance(record("tx-2"), TransactionState.PERSISTED)
    assert registry.reconcile([record(), persisted]) == {
        SubsystemName("file+audit"): (RecoveryAction.ABORT_PREPARED, RecoveryAction.COMPLETE_AUDIT)
    }


def test_registry_rejects_duplicate_authority():
    class Reconciler:
        subsystem = SubsystemName("sqlite+audit")
        def reconcile(self, records):
            return ()
    registry = ReconciliationRegistry()
    registry.register(Reconciler())
    with pytest.raises(TransactionError):
        registry.register(Reconciler())


def test_borrowed_resource_double_close_does_not_cascade():
    class Authority:
        def __init__(self):
            self.closed = 0
        def close(self):
            self.closed += 1
    authority = Authority()
    handle = ResourceHandle(authority)
    handle.close(); handle.close()
    assert authority.closed == 0
    owned = ResourceHandle(authority, ResourceOwnership.OWNED)
    owned.close(); owned.close()
    assert authority.closed == 2


@pytest.mark.parametrize("fail_state", [TransactionState.PERSISTED, TransactionState.AUDIT_COMMITTED, TransactionState.PUBLISHED])
def test_fixture_failpoints_recover_each_externally_observable_transition(fail_state):
    rec = record("tx-fail")
    if fail_state is TransactionState.PERSISTED:
        assert rec.recovery_action() is RecoveryAction.ABORT_PREPARED
        return
    rec = advance(rec, TransactionState.PERSISTED)
    if fail_state is TransactionState.AUDIT_COMMITTED:
        assert rec.recovery_action() is RecoveryAction.COMPLETE_AUDIT
        return
    rec = advance(rec, TransactionState.AUDIT_COMMITTED)
    assert rec.recovery_action() is RecoveryAction.PUBLISH_COMMITTED
