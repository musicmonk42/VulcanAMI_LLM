from __future__ import annotations

import json
from datetime import datetime, timezone

from vulcan.microkernel.constitutional_journal import (
    ConstitutionalDatabase,
    ConstitutionalJournal,
    JournalEvent,
)
from vulcan.microkernel.episode import ActorBinding
from vulcan.persistence.journal_audit import JournalAuditProjector


def test_projection_is_disposable_deterministic_and_receipt_ordered(tmp_path):
    database = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    actor = ActorBinding._from_verified_identity(
        tenant="tenant-a", issuer="issuer-a", subject="operator"
    )
    with database.transaction() as uow:
        actor_digest = journal.bind_actor(uow, actor)
        uow.emit(
            JournalEvent(
                "episode.transitioned",
                actor_digest,
                "c" * 64,
                {"episode_id": "episode-1", "episode_digest": "d" * 64},
                datetime(2026, 9, 10, tzinfo=timezone.utc),
            )
        )
    path = tmp_path / "audit" / "events.jsonl"
    projector = JournalAuditProjector(database, path)
    first = projector.rebuild()
    second = projector.rebuild()
    assert first == second == path.read_bytes()
    document = json.loads(first)
    assert (document["commit_seq"], document["event_ordinal"]) == (1, 0)
    assert document["receipt_digest"] != document["previous_receipt_digest"]
    events = projector.events_for_episode("episode-1")
    assert len(events) == 1 and events[0].event_hash == document["receipt_digest"]

    path.unlink()
    projector.readiness()
    projector.deep_verify()
    assert not path.exists()
    assert projector.rebuild() == first
    projector.close()
    database.close()

    reopened = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    cold = JournalAuditProjector(reopened, path)
    assert cold.rebuild() == first
    cold.close()
    reopened.close()


def test_projection_delivery_failure_does_not_change_readiness(tmp_path, monkeypatch):
    database = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    projector = JournalAuditProjector(database, tmp_path / "audit.jsonl")
    monkeypatch.setattr(
        "os.replace", lambda *_args: (_ for _ in ()).throw(OSError("down"))
    )
    try:
        try:
            projector.rebuild()
        except OSError as exc:
            assert str(exc) == "down"
        projector.readiness()
    finally:
        projector.close()
        database.close()
