from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vulcan.constitution.primitives import canonical_json
from vulcan.microkernel.constitutional_journal import (
    EXPECTED_SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    ConstitutionalDatabase,
    ConstitutionalJournal,
    JournalError,
    JournalEvent,
    NestedUnitOfWorkError,
    SuccessorError,
)
from vulcan.microkernel.episode import ActorBinding

D = "d" * 64
C = "c" * 64
NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)
ACTOR = ActorBinding._from_verified_identity(
    tenant="tenant-a", issuer="issuer-a", subject="alice"
)


def event(actor_digest: str, label: str = "commit") -> JournalEvent:
    return JournalEvent("journal.commit", actor_digest, C, {"label": label}, NOW)


def genesis(journal: ConstitutionalJournal, uow, suffix: str = "1"):
    actor_digest = journal.bind_actor(uow, ACTOR)
    assert (
        journal.bind_command(
            uow,
            command_id=f"command-{suffix}",
            actor_digest=actor_digest,
            credential_provenance_digest=C,
            request_id=f"request-{suffix}",
            request_digest=(suffix[-1] if suffix[-1] in "0123456789abcdef" else "a")
            * 64,
            idempotency_key=f"idem-{suffix}",
        )
        == "created"
    )
    context = journal.put_artifact(
        uow, kind="admitted-context.v1", content=f"context-{suffix}".encode()
    )
    journal.create_episode(
        uow,
        episode_id=f"episode-{suffix}",
        command_id=f"command-{suffix}",
        actor_digest=actor_digest,
        context_digest=context,
    )
    return actor_digest


def test_multi_repository_commit_and_rollback(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    with db.transaction() as uow:
        actor = genesis(journal, uow)
        journal.append_transition(
            uow,
            episode_id="episode-1",
            from_state="perceived",
            to_state="interpreted",
            transition_digest="1" * 64,
            actor_digest=actor,
            credential_provenance_digest=C,
        )
        journal.append_transition(
            uow,
            episode_id="episode-1",
            from_state="interpreted",
            to_state="failed",
            transition_digest="6" * 64,
            actor_digest=actor,
            credential_provenance_digest=C,
        )
        epistemic_artifact = journal.put_artifact(
            uow, kind="epistemic-commit.v1", content=b"epistemic"
        )
        journal.commit_epistemic(
            uow,
            episode_id="episode-1",
            epistemic_digest="2" * 64,
            artifact_digest=epistemic_artifact,
            prior_digest=None,
            actor_digest=actor,
            credential_provenance_digest=C,
        )
        journal.advance_lineage(
            uow,
            branch_id="branch-main",
            episode_id="episode-1",
            expected_head_digest="0" * 64,
            new_head_digest="3" * 64,
        )
        result = journal.put_artifact(uow, kind="terminal-result.v1", content=b"result")
        journal.record_terminal(
            uow,
            episode_id="episode-1",
            result_digest=result,
            actor_digest=actor,
            credential_provenance_digest=C,
            terminal_state="failed",
        )
        uow.emit(event(actor))

    with pytest.raises(RuntimeError, match="abort"):
        with db.transaction() as uow:
            actor = genesis(journal, uow, "2")
            uow.emit(event(actor, "rolled-back"))
            raise RuntimeError("abort")

    with db.transaction() as uow:
        assert uow.query("SELECT count(*) FROM episodes")[0][0] == 1
        assert uow.query("SELECT count(*) FROM terminal_results")[0][0] == 1
        actor = journal.bind_actor(uow, ACTOR)
        uow.emit(event(actor, "inspection"))
    db.close()


def test_nested_hidden_commit_empty_commit_and_dangling_reference_fail(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    with pytest.raises(JournalError, match="requires an outbox"):
        with db.transaction():
            pass
    with db.transaction() as uow:
        with pytest.raises(NestedUnitOfWorkError):
            with db.transaction():
                pass
        for statement in (
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT hidden",
            "PRAGMA foreign_keys=OFF",
        ):
            with pytest.raises(JournalError, match="cannot control"):
                uow._execute(statement)
        with pytest.raises(JournalError, match="read-only"):
            uow.query("DELETE FROM actors")
        with pytest.raises(sqlite3.IntegrityError):
            uow._execute(
                "INSERT INTO journal_commits VALUES (?,?)",
                (uow.commit_seq, "vulcan-constitutional-journal/1"),
            )
        actor = ConstitutionalJournal.bind_actor(uow, ACTOR)
        with pytest.raises(sqlite3.IntegrityError):
            uow._execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?,-1)",
                ("dangling", "absent", actor, D, "perceived"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="invalid episode successor"):
            uow._execute(
                "INSERT INTO episode_transitions VALUES (?,?,?,?,?,?,?,?)",
                (
                    "missing",
                    9,
                    "perceived",
                    "failed",
                    "7" * 64,
                    actor,
                    C,
                    uow.commit_seq,
                ),
            )
        uow.emit(event(actor))
        with pytest.raises(sqlite3.IntegrityError):
            uow._execute(
                "INSERT INTO transactional_outbox VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    uow.commit_seq,
                    0,
                    "duplicate",
                    actor,
                    C,
                    "{}",
                    "2026-09-10T00:00:00Z",
                    "0" * 64,
                    "8" * 64,
                ),
            )
    db.close()


def test_credential_material_is_rejected_at_persistence_boundaries(tmp_path):
    with pytest.raises(ValueError, match="credential material"):
        JournalEvent("journal.commit", D, C, {"authorization": "Bearer raw"}, NOW)
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    with db.transaction() as uow:
        actor = journal.bind_actor(uow, ACTOR)
        with pytest.raises(ValueError, match="credential artifacts"):
            journal.put_artifact(uow, kind="raw-token", content=b"credential")
        uow.emit(event(actor))
    assert b"Bearer raw" not in db.path.read_bytes()
    db.close()


def test_successor_idempotency_and_lineage_constraints(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    with db.transaction() as uow:
        actor = genesis(journal, uow)
        with pytest.raises(SuccessorError):
            journal.append_transition(
                uow,
                episode_id="episode-1",
                from_state="interpreted",
                to_state="planned",
                transition_digest="4" * 64,
                actor_digest=actor,
                credential_provenance_digest=C,
            )
        with pytest.raises(SuccessorError, match="state transition"):
            journal.append_transition(
                uow,
                episode_id="episode-1",
                from_state="perceived",
                to_state="consolidated",
                transition_digest="5" * 64,
                actor_digest=actor,
                credential_provenance_digest=C,
            )
        uow.emit(event(actor))
    with db.transaction() as uow:
        actor = journal.bind_actor(uow, ACTOR)
        assert (
            journal.bind_command(
                uow,
                command_id="different-request-id",
                actor_digest=actor,
                credential_provenance_digest=D,
                request_id="request-replay",
                request_digest="1" * 64,
                idempotency_key="idem-1",
            )
            == "replay"
        )
        with pytest.raises(JournalError, match="different command"):
            journal.bind_command(
                uow,
                command_id="conflict",
                actor_digest=actor,
                credential_provenance_digest=D,
                request_id="request-conflict",
                request_digest="9" * 64,
                idempotency_key="idem-1",
            )
        uow.emit(event(actor, "replay"))
    db.close()


def test_active_and_past_lineage_membership(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    with db.transaction() as uow:
        actor = genesis(journal, uow, "1")
        journal.advance_lineage(
            uow,
            branch_id="branch-main",
            episode_id="episode-1",
            expected_head_digest="0" * 64,
            new_head_digest="1" * 64,
        )
        uow.emit(event(actor, "first"))
    with db.transaction() as uow:
        actor = genesis(journal, uow, "2")
        journal.advance_lineage(
            uow,
            branch_id="branch-main",
            episode_id="episode-2",
            expected_head_digest="1" * 64,
            new_head_digest="2" * 64,
        )
        statuses = dict(uow.query("SELECT episode_id,status FROM lineage_membership"))
        assert statuses == {"episode-1": "past", "episode-2": "active"}
        uow.emit(event(actor, "second"))
    db.close()


def test_concurrent_sequence_allocation_receipt_chain_and_close(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()

    def write(index: int) -> tuple[int, str, str]:
        with db.transaction() as uow:
            actor = genesis(journal, uow, f"c{index}")
            receipt = uow.emit(event(actor, f"worker-{index}"))
            return uow.commit_seq, receipt, actor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(8)))
    assert sorted(seq for seq, _, _ in results) == list(range(1, 9))
    with db.transaction() as uow:
        rows = uow.query(
            "SELECT commit_seq,event_ordinal,previous_receipt_digest,receipt_digest "
            "FROM transactional_outbox ORDER BY commit_seq,event_ordinal"
        )
        assert [(row[0], row[1]) for row in rows] == [(i, 0) for i in range(1, 9)]
        assert rows[0][2] == "0" * 64
        assert all(
            rows[index][2] == rows[index - 1][3] for index in range(1, len(rows))
        )
        actor = journal.bind_actor(uow, ACTOR)
        last_receipt = uow.emit(event(actor, "inspection"))
    with pytest.raises(JournalError, match="no longer active"):
        uow.query("SELECT 1")
    gc.disable()
    try:
        db.close()
        with pytest.raises(JournalError):
            with db.transaction():
                pass
    finally:
        gc.enable()
    restarted = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    with restarted.transaction() as uow:
        actor = journal.bind_actor(uow, ACTOR)
        receipt = uow.emit(event(actor, "after-restart"))
        row = uow.query(
            "SELECT previous_receipt_digest FROM transactional_outbox "
            "WHERE receipt_digest=?",
            (receipt,),
        )[0]
        assert uow.commit_seq == 10
        assert row[0] == last_receipt
    restarted.close()


def test_event_ordinal_and_receipt_not_timestamp_are_ordering_authority(tmp_path):
    db = ConstitutionalDatabase(tmp_path / "constitutional.sqlite3")
    journal = ConstitutionalJournal()
    early = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as uow:
        actor = genesis(journal, uow)
        uow.emit(event(actor, "later-timestamp-first"))
        uow.emit(JournalEvent("journal.commit", actor, C, {"label": "early"}, early))
    with db.transaction() as uow:
        rows = uow.query(
            "SELECT * FROM transactional_outbox ORDER BY commit_seq,event_ordinal"
        )
        assert [row["event_ordinal"] for row in rows] == [0, 1]
        assert rows[0]["occurred_at"] > rows[1]["occurred_at"]
        previous = "0" * 64
        for row in rows:
            body = {
                "actor_digest": row["actor_digest"],
                "commit_seq": row["commit_seq"],
                "credential_provenance_digest": row["credential_provenance_digest"],
                "event_ordinal": row["event_ordinal"],
                "event_type": row["event_type"],
                "occurred_at": row["occurred_at"],
                "payload": json.loads(row["payload"]),
                "previous_receipt_digest": previous,
                "schema_version": SCHEMA_VERSION,
            }
            assert row["previous_receipt_digest"] == previous
            assert (
                row["receipt_digest"]
                == hashlib.sha256(canonical_json(body)).hexdigest()
            )
            previous = row["receipt_digest"]
        actor = journal.bind_actor(uow, ACTOR)
        uow.emit(event(actor, "inspection"))
    db.close()


def test_pragmas_and_schema_fingerprint_are_stable(tmp_path):
    db = ConstitutionalDatabase(
        tmp_path / "constitutional.sqlite3", busy_timeout_ms=1234
    )
    first = db.fingerprint
    second = db.fingerprint
    assert first == second
    assert first == EXPECTED_SCHEMA_FINGERPRINT
    schema_evidence = json.loads(
        Path("config/constitutional-journal-schema.json").read_text()
    )
    assert schema_evidence["schema_fingerprint"] == first
    assert schema_evidence["production_wiring"] == "not-composed-no-dual-write"
    assert os.stat(db.path).st_mode & 0o777 == 0o600
    assert dict(db.sqlite_settings) == {
        "journal_mode": "wal",
        "synchronous": 2,
        "foreign_keys": 1,
        "trusted_schema": 0,
        "busy_timeout": 1234,
    }
    connection = sqlite3.connect(db.path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }
        assert not any(
            forbidden in name
            for name in names
            for forbidden in ("memory", "effect", "reafference", "autobiograph")
        )
        # Connection-local safety settings are checked by ConstitutionalDatabase
        # at every owned connection opening; external connections do not inherit them.
    finally:
        connection.close()
    db.close()
    connection = sqlite3.connect(db.path)
    connection.execute("CREATE TABLE unauthorized_schema_change(value TEXT)")
    connection.close()
    with pytest.raises(JournalError, match="schema fingerprint mismatch"):
        ConstitutionalDatabase(db.path)


def test_database_rejects_symlink_path(tmp_path):
    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        ConstitutionalDatabase(link)


def test_journal_owner_is_not_production_composed_or_dual_written() -> None:
    manifest = json.loads(Path("config/wheel-inclusion-manifest.json").read_text())
    assert "vulcan.microkernel.constitutional_journal" not in {
        row["module"] for row in manifest["files"]
    }
    for path in (
        Path("src/vulcan/runtime/api.py"),
        Path("src/vulcan/runtime/phase_a_composition.py"),
    ):
        assert "constitutional_journal" not in path.read_text(encoding="utf-8")
    implementation = Path("src/vulcan/microkernel/constitutional_journal.py").read_text(
        encoding="utf-8"
    )
    repositories = implementation.split("class ConstitutionalJournal:", 1)[1]
    assert "sqlite3.connect" not in repositories
    assert '.execute("COMMIT"' not in repositories
    assert '.execute("ROLLBACK"' not in repositories
