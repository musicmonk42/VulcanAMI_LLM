from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pytest

from vulcan.constitution.primitives import canonical_json
from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode, EpisodeRef
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.lineage import (
    AUTHORITY_KINDS,
    AuthoritySnapshotRef,
    LineageConflict,
    LineageError,
    LineageIntegrityError,
    LineageState,
    LineageStore,
    LineageTransactionService,
)
from vulcan.microkernel.principals import Principal, PrincipalKind


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def service(path):
    episode_store = EpisodeStore(path)
    store = LineageStore(path)
    principal = Principal(
        PrincipalKind.SYSTEM_KERNEL, "lineage-test-kernel", digest("release")
    )
    return episode_store, store, LineageTransactionService(store, principal)


def episode(name: str) -> CognitiveEpisode:
    return CognitiveEpisode.create(
        actor=ActorBinding("kernel-test", digest("principal"), "CognitiveKernel"),
        request_id=f"request-{name}",
        input_digest=digest(name),
        episode_id=f"case-{name}",
    )


def snapshots(seed: str = "snapshot") -> tuple[AuthoritySnapshotRef, ...]:
    return tuple(
        AuthoritySnapshotRef(
            kind,
            digest(f"{seed}-{kind}"),
            "1",
            "test-snapshot.v1",
            f"owner-{kind}",
            f"release-{kind}",
        )
        for kind in AUTHORITY_KINDS
    )


def test_event_replay_reproduces_exact_head_and_rejects_tamper(tmp_path):
    episodes, store, tx = service(tmp_path / "lineage.sqlite3")
    head = tx.genesis("lineage-test", "branch-main", "instance-one")
    head = tx.admit_episode(
        "branch-main",
        head.digest,
        episode("one"),
        snapshots("first"),
        episodes,
    )
    assert store.replay("branch-main") == head

    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE lineage_events SET document=replace(document, 'world', 'w0rld') WHERE tick=1"
        )
    with pytest.raises(LineageIntegrityError):
        LineageStore(store.path)


def test_concurrent_writers_cannot_both_become_current(tmp_path):
    episodes, store, tx = service(tmp_path / "lineage.sqlite3")
    head = tx.genesis("lineage-test", "branch-main", "instance-one")

    def admit(n: int):
        return tx.admit_episode(
            "branch-main",
            head.digest,
            episode(str(n)),
            snapshots(str(n)),
            episodes,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.exception()
            for future in (pool.submit(admit, 1), pool.submit(admit, 2))
        ]
    assert sum(error is None for error in results) == 1
    assert sum(isinstance(error, LineageConflict) for error in results) == 1
    assert store.load("branch-main").tick == 1


def test_restart_suspend_resume_clone_fork_and_merge_are_explicit(tmp_path):
    episodes, store, tx = service(tmp_path / "lineage.sqlite3")
    main = tx.genesis("lineage-test", "branch-main", "instance-one")
    main = tx.restart("branch-main", main.digest, "instance-two")
    assert main.instance_id == "instance-two"
    main = tx.suspend("branch-main", main.digest)
    with pytest.raises(LineageError):
        tx.admit_episode("branch-main", main.digest, episode("denied"), (), episodes)
    main = tx.resume("branch-main", main.digest, "instance-three")

    fork = tx.fork("branch-main", main.digest, "branch-fork", "instance-fork")
    clone = tx.clone("branch-main", main.digest, "branch-clone", "instance-clone")
    assert fork.branch_id != main.branch_id and fork.instance_id != main.instance_id
    assert clone.branch_id != main.branch_id and clone.instance_id != main.instance_id
    merged = tx.merge("branch-main", main.digest, "branch-fork", fork.digest)
    assert merged.tick == main.tick + 1

    with sqlite3.connect(store.path) as db:
        operations = {
            row[0] for row in db.execute("SELECT operation FROM lineage_events")
        }
    assert {"restart", "suspend", "resume", "fork", "clone"} <= operations
    assert any(operation.startswith("merge:branch-fork:") for operation in operations)


def test_non_kernel_cannot_be_lineage_authority(tmp_path):
    store = LineageStore(tmp_path / "lineage.sqlite3")
    with pytest.raises(LineageError):
        LineageTransactionService(
            store,
            Principal(PrincipalKind.HUMAN, "human-user", digest("release")),
        )


def test_crash_before_commit_leaves_replayable_prior_head(tmp_path):
    path = tmp_path / "lineage.sqlite3"
    _, store, tx = service(path)
    head = tx.genesis("lineage-test", "branch-main", "instance-one")

    def crash(point: str) -> None:
        if point == "before_commit":
            raise OSError("simulated crash")

    crashing = LineageTransactionService(
        LineageStore(path, failpoint=crash), tx.principal
    )
    with pytest.raises(OSError, match="simulated crash"):
        crashing.restart("branch-main", head.digest, "instance-two")
    restarted = LineageStore(path)
    assert restarted.load("branch-main") == head
    assert restarted.replay("branch-main") == head


def test_admission_crash_rolls_back_episode_and_lineage_together(tmp_path):
    path = tmp_path / "lineage.sqlite3"
    episodes, store, tx = service(path)
    head = tx.genesis("lineage-test", "branch-main", "instance-one")

    def crash(point: str) -> None:
        if point == "after_episode_genesis":
            raise OSError("simulated admission crash")

    crashing_store = LineageStore(path, failpoint=crash)
    crashing = LineageTransactionService(crashing_store, tx.principal)
    with pytest.raises(OSError, match="simulated admission crash"):
        crashing.admit_episode(
            "branch-main",
            head.digest,
            episode("crash"),
            snapshots("crash"),
            episodes,
        )
    assert LineageStore(path).load("branch-main") == head
    with pytest.raises(KeyError):
        episodes.load("case-crash")


def test_operation_and_authority_are_integrity_protected(tmp_path):
    _, store, tx = service(tmp_path / "lineage.sqlite3")
    tx.genesis("lineage-test", "branch-main", "instance-one")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE lineage_events SET operation='clone' WHERE branch_id='branch-main'"
        )
    with pytest.raises(LineageIntegrityError):
        LineageStore(store.path)


def test_lineage_documents_exclude_request_and_conversation_identity(tmp_path):
    path = tmp_path / "lineage.sqlite3"
    episodes, store, tx = service(path)
    head = tx.genesis("lineage-test", "branch-main", "instance-one")
    admitted = CognitiveEpisode.create(
        actor=ActorBinding("kernel-test", digest("principal"), "CognitiveKernel"),
        request_id="private-request-identity",
        conversation_id="private-conversation-identity",
        input_digest=digest("private-input"),
        episode_id="case-private",
    )
    tx.admit_episode(
        "branch-main", head.digest, admitted, snapshots("privacy"), episodes
    )
    with sqlite3.connect(store.path) as db:
        lineage_documents = "".join(
            row[0] for row in db.execute("SELECT document FROM lineage_events")
        )
    assert "private-request-identity" not in lineage_documents
    assert "private-conversation-identity" not in lineage_documents


def test_nonterminal_episode_cannot_enter_past_history(tmp_path):
    path = tmp_path / "lineage.sqlite3"
    episodes, store, tx = service(path)
    head = tx.genesis("lineage-test", "branch-main", "instance-one")
    admitted = episode("active")
    tx.admit_episode(
        "branch-main", head.digest, admitted, snapshots("active"), episodes
    )
    with pytest.raises(LineageIntegrityError, match="terminal"):
        tx.complete_episode("branch-main", admitted, episodes)


def test_pre_event_digest_schema_is_migrated_and_verified(tmp_path):
    path = tmp_path / "lineage.sqlite3"
    EpisodeStore(path)
    head = LineageState("lineage-test", "branch-main", "instance-one", 0, "0" * 64)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE lineage_heads(branch_id TEXT PRIMARY KEY, lineage_id TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, document TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE lineage_events(branch_id TEXT NOT NULL, tick INTEGER NOT NULL, prior_digest TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, operation TEXT NOT NULL, authority_digest TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(branch_id,tick))"
        )
        document = canonical_json(head.to_json()).decode()
        db.execute(
            "INSERT INTO lineage_heads VALUES(?,?,?,?)",
            (head.branch_id, head.lineage_id, head.digest, document),
        )
        db.execute(
            "INSERT INTO lineage_events VALUES(?,?,?,?,?,?,?)",
            (
                head.branch_id,
                head.tick,
                head.prior_state_digest,
                head.digest,
                "genesis",
                digest("principal"),
                document,
            ),
        )
    migrated = LineageStore(path)
    assert migrated.replay("branch-main") == head
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(lineage_events)")}
    assert "event_digest" in columns
