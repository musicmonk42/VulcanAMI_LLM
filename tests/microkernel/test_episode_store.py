from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.episode_store import (
    EpisodeConflict,
    EpisodeIntegrityError,
    EpisodeStore,
)
from vulcan.microkernel.state_machine import EpisodeState


def episode() -> CognitiveEpisode:
    return CognitiveEpisode.create(
        actor=ActorBinding("test-actor", "a" * 64, "CognitiveKernel"),
        request_id="request-private",
        input_digest="b" * 64,
        episode_id="case-durable",
    )


def failed(prior: CognitiveEpisode, reason: str = "failed") -> CognitiveEpisode:
    return prior.transition(
        EpisodeState.FAILED, reason=reason, authority="CognitiveKernel"
    )


def test_restart_reconstructs_identical_digest_and_reconciles_outbox(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    original = episode()
    EpisodeStore(path).create(original)
    delivered = []

    restarted = EpisodeStore(
        path, outbox_sink=lambda kind, body: delivered.append((kind, body))
    )

    assert restarted.load(original.episode_id).digest == original.digest
    assert restarted.replay(original.episode_id).digest == original.digest
    assert delivered[0][0] == "episode.transition"
    assert delivered[0][1]["episode_digest"] == original.digest
    assert (
        EpisodeStore(
            path, outbox_sink=lambda *_: delivered.append("duplicate")
        ).deliver_outbox()
        == 0
    )


def test_competing_compare_and_swap_has_one_winner(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    original = episode()
    store.create(original)
    alternatives = [failed(original, f"competitor-{index}") for index in range(2)]

    def advance(candidate):
        try:
            store.advance(original.episode_id, original.digest, candidate)
            return "won"
        except EpisodeConflict:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(advance, alternatives))
    assert sorted(outcomes) == ["lost", "won"]
    assert store.load(original.episode_id).digest in {
        item.digest for item in alternatives
    }


def test_duplicate_retry_is_idempotent_but_identity_collision_is_rejected(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    original = episode()
    next_episode = failed(original)
    store.create(original)
    store.create(original)
    store.advance(original.episode_id, original.digest, next_episode)
    store.advance(original.episode_id, original.digest, next_episode)
    assert store.load(original.episode_id).digest == next_episode.digest

    collision = CognitiveEpisode.create(
        actor=original.actor,
        request_id="different-request",
        input_digest="c" * 64,
        episode_id=original.episode_id,
    )
    with pytest.raises(EpisodeConflict, match="different digest"):
        store.create(collision)


@pytest.mark.parametrize(
    ("point", "durable"),
    [
        ("before_db_write", False),
        ("before_head_cas", False),
        ("after_head_cas", False),
        ("after_db_write", False),
        ("before_commit", False),
        ("after_commit", True),
    ],
)
def test_crash_failpoints_preserve_transaction_boundary(tmp_path, point, durable):
    original = episode()

    def crash(name):
        if name == point:
            raise RuntimeError(f"crash:{point}")

    store = EpisodeStore(tmp_path / "episodes.sqlite3", failpoint=crash)
    with pytest.raises(RuntimeError, match=point):
        store.create(original)
    restarted = EpisodeStore(tmp_path / "episodes.sqlite3")
    if durable:
        assert restarted.load(original.episode_id).digest == original.digest
    else:
        with pytest.raises(KeyError):
            restarted.load(original.episode_id)


@pytest.mark.parametrize("point", ["before_outbox_delivery", "after_outbox_delivery"])
def test_outbox_delivery_crash_is_reconciled_on_restart(tmp_path, point):
    path = tmp_path / "episodes.sqlite3"
    original = episode()
    EpisodeStore(path).create(original)
    deliveries = []

    def crash(name):
        if name == point:
            raise RuntimeError(f"crash:{point}")

    with pytest.raises(RuntimeError, match=point):
        EpisodeStore(
            path,
            outbox_sink=lambda kind, body: deliveries.append(body["episode_digest"]),
            failpoint=crash,
        )
    EpisodeStore(
        path, outbox_sink=lambda kind, body: deliveries.append(body["episode_digest"])
    )
    assert deliveries[-1] == original.digest
    assert len(deliveries) == (1 if point == "before_outbox_delivery" else 2)


def test_tampered_stored_json_fails_startup_verification(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    original = episode()
    EpisodeStore(path).create(original)
    with sqlite3.connect(path) as connection:
        document = json.loads(
            connection.execute("SELECT document FROM episode_heads").fetchone()[0]
        )
        document["request"]["input_digest"] = "c" * 64
        connection.execute(
            "UPDATE episode_heads SET document=?",
            (json.dumps(document, sort_keys=True, separators=(",", ":")),),
        )
    with pytest.raises(EpisodeIntegrityError):
        EpisodeStore(path)


def test_tampered_or_missing_outbox_fails_startup_verification(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    original = episode()
    EpisodeStore(path).create(original)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE episode_outbox SET payload=?", ('{"episode_id":"forged"}',)
        )
    with pytest.raises(EpisodeIntegrityError, match="outbox payload"):
        EpisodeStore(path)

    path.unlink()
    EpisodeStore(path).create(original)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM episode_outbox")
    with pytest.raises(EpisodeIntegrityError, match="relational integrity"):
        EpisodeStore(path)


def test_store_never_persists_raw_request_or_provider_text(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    original = CognitiveEpisode.create(
        actor=ActorBinding("test-actor", "a" * 64, "CognitiveKernel"),
        request_id="request-private",
        raw_request="raw super secret prompt",
        episode_id="case-private",
    )
    EpisodeStore(path).create(original)
    stored = path.read_bytes()
    assert b"raw super secret prompt" not in stored
    assert b"provider completion" not in stored


def test_advance_rejects_forged_immutable_identity_and_history(tmp_path):
    from dataclasses import replace

    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    original = episode()
    store.create(original)
    valid = failed(original)
    forged_actor = replace(
        valid,
        actor=ActorBinding("other-actor", "c" * 64, "CognitiveKernel"),
    )
    with pytest.raises(EpisodeIntegrityError, match="immutable episode identity"):
        store.advance(original.episode_id, original.digest, forged_actor)

    other_genesis = CognitiveEpisode.create(
        actor=original.actor,
        request_id=original.request.request_id,
        input_digest=original.request.input_digest,
        episode_id=original.episode_id,
    )
    forged_history = replace(
        valid, transitions=(other_genesis.transitions[0], valid.transitions[-1])
    )
    with pytest.raises(EpisodeIntegrityError, match="immutable transition history"):
        store.advance(original.episode_id, original.digest, forged_history)


def test_decoder_rejects_duplicate_keys_and_non_utc_timestamps():
    from vulcan.microkernel.episode_store import episode_from_document

    document = episode().canonical_json()
    duplicate = document.replace('{"actor":', '{"actor":null,"actor":', 1)
    with pytest.raises(EpisodeIntegrityError, match="duplicate persisted JSON key"):
        episode_from_document(duplicate)

    value = json.loads(document)
    value["transitions"][0]["at"] = "2026-01-01T00:00:00"
    tampered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    with pytest.raises(EpisodeIntegrityError, match="timestamp is not UTC"):
        episode_from_document(tampered)


def test_durable_memory_database_is_rejected():
    with pytest.raises(ValueError, match="durable filesystem path"):
        EpisodeStore(":memory:")


def test_database_permissions_and_symlink_rejection(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    EpisodeStore(path)
    assert path.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(path)
    with pytest.raises(EpisodeIntegrityError, match="symlink"):
        EpisodeStore(link)


@pytest.mark.parametrize(
    ("point", "durable"),
    [
        ("before_db_write", False),
        ("before_head_cas", False),
        ("after_head_cas", False),
        ("after_db_write", False),
        ("before_commit", False),
        ("after_commit", True),
    ],
)
def test_advance_crash_failpoints_replay_last_committed_head(tmp_path, point, durable):
    path = tmp_path / "episodes.sqlite3"
    original = episode()
    EpisodeStore(path).create(original)

    def crash(name):
        if name == point:
            raise RuntimeError(f"crash:{point}")

    store = EpisodeStore(path, failpoint=crash)
    successor = failed(original)
    with pytest.raises(RuntimeError, match=point):
        store.advance(original.episode_id, original.digest, successor)
    recovered = EpisodeStore(path).replay(original.episode_id)
    assert recovered.digest == (successor.digest if durable else original.digest)
