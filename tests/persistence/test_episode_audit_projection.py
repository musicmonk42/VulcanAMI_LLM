from __future__ import annotations

import json
import hashlib

import pytest

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.state_machine import ALLOWED_TRANSITIONS, EpisodeState
from vulcan.runtime.audit import AuditError, CanonicalAudit


def _episode() -> CognitiveEpisode:
    return CognitiveEpisode.create(
        actor=ActorBinding("actor", "a" * 64, "CognitiveKernel"),
        request_id="request-private",
        input_digest="b" * 64,
        episode_id="case-audit",
    )


def test_outbox_projects_authoritative_digest_chain_and_deduplicates_restart(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    audit = CanonicalAudit(audit_path)
    path = tmp_path / "episodes.sqlite3"
    store = EpisodeStore(path, outbox_sink=audit.append_episode_transition)
    genesis = _episode()
    store.create(genesis)
    terminal = genesis.transition(
        EpisodeState.CANCELLED, reason="cancelled", authority="CognitiveKernel"
    )
    store.advance(genesis.episode_id, genesis.digest, terminal)

    rows = audit.events_for_episode(genesis.episode_id)
    assert [row.data["to_state"] for row in rows] == ["perceived", "cancelled"]
    assert rows[1].data["prior_episode_digest"] == genesis.digest
    assert rows[1].data["resulting_episode_digest"] == terminal.digest
    assert len({row.data["transition_digest"] for row in rows}) == 2

    # Simulate append succeeding before the outbox delivered mark.  Delivery is
    # at-least-once, while the externally visible audit effect is exactly once.
    with store._connection() as connection:
        connection.execute("UPDATE episode_outbox SET delivered_at=NULL WHERE id=2")
    assert store.deliver_outbox() == 1
    assert len(audit.events_for_episode(genesis.episode_id)) == 2


def test_restart_reconciles_commit_that_preceded_audit_append(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    genesis = _episode()
    EpisodeStore(path).create(genesis)
    audit = CanonicalAudit(tmp_path / "audit.jsonl")

    EpisodeStore(path, outbox_sink=audit.append_episode_transition)

    rows = audit.events_for_episode(genesis.episode_id)
    assert len(rows) == 1
    assert rows[0].data["resulting_episode_digest"] == genesis.digest


def test_episode_audit_rejects_out_of_order_wrong_prior_and_duplicate_tamper(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit.jsonl")
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    genesis = _episode()
    store.create(genesis)
    with store._connection() as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM episode_outbox").fetchone()[0]
        )
    audit.append_episode_transition("episode.transitioned", payload)
    successor = genesis.transition(
        EpisodeState.CANCELLED, reason="cancelled", authority="CognitiveKernel"
    )
    store.advance(genesis.episode_id, genesis.digest, successor)
    with store._connection() as connection:
        next_payload = json.loads(
            connection.execute("SELECT payload FROM episode_outbox WHERE id=2").fetchone()[0]
        )
    out_of_order = CanonicalAudit(tmp_path / "out-of-order.jsonl")
    with pytest.raises(AuditError, match="start at genesis"):
        out_of_order.append_episode_transition("episode.transitioned", next_payload)

    wrong_prior = dict(next_payload)
    wrong_prior["prior_episode_digest"] = "f" * 64
    wrong_prior["transition"] = dict(next_payload["transition"])
    wrong_prior["transition"]["prior_digest"] = "f" * 64
    unsigned = dict(wrong_prior["transition"])
    unsigned.pop("event_digest")
    digest = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    wrong_prior["transition"]["event_digest"] = digest
    wrong_prior["transition_digest"] = digest
    with pytest.raises(AuditError, match="prior digest"):
        audit.append_episode_transition("episode.transitioned", wrong_prior)

    duplicate_tamper = dict(payload)
    duplicate_tamper["authority_references"] = ["forged"]
    with pytest.raises(AuditError, match="different data"):
        audit.append_episode_transition("episode.transitioned", duplicate_tamper)


def test_legacy_case_events_remain_readable_but_do_not_set_episode_head(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit.jsonl")
    audit.append(
        "case.started", {"case_id": "case-legacy", "request_digest": "d" * 64}
    )
    audit.append(
        "case.failed", {"case_id": "case-legacy", "request_digest": "d" * 64}
    )
    assert [event.event_type for event in audit.events_for_case("case-legacy")] == [
        "case.started", "case.failed"
    ]
    assert "case-legacy" not in audit._episode_heads


def test_cancellation_is_legal_from_every_nonterminal_preconsolidation_state():
    cancellable = set(EpisodeState).difference(
        {
            EpisodeState.CONSOLIDATED,
            EpisodeState.ABSTAINED,
            EpisodeState.BLOCKED,
            EpisodeState.FAILED,
            EpisodeState.CANCELLED,
        }
    )
    assert all(
        EpisodeState.CANCELLED in ALLOWED_TRANSITIONS[state]
        for state in cancellable
    )
