from datetime import datetime, timezone

import pytest

from vulcan.microkernel.episode import ActorBinding, ArtifactRef, CognitiveEpisode, canonical_digest, digest_text
from vulcan.microkernel.state_machine import EpisodeState, EpisodeTransitionError
from vulcan.runtime.case import CognitiveCase, episode_from_case


def fixed_clock():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def actor():
    return ActorBinding("user:1", "a" * 64, "microkernel")


def ref(kind="evidence"):
    return ArtifactRef(f"{kind}:1", "b" * 64, kind)


def test_episode_does_not_persist_raw_input_and_serializes_canonically():
    ep = CognitiveEpisode.create(actor=actor(), request_id="r1", raw_request="secret token", clock=fixed_clock)
    encoded = ep.canonical_json()
    assert "secret token" not in encoded
    assert ep.request.input_digest == digest_text("secret token")
    assert ep.to_json()["schema_version"] == "cognitive-episode.v1"
    assert ep.digest == canonical_digest(ep.to_json(include_digest=False))


def test_digest_chain_binds_transition_to_prior_episode_digest():
    ep = CognitiveEpisode.create(actor=actor(), request_id="r1", input_digest="c" * 64, clock=fixed_clock)
    prior = ep.digest
    ep2 = ep.transition(EpisodeState.INTERPRETED, reason="schema-valid", authority="microkernel", clock=fixed_clock, interpretation={"intent": "chat"}, evidence_refs=[ref()])
    assert ep2.transitions[-1].prior_digest == prior
    assert ep2.transitions[-1].event_digest == canonical_digest(ep2.transitions[-1].to_json(include_digest=False))
    assert ep2.digest != prior


def test_invalid_transition_and_missing_authority_fail_closed():
    ep = CognitiveEpisode.create(actor=actor(), request_id="r1", input_digest="c" * 64, clock=fixed_clock)
    with pytest.raises(EpisodeTransitionError):
        ep.transition(EpisodeState.EXECUTED, reason="skip", authority="microkernel", clock=fixed_clock)
    with pytest.raises(EpisodeTransitionError):
        ep.transition(EpisodeState.INTERPRETED, reason="no-authority", authority="", clock=fixed_clock)


def test_episode_is_immutable_and_collections_are_not_externally_mutable():
    ep = CognitiveEpisode.create(actor=actor(), request_id="r1", input_digest="c" * 64, clock=fixed_clock)
    with pytest.raises(Exception):
        ep.state = EpisodeState.FAILED
    with pytest.raises(TypeError):
        ep.interpretation["x"] = "y"
    assert isinstance(ep.transitions, tuple)


def test_lifecycle_success_path_property():
    ep = CognitiveEpisode.create(actor=actor(), request_id="r1", input_digest="c" * 64, clock=fixed_clock)
    authorization = ArtifactRef("authorization:1", "d" * 64, "response-authorization.compat.v1")
    response = ArtifactRef("response:1", "e" * 64, "response-ir.v3")
    consolidation = ArtifactRef("consolidation:1", "f" * 64, "episode-consolidation.v1")
    for state in [EpisodeState.INTERPRETED, EpisodeState.GROUNDED, EpisodeState.DELIBERATING, EpisodeState.EPISTEMICALLY_COMMITTED, EpisodeState.NORMATIVELY_AUTHORIZED, EpisodeState.EXECUTED, EpisodeState.OBSERVED, EpisodeState.COMMUNICATED, EpisodeState.CONSOLIDATED]:
        updates = {}
        if state is EpisodeState.NORMATIVELY_AUTHORIZED:
            updates["authorization"] = authorization
        if state is EpisodeState.EXECUTED:
            updates.update(response=response, effects=(response,))
        if state is EpisodeState.CONSOLIDATED:
            updates["consolidation_refs"] = (consolidation,)
        ep = ep.transition(state, reason=state.value, authority="microkernel", clock=fixed_clock, snapshot_ids=["snap-1"], evidence_refs=[ref()], **updates)
    assert ep.state is EpisodeState.CONSOLIDATED
    with pytest.raises(EpisodeTransitionError):
        ep.transition(EpisodeState.FAILED, reason="late", authority="microkernel", clock=fixed_clock)


def test_legacy_cognitive_case_adapter_preserves_request_ledger_identity():
    case = CognitiveCase.create(request_id="request", conversation_id="conversation", input_digest="d" * 64)
    ep = episode_from_case(case)
    assert ep.episode_id == case.case_id
    assert ep.request.request_id == case.request_id
    assert ep.conversation_id == case.conversation_id
    assert ep.request.input_digest == case.input_hash


def test_snapshot_rebase_requires_typed_digest_bound_authority_evidence():
    from vulcan.microkernel.episode import SnapshotBundleRef, SnapshotRebaseCommand

    old = SnapshotBundleRef("snapshot-old", "1" * 64)
    new = SnapshotBundleRef("snapshot-new", "2" * 64)
    ep = CognitiveEpisode.create(actor=actor(), request_id="request-rebase", input_digest="c" * 64, snapshot_bundle=old, clock=fixed_clock)
    with pytest.raises(EpisodeTransitionError, match="typed authorized rebase"):
        ep.transition(EpisodeState.INTERPRETED, reason="please rebase this transition", authority="microkernel", snapshot_ids=(new.bundle_id, new.state_digest), clock=fixed_clock)
    evidence = ArtifactRef("rebase-authorization:1", "3" * 64, "snapshot-rebase-authorization.v1")
    command = SnapshotRebaseCommand(old_bundle=old, new_bundle=new, reason_code="state-expired", authority_evidence=evidence, prior_episode_digest=ep.digest)
    rebased = ep.transition(EpisodeState.INTERPRETED, reason="state authority replacement", authority="microkernel", snapshot_ids=(new.bundle_id, new.state_digest), evidence_refs=(), rebase=command, clock=fixed_clock)
    assert rebased.snapshot_bundle == new
    assert rebased.transitions[-1].prior_digest == ep.digest
    assert rebased.transitions[-1].evidence_refs == (evidence,)


def test_snapshot_rebase_rejects_replay_against_a_different_episode_digest():
    from vulcan.microkernel.episode import SnapshotBundleRef, SnapshotRebaseCommand

    old = SnapshotBundleRef("snapshot-old", "1" * 64)
    new = SnapshotBundleRef("snapshot-new", "2" * 64)
    ep = CognitiveEpisode.create(actor=actor(), request_id="request-replay", input_digest="c" * 64, snapshot_bundle=old, clock=fixed_clock)
    command = SnapshotRebaseCommand(old, new, "state-expired", ArtifactRef("rebase-authorization:1", "3" * 64, "snapshot-rebase-authorization.v1"), "9" * 64)
    with pytest.raises(EpisodeTransitionError, match="invalid typed snapshot rebase"):
        ep.transition(EpisodeState.INTERPRETED, reason="replacement", authority="microkernel", snapshot_ids=(new.bundle_id, new.state_digest), rebase=command, clock=fixed_clock)
