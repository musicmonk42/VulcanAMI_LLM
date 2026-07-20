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
    for state in [EpisodeState.INTERPRETED, EpisodeState.GROUNDED, EpisodeState.DELIBERATING, EpisodeState.EPISTEMICALLY_COMMITTED, EpisodeState.NORMATIVELY_AUTHORIZED, EpisodeState.EXECUTED, EpisodeState.OBSERVED, EpisodeState.COMMUNICATED, EpisodeState.CONSOLIDATED]:
        ep = ep.transition(state, reason=state.value, authority="microkernel", clock=fixed_clock, snapshot_ids=["snap-1"], evidence_refs=[ref()])
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
