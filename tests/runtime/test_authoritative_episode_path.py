from __future__ import annotations

from types import SimpleNamespace

import pytest

from vulcan.microkernel.snapshots import AttributeSnapshotProvider, construct_snapshot_bundle
from vulcan.microkernel.state_machine import EpisodeState
from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.constitutional_kernel import ConstitutionalCognitiveKernel
from vulcan.runtime.finalization import FinalizationDecision, FinalizationResult
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest
from vulcan.runtime.semantic import Utterance


class _Finalizer:
    async def finalize(self, artifact):
        return FinalizationResult(
            FinalizationDecision.ALLOW,
            artifact,
            artifact.text,
        )


def _admitter(episode_id: str):
    owners = [SimpleNamespace(version=f"owner-{index}") for index in range(9)]
    providers = tuple(
        AttributeSnapshotProvider(owner, owner_name=f"test-owner-{index}")
        for index, owner in enumerate(owners)
    )
    return construct_snapshot_bundle(episode_id=episode_id, providers=providers)


def _kernel() -> ConstitutionalCognitiveKernel:
    inner = CognitiveKernel(
        state_authority=SimpleNamespace(version="world-1"),
        finalizer=_Finalizer(),
    )
    return ConstitutionalCognitiveKernel.from_kernel(
        inner,
        snapshot_admitter=_admitter,
    )


@pytest.mark.asyncio
async def test_successful_request_binds_snapshot_and_consolidates_episode():
    utterance = Utterance.from_text("2 + 3 * 4")
    case = CognitiveCase.create(
        request_id="request-1",
        conversation_id="conversation-1",
        input_digest=utterance.digest,
    )

    result = await _kernel().handle(
        KernelRequest(utterance, "conversation-1"),
        case,
    )

    assert result.status is CognitiveCaseStatus.SUCCESS
    assert case.case_id.startswith("case-")
    assert case.episode is not None
    assert case.episode.episode_id == case.case_id
    assert case.episode.snapshot_bundle is not None
    assert case.state_snapshot_id == case.episode.snapshot_bundle.state_digest
    assert case.episode.state is EpisodeState.CONSOLIDATED
    assert [event.to_state for event in case.episode.transitions] == [
        EpisodeState.PERCEIVED,
        EpisodeState.INTERPRETED,
        EpisodeState.GROUNDED,
        EpisodeState.DELIBERATING,
        EpisodeState.EPISTEMICALLY_COMMITTED,
        EpisodeState.NORMATIVELY_AUTHORIZED,
        EpisodeState.EXECUTED,
        EpisodeState.OBSERVED,
        EpisodeState.COMMUNICATED,
        EpisodeState.CONSOLIDATED,
    ]
    assert [event.from_state for event in case.episode.transitions[1:]] == [
        EpisodeState.PERCEIVED,
        EpisodeState.INTERPRETED,
        EpisodeState.GROUNDED,
        EpisodeState.DELIBERATING,
        EpisodeState.EPISTEMICALLY_COMMITTED,
        EpisodeState.NORMATIVELY_AUTHORIZED,
        EpisodeState.EXECUTED,
        EpisodeState.OBSERVED,
        EpisodeState.COMMUNICATED,
    ]
    assert case.episode.claims
    assert case.episode.derivations
    assert case.episode.authorization is not None
    assert case.episode.response is not None
    assert case.episode.effects == (case.episode.response,)
    assert case.episode.consolidation_refs
    assert case.snapshot_bundle is not None
    assert case.snapshot_bundle.released is True


@pytest.mark.asyncio
async def test_abstention_is_an_authoritative_terminal_episode():
    utterance = Utterance.from_text("tell me a secret")
    case = CognitiveCase.create(
        request_id="request-2",
        conversation_id=None,
        input_digest=utterance.digest,
    )

    result = await _kernel().handle(KernelRequest(utterance, None), case)

    assert result.status is CognitiveCaseStatus.ABSTAINED
    assert case.episode is not None
    assert case.episode.state is EpisodeState.ABSTAINED
    assert case.episode.snapshot_bundle is not None
    assert case.episode.claims
    assert case.episode.derivations
    assert case.episode.response is not None
    assert case.snapshot_bundle is not None
    assert case.snapshot_bundle.released is True


def test_case_identifier_and_episode_identifier_are_one_identity():
    case = CognitiveCase.create(
        request_id="request-3",
        conversation_id=None,
        input_digest="d" * 64,
    )
    assert case.case_id.startswith("case-")
    assert case.episode is not None
    assert case.episode.episode_id == case.case_id
