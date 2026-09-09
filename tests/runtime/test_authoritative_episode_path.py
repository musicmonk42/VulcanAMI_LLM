from __future__ import annotations

from types import SimpleNamespace

import pytest

from vulcan.microkernel.episode import CognitiveEpisode
from vulcan.microkernel.snapshots import construct_snapshot_bundle
from vulcan.testing.snapshots import AttributeSnapshotProvider
from vulcan.microkernel.state_machine import EpisodeState
from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.constitutional_kernel import ConstitutionalCognitiveKernel
from vulcan.runtime.finalization import FinalizationDecision, FinalizationResult
from vulcan.runtime.kernel import CognitiveKernel, KernelRequest
from vulcan.runtime.semantic import Utterance, canonical_digest


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
    kernel = _kernel()
    case = kernel.create_case(
        request_id="request-1",
        conversation_id="conversation-1",
        input_digest=utterance.digest,
    )

    result = await kernel.handle(
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
        EpisodeState.COMMUNICATED,
    ]
    assert case.episode.claims
    assert case.episode.derivations
    assert case.episode.authorization is not None
    assert case.episode.response is not None
    assert case.episode.effects == ()
    assert all(len(event.authority) == 64 for event in case.episode.transitions[1:])
    assert case.episode.consolidation_refs
    assert case.snapshot_bundle is not None
    assert case.snapshot_bundle.released is True


@pytest.mark.asyncio
async def test_abstention_is_an_authoritative_terminal_episode():
    utterance = Utterance.from_text("tell me a secret")
    kernel = _kernel()
    case = kernel.create_case(
        request_id="request-2",
        conversation_id=None,
        input_digest=utterance.digest,
    )

    result = await kernel.handle(KernelRequest(utterance, None), case)

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


def test_compatibility_digest_canonicalizes_lists_and_tuples_identically():
    list_payload = {"claim_digests": ["a" * 64, "b" * 64]}
    tuple_payload = {"claim_digests": ("a" * 64, "b" * 64)}

    assert canonical_digest(list_payload) == canonical_digest(tuple_payload)


class _CountingLease:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


def test_admission_failure_releases_bundle_once():
    lease = _CountingLease()

    def wrong_identity(episode_id):
        bundle = _admitter(episode_id)
        object.__setattr__(bundle, "episode_id", "case-wrong")
        object.__setattr__(bundle, "leases", (lease,))
        return bundle

    kernel = ConstitutionalCognitiveKernel.from_kernel(
        CognitiveKernel(state_authority=object(), finalizer=_Finalizer()),
        snapshot_admitter=wrong_identity,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        kernel.create_case(
            request_id="request-admission", conversation_id=None, input_digest="d" * 64
        )
    assert lease.closes == 1


@pytest.mark.asyncio
async def test_constitutional_kernel_rejects_direct_case_bypass():
    utterance = Utterance.from_text("2+2")
    case = CognitiveCase.create(
        request_id="request-bypass", conversation_id=None, input_digest=utterance.digest
    )
    with pytest.raises(RuntimeError, match="unadmitted"):
        await _kernel().handle(KernelRequest(utterance, None), case)


@pytest.mark.asyncio
async def test_concurrent_admissions_have_distinct_genesis_bound_episodes():
    kernel = _kernel()
    utterance = Utterance.from_text("2+2")
    cases = [
        kernel.create_case(
            request_id=f"request-{i}",
            conversation_id=None,
            input_digest=utterance.digest,
        )
        for i in range(12)
    ]
    genesis = [case.episode.digest for case in cases]
    await __import__("asyncio").gather(
        *(kernel.handle(KernelRequest(utterance, None), case) for case in cases)
    )
    assert len({case.case_id for case in cases}) == len(cases)
    assert len(set(genesis)) == len(cases)
    assert all(case.snapshot_bundle.released for case in cases)


def test_invalid_and_duplicate_artifact_references_fail_closed():
    from vulcan.microkernel.episode import ArtifactRef

    with pytest.raises(ValueError):
        ArtifactRef("response-ok", "not-a-digest", "response-ir.v3")
    ref = ArtifactRef("claim-valid", "d" * 64, "semantic-claim.v2")
    case = CognitiveCase.create(
        request_id="request-duplicates", conversation_id=None, input_digest="d" * 64
    )
    with pytest.raises(ValueError, match="duplicate"):
        __import__("dataclasses").replace(case.episode, claims=(ref, ref))


@pytest.mark.asyncio
async def test_cancellation_releases_admitted_leases_exactly_once():
    import asyncio

    lease = _CountingLease()

    def admitted(episode_id):
        bundle = _admitter(episode_id)
        object.__setattr__(bundle, "leases", (lease,))
        return bundle

    class CancelInput:
        async def propose(self, utterance):
            raise asyncio.CancelledError

    kernel = ConstitutionalCognitiveKernel.from_kernel(
        CognitiveKernel(
            state_authority=object(),
            finalizer=_Finalizer(),
            language_input=CancelInput(),
        ),
        snapshot_admitter=admitted,
    )
    utterance = Utterance.from_text("2+2")
    case = kernel.create_case(
        request_id="request-cancel", conversation_id=None, input_digest=utterance.digest
    )
    with pytest.raises(asyncio.CancelledError):
        await kernel.handle(KernelRequest(utterance, None), case)
    assert case.episode.state is EpisodeState.CANCELLED
    assert lease.closes == 1


def test_compatibility_ledger_cannot_invoke_episode_transition(monkeypatch):
    import asyncio

    from vulcan.runtime.semantic import (
        DeterministicLanguageInput,
        accept,
        build_graphix_plan,
        compile_graphix_plan,
        execute_graphix_plan,
        validate_proposal,
    )

    utterance = Utterance.from_text("2+2")
    case = _kernel().create_case(
        request_id="request-atomic", conversation_id=None, input_digest=utterance.digest
    )
    bundle = asyncio.run(DeterministicLanguageInput().propose(utterance))
    case.interpretation = validate_proposal(utterance, bundle)
    case.accepted_interpretation = accept(case.interpretation)
    plan = build_graphix_plan(
        case.accepted_interpretation,
        request_digest=utterance.digest,
        state_snapshot_id=case.state_snapshot_id,
        domain_snapshot_id="domain:none",
    )
    compiled = compile_graphix_plan(
        plan,
        request_digest=utterance.digest,
        state_snapshot_id=case.state_snapshot_id,
        domain_snapshot_id="domain:none",
    )
    claim, derivation, evidence = execute_graphix_plan(
        compiled,
        request_digest=utterance.digest,
        state_snapshot_id=case.state_snapshot_id,
        domain_snapshot_id="domain:none",
        case_id=case.case_id,
        domain=None,
    )
    before = case.episode

    def deny_case_promotion(self, target, **kwargs):
        raise RuntimeError("compatibility case attempted authority promotion")

    monkeypatch.setattr(CognitiveEpisode, "transition", deny_case_promotion)
    with pytest.raises(RuntimeError, match="direct case ledger mutation is prohibited"):
        case.append_ledger(claim=claim, derivation=derivation, evidence=evidence)
    assert case.episode == before
    assert case.claims == ()
    assert case.derivations == ()
    assert case.evidence == ()


def test_cognitive_case_has_no_authority_promotion_or_persistence_logic():
    import inspect

    source = inspect.getsource(CognitiveCase)
    assert ".transition(" not in source
    assert ".advance(" not in source
    assert "response-authorization.compat.v1" not in source


def _durable_kernel(path, *, finalizer=None, failpoint=None):
    from vulcan.microkernel.episode_store import EpisodeStore

    store = EpisodeStore(path, failpoint=failpoint)
    inner = CognitiveKernel(
        state_authority=SimpleNamespace(version="world-1"),
        finalizer=finalizer or _Finalizer(),
    )
    return (
        ConstitutionalCognitiveKernel.from_kernel(
            inner, snapshot_admitter=_admitter, episode_store=store
        ),
        store,
    )


@pytest.mark.asyncio
async def test_response_is_withheld_when_episode_persistence_fails(tmp_path):
    armed = {"value": False}

    def fail(name):
        if armed["value"] and name == "before_commit":
            raise OSError("episode disk unavailable")

    kernel, store = _durable_kernel(tmp_path / "episodes.sqlite3", failpoint=fail)
    utterance = Utterance.from_text("2+2")
    case = kernel.create_case(
        request_id="request-persist-failure",
        conversation_id=None,
        input_digest=utterance.digest,
    )
    genesis = case.episode.digest
    armed["value"] = True

    with pytest.raises(OSError, match="disk unavailable"):
        await kernel.handle(KernelRequest(utterance, None), case)

    assert store.load(case.case_id).digest == genesis
    assert case.terminal_status is CognitiveCaseStatus.OPEN


@pytest.mark.asyncio
async def test_cancellation_after_finalization_before_transport_keeps_durable_head(
    tmp_path,
):
    import asyncio

    class CancelAtTransportFinalizer:
        async def finalize(self, artifact):
            task = asyncio.current_task()
            assert task is not None
            task.get_loop().call_soon(task.cancel)
            return FinalizationResult(
                FinalizationDecision.ALLOW, artifact, artifact.text
            )

    kernel, store = _durable_kernel(
        tmp_path / "episodes.sqlite3", finalizer=CancelAtTransportFinalizer()
    )
    utterance = Utterance.from_text("2+2")
    case = kernel.create_case(
        request_id="request-cancel-before-transport",
        conversation_id=None,
        input_digest=utterance.digest,
    )

    with pytest.raises(asyncio.CancelledError):
        await kernel.handle(KernelRequest(utterance, None), case)

    assert case.episode.state is EpisodeState.CONSOLIDATED
    assert store.load(case.case_id).digest == case.episode.digest
    assert case.snapshot_bundle.released is True


@pytest.mark.asyncio
async def test_transport_rejects_nonterminal_delegate_result(tmp_path):
    class InvalidDelegate:
        async def handle(self, request, case):
            from vulcan.runtime.kernel import KernelResult

            return KernelResult(
                "not-authorized", None, CognitiveCaseStatus.OPEN, "allow"
            )

        def capabilities(self):
            return ()

    from vulcan.microkernel.episode_store import EpisodeStore

    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    kernel = ConstitutionalCognitiveKernel.from_kernel(
        InvalidDelegate(), snapshot_admitter=_admitter, episode_store=store
    )
    utterance = Utterance.from_text("2+2")
    case = kernel.create_case(
        request_id="request-nonterminal",
        conversation_id=None,
        input_digest=utterance.digest,
    )
    with pytest.raises(RuntimeError, match="not terminal"):
        await kernel.handle(KernelRequest(utterance, None), case)
