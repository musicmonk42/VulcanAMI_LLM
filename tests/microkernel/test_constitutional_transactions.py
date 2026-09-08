from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from vulcan.constitution.primitives import AuthorityLevel
from vulcan.microkernel.authority import (
    AuthorityError,
    AuthorityGrant,
    EvidenceRecord,
    AuditSink,
    Operation,
    promote_authority,
)
from vulcan.microkernel.episode import (
    ActorBinding,
    ArtifactRef,
    CognitiveEpisode,
    SnapshotBundleRef,
    canonical_digest,
)
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.capability_tokens import CapabilityTokenIssuer
from vulcan.microkernel.principals import Principal, PrincipalKind, digest
from vulcan.microkernel.transactions import (
    CommandAuthority,
    ConstitutionalTransactionService,
    EffectAuthorization,
    PublicationAuthorization,
    TerminalOutcome,
)

D = "a" * 64
SNAPSHOT = "b" * 64


def principal(kind=PrincipalKind.SYSTEM_KERNEL, name="kernel"):
    return Principal(kind, name, "c" * 64)


def admitted(store, episode_id="case-constitutional"):
    episode = CognitiveEpisode.create(
        actor=ActorBinding("request-actor", D, "requester"),
        request_id="request",
        input_digest=D,
        episode_id=episode_id,
        snapshot_bundle=SnapshotBundleRef("snapshot-bundle", SNAPSHOT),
    )
    store.create(episode)
    return episode


def command(store, p, level=AuthorityLevel.EXECUTED_EFFECT):
    head = store.load("case-constitutional")
    evidence = EvidenceRecord(p.identity_digest, D, D, datetime.now(timezone.utc))
    grant = promote_authority(
        current=AuthorityLevel.UNTRUSTED_PROPOSAL,
        target=level,
        principal=p,
        evidence=evidence,
    )
    return CommandAuthority(p, grant, evidence, D, D, SNAPSHOT, head.digest)


def ref(name, kind="claim.v1"):
    return ArtifactRef(name, canonical_digest({"name": name}), kind)


def test_publication_is_digest_bound_communication_not_executed_effect(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    service = ConstitutionalTransactionService(store)
    kernel = principal()
    service.record_interpretation(
        "case-constitutional", command(store, kernel), {"intent": "arithmetic"}
    )
    service.ground_candidate("case-constitutional", command(store, kernel))
    service.open_deliberation(
        "case-constitutional", command(store, kernel), (ref("plan:1", "plan.v1"),)
    )
    claim, evidence, derivation = (
        ref("claim:1"),
        ref("evidence:1", "evidence.v1"),
        ref("derivation:1", "derivation.v1"),
    )
    committed = service.commit_epistemic_artifact(
        "case-constitutional",
        command(store, kernel),
        claims=(claim,),
        evidence=(evidence,),
        derivations=(derivation,),
    )
    rendered = ref("response:1", "response-ir.v3")
    authorization = PublicationAuthorization(
        committed.digest,
        D,
        D,
        7,
        D,
        rendered.digest,
        D,
        D,
        kernel.identity_digest,
        kernel.release_digest,
    )
    service.authorize_response_publication(
        "case-constitutional",
        command(store, kernel),
        authorization=authorization,
        response=rendered,
    )
    communicated = service.communicate("case-constitutional", command(store, kernel))
    assert communicated.state.value == "communicated"
    assert communicated.effects == ()
    final = service.consolidate(
        "case-constitutional",
        command(store, kernel),
        ref("consolidation:1", "episode-consolidation.v1"),
    )
    assert final.state.value == "consolidated"
    assert (
        EpisodeStore(tmp_path / "episodes.sqlite3").replay(final.episode_id).digest
        == final.digest
    )


@pytest.mark.parametrize(
    "kind",
    [
        PrincipalKind.LANGUAGE_PROVIDER,
        PrincipalKind.REASONER,
        PrincipalKind.EXTERNAL_PROVIDER,
        PrincipalKind.TOOL,
    ],
)
def test_proposal_organs_cannot_promote_themselves(tmp_path, kind):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    organ = principal(kind, kind.value)
    evidence = EvidenceRecord(organ.identity_digest, D, D, datetime.now(timezone.utc))
    fake_grant = AuthorityGrant(
        organ.identity_digest,
        AuthorityLevel.EXECUTED_EFFECT,
        digest(evidence.to_json()),
    )
    auth = CommandAuthority(
        organ,
        fake_grant,
        evidence,
        D,
        D,
        SNAPSHOT,
        store.load("case-constitutional").digest,
    )
    with pytest.raises(AuthorityError, match="SYSTEM_KERNEL"):
        ConstitutionalTransactionService(store).record_interpretation(
            "case-constitutional", auth, {"attacker": kind.value}
        )


def test_stale_replay_and_concurrent_writers_fail_closed(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    service = ConstitutionalTransactionService(store)
    kernel = principal()
    stale = command(store, kernel)
    service.record_interpretation("case-constitutional", stale, {"intent": "one"})
    with pytest.raises(AuthorityError, match="stale"):
        service.record_interpretation(
            "case-constitutional", stale, {"intent": "replay"}
        )

    shared = command(store, kernel)

    def race():
        try:
            service.ground_candidate("case-constitutional", shared)
            return "committed"
        except AuthorityError:
            return "denied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: race(), range(2)))
    assert results.count("committed") == 1
    assert results.count("denied") == 1


def test_cancellation_is_a_durable_typed_terminal_command(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    service = ConstitutionalTransactionService(store)
    cancelled = service.terminalize(
        "case-constitutional", command(store, principal()), TerminalOutcome.CANCELLATION
    )
    assert cancelled.state.value == "cancelled"
    assert (
        EpisodeStore(tmp_path / "episodes.sqlite3").load(cancelled.episode_id).digest
        == cancelled.digest
    )


def test_command_rejects_detached_grant_evidence_and_snapshot(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    kernel = principal()
    valid = command(store, kernel)
    with pytest.raises(AuthorityError, match="evidence binding"):
        CommandAuthority(
            kernel,
            AuthorityGrant(kernel.identity_digest, valid.grant.level, D),
            valid.evidence,
            valid.validation_digest,
            valid.policy_digest,
            valid.snapshot_digest,
            valid.expected_prior_episode_digest,
        )
    wrong_snapshot = CommandAuthority(
        kernel,
        valid.grant,
        valid.evidence,
        valid.validation_digest,
        valid.policy_digest,
        "f" * 64,
        valid.expected_prior_episode_digest,
    )
    with pytest.raises(AuthorityError, match="admitted snapshot"):
        ConstitutionalTransactionService(store).record_interpretation(
            "case-constitutional", wrong_snapshot, {"intent": "tamper"}
        )


def test_publication_rejects_policy_and_exact_output_tampering(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    service = ConstitutionalTransactionService(store)
    kernel = principal()
    service.record_interpretation(
        "case-constitutional", command(store, kernel), {"intent": "test"}
    )
    service.ground_candidate("case-constitutional", command(store, kernel))
    service.open_deliberation("case-constitutional", command(store, kernel))
    committed = service.commit_epistemic_artifact(
        "case-constitutional",
        command(store, kernel),
        claims=(ref("claim:tamper"),),
        evidence=(),
        derivations=(ref("derivation:tamper", "derivation.v1"),),
    )
    response = ref("response:tamper", "response-ir.v3")
    tampered = PublicationAuthorization(
        committed.digest,
        D,
        "f" * 64,
        1,
        D,
        canonical_digest({"different": "rendered text"}),
        D,
        D,
        kernel.identity_digest,
        kernel.release_digest,
    )
    with pytest.raises(AuthorityError, match="policy mismatch"):
        service.authorize_response_publication(
            "case-constitutional",
            command(store, kernel),
            authorization=tampered,
            response=response,
        )


def test_external_effect_requires_scoped_single_use_capability_and_receipt(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.sqlite3")
    admitted(store)
    service = ConstitutionalTransactionService(store)
    kernel = principal()
    service.record_interpretation(
        "case-constitutional", command(store, kernel), {"intent": "effect"}
    )
    service.ground_candidate("case-constitutional", command(store, kernel))
    service.open_deliberation("case-constitutional", command(store, kernel))
    service.commit_epistemic_artifact(
        "case-constitutional",
        command(store, kernel),
        claims=(ref("claim:effect"),),
        evidence=(),
        derivations=(ref("derivation:effect", "derivation.v1"),),
    )
    auth = command(store, kernel)
    issuer = CapabilityTokenIssuer()
    audit = AuditSink()
    now = datetime.now(timezone.utc)
    token = issuer.issue(
        principal=kernel,
        grant=auth.grant,
        operation=Operation.EXECUTE_EFFECT,
        episode_id="case-constitutional",
        resource_digest=D,
        expires_at=now + timedelta(minutes=1),
        audit=audit,
        clock=lambda: now,
    )
    effect = EffectAuthorization(
        D,
        D,
        token.token_digest,
        D,
        SNAPSHOT,
        kernel.identity_digest,
        kernel.release_digest,
    )
    service.authorize_effect("case-constitutional", auth, effect)
    auth = command(store, kernel)
    executed = service.record_execution(
        "case-constitutional",
        auth,
        ref("receipt:effect", "execution-receipt.v1"),
        authorization=effect,
        capability=token,
        issuer=issuer,
        resource_digest=D,
        clock=lambda: now,
    )
    assert executed.state.value == "executed"
    with pytest.raises(AuthorityError, match="replay"):
        issuer.consume(
            token=token,
            principal=kernel,
            operation=Operation.EXECUTE_EFFECT,
            episode_id="case-constitutional",
            resource_digest=D,
            now=now,
        )
    observed = service.record_observation(
        "case-constitutional",
        command(store, kernel),
        ref("observation:effect", "effect-observation.v1"),
    )
    final = service.consolidate(
        "case-constitutional",
        command(store, kernel),
        ref("consolidation:effect", "episode-consolidation.v1"),
    )
    assert observed.effects == final.effects
    assert final.state.value == "consolidated"
