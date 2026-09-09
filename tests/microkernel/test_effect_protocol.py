from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import sqlite3

import pytest

from vulcan.microkernel.effects import (
    AuthorizedPolicy,
    CapabilityToken,
    EffectConflict,
    EffectCrash,
    EffectIntent,
    EffectOutcome,
    EffectRejected,
    EffectStore,
    EffectTransactionService,
    ExpectedEffect,
    PolicyProposal,
    ReversibleSandbox,
)
from vulcan.microkernel.principals import Principal, PrincipalKind


D = sha256(b"fixture").hexdigest()
NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def principal(kind: PrincipalKind = PrincipalKind.SYSTEM_KERNEL) -> Principal:
    return Principal(kind, f"{kind.value}-one", sha256(b"release").hexdigest())


def artifacts(
    kernel: Principal,
    *,
    branch: str = "branch-primary",
    episode: str = "case-one",
    before: str | None = None,
):
    value = "Ada"
    proposal = PolicyProposal(
        "proposal-one",
        episode,
        "lineage-primary",
        branch,
        "kv/name",
        "put",
        sha256(value.encode()).hexdigest(),
    )
    policy = AuthorizedPolicy(
        "policy-one", proposal.digest, kernel.identity_digest, kernel.release_digest, 1
    )
    expected = ExpectedEffect(
        "kv/name",
        "put",
        sha256(
            b'{"resource":"kv/name","value":null}'
            if before is None
            else f'{{"resource":"kv/name","value":"{before}"}}'.encode()
        ).hexdigest(),
        sha256(b'{"resource":"kv/name","value":"Ada"}').hexdigest(),
        True,
    )
    intent = EffectIntent(
        "intent-one",
        kernel.identity_digest,
        kernel.release_digest,
        "lineage-primary",
        branch,
        D,
        episode,
        D,
        policy.digest,
        expected.digest,
        "kv/name",
        "put",
        value,
        1,
        "idem-one",
    )
    return proposal, policy, expected, intent


def service(store, kernel, sandbox=None, *, clock=lambda: NOW, validator=None):
    return EffectTransactionService(
        store,
        kernel,
        sandbox or ReversibleSandbox(),
        clock=clock,
        validate_heads=validator or (lambda *args: None),
    )


def authorize(tx, values, *, expiry=NOW + timedelta(hours=1), nonce="nonce-one"):
    proposal, policy, expected, intent = values
    return tx.authorize(
        proposal, policy, expected, intent, expires_at=expiry, nonce=nonce
    )


def test_contracts_preserve_authority_distinctions(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    proposal, policy, expected, intent = values
    tx = service(EffectStore(tmp_path / "effects.sqlite3"), kernel)
    token = authorize(tx, values)
    assert proposal.authority == "untrusted_proposal"
    assert policy.proposal_digest == proposal.digest
    assert token.effect_digest == intent.expected_effect_digest == expected.digest
    with pytest.raises(ValueError, match="cannot carry authority"):
        replace(proposal, authority="authorized_plan")
    with pytest.raises(ValueError, match="allowlisted"):
        replace(intent, operation="shell")


def test_authorization_validates_exact_episode_and_lineage_heads(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    calls = []
    tx = service(
        EffectStore(tmp_path / "effects.sqlite3"),
        kernel,
        validator=lambda *args: calls.append(args),
    )
    authorize(tx, values)
    assert calls == [("case-one", D, "lineage-primary", "branch-primary", D)]


def test_stale_authority_head_fails_before_capability_is_issued(tmp_path):
    kernel = principal()
    values = artifacts(kernel)

    def reject(*args):
        raise EffectRejected("stale episode or lineage head")

    store = EffectStore(tmp_path / "effects.sqlite3")
    with pytest.raises(EffectRejected, match="stale"):
        authorize(service(store, kernel, validator=reject), values)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM effect_capabilities").fetchone()[0] == 0


@pytest.mark.parametrize("artifact", ["proposal", "policy", "expected", "intent"])
def test_authorization_rejects_mismatched_authority_chain(tmp_path, artifact):
    kernel = principal()
    proposal, policy, expected, intent = artifacts(kernel)
    if artifact == "proposal":
        proposal = replace(proposal, resource="kv/other")
    elif artifact == "policy":
        policy = replace(policy, budget=2)
    elif artifact == "expected":
        expected = replace(expected, resource="kv/other")
    else:
        intent = replace(intent, value="model supplied mismatch")
    tx = service(EffectStore(tmp_path / "effects.sqlite3"), kernel)
    with pytest.raises(EffectRejected, match="mismatch"):
        authorize(tx, (proposal, policy, expected, intent))


def test_unissued_capability_cannot_execute(tmp_path):
    kernel = principal()
    proposal, policy, expected, intent = artifacts(kernel)
    token = CapabilityToken(
        "forged",
        kernel.identity_digest,
        kernel.release_digest,
        intent.lineage_id,
        intent.branch_id,
        intent.episode_id,
        policy.digest,
        expected.digest,
        intent.resource,
        intent.operation,
        (NOW + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "forged-nonce",
        intent.budget,
        intent.idempotency_key,
    )
    tx = service(EffectStore(tmp_path / "effects.sqlite3"), kernel)
    with pytest.raises(EffectRejected, match="not issued"):
        tx.execute(intent, token)


def test_authorization_evidence_tamper_fails_on_restart(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    path = tmp_path / "effects.sqlite3"
    authorize(service(EffectStore(path), kernel), values)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE effect_authorizations SET expected_document='{}'")
    with pytest.raises(Exception, match="authorization evidence"):
        EffectStore(path)


def test_durable_order_and_audit_bindings(tmp_path):
    events = []
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]
    store = EffectStore(
        tmp_path / "effects.sqlite3",
        outbox_sink=lambda topic, body: events.append((topic, body)),
    )
    tx = service(store, kernel)
    receipt = tx.execute(intent, authorize(tx, values))
    assert receipt.outcome is EffectOutcome.SUCCEEDED
    assert store.state(intent.digest) == "succeeded"
    assert [topic for topic, _ in events] == [
        "effect.capability.issued",
        "effect.intent.committed",
        "effect.attempt.started",
        "effect.receipt.committed",
    ]
    assert events[1][1]["episode_id"] == intent.episode_id
    assert events[1][1]["lineage_id"] == intent.lineage_id


def test_capability_is_single_use_and_concurrency_safe(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]
    sandbox = ReversibleSandbox()
    tx = service(EffectStore(tmp_path / "effects.sqlite3"), kernel, sandbox)
    token = authorize(tx, values)

    def consume():
        try:
            return tx.execute(intent, token).outcome
        except (EffectConflict, EffectRejected):
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: consume(), range(2)))
    assert outcomes.count(EffectOutcome.SUCCEEDED) == 1
    assert outcomes.count("conflict") == 1
    assert sandbox.values == {"kv/name": "Ada"}


@pytest.mark.parametrize(
    "changed",
    [
        "resource",
        "episode_id",
        "branch_id",
        "release_digest",
        "policy_digest",
        "effect_digest",
        "budget",
        "idempotency_key",
    ],
)
def test_wrong_capability_binding_fails_before_intent(tmp_path, changed):
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]
    store = EffectStore(tmp_path / "effects.sqlite3")
    tx = service(store, kernel)
    token = authorize(tx, values)
    replacement = (
        D if changed.endswith("digest") else (2 if changed == "budget" else "wrong")
    )
    forged = replace(token, **{changed: replacement})
    with pytest.raises(EffectRejected):
        tx.execute(intent, forged)
    with pytest.raises(Exception, match="unknown effect intent"):
        store.state(intent.digest)


def test_expiry_missing_head_authority_and_non_kernel_fail_closed(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    store = EffectStore(tmp_path / "effects.sqlite3")
    tx = service(store, kernel)
    token = authorize(tx, values)
    expired = service(store, kernel, clock=lambda: NOW + timedelta(days=1))
    with pytest.raises(EffectRejected, match="expiry"):
        expired.execute(values[-1], token)
    with pytest.raises(EffectRejected, match="validator"):
        EffectTransactionService(store, kernel, ReversibleSandbox())
    with pytest.raises(EffectRejected, match="SYSTEM_KERNEL"):
        service(store, principal(PrincipalKind.LANGUAGE_PROVIDER))


@pytest.mark.parametrize(
    "boundary",
    [
        "after_intent_before_commit",
        "after_intent_commit",
        "after_attempt_before_commit",
        "after_attempt_commit",
        "after_external_execution",
        "after_receipt_before_commit",
    ],
)
def test_crash_at_every_execution_boundary_is_closed_or_ambiguous(tmp_path, boundary):
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]

    def crash(name):
        if name == boundary:
            raise EffectCrash(name)

    path = tmp_path / "effects.sqlite3"
    store = EffectStore(path, failpoint=crash)
    tx = service(store, kernel)
    token = authorize(tx, values)
    with pytest.raises(EffectCrash):
        tx.execute(intent, token)
    restarted = EffectStore(path)
    if boundary in {
        "after_attempt_commit",
        "after_external_execution",
        "after_receipt_before_commit",
    }:
        assert restarted.mark_ambiguous_attempts() == 1
        assert restarted.state(intent.digest) == "ambiguous"
    elif boundary in {"after_intent_commit", "after_attempt_before_commit"}:
        assert restarted.state(intent.digest) == "authorized"
    else:
        with pytest.raises(Exception, match="unknown effect intent"):
            restarted.state(intent.digest)


def test_crash_during_capability_issue_rolls_back(tmp_path):
    kernel = principal()
    values = artifacts(kernel)

    def crash(name):
        if name == "after_capability_before_commit":
            raise EffectCrash(name)

    path = tmp_path / "effects.sqlite3"
    with pytest.raises(EffectCrash):
        authorize(service(EffectStore(path, failpoint=crash), kernel), values)
    # The same nonce can be issued after restart because the failed transaction vanished.
    authorize(service(EffectStore(path), kernel), values)


def test_idempotent_target_records_ambiguity_without_blind_retry(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]
    calls = 0
    sandbox = ReversibleSandbox(idempotent=True)

    def crash(name):
        nonlocal calls
        if name == "after_external_execution":
            calls += 1
            raise EffectCrash(name)

    path = tmp_path / "effects.sqlite3"
    tx = service(EffectStore(path, failpoint=crash), kernel, sandbox)
    token = authorize(tx, values)
    with pytest.raises(EffectCrash):
        tx.execute(intent, token)
    assert sandbox.values["kv/name"] == "Ada" and calls == 1
    service(EffectStore(path), kernel, sandbox)
    assert EffectStore(path).state(intent.digest) == "ambiguous" and calls == 1


def test_non_idempotent_ambiguity_requires_operator_reconciliation(tmp_path):
    kernel = principal()
    values = artifacts(kernel)
    intent = values[-1]
    path = tmp_path / "effects.sqlite3"
    sandbox = ReversibleSandbox(idempotent=False)

    def crash(name):
        if name == "after_external_execution":
            raise EffectCrash(name)

    tx = service(EffectStore(path, failpoint=crash), kernel, sandbox)
    token = authorize(tx, values)
    with pytest.raises(EffectCrash):
        tx.execute(intent, token)
    restarted = service(EffectStore(path), kernel, sandbox)
    with pytest.raises(EffectRejected, match="operator"):
        restarted.reconcile(
            intent.digest,
            observed_digest=D,
            operator=principal(PrincipalKind.AUDITOR),
            succeeded=True,
        )
    restarted.reconcile(
        intent.digest,
        observed_digest=D,
        operator=principal(PrincipalKind.OPERATOR),
        succeeded=True,
    )
    assert restarted.store.state(intent.digest) == "succeeded"


def test_reversible_effect_can_be_compensated(tmp_path):
    kernel = principal()
    values = artifacts(kernel, before="Grace")
    intent = values[-1]
    sandbox = ReversibleSandbox()
    sandbox.values["kv/name"] = "Grace"
    tx = service(EffectStore(tmp_path / "effects.sqlite3"), kernel, sandbox)
    receipt = tx.execute(intent, authorize(tx, values))
    compensation = tx.compensate(intent, receipt, "Grace")
    assert compensation.outcome is EffectOutcome.COMPENSATED
    assert sandbox.values["kv/name"] == "Grace"
    assert tx.store.state(intent.digest) == "compensated"
