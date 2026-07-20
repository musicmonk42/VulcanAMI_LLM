from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from vulcan.microkernel.authority import AuthorityError, AuthorityLevel, AuditSink, EvidenceRecord, Operation, promote_authority, require_authority
from vulcan.microkernel.capability_tokens import CapabilityToken, CapabilityTokenIssuer
from vulcan.microkernel.principals import Principal, PrincipalKind

RELEASE = "a" * 64
RESOURCE = "b" * 64
POLICY = "c" * 64
VALIDATION = "d" * 64


def now():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def principal(kind=PrincipalKind.SYSTEM_KERNEL, pid="kernel"):
    return Principal(kind=kind, principal_id=pid, release_digest=RELEASE, metadata={"role": "admin"})


def evidence(p=None):
    p = p or principal()
    return EvidenceRecord(p.identity_digest, VALIDATION, POLICY, now(), ("artifact:validation",))


def grant(level=AuthorityLevel.EXECUTED_EFFECT, p=None):
    p = p or principal()
    return promote_authority(current=AuthorityLevel.UNTRUSTED_PROPOSAL, target=level, principal=p, evidence=evidence(p))


def test_principal_kinds_are_explicit_and_metadata_does_not_grant_authority():
    assert {kind.name for kind in PrincipalKind} == {"HUMAN", "SYSTEM_KERNEL", "LANGUAGE_PROVIDER", "REASONER", "RETRIEVER", "TOOL", "POLICY_AUTHORITY", "OPERATOR", "AUDITOR", "EXTERNAL_PROVIDER"}
    attacker = principal(PrincipalKind.LANGUAGE_PROVIDER, "llm")
    with pytest.raises(AuthorityError):
        promote_authority(current=AuthorityLevel.UNTRUSTED_PROPOSAL, target=AuthorityLevel.EXECUTED_EFFECT, principal=attacker, evidence=evidence())


def test_monotonic_promotion_and_complete_evidence_required():
    p = principal()
    g = promote_authority(current=AuthorityLevel.VALIDATED_CANDIDATE, target=AuthorityLevel.COMMITTED_BELIEF, principal=p, evidence=evidence(p))
    assert g.level is AuthorityLevel.COMMITTED_BELIEF
    with pytest.raises(AuthorityError):
        promote_authority(current=AuthorityLevel.AUTHORIZED_PLAN, target=AuthorityLevel.COMMITTED_BELIEF, principal=p, evidence=evidence(p))
    with pytest.raises(ValueError):
        EvidenceRecord(p.identity_digest, "bad", POLICY, now())


def test_default_deny_unknown_operation_and_low_authority_effect():
    p = principal()
    audit = AuditSink()
    with pytest.raises(AuthorityError):
        require_authority(principal=p, grant=grant(AuthorityLevel.VALIDATED_CANDIDATE, p), operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, audit=audit, clock=now)
    assert audit.events[-1].decision == "denied"
    with pytest.raises(AuthorityError):
        require_authority(principal=p, grant=grant(AuthorityLevel.EXECUTED_EFFECT, p), operation="metadata.says.admin", episode_id="e1", resource_digest=RESOURCE, audit=audit, clock=now)


def test_high_risk_grants_are_audited_without_raw_resource_or_secret():
    p = principal()
    audit = AuditSink()
    require_authority(principal=p, grant=grant(AuthorityLevel.EXECUTED_EFFECT, p), operation=Operation.EXECUTE_EFFECT, episode_id="episode-1", resource_digest=RESOURCE, audit=audit, clock=now)
    event = audit.events[-1].to_json()
    assert event["decision"] == "granted"
    assert event["resource_digest"] == RESOURCE
    assert "secret" not in str(event).lower()


def test_token_scope_expiry_replay_and_serialization_rejection():
    p = principal()
    issuer = CapabilityTokenIssuer()
    audit = AuditSink()
    token = issuer.issue(principal=p, grant=grant(AuthorityLevel.EXECUTED_EFFECT, p), operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, expires_at=now() + timedelta(minutes=5), audit=audit, clock=now)
    with pytest.raises(AuthorityError):
        CapabilityToken.from_json(token.to_json())
    with pytest.raises(AuthorityError):
        issuer.consume(token=token, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="wrong", resource_digest=RESOURCE, now=now())
    with pytest.raises(AuthorityError):
        issuer.consume(token=token, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest="e" * 64, now=now())
    issuer.consume(token=token, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, now=now())
    with pytest.raises(AuthorityError):
        issuer.consume(token=token, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, now=now())
    expired = issuer.issue(principal=p, grant=grant(AuthorityLevel.EXECUTED_EFFECT, p), operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, expires_at=now(), audit=audit, clock=now)
    with pytest.raises(AuthorityError):
        issuer.consume(token=expired, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, now=now())


def test_principal_spoofing_and_confused_deputy_denied():
    kernel = principal()
    spoof = principal(PrincipalKind.TOOL, "kernel")
    issuer = CapabilityTokenIssuer()
    audit = AuditSink()
    token = issuer.issue(principal=kernel, grant=grant(AuthorityLevel.EXECUTED_EFFECT, kernel), operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, expires_at=now() + timedelta(minutes=1), audit=audit, clock=now)
    with pytest.raises(AuthorityError):
        issuer.consume(token=token, principal=spoof, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, now=now())


def test_concurrent_token_consume_allows_exactly_one_effect_owner():
    p = principal()
    issuer = CapabilityTokenIssuer()
    token = issuer.issue(principal=p, grant=grant(AuthorityLevel.EXECUTED_EFFECT, p), operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, expires_at=now() + timedelta(minutes=1), audit=AuditSink(), clock=now)

    def consume_once():
        try:
            issuer.consume(token=token, principal=p, operation=Operation.EXECUTE_EFFECT, episode_id="e1", resource_digest=RESOURCE, now=now())
            return True
        except AuthorityError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: consume_once(), range(16)))
    assert results.count(True) == 1
    assert results.count(False) == 15
