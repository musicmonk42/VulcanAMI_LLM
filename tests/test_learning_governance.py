from datetime import datetime, timedelta, timezone
import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest

from vulcan.learning_bandit import ACTIONS, ShadowLinUCBToolBandit
from vulcan.learning_governance import (
    ActivationStatus,
    AlignmentApprovalVerifier,
    DurableInfluenceLedger,
    GovernedLearningActivator,
    issue_alignment_approval,
)
from vulcan.learning_observation import ObservationContext, ProvenanceType, TerminalStatus, construct_observation, digest_json
from vulcan.runtime.audit import CanonicalAudit, AuditError


def h(x): return digest_json({"x": x})

def make_obs(bandit, label, tool="graphix_retrieval", status=TerminalStatus.VALIDATED_SUCCESS):
    now=datetime.now(timezone.utc).replace(microsecond=123456)
    dist={a: round(1/len(ACTIONS), 12) for a in ACTIONS}
    ctx=ObservationContext(case_id=f"case-{label}", case_digest=h("case"+str(label)), request_digest=h("req"+str(label)), tenant_digest=h("tenant"), alignment_revision=1, alignment_digest=h("align"), csiu_policy_digest=h("csiu-p"), csiu_snapshot_digest=h("csiu-s"), domain_snapshot_digest=h("domain"), runtime_owner_id="learning-owner", acquisition_time=now)
    ob, elig=construct_observation(context=ctx, selected_plan_digest=h("plan"+str(label)), selected_tool_id=tool, selection_distribution_digest=bandit.distribution_digest(dist), action_propensity=dist[tool], terminal_status=status, ledger_digest=h("ledger"+str(label)), evidence_digest=h("evidence"+str(label)), provenance_type=ProvenanceType.DERIVATION, terminal_case_validated=True, ledger_validated=True, evidence_integrity_validated=True, bindings_match=True, alignment_matches_lease=True, csiu_bindings_valid=True, clock=lambda: now)
    assert elig.status.value == "eligible_positive" or status is not TerminalStatus.VALIDATED_SUCCESS
    return ob


def trained_bandit(n=8):
    b=ShadowLinUCBToolBandit(alpha=0.1)
    obs=[]
    for i in range(n):
        ob=make_obs(b, f"train-{i}", "graphix_retrieval")
        b.select_shadow(ob); b.update_from_observation(ob); obs.append(ob.canonical_observation_digest)
    probe=make_obs(b, "probe", "graphix_retrieval")
    b.select_shadow(probe); obs.append(probe.canonical_observation_digest)
    return b, obs


def activator(tmp_path, bandit, *, single=0.5, cumulative=0.8):
    audit=CanonicalAudit(tmp_path/"audit.jsonl")
    ledger=DurableInfluenceLedger(tmp_path/"influence.json", single_cap=single, cumulative_cap=cumulative)
    return GovernedLearningActivator(bandit=bandit, audit=audit, ledger=ledger, verifier=AlignmentApprovalVerifier())


def approval_for(proposal, seconds=60, issuer="alignment-reviewer"):
    exp=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(timespec="microseconds").replace("+00:00","Z")
    return issue_alignment_approval(proposal=proposal, issuer_id=issuer, expires_at_utc=exp)


def proposal_for(act, obs):
    return act.propose(alignment_revision=1, alignment_digest=h("align"), csiu_policy_digest=h("csiu-p"), csiu_snapshot_digest=h("csiu-s"), observation_digests=obs)


def test_candidate_within_cap_valid_review_activates_exactly_once(tmp_path):
    b, obs=trained_bandit(4); act=activator(tmp_path,b,single=1.0,cumulative=1.0)
    p=proposal_for(act, obs); a=approval_for(p)
    r1=act.activate(p,a); r2=act.activate(p,a)
    assert r1.status is ActivationStatus.ACTIVATED
    assert r2.status is ActivationStatus.ACTIVATED
    assert r1.active_policy_digest == p.candidate_policy_digest
    assert b.active_policy_digest == p.candidate_policy_digest
    assert act.ledger.consumed() == pytest.approx(p.charged_influence)


def test_candidate_above_single_cap_is_blocked(tmp_path):
    b, obs=trained_bandit(3); act=activator(tmp_path,b,single=0.0001,cumulative=1.0)
    p=proposal_for(act, obs); a=approval_for(p)
    with pytest.raises(ValueError, match="single influence cap"):
        act.activate(p,a)
    assert b.active_policy_digest == p.active_policy_digest


def test_multiple_candidates_exceed_cumulative_cap_and_restart_preserves_budget(tmp_path):
    b, obs=trained_bandit(3); act=activator(tmp_path,b,single=1.0,cumulative=0.6)
    p=proposal_for(act, obs); act.activate(p, approval_for(p))
    used=act.ledger.consumed(); act.audit.close()
    b2, obs2=trained_bandit(3); act2=activator(tmp_path,b2,single=1.0,cumulative=used + 0.000001)
    p2=proposal_for(act2, obs2)
    with pytest.raises(ValueError, match="cumulative influence cap"):
        act2.activate(p2, approval_for(p2))
    assert act2.ledger.consumed() == pytest.approx(used)


def test_concurrent_activation_race_produces_one_winner(tmp_path):
    b, obs=trained_bandit(5); act=activator(tmp_path,b,single=1.0,cumulative=1.0)
    p=proposal_for(act, obs)
    approvals=[approval_for(p, issuer=f"reviewer-{i}") for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda ap: act.activate(p, ap), approvals))
    assert sum(r.status is ActivationStatus.ACTIVATED for r in results) == 2
    assert act.ledger.consumed() == pytest.approx(p.charged_influence)


def test_stale_active_revision_fails_cas(tmp_path):
    b, obs=trained_bandit(2); act=activator(tmp_path,b,single=1.0,cumulative=1.0)
    p=proposal_for(act, obs)
    b.activate_candidate(expected_active_digest=p.active_policy_digest, expected_candidate_digest=p.candidate_policy_digest)
    r=act.activate(p, approval_for(p))
    assert r.status is ActivationStatus.STALE


def test_altered_reused_and_expired_approvals_fail(tmp_path):
    b, obs=trained_bandit(2); act=activator(tmp_path,b,single=1.0,cumulative=1.0)
    p=proposal_for(act, obs)
    altered=dataclasses.replace(approval_for(p), charged_influence=p.charged_influence/2, approval_digest="")
    with pytest.raises(ValueError, match="approval binding"):
        act.activate(p, altered)
    expired=approval_for(p, seconds=-1)
    with pytest.raises(ValueError, match="approval expired"):
        act.activate(p, expired)
    used=approval_for(p, issuer="reviewer-used")
    assert act.activate(p, used).status is ActivationStatus.ACTIVATED
    with pytest.raises(ValueError, match="approval reused"):
        act.verifier.verify(used, p)


def test_learning_owner_cannot_issue_approval(tmp_path):
    b, obs=trained_bandit(1); act=activator(tmp_path,b,single=1.0,cumulative=1.0); p=proposal_for(act, obs)
    with pytest.raises(ValueError, match="learning owner cannot issue"):
        approval_for(p, issuer="learning-owner-1")


def test_audit_failure_publishes_nothing(tmp_path):
    class FailingAudit(CanonicalAudit):
        def append(self, t, d): raise AuditError("audit down")
    b, obs=trained_bandit(1); audit=FailingAudit(tmp_path/"audit.jsonl"); ledger=DurableInfluenceLedger(tmp_path/"i.json", single_cap=1, cumulative_cap=1)
    act=GovernedLearningActivator(bandit=b, audit=audit, ledger=ledger, verifier=AlignmentApprovalVerifier())
    p=proposal_for(act, obs)
    with pytest.raises(AuditError): act.activate(p, approval_for(p))
    assert b.active_policy_digest == p.active_policy_digest
    assert ledger.consumed() == 0


def test_distribution_change_recomputed_and_underreporting_detected(tmp_path):
    b, obs=trained_bandit(3); act=activator(tmp_path,b,single=1.0,cumulative=1.0)
    p=proposal_for(act, obs)
    under=dataclasses.replace(p, charged_influence=0.0, proposal_digest="")
    r=act.activate(under, approval_for(under))
    assert r.status is ActivationStatus.BLOCKED
    assert "underreporting" in r.reason


def test_mathematical_compatibility_methods_do_not_mutate_private_weights():
    pytest.importorskip("numpy")
    from vulcan.learning.mathematical_accuracy_integration import MathematicalAccuracyIntegration
    class LS:
        def __init__(self): self.tool_weight_adjustments={"graphix_arithmetic": 0.1}
    m=MathematicalAccuracyIntegration(); ls=LS(); before=dict(ls.tool_weight_adjustments)
    m.reward_tool("graphix_arithmetic", ls)
    assert ls.tool_weight_adjustments == before
