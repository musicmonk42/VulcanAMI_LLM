import hashlib, os, time
from datetime import datetime, timezone, timedelta

from vulcan.world_model.meta_reasoning.self_improvement_drive import compose_self_improvement_drive
from vulcan.world_model.meta_reasoning.csiu_enforcement import METRIC_ORDER, CSIUMetricSnapshot, reset_csiu_enforcer
from vulcan.world_model.meta_reasoning.governed_transaction import (
    ApprovalStore, ImprovementPolicy, ImprovementProposal, VerificationGate,
    SCHEMA_VERSION, DISABLED_ENV, inspect_repository, TransactionStatus,
    ApprovalIssuer, ClosedApprovalVerifier,
)
from vulcan.runtime.audit import CanonicalAudit

class MutableVerifier(ClosedApprovalVerifier):
    def __init__(self):
        self.principals={}
        super().__init__(self.principals)
    def is_authorized(self, principal, bindings):
        self._principals=dict(self.principals)
        return super().is_authorized(principal, bindings)


def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def cfg():
    return {"drives":{"self_improvement":{"enabled":True,"priority":1,"objectives":[{"type":"bugfix","weight":1.0}],"constraints":{"require_human_approval":True,"governed_transactions_enabled":True,"unattended_application_permitted":True,"independent_approval_required":True,"max_changes_per_session":5},"triggers":[],"resource_limits":{}}}}

def mk_drive(tmp_path, gate):
    reset_csiu_enforcer()
    repo=tmp_path/'repo'; (repo/'.git').mkdir(parents=True); (repo/'src').mkdir(); (repo/'src/t.py').write_text('X=1\n')
    pol=ImprovementPolicy('auto-apply-policy/2',True,repo,('bugfix',),('src/*.py',),(),1,1000,10,True,{'deterministic':('rel',)},(gate,),5,2000,True)
    snap=inspect_repository(repo,('src/*.py',))
    prop=ImprovementProposal(SCHEMA_VERSION,'prop-1','bugfix','src/t.py',sha('X=1\n'),'X=1\n','X=2\n',sha('X=2\n'),snap.digest,'deterministic','rel','proof',pol.digest,'')
    policy_file=tmp_path/'auto_policy.json'
    policy_file.write_text('{"auto_apply": {"enabled": true}}', encoding='utf-8')
    os.environ['VULCAN_AUTO_APPLY_POLICY']=str(policy_file)
    verifier=MutableVerifier()
    store=ApprovalStore(tmp_path/'approvals.json')
    audit=CanonicalAudit(tmp_path/'audit.jsonl')
    drive=compose_self_improvement_drive(config_path=cfg(), state_path=str(tmp_path/'state.json'), improvement_policy=pol, approval_store=store, approval_verifier=verifier, audit_owner=audit)
    obj=drive.objectives[0]
    drive.should_trigger=lambda ctx: True
    drive.select_objective=lambda: obj
    drive.generate_improvement_action=lambda objective: {'id':'drive-plan','type':'improvement','governed_proposal':prop.__dict__.copy(),'objective_weights':{'bugfix':1.0}}
    base=datetime.now(timezone.utc)-timedelta(minutes=20); calls={'n':0}
    phase={'n':1}
    def provider(k):
        calls['n']+=1
        start=base-timedelta(minutes=5) if phase['n']==1 else base+timedelta(minutes=1)
        end=base if phase['n']==1 else base+timedelta(minutes=6)
        if k=='metrics.window_start': return start.isoformat().replace('+00:00','Z')
        if k=='metrics.window_end': return end.isoformat().replace('+00:00','Z')
        if k=='metrics.sample_count': return 30
        if k=='metrics.aggregation_method': return 'mean'
        if k=='metrics.metric_definition_version': return drive._csiu_enforcer.policy.metric_definition_version
        if k=='metrics.provider_id': return 'typed-provider'
        if k=='metrics.provenance_digest': return ('a' if phase['n']==1 else 'b')*64
        if phase['n']==1: return 0.5
        return 0.6 if k.endswith(("alignment_coherence_idx","intent_clarity_score","empathy_index","user_satisfaction")) else 0.4
    drive.set_metrics_provider(provider)
    d1=drive.observe_csiu_telemetry()
    assert d1.reason_code=="baseline_established"
    assert drive._csiu_enforcer.check_cumulative_influence()["cumulative_influence"]==0
    phase['n']=2
    return repo, drive, prop, verifier.principals

def issue_for(drive, approval_id, verifier_map):
    pending=next(a for a in drive.state.pending_approvals if a["approval_id"]==approval_id)
    bindings={"approval_id":approval_id,"proposal_digest":pending["proposal_digest"],"policy_digest":pending["policy_digest"],"original_source_digest":pending["original_source_digest"],"required_scope":"self_improvement.approve"}
    principal=ApprovalIssuer().issue_principal("independent-reviewer",("self_improvement.approve",),bindings=bindings,ttl_seconds=60)
    verifier_map[principal.principal_id]=principal
    return principal

def test_governed_drive_e2e_step_approval_resume_success(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','0'); monkeypatch.setenv(DISABLED_ENV,'0')
    repo, drive, prop, verifier_map = mk_drive(tmp_path, VerificationGate('pycompile',('python','-m','py_compile','src/t.py')))
    original=(repo/'src/t.py').read_bytes()
    action=drive.step({'force':True})
    assert action['status']=='pending_governed_approval' and action['_wait_for_approval'] is True
    assert action['approval_id'] and action['proposal_digest']==prop.digest()
    assert (repo/'src/t.py').read_bytes()==original
    assert drive.approve_governed_pending(action['approval_id'], issue_for(drive, action['approval_id'], verifier_map))
    resumed=drive.resume_governed_pending(action['approval_id'])
    assert resumed['status']==TransactionStatus.APPLIED_AND_VERIFIED.value
    assert (repo/'src/t.py').read_text()=='X=2\n'
    assert drive.state.improvements_this_session==1
    assert drive._last_governed_transaction_result.status==TransactionStatus.APPLIED_AND_VERIFIED
    assert drive._csiu_last_decision is not None
    assert (tmp_path/'state.json').exists() and (tmp_path/'audit.jsonl').exists()
    drive.audit_owner.readiness(); events=drive.audit_owner.events_for_proposal(prop.digest())
    lifecycle=[e.event_type for e in events]
    assert lifecycle[-6:]==["improvement.proposed","improvement.approved","improvement.apply_prepared","improvement.candidate_installed","improvement.gate_completed","improvement.applied"]
    assert all(e.data["proposal_digest"]==prop.digest() and e.data["policy_digest"]==drive.improvement_policy.digest and e.data["original_digest"]==prop.expected_original_sha256 and e.data["candidate_digest"]==prop.candidate_sha256 for e in events)
    drive.audit_owner.close(); reopened=CanonicalAudit(tmp_path/'audit.jsonl'); reopened.readiness(); assert len(reopened.events_for_proposal(prop.digest()))==len(events); reopened.close()
    assert drive.approval_store.load(action['approval_id']).state=='consumed'
    again=drive.resume_governed_pending(action['approval_id'])
    assert drive.state.improvements_this_session==1 and again['applied'] is False
    drive.audit_owner.close()

def test_governed_drive_e2e_step_approval_resume_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','0'); monkeypatch.setenv(DISABLED_ENV,'0')
    repo, drive, prop, verifier_map = mk_drive(tmp_path, VerificationGate('fail',('python','-c','import sys; sys.exit(1)')))
    action=drive.step({'force':True})
    assert action['status']=='pending_governed_approval'
    assert drive.approve_governed_pending(action['approval_id'], issue_for(drive, action['approval_id'], verifier_map))
    resumed=drive.resume_governed_pending(action['approval_id'])
    assert resumed['status']==TransactionStatus.VERIFICATION_FAILED_ROLLBACK_SUCCEEDED.value
    assert (repo/'src/t.py').read_text()=='X=1\n'
    assert drive.state.improvements_this_session==0
    assert len([a for a in drive.state.pending_approvals if a.get('status')=='pending_governed_approval'])==0
    drive.audit_owner.readiness(); events=drive.audit_owner.events_for_proposal(prop.digest())
    assert [e.event_type for e in events][-1]=="improvement.rollback_completed"
    drive.audit_owner.close()
