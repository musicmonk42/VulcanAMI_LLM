import hashlib, os, time
from datetime import datetime, timezone, timedelta

import json
from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive
from vulcan.world_model.meta_reasoning.csiu_enforcement import METRIC_ORDER, CSIUMetricSnapshot, reset_csiu_enforcer
from vulcan.world_model.meta_reasoning.governed_transaction import (
    ApprovalStore, ImprovementPolicy, ImprovementProposal, VerificationGate,
    SCHEMA_VERSION, DISABLED_ENV, inspect_repository, TransactionStatus,
)


def sha(s): return hashlib.sha256(s.encode()).hexdigest()

class TrustedAuthority:
    def is_authorized(self, principal, approval_id):
        return principal == "independent-reviewer" and bool(approval_id)

class AuditAdapter:
    def __init__(self, path):
        self.path=path; self.owner_id=f"runtime-audit-adapter:{path}"; path.write_text("")
    def record_event(self, event, payload):
        self.path.write_text(self.path.read_text()+json.dumps({"event":event,"payload":payload},sort_keys=True,default=str)+"\n")
    def close(self): pass

def cfg():
    return {"drives":{"self_improvement":{"enabled":True,"priority":1,"objectives":[{"type":"bugfix","weight":1.0}],"constraints":{"require_human_approval":True,"max_changes_per_session":5},"triggers":[],"resource_limits":{}}}}

def mk_drive(tmp_path, gate):
    reset_csiu_enforcer()
    repo=tmp_path/'repo'; (repo/'.git').mkdir(parents=True); (repo/'src').mkdir(); (repo/'src/t.py').write_text('X=1\n')
    pol=ImprovementPolicy('auto-apply-policy/2',True,repo,('bugfix',),('src/*.py',),(),1,1000,10,True,{'deterministic':('rel',)},(gate,),5,2000,True)
    snap=inspect_repository(repo,('src/*.py',))
    prop=ImprovementProposal(SCHEMA_VERSION,'prop-1','bugfix','src/t.py',sha('X=1\n'),'X=1\n','X=2\n',sha('X=2\n'),snap.digest,'deterministic','rel','proof',pol.digest,'')
    drive=SelfImprovementDrive(config_path=cfg(), state_path=str(tmp_path/'state.json'))
    drive.improvement_policy=pol; drive.approval_store=ApprovalStore(tmp_path/'approvals.json'); drive.approval_authority=TrustedAuthority(); drive.audit_owner=AuditAdapter(tmp_path/'audit.jsonl'); drive._auto_apply_enabled=True
    obj=drive.objectives[0]
    drive.should_trigger=lambda ctx: True
    drive.select_objective=lambda: obj
    drive.generate_improvement_action=lambda objective: {'id':'drive-plan','type':'improvement','governed_proposal':prop.__dict__.copy(),'objective_weights':{'bugfix':1.0}}
    base=datetime.now(timezone.utc)-timedelta(minutes=20); calls={'n':0}
    drive._csiu_previous_snapshot = CSIUMetricSnapshot(metrics={k:0.5 for k in METRIC_ORDER}, window_start=(base-timedelta(minutes=5)).isoformat().replace('+00:00','Z'), window_end=base.isoformat().replace('+00:00','Z'), sample_count=30, aggregation_method='mean', metric_definition_version=drive._csiu_enforcer.policy.metric_definition_version, provider_id='typed-provider', provenance_digest='a'*64, policy_digest=drive._csiu_enforcer.policy.policy_digest, privacy_cohort={"kind":"aggregate"})
    def provider(k):
        calls['n']+=1
        if k=='metrics.window_start': return (base+timedelta(minutes=6)-timedelta(minutes=5)).isoformat().replace('+00:00','Z')
        if k=='metrics.window_end': return (base+timedelta(minutes=6)).isoformat().replace('+00:00','Z')
        if k=='metrics.sample_count': return 30
        if k=='metrics.aggregation_method': return 'mean'
        if k=='metrics.metric_definition_version': return drive._csiu_enforcer.policy.metric_definition_version
        if k=='metrics.provider_id': return 'typed-provider'
        if k=='metrics.provenance_digest': return 'b'*64
        return 0.6 if k.endswith(("alignment_coherence_idx","intent_clarity_score","empathy_index","user_satisfaction")) else 0.4
    drive.set_metrics_provider(provider)
    return repo, drive, prop

def test_governed_drive_e2e_step_approval_resume_success(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','0'); monkeypatch.setenv(DISABLED_ENV,'0')
    repo, drive, prop = mk_drive(tmp_path, VerificationGate('pycompile',('python','-m','py_compile','src/t.py')))
    original=(repo/'src/t.py').read_bytes()
    action=drive.step({'force':True})
    assert action['status']=='pending_governed_approval' and action['_wait_for_approval'] is True
    assert action['approval_id'] and action['proposal_digest']==prop.digest()
    assert (repo/'src/t.py').read_bytes()==original
    assert drive.approve_governed_pending(action['approval_id'],'independent-reviewer')
    resumed=drive.resume_governed_pending(action['approval_id'])
    assert resumed['status']==TransactionStatus.APPLIED_AND_VERIFIED.value
    assert (repo/'src/t.py').read_text()=='X=2\n'
    assert drive.state.improvements_this_session==1
    assert drive._last_governed_transaction_result.status==TransactionStatus.APPLIED_AND_VERIFIED
    assert drive._csiu_last_decision is not None
    assert (tmp_path/'state.json').exists() and (tmp_path/'audit.jsonl').exists()
    assert drive.approval_store.load(action['approval_id']).state=='consumed'
    again=drive.resume_governed_pending(action['approval_id'])
    assert drive.state.improvements_this_session==1 and again['applied'] is False
    drive.audit_owner.close()

def test_governed_drive_e2e_step_approval_resume_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv('INTRINSIC_CSIU_OFF','0'); monkeypatch.setenv(DISABLED_ENV,'0')
    repo, drive, prop = mk_drive(tmp_path, VerificationGate('fail',('python','-c','import sys; sys.exit(1)')))
    action=drive.step({'force':True})
    assert action['status']=='pending_governed_approval'
    assert drive.approve_governed_pending(action['approval_id'],'independent-reviewer')
    resumed=drive.resume_governed_pending(action['approval_id'])
    assert resumed['status']==TransactionStatus.VERIFICATION_FAILED_ROLLBACK_SUCCEEDED.value
    assert (repo/'src/t.py').read_text()=='X=1\n'
    assert drive.state.improvements_this_session==0
    assert len([a for a in drive.state.pending_approvals if a.get('status')=='pending_governed_approval'])==0
    drive.audit_owner.close()
