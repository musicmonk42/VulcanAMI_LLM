import hashlib,json,os,time,threading
from pathlib import Path
from vulcan.world_model.meta_reasoning.governed_transaction import *
from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
class Audit:
    def __init__(self): self.events=[]
    def record_event(self,e,p): self.events.append((e,p))

def mk(tmp_path):
    repo=tmp_path/'r'; (repo/'.git').mkdir(parents=True); (repo/'src').mkdir(); (repo/'src/t.py').write_text('X=1\n')
    pol=ImprovementPolicy('auto-apply-policy/2',True,repo,('bugfix',),('src/*.py',),(),1,1000,10,True,{'human':('rel',)},(VerificationGate('ok',('python','-m','py_compile','src/t.py')),),5,2000,True,digest='pd')
    snap=inspect_repository(repo,('src/*.py',))
    p=ImprovementProposal(SCHEMA_VERSION,'p','bugfix','src/t.py',sha('X=1\n'),'X=1\n','X=2\n',sha('X=2\n'),snap.digest,'human','rel','why',pol.digest,'')
    return repo,pol,snap,p

def test_approval_digest_ignores_server_envelope_and_reuse_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLED_ENV,'0')
    repo,pol,snap,p=mk(tmp_path)
    aid='srv-1'; bound=p.digest(); p2=ImprovementProposal(**{**p.__dict__,'approval_id':aid})
    assert p2.digest()==bound
    store=ApprovalStore(tmp_path/'a.json'); store.save(ApprovalRecord(aid,bound,pol.digest,p.expected_original_sha256,'independent',time.time(),time.time()+60))
    r=GovernedSelfImprovementTransaction(pol,Audit(),store).apply(p2,snap)
    assert r.status==TransactionStatus.APPLIED_AND_VERIFIED and r.proposal_digest==bound
    (repo/'src/t.py').write_text('X=1\n')
    assert GovernedSelfImprovementTransaction(pol,Audit(),store).apply(p2,snap).status==TransactionStatus.REJECTED_BEFORE_INSTALLATION
    bad=ImprovementProposal(**{**p2.__dict__,'candidate_content':'X=3\n','candidate_sha256':sha('X=3\n')})
    assert GovernedSelfImprovementTransaction(pol,Audit(),store).apply(bad,snap).status==TransactionStatus.REJECTED_BEFORE_INSTALLATION

def test_concurrent_claim_exactly_one(tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLED_ENV,'0')
    repo,pol,snap,p=mk(tmp_path); p=ImprovementProposal(**{**p.__dict__,'approval_id':'a'})
    store=ApprovalStore(tmp_path/'a.json'); store.save(ApprovalRecord('a',p.digest(),pol.digest,p.expected_original_sha256,'independent',time.time(),time.time()+60))
    res=[]
    def worker(): res.append(GovernedSelfImprovementTransaction(pol,Audit(),ApprovalStore(tmp_path/'a.json')).apply(p,snap).status)
    ts=[threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert res.count(TransactionStatus.APPLIED_AND_VERIFIED)==1

def test_trusted_approval_principal_only_and_bound(tmp_path):
    d=SelfImprovementDrive(config_path={"drives":{"self_improvement":{"enabled":True,"objectives":[{"type":"bugfix","weight":1.0}],"constraints":{},"triggers":[],"resource_limits":{}}}}, state_path=str(tmp_path/"s.json"))
    d.approval_store=ApprovalStore(tmp_path/"approvals.json")
    p_digest=sha("proposal"); pol_digest=sha("policy"); src_digest=sha("source"); aid="a"
    d.state.pending_approvals.append({"approval_id":aid,"status":"pending_governed_approval","proposal_digest":p_digest,"policy_digest":pol_digest,"original_source_digest":src_digest})
    bindings={"approval_id":aid,"proposal_digest":p_digest,"policy_digest":pol_digest,"original_source_digest":src_digest,"required_scope":"self_improvement.approve"}
    issuer=ApprovalIssuer()
    good=issuer.issue_principal("reviewer",("self_improvement.approve",),bindings=bindings,ttl_seconds=60)
    d.approval_verifier=ClosedApprovalVerifier({good.principal_id:good})
    assert not d.approve_governed_pending(aid,"reviewer")
    assert not d.approve_governed_pending(aid,b"reviewer")
    assert not d.approve_governed_pending(aid,{"principal_id":"reviewer"})
    class Duck:
        principal_id="reviewer"; scopes=("self_improvement.approve",); expires_at=time.time()+60
    assert not d.approve_governed_pending(aid,Duck())
    class Attacker:
        def is_authorized(self,*a): return True
    d.approval_verifier=Attacker()
    assert not d.approve_governed_pending(aid,good)
    d.approval_verifier=ClosedApprovalVerifier({good.principal_id:good})
    wrong_scope=issuer.issue_principal("wrong",("other",),bindings=bindings,ttl_seconds=60)
    d.approval_verifier=ClosedApprovalVerifier({"wrong":wrong_scope}); assert not d.approve_governed_pending(aid,wrong_scope)
    expired=issuer.issue_principal("expired",("self_improvement.approve",),bindings=bindings,ttl_seconds=-1)
    d.approval_verifier=ClosedApprovalVerifier({"expired":expired}); assert not d.approve_governed_pending(aid,expired)
    altered=issuer.issue_principal("altered",("self_improvement.approve",),bindings={**bindings,"proposal_digest":sha("altered")},ttl_seconds=60)
    d.approval_verifier=ClosedApprovalVerifier({"altered":altered}); assert not d.approve_governed_pending(aid,altered)
    d.approval_verifier=ClosedApprovalVerifier({good.principal_id:good})
    assert not any(hasattr(d.approval_verifier,n) for n in ("issue_principal","register_principal"))
    assert not any(hasattr(d,n) for n in ("issue_principal","register_principal"))
    assert d.approve_governed_pending(aid,good)
