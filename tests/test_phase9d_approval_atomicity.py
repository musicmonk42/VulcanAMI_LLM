import hashlib,json,os,time,threading
from pathlib import Path
from vulcan.world_model.meta_reasoning.governed_transaction import *

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
