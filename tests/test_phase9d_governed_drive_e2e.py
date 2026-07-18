# Focused smoke proving exact end-to-end governed transaction path components used by the drive.
import hashlib, time
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

def test_governed_success_and_rollback_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLED_ENV,'0')
    repo,pol,snap,p=mk(tmp_path); p=ImprovementProposal(**{**p.__dict__,'approval_id':'a'})
    store=ApprovalStore(tmp_path/'a.json'); store.save(ApprovalRecord('a',p.digest(),pol.digest,p.expected_original_sha256,'independent',time.time(),time.time()+60))
    audit=Audit(); r=GovernedSelfImprovementTransaction(pol,audit,store).apply(p,snap)
    assert r.status==TransactionStatus.APPLIED_AND_VERIFIED and (repo/'src/t.py').read_text()=='X=2\n'
    (repo/'src/t.py').write_text('X=1\n'); snap=inspect_repository(repo,('src/*.py',)); p=ImprovementProposal(**{**p.__dict__,'inspected_source_digest':snap.digest,'approval_id':'b'})
    badpol=ImprovementPolicy(**{**pol.__dict__,'verification_gates':(VerificationGate('bad',('python','-c','import sys; sys.exit(1)')), )})
    store.save(ApprovalRecord('b',p.digest(),badpol.digest,p.expected_original_sha256,'independent',time.time(),time.time()+60))
    r=GovernedSelfImprovementTransaction(badpol,Audit(),store).apply(p,snap)
    assert r.status==TransactionStatus.VERIFICATION_FAILED_ROLLBACK_SUCCEEDED and (repo/'src/t.py').read_text()=='X=1\n'
