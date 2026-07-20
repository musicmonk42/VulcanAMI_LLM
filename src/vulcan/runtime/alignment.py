"""Versioned evidence-bound alignment policy registry."""
from __future__ import annotations
import copy, fcntl, hashlib, json, os, re, tempfile, threading, unicodedata, uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from .semantic import EpistemicStatus
from vulcan.persistence.transactions import TransactionId

SCHEMA_VERSION="vulcan-alignment/1"; POLICY_ID="canonical-evidence-bound"; MAX_HIST=16
STATUSES={s.value for s in EpistemicStatus}; BEH={"abstain","allow_unknown"}
REASONS={"too_many_claims","status_not_permitted","missing_evidence","missing_citation","bad_integrity","expired_evidence","missing_valid_until","dangling_reference","unknown_abstain","passed"}
_HEX64=re.compile(r"^[0-9a-f]{64}$")

@dataclass(frozen=True)
class AlignmentPolicy:
    schema_version:str; policy_id:str; revision:int; permitted_epistemic_statuses:tuple[str,...]; explicit_unknown_behavior:str; require_citations:bool; require_verified_integrity:bool; require_temporal_validity:bool; max_claims_per_response:int; policy_digest:str=""
@dataclass(frozen=True)
class AlignmentDecision:
    accepted:bool; reason_codes:tuple[str,...]; policy_digest:str; policy_revision:int; evaluated_at:datetime
    def replay(self, *, reevaluate: bool=False, registry: "AlignmentRegistry|None"=None, claims=(), evidence=(), derivations=()) -> "AlignmentDecision":
        if not reevaluate: return self
        if registry is None: raise ValueError("registry required for reevaluation")
        return registry.decide(claims, evidence, derivations)
@dataclass(frozen=True)
class AdminPrincipal:
    principal_id:str; principal_digest:str; roles:tuple[str,...]=( "alignment-admin", )
@dataclass(frozen=True)
class LeaseDiagnostics:
    lease_id:str; policy_digest:str; revision:int; active:bool; released:bool

def trusted_admin_principal(principal_id: str) -> AdminPrincipal:
    if not principal_id or len(principal_id)>128 or any(ord(c)<32 for c in principal_id): raise ValueError("invalid principal id")
    return AdminPrincipal(principal_id, hashlib.sha256(principal_id.encode()).hexdigest())
def _validate_principal(p: AdminPrincipal) -> AdminPrincipal:
    if not isinstance(p, AdminPrincipal): raise TypeError("trusted AdminPrincipal required")
    if "alignment-admin" not in p.roles or not _HEX64.match(p.principal_digest): raise PermissionError("principal lacks alignment authority")
    return p

def _canon_obj(v):
    if isinstance(v,str):
        n=unicodedata.normalize("NFC",v)
        if len(n)>512 or any(ord(c)<32 for c in n): raise ValueError("invalid string")
        return n
    if type(v) is bool or v is None: return v
    if type(v) is int: return v
    if isinstance(v,float): raise ValueError("non-finite number")
    if isinstance(v,tuple): return [_canon_obj(x) for x in v]
    if isinstance(v,list): return [_canon_obj(x) for x in v]
    if isinstance(v,dict): return {str(k):_canon_obj(val) for k,val in sorted(v.items())}
    raise ValueError("unsupported policy value")
def _bytes(o): return json.dumps(_canon_obj(o),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _digest_dict(d):
    x=dict(d); x.pop("policy_digest",None); return hashlib.sha256(_bytes(x)).hexdigest()
def _loads(b):
    def pairs(p):
        d={}
        for k,v in p:
            if k in d: raise ValueError("duplicate JSON key")
            d[k]=v
        return d
    return json.loads(b.decode() if isinstance(b,bytes) else b, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(ValueError("non-finite number")))
def validate_policy(o:dict[str,Any])->AlignmentPolicy:
    allowed=set(AlignmentPolicy.__dataclass_fields__)
    if set(o)!=allowed or o.get("schema_version")!=SCHEMA_VERSION or o.get("policy_id")!=POLICY_ID: raise ValueError("invalid policy schema")
    if type(o["revision"]) is not int or o["revision"]<1: raise ValueError("invalid revision")
    sts=o["permitted_epistemic_statuses"]
    if not isinstance(sts,list) or not sts or len(sts)>len(STATUSES) or len(sts)!=len(set(sts)) or not set(sts)<=STATUSES: raise ValueError("invalid statuses")
    if o["explicit_unknown_behavior"] not in BEH: raise ValueError("invalid unknown behavior")
    for k in ("require_citations","require_verified_integrity","require_temporal_validity"):
        if type(o[k]) is not bool: raise ValueError("invalid boolean")
    if type(o["max_claims_per_response"]) is not int or not 1<=o["max_claims_per_response"]<=64: raise ValueError("invalid max claims")
    if o["explicit_unknown_behavior"]=="allow_unknown" and "unknown" not in sts: raise ValueError("impossible policy")
    if o["policy_digest"]!=_digest_dict(o): raise ValueError("policy digest mismatch")
    return AlignmentPolicy(**{**o,"permitted_epistemic_statuses":tuple(sts)})
def default_policy():
    d={"schema_version":SCHEMA_VERSION,"policy_id":POLICY_ID,"revision":1,"permitted_epistemic_statuses":["computed","retrieved","proven"],"explicit_unknown_behavior":"abstain","require_citations":True,"require_verified_integrity":True,"require_temporal_validity":True,"max_claims_per_response":8,"policy_digest":""}
    d["policy_digest"]=_digest_dict(d); return validate_policy(d)

class AlignmentLease:
    def __init__(self, reg:"AlignmentRegistry", pol:AlignmentPolicy):
        self._reg=reg; self.policy=pol; self.policy_digest=pol.policy_digest; self.revision=pol.revision; self.lease_id=uuid.uuid4().hex; self._closed=False
    def close(self):
        if self._closed: return False
        self._closed=True; self._reg._release_lease(self.lease_id, self.policy_digest); return True
    def diagnostics(self): return self._reg.lease_diagnostics(self.lease_id)
    def __enter__(self): return self
    def __exit__(self,*a): self.close()

class AlignmentRegistry:
    def __init__(self,path: str|os.PathLike[str], *, audit=None, clock:Callable[[],datetime]|None=None, failpoint:Callable[[str],None]|None=None, retention_limit:int=MAX_HIST):
        self.path=Path(path); self.audit=audit; self.clock=clock or (lambda: datetime.now(timezone.utc)); self.failpoint=failpoint; self.retention_limit=retention_limit
        self._lock=threading.RLock(); self._leases:dict[str,str]={}; self._lease_counts:dict[str,int]={}; self._hist={}; self.close_count=0; self._closed=False
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink(): raise ValueError("symlinked policy path")
        self._dir=self.path.parent/("."+self.path.name+".registry"); self._dir.mkdir(exist_ok=True)
        self._lock_handle=(self._dir/"writer.lock").open("a+")
        try: fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError("alignment registry already open in another process") from e
        self._recover()
    def _fp(self,name):
        if self.failpoint: self.failpoint(name)
    def close(self):
        with self._lock:
            if self._closed: return
            self._closed=True; self.close_count+=1; fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN); self._lock_handle.close()
    def _ensure_open(self):
        if self._closed: raise RuntimeError("alignment registry closed")
    def active(self): self._ensure_open(); return self._active
    def active_metadata(self): self._ensure_open(); return {"policy_digest":self._active.policy_digest,"revision":self._active.revision}
    def lease(self):
        with self._lock:
            self._ensure_open(); l=AlignmentLease(self,self._active); self._leases[l.lease_id]=l.policy_digest; self._lease_counts[l.policy_digest]=self._lease_counts.get(l.policy_digest,0)+1; return l
    def _release_lease(self, lease_id, digest):
        with self._lock:
            if self._leases.get(lease_id)!=digest: return
            del self._leases[lease_id]; n=self._lease_counts.get(digest,0)-1
            if n>0: self._lease_counts[digest]=n
            else: self._lease_counts.pop(digest,None)
    def release(self,d): raise RuntimeError("leases must be released by lease object")
    def lease_diagnostics(self, lease_id:str):
        with self._lock:
            d=self._leases.get(lease_id); pol=self._hist.get(d) if d else None
            return LeaseDiagnostics(lease_id,d or "", pol.revision if pol else 0, d is not None, d is None)
    def activate(self, candidate:dict[str,Any], *, expected_previous_digest:str, principal:AdminPrincipal|None=None, actor_id:str|None=None, transaction_id:TransactionId|str|None=None):
        if principal is None and actor_id is not None: principal=trusted_admin_principal(actor_id)
        return self.update(candidate, expected_previous_digest=expected_previous_digest, principal=principal, transaction_id=transaction_id)
    def update(self, candidate:dict[str,Any], *, expected_previous_digest:str, principal:AdminPrincipal|None=None, actor_id:str|None=None, transaction_id:TransactionId|str|None=None):
        if principal is None and actor_id is not None: principal=trusted_admin_principal(actor_id)
        principal=_validate_principal(principal or trusted_admin_principal("system")); pol=validate_policy(copy.deepcopy(candidate)); txid=str(transaction_id or uuid.uuid4())
        with self._lock:
            self._ensure_open(); prior=self._active
            if expected_previous_digest!=prior.policy_digest: raise ValueError("stale CAS")
            if pol.revision<=prior.revision or pol.policy_digest in self._hist: raise ValueError("revision reuse")
            ev={"transaction_id":txid,"policy_id":pol.policy_id,"revision":pol.revision,"expected_prior_digest":prior.policy_digest,"proposed_new_digest":pol.policy_digest,"policy_digest":pol.policy_digest,"actor_digest":principal.principal_digest}
            self._write_tx(txid,"prepared",pol,prior.policy_digest)
            if self.audit: self.audit.append("alignment.activation_prepared", ev)
            self._fp("after_prepare")
            cand_path=self._candidate_path(pol.policy_digest); self._persist_path(cand_path,pol); self._fp("after_persist_candidate")
            if validate_policy(_loads(cand_path.read_bytes())).policy_digest!=pol.policy_digest: raise RuntimeError("candidate verification failed")
            if self.audit: self.audit.append("alignment.activation_committed", ev)
            self._write_tx(txid,"audit_committed",pol,prior.policy_digest); self._fp("after_audit_commit")
            self._atomic_write(self.path, cand_path.read_bytes()); self._write_pointer(pol.policy_digest); self._active=pol; self._hist[pol.policy_digest]=pol
            self._write_tx(txid,"published",pol,prior.policy_digest); self._fp("after_publish")
            self._evict(); return pol
    def _candidate_path(self,d): return self._dir/(d+".json")
    def _atomic_write(self,path,raw:bytes):
        fd,tmp=tempfile.mkstemp(prefix=".alignment.",suffix=".tmp",dir=path.parent)
        with os.fdopen(fd,"wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path); self._fsync_dir(path.parent)
    def _persist_path(self,path,pol): self._atomic_write(path,_bytes(asdict(pol)))
    def _write_pointer(self,d): self._atomic_write(self._dir/"active", (d+"\n").encode())
    def _write_tx(self,txid,state,pol,prior): self._atomic_write(self._dir/(txid+".tx"), _bytes({"transaction_id":txid,"state":state,"prior_digest":prior,"proposed_digest":pol.policy_digest})+b"\n")
    def _fsync_dir(self,p):
        try: dd=os.open(p,os.O_DIRECTORY); os.fsync(dd); os.close(dd)
        except OSError: pass
    def _recover(self):
        pol=None
        if self.path.exists():
            validate_policy(_loads(self.path.read_bytes()))
        active_file=self._dir/"active"
        if active_file.exists():
            d=active_file.read_text().strip(); cp=self._candidate_path(d)
            if cp.exists(): pol=validate_policy(_loads(cp.read_bytes()))
        if pol is None and self.path.exists(): pol=validate_policy(_loads(self.path.read_bytes()))
        if pol is None: pol=default_policy(); self._persist_path(self.path,pol)
        self._active=pol; self._hist[pol.policy_digest]=pol; self._write_pointer(pol.policy_digest); self._persist_path(self._candidate_path(pol.policy_digest),pol)
        # publish any transaction that reached committed audit but not pointer
        for tx in self._dir.glob("*.tx"):
            raw=_loads(tx.read_bytes()); d=raw.get("proposed_digest")
            if raw.get("state")=="audit_committed" and isinstance(d,str) and self._candidate_path(d).exists():
                c=validate_policy(_loads(self._candidate_path(d).read_bytes())); self._atomic_write(self.path, self._candidate_path(d).read_bytes()); self._write_pointer(d); self._active=c; self._hist[d]=c; self._write_tx(raw["transaction_id"],"published",c,raw.get("prior_digest"))
    def _evict(self):
        while len(self._hist)>self.retention_limit:
            for d in list(self._hist):
                if d!=self._active.policy_digest and d not in self._lease_counts:
                    self._hist.pop(d,None); break
            else: break
    def _persist(self,pol): self._persist_path(self.path,pol)
    def readiness(self):
        self._ensure_open()
        if validate_policy(_loads(self.path.read_bytes())).policy_digest!=self._active.policy_digest: raise RuntimeError("active policy persistence mismatch")
        return True
    def decide(self, claims, evidence, derivations, policy:AlignmentPolicy|None=None, *, evaluated_at:datetime|None=None):
        self._ensure_open(); p=policy or self._active; now=evaluated_at or self.clock(); reasons=[]; eids={e.artifact_id:e for e in evidence}; dids={d.derivation_id:d for d in derivations}
        if len(claims)>p.max_claims_per_response: reasons.append("too_many_claims")
        for c in claims:
            st=c.status.value
            if st not in p.permitted_epistemic_statuses: reasons.append("unknown_abstain" if st=="unknown" else "status_not_permitted")
            if st in {"retrieved","observed","proven"}:
                if not c.evidence_ids or not set(c.evidence_ids)<=set(eids): reasons.append("missing_evidence")
                if not set(c.derivation_ids)<=set(dids): reasons.append("dangling_reference")
                for eid in c.evidence_ids:
                    e=eids.get(eid)
                    if not e: continue
                    if p.require_citations and not (e.citation and eid in c.citation_ids): reasons.append("missing_citation")
                    if p.require_verified_integrity and e.source_integrity!="digest-verified": reasons.append("bad_integrity")
                    if p.require_temporal_validity:
                        if not e.valid_until: reasons.append("missing_valid_until")
                        elif e.valid_until<now: reasons.append("expired_evidence")
        reasons=tuple(sorted(set(reasons))) or ("passed",)
        return AlignmentDecision(reasons==("passed",), reasons, p.policy_digest, p.revision, now)
