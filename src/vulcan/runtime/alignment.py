"""Versioned evidence-bound alignment policy registry."""
from __future__ import annotations
import copy, hashlib, json, os, re, tempfile, threading, unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .semantic import EpistemicStatus

SCHEMA_VERSION="vulcan-alignment/1"; POLICY_ID="canonical-evidence-bound"; MAX_HIST=16
STATUSES={s.value for s in EpistemicStatus}; BEH={"abstain","allow_unknown"}
REASONS={"too_many_claims","status_not_permitted","missing_evidence","missing_citation","bad_integrity","expired_evidence","dangling_reference","unknown_abstain","passed"}
@dataclass(frozen=True)
class AlignmentPolicy:
    schema_version:str; policy_id:str; revision:int; permitted_epistemic_statuses:tuple[str,...]; explicit_unknown_behavior:str; require_citations:bool; require_verified_integrity:bool; require_temporal_validity:bool; max_claims_per_response:int; policy_digest:str=""
@dataclass(frozen=True)
class AlignmentDecision:
    accepted:bool; reason_codes:tuple[str,...]; policy_digest:str; policy_revision:int

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

class AlignmentRegistry:
    def __init__(self,path: str|os.PathLike[str], *, audit=None):
        self.path=Path(path); self.audit=audit; self._lock=threading.RLock(); self._leases={}; self._hist={}; self.close_count=0
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink(): raise ValueError("symlinked policy path")
        if self.path.exists(): pol=validate_policy(_loads(self.path.read_bytes()))
        else:
            pol=default_policy(); self._persist(pol)
        self._active=pol; self._hist[pol.policy_digest]=pol
    def close(self): self.close_count+=1
    def active(self): return self._active
    def active_metadata(self): return {"policy_digest":self._active.policy_digest,"revision":self._active.revision}
    def lease(self):
        reg=self; pol=self._active
        class L:
            policy=pol; policy_digest=pol.policy_digest; revision=pol.revision
            def close(self): reg.release(pol.policy_digest)
            def __enter__(self): return self
            def __exit__(self,*a): self.close()
        with self._lock: self._leases[pol.policy_digest]=self._leases.get(pol.policy_digest,0)+1
        return L()
    def release(self,d):
        with self._lock:
            n=self._leases.get(d,0)-1
            if n>0: self._leases[d]=n
            else: self._leases.pop(d,None)
    def update(self, candidate:dict[str,Any], *, expected_previous_digest:str, actor_id:str|None=None):
        pol=validate_policy(copy.deepcopy(candidate))
        with self._lock:
            if expected_previous_digest!=self._active.policy_digest: raise ValueError("stale CAS")
            if pol.revision<=self._active.revision or pol.policy_digest in self._hist: raise ValueError("revision reuse")
            event={"policy_id":pol.policy_id,"revision":pol.revision,"expected_prior_digest":self._active.policy_digest,"proposed_new_digest":pol.policy_digest,"actor_id":actor_id or "system"}
            if self.audit: self.audit.append("alignment.activation_prepared", event)
            prior=self._active
            try:
                self._persist(pol); self._active=pol; self._hist[pol.policy_digest]=pol
                if self.audit: self.audit.append("alignment.activation_committed", {**event,"active_policy_digest":pol.policy_digest})
                self._evict(); return pol
            except Exception:
                self._active=prior
                try: self._persist(prior)
                except Exception: pass
                self._hist.pop(pol.policy_digest, None)
                if self.audit:
                    try: self.audit.append("alignment.activation_aborted", {**event,"result_category":"aborted"})
                    except Exception: pass
                raise
    def _evict(self):
        for d in list(self._hist)[:-MAX_HIST]:
            if d in self._leases: continue
            if d!=self._active.policy_digest: self._hist.pop(d,None)
    def _persist(self,pol):
        raw=_bytes(asdict(pol)); fd,tmp=tempfile.mkstemp(prefix=".alignment.",suffix=".tmp",dir=self.path.parent)
        with os.fdopen(fd,"wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,self.path)
        try: dd=os.open(self.path.parent,os.O_DIRECTORY); os.fsync(dd); os.close(dd)
        except OSError: pass
    def readiness(self):
        if validate_policy(_loads(self.path.read_bytes())).policy_digest!=self._active.policy_digest: raise RuntimeError("active policy persistence mismatch")
        return True
    def decide(self, claims, evidence, derivations, policy:AlignmentPolicy|None=None):
        p=policy or self._active; reasons=[]; eids={e.artifact_id:e for e in evidence}; dids={d.derivation_id:d for d in derivations}
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
                    if p.require_temporal_validity and e.valid_until and e.valid_until<datetime.now(timezone.utc): reasons.append("expired_evidence")
        reasons=tuple(sorted(set(reasons))) or ("passed",)
        return AlignmentDecision(reasons==("passed",), reasons, p.policy_digest, p.revision)
