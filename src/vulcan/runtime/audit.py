"""Dependency-light append-only canonical audit owner (vulcan-audit/1)."""
from __future__ import annotations
import copy, fcntl, hashlib, json, math, os, re, threading, unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION="vulcan-audit/1"
MAX_FILE_BYTES=8_000_000; MAX_LINE_BYTES=64_000; MAX_EVENTS=100_000; MAX_DEPTH=8; MAX_ITEMS=256; MAX_STRING=2048
_FIELDS={"schema_version","sequence","event_type","timestamp","previous_hash","data","event_hash"}
_EVENT=re.compile(r"(?:case|domain|alignment|runtime|memory|csiu|improvement|learning)\.[a-z][a-z0-9_]{0,31}")
_ALLOWED={"case.started","case.interpreted","case.plan_compiled","case.ledger_committed","case.alignment_decided","case.finalized","case.completed","case.abstained","case.blocked","case.finalization_error","case.cancelled","case.failed","domain.activation_prepared","domain.activation_committed","domain.activation_aborted","alignment.activation_prepared","alignment.activation_committed","alignment.activation_aborted","memory.write_prepared","memory.write_committed","memory.write_aborted","runtime.ready","csiu.snapshot_validated","csiu.snapshot_rejected","csiu.decision_prepared","csiu.influence_applied","csiu.influence_blocked","csiu.decision_aborted","csiu.weight_proposed","csiu.alignment_proposed","csiu.kill_switch_changed","improvement.proposed","improvement.approved","improvement.apply_prepared","improvement.candidate_installed","improvement.gate_completed","improvement.applied","improvement.aborted","improvement.rollback_completed","improvement.manual_recovery_required","learning.update_prepared","learning.update_aborted","learning.update_committed","learning.update_published","learning.manual_recovery_required","learning.policy_activation_prepared","learning.policy_activation_committed","learning.policy_activation_aborted"}
_TRANS={None:{"case.started"},"case.started":{"case.interpreted","case.failed"},"case.interpreted":{"case.plan_compiled","case.failed"},"case.plan_compiled":{"case.ledger_committed","case.failed"},"case.ledger_committed":{"case.alignment_decided","case.failed"},"case.alignment_decided":{"case.finalized","case.abstained","case.failed"},"case.finalized":{"case.completed","case.abstained","case.blocked","case.finalization_error","case.cancelled","case.failed"},"case.abstained":set(),"case.completed":set(),"case.failed":set()}

class AuditError(RuntimeError): pass
@dataclass(frozen=True)
class AuditEvent:
    schema_version:str; sequence:int; event_type:str; timestamp:str; previous_hash:str; data:dict[str,Any]; event_hash:str

def _loads(raw:bytes)->Any:
    def pairs(p):
        d={}
        for k,v in p:
            if k in d: raise AuditError("duplicate JSON key")
            d[k]=v
        return d
    return json.loads(raw.decode(), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(AuditError("non-finite number")))
def _bound(v, depth=0):
    if depth>MAX_DEPTH: raise AuditError("nested data bound")
    if isinstance(v,str):
        n=unicodedata.normalize("NFC",v)
        if len(n)>MAX_STRING or any(ord(c)<32 for c in n): raise AuditError("invalid string")
        return n
    if type(v) in (int,bool) or v is None: return v
    if isinstance(v,float):
        if not math.isfinite(v): raise AuditError("non-finite number")
        return v
    if isinstance(v,dict):
        if len(v)>MAX_ITEMS: raise AuditError("object bound")
        return { _bound(str(k),depth+1): _bound(val,depth+1) for k,val in sorted(v.items()) }
    if isinstance(v,(list,tuple)):
        if len(v)>MAX_ITEMS: raise AuditError("array bound")
        return [_bound(x,depth+1) for x in v]
    raise AuditError("unsupported data")
def _canonical(o): return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _hash_event(o):
    d=dict(o); d.pop("event_hash",None); return hashlib.sha256(_canonical(d)).hexdigest()
def _ts(s):
    if not isinstance(s,str) or not s.endswith("Z"): raise AuditError("invalid timestamp")
    try: return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)
    except ValueError: raise AuditError("invalid timestamp")
def _validate_type(t):
    if not isinstance(t,str) or len(t)>40 or not _EVENT.fullmatch(t) or t not in _ALLOWED: raise AuditError("invalid event type")

def _case_data(t,d):
    cid=d.get("case_id"); rd=d.get("request_digest")
    if not isinstance(cid,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}",cid): raise AuditError("invalid case id")
    if not isinstance(rd,str) or not re.fullmatch(r"[0-9a-f]{64}",rd): raise AuditError("invalid request digest")
    forbidden=("raw_prompt","prompt","authorization","jwt","token","secret","password","stack","exception_text")
    if any(k in d for k in forbidden): raise AuditError("forbidden case data")

class CanonicalAudit:
    def __init__(self,path: str|os.PathLike[str]):
        self.path=Path(path); self.lock_path=self.path.with_suffix(self.path.suffix+".lock"); self._lock=threading.RLock(); self._closed=False; self._seq=0; self._prev="0"*64; self._case_state={}; self.owner_id=f"audit:{self.path}"
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            if self.path.is_symlink() or self.lock_path.is_symlink(): raise AuditError("symlinked audit path")
            self._lfd=os.open(self.lock_path, os.O_CREAT|os.O_RDWR,0o600)
            try: fcntl.flock(self._lfd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise AuditError("second writer")
            if self.path.exists(): self.verify()
            else: self.path.touch(mode=0o600)
        except Exception:
            if hasattr(self,"_lfd"):
                try: fcntl.flock(self._lfd, fcntl.LOCK_UN); os.close(self._lfd)
                except OSError: pass
            raise
    def close(self):
        with self._lock:
            if self._closed: return
            self._closed=True; fcntl.flock(self._lfd, fcntl.LOCK_UN); os.close(self._lfd)
    def readiness(self): self.verify(); return True
    def append(self,event_type:str,data:dict[str,Any])->AuditEvent:
        _validate_type(event_type); frozen=_bound(copy.deepcopy(data))
        if event_type.startswith("case."): _case_data(event_type,frozen)
        raw_probe=_canonical(frozen)
        if len(raw_probe)>MAX_LINE_BYTES//2: raise AuditError("event data bound")
        with self._lock:
            if self._closed: raise AuditError("audit closed")
            prev_state=None
            if event_type.startswith("case."):
                prev_state=self._case_state.get(frozen["case_id"])
                if event_type not in _TRANS.get(prev_state,set()): raise AuditError("invalid case lifecycle")
            ev={"schema_version":SCHEMA_VERSION,"sequence":self._seq+1,"event_type":event_type,"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"previous_hash":self._prev,"data":frozen}
            ev["event_hash"]=_hash_event(ev); line=_canonical(ev)+b"\n"
            if len(line)>MAX_LINE_BYTES: raise AuditError("line bound")
            with open(self.path,"ab",buffering=0) as f:
                f.write(line); f.flush(); os.fsync(f.fileno())
            self._seq+=1; self._prev=ev["event_hash"]
            if event_type.startswith("case."): self._case_state[frozen["case_id"]]=event_type
            return AuditEvent(**ev)
    def verify(self):
        if self.path.is_symlink(): raise AuditError("symlinked audit path")
        if self.path.exists() and self.path.stat().st_size>MAX_FILE_BYTES: raise AuditError("audit file bound")
        seq=0; prev="0"*64; states={}; prepared={}
        data=self.path.read_bytes() if self.path.exists() else b""
        if data and not data.endswith(b"\n"): raise AuditError("incomplete final line")
        for line in data.splitlines():
            if not line or len(line)>MAX_LINE_BYTES: raise AuditError("malformed or oversized line")
            o=_loads(line)
            if set(o)!=_FIELDS or o.get("schema_version")!=SCHEMA_VERSION: raise AuditError("unknown event fields")
            _validate_type(o["event_type"]); _ts(o["timestamp"])
            if type(o["sequence"]) is not int or o["sequence"]!=seq+1: raise AuditError("sequence gap or reuse")
            if o["previous_hash"]!=prev or o["event_hash"]!=_hash_event(o): raise AuditError("audit hash mismatch")
            d=_bound(o["data"])
            if o["event_type"].startswith("case."):
                _case_data(o["event_type"],d); cid=d["case_id"]
                if o["event_type"] not in _TRANS.get(states.get(cid),set()): raise AuditError("invalid case lifecycle")
                states[cid]=o["event_type"]
            elif o["event_type"].endswith("activation_prepared"):
                key=(o["event_type"].rsplit("_",1)[0], d.get("domain") or d.get("policy_id"), d.get("proposed_new_digest"))
                prepared[key]=True
            elif o["event_type"] == "memory.write_prepared":
                prepared[("memory.write", d.get("operation_type"))]=True
            elif o["event_type"].endswith(("activation_committed","activation_aborted")):
                key=(o["event_type"].rsplit("_",1)[0], d.get("domain") or d.get("policy_id"), d.get("proposed_new_digest"))
                prepared.pop(key, None)
            elif o["event_type"] in {"memory.write_committed","memory.write_aborted"}:
                prepared.pop(("memory.write", d.get("operation_type")), None)
            seq=o["sequence"]; prev=o["event_hash"]
            if seq>MAX_EVENTS: raise AuditError("event bound")
        if prepared: raise AuditError("unresolved prepared transaction")
        self._seq=seq; self._prev=prev; self._case_state=states
    def _events_matching(self, prefix:str, field:str, value:str):
        self.verify(); out=[]
        for line in self.path.read_bytes().splitlines():
            o=_loads(line)
            if o["event_type"].startswith(prefix) and o["data"].get(field)==value: out.append(AuditEvent(**o))
        return tuple(out)
    def events_for_case(self,case_id:str): return self._events_matching("case.", "case_id", case_id)
    def events_for_domain(self,domain:str): return self._events_matching("domain.", "domain", domain)
    def events_for_alignment(self,policy_id:str): return self._events_matching("alignment.", "policy_id", policy_id)
    def events_for_memory_record(self,record_id:str): return self._events_matching("memory.", "record_id", record_id)
    def events_for_proposal(self, proposal_digest:str): return self._events_matching("improvement.", "proposal_digest", proposal_digest)
    def record_event(self, event_type:str, data:dict[str,Any]):
        return self.append(event_type, data)
