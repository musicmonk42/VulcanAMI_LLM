"""Segmented canonical audit owner (vulcan-audit/2) with v1 read migration."""
from __future__ import annotations
import copy, fcntl, hashlib, json, math, os, re, threading, unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from vulcan.persistence.audit.events import EVENT_FAMILY_BY_TYPE, TERMINAL_CASE_EVENTS, PREPARED_EVENTS, COMMIT_EVENTS, ABORT_EVENTS, validate_event_data
from vulcan.persistence.audit.index import AuditPage, add_event as _index_add_event, empty_index as _empty_index, paginate_sequences as _paginate_sequences, read_index as _read_index, write_index as _write_index, FIELD_BY_INDEX
from vulcan.persistence.audit.reconcile import reconcile_transactions

SCHEMA_VERSION="vulcan-audit/2"; LEGACY_SCHEMA_VERSION="vulcan-audit/1"
MAX_LINE_BYTES=64_000; MAX_DEPTH=8; MAX_ITEMS=256; MAX_STRING=2048
DEFAULT_SEGMENT_BYTES=8_000_000; DEFAULT_SEGMENT_EVENTS=100_000
_FIELDS={"schema_version","sequence","event_type","timestamp","previous_hash","data","event_hash"}
_V2_FIELDS=_FIELDS|{"segment","segment_sequence"}
_CLOSE_FIELDS={"schema_version","record_type","segment","first_sequence","last_sequence","event_count","previous_segment_digest","last_event_hash","segment_digest","timestamp"}
_MANIFEST_FIELDS={"schema_version","active_segment","next_sequence","previous_event_hash","closed_segments","legacy_source_digest","created_at","updated_at"}
_EVENT=re.compile(r"(?:case|capability|domain|alignment|runtime|memory|csiu|improvement|learning|audit|release|consent|relationship)\.[a-z][a-z0-9_]{0,31}")
_ALLOWED=set(EVENT_FAMILY_BY_TYPE)
_TRANS={None:{"case.started"},"case.started":{"case.interpreted","case.failed"},"case.interpreted":{"case.plan_compiled","case.failed"},"case.plan_compiled":{"case.ledger_committed","case.failed"},"case.ledger_committed":{"case.alignment_decided","case.failed"},"case.alignment_decided":{"case.finalized","case.abstained","case.failed"},"case.finalized":{"case.completed","case.abstained","case.blocked","case.finalization_error","case.cancelled","case.failed"},"case.abstained":set(),"case.completed":set(),"case.failed":set()}

class AuditError(RuntimeError): pass
@dataclass(frozen=True)
class AuditEvent:
    schema_version:str; sequence:int; event_type:str; timestamp:str; previous_hash:str; data:dict[str,Any]; event_hash:str; segment:int=0; segment_sequence:int=0
@dataclass(frozen=True)
class AuditDurabilityProfile:
    fsync_events: bool=True; fsync_manifest: bool=True

class Failpoint:
    def hit(self, name:str)->None: return None

def _loads(raw:bytes)->Any:
    def pairs(p):
        d={}
        for k,v in p:
            if k in d: raise AuditError("duplicate JSON key")
            d[k]=v
        return d
    try: return json.loads(raw.decode(), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in()).throw(AuditError("non-finite number")))
    except json.JSONDecodeError as exc: raise AuditError("invalid json") from exc
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
        return {_bound(str(k),depth+1): _bound(val,depth+1) for k,val in sorted(v.items())}
    if isinstance(v,(list,tuple)):
        if len(v)>MAX_ITEMS: raise AuditError("array bound")
        return [_bound(x,depth+1) for x in v]
    raise AuditError("unsupported data")
def _canonical(o): return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def _hash_event(o):
    d=dict(o); d.pop("event_hash",None); return _sha(_canonical(d))
def _ts(s):
    if not isinstance(s,str) or not s.endswith("Z"): raise AuditError("invalid timestamp")
    try: return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)
    except ValueError as exc: raise AuditError("invalid timestamp") from exc
def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def _validate_type(t):
    if not isinstance(t,str) or len(t)>40 or not _EVENT.fullmatch(t) or t not in _ALLOWED: raise AuditError("invalid event type")
def _case_data(t,d):
    cid=d.get("case_id"); rd=d.get("request_digest")
    if not isinstance(cid,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}",cid): raise AuditError("invalid case id")
    if not isinstance(rd,str) or not re.fullmatch(r"[0-9a-f]{64}",rd): raise AuditError("invalid request digest")
    if any(k in d for k in ("raw_prompt","prompt","authorization","jwt","token","secret","password","stack","exception_text")): raise AuditError("forbidden case data")

def _reject_path(p:Path, must_dir:bool=False):
    if p.is_symlink(): raise AuditError("symlinked audit path")
    if p.exists() and must_dir and not p.is_dir(): raise AuditError("audit path replacement")

class CanonicalAudit:
    def __init__(self,path: str|os.PathLike[str], *, segment_max_bytes:int=DEFAULT_SEGMENT_BYTES, segment_max_events:int=DEFAULT_SEGMENT_EVENTS, durability:AuditDurabilityProfile=AuditDurabilityProfile(), failpoint:Failpoint|None=None):
        self.legacy_path=Path(path); self.root=self.legacy_path if self.legacy_path.suffix=="" else self.legacy_path.with_suffix(self.legacy_path.suffix+".d")
        self.segment_max_bytes=segment_max_bytes; self.segment_max_events=segment_max_events; self.durability=durability; self.failpoint=failpoint or Failpoint(); self.lock_path=self.root.with_suffix(self.root.suffix+".lock")
        self._lock=threading.RLock(); self._closed=False; self._seq=0; self._prev="0"*64; self._case_state={}; self._tx_prepared=set(); self._tx_terminal=set(); self._active=1; self._seg_events=0; self._index=_empty_index(); self.owner_id=f"audit:{self.root}"
        try:
            self.root.parent.mkdir(parents=True,exist_ok=True); _reject_path(self.root); _reject_path(self.lock_path)
            self._lfd=os.open(self.lock_path, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            try: fcntl.flock(self._lfd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc: raise AuditError("second writer") from exc
            self.root.mkdir(mode=0o700,exist_ok=True); _reject_path(self.root, True)
            if not self._manifest().exists(): self._initialize_or_migrate()
            self.verify()
        except Exception:
            if hasattr(self,"_lfd"):
                try: fcntl.flock(self._lfd, fcntl.LOCK_UN); os.close(self._lfd)
                except OSError: pass
            raise
    def _manifest(self): return self.root/"manifest.json"
    def _seg(self,n:int): return self.root/f"segment-{n:06d}.jsonl"
    def _write_manifest(self,m:dict[str,Any]):
        m["updated_at"]=_now(); raw=_canonical(m)+b"\n"; tmp=self.root/"manifest.json.tmp"
        tmp.unlink(missing_ok=True)
        fd=os.open(tmp, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        try:
            os.write(fd,raw)
            if self.durability.fsync_manifest: os.fsync(fd)
        finally: os.close(fd)
        self.failpoint.hit("before_manifest_replace"); os.replace(tmp,self._manifest()); self.failpoint.hit("after_manifest_replace")
        if self.durability.fsync_manifest:
            dfd=os.open(self.root,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
    def _initialize_or_migrate(self):
        created=_now(); legacy_digest=None; seq=0; prev="0"*64
        m={"schema_version":SCHEMA_VERSION,"active_segment":1,"next_sequence":1,"previous_event_hash":prev,"closed_segments":[],"legacy_source_digest":None,"created_at":created,"updated_at":created}
        self._write_manifest(m)
        if self.legacy_path.exists() and self.legacy_path.is_file() and not self.legacy_path.is_symlink() and self.legacy_path.stat().st_size:
            data=self.legacy_path.read_bytes(); legacy_digest=_sha(data); seq,prev=self._verify_legacy(data)
            self._append_raw({"schema_version":SCHEMA_VERSION,"segment":1,"segment_sequence":1,"sequence":1,"event_type":"audit.migration_boundary","timestamp":_now(),"previous_hash":"0"*64,"data":{"legacy_schema_version":LEGACY_SCHEMA_VERSION,"legacy_source_digest":legacy_digest,"legacy_events":seq}})
            m["next_sequence"]=2; m["previous_event_hash"]=self._prev; m["legacy_source_digest"]=legacy_digest; self._write_manifest(m)
    def _append_raw(self,ev):
        ev["event_hash"]=_hash_event(ev); line=_canonical(ev)+b"\n"
        if len(line)>MAX_LINE_BYTES: raise AuditError("line bound")
        p=self._seg(ev["segment"]); _reject_path(p)
        fd=os.open(p, os.O_CREAT|os.O_APPEND|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        try:
            self.failpoint.hit("before_append_write"); os.write(fd,line); self.failpoint.hit("after_append_write")
            if self.durability.fsync_events: self.failpoint.hit("before_append_fsync"); os.fsync(fd); self.failpoint.hit("after_append_fsync")
        finally: os.close(fd)
        self._seq=ev["sequence"]; self._prev=ev["event_hash"]; self._seg_events=ev["segment_sequence"]
        return ev
    def close(self):
        with self._lock:
            if self._closed: return
            m=self._read_manifest(); m["next_sequence"]=self._seq+1; m["previous_event_hash"]=self._prev; self._write_manifest(m); self._persist_index(); self._closed=True; fcntl.flock(self._lfd, fcntl.LOCK_UN); os.close(self._lfd)
    def readiness(self): self.verify(); return True

    def _adapt_legacy_event_data(self, event_type:str, data:dict[str,Any])->dict[str,Any]:
        if event_type.startswith(("alignment.","domain.","memory.","csiu.","learning.","improvement.","release.","capability.")):
            adapted=dict(data)
            if "actor_digest" not in adapted: adapted["actor_digest"]=_sha(_canonical({"legacy_actor":"system"}))
            if "transaction_id" not in adapted and (event_type in PREPARED_EVENTS or event_type in COMMIT_EVENTS or event_type in ABORT_EVENTS):
                basis={k:v for k,v in adapted.items() if k not in {"transaction_id"}}
                adapted["transaction_id"]="legacy-"+_sha(_canonical({"event_type":event_type,"data":basis}))[:32]
            return adapted
        return data
    def append(self,event_type:str,data:dict[str,Any])->AuditEvent:
        _validate_type(event_type); frozen=_bound(copy.deepcopy(data)); frozen=self._adapt_legacy_event_data(event_type, frozen)
        try: validate_event_data(event_type, frozen)
        except ValueError as exc: raise AuditError(str(exc)) from exc
        if event_type.startswith("case."): _case_data(event_type,frozen)
        with self._lock:
            if self._closed: raise AuditError("audit closed")
            tx=frozen.get("transaction_id")
            if event_type in PREPARED_EVENTS:
                if tx in self._tx_prepared: raise AuditError("duplicate transaction prepare")
            if event_type in COMMIT_EVENTS or event_type in ABORT_EVENTS:
                if tx not in self._tx_prepared: raise AuditError("transaction terminal without prepare")
                if tx in self._tx_terminal: raise AuditError("duplicate transaction terminal")
            if event_type.startswith("case.") and event_type not in _TRANS.get(self._case_state.get(frozen["case_id"]),set()): raise AuditError("invalid case lifecycle")
            if self._seg_events>=self.segment_max_events: self._rotate()
            ev={"schema_version":SCHEMA_VERSION,"segment":self._active,"segment_sequence":self._seg_events+1,"sequence":self._seq+1,"event_type":event_type,"timestamp":_now(),"previous_hash":self._prev,"data":frozen}
            probe=_canonical({**ev,"event_hash":"0"*64})+b"\n"
            if len(probe)>MAX_LINE_BYTES or self._seg(self._active).exists() and self._seg(self._active).stat().st_size+len(probe)>self.segment_max_bytes: self._rotate(); ev.update({"segment":self._active,"segment_sequence":1})
            try:
                ev=self._append_raw(ev)
            except (AuditError, OSError):
                self.close()
                raise
            if self.durability.fsync_manifest:
                m=self._read_manifest(); m["next_sequence"]=self._seq+1; m["previous_event_hash"]=self._prev; self._write_manifest(m)
            _index_add_event(self._index, ev["sequence"], frozen)
            if self.durability.fsync_manifest: self._persist_index()
            if event_type in PREPARED_EVENTS: self._tx_prepared.add(tx)
            if event_type in COMMIT_EVENTS or event_type in ABORT_EVENTS: self._tx_terminal.add(tx)
            if event_type.startswith("case."): self._case_state[frozen["case_id"]]=event_type
            return AuditEvent(**ev)
    def _read_manifest(self):
        o=_loads(self._manifest().read_bytes());
        if set(o)!=_MANIFEST_FIELDS or o["schema_version"]!=SCHEMA_VERSION: raise AuditError("manifest mismatch")
        return o
    def _rotate(self):
        m=self._read_manifest(); p=self._seg(self._active); body=p.read_bytes() if p.exists() else b""; digest=_sha(body); close={"schema_version":SCHEMA_VERSION,"record_type":"segment_close","segment":self._active,"first_sequence":self._seq-self._seg_events+1,"last_sequence":self._seq,"event_count":self._seg_events,"previous_segment_digest":m["closed_segments"][-1]["segment_digest"] if m["closed_segments"] else "0"*64,"last_event_hash":self._prev,"segment_digest":digest,"timestamp":_now()}
        fd=os.open(p,os.O_APPEND|os.O_WRONLY|os.O_NOFOLLOW); os.write(fd,_canonical(close)+b"\n"); os.fsync(fd); os.close(fd)
        m["closed_segments"].append(close); self._active+=1; self._seg_events=0; m["active_segment"]=self._active; self._write_manifest(m)
    def _verify_legacy(self,data:bytes):
        seq=0; prev="0"*64
        if data and not data.endswith(b"\n"): raise AuditError("incomplete final line")
        for line in data.splitlines():
            o=_loads(line)
            if set(o)!=_FIELDS or o.get("schema_version")!=LEGACY_SCHEMA_VERSION: raise AuditError("unknown legacy event fields")
            if o["sequence"]!=seq+1 or o["previous_hash"]!=prev or o["event_hash"]!=_hash_event(o): raise AuditError("legacy audit hash mismatch")
            seq=o["sequence"]; prev=o["event_hash"]
        return seq,prev
    def _index_path(self): return self.root/"index.json"
    def _source_digest(self):
        h=hashlib.sha256()
        for n in range(1,self._active+1):
            p=self._seg(n)
            if p.exists(): h.update(p.name.encode()+b"\0"+p.read_bytes())
        return h.hexdigest()
    def _persist_index(self):
        self.failpoint.hit("before_index_write"); _write_index(self._index_path(), source_digest=self._source_digest(), index=self._index, canonical=_canonical, sha=_sha); self.failpoint.hit("after_index_write")
    def _load_or_rebuild_index(self):
        source=self._source_digest(); idx=_read_index(self._index_path(), source_digest=source, canonical=_canonical, sha=_sha, loads=_loads)
        if idx is None:
            idx=_empty_index()
            for ev in self.events(limit=100_000): _index_add_event(idx, ev.sequence, ev.data)
            self._index=idx; self._persist_index()
        else: self._index=idx
    def verify(self):
        _reject_path(self.root, True); m=self._read_manifest(); seq=0; prev="0"*64; states={}; tx_prepared=set(); tx_terminal=set(); terminal_counts={}; closed_cases=set(); seg_prev="0"*64; rebuilt_index=_empty_index(); events_for_reconcile=[]
        for n in range(1,m["active_segment"]+1):
            p=self._seg(n); _reject_path(p); data=p.read_bytes() if p.exists() else b""
            if data and not data.endswith(b"\n"): raise AuditError("incomplete final line")
            lines=data.splitlines(); close=None
            if lines:
                maybe=_loads(lines[-1])
                if maybe.get("record_type")=="segment_close": close=maybe; lines=lines[:-1]
            count=0
            for line in lines:
                if not line or len(line)>MAX_LINE_BYTES: raise AuditError("malformed or oversized line")
                o=_loads(line)
                if set(o)!=_V2_FIELDS or o["schema_version"]!=SCHEMA_VERSION: raise AuditError("unknown event fields")
                _validate_type(o["event_type"]); _ts(o["timestamp"])
                if o["segment"]!=n or o["segment_sequence"]!=count+1 or o["sequence"]!=seq+1: raise AuditError("sequence gap or reuse")
                if o["previous_hash"]!=prev or o["event_hash"]!=_hash_event(o): raise AuditError("audit hash mismatch")
                d=_bound(o["data"])
                if o["event_type"].startswith("case."):
                    _case_data(o["event_type"],d); cid=d["case_id"]
                    if o["event_type"] not in _TRANS.get(states.get(cid),set()): raise AuditError("invalid case lifecycle")
                    if n < m["active_segment"]: closed_cases.add(cid)
                    if o["event_type"] in TERMINAL_CASE_EVENTS: terminal_counts[cid]=terminal_counts.get(cid,0)+1
                    states[cid]=o["event_type"]
                tx=d.get("transaction_id")
                if o["event_type"] in PREPARED_EVENTS: tx_prepared.add(tx)
                if o["event_type"] in COMMIT_EVENTS or o["event_type"] in ABORT_EVENTS: tx_terminal.add(tx)
                _index_add_event(rebuilt_index, o["sequence"], d); events_for_reconcile.append(AuditEvent(**o))
                seq=o["sequence"]; prev=o["event_hash"]; count+=1
            if n<m["active_segment"]:
                if close is None or set(close)!=_CLOSE_FIELDS or close["previous_segment_digest"]!=seg_prev or close["segment_digest"]!=_sha(b"\n".join(lines)+(b"\n" if lines else b"")): raise AuditError("segment close mismatch")
                seg_prev=close["segment_digest"]
            elif close is not None: raise AuditError("active segment is closed")
        if m["next_sequence"]!=seq+1 or m["previous_event_hash"]!=prev:
            if m["next_sequence"]<=seq+1 and (m["previous_event_hash"] in {prev,"0"*64} or m["next_sequence"]==1):
                m["next_sequence"]=seq+1; m["previous_event_hash"]=prev; self._write_manifest(m)
            else: raise AuditError("manifest sequence mismatch")
        try: reconcile_transactions(tuple(events_for_reconcile))
        except ValueError as exc: raise AuditError(str(exc)) from exc
        self._seq=seq; self._prev=prev; self._case_state=states; self._tx_prepared=tx_prepared; self._tx_terminal=tx_terminal; self._active=m["active_segment"]; self._seg_events=sum(1 for _ in (self._seg(self._active).read_bytes().splitlines() if self._seg(self._active).exists() else [])); self._index=rebuilt_index
        source=self._source_digest(); existing=_read_index(self._index_path(), source_digest=source, canonical=_canonical, sha=_sha, loads=_loads)
        if existing is None or existing != rebuilt_index:
            self._persist_index()
    def deep_verify(self):
        self.verify()
        for cid,state in self._case_state.items():
            if state not in TERMINAL_CASE_EVENTS: raise AuditError("nonrecoverable case missing terminal event")
        return True
    def events(self, *, limit:int=1000):
        if limit<0 or limit>100_000: raise AuditError("export bound")
        self.verify(); out=[]
        for n in range(1,self._active+1):
            for line in (self._seg(n).read_bytes().splitlines() if self._seg(n).exists() else []):
                o=_loads(line)
                if o.get("record_type")=="segment_close": continue
                out.append(AuditEvent(**o))
                if len(out)>=limit: return tuple(out)
        return tuple(out)
    def export_archive(self, dest: str|os.PathLike[str], *, max_bytes:int=64_000_000)->str:
        self.verify(); total=sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        if total>max_bytes: raise AuditError("archive bound")
        target=Path(dest); fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600); h=hashlib.sha256()
        try:
            for name in ["manifest.json"]+[f"segment-{i:06d}.jsonl" for i in range(1,self._active+1)]:
                b=(self.root/name).read_bytes(); h.update(name.encode()+b"\0"+b); os.write(fd,b"--"+name.encode()+b"\n"+b)
            os.fsync(fd)
        finally: os.close(fd)
        return h.hexdigest()
    def _event_by_sequence(self, wanted:set[int]):
        found=[]
        for n in range(1,self._active+1):
            for line in (self._seg(n).read_bytes().splitlines() if self._seg(n).exists() else []):
                o=_loads(line)
                if o.get("record_type")=="segment_close": continue
                if o["sequence"] in wanted: found.append(AuditEvent(**o))
        return tuple(sorted(found, key=lambda e:e.sequence))
    def _events_matching(self,prefix,field,value):
        self.verify(); seqs=self._index.get(next(k for k,v in FIELD_BY_INDEX.items() if v==field),{}).get(value,[])
        return self._event_by_sequence(set(seqs))
    def page_by_index(self,index_name,value,*,cursor=None,limit=100):
        self.verify(); seqs=self._index.get(index_name,{}).get(value,[])
        try: page,next_cursor=_paginate_sequences(seqs,cursor=cursor,limit=limit)
        except ValueError as exc: raise AuditError(str(exc)) from exc
        return AuditPage(self._event_by_sequence(set(page)), next_cursor)
    def events_for_case(self,case_id): return self._events_matching("case.","case_id",case_id)
    def events_for_domain(self,domain): return self._events_matching("domain.","domain",domain)
    def events_for_alignment(self,policy_id): return self._events_matching("alignment.","policy_id",policy_id)
    def events_for_memory_record(self,record_id): return self._events_matching("memory.","record_id",record_id)
    def events_for_proposal(self,proposal_digest): return self._events_matching("improvement.","proposal_digest",proposal_digest)
    def record_event(self,event_type,data): return self.append(event_type,data)
