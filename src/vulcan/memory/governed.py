"""Canonical durable authority for a deliberately small preference surface.

This module has no vector, graph, transcript, or model-memory fallback.  It is
an exact-key SQLite store whose authoritative ``memory_heads`` rows prevent a
missing historical revision from becoming effective after a restore or purge.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

SCHEMA_VERSION = "governed-memory/2"
POLICY_VERSION = "memory-policy/2"
MAX_RESULTS = 20

@dataclass(frozen=True)
class MemoryRuntimeConfig:
    """Validated deployment input; environment parsing is isolated here."""
    enabled: bool
    path: Path | None = None
    durable_root: Path | None = None
    replicas: int = 1
    backend: str = "sqlite"
    policy_version: str = POLICY_VERSION

    @classmethod
    def from_environment(cls) -> "MemoryRuntimeConfig":
        enabled = os.getenv("VULCAN_MEMORY_ENABLED", "0") == "1"
        if not enabled:
            return cls(False)
        path, root = os.getenv("VULCAN_MEMORY_SQLITE_PATH"), os.getenv("VULCAN_MEMORY_DURABLE_ROOT")
        if not path or not root:
            raise RuntimeError("durable memory requires VULCAN_MEMORY_SQLITE_PATH and VULCAN_MEMORY_DURABLE_ROOT")
        try:
            replicas = int(os.getenv("VULCAN_RUNTIME_REPLICAS", "1"))
        except ValueError as exc:
            raise RuntimeError("invalid durable-memory replica count") from exc
        return cls(True, Path(path), Path(root), replicas, os.getenv("VULCAN_MEMORY_BACKEND", "sqlite"), os.getenv("VULCAN_MEMORY_POLICY_VERSION", POLICY_VERSION)).validated()

    def validated(self) -> "MemoryRuntimeConfig":
        if not self.enabled: return self
        if self.backend != "sqlite" or self.replicas != 1 or self.policy_version != POLICY_VERSION:
            raise RuntimeError("unsupported governed-memory topology or policy")
        if self.path is None or self.durable_root is None or not self.path.is_absolute() or not self.durable_root.is_absolute():
            raise RuntimeError("governed-memory paths must be absolute")
        root = self.durable_root.resolve(strict=True)
        if root in {Path("/tmp"), Path.home(), Path.cwd()} or root.is_symlink() or not root.is_dir():
            raise RuntimeError("unsafe governed-memory durable root")
        candidate = self.path.resolve(strict=False)
        if root not in (candidate, *candidate.parents) or (candidate.exists() and not candidate.is_file()):
            raise RuntimeError("governed-memory database escapes durable root")
        return self

class MemoryKind(str, Enum): EXPLICIT_PREFERENCE = "explicit_preference"
class MemoryLifecycle(str, Enum): ACTIVE="active"; SUPERSEDED="superseded"; TOMBSTONED="tombstoned"; PURGED="purged"
class MemoryOperation(str, Enum): CREATE="create"; READ="read"; CORRECT="correct"; FORGET="forget"; LIST="list"; EXPORT="export"; MIGRATE="migrate"
class MemoryReason(str, Enum):
    COMMITTED="committed"; CONFLICT="conflict"; NOT_FOUND="not_found"; UNAUTHORIZED="unauthorized"; MEMORY_DISABLED="memory_disabled"; POLICY_REJECTED="policy_rejected"; EMPTY="empty"; IDEMPOTENCY_CONFLICT="idempotency_conflict"
class PolicyDecision(str, Enum): ALLOW="allow"; REJECT="reject"
class DeletionState(str, Enum): REQUESTED="requested"; TOMBSTONED="tombstoned"; COMPLETED="completed"; REJECTED="rejected"

# Closed, typed, non-free-form product values.  ``color`` remains only for the
# pre-existing supported test/product surface; it is not arbitrary text.
_PREFERENCES: dict[str, frozenset[str]] = {
    "locale": frozenset({"en", "en-us", "en-gb", "fr", "de", "es"}),
    "response_style": frozenset({"concise", "balanced", "detailed"}),
    "unit_system": frozenset({"metric", "imperial"}),
    "color": frozenset({"blue", "red", "green"}),
}

def _identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.replace("-", "").replace("_", "").isalnum()
def _time(clock: Callable[[], datetime]) -> str:
    value=clock()
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("memory clock must return UTC-aware time")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
def _digest(values: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def _normal(key: str, value: str) -> tuple[str, str]:
    if not isinstance(key, str) or not isinstance(value, str) or any(ord(c)<32 for c in key+value): raise ValueError("invalid preference text")
    key, value = key.strip().lower(), value.strip().lower()
    if key not in _PREFERENCES or value not in _PREFERENCES[key]: raise ValueError("unsupported typed preference")
    return key, value

@dataclass(frozen=True)
class MemoryActorContext:
    """Server-owned trusted principal binding; never decoded from wire data."""
    tenant_id: str; subject_id: str; actor_id: str; purpose: str = "personalization"; request_id: str = "system"
    def __post_init__(self) -> None:
        if not all(_identifier(x) for x in (self.tenant_id,self.subject_id,self.actor_id,self.purpose,self.request_id)): raise ValueError("invalid trusted memory actor")
MemoryActor = MemoryActorContext # compatibility name; construction remains domain-only.

@dataclass(frozen=True)
class MemoryWriteProposal:
    kind: MemoryKind; namespace: str; key: str; value: str; idempotency_key: str; case_id: str|None=None
    def __post_init__(self) -> None:
        if self.kind is not MemoryKind.EXPLICIT_PREFERENCE or not _identifier(self.namespace) or not _identifier(self.idempotency_key): raise ValueError("invalid memory proposal")
        key,value=_normal(self.key,self.value); object.__setattr__(self,"key",key); object.__setattr__(self,"value",value)
        if self.case_id is not None and not _identifier(self.case_id): raise ValueError("invalid case reference")
@dataclass(frozen=True)
class MemoryReadRequest:
    actor: MemoryActorContext; namespace: str; query: str; maximum_results: int=5
    def __post_init__(self) -> None:
        if not _identifier(self.namespace) or self.query not in _PREFERENCES or not 1<=self.maximum_results<=MAX_RESULTS: raise ValueError("memory reads require an allowlisted exact key")
@dataclass(frozen=True)
class MemoryRecord:
    record_id:str; revision:int; tenant_id:str; subject_id:str; actor_id:str; purpose:str; namespace:str; key:str; value:str|None; kind:MemoryKind; lifecycle:MemoryLifecycle; policy_version:str; created_at:str; expires_at:str; deletion_epoch:int; digest:str; supersedes:str|None=None
@dataclass(frozen=True)
class DeletionReceipt:
    operation_id:str; state:DeletionState; record_id:str; revision:int; deletion_epoch:int; policy_version:str; required_locations:tuple[str,...]; completed_locations:tuple[str,...]; remaining_obligations:tuple[str,...]=()
@dataclass(frozen=True)
class MemoryCommitResult:
    reason:MemoryReason; record:MemoryRecord|None=None; reconciliation_pending:bool=False; deletion_receipt:DeletionReceipt|None=None
@dataclass(frozen=True)
class MemoryPolicyResult:
    decision:PolicyDecision; decision_id:str; operation:MemoryOperation; version:str=POLICY_VERSION

class MemoryPolicyPort(Protocol):
    version:str
    def decide(self, operation:MemoryOperation, actor:MemoryActorContext, proposal:MemoryWriteProposal|None=None)->MemoryPolicyResult: ...
    def retention(self, actor:MemoryActorContext, proposal:MemoryWriteProposal)->timedelta: ...
class DefaultMemoryPolicy:
    version=POLICY_VERSION
    def decide(self, operation, actor, proposal=None):
        allowed=actor.purpose=="personalization" and operation in {MemoryOperation.CREATE,MemoryOperation.READ,MemoryOperation.CORRECT,MemoryOperation.FORGET,MemoryOperation.LIST}
        if proposal is not None:
            try: _normal(proposal.key,proposal.value)
            except ValueError: allowed=False
        return MemoryPolicyResult(PolicyDecision.ALLOW if allowed else PolicyDecision.REJECT, f"{self.version}:{operation.value}", operation)
    def retention(self, actor, proposal): return timedelta(days=365)

class GovernedMemoryPort(Protocol):
    def remember(self, actor:MemoryActorContext, proposal:MemoryWriteProposal)->MemoryCommitResult: ...
    def retrieve(self, request:MemoryReadRequest)->tuple[MemoryRecord,...]: ...
    def correct(self, actor:MemoryActorContext, record_id:str, base_revision:int, proposal:MemoryWriteProposal)->MemoryCommitResult: ...
    def forget(self, actor:MemoryActorContext, record_id:str, revision:int, idempotency_key:str|None=None)->MemoryCommitResult: ...
    def readiness(self)->None: ...
    def close(self)->None: ...

class SQLiteMemoryRepository:
    """Serialized SQLite repository with immutable revisions and authoritative heads."""
    def __init__(self,path:str,*,policy:MemoryPolicyPort|None=None,clock:Callable[[],datetime]|None=None, durable_root: str | None = None)->None:
        if not path or path==":memory:": raise ValueError("durable memory requires an explicit filesystem path")
        self._path=Path(path).resolve();
        if durable_root is not None:
            root=Path(durable_root).resolve(strict=True)
            if root not in (self._path, *self._path.parents): raise RuntimeError("memory database is outside durable root")
        self._path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);
        if self._path.exists() and not self._path.is_file(): raise RuntimeError("memory database is not a regular file")
        self._ownership=open(str(self._path)+".lock", "a+", encoding="utf-8")
        try: fcntl.flock(self._ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: self._ownership.close(); raise RuntimeError("governed-memory writer is already owned") from exc
        self._lock=threading.RLock(); self._closed=False; self._policy=policy or DefaultMemoryPolicy(); self._clock=clock or (lambda:datetime.now(timezone.utc))
        self._db=sqlite3.connect(str(self._path),check_same_thread=False,isolation_level=None); self._db.execute("PRAGMA foreign_keys=ON"); self._db.execute("PRAGMA busy_timeout=5000"); self._db.execute("PRAGMA journal_mode=WAL"); self._migrate(); self.readiness()
    def _migrate(self)->None:
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_schema(version TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS memory_revisions(record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,actor_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,value TEXT,kind TEXT NOT NULL CHECK(kind='explicit_preference'),lifecycle TEXT NOT NULL CHECK(lifecycle IN ('active','superseded','tombstoned','purged')),policy_version TEXT NOT NULL,created_at TEXT NOT NULL,expires_at TEXT NOT NULL,deletion_epoch INTEGER NOT NULL CHECK(deletion_epoch>=0),digest TEXT NOT NULL,supersedes TEXT,PRIMARY KEY(record_id,revision));
            CREATE TABLE IF NOT EXISTS memory_heads(record_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,purpose TEXT NOT NULL,namespace TEXT NOT NULL,key_name TEXT NOT NULL,current_revision INTEGER NOT NULL,current_lifecycle TEXT NOT NULL CHECK(current_lifecycle IN ('active','tombstoned','purged')),deletion_epoch INTEGER NOT NULL,UNIQUE(tenant_id,subject_id,purpose,namespace,key_name),FOREIGN KEY(record_id,current_revision) REFERENCES memory_revisions(record_id,revision));
            CREATE TABLE IF NOT EXISTS memory_idempotency(tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,operation TEXT NOT NULL,request_digest TEXT NOT NULL,record_id TEXT,PRIMARY KEY(tenant_id,subject_id,idempotency_key));
            CREATE TABLE IF NOT EXISTS memory_journal(sequence INTEGER PRIMARY KEY AUTOINCREMENT,operation TEXT NOT NULL,record_id TEXT NOT NULL,revision INTEGER NOT NULL,tenant_id TEXT NOT NULL,subject_id TEXT NOT NULL,request_digest TEXT NOT NULL,committed_at TEXT NOT NULL);
            """)
            versions=self._db.execute("SELECT version FROM memory_schema").fetchall()
            if not versions: self._db.execute("INSERT INTO memory_schema VALUES(?)",(SCHEMA_VERSION,))
            elif versions != [(SCHEMA_VERSION,)]: raise RuntimeError("unsupported governed-memory schema version")
    def readiness(self)->None:
        with self._lock:
            if self._closed: raise RuntimeError("memory repository is closed")
            if self._db.execute("PRAGMA integrity_check").fetchone() != ("ok",): raise RuntimeError("memory repository integrity check failed")
            if self._db.execute("SELECT version FROM memory_schema").fetchone() != (SCHEMA_VERSION,): raise RuntimeError("memory schema verification failed")
            bad=self._db.execute("SELECT 1 FROM memory_heads h LEFT JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE r.record_id IS NULL LIMIT 1").fetchone()
            if bad: raise RuntimeError("memory head corruption fails closed")
    def _record(self,row):
        data=list(row); data[9]=MemoryKind(data[9]); data[10]=MemoryLifecycle(data[10]); return MemoryRecord(*data)
    def _decision(self,op,actor,proposal=None): return self._policy.decide(op,actor,proposal)
    def _envelope(self,op,actor,target,proposal=None,base=None): return _digest({"operation":op.value,"tenant":actor.tenant_id,"subject":actor.subject_id,"actor":actor.actor_id,"purpose":actor.purpose,"target":target,"base":base,"proposal":None if proposal is None else [proposal.namespace,proposal.key,proposal.value],"policy":self._policy.version,"schema":SCHEMA_VERSION})
    def _idempotent(self,actor,key,op,digest):
        row=self._db.execute("SELECT operation,request_digest,record_id FROM memory_idempotency WHERE tenant_id=? AND subject_id=? AND idempotency_key=?",(actor.tenant_id,actor.subject_id,key)).fetchone()
        if not row:return None
        if row[0]!=op.value or row[1]!=digest:return MemoryCommitResult(MemoryReason.IDEMPOTENCY_CONFLICT)
        return self._current(actor,row[2]) if row[2] else MemoryCommitResult(MemoryReason.COMMITTED)
    def _current(self,actor,record_id):
        row=self._db.execute("SELECT r.record_id,r.revision,r.tenant_id,r.subject_id,r.actor_id,r.purpose,r.namespace,r.key_name,r.value,r.kind,r.lifecycle,r.policy_version,r.created_at,r.expires_at,r.deletion_epoch,r.digest,r.supersedes FROM memory_heads h JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE h.record_id=? AND h.tenant_id=? AND h.subject_id=? AND h.purpose=?",(record_id,actor.tenant_id,actor.subject_id,actor.purpose)).fetchone()
        return MemoryCommitResult(MemoryReason.COMMITTED,self._record(row)) if row else MemoryCommitResult(MemoryReason.NOT_FOUND)
    def _insert_revision(self,data):
        self._db.execute("INSERT INTO memory_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes")))
    def _journal(self,op,data,digest): self._db.execute("INSERT INTO memory_journal(operation,record_id,revision,tenant_id,subject_id,request_digest,committed_at) VALUES(?,?,?,?,?,?,?)",(op.value,data["record_id"],data["revision"],data["tenant_id"],data["subject_id"],digest,_time(self._clock)))
    def commit(self,actor,proposal):
        if self._decision(MemoryOperation.CREATE,actor,proposal).decision is PolicyDecision.REJECT:return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
        digest=self._envelope(MemoryOperation.CREATE,actor,f"{proposal.namespace}:{proposal.key}",proposal)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior=self._idempotent(actor,proposal.idempotency_key,MemoryOperation.CREATE,digest)
                if prior is not None:self._db.execute("COMMIT");return prior
                exists=self._db.execute("SELECT 1 FROM memory_heads WHERE tenant_id=? AND subject_id=? AND purpose=? AND namespace=? AND key_name=?",(actor.tenant_id,actor.subject_id,actor.purpose,proposal.namespace,proposal.key)).fetchone()
                if exists:self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                now=_time(self._clock); rid="mem-"+uuid4().hex; data={"record_id":rid,"revision":1,"tenant_id":actor.tenant_id,"subject_id":actor.subject_id,"actor_id":actor.actor_id,"purpose":actor.purpose,"namespace":proposal.namespace,"key":proposal.key,"value":proposal.value,"kind":proposal.kind.value,"lifecycle":"active","policy_version":self._policy.version,"created_at":now,"expires_at":(datetime.fromisoformat(now.replace("Z","+00:00"))+self._policy.retention(actor,proposal)).isoformat().replace("+00:00","Z"),"deletion_epoch":0,"supersedes":None};data["digest"]=_digest(data);self._insert_revision(data);self._db.execute("INSERT INTO memory_heads VALUES(?,?,?,?,?,?,?,?,?)",(rid,actor.tenant_id,actor.subject_id,actor.purpose,proposal.namespace,proposal.key,1,"active",0));self._db.execute("INSERT INTO memory_idempotency VALUES(?,?,?,?,?,?)",(actor.tenant_id,actor.subject_id,proposal.idempotency_key,"create",digest,rid));self._journal(MemoryOperation.CREATE,data,digest);self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.COMMITTED,self._record(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes"))))
            except Exception:self._db.execute("ROLLBACK");raise
    def read(self,request):
        if self._decision(MemoryOperation.READ,request.actor).decision is PolicyDecision.REJECT:return ()
        with self._lock:
            now=_time(self._clock); rows=self._db.execute("SELECT r.record_id,r.revision,r.tenant_id,r.subject_id,r.actor_id,r.purpose,r.namespace,r.key_name,r.value,r.kind,r.lifecycle,r.policy_version,r.created_at,r.expires_at,r.deletion_epoch,r.digest,r.supersedes FROM memory_heads h JOIN memory_revisions r ON r.record_id=h.record_id AND r.revision=h.current_revision WHERE h.tenant_id=? AND h.subject_id=? AND h.purpose=? AND h.namespace=? AND h.key_name=? AND h.current_lifecycle='active' AND r.expires_at>? LIMIT ?",(request.actor.tenant_id,request.actor.subject_id,request.actor.purpose,request.namespace,request.query,now,request.maximum_results)).fetchall();return tuple(self._record(r) for r in rows)
    def correct(self,actor,record_id,base_revision,proposal): return self._advance(MemoryOperation.CORRECT,actor,record_id,base_revision,proposal,proposal.idempotency_key)
    def tombstone(self,actor,record_id,revision,idempotency_key=None): return self._advance(MemoryOperation.FORGET,actor,record_id,revision,None,idempotency_key or f"forget-{record_id}-{revision}")
    def _advance(self,op,actor,rid,base,proposal,key):
        if not _identifier(rid) or base<1:raise ValueError("invalid memory revision")
        if self._decision(op,actor,proposal).decision is PolicyDecision.REJECT:return MemoryCommitResult(MemoryReason.POLICY_REJECTED)
        digest=self._envelope(op,actor,rid,proposal,base)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prior=self._idempotent(actor,key,op,digest)
                if prior is not None:self._db.execute("COMMIT");return prior
                cur=self._current(actor,rid)
                if cur.reason is not MemoryReason.COMMITTED:self._db.execute("COMMIT");return cur
                old=cur.record
                if old is None or old.revision!=base or old.lifecycle is not MemoryLifecycle.ACTIVE:self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                if proposal and (proposal.namespace!=old.namespace or proposal.key!=old.key):self._db.execute("COMMIT");return MemoryCommitResult(MemoryReason.CONFLICT)
                now=_time(self._clock); deletion=old.deletion_epoch+(op is MemoryOperation.FORGET); lifecycle="tombstoned" if op is MemoryOperation.FORGET else "active"; data={"record_id":rid,"revision":base+1,"tenant_id":old.tenant_id,"subject_id":old.subject_id,"actor_id":actor.actor_id,"purpose":old.purpose,"namespace":old.namespace,"key":old.key,"value":None if op is MemoryOperation.FORGET else proposal.value,"kind":old.kind.value,"lifecycle":lifecycle,"policy_version":self._policy.version,"created_at":now,"expires_at":old.expires_at,"deletion_epoch":deletion,"supersedes":f"{rid}:{base}"};data["digest"]=_digest(data);self._insert_revision(data)
                # Erase every retained payload revision in the same transaction; the head/tombstone remains.
                if op is MemoryOperation.FORGET:self._db.execute("UPDATE memory_revisions SET value=NULL WHERE record_id=?",(rid,))
                self._db.execute("UPDATE memory_heads SET current_revision=?,current_lifecycle=?,deletion_epoch=? WHERE record_id=?",(base+1,lifecycle,deletion,rid));self._db.execute("INSERT INTO memory_idempotency VALUES(?,?,?,?,?,?)",(actor.tenant_id,actor.subject_id,key,op.value,digest,rid));self._journal(op,data,digest);self._db.execute("COMMIT")
                rec=self._record(tuple(data[k] for k in ("record_id","revision","tenant_id","subject_id","actor_id","purpose","namespace","key","value","kind","lifecycle","policy_version","created_at","expires_at","deletion_epoch","digest","supersedes")))
                receipt=None if op is not MemoryOperation.FORGET else DeletionReceipt("del-"+uuid4().hex,DeletionState.COMPLETED,rid,base+1,deletion,self._policy.version,("sqlite_payloads","canonical_head"),("sqlite_payloads","canonical_head"))
                return MemoryCommitResult(MemoryReason.COMMITTED,rec,False,receipt)
            except Exception:self._db.execute("ROLLBACK");raise
    def close(self):
        with self._lock:
            if not self._closed:self._db.close();fcntl.flock(self._ownership.fileno(), fcntl.LOCK_UN);self._ownership.close();self._closed=True

class GovernedMemoryService:
    def __init__(self,repository:SQLiteMemoryRepository)->None:self._repository=repository
    def remember(self,actor,proposal):return self._repository.commit(actor,proposal)
    def retrieve(self,request):return self._repository.read(request)
    def correct(self,actor,record_id,base_revision,proposal):return self._repository.correct(actor,record_id,base_revision,proposal)
    def forget(self,actor,record_id,revision,idempotency_key=None):return self._repository.tombstone(actor,record_id,revision,idempotency_key)
    def readiness(self):self._repository.readiness()
    def close(self):self._repository.close()
class DisabledMemoryService:
    def remember(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def retrieve(self,*args,**kwargs):return ()
    def correct(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def forget(self,*args,**kwargs):return MemoryCommitResult(MemoryReason.MEMORY_DISABLED)
    def readiness(self):return None
    def close(self):return None

def compose_governed_memory(config: MemoryRuntimeConfig | None = None)->GovernedMemoryPort:
    config = (config or MemoryRuntimeConfig.from_environment()).validated()
    if not config.enabled:return DisabledMemoryService()
    assert config.path is not None and config.durable_root is not None
    return GovernedMemoryService(SQLiteMemoryRepository(str(config.path), durable_root=str(config.durable_root)))
