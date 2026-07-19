"""Transactional exactly-once-effect outbox for canonical learning observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from vulcan.learning_observation import LearningObservation, validate_observation

SCHEMA_VERSION = "vulcan-learning-outbox/1"
SCHEMA_DIGEST = hashlib.sha256(SCHEMA_VERSION.encode()).hexdigest()


class LearningUpdateState(Enum):
    PREPARED = "prepared"
    CANDIDATE_PERSISTED = "candidate_persisted"
    AUDIT_COMMITTED = "audit_committed"
    PUBLISHED = "published"
    ABORTED = "aborted"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


class LearningDeliveryStatus(Enum):
    PUBLISHED = "published"
    REPLAYED = "replayed"
    CONFLICT = "conflict"
    STALE_REVISION = "stale_revision"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"
    ABORTED = "aborted"


@dataclass(frozen=True)
class LearningDeliveryResult:
    status: LearningDeliveryStatus
    transaction_id: str
    observation_digest: str
    active_revision: int | None
    state: LearningUpdateState
    reason: str = ""


class LearningOutboxError(RuntimeError): pass
class IdempotencyConflictError(LearningOutboxError): pass
class ManualRecoveryRequiredError(LearningOutboxError): pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _row_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


class LearningObservationOutbox:
    """SQLite-backed transaction owner for non-authoritative learning candidates.

    Semantics: a candidate has no authoritative effect until prepared audit,
    candidate/outbox persistence, committed audit, CAS publication, and ack complete.
    Committed-but-unpublished rows are recoverable; readiness fails if recovery cannot
    reconcile every durable row and digest.
    """

    def __init__(self, db_path: str | os.PathLike[str], *, audit: Any, failpoint: str | None = None) -> None:
        self.path = Path(db_path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.audit = audit
        self.failpoint = failpoint
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise LearningOutboxError("symlinked learning outbox path")
        self._lfd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lfd)
            raise LearningOutboxError("second writer") from exc
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=5.0)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()
        self.recover()

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        first = None
        try:
            self.conn.close()
        except Exception as exc:
            first = exc
        try:
            close = getattr(self.audit, "close", None)
            if close: close()
        except Exception as exc:
            if first is None: first = exc
        finally:
            fcntl.flock(self._lfd, fcntl.LOCK_UN); os.close(self._lfd)
        if first is not None:
            raise first

    def readiness(self) -> bool:
        self.recover()
        bad = self.conn.execute("SELECT COUNT(*) FROM transactions WHERE state IN (?,?)", (LearningUpdateState.CANDIDATE_PERSISTED.value, LearningUpdateState.MANUAL_RECOVERY_REQUIRED.value)).fetchone()[0]
        if bad:
            raise ManualRecoveryRequiredError("learning outbox has unresolved transactions")
        return True

    def _init_schema(self) -> None:
        self.conn.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version','vulcan-learning-outbox/1');
        INSERT OR IGNORE INTO meta(key,value) VALUES('schema_digest','%s');
        CREATE TABLE IF NOT EXISTS transactions(
          transaction_id TEXT PRIMARY KEY,
          observation_id TEXT NOT NULL UNIQUE,
          observation_digest TEXT NOT NULL UNIQUE,
          observation_json TEXT NOT NULL,
          candidate_digest TEXT NOT NULL,
          state TEXT NOT NULL,
          expected_revision INTEGER NOT NULL,
          active_revision INTEGER,
          acknowledged INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          row_digest TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS active_head(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, observation_digest TEXT);
        INSERT OR IGNORE INTO active_head(id,revision,observation_digest) VALUES(1,0,NULL);
        COMMIT;
        """ % SCHEMA_DIGEST)
        version = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        digest = self.conn.execute("SELECT value FROM meta WHERE key='schema_digest'").fetchone()[0]
        if version != SCHEMA_VERSION or digest != SCHEMA_DIGEST:
            raise LearningOutboxError("learning outbox schema mismatch")

    def deliver(self, observation: LearningObservation, *, expected_revision: int | None = None) -> LearningDeliveryResult:
        if self._closed: raise LearningOutboxError("learning outbox is closed")
        validate_observation(observation)
        obs_digest = observation.canonical_observation_digest
        current_rev = self.conn.execute("SELECT revision FROM active_head WHERE id=1").fetchone()[0]
        expected = current_rev if expected_revision is None else expected_revision
        prior = self._find_prior(observation.observation_id, obs_digest)
        if prior is not None:
            return prior
        self._fail("before_prepared_audit")
        txid = f"learn-tx-{obs_digest[:32]}"
        try:
            self.audit.append("learning.update_prepared", {"transaction_id": txid, "observation_digest": obs_digest})
            self._fail("after_prepared_audit")
            self._persist_candidate(txid, observation, expected)
            self._fail("after_candidate_persistence")
            self.audit.append("learning.update_committed", {"transaction_id": txid, "observation_digest": obs_digest})
            self._fail("during_committed_audit")
            self.conn.execute("UPDATE transactions SET state=?, updated_at=?, row_digest=? WHERE transaction_id=?", (LearningUpdateState.AUDIT_COMMITTED.value, _utc(), "pending", txid))
            self._refresh_row_digest(txid)
            self._fail("after_committed_audit")
            return self._publish(txid, obs_digest, expected)
        except Exception:
            if self._transaction_exists(txid):
                try:
                    self.conn.execute("UPDATE transactions SET state=?, updated_at=? WHERE transaction_id=? AND state=?", (LearningUpdateState.ABORTED.value, _utc(), txid, LearningUpdateState.CANDIDATE_PERSISTED.value))
                    self._refresh_row_digest(txid)
                    self.audit.append("learning.update_aborted", {"transaction_id": txid, "observation_digest": obs_digest})
                except Exception:
                    pass
            raise

    def recover(self) -> None:
        self._verify_rows()
        rows = self.conn.execute("SELECT transaction_id,observation_digest,expected_revision,state,acknowledged FROM transactions ORDER BY created_at,transaction_id").fetchall()
        for txid, obs_digest, expected, state, ack in rows:
            st = LearningUpdateState(state)
            if st is LearningUpdateState.AUDIT_COMMITTED:
                try: self._publish(txid, obs_digest, expected, recovery=True)
                except Exception:
                    self.conn.execute("UPDATE transactions SET state=?, updated_at=? WHERE transaction_id=?", (LearningUpdateState.MANUAL_RECOVERY_REQUIRED.value, _utc(), txid)); self._refresh_row_digest(txid)
            elif st is LearningUpdateState.PUBLISHED and not ack:
                self._ack(txid, obs_digest)
            elif st is LearningUpdateState.CANDIDATE_PERSISTED:
                raise ManualRecoveryRequiredError("dangling prepared learning transaction")

    def _publish(self, txid: str, obs_digest: str, expected: int, *, recovery: bool=False) -> LearningDeliveryResult:
        self._fail("during_cas_publication")
        with self.conn:
            cur = self.conn.execute("UPDATE active_head SET revision=revision+1, observation_digest=? WHERE id=1 AND revision=?", (obs_digest, expected))
            if cur.rowcount != 1:
                self.conn.execute("UPDATE transactions SET state=?, updated_at=? WHERE transaction_id=?", (LearningUpdateState.MANUAL_RECOVERY_REQUIRED.value, _utc(), txid)); self._refresh_row_digest(txid)
                return LearningDeliveryResult(LearningDeliveryStatus.STALE_REVISION, txid, obs_digest, None, LearningUpdateState.MANUAL_RECOVERY_REQUIRED, "active revision changed")
            rev = self.conn.execute("SELECT revision FROM active_head WHERE id=1").fetchone()[0]
            self.conn.execute("UPDATE transactions SET state=?, active_revision=?, updated_at=? WHERE transaction_id=?", (LearningUpdateState.PUBLISHED.value, rev, _utc(), txid)); self._refresh_row_digest(txid)
        self._fail("after_publication_before_ack")
        self._ack(txid, obs_digest)
        self._fail("during_final_audit")
        self.audit.append("learning.update_published", {"transaction_id": txid, "observation_digest": obs_digest})
        return LearningDeliveryResult(LearningDeliveryStatus.PUBLISHED, txid, obs_digest, rev, LearningUpdateState.PUBLISHED)

    def _ack(self, txid: str, obs_digest: str) -> None:
        self.conn.execute("UPDATE transactions SET acknowledged=1, updated_at=? WHERE transaction_id=?", (_utc(), txid)); self._refresh_row_digest(txid)

    def _persist_candidate(self, txid: str, observation: LearningObservation, expected: int) -> None:
        self._fail("during_candidate_persistence")
        payload = observation.canonical_json()
        candidate_digest = hashlib.sha256(("candidate:" + observation.canonical_observation_digest).encode()).hexdigest()
        row = {"transaction_id": txid,"observation_id": observation.observation_id,"observation_digest": observation.canonical_observation_digest,"candidate_digest": candidate_digest,"state": LearningUpdateState.CANDIDATE_PERSISTED.value,"expected_revision": expected,"active_revision": None,"acknowledged": 0}
        rd = _row_digest(row)
        with self.conn:
            self.conn.execute("INSERT INTO transactions(transaction_id,observation_id,observation_digest,observation_json,candidate_digest,state,expected_revision,created_at,updated_at,row_digest) VALUES(?,?,?,?,?,?,?,?,?,?)", (txid, observation.observation_id, observation.canonical_observation_digest, payload, candidate_digest, LearningUpdateState.CANDIDATE_PERSISTED.value, expected, _utc(), _utc(), rd))

    def _find_prior(self, observation_id: str, obs_digest: str) -> LearningDeliveryResult | None:
        row = self.conn.execute("SELECT transaction_id,observation_id,observation_digest,state,active_revision FROM transactions WHERE observation_id=? OR observation_digest=?", (observation_id, obs_digest)).fetchone()
        if row is None: return None
        txid, oid, odig, state, rev = row
        if oid == observation_id and odig != obs_digest:
            raise IdempotencyConflictError("observation id reused with different digest")
        st = LearningUpdateState(state)
        status = LearningDeliveryStatus.REPLAYED if st is LearningUpdateState.PUBLISHED else LearningDeliveryStatus.MANUAL_RECOVERY_REQUIRED
        return LearningDeliveryResult(status, txid, odig, rev, st, "idempotent replay")

    def _transaction_exists(self, txid: str) -> bool:
        return self.conn.execute("SELECT 1 FROM transactions WHERE transaction_id=?", (txid,)).fetchone() is not None

    def _refresh_row_digest(self, txid: str) -> None:
        row = self.conn.execute("SELECT transaction_id,observation_id,observation_digest,candidate_digest,state,expected_revision,active_revision,acknowledged FROM transactions WHERE transaction_id=?", (txid,)).fetchone()
        if row:
            keys = ["transaction_id","observation_id","observation_digest","candidate_digest","state","expected_revision","active_revision","acknowledged"]
            self.conn.execute("UPDATE transactions SET row_digest=? WHERE transaction_id=?", (_row_digest(dict(zip(keys,row))), txid))

    def _verify_rows(self) -> None:
        for row in self.conn.execute("SELECT transaction_id,observation_id,observation_digest,observation_json,candidate_digest,state,expected_revision,active_revision,acknowledged,row_digest FROM transactions").fetchall():
            txid, oid, odig, ojson, cdig, state, exp, rev, ack, rd = row
            from vulcan.learning_observation import observation_from_canonical_json
            obs = observation_from_canonical_json(ojson)
            if obs.observation_id != oid or obs.canonical_observation_digest != odig:
                raise ManualRecoveryRequiredError("learning outbox observation digest mismatch")
            keys = ["transaction_id","observation_id","observation_digest","candidate_digest","state","expected_revision","active_revision","acknowledged"]
            if _row_digest(dict(zip(keys,(txid,oid,odig,cdig,state,exp,rev,ack)))) != rd:
                raise ManualRecoveryRequiredError("learning outbox row digest mismatch")
            LearningUpdateState(state)

    def _fail(self, name: str) -> None:
        if self.failpoint == name:
            raise LearningOutboxError(name)
