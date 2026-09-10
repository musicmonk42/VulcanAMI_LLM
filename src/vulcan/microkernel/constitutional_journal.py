"""Caller-owned constitutional SQLite unit of work.

This module owns the Phase-A constitutional database. Lower repositories receive
its unit of work and cannot open, commit, or close hidden transactions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from vulcan.constitution.primitives import canonical_json
from vulcan.microkernel.episode import ActorBinding
from vulcan.microkernel.state_machine import (
    EpisodeState,
    EpisodeTransitionError,
    ensure_transition,
)

SCHEMA_VERSION = "vulcan-constitutional-journal/1"
EXPECTED_SCHEMA_FINGERPRINT = (
    "aa7f197387599d7d6ecf56cea44f3779d44e4208b3744174fc8431b251c24c89"
)
_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TRANSACTION_SQL = re.compile(
    r"^\s*(?:BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE|VACUUM|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)
_READ_SQL = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_CREDENTIAL_FIELD = re.compile(
    r"(?:^|[_-])(?:authorization|bearer|raw[_-]?token|secret|signing[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(r"^\s*bearer\s+\S+", re.IGNORECASE)
_LIFECYCLE_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, "SQLITE_TRANSACTION", None),
        getattr(sqlite3, "SQLITE_SAVEPOINT", None),
        getattr(sqlite3, "SQLITE_PRAGMA", None),
        getattr(sqlite3, "SQLITE_ATTACH", None),
        getattr(sqlite3, "SQLITE_DETACH", None),
    )
    if action is not None
)
_WRITE_ACTIONS = frozenset(
    action
    for name in (
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",
    )
    if (action := getattr(sqlite3, name, None)) is not None
)


class JournalError(RuntimeError):
    """Base failure for the constitutional journal boundary."""


class JournalClosedError(JournalError):
    pass


class NestedUnitOfWorkError(JournalError):
    pass


class SuccessorError(JournalError):
    pass


class IdempotencyConflict(JournalError):
    """An actor/operation/key tuple was reused with non-identical facts."""


Failpoint = Callable[[str], None]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _hex(value: str, name: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _utc(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise ValueError("UTC timestamp required")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_nonsecret_payload(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or (
                _CREDENTIAL_FIELD.search(key) and not key.lower().endswith("_digest")
            ):
                raise ValueError(
                    "credential material is prohibited from journal payloads"
                )
            _validate_nonsecret_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_nonsecret_payload(item)
    elif isinstance(value, str) and _CREDENTIAL_VALUE.search(value):
        raise ValueError("credential material is prohibited from journal payloads")
    canonical_json(value)


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        for field in row:
            raw = (field or "").encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _strict_json(text: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise JournalError("duplicate key in canonical journal JSON")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (TypeError, json.JSONDecodeError) as exc:
        raise JournalError("invalid journal JSON") from exc
    try:
        canonical = canonical_json(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JournalError("invalid canonical journal JSON") from exc
    if canonical != text:
        raise JournalError("noncanonical journal JSON")
    return value


def verify_integrity(connection: sqlite3.Connection) -> None:
    """Verify authoritative chains and derived heads before the owner is usable."""
    metadata = connection.execute(
        "SELECT value FROM journal_metadata WHERE key='schema_version'"
    ).fetchone()
    if metadata is None or metadata[0] != SCHEMA_VERSION:
        raise JournalError("constitutional journal schema version mismatch")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise JournalError("constitutional journal has dangling references")
    actors = connection.execute("SELECT actor_digest,document FROM actors").fetchall()
    for actor_digest, document in actors:
        if _digest(_strict_json(document)) != actor_digest:
            raise JournalError("actor document digest mismatch")
    commands = connection.execute(
        "SELECT command_id,actor_digest,credential_provenance_digest,request_id,"
        "request_digest,operation,idempotency_key,command_digest FROM commands"
    ).fetchall()
    for row in commands:
        facts = {
            "actor_digest": row["actor_digest"],
            "command_id": row["command_id"],
            "credential_provenance_digest": row["credential_provenance_digest"],
            "idempotency_key": row["idempotency_key"],
            "operation": row["operation"],
            "request_digest": row["request_digest"],
            "request_id": row["request_id"],
            "schema_version": SCHEMA_VERSION,
        }
        if _digest(facts) != row["command_digest"]:
            raise JournalError("command fact digest mismatch")
    artifacts = connection.execute(
        "SELECT artifact_digest,content FROM artifacts"
    ).fetchall()
    for artifact_digest, content in artifacts:
        if hashlib.sha256(bytes(content)).hexdigest() != artifact_digest:
            raise JournalError("artifact content digest mismatch")
    transition_receipts = connection.execute(
        "SELECT artifact_digest,content FROM artifacts "
        "WHERE kind='transition-receipt.v1'"
    ).fetchall()
    for artifact_digest, content in transition_receipts:
        receipt = _strict_json(bytes(content).decode("utf-8"))
        required = {
            "actor_digest",
            "artifact_digests",
            "command_id",
            "committed_at",
            "commit_seq",
            "constitution_digest",
            "credential_provenance_digest",
            "episode_id",
            "expires_at_epoch",
            "issued_at_epoch",
            "nonce_digest",
            "operation",
            "policy_digest",
            "predecessor_digest",
            "qualified_release_digest",
            "resulting_head_digest",
            "schema_version",
            "snapshot_digest",
            "trust_root_digest",
            "validation_digest",
            "verifier_digest",
        }
        if (
            set(receipt) != required
            or receipt["schema_version"] != "vulcan-transition-receipt/1"
            or receipt["operation"]
            not in {"admission", "validation", "epistemic_commit", "publication"}
            or not isinstance(receipt["commit_seq"], int)
            or not isinstance(receipt["artifact_digests"], list)
        ):
            raise JournalError("transition receipt schema is invalid")
        committed = datetime.fromisoformat(receipt["committed_at"]).timestamp()
        if not receipt["issued_at_epoch"] <= committed <= receipt["expires_at_epoch"]:
            raise JournalError("transition receipt permit was expired at commit")
        for name in (
            "actor_digest",
            "constitution_digest",
            "credential_provenance_digest",
            "nonce_digest",
            "policy_digest",
            "predecessor_digest",
            "qualified_release_digest",
            "resulting_head_digest",
            "snapshot_digest",
            "trust_root_digest",
            "validation_digest",
            "verifier_digest",
        ):
            try:
                _hex(receipt[name], name.replace("_", " "))
            except (TypeError, ValueError) as exc:
                raise JournalError(
                    "transition receipt digest binding is invalid"
                ) from exc
        command = connection.execute(
            "SELECT actor_digest,credential_provenance_digest,commit_seq "
            "FROM commands WHERE command_id=?",
            (receipt["command_id"],),
        ).fetchone()
        if (
            command is None
            or command["actor_digest"] != receipt["actor_digest"]
            or command["credential_provenance_digest"]
            != receipt["credential_provenance_digest"]
        ):
            raise JournalError("transition receipt command binding is invalid")
        episode = connection.execute(
            "SELECT command_id,actor_digest FROM episodes WHERE episode_id=?",
            (receipt["episode_id"],),
        ).fetchone()
        if (
            episode is None
            or episode["command_id"] != receipt["command_id"]
            or episode["actor_digest"] != receipt["actor_digest"]
        ):
            raise JournalError("transition receipt episode binding is invalid")
        if receipt["operation"] == "admission":
            if int(command["commit_seq"]) != int(receipt["commit_seq"]):
                raise JournalError("admission receipt commit binding is invalid")
            admission = connection.execute(
                "SELECT 1 FROM transactional_outbox "
                "WHERE commit_seq=? AND event_type='episode.admitted' "
                "AND json_extract(payload,'$.episode_id')=? "
                "AND json_extract(payload,'$.episode_digest')=?",
                (
                    receipt["commit_seq"],
                    receipt["episode_id"],
                    receipt["resulting_head_digest"],
                ),
            ).fetchone()
            if admission is None or receipt["predecessor_digest"] != "0" * 64:
                raise JournalError("admission receipt genesis binding is invalid")
        else:
            transition = connection.execute(
                "SELECT actor_digest,credential_provenance_digest,commit_seq,to_state,"
                "transition_ordinal "
                "FROM episode_transitions WHERE episode_id=? AND transition_digest=?",
                (receipt["episode_id"], receipt["resulting_head_digest"]),
            ).fetchone()
            if (
                transition is None
                or transition["actor_digest"] != receipt["actor_digest"]
                or transition["credential_provenance_digest"]
                != receipt["credential_provenance_digest"]
                or int(transition["commit_seq"]) != int(receipt["commit_seq"])
            ):
                raise JournalError("transition receipt successor binding is invalid")
            expected_operation = (
                "epistemic_commit"
                if transition["to_state"] == "epistemically_committed"
                else (
                    "publication"
                    if transition["to_state"]
                    in {"normatively_authorized", "communicated", "consolidated"}
                    else "validation"
                )
            )
            if receipt["operation"] != expected_operation:
                raise JournalError("transition receipt edge binding is invalid")
            ordinal = int(transition["transition_ordinal"])
            if ordinal == 0:
                predecessor = connection.execute(
                    "SELECT json_extract(a.content,'$.resulting_head_digest') "
                    "FROM artifacts a WHERE a.kind='transition-receipt.v1' "
                    "AND json_extract(a.content,'$.episode_id')=? "
                    "AND json_extract(a.content,'$.operation')='admission'",
                    (receipt["episode_id"],),
                ).fetchone()
            else:
                predecessor = connection.execute(
                    "SELECT transition_digest FROM episode_transitions "
                    "WHERE episode_id=? AND transition_ordinal=?",
                    (receipt["episode_id"], ordinal - 1),
                ).fetchone()
            if predecessor is None or predecessor[0] != receipt["predecessor_digest"]:
                raise JournalError("transition receipt predecessor binding is invalid")
        for member_digest in receipt["artifact_digests"]:
            if (
                connection.execute(
                    "SELECT 1 FROM artifacts WHERE artifact_digest=?", (member_digest,)
                ).fetchone()
                is None
            ):
                raise JournalError("transition receipt artifact binding is dangling")
        reference = connection.execute(
            "SELECT 1 FROM transactional_outbox WHERE event_type='transition.receipt.recorded' "
            "AND json_extract(payload,'$.receipt_artifact_digest')=?",
            (artifact_digest,),
        ).fetchone()
        if reference is None:
            raise JournalError("transition receipt is not receipt-chain bound")
    terminal_rows = connection.execute(
        "SELECT t.terminal_state,a.content,ed.document FROM terminal_results t "
        "JOIN artifacts a ON a.artifact_digest=t.result_digest "
        "LEFT JOIN episode_documents ed ON ed.episode_id=t.episode_id "
        "WHERE a.kind='terminal-result.v2'"
    ).fetchall()
    for terminal_state, content, episode_document in terminal_rows:
        try:
            result = _strict_json(bytes(content).decode("utf-8"))
            status = result["status"]
            response = result["response"]
        except (KeyError, TypeError, UnicodeDecodeError) as exc:
            raise JournalError("terminal result document is invalid") from exc
        allowed = {
            "consolidated": {"success", "abstained"},
            "blocked": {"blocked"},
            "failed": {"failed", "finalization_error"},
            "cancelled": {"cancelled"},
        }
        if not isinstance(status, str) or status not in allowed.get(
            terminal_state, set()
        ):
            raise JournalError("terminal result status diverged")
        if response is not None:
            if not isinstance(response, str) or episode_document is None:
                raise JournalError("terminal response document is invalid")
            from .episode_store import episode_from_document

            episode = episode_from_document(episode_document)
            if (
                episode.response is None
                or hashlib.sha256(response.encode("utf-8")).hexdigest()
                != episode.response.digest
            ):
                raise JournalError("terminal response bytes diverged")
    commits = [
        int(row[0])
        for row in connection.execute(
            "SELECT commit_seq FROM journal_commits ORDER BY commit_seq"
        )
    ]
    if commits != list(range(1, len(commits) + 1)):
        raise JournalError("commit sequence is not contiguous and monotonic")
    next_value = connection.execute(
        "SELECT next_value FROM journal_sequence WHERE singleton=1"
    ).fetchone()
    if next_value is None or int(next_value[0]) != len(commits) + 1:
        raise JournalError("commit sequence allocator diverged")
    rows = connection.execute(
        "SELECT * FROM transactional_outbox ORDER BY commit_seq,event_ordinal"
    ).fetchall()
    previous = "0" * 64
    by_commit: dict[int, int] = {}
    for row in rows:
        sequence = int(row["commit_seq"])
        ordinal = int(row["event_ordinal"])
        if ordinal != by_commit.get(sequence, 0):
            raise JournalError("event ordinal chain is invalid")
        by_commit[sequence] = ordinal + 1
        payload = _strict_json(row["payload"])
        _validate_nonsecret_payload(payload)
        body = {
            "actor_digest": row["actor_digest"],
            "commit_seq": sequence,
            "credential_provenance_digest": row["credential_provenance_digest"],
            "event_ordinal": ordinal,
            "event_type": row["event_type"],
            "occurred_at": row["occurred_at"],
            "payload": payload,
            "previous_receipt_digest": previous,
            "schema_version": SCHEMA_VERSION,
        }
        if row["previous_receipt_digest"] != previous or row[
            "receipt_digest"
        ] != _digest(body):
            raise JournalError("transactional receipt chain is invalid")
        previous = row["receipt_digest"]
    if set(by_commit) != set(commits):
        raise JournalError("every constitutional commit requires outbox evidence")
    episodes = connection.execute(
        "SELECT episode_id,state,head_transition_ordinal,actor_digest FROM episodes"
    ).fetchall()
    for episode_id, state, head_ordinal, actor_digest in episodes:
        transitions = connection.execute(
            "SELECT transition_ordinal,from_state,to_state,actor_digest "
            "FROM episode_transitions WHERE episode_id=? ORDER BY transition_ordinal",
            (episode_id,),
        ).fetchall()
        prior = EpisodeState.PERCEIVED
        for expected_ordinal, row in enumerate(transitions):
            ordinal, from_state, to_state, transition_actor = row
            if (
                int(ordinal) != expected_ordinal
                or from_state != prior.value
                or transition_actor != actor_digest
            ):
                raise JournalError("episode transition chain is invalid")
            try:
                target = EpisodeState(to_state)
                ensure_transition(prior, target)
            except (ValueError, EpisodeTransitionError) as exc:
                raise JournalError("episode transition chain is invalid") from exc
            prior = target
        expected = (
            (-1, EpisodeState.PERCEIVED.value)
            if not transitions
            else (int(transitions[-1][0]), transitions[-1][2])
        )
        if (int(head_ordinal), state) != expected:
            raise JournalError("episode head diverged from transition chain")
    branches = connection.execute(
        "SELECT branch_id,head_episode_id FROM lineage_branches"
    ).fetchall()
    for branch_id, head_episode_id in branches:
        active = connection.execute(
            "SELECT episode_id FROM lineage_membership "
            "WHERE branch_id=? AND status='active'",
            (branch_id,),
        ).fetchall()
        membership = connection.execute(
            "SELECT status FROM lineage_membership WHERE branch_id=? AND episode_id=?",
            (branch_id, head_episode_id),
        ).fetchone()
        if (
            membership is None
            or (
                membership[0] == "active"
                and (len(active) != 1 or active[0][0] != head_episode_id)
            )
            or (membership[0] == "past" and active)
        ):
            raise JournalError("lineage head diverged from active membership")


@dataclass(frozen=True, slots=True)
class JournalEvent:
    event_type: str
    actor_digest: str
    credential_provenance_digest: str
    payload: Mapping[str, object]
    occurred_at: datetime

    def __post_init__(self) -> None:
        _id(self.event_type, "event type")
        _hex(self.actor_digest, "actor digest")
        _hex(self.credential_provenance_digest, "credential provenance digest")
        _validate_nonsecret_payload(self.payload)
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        _utc(self.occurred_at)


class UnitOfWork:
    """A transaction handle owned and finalized only by ConstitutionalDatabase."""

    __slots__ = (
        "__connection",
        "__active",
        "__commit_seq",
        "__event_ordinal",
        "__previous_receipt",
        "__failpoint",
    )

    def __init__(self, connection: sqlite3.Connection, failpoint: Failpoint):
        self.__connection = connection
        self.__failpoint = failpoint
        self.__active = True
        row = connection.execute(
            "UPDATE journal_sequence SET next_value=next_value+1 WHERE singleton=1 "
            "RETURNING next_value-1"
        ).fetchone()
        if row is None:
            raise JournalError("commit sequence allocator unavailable")
        self.__commit_seq = int(row[0])
        connection.execute(
            "INSERT INTO journal_commits(commit_seq, schema_version) VALUES (?, ?)",
            (self.__commit_seq, SCHEMA_VERSION),
        )
        prior = connection.execute(
            "SELECT receipt_digest FROM transactional_outbox "
            "ORDER BY commit_seq DESC,event_ordinal DESC LIMIT 1"
        ).fetchone()
        self.__previous_receipt = prior[0] if prior else "0" * 64
        self.__event_ordinal = 0

    @property
    def commit_seq(self) -> int:
        self._require_active()
        return self.__commit_seq

    def _require_active(self) -> None:
        if not self.__active:
            raise JournalError("unit of work is no longer active")

    def _execute(self, sql: str, parameters: tuple[object, ...] = ()) -> sqlite3.Cursor:
        """Execute data SQL while denying transaction/lifecycle escape hatches."""
        self._require_active()
        if not isinstance(sql, str) or _TRANSACTION_SQL.match(sql):
            raise JournalError("unit of work cannot control its transaction")
        self.__failpoint("before_sql")
        cursor = self.__authorized_execute(sql, parameters, readonly=False)
        self.__failpoint("after_sql")
        return cursor

    def __authorized_execute(
        self, sql: str, parameters: tuple[object, ...], *, readonly: bool
    ) -> sqlite3.Cursor:
        def authorize(action, arg1, arg2, database, trigger):
            denied = _LIFECYCLE_ACTIONS | (_WRITE_ACTIONS if readonly else frozenset())
            return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

        self.__connection.set_authorizer(authorize)
        try:
            return self.__connection.execute(sql, parameters)
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc).lower():
                raise JournalError("unit of work rejected unauthorized SQL") from exc
            raise
        finally:
            self.__connection.set_authorizer(None)

    def query(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> tuple[sqlite3.Row, ...]:
        """Return an immutable read snapshot without exposing a connection."""
        self._require_active()
        if not isinstance(sql, str) or _READ_SQL.match(sql) is None:
            raise JournalError("unit of work query must be read-only")
        try:
            cursor = self.__authorized_execute(sql, parameters, readonly=True)
        except JournalError as exc:
            raise JournalError("unit of work query must be read-only") from exc
        return tuple(cursor.fetchall())

    def emit(self, event: JournalEvent) -> str:
        self._require_active()
        ordinal = self.__event_ordinal
        body = {
            "actor_digest": event.actor_digest,
            "commit_seq": self.__commit_seq,
            "credential_provenance_digest": event.credential_provenance_digest,
            "event_ordinal": ordinal,
            "event_type": event.event_type,
            "occurred_at": _utc(event.occurred_at),
            "payload": dict(event.payload),
            "previous_receipt_digest": self.__previous_receipt,
            "schema_version": SCHEMA_VERSION,
        }
        receipt = _digest(body)
        self.__failpoint("before_outbox_insert")
        self.__connection.execute(
            "INSERT INTO transactional_outbox VALUES (?,?,?,?,?,?,?,?,?)",
            (
                self.__commit_seq,
                ordinal,
                event.event_type,
                event.actor_digest,
                event.credential_provenance_digest,
                canonical_json(dict(event.payload)).decode("utf-8"),
                body["occurred_at"],
                self.__previous_receipt,
                receipt,
            ),
        )
        self.__failpoint("after_outbox_insert")
        self.__event_ordinal += 1
        self.__previous_receipt = receipt
        return receipt

    def _finish(self) -> None:
        self.__active = False

    @property
    def _event_count(self) -> int:
        return self.__event_ordinal


class ConstitutionalDatabase:
    """The only connection and transaction owner for the constitutional journal."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        failpoint: Failpoint | None = None,
        require_transition_receipts: bool = False,
    ):
        if not 1 <= busy_timeout_ms <= 30_000:
            raise ValueError("busy timeout must be bounded")
        supplied_path = Path(path)
        if supplied_path.is_symlink():
            raise ValueError("constitutional database path cannot be a symlink")
        self.path = supplied_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_file():
            raise ValueError("constitutional database path must be a regular file")
        self.__busy_timeout_ms = busy_timeout_ms
        self.__failpoint = failpoint or (lambda _name: None)
        self.__require_transition_receipts = require_transition_receipts
        self.__lock = threading.RLock()
        self.__local = threading.local()
        self.__active = 0
        self.__closed = False
        connection = self.__connect()
        try:
            os.chmod(self.path, 0o600)
            self.__create_schema(connection)
            actual = schema_fingerprint(connection)
            if actual != EXPECTED_SCHEMA_FINGERPRINT:
                raise JournalError(
                    f"constitutional schema fingerprint mismatch: {actual}"
                )
            verify_integrity(connection)
            self.__verify_receipt_completeness(connection)
        finally:
            connection.close()

    def __connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.__busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute(f"PRAGMA busy_timeout={self.__busy_timeout_ms}")
        settings = {
            name: connection.execute(f"PRAGMA {name}").fetchone()[0]
            for name in (
                "journal_mode",
                "synchronous",
                "foreign_keys",
                "trusted_schema",
            )
        }
        if str(settings["journal_mode"]).lower() != "wal":
            connection.close()
            raise JournalError("WAL could not be enabled")
        if (
            int(settings["synchronous"]) != 2
            or int(settings["foreign_keys"]) != 1
            or int(settings["trusted_schema"]) != 0
        ):
            connection.close()
            raise JournalError("required SQLite safety pragmas are unavailable")
        return connection

    def __verify_receipt_completeness(self, connection: sqlite3.Connection) -> None:
        if not self.__require_transition_receipts:
            return
        expected = connection.execute(
            "SELECT (SELECT count(*) FROM episodes) + "
            "(SELECT count(*) FROM episode_transitions)"
        ).fetchone()[0]
        actual = connection.execute(
            "SELECT count(*) FROM artifacts WHERE kind='transition-receipt.v1'"
        ).fetchone()[0]
        if actual != expected:
            raise JournalError("every episode mutation requires a transition receipt")

    @staticmethod
    def __create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS journal_metadata(
              key TEXT PRIMARY KEY, value TEXT NOT NULL
            ) STRICT;
            INSERT OR IGNORE INTO journal_metadata VALUES ('schema_version','vulcan-constitutional-journal/1');
            CREATE TABLE IF NOT EXISTS journal_sequence(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              next_value INTEGER NOT NULL CHECK(next_value>0)
            ) STRICT;
            INSERT OR IGNORE INTO journal_sequence VALUES(1,1);
            CREATE TABLE IF NOT EXISTS journal_commits(
              commit_seq INTEGER PRIMARY KEY CHECK(commit_seq>0),
              schema_version TEXT NOT NULL CHECK(schema_version='vulcan-constitutional-journal/1')
            ) STRICT;
            CREATE TABLE IF NOT EXISTS actors(
              actor_digest TEXT PRIMARY KEY CHECK(length(actor_digest)=64 AND actor_digest NOT GLOB '*[^0-9a-f]*'),
              document TEXT NOT NULL UNIQUE CHECK(json_valid(document))
            ) STRICT;
            CREATE TABLE IF NOT EXISTS commands(
              command_id TEXT PRIMARY KEY,
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              credential_provenance_digest TEXT NOT NULL CHECK(length(credential_provenance_digest)=64 AND credential_provenance_digest NOT GLOB '*[^0-9a-f]*'),
              request_id TEXT NOT NULL CHECK(length(request_id) BETWEEN 1 AND 128),
              request_digest TEXT NOT NULL CHECK(length(request_digest)=64 AND request_digest NOT GLOB '*[^0-9a-f]*'),
              operation TEXT NOT NULL CHECK(length(operation) BETWEEN 1 AND 128),
              idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 128),
              command_digest TEXT NOT NULL UNIQUE CHECK(length(command_digest)=64 AND command_digest NOT GLOB '*[^0-9a-f]*'),
              commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq),
              UNIQUE(actor_digest,operation,idempotency_key)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS artifacts(
              artifact_digest TEXT PRIMARY KEY CHECK(length(artifact_digest)=64 AND artifact_digest NOT GLOB '*[^0-9a-f]*'),
              kind TEXT NOT NULL,
              content BLOB NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS admitted_contexts(
              context_digest TEXT PRIMARY KEY REFERENCES artifacts(artifact_digest),
              command_id TEXT NOT NULL UNIQUE REFERENCES commands(command_id)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS episodes(
              episode_id TEXT PRIMARY KEY,
              command_id TEXT NOT NULL UNIQUE REFERENCES commands(command_id),
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              admitted_context_digest TEXT NOT NULL REFERENCES admitted_contexts(context_digest),
              state TEXT NOT NULL CHECK(state IN ('perceived','interpreted','grounded','deliberating','epistemically_committed','normatively_authorized','executed','observed','communicated','consolidated','abstained','blocked','failed','cancelled')),
              head_transition_ordinal INTEGER NOT NULL DEFAULT -1 CHECK(head_transition_ordinal>=-1)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS episode_documents(
              episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
              head_digest TEXT NOT NULL UNIQUE CHECK(length(head_digest)=64 AND head_digest NOT GLOB '*[^0-9a-f]*'),
              document TEXT NOT NULL CHECK(json_valid(document))
            ) STRICT;
            CREATE TABLE IF NOT EXISTS episode_transitions(
              episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
              transition_ordinal INTEGER NOT NULL CHECK(transition_ordinal>=0),
              from_state TEXT NOT NULL CHECK(from_state IN ('perceived','interpreted','grounded','deliberating','epistemically_committed','normatively_authorized','executed','observed','communicated','consolidated','abstained','blocked','failed','cancelled')),
              to_state TEXT NOT NULL CHECK(to_state IN ('perceived','interpreted','grounded','deliberating','epistemically_committed','normatively_authorized','executed','observed','communicated','consolidated','abstained','blocked','failed','cancelled')),
              transition_digest TEXT NOT NULL UNIQUE CHECK(length(transition_digest)=64 AND transition_digest NOT GLOB '*[^0-9a-f]*'),
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              credential_provenance_digest TEXT NOT NULL CHECK(length(credential_provenance_digest)=64 AND credential_provenance_digest NOT GLOB '*[^0-9a-f]*'),
              commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq),
              PRIMARY KEY(episode_id,transition_ordinal)
            ) STRICT;
            CREATE TRIGGER IF NOT EXISTS enforce_transition_successor
            BEFORE INSERT ON episode_transitions BEGIN
              SELECT CASE WHEN NOT EXISTS(
                SELECT 1 FROM episodes e WHERE e.episode_id=NEW.episode_id
                AND NEW.transition_ordinal=e.head_transition_ordinal+1
                AND NEW.from_state=e.state AND NEW.actor_digest=e.actor_digest
              ) THEN RAISE(ABORT,'invalid episode successor') END;
            END;
            CREATE TRIGGER IF NOT EXISTS advance_episode_head
            AFTER INSERT ON episode_transitions BEGIN
              UPDATE episodes SET state=NEW.to_state,
                head_transition_ordinal=NEW.transition_ordinal
                WHERE episode_id=NEW.episode_id;
            END;
            CREATE TABLE IF NOT EXISTS epistemic_commits(
              epistemic_digest TEXT PRIMARY KEY CHECK(length(epistemic_digest)=64 AND epistemic_digest NOT GLOB '*[^0-9a-f]*'),
              episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
              artifact_digest TEXT NOT NULL REFERENCES artifacts(artifact_digest),
              prior_digest TEXT REFERENCES epistemic_commits(epistemic_digest),
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              credential_provenance_digest TEXT NOT NULL CHECK(length(credential_provenance_digest)=64 AND credential_provenance_digest NOT GLOB '*[^0-9a-f]*'),
              commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS epistemic_heads(
              episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
              epistemic_digest TEXT NOT NULL UNIQUE REFERENCES epistemic_commits(epistemic_digest)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS epistemic_documents(
              epistemic_digest TEXT PRIMARY KEY REFERENCES epistemic_commits(epistemic_digest),
              document BLOB NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS lineage_branches(
              branch_id TEXT PRIMARY KEY,
              head_episode_id TEXT REFERENCES episodes(episode_id),
              head_digest TEXT NOT NULL CHECK(length(head_digest)=64 AND head_digest NOT GLOB '*[^0-9a-f]*')
            ) STRICT;
            CREATE TABLE IF NOT EXISTS lineage_membership(
              branch_id TEXT NOT NULL REFERENCES lineage_branches(branch_id),
              episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
              status TEXT NOT NULL CHECK(status IN ('active','past')),
              admitted_commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq),
              PRIMARY KEY(branch_id,episode_id)
            ) STRICT;
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_episode_per_branch
              ON lineage_membership(branch_id) WHERE status='active';
            CREATE TABLE IF NOT EXISTS terminal_results(
              episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
              result_digest TEXT NOT NULL UNIQUE REFERENCES artifacts(artifact_digest),
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              credential_provenance_digest TEXT NOT NULL CHECK(length(credential_provenance_digest)=64 AND credential_provenance_digest NOT GLOB '*[^0-9a-f]*'),
              terminal_state TEXT NOT NULL CHECK(terminal_state IN ('consolidated','abstained','blocked','failed','cancelled')),
              commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS transactional_outbox(
              commit_seq INTEGER NOT NULL REFERENCES journal_commits(commit_seq),
              event_ordinal INTEGER NOT NULL CHECK(event_ordinal>=0),
              event_type TEXT NOT NULL,
              actor_digest TEXT NOT NULL REFERENCES actors(actor_digest),
              credential_provenance_digest TEXT NOT NULL CHECK(length(credential_provenance_digest)=64 AND credential_provenance_digest NOT GLOB '*[^0-9a-f]*'),
              payload TEXT NOT NULL CHECK(json_valid(payload)),
              occurred_at TEXT NOT NULL,
              previous_receipt_digest TEXT NOT NULL CHECK(length(previous_receipt_digest)=64 AND previous_receipt_digest NOT GLOB '*[^0-9a-f]*'),
              receipt_digest TEXT NOT NULL UNIQUE CHECK(length(receipt_digest)=64 AND receipt_digest NOT GLOB '*[^0-9a-f]*'),
              PRIMARY KEY(commit_seq,event_ordinal)
            ) STRICT;
            COMMIT;
            """)

    def transaction(self) -> "_TransactionContext":
        return _TransactionContext(self)

    def read(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> tuple[sqlite3.Row, ...]:
        """Execute a read through an owner-managed, write-denied connection."""
        if not isinstance(sql, str) or _READ_SQL.match(sql) is None:
            raise JournalError("constitutional database read must be read-only")
        with self.__lock:
            if self.__closed:
                raise JournalClosedError("constitutional database is closed")
            self.__active += 1
        connection: sqlite3.Connection | None = None
        try:
            connection = self.__connect()

            def authorize(action, arg1, arg2, database, trigger):
                denied = _LIFECYCLE_ACTIONS | _WRITE_ACTIONS
                return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)
            try:
                return tuple(connection.execute(sql, parameters).fetchall())
            finally:
                connection.set_authorizer(None)
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc).lower():
                raise JournalError(
                    "constitutional database read must be read-only"
                ) from exc
            raise
        finally:
            if connection is not None:
                connection.close()
            with self.__lock:
                self.__active -= 1

    def _enter(self) -> tuple[sqlite3.Connection, UnitOfWork]:
        with self.__lock:
            if self.__closed:
                raise JournalClosedError("constitutional database is closed")
            if getattr(self.__local, "active", False):
                raise NestedUnitOfWorkError("nested unit of work is prohibited")
            self.__local.active = True
            self.__active += 1
        connection: sqlite3.Connection | None = None
        try:
            connection = self.__connect()
            self.__failpoint("before_begin")
            connection.execute("BEGIN IMMEDIATE")
            self.__failpoint("after_begin")
            return connection, UnitOfWork(connection, self.__failpoint)
        except BaseException:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                connection.close()
            with self.__lock:
                self.__local.active = False
                self.__active -= 1
            raise

    def _exit(
        self, connection: sqlite3.Connection, uow: UnitOfWork, success: bool
    ) -> None:
        committed = False
        try:
            if success and uow._event_count == 0:
                connection.execute("ROLLBACK")
                raise JournalError("a constitutional commit requires an outbox event")
            if success:
                self.__failpoint("before_commit")
                connection.execute("COMMIT")
                committed = True
                self.__failpoint("after_commit")
            else:
                connection.execute("ROLLBACK")
        except BaseException:
            if not committed and connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            uow._finish()
            connection.close()
            with self.__lock:
                self.__local.active = False
                self.__active -= 1

    def close(self) -> None:
        with self.__lock:
            if self.__active:
                raise JournalError("cannot close with an active unit of work")
            self.__closed = True

    @property
    def fingerprint(self) -> str:
        with self.__lock:
            if self.__closed:
                raise JournalClosedError("constitutional database is closed")
        connection = self.__connect()
        try:
            return schema_fingerprint(connection)
        finally:
            connection.close()

    @property
    def sqlite_settings(self) -> Mapping[str, object]:
        with self.__lock:
            if self.__closed:
                raise JournalClosedError("constitutional database is closed")
        connection = self.__connect()
        try:
            values = {
                name: connection.execute(f"PRAGMA {name}").fetchone()[0]
                for name in (
                    "journal_mode",
                    "synchronous",
                    "foreign_keys",
                    "trusted_schema",
                    "busy_timeout",
                )
            }
            return MappingProxyType(values)
        finally:
            connection.close()

    def verify(self) -> None:
        with self.__lock:
            if self.__closed:
                raise JournalClosedError("constitutional database is closed")
            if self.__active:
                raise JournalError("cannot verify during an active unit of work")
        connection = self.__connect()
        try:
            verify_integrity(connection)
            self.__verify_receipt_completeness(connection)
        finally:
            connection.close()


class _TransactionContext:
    __slots__ = ("__database", "__connection", "__uow")

    def __init__(self, database: ConstitutionalDatabase):
        self.__database = database
        self.__connection: sqlite3.Connection | None = None
        self.__uow: UnitOfWork | None = None

    def __enter__(self) -> UnitOfWork:
        self.__connection, self.__uow = self.__database._enter()
        return self.__uow

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.__connection is None or self.__uow is None:
            raise JournalError("unit of work was not entered")
        self.__database._exit(self.__connection, self.__uow, exc_type is None)


class ConstitutionalJournal:
    """Repositories over a caller-owned UnitOfWork; never opens or finalizes it."""

    @staticmethod
    def bind_actor(uow: UnitOfWork, actor: ActorBinding) -> str:
        document = canonical_json(actor.to_json()).decode("utf-8")
        digest = _digest(actor.to_json())
        uow._execute("INSERT OR IGNORE INTO actors VALUES (?,?)", (digest, document))
        row = uow._execute(
            "SELECT document FROM actors WHERE actor_digest=?", (digest,)
        ).fetchone()
        if row is None or row[0] != document:
            raise JournalError("actor digest collision")
        return digest

    @staticmethod
    def bind_command(
        uow: UnitOfWork,
        *,
        command_id: str,
        actor_digest: str,
        credential_provenance_digest: str,
        request_id: str,
        request_digest: str,
        operation: str,
        idempotency_key: str,
    ) -> str:
        for value, name in (
            (command_id, "command id"),
            (request_id, "request id"),
            (idempotency_key, "idempotency key"),
            (operation, "operation"),
        ):
            _id(value, name)
        _hex(actor_digest, "actor digest")
        _hex(credential_provenance_digest, "credential provenance digest")
        _hex(request_digest, "request digest")
        facts = {
            "actor_digest": actor_digest,
            "command_id": command_id,
            "credential_provenance_digest": credential_provenance_digest,
            "idempotency_key": idempotency_key,
            "operation": operation,
            "request_digest": request_digest,
            "request_id": request_id,
            "schema_version": SCHEMA_VERSION,
        }
        command_digest = _digest(facts)
        existing = uow._execute(
            "SELECT command_id,request_id,request_digest,credential_provenance_digest,"
            "command_digest "
            "FROM commands WHERE actor_digest=? AND operation=? AND idempotency_key=?",
            (actor_digest, operation, idempotency_key),
        ).fetchone()
        if existing is not None:
            persisted_facts = {
                "actor_digest": actor_digest,
                "command_id": existing["command_id"],
                "credential_provenance_digest": existing[
                    "credential_provenance_digest"
                ],
                "idempotency_key": idempotency_key,
                "operation": operation,
                "request_digest": existing["request_digest"],
                "request_id": existing["request_id"],
                "schema_version": SCHEMA_VERSION,
            }
            if _digest(persisted_facts) != existing["command_digest"]:
                raise JournalError("persisted command fact digest mismatch")
            if existing["request_digest"] != request_digest:
                raise IdempotencyConflict(
                    "idempotency key is bound to different command facts"
                )
            # Request IDs are transport facts and intentionally do not alter the
            # stable replay identity. Return the canonical persisted command.
            return f"replay:{existing['command_id']}"
        uow._execute(
            "INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,?)",
            (
                command_id,
                actor_digest,
                credential_provenance_digest,
                request_id,
                request_digest,
                operation,
                idempotency_key,
                command_digest,
                uow.commit_seq,
            ),
        )
        return "created"

    @staticmethod
    def put_artifact(uow: UnitOfWork, *, kind: str, content: bytes) -> str:
        _id(kind, "artifact kind")
        if _CREDENTIAL_FIELD.search(kind):
            raise ValueError("credential artifacts are prohibited")
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        uow._execute(
            "INSERT OR IGNORE INTO artifacts VALUES (?,?,?)", (digest, kind, content)
        )
        row = uow._execute(
            "SELECT kind,content FROM artifacts WHERE artifact_digest=?", (digest,)
        ).fetchone()
        if row is None or row[0] != kind or bytes(row[1]) != content:
            raise JournalError("artifact digest collision")
        return digest

    @staticmethod
    def create_episode(
        uow: UnitOfWork,
        *,
        episode_id: str,
        command_id: str,
        actor_digest: str,
        context_digest: str,
        initial_state: str = "perceived",
    ) -> None:
        _id(episode_id, "episode id")
        _id(command_id, "command id")
        _hex(actor_digest, "actor digest")
        _hex(context_digest, "context digest")
        _id(initial_state, "episode state")
        if initial_state != EpisodeState.PERCEIVED.value:
            raise ValueError("episode genesis must start perceived")
        uow._execute(
            "INSERT INTO admitted_contexts VALUES (?,?)", (context_digest, command_id)
        )
        uow._execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,-1)",
            (episode_id, command_id, actor_digest, context_digest, initial_state),
        )

    @staticmethod
    def append_transition(
        uow: UnitOfWork,
        *,
        episode_id: str,
        from_state: str,
        to_state: str,
        transition_digest: str,
        actor_digest: str,
        credential_provenance_digest: str,
    ) -> None:
        _hex(transition_digest, "transition digest")
        _hex(actor_digest, "actor digest")
        _hex(credential_provenance_digest, "credential provenance digest")
        row = uow._execute(
            "SELECT state,head_transition_ordinal,actor_digest FROM episodes WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise SuccessorError("episode is missing")
        if row[0] != from_state or row[2] != actor_digest:
            raise SuccessorError("invalid transition predecessor")
        try:
            ensure_transition(EpisodeState(from_state), EpisodeState(to_state))
        except (ValueError, EpisodeTransitionError) as exc:
            raise SuccessorError("invalid episode state transition") from exc
        ordinal = int(row[1]) + 1
        uow._execute(
            "INSERT INTO episode_transitions VALUES (?,?,?,?,?,?,?,?)",
            (
                episode_id,
                ordinal,
                from_state,
                to_state,
                transition_digest,
                actor_digest,
                credential_provenance_digest,
                uow.commit_seq,
            ),
        )

    @staticmethod
    def commit_epistemic(
        uow: UnitOfWork,
        *,
        episode_id: str,
        epistemic_digest: str,
        artifact_digest: str,
        prior_digest: str | None,
        actor_digest: str,
        credential_provenance_digest: str,
    ) -> None:
        _hex(epistemic_digest, "epistemic digest")
        _hex(artifact_digest, "artifact digest")
        _hex(actor_digest, "actor digest")
        _hex(credential_provenance_digest, "credential provenance digest")
        if prior_digest is not None:
            _hex(prior_digest, "prior epistemic digest")
        head = uow._execute(
            "SELECT epistemic_digest FROM epistemic_heads WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        actual = head[0] if head else None
        if actual != prior_digest:
            raise SuccessorError("epistemic head conflict")
        uow._execute(
            "INSERT INTO epistemic_commits VALUES (?,?,?,?,?,?,?)",
            (
                epistemic_digest,
                episode_id,
                artifact_digest,
                prior_digest,
                actor_digest,
                credential_provenance_digest,
                uow.commit_seq,
            ),
        )
        if head is None:
            uow._execute(
                "INSERT INTO epistemic_heads VALUES (?,?)",
                (episode_id, epistemic_digest),
            )
        else:
            changed = uow._execute(
                "UPDATE epistemic_heads SET epistemic_digest=? WHERE episode_id=? AND epistemic_digest=?",
                (epistemic_digest, episode_id, prior_digest),
            ).rowcount
            if changed != 1:
                raise SuccessorError("epistemic successor conflict")

    @staticmethod
    def advance_lineage(
        uow: UnitOfWork,
        *,
        branch_id: str,
        episode_id: str,
        expected_head_digest: str,
        new_head_digest: str,
    ) -> None:
        _id(branch_id, "branch id")
        _id(episode_id, "episode id")
        _hex(expected_head_digest, "expected lineage head")
        _hex(new_head_digest, "new lineage head")
        row = uow._execute(
            "SELECT head_digest FROM lineage_branches WHERE branch_id=?", (branch_id,)
        ).fetchone()
        if row is None:
            if expected_head_digest != "0" * 64:
                raise SuccessorError("lineage head is missing")
            uow._execute(
                "INSERT INTO lineage_branches VALUES (?,?,?)",
                (branch_id, episode_id, new_head_digest),
            )
        else:
            if row[0] != expected_head_digest:
                raise SuccessorError("lineage head conflict")
            uow._execute(
                "UPDATE lineage_membership SET status='past' WHERE branch_id=? AND status='active'",
                (branch_id,),
            )
            uow._execute(
                "UPDATE lineage_branches SET head_episode_id=?,head_digest=? WHERE branch_id=?",
                (episode_id, new_head_digest, branch_id),
            )
        uow._execute(
            "INSERT INTO lineage_membership VALUES (?,?,?,?)",
            (branch_id, episode_id, "active", uow.commit_seq),
        )

    @staticmethod
    def record_terminal(
        uow: UnitOfWork,
        *,
        episode_id: str,
        result_digest: str,
        actor_digest: str,
        credential_provenance_digest: str,
        terminal_state: str,
    ) -> None:
        _hex(result_digest, "result digest")
        _hex(actor_digest, "actor digest")
        _hex(credential_provenance_digest, "credential provenance digest")
        row = uow._execute(
            "SELECT state,actor_digest FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if row is None or row[0] != terminal_state or row[1] != actor_digest:
            raise SuccessorError("terminal result does not match episode head")
        if not EpisodeState(terminal_state).is_terminal:
            raise SuccessorError("terminal result requires a terminal state")
        uow._execute(
            "INSERT INTO terminal_results VALUES (?,?,?,?,?,?)",
            (
                episode_id,
                result_digest,
                actor_digest,
                credential_provenance_digest,
                terminal_state,
                uow.commit_seq,
            ),
        )
