"""Exclusive, one-way import of legacy constitutional stores.

The migrator never edits a source database and never installs a partially
verified target.  Legacy identity strings are evidence only and are explicitly
classified ``LEGACY_UNVERIFIED``; the authenticated migration workload is the
actor of every import transaction.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from vulcan.constitution.primitives import canonical_json
from vulcan.microkernel.constitutional_journal import (
    ConstitutionalDatabase,
    ConstitutionalJournal,
    JournalEvent,
)
from vulcan.microkernel.episode import ActorBinding
from vulcan.microkernel.episode_store import episode_from_document
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.lineage import _decode as decode_lineage


class MigrationError(RuntimeError):
    pass


class MigrationReconciliationError(MigrationError):
    pass


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    role: str
    path: str
    sha256: str
    schema: tuple[str, ...]
    row_counts: tuple[tuple[str, int], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "role": self.role,
            "row_counts": dict(self.row_counts),
            "schema": list(self.schema),
            "sha256": self.sha256,
        }


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        connection.close()
        raise MigrationReconciliationError(f"SQLite integrity failed: {path}")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        connection.close()
        raise MigrationReconciliationError(f"foreign-key integrity failed: {path}")
    return connection


def _tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier obtained from trusted schema introspection."""
    return '"' + identifier.replace('"', '""') + '"'


def _evidence(role: str, path: Path, connection: sqlite3.Connection) -> SourceEvidence:
    tables = _tables(connection)
    counts = tuple(
        (
            table,
            int(
                connection.execute(
                    f"SELECT count(*) FROM {_quote_identifier(table)}"
                ).fetchone()[0]
            ),
        )
        for table in tables
    )
    return SourceEvidence(
        role,
        str(path.resolve()),
        _logical_digest(connection),
        tables,
        counts,
    )


def _logical_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table in _tables(connection):
        digest.update(table.encode("utf-8") + b"\0")
        for document in _rows(connection, table):
            digest.update(len(document).to_bytes(8, "big"))
            digest.update(document)
    return digest.hexdigest()


def legacy_source_digest(path: str | Path) -> str:
    """Return the canonical logical digest operators must approve."""
    connection = _connect_readonly(Path(path))
    try:
        return _logical_digest(connection)
    finally:
        connection.close()


def _cell(value: object) -> object:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if value is None or type(value) in (str, int, float):
        return value
    raise MigrationReconciliationError("legacy row contains unsupported value")


def _rows(connection: sqlite3.Connection, table: str) -> tuple[bytes, ...]:
    columns = tuple(
        row[1]
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    )
    if not columns:
        raise MigrationReconciliationError(f"legacy table has no columns: {table}")
    quoted_table = _quote_identifier(table)
    order = ",".join(_quote_identifier(column) for column in columns)
    documents = []
    for row in connection.execute(f"SELECT * FROM {quoted_table} ORDER BY {order}"):
        documents.append(
            canonical_json({column: _cell(row[column]) for column in columns})
        )
    return tuple(documents)


def _legacy_attribution(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                "LEGACY_UNVERIFIED"
                if key == "classification"
                else _legacy_attribution(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_legacy_attribution(item) for item in value]
    return value


def _import_document(
    document: bytes, *, source: SourceEvidence, table: str, ordinal: int
) -> tuple[bytes, str]:
    lowered = document.lower()
    if any(
        marker in lowered
        for marker in (b"bearer ", b'"raw_token"', b'"signing_key"', b'"secret"')
    ):
        raise MigrationReconciliationError("legacy source contains credential material")
    digest = hashlib.sha256(document).hexdigest()
    return (
        canonical_json(
            {
                "legacy_attribution": "LEGACY_UNVERIFIED",
                "ordinal": ordinal,
                "record": _legacy_attribution(json.loads(document)),
                "source_digest": source.sha256,
                "source_role": source.role,
                "source_row_digest": digest,
                "table": table,
            }
        ),
        digest,
    )


def _validate_episode_source(connection: sqlite3.Connection) -> set[str]:
    required = {"episode_heads", "episode_transitions", "lineage_heads"}
    if not required <= set(_tables(connection)):
        raise MigrationReconciliationError(
            "legacy episode/lineage schema is incomplete"
        )
    episodes = set()
    for row in connection.execute(
        "SELECT episode_id,digest,document FROM episode_heads"
    ):
        episode = episode_from_document(row["document"])
        if episode.episode_id != row["episode_id"] or episode.digest != row["digest"]:
            raise MigrationReconciliationError("legacy episode head diverged")
        episodes.add(episode.episode_id)
        transitions = connection.execute(
            "SELECT ordinal,prior_digest,digest,episode_document "
            "FROM episode_transitions WHERE episode_id=? ORDER BY ordinal",
            (episode.episode_id,),
        ).fetchall()
        prior = "0" * 64
        for ordinal, transition in enumerate(transitions):
            current = episode_from_document(transition["episode_document"])
            if (
                transition["ordinal"] != ordinal
                or transition["prior_digest"] != prior
                or current.digest != transition["digest"]
                or current.transitions[-1].prior_digest != prior
            ):
                raise MigrationReconciliationError(
                    "legacy episode transition chain diverged"
                )
            prior = current.digest
        if not transitions or prior != episode.digest:
            raise MigrationReconciliationError("legacy episode head is not replayable")
    for row in connection.execute("SELECT document,digest FROM lineage_heads"):
        lineage = decode_lineage(row["document"])
        if lineage.digest != row["digest"]:
            raise MigrationReconciliationError("legacy lineage head diverged")
        referenced = {
            ref.episode_id
            for ref in (*lineage.active_episode_refs, *lineage.past_episode_refs)
        }
        if not referenced <= episodes:
            raise MigrationReconciliationError("lineage references unknown episodes")
        events = connection.execute(
            "SELECT tick,prior_digest,digest,document FROM lineage_events "
            "WHERE branch_id=? ORDER BY tick",
            (lineage.branch_id,),
        ).fetchall()
        prior = "0" * 64
        for tick, event in enumerate(events):
            state = decode_lineage(event["document"])
            if (
                event["tick"] != tick
                or event["prior_digest"] != prior
                or event["digest"] != state.digest
                or state.prior_state_digest != prior
            ):
                raise MigrationReconciliationError("legacy lineage chain diverged")
            prior = state.digest
        if not events or prior != lineage.digest:
            raise MigrationReconciliationError("legacy lineage head is not replayable")
    return episodes


def _validate_epistemic_source(
    connection: sqlite3.Connection, episode_ids: set[str]
) -> None:
    required = {"epistemic_commits", "epistemic_heads"}
    if not required <= set(_tables(connection)):
        raise MigrationReconciliationError("legacy epistemic schema is incomplete")
    referenced = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT episode_id FROM epistemic_commits"
        )
    }
    if not referenced <= episode_ids:
        raise MigrationReconciliationError(
            "epistemic commits reference unknown episodes"
        )
    commits = {}
    for row in connection.execute(
        "SELECT commit_digest,episode_id,prior_digest,snapshot_digest,document "
        "FROM epistemic_commits ORDER BY rowid"
    ):
        commit = EpistemicStore._decode(row["document"])
        if (
            commit.commit_digest != row["commit_digest"]
            or commit.episode_id != row["episode_id"]
            or commit.prior_commit_digest != row["prior_digest"]
            or commit.snapshot_digest != row["snapshot_digest"]
            or (
                commit.prior_commit_digest is not None
                and commit.prior_commit_digest not in commits
            )
        ):
            raise MigrationReconciliationError("legacy epistemic chain diverged")
        commits[commit.commit_digest] = commit
    for row in connection.execute(
        "SELECT h.commit_digest,c.document FROM epistemic_heads h "
        "JOIN epistemic_commits c ON c.commit_digest=h.commit_digest"
    ):
        commit = EpistemicStore._decode(row["document"])
        if commit.commit_digest != row["commit_digest"]:
            raise MigrationReconciliationError("legacy epistemic head diverged")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(report) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def migrate_legacy_stores(
    *,
    episode_database: str | Path,
    epistemic_database: str | Path,
    target_database: str | Path,
    report_path: str | Path,
    migration_actor: ActorBinding,
    credential_provenance_digest: str,
    migration_time: datetime,
    expected_source_digests: Mapping[str, str],
) -> dict[str, object]:
    """Validate, import into a temporary journal, verify, fsync, and install."""
    if migration_actor.classification != "AUTHENTICATED":
        raise MigrationError("authenticated migration workload actor required")
    if (
        migration_time.tzinfo is None
        or migration_time.utcoffset() != timezone.utc.utcoffset(migration_time)
    ):
        raise MigrationError("UTC migration time required")
    if len(credential_provenance_digest) != 64 or any(
        c not in "0123456789abcdef" for c in credential_provenance_digest
    ):
        raise MigrationError("credential provenance digest required")
    if set(expected_source_digests) != {"episode-lineage", "epistemic"}:
        raise MigrationError("both expected source digests are required")
    sources = (
        ("episode-lineage", Path(episode_database)),
        ("epistemic", Path(epistemic_database)),
    )
    target = Path(target_database)
    report_file = Path(report_path)
    temporary = target.with_name(f".{target.name}.migration.tmp")
    if target.exists() or temporary.exists():
        raise MigrationError("one-way migration target already exists")
    report: dict[str, object] = {
        "classification_rule": "historical labels are LEGACY_UNVERIFIED attribution only",
        "schema_version": "vulcan-legacy-migration-report/1",
        "status": "RECONCILIATION_REQUIRED",
    }
    database: ConstitutionalDatabase | None = None
    try:
        with ExitStack() as stack:
            connections = {}
            evidence = []
            for role, path in sources:
                if path.is_symlink() or not path.is_file():
                    raise MigrationReconciliationError(f"unsafe legacy source: {path}")
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                stack.callback(os.close, descriptor)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise MigrationReconciliationError(
                        f"legacy source is not exclusively available: {path}"
                    ) from exc
                connection = _connect_readonly(path)
                stack.callback(connection.close)
                connections[role] = connection
                source_evidence = _evidence(role, path, connection)
                evidence.append(source_evidence)
                report["sources"] = [item.to_json() for item in evidence]
                if source_evidence.sha256 != expected_source_digests[role]:
                    raise MigrationReconciliationError(
                        f"legacy source digest mismatch: {role}"
                    )
            episode_ids = _validate_episode_source(connections["episode-lineage"])
            _validate_epistemic_source(connections["epistemic"], episode_ids)
            target.parent.mkdir(parents=True, exist_ok=True)
            database = ConstitutionalDatabase(temporary)
            journal = ConstitutionalJournal()
            actor_digest: str | None = None
            for source in sorted(evidence, key=lambda item: item.role):
                connection = connections[source.role]
                with database.transaction() as uow:
                    actor_digest = journal.bind_actor(uow, migration_actor)
                    source_artifact = journal.put_artifact(
                        uow,
                        kind="legacy-source-evidence.v1",
                        content=canonical_json(source.to_json()),
                    )
                    uow.emit(
                        JournalEvent(
                            "migration.source.validated",
                            actor_digest,
                            credential_provenance_digest,
                            {
                                "artifact_digest": source_artifact,
                                "legacy_attribution": "LEGACY_UNVERIFIED",
                                "source_digest": source.sha256,
                                "source_role": source.role,
                            },
                            migration_time,
                        )
                    )
                for table in source.schema:
                    for ordinal, document in enumerate(_rows(connection, table)):
                        import_document, row_digest = _import_document(
                            document,
                            source=source,
                            table=table,
                            ordinal=ordinal,
                        )
                        with database.transaction() as uow:
                            actor_digest = journal.bind_actor(uow, migration_actor)
                            artifact = journal.put_artifact(
                                uow,
                                kind="legacy-record.v1",
                                content=import_document,
                            )
                            uow.emit(
                                JournalEvent(
                                    "migration.record.imported",
                                    actor_digest,
                                    credential_provenance_digest,
                                    {
                                        "artifact_digest": artifact,
                                        "legacy_attribution": "LEGACY_UNVERIFIED",
                                        "ordinal": ordinal,
                                        "source_row_digest": row_digest,
                                        "source_digest": source.sha256,
                                        "source_role": source.role,
                                        "table": table,
                                    },
                                    migration_time,
                                )
                            )
            database.verify()
            expected_imports = len(evidence) + sum(
                count for source in evidence for _, count in source.row_counts
            )
            imported = database.read(
                "SELECT count(*) AS n FROM transactional_outbox "
                "WHERE event_type LIKE 'migration.%'"
            )[0]["n"]
            if imported != expected_imports:
                raise MigrationReconciliationError("migration row count diverged")
            report["imported_records"] = imported
            database.close()
            database = None
        checkpoint = sqlite3.connect(temporary)
        checkpoint.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpoint.close()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        for _, source_path in sources:
            for suffix in ("", "-wal", "-shm"):
                member = Path(f"{source_path}{suffix}")
                if member.exists():
                    os.chmod(member, 0o440)
        installed = ConstitutionalDatabase(target)
        installed.verify()
        installed.close()
        report.update(
            {
                "migration_actor_digest": actor_digest,
                "status": "INSTALLED",
                "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        _write_report(report_file, report)
        return report
    except BaseException as exc:
        if database is not None:
            try:
                database.close()
            except BaseException:
                pass
        temporary.unlink(missing_ok=True)
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report_file, report)
        raise
