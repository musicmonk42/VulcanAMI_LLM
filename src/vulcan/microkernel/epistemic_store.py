"""Durable per-episode Graphix Epistemic authority.

The database transaction owns the canonical commit bytes, the episode-local
head, scoped claim/evidence indexes, and one idempotent audit-outbox row.  It is
deliberately independent from governed memory and from the episode store.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from vulcan.graphix.epistemic import EpistemicCommit, dumps_commit, loads_commit


class EpistemicStoreError(RuntimeError):
    pass


class EpistemicConflict(EpistemicStoreError):
    pass


class EpistemicIntegrityError(EpistemicStoreError):
    pass


Failpoint = Callable[[str], None]
OutboxSink = Callable[[str, dict[str, object]], None]


class EpistemicStore:
    """SQLite CAS store with no cross-episode total ordering."""

    def __init__(
        self,
        path: str | Path,
        *,
        outbox_sink: OutboxSink | None = None,
        failpoint: Failpoint | None = None,
    ) -> None:
        database = Path(path)
        if str(path) == ":memory:":
            raise ValueError("EpistemicStore requires a durable filesystem path")
        if database.is_symlink() or (database.exists() and not database.is_file()):
            raise EpistemicIntegrityError("invalid epistemic database path")
        database.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(database)
        self._sink = outbox_sink
        self._failpoint = failpoint
        with self._connection() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS epistemic_commits (
                    commit_id TEXT PRIMARY KEY,
                    commit_digest TEXT NOT NULL UNIQUE,
                    episode_id TEXT NOT NULL,
                    prior_digest TEXT,
                    snapshot_digest TEXT NOT NULL,
                    document BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS epistemic_heads (
                    episode_id TEXT PRIMARY KEY,
                    commit_digest TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(commit_digest) REFERENCES epistemic_commits(commit_digest)
                );
                CREATE TABLE IF NOT EXISTS epistemic_claim_index (
                    episode_id TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    commit_digest TEXT NOT NULL,
                    PRIMARY KEY(episode_id, claim_id)
                );
                CREATE TABLE IF NOT EXISTS epistemic_evidence_index (
                    episode_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    commit_digest TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    provenance_id TEXT NOT NULL,
                    PRIMARY KEY(episode_id, evidence_id)
                );
                CREATE TABLE IF NOT EXISTS epistemic_audit_outbox (
                    event_id TEXT PRIMARY KEY,
                    commit_digest TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0
                );
                """)
        self.reconcile()
        self.deliver_outbox()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def _trip(self, name: str) -> None:
        if self._failpoint is not None:
            self._failpoint(name)

    def head(self, episode_id: str) -> EpistemicCommit | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT c.document FROM epistemic_heads h JOIN epistemic_commits c "
                "ON c.commit_digest=h.commit_digest WHERE h.episode_id=?",
                (episode_id,),
            ).fetchone()
        return None if row is None else self._decode(row[0])

    def append(
        self, commit: EpistemicCommit, expected_prior_digest: str | None
    ) -> EpistemicCommit:
        if commit.prior_commit_digest != expected_prior_digest:
            raise EpistemicConflict("candidate prior digest does not match CAS input")
        document = dumps_commit(commit)
        event_id = f"epistemic.commit:{commit.commit_id}"
        payload = self._outbox_payload(commit, event_id)
        persisted = commit
        inserted = False
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT commit_digest, document FROM epistemic_commits WHERE commit_id=?",
                    (commit.commit_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != commit.commit_digest or existing[1] != document:
                        raise EpistemicConflict("commit id collision")
                    persisted = self._decode(existing[1])
                    connection.commit()
                else:
                    row = connection.execute(
                        "SELECT commit_digest FROM epistemic_heads WHERE episode_id=?",
                        (commit.episode_id,),
                    ).fetchone()
                    actual = None if row is None else row[0]
                    if actual != expected_prior_digest:
                        raise EpistemicConflict(
                            "epistemic head compare-and-swap failed"
                        )
                    self._validate_references(connection, commit)
                    self._trip("before_commit")
                    connection.execute(
                        "INSERT INTO epistemic_commits VALUES (?,?,?,?,?,?)",
                        (
                            commit.commit_id,
                            commit.commit_digest,
                            commit.episode_id,
                            commit.prior_commit_digest,
                            commit.snapshot_digest,
                            document,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO epistemic_heads VALUES (?,?) ON CONFLICT(episode_id) "
                        "DO UPDATE SET commit_digest=excluded.commit_digest",
                        (commit.episode_id, commit.commit_digest),
                    )
                    for claim in commit.claims:
                        connection.execute(
                            "INSERT INTO epistemic_claim_index VALUES (?,?,?)",
                            (commit.episode_id, claim.claim_id, commit.commit_digest),
                        )
                    for evidence in commit.evidence:
                        connection.execute(
                            "INSERT INTO epistemic_evidence_index VALUES (?,?,?,?,?,?)",
                            (
                                commit.episode_id,
                                evidence.evidence_id,
                                commit.commit_digest,
                                evidence.snapshot_digest,
                                evidence.content_digest,
                                evidence.provenance_id,
                            ),
                        )
                    connection.execute(
                        "INSERT INTO epistemic_audit_outbox(event_id,commit_digest,payload) VALUES (?,?,?)",
                        (event_id, commit.commit_digest, payload),
                    )
                    connection.commit()
                    inserted = True
            except BaseException:
                connection.rollback()
                raise
        if inserted:
            self._trip("after_db_commit")
        self.deliver_outbox()
        return persisted

    @staticmethod
    def _outbox_payload(commit: EpistemicCommit, event_id: str) -> str:
        bare = lambda value: value.removeprefix("sha256:")
        return json.dumps(
            {
                "case_id": commit.case_id,
                "commit_digest": bare(commit.commit_digest),
                "commit_id": commit.commit_id,
                "episode_id": commit.episode_id,
                "event_id": event_id,
                "policy_digest": bare(commit.policy_digest),
                "snapshot_digest": bare(commit.snapshot_digest),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _validate_references(
        self, connection: sqlite3.Connection, commit: EpistemicCommit
    ) -> None:
        previous_claims = {
            row[0]
            for row in connection.execute(
                "SELECT claim_id FROM epistemic_claim_index WHERE episode_id=?",
                (commit.episode_id,),
            )
        }
        current_claims = {claim.claim_id for claim in commit.claims}
        previous_evidence = {
            row[0]
            for row in connection.execute(
                "SELECT evidence_id FROM epistemic_evidence_index WHERE episode_id=?",
                (commit.episode_id,),
            )
        }
        for claim in commit.claims:
            if claim.claim_id in previous_claims:
                raise EpistemicConflict("claim identifiers are immutable")
            if not set((*claim.contested_by, *claim.supersedes)) <= (
                previous_claims | current_claims
            ):
                raise EpistemicIntegrityError("dangling contested/superseded claim")
        for evidence in commit.evidence:
            if evidence.evidence_id in previous_evidence:
                raise EpistemicConflict("evidence identifiers are immutable")
            if evidence.source_episode_id is None:
                continue
            source = connection.execute(
                "SELECT snapshot_digest,content_digest,provenance_id FROM epistemic_evidence_index "
                "WHERE episode_id=? AND evidence_id=?",
                (evidence.source_episode_id, evidence.evidence_id),
            ).fetchone()
            if source != (
                evidence.snapshot_digest,
                evidence.content_digest,
                evidence.provenance_id,
            ):
                raise EpistemicIntegrityError(
                    "cross-episode evidence does not preserve source provenance"
                )

    def require_committed_claim(
        self, episode_id: str, claim_id: str
    ) -> EpistemicCommit:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT c.document FROM epistemic_claim_index i JOIN epistemic_commits c "
                "ON c.commit_digest=i.commit_digest WHERE i.episode_id=? AND i.claim_id=?",
                (episode_id, claim_id),
            ).fetchone()
        if row is None:
            raise EpistemicIntegrityError("claim is not committed")
        return self._decode(row[0])

    def deliver_outbox(self) -> None:
        if self._sink is None:
            return
        while True:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT event_id,payload FROM epistemic_audit_outbox "
                    "WHERE delivered=0 ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row is None:
                    connection.commit()
                    return
                event_id, payload = row
                self._sink(event_id, json.loads(payload))
                connection.execute(
                    "UPDATE epistemic_audit_outbox SET delivered=1 WHERE event_id=?",
                    (event_id,),
                )
                connection.commit()

    def reconcile(self) -> None:
        """Verify chains and references, then safely rebuild derived indexes."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT rowid,commit_id,commit_digest,episode_id,prior_digest,snapshot_digest,document "
                "FROM epistemic_commits ORDER BY rowid"
            ).fetchall()
            commits: dict[str, EpistemicCommit] = {}
            heads: dict[str, str] = {}
            claims: list[tuple[str, str, str]] = []
            evidence: list[tuple[str, str, str, str, str, str]] = []
            for _, commit_id, digest, episode_id, prior, snapshot, document in rows:
                commit = self._decode(document)
                if (
                    commit.commit_id,
                    commit.commit_digest,
                    commit.episode_id,
                    commit.prior_commit_digest,
                    commit.snapshot_digest,
                ) != (commit_id, digest, episode_id, prior, snapshot):
                    raise EpistemicIntegrityError(
                        "commit columns diverge from canonical bytes"
                    )
                if prior != heads.get(episode_id):
                    raise EpistemicIntegrityError("per-episode commit chain is corrupt")
                commits[digest] = commit
                heads[episode_id] = digest
                claims.extend((episode_id, c.claim_id, digest) for c in commit.claims)
                evidence.extend(
                    (
                        episode_id,
                        e.evidence_id,
                        digest,
                        e.snapshot_digest,
                        e.content_digest,
                        e.provenance_id,
                    )
                    for e in commit.evidence
                )
            persisted_heads = dict(
                connection.execute(
                    "SELECT episode_id,commit_digest FROM epistemic_heads"
                )
            )
            if persisted_heads != heads:
                raise EpistemicIntegrityError("persisted epistemic heads are corrupt")
            # Semantic references are checked against reconstructed authoritative data.
            claim_sets: dict[str, set[str]] = {}
            evidence_sets: dict[str, set[str]] = {}
            evidence_map = {(e[0], e[1]): e[3:] for e in evidence}
            for commit in commits.values():
                known = claim_sets.setdefault(commit.episode_id, set())
                known_evidence = evidence_sets.setdefault(commit.episode_id, set())
                current = {claim.claim_id for claim in commit.claims}
                current_evidence = {item.evidence_id for item in commit.evidence}
                if known_evidence & current_evidence:
                    raise EpistemicIntegrityError(
                        "persisted evidence identifier is not immutable"
                    )
                for claim in commit.claims:
                    if (
                        claim.claim_id in known
                        or not set((*claim.contested_by, *claim.supersedes))
                        <= known | current
                    ):
                        raise EpistemicIntegrityError(
                            "persisted claim reference is corrupt"
                        )
                for item in commit.evidence:
                    if item.source_episode_id is not None and evidence_map.get(
                        (item.source_episode_id, item.evidence_id)
                    ) != (
                        item.snapshot_digest,
                        item.content_digest,
                        item.provenance_id,
                    ):
                        raise EpistemicIntegrityError(
                            "persisted evidence reference is corrupt"
                        )
                known.update(current)
                known_evidence.update(current_evidence)
            outbox_rows = connection.execute(
                "SELECT event_id,commit_digest,payload,delivered FROM epistemic_audit_outbox"
            ).fetchall()
            expected_outbox = {
                f"epistemic.commit:{commit.commit_id}": (
                    commit.commit_digest,
                    self._outbox_payload(
                        commit, f"epistemic.commit:{commit.commit_id}"
                    ),
                )
                for commit in commits.values()
            }
            if len(outbox_rows) != len(expected_outbox):
                raise EpistemicIntegrityError("epistemic audit outbox is incomplete")
            for event_id, digest, payload, delivered in outbox_rows:
                if expected_outbox.get(event_id) != (
                    digest,
                    payload,
                ) or delivered not in (
                    0,
                    1,
                ):
                    raise EpistemicIntegrityError("epistemic audit outbox is corrupt")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM epistemic_claim_index")
            connection.execute("DELETE FROM epistemic_evidence_index")
            connection.executemany(
                "INSERT INTO epistemic_claim_index VALUES (?,?,?)", claims
            )
            connection.executemany(
                "INSERT INTO epistemic_evidence_index VALUES (?,?,?,?,?,?)", evidence
            )
            connection.commit()

    @staticmethod
    def _decode(document: bytes | str) -> EpistemicCommit:
        try:
            return loads_commit(document)
        except Exception as exc:
            raise EpistemicIntegrityError("invalid canonical epistemic commit") from exc

    def close(self) -> None:
        """Flush pending audit effects; connections are operation-scoped."""
        self.deliver_outbox()
