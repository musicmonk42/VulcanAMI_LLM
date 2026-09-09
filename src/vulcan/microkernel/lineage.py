"""Durable causal lineage above bounded cognitive episodes.

Lineage records continuity facts only.  They are deliberately unrelated to user
or conversation identity and make no claim about personal identity or
consciousness.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from vulcan.constitution.primitives import Digest, canonical_json

from .episode import ArtifactRef, CognitiveEpisode, EpisodeRef, canonical_digest
from .episode_store import EpisodeStore, _insert_episode_genesis
from .principals import Principal

GENESIS_DIGEST = "0" * 64
AUTHORITY_KINDS = (
    "world",
    "self",
    "social",
    "normative",
    "domain",
    "memory",
    "capability",
    "csiu",
    "alignment",
)


class LineageError(RuntimeError):
    """Base fail-closed lineage error."""


class LineageConflict(LineageError):
    """The branch head no longer matches the command's expected head."""


class LineageIntegrityError(LineageError):
    """Persisted lineage history is invalid or has been modified."""


def _identifier(value: str, kind: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(f"{kind}-")
        or len(value) > 128
    ):
        raise ValueError(f"invalid {kind} identity")
    if any(not (c.isalnum() or c in "-_.:") for c in value):
        raise ValueError(f"invalid {kind} identity")
    return value


def _digest(value: str) -> str:
    return str(Digest.from_legacy_hex(value).hex)


@dataclass(frozen=True)
class AuthoritySnapshotRef:
    kind: str
    digest: str
    revision: str
    schema_version: str
    owner: str
    release_id: str

    def __post_init__(self) -> None:
        if self.kind not in AUTHORITY_KINDS:
            raise ValueError("unknown snapshot authority kind")
        for name in ("revision", "schema_version", "owner", "release_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError(f"snapshot authority {name} is invalid")
        _digest(self.digest)

    def to_json(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "owner": self.owner,
            "release_id": self.release_id,
            "revision": self.revision,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class LineageHeadRef:
    branch_id: str
    digest: str

    def __post_init__(self) -> None:
        _identifier(self.branch_id, "branch")
        _digest(self.digest)

    def to_json(self) -> dict[str, str]:
        return {"branch_id": self.branch_id, "digest": self.digest}


@dataclass(frozen=True)
class LineageState:
    lineage_id: str
    branch_id: str
    instance_id: str
    tick: int
    prior_state_digest: str
    current_authority_snapshots: tuple[AuthoritySnapshotRef, ...] = ()
    active_commitments: tuple[ArtifactRef, ...] = ()
    active_episode_refs: tuple[EpisodeRef, ...] = ()
    past_episode_refs: tuple[EpisodeRef, ...] = ()
    pending_effect_refs: tuple[ArtifactRef, ...] = ()
    branch_parents: tuple[LineageHeadRef, ...] = ()
    suspended: bool = False
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.lineage_id, "lineage")
        _identifier(self.branch_id, "branch")
        _identifier(self.instance_id, "instance")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("lineage tick cannot be negative")
        if type(self.suspended) is not bool:
            raise ValueError("lineage suspension state must be Boolean")
        _digest(self.prior_state_digest)
        for values in (self.active_commitments, self.pending_effect_refs):
            identities = [value.artifact_id for value in values]
            if len(identities) != len(set(identities)):
                raise ValueError("duplicate lineage reference")
        kinds = [ref.kind for ref in self.current_authority_snapshots]
        if len(kinds) != len(set(kinds)):
            raise ValueError("duplicate authority snapshot kind")
        if kinds and tuple(kinds) != AUTHORITY_KINDS:
            raise ValueError("lineage requires the ordered nine-authority snapshot set")
        episode_ids = [
            r.episode_id for r in (*self.active_episode_refs, *self.past_episode_refs)
        ]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode cannot be both active and past")
        parent_branches = [ref.branch_id for ref in self.branch_parents]
        if len(parent_branches) != len(set(parent_branches)):
            raise ValueError("duplicate branch parent")
        if self.branch_id in parent_branches:
            raise ValueError("branch cannot be its own parent")
        object.__setattr__(self, "digest", canonical_digest(self.to_json(False)))

    @property
    def current_state_digest(self) -> str:
        return self.digest

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "active_commitments": [ref.to_json() for ref in self.active_commitments],
            "active_episode_refs": [r.to_json() for r in self.active_episode_refs],
            "branch_id": self.branch_id,
            "branch_parents": [ref.to_json() for ref in self.branch_parents],
            "current_authority_snapshots": [
                r.to_json() for r in self.current_authority_snapshots
            ],
            "instance_id": self.instance_id,
            "lineage_id": self.lineage_id,
            "past_episode_refs": [r.to_json() for r in self.past_episode_refs],
            "pending_effect_refs": [ref.to_json() for ref in self.pending_effect_refs],
            "prior_state_digest": self.prior_state_digest,
            "schema_version": "lineage-state.v1",
            "suspended": self.suspended,
            "tick": self.tick,
        }
        if include_digest:
            value["current_state_digest"] = self.digest
        return value

    def successor(self, **changes: object) -> "LineageState":
        if self.suspended and changes.get("suspended", True):
            raise LineageError("suspended lineage must be resumed before advancement")
        return replace(  # type: ignore[arg-type]
            self, tick=self.tick + 1, prior_state_digest=self.digest, **changes
        )


def _decode(document: str) -> LineageState:
    try:
        raw = json.loads(document)
        if raw.get("schema_version") != "lineage-state.v1":
            raise ValueError("unsupported lineage schema")
        state = LineageState(
            lineage_id=raw["lineage_id"],
            branch_id=raw["branch_id"],
            instance_id=raw["instance_id"],
            tick=raw["tick"],
            prior_state_digest=raw["prior_state_digest"],
            current_authority_snapshots=tuple(
                AuthoritySnapshotRef(**x) for x in raw["current_authority_snapshots"]
            ),
            active_commitments=tuple(
                ArtifactRef(**x) for x in raw["active_commitments"]
            ),
            active_episode_refs=tuple(
                EpisodeRef(x["episode_id"], x["digest"])
                for x in raw["active_episode_refs"]
            ),
            past_episode_refs=tuple(
                EpisodeRef(x["episode_id"], x["digest"])
                for x in raw["past_episode_refs"]
            ),
            pending_effect_refs=tuple(
                ArtifactRef(**x) for x in raw["pending_effect_refs"]
            ),
            branch_parents=tuple(LineageHeadRef(**x) for x in raw["branch_parents"]),
            suspended=raw["suspended"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LineageIntegrityError("invalid lineage document") from exc
    if (
        raw.get("current_state_digest") != state.digest
        or canonical_json(raw).decode() != document
    ):
        raise LineageIntegrityError("lineage digest or canonical document mismatch")
    return state


class LineageStore:
    """SQLite event stream with exactly one compare-and-swap head per branch."""

    def __init__(
        self, path: str | Path, *, failpoint: Callable[[str], None] | None = None
    ) -> None:
        self.path = str(path)
        self._failpoint = failpoint
        p = Path(path)
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise LineageIntegrityError("lineage database path is unsafe")
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS lineage_heads(branch_id TEXT PRIMARY KEY, lineage_id TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lineage_events(branch_id TEXT NOT NULL, tick INTEGER NOT NULL, prior_digest TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, operation TEXT NOT NULL, authority_digest TEXT NOT NULL, document TEXT NOT NULL, event_digest TEXT NOT NULL UNIQUE, PRIMARY KEY(branch_id,tick));
            """)
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(lineage_events)")
            }
            if "event_digest" not in columns:
                db.execute("ALTER TABLE lineage_events ADD COLUMN event_digest TEXT")
                rows = db.execute("SELECT * FROM lineage_events").fetchall()
                for row in rows:
                    db.execute(
                        "UPDATE lineage_events SET event_digest=? WHERE branch_id=? AND tick=?",
                        (self._event_digest(row), row["branch_id"], row["tick"]),
                    )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS lineage_event_digest_unique "
                "ON lineage_events(event_digest)"
            )
        os.chmod(p, 0o600)
        self.verify_all()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    def create(
        self, state: LineageState, *, operation: str, authority_digest: str
    ) -> LineageState:
        if state.tick != 0 or state.prior_state_digest != GENESIS_DIGEST:
            raise LineageIntegrityError("lineage genesis is invalid")
        return self._write(None, state, operation, authority_digest)

    def advance(
        self,
        expected: str,
        state: LineageState,
        *,
        operation: str,
        authority_digest: str,
    ) -> LineageState:
        if state.prior_state_digest != expected:
            raise LineageIntegrityError(
                "lineage successor is not bound to expected head"
            )
        return self._write(expected, state, operation, authority_digest)

    def _write(
        self,
        expected: str | None,
        state: LineageState,
        operation: str,
        authority_digest: str,
    ) -> LineageState:
        _digest(authority_digest)
        document = canonical_json(state.to_json()).decode()
        _decode(document)
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                self._write_on_connection(
                    db, expected, state, operation, authority_digest, document
                )
                if self._failpoint is not None:
                    self._failpoint("before_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return state

    def _write_on_connection(
        self,
        db: sqlite3.Connection,
        expected: str | None,
        state: LineageState,
        operation: str,
        authority_digest: str,
        document: str | None = None,
    ) -> None:
        if not operation or len(operation) > 256 or any(ord(c) < 32 for c in operation):
            raise LineageIntegrityError("invalid lineage operation")
        _digest(authority_digest)
        document = document or canonical_json(state.to_json()).decode()
        row = db.execute(
            "SELECT digest,document FROM lineage_heads WHERE branch_id=?",
            (state.branch_id,),
        ).fetchone()
        actual = row["digest"] if row else None
        if actual != expected:
            raise LineageConflict("stale lineage branch head")
        if row is not None:
            prior = _decode(row["document"])
            if state.tick != prior.tick + 1 or state.lineage_id != prior.lineage_id:
                raise LineageIntegrityError(
                    "lineage identity or monotonic tick changed"
                )
        if expected is None:
            db.execute(
                "INSERT INTO lineage_heads VALUES(?,?,?,?)",
                (state.branch_id, state.lineage_id, state.digest, document),
            )
        else:
            changed = db.execute(
                "UPDATE lineage_heads SET digest=?,document=? WHERE branch_id=? AND digest=?",
                (state.digest, document, state.branch_id, expected),
            ).rowcount
            if changed != 1:
                raise LineageConflict("lineage compare-and-swap failed")
        event_digest = self._event_digest(
            {
                "authority_digest": authority_digest,
                "branch_id": state.branch_id,
                "operation": operation,
                "prior_digest": state.prior_state_digest,
                "digest": state.digest,
                "tick": state.tick,
            }
        )
        db.execute(
            "INSERT INTO lineage_events VALUES(?,?,?,?,?,?,?,?)",
            (
                state.branch_id,
                state.tick,
                state.prior_state_digest,
                state.digest,
                operation,
                authority_digest,
                document,
                event_digest,
            ),
        )

    @staticmethod
    def _event_digest(row: Mapping[str, object]) -> str:
        return canonical_digest(
            {
                "authority_digest": row["authority_digest"],
                "branch_id": row["branch_id"],
                "operation": row["operation"],
                "prior_state_digest": row["prior_digest"],
                "state_digest": row["digest"],
                "tick": row["tick"],
            }
        )

    def admit_episode_atomically(
        self,
        episode_store: EpisodeStore,
        expected: str,
        state: LineageState,
        episode: CognitiveEpisode,
        *,
        authority_digest: str,
    ) -> LineageState:
        """Commit episode genesis and its lineage admission in one SQLite txn."""
        if Path(episode_store.path).resolve() != Path(self.path).resolve():
            raise LineageIntegrityError(
                "atomic admission requires one constitutional database"
            )
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                _insert_episode_genesis(db, episode)
                if self._failpoint is not None:
                    self._failpoint("after_episode_genesis")
                self._write_on_connection(
                    db, expected, state, "episode.admitted", authority_digest
                )
                if self._failpoint is not None:
                    self._failpoint("before_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        episode_store.deliver_outbox()
        return state

    def load(self, branch_id: str) -> LineageState:
        with self._connect() as db:
            row = db.execute(
                "SELECT document FROM lineage_heads WHERE branch_id=?", (branch_id,)
            ).fetchone()
        if row is None:
            raise LineageError("unknown lineage branch")
        return _decode(row["document"])

    def replay(self, branch_id: str) -> LineageState:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM lineage_events WHERE branch_id=? ORDER BY tick",
                (branch_id,),
            ).fetchall()
        if not rows:
            raise LineageError("unknown lineage branch")
        prior = GENESIS_DIGEST
        for ordinal, row in enumerate(rows):
            state = _decode(row["document"])
            expected_event_digest = self._event_digest(row)
            if (
                state.tick != ordinal
                or row["prior_digest"] != prior
                or state.prior_state_digest != prior
                or row["digest"] != state.digest
                or row["event_digest"] != expected_event_digest
            ):
                raise LineageIntegrityError("lineage event chain is invalid")
            prior = state.digest
        return state

    def verify_all(self) -> None:
        with self._connect() as db:
            branches = [r[0] for r in db.execute("SELECT branch_id FROM lineage_heads")]
            orphan = db.execute(
                "SELECT 1 FROM lineage_events e LEFT JOIN lineage_heads h USING(branch_id) "
                "WHERE h.branch_id IS NULL LIMIT 1"
            ).fetchone()
        if orphan is not None:
            raise LineageIntegrityError("lineage event has no branch head")
        for branch in branches:
            if self.replay(branch).digest != self.load(branch).digest:
                raise LineageIntegrityError("lineage replay does not reproduce head")


class LineageTransactionService:
    """Kernel-only lifecycle operations for causal lineage."""

    def __init__(self, store: LineageStore, principal: Principal) -> None:
        if not principal.is_kernel:
            raise LineageError("only SYSTEM_KERNEL may advance lineage")
        self.store, self.principal = store, principal

    def genesis(
        self, lineage_id: str, branch_id: str, instance_id: str
    ) -> LineageState:
        state = LineageState(lineage_id, branch_id, instance_id, 0, GENESIS_DIGEST)
        return self.store.create(
            state, operation="genesis", authority_digest=self.principal.identity_digest
        )

    def admit_episode(
        self,
        branch_id: str,
        expected: str,
        episode: CognitiveEpisode,
        snapshots: Sequence[AuthoritySnapshotRef],
        episode_store: EpisodeStore,
    ) -> LineageState:
        head = self.store.load(branch_id)
        if head.digest != expected:
            raise LineageConflict("stale lineage branch head")
        if head.suspended:
            raise LineageError("cannot admit an episode while suspended")
        successor = head.successor(
            current_authority_snapshots=tuple(snapshots),
            active_episode_refs=(
                *head.active_episode_refs,
                EpisodeRef(episode.episode_id, episode.digest),
            ),
        )
        return self.store.admit_episode_atomically(
            episode_store,
            expected,
            successor,
            episode,
            authority_digest=self.principal.identity_digest,
        )

    def complete_episode(
        self, branch_id: str, episode: CognitiveEpisode, episode_store: EpisodeStore
    ) -> LineageState:
        """Move a terminal bounded transaction from active to immutable history."""
        durable = episode_store.load(episode.episode_id)
        if durable.digest != episode.digest or not durable.state.is_terminal:
            raise LineageIntegrityError(
                "only the exact durable terminal episode may enter lineage history"
            )
        episode_ref = EpisodeRef(episode.episode_id, episode.digest)
        for _ in range(16):
            head = self.store.load(branch_id)
            active = tuple(
                ref
                for ref in head.active_episode_refs
                if ref.episode_id != episode_ref.episode_id
            )
            if len(active) == len(head.active_episode_refs):
                if episode_ref in head.past_episode_refs:
                    return head
                raise LineageError("episode is not active on this branch")
            past = (*head.past_episode_refs, episode_ref)
            try:
                return self.store.advance(
                    head.digest,
                    head.successor(active_episode_refs=active, past_episode_refs=past),
                    operation="episode.completed",
                    authority_digest=self.principal.identity_digest,
                )
            except LineageConflict:
                continue
        raise LineageConflict("lineage completion contention did not converge")

    def suspend(self, branch_id: str, expected: str) -> LineageState:
        head = self.store.load(branch_id)
        return self.store.advance(
            expected,
            head.successor(suspended=True),
            operation="suspend",
            authority_digest=self.principal.identity_digest,
        )

    def resume(self, branch_id: str, expected: str, instance_id: str) -> LineageState:
        head = self.store.load(branch_id)
        if not head.suspended:
            raise LineageError("only a suspended branch may resume")
        return self.store.advance(
            expected,
            replace(
                head,
                tick=head.tick + 1,
                prior_state_digest=head.digest,
                instance_id=_identifier(instance_id, "instance"),
                suspended=False,
            ),
            operation="resume",
            authority_digest=self.principal.identity_digest,
        )

    def restart(self, branch_id: str, expected: str, instance_id: str) -> LineageState:
        """Record process replacement without pretending it is the old instance."""
        head = self.store.load(branch_id)
        if head.digest != expected:
            raise LineageConflict("stale restart head")
        return self.store.advance(
            expected,
            replace(
                head,
                tick=head.tick + 1,
                prior_state_digest=head.digest,
                instance_id=_identifier(instance_id, "instance"),
            ),
            operation="restart",
            authority_digest=self.principal.identity_digest,
        )

    def fork(
        self,
        source_branch: str,
        expected: str,
        branch_id: str,
        instance_id: str,
        *,
        operation: str = "fork",
    ) -> LineageState:
        source = self.store.load(source_branch)
        if source.digest != expected:
            raise LineageConflict("stale source branch head")
        # A new identity prevents a clone from impersonating its source branch.
        child = LineageState(
            lineage_id=source.lineage_id,
            branch_id=_identifier(branch_id, "branch"),
            instance_id=_identifier(instance_id, "instance"),
            tick=0,
            prior_state_digest=GENESIS_DIGEST,
            current_authority_snapshots=source.current_authority_snapshots,
            active_commitments=source.active_commitments,
            past_episode_refs=source.past_episode_refs,
            pending_effect_refs=source.pending_effect_refs,
            branch_parents=(LineageHeadRef(source.branch_id, source.digest),),
        )
        return self.store.create(
            child, operation=operation, authority_digest=self.principal.identity_digest
        )

    def clone(
        self, source_branch: str, expected: str, branch_id: str, instance_id: str
    ) -> LineageState:
        return self.fork(
            source_branch, expected, branch_id, instance_id, operation="clone"
        )

    def merge(
        self,
        target_branch: str,
        target_expected: str,
        source_branch: str,
        source_expected: str,
    ) -> LineageState:
        target, source = self.store.load(target_branch), self.store.load(source_branch)
        if target.digest != target_expected or source.digest != source_expected:
            raise LineageConflict("stale merge head")
        commitments = tuple(
            dict.fromkeys((*target.active_commitments, *source.active_commitments))
        )
        effects = tuple(
            dict.fromkeys((*target.pending_effect_refs, *source.pending_effect_refs))
        )
        past_by_id = {ref.episode_id: ref for ref in target.past_episode_refs}
        for ref in source.past_episode_refs:
            existing = past_by_id.get(ref.episode_id)
            if existing is not None and existing != ref:
                raise LineageIntegrityError("merge has conflicting episode history")
            past_by_id.setdefault(ref.episode_id, ref)
        return self.store.advance(
            target_expected,
            target.successor(
                active_commitments=commitments,
                pending_effect_refs=effects,
                past_episode_refs=tuple(past_by_id.values()),
                branch_parents=tuple(
                    dict.fromkeys(
                        (
                            *target.branch_parents,
                            LineageHeadRef(source.branch_id, source.digest),
                        )
                    )
                ),
            ),
            operation=f"merge:{source.branch_id}:{source.digest}",
            authority_digest=self.principal.identity_digest,
        )
