"""Durable compare-and-swap authority for :class:`CognitiveEpisode`.

The store deliberately persists only the privacy-preserving canonical episode
document.  Request/provider bytes and reasoning traces are not accepted by this
interface.  Each head change, immutable transition, and audit-outbox item is one
SQLite transaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .episode import (
    ActorBinding,
    ArtifactRef,
    CognitiveEpisode,
    EpisodeRef,
    RequestBinding,
    SnapshotBundleRef,
    TransitionEvent,
)
from .state_machine import EpisodeState, EpisodeTransitionError, ensure_transition

_SCHEMA_VERSION = 2


class EpisodeStoreError(RuntimeError):
    """Base class for fail-closed durable episode errors."""


class EpisodeConflict(EpisodeStoreError):
    """The persisted head did not match the caller's expected digest."""


class EpisodeIntegrityError(EpisodeStoreError):
    """Persisted bytes do not reconstruct their claimed digest chain."""


Failpoint = Callable[[str], None]
OutboxSink = Callable[[str, dict[str, object]], None]


def _artifact(value: dict[str, str]) -> ArtifactRef:
    return ArtifactRef(value["artifact_id"], value["digest"], value["kind"])


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EpisodeIntegrityError(f"duplicate persisted JSON key: {key}")
        value[key] = item
    return value


def episode_from_document(document: str) -> CognitiveEpisode:
    """Strictly reconstruct and digest-check a canonical persisted document."""
    try:
        value = json.loads(document, object_pairs_hook=_reject_duplicate_keys)
        transitions = tuple(
            TransitionEvent(
                event_id=item["event_id"],
                from_state=EpisodeState(item["from_state"]),
                to_state=EpisodeState(item["to_state"]),
                at=_utc_datetime(item["at"]),
                reason=item["reason"],
                authority=item["authority"],
                prior_digest=item["prior_digest"],
                snapshot_ids=tuple(item["snapshot_ids"]),
                evidence_refs=tuple(_artifact(ref) for ref in item["evidence_refs"]),
            )
            for item in value["transitions"]
        )
        actor = ActorBinding(**value["actor"])
        request = RequestBinding(**value["request"])
        parent = EpisodeRef(**value["parent"]) if value["parent"] else None
        snapshot = (
            SnapshotBundleRef(**value["snapshot_bundle"])
            if value["snapshot_bundle"]
            else None
        )
        episode = CognitiveEpisode(
            episode_id=value["episode_id"],
            actor=actor,
            request=request,
            state=EpisodeState(value["state"]),
            schema_version=value["schema_version"],
            conversation_id=value["conversation_id"],
            parent=parent,
            snapshot_bundle=snapshot,
            interpretation=value["interpretation"],
            claims=tuple(_artifact(item) for item in value["claims"]),
            evidence=tuple(_artifact(item) for item in value["evidence"]),
            derivations=tuple(_artifact(item) for item in value["derivations"]),
            candidate_plans=tuple(_artifact(item) for item in value["candidate_plans"]),
            authorization=(
                _artifact(value["authorization"]) if value["authorization"] else None
            ),
            effects=tuple(_artifact(item) for item in value["effects"]),
            response=_artifact(value["response"]) if value["response"] else None,
            consolidation_refs=tuple(
                _artifact(item) for item in value["consolidation_refs"]
            ),
            transitions=transitions,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EpisodeIntegrityError("invalid persisted episode document") from exc
    if value.get("digest") != episode.digest or episode.canonical_json() != document:
        raise EpisodeIntegrityError(
            "persisted episode digest or canonical JSON is invalid"
        )
    _verify_chain(episode)
    return episode


def _verify_chain(episode: CognitiveEpisode) -> None:
    if not episode.transitions:
        raise EpisodeIntegrityError("episode has no genesis transition")
    for index, event in enumerate(episode.transitions):
        if index == 0:
            if (
                event.prior_digest != "0" * 64
                or event.from_state is not EpisodeState.PERCEIVED
                or event.to_state is not EpisodeState.PERCEIVED
            ):
                raise EpisodeIntegrityError("genesis prior digest is invalid")
        if index and event.from_state is not episode.transitions[index - 1].to_state:
            raise EpisodeIntegrityError("transition state chain is invalid")
    if episode.state is not episode.transitions[-1].to_state:
        raise EpisodeIntegrityError("episode state does not match transition head")


def _utc_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise EpisodeIntegrityError("persisted transition timestamp is not text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EpisodeIntegrityError("persisted transition timestamp is not UTC")
    return parsed


def _validate_successor(prior: CognitiveEpisode, successor: CognitiveEpisode) -> None:
    """Reject forged history, identity mutation, rollback, and state skipping."""
    if len(successor.transitions) != len(prior.transitions) + 1:
        raise EpisodeIntegrityError("advance requires exactly one new transition")
    if successor.transitions[:-1] != prior.transitions:
        raise EpisodeIntegrityError("successor rewrites immutable transition history")
    event = successor.transitions[-1]
    if event.prior_digest != prior.digest or event.from_state is not prior.state:
        raise EpisodeIntegrityError("successor is not bound to the durable prior head")
    try:
        ensure_transition(prior.state, successor.state)
    except EpisodeTransitionError as exc:
        raise EpisodeIntegrityError("successor state transition is invalid") from exc
    if event.to_state is not successor.state:
        raise EpisodeIntegrityError("successor event and state disagree")
    immutable = (
        "episode_id",
        "actor",
        "request",
        "schema_version",
        "conversation_id",
        "parent",
    )
    if any(getattr(prior, name) != getattr(successor, name) for name in immutable):
        raise EpisodeIntegrityError("successor mutates immutable episode identity")
    for name in (
        "claims",
        "evidence",
        "derivations",
        "candidate_plans",
        "effects",
        "consolidation_refs",
    ):
        old = getattr(prior, name)
        new = getattr(successor, name)
        if new[: len(old)] != old:
            raise EpisodeIntegrityError(f"successor rewrites {name}")
    for name in ("authorization", "response"):
        old = getattr(prior, name)
        if old is not None and getattr(successor, name) != old:
            raise EpisodeIntegrityError(f"successor rewrites {name}")
    if prior.interpretation and successor.interpretation != prior.interpretation:
        raise EpisodeIntegrityError("successor rewrites interpretation")
    if prior.snapshot_bundle != successor.snapshot_bundle and not any(
        ref.kind == "snapshot-rebase-authorization.v1" for ref in event.evidence_refs
    ):
        raise EpisodeIntegrityError(
            "successor changes snapshot without rebase evidence"
        )


def _outbox_payload(episode: CognitiveEpisode, event: TransitionEvent) -> str:
    snapshot_digest = (
        episode.snapshot_bundle.state_digest
        if episode.snapshot_bundle is not None
        else "0" * 64
    )
    return json.dumps(
        {
            "episode_id": episode.episode_id,
            "transition_digest": event.event_digest,
            "transition": event.to_json(),
            "from_state": event.from_state.value,
            "to_state": event.to_state.value,
            "prior_episode_digest": event.prior_digest,
            "resulting_episode_digest": episode.digest,
            # Read compatibility for v1 EpisodeStore sink consumers.  Remove
            # after all external sinks consume resulting_episode_digest.
            "episode_digest": episode.digest,
            "snapshot_bundle_digest": snapshot_digest,
            "authority_references": [event.authority],
            "policy_references": [
                ref.to_json()
                for ref in event.evidence_refs
                if ref.kind == "constitutional-policy-reference.v1"
            ],
            "evidence_references": [ref.to_json() for ref in event.evidence_refs],
            "transaction_id": event.event_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class EpisodeStore:
    """SQLite episode head, immutable transitions, and transactional outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        outbox_sink: OutboxSink | None = None,
        failpoint: Failpoint | None = None,
    ) -> None:
        database_path = Path(path)
        self.path = str(database_path)
        if self.path == ":memory:":
            raise ValueError("EpisodeStore requires a durable filesystem path")
        if database_path.is_symlink():
            raise EpisodeIntegrityError("episode database path must not be a symlink")
        if database_path.exists() and not database_path.is_file():
            raise EpisodeIntegrityError("episode database path is not a regular file")
        self._outbox_sink = outbox_sink
        self._failpoint = failpoint
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS episode_heads (
                    episode_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL UNIQUE,
                    document TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_transitions (
                    episode_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    prior_digest TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_document TEXT NOT NULL,
                    episode_document TEXT NOT NULL,
                    PRIMARY KEY (episode_id, ordinal),
                    UNIQUE (episode_id, digest)
                );
                CREATE TABLE IF NOT EXISTS episode_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    transition_digest TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    delivered_at TEXT
                );
                """)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, _SCHEMA_VERSION):
                raise EpisodeIntegrityError("unsupported episode store schema version")
            if version == 1:
                rows = connection.execute(
                    """SELECT o.id, t.episode_document
                       FROM episode_outbox o
                       JOIN episode_transitions t
                         ON t.episode_id=o.episode_id AND t.digest=o.transition_digest"""
                ).fetchall()
                for row in rows:
                    migrated_episode = episode_from_document(row["episode_document"])
                    connection.execute(
                        "UPDATE episode_outbox SET payload=?, delivered_at=NULL WHERE id=?",
                        (
                            _outbox_payload(
                                migrated_episode, migrated_episode.transitions[-1]
                            ),
                            row["id"],
                        ),
                    )
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        os.chmod(database_path, 0o600)
        self.verify_all()
        self.deliver_outbox()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            # Durability settings are connection-scoped and therefore belong on
            # every operation, not only the schema-initialization connection.
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA trusted_schema=OFF")
            yield connection
        finally:
            connection.close()

    def _hit(self, name: str) -> None:
        if self._failpoint is not None:
            self._failpoint(name)

    def create(self, episode: CognitiveEpisode) -> CognitiveEpisode:
        """Persist genesis, idempotently only when its identity and digest agree."""
        if len(episode.transitions) != 1:
            raise EpisodeIntegrityError(
                "create requires exactly one genesis transition"
            )
        self._write(None, episode)
        return episode

    def advance(
        self,
        episode_id: str,
        expected_prior_digest: str,
        next_episode: CognitiveEpisode,
    ) -> CognitiveEpisode:
        if next_episode.episode_id != episode_id:
            raise EpisodeIntegrityError("episode identity changed during advancement")
        if (
            not next_episode.transitions
            or next_episode.transitions[-1].prior_digest != expected_prior_digest
        ):
            raise EpisodeIntegrityError("next transition is not bound to expected head")
        self._write(expected_prior_digest, next_episode)
        return next_episode

    def _write(self, expected: str | None, episode: CognitiveEpisode) -> None:
        document = episode.canonical_json()
        # Treat even in-process aggregate instances as untrusted store input.
        episode_from_document(document)
        event = episode.transitions[-1]
        event_document = json.dumps(
            event.to_json(), sort_keys=True, separators=(",", ":")
        )
        payload = _outbox_payload(episode, event)
        self._hit("before_db_write")
        committed = False
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT digest FROM episode_heads WHERE episode_id=?",
                    (episode.episode_id,),
                ).fetchone()
                if expected is None:
                    if row is not None:
                        if row["digest"] == episode.digest:
                            connection.rollback()
                            return
                        raise EpisodeConflict(
                            "episode identity already has a different digest"
                        )
                    self._hit("before_head_cas")
                    connection.execute(
                        "INSERT INTO episode_heads VALUES (?, ?, ?)",
                        (episode.episode_id, episode.digest, document),
                    )
                else:
                    if row is None:
                        raise EpisodeConflict("episode head is missing")
                    if row["digest"] != expected:
                        duplicate = connection.execute(
                            "SELECT 1 FROM episode_transitions WHERE episode_id=? AND prior_digest=? AND digest=?",
                            (episode.episode_id, expected, episode.digest),
                        ).fetchone()
                        if duplicate:
                            connection.rollback()
                            return
                        raise EpisodeConflict("episode head compare-and-swap failed")
                    prior_row = connection.execute(
                        "SELECT document FROM episode_heads WHERE episode_id=?",
                        (episode.episode_id,),
                    ).fetchone()
                    if prior_row is None:
                        raise EpisodeConflict("episode head disappeared")
                    _validate_successor(
                        episode_from_document(prior_row["document"]), episode
                    )
                    self._hit("before_head_cas")
                    changed = connection.execute(
                        "UPDATE episode_heads SET digest=?, document=? WHERE episode_id=? AND digest=?",
                        (episode.digest, document, episode.episode_id, expected),
                    ).rowcount
                    if changed != 1:
                        raise EpisodeConflict("episode head compare-and-swap failed")
                self._hit("after_head_cas")
                connection.execute(
                    "INSERT INTO episode_transitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        episode.episode_id,
                        len(episode.transitions) - 1,
                        event.prior_digest,
                        episode.digest,
                        event.event_id,
                        event_document,
                        document,
                    ),
                )
                connection.execute(
                    "INSERT INTO episode_outbox (episode_id, transition_digest, payload) VALUES (?, ?, ?)",
                    (episode.episode_id, episode.digest, payload),
                )
                self._hit("after_db_write")
                self._hit("before_commit")
                connection.commit()
                committed = True
            except BaseException:
                if not committed:
                    connection.rollback()
                raise
        self._hit("after_commit")
        self.deliver_outbox()

    def load(self, episode_id: str) -> CognitiveEpisode:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT digest, document FROM episode_heads WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        episode = episode_from_document(row["document"])
        if episode.digest != row["digest"]:
            raise EpisodeIntegrityError(
                "stored head digest does not match its document"
            )
        return episode

    def replay(self, episode_id: str) -> CognitiveEpisode:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT ordinal, prior_digest, digest, event_id, event_document, episode_document FROM episode_transitions WHERE episode_id=? ORDER BY ordinal",
                (episode_id,),
            ).fetchall()
        if not rows:
            raise KeyError(episode_id)
        prior: CognitiveEpisode | None = None
        for ordinal, row in enumerate(rows):
            if row["ordinal"] != ordinal:
                raise EpisodeIntegrityError("transition ordinal chain has a gap")
            current = episode_from_document(row["episode_document"])
            expected_event = json.dumps(
                current.transitions[-1].to_json(),
                sort_keys=True,
                separators=(",", ":"),
            )
            if (
                current.digest != row["digest"]
                or row["prior_digest"] != current.transitions[-1].prior_digest
                or row["event_id"] != current.transitions[-1].event_id
                or row["event_document"] != expected_event
            ):
                raise EpisodeIntegrityError("transition row digest is invalid")
            if prior is not None and row["prior_digest"] != prior.digest:
                raise EpisodeIntegrityError(
                    "transition prior digest does not match replay head"
                )
            prior = current
        if prior is None:  # defensive: rows was checked above
            raise EpisodeIntegrityError("episode replay produced no head")
        head = self.load(episode_id)
        if prior.digest != head.digest:
            raise EpisodeIntegrityError("replayed digest does not match stored head")
        return prior

    def verify_all(self) -> None:
        with self._connection() as connection:
            identities = [
                row[0]
                for row in connection.execute("SELECT episode_id FROM episode_heads")
            ]
            orphan_transition = connection.execute(
                "SELECT 1 FROM episode_transitions t LEFT JOIN episode_heads h USING (episode_id) WHERE h.episode_id IS NULL LIMIT 1"
            ).fetchone()
            missing_outbox = connection.execute("""SELECT 1 FROM episode_transitions t
                   LEFT JOIN episode_outbox o
                     ON o.episode_id=t.episode_id AND o.transition_digest=t.digest
                   WHERE o.id IS NULL LIMIT 1""").fetchone()
            outbox_rows = connection.execute(
                """SELECT o.episode_id, o.transition_digest, o.payload,
                          t.episode_document
                   FROM episode_outbox o
                   LEFT JOIN episode_transitions t
                     ON t.episode_id=o.episode_id AND t.digest=o.transition_digest"""
            ).fetchall()
        if orphan_transition is not None or missing_outbox is not None:
            raise EpisodeIntegrityError("episode store relational integrity is invalid")
        for episode_id in identities:
            self.replay(episode_id)
        for row in outbox_rows:
            if row["episode_document"] is None:
                raise EpisodeIntegrityError("outbox row has no immutable transition")
            episode = episode_from_document(row["episode_document"])
            expected = _outbox_payload(episode, episode.transitions[-1])
            try:
                decoded = json.loads(
                    row["payload"], object_pairs_hook=_reject_duplicate_keys
                )
            except (TypeError, json.JSONDecodeError) as exc:
                raise EpisodeIntegrityError("outbox payload is invalid") from exc
            canonical = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
            if (
                row["episode_id"] != episode.episode_id
                or row["transition_digest"] != episode.digest
                or canonical != expected
                or row["payload"] != canonical
            ):
                raise EpisodeIntegrityError("outbox payload does not match transition")

    def deliver_outbox(self) -> int:
        if self._outbox_sink is None:
            return 0
        delivered = 0
        while True:
            with self._connection() as connection:
                row = connection.execute(
                    """SELECT o.id, o.episode_id, o.transition_digest, o.payload,
                              t.episode_document
                       FROM episode_outbox o
                       LEFT JOIN episode_transitions t
                         ON t.episode_id=o.episode_id
                        AND t.digest=o.transition_digest
                       WHERE o.delivered_at IS NULL ORDER BY o.id LIMIT 1"""
                ).fetchone()
            if row is None:
                return delivered
            if row["episode_document"] is None:
                raise EpisodeIntegrityError("outbox row has no immutable transition")
            episode = episode_from_document(row["episode_document"])
            try:
                payload = json.loads(
                    row["payload"], object_pairs_hook=_reject_duplicate_keys
                )
            except (TypeError, json.JSONDecodeError) as exc:
                raise EpisodeIntegrityError("outbox payload is invalid") from exc
            if (
                row["episode_id"] != episode.episode_id
                or row["transition_digest"] != episode.digest
                or row["payload"] != _outbox_payload(episode, episode.transitions[-1])
            ):
                raise EpisodeIntegrityError("outbox payload does not match transition")
            self._hit("before_outbox_delivery")
            self._outbox_sink("episode.transitioned", payload)
            self._hit("after_outbox_delivery")
            with self._connection() as connection:
                connection.execute(
                    "UPDATE episode_outbox SET delivered_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=? AND delivered_at IS NULL",
                    (row["id"],),
                )
            delivered += 1

    def close(self) -> None:
        """Connections are operation-scoped; provided for container ownership."""
