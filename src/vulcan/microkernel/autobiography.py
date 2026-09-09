"""Kernel-owned causal autobiography, distinct from immutable audit.

The store contains approved, content-addressed projections only.  It never
accepts provider text or prose summaries and it does not infer continuity:
every record binds the durable lineage and episode/reafference chain supplied
by the constitutional kernel.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, cast

from vulcan.constitution.primitives import Digest, canonical_json

from .principals import Principal


class AutobiographyError(RuntimeError):
    """Causal memory failed closed."""


class AutobiographyConflict(AutobiographyError):
    """A replay, stale writer, or conflicting biography was rejected."""


class AutobiographyCrash(BaseException):
    """Test-only process termination at a durable transaction boundary."""


class BiographyChainAuthority(Protocol):
    """Validates exact references against durable constitutional authorities."""

    def verify(self, episode: "AutobiographicalEpisode") -> None: ...


class AutobiographyFailpoint(Protocol):
    def hit(self, name: str) -> None: ...


class NoopAutobiographyFailpoint:
    def hit(self, name: str) -> None:
        return None


def _strict_json(document: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise AutobiographyError("duplicate persisted JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(document, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AutobiographyError("invalid persisted JSON") from exc
    if not isinstance(value, dict) or canonical_json(value).decode() != document:
        raise AutobiographyError("persisted JSON is not canonical")
    return value


def _retention(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class FactKind(str, Enum):
    OBSERVATION = "observation"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    OUTCOME = "outcome"


class MemoryState(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    TOMBSTONED = "tombstoned"


def _digest(value: object) -> str:
    return cast(str, Digest.of_bytes(canonical_json(value)).hex)


def _valid_digest(value: str) -> None:
    Digest.from_legacy_hex(value)


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class CausalFact:
    """An approved typed fact reference, never raw provider content."""

    fact_id: str
    kind: FactKind
    artifact_ref: str
    source_episode_ref: str
    causal_parents: tuple[str, ...] = ()
    policy_refs: tuple[str, ...] = ()
    model_refs: tuple[str, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.fact_id, "fact id")
        if not isinstance(self.kind, FactKind):
            raise ValueError("invalid fact kind")
        for value in (
            self.artifact_ref,
            self.source_episode_ref,
            *self.causal_parents,
            *self.policy_refs,
            *self.model_refs,
        ):
            _valid_digest(value)
        for values in (self.causal_parents, self.policy_refs, self.model_refs):
            if type(values) is not tuple or len(values) != len(set(values)):
                raise ValueError("fact references must be unique immutable tuples")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "artifact_ref": self.artifact_ref,
            "causal_parents": self.causal_parents,
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "model_refs": self.model_refs,
            "policy_refs": self.policy_refs,
            "schema_version": "causal-fact.v1",
            "source_episode_ref": self.source_episode_ref,
        }
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class AutobiographicalEpisode:
    memory_id: str
    tenant_id: str
    person_id: str
    purpose: str
    consent_ref: str
    privacy_policy_ref: str
    retention_until: str
    prior_lineage_state: str
    episode_ref: str
    interpretation_ref: str
    committed_belief_refs: tuple[str, ...]
    policy_ref: str
    expected_effect_ref: str
    intent_ref: str
    receipt_ref: str
    observation_ref: str
    reafference_ref: str
    correction_refs: tuple[str, ...]
    next_lineage_state: str
    facts: tuple[CausalFact, ...]
    revision: int = 1
    supersedes: str | None = None
    state: MemoryState = MemoryState.ACTIVE
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.memory_id, "memory id"),
            (self.tenant_id, "tenant id"),
            (self.person_id, "person id"),
            (self.purpose, "purpose"),
            (self.retention_until, "retention"),
        ):
            _text(value, name)
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be positive")
        try:
            retention = datetime.fromisoformat(
                self.retention_until.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("invalid retention") from exc
        if not self.retention_until.endswith(
            "Z"
        ) or retention.utcoffset() != timezone.utc.utcoffset(retention):
            raise ValueError("retention must be canonical UTC")
        if not isinstance(self.state, MemoryState):
            raise ValueError("invalid memory state")
        refs: Iterable[str] = (
            self.consent_ref,
            self.privacy_policy_ref,
            self.prior_lineage_state,
            self.episode_ref,
            self.interpretation_ref,
            *self.committed_belief_refs,
            self.policy_ref,
            self.expected_effect_ref,
            self.intent_ref,
            self.receipt_ref,
            self.observation_ref,
            self.reafference_ref,
            *self.correction_refs,
            self.next_lineage_state,
        )
        for value in refs:
            _valid_digest(value)
        if self.supersedes is not None:
            _valid_digest(self.supersedes)
        if type(self.facts) is not tuple or not self.facts:
            raise ValueError("autobiography requires typed facts")
        if not all(isinstance(fact, CausalFact) for fact in self.facts):
            raise ValueError("autobiography facts must be typed")
        kinds = tuple(fact.kind for fact in self.facts)
        if kinds != (
            FactKind.OBSERVATION,
            FactKind.INFERENCE,
            FactKind.PREDICTION,
            FactKind.OUTCOME,
        ):
            raise ValueError(
                "facts must be one ordered observation, inference, prediction, and outcome"
            )
        if len({fact.fact_id for fact in self.facts}) != len(self.facts) or len(
            {fact.digest for fact in self.facts}
        ) != len(self.facts):
            raise ValueError("autobiography contains duplicate facts")
        preceding: set[str] = set()
        for fact in self.facts:
            if fact.source_episode_ref != self.episode_ref:
                raise ValueError("fact belongs to a different episode")
            if any(parent not in preceding for parent in fact.causal_parents):
                raise ValueError(
                    "fact causal parent is absent, cyclic, or out of order"
                )
            preceding.add(fact.digest)
        observation, inference, prediction, outcome = self.facts
        if observation.causal_parents:
            raise ValueError("observation cannot infer its own causal parent")
        if inference.causal_parents != (observation.digest,):
            raise ValueError("inference must derive from the observation")
        if prediction.causal_parents != (inference.digest,):
            raise ValueError("prediction must derive from the interpretation")
        if prediction.digest not in outcome.causal_parents:
            raise ValueError("outcome must evaluate the prediction")
        if (
            self.policy_ref not in inference.policy_refs
            or self.policy_ref not in prediction.policy_refs
        ):
            raise ValueError(
                "interpretation and prediction must bind the governing policy"
            )
        if not inference.model_refs:
            raise ValueError("interpretation must bind a causal model")
        if (
            observation.artifact_ref != self.observation_ref
            or inference.artifact_ref != self.interpretation_ref
            or prediction.artifact_ref != self.expected_effect_ref
            or outcome.artifact_ref != self.reafference_ref
        ):
            raise ValueError("typed facts do not bind the causal episode references")
        if self.revision == 1 and (self.supersedes or self.correction_refs):
            raise ValueError("genesis autobiography cannot claim a correction")
        if self.revision > 1 and (self.supersedes is None or not self.correction_refs):
            raise ValueError(
                "corrected autobiography requires prior and evidence references"
            )
        if self.prior_lineage_state == self.next_lineage_state:
            raise ValueError("autobiography must record a lineage transition")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            "committed_belief_refs": self.committed_belief_refs,
            "consent_ref": self.consent_ref,
            "correction_refs": self.correction_refs,
            "episode_ref": self.episode_ref,
            "expected_effect_ref": self.expected_effect_ref,
            "facts": tuple(f.to_json() for f in self.facts),
            "intent_ref": self.intent_ref,
            "interpretation_ref": self.interpretation_ref,
            "memory_id": self.memory_id,
            "next_lineage_state": self.next_lineage_state,
            "observation_ref": self.observation_ref,
            "person_id": self.person_id,
            "policy_ref": self.policy_ref,
            "privacy_policy_ref": self.privacy_policy_ref,
            "prior_lineage_state": self.prior_lineage_state,
            "purpose": self.purpose,
            "reafference_ref": self.reafference_ref,
            "receipt_ref": self.receipt_ref,
            "retention_until": self.retention_until,
            "revision": self.revision,
            "schema_version": "autobiographical-episode.v1",
            "state": self.state.value,
            "supersedes": self.supersedes,
            "tenant_id": self.tenant_id,
        }
        if include_digest:
            value["digest"] = self.digest
        return value


class AutobiographicalMemoryStore:
    """DB-first revision ledger with a transactional, idempotent audit outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        chain_authority: BiographyChainAuthority,
        failpoint: AutobiographyFailpoint | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if chain_authority is None:
            raise AutobiographyError("durable causal-chain authority is required")
        self.path = str(path)
        self._chain_authority = chain_authority
        self._failpoint = failpoint or NoopAutobiographyFailpoint()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise AutobiographyError("unsafe autobiography database path")
        with self._connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS autobiography(
              digest TEXT PRIMARY KEY, memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
              tenant_id TEXT NOT NULL, person_id TEXT NOT NULL, purpose TEXT NOT NULL,
              prior_lineage TEXT NOT NULL, next_lineage TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('active','superseded','tombstoned')),
              supersedes TEXT, document TEXT,
              UNIQUE(tenant_id,person_id,memory_id,revision));
            CREATE UNIQUE INDEX IF NOT EXISTS autobiography_active
              ON autobiography(tenant_id,person_id,memory_id) WHERE state='active';
            CREATE TABLE IF NOT EXISTS autobiography_outbox(
              event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, payload TEXT NOT NULL,
              delivered INTEGER NOT NULL DEFAULT 0 CHECK(delivered IN (0,1)));
            """)
        os.chmod(p, 0o600)
        self._verify_schema()
        self.verify()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _verify_schema(self) -> None:
        """Reject legacy/partially-created schemas before accepting authority."""
        with self._connect() as db:
            columns = {
                str(row["name"]): (str(row["type"]), int(row["notnull"]))
                for row in db.execute("PRAGMA table_info(autobiography)").fetchall()
            }
            active_index = tuple(
                str(row["name"])
                for row in db.execute(
                    "PRAGMA index_info(autobiography_active)"
                ).fetchall()
            )
        expected_columns = {
            "digest",
            "memory_id",
            "revision",
            "tenant_id",
            "person_id",
            "purpose",
            "prior_lineage",
            "next_lineage",
            "state",
            "supersedes",
            "document",
        }
        if (
            set(columns) != expected_columns
            or columns.get("document") != ("TEXT", 0)
            or active_index != ("tenant_id", "person_id", "memory_id")
        ):
            raise AutobiographyError(
                "incompatible autobiography schema; governed migration required"
            )

    def commit(self, principal: Principal, episode: AutobiographicalEpisode) -> str:
        if not principal.is_kernel:
            raise AutobiographyError(
                "only the cognitive kernel may commit autobiography"
            )
        if (
            episode.revision != 1
            or episode.supersedes is not None
            or episode.state is not MemoryState.ACTIVE
        ):
            raise AutobiographyError("genesis memory has invalid revision state")
        self._write(episode, "autobiography.committed", expected_digest=None)
        return episode.digest

    def correct(
        self,
        principal: Principal,
        prior_digest: str,
        replacement: AutobiographicalEpisode,
    ) -> str:
        if not principal.is_kernel:
            raise AutobiographyError(
                "only the cognitive kernel may correct autobiography"
            )
        prior = self.load_digest(
            prior_digest,
            tenant_id=replacement.tenant_id,
            person_id=replacement.person_id,
        )
        if (
            replacement.memory_id,
            replacement.tenant_id,
            replacement.person_id,
            replacement.purpose,
            replacement.consent_ref,
            replacement.privacy_policy_ref,
        ) != (
            prior.memory_id,
            prior.tenant_id,
            prior.person_id,
            prior.purpose,
            prior.consent_ref,
            prior.privacy_policy_ref,
        ):
            raise AutobiographyError("correction cannot change privacy scope")
        if (
            replacement.revision != prior.revision + 1
            or replacement.supersedes != prior.digest
        ):
            raise AutobiographyConflict(
                "correction does not extend the active revision"
            )
        self._write(
            replacement, "autobiography.corrected", expected_digest=prior.digest
        )
        return replacement.digest

    def tombstone(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        person_id: str,
        memory_id: str,
        deletion_authorization_ref: str,
    ) -> str:
        if not principal.is_kernel:
            raise AutobiographyError(
                "only the cognitive kernel may tombstone autobiography"
            )
        prior = self.load(memory_id, tenant_id=tenant_id, person_id=person_id)
        _valid_digest(deletion_authorization_ref)
        revision = prior.revision + 1
        scope_digest = _digest(
            {"person_id": person_id, "purpose": prior.purpose, "tenant_id": tenant_id}
        )
        tombstone_digest = _digest(
            {
                "deletion_authorization_ref": deletion_authorization_ref,
                "memory_id": memory_id,
                "prior_digest": prior.digest,
                "revision": revision,
                "schema_version": "autobiography-tombstone.v1",
                "scope_digest": scope_digest,
            }
        )
        payload = canonical_json(
            {
                "record_digest": tombstone_digest,
                "revision": revision,
                "scope_digest": scope_digest,
                "state": MemoryState.TOMBSTONED.value,
            }
        ).decode()
        event_id = _digest(
            {"event_type": "autobiography.tombstoned", "digest": tombstone_digest}
        )
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                changed = db.execute(
                    "UPDATE autobiography SET state='superseded',document=NULL "
                    "WHERE tenant_id=? AND person_id=? AND memory_id=? AND document IS NOT NULL",
                    (tenant_id, person_id, memory_id),
                ).rowcount
                if changed < 1:
                    raise AutobiographyConflict("autobiography was already deleted")
                db.execute(
                    "INSERT INTO autobiography VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                    (
                        tombstone_digest,
                        memory_id,
                        revision,
                        tenant_id,
                        person_id,
                        prior.purpose,
                        prior.prior_lineage_state,
                        prior.next_lineage_state,
                        MemoryState.TOMBSTONED.value,
                        prior.digest,
                    ),
                )
                db.execute(
                    "INSERT INTO autobiography_outbox(event_id,event_type,payload) VALUES(?,?,?)",
                    (event_id, "autobiography.tombstoned", payload),
                )
                self._failpoint.hit("before_commit")
                db.commit()
                self._failpoint.hit("after_commit")
            except BaseException as exc:
                db.rollback()
                if isinstance(exc, (AutobiographyConflict, AutobiographyCrash)):
                    raise
                if isinstance(exc, sqlite3.IntegrityError):
                    raise AutobiographyConflict("conflicting deletion") from exc
                raise
        return tombstone_digest

    def _write(
        self,
        episode: AutobiographicalEpisode,
        event_type: str,
        expected_digest: str | None,
    ) -> None:
        self._chain_authority.verify(episode)
        document = canonical_json(episode.to_json()).decode()
        scope_digest = _digest(
            {
                "tenant_id": episode.tenant_id,
                "person_id": episode.person_id,
                "purpose": episode.purpose,
            }
        )
        payload = canonical_json(
            {
                "record_digest": episode.digest,
                "revision": episode.revision,
                "scope_digest": scope_digest,
                "state": episode.state.value,
            }
        ).decode()
        event_id = _digest({"event_type": event_type, "digest": episode.digest})
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                if expected_digest is not None:
                    changed = db.execute(
                        "UPDATE autobiography SET state='superseded' WHERE digest=? AND state='active'",
                        (expected_digest,),
                    ).rowcount
                    if changed != 1:
                        raise AutobiographyConflict("stale autobiography correction")
                db.execute(
                    "INSERT INTO autobiography VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        episode.digest,
                        episode.memory_id,
                        episode.revision,
                        episode.tenant_id,
                        episode.person_id,
                        episode.purpose,
                        episode.prior_lineage_state,
                        episode.next_lineage_state,
                        episode.state.value,
                        episode.supersedes,
                        document,
                    ),
                )
                db.execute(
                    "INSERT INTO autobiography_outbox(event_id,event_type,payload) VALUES(?,?,?)",
                    (event_id, event_type, payload),
                )
                self._failpoint.hit("before_commit")
                db.commit()
                self._failpoint.hit("after_commit")
            except BaseException as exc:
                db.rollback()
                if isinstance(exc, (AutobiographyConflict, AutobiographyCrash)):
                    raise
                if isinstance(exc, sqlite3.IntegrityError):
                    raise AutobiographyConflict(
                        "duplicate or conflicting autobiography"
                    ) from exc
                raise

    def load(
        self, memory_id: str, *, tenant_id: str, person_id: str
    ) -> AutobiographicalEpisode:
        with self._connect() as db:
            row = db.execute(
                "SELECT document FROM autobiography WHERE memory_id=? AND tenant_id=? AND person_id=? AND state='active'",
                (memory_id, tenant_id, person_id),
            ).fetchone()
        if row is None:
            raise AutobiographyError("active autobiography is absent")
        episode = _decode(row["document"])
        self._chain_authority.verify(episode)
        if _retention(episode.retention_until) <= self._now():
            raise AutobiographyError("autobiography retention has expired")
        return episode

    def load_digest(
        self, digest: str, *, tenant_id: str, person_id: str
    ) -> AutobiographicalEpisode:
        _valid_digest(digest)
        with self._connect() as db:
            row = db.execute(
                "SELECT document FROM autobiography WHERE digest=? AND tenant_id=? AND person_id=?",
                (digest, tenant_id, person_id),
            ).fetchone()
        if row is None or row["document"] is None:
            raise AutobiographyError("autobiography revision is absent")
        episode = _decode(row["document"])
        self._chain_authority.verify(episode)
        if _retention(episode.retention_until) <= self._now():
            raise AutobiographyError("autobiography retention has expired")
        return episode

    def retrieve_causal(
        self,
        *,
        tenant_id: str,
        person_id: str,
        purpose: str,
        policy_refs: tuple[str, ...] = (),
        model_refs: tuple[str, ...] = (),
        causal_refs: tuple[str, ...] = (),
        limit: int = 20,
    ) -> tuple[AutobiographicalEpisode, ...]:
        """Retrieve active records sharing explicit causal/model/policy edges.

        Text and embeddings are deliberately absent from this authority path.
        """
        if (
            not (policy_refs or model_refs or causal_refs)
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("causal retrieval requires bounded causal selectors")
        selectors = set((*policy_refs, *model_refs, *causal_refs))
        for value in selectors:
            _valid_digest(value)
        with self._connect() as db:
            rows = db.execute(
                "SELECT document FROM autobiography WHERE tenant_id=? AND person_id=? AND purpose=? AND state='active' ORDER BY revision DESC",
                (tenant_id, person_id, purpose),
            ).fetchall()
        ranked: list[tuple[int, AutobiographicalEpisode]] = []
        for row in rows:
            if row["document"] is None:
                continue
            episode = _decode(row["document"])
            self._chain_authority.verify(episode)
            if _retention(episode.retention_until) <= self._now():
                continue
            edges = {
                episode.policy_ref,
                episode.interpretation_ref,
                episode.reafference_ref,
                *episode.correction_refs,
            }
            for fact in episode.facts:
                edges.update(
                    (
                        fact.digest,
                        *fact.causal_parents,
                        *fact.policy_refs,
                        *fact.model_refs,
                    )
                )
            score = len(selectors & edges)
            if score:
                ranked.append((score, episode))
        ranked.sort(key=lambda item: (-item[0], item[1].digest))
        return tuple(episode for _, episode in ranked[:limit])

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise AutobiographyError("autobiography clock must return aware time")
        return value.astimezone(timezone.utc)

    def drain_outbox(
        self, append: Callable[[str, Mapping[str, object]], object]
    ) -> int:
        delivered = 0
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM autobiography_outbox WHERE delivered=0 ORDER BY rowid"
            ).fetchall()
        for row in rows:
            payload = _strict_json(row["payload"])
            try:
                append(
                    row["event_type"],
                    {**payload, "transaction_id": row["event_id"]},
                )
            except RuntimeError as exc:
                if "duplicate transaction" not in str(exc):
                    raise AutobiographyError(
                        "autobiography outbox delivery failed"
                    ) from exc
            with self._connect() as db:
                changed = db.execute(
                    "UPDATE autobiography_outbox SET delivered=1 WHERE event_id=? AND delivered=0",
                    (row["event_id"],),
                ).rowcount
            delivered += changed
        return delivered

    def verify(self) -> None:
        with self._connect() as db:
            check = db.execute("PRAGMA quick_check").fetchone()
            rows = db.execute(
                "SELECT * FROM autobiography ORDER BY tenant_id,person_id,memory_id,revision"
            ).fetchall()
            outbox = db.execute("SELECT * FROM autobiography_outbox").fetchall()
        if check is None or check[0] != "ok":
            raise AutobiographyError("autobiography database integrity failed")
        prior_by_id: dict[tuple[str, str, str], sqlite3.Row] = {}
        record_by_digest: dict[str, sqlite3.Row] = {}
        for row in rows:
            key = (row["tenant_id"], row["person_id"], row["memory_id"])
            prior = prior_by_id.get(key)
            if prior is None and (
                row["revision"] != 1 or row["supersedes"] is not None
            ):
                raise AutobiographyError("autobiography revision chain has no genesis")
            if prior is not None and (
                row["revision"] != prior["revision"] + 1
                or row["supersedes"] != prior["digest"]
                or row["prior_lineage"] != prior["prior_lineage"]
                or row["next_lineage"] != prior["next_lineage"]
            ):
                raise AutobiographyError("autobiography revision chain is invalid")
            if row["document"] is not None:
                episode = _decode(row["document"])
                self._chain_authority.verify(episode)
                if episode.digest != row["digest"] or (
                    row["state"] != MemoryState.SUPERSEDED.value
                    and episode.state.value != row["state"]
                ):
                    raise AutobiographyError("persisted autobiography digest mismatch")
                if (
                    episode.memory_id != row["memory_id"]
                    or episode.tenant_id != row["tenant_id"]
                    or episode.person_id != row["person_id"]
                    or episode.purpose != row["purpose"]
                    or episode.revision != row["revision"]
                    or episode.supersedes != row["supersedes"]
                ):
                    raise AutobiographyError("persisted autobiography index mismatch")
            elif row["state"] not in {
                MemoryState.SUPERSEDED.value,
                MemoryState.TOMBSTONED.value,
            }:
                raise AutobiographyError("live autobiography document is missing")
            prior_by_id[key] = row
            record_by_digest[row["digest"]] = row
        for final in prior_by_id.values():
            if final["state"] == MemoryState.ACTIVE.value:
                if final["document"] is None:
                    raise AutobiographyError("active autobiography document is missing")
            elif (
                final["state"] != MemoryState.TOMBSTONED.value
                or final["document"] is not None
            ):
                raise AutobiographyError(
                    "autobiography chain has no active head or tombstone"
                )
        for row in outbox:
            payload = _strict_json(row["payload"])
            if set(payload) != {"record_digest", "revision", "scope_digest", "state"}:
                raise AutobiographyError("invalid autobiography outbox schema")
            record = record_by_digest.get(str(payload["record_digest"]))
            if (
                record is None
                or row["event_id"]
                != _digest(
                    {
                        "event_type": row["event_type"],
                        "digest": payload["record_digest"],
                    }
                )
                or payload["revision"] != record["revision"]
                or not (
                    payload["state"] == record["state"]
                    or (
                        payload["state"] == MemoryState.ACTIVE.value
                        and record["state"] == MemoryState.SUPERSEDED.value
                    )
                )
                or payload["scope_digest"]
                != _digest(
                    {
                        "person_id": record["person_id"],
                        "purpose": record["purpose"],
                        "tenant_id": record["tenant_id"],
                    }
                )
            ):
                raise AutobiographyError(
                    "autobiography outbox does not match its record"
                )


def _decode(document: str) -> AutobiographicalEpisode:
    try:
        raw = _strict_json(document)
        if raw.pop("schema_version") != "autobiographical-episode.v1":
            raise ValueError
        digest = str(raw.pop("digest"))
        raw_facts = raw.pop("facts")
        if not isinstance(raw_facts, list):
            raise ValueError
        facts: list[CausalFact] = []
        for untyped_value in raw_facts:
            if not isinstance(untyped_value, dict):
                raise ValueError
            value = dict(untyped_value)
            fact_digest = value.pop("digest")
            if value.pop("schema_version") != "causal-fact.v1":
                raise ValueError
            fact = CausalFact(
                fact_id=str(value.pop("fact_id")),
                kind=FactKind(_string(value.pop("kind"))),
                artifact_ref=str(value.pop("artifact_ref")),
                source_episode_ref=str(value.pop("source_episode_ref")),
                causal_parents=_string_tuple(value.pop("causal_parents")),
                policy_refs=_string_tuple(value.pop("policy_refs")),
                model_refs=_string_tuple(value.pop("model_refs")),
            )
            if value or fact.digest != fact_digest:
                raise ValueError
            facts.append(fact)
        episode = AutobiographicalEpisode(
            memory_id=str(raw.pop("memory_id")),
            tenant_id=str(raw.pop("tenant_id")),
            person_id=str(raw.pop("person_id")),
            purpose=str(raw.pop("purpose")),
            consent_ref=str(raw.pop("consent_ref")),
            privacy_policy_ref=str(raw.pop("privacy_policy_ref")),
            retention_until=str(raw.pop("retention_until")),
            prior_lineage_state=str(raw.pop("prior_lineage_state")),
            episode_ref=str(raw.pop("episode_ref")),
            interpretation_ref=str(raw.pop("interpretation_ref")),
            committed_belief_refs=_string_tuple(raw.pop("committed_belief_refs")),
            policy_ref=str(raw.pop("policy_ref")),
            expected_effect_ref=str(raw.pop("expected_effect_ref")),
            intent_ref=str(raw.pop("intent_ref")),
            receipt_ref=str(raw.pop("receipt_ref")),
            observation_ref=str(raw.pop("observation_ref")),
            reafference_ref=str(raw.pop("reafference_ref")),
            correction_refs=_string_tuple(raw.pop("correction_refs")),
            next_lineage_state=str(raw.pop("next_lineage_state")),
            facts=tuple(facts),
            revision=_integer(raw.pop("revision")),
            supersedes=_optional_string(raw.pop("supersedes")),
            state=MemoryState(_string(raw.pop("state"))),
        )
        if raw or episode.digest != digest:
            raise ValueError
        return episode
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AutobiographyError("invalid persisted autobiography document") from exc


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected string array")
    return tuple(value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected integer")
    return cast(int, value)


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("expected optional string")
    return cast(str | None, value)
