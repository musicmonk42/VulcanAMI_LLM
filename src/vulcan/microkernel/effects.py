"""Durable, fail-closed execution of narrowly scoped reversible effects.

This module is the sole effect authority.  Effect payloads are typed data for a
code-owned sandbox; provider/model text is deliberately not accepted here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol
from uuid import uuid4

from vulcan.constitution.primitives import Digest, canonical_json

from .principals import Principal

ZERO_DIGEST = "0" * 64
ALLOWED_OPERATIONS = frozenset({"put", "delete"})


class EffectError(RuntimeError):
    """Base fail-closed effect protocol error."""


class EffectRejected(EffectError):
    """The requested operation is not authorized by the capability."""


class EffectConflict(EffectError):
    """A nonce, idempotency key, or durable state was replayed."""


class AuthorityHeadValidator(Protocol):
    """Validate exact episode and lineage heads at authorization time."""

    def __call__(
        self,
        episode_id: str,
        episode_digest: str,
        lineage_id: str,
        branch_id: str,
        lineage_head_digest: str,
    ) -> None: ...


class EffectCrash(BaseException):
    """Testable process-crash signal; never converted into a retry."""


class EffectOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    COMPENSATED = "compensated"
    AMBIGUOUS = "ambiguous"


def _digest(value: object) -> str:
    return Digest.of_bytes(canonical_json(value)).hex


def _check_digest(value: str) -> None:
    Digest.from_legacy_hex(value)


def _check_text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"invalid {name}")


def _strict_document(document: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EffectError("duplicate persisted JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(document, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EffectError("invalid persisted effect document") from exc
    if not isinstance(value, dict) or canonical_json(value).decode() != document:
        raise EffectError("effect document is not canonical JSON")
    return value


@dataclass(frozen=True)
class PolicyProposal:
    proposal_id: str
    episode_id: str
    lineage_id: str
    branch_id: str
    resource: str
    operation: str
    value_digest: str
    authority: str = "untrusted_proposal"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "episode_id",
            "lineage_id",
            "branch_id",
            "resource",
            "operation",
        ):
            _check_text(getattr(self, name), name)
        _check_digest(self.value_digest)
        if self.authority != "untrusted_proposal":
            raise ValueError("policy proposals cannot carry authority")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            k: getattr(self, k)
            for k in (
                "authority",
                "branch_id",
                "episode_id",
                "lineage_id",
                "operation",
                "proposal_id",
                "resource",
                "value_digest",
            )
        }
        value["schema_version"] = "policy-proposal.v1"
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class AuthorizedPolicy:
    policy_id: str
    proposal_digest: str
    principal_digest: str
    release_digest: str
    budget: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_text(self.policy_id, "policy_id")
        for value in (self.proposal_digest, self.principal_digest, self.release_digest):
            _check_digest(value)
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("policy budget must be positive")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            "budget": self.budget,
            "policy_id": self.policy_id,
            "principal_digest": self.principal_digest,
            "proposal_digest": self.proposal_digest,
            "release_digest": self.release_digest,
            "schema_version": "authorized-policy.v1",
        }
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class ExpectedEffect:
    resource: str
    operation: str
    before_digest: str
    after_digest: str
    reversible: bool
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_text(self.resource, "resource")
        _check_text(self.operation, "operation")
        _check_digest(self.before_digest)
        _check_digest(self.after_digest)
        if type(self.reversible) is not bool:
            raise ValueError("reversible must be Boolean")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            "after_digest": self.after_digest,
            "before_digest": self.before_digest,
            "operation": self.operation,
            "resource": self.resource,
            "reversible": self.reversible,
            "schema_version": "expected-effect.v1",
        }
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class EffectIntent:
    intent_id: str
    principal_digest: str
    release_digest: str
    lineage_id: str
    branch_id: str
    lineage_head_digest: str
    episode_id: str
    episode_digest: str
    policy_digest: str
    expected_effect_digest: str
    resource: str
    operation: str
    value: str | None
    budget: int
    idempotency_key: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "intent_id",
            "lineage_id",
            "branch_id",
            "episode_id",
            "resource",
            "operation",
            "idempotency_key",
        ):
            _check_text(getattr(self, name), name)
        for value in (
            self.principal_digest,
            self.release_digest,
            self.lineage_head_digest,
            self.episode_digest,
            self.policy_digest,
            self.expected_effect_digest,
        ):
            _check_digest(value)
        if self.operation not in ALLOWED_OPERATIONS:
            raise ValueError("operation is not sandbox allowlisted")
        if self.value is not None and (
            not isinstance(self.value, str) or len(self.value) > 4096
        ):
            raise ValueError("sandbox value is invalid")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("intent budget must be positive")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            k: getattr(self, k)
            for k in (
                "branch_id",
                "budget",
                "episode_digest",
                "episode_id",
                "expected_effect_digest",
                "idempotency_key",
                "intent_id",
                "lineage_head_digest",
                "lineage_id",
                "operation",
                "policy_digest",
                "principal_digest",
                "release_digest",
                "resource",
                "value",
            )
        }
        value["schema_version"] = "effect-intent.v1"
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class CapabilityToken:
    token_id: str
    principal_digest: str
    release_digest: str
    lineage_id: str
    branch_id: str
    episode_id: str
    policy_digest: str
    effect_digest: str
    resource: str
    operation: str
    expires_at: str
    nonce: str
    budget: int
    idempotency_key: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "token_id",
            "lineage_id",
            "branch_id",
            "episode_id",
            "resource",
            "operation",
            "expires_at",
            "nonce",
            "idempotency_key",
        ):
            _check_text(getattr(self, name), name)
        for value in (
            self.principal_digest,
            self.release_digest,
            self.policy_digest,
            self.effect_digest,
        ):
            _check_digest(value)
        if self.operation not in ALLOWED_OPERATIONS:
            raise ValueError("operation is not allowlisted")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("capability budget must be positive")
        parsed = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
            parsed
        ):
            raise ValueError("expiry must be UTC")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            k: getattr(self, k)
            for k in (
                "branch_id",
                "budget",
                "effect_digest",
                "episode_id",
                "expires_at",
                "idempotency_key",
                "lineage_id",
                "nonce",
                "operation",
                "policy_digest",
                "principal_digest",
                "release_digest",
                "resource",
                "token_id",
            )
        }
        value["schema_version"] = "capability-token.v1"
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class EffectAttempt:
    attempt_id: str
    intent_digest: str
    capability_digest: str
    started_at: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_text(self.attempt_id, "attempt_id")
        _check_text(self.started_at, "started_at")
        _check_digest(self.intent_digest)
        _check_digest(self.capability_digest)
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            "attempt_id": self.attempt_id,
            "capability_digest": self.capability_digest,
            "intent_digest": self.intent_digest,
            "schema_version": "effect-attempt.v1",
            "started_at": self.started_at,
        }
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class EffectReceipt:
    receipt_id: str
    attempt_digest: str
    intent_digest: str
    outcome: EffectOutcome
    observed_digest: str
    detail_code: str
    compensated_by: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_text(self.receipt_id, "receipt_id")
        _check_text(self.detail_code, "detail_code")
        _check_digest(self.attempt_digest)
        _check_digest(self.intent_digest)
        _check_digest(self.observed_digest)
        if self.compensated_by is not None:
            _check_digest(self.compensated_by)
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value = {
            "attempt_digest": self.attempt_digest,
            "compensated_by": self.compensated_by,
            "detail_code": self.detail_code,
            "intent_digest": self.intent_digest,
            "observed_digest": self.observed_digest,
            "outcome": self.outcome.value,
            "receipt_id": self.receipt_id,
            "schema_version": "effect-receipt.v1",
        }
        if include_digest:
            value["digest"] = self.digest
        return value


class ReversibleSandbox:
    """Deterministic in-memory key/value target; no host, shell, or network access."""

    def __init__(self, *, idempotent: bool = True) -> None:
        self.idempotent = idempotent
        self.values: dict[str, str] = {}
        self.applied: dict[str, str] = {}

    def execute(self, intent: EffectIntent) -> tuple[str | None, str]:
        if (
            intent.resource.startswith("/")
            or ".." in intent.resource
            or intent.operation not in ALLOWED_OPERATIONS
        ):
            raise EffectRejected("resource is outside the sandbox")
        if self.idempotent and intent.idempotency_key in self.applied:
            return None, self.applied[intent.idempotency_key]
        before = self.values.get(intent.resource)
        if intent.operation == "put":
            if intent.value is None:
                raise EffectRejected("put requires a value")
            self.values[intent.resource] = intent.value
        else:
            self.values.pop(intent.resource, None)
        observed = _digest(
            {"resource": intent.resource, "value": self.values.get(intent.resource)}
        )
        if self.idempotent:
            self.applied[intent.idempotency_key] = observed
        return before, observed

    def compensate(self, intent: EffectIntent, before: str | None) -> str:
        if before is None:
            self.values.pop(intent.resource, None)
        else:
            self.values[intent.resource] = before
        return _digest(
            {"resource": intent.resource, "value": self.values.get(intent.resource)}
        )


class EffectStore:
    """SQLite intent/attempt/receipt log with a transactional audit outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        outbox_sink: Callable[[str, Mapping[str, object]], None] | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.path, self.outbox_sink, self.failpoint = str(path), outbox_sink, failpoint
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise EffectError("unsafe effect database path")
        with self._connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS effect_intents(digest TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, nonce TEXT UNIQUE NOT NULL, episode_id TEXT NOT NULL, episode_digest TEXT NOT NULL, lineage_id TEXT NOT NULL, branch_id TEXT NOT NULL, lineage_head_digest TEXT NOT NULL, document TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS effect_capabilities(digest TEXT PRIMARY KEY, nonce TEXT UNIQUE NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, document TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS effect_authorizations(capability_digest TEXT PRIMARY KEY, proposal_document TEXT NOT NULL, policy_document TEXT NOT NULL, expected_document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS effect_attempts(digest TEXT PRIMARY KEY, intent_digest TEXT UNIQUE NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS effect_receipts(digest TEXT PRIMARY KEY, intent_digest TEXT UNIQUE NOT NULL, document TEXT NOT NULL, outcome TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS effect_outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, artifact_digest TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
            """)
        os.chmod(p, 0o600)
        self.verify()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    def _event(
        self,
        db: sqlite3.Connection,
        topic: str,
        digest: str,
        payload: Mapping[str, object],
    ) -> None:
        db.execute(
            "INSERT INTO effect_outbox(topic,artifact_digest,payload) VALUES(?,?,?)",
            (topic, digest, canonical_json(payload).decode()),
        )

    def commit_intent(self, intent: EffectIntent, token: CapabilityToken) -> None:
        document = canonical_json(intent.to_json()).decode()
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                capability = db.execute(
                    "SELECT state,document FROM effect_capabilities WHERE digest=?",
                    (token.digest,),
                ).fetchone()
                if capability is None or capability["state"] != "issued":
                    raise EffectRejected(
                        "capability was not issued or was already consumed"
                    )
                if capability["document"] != canonical_json(token.to_json()).decode():
                    raise EffectRejected(
                        "capability document does not match issued token"
                    )
                db.execute(
                    "INSERT INTO effect_intents VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        intent.digest,
                        intent.idempotency_key,
                        token.nonce,
                        intent.episode_id,
                        intent.episode_digest,
                        intent.lineage_id,
                        intent.branch_id,
                        intent.lineage_head_digest,
                        document,
                        "authorized",
                    ),
                )
                changed = db.execute(
                    "UPDATE effect_capabilities SET state='consumed' WHERE digest=? AND state='issued'",
                    (token.digest,),
                ).rowcount
                if changed != 1:
                    raise EffectConflict("capability was concurrently consumed")
                self._event(
                    db,
                    "effect.intent.committed",
                    intent.digest,
                    {
                        "branch_id": intent.branch_id,
                        "episode_id": intent.episode_id,
                        "intent_digest": intent.digest,
                        "lineage_id": intent.lineage_id,
                    },
                )
                if self.failpoint:
                    self.failpoint("after_intent_before_commit")
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise EffectConflict(
                    "capability nonce or idempotency key was replayed"
                ) from exc
            except BaseException:
                db.rollback()
                raise
        self.deliver_outbox()

    def issue_capability(
        self,
        token: CapabilityToken,
        proposal: PolicyProposal,
        policy: AuthorizedPolicy,
        expected: ExpectedEffect,
    ) -> None:
        """Persist a kernel-authorized capability before it can be consumed."""
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO effect_capabilities VALUES(?,?,?,?, 'issued')",
                    (
                        token.digest,
                        token.nonce,
                        token.idempotency_key,
                        canonical_json(token.to_json()).decode(),
                    ),
                )
                db.execute(
                    "INSERT INTO effect_authorizations VALUES(?,?,?,?)",
                    (
                        token.digest,
                        canonical_json(proposal.to_json()).decode(),
                        canonical_json(policy.to_json()).decode(),
                        canonical_json(expected.to_json()).decode(),
                    ),
                )
                self._event(
                    db,
                    "effect.capability.issued",
                    token.digest,
                    {
                        "capability_digest": token.digest,
                        "episode_id": token.episode_id,
                        "lineage_id": token.lineage_id,
                    },
                )
                if self.failpoint:
                    self.failpoint("after_capability_before_commit")
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise EffectConflict(
                    "capability nonce or idempotency key was replayed"
                ) from exc
            except BaseException:
                db.rollback()
                raise
        self.deliver_outbox()

    def expected_effect(self, capability_digest: str) -> ExpectedEffect:
        with self._connect() as db:
            row = db.execute(
                "SELECT expected_document FROM effect_authorizations WHERE capability_digest=?",
                (capability_digest,),
            ).fetchone()
        if row is None:
            raise EffectRejected("capability authorization evidence is missing")
        raw = _strict_document(row["expected_document"])
        try:
            expected = ExpectedEffect(
                resource=str(raw["resource"]),
                operation=str(raw["operation"]),
                before_digest=str(raw["before_digest"]),
                after_digest=str(raw["after_digest"]),
                reversible=raw["reversible"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EffectError("invalid expected-effect authorization") from exc
        if raw.get("digest") != expected.digest:
            raise EffectError("expected-effect authorization digest mismatch")
        return expected

    def start_attempt(self, attempt: EffectAttempt) -> None:
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT state FROM effect_intents WHERE digest=?",
                    (attempt.intent_digest,),
                ).fetchone()
                if row is None or row["state"] != "authorized":
                    raise EffectConflict("intent is absent or already attempted")
                db.execute(
                    "INSERT INTO effect_attempts VALUES(?,?,?)",
                    (
                        attempt.digest,
                        attempt.intent_digest,
                        canonical_json(attempt.to_json()).decode(),
                    ),
                )
                db.execute(
                    "UPDATE effect_intents SET state='attempting' WHERE digest=?",
                    (attempt.intent_digest,),
                )
                self._event(
                    db, "effect.attempt.started", attempt.digest, attempt.to_json()
                )
                if self.failpoint:
                    self.failpoint("after_attempt_before_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        self.deliver_outbox()

    def finish(self, receipt: EffectReceipt) -> None:
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT state FROM effect_intents WHERE digest=?",
                    (receipt.intent_digest,),
                ).fetchone()
                if row is None or row["state"] != "attempting":
                    raise EffectConflict("receipt has no active attempt")
                db.execute(
                    "INSERT INTO effect_receipts VALUES(?,?,?,?)",
                    (
                        receipt.digest,
                        receipt.intent_digest,
                        canonical_json(receipt.to_json()).decode(),
                        receipt.outcome.value,
                    ),
                )
                db.execute(
                    "UPDATE effect_intents SET state=? WHERE digest=?",
                    (receipt.outcome.value, receipt.intent_digest),
                )
                self._event(
                    db, "effect.receipt.committed", receipt.digest, receipt.to_json()
                )
                if self.failpoint:
                    self.failpoint("after_receipt_before_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        self.deliver_outbox()

    def mark_ambiguous_attempts(self) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT a.document FROM effect_attempts a JOIN effect_intents i ON i.digest=a.intent_digest LEFT JOIN effect_receipts r ON r.intent_digest=i.digest WHERE i.state='attempting' AND r.digest IS NULL"
            ).fetchall()
            for row in rows:
                raw = json.loads(row["document"])
                receipt = EffectReceipt(
                    f"receipt-{uuid4().hex}",
                    raw["digest"],
                    raw["intent_digest"],
                    EffectOutcome.AMBIGUOUS,
                    ZERO_DIGEST,
                    "restart_unresolved",
                )
                db.execute(
                    "INSERT INTO effect_receipts VALUES(?,?,?,?)",
                    (
                        receipt.digest,
                        receipt.intent_digest,
                        canonical_json(receipt.to_json()).decode(),
                        receipt.outcome.value,
                    ),
                )
                db.execute(
                    "UPDATE effect_intents SET state='ambiguous' WHERE digest=?",
                    (receipt.intent_digest,),
                )
                self._event(
                    db, "effect.receipt.committed", receipt.digest, receipt.to_json()
                )
            db.commit()
        self.deliver_outbox()
        return len(rows)

    def state(self, intent_digest: str) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT state FROM effect_intents WHERE digest=?", (intent_digest,)
            ).fetchone()
        if row is None:
            raise EffectError("unknown effect intent")
        return str(row["state"])

    def validate_reafference_evidence(
        self, receipt_digest: str, expected_effect_digest: str
    ) -> None:
        """Validate the exact successful receipt-to-expectation chain for observation."""
        _check_digest(receipt_digest)
        _check_digest(expected_effect_digest)
        with self._connect() as db:
            row = db.execute(
                "SELECT r.document AS receipt_document,i.document AS intent_document,"
                "a.expected_document AS expected_document "
                "FROM effect_receipts r "
                "JOIN effect_intents i ON i.digest=r.intent_digest "
                "JOIN effect_capabilities c ON c.nonce=i.nonce "
                "JOIN effect_authorizations a ON a.capability_digest=c.digest "
                "WHERE r.digest=?",
                (receipt_digest,),
            ).fetchone()
        if row is None:
            raise EffectRejected("receipt evidence is missing")
        receipt = _strict_document(row["receipt_document"])
        intent = _strict_document(row["intent_document"])
        expected = _strict_document(row["expected_document"])
        receipt_claim = receipt.get("digest")
        intent_claim = intent.get("digest")
        expected_claim = expected.get("digest")
        for document, claim in (
            (receipt, receipt_claim),
            (intent, intent_claim),
            (expected, expected_claim),
        ):
            if not isinstance(claim, str):
                raise EffectRejected("receipt evidence has an invalid digest")
            unsigned = dict(document)
            unsigned.pop("digest")
            if _digest(unsigned) != claim:
                raise EffectRejected("receipt evidence failed digest verification")
        if (
            receipt_claim != receipt_digest
            or receipt.get("outcome") != EffectOutcome.SUCCEEDED.value
            or receipt.get("intent_digest") != intent_claim
            or intent.get("expected_effect_digest") != expected_effect_digest
            or expected_claim != expected_effect_digest
            or receipt.get("observed_digest") != expected.get("after_digest")
            or receipt.get("observed_digest") == ZERO_DIGEST
        ):
            raise EffectRejected(
                "receipt is not a successful observation-eligible effect chain"
            )

    def verify(self) -> None:
        with self._connect() as db:
            for table in (
                "effect_capabilities",
                "effect_intents",
                "effect_attempts",
                "effect_receipts",
            ):
                for row in db.execute(f"SELECT digest,document FROM {table}"):
                    raw = _strict_document(row["document"])
                    claimed = raw.pop("digest", None)
                    if claimed != row["digest"] or _digest(raw) != claimed:
                        raise EffectError(
                            "persisted effect artifact failed digest verification"
                        )
            for row in db.execute(
                "SELECT i.state AS intent_state,a.digest AS attempt_digest,"
                "a.document AS attempt_document,r.outcome AS receipt_outcome,"
                "r.document AS receipt_document,c.state AS capability_state "
                "FROM effect_intents i "
                "LEFT JOIN effect_attempts a ON a.intent_digest=i.digest "
                "LEFT JOIN effect_receipts r ON r.intent_digest=i.digest "
                "LEFT JOIN effect_capabilities c ON c.nonce=i.nonce"
            ):
                state = row["intent_state"]
                if state == "authorized" and row["attempt_digest"] is not None:
                    raise EffectError("authorized intent unexpectedly has an attempt")
                if state != "authorized" and row["attempt_digest"] is None:
                    raise EffectError("advanced intent is missing its attempt")
                if row["attempt_digest"] is not None:
                    attempt = _strict_document(row["attempt_document"])
                    if row["capability_state"] != "consumed":
                        raise EffectError("attempt capability was not consumed")
                    if row["receipt_document"] is not None:
                        receipt = _strict_document(row["receipt_document"])
                        if receipt["attempt_digest"] != attempt["digest"]:
                            raise EffectError("receipt is bound to the wrong attempt")
                if (
                    state
                    in {
                        outcome.value
                        for outcome in EffectOutcome
                        if outcome is not EffectOutcome.COMPENSATED
                    }
                    and state != "attempting"
                ):
                    if row["receipt_outcome"] != state and not (
                        state in {"succeeded", "failed"}
                        and row["receipt_outcome"] == EffectOutcome.AMBIGUOUS.value
                    ):
                        raise EffectError("intent state and receipt outcome disagree")
            for row in db.execute("SELECT payload FROM effect_outbox"):
                _strict_document(row["payload"])
            for row in db.execute(
                "SELECT a.*,c.digest FROM effect_authorizations a "
                "LEFT JOIN effect_capabilities c ON c.digest=a.capability_digest"
            ):
                if row["digest"] is None:
                    raise EffectError("authorization has no capability")
                for column in (
                    "proposal_document",
                    "policy_document",
                    "expected_document",
                ):
                    document = _strict_document(row[column])
                    claimed = document.pop("digest", None)
                    if not isinstance(claimed, str) or _digest(document) != claimed:
                        raise EffectError("authorization evidence digest mismatch")

    def deliver_outbox(self) -> None:
        if self.outbox_sink is None:
            return
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM effect_outbox WHERE delivered=0 ORDER BY id"
            ).fetchall()
        for row in rows:
            self.outbox_sink(row["topic"], json.loads(row["payload"]))
            with self._connect() as db:
                db.execute(
                    "UPDATE effect_outbox SET delivered=1 WHERE id=?", (row["id"],)
                )


class EffectTransactionService:
    """Kernel-only authorize/execute boundary for the deterministic sandbox."""

    def __init__(
        self,
        store: EffectStore,
        principal: Principal,
        sandbox: ReversibleSandbox,
        *,
        clock: Callable[[], datetime] | None = None,
        validate_heads: AuthorityHeadValidator | None = None,
    ) -> None:
        if not principal.is_kernel:
            raise EffectRejected("only SYSTEM_KERNEL may authorize effects")
        self.store, self.principal, self.sandbox = store, principal, sandbox
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if validate_heads is None:
            raise EffectRejected("episode and lineage head validator is required")
        self.validate_heads = validate_heads
        self.store.mark_ambiguous_attempts()

    def authorize(
        self,
        proposal: PolicyProposal,
        policy: AuthorizedPolicy,
        expected: ExpectedEffect,
        intent: EffectIntent,
        *,
        expires_at: datetime,
        nonce: str,
    ) -> CapabilityToken:
        """Validate the complete authority chain and durably issue one capability."""
        if (
            expires_at.tzinfo is None
            or expires_at.utcoffset() != timezone.utc.utcoffset(expires_at)
        ):
            raise EffectRejected("capability expiry must be UTC")
        if expires_at <= self.clock():
            raise EffectRejected("capability expiry must be in the future")
        if (
            proposal.authority != "untrusted_proposal"
            or policy.proposal_digest != proposal.digest
            or policy.principal_digest != self.principal.identity_digest
            or policy.release_digest != self.principal.release_digest
            or policy.budget != intent.budget
            or intent.policy_digest != policy.digest
            or intent.expected_effect_digest != expected.digest
            or proposal.episode_id != intent.episode_id
            or proposal.lineage_id != intent.lineage_id
            or proposal.branch_id != intent.branch_id
            or proposal.resource != intent.resource
            or proposal.operation != intent.operation
            or expected.resource != intent.resource
            or expected.operation != intent.operation
            or proposal.value_digest
            != Digest.of_bytes((intent.value or "").encode("utf-8")).hex
        ):
            raise EffectRejected(
                "policy proposal, authorization, expectation, or intent mismatch"
            )
        self.validate_heads(
            intent.episode_id,
            intent.episode_digest,
            intent.lineage_id,
            intent.branch_id,
            intent.lineage_head_digest,
        )
        token = CapabilityToken(
            token_id=f"capability-{uuid4().hex}",
            principal_digest=self.principal.identity_digest,
            release_digest=self.principal.release_digest,
            lineage_id=intent.lineage_id,
            branch_id=intent.branch_id,
            episode_id=intent.episode_id,
            policy_digest=policy.digest,
            effect_digest=expected.digest,
            resource=intent.resource,
            operation=intent.operation,
            expires_at=expires_at.isoformat().replace("+00:00", "Z"),
            nonce=nonce,
            budget=intent.budget,
            idempotency_key=intent.idempotency_key,
        )
        self.store.issue_capability(token, proposal, policy, expected)
        return token

    def execute(self, intent: EffectIntent, token: CapabilityToken) -> EffectReceipt:
        now = self.clock()
        bindings = (
            token.principal_digest == self.principal.identity_digest,
            token.release_digest == self.principal.release_digest,
            token.lineage_id == intent.lineage_id,
            token.branch_id == intent.branch_id,
            token.episode_id == intent.episode_id,
            token.policy_digest == intent.policy_digest,
            token.effect_digest == intent.expected_effect_digest,
            token.resource == intent.resource,
            token.operation == intent.operation,
            token.budget == intent.budget,
            token.idempotency_key == intent.idempotency_key,
        )
        expiry = datetime.fromisoformat(token.expires_at.replace("Z", "+00:00"))
        if not all(bindings) or now >= expiry:
            raise EffectRejected("capability binding or expiry rejected")
        if (
            intent.principal_digest != self.principal.identity_digest
            or intent.release_digest != self.principal.release_digest
        ):
            raise EffectRejected("intent principal/release mismatch")
        self.store.commit_intent(intent, token)
        if self.store.failpoint:
            self.store.failpoint("after_intent_commit")
        attempt = EffectAttempt(
            f"attempt-{uuid4().hex}",
            intent.digest,
            token.digest,
            now.isoformat().replace("+00:00", "Z"),
        )
        self.store.start_attempt(attempt)
        if self.store.failpoint:
            self.store.failpoint("after_attempt_commit")
        before: str | None = None
        try:
            expected = self.store.expected_effect(token.digest)
            current = self.sandbox.values.get(intent.resource)
            current_digest = _digest({"resource": intent.resource, "value": current})
            if current_digest != expected.before_digest:
                raise EffectRejected(
                    "sandbox precondition does not match expected effect"
                )
            before, observed = self.sandbox.execute(intent)
            if observed != expected.after_digest:
                self.sandbox.compensate(intent, before)
                raise RuntimeError("sandbox consequence does not match expected effect")
            if self.store.failpoint:
                self.store.failpoint("after_external_execution")
            receipt = EffectReceipt(
                f"receipt-{uuid4().hex}",
                attempt.digest,
                intent.digest,
                EffectOutcome.SUCCEEDED,
                observed,
                "sandbox_applied",
            )
        except EffectRejected:
            receipt = EffectReceipt(
                f"receipt-{uuid4().hex}",
                attempt.digest,
                intent.digest,
                EffectOutcome.REJECTED,
                ZERO_DIGEST,
                "sandbox_rejected",
            )
        except Exception:
            receipt = EffectReceipt(
                f"receipt-{uuid4().hex}",
                attempt.digest,
                intent.digest,
                EffectOutcome.FAILED,
                ZERO_DIGEST,
                "sandbox_failed",
            )
        self.store.finish(receipt)
        return receipt

    def compensate(
        self, intent: EffectIntent, original: EffectReceipt, before: str | None
    ) -> EffectReceipt:
        if original.outcome is not EffectOutcome.SUCCEEDED:
            raise EffectRejected("only successful effects can be compensated")
        observed = self.sandbox.compensate(intent, before)
        receipt = EffectReceipt(
            f"receipt-{uuid4().hex}",
            original.attempt_digest,
            intent.digest,
            EffectOutcome.COMPENSATED,
            observed,
            "sandbox_rolled_back",
            original.digest,
        )
        # Compensation is an additional audit artifact, not a second execution receipt.
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.store._event(
                db, "effect.compensated", receipt.digest, receipt.to_json()
            )
            db.execute(
                "UPDATE effect_intents SET state='compensated' WHERE digest=?",
                (intent.digest,),
            )
            db.commit()
        self.store.deliver_outbox()
        return receipt

    def reconcile(
        self,
        intent_digest: str,
        *,
        observed_digest: str,
        operator: Principal,
        succeeded: bool,
    ) -> None:
        if operator.kind.value != "operator":
            raise EffectRejected("operator reconciliation required")
        _check_digest(observed_digest)
        if self.store.state(intent_digest) != EffectOutcome.AMBIGUOUS.value:
            raise EffectConflict("only ambiguous effects can be reconciled")
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE effect_intents SET state=? WHERE digest=?",
                ("succeeded" if succeeded else "failed", intent_digest),
            )
            self.store._event(
                db,
                "effect.reconciled",
                _digest(
                    {
                        "intent": intent_digest,
                        "operator": operator.identity_digest,
                        "observed": observed_digest,
                        "succeeded": succeeded,
                    }
                ),
                {
                    "intent_digest": intent_digest,
                    "observed_digest": observed_digest,
                    "operator_digest": operator.identity_digest,
                    "succeeded": succeeded,
                },
            )
            db.commit()
        self.store.deliver_outbox()
