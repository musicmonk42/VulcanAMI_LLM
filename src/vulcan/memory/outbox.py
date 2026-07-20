"""Transactional outbox contracts for governed memory audit publication."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Protocol


class MemoryOutboxError(RuntimeError):
    """Raised when a committed memory audit event cannot be delivered safely."""


class MemoryOutboxFailpoint(Protocol):
    def hit(self, name: str) -> None: ...


class NoopMemoryOutboxFailpoint:
    def hit(self, name: str) -> None:
        return None


@dataclass(frozen=True)
class MemoryAuditOutboxEvent:
    event_id: str
    event_type: str
    operation: str
    record_id: str
    revision: int
    tenant_id: str
    subject_id: str
    purpose: str
    namespace: str
    key_name: str
    prior_record_digest: str | None
    new_record_digest: str
    deletion_epoch: int
    request_digest: str
    payload_digest: str


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def deterministic_event_id(*, request_digest: str, record_id: str, revision: int, event_type: str) -> str:
    return sha256_hex(canonical_json_bytes({"event_type": event_type, "record_id": record_id, "request_digest": request_digest, "revision": revision}))


def payload_digest(payload: dict[str, object]) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def deliver_idempotently(*, append: Callable[[str, dict[str, object]], object], event_type: str, payload: dict[str, object]) -> None:
    """Deliver once in effect: duplicate transaction terminal errors are success.

    CanonicalAudit enforces transaction_id uniqueness.  A crash after append and
    before marking the DB row delivered will retry the same deterministic
    transaction_id; the duplicate terminal/prepared response is the proof that
    the audit effect already exists.
    """
    try:
        append(event_type, payload)
    except RuntimeError as exc:
        message = str(exc)
        if "duplicate transaction" in message:
            return
        raise MemoryOutboxError("memory audit outbox delivery failed") from exc
