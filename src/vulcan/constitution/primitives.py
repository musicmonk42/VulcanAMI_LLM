"""Canonical constitutional primitives; standard-library dependencies only."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
import unicodedata
from typing import Any

_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$", re.ASCII)
_MAX_SAFE_INTEGER = 2**53 - 1


class Digest(str):
    """Validated digest with the sole canonical wire form ``sha256:<hex>``."""

    def __new__(cls, value: str) -> "Digest":
        if (
            not isinstance(value, str)
            or not value.startswith("sha256:")
            or not _HEX.fullmatch(value[7:])
        ):
            raise ValueError("digest must be sha256:<64 lowercase hex>")
        return str.__new__(cls, value)

    @classmethod
    def of_bytes(cls, value: bytes) -> "Digest":
        if not isinstance(value, bytes):
            raise TypeError("digest input must be bytes")
        return cls("sha256:" + hashlib.sha256(value).hexdigest())

    @classmethod
    def of_json(cls, value: object) -> "Digest":
        return cls.of_bytes(canonical_json(value))

    @classmethod
    def from_legacy_hex(cls, value: str) -> "Digest":
        if not isinstance(value, str) or not _HEX.fullmatch(value):
            raise ValueError("legacy digest must be 64 lowercase hex characters")
        return cls("sha256:" + value)

    @property
    def hex(self) -> str:
        """Explicit adapter for persisted bare-hex contracts."""
        return self[7:]


class _Id(str):
    kind = "identifier"

    def __new__(cls, value: str):
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise ValueError(f"invalid {cls.kind}")
        return str.__new__(cls, value)


class EpisodeId(_Id):
    kind = "episode_id"


class ArtifactId(_Id):
    kind = "artifact_id"


class PrincipalId(_Id):
    kind = "principal_id"


class SnapshotId(_Id):
    kind = "snapshot_id"


class CommitId(_Id):
    kind = "commit_id"


class LineageId(_Id):
    kind = "lineage_id"


class AuthorityLevel(Enum):
    """Authority values that cannot compare equal to unvalidated strings."""

    UNTRUSTED_PROPOSAL = "UNTRUSTED_PROPOSAL"
    VALIDATED_CANDIDATE = "VALIDATED_CANDIDATE"
    COMMITTED_BELIEF = "COMMITTED_BELIEF"
    AUTHORIZED_PLAN = "AUTHORIZED_PLAN"
    EXECUTED_EFFECT = "EXECUTED_EFFECT"

    @property
    def rank(self) -> int:
        return _AUTHORITY_ORDER[self]

    def dominates(self, required: "AuthorityLevel") -> bool:
        if not isinstance(required, AuthorityLevel):
            raise TypeError("authority comparisons require AuthorityLevel values")
        return self.rank >= required.rank


_AUTHORITY_ORDER = {
    AuthorityLevel.UNTRUSTED_PROPOSAL: 0,
    AuthorityLevel.VALIDATED_CANDIDATE: 1,
    AuthorityLevel.COMMITTED_BELIEF: 2,
    AuthorityLevel.AUTHORIZED_PLAN: 3,
    AuthorityLevel.EXECUTED_EFFECT: 4,
}


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number rejected: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _validate(value: object) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("integer outside interoperable JSON bound")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number rejected")
        return
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError("noncanonical Unicode rejected; NFC is required")
        if any(
            ord(character) < 0x20 or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        ):
            raise ValueError("JSON strings may not contain controls or surrogates")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _validate(key)
            _validate(item)
        return
    raise TypeError(f"unsupported canonical JSON object: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    _validate(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def canonical_json_loads(value: bytes | str) -> object:
    result = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    _validate(result)
    return result


def require_utc(value: datetime, *, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def canonical_timestamp(value: datetime) -> str:
    """Return the repository wire timestamp at millisecond precision."""
    utc = require_utc(value)
    utc = utc.replace(microsecond=(utc.microsecond // 1000) * 1000)
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    """Parse a timezone-bearing ISO timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError("invalid ISO timestamp") from exc
    utc = require_utc(parsed)
    if not value.endswith("Z"):
        raise ValueError("canonical timestamp must end with Z")
    if utc.microsecond % 1000:
        raise ValueError("canonical timestamp precision must be milliseconds")
    return utc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
