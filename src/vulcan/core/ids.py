"""Canonical AMI identifier constructors and validators."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from uuid import UUID, uuid4

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SUFFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
_LEGACY_CASE_RE = re.compile(r"^case-[0-9a-f]{32}$")


class IdKind(str, Enum):
    EPISODE = "episode"
    CASE = "case"
    REQUEST = "request"
    TRANSACTION = "txn"
    ACTOR = "actor"
    CAPABILITY = "cap"
    POLICY = "policy"
    SNAPSHOT = "snapshot"
    EVIDENCE = "evidence"
    SELECTION = "selection"
    OBSERVATION = "observation"
    PROPOSAL = "proposal"
    APPROVAL = "approval"


_PREFIX = {
    IdKind.EPISODE: "ep",
    IdKind.CASE: "case",
    IdKind.REQUEST: "req",
    IdKind.TRANSACTION: "txn",
    IdKind.ACTOR: "actor",
    IdKind.CAPABILITY: "cap",
    IdKind.POLICY: "pol",
    IdKind.SNAPSHOT: "snap",
    IdKind.EVIDENCE: "ev",
    IdKind.SELECTION: "sel",
    IdKind.OBSERVATION: "obs",
    IdKind.PROPOSAL: "prop",
    IdKind.APPROVAL: "appr",
}


@dataclass(frozen=True, slots=True)
class CanonicalId:
    kind: IdKind
    value: str

    def __str__(self) -> str:
        return self.value


def _prefix(kind: IdKind) -> str:
    return _PREFIX[kind]


def new_id(kind: IdKind) -> CanonicalId:
    return CanonicalId(kind, f"{_prefix(kind)}_{uuid4().hex}")


def from_digest(kind: IdKind, digest: str) -> CanonicalId:
    if _HEX64_RE.fullmatch(digest) is None:
        raise ValueError("authoritative digest identifiers require full lowercase SHA-256")
    return CanonicalId(kind, f"{_prefix(kind)}_sha256_{digest}")


def from_slug(kind: IdKind, slug: str) -> CanonicalId:
    if _SUFFIX_RE.fullmatch(slug) is None:
        raise ValueError("slug identifier suffix is invalid")
    return CanonicalId(kind, f"{_prefix(kind)}_{slug}")


def validate_id(kind: IdKind, value: str, *, allow_legacy_case: bool = False) -> CanonicalId:
    prefix = _prefix(kind)
    if allow_legacy_case and kind is IdKind.CASE and _LEGACY_CASE_RE.fullmatch(value):
        return CanonicalId(kind, value)
    if value.startswith(f"{prefix}_sha256_"):
        digest = value.removeprefix(f"{prefix}_sha256_")
        if _HEX64_RE.fullmatch(digest) is None:
            raise ValueError("digest id must include full lowercase SHA-256")
        return CanonicalId(kind, value)
    if not value.startswith(f"{prefix}_"):
        raise ValueError(f"{kind.value} id has wrong prefix")
    suffix = value.removeprefix(f"{prefix}_")
    if len(suffix) == 32:
        try:
            UUID(hex=suffix)
        except ValueError as exc:
            raise ValueError("uuid id suffix is invalid") from exc
        if suffix != suffix.lower():
            raise ValueError("uuid id suffix must be lowercase")
        return CanonicalId(kind, value)
    if _SUFFIX_RE.fullmatch(suffix):
        return CanonicalId(kind, value)
    raise ValueError("identifier suffix is invalid")


def short_display(value: str, width: int = 12) -> str:
    if width < 8:
        raise ValueError("display width too small")
    return value if len(value) <= width else value[:width] + "…"
