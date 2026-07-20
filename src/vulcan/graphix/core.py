"""Immutable Graphix Core envelope shared by cognitive dialects.

Graphix Core carries authority-context as data only. It never carries executable
semantics, dynamic imports, class paths, shell commands, or raw model authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping, Sequence

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_RELEASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/@-]{2,127}$")
_EXT_KEY_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9-]*){2,}$")
FORBIDDEN_EXTENSION_KEYS = frozenset({"authority", "authority_level", "policy", "executable", "command", "code", "callable", "class_path", "import"})
MAX_REFERENCES = 32
MAX_CONSENT_REFERENCES = 32
MAX_EXTENSIONS = 16
MAX_EXTENSION_VALUE_BYTES = 4096

class GraphixCoreError(ValueError):
    """Base class for exact Graphix Core contract failures."""
class UnknownFieldError(GraphixCoreError): pass
class DigestMismatchError(GraphixCoreError): pass
class ForbiddenExecutableSemanticsError(GraphixCoreError): pass
class ExtensionCollisionError(GraphixCoreError): pass
class UnsupportedDialectError(GraphixCoreError): pass

class AuthorityLevel(str, Enum):
    UNTRUSTED_PROPOSAL = "UNTRUSTED_PROPOSAL"
    VALIDATED_CANDIDATE = "VALIDATED_CANDIDATE"
    COMMITTED_BELIEF = "COMMITTED_BELIEF"
    AUTHORIZED_PLAN = "AUTHORIZED_PLAN"
    EXECUTED_EFFECT = "EXECUTED_EFFECT"
class EpistemicStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    PROPOSED = "PROPOSED"
    CONTESTED = "CONTESTED"
    RETIRED = "RETIRED"
class PrivacyClass(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PERSONAL = "PERSONAL"
    SENSITIVE_PERSONAL = "SENSITIVE_PERSONAL"
class SourceKind(str, Enum):
    EPISODE = "EPISODE"
    ARTIFACT = "ARTIFACT"
    SNAPSHOT = "SNAPSHOT"
    CONSENT = "CONSENT"
    CONTROL = "CONTROL"
    EXTERNAL = "EXTERNAL"

@dataclass(frozen=True, slots=True)
class PrincipalRelease:
    principal_id: str
    release_id: str
    def __post_init__(self) -> None:
        _require_match("principal_id", self.principal_id, _ID_RE)
        _require_match("release_id", self.release_id, _RELEASE_RE)

@dataclass(frozen=True, slots=True)
class SourceReference:
    kind: SourceKind
    reference_id: str
    digest: str | None = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(SourceKind, self.kind, "source.kind"))
        _require_match("source.reference_id", self.reference_id, _ID_RE)
        if self.digest is not None: _require_digest("source.digest", self.digest)

@dataclass(frozen=True, slots=True)
class ExtensionDeclaration:
    namespace: str
    schema_version: int
    digest: str
    value: Mapping[str, object] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not _EXT_KEY_RE.match(self.namespace):
            raise ExtensionCollisionError("extension namespace must be reverse-DNS style with at least three labels")
        low = self.namespace.lower()
        if any(part in FORBIDDEN_EXTENSION_KEYS for part in low.replace("-", ".").split(".")):
            raise ExtensionCollisionError("extension namespace collides with authority/executable semantics")
        _require_positive_version("extension.schema_version", self.schema_version)
        _require_digest("extension.digest", self.digest)
        frozen = _freeze_extension(self.value)
        object.__setattr__(self, "value", frozen)

@dataclass(frozen=True, slots=True)
class GraphixEnvelope:
    dialect: str
    schema_version: int
    node_artifact_id: str
    episode_id: str
    content_digest: str
    proposer: PrincipalRelease
    authority_level: AuthorityLevel
    source_references: tuple[SourceReference, ...]
    snapshot_bundle_digest: str
    epistemic_status: EpistemicStatus
    privacy_class: PrivacyClass
    purpose: str
    consent_references: tuple[str, ...]
    valid_from: datetime
    valid_until: datetime | None
    extensions: tuple[ExtensionDeclaration, ...] = ()
    def __post_init__(self) -> None:
        _require_match("dialect", self.dialect, _TOKEN_RE)
        _require_positive_version("schema_version", self.schema_version)
        _require_match("node_artifact_id", self.node_artifact_id, _ID_RE)
        _require_match("episode_id", self.episode_id, _ID_RE)
        _require_digest("content_digest", self.content_digest)
        _require_digest("snapshot_bundle_digest", self.snapshot_bundle_digest)
        object.__setattr__(self, "authority_level", _enum(AuthorityLevel, self.authority_level, "authority_level"))
        object.__setattr__(self, "epistemic_status", _enum(EpistemicStatus, self.epistemic_status, "epistemic_status"))
        object.__setattr__(self, "privacy_class", _enum(PrivacyClass, self.privacy_class, "privacy_class"))
        if not (1 <= len(self.purpose) <= 256): raise GraphixCoreError("purpose length outside bounds")
        if len(self.source_references) > MAX_REFERENCES: raise GraphixCoreError("too many source references")
        if len(self.consent_references) > MAX_CONSENT_REFERENCES: raise GraphixCoreError("too many consent references")
        for cref in self.consent_references: _require_match("consent_reference", cref, _ID_RE)
        if self.valid_from.tzinfo is None: raise GraphixCoreError("valid_from must be timezone-aware")
        if self.valid_until is not None and (self.valid_until.tzinfo is None or self.valid_until <= self.valid_from):
            raise GraphixCoreError("valid_until must be timezone-aware and after valid_from")
        if len(self.extensions) > MAX_EXTENSIONS: raise GraphixCoreError("too many extensions")
        namespaces = [e.namespace for e in self.extensions]
        if len(namespaces) != len(set(namespaces)): raise ExtensionCollisionError("duplicate extension namespace")

def _require_match(name: str, value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None: raise GraphixCoreError(f"invalid {name}")
def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None: raise DigestMismatchError(f"invalid {name}")
def _require_positive_version(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 9999: raise GraphixCoreError(f"invalid {name}")
def _enum(cls: type[Enum], value: object, name: str) -> Enum:
    try: return cls(value)
    except ValueError as exc: raise GraphixCoreError(f"invalid {name}") from exc

def _freeze_extension(value: Mapping[str, object]) -> Mapping[str, object]:
    from vulcan.graphix.codec import extension_digest, validate_json_value
    validate_json_value(value, allow_extension_objects=True)
    if extension_digest(value) == "sha256:" + "0"*64: raise GraphixCoreError("unreachable digest")
    return MappingProxyType(dict(value))
