"""Principal identities for the cognitive microkernel authority boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping


class PrincipalKind(str, Enum):
    HUMAN = "human"
    SYSTEM_KERNEL = "system_kernel"
    LANGUAGE_PROVIDER = "language_provider"
    REASONER = "reasoner"
    RETRIEVER = "retriever"
    TOOL = "tool"
    POLICY_AUTHORITY = "policy_authority"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    EXTERNAL_PROVIDER = "external_provider"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_metadata(value: Mapping[str, str] | None) -> Mapping[str, str]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class Principal:
    """Immutable principal identity; metadata never grants authority."""

    kind: PrincipalKind
    principal_id: str
    release_digest: str
    display_name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id is required")
        if len(self.release_digest) != 64 or any(c not in "0123456789abcdef" for c in self.release_digest):
            raise ValueError("release_digest must be a lowercase sha256 hex digest")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "identity_digest", digest(self.to_json(include_digest=False)))

    @property
    def is_kernel(self) -> bool:
        return self.kind is PrincipalKind.SYSTEM_KERNEL

    def to_json(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "display_name": self.display_name,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
            "principal_id": self.principal_id,
            "release_digest": self.release_digest,
        }
        if include_digest:
            payload["identity_digest"] = self.identity_digest
        return payload
