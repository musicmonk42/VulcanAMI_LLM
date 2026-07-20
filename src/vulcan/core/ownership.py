"""Resource ownership contracts for authority-bearing injected resources."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ResourceOwnership(str, Enum):
    """Whether a resource wrapper owns the underlying authority."""

    BORROWED = "borrowed"
    OWNED = "owned"


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ResourceHandle:
    """Immutable resource ownership marker.

    Injected authorities are borrowed by default.  A close on a borrowed handle
    intentionally does not cascade to the wrapped resource.
    """

    resource: object
    ownership: ResourceOwnership = ResourceOwnership.BORROWED

    def close(self) -> None:
        if self.ownership is ResourceOwnership.OWNED and isinstance(self.resource, Closeable):
            self.resource.close()

    @property
    def is_owned(self) -> bool:
        return self.ownership is ResourceOwnership.OWNED
