"""Closed registration for deterministic Graphix operations.

Registrations are module constants, not request data.  A Graphix artifact can
select a name from this allowlist but cannot supply code, imports or class paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True, slots=True)
class RegisteredOperation:
    name: str
    operand: str
    executor: Callable[..., object]


class OperationRegistry:
    def __init__(self, operations: tuple[RegisteredOperation, ...]) -> None:
        items: dict[str, RegisteredOperation] = {}
        for operation in operations:
            if operation.name in items or operation.name not in {
                "arithmetic",
                "lookup",
            }:
                raise ValueError("duplicate or non-canonical Graphix operation")
            if operation.operand not in {"expression", "key"}:
                raise ValueError("non-canonical Graphix operand")
            items[operation.name] = operation
        self._items: Mapping[str, RegisteredOperation] = MappingProxyType(items)

    def require(self, name: str) -> RegisteredOperation:
        try:
            return self._items[name]
        except KeyError:
            raise ValueError("operation is not registered") from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def _arithmetic_executor(
    *, accepted: object, execute: Callable[..., object], **_: object
) -> object:
    return execute(accepted)


def _lookup_executor(*, lookup: Callable[..., object], **context: object) -> object:
    return lookup(**context)


CANONICAL_OPERATIONS = OperationRegistry(
    (
        RegisteredOperation("arithmetic", "expression", _arithmetic_executor),
        RegisteredOperation("lookup", "key", _lookup_executor),
    )
)
