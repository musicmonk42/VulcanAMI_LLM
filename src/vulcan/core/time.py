"""Injected UTC clock primitives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

UTC_PRECISION = "milliseconds"


class Clock(Protocol):
    def now(self) -> datetime: ...


def canonical_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    return utc.replace(microsecond=(utc.microsecond // 1000) * 1000)


def format_utc(value: datetime) -> str:
    utc = canonical_utc(value)
    text = utc.isoformat(timespec="milliseconds")
    return text.replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("timestamp must end with Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError("timestamp must be UTC")
    if parsed.microsecond % 1000 != 0:
        raise ValueError("timestamp precision must be milliseconds")
    return parsed


@dataclass(frozen=True, slots=True)
class SystemClock:
    def now(self) -> datetime:
        return canonical_utc(datetime.now(timezone.utc))


@dataclass(slots=True)
class DeterministicClock:
    current: datetime

    def __post_init__(self) -> None:
        self.current = canonical_utc(self.current)

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> datetime:
        if delta.total_seconds() < 0:
            raise ValueError("clock cannot move backwards")
        self.current = canonical_utc(self.current + delta)
        return self.current
