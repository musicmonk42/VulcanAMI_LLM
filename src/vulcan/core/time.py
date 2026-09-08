"""Injected UTC clock primitives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from vulcan.constitution.primitives import canonical_timestamp, parse_timestamp, require_utc

UTC_PRECISION = "milliseconds"


class Clock(Protocol):
    def now(self) -> datetime: ...


def canonical_utc(value: datetime) -> datetime:
    utc = require_utc(value)
    return utc.replace(microsecond=(utc.microsecond // 1000) * 1000)


def format_utc(value: datetime) -> str:
    return canonical_timestamp(value)


def parse_utc(value: str) -> datetime:
    return parse_timestamp(value)


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
