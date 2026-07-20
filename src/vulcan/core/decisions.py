"""Closed typed decision results for authority boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class DecisionOutcome(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


class DecisionCategory(str, Enum):
    SAFETY = "safety"
    AUTHORIZATION = "authorization"
    POLICY = "policy"
    TERMINALIZATION = "terminalization"
    DELIVERY = "delivery"
    READINESS = "readiness"


@dataclass(frozen=True, slots=True)
class Decision:
    category: DecisionCategory
    outcome: DecisionOutcome
    reason_code: str
    subject_id: str
    evidence_digest: str | None = None
    details: Mapping[str, str] = MappingProxyType({})

    @property
    def allowed(self) -> bool:
        return self.outcome is DecisionOutcome.ALLOW

    @classmethod
    def allow(cls, category: DecisionCategory, subject_id: str, reason_code: str = "allowed") -> "Decision":
        return cls(category, DecisionOutcome.ALLOW, reason_code, subject_id)

    @classmethod
    def deny(cls, category: DecisionCategory, outcome: DecisionOutcome, subject_id: str, reason_code: str) -> "Decision":
        if outcome is DecisionOutcome.ALLOW:
            raise ValueError("deny cannot produce ALLOW")
        return cls(category, outcome, reason_code, subject_id)
