"""Request-scoped, privacy-preserving cognitive unit of work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import uuid4


class CognitiveCaseStatus(str, Enum):
    OPEN = "open"
    SUCCESS = "success"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CaseEvent:
    stage: str
    at: datetime
    detail: str | None = None


@dataclass
class CognitiveCase:
    """The sole mutable request record; it never retains the raw prompt."""

    request_id: str
    conversation_id: str | None
    input_hash: str
    case_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = "1"
    privacy_classification: str = "request-confidential"
    state_snapshot_id: str | None = None
    routing_proposal: Any | None = None
    routing_decision: str | None = None
    selected_components: tuple[str, ...] = ()
    terminal_status: CognitiveCaseStatus = CognitiveCaseStatus.OPEN
    failure_kind: str | None = None
    events: list[CaseEvent] = field(default_factory=list)

    @classmethod
    def create(cls, *, request_id: str, conversation_id: str | None, message: str) -> "CognitiveCase":
        case = cls(request_id=request_id, conversation_id=conversation_id,
                   input_hash=sha256(message.encode("utf-8")).hexdigest())
        case.record("created")
        return case

    def record(self, stage: str, detail: str | None = None) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot append an event after a cognitive case is closed")
        self.events.append(CaseEvent(stage, datetime.now(timezone.utc), detail))

    def close(self, status: CognitiveCaseStatus, failure_kind: str | None = None) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cognitive case closed more than once")
        self.failure_kind = failure_kind
        self.record("terminal", status.value)
        self.terminal_status = status
