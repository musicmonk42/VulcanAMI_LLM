"""Request-scoped, privacy-preserving cognitive unit of work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic import (AcceptedInterpretation, Claim, ClarificationRequest, Derivation,
                           EvidenceArtifact, InterpretationBundle, ResponseIR)
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
    interpretation: "InterpretationBundle | None" = None
    accepted_interpretation: "AcceptedInterpretation | None" = None
    clarification: "ClarificationRequest | None" = None
    _evidence: list["EvidenceArtifact"] = field(default_factory=list, repr=False)
    _claims: list["Claim"] = field(default_factory=list, repr=False)
    _derivations: list["Derivation"] = field(default_factory=list, repr=False)
    response_ir: "ResponseIR | None" = None
    selected_components: tuple[str, ...] = ()
    terminal_status: CognitiveCaseStatus = CognitiveCaseStatus.OPEN
    failure_kind: str | None = None
    finalization_status: str | None = None
    render_artifact: object | None = field(default=None, repr=False)
    events: list[CaseEvent] = field(default_factory=list)

    @classmethod
    def create(cls, *, request_id: str, conversation_id: str | None, input_digest: str | None = None,
               message: str | None = None) -> "CognitiveCase":
        # ``message`` is retained only for in-process compatibility callers and is
        # immediately digested; the case never retains raw request content.
        if input_digest is None:
            if message is None:
                raise ValueError("input_digest is required")
            input_digest = sha256(message.encode("utf-8")).hexdigest()
        case = cls(request_id=request_id, conversation_id=conversation_id, input_hash=input_digest)
        case.record("created")
        return case

    def record(self, stage: str, detail: str | None = None) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot append an event after a cognitive case is closed")
        self.events.append(CaseEvent(stage, datetime.now(timezone.utc), detail))

    @property
    def evidence(self) -> tuple["EvidenceArtifact", ...]:
        return tuple(self._evidence)

    @property
    def claims(self) -> tuple["Claim", ...]:
        return tuple(self._claims)

    @property
    def derivations(self) -> tuple["Derivation", ...]:
        return tuple(self._derivations)

    def append_ledger(self, *, claim: "Claim", derivation: "Derivation", evidence: tuple["EvidenceArtifact", ...] = ()) -> None:
        """Atomically validate and commit one request-local ledger transaction."""
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot mutate a closed case ledger")
        from .semantic import validate_ledger
        proposed_evidence = (*self._evidence, *evidence)
        proposed_derivations = (*self._derivations, derivation)
        proposed_claims = (*self._claims, claim)
        validate_ledger(proposed_evidence, proposed_derivations, proposed_claims)
        self._evidence.extend(evidence)
        self._derivations.append(derivation)
        self._claims.append(claim)

    def record_finalization(self, decision: str) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot finalize a closed cognitive case")
        if self.finalization_status is not None:
            raise RuntimeError("response finalized more than once")
        self.finalization_status = decision
        self.record("finalized", decision)

    def close(self, status: CognitiveCaseStatus, failure_kind: str | None = None) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cognitive case closed more than once")
        self.failure_kind = failure_kind
        self.record("terminal", status.value)
        self.terminal_status = status
