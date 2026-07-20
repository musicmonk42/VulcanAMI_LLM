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

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.state_machine import ALLOWED_TRANSITIONS, EpisodeState


class CognitiveCaseStatus(str, Enum):
    OPEN = "open"
    SUCCESS = "success"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    FINALIZATION_ERROR = "finalization_error"
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
    interpretation: "InterpretationBundle | None" = field(default=None, repr=False)
    accepted_interpretation: "AcceptedInterpretation | None" = field(default=None, repr=False)
    clarification: "ClarificationRequest | None" = field(default=None, repr=False)
    _evidence: list["EvidenceArtifact"] = field(default_factory=list, repr=False)
    _claims: list["Claim"] = field(default_factory=list, repr=False)
    _derivations: list["Derivation"] = field(default_factory=list, repr=False)
    response_ir: "ResponseIR | None" = field(default=None, repr=False)
    selected_components: tuple[str, ...] = ()
    terminal_status: CognitiveCaseStatus = CognitiveCaseStatus.OPEN
    failure_kind: str | None = None
    finalization_status: str | None = None
    render_artifact: object | None = field(default=None, repr=False)
    events: list[CaseEvent] = field(default_factory=list)
    episode: CognitiveEpisode | None = field(default=None, repr=False)

    @classmethod
    def create(cls, *, request_id: str, conversation_id: str | None, input_digest: str | None = None,
               message: str | None = None) -> "CognitiveCase":
        # ``message`` is retained only for in-process compatibility callers and is
        # immediately digested; the case never retains raw request content.
        if input_digest is None:
            if message is None:
                raise ValueError("input_digest is required")
            input_digest = sha256(message.encode("utf-8")).hexdigest()
        episode = CognitiveEpisode.create(
            actor=ActorBinding(
                actor_id="legacy-runtime",
                principal_digest=sha256(request_id.encode("utf-8")).hexdigest(),
                authority="CognitiveCaseAdapter",
            ),
            request_id=request_id,
            input_digest=input_digest,
            conversation_id=conversation_id,
        )
        case = cls(request_id=request_id, conversation_id=conversation_id, input_hash=input_digest, case_id=episode.episode_id, episode=episode)
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
        validate_ledger(proposed_evidence, proposed_derivations, proposed_claims, case_id=self.case_id)
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
        if self.episode is not None:
            target = {
                CognitiveCaseStatus.SUCCESS: EpisodeState.COMMUNICATED,
                CognitiveCaseStatus.ABSTAINED: EpisodeState.ABSTAINED,
                CognitiveCaseStatus.BLOCKED: EpisodeState.BLOCKED,
                CognitiveCaseStatus.FINALIZATION_ERROR: EpisodeState.FAILED,
                CognitiveCaseStatus.FAILED: EpisodeState.FAILED,
                CognitiveCaseStatus.CANCELLED: EpisodeState.CANCELLED,
            }.get(status)
            if target is not None and target in ALLOWED_TRANSITIONS[self.episode.state]:
                self.episode = self.episode.transition(
                    target, reason=failure_kind or status.value, authority="CognitiveCaseAdapter"
                )


def episode_from_case(case: CognitiveCase) -> CognitiveEpisode:
    """Compatibility adapter exposing the authoritative episode for a legacy case."""
    if case.episode is not None:
        return case.episode
    return CognitiveEpisode.create(
        actor=ActorBinding(
            actor_id="legacy-runtime",
            principal_digest=sha256(case.request_id.encode("utf-8")).hexdigest(),
            authority="CognitiveCaseAdapter",
        ),
        request_id=case.request_id,
        input_digest=case.input_hash,
        conversation_id=case.conversation_id,
    )
