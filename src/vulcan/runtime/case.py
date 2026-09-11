"""Request-scoped, privacy-preserving projection of a CognitiveEpisode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from vulcan.microkernel.snapshots import GraphixEvaluationContext, SnapshotBundle

    from vulcan.graphix.runtime import (
        AcceptedInterpretation,
        ClarificationRequest,
        InterpretationBundle,
        ResponseIR,
    )

from vulcan.microkernel.episode import ActorBinding, ArtifactRef, CognitiveEpisode
from vulcan.microkernel.state_machine import EpisodeState


class CognitiveCaseStatus(str, Enum):
    OPEN = "open"
    SUCCESS = "success"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    FINALIZATION_ERROR = "finalization_error"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_EPISODE_STATE: dict[CognitiveCaseStatus, EpisodeState] = {
    CognitiveCaseStatus.SUCCESS: EpisodeState.CONSOLIDATED,
    # A released abstention is still a communication transaction. Its semantic
    # outcome remains ABSTAINED while its authoritative lifecycle consolidates.
    CognitiveCaseStatus.ABSTAINED: EpisodeState.CONSOLIDATED,
    CognitiveCaseStatus.BLOCKED: EpisodeState.BLOCKED,
    CognitiveCaseStatus.FINALIZATION_ERROR: EpisodeState.FAILED,
    CognitiveCaseStatus.FAILED: EpisodeState.FAILED,
    CognitiveCaseStatus.CANCELLED: EpisodeState.CANCELLED,
}


@dataclass(frozen=True)
class CaseEvent:
    stage: str
    at: datetime
    detail: str | None = None


@dataclass
class CognitiveCase:
    """Mutable compatibility workspace backed by one immutable episode.

    The semantic kernel still populates request-local Python objects. This class
    validates and projects those objects only. It cannot advance, persist, or
    terminalize the authoritative episode.
    """

    request_id: str
    conversation_id: str | None
    input_hash: str
    case_id: str = field(default_factory=lambda: f"case-{uuid4().hex}")
    schema_version: str = "1"
    privacy_classification: str = "request-confidential"
    state_snapshot_id: str | None = None
    interpretation: "InterpretationBundle | None" = field(default=None, repr=False)
    accepted_interpretation: "AcceptedInterpretation | None" = field(
        default=None, repr=False
    )
    clarification: "ClarificationRequest | None" = field(default=None, repr=False)
    response_ir: "ResponseIR | None" = field(default=None, repr=False)
    selected_components: tuple[str, ...] = ()
    terminal_status: CognitiveCaseStatus = CognitiveCaseStatus.OPEN
    failure_kind: str | None = None
    finalization_status: str | None = None
    render_artifact: object | None = field(default=None, repr=False)
    events: list[CaseEvent] = field(default_factory=list)
    episode: CognitiveEpisode | None = field(default=None, repr=False)
    _snapshot_bundle: "SnapshotBundle | None" = field(default=None, repr=False)
    _evaluation_context: "GraphixEvaluationContext | None" = field(
        default=None, repr=False
    )

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        conversation_id: str | None,
        input_digest: str | None = None,
        message: str | None = None,
        actor: ActorBinding | None = None,
        case_id: str | None = None,
    ) -> "CognitiveCase":
        if input_digest is None:
            if message is None:
                raise ValueError("input_digest is required")
            input_digest = sha256(message.encode("utf-8")).hexdigest()
        resolved_case_id = case_id or f"case-{uuid4().hex}"
        actor_binding = actor or ActorBinding(
            actor_id="legacy-unverified",
            principal_digest=sha256(b"legacy-unverified").hexdigest(),
            authority="LegacyCompatibility",
        )
        episode = CognitiveEpisode.create(
            actor=actor_binding,
            request_id=request_id,
            input_digest=input_digest,
            conversation_id=conversation_id,
            episode_id=resolved_case_id,
        )
        case = cls(
            request_id=request_id,
            conversation_id=conversation_id,
            input_hash=input_digest,
            case_id=resolved_case_id,
            episode=episode,
        )
        case.record("created")
        return case

    @classmethod
    def from_admitted_episode(
        cls,
        *,
        episode: CognitiveEpisode,
        bundle: "SnapshotBundle",
    ) -> "CognitiveCase":
        """Create the compatibility projection after authoritative admission."""
        if episode.snapshot_bundle != bundle.bundle_ref():
            raise ValueError("episode and snapshot bundle do not match")
        if episode.episode_id != bundle.episode_id:
            raise ValueError("episode and snapshot identities do not match")
        case = cls(
            request_id=episode.request.request_id,
            conversation_id=episode.conversation_id,
            input_hash=episode.request.input_digest,
            case_id=episode.episode_id,
            state_snapshot_id=bundle.digest,
            episode=episode,
            _snapshot_bundle=bundle,
            _evaluation_context=(
                bundle.evaluation_context()
                if bundle.domain.live_view is not None
                and bundle.alignment.live_view is not None
                else None
            ),
        )
        case.record("created")
        case.record("snapshot_admitted", bundle.bundle_id)
        return case

    def record(self, stage: str, detail: str | None = None) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError(
                "cannot append an event after a cognitive case is closed"
            )
        self.events.append(CaseEvent(stage, datetime.now(timezone.utc), detail))

    @property
    def snapshot_bundle(self) -> "SnapshotBundle | None":
        return self._snapshot_bundle

    @property
    def evaluation_context(self) -> "GraphixEvaluationContext":
        if self._evaluation_context is None:
            raise RuntimeError("case has no admitted evaluation context")
        return self._evaluation_context

    def bind_snapshot_bundle(self, bundle: "SnapshotBundle") -> None:
        """Migration adapter for pre-admission callers; never used in production.

        Remove when direct callers construct cases through EpisodeAdmissionService.
        """
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot bind a snapshot bundle to a closed case")
        if self._snapshot_bundle is not None:
            raise RuntimeError("snapshot bundle already bound")
        if bundle.episode_id != self.case_id:
            raise ValueError("snapshot bundle/case identity mismatch")
        if self.episode is None:
            raise RuntimeError("authoritative episode is unavailable")
        self.episode = self.episode.bind_snapshot_bundle_for_migration(
            bundle.bundle_ref()
        )
        self._snapshot_bundle = bundle
        self.state_snapshot_id = bundle.digest
        self.record("snapshot_admitted", bundle.bundle_id)

    def release_snapshot_bundle(self) -> None:
        if self._snapshot_bundle is not None:
            self._snapshot_bundle.close()

    def record_finalization(self, decision: str) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot finalize a closed cognitive case")
        if self.finalization_status is not None:
            raise RuntimeError("response finalized more than once")
        self.finalization_status = decision
        self.record("finalized", decision)

    def mirror_terminal(
        self,
        status: CognitiveCaseStatus,
        failure_kind: str | None = None,
    ) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("terminal episode projected more than once")
        if self.episode is None or not self.episode.state.is_terminal:
            raise RuntimeError("durable episode must terminalize first")
        expected = _TERMINAL_EPISODE_STATE[status]
        if self.episode.state is not expected:
            raise RuntimeError("case and episode terminal states disagree")
        self.failure_kind = failure_kind
        self.record("terminal", status.value)
        self.terminal_status = status

    def project_episode(self, episode: CognitiveEpisode) -> None:
        """Project an already committed transaction-service result."""
        if episode.episode_id != self.case_id:
            raise ValueError("episode projection identity mismatch")
        self.episode = episode
        self.record("episode_transition", episode.state.value)


def episode_from_case(case: CognitiveCase) -> CognitiveEpisode:
    """Compatibility adapter exposing the authoritative episode for a case."""
    if case.episode is not None:
        return case.episode
    return CognitiveEpisode.create(
        actor=ActorBinding(
            actor_id="legacy-unverified",
            principal_digest=sha256(b"legacy-unverified").hexdigest(),
            authority="LegacyCompatibility",
        ),
        request_id=case.request_id,
        input_digest=case.input_hash,
        conversation_id=case.conversation_id,
        episode_id=case.case_id,
    )
