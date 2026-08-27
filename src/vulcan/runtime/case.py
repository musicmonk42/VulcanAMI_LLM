"""Request-scoped, privacy-preserving projection of a CognitiveEpisode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from vulcan.microkernel.snapshots import SnapshotBundle

    from .semantic import (
        AcceptedInterpretation,
        Claim,
        ClarificationRequest,
        Derivation,
        EvidenceArtifact,
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
    CognitiveCaseStatus.ABSTAINED: EpisodeState.ABSTAINED,
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

    The current semantic kernel still populates request-local Python objects. This
    class translates those objects into digest-bound episode artifacts at the
    existing ledger and close boundaries, so compatibility state no longer creates
    an independent lifecycle authority.
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
    _snapshot_bundle: "SnapshotBundle | None" = field(default=None, repr=False)

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
            actor_id="canonical-runtime",
            principal_digest=sha256(request_id.encode("utf-8")).hexdigest(),
            authority="CognitiveKernel",
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

    @property
    def snapshot_bundle(self) -> "SnapshotBundle | None":
        return self._snapshot_bundle

    def bind_snapshot_bundle(self, bundle: "SnapshotBundle") -> None:
        """Complete admission by binding one real state bundle exactly once."""
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot bind a snapshot bundle to a closed case")
        if self._snapshot_bundle is not None:
            raise RuntimeError("snapshot bundle already bound")
        if bundle.episode_id != self.case_id:
            raise ValueError("snapshot bundle/case identity mismatch")
        if self.episode is None:
            raise RuntimeError("authoritative episode is unavailable")
        self.episode = self.episode.bind_snapshot_bundle(bundle.bundle_ref())
        self._snapshot_bundle = bundle
        self.state_snapshot_id = bundle.digest
        self.record("snapshot_admitted", bundle.bundle_id)

    def release_snapshot_bundle(self) -> None:
        if self._snapshot_bundle is not None:
            self._snapshot_bundle.close()

    def append_ledger(
        self,
        *,
        claim: "Claim",
        derivation: "Derivation",
        evidence: tuple["EvidenceArtifact", ...] = (),
    ) -> None:
        """Validate compatibility objects and bind them into the episode."""
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot mutate a closed case ledger")
        from .semantic import validate_ledger

        proposed_evidence = (*self._evidence, *evidence)
        proposed_derivations = (*self._derivations, derivation)
        proposed_claims = (*self._claims, claim)
        validate_ledger(
            proposed_evidence,
            proposed_derivations,
            proposed_claims,
            case_id=self.case_id,
        )
        self._evidence.extend(evidence)
        self._derivations.append(derivation)
        self._claims.append(claim)
        self._ensure_interpreted()
        if self.accepted_interpretation is not None:
            self._advance(EpisodeState.GROUNDED, "accepted interpretation grounded")
            self._advance(EpisodeState.DELIBERATING, "bounded plan deliberated")
            claims, derivations, evidence_refs = self._ledger_refs()
            self._advance(
                EpisodeState.EPISTEMICALLY_COMMITTED,
                "validated compatibility ledger committed to episode",
                claims=claims,
                derivations=derivations,
                evidence=evidence_refs,
                evidence_refs=evidence_refs,
            )

    def record_finalization(self, decision: str) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cannot finalize a closed cognitive case")
        if self.finalization_status is not None:
            raise RuntimeError("response finalized more than once")
        self.finalization_status = decision
        self.record("finalized", decision)

    def close(
        self,
        status: CognitiveCaseStatus,
        failure_kind: str | None = None,
    ) -> None:
        if self.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("cognitive case closed more than once")
        self._terminalize_episode(status, failure_kind or status.value)
        self.failure_kind = failure_kind
        self.record("terminal", status.value)
        self.terminal_status = status

    def _ensure_interpreted(self) -> None:
        if self.episode is None or self.episode.state is not EpisodeState.PERCEIVED:
            return
        if self.interpretation is None:
            return
        mapping = {
            "parser_identity": str(
                getattr(self.interpretation, "parser_identity", "unknown")
            ),
            "candidate_count": str(
                len(getattr(self.interpretation, "candidates", ()))
            ),
            "ontology_version": str(
                getattr(self.interpretation, "ontology_version", "unknown")
            ),
        }
        self._advance(
            EpisodeState.INTERPRETED,
            "validated semantic ingress",
            interpretation=mapping,
        )

    def _advance(
        self,
        target: EpisodeState,
        reason: str,
        **updates: Any,
    ) -> None:
        if self.episode is None:
            raise RuntimeError("authoritative episode is unavailable")
        if self.episode.state is target:
            return
        snapshot_ids = (self.state_snapshot_id,) if self.state_snapshot_id else ()
        self.episode = self.episode.transition(
            target,
            reason=reason,
            authority="CognitiveKernel",
            snapshot_ids=snapshot_ids,
            **updates,
        )
        self.record("episode_transition", target.value)

    def _ledger_refs(
        self,
    ) -> tuple[
        tuple[ArtifactRef, ...],
        tuple[ArtifactRef, ...],
        tuple[ArtifactRef, ...],
    ]:
        from .semantic import canonical_digest

        claims = tuple(
            ArtifactRef(claim.claim_id, canonical_digest(claim), "semantic-claim.v2")
            for claim in self.claims
        )
        derivations = tuple(
            ArtifactRef(
                derivation.derivation_id,
                canonical_digest(derivation),
                "semantic-derivation.v2",
            )
            for derivation in self.derivations
        )
        evidence = tuple(
            ArtifactRef(
                item.artifact_id,
                item.content_digest,
                "semantic-evidence.v2",
            )
            for item in self.evidence
        )
        return claims, derivations, evidence

    def _response_ref(self) -> ArtifactRef | None:
        from .semantic import canonical_digest

        if self.response_ir is None or self.render_artifact is None:
            return None
        return ArtifactRef(
            self.response_ir.response_id,
            str(
                getattr(
                    self.render_artifact,
                    "ir_digest",
                    canonical_digest(self.response_ir),
                )
            ),
            "response-ir.v3",
        )

    def _terminalize_episode(self, status: CognitiveCaseStatus, reason: str) -> None:
        if self.episode is None:
            raise RuntimeError("authoritative episode is unavailable")
        if self.episode.state.is_terminal:
            expected = _TERMINAL_EPISODE_STATE[status]
            if self.episode.state is not expected:
                raise RuntimeError("case and episode terminal states disagree")
            return

        self._ensure_interpreted()
        claims, derivations, evidence = self._ledger_refs()
        response = self._response_ref()

        if status is CognitiveCaseStatus.SUCCESS:
            if self.episode.state is not EpisodeState.EPISTEMICALLY_COMMITTED:
                raise RuntimeError("successful case lacks an epistemic episode commit")
            if response is None or self.finalization_status != "allow":
                raise RuntimeError("successful case lacks an allowed response artifact")
            from .semantic import canonical_digest

            authorization = ArtifactRef(
                f"authorization-{self.case_id}",
                canonical_digest(
                    {
                        "case_id": self.case_id,
                        "finalization": self.finalization_status,
                        "claim_digests": [ref.digest for ref in claims],
                    }
                ),
                "response-authorization.compat.v1",
            )
            self._advance(
                EpisodeState.NORMATIVELY_AUTHORIZED,
                "alignment and safety allowed response publication",
                authorization=authorization,
            )
            self._advance(
                EpisodeState.EXECUTED,
                "authorized response artifact executed",
                response=response,
                effects=(response,),
            )
            self._advance(
                EpisodeState.OBSERVED,
                "rendered response artifact observed and digest bound",
                evidence_refs=(response,),
            )
            self._advance(
                EpisodeState.COMMUNICATED,
                "response released to transport",
                response=response,
            )
            consolidation = ArtifactRef(
                f"consolidation-{self.case_id}",
                canonical_digest(
                    {
                        "episode": self.episode.digest,
                        "claims": [ref.digest for ref in claims],
                        "derivations": [ref.digest for ref in derivations],
                        "evidence": [ref.digest for ref in evidence],
                        "response": response.digest,
                    }
                ),
                "episode-consolidation.v1",
            )
            self._advance(
                EpisodeState.CONSOLIDATED,
                "authoritative episode consolidated",
                consolidation_refs=(consolidation,),
            )
            return

        target = _TERMINAL_EPISODE_STATE[status]
        existing_claims = {ref.artifact_id for ref in self.episode.claims}
        existing_derivations = {ref.artifact_id for ref in self.episode.derivations}
        existing_evidence = {ref.artifact_id for ref in self.episode.evidence}
        self._advance(
            target,
            reason,
            response=response,
            claims=tuple(ref for ref in claims if ref.artifact_id not in existing_claims),
            derivations=tuple(
                ref for ref in derivations if ref.artifact_id not in existing_derivations
            ),
            evidence=tuple(
                ref for ref in evidence if ref.artifact_id not in existing_evidence
            ),
            evidence_refs=evidence,
        )


def episode_from_case(case: CognitiveCase) -> CognitiveEpisode:
    """Compatibility adapter exposing the authoritative episode for a case."""
    if case.episode is not None:
        return case.episode
    return CognitiveEpisode.create(
        actor=ActorBinding(
            actor_id="canonical-runtime",
            principal_digest=sha256(case.request_id.encode("utf-8")).hexdigest(),
            authority="CognitiveKernel",
        ),
        request_id=case.request_id,
        input_digest=case.input_hash,
        conversation_id=case.conversation_id,
        episode_id=case.case_id,
    )
