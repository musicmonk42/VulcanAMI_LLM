"""Principal- and digest-bound constitutional episode transactions.

This service is the sole production authority promoter.  Cognitive organs may
construct command inputs, but cannot advance an episode or manufacture a grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from vulcan.constitution.primitives import AuthorityLevel

from .authority import AuthorityError, AuthorityGrant, EvidenceRecord, Operation
from .capability_tokens import CapabilityToken, CapabilityTokenIssuer
from .episode import ArtifactRef, CognitiveEpisode, canonical_digest
from .episode_store import EpisodeConflict, EpisodeStore
from .principals import Principal, digest
from .state_machine import EpisodeState


def _hex(value: str, name: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AuthorityError(f"{name} must be a lowercase sha256 digest")


@dataclass(frozen=True)
class CommandAuthority:
    principal: Principal
    grant: AuthorityGrant
    evidence: EvidenceRecord
    validation_digest: str
    policy_digest: str
    snapshot_digest: str
    expected_prior_episode_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal, Principal):
            raise AuthorityError("typed principal identity is required")
        if not isinstance(self.grant, AuthorityGrant):
            raise AuthorityError("typed current authority grant is required")
        if not isinstance(self.evidence, EvidenceRecord):
            raise AuthorityError("typed authority evidence is required")
        for name in (
            "validation_digest",
            "policy_digest",
            "snapshot_digest",
            "expected_prior_episode_digest",
        ):
            _hex(getattr(self, name), name)
        if (
            self.evidence.validator_principal_digest != self.principal.identity_digest
            or self.evidence.validation_digest != self.validation_digest
            or self.evidence.policy_digest != self.policy_digest
            or self.grant.evidence_digest != digest(self.evidence.to_json())
        ):
            raise AuthorityError("authority grant evidence binding mismatch")


@dataclass(frozen=True)
class PublicationAuthorization:
    committed_epistemic_head: str
    alignment_decision_digest: str
    policy_digest: str
    policy_revision: int
    finalizer_decision_digest: str
    rendered_text_digest: str
    privacy_context_digest: str
    consent_context_digest: str
    principal_digest: str
    release_digest: str

    def __post_init__(self) -> None:
        for name in (
            "committed_epistemic_head",
            "alignment_decision_digest",
            "policy_digest",
            "finalizer_decision_digest",
            "rendered_text_digest",
            "privacy_context_digest",
            "consent_context_digest",
            "principal_digest",
            "release_digest",
        ):
            _hex(getattr(self, name), name)
        if self.policy_revision < 0:
            raise ValueError("policy revision cannot be negative")

    @property
    def digest(self) -> str:
        return canonical_digest(self.__dict__)

    def ref(self, episode_id: str) -> ArtifactRef:
        return ArtifactRef(
            f"publication-authorization:{episode_id}",
            self.digest,
            "response-publication-authorization.v1",
        )


@dataclass(frozen=True)
class EffectAuthorization:
    intent_digest: str
    resource_digest: str
    capability_digest: str
    policy_digest: str
    snapshot_digest: str
    principal_digest: str
    release_digest: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _hex(getattr(self, name), name)

    @property
    def digest(self) -> str:
        return canonical_digest(self.__dict__)

    def ref(self, episode_id: str) -> ArtifactRef:
        return ArtifactRef(
            f"effect-authorization:{episode_id}",
            self.digest,
            "effect-authorization.v1",
        )


class TerminalOutcome(str, Enum):
    ABSTENTION = "abstention"
    BLOCK = "block"
    FAILURE = "failure"
    CANCELLATION = "cancellation"


class ConstitutionalTransactionService:
    """Validate a command against the durable head and advance it by CAS."""

    def __init__(self, store: EpisodeStore) -> None:
        if not isinstance(store, EpisodeStore):
            raise TypeError("a durable EpisodeStore is required")
        self._store = store

    def _advance(
        self,
        episode_id: str,
        auth: CommandAuthority,
        target: EpisodeState,
        *,
        reason: str,
        minimum: AuthorityLevel,
        **updates: object,
    ) -> CognitiveEpisode:
        if not auth.principal.is_kernel:
            raise AuthorityError("only SYSTEM_KERNEL may promote episode authority")
        if auth.grant.principal_digest != auth.principal.identity_digest:
            raise AuthorityError("authority grant principal mismatch")
        if not auth.grant.level.dominates(minimum):
            raise AuthorityError("current authority grant is insufficient")
        head = self._store.load(episode_id)
        if head.digest != auth.expected_prior_episode_digest:
            raise AuthorityError("expected prior episode digest is stale")
        if (
            head.snapshot_bundle is None
            or head.snapshot_bundle.state_digest != auth.snapshot_digest
        ):
            raise AuthorityError("command snapshot digest is not the admitted snapshot")
        evidence = ArtifactRef(
            f"command-evidence:{head.episode_id}:{len(head.transitions)}",
            canonical_digest(
                {
                    "grant": auth.grant.evidence_digest,
                    "policy": auth.policy_digest,
                    "principal": auth.principal.identity_digest,
                    "release": auth.principal.release_digest,
                    "snapshot": auth.snapshot_digest,
                    "validation": auth.validation_digest,
                }
            ),
            "constitutional-command-evidence.v1",
        )
        policy_reference = ArtifactRef(
            f"policy-reference:{head.episode_id}:{len(head.transitions)}",
            auth.policy_digest,
            "constitutional-policy-reference.v1",
        )
        successor = head.transition(
            target,
            reason=reason,
            authority=auth.principal.identity_digest,
            snapshot_ids=(auth.snapshot_digest,),
            evidence_refs=(evidence, policy_reference),
            **updates,
        )
        try:
            return self._store.advance(episode_id, head.digest, successor)
        except EpisodeConflict as exc:
            raise AuthorityError("episode head compare-and-swap failed") from exc
        except BaseException:
            # A commit may succeed and its acknowledgement may be lost. Re-read
            # before propagating so callers never terminalize from a stale head.
            try:
                committed = self._store.load(episode_id)
            except BaseException:
                raise
            if committed.digest == successor.digest:
                return committed
            raise

    def record_interpretation(
        self, episode_id: str, auth: CommandAuthority, interpretation: Mapping[str, str]
    ) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.INTERPRETED,
            reason="interpretation recorded",
            minimum=AuthorityLevel.VALIDATED_CANDIDATE,
            interpretation=interpretation,
        )

    def ground_candidate(
        self, episode_id: str, auth: CommandAuthority
    ) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.GROUNDED,
            reason="candidate grounded",
            minimum=AuthorityLevel.VALIDATED_CANDIDATE,
        )

    def open_deliberation(
        self, episode_id: str, auth: CommandAuthority, plans: Sequence[ArtifactRef] = ()
    ) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.DELIBERATING,
            reason="deliberation opened",
            minimum=AuthorityLevel.VALIDATED_CANDIDATE,
            candidate_plans=plans,
        )

    def commit_epistemic_artifact(
        self,
        episode_id: str,
        auth: CommandAuthority,
        *,
        claims: Sequence[ArtifactRef],
        evidence: Sequence[ArtifactRef],
        derivations: Sequence[ArtifactRef],
    ) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.EPISTEMICALLY_COMMITTED,
            reason="epistemic artifact committed",
            minimum=AuthorityLevel.COMMITTED_BELIEF,
            claims=claims,
            evidence=evidence,
            derivations=derivations,
        )

    def authorize_response_publication(
        self,
        episode_id: str,
        auth: CommandAuthority,
        *,
        authorization: PublicationAuthorization,
        response: ArtifactRef,
    ) -> CognitiveEpisode:
        if (
            authorization.principal_digest != auth.principal.identity_digest
            or authorization.release_digest != auth.principal.release_digest
        ):
            raise AuthorityError(
                "publication authorization principal or release mismatch"
            )
        if authorization.policy_digest != auth.policy_digest:
            raise AuthorityError("publication authorization policy mismatch")
        return self._advance(
            episode_id,
            auth,
            EpisodeState.NORMATIVELY_AUTHORIZED,
            reason="response publication authorized",
            minimum=AuthorityLevel.AUTHORIZED_PLAN,
            authorization=authorization.ref(episode_id),
            response=response,
        )

    def authorize_effect(
        self,
        episode_id: str,
        auth: CommandAuthority,
        authorization: EffectAuthorization,
    ) -> CognitiveEpisode:
        if not isinstance(authorization, EffectAuthorization):
            raise AuthorityError("typed effect authorization required")
        if (
            authorization.principal_digest != auth.principal.identity_digest
            or authorization.release_digest != auth.principal.release_digest
            or authorization.policy_digest != auth.policy_digest
            or authorization.snapshot_digest != auth.snapshot_digest
        ):
            raise AuthorityError("effect authorization context mismatch")
        return self._advance(
            episode_id,
            auth,
            EpisodeState.NORMATIVELY_AUTHORIZED,
            reason="effect authorized",
            minimum=AuthorityLevel.AUTHORIZED_PLAN,
            authorization=authorization.ref(episode_id),
        )

    def record_execution(
        self,
        episode_id: str,
        auth: CommandAuthority,
        receipt: ArtifactRef,
        *,
        authorization: EffectAuthorization,
        capability: CapabilityToken,
        issuer: CapabilityTokenIssuer,
        resource_digest: str,
        clock,
    ) -> CognitiveEpisode:
        if receipt.kind != "execution-receipt.v1":
            raise AuthorityError("typed execution receipt required")
        if not isinstance(issuer, CapabilityTokenIssuer):
            raise AuthorityError("typed capability issuer required")
        head = self._store.load(episode_id)
        if (
            not isinstance(capability, CapabilityToken)
            or capability.token_digest != authorization.capability_digest
            or head.authorization != authorization.ref(episode_id)
        ):
            raise AuthorityError("execution capability is not bound to authorization")
        issuer.consume(
            token=capability,
            principal=auth.principal,
            operation=Operation.EXECUTE_EFFECT,
            episode_id=episode_id,
            resource_digest=resource_digest,
            now=clock(),
        )
        return self._advance(
            episode_id,
            auth,
            EpisodeState.EXECUTED,
            reason="effect execution recorded",
            minimum=AuthorityLevel.EXECUTED_EFFECT,
            effects=(receipt,),
        )

    def record_observation(
        self, episode_id: str, auth: CommandAuthority, observation: ArtifactRef
    ) -> CognitiveEpisode:
        if observation.kind != "effect-observation.v1":
            raise AuthorityError("typed effect observation required")
        return self._advance(
            episode_id,
            auth,
            EpisodeState.OBSERVED,
            reason="effect observation recorded",
            minimum=AuthorityLevel.EXECUTED_EFFECT,
            evidence=(observation,),
        )

    def communicate(self, episode_id: str, auth: CommandAuthority) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.COMMUNICATED,
            reason="authorized response communicated",
            minimum=AuthorityLevel.AUTHORIZED_PLAN,
        )

    def consolidate(
        self, episode_id: str, auth: CommandAuthority, artifact: ArtifactRef
    ) -> CognitiveEpisode:
        return self._advance(
            episode_id,
            auth,
            EpisodeState.CONSOLIDATED,
            reason="episode consolidated",
            minimum=AuthorityLevel.COMMITTED_BELIEF,
            consolidation_refs=(artifact,),
        )

    def terminalize(
        self,
        episode_id: str,
        auth: CommandAuthority,
        outcome: TerminalOutcome,
        *,
        response: ArtifactRef | None = None,
        publication: PublicationAuthorization | None = None,
    ) -> CognitiveEpisode:
        if (response is None) is not (publication is None):
            raise AuthorityError(
                "terminal publication requires both response and authorization"
            )
        if publication is not None and (
            publication.principal_digest != auth.principal.identity_digest
            or publication.release_digest != auth.principal.release_digest
            or publication.policy_digest != auth.policy_digest
        ):
            raise AuthorityError("terminal publication authorization mismatch")
        targets = {
            TerminalOutcome.ABSTENTION: EpisodeState.ABSTAINED,
            TerminalOutcome.BLOCK: EpisodeState.BLOCKED,
            TerminalOutcome.FAILURE: EpisodeState.FAILED,
            TerminalOutcome.CANCELLATION: EpisodeState.CANCELLED,
        }
        return self._advance(
            episode_id,
            auth,
            targets[outcome],
            reason=outcome.value,
            minimum=AuthorityLevel.VALIDATED_CANDIDATE,
            response=response,
            authorization=(
                publication.ref(episode_id) if publication is not None else None
            ),
        )
