"""Principal- and digest-bound constitutional episode transactions.

This service is the sole production authority promoter.  Cognitive organs may
construct command inputs, but cannot advance an episode or manufacture a grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from vulcan.constitution.primitives import AuthorityLevel
from vulcan.graphix.epistemic import EpistemicCommit, commit_to_dict

from .authority import AuthorityError, AuthorityGrant, EvidenceRecord, Operation
from .capability_tokens import CapabilityToken, CapabilityTokenIssuer
from .episode import ArtifactRef, CognitiveEpisode, canonical_digest
from .episode_store import EpisodeConflict, EpisodeStore
from .epistemic_store import EpistemicStore
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

    def __init__(
        self, store: EpisodeStore, epistemic_store: EpistemicStore | None = None
    ) -> None:
        if not isinstance(store, EpisodeStore):
            raise TypeError("a durable EpisodeStore is required")
        self._store = store
        self._epistemic_store = epistemic_store

    def epistemic_head(self, episode_id: str) -> EpistemicCommit | None:
        if self._epistemic_store is None:
            raise AuthorityError("durable epistemic authority is not bound")
        return self._epistemic_store.head(episode_id)

    def episode_head(self, episode_id: str) -> CognitiveEpisode:
        """Return the verified durable episode head used by projection readers."""
        return self._store.load(episode_id)

    def commit_epistemic_candidate(
        self,
        episode_id: str,
        auth: CommandAuthority,
        candidate: EpistemicCommit,
    ) -> CognitiveEpisode:
        """Commit Graphix Epistemic bytes before projecting them into the episode."""
        if self._epistemic_store is None:
            raise AuthorityError("durable epistemic authority is not bound")
        if (
            not auth.principal.is_kernel
            or auth.grant.principal_digest != auth.principal.identity_digest
            or not auth.grant.level.dominates(AuthorityLevel.COMMITTED_BELIEF)
        ):
            raise AuthorityError("only SYSTEM_KERNEL may commit epistemic candidates")
        head = self._store.load(episode_id)
        snapshot = "sha256:" + auth.snapshot_digest
        if (
            candidate.episode_id != episode_id
            or candidate.case_id != episode_id
            or candidate.snapshot_digest != snapshot
            or candidate.authority_principal_id
            != f"principal:{auth.principal.identity_digest[:32]}"
            or candidate.authority_release_digest
            != f"sha256:{auth.principal.release_digest}"
            or candidate.validation_digest != f"sha256:{auth.validation_digest}"
            or candidate.policy_digest != f"sha256:{auth.policy_digest}"
            or candidate.authority_evidence_digest
            != f"sha256:{auth.grant.evidence_digest}"
            or head.snapshot_bundle is None
            or head.snapshot_bundle.state_digest != auth.snapshot_digest
            or head.digest != auth.expected_prior_episode_digest
        ):
            raise AuthorityError("epistemic candidate authority binding mismatch")
        durable_head = self._epistemic_store.head(episode_id)
        if durable_head is not None and self._same_epistemic_payload(
            durable_head, candidate
        ):
            # Recovery for the DB-first crash window: epistemic persistence may
            # have succeeded before its episode projection was acknowledged.
            committed = durable_head
        else:
            committed = self._epistemic_store.append(
                candidate, candidate.prior_commit_digest
            )
        claims = tuple(
            ArtifactRef(
                item.claim_id,
                committed.commit_digest.removeprefix("sha256:"),
                "graphix-epistemic-claim.v1",
            )
            for item in committed.claims
        )
        evidence = tuple(
            ArtifactRef(
                item.evidence_id,
                item.content_digest.removeprefix("sha256:"),
                "graphix-epistemic-evidence.v1",
            )
            for item in committed.evidence
        )
        derivations = tuple(
            ArtifactRef(
                item.derivation_id,
                committed.commit_digest.removeprefix("sha256:"),
                "graphix-epistemic-derivation.v1",
            )
            for item in committed.derivations
        )
        return self._advance(
            episode_id,
            auth,
            EpisodeState.EPISTEMICALLY_COMMITTED,
            reason="durable Graphix Epistemic head committed",
            minimum=AuthorityLevel.COMMITTED_BELIEF,
            claims=claims,
            evidence=evidence,
            derivations=derivations,
        )

    @staticmethod
    def _same_epistemic_payload(
        committed: EpistemicCommit, candidate: EpistemicCommit
    ) -> bool:
        def stable_document(value: EpistemicCommit) -> dict[str, object]:
            document = commit_to_dict(value)
            for field in (
                "authority_evidence_digest",
                "commit_digest",
                "commit_id",
                "committed_at",
                "prior_commit_digest",
            ):
                document.pop(field)
            return document

        return stable_document(committed) == stable_document(candidate)

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
        if self._epistemic_store is not None:
            raise AuthorityError(
                "compatibility artifact commitment is disabled when durable "
                "epistemic authority is bound"
            )
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
        if response.digest != authorization.rendered_text_digest:
            raise AuthorityError(
                "publication response is not the authorized exact text"
            )
        if self._epistemic_store is not None:
            epistemic_head = self.epistemic_head(episode_id)
            if (
                epistemic_head is None
                or authorization.committed_epistemic_head
                != epistemic_head.commit_digest.removeprefix("sha256:")
            ):
                raise AuthorityError("publication is not bound to the epistemic head")
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
        head = self._store.load(episode_id)
        if head.authorization is None or head.response is None:
            raise AuthorityError("communication requires publication evidence")
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
        if (
            publication is not None
            and response is not None
            and response.digest != publication.rendered_text_digest
        ):
            raise AuthorityError("terminal response is not the authorized exact text")
        if publication is not None and self._epistemic_store is not None:
            epistemic_head = self.epistemic_head(episode_id)
            if (
                epistemic_head is None
                or publication.committed_epistemic_head
                != epistemic_head.commit_digest.removeprefix("sha256:")
            ):
                raise AuthorityError(
                    "terminal publication is not bound to the epistemic head"
                )
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
