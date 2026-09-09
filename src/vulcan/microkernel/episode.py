"""Immutable, versioned CognitiveEpisode aggregate.

The episode is the authoritative request-scoped cognitive contract. Raw input is
accepted only at construction time for digesting and is never retained in the
serialized aggregate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence
from uuid import uuid4

from vulcan.constitution.primitives import (
    ArtifactId,
    Digest,
    EpisodeId,
    PrincipalId,
    SnapshotId,
)
from vulcan.constitution.primitives import canonical_json as _canonical_json

from .state_machine import EpisodeState, EpisodeTransitionError, ensure_transition

SCHEMA_VERSION = "cognitive-episode.v1"
GENESIS_DIGEST = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")


class Clock(Protocol):
    def __call__(self) -> datetime: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _digest_bytes(value: bytes) -> str:
    return Digest.of_bytes(value).hex


def digest_text(text: str) -> str:
    return _digest_bytes(text.encode("utf-8"))


def _canon(value: object) -> str:
    return _canonical_json(value).decode("utf-8")


def canonical_digest(value: object) -> str:
    return _digest_bytes(_canon(value).encode("utf-8"))


def _freeze_mapping(value: Mapping[str, str] | None) -> Mapping[str, str]:
    return MappingProxyType(dict(value or {}))


def _episode_id(value: str) -> str:
    return str(EpisodeId(value))


@dataclass(frozen=True)
class ActorBinding:
    actor_id: str
    principal_digest: str
    authority: str

    def __post_init__(self) -> None:
        PrincipalId(self.actor_id)
        Digest.from_legacy_hex(self.principal_digest)
        if not isinstance(self.authority, str) or not _ID.fullmatch(self.authority):
            raise ValueError("invalid actor authority")

    def to_json(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "authority": self.authority,
            "principal_digest": self.principal_digest,
        }


@dataclass(frozen=True)
class RequestBinding:
    request_id: str
    input_digest: str
    projection_digest: str | None = None
    retention_policy: str = (
        "raw-request-working-memory-only; "
        "durable-episode-digests-and-approved-projections"
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or any(ord(character) < 32 for character in self.request_id)
        ):
            raise ValueError("invalid request identity")
        Digest.from_legacy_hex(self.input_digest)
        if self.projection_digest is not None:
            Digest.from_legacy_hex(self.projection_digest)
        if not isinstance(self.retention_policy, str) or not self.retention_policy:
            raise ValueError("retention policy is required")

    def to_json(self) -> dict[str, str | None]:
        return {
            "input_digest": self.input_digest,
            "projection_digest": self.projection_digest,
            "request_id": self.request_id,
            "retention_policy": self.retention_policy,
        }


@dataclass(frozen=True)
class SnapshotBundleRef:
    bundle_id: str
    state_digest: str

    def __post_init__(self) -> None:
        SnapshotId(self.bundle_id)
        Digest.from_legacy_hex(self.state_digest)

    def to_json(self) -> dict[str, str]:
        return {"bundle_id": self.bundle_id, "state_digest": self.state_digest}


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    digest: str

    def __post_init__(self) -> None:
        EpisodeId(self.episode_id)
        Digest.from_legacy_hex(self.digest)

    def to_json(self) -> dict[str, str]:
        return {"digest": self.digest, "episode_id": self.episode_id}


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    digest: str
    kind: str

    def __post_init__(self) -> None:
        ArtifactId(self.artifact_id)
        Digest.from_legacy_hex(self.digest)
        if not isinstance(self.kind, str) or not _ID.fullmatch(self.kind):
            raise ValueError("invalid artifact kind")

    def to_json(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class SnapshotRebaseCommand:
    """Typed evidence for a microkernel-authorized snapshot rebase."""

    old_bundle: SnapshotBundleRef
    new_bundle: SnapshotBundleRef
    reason_code: str
    authority_evidence: ArtifactRef
    prior_episode_digest: str

    def __post_init__(self) -> None:
        if self.old_bundle == self.new_bundle:
            raise ValueError("snapshot rebase must change the bundle")
        if not isinstance(self.reason_code, str) or not _ID.fullmatch(self.reason_code):
            raise ValueError("invalid snapshot rebase reason code")
        Digest.from_legacy_hex(self.prior_episode_digest)
        if self.authority_evidence.kind != "snapshot-rebase-authorization.v1":
            raise ValueError("invalid snapshot rebase authority evidence kind")

    def authorizes(
        self,
        current: SnapshotBundleRef,
        snapshot_ids: Sequence[str],
        episode_digest: str,
    ) -> bool:
        return (
            self.old_bundle == current
            and self.prior_episode_digest == episode_digest
            and self.new_bundle.bundle_id in snapshot_ids
            and self.new_bundle.state_digest in snapshot_ids
        )


@dataclass(frozen=True)
class TransitionEvent:
    event_id: str
    from_state: EpisodeState
    to_state: EpisodeState
    at: datetime
    reason: str
    authority: str
    prior_digest: str
    snapshot_ids: tuple[str, ...] = ()
    evidence_refs: tuple[ArtifactRef, ...] = ()
    event_digest: str = field(init=False)

    def __post_init__(self) -> None:
        ArtifactId(self.event_id)
        Digest.from_legacy_hex(self.prior_digest)
        if not self.reason or not self.authority:
            raise ValueError("transition reason and authority are required")
        object.__setattr__(self, "snapshot_ids", tuple(self.snapshot_ids))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(
            self,
            "event_digest",
            canonical_digest(self.to_json(include_digest=False)),
        )

    def to_json(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "at": self.at.astimezone(timezone.utc).isoformat(),
            "authority": self.authority,
            "event_id": self.event_id,
            "evidence_refs": [ref.to_json() for ref in self.evidence_refs],
            "from_state": self.from_state.value,
            "prior_digest": self.prior_digest,
            "reason": self.reason,
            "snapshot_ids": list(self.snapshot_ids),
            "to_state": self.to_state.value,
        }
        if include_digest:
            payload["event_digest"] = self.event_digest
        return payload


@dataclass(frozen=True)
class CognitiveEpisode:
    episode_id: str
    actor: ActorBinding
    request: RequestBinding
    state: EpisodeState = EpisodeState.PERCEIVED
    schema_version: str = SCHEMA_VERSION
    conversation_id: str | None = None
    parent: EpisodeRef | None = None
    lineage_head: ArtifactRef | None = None
    snapshot_bundle: SnapshotBundleRef | None = None
    interpretation: Mapping[str, str] = field(default_factory=dict)
    claims: tuple[ArtifactRef, ...] = ()
    evidence: tuple[ArtifactRef, ...] = ()
    derivations: tuple[ArtifactRef, ...] = ()
    candidate_plans: tuple[ArtifactRef, ...] = ()
    authorization: ArtifactRef | None = None
    effects: tuple[ArtifactRef, ...] = ()
    response: ArtifactRef | None = None
    consolidation_refs: tuple[ArtifactRef, ...] = ()
    transitions: tuple[TransitionEvent, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _episode_id(self.episode_id)
        object.__setattr__(self, "interpretation", _freeze_mapping(self.interpretation))
        for name in (
            "claims",
            "evidence",
            "derivations",
            "candidate_plans",
            "effects",
            "consolidation_refs",
            "transitions",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        reference_groups = (
            "claims",
            "evidence",
            "derivations",
            "candidate_plans",
            "effects",
            "consolidation_refs",
        )
        for name in reference_groups:
            refs = getattr(self, name)
            identities = [ref.artifact_id for ref in refs]
            if len(identities) != len(set(identities)):
                raise ValueError(f"duplicate artifact reference in {name}")
        semantic_refs = (
            *self.claims,
            *self.evidence,
            *self.derivations,
            *self.candidate_plans,
            *self.consolidation_refs,
        )
        semantic_ids = [ref.artifact_id for ref in semantic_refs]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("artifact identity reused across semantic roles")
        if self.authorization is not None and self.authorization.kind not in {
            "response-publication-authorization.v1",
            "effect-authorization.v1",
            "response-authorization.compat.v1",
        }:
            raise ValueError("authorization artifact has the wrong kind")
        if (
            self.lineage_head is not None
            and self.lineage_head.kind != "lineage-head.v1"
        ):
            raise ValueError("lineage binding has the wrong kind")
        if self.response is not None and self.response.kind not in {
            "published-response.v1",
            "response-ir.v3",  # persisted pre-projection compatibility
        }:
            raise ValueError("response artifact has the wrong kind")
        if any(
            ref.kind != "execution-receipt.v1"
            and not (
                ref.kind == "response-ir.v3"
                and self.authorization is not None
                and self.authorization.kind == "response-authorization.compat.v1"
            )
            for ref in self.effects
        ):
            raise ValueError("effect has the wrong kind")
        if any(
            ref.kind != "episode-consolidation.v1" for ref in self.consolidation_refs
        ):
            raise ValueError("consolidation artifact has the wrong kind")
        if (
            self.state
            in {
                EpisodeState.NORMATIVELY_AUTHORIZED,
                EpisodeState.EXECUTED,
                EpisodeState.OBSERVED,
                EpisodeState.COMMUNICATED,
                EpisodeState.CONSOLIDATED,
            }
            and self.authorization is None
        ):
            raise ValueError("authorized success state requires authorization evidence")
        if self.state is EpisodeState.COMMUNICATED and (
            self.response is None
            or self.authorization is None
            or self.authorization.kind
            not in {
                "response-publication-authorization.v1",
                "response-authorization.compat.v1",
            }
        ):
            raise ValueError(
                "communication requires response publication authorization"
            )
        if (
            self.state is EpisodeState.CONSOLIDATED
            and self.authorization is not None
            and self.authorization.kind == "response-publication-authorization.v1"
            and self.response is None
        ):
            raise ValueError("published consolidation requires a response artifact")
        if self.state in {EpisodeState.EXECUTED, EpisodeState.OBSERVED} and (
            self.authorization is None
            or self.authorization.kind
            not in {
                "effect-authorization.v1",
                "response-authorization.compat.v1",
            }
            or not self.effects
        ):
            raise ValueError("executed effect requires authorization and a receipt")
        if self.state is EpisodeState.CONSOLIDATED and not self.consolidation_refs:
            raise ValueError("consolidated state requires a consolidation artifact")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(self.to_json(include_digest=False)),
        )

    @classmethod
    def create(
        cls,
        *,
        actor: ActorBinding,
        request_id: str,
        input_digest: str | None = None,
        raw_request: bytes | str | None = None,
        conversation_id: str | None = None,
        parent: EpisodeRef | None = None,
        lineage_head: ArtifactRef | None = None,
        snapshot_bundle: SnapshotBundleRef | None = None,
        projection_digest: str | None = None,
        episode_id: str | None = None,
        clock: Clock = utc_now,
    ) -> "CognitiveEpisode":
        if input_digest is None:
            if raw_request is None:
                raise ValueError("input_digest or raw_request is required")
            raw = (
                raw_request.encode("utf-8")
                if isinstance(raw_request, str)
                else raw_request
            )
            input_digest = _digest_bytes(raw)
        resolved_episode_id = _episode_id(episode_id or str(uuid4()))
        episode = cls(
            episode_id=resolved_episode_id,
            actor=actor,
            request=RequestBinding(request_id, input_digest, projection_digest),
            conversation_id=conversation_id,
            parent=parent,
            lineage_head=lineage_head,
            snapshot_bundle=snapshot_bundle,
        )
        return episode._append_event(
            EpisodeState.PERCEIVED,
            reason="created",
            authority=actor.authority,
            clock=clock,
            snapshot_ids=(
                (snapshot_bundle.bundle_id, snapshot_bundle.state_digest)
                if snapshot_bundle is not None
                else ()
            ),
        )

    def bind_snapshot_bundle_for_migration(
        self, snapshot_bundle: SnapshotBundleRef
    ) -> "CognitiveEpisode":
        """Legacy pre-admission adapter; remove with all pre-#1080 callers.

        Runtime admission creates the episode identifier before all mutable state
        authorities can be leased. Binding is therefore allowed exactly once while
        the episode is still at its genesis ``PERCEIVED`` state. It is not a second
        authority transition; it completes admission and changes the episode digest.
        """
        if self.state is not EpisodeState.PERCEIVED or len(self.transitions) != 1:
            raise EpisodeTransitionError(
                "snapshot bundle must be bound before the first semantic transition"
            )
        if self.snapshot_bundle is not None:
            raise EpisodeTransitionError("snapshot bundle already bound")
        return replace(self, snapshot_bundle=snapshot_bundle)

    # Compatibility alias. Production composition never calls this method.
    bind_snapshot_bundle = bind_snapshot_bundle_for_migration

    def transition(
        self,
        target: EpisodeState,
        *,
        reason: str,
        authority: str,
        clock: Clock = utc_now,
        snapshot_ids: Sequence[str] = (),
        evidence_refs: Sequence[ArtifactRef] = (),
        interpretation: Mapping[str, str] | None = None,
        claims: Sequence[ArtifactRef] = (),
        evidence: Sequence[ArtifactRef] = (),
        derivations: Sequence[ArtifactRef] = (),
        candidate_plans: Sequence[ArtifactRef] = (),
        authorization: ArtifactRef | None = None,
        effects: Sequence[ArtifactRef] = (),
        response: ArtifactRef | None = None,
        consolidation_refs: Sequence[ArtifactRef] = (),
        rebase: "SnapshotRebaseCommand | None" = None,
    ) -> "CognitiveEpisode":
        if not authority:
            raise EpisodeTransitionError("transition authority is required")
        if rebase is not None and (
            self.snapshot_bundle is None
            or not rebase.authorizes(self.snapshot_bundle, snapshot_ids, self.digest)
        ):
            raise EpisodeTransitionError("invalid typed snapshot rebase")
        if self.snapshot_bundle is not None:
            allowed = {
                self.snapshot_bundle.bundle_id,
                self.snapshot_bundle.state_digest,
            }
            unknown = [sid for sid in snapshot_ids if sid not in allowed]
            if unknown and (
                rebase is None
                or not rebase.authorizes(
                    self.snapshot_bundle, snapshot_ids, self.digest
                )
            ):
                raise EpisodeTransitionError(
                    "mixed snapshot versions require a typed authorized rebase"
                )
        ensure_transition(self.state, target)
        prior_digest = self.digest
        updated = replace(
            self,
            state=target,
            snapshot_bundle=(
                rebase.new_bundle if rebase is not None else self.snapshot_bundle
            ),
            interpretation=(
                _freeze_mapping(interpretation)
                if interpretation is not None
                else self.interpretation
            ),
            claims=(*self.claims, *tuple(claims)),
            evidence=(*self.evidence, *tuple(evidence)),
            derivations=(*self.derivations, *tuple(derivations)),
            candidate_plans=(*self.candidate_plans, *tuple(candidate_plans)),
            authorization=(
                authorization if authorization is not None else self.authorization
            ),
            effects=(*self.effects, *tuple(effects)),
            response=response if response is not None else self.response,
            consolidation_refs=(
                *self.consolidation_refs,
                *tuple(consolidation_refs),
            ),
        )
        return updated._append_event(
            target,
            reason=reason,
            authority=authority,
            clock=clock,
            snapshot_ids=tuple(snapshot_ids),
            evidence_refs=(
                *tuple(evidence_refs),
                *((rebase.authority_evidence,) if rebase is not None else ()),
            ),
            prior_digest=prior_digest,
            from_state=self.state,
        )

    def _append_event(
        self,
        target: EpisodeState,
        *,
        reason: str,
        authority: str,
        clock: Clock,
        snapshot_ids: Sequence[str] = (),
        evidence_refs: Sequence[ArtifactRef] = (),
        prior_digest: str | None = None,
        from_state: EpisodeState | None = None,
    ) -> "CognitiveEpisode":
        event = TransitionEvent(
            str(uuid4()),
            from_state if from_state is not None else self.state,
            target,
            clock(),
            reason,
            authority,
            prior_digest or (self.digest if self.transitions else GENESIS_DIGEST),
            tuple(snapshot_ids),
            tuple(evidence_refs),
        )
        return replace(self, transitions=(*self.transitions, event))

    def to_json(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "actor": self.actor.to_json(),
            "authorization": (
                self.authorization.to_json() if self.authorization else None
            ),
            "candidate_plans": [x.to_json() for x in self.candidate_plans],
            "claims": [x.to_json() for x in self.claims],
            "consolidation_refs": [x.to_json() for x in self.consolidation_refs],
            "conversation_id": self.conversation_id,
            "derivations": [x.to_json() for x in self.derivations],
            "effects": [x.to_json() for x in self.effects],
            "episode_id": self.episode_id,
            "evidence": [x.to_json() for x in self.evidence],
            "interpretation": dict(self.interpretation),
            "lineage_head": self.lineage_head.to_json() if self.lineage_head else None,
            "parent": self.parent.to_json() if self.parent else None,
            "request": self.request.to_json(),
            "response": self.response.to_json() if self.response else None,
            "schema_version": self.schema_version,
            "snapshot_bundle": (
                self.snapshot_bundle.to_json() if self.snapshot_bundle else None
            ),
            "state": self.state.value,
            "transitions": [event.to_json() for event in self.transitions],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def canonical_json(self) -> str:
        return _canon(self.to_json())
