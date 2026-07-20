"""Immutable, versioned CognitiveEpisode aggregate.

The episode is the authoritative request-scoped cognitive contract. Raw input is
accepted only at construction time for digesting and is never retained in the
serialized aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence
from uuid import uuid4

from .state_machine import EpisodeState, EpisodeTransitionError, ensure_transition

SCHEMA_VERSION = "cognitive-episode.v1"
GENESIS_DIGEST = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Clock(Protocol):
    def __call__(self) -> datetime: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_text(text: str) -> str:
    return _digest_bytes(text.encode("utf-8"))


def _canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: object) -> str:
    return _digest_bytes(_canon(value).encode("utf-8"))


def _freeze_mapping(value: Mapping[str, str] | None) -> Mapping[str, str]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class ActorBinding:
    actor_id: str
    principal_digest: str
    authority: str

    def to_json(self) -> dict[str, str]:
        return {"actor_id": self.actor_id, "authority": self.authority, "principal_digest": self.principal_digest}


@dataclass(frozen=True)
class RequestBinding:
    request_id: str
    input_digest: str
    projection_digest: str | None = None
    retention_policy: str = "raw-request-working-memory-only; durable-episode-digests-and-approved-projections"

    def to_json(self) -> dict[str, str | None]:
        return {"input_digest": self.input_digest, "projection_digest": self.projection_digest, "request_id": self.request_id, "retention_policy": self.retention_policy}


@dataclass(frozen=True)
class SnapshotBundleRef:
    bundle_id: str
    state_digest: str

    def __post_init__(self) -> None:
        if not self.bundle_id or not _HEX64.fullmatch(self.state_digest):
            raise ValueError("validated snapshot bundle identity is required")

    def to_json(self) -> dict[str, str]:
        return {"bundle_id": self.bundle_id, "state_digest": self.state_digest}


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    digest: str

    def to_json(self) -> dict[str, str]:
        return {"digest": self.digest, "episode_id": self.episode_id}


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    digest: str
    kind: str

    def to_json(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "digest": self.digest, "kind": self.kind}


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
        object.__setattr__(self, "snapshot_ids", tuple(self.snapshot_ids))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "event_digest", canonical_digest(self.to_json(include_digest=False)))

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
        object.__setattr__(self, "interpretation", _freeze_mapping(self.interpretation))
        for name in ("claims", "evidence", "derivations", "candidate_plans", "effects", "consolidation_refs", "transitions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "digest", canonical_digest(self.to_json(include_digest=False)))

    @classmethod
    def create(cls, *, actor: ActorBinding, request_id: str, input_digest: str | None = None, raw_request: bytes | str | None = None,
               conversation_id: str | None = None, parent: EpisodeRef | None = None, snapshot_bundle: SnapshotBundleRef | None = None,
               projection_digest: str | None = None, clock: Clock = utc_now) -> "CognitiveEpisode":
        if input_digest is None:
            if raw_request is None:
                raise ValueError("input_digest or raw_request is required")
            raw = raw_request.encode("utf-8") if isinstance(raw_request, str) else raw_request
            input_digest = _digest_bytes(raw)
        episode = cls(episode_id=str(uuid4()), actor=actor, request=RequestBinding(request_id, input_digest, projection_digest), conversation_id=conversation_id, parent=parent, snapshot_bundle=snapshot_bundle)
        return episode._append_event(EpisodeState.PERCEIVED, reason="created", authority=actor.authority, clock=clock)

    def transition(self, target: EpisodeState, *, reason: str, authority: str, clock: Clock = utc_now,
                   snapshot_ids: Sequence[str] = (), evidence_refs: Sequence[ArtifactRef] = (),
                   interpretation: Mapping[str, str] | None = None, claims: Sequence[ArtifactRef] = (), evidence: Sequence[ArtifactRef] = (),
                   derivations: Sequence[ArtifactRef] = (), candidate_plans: Sequence[ArtifactRef] = (), authorization: ArtifactRef | None = None,
                   effects: Sequence[ArtifactRef] = (), response: ArtifactRef | None = None, consolidation_refs: Sequence[ArtifactRef] = ()) -> "CognitiveEpisode":
        if not authority:
            raise EpisodeTransitionError("transition authority is required")
        if self.snapshot_bundle is not None:
            allowed = {self.snapshot_bundle.bundle_id, self.snapshot_bundle.state_digest}
            unknown = [sid for sid in snapshot_ids if sid not in allowed]
            if unknown and "rebase" not in reason.lower() and "transition" not in reason.lower():
                raise EpisodeTransitionError("mixed snapshot versions require explicit transition/rebase event")
        ensure_transition(self.state, target)
        prior_digest = self.digest
        updated = replace(
            self,
            state=target,
            interpretation=_freeze_mapping(interpretation) if interpretation is not None else self.interpretation,
            claims=(*self.claims, *tuple(claims)),
            evidence=(*self.evidence, *tuple(evidence)),
            derivations=(*self.derivations, *tuple(derivations)),
            candidate_plans=(*self.candidate_plans, *tuple(candidate_plans)),
            authorization=authorization if authorization is not None else self.authorization,
            effects=(*self.effects, *tuple(effects)),
            response=response if response is not None else self.response,
            consolidation_refs=(*self.consolidation_refs, *tuple(consolidation_refs)),
        )
        return updated._append_event(target, reason=reason, authority=authority, clock=clock, snapshot_ids=tuple(snapshot_ids), evidence_refs=tuple(evidence_refs), prior_digest=prior_digest)

    def _append_event(self, target: EpisodeState, *, reason: str, authority: str, clock: Clock, snapshot_ids: Sequence[str] = (), evidence_refs: Sequence[ArtifactRef] = (), prior_digest: str | None = None) -> "CognitiveEpisode":
        event = TransitionEvent(str(uuid4()), self.state, target, clock(), reason, authority, prior_digest or (self.digest if self.transitions else GENESIS_DIGEST), tuple(snapshot_ids), tuple(evidence_refs))
        return replace(self, transitions=(*self.transitions, event))

    def to_json(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "actor": self.actor.to_json(), "authorization": self.authorization.to_json() if self.authorization else None,
            "candidate_plans": [x.to_json() for x in self.candidate_plans], "claims": [x.to_json() for x in self.claims],
            "consolidation_refs": [x.to_json() for x in self.consolidation_refs], "conversation_id": self.conversation_id,
            "derivations": [x.to_json() for x in self.derivations], "effects": [x.to_json() for x in self.effects],
            "episode_id": self.episode_id, "evidence": [x.to_json() for x in self.evidence], "interpretation": dict(self.interpretation),
            "parent": self.parent.to_json() if self.parent else None, "request": self.request.to_json(), "response": self.response.to_json() if self.response else None,
            "schema_version": self.schema_version, "snapshot_bundle": self.snapshot_bundle.to_json() if self.snapshot_bundle else None,
            "state": self.state.value, "transitions": [event.to_json() for event in self.transitions],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def canonical_json(self) -> str:
        return _canon(self.to_json())
