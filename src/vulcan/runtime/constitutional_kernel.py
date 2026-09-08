"""Atomic episode admission around the compatibility semantic kernel."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.snapshots import SnapshotBundle

from .case import CognitiveCase
from .kernel import KernelRequest, KernelResult


class SnapshotAdmitter(Protocol):
    def __call__(self, episode_id: str) -> SnapshotBundle: ...


class KernelDelegate(Protocol):
    async def handle(
        self, request: KernelRequest, case: CognitiveCase
    ) -> KernelResult: ...
    def capabilities(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class EpisodeAdmissionService:
    """Own the fail-closed episode/snapshot genesis transaction."""

    snapshot_admitter: SnapshotAdmitter

    def admit(
        self,
        *,
        request_id: str,
        conversation_id: str | None,
        input_digest: str,
        actor: ActorBinding | None = None,
    ) -> CognitiveCase:
        episode_id = f"case-{uuid4().hex}"
        bundle = self.snapshot_admitter(episode_id)
        try:
            bundle.validate_active(datetime.now(timezone.utc))
            if bundle.episode_id != episode_id:
                raise ValueError("snapshot bundle/episode identity mismatch")
            binding = actor or ActorBinding(
                actor_id="canonical-runtime",
                principal_digest=sha256(request_id.encode("utf-8")).hexdigest(),
                authority="CognitiveKernel",
            )
            episode = CognitiveEpisode.create(
                actor=binding,
                request_id=request_id,
                input_digest=input_digest,
                conversation_id=conversation_id,
                episode_id=episode_id,
                snapshot_bundle=bundle.bundle_ref(),
            )
            return CognitiveCase.from_admitted_episode(episode=episode, bundle=bundle)
        except BaseException:
            bundle.close()
            raise


class ConstitutionalCognitiveKernel:
    """Composition adapter enforcing admission before semantic delegation.

    It owns neither the delegate's resources nor its close lifecycle. Remove this
    adapter when the semantic compatibility kernel itself consumes episodes.
    """

    def __init__(self, delegate: KernelDelegate, admission: EpisodeAdmissionService):
        if not callable(getattr(delegate, "handle", None)):
            raise TypeError("constitutional wrapper requires a kernel delegate")
        self._delegate = delegate
        self.admission = admission

    @classmethod
    def from_kernel(
        cls,
        kernel: KernelDelegate,
        *,
        snapshot_admitter: SnapshotAdmitter,
    ) -> "ConstitutionalCognitiveKernel":
        if not callable(snapshot_admitter):
            raise TypeError("snapshot_admitter must be callable")
        return cls(kernel, EpisodeAdmissionService(snapshot_admitter))

    @property
    def calls(self) -> int:
        return int(getattr(self._delegate, "calls", 0))

    def capabilities(self) -> tuple[str, ...]:
        return self._delegate.capabilities()

    def create_case(
        self, *, request_id: str, conversation_id: str | None, input_digest: str
    ) -> CognitiveCase:
        return self.admission.admit(
            request_id=request_id,
            conversation_id=conversation_id,
            input_digest=input_digest,
        )

    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.snapshot_bundle is None or case.episode is None:
            raise RuntimeError("constitutional kernel rejects an unadmitted case")
        if case.episode.snapshot_bundle != case.snapshot_bundle.bundle_ref():
            raise RuntimeError("case snapshot projection diverged from episode")
        try:
            return await self._delegate.handle(request, case)
        finally:
            case.release_snapshot_bundle()
