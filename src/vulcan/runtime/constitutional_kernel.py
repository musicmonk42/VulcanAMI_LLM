"""Atomic episode admission around the compatibility semantic kernel."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from vulcan.microkernel.episode import ActorBinding, CognitiveEpisode
from vulcan.microkernel.episode_store import EpisodeStore
from vulcan.microkernel.epistemic_store import EpistemicStore
from vulcan.microkernel.snapshots import SnapshotBundle
from vulcan.microkernel.state_machine import EpisodeState
from vulcan.microkernel.principals import Principal, PrincipalKind
from vulcan.microkernel.transactions import ConstitutionalTransactionService

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
    store: EpisodeStore

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
            if self.store is not None:
                self.store.create(episode)
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
        episode_store: EpisodeStore | None = None,
        epistemic_store: EpistemicStore | None = None,
    ) -> "ConstitutionalCognitiveKernel":
        if not callable(snapshot_admitter):
            raise TypeError("snapshot_admitter must be callable")
        # Direct-kernel callers retain a named compatibility store, but still use
        # durable SQLite transactions. Production composition always supplies its
        # governed durable-root store.
        if episode_store is None:
            path = tempfile.NamedTemporaryFile(
                prefix="vulcan-episode-compat-", suffix=".sqlite3", delete=False
            ).name
            episode_store = EpisodeStore(path)
        if epistemic_store is None:
            path = tempfile.NamedTemporaryFile(
                prefix="vulcan-epistemic-compat-", suffix=".sqlite3", delete=False
            ).name
            epistemic_store = EpistemicStore(path)
        service = ConstitutionalTransactionService(episode_store, epistemic_store)
        principal = Principal(
            PrincipalKind.SYSTEM_KERNEL,
            "constitutional-cognitive-kernel",
            sha256(b"vulcan-constitutional-kernel-v1").hexdigest(),
        )
        binder = getattr(kernel, "bind_transaction_service", None)
        if callable(binder):
            binder(service, principal)
        return cls(kernel, EpisodeAdmissionService(snapshot_admitter, episode_store))

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
            result = await self._delegate.handle(request, case)
            # Explicit cancellation boundary: a finalized response is still not
            # transport-released until the durable head check below completes.
            await asyncio.sleep(0)
            if self.admission.store is not None:
                durable = self.admission.store.load(case.case_id)
                if case.episode is None or durable.digest != case.episode.digest:
                    raise RuntimeError(
                        "transport withheld: durable episode head diverged"
                    )
                if not durable.state.is_terminal:
                    raise RuntimeError(
                        "transport withheld: durable episode is not terminal"
                    )
                if result.status is not case.terminal_status:
                    raise RuntimeError("transport withheld: result status diverged")
                required_by_status = {
                    "success": EpisodeState.CONSOLIDATED,
                    "abstained": EpisodeState.ABSTAINED,
                    "blocked": EpisodeState.BLOCKED,
                    "finalization_error": EpisodeState.FAILED,
                    "failed": EpisodeState.FAILED,
                    "cancelled": EpisodeState.CANCELLED,
                }
                required = required_by_status.get(result.status.value)
                if required is None or durable.state is not required:
                    raise RuntimeError(
                        "transport withheld: durable terminal state missing"
                    )
            return result
        finally:
            case.release_snapshot_bundle()
