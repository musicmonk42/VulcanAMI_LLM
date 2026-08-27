"""Snapshot-admitting constitutional wrapper for the compatibility kernel."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Callable

from vulcan.microkernel.snapshots import SnapshotBundle

from .case import CognitiveCase
from .kernel import CognitiveKernel, KernelRequest, KernelResult

SnapshotAdmitter = Callable[[str], SnapshotBundle]


class ConstitutionalCognitiveKernel(CognitiveKernel):
    """Run the existing semantic kernel inside an admitted episode context.

    This is a strangler adapter. It preserves the existing bounded semantic path
    while making the composed production route use the real multi-authority
    snapshot bundle and the authoritative ``CognitiveEpisode`` lifecycle.
    """

    @classmethod
    def from_kernel(
        cls,
        kernel: CognitiveKernel,
        *,
        snapshot_admitter: SnapshotAdmitter,
    ) -> "ConstitutionalCognitiveKernel":
        if not isinstance(kernel, CognitiveKernel):
            raise TypeError("constitutional wrapper requires CognitiveKernel")
        if not callable(snapshot_admitter):
            raise TypeError("snapshot_admitter must be callable")
        wrapped = cls.__new__(cls)
        wrapped.__dict__ = dict(kernel.__dict__)
        wrapped._constitutional_snapshot_admitter = snapshot_admitter
        wrapped._active_snapshot_digest = ContextVar(
            f"vulcan_snapshot_{id(wrapped)}",
            default=None,
        )
        return wrapped

    async def handle(
        self,
        request: KernelRequest,
        case: CognitiveCase,
    ) -> KernelResult:
        bundle = self._constitutional_snapshot_admitter(case.case_id)
        try:
            bundle.validate_active(datetime.now(timezone.utc))
            case.bind_snapshot_bundle(bundle)
        except BaseException:
            bundle.close()
            raise
        token = self._active_snapshot_digest.set(bundle.digest)
        try:
            return await super().handle(request, case)
        finally:
            self._active_snapshot_digest.reset(token)
            case.release_snapshot_bundle()

    def _snapshot_id(self) -> str:
        active = self._active_snapshot_digest.get()
        if active is not None:
            return active
        return super()._snapshot_id()
