"""Framework-independent cognitive orchestration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .case import CognitiveCase, CognitiveCaseStatus


@dataclass(frozen=True)
class KernelRequest:
    message: str
    conversation_id: str | None
    payload: Any


@dataclass(frozen=True)
class KernelResult:
    payload: dict[str, Any]
    status: CognitiveCaseStatus


LegacyExecutor = Callable[[KernelRequest, CognitiveCase], Awaitable[dict[str, Any]]]


class CognitiveKernel:
    """The only production authority which invokes cognitive orchestration.

    The legacy executor is deliberately an injected adapter.  This lets the
    strangler migration preserve response compatibility without allowing an
    endpoint, provider, or bridge to become another coordinator.
    """

    def __init__(self, *, state_authority: Any, executor: LegacyExecutor) -> None:
        self._state_authority = state_authority
        self._executor = executor
        self.calls = 0

    async def handle(self, request: KernelRequest, case: CognitiveCase) -> KernelResult:
        if case.terminal_status is not CognitiveCaseStatus.OPEN:
            raise RuntimeError("kernel received a closed cognitive case")
        self.calls += 1
        case.state_snapshot_id = self._snapshot_id()
        case.record("kernel_entered")
        try:
            # Item 2's router remains the source of an untrusted proposal;
            # this kernel never treats proposal confidence as authority.
            result = await self._executor(request, case)
            status = (CognitiveCaseStatus.ABSTAINED if
                      result.get("metadata", {}).get("source", "").startswith("deterministic_")
                      else CognitiveCaseStatus.SUCCESS)
            case.close(status)
            return KernelResult(payload=result, status=status)
        except BaseException as exc:
            status = CognitiveCaseStatus.CANCELLED if type(exc).__name__ == "CancelledError" else CognitiveCaseStatus.FAILED
            case.close(status, type(exc).__name__)
            raise

    def _snapshot_id(self) -> str:
        candidate = getattr(self._state_authority, "version", None)
        if candidate is None:
            candidate = getattr(self._state_authority, "snapshot_id", None)
        return f"world-state:{candidate if candidate is not None else id(self._state_authority)}"
