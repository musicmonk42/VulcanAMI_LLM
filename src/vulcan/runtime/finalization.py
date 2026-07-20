"""Framework-independent mandatory response finalization."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from vulcan.safety.safety_types import ResponseSafetyContext, ResponseSafetyPort, ResponseSafetyStatus

from .semantic import RenderArtifact


class FinalizationDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class FinalizationResult:
    decision: FinalizationDecision
    artifact: RenderArtifact
    public_text: str


class ResponseFinalizerPort(Protocol):
    async def finalize(self, artifact: RenderArtifact, context: ResponseSafetyContext | None = None) -> FinalizationResult: ...


class SafetyResponseFinalizer:
    """Final fail-closed gate that accepts only typed ResponseSafetyPort decisions."""

    _SAFE_FALLBACK = "I generated a response, but it could not be safely returned. Please rephrase your request."

    def __init__(self, safety: ResponseSafetyPort) -> None:
        if not hasattr(safety, "evaluate_response"):
            raise RuntimeError("finalizer requires a typed response safety port")
        self._safety = safety

    async def finalize(self, artifact: RenderArtifact, context: ResponseSafetyContext | None = None) -> FinalizationResult:
        ctx = context or ResponseSafetyContext(
            case_id="unknown",
            episode_id="unknown",
            response_ir_digest=artifact.ir_digest,
            rendered_text_digest=_text_digest(artifact.text),
            policy_identity="unknown",
            policy_release="unknown",
            actor_risk="unknown",
        )
        decision = await self._safety.evaluate_response(artifact.text, ctx)
        if decision.status is ResponseSafetyStatus.ALLOW:
            return FinalizationResult(FinalizationDecision.ALLOW, artifact, artifact.text)
        if decision.status is ResponseSafetyStatus.CANCELLED:
            return FinalizationResult(FinalizationDecision.CANCELLED, artifact, self._SAFE_FALLBACK)
        if decision.status in {ResponseSafetyStatus.ERROR, ResponseSafetyStatus.TIMEOUT, ResponseSafetyStatus.UNAVAILABLE}:
            return FinalizationResult(FinalizationDecision.ERROR, artifact, self._SAFE_FALLBACK)
        return FinalizationResult(FinalizationDecision.BLOCK, artifact, self._SAFE_FALLBACK)


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
