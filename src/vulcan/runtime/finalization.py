"""Framework-independent mandatory response finalization."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from .semantic import RenderArtifact
class FinalizationDecision(str, Enum): ALLOW="allow"; BLOCK="block"; ERROR="error"; CANCELLED="cancelled"
@dataclass(frozen=True)
class FinalizationResult:
    decision: FinalizationDecision
    artifact: RenderArtifact
    public_text: str
class ResponseFinalizerPort(Protocol):
    async def finalize(self, artifact: RenderArtifact) -> FinalizationResult: ...
class SafetyResponseFinalizer:
    def __init__(self, safety: Any) -> None: self._safety = safety
    async def finalize(self, artifact: RenderArtifact) -> FinalizationResult:
        try:
            verdict = self._safety.validate_action({"type":"response", "content":artifact.text})
            if hasattr(verdict, "__await__"): verdict = await verdict
            allowed = verdict[0] if isinstance(verdict, tuple) else verdict is True
        except Exception:
            allowed = False
        if allowed: return FinalizationResult(FinalizationDecision.ALLOW, artifact, artifact.text)
        return FinalizationResult(FinalizationDecision.BLOCK, artifact, "I generated a response, but it could not be safely returned. Please rephrase your request.")
