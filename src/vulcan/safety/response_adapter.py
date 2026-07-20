"""Typed adapter from canonical response finalization to EnhancedSafetyValidator."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from .safety_types import (
    ResponseSafetyContext,
    ResponseSafetyDecision,
    ResponseSafetyPort,
    ResponseSafetyStatus,
    SafetyReport,
    SafetyValidator,
)


class EnhancedSafetyResponseAdapter(ResponseSafetyPort):
    """Invoke the concrete post-generation validation path and fail closed."""

    adapter_identity = "enhanced-safety-response-adapter/1"

    def __init__(self, validator: Any, *, timeout_seconds: float = 2.0) -> None:
        self.validator = validator
        self.timeout_seconds = timeout_seconds
        if timeout_seconds <= 0:
            raise ValueError("response safety timeout must be positive")

    def readiness(self) -> bool:
        self._require_concrete_validator()
        return True

    async def evaluate_response(self, response_text: str, context: ResponseSafetyContext) -> ResponseSafetyDecision:
        try:
            self._require_concrete_validator()
        except Exception as exc:
            return self._decision(ResponseSafetyStatus.UNAVAILABLE, str(exc), 0.0, context)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.validator.validate_response, response_text, self._redacted_query(context)),
                timeout=self.timeout_seconds,
            )
        except asyncio.CancelledError:
            return self._decision(ResponseSafetyStatus.CANCELLED, "response safety evaluation cancelled", 0.0, context)
        except TimeoutError:
            return self._decision(ResponseSafetyStatus.TIMEOUT, "response safety evaluation timed out; worker may complete later", 0.0, context)
        except Exception as exc:
            return self._decision(ResponseSafetyStatus.ERROR, f"response validator exception: {type(exc).__name__}", 0.0, context)
        return self._normalize(result, response_text, context)

    def _require_concrete_validator(self) -> None:
        if self.validator is None:
            raise RuntimeError("response safety validator is missing")
        method = getattr(self.validator, "validate_response", None)
        if not callable(method):
            raise RuntimeError("response safety validator lacks validate_response")
        owner = getattr(method, "__func__", method)
        base = getattr(SafetyValidator, "validate_response", None)
        if base is not None and owner is base:
            raise RuntimeError("response safety validator is inherited base stub")
        if type(self.validator) is SafetyValidator:
            raise RuntimeError("response safety validator is base stub")
        try:
            from .safety_validator import EnhancedSafetyValidator
        except Exception as exc:  # pragma: no cover - import failure is unavailable
            raise RuntimeError("concrete EnhancedSafetyValidator path is unavailable") from exc
        if not isinstance(self.validator, EnhancedSafetyValidator):
            raise RuntimeError("response safety validator is not the concrete EnhancedSafetyValidator path")

    def _normalize(self, result: Any, response_text: str, context: ResponseSafetyContext) -> ResponseSafetyDecision:
        if not isinstance(result, SafetyReport):
            return self._decision(ResponseSafetyStatus.ERROR, "malformed response safety verdict", 0.0, context)
        audit = {"adapter_identity": self.adapter_identity, "context": asdict(context), "violations": [getattr(v, "value", str(v)) for v in result.violations], "metadata": dict(result.metadata)}
        modified = result.metadata.get("modified_response") or result.metadata.get("modified_text")
        if modified is not None and modified != response_text:
            return ResponseSafetyDecision(ResponseSafetyStatus.MODIFIED, "response safety attempted to modify rendered text", result.confidence, audit)
        if result.safe is True:
            return ResponseSafetyDecision(ResponseSafetyStatus.ALLOW, "; ".join(result.reasons) or "response allowed", result.confidence, audit)
        return ResponseSafetyDecision(ResponseSafetyStatus.BLOCK, "; ".join(result.reasons) or "response blocked", result.confidence, audit)

    def _decision(self, status: ResponseSafetyStatus, reason: str, confidence: float, context: ResponseSafetyContext) -> ResponseSafetyDecision:
        return ResponseSafetyDecision(status, reason, confidence, {"adapter_identity": self.adapter_identity, "context": asdict(context)})

    @staticmethod
    def _redacted_query(context: ResponseSafetyContext) -> str:
        return f"case={context.case_id};episode={context.episode_id};ir={context.response_ir_digest};policy={context.policy_identity}@{context.policy_release};actor_risk={context.actor_risk}"
