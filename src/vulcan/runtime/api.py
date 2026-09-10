"""The sole framework-independent application facade for Phase A."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from vulcan.graphix.runtime import Utterance

from .auth import AuthenticatedPrincipal
from .kernel import KernelRequest
from .settings import RuntimeSettings


class CommandKind(str, Enum):
    CHAT = "chat"


class QueryKind(str, Enum):
    READINESS = "readiness"
    INTEGRITY = "integrity"
    CAPABILITIES = "capabilities"
    EPISODE_AUDIT = "episode_audit"


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedAuthenticationContext:
    """Opaque, request-local authentication proof; never persisted as authority."""

    __principal: AuthenticatedPrincipal

    @classmethod
    def from_verified_principal(
        cls, principal: AuthenticatedPrincipal
    ) -> "VerifiedAuthenticationContext":
        if not isinstance(principal, AuthenticatedPrincipal):
            raise TypeError("verified authentication principal required")
        return cls(principal)

    @classmethod
    def public(cls) -> "VerifiedAuthenticationContext":
        return cls(
            AuthenticatedPrincipal(
                "public-route",
                "public",
                "vulcan-runtime",
                ("vulcan-runtime",),
                frozenset(),
                "public-route-context",
                "internal",
            )
        )

    def require(self, scope: str) -> None:
        self.__principal.require(scope)


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    max_steps: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 256 or not 1 <= self.max_output_bytes <= 65536:
            raise ValueError("invalid execution budget")


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    kind: CommandKind
    request_digest: str
    authentication: VerifiedAuthenticationContext
    idempotency_key: str
    deadline: datetime
    budget: ExecutionBudget
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_digest,
            self.authentication,
            self.idempotency_key,
            self.deadline,
        )
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True, slots=True)
class QueryEnvelope:
    kind: QueryKind
    request_digest: str
    authentication: VerifiedAuthenticationContext
    idempotency_key: str
    deadline: datetime
    budget: ExecutionBudget
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_digest,
            self.authentication,
            self.idempotency_key,
            self.deadline,
        )
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


def _validate_envelope(digest: str, auth: object, key: str, deadline: datetime) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid request digest")
    if not isinstance(auth, VerifiedAuthenticationContext):
        raise TypeError("verified authentication context required")
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", key) is None:
        raise ValueError("invalid idempotency key")
    if deadline.tzinfo is None or deadline.utcoffset() != timezone.utc.utcoffset(
        deadline
    ):
        raise ValueError("UTC deadline required")


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("envelope payload must be a mapping")
    frozen: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise ValueError("invalid payload key")
        if value is not None and type(value) not in (str, int, bool):
            raise TypeError("payload values must be immutable primitives")
        frozen[key] = value
    return MappingProxyType(frozen)


class RuntimeAPI:
    """Capability-minimized facade. Public operations are exactly execute/query."""

    __slots__ = ("__runtime",)

    def __init__(self, runtime: object):
        self.__runtime = runtime

    @classmethod
    def _from_settings(cls, settings: RuntimeSettings) -> "RuntimeAPI":
        from .phase_a_composition import compose_phase_a_runtime

        _validate_phase_a_settings(settings)
        return cls(compose_phase_a_runtime(settings))

    @classmethod
    def _from_environment(cls) -> tuple["RuntimeAPI", object]:
        from .settings import load_runtime_settings

        settings = load_runtime_settings()
        return cls._from_settings(settings), settings.auth_config()

    async def _close(self) -> None:
        await self.__runtime.close()

    async def execute(self, envelope: CommandEnvelope) -> Mapping[str, object]:
        if envelope.deadline <= datetime.now(timezone.utc):
            raise TimeoutError("request deadline exceeded")
        if envelope.kind is not CommandKind.CHAT:
            raise ValueError("unsupported command")
        envelope.authentication.require("reason:write")
        message = envelope.payload.get("message")
        conversation_id = envelope.payload.get("conversation_id")
        if not isinstance(message, str):
            raise TypeError("chat message required")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise TypeError("invalid conversation id")
        utterance = Utterance.from_text(message)
        runtime = self.__runtime
        await runtime.admission()
        case = runtime.kernel.create_case(
            request_id=envelope.idempotency_key,
            conversation_id=conversation_id,
            input_digest=utterance.digest,
        )
        remaining = (envelope.deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("request deadline exceeded")
        async with asyncio.timeout(remaining):
            result = await runtime.kernel.handle(
                KernelRequest(utterance, conversation_id), case
            )
        if len(result.response.encode("utf-8")) > envelope.budget.max_output_bytes:
            raise RuntimeError("response exceeds authorized output budget")
        output = result.transport(
            case_id=case.case_id,
            runtime_id=runtime.runtime_id,
            snapshot_id=case.state_snapshot_id,
        )
        output["status"] = result.status.value
        return MappingProxyType(output)

    async def query(self, envelope: QueryEnvelope) -> Mapping[str, object]:
        if envelope.deadline <= datetime.now(timezone.utc):
            raise TimeoutError("request deadline exceeded")
        runtime = self.__runtime
        if envelope.kind is QueryKind.READINESS:
            await runtime.shallow_readiness()
            return MappingProxyType({"status": "ready"})
        if envelope.kind is QueryKind.INTEGRITY:
            envelope.authentication.require("operator:read")
            await runtime.deep_integrity()
            return MappingProxyType(
                {"status": "passed", "runtime_id": runtime.runtime_id}
            )
        if envelope.kind is QueryKind.CAPABILITIES:
            from .capabilities import public_capability_response

            return MappingProxyType(
                public_capability_response(runtime.capability_authority)
            )
        if envelope.kind is QueryKind.EPISODE_AUDIT:
            envelope.authentication.require("audit:read")
            episode_id = envelope.payload.get("episode_id")
            if not isinstance(episode_id, str):
                raise TypeError("episode id required")
            events = runtime.audit.events_for_episode(episode_id)
            return MappingProxyType(
                {
                    "episode_id": episode_id,
                    "events": tuple(
                        {
                            "schema_version": event.schema_version,
                            "sequence": event.sequence,
                            "event_type": event.event_type,
                            "timestamp": event.timestamp,
                            "previous_hash": event.previous_hash,
                            "data": event.data,
                            "event_hash": event.event_hash,
                        }
                        for event in events[:64]
                    ),
                }
            )
        raise ValueError("unsupported query")


def _validate_phase_a_settings(settings: RuntimeSettings) -> None:
    rejected = {
        "memory": settings.memory_enabled,
        "learning": settings.learning_enabled,
        "csiu": settings.csiu_enabled,
        "self_improvement": settings.self_improvement_enabled,
        "openai": settings.openai_enabled,
        "anthropic": settings.anthropic_enabled,
        "transformer": settings.language_mode.value != "deterministic_only",
    }
    enabled = sorted(name for name, value in rejected.items() if value)
    if enabled:
        raise ValueError(
            f"Phase-A profile rejects configured owners: {', '.join(enabled)}"
        )
