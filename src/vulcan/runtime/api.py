"""The sole framework-independent application facade for Phase A."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from vulcan.graphix.runtime import Utterance
from vulcan.microkernel.episode import ActorBinding
from vulcan.microkernel.episode import canonical_digest as canonical_episode_digest

from .auth import AuthenticatedPrincipal, CredentialProvenance
from .kernel import KernelRequest
from .settings import RuntimeSettings

_AUTHENTICATION_CONTEXT_TOKEN = object()


class CommandKind(str, Enum):
    CHAT = "chat"


class QueryKind(str, Enum):
    READINESS = "readiness"
    INTEGRITY = "integrity"
    CAPABILITIES = "capabilities"
    EPISODE_AUDIT = "episode_audit"


@dataclass(frozen=True, slots=True, repr=False, init=False)
class VerifiedAuthenticationContext:
    """Opaque verified actor plus nonpersistent credential provenance."""

    __actor: ActorBinding
    __credential: CredentialProvenance | None
    __scopes: frozenset[str]

    def __init__(
        self,
        actor: ActorBinding,
        credential: CredentialProvenance | None,
        scopes: frozenset[str],
        *,
        _construction_token: object = None,
    ) -> None:
        if _construction_token is not _AUTHENTICATION_CONTEXT_TOKEN:
            raise TypeError("authentication context is adapter-created")
        object.__setattr__(self, "_VerifiedAuthenticationContext__actor", actor)
        object.__setattr__(
            self, "_VerifiedAuthenticationContext__credential", credential
        )
        object.__setattr__(self, "_VerifiedAuthenticationContext__scopes", scopes)

    @classmethod
    def from_verified_principal(
        cls, principal: AuthenticatedPrincipal
    ) -> "VerifiedAuthenticationContext":
        if not isinstance(principal, AuthenticatedPrincipal):
            raise TypeError("verified authentication principal required")
        return cls(
            principal.actor,
            CredentialProvenance.from_principal(principal),
            principal.scopes,
            _construction_token=_AUTHENTICATION_CONTEXT_TOKEN,
        )

    @classmethod
    def internal_system_query(cls) -> "VerifiedAuthenticationContext":
        return cls(
            ActorBinding.internal_system_query(),
            None,
            frozenset(),
            _construction_token=_AUTHENTICATION_CONTEXT_TOKEN,
        )

    def require(self, scope: str) -> None:
        if scope not in self.__scopes:
            from .auth import AuthorizationError

            raise AuthorizationError("missing required scope")

    def _actor_binding(self) -> ActorBinding:
        return self.__actor

    def _credential_provenance_digest(self) -> str:
        if self.__credential is None:
            raise TypeError("mutating commands require credential provenance")
        value = self.__credential
        return canonical_episode_digest(
            {
                "adapter_release": value.adapter_release,
                "authenticated_at": value.authenticated_at.isoformat().replace(
                    "+00:00", "Z"
                ),
                "key_id": value.key_id,
                "method": value.method,
                "scopes": sorted(value.scopes),
                "token_id": value.token_id,
            }
        )


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
    request_id: str
    idempotency_key: str
    deadline: datetime
    budget: ExecutionBudget
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_digest,
            self.authentication,
            self.request_id,
            self.idempotency_key,
            self.deadline,
        )
        payload = _freeze_payload(self.payload)
        object.__setattr__(self, "payload", payload)
        _require_payload_digest(self.kind.value, payload, self.request_digest)


@dataclass(frozen=True, slots=True)
class QueryEnvelope:
    kind: QueryKind
    request_digest: str
    authentication: VerifiedAuthenticationContext
    request_id: str
    idempotency_key: str
    deadline: datetime
    budget: ExecutionBudget
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_envelope(
            self.request_digest,
            self.authentication,
            self.request_id,
            self.idempotency_key,
            self.deadline,
        )
        payload = _freeze_payload(self.payload)
        object.__setattr__(self, "payload", payload)
        _require_payload_digest(self.kind.value, payload, self.request_digest)


def _validate_envelope(
    digest: str, auth: object, request_id: str, key: str, deadline: datetime
) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid request digest")
    if not isinstance(auth, VerifiedAuthenticationContext):
        raise TypeError("verified authentication context required")
    if (
        not isinstance(request_id, str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id) is None
    ):
        raise ValueError("invalid request id")
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


def request_digest(kind: CommandKind | QueryKind, payload: Mapping[str, object]) -> str:
    frozen = _freeze_payload(payload)
    document = {"kind": kind.value, "payload": dict(frozen)}
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_payload_digest(
    kind: str, payload: Mapping[str, object], supplied_digest: str
) -> None:
    expected = request_digest(
        CommandKind(kind) if kind == CommandKind.CHAT.value else QueryKind(kind),
        payload,
    )
    if supplied_digest != expected:
        raise ValueError("request digest does not bind envelope payload")


def _bounded_output(
    payload: dict[str, object], budget: ExecutionBudget
) -> Mapping[str, object]:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > budget.max_output_bytes:
        raise RuntimeError("response exceeds authorized output budget")
    return MappingProxyType(payload)


def _remaining(deadline: datetime) -> float:
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        raise TimeoutError("request deadline exceeded")
    return seconds


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
        if envelope.budget.max_steps < 8:
            raise RuntimeError(
                "execution budget is insufficient for constitutional arithmetic"
            )
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
        replay = getattr(runtime.kernel.transaction_service, "replay", None)
        if callable(replay):
            prior = replay(
                actor=envelope.authentication._actor_binding(),
                request_digest=envelope.request_digest,
                idempotency_key=envelope.idempotency_key,
            )
            if prior is not None:
                episode, response, status = prior
                return _bounded_output(
                    {
                        "response": response,
                        "metadata": {
                            "case_id": episode.episode_id,
                            "runtime_id": runtime.runtime_id,
                            "state_snapshot_id": (
                                None
                                if episode.snapshot_bundle is None
                                else episode.snapshot_bundle.state_digest
                            ),
                            "semantic_schema_version": "response-ir/3",
                            "terminal_status": status,
                            "response_released": response is not None,
                            "finalized": True,
                            "finalization_safety_decision": "allow",
                        },
                        "status": status,
                    },
                    envelope.budget,
                )
        case = runtime.kernel.create_case(
            request_id=envelope.request_id,
            conversation_id=conversation_id,
            input_digest=utterance.digest,
            actor=envelope.authentication._actor_binding(),
            credential_provenance_digest=envelope.authentication._credential_provenance_digest(),
            request_digest=envelope.request_digest,
            idempotency_key=envelope.idempotency_key,
        )
        async with asyncio.timeout(_remaining(envelope.deadline)):
            result = await runtime.kernel.handle(
                KernelRequest(utterance, conversation_id), case
            )
        output = result.transport(
            case_id=case.case_id,
            runtime_id=runtime.runtime_id,
            snapshot_id=case.state_snapshot_id,
        )
        output["status"] = result.status.value
        return _bounded_output(output, envelope.budget)

    async def query(self, envelope: QueryEnvelope) -> Mapping[str, object]:
        if envelope.deadline <= datetime.now(timezone.utc):
            raise TimeoutError("request deadline exceeded")
        runtime = self.__runtime
        if envelope.kind is QueryKind.READINESS:
            async with asyncio.timeout(_remaining(envelope.deadline)):
                await runtime.shallow_readiness()
            return _bounded_output({"status": "ready"}, envelope.budget)
        if envelope.kind is QueryKind.INTEGRITY:
            envelope.authentication.require("operator:read")
            async with asyncio.timeout(_remaining(envelope.deadline)):
                await runtime.deep_integrity()
            return _bounded_output(
                {"status": "passed", "runtime_id": runtime.runtime_id},
                envelope.budget,
            )
        if envelope.kind is QueryKind.CAPABILITIES:
            from .capabilities import public_capability_response

            async with asyncio.timeout(_remaining(envelope.deadline)):
                result = await asyncio.to_thread(
                    public_capability_response, runtime.capability_authority
                )
            return _bounded_output(result, envelope.budget)
        if envelope.kind is QueryKind.EPISODE_AUDIT:
            envelope.authentication.require("audit:read")
            episode_id = envelope.payload.get("episode_id")
            if not isinstance(episode_id, str):
                raise TypeError("episode id required")
            async with asyncio.timeout(_remaining(envelope.deadline)):
                episode = await asyncio.to_thread(
                    runtime.episode_store.load, episode_id
                )
                requester = envelope.authentication._actor_binding()
                if (
                    requester.classification != "AUTHENTICATED"
                    or episode.actor.classification != "AUTHENTICATED"
                    or (requester.tenant, requester.issuer)
                    != (episode.actor.tenant, episode.actor.issuer)
                ):
                    from .auth import AuthorizationError

                    raise AuthorizationError("cross-tenant episode access denied")
                events = await asyncio.to_thread(
                    runtime.audit.events_for_episode, episode_id
                )
            return _bounded_output(
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
                },
                envelope.budget,
            )
        raise ValueError("unsupported query")


def _validate_phase_a_settings(settings: RuntimeSettings) -> None:
    from .phase_a_composition import reject_forbidden_settings

    reject_forbidden_settings(settings)
