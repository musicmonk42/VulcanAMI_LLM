"""The sole Phase-A ASGI adapter; all cognition crosses ``RuntimeAPI``."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .api import (
    CommandEnvelope,
    CommandKind,
    ExecutionBudget,
    QueryEnvelope,
    QueryKind,
    RuntimeAPI,
    VerifiedAuthenticationContext,
    request_digest,
)
from .api_models import ReasonRequest
from .auth import AuthError, AuthorizationError, authenticate_bearer
from .errors import (
    ApiContractError,
    ApiErrorCategory,
)
from .route_manifest import generate_route_manifest

MAX_BODY = 16_384
_TRANSACTION_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.api = None
    api = None
    try:
        api, app.state.auth_config = RuntimeAPI._from_environment()
        app.state.api = api
        await api.query(_public_query(QueryKind.READINESS))
        app.state.ready = True
        yield
    finally:
        app.state.ready = False
        app.state.api = None
        if api is not None:
            await api._close()


def _public_query(kind: QueryKind) -> QueryEnvelope:
    payload: dict[str, object] = {}
    return QueryEnvelope(
        kind,
        request_digest(kind, payload),
        VerifiedAuthenticationContext.internal_system_query(),
        f"system-{kind.value}",
        f"public-{kind.value}",
        datetime.now(timezone.utc) + timedelta(seconds=5),
        ExecutionBudget(1, 65536),
        payload,
    )


def _authentication(request: Request, scope: str) -> VerifiedAuthenticationContext:
    try:
        principal = authenticate_bearer(
            request.headers.get("authorization"), request.app.state.auth_config
        )
        principal.require(scope)
        return VerifiedAuthenticationContext.from_verified_principal(principal)
    except AuthorizationError:
        raise ApiContractError(403, ApiErrorCategory.FORBIDDEN, "forbidden") from None
    except AuthError:
        raise ApiContractError(
            401, ApiErrorCategory.AUTHENTICATION_REQUIRED, "authentication required"
        ) from None


def _request_id(request: Request) -> str:
    value = request.headers.get("x-request-id")
    if value is None or _TRANSACTION_ID.fullmatch(value) is None:
        raise ApiContractError(
            400, ApiErrorCategory.SCHEMA_INVALID, "X-Request-ID required"
        )
    return value


def _api(request: Request) -> RuntimeAPI:
    api = getattr(request.app.state, "api", None)
    if not getattr(request.app.state, "ready", False) or not isinstance(
        api, RuntimeAPI
    ):
        raise ApiContractError(
            503, ApiErrorCategory.RUNTIME_NOT_READY, "runtime not ready"
        )
    return api


async def _body(request: Request) -> dict[str, object]:
    if (
        request.headers.get("content-type", "").split(";", 1)[0].lower()
        != "application/json"
    ):
        raise ApiContractError(
            415, ApiErrorCategory.CONTENT_TYPE_UNSUPPORTED, "unsupported content type"
        )
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise ApiContractError(
            413, ApiErrorCategory.BODY_TOO_LARGE, "request body too large"
        )

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ApiContractError(
            400, ApiErrorCategory.MALFORMED_JSON, "malformed JSON"
        ) from None
    if not isinstance(value, dict):
        raise ApiContractError(
            400, ApiErrorCategory.SCHEMA_INVALID, "JSON object required"
        )
    return value


def create_app() -> FastAPI:
    app = FastAPI(
        title="VULCAN Phase-A runtime",
        version="8.0",
        lifespan=lifespan,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    app.state.ready = False
    app.state.api = None

    @app.exception_handler(ApiContractError)
    async def api_error(request: Request, exc: ApiContractError):
        return JSONResponse(
            content={"error": {"code": exc.category.value, "message": exc.message}},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, exc: Exception):
        return JSONResponse(
            content={
                "error": {
                    "code": ApiErrorCategory.SCHEMA_INVALID.value,
                    "message": "schema validation failed",
                }
            },
            status_code=400,
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error(request: Request, exc: AuthorizationError):
        return JSONResponse(
            content={"error": {"code": "forbidden", "message": "forbidden"}},
            status_code=403,
        )

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        try:
            return dict(await _api(request).query(_public_query(QueryKind.READINESS)))
        except ApiContractError:
            return JSONResponse(
                content={"status": "not_ready", "code": "runtime_not_ready"},
                status_code=503,
            )

    @app.get("/health/integrity")
    async def integrity(request: Request):
        payload: dict[str, object] = {}
        envelope = QueryEnvelope(
            QueryKind.INTEGRITY,
            request_digest(QueryKind.INTEGRITY, payload),
            _authentication(request, "operator:read"),
            _request_id(request),
            "integrity",
            datetime.now(timezone.utc) + timedelta(seconds=5),
            ExecutionBudget(1, 65536),
            payload,
        )
        return dict(await _api(request).query(envelope))

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        return dict(await _api(request).query(_public_query(QueryKind.CAPABILITIES)))

    @app.post("/v1/chat")
    async def chat(request: Request):
        body = ReasonRequest.model_validate(await _body(request))
        payload = {"message": body.message, "conversation_id": body.conversation_id}
        key = request.headers.get("idempotency-key")
        if key is None or _TRANSACTION_ID.fullmatch(key) is None:
            raise ApiContractError(
                400, ApiErrorCategory.SCHEMA_INVALID, "Idempotency-Key required"
            )
        envelope = CommandEnvelope(
            CommandKind.CHAT,
            request_digest(CommandKind.CHAT, payload),
            _authentication(request, "reason:write"),
            _request_id(request),
            key,
            datetime.now(timezone.utc) + timedelta(seconds=30),
            ExecutionBudget(64, 65536),
            payload,
        )
        return dict(await _api(request).execute(envelope))

    @app.get("/v1/audit/cases/{episode_id}")
    async def audit_case(episode_id: str, request: Request):
        payload = {"episode_id": episode_id}
        envelope = QueryEnvelope(
            QueryKind.EPISODE_AUDIT,
            request_digest(QueryKind.EPISODE_AUDIT, payload),
            _authentication(request, "audit:read"),
            _request_id(request),
            f"audit-{episode_id}",
            datetime.now(timezone.utc) + timedelta(seconds=5),
            ExecutionBudget(1, 65536),
            payload,
        )
        result = dict(await _api(request).query(envelope))
        if not result["events"]:
            raise ApiContractError(404, ApiErrorCategory.NOT_FOUND, "episode not found")
        return result

    app.state.route_manifest = generate_route_manifest(app)
    return app


app = create_app()
