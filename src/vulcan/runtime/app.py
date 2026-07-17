"""Canonical statically-composed ASGI application selected by Docker."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os
import base64
import hashlib
import hmac
import json
import time
from vulcan.api.models import UnifiedChatRequest

from .case import CognitiveCase
from .composition import compose_runtime
from .kernel import KernelRequest
from .semantic import Utterance


def _runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.closed:
        raise HTTPException(status_code=503, detail="Cognitive runtime is not ready")
    return runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Routes are registered below, before this point.  Readiness is not set
    # until all required services form one RuntimeContainer.
    app.state.ready = False
    app.state.runtime = None
    runtime = None
    try:
        runtime = await asyncio.to_thread(compose_runtime)
        await runtime.readiness()
        app.state.runtime = runtime
        app.state.ready = True
        yield
    except BaseException:
        # Composition can succeed while a subsequent graph health check fails.
        # Do not leak that partially composed owner graph or publish readiness.
        if runtime is not None:
            try:
                await runtime.close()
            except BaseException:
                pass
        app.state.runtime = None
        raise
    finally:
        app.state.ready = False
        active_runtime = getattr(app.state, "runtime", None)
        if active_runtime is not None:
            await active_runtime.close()
        app.state.runtime = None


def _jwt_secret() -> str | None:
    return os.getenv("GRAPHIX_JWT_SECRET") or os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")

def _decode_segment(segment: str) -> dict[str, object]:
    padding = "=" * (-len(segment) % 4)
    decoded = base64.urlsafe_b64decode((segment + padding).encode("ascii"))
    value = json.loads(decoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JWT segment is not an object")
    return value

def _authenticate_bearer(value: str) -> bool:
    """Verify only configured HS256 bearer tokens; never treat presence as auth."""
    secret = _jwt_secret()
    if not secret or not value.startswith("Bearer "):
        return False
    parts = value[7:].split(".")
    if len(parts) != 3:
        return False
    try:
        header, payload = _decode_segment(parts[0]), _decode_segment(parts[1])
        if header.get("alg") != "HS256" or not isinstance(payload.get("sub"), str):
            return False
        expected = base64.urlsafe_b64encode(hmac.new(secret.encode("utf-8"), f"{parts[0]}.{parts[1]}".encode("ascii"), hashlib.sha256).digest()).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(expected, parts[2]):
            return False
        expiry = payload.get("exp")
        return isinstance(expiry, (int, float)) and not isinstance(expiry, bool) and expiry > time.time()
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return False

class ServingBoundaryMiddleware(BaseHTTPMiddleware):
    """Canonical transport policy: health is public; chat is authenticated and bounded."""
    max_body_bytes = 16_384
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in {"/health/live", "/health/ready"}:
            length = request.headers.get("content-length")
            # Chunked bodies are rejected here; deployments must enforce the same
            # bound upstream before forwarding a request without Content-Length.
            if length is None or not length.isdigit() or int(length) > self.max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "request body exceeds canonical bound"})
            if not _authenticate_bearer(request.headers.get("authorization", "")):
                return JSONResponse(status_code=401, content={"detail": "authentication required"}, headers={"WWW-Authenticate": "Bearer"})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

def generate_route_manifest() -> tuple[dict[str, object], ...]:
    return tuple({"path": path, "method": method, "classification": "public" if path.startswith("/health/") else "protected", "authentication_required": not path.startswith("/health/"), "authorization": "none" if path.startswith("/health/") else "authenticated"} for path, method in (("/health/live", "GET"), ("/health/ready", "GET"), ("/v1/chat", "POST"), ("/v1/chat/orchestrated", "POST"), ("/vulcan/v1/chat", "POST")))

def create_app() -> FastAPI:
    app = FastAPI(title="VULCAN canonical runtime", version="4.0", lifespan=lifespan)
    app.add_middleware(ServingBoundaryMiddleware)
    app.state.route_manifest = generate_route_manifest()

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        runtime = getattr(request.app.state, "runtime", None)
        if not getattr(request.app.state, "ready", False) or runtime is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        try:
            await runtime.readiness()
            capabilities = runtime.capabilities()
        except Exception:
            app = request.app
            app.state.ready = False
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "runtime_id": runtime.runtime_id, "capabilities": list(capabilities)}

    async def chat_handler(request: Request, body: UnifiedChatRequest):
        runtime = _runtime(request)
        if body.history or body.enable_reasoning or body.enable_memory or body.enable_planning or body.enable_causal:
            raise HTTPException(status_code=422, detail="history, memory, planning, causal, and general reasoning are unavailable on the canonical runtime")
        utterance = Utterance.from_text(body.message)
        case = CognitiveCase.create(request_id=request.headers.get("X-Request-ID", str(uuid4())), conversation_id=body.conversation_id, input_digest=utterance.digest)
        result = await runtime.kernel.handle(KernelRequest(utterance, body.conversation_id), case)
        return result.transport(case_id=case.case_id, runtime_id=runtime.runtime_id, snapshot_id=case.state_snapshot_id)

    app.post("/v1/chat", response_model=None)(chat_handler)
    app.post("/v1/chat/orchestrated", response_model=None)(chat_handler)
    app.post("/vulcan/v1/chat", response_model=None)(chat_handler)
    return app


app = create_app()
