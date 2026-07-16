"""Canonical statically-composed ASGI application selected by Docker."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from vulcan.api.models import UnifiedChatRequest

from .case import CognitiveCase
from .composition import compose_runtime
from .kernel import KernelRequest


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
    try:
        runtime = await asyncio.to_thread(compose_runtime)
        app.state.runtime = runtime
        app.state.ready = True
        yield
    finally:
        app.state.ready = False
        runtime = getattr(app.state, "runtime", None)
        if runtime is not None:
            await runtime.close()
        app.state.runtime = None


def create_app() -> FastAPI:
    app = FastAPI(title="VULCAN canonical runtime", version="3.0", lifespan=lifespan)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        runtime = getattr(request.app.state, "runtime", None)
        if not getattr(request.app.state, "ready", False) or runtime is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "runtime_id": runtime.runtime_id}

    async def chat_handler(request: Request, body: UnifiedChatRequest):
        runtime = _runtime(request)
        case = CognitiveCase.create(request_id=request.headers.get("X-Request-ID", str(uuid4())),
                                    conversation_id=body.conversation_id, message=body.message)
        command = KernelRequest(message=body.message, conversation_id=body.conversation_id, payload=request)
        result = await runtime.kernel.handle(command, case)
        result.payload.setdefault("metadata", {}).update({"case_id": case.case_id,
                                                             "runtime_id": runtime.runtime_id,
                                                             "state_snapshot_id": case.state_snapshot_id})
        return result.payload

    # Compatibility aliases deliberately reference the same callable.
    app.post("/v1/chat", response_model=None)(chat_handler)
    app.post("/v1/chat/orchestrated", response_model=None)(chat_handler)
    app.post("/vulcan/v1/chat", response_model=None)(chat_handler)
    return app


app = create_app()
