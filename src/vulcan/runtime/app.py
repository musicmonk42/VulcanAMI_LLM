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
        utterance = Utterance.from_text(body.message)
        case = CognitiveCase.create(request_id=request.headers.get("X-Request-ID", str(uuid4())),
                                    conversation_id=body.conversation_id, input_digest=utterance.digest)
        result = await runtime.kernel.handle(KernelRequest(utterance, body.conversation_id), case)
        transport = result.transport(case_id=case.case_id, runtime_id=runtime.runtime_id,
                                     snapshot_id=case.state_snapshot_id)
        safety_payload = {"type": "response", "content": result.response}
        try:
            decision = await asyncio.to_thread(runtime.safety.validate_action, safety_payload)
            allowed = decision[0] if isinstance(decision, tuple) else (decision if isinstance(decision, bool) else False)
        except Exception:
            allowed = False
        transport["metadata"]["finalized"] = True
        transport["metadata"]["finalization_safety_decision"] = "allow" if allowed else "block"
        if not allowed:
            transport["response"] = "I generated a response, but it could not be safely returned. Please rephrase your request."
            transport["safety_status"] = "output_filtered"
        case.record_finalization(transport["metadata"]["finalization_safety_decision"])
        return transport

    # Compatibility aliases deliberately reference the same callable.
    app.post("/v1/chat", response_model=None)(chat_handler)
    app.post("/v1/chat/orchestrated", response_model=None)(chat_handler)
    app.post("/vulcan/v1/chat", response_model=None)(chat_handler)
    return app


app = create_app()
