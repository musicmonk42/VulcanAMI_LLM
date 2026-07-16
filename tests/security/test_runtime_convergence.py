"""Hard gates for remediation sequence item 3's single-authority runtime."""

import asyncio
from types import SimpleNamespace

import pytest

from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.container import RuntimeContainer
from vulcan.runtime.kernel import KernelRequest


class _Safety:
    def validate_action(self, _payload):
        return True


_DEFAULT_WORLD = object()


def _deployment(world=_DEFAULT_WORLD):
    if world is _DEFAULT_WORLD:
        world = SimpleNamespace(version="7")
    return SimpleNamespace(collective=SimpleNamespace(deps=SimpleNamespace(
        world_model=world, safety_validator=_Safety(), memory=None
    )))


@pytest.mark.asyncio
async def test_one_container_injects_one_world_state_and_kernel_closes_case():
    calls = 0

    async def executor(_request, _case):
        nonlocal calls
        calls += 1
        return {"response": "bounded", "metadata": {}}

    deployment = _deployment()
    runtime = RuntimeContainer.new(deployment=deployment, executor=executor)
    case = CognitiveCase.create(request_id="request", conversation_id="conversation", message="secret prompt")
    result = await runtime.kernel.handle(KernelRequest("secret prompt", "conversation", object()), case)

    assert runtime.world_state is deployment.collective.deps.world_model
    assert runtime.kernel._state_authority is runtime.world_state
    assert calls == runtime.kernel.calls == 1
    assert case.terminal_status is result.status is CognitiveCaseStatus.SUCCESS
    assert "secret prompt" not in repr(case)


@pytest.mark.asyncio
async def test_cases_are_isolated_under_concurrency():
    async def executor(_request, _case):
        await asyncio.sleep(0)
        return {"response": "ok", "metadata": {}}

    runtime = RuntimeContainer.new(deployment=_deployment(), executor=executor)
    cases = [CognitiveCase.create(request_id=str(index), conversation_id=None, message=f"prompt-{index}") for index in range(20)]
    await asyncio.gather(*(runtime.kernel.handle(KernelRequest("x", None, object()), case) for case in cases))
    assert len({case.case_id for case in cases}) == 20
    assert all(case.terminal_status is CognitiveCaseStatus.SUCCESS for case in cases)
    assert runtime.kernel.calls == 20


def test_container_fails_closed_without_canonical_world_state():
    with pytest.raises(RuntimeError, match="World State"):
        RuntimeContainer.new(deployment=_deployment(world=None), executor=lambda *_: None)


def test_production_docker_uses_canonical_package_identity():
    dockerfile = open("Dockerfile", encoding="utf-8").read()
    assert "uvicorn vulcan.runtime.app:app" in dockerfile
    assert "ENV PYTHONPATH=/app/src" in dockerfile
    assert "uvicorn src.full_platform:app" not in dockerfile


def test_chat_aliases_are_one_handler_and_legacy_routes_are_not_deployed():
    from vulcan.runtime.app import create_app

    app = create_app()
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    assert routes["/v1/chat"] is routes["/v1/chat/orchestrated"] is routes["/vulcan/v1/chat"]
