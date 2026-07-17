"""Hard gates for remediation sequence item 3's single-authority runtime."""

import asyncio
from types import SimpleNamespace

import pytest

from vulcan.runtime.case import CognitiveCase, CognitiveCaseStatus
from vulcan.runtime.container import RuntimeContainer
from vulcan.runtime.kernel import KernelRequest
from vulcan.runtime.semantic import Utterance


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
    deployment = _deployment()
    runtime = RuntimeContainer.new(deployment=deployment)
    utterance = Utterance.from_text("2 + 2")
    case = CognitiveCase.create(request_id="request", conversation_id="conversation", input_digest=utterance.digest)
    result = await runtime.kernel.handle(KernelRequest(utterance, "conversation"), case)

    assert runtime.world_state is deployment.collective.deps.world_model
    assert runtime.kernel._state_authority is runtime.world_state
    assert runtime.kernel.calls == 1
    assert case.terminal_status is result.status is CognitiveCaseStatus.SUCCESS
    assert "secret prompt" not in repr(case)


@pytest.mark.asyncio
async def test_cases_are_isolated_under_concurrency():
    runtime = RuntimeContainer.new(deployment=_deployment())
    utterance = Utterance.from_text("1+1")
    cases = [CognitiveCase.create(request_id=str(index), conversation_id=None, input_digest=utterance.digest) for index in range(20)]
    await asyncio.gather(*(runtime.kernel.handle(KernelRequest(utterance, None), case) for case in cases))
    assert len({case.case_id for case in cases}) == 20
    assert all(case.terminal_status is CognitiveCaseStatus.SUCCESS for case in cases)
    assert runtime.kernel.calls == 20


def test_container_fails_closed_without_canonical_world_state():
    with pytest.raises(RuntimeError, match="World State"):
        RuntimeContainer.new(deployment=_deployment(world=None))


@pytest.mark.asyncio
async def test_readiness_fails_when_an_owned_port_reports_unhealthy():
    runtime = RuntimeContainer.new(deployment=_deployment())
    runtime.memory.readiness = lambda: False
    with pytest.raises(RuntimeError, match="memory is unhealthy"):
        await runtime.readiness()


@pytest.mark.asyncio
async def test_close_attempts_every_owner_after_a_close_failure():
    closed: list[str] = []

    class _Resource:
        def __init__(self, name: str, broken: bool = False):
            self.name, self.broken = name, broken
        def close(self):
            closed.append(self.name)
            if self.broken:
                raise RuntimeError(self.name)

    runtime = RuntimeContainer.new(deployment=_deployment())
    runtime.language_output = _Resource("output", broken=True)
    runtime.language_input = _Resource("input")
    runtime.memory = _Resource("memory")
    runtime.kernel = _Resource("kernel")
    runtime.safety = _Resource("safety")
    runtime.world_state = _Resource("world")
    runtime.deployment = _Resource("deployment")
    with pytest.raises(RuntimeError, match="output"):
        await runtime.close()
    assert closed == ["output", "input", "memory", "kernel", "safety", "world", "deployment"]


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
