from types import SimpleNamespace

import pytest

from vulcan.runtime.container import RuntimeContainer
from vulcan.runtime.health import HealthStateMachine, ProcessState


class Owner:
    def __init__(self, healthy=True, message="transient dependency unavailable"):
        self.healthy = healthy
        self.message = message
        self.readiness_calls = 0
    def readiness(self):
        self.readiness_calls += 1
        if self.healthy is True:
            return True
        if self.healthy is False:
            return False
        raise RuntimeError(self.message)
    def capabilities(self):
        return ()
    def capability_matrix(self):
        return ()


def runtime_with_owner(owner, *, root=None):
    h = HealthStateMachine(); h.admit()
    return RuntimeContainer(
        runtime_id="runtime-1",
        deployment=owner,
        world_state=owner,
        kernel=SimpleNamespace(capabilities=lambda: ("bounded-arithmetic",)),
        safety=owner,
        memory=owner,
        language_input=owner,
        language_output=owner,
        language_config=SimpleNamespace(mode="deterministic_only"),
        audit=owner,
        alignment=owner,
        domain_registry=owner,
        durable_root=root,
        self_improvement=SimpleNamespace(capabilities=lambda: (), readiness=lambda: True),
        learning_owner=SimpleNamespace(owner_id="learning", capability=SimpleNamespace(value="shadow"), readiness=owner.readiness, capability_matrix=lambda: ()),
        settings=SimpleNamespace(public_diagnostics=False, environment=SimpleNamespace(value="production"), schema=lambda: {"schema_version":"test"}),
        health=h,
    )


@pytest.mark.asyncio
async def test_transient_readiness_degrades_and_recovers(tmp_path):
    owner = Owner(healthy=True)
    rt = runtime_with_owner(owner, root=tmp_path)
    await rt.shallow_readiness()
    assert rt.health.state is ProcessState.READY
    rt.memory = None
    with pytest.raises(RuntimeError, match="memory is unavailable"):
        await rt.shallow_readiness()
    assert rt.health.state is ProcessState.DEGRADED
    rt.memory = owner
    await rt.shallow_readiness()
    assert rt.health.state is ProcessState.READY
    assert owner.readiness_calls == 0


@pytest.mark.asyncio
async def test_irreversible_corruption_fails_closed(tmp_path):
    owner = Owner(healthy=None, message="audit hash mismatch corruption")
    rt = runtime_with_owner(owner, root=tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        await rt.deep_integrity()
    assert rt.health.state is ProcessState.FAILED
    with pytest.raises(RuntimeError, match="closed"):
        await rt.admission()


@pytest.mark.asyncio
async def test_drain_and_shutdown_are_not_reopened(tmp_path):
    owner = Owner(healthy=True)
    rt = runtime_with_owner(owner, root=tmp_path)
    await rt.shallow_readiness()
    rt.health.drain()
    with pytest.raises(RuntimeError, match="closed"):
        await rt.admission()
    await rt.close()
    assert rt.health.state is ProcessState.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        await rt.shallow_readiness()


@pytest.mark.asyncio
async def test_readiness_cost_does_not_scale_with_deep_history(tmp_path):
    owner = Owner(healthy=True)
    rt = runtime_with_owner(owner, root=tmp_path)
    for _ in range(50):
        await rt.shallow_readiness()
    assert owner.readiness_calls == 0
    await rt.deep_integrity()
    assert owner.readiness_calls > 0
