from types import SimpleNamespace

import pytest

from vulcan.runtime.container import RuntimeContainer


class Owner:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.readiness_calls = 0
    def readiness(self):
        self.readiness_calls += 1
        return self.healthy
    def capabilities(self):
        return ()
    def capability_matrix(self):
        return ()


def runtime_with_owner(owner):
    runtime = RuntimeContainer(
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
        durable_root=object(),
        self_improvement=SimpleNamespace(capabilities=lambda: (), readiness=lambda: True),
        learning_owner=SimpleNamespace(owner_id="learning", capability=SimpleNamespace(value="shadow"), readiness=owner.readiness, capability_matrix=lambda: ()),
        settings=SimpleNamespace(public_diagnostics=False, environment=SimpleNamespace(value="production"), schema=lambda: {"schema_version":"test"}),
    )
    return runtime


@pytest.mark.asyncio
async def test_shallow_readiness_does_not_perform_deep_integrity_checks():
    owner = Owner(healthy=False)
    runtime = runtime_with_owner(owner)

    await runtime.shallow_readiness()

    assert owner.readiness_calls == 0


@pytest.mark.asyncio
async def test_deep_integrity_invokes_owned_checks_and_fails_closed():
    owner = Owner(healthy=False)
    runtime = runtime_with_owner(owner)

    with pytest.raises(RuntimeError, match="deployment is unhealthy"):
        await runtime.deep_integrity()
    assert owner.readiness_calls == 1


def test_ready_endpoint_is_shallow_and_integrity_endpoint_is_deep():
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from vulcan.runtime.app import create_app

    owner = Owner(healthy=False)
    runtime = runtime_with_owner(owner)
    app = create_app()
    app.state.ready = True
    app.state.runtime = runtime

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        integrity = client.get("/health/integrity")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert integrity.status_code == 503
    assert integrity.json()["status"] == "failed"
    assert owner.readiness_calls == 1
