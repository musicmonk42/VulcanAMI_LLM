from __future__ import annotations

import ast
import asyncio
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vulcan.runtime.composition import (
    CompositionSpecification,
    LegacyWorldReadOnlyAdapter,
    compose_runtime,
)
from vulcan.runtime.errors import StartupErrorCategory, StartupFailure
from vulcan.runtime.settings import (
    OpaqueSecret,
    RuntimeSettings,
    SecretSource,
    VulcanEnvironment,
    durable_root_paths,
)
from vulcan.safety.safety_types import ResponseSafetyDecision, ResponseSafetyStatus


def settings(
    tmp_path: Path, *, env=VulcanEnvironment.production, stub=False
) -> RuntimeSettings:
    root = (tmp_path / "durable").resolve()
    root.mkdir()
    return RuntimeSettings(
        environment=env,
        jwt_issuer="vulcan",
        jwt_audience="vulcan-runtime",
        jwt_secret=OpaqueSecret(
            SecretSource.direct, "A" * 40 + "1!bcdefgh", "VULCAN_JWT_SECRET"
        ),
        durable_root=root,
        durable_paths=durable_root_paths(root),
        approval_hmac_secret=OpaqueSecret(
            SecretSource.direct, "B" * 40 + "1!cdefghi", "VULCAN_APPROVAL_HMAC_SECRET"
        ),
        memory_sqlite_path=root / "memory" / "memory.sqlite",
        development_stub_mode=stub,
    )


class Owner:
    owner_id = "owner"
    capability = SimpleNamespace(value="shadow")
    domain_snapshot_id = "d" * 64

    def __init__(self, *args, **kwargs):
        self.closed = False

    def readiness(self):
        return True

    def close(self):
        self.closed = True

    def capabilities(self):
        return ()

    def append_episode_transition(self, *args):
        return None

    def append_epistemic_commit(self, *args):
        return None

    def append(self, *args):
        return None

    def lease(self):
        return SimpleNamespace(
            domain_snapshot_id=self.domain_snapshot_id,
            policy_digest="a" * 64,
            revision=1,
            close=lambda: None,
        )

    def active_metadata(self):
        return {"policy_digest": "a" * 64, "revision": 1}

    def snapshot_state(self):
        return "0", {"enabled": True, "digest": "b" * 64}


class World(Owner):
    snapshot_id = "world-v1"


class Safety(Owner):
    pass


class SafetyPort(Owner):
    async def evaluate_response(self, text, context):
        return ResponseSafetyDecision(ResponseSafetyStatus.ALLOW, "test", 1.0, {})


def test_legacy_world_adapter_exposes_readiness_and_identity_only():
    legacy = World()
    legacy.reason = lambda query: "untrusted"
    legacy.domain = object()
    adapter = LegacyWorldReadOnlyAdapter(legacy)
    assert adapter.readiness()
    assert adapter.snapshot_id == "world-v1"
    assert not hasattr(adapter, "reason")
    assert not hasattr(adapter, "domain")
    adapter.close()
    assert legacy.closed


def spec(**changes):
    values = dict(
        world_proposal_factory=World,
        safety_validator_factory=Safety,
        audit_projector_factory=Owner,
        governed_memory_factory=lambda config, **kwargs: Owner(),
        alignment_factory=Owner,
        domain_lookup_factory=Owner,
        learning_factory=lambda **kwargs: Owner(),
        safety_port_factory=lambda safety: SafetyPort(),
    )
    values.update(changes)
    return CompositionSpecification(**values)


@pytest.mark.asyncio
async def test_typed_composition_is_ready_and_has_close_graph(tmp_path):
    runtime = compose_runtime(settings(tmp_path), spec())
    assert runtime.deployment is None
    assert runtime.transaction_service is not None
    assert runtime.ownership_close_order[-2:] == ("safety", "world_proposal")
    assert len(runtime.ownership_close_order) == len(set(runtime.ownership_close_order))
    await runtime.readiness()
    await runtime.close()


@pytest.mark.asyncio
async def test_every_declared_factory_is_invoked_exactly_once(tmp_path):
    configured = spec()
    calls: Counter[str] = Counter()
    replacements = {}
    for declared in fields(configured):
        original = getattr(configured, declared.name)

        def counted(*args, _name=declared.name, _factory=original, **kwargs):
            calls[_name] += 1
            return _factory(*args, **kwargs)

        replacements[declared.name] = counted
    runtime = compose_runtime(settings(tmp_path), replace(configured, **replacements))
    try:
        assert calls == Counter({declared.name: 1 for declared in fields(configured)})
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_typed_composition_restarts_from_the_same_durable_root(tmp_path):
    configured = settings(tmp_path)
    first = compose_runtime(configured, spec())
    await first.close()
    restarted = compose_runtime(configured, spec())
    try:
        await restarted.readiness()
        assert restarted.episode_store is not first.episode_store
        assert restarted.epistemic_store is not first.epistemic_store
    finally:
        await restarted.close()


def test_missing_owner_fails_closed_with_category(tmp_path):
    with pytest.raises(StartupFailure) as failure:
        compose_runtime(settings(tmp_path), spec(world_proposal_factory=lambda: None))
    assert failure.value.category is StartupErrorCategory.WORLD_MISSING


def test_duplicate_ownership_is_rejected(tmp_path):
    shared = Owner()
    bad = spec(
        audit_projector_factory=lambda *a, **k: shared,
        governed_memory_factory=lambda *a, **k: shared,
    )
    with pytest.raises(StartupFailure, match="duplicate ownership"):
        compose_runtime(settings(tmp_path), bad)


def test_duplicate_edge_owner_is_rejected_before_authorities_construct(tmp_path):
    shared = World()
    with pytest.raises(StartupFailure, match="duplicate ownership"):
        compose_runtime(
            settings(tmp_path),
            spec(
                world_proposal_factory=lambda: shared,
                safety_validator_factory=lambda: shared,
            ),
        )
    assert shared.closed


def test_partial_construction_closes_earlier_owner(tmp_path):
    world = World()

    def fail():
        raise RuntimeError("safety construction failed")

    with pytest.raises(StartupFailure):
        compose_runtime(
            settings(tmp_path),
            spec(world_proposal_factory=lambda: world, safety_validator_factory=fail),
        )
    assert world.closed


def test_startup_cancellation_is_not_reclassified_and_still_cleans_up(tmp_path):
    world = World()

    def cancel():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        compose_runtime(
            settings(tmp_path),
            spec(world_proposal_factory=lambda: world, safety_validator_factory=cancel),
        )
    assert world.closed


@pytest.mark.asyncio
async def test_close_is_best_effort_and_preserves_first_failure(tmp_path):
    runtime = compose_runtime(settings(tmp_path), spec())
    called = []

    class Broken:
        def close(self):
            called.append("broken")
            raise RuntimeError("close failed")

    class Later:
        def close(self):
            called.append("later")

    runtime.language_output = Broken()
    runtime.language_input = Later()
    with pytest.raises(RuntimeError, match="close failed"):
        await runtime.close()
    assert called == ["broken", "later"]


def test_no_legacy_deployment_import_or_setattr_in_canonical_root():
    path = Path("src/vulcan/runtime/composition.py")
    tree = ast.parse(path.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all("orchestrator.deployment" not in ast.unparse(node) for node in imports)
    assert "ProductionDeployment" not in path.read_text()
    assert "setattr(" not in Path("src/vulcan/runtime/container.py").read_text()
    assert "runtime.kernel =" not in path.read_text()


@pytest.mark.asyncio
async def test_development_stub_composes_but_never_becomes_ready(tmp_path):
    runtime = compose_runtime(
        settings(tmp_path, env=VulcanEnvironment.development, stub=True), spec()
    )
    try:
        with pytest.raises(RuntimeError, match="world_state is unhealthy"):
            await runtime.readiness()
    finally:
        await runtime.close()


def test_production_forbids_development_stub_setting(tmp_path):
    with pytest.raises(StartupFailure) as failure:
        compose_runtime(settings(tmp_path, stub=True), spec())
    assert failure.value.category is StartupErrorCategory.SETTINGS_INVALID
