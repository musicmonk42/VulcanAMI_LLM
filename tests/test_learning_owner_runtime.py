import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vulcan.learning_owner import (
    LearningCapabilityStatus,
    LearningOwner,
    LearningOwnerBackpressureError,
    LearningOwnerClosedError,
)
from vulcan.runtime.container import RuntimeContainer


class _Safety:
    def validate_action(self, _payload):
        return True
    def close(self):
        pass


def _deployment(world=None, continual=None):
    if world is None:
        world = SimpleNamespace(version="7")
    deps = SimpleNamespace(world_model=world, safety_validator=_Safety(), memory=None, continual=continual)
    return SimpleNamespace(collective=SimpleNamespace(deps=deps), runtime_root=None, data_dir=None)


@pytest.fixture
def durable_root(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setenv("VULCAN_RUNTIME_DURABLE_ROOT", str(root))
    return root


def test_runtime_composition_publishes_one_learning_owner_identity(durable_root):
    deployment = _deployment()
    runtime = RuntimeContainer.new(deployment=deployment)

    owner = runtime.learning_owner
    assert owner is not None
    assert owner.capability is LearningCapabilityStatus.SHADOW
    assert deployment.learning_owner is owner
    assert deployment.learning_system is owner
    assert deployment.collective.deps.learning_owner is owner
    assert deployment.collective.deps.learning_system is owner
    assert owner.owner_id == deployment.collective.deps.learning_owner.owner_id
    assert "learning:shadow" in runtime.capabilities()
    assert "learning:active" not in runtime.capabilities()


@pytest.mark.asyncio
async def test_learning_owner_readiness_and_close_semantics(durable_root):
    runtime = RuntimeContainer.new(deployment=_deployment())
    owner = runtime.learning_owner
    assert owner.readiness() is True
    await runtime.close()
    await runtime.close()
    assert owner.capability is LearningCapabilityStatus.CLOSED
    with pytest.raises(LearningOwnerClosedError):
        owner.readiness()


def test_learning_owner_queue_capacity_and_overflow_are_deterministic():
    owner = LearningOwner(observation_capacity=2, work_capacity=1, isolated_test_owner=True)
    assert owner.submit_work({"n": 1}) == owner.owner_id
    with pytest.raises(LearningOwnerBackpressureError):
        owner.submit_work({"n": 2})
    # Observation queue capacity is enforced by typed observation tests.
    snap = owner.status_snapshot()
    assert snap.observation_queue.capacity == 2
    assert snap.observation_queue.pending == 0
    assert snap.work_queue.capacity == 1
    assert snap.work_queue.pending == 1


@pytest.mark.asyncio
async def test_learning_owner_submission_after_close_fails():
    owner = LearningOwner(isolated_test_owner=True)
    await owner.close()
    with pytest.raises(LearningOwnerClosedError):
        owner.submit_work({"closed": True})


def test_learning_owner_readiness_fails_after_worker_failure():
    owner = LearningOwner(isolated_test_owner=True)
    owner.mark_worker_failed()
    with pytest.raises(RuntimeError, match="worker is unhealthy"):
        owner.readiness()
    assert owner.capability is LearningCapabilityStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_learning_owner_closes_shared_subordinate_once_and_attempts_all():
    calls = []

    class Resource:
        def __init__(self, name, broken=False):
            self.name = name
            self.broken = broken
        def close(self):
            calls.append(self.name)
            if self.broken:
                raise RuntimeError(self.name)

    shared = Resource("shared", broken=True)
    other = Resource("other")
    owner = LearningOwner(resources={"a": shared, "b": shared, "c": other}, isolated_test_owner=True)
    with pytest.raises(RuntimeError, match="shared"):
        await owner.close()
    assert calls == ["shared", "other"]
    await owner.close()
    assert calls == ["shared", "other"]


def test_second_production_identity_request_is_rejected():
    with pytest.raises(RuntimeError, match="runtime-owned"):
        LearningOwner(owner_id="caller-controlled")
    test_owner = LearningOwner(owner_id="isolated", isolated_test_owner=True)
    assert test_owner.owner_id == "isolated"


def test_static_production_learning_construction_is_canonical_only():
    production_roots = [Path("src")]
    constructors = []
    for root in production_roots:
        for path in root.rglob("*.py"):
            if "tests" in path.parts or path.name == "README.py":
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = ""
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name in {"LearningOwner", "UnifiedLearningSystem"}:
                        constructors.append((str(path), node.lineno, name))
    assert constructors == [("src/vulcan/runtime/container.py", 208, "LearningOwner")]


def test_dependency_light_owner_import_keeps_heavy_modules_unloaded():
    code = """
import sys
import vulcan.learning_owner
heavy = {'torch','numpy','sklearn','networkx','fastapi','aiohttp'}
loaded = sorted(name for name in heavy if name in sys.modules)
print(','.join(loaded))
raise SystemExit(1 if loaded else 0)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run([sys.executable, "-c", code], cwd="/tmp", env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
