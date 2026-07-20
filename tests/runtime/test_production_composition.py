from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from vulcan.runtime.composition import DevelopmentStubDeployment, compose_runtime
from vulcan.runtime.errors import StartupErrorCategory, StartupFailure
from vulcan.runtime.settings import RuntimeSettings, VulcanEnvironment, durable_root_paths, OpaqueSecret, SecretSource


def settings(tmp_path: Path, *, env: VulcanEnvironment = VulcanEnvironment.production, stub: bool = False) -> RuntimeSettings:
    root = (tmp_path / "durable").resolve()
    root.mkdir()
    return RuntimeSettings(
        environment=env,
        jwt_issuer="vulcan",
        jwt_audience="vulcan-runtime",
        jwt_secret=OpaqueSecret(SecretSource.direct, "A" * 40 + "1!bcdefgh", "VULCAN_JWT_SECRET"),
        durable_root=root,
        durable_paths=durable_root_paths(root),
        approval_hmac_secret=OpaqueSecret(SecretSource.direct, "B" * 40 + "1!cdefghi", "VULCAN_APPROVAL_HMAC_SECRET"),
        memory_sqlite_path=root / "memory" / "memory.sqlite",
        development_stub_mode=stub,
    )


class DummyOwner:
    owner_id = "dummy-owner"
    capability = SimpleNamespace(value="shadow")
    def readiness(self): return True
    def close(self): return None
    def capabilities(self): return ()
    def capability_matrix(self): return ()


def lightweight_container(monkeypatch: pytest.MonkeyPatch) -> None:
    import vulcan.runtime.container as container
    monkeypatch.setattr(container, "compose_governed_memory", lambda config: DummyOwner())
    monkeypatch.setattr(container, "CanonicalAudit", lambda path: DummyOwner())
    monkeypatch.setattr(container, "AlignmentRegistry", lambda path, audit=None: DummyOwner())
    monkeypatch.setattr(container, "PersistentDomainRegistry", lambda path, audit=None: DummyOwner())
    monkeypatch.setattr(container, "compose_self_improvement_runtime", lambda **kwargs: SimpleNamespace(drive=DummyOwner(), capabilities=lambda: (), close=lambda: None))
    monkeypatch.setattr(container, "ShadowLinUCBToolBandit", lambda: DummyOwner())
    monkeypatch.setattr(container, "LearningOwner", lambda **kwargs: DummyOwner())


class GoodWorld:
    def readiness(self):
        return True


class GoodSafety:
    def readiness(self):
        return True

    def validate(self, *args, **kwargs):
        return True


class Deployment:
    def __init__(self, config=None, *, world=GoodWorld(), safety=GoodSafety()):
        self.collective = SimpleNamespace(deps=SimpleNamespace(world_model=world, safety_validator=safety, continual=None))

    def readiness(self):
        return True


def install_deployment(monkeypatch: pytest.MonkeyPatch, deployment_cls=Deployment) -> None:
    config = types.ModuleType("vulcan.config")
    config.get_config = lambda: {"authoritative": True}
    deployment = types.ModuleType("vulcan.orchestrator.deployment")
    deployment.ProductionDeployment = deployment_cls
    monkeypatch.setitem(sys.modules, "vulcan.config", config)
    monkeypatch.setitem(sys.modules, "vulcan.orchestrator.deployment", deployment)


@pytest.mark.asyncio
async def test_real_composition_contains_no_fallback_types(monkeypatch, tmp_path):
    install_deployment(monkeypatch)
    lightweight_container(monkeypatch)
    runtime = compose_runtime(settings(tmp_path))
    try:
        assert "Fallback" not in type(runtime.deployment).__name__
        assert "Fallback" not in type(runtime.world_state).__name__
        assert "Fallback" not in type(runtime.safety).__name__
        await runtime.readiness()
    finally:
        await runtime.close()


def test_missing_world_fails_with_original_category(monkeypatch, tmp_path):
    class MissingWorld(Deployment):
        def __init__(self, config=None):
            super().__init__(config, world=None)

    install_deployment(monkeypatch, MissingWorld)
    lightweight_container(monkeypatch)
    with pytest.raises(StartupFailure) as excinfo:
        compose_runtime(settings(tmp_path))
    assert excinfo.value.category is StartupErrorCategory.WORLD_MISSING
    assert excinfo.value.public_code == "world_missing"


def test_missing_safety_fails_with_original_category(monkeypatch, tmp_path):
    class MissingSafety(Deployment):
        def __init__(self, config=None):
            super().__init__(config, safety=None)

    install_deployment(monkeypatch, MissingSafety)
    lightweight_container(monkeypatch)
    with pytest.raises(StartupFailure) as excinfo:
        compose_runtime(settings(tmp_path))
    assert excinfo.value.category is StartupErrorCategory.SAFETY_MISSING


def test_deployment_constructor_failure_preserves_cause(monkeypatch, tmp_path):
    class Broken:
        def __init__(self, config=None):
            raise ValueError("malicious constructor escalation")

    install_deployment(monkeypatch, Broken)
    with pytest.raises(StartupFailure) as excinfo:
        compose_runtime(settings(tmp_path))
    assert excinfo.value.category is StartupErrorCategory.DEPLOYMENT_CONSTRUCTION_FAILED
    assert isinstance(excinfo.value.cause, ValueError)


def test_development_stub_is_explicit_and_not_production_ready(monkeypatch, tmp_path):
    lightweight_container(monkeypatch)
    runtime = compose_runtime(settings(tmp_path, env=VulcanEnvironment.development, stub=True))
    assert isinstance(runtime.deployment, DevelopmentStubDeployment)
    assert runtime.deployment.production_ready is False
    with pytest.raises(RuntimeError):
        import asyncio
        asyncio.run(runtime.readiness())


def test_production_forbids_development_stub_setting(tmp_path):
    from vulcan.runtime.settings import SettingsError, load_runtime_settings
    root = (tmp_path / "prod-root").resolve()
    env = {
        "VULCAN_ENV": "production",
        "VULCAN_RUNTIME_DURABLE_ROOT": str(root),
        "VULCAN_JWT_SECRET": "Abcd1234!" * 5,
        "VULCAN_APPROVAL_HMAC_SECRET": "Bcde1234!" * 5,
        "VULCAN_DEVELOPMENT_STUB_MODE": "true",
    }
    with pytest.raises(SettingsError, match="development stub mode"):
        load_runtime_settings(env)


def test_server_dependency_absence_fails_import_without_fake_frameworks():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", "import vulcan.runtime.app"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode == 0:
        pytest.skip("server dependencies are installed in this environment")
    assert "ModuleNotFoundError" in result.stderr
    assert "fastapi" in result.stderr or "pydantic" in result.stderr
