from __future__ import annotations

import os
from pathlib import Path

import pytest

from vulcan.runtime.settings import SettingsError, load_runtime_settings, validate_durable_root

SECRET = "AbCdEfGhIjKlMnOpQrStUvWxYz7890+/safe"
APPROVAL = "YnOpQrStUvWxAbCdEfGhIjKlM7890+/approval"


def _env(root: Path, **overrides: str) -> dict[str, str]:
    env = {
        "VULCAN_ENV": "production",
        "VULCAN_JWT_SECRET": SECRET,
        "VULCAN_APPROVAL_HMAC_SECRET": APPROVAL,
        "VULCAN_RUNTIME_DURABLE_ROOT": str(root),
    }
    env.update(overrides)
    return env


def test_runtime_settings_places_canonical_owners_under_durable_root(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    root.mkdir(mode=0o700)
    settings = load_runtime_settings(_env(root))

    assert settings.durable_root == root.resolve()
    assert settings.memory_sqlite_path == root.resolve() / "memory" / "memory.sqlite"
    assert settings.durable_paths.audit == root.resolve() / "audit"
    assert settings.durable_paths.alignment == root.resolve() / "alignment"
    assert settings.durable_paths.domains == root.resolve() / "domains"
    assert settings.durable_paths.governed_memory == root.resolve() / "memory"
    assert settings.durable_paths.learning_outbox == root.resolve() / "learning" / "outbox"
    assert settings.durable_paths.csiu == root.resolve() / "csiu"
    assert settings.durable_paths.approval == root.resolve() / "approval"
    assert settings.durable_paths.improvement == root.resolve() / "improvement"


def test_runtime_validation_as_uid_1001_writes_fsyncs_and_reopens(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    root.mkdir(mode=0o700)
    result = validate_durable_root(root, expected_uid=os.getuid() if os.name != "nt" else None, expected_gid=os.getgid() if os.name != "nt" else None)
    assert result.category == "ok"

    marker = root / "audit" / "restart.marker"
    marker.write_text("preserved", encoding="utf-8")
    with marker.open("r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    assert (root / "audit" / "restart.marker").read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize("bad", ["/tmp", ".", "relative/path"])
def test_durable_root_rejects_tmp_source_tree_and_relative_paths(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SettingsError):
        load_runtime_settings(_env(tmp_path / "unused", VULCAN_RUNTIME_DURABLE_ROOT=bad))


def test_durable_root_rejects_symlink_components(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SettingsError, match="symlinked"):
        load_runtime_settings(_env(link / "state"))


def test_validate_durable_root_fails_closed_for_read_only_or_missing_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "durable"
    root.mkdir(mode=0o700)

    def deny_fsync(_fd: int) -> None:
        raise OSError("fsync unavailable")

    monkeypatch.setattr(os, "fsync", deny_fsync)
    result = validate_durable_root(root)
    assert result.category == "fsync_unavailable"
    assert str(root) not in result.public_message


def test_dockerfile_declares_canonical_volume_uid_and_restrictive_permissions() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "VULCAN_RUNTIME_DURABLE_ROOT=/var/lib/vulcan" in dockerfile
    assert "VOLUME [\"/var/lib/vulcan\"]" in dockerfile
    assert "chown -R graphix:graphix /var/lib/vulcan" in dockerfile
    assert "chmod 0700 /var/lib/vulcan" in dockerfile


def test_compose_uses_single_canonical_durable_volume_and_ephemeral_cache() -> None:
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "vulcan_durable_data:" in compose
    assert "VULCAN_RUNTIME_DURABLE_ROOT=/var/lib/vulcan" in compose
    assert "VULCAN_RUNTIME_REPLICAS=1" in compose
    assert "VULCAN_CACHE_ROOT=/tmp/vulcan-cache" in compose
    assert "vulcan_durable_data:/var/lib/vulcan" in compose


def test_helm_templates_mount_canonical_pvc_env_and_single_replica_guard() -> None:
    values = Path("helm/vulcanami/values.yaml").read_text(encoding="utf-8")
    deployment = Path("helm/vulcanami/templates/deployment.yaml").read_text(encoding="utf-8")
    pvc = Path("helm/vulcanami/templates/pvc.yaml").read_text(encoding="utf-8")

    assert "durableRoot: /var/lib/vulcan" in values
    assert "replicaCount: 1" in values
    assert "enabled: false" in values
    assert "readOnlyRootFilesystem: true" in values
    assert "fail \"canonical durable-root sqlite backend requires replicaCount=1" in deployment
    assert "mountPath: {{ .Values.runtime.durableRoot }}" in deployment
    assert "claimName: {{ include \"vulcanami.fullname\" . }}-durable-root" in deployment
    assert "kind: PersistentVolumeClaim" in pvc
    assert "-durable-root" in pvc
