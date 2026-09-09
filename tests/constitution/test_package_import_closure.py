from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from scripts.ci.check_production_imports import check
from vulcan.platform import (
    UnsupportedServingPlatform,
    require_canonical_serving_platform,
)


def test_production_import_graph_is_closed() -> None:
    visited, errors = check()
    assert "vulcan.runtime.app" in visited
    assert errors == []


def test_policy_denies_legacy_and_research_authorities() -> None:
    policy = json.loads(Path("config/production-import-policy.json").read_text())
    denied = set(policy["denied_prefixes"])
    assert {
        "src.vulcan",
        "vulcan.orchestrator",
        "vulcan.runtime.semantic",
        "vulcan.npt",
    } <= denied


def test_src_vulcan_identity_fails_closed() -> None:
    spec = importlib.util.spec_from_file_location(
        "src.vulcan",
        Path("src/vulcan/__init__.py"),
        submodule_search_locations=[str(Path("src/vulcan"))],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError, match="sole package identity"):
        spec.loader.exec_module(module)


def test_non_linux_serving_fails_before_composition() -> None:
    with pytest.raises(UnsupportedServingPlatform, match="Linux/Docker"):
        require_canonical_serving_platform("win32")
    require_canonical_serving_platform("linux")


def test_release_evidence_root_requires_absolute_nonsymlink(
    tmp_path, monkeypatch
) -> None:
    from vulcan.runtime.capabilities import release_evidence_root

    monkeypatch.setenv("VULCAN_RELEASE_EVIDENCE_ROOT", "relative")
    with pytest.raises(ValueError, match="invalid release evidence root"):
        release_evidence_root()
    target = tmp_path / "evidence"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("VULCAN_RELEASE_EVIDENCE_ROOT", str(link))
    with pytest.raises(ValueError, match="invalid release evidence root"):
        release_evidence_root()
    monkeypatch.setenv("VULCAN_RELEASE_EVIDENCE_ROOT", str(target))
    assert release_evidence_root() == target.resolve()


def test_installed_runtime_uses_explicit_release_evidence_root(
    tmp_path, monkeypatch
) -> None:
    from vulcan.runtime.capabilities import load_capability_registry

    referenced = (
        "config/capabilities.yaml",
        "config/architecture-status.json",
        "tests/security/test_language_contracts.py",
        "docs/architecture/ami-invariants.yaml",
        "docs/architecture/adr-006-local-language-interface.md",
        "docs/governance/controls.yaml",
        "docs/governance/impact-assessment.yaml",
    )
    for name in referenced:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    monkeypatch.setenv("VULCAN_RELEASE_EVIDENCE_ROOT", str(tmp_path))
    registry = load_capability_registry()
    assert "cap.bounded_arithmetic" in registry.records
