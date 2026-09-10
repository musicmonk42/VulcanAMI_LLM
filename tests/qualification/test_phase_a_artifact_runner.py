from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path("scripts/qualification/qualify_phase_a_artifact.py")
spec = importlib.util.spec_from_file_location("phase_a_qualification", PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_mutable_image_references_are_rejected():
    for value in ("vulcan:latest", "sha256:" + "a" * 64, "vulcan@sha256:short"):
        with pytest.raises(ValueError, match="repository@sha256"):
            module.immutable_image(value)
    assert module.immutable_image("registry.example/vulcan@sha256:" + "a" * 64)


def test_scenario_execution_is_argv_based_and_subject_bound(tmp_path):
    subject = "registry.example/vulcan@sha256:" + "b" * 64
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"wheel")
    observation = module.execute(
        ["python", "-c", "print('ok')", "{subject}", "{wheel}"],
        subject=subject,
        wheel=wheel,
    )
    assert observation["passed"] is True
    assert observation["exit_status"] == 0
    assert observation["stdout_sha256"] == module.digest(b"ok\n")
    assert "stdout" not in observation and "stderr" not in observation
    with pytest.raises(ValueError, match="must name"):
        module.execute(["python", "-c", "pass"], subject=subject, wheel=wheel)


def test_catalog_and_binding_sets_are_exact():
    assert "installed-normal" in module.REQUIRED_SCENARIOS
    assert "installed-optimized" in module.REQUIRED_SCENARIOS
    assert "sbom" in module.REQUIRED_SCENARIOS
    assert "authority_verification" in module.REQUIRED_BINDINGS
    assert "actor_binding_vectors" in module.REQUIRED_BINDINGS


def test_e2e_uses_admitted_arithmetic_contract_and_immutable_subject():
    script = Path("scripts/e2e/run_runtime_qualification.sh").read_text()
    assert '"message":"2 + 2"' in script
    assert 'd.get("response") == "The computed result is 4."' in script
    assert "IMAGE_TAG" not in script
    assert "@sha256:[0-9a-f]{64}" in script
    assert "--read-only" in script


def test_candidate_builder_refuses_without_container_engine(tmp_path, monkeypatch):
    builder_path = Path("scripts/qualification/build_candidate_images.py")
    builder_spec = importlib.util.spec_from_file_location(
        "candidate_builder", builder_path
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec.loader is not None
    # The script-local qualification helper is importable when invoked normally.
    import sys

    sys.path.insert(0, str(builder_path.parent))
    try:
        builder_spec.loader.exec_module(builder)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(builder.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(builder_path),
            "--base",
            "registry.example/python@sha256:" + "a" * 64,
            "--wheel",
            str(tmp_path / "candidate.whl"),
            "--wheel-sha256",
            module.digest(b"wheel"),
            "--wheelhouse",
            str(tmp_path / "wheelhouse"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    (tmp_path / "candidate.whl").write_bytes(b"wheel")
    (tmp_path / "wheelhouse").mkdir()
    with pytest.raises(RuntimeError, match="NOT_EXECUTED"):
        builder.main()
    assert not (tmp_path / "result.json").exists()
