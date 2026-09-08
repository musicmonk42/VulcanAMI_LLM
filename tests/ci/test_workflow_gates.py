from __future__ import annotations

from pathlib import Path

from scripts.ci.verify_dependency_inputs import validate_hashed_requirements
from scripts.ci.workflow_lint import lint_file

ROOT = Path(__file__).resolve().parents[2]


def test_required_workflows_have_no_successful_fallback_bypass() -> None:
    for rel in (
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/docker.yml",
    ):
        assert lint_file(ROOT / rel) == []


def test_negative_fixture_failing_scanner_step_is_not_masked(tmp_path: Path) -> None:
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        """
name: bad
jobs:
  dependency-light-unit-contract:
    steps:
      - uses: actions/checkout@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      - run: python failing_scanner.py || true
      - run: python scripts/ci/write_evidence.py --job bad --command bad --output evidence/bad.json
  full-integration: {steps: []}
  architecture-fitness: {steps: []}
  static-typing: {steps: []}
  lint-format: {steps: []}
  optimized-python: {steps: []}
""",
        encoding="utf-8",
    )
    errors = lint_file(workflow)
    assert any("|| true" in error for error in errors)


def test_negative_fixture_unpinned_action_is_rejected(tmp_path: Path) -> None:
    workflow = tmp_path / "security.yml"
    workflow.write_text(
        """
name: bad
jobs:
  secret-scan:
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/ci/write_evidence.py --job secret --command secret --output evidence/secret.json
  sast: {steps: []}
  dependency-vulnerability-policy: {steps: []}
""",
        encoding="utf-8",
    )
    errors = lint_file(workflow)
    assert any("not pinned" in error for error in errors)


def test_negative_fixture_empty_sarif_substitution_is_rejected(tmp_path: Path) -> None:
    workflow = tmp_path / "docker.yml"
    workflow.write_text(
        """
name: bad
jobs:
  image-e2e:
    steps:
      - uses: actions/checkout@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      - run: echo 'empty SARIF'; touch trivy-results.sarif
      - run: python scripts/ci/write_evidence.py --job image --command image --output evidence/image.json
  supply-chain-evidence: {steps: []}
""",
        encoding="utf-8",
    )
    errors = lint_file(workflow)
    assert any("empty SARIF" in error or "touch trivy" in error for error in errors)


def test_local_gate_exposes_every_required_real_check() -> None:
    from scripts.ci.run_constitutional_gate import CHECKS

    assert set(CHECKS) == {
        "workflow",
        "unit",
        "typing",
        "format",
        "integration",
        "security",
    }
    flattened = [
        " ".join(command) for commands in CHECKS.values() for command in commands
    ]
    assert any(" -m mypy " in f" {command} " for command in flattened)
    assert any(" -m black --check " in f" {command} " for command in flattened)
    assert any(" -m isort --check-only " in f" {command} " for command in flattened)
    assert any("test_authoritative_episode_path.py" in command for command in flattened)
    assert any(" -O -m pytest " in f" {command} " for command in flattened)
    source = (ROOT / "scripts/ci/run_constitutional_gate.py").read_text(
        encoding="utf-8"
    )
    assert "--bootstrap" in source
    assert "--require-hashes" in source


def test_required_jobs_use_python_311_and_hash_locked_install() -> None:
    for rel in (
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/docker.yml",
        ".github/workflows/runtime-e2e.yml",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c" in text
        assert "3.11" in text
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "--require-hashes -r requirements-constitutional.txt" in ci
    assert "pip install pytest" not in ci


def test_legacy_provider_and_scalability_workflows_are_not_required() -> None:
    for rel in (
        ".github/workflows/security-smoke.yml",
        ".github/workflows/scalability_test.yml",
        ".github/workflows/azure-kubernetes-service-helm.yml",
        ".github/workflows/tencent.yml",
        ".github/workflows/deploy.yml",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        trigger = text[text.index("on:") : text.index("jobs:")]
        assert "workflow_dispatch:" in trigger
        assert "pull_request:" not in trigger
        assert "push:" not in trigger


def test_constitutional_lock_rejects_unhashed_and_unpinned_requirements() -> None:
    assert validate_hashed_requirements("--only-binary=:all:\npytest==9.0.1\n")
    assert validate_hashed_requirements(
        "--only-binary=:all:\npytest>=9 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    )
    assert validate_hashed_requirements(
        "--only-binary=:all:\npytest==9.0.1 \\\n    --hash=sha256:not-a-digest\n"
    )


def test_committed_constitutional_lock_is_fully_hashed() -> None:
    lock = (ROOT / "requirements-constitutional.txt").read_text(encoding="utf-8")
    assert validate_hashed_requirements(lock) == []
