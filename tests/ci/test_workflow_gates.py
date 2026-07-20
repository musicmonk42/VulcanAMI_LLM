from __future__ import annotations

from pathlib import Path

from scripts.ci.workflow_lint import lint_file

ROOT = Path(__file__).resolve().parents[2]


def test_required_workflows_have_no_successful_fallback_bypass() -> None:
    for rel in (".github/workflows/ci.yml", ".github/workflows/security.yml", ".github/workflows/docker.yml"):
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
