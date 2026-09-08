from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from vulcan.assurance.inventory import (
    InventoryConfig,
    _validated_truth_map,
    build_inventory,
    canonical_json,
    render_readme_capabilities,
    replace_readme_capabilities,
)

ROOT = Path(__file__).resolve().parents[2]


def _minimal_root(tmp_path: Path, component: dict[str, object]) -> Path:
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "docs" / "architecture").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    (tmp_path / "tests" / "test_component.py").write_text(
        "def test_evidence(): pass\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "architecture" / "contract.md").write_text(
        "# Contract\n", encoding="utf-8"
    )
    (tmp_path / "config" / "architecture-status.json").write_text(
        json.dumps({"schema_version": 1, "components": [component]}), encoding="utf-8"
    )
    (tmp_path / "config" / "capabilities.yaml").write_text(
        json.dumps({"schema_version": 1, "capabilities": []}), encoding="utf-8"
    )
    (tmp_path / "docs" / "documentation-status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [
                    {
                        "path": "docs/architecture/contract.md",
                        "status": "normative",
                        "purpose": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _component(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "component_id": "component.test",
        "package_path": "src",
        "sole_owner": "TestOwner",
        "authority_ceiling": "VALIDATED_CANDIDATE",
        "production_reachability": "not-reachable",
        "state_authority": "none",
        "snapshot_implementation": "none",
        "persistence_owner": "none",
        "audit_evidence": "test",
        "tests": ["tests/test_component.py"],
        "maturity": "M2",
        "maturity_evidence": ["evidence.txt"],
        "public_capability_claim": False,
        "capability_ids": [],
        "compatibility_status": "current",
        "removal_condition": "replace with a stricter test fixture",
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"production_reachability": "unknown"},
            "invalid production reachability",
        ),
        ({"maturity_evidence": []}, "maturity_evidence must not be empty"),
        (
            {"maturity": "M2", "public_capability_claim": True},
            "public capability claim below M3",
        ),
    ],
)
def test_truth_map_fails_closed_on_unsupported_claims(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    root = _minimal_root(tmp_path, _component(**override))
    with pytest.raises(ValueError, match=message):
        _validated_truth_map(root)


def test_truth_map_rejects_duplicate_component_owners(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path, _component())
    manifest = json.loads((root / "config" / "architecture-status.json").read_text())
    manifest["components"].append(_component(sole_owner="AnotherOwner"))
    (root / "config" / "architecture-status.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate component owner declaration"):
        _validated_truth_map(root)


def test_truth_map_rejects_overlapping_paths_with_different_owners(
    tmp_path: Path,
) -> None:
    root = _minimal_root(tmp_path, _component(package_path="src"))
    (root / "src" / "nested.py").write_text("# nested\n", encoding="utf-8")
    manifest = json.loads((root / "config" / "architecture-status.json").read_text())
    manifest["components"].append(
        _component(
            component_id="component.nested",
            package_path="src/nested.py",
            sole_owner="AnotherOwner",
        )
    )
    (root / "config" / "architecture-status.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="overlapping component owners"):
        _validated_truth_map(root)


def test_every_architecture_document_has_one_status() -> None:
    _, documents = _validated_truth_map(ROOT)
    indexed = {item["path"] for item in documents}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "docs" / "architecture").rglob("*")
        if path.is_file()
    }
    assert indexed == actual | {
        "CURRENT_DIRECTION.md",
        "AGENTS.md",
        "docs/roadmap/constitutional-convergence-plan.md",
    }


def test_generated_inventory_is_deterministic_and_current() -> None:
    generated = json.loads(
        (ROOT / "docs" / "generated" / "architecture-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    components, documents = _validated_truth_map(ROOT)
    assert generated["architecture_status"] == components
    assert generated["documentation_status"] == documents
    assert all(item["maturity_evidence_sha256"] for item in components)


def test_minimal_inventory_generation_is_deterministic(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path, _component())
    first = build_inventory(InventoryConfig(root))
    second = build_inventory(InventoryConfig(root))
    assert canonical_json(first) == canonical_json(second)


def test_readme_public_claims_are_generated_from_m3_canonical_records() -> None:
    components, _ = _validated_truth_map(ROOT)
    public = [item for item in components if item.get("public_capability_claim")]
    assert public
    assert all(int(str(item["maturity"])[1:]) >= 3 for item in public)
    assert all(item["production_reachability"] == "canonical" for item in public)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generated = json.loads(
        (ROOT / "docs" / "generated" / "architecture-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        replace_readme_capabilities(readme, render_readme_capabilities(generated))
        == readme
    )


def test_readme_capability_drift_is_detected() -> None:
    generated = json.loads(
        (ROOT / "docs" / "generated" / "architecture-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tampered = readme.replace("**M3 — Canonical**", "**M2 — Production**", 1)
    assert (
        replace_readme_capabilities(tampered, render_readme_capabilities(generated))
        != tampered
    )


def test_truth_map_rejects_traversal_and_noncanonical_m3(tmp_path: Path) -> None:
    root = _minimal_root(tmp_path / "traversal", _component(package_path="../outside"))
    with pytest.raises(ValueError, match="normalized repository-relative path"):
        _validated_truth_map(root)

    root = _minimal_root(tmp_path / "maturity", _component(maturity="M3"))
    with pytest.raises(ValueError, match=r"M3\+ component is not canonical"):
        _validated_truth_map(root)


def test_inventory_rejects_undeclared_deployment_entrypoint(tmp_path: Path) -> None:
    root = _minimal_root(
        tmp_path,
        _component(
            package_path="src/known.py",
            production_reachability="canonical",
            maturity="M3",
        ),
    )
    (root / "src" / "known.py").write_text("# declared\n", encoding="utf-8")
    dockerfile = root / "docker" / "api" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text('CMD ["uvicorn", "undeclared.app:app"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown production entrypoint component"):
        build_inventory(InventoryConfig(root))


def test_readme_generated_region_requires_unique_markers() -> None:
    with pytest.raises(ValueError, match="exactly one generated capability block"):
        replace_readme_capabilities("# no markers\n", "generated")


def test_license_and_canonical_run_metadata_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["license"] == "GPL-3.0-only"
    assert project["license-files"] == ["LICENSE"]
    assert (
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        in project["classifiers"]
    )
    assert any(
        requirement.startswith("uvicorn>=")
        for requirement in project["optional-dependencies"]["server"]
    )
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in license_text[:100]
    assert "Version 3, 29 June 2007" in license_text[:150]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m uvicorn vulcan.runtime.app:app" in readme
    assert "GNU General Public License version 3" in readme
