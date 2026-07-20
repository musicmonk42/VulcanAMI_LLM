from __future__ import annotations

import json
from pathlib import Path

from vulcan.assurance.inventory import InventoryConfig, build_inventory, inventory_digest, render_markdown

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_JSON = ROOT / "docs" / "generated" / "architecture-inventory.json"
INVENTORY_MD = ROOT / "docs" / "generated" / "architecture-inventory.md"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fixture_inventory_detects_entrypoints_aliases_cycles_and_fallbacks(tmp_path: Path) -> None:
    write(tmp_path / "src" / "vulcan" / "runtime" / "app.py", """
from src.vulcan.foo import bar
from vulcan.world_model import thing
import os
from threading import Thread
from unittest.mock import MagicMock

@app.get('/health')
def health():
    return {'ok': True}

def get_global_instance():
    return object()

if __name__ == '__main__':
    print(os.getenv('PORT'))

try:
    Thread(target=health).start()
except Exception:
    MagicMock()
""")
    write(tmp_path / "src" / "vulcan" / "world_model" / "cycle_a.py", "from vulcan.world_model.cycle_b import b\n")
    write(tmp_path / "src" / "vulcan" / "world_model" / "cycle_b.py", "from vulcan.world_model.cycle_a import a\n")
    write(tmp_path / "docker" / "api" / "Dockerfile", "FROM python:3.12\nCMD [\"uvicorn\", \"vulcan.runtime.app:app\"]\n")
    write(tmp_path / "docker-compose.prod.yml", "services:\n  api:\n    image: vulcan:test\n    command: python -m vulcan.runtime.app\n")
    write(tmp_path / "helm" / "vulcanami" / "templates" / "deployment.yaml", "image: vulcan:test\ncommand: [\"python\"]\nlivenessProbe: {}\nreadinessProbe: {}\n")

    inventory = build_inventory(InventoryConfig(root=tmp_path, max_files=100, max_bytes_per_file=100_000))
    assert any(item["kind"] == "python_main_guard" for item in inventory["entrypoints"])
    assert any(item["module"].startswith("src.vulcan") for item in inventory["import_identities"])
    assert any(item["module"].startswith("vulcan") for item in inventory["import_identities"])
    assert any(item["id"] == "duplicate_package_identities" for item in inventory["findings"])
    assert any(item["kind"] == "except_exception" for item in inventory["fallbacks"])
    assert any(item["kind"] == "MagicMock" for item in inventory["mocks"])
    assert any(item["call"].endswith("Thread") for item in inventory["workers"])
    assert any(item["key"] == "PORT" for item in inventory["environment_readers"])


def test_repository_inventory_is_current_and_deterministic() -> None:
    expected = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    regenerated = build_inventory(InventoryConfig(root=ROOT))
    assert regenerated == expected
    assert expected["digest"] == inventory_digest(expected)
    assert render_markdown(expected) == INVENTORY_MD.read_text(encoding="utf-8")


def test_docker_compose_and_helm_entrypoints_are_separate() -> None:
    inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    text_entrypoints = inventory["text_entrypoints"]
    assert text_entrypoints["docker"], "Docker CMD/ENTRYPOINT directives must be represented"
    assert text_entrypoints["compose"], "Compose image/command/entrypoint keys must be represented"
    assert text_entrypoints["helm"], "Helm image/command/probe keys must be represented"
    assert {"docker", "compose", "helm"} == set(text_entrypoints)


def test_inventory_marks_reachability_and_known_architecture_findings() -> None:
    inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    reachability = {item["reachability"] for section in ("routes", "fallbacks", "mocks", "workers") for item in inventory[section]}
    assert {"production-reachable", "research-only", "test-only", "unknown"} & reachability
    finding_ids = {item["id"] for item in inventory["findings"]}
    assert "duplicate_package_identities" in finding_ids
    assert "competing_cognitive_orchestrator" in finding_ids
    assert "production_mock" not in finding_ids


def test_inventory_has_bounded_refs_and_no_arbitrary_import_side_effect_contract() -> None:
    inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    assert inventory["bounded_traversal"]["max_files"] <= 6000
    for section in ("entrypoints", "routes", "import_identities", "fallbacks", "persistence"):
        for item in inventory[section][:200]:
            assert item["path"]
            assert item["line_start"] >= 1
            assert item["line_end"] >= item["line_start"]
