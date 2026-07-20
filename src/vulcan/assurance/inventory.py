"""Deterministic architecture inventory generator using AST, never imports targets."""
from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

PRODUCTION_ENTRYPOINT_FILES = ("docker-compose.prod.yml", "docker/api/Dockerfile", "helm/vulcanami/templates/deployment.yaml")
PYTHON_ROOTS = ("src", "scripts")
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "src/data", "evolution_champions", "output"}
ALLOWLISTED_COMPATIBILITY_SHIMS = {
    "src/vulcan/runtime/legacy_adapter.py",
    "src/vulcan/runtime/app.py",
}
OWNERSHIP_TERMS = ("runtime", "audit", "memory", "alignment", "domain", "learning", "csiu", "language", "reasoner", "world_model", "self_improvement")


@dataclass(frozen=True, slots=True)
class InventoryConfig:
    root: Path
    max_files: int = 6000
    max_bytes_per_file: int = 1_000_000


def canonical_json(data: dict[str, object]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def inventory_digest(data: dict[str, object]) -> str:
    without_digest = {key: value for key, value in data.items() if key != "digest"}
    return hashlib.sha256(canonical_json(without_digest)).hexdigest()


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDED_PARTS) or any(str(path).startswith(part + "/") for part in EXCLUDED_PARTS)


def bounded_files(root: Path, patterns: Sequence[str], max_files: int) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            rel = path.relative_to(root)
            if path.is_file() and not _is_excluded(rel):
                found.append(rel)
                if len(found) > max_files:
                    raise RuntimeError("architecture inventory traversal exceeded file bound")
    return sorted(set(found), key=lambda p: p.as_posix())


def _line_ref(path: Path, node: ast.AST) -> dict[str, object]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return {"path": path.as_posix(), "line_start": start, "line_end": end}


def _literal_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _decorator_route(dec: ast.AST) -> tuple[str, str] | None:
    if not isinstance(dec, ast.Call):
        return None
    name = _call_name(dec.func)
    method = name.rsplit(".", 1)[-1]
    if method not in {"get", "post", "put", "delete", "patch", "route", "websocket"}:
        return None
    route = _literal_text(dec.args[0]) if dec.args else None
    if route is None:
        return None
    return method.upper(), route


def _is_entrypoint_guard(node: ast.If) -> bool:
    left = getattr(node.test, "left", None)
    comparators = getattr(node.test, "comparators", [])
    return isinstance(left, ast.Name) and left.id == "__name__" and any(_literal_text(item) == "__main__" for item in comparators)


def scan_python_file(root: Path, rel: Path, max_bytes: int) -> dict[str, list[dict[str, object]]]:
    full = root / rel
    raw = full.read_bytes()
    if len(raw) > max_bytes:
        return {"skipped": [{"path": rel.as_posix(), "reason": "file_too_large"}]}
    text = raw.decode("utf-8", errors="replace")
    tree = ast.parse(text, filename=rel.as_posix())
    result: dict[str, list[dict[str, object]]] = {
        "entrypoints": [], "routes": [], "imports": [], "singletons": [], "workers": [],
        "fallbacks": [], "mocks": [], "persistence": [], "env_readers": [], "capability_claims": [], "owners": []
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"vulcan", "src.vulcan"} or alias.name.startswith(("vulcan.", "src.vulcan.")):
                    result["imports"].append({**_line_ref(rel, node), "module": alias.name})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in {"vulcan", "src.vulcan"} or node.module.startswith(("vulcan.", "src.vulcan.")):
                result["imports"].append({**_line_ref(rel, node), "module": node.module})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                route = _decorator_route(dec)
                if route:
                    method, path = route
                    result["routes"].append({**_line_ref(rel, node), "method": method, "route": path, "handler": node.name})
            lowered = node.name.lower()
            if lowered.startswith(("get_", "create_")) and any(term in lowered for term in ("singleton", "global", "instance")):
                result["singletons"].append({**_line_ref(rel, node), "name": node.name})
        elif isinstance(node, ast.If) and _is_entrypoint_guard(node):
            result["entrypoints"].append({**_line_ref(rel, node), "kind": "python_main_guard"})
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name.endswith(("Thread", "Process")) or name.endswith(("create_task", "run_in_executor")):
                result["workers"].append({**_line_ref(rel, node), "call": name})
            if name in {"os.getenv", "os.environ.get"} or name.endswith("getenv"):
                key = _literal_text(node.args[0]) if node.args else None
                result["env_readers"].append({**_line_ref(rel, node), "call": name, "key": key})
            if "sqlite" in name.lower() or name.endswith(("flock", "lockf")):
                result["persistence"].append({**_line_ref(rel, node), "kind": "call", "value": name})
            if name.endswith("MagicMock"):
                result["mocks"].append({**_line_ref(rel, node), "kind": "MagicMock"})
        elif isinstance(node, ast.ExceptHandler):
            if node.type is not None and _call_name(node.type) == "Exception":
                result["fallbacks"].append({**_line_ref(rel, node), "kind": "except_exception"})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(token in lowered for token in ("sqlite", ".db", ".sqlite", "flock")):
                result["persistence"].append({**_line_ref(rel, node), "kind": "literal", "value": node.value[:160]})
            if "capability" in lowered and any(marker in lowered for marker in ("available", "enabled", "supported", "production")):
                result["capability_claims"].append({**_line_ref(rel, node), "text": node.value[:160]})
    rel_text = rel.as_posix().lower()
    for term in OWNERSHIP_TERMS:
        if term in rel_text.replace("-", "_"):
            result["owners"].append({"path": rel.as_posix(), "owner_domain": term, "reachability": "unknown"})
    return result


def scan_text_entrypoints(root: Path) -> dict[str, list[dict[str, object]]]:
    items: dict[str, list[dict[str, object]]] = {"docker": [], "compose": [], "helm": []}
    for rel in bounded_files(root, ["docker*/**/Dockerfile", "Dockerfile", "docker-compose*.yml", "docker-compose*.yaml", "helm/**/*.yaml", "helm/**/*.yml"], 1000):
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            lower = stripped.lower()
            if rel.name == "Dockerfile" and (stripped.startswith("CMD") or stripped.startswith("ENTRYPOINT")):
                items["docker"].append({"path": rel.as_posix(), "line_start": idx, "line_end": idx, "directive": stripped.split(None, 1)[0], "value": stripped})
            if rel.name.startswith("docker-compose") and re.match(r"^(command|entrypoint|image):", stripped):
                items["compose"].append({"path": rel.as_posix(), "line_start": idx, "line_end": idx, "key": stripped.split(":", 1)[0], "value": stripped})
            if rel.parts and rel.parts[0] == "helm" and any(key in lower for key in ("image:", "command:", "args:", "livenessprobe:", "readinessprobe:")):
                items["helm"].append({"path": rel.as_posix(), "line_start": idx, "line_end": idx, "key": stripped.split(":", 1)[0], "value": stripped})
    return items


def _reachability(path: str, production_modules: set[str]) -> str:
    if "/tests/" in f"/{path}" or path.startswith("tests/"):
        return "test-only"
    if any(part in path for part in ("archive", "examples/", "docs/", "scripts/")):
        return "research-only"
    module = path.removesuffix(".py").replace("/", ".")
    if module.startswith("src."):
        module = module[4:]
    if module in production_modules or any(module.startswith(prefix + ".") for prefix in production_modules):
        return "production-reachable"
    return "unknown"


def build_inventory(config: InventoryConfig) -> dict[str, object]:
    root = config.root.resolve()
    py_files = bounded_files(root, ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py"], config.max_files)
    text_entrypoints = scan_text_entrypoints(root)
    production_modules = {"vulcan.runtime.app", "vulcan.runtime"}
    inventory: dict[str, object] = {
        "schema_version": 1,
        "generator": "scripts/architecture_inventory.py",
        "bounded_traversal": {"max_files": config.max_files, "max_bytes_per_file": config.max_bytes_per_file},
        "production_entrypoint_files": list(PRODUCTION_ENTRYPOINT_FILES),
        "allowlisted_compatibility_shims": sorted(ALLOWLISTED_COMPATIBILITY_SHIMS),
        "entrypoints": [], "routes": [], "import_identities": [], "singletons": [], "workers": [],
        "fallbacks": [], "mocks": [], "persistence": [], "environment_readers": [], "capability_claims": [],
        "ownership_graph": [], "text_entrypoints": text_entrypoints, "findings": [],
    }
    for rel in py_files:
        scanned = scan_python_file(root, rel, config.max_bytes_per_file)
        reachability = _reachability(rel.as_posix(), production_modules)
        for key, target in (("entrypoints", "entrypoints"), ("routes", "routes"), ("imports", "import_identities"), ("singletons", "singletons"), ("workers", "workers"), ("fallbacks", "fallbacks"), ("mocks", "mocks"), ("persistence", "persistence"), ("env_readers", "environment_readers"), ("capability_claims", "capability_claims")):
            for item in scanned.get(key, []):
                item = dict(item)
                item["reachability"] = reachability
                if target in {"fallbacks", "mocks"}:
                    item["review"] = "allowlisted" if item["path"] in ALLOWLISTED_COMPATIBILITY_SHIMS else "unreviewed"
                inventory[target].append(item)
        for item in scanned.get("owners", []):
            item = dict(item)
            item["reachability"] = reachability
            inventory["ownership_graph"].append(item)
    modules = {item["module"] for item in inventory["import_identities"]}
    if any(str(module).startswith("src.vulcan") for module in modules) and any(str(module).startswith("vulcan") for module in modules):
        inventory["findings"].append({"id": "duplicate_package_identities", "severity": "high", "description": "Both src.vulcan and vulcan import identities are present."})
    for path in sorted({item["path"] for item in inventory["routes"] if any(name in item["path"].lower() for name in ("orchestrator", "world_model", "unified_chat", "runtime"))}):
        inventory["findings"].append({"id": "competing_cognitive_orchestrator", "severity": "medium", "path": path})
    if any(item.get("reachability") == "production-reachable" and item.get("review") == "unreviewed" for item in inventory["mocks"]):
        inventory["findings"].append({"id": "production_mock", "severity": "critical", "description": "Production-reachable MagicMock use is not allowlisted."})
    if any(item.get("reachability") == "production-reachable" and item.get("review") == "unreviewed" for item in inventory["fallbacks"]):
        inventory["findings"].append({"id": "production_except_exception_fallback", "severity": "high", "description": "Production-reachable broad Exception fallbacks require review."})
    for key, value in list(inventory.items()):
        if isinstance(value, list):
            inventory[key] = sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
    inventory["digest"] = inventory_digest(inventory)
    return inventory


def render_markdown(inventory: dict[str, object]) -> str:
    lines = ["# Generated architecture inventory", "", f"Digest: `{inventory['digest']}`", "", "This file is generated from `docs/generated/architecture-inventory.json`.", ""]
    for section in ("entrypoints", "routes", "import_identities", "singletons", "workers", "fallbacks", "mocks", "persistence", "environment_readers", "capability_claims", "ownership_graph"):
        items = inventory[section]
        lines.extend([f"## {section.replace('_', ' ').title()}", "", f"Count: {len(items)}", ""])
        for item in items[:50]:
            path = item.get("path", "")
            line = item.get("line_start", "")
            detail = ", ".join(f"{k}={item[k]!r}" for k in sorted(item) if k not in {"path", "line_start", "line_end"})
            lines.append(f"- `{path}:{line}` {detail}")
        if len(items) > 50:
            lines.append(f"- ... {len(items) - 50} more in JSON")
        lines.append("")
    lines.extend(["## Text Entrypoints", ""])
    for kind in sorted(inventory["text_entrypoints"]):
        items = inventory["text_entrypoints"][kind]
        lines.append(f"### {kind}")
        for item in items:
            lines.append(f"- `{item['path']}:{item['line_start']}` {item.get('value', '')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
