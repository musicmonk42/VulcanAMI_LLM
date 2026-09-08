"""Deterministic architecture inventory generator using AST, never imports targets."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

PRODUCTION_ENTRYPOINT_FILES = (
    "docker-compose.prod.yml",
    "docker/api/Dockerfile",
    "helm/vulcanami/templates/deployment.yaml",
)
PYTHON_ROOTS = ("src", "scripts")
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "src/data",
    "evolution_champions",
    "output",
}
ALLOWLISTED_COMPATIBILITY_SHIMS = {
    "src/vulcan/runtime/legacy_adapter.py",
    "src/vulcan/runtime/app.py",
}
OWNERSHIP_TERMS = (
    "runtime",
    "audit",
    "memory",
    "alignment",
    "domain",
    "learning",
    "csiu",
    "language",
    "reasoner",
    "world_model",
    "self_improvement",
)
MATURITY_LEVELS = {f"M{level}": level for level in range(6)}
DOCUMENT_STATUSES = {
    "normative",
    "current",
    "superseded",
    "historical",
    "research-only",
}
AUTHORITY_LEVELS = {
    "NONE",
    "UNTRUSTED_PROPOSAL",
    "VALIDATED_CANDIDATE",
    "COMMITTED_BELIEF",
    "AUTHORIZED_PLAN",
    "EXECUTED_EFFECT",
}
REACHABILITY_LEVELS = {
    "canonical",
    "reachable-compatibility",
    "not-reachable",
}
COMPATIBILITY_STATUSES = {
    "canonical",
    "compatibility",
    "pre-canonical",
    "historical",
    "current",
}
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class InventoryConfig:
    root: Path
    max_files: int = 6000
    max_bytes_per_file: int = 1_000_000
    require_truth_map: bool = True


def canonical_json(data: dict[str, object]) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def inventory_digest(data: dict[str, object]) -> str:
    without_digest = {key: value for key, value in data.items() if key != "digest"}
    return hashlib.sha256(canonical_json(without_digest)).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON-compatible manifest while rejecting duplicate keys."""
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _repository_path(root: Path, value: object, *, field: str, file: bool) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX repository-relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != value:
        raise ValueError(
            f"{field} must be a normalized repository-relative path: {value!r}"
        )
    candidate = root / rel
    exists = candidate.is_file() if file else candidate.exists()
    if not exists:
        raise ValueError(
            f"{field} references missing {'file' if file else 'path'}: {value}"
        )
    return value


def _string_list(value: object, *, field: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate entries")
    return value


def _required_string(item: Mapping[str, object], field: str, component_id: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{component_id} requires non-empty {field}")
    if any(character in value for character in ("\r", "\n", "|")):
        raise ValueError(f"{component_id}.{field} contains unsafe Markdown characters")
    return value


def _validated_truth_map(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = _load_json(root / "config" / "architecture-status.json")
    docs_source = _load_json(root / "docs" / "documentation-status.json")
    if source.get("schema_version") != 1 or docs_source.get("schema_version") != 1:
        raise ValueError("unsupported architecture-status manifest schema version")
    components = source.get("components")
    documents = docs_source.get("documents")
    if not isinstance(components, list) or not isinstance(documents, list):
        raise ValueError(
            "architecture and documentation manifests require list entries"
        )

    ids: set[str] = set()
    capability_owners: dict[str, str] = {}
    component_paths: list[tuple[str, str, str]] = []
    validated_components: list[dict[str, object]] = []
    required = {
        "component_id",
        "package_path",
        "sole_owner",
        "authority_ceiling",
        "production_reachability",
        "state_authority",
        "snapshot_implementation",
        "persistence_owner",
        "audit_evidence",
        "tests",
        "maturity",
        "compatibility_status",
        "removal_condition",
        "maturity_evidence",
    }
    for raw in components:
        if not isinstance(raw, dict) or required - raw.keys():
            missing = (
                sorted(required - raw.keys())
                if isinstance(raw, dict)
                else sorted(required)
            )
            raise ValueError(f"component entry missing required fields: {missing}")
        item = dict(raw)
        component_id = _required_string(item, "component_id", "component")
        if _IDENTIFIER_RE.fullmatch(component_id) is None:
            raise ValueError(f"invalid component_id: {component_id}")
        if component_id in ids:
            raise ValueError(f"duplicate component owner declaration: {component_id}")
        ids.add(component_id)
        sole_owner = _required_string(item, "sole_owner", component_id)
        package_path = _repository_path(
            root, item["package_path"], field=f"{component_id}.package_path", file=False
        )
        for other_path, other_owner, other_id in component_paths:
            overlaps = (
                package_path == other_path
                or package_path.startswith(other_path.rstrip("/") + "/")
                or other_path.startswith(package_path.rstrip("/") + "/")
            )
            if overlaps and sole_owner != other_owner:
                raise ValueError(
                    f"overlapping component owners: {other_id} and {component_id}"
                )
        component_paths.append((package_path, sole_owner, component_id))
        authority = _required_string(item, "authority_ceiling", component_id)
        if authority not in AUTHORITY_LEVELS:
            raise ValueError(
                f"invalid authority ceiling for {component_id}: {authority}"
            )
        for field in (
            "state_authority",
            "snapshot_implementation",
            "persistence_owner",
            "audit_evidence",
            "removal_condition",
        ):
            _required_string(item, field, component_id)
        compatibility = _required_string(item, "compatibility_status", component_id)
        if compatibility not in COMPATIBILITY_STATUSES:
            raise ValueError(
                f"invalid compatibility status for {component_id}: {compatibility}"
            )
        maturity = str(item["maturity"])
        if maturity not in MATURITY_LEVELS:
            raise ValueError(f"unknown maturity for {component_id}: {maturity}")
        evidence = _string_list(
            item["maturity_evidence"],
            field=f"{component_id}.maturity_evidence",
            nonempty=True,
        )
        tests = _string_list(
            item["tests"],
            field=f"{component_id}.tests",
            nonempty=MATURITY_LEVELS[maturity] >= 2,
        )
        for rel in [*evidence, *tests]:
            _repository_path(
                root, rel, field=f"{component_id}.maturity_evidence", file=True
            )
        item["maturity_evidence_sha256"] = {
            rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
            for rel in sorted(set([*evidence, *tests]))
        }
        reachability = str(item["production_reachability"])
        if reachability not in REACHABILITY_LEVELS:
            raise ValueError(
                f"invalid production reachability for {component_id}: {reachability}"
            )
        if MATURITY_LEVELS[maturity] >= 3 and reachability != "canonical":
            raise ValueError(f"M3+ component is not canonical: {component_id}")
        if bool(item.get("public_capability_claim")) and MATURITY_LEVELS[maturity] < 3:
            raise ValueError(f"public capability claim below M3: {component_id}")
        if bool(item.get("public_capability_claim")):
            _required_string(item, "public_name", component_id)
            _required_string(item, "public_limitations", component_id)
        for capability_id in _string_list(
            item.get("capability_ids", []), field=f"{component_id}.capability_ids"
        ):
            if capability_id in capability_owners:
                raise ValueError(f"duplicate capability owner: {capability_id}")
            capability_owners[capability_id] = component_id
        validated_components.append(item)

    capability_manifest = _load_json(root / "config" / "capabilities.yaml")
    capabilities = capability_manifest.get("capabilities", [])
    if not isinstance(capabilities, list):
        raise ValueError("capability manifest requires a capabilities list")
    known_capabilities = {
        str(item["capability_id"])
        for item in capabilities
        if isinstance(item, dict) and "capability_id" in item
    }
    unknown_capabilities = set(capability_owners) - known_capabilities
    if unknown_capabilities:
        raise ValueError(
            f"architecture status references unknown capabilities: {sorted(unknown_capabilities)}"
        )
    for capability in capabilities:
        if not isinstance(capability, dict):
            raise ValueError("capability entry must be an object")
        if capability.get("status") in {"ACTIVE", "DEGRADED"}:
            capability_id = str(capability["capability_id"])
            owner = capability_owners.get(capability_id)
            if owner is None:
                raise ValueError(
                    f"public capability has no architecture owner: {capability_id}"
                )
            component = next(
                item for item in components if item["component_id"] == owner
            )
            if MATURITY_LEVELS[str(component["maturity"])] < 3:
                raise ValueError(f"public capability below M3: {capability_id}")
            if component["production_reachability"] != "canonical":
                raise ValueError(f"public capability is not canonical: {capability_id}")

    doc_paths: set[str] = set()
    for raw in documents:
        if not isinstance(raw, dict):
            raise ValueError("documentation status entry must be an object")
        path = _repository_path(
            root, raw.get("path"), field="documentation.path", file=True
        )
        status = str(raw.get("status", ""))
        if path in doc_paths:
            raise ValueError(f"duplicate documentation status: {path}")
        doc_paths.add(path)
        if status not in DOCUMENT_STATUSES:
            raise ValueError(f"invalid documentation status for {path}: {status}")
        _required_string(raw, "purpose", path)
    architecture_docs = {
        path.relative_to(root).as_posix()
        for path in (root / "docs" / "architecture").rglob("*")
        if path.is_file()
    }
    missing_docs = architecture_docs - doc_paths
    if missing_docs:
        raise ValueError(
            f"architecture documents missing status: {sorted(missing_docs)}"
        )
    return sorted(
        validated_components, key=lambda item: str(item["component_id"])
    ), sorted((dict(item) for item in documents), key=lambda item: str(item["path"]))


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDED_PARTS) or any(
        str(path).startswith(part + "/") for part in EXCLUDED_PARTS
    )


def bounded_files(root: Path, patterns: Sequence[str], max_files: int) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            rel = path.relative_to(root)
            if path.is_file() and not _is_excluded(rel):
                found.append(rel)
                if len(found) > max_files:
                    raise RuntimeError(
                        "architecture inventory traversal exceeded file bound"
                    )
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
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and any(_literal_text(item) == "__main__" for item in comparators)
    )


def scan_python_file(
    root: Path, rel: Path, max_bytes: int
) -> dict[str, list[dict[str, object]]]:
    full = root / rel
    raw = full.read_bytes()
    if len(raw) > max_bytes:
        return {"skipped": [{"path": rel.as_posix(), "reason": "file_too_large"}]}
    text = raw.decode("utf-8", errors="replace")
    tree = ast.parse(text, filename=rel.as_posix())
    result: dict[str, list[dict[str, object]]] = {
        "entrypoints": [],
        "routes": [],
        "imports": [],
        "singletons": [],
        "workers": [],
        "fallbacks": [],
        "mocks": [],
        "persistence": [],
        "env_readers": [],
        "capability_claims": [],
        "owners": [],
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"vulcan", "src.vulcan"} or alias.name.startswith(
                    ("vulcan.", "src.vulcan.")
                ):
                    result["imports"].append(
                        {**_line_ref(rel, node), "module": alias.name}
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in {"vulcan", "src.vulcan"} or node.module.startswith(
                ("vulcan.", "src.vulcan.")
            ):
                result["imports"].append(
                    {**_line_ref(rel, node), "module": node.module}
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                route = _decorator_route(dec)
                if route:
                    method, path = route
                    result["routes"].append(
                        {
                            **_line_ref(rel, node),
                            "method": method,
                            "route": path,
                            "handler": node.name,
                        }
                    )
            lowered = node.name.lower()
            if lowered.startswith(("get_", "create_")) and any(
                term in lowered for term in ("singleton", "global", "instance")
            ):
                result["singletons"].append({**_line_ref(rel, node), "name": node.name})
        elif isinstance(node, ast.If) and _is_entrypoint_guard(node):
            result["entrypoints"].append(
                {**_line_ref(rel, node), "kind": "python_main_guard"}
            )
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name.endswith(("Thread", "Process")) or name.endswith(
                ("create_task", "run_in_executor")
            ):
                result["workers"].append({**_line_ref(rel, node), "call": name})
            if name in {"os.getenv", "os.environ.get"} or name.endswith("getenv"):
                key = _literal_text(node.args[0]) if node.args else None
                result["env_readers"].append(
                    {**_line_ref(rel, node), "call": name, "key": key}
                )
            if "sqlite" in name.lower() or name.endswith(("flock", "lockf")):
                result["persistence"].append(
                    {**_line_ref(rel, node), "kind": "call", "value": name}
                )
            if name.endswith("MagicMock"):
                result["mocks"].append({**_line_ref(rel, node), "kind": "MagicMock"})
        elif isinstance(node, ast.ExceptHandler):
            if node.type is not None and _call_name(node.type) == "Exception":
                result["fallbacks"].append(
                    {**_line_ref(rel, node), "kind": "except_exception"}
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(token in lowered for token in ("sqlite", ".db", ".sqlite", "flock")):
                result["persistence"].append(
                    {
                        **_line_ref(rel, node),
                        "kind": "literal",
                        "value": node.value[:160],
                    }
                )
            if "capability" in lowered and any(
                marker in lowered
                for marker in ("available", "enabled", "supported", "production")
            ):
                result["capability_claims"].append(
                    {**_line_ref(rel, node), "text": node.value[:160]}
                )
    rel_text = rel.as_posix().lower()
    for term in OWNERSHIP_TERMS:
        if term in rel_text.replace("-", "_"):
            result["owners"].append(
                {
                    "path": rel.as_posix(),
                    "owner_domain": term,
                    "reachability": "unknown",
                }
            )
    return result


def scan_text_entrypoints(root: Path) -> dict[str, list[dict[str, object]]]:
    items: dict[str, list[dict[str, object]]] = {
        "docker": [],
        "compose": [],
        "helm": [],
    }
    for rel in bounded_files(
        root,
        [
            "docker*/**/Dockerfile",
            "Dockerfile",
            "docker-compose*.yml",
            "docker-compose*.yaml",
            "helm/**/*.yaml",
            "helm/**/*.yml",
        ],
        1000,
    ):
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            lower = stripped.lower()
            if rel.name == "Dockerfile" and (
                stripped.startswith("CMD") or stripped.startswith("ENTRYPOINT")
            ):
                items["docker"].append(
                    {
                        "path": rel.as_posix(),
                        "line_start": idx,
                        "line_end": idx,
                        "directive": stripped.split(None, 1)[0],
                        "value": stripped,
                    }
                )
            if rel.name.startswith("docker-compose") and re.match(
                r"^(command|entrypoint|image):", stripped
            ):
                items["compose"].append(
                    {
                        "path": rel.as_posix(),
                        "line_start": idx,
                        "line_end": idx,
                        "key": stripped.split(":", 1)[0],
                        "value": stripped,
                    }
                )
            if (
                rel.parts
                and rel.parts[0] == "helm"
                and any(
                    key in lower
                    for key in (
                        "image:",
                        "command:",
                        "args:",
                        "livenessprobe:",
                        "readinessprobe:",
                    )
                )
            ):
                items["helm"].append(
                    {
                        "path": rel.as_posix(),
                        "line_start": idx,
                        "line_end": idx,
                        "key": stripped.split(":", 1)[0],
                        "value": stripped,
                    }
                )
    return items


def _reachability(path: str, production_paths: tuple[str, ...]) -> str:
    if "/tests/" in f"/{path}" or path.startswith("tests/"):
        return "test-only"
    if any(part in path for part in ("archive", "examples/", "docs/", "scripts/")):
        return "research-only"
    if any(
        path == prefix or path.startswith(prefix.rstrip("/") + "/")
        for prefix in production_paths
    ):
        return "production-reachable"
    return "unknown"


def _entrypoint_python_paths(text_entrypoints: Mapping[str, object]) -> set[str]:
    """Resolve statically visible uvicorn module targets to repository paths."""
    paths: set[str] = set()
    for entries in text_entrypoints.values():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", ""))
            for module in re.findall(
                r"(?:uvicorn(?:\s+|[^\w]+))([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+):[a-zA-Z_]\w*",
                value,
            ):
                module_path = module.replace(".", "/") + ".py"
                paths.add(
                    module_path
                    if module_path.startswith("src/")
                    else f"src/{module_path}"
                )
    return paths


def build_inventory(config: InventoryConfig) -> dict[str, object]:
    root = config.root.resolve()
    py_files = bounded_files(
        root, ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py"], config.max_files
    )
    text_entrypoints = scan_text_entrypoints(root)
    if config.require_truth_map:
        components, documents = _validated_truth_map(root)
    else:
        components, documents = [], []
    production_paths = tuple(
        str(item["package_path"]).rstrip("/")
        for item in components
        if item["production_reachability"] in {"canonical", "reachable-compatibility"}
    )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "generator": "scripts/architecture_inventory.py",
        "bounded_traversal": {
            "max_files": config.max_files,
            "max_bytes_per_file": config.max_bytes_per_file,
        },
        "production_entrypoint_files": list(PRODUCTION_ENTRYPOINT_FILES),
        "allowlisted_compatibility_shims": sorted(ALLOWLISTED_COMPATIBILITY_SHIMS),
        "entrypoints": [],
        "routes": [],
        "import_identities": [],
        "singletons": [],
        "workers": [],
        "fallbacks": [],
        "mocks": [],
        "persistence": [],
        "environment_readers": [],
        "capability_claims": [],
        "ownership_graph": [],
        "text_entrypoints": text_entrypoints,
        "findings": [],
        "architecture_status": components,
        "documentation_status": documents,
    }
    for rel in py_files:
        scanned = scan_python_file(root, rel, config.max_bytes_per_file)
        reachability = _reachability(rel.as_posix(), production_paths)
        for key, target in (
            ("entrypoints", "entrypoints"),
            ("routes", "routes"),
            ("imports", "import_identities"),
            ("singletons", "singletons"),
            ("workers", "workers"),
            ("fallbacks", "fallbacks"),
            ("mocks", "mocks"),
            ("persistence", "persistence"),
            ("env_readers", "environment_readers"),
            ("capability_claims", "capability_claims"),
        ):
            for item in scanned.get(key, []):
                item = dict(item)
                item["reachability"] = reachability
                if target in {"fallbacks", "mocks"}:
                    item["review"] = (
                        "allowlisted"
                        if item["path"] in ALLOWLISTED_COMPATIBILITY_SHIMS
                        else "unreviewed"
                    )
                inventory[target].append(item)
        for item in scanned.get("owners", []):
            item = dict(item)
            item["reachability"] = reachability
            inventory["ownership_graph"].append(item)
    covered_roots = tuple(
        str(item["package_path"]).rstrip("/")
        for item in components
        if item["production_reachability"] in {"canonical", "reachable-compatibility"}
    )
    reachable_paths = {
        str(item["path"])
        for key in (
            "entrypoints",
            "routes",
            "import_identities",
            "singletons",
            "workers",
            "fallbacks",
            "mocks",
            "persistence",
            "environment_readers",
            "capability_claims",
        )
        for item in inventory[key]
        if item.get("reachability") == "production-reachable"
    }
    unexplained = sorted(
        path for path in reachable_paths if not path.startswith(covered_roots)
    )
    unexplained_entrypoints = sorted(
        path
        for path in _entrypoint_python_paths(text_entrypoints)
        if not any(
            path == root_path or path.startswith(root_path.rstrip("/") + "/")
            for root_path in covered_roots
        )
    )
    if config.require_truth_map and unexplained:
        raise ValueError(f"unknown production-reachable components: {unexplained}")
    if config.require_truth_map and unexplained_entrypoints:
        raise ValueError(
            f"unknown production entrypoint components: {unexplained_entrypoints}"
        )
    modules = {item["module"] for item in inventory["import_identities"]}
    if any(str(module).startswith("src.vulcan") for module in modules) and any(
        str(module).startswith("vulcan") for module in modules
    ):
        inventory["findings"].append(
            {
                "id": "duplicate_package_identities",
                "severity": "high",
                "description": "Both src.vulcan and vulcan import identities are present.",
            }
        )
    for path in sorted(
        {
            item["path"]
            for item in inventory["routes"]
            if any(
                name in item["path"].lower()
                for name in ("orchestrator", "world_model", "unified_chat", "runtime")
            )
        }
    ):
        inventory["findings"].append(
            {
                "id": "competing_cognitive_orchestrator",
                "severity": "medium",
                "path": path,
            }
        )
    if any(
        item.get("reachability") == "production-reachable"
        and item.get("review") == "unreviewed"
        for item in inventory["mocks"]
    ):
        inventory["findings"].append(
            {
                "id": "production_mock",
                "severity": "critical",
                "description": "Production-reachable MagicMock use is not allowlisted.",
            }
        )
    if any(
        item.get("reachability") == "production-reachable"
        and item.get("review") == "unreviewed"
        for item in inventory["fallbacks"]
    ):
        inventory["findings"].append(
            {
                "id": "production_except_exception_fallback",
                "severity": "high",
                "description": "Production-reachable broad Exception fallbacks require review.",
            }
        )
    for key, value in list(inventory.items()):
        if isinstance(value, list) and key not in {
            "architecture_status",
            "documentation_status",
        }:
            inventory[key] = sorted(
                value, key=lambda item: json.dumps(item, sort_keys=True)
            )
    inventory["digest"] = inventory_digest(inventory)
    return inventory


def render_markdown(inventory: dict[str, object]) -> str:
    lines = [
        "# Generated architecture inventory",
        "",
        f"Digest: `{inventory['digest']}`",
        "",
        "This file is generated from `docs/generated/architecture-inventory.json`.",
        "",
    ]
    lines.extend(
        [
            "## Architecture truth map",
            "",
            "| Component | Path | Sole owner | Ceiling | Reachability | State authority | Snapshot | Persistence | Audit | Tests | Maturity | Compatibility | Removal condition |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in inventory["architecture_status"]:
        tests = "<br>".join(f"`{value}`" for value in item["tests"])
        lines.append(
            f"| `{item['component_id']}` | `{item['package_path']}` | {item['sole_owner']} | {item['authority_ceiling']} | {item['production_reachability']} | {item['state_authority']} | {item['snapshot_implementation']} | {item['persistence_owner']} | {item['audit_evidence']} | {tests} | **{item['maturity']}** | {item['compatibility_status']} | {item['removal_condition']} |"
        )
    lines.extend(
        [
            "",
            "## Normative documentation index",
            "",
            "| Document | Status | Purpose |",
            "|---|---|---|",
        ]
    )
    for item in inventory["documentation_status"]:
        target = (
            f"../{item['path'].removeprefix('docs/')}"
            if str(item["path"]).startswith("docs/")
            else f"../../{item['path']}"
        )
        lines.append(
            f"| [`{item['path']}`]({target}) | **{item['status']}** | {item['purpose']} |"
        )
    lines.append("")
    for section in (
        "entrypoints",
        "routes",
        "import_identities",
        "singletons",
        "workers",
        "fallbacks",
        "mocks",
        "persistence",
        "environment_readers",
        "capability_claims",
        "ownership_graph",
    ):
        items = inventory[section]
        lines.extend(
            [f"## {section.replace('_', ' ').title()}", "", f"Count: {len(items)}", ""]
        )
        for item in items[:50]:
            path = item.get("path", "")
            line = item.get("line_start", "")
            detail = ", ".join(
                f"{k}={item[k]!r}"
                for k in sorted(item)
                if k not in {"path", "line_start", "line_end"}
            )
            lines.append(f"- `{path}:{line}` {detail}")
        if len(items) > 50:
            lines.append(f"- ... {len(items) - 50} more in JSON")
        lines.append("")
    lines.extend(["## Text Entrypoints", ""])
    for kind in sorted(inventory["text_entrypoints"]):
        items = inventory["text_entrypoints"][kind]
        lines.append(f"### {kind}")
        for item in items:
            lines.append(
                f"- `{item['path']}:{item['line_start']}` {item.get('value', '')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_readme_capabilities(inventory: Mapping[str, object]) -> str:
    """Render the README capability block exclusively from validated M3+ records."""
    lines = [
        "<!-- BEGIN GENERATED ARCHITECTURE CAPABILITIES -->",
        "| Public capability | Canonical owner | Maturity | Limitations |",
        "|---|---|---|---|",
    ]
    components = inventory.get("architecture_status", [])
    if not isinstance(components, list):
        raise ValueError("inventory architecture_status must be a list")
    for item in components:
        if isinstance(item, dict) and item.get("public_capability_claim"):
            lines.append(
                f"| {item['public_name']} | `{item['sole_owner']}` | **{item['maturity']} — Canonical** | {item['public_limitations']} |"
            )
    lines.append("<!-- END GENERATED ARCHITECTURE CAPABILITIES -->")
    return "\n".join(lines)


def replace_readme_capabilities(readme: str, block: str) -> str:
    """Replace exactly one generated README region, failing closed on ambiguity."""
    start_marker = "<!-- BEGIN GENERATED ARCHITECTURE CAPABILITIES -->"
    end_marker = "<!-- END GENERATED ARCHITECTURE CAPABILITIES -->"
    if readme.count(start_marker) != 1 or readme.count(end_marker) != 1:
        raise ValueError("README must contain exactly one generated capability block")
    start = readme.index(start_marker)
    end = readme.index(end_marker, start) + len(end_marker)
    return readme[:start] + block + readme[end:]
