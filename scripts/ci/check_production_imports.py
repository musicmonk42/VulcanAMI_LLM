#!/usr/bin/env python3
"""Fail closed unless canonical serving's transitive local imports match policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
POLICY_PATH = ROOT / "config" / "production-import-policy.json"


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    package = SOURCE / relative / "__init__.py"
    module_file = (SOURCE / relative).with_suffix(".py")
    return (
        package if package.is_file() else module_file if module_file.is_file() else None
    )


def _imports(module: str, path: Path) -> set[str]:
    found: set[str] = set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                name = ".".join(filter(None, (base, node.module or "")))
            else:
                name = node.module or ""
            if name:
                found.add(name)
    return found


def check() -> tuple[set[str], list[str]]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    allowed = tuple(policy["allowed_prefixes"])
    denied = tuple(policy["denied_prefixes"])
    pending = [policy["entrypoint"]]
    visited: set[str] = set()
    errors: list[str] = []
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        parts = module.split(".")
        pending.extend(
            ".".join(parts[:index])
            for index in range(1, len(parts))
            if ".".join(parts[:index]) not in visited
        )
        path = _path(module)
        if path is None:
            continue
        for imported in _imports(module, path):
            if _matches(imported, denied):
                errors.append(f"{module} imports denied {imported}")
            if imported.startswith("vulcan"):
                if not _matches(imported, allowed):
                    errors.append(f"{module} imports non-allowlisted {imported}")
                elif _path(imported) is not None:
                    pending.append(imported)
    for module in visited:
        path = _path(module)
        if path is not None and any(
            imported == "src.vulcan" or imported.startswith("src.vulcan.")
            for imported in _imports(module, path)
        ):
            errors.append(f"deprecated package identity in {path.relative_to(ROOT)}")
    return visited, sorted(set(errors))


def main() -> int:
    visited, errors = check()
    if errors:
        print("\n".join(errors))
        return 1
    print(f"production import closure verified: {len(visited)} local modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
