"""Fresh-process qualification for canonical audit imports and ASGI bootstrap."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _run_isolated(
    code: str, *, optimized: bool = False, env: dict[str, str] | None = None
):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend(["-I", "-c", code])
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    child_env.update(env or {})
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=child_env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "imports",
    [
        ("vulcan.runtime.audit", "vulcan.persistence.audit"),
        ("vulcan.persistence.audit", "vulcan.runtime.audit"),
        ("vulcan.persistence.audit.store", "vulcan.runtime.audit"),
    ],
)
def test_audit_import_orders_are_acyclic(imports: tuple[str, ...]) -> None:
    code = textwrap.dedent(f"""
        import importlib
        import sys
        sys.path.insert(0, {str(SOURCE_ROOT)!r})
        modules = [importlib.import_module(name) for name in {imports!r}]
        if not modules:
            raise RuntimeError("audit modules were not imported")
        if any(name == "src.vulcan" or name.startswith("src.vulcan.") for name in sys.modules):
            raise RuntimeError("deprecated src.vulcan package identity loaded")
        """)
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("optimized", [False, True])
def test_canonical_app_constructs_and_lifespan_closes_in_fresh_process(
    tmp_path: Path, optimized: bool
) -> None:
    durable_root = tmp_path / ("optimized" if optimized else "normal")
    durable_root.mkdir(mode=0o700)
    code = textwrap.dedent(f"""
        import asyncio
        import sys
        sys.path.insert(0, {str(SOURCE_ROOT)!r})
        from vulcan.runtime.app import app, create_app

        async def qualify():
            fresh = create_app()
            if fresh is app:
                raise RuntimeError("create_app returned the module singleton")
            async with fresh.router.lifespan_context(fresh):
                if fresh.state.ready is not True or fresh.state.runtime is None:
                    raise RuntimeError("runtime did not become ready")
            if fresh.state.ready is not False or fresh.state.runtime is not None:
                raise RuntimeError("runtime resources were not released")

        asyncio.run(qualify())
        if any(name == "src.vulcan" or name.startswith("src.vulcan.") for name in sys.modules):
            raise RuntimeError("deprecated src.vulcan package identity loaded")
        """)
    result = _run_isolated(
        code,
        optimized=optimized,
        env={
            "CI": "true",
            "VULCAN_ENV": "development",
            "VULCAN_JWT_SECRET": "Fresh-Process-Secret-0123456789!abcdef",
            "VULCAN_RUNTIME_DURABLE_ROOT": str(durable_root),
            "VULCAN_MEMORY_ENABLED": "false",
            "VULCAN_CSIU_ENABLED": "false",
            "VULCAN_LEARNING_ENABLED": "false",
        },
    )
    assert result.returncode == 0, result.stderr


def test_regression_fixture_recreates_and_detects_eager_import_cycle(
    tmp_path: Path,
) -> None:
    package = tmp_path / "cycle_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("from .first import FIRST\n")
    (package / "first.py").write_text("from .second import SECOND\nFIRST = 'first'\n")
    (package / "second.py").write_text("from .first import FIRST\nSECOND = 'second'\n")
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import cycle_fixture
        """)
    result = _run_isolated(code)
    assert result.returncode != 0
    assert "partially initialized module" in result.stderr


def _local_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_audit_ownership_cannot_regress_to_runtime_persistence_cycle() -> None:
    runtime_audit = SOURCE_ROOT / "vulcan" / "runtime" / "audit.py"
    persistence_files = tuple(
        (SOURCE_ROOT / "vulcan" / "persistence" / "audit").glob("*.py")
    )

    runtime_tree = ast.parse(runtime_audit.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in runtime_tree.body
    )
    assert all(
        not any(name == "vulcan.runtime.audit" for name in _local_imports(path))
        for path in persistence_files
    )
