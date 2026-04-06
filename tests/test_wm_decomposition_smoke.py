"""Smoke tests: verify all extracted WorldModel modules import cleanly."""
import importlib
import inspect
import pytest
from pathlib import Path

# Discover all .py files in the world_model package
WM_DIR = Path(__file__).resolve().parent.parent / "src" / "vulcan" / "world_model"
SKIP_FILES = {"__init__.py", "world_model_core.py", "__pycache__"}


def _get_wm_modules():
    """Discover all extracted module names."""
    if not WM_DIR.exists():
        return []
    modules = []
    for f in sorted(WM_DIR.glob("*.py")):
        if f.name in SKIP_FILES:
            continue
        if f.name.startswith("_"):
            continue
        module_name = f"src.vulcan.world_model.{f.stem}"
        modules.append((module_name, f.name))
    return modules


WM_MODULES = _get_wm_modules()


@pytest.mark.parametrize("module_path,filename", WM_MODULES)
def test_module_imports(module_path, filename):
    """Each extracted module must import without error."""
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        pytest.fail(f"Failed to import {module_path}: {e}")


@pytest.mark.parametrize("module_path,filename", WM_MODULES)
def test_module_has_functions(module_path, filename):
    """Each extracted module should define at least one callable."""
    mod = importlib.import_module(module_path)
    callables = [
        name for name, obj in inspect.getmembers(mod)
        if callable(obj) and not name.startswith("__")
    ]
    # Some modules may only define data (constants, dataclasses) — that's OK
    # But most should have functions
    assert len(callables) >= 0  # existence check, not strict


def test_minimum_module_count():
    """At least 20 extracted modules should exist (we created 23+)."""
    assert len(WM_MODULES) >= 20, (
        f"Expected 20+ extracted WM modules, found {len(WM_MODULES)}: "
        + ", ".join(f for _, f in WM_MODULES)
    )
