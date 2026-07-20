from __future__ import annotations

from pathlib import Path

PRODUCTION_RUNTIME = Path("src/vulcan/runtime")
BANNED = ("MagicMock", "_FallbackDeployment", "_FallbackWorld", "_FallbackSafety", "class FastAPI", "class BaseModel")


def test_no_magicmock_or_fallback_classes_in_runtime_import_closure():
    offenders = []
    for path in PRODUCTION_RUNTIME.rglob("*.py"):
        if path.name == "route_manifest.py":
            text = path.read_text()
            assert "fastapi" not in text.lower()
            assert "pydantic" not in text.lower()
        text = path.read_text()
        for banned in BANNED:
            if banned in text:
                offenders.append(f"{path}:{banned}")
    assert offenders == []
