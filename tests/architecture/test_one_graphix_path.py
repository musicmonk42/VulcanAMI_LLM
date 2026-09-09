"""Import-closure and provenance gates for Wave 2.2."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vulcan.graphix.dialects import (
    CANONICAL_DIALECTS,
)
from vulcan.graphix.operations import CANONICAL_OPERATIONS
from vulcan.graphix.runtime import (
    AcceptedInterpretation,
    build_graphix_plan,
    compile_graphix_plan,
    execute_graphix_plan,
)


def test_production_kernel_does_not_import_retired_runtime_semantic() -> None:
    source = Path("src/vulcan/runtime/kernel.py").read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all("runtime.semantic" not in ast.unparse(node) for node in imports)
    assert "vulcan.graphix.runtime" in source


def test_all_cognitive_dialects_are_explicit_and_graphix_cannot_commit() -> None:
    assert len(CANONICAL_DIALECTS) == 5


def test_live_plan_requires_validated_snapshot_bound_graphix_artifact() -> None:
    request = "1" * 64
    snapshot = "2" * 64
    plan = build_graphix_plan(
        AcceptedInterpretation(0, "arithmetic", "2+2"),
        request_digest=request,
        state_snapshot_id=snapshot,
    )
    compiled = compile_graphix_plan(
        plan,
        request_digest=request,
        state_snapshot_id=snapshot,
        domain_snapshot_id="domain:none",
        episode_id="case-canonical-test",
    )
    assert compiled.plan_artifact_digest is not None
    assert compiled.validation_digest is not None
    assert compiled.snapshot_bundle_digest == "sha256:" + snapshot
    with pytest.raises(ValueError, match="validated canonical"):
        execute_graphix_plan(
            compiled,
            request_digest=request,
            state_snapshot_id="3" * 64,
            domain_snapshot_id="domain:none",
            require_canonical_artifact=True,
        )


@pytest.mark.parametrize(
    "name", ["command", "import", "class_path", "__import__", "memory_write"]
)
def test_dynamic_or_authority_bearing_operations_are_not_registered(name: str) -> None:
    with pytest.raises(ValueError, match="not registered"):
        CANONICAL_OPERATIONS.require(name)


def test_only_bounded_arithmetic_and_typed_lookup_are_registered() -> None:
    assert CANONICAL_OPERATIONS.names == ("arithmetic", "lookup")
