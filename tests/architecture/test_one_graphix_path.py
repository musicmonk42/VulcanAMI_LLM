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


def test_retired_runtime_semantic_adapters_are_deleted() -> None:
    assert not Path("src/vulcan/runtime/semantic.py").exists()
    assert not Path("src/vulcan/runtime/epistemic_adapter.py").exists()
    production = Path("src/vulcan").rglob("*.py")
    retired = ("runtime.semantic", "runtime.epistemic_adapter")
    for path in production:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = (
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        assert all(
            not any(name in statement for name in retired) for statement in imports
        )


def test_case_has_no_compatibility_epistemic_ledger() -> None:
    source = Path("src/vulcan/runtime/case.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "_claims",
        "_evidence",
        "_derivations",
        "append_ledger",
        "project_committed_ledger",
    }
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    assert forbidden.isdisjoint(definitions | assignments)


def test_kernel_and_alignment_use_durable_epistemic_head_directly() -> None:
    kernel = Path("src/vulcan/runtime/kernel.py").read_text(encoding="utf-8")
    alignment = Path("src/vulcan/runtime/alignment.py").read_text(encoding="utf-8")
    assert "build_epistemic_candidate" in kernel
    assert "epistemic_head = self._transactions.epistemic_head" in kernel
    assert "project_committed_ledger" not in kernel
    assert "adapt_runtime_semantic_candidate" not in kernel
    assert "commit: EpistemicCommit" in alignment


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
