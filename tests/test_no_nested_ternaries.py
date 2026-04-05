"""AST-based test: no nested ternary expressions in production code."""
import ast
import pytest
from pathlib import Path


def _find_nested_ternaries(filepath: Path) -> list[tuple[int, str]]:
    """Find nested ternary (IfExp inside IfExp) expressions."""
    violations = []
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            # Check if body or orelse contains another IfExp
            for child in ast.walk(node.body):
                if child is not node and isinstance(child, ast.IfExp):
                    line = getattr(node, "lineno", 0)
                    violations.append((line, "nested ternary in body"))
                    break
            for child in ast.walk(node.orelse):
                if child is not node and isinstance(child, ast.IfExp):
                    line = getattr(node, "lineno", 0)
                    violations.append((line, "nested ternary in else"))
                    break
    return violations


def test_no_nested_ternaries_in_source():
    """All .py files under src/ must have zero nested ternary expressions."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if not src_dir.exists():
        pytest.skip("src/ directory not found")

    all_violations = []
    excluded = {"__pycache__", ".git", "node_modules"}

    for py_file in src_dir.rglob("*.py"):
        if set(py_file.parts).intersection(excluded):
            continue
        violations = _find_nested_ternaries(py_file)
        for line, desc in violations:
            rel = py_file.relative_to(repo_root)
            all_violations.append(f"{rel}:{line}: {desc}")

    assert not all_violations, (
        f"Found {len(all_violations)} nested ternary expression(s):\n"
        + "\n".join(all_violations)
    )
