"""Static analysis: no hardcoded secrets in production source."""
import ast
import os
import re
import pytest
from pathlib import Path

# Patterns that indicate hardcoded secrets
SECRET_PATTERNS = [
    re.compile(r'api_key\s*=\s*["\'][^"\']+["\']'),
    re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
    re.compile(r'secret\s*=\s*["\'][^"\']+["\']'),
    re.compile(r'secret_key\s*=\s*["\'][^"\']+["\']'),
    re.compile(r'token\s*=\s*["\'][^"\']+["\']'),
]

# Allowlisted patterns (not actual secrets) — both quote styles
ALLOWLIST = [
    'api_key=""',
    "api_key=''",
    'api_key=None',
    'api_key="mock-key"',
    "api_key='mock-key'",
    'password=""',
    "password=''",
    'secret=""',
    "secret=''",
]

EXCLUDED_DIRS = {"tests", "test", "__pycache__", ".git", "node_modules"}


def _is_in_test_or_main_block(filepath: str, line_no: int) -> bool:
    """Check if a line is inside a test file or __main__ guard."""
    # Skip test directories entirely (handled by dir exclusion)
    # Check for __main__ blocks
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Walk backwards from line to find if we're in __main__
        for i in range(line_no - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped == 'if __name__ == "__main__":' or stripped == "if __name__ == '__main__':":
                return True
            if not stripped.startswith("#") and not stripped == "" and not stripped.startswith("def ") and not stripped.startswith("class "):
                if i < line_no - 20:
                    break
    except Exception:
        pass
    return False


def get_production_python_files():
    """Get all .py files under src/ excluding test directories."""
    # Use repo root relative to this test file's location
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if not src_dir.exists():
        pytest.skip("src/ directory not found")
    files = []
    for py_file in src_dir.rglob("*.py"):
        parts = set(py_file.parts)
        if not parts.intersection(EXCLUDED_DIRS):
            files.append(py_file)
    return files


def test_no_hardcoded_secrets_in_production_code():
    """Scan production source for hardcoded credential patterns."""
    violations = []
    for filepath in get_production_python_files():
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in SECRET_PATTERNS:
                match = pattern.search(line)
                if match:
                    matched_text = match.group(0)
                    # Check allowlist
                    if any(allow in matched_text for allow in ALLOWLIST):
                        continue
                    # Check if in __main__ block
                    if _is_in_test_or_main_block(str(filepath), line_no):
                        continue
                    violations.append(
                        f"{filepath}:{line_no}: {matched_text}"
                    )
    assert not violations, (
        f"Found {len(violations)} hardcoded secret(s) in production code:\n"
        + "\n".join(violations)
    )
