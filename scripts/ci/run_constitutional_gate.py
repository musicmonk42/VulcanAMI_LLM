#!/usr/bin/env python3
"""Run the reproducible, dependency-light constitutional assurance gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
CHECKS: dict[str, tuple[tuple[str, ...], ...]] = {
    "workflow": (
        (PYTHON, "scripts/architecture_inventory.py", "--check"),
        (PYTHON, "scripts/ci/workflow_lint.py"),
        (PYTHON, "scripts/ci/verify_dependency_inputs.py"),
    ),
    "unit": (
        (
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/architecture/test_ami_constitution.py",
            "tests/assurance/test_control_catalog.py",
        ),
    ),
    "typing": ((PYTHON, "-m", "mypy", "scripts/ci"),),
    "format": (
        (
            PYTHON,
            "-m",
            "black",
            "--check",
            "--target-version",
            "py311",
            "scripts/ci/run_constitutional_gate.py",
            "scripts/ci/secret_scan.py",
            "scripts/ci/verify_dependency_inputs.py",
            "scripts/ci/workflow_lint.py",
            "tests/ci",
        ),
        (
            PYTHON,
            "-m",
            "isort",
            "--check-only",
            "scripts/ci/run_constitutional_gate.py",
            "scripts/ci/secret_scan.py",
            "scripts/ci/verify_dependency_inputs.py",
            "scripts/ci/workflow_lint.py",
            "tests/ci",
        ),
    ),
    "integration": (
        (
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/runtime/test_authoritative_episode_path.py",
            "tests/microkernel/test_episode_store.py",
            "tests/architecture/test_ami_constitution.py",
            "tests/ci/test_workflow_gates.py",
        ),
    ),
    "security": (
        (
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "tests/security/test_language_contracts.py",
            "tests/runtime/test_authoritative_episode_path.py",
            "tests/microkernel/test_episode_store.py",
            "tests/architecture/test_ami_constitution.py",
        ),
        (
            PYTHON,
            "-O",
            "-m",
            "pytest",
            "-q",
            "tests/security/test_language_contracts.py",
            "tests/runtime/test_authoritative_episode_path.py",
            "tests/microkernel/test_episode_store.py",
            "tests/architecture/test_ami_constitution.py",
        ),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=tuple(CHECKS), action="append")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="create .venv-constitutional and install the hash-locked toolchain",
    )
    args = parser.parse_args()
    if args.bootstrap:
        environment = ROOT / ".venv-constitutional"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                "requirements-constitutional.txt",
            ],
            cwd=ROOT,
            check=True,
        )
        bootstrap_command = [str(python), __file__]
        for check in args.check or ():
            bootstrap_command.extend(("--check", check))
        return subprocess.run(bootstrap_command, cwd=ROOT, check=False).returncode
    selected = args.check or list(CHECKS)
    for name in selected:
        for check_command in CHECKS[name]:
            print(f"[{name}] {' '.join(check_command)}", flush=True)
            completed = subprocess.run(check_command, cwd=ROOT, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
