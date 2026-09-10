#!/usr/bin/env python3
"""Positive-inclusion wheel build configured by the reviewed closure manifest."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "config/wheel-inclusion-manifest.json").read_text())
ALLOWED = {row["module"] for row in MANIFEST["files"]}
PACKAGES = {
    row["module"] for row in MANIFEST["files"] if row["path"].endswith("/__init__.py")
}

subprocess.run(
    [sys.executable, "scripts/ci/generate_wheel_manifest.py", "--check"],
    cwd=ROOT,
    check=True,
)


class PositiveBuildPy(build_py):
    def run(self):
        super().run()
        for resource in MANIFEST["resources"]:
            source = ROOT / resource["path"]
            destination = Path(self.build_lib) / resource["wheel_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def find_package_modules(self, package, package_dir):
        found = super().find_package_modules(package, package_dir)
        selected = []
        for package, module, path in found:
            identity = package if module == "__init__" else f"{package}.{module}"
            if identity in ALLOWED:
                selected.append((package, module, path))
        return selected

    def find_all_modules(self):
        selected = super().find_all_modules()
        included = {p if m == "__init__" else f"{p}.{m}" for p, m, _ in selected}
        missing = ALLOWED - included
        if missing:
            raise RuntimeError(
                f"wheel manifest modules missing from source: {sorted(missing)}"
            )
        return selected


setup(packages=sorted(PACKAGES), cmdclass={"build_py": PositiveBuildPy})
