#!/usr/bin/env python3
"""Reproducibly build and adversarially verify the Phase-A wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPOCH = "1704067200"


def run(cmd, cwd, ok=True, env=None):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if ok and r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return r


def copy_source(dest):
    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(
            ".git", "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )


def build(source, out):
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = EPOCH
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "-w",
            str(out),
            ".",
        ],
        source,
        env=env,
    )
    return next(out.glob("*.whl"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        wheels = []
        for name in ("a", "b"):
            src = t / f"source-{name}"
            out = t / f"wheel-{name}"
            copy_source(src)
            out.mkdir()
            wheels.append(build(src, out))
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/ci/verify_wheel.py"),
                    str(wheels[-1]),
                ],
                ROOT,
            )
        if sha(wheels[0]) != sha(wheels[1]):
            raise RuntimeError("non-deterministic wheel bytes")
        final = a.output / wheels[0].name
        shutil.copyfile(wheels[0], final)
        badsrc = t / "undeclared"
        copy_source(badsrc)
        (badsrc / "src/vulcan/runtime/app.py").write_text(
            (badsrc / "src/vulcan/runtime/app.py").read_text()
            + "\nimport vulcan.memory\n"
        )
        badout = t / "badout"
        badout.mkdir()
        if build_result := run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "-w",
                str(badout),
                ".",
            ],
            badsrc,
            ok=False,
        ):
            if build_result.returncode == 0:
                raise RuntimeError("undeclared import did not block build")
        mutated = t / "mutated.whl"
        with zipfile.ZipFile(final) as source, zipfile.ZipFile(mutated, "w") as target:
            for info in source.infolist():
                raw = source.read(info.filename)
                if info.filename == "vulcan/runtime/api.py":
                    raw = b"mutated"
                target.writestr(info, raw)
        if (
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/ci/verify_wheel.py"),
                    str(mutated),
                ],
                ROOT,
                ok=False,
            ).returncode
            == 0
        ):
            raise RuntimeError("mutated wheel passed RECORD verification")
        env = t / "installed"
        venv.EnvBuilder(with_pip=True).create(env)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                str(final),
            ],
            t,
        )
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(ROOT / "requirements-runtime.lock"),
            ],
            t,
        )
        probe = "import pathlib,vulcan.runtime.api as a; p=pathlib.Path(a.__file__).resolve();\nif 'workspace' in str(p): raise RuntimeError('module resolved from checkout')\nfor n in ('vulcan.memory','vulcan.research','vulcan.improvement'):\n try: __import__(n); raise RuntimeError(n)\n except ModuleNotFoundError: pass"
        run([str(python), "-I", "-c", probe], t)
        run([str(python), "-O", "-I", "-c", probe], t)
        durable = t / "durable"
        durable.mkdir(mode=0o700)
        lifecycle = """import asyncio,os
from datetime import datetime,timedelta,timezone
from vulcan.runtime.app import app
from vulcan.runtime.api import CommandEnvelope,CommandKind,ExecutionBudget,VerifiedAuthenticationContext,request_digest
from vulcan.runtime.auth import AuthenticatedPrincipal
async def main():
 async with app.router.lifespan_context(app):
  payload={'message':'2 + 2','conversation_id':None}
  principal=AuthenticatedPrincipal('subject','tenant','vulcan',('vulcan-runtime',),frozenset({'reason:write'}),'0123456789abcdef','v1')
  envelope=CommandEnvelope(CommandKind.CHAT,request_digest(CommandKind.CHAT,payload),VerifiedAuthenticationContext.from_verified_principal(principal),'installed-arithmetic',datetime.now(timezone.utc)+timedelta(seconds=30),ExecutionBudget(64,4096),payload)
  result=await app.state.api.execute(envelope)
  if result['response']!='The computed result is 4.': raise RuntimeError('installed arithmetic mismatch')
 if app.state.api is not None or app.state.ready: raise RuntimeError('lifespan resources not closed')
asyncio.run(main())"""
        child_env = os.environ.copy()
        child_env.update(
            VULCAN_ENV="production",
            VULCAN_JWT_SECRET="Installed-Wheel-Secret-0123456789!abcdef",
            VULCAN_RUNTIME_DURABLE_ROOT=str(durable),
            VULCAN_MEMORY_ENABLED="false",
            VULCAN_CSIU_ENABLED="false",
            VULCAN_LEARNING_ENABLED="false",
        )
        run([str(python), "-I", "-c", lifecycle], t, env=child_env)
        run([str(python), "-O", "-I", "-c", lifecycle], t, env=child_env)
        run([str(python), "-m", "pip", "check"], t)
        manifest = json.loads(
            (ROOT / "config/wheel-inclusion-manifest.json").read_text()
        )
        file_count = len(manifest["files"])
        loc = sum(row["loc"] for row in manifest["files"])
        toolchain = {
            "python": sys.version.split()[0],
            "pip": "24.0",
            "setuptools": "79.0.1",
        }
        report = {
            "schema": "vulcan-wheel-build/v1",
            "source_digest": hashlib.sha256(
                b"".join((ROOT / r["path"]).read_bytes() for r in manifest["files"])
            ).hexdigest(),
            "build_lock_digest": sha(ROOT / "requirements-build.lock"),
            "runtime_lock_digest": sha(ROOT / "requirements-runtime.lock"),
            "wheel_digest": sha(final),
            "manifest_digest": manifest["manifest_digest"],
            "toolchain": toolchain,
            "toolchain_digest": hashlib.sha256(
                json.dumps(toolchain, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "file_count": file_count,
            "file_count_digest": hashlib.sha256(str(file_count).encode()).hexdigest(),
            "loc": loc,
            "loc_digest": hashlib.sha256(str(loc).encode()).hexdigest(),
        }
        (a.output / "build-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
