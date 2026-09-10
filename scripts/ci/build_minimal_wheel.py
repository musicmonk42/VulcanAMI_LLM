#!/usr/bin/env python3
"""Reproducibly build and adversarially verify the Phase-A wheel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPOCH = "1704067200"
EXPECTED_TOOLCHAIN = {"pip": "24.0", "setuptools": "79.0.1"}


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


def toolchain_versions():
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{name: metadata.version(name) for name in EXPECTED_TOOLCHAIN},
    }


def require_locked_toolchain():
    actual = toolchain_versions()
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in EXPECTED_TOOLCHAIN.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "build interpreter does not match requirements-build.lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    toolchain = require_locked_toolchain()
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        raise RuntimeError("output directory must be empty")
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
        # Keep the candidate private until every negative control and both
        # installed-artifact qualifications pass.  A failed run must never
        # leave a wheel in the requested release directory.
        candidate = wheels[0]
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
        with (
            zipfile.ZipFile(candidate) as source,
            zipfile.ZipFile(mutated, "w") as target,
        ):
            record_name = next(
                name for name in source.namelist() if name.endswith(".dist-info/RECORD")
            )
            for info in source.infolist():
                raw = source.read(info.filename)
                if info.filename == "vulcan/runtime/api.py":
                    raw = b"mutated"
                # Re-sign the archive's RECORD below.  This proves that RECORD is
                # an integrity inventory, not authorization to change reviewed code.
                if info.filename == record_name:
                    continue
                target.writestr(info, raw)
            names = target.namelist() + [record_name]
            rows = []
            for name in names:
                if name == record_name:
                    rows.append(f"{name},,")
                    continue
                raw = target.read(name)
                value = (
                    base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
                    .rstrip(b"=")
                    .decode()
                )
                rows.append(f"{name},sha256={value},{len(raw)}")
            target.writestr(record_name, "\n".join(rows) + "\n")
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
  principal=AuthenticatedPrincipal._from_verified_adapter('subject','tenant','vulcan',('vulcan-runtime',),frozenset({'reason:write'}),'0123456789abcdef','v1',datetime.now(timezone.utc))
  envelope=CommandEnvelope(CommandKind.CHAT,request_digest(CommandKind.CHAT,payload),VerifiedAuthenticationContext.from_verified_principal(principal),'installed-request','installed-arithmetic',datetime.now(timezone.utc)+timedelta(seconds=30),ExecutionBudget(64,4096),payload)
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
        for mode, flags in (("normal", []), ("optimized", ["-O"])):
            env = t / f"installed-{mode}"
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
                    str(candidate),
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
            probe = f"""import pathlib,sys,vulcan.runtime.api as a
p=pathlib.Path(a.__file__).resolve()
root=pathlib.Path({str(ROOT)!r}).resolve()
if p.is_relative_to(root) or any(pathlib.Path(x or '.').resolve().is_relative_to(root) for x in sys.path): raise RuntimeError('checkout present in import path')
if not p.is_relative_to(pathlib.Path(sys.prefix).resolve()): raise RuntimeError('module did not resolve from pristine environment')
for n in ('vulcan.memory','vulcan.research','vulcan.improvement','vulcan.effects','vulcan.npt'):
 try: __import__(n); raise RuntimeError(n)
 except ModuleNotFoundError: pass"""
            run([str(python), *flags, "-I", "-c", probe], t)
            run([str(python), *flags, "-I", "-c", lifecycle], t, env=child_env)
            run([str(python), "-m", "pip", "check"], t)
        manifest = json.loads(
            (ROOT / "config/wheel-inclusion-manifest.json").read_text()
        )
        file_count = len(manifest["files"])
        loc = sum(row["loc"] for row in manifest["files"])
        source_inputs = [
            {"path": row["path"], "sha256": row["sha256"]} for row in manifest["files"]
        ] + [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in manifest["resources"]
        ]
        report = {
            "schema": "vulcan-wheel-build/v1",
            "source_digest": canonical_digest(source_inputs),
            "build_lock_digest": sha(ROOT / "requirements-build.lock"),
            "runtime_lock_digest": sha(ROOT / "requirements-runtime.lock"),
            "wheel_digest": sha(candidate),
            "manifest_digest": manifest["manifest_digest"],
            "toolchain": toolchain,
            "toolchain_digest": canonical_digest(toolchain),
            "file_count": file_count,
            "file_count_digest": hashlib.sha256(str(file_count).encode()).hexdigest(),
            "loc": loc,
            "loc_digest": hashlib.sha256(str(loc).encode()).hexdigest(),
        }
        shutil.copyfile(candidate, a.output / candidate.name)
        (a.output / "build-evidence.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
