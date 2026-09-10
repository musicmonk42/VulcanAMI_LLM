#!/usr/bin/env python3
"""Build two untagged images from one frozen wheel and compare immutable IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from qualify_phase_a_artifact import immutable_image

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["DOCKER_BUILDKIT"] = "1"
    result = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, env=environment
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = immutable_image(args.base)
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required; image qualification was NOT_EXECUTED")
    if sha(args.wheel) != args.wheel_sha256:
        raise RuntimeError("frozen wheel digest mismatch")
    if args.output.exists():
        raise RuntimeError("output must not already exist")
    base_inspection = json.loads(
        run(["docker", "image", "inspect", base], ROOT).stdout
    )[0]
    if base not in base_inspection.get("RepoDigests", []):
        raise RuntimeError("local base image does not match the immutable reference")
    with tempfile.TemporaryDirectory() as temporary:
        context = Path(temporary)
        shutil.copy2(args.wheel, context / "candidate.whl")
        shutil.copy2(ROOT / "requirements-runtime.lock", context)
        shutil.copytree(args.wheelhouse, context / "wheelhouse")
        shutil.copy2(ROOT / "Dockerfile.candidate", context / "Dockerfile")
        ids = []
        inspections = []
        for name in ("a", "b"):
            iid = context / f"{name}.iid"
            run(
                [
                    "docker",
                    "build",
                    "--pull=false",
                    "--no-cache",
                    "--network=none",
                    "--build-arg",
                    f"PYTHON_BASE={base}",
                    "--build-arg",
                    f"WHEEL_SHA256={args.wheel_sha256}",
                    "--build-arg",
                    "SOURCE_DATE_EPOCH=1704067200",
                    "--iidfile",
                    str(iid),
                    ".",
                ],
                context,
            )
            image_id = iid.read_text().strip()
            if not image_id.startswith("sha256:"):
                raise RuntimeError("builder did not return an immutable image ID")
            ids.append(image_id)
            inspection = json.loads(
                run(["docker", "image", "inspect", image_id], context).stdout
            )[0]
            inspections.append(
                {
                    "architecture": inspection["Architecture"],
                    "config": inspection["Config"],
                    "os": inspection["Os"],
                    "rootfs": inspection["RootFS"],
                }
            )
        if ids[0] != ids[1] or inspections[0] != inspections[1]:
            raise RuntimeError("independent image builds differ")
        evidence = {
            "base": base,
            "candidate_image": ids[0],
            "normalized_inspect_sha256": hashlib.sha256(
                json.dumps(
                    inspections[0], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "schema": "vulcan-image-rebuild-comparison/1",
            "wheel_sha256": args.wheel_sha256,
        }
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            os.write(descriptor, raw.encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"image qualification refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
