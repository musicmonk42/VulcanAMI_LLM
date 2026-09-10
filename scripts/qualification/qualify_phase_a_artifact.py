#!/usr/bin/env python3
"""Execute and attest all critical checks against one immutable Phase-A subject."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = re.compile(r"^[^\s@]+@sha256:([0-9a-f]{64})$")
HEX = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SCENARIOS = frozenset(
    {
        "wheel-verify",
        "image-rebuild-a",
        "image-rebuild-b",
        "image-compare",
        "installed-normal",
        "installed-optimized",
        "boot-lifespan",
        "six-routes",
        "auth-actor-idempotency",
        "arithmetic-publication",
        "journal-replay",
        "audit-rebuild",
        "authority-counterexamples",
        "kill-restart",
        "tcb-manifest",
        "sbom",
        "nonroot-readonly",
    }
)
REQUIRED_BINDINGS = frozenset(
    {
        "source",
        "runtime_lock",
        "build_lock",
        "builder_base",
        "runtime_base",
        "actor_binding_schema",
        "actor_binding_vectors",
        "journal_schema",
        "authority_verification",
        "migration_policy",
        "route_manifest",
        "import_manifest",
        "test_catalog",
        "harness",
    }
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def immutable_image(value: str) -> str:
    if IMAGE.fullmatch(value) is None:
        raise ValueError("qualification subject must be repository@sha256:digest")
    return value


def require_clean_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    if status.returncode or status.stdout:
        raise RuntimeError("qualification requires a clean source tree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("source revision is not immutable")
    return revision


def load_document(path: Path) -> object:
    return json.loads(path.read_text())


def derive_bindings(
    *, revision: str, catalog: Path, builder_base: str, runtime_base: str
) -> dict[str, str]:
    actor = load_document(ROOT / "config/actor-binding-schema.json")
    return {
        "source": digest(revision.encode()),
        "runtime_lock": digest((ROOT / "requirements-runtime.lock").read_bytes()),
        "build_lock": digest((ROOT / "requirements-build.lock").read_bytes()),
        "builder_base": immutable_image(builder_base).rsplit(":", 1)[1],
        "runtime_base": immutable_image(runtime_base).rsplit(":", 1)[1],
        "actor_binding_schema": digest(
            (ROOT / "config/actor-binding-schema.json").read_bytes()
        ),
        "actor_binding_vectors": digest(canonical(actor["golden_vectors"])),
        "journal_schema": digest(
            (ROOT / "config/constitutional-journal-schema.json").read_bytes()
        ),
        "authority_verification": digest(
            (ROOT / "config/authority-verification-history.json").read_bytes()
        ),
        "migration_policy": digest(
            (
                ROOT / "docs/architecture/legacy-migration-and-audit-projection.md"
            ).read_bytes()
        ),
        "route_manifest": digest(
            (ROOT / "src/vulcan/runtime/route_manifest.py").read_bytes()
        ),
        "import_manifest": digest(
            (ROOT / "config/production-import-policy.json").read_bytes()
        ),
        "test_catalog": digest(catalog.read_bytes()),
        "harness": digest(
            (ROOT / "scripts/qualification/journal_recovery_harness.py").read_bytes()
        ),
    }


def execute(argv: list[str], *, subject: str, wheel: Path) -> dict[str, object]:
    expanded = [
        part.replace("{subject}", subject).replace("{wheel}", str(wheel))
        for part in argv
    ]
    if not expanded or subject not in expanded:
        raise ValueError("every scenario argv must name the immutable image subject")
    result = subprocess.run(expanded, cwd=ROOT, capture_output=True)
    return {
        "argv": expanded,
        "exit_status": result.returncode,
        "stdout_sha256": digest(result.stdout),
        "stderr_sha256": digest(result.stderr),
        "passed": result.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--builder-base", required=True)
    parser.add_argument("--runtime-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subject = immutable_image(args.subject)
    if args.output.exists():
        raise RuntimeError("qualification output must not already exist")
    if not args.wheel.is_file() or not HEX.fullmatch(args.wheel_sha256):
        raise ValueError("frozen candidate wheel and immutable digest are required")
    actual_wheel = digest(args.wheel.read_bytes())
    if actual_wheel != args.wheel_sha256:
        raise RuntimeError("candidate wheel digest mismatch")
    revision = require_clean_revision()
    catalog = load_document(args.catalog)
    bindings = load_document(args.bindings)
    derived_bindings = derive_bindings(
        revision=revision,
        catalog=args.catalog,
        builder_base=args.builder_base,
        runtime_base=args.runtime_base,
    )
    if not isinstance(catalog, dict) or set(catalog) != REQUIRED_SCENARIOS:
        raise ValueError("critical scenario catalog is incomplete or contains aliases")
    if not isinstance(bindings, dict) or set(bindings) != REQUIRED_BINDINGS:
        raise ValueError("qualification bindings are incomplete")
    if bindings != derived_bindings:
        raise ValueError("qualification bindings do not match reviewed inputs")
    if any(
        not isinstance(value, str) or not HEX.fullmatch(value)
        for value in bindings.values()
    ):
        raise ValueError("every qualification binding must be a sha256 digest")
    observations = {}
    for name in sorted(REQUIRED_SCENARIOS):
        scenario = catalog[name]
        if not isinstance(scenario, list) or not all(
            isinstance(v, str) for v in scenario
        ):
            raise ValueError(f"scenario {name} must be an argv array")
        observations[name] = execute(
            scenario, subject=subject, wheel=args.wheel.resolve()
        )
    if not all(item["passed"] for item in observations.values()):
        failed = sorted(
            name for name, item in observations.items() if not item["passed"]
        )
        raise RuntimeError("critical qualification failed: " + ",".join(failed))
    bundle = {
        "bindings": bindings,
        "catalog_sha256": digest(args.catalog.read_bytes()),
        "image_digest": subject.rsplit("@", 1)[1],
        "observations": observations,
        "phase_b_admission": "ELIGIBLE",
        "schema": "vulcan-phase-a-qualification/1",
        "source_commit": revision,
        "tree_clean": True,
        "wheel_sha256": actual_wheel,
    }
    bundle["bundle_sha256"] = digest(canonical(bundle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output, flags, 0o444)
    try:
        os.write(fd, canonical(bundle) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"qualification refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
