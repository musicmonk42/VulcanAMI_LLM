#!/usr/bin/env python3
"""Verify exact wheel inclusion and every RECORD member digest."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/wheel-inclusion-manifest.json"


def digest(raw):
    return hashlib.sha256(raw).digest()


def encoded(raw):
    return "sha256=" + base64.urlsafe_b64encode(digest(raw)).rstrip(b"=").decode()


def verify(wheel: Path):
    manifest = json.loads(MANIFEST.read_text())
    expected_digests = {
        row["path"].removeprefix("src/"): row["sha256"] for row in manifest["files"]
    }
    expected_digests.update(
        {row["wheel_path"]: row["sha256"] for row in manifest["resources"]}
    )
    expected = set(expected_digests)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate wheel member")
        if any(
            PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts
            for n in names
        ):
            raise RuntimeError("unsafe wheel member")
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError("wheel must contain exactly one RECORD")
        dist_root = records[0].rsplit("/", 1)[0]
        expected_metadata = {
            f"{dist_root}/METADATA",
            f"{dist_root}/WHEEL",
            f"{dist_root}/top_level.txt",
            f"{dist_root}/licenses/LICENSE",
            records[0],
        }
        metadata = {n for n in names if ".dist-info/" in n}
        if metadata != expected_metadata:
            raise RuntimeError("wheel metadata inclusion mismatch")
        actual = set(names) - metadata
        if actual != expected:
            raise RuntimeError(
                f"wheel inclusion mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
            )
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        record = {path: (value, size) for path, value, size in rows}
        if set(record) != set(names):
            raise RuntimeError("RECORD membership mismatch")
        for name in names:
            value, size = record[name]
            raw = archive.read(name)
            if name == records[0]:
                if value or size:
                    raise RuntimeError("RECORD self-entry must be empty")
            elif value != encoded(raw) or size != str(len(raw)):
                raise RuntimeError(f"RECORD verification failed: {name}")
        members = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(actual)
        }
        mismatched = {
            name: {"expected": expected_digests[name], "actual": members[name]}
            for name in sorted(actual)
            if members[name] != expected_digests[name]
        }
        if mismatched:
            raise RuntimeError(
                "wheel content differs from reviewed inclusion manifest: "
                + json.dumps(mismatched, sort_keys=True)
            )
    return {
        "schema": "vulcan-wheel-verification/v1",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "file_count": len(members),
        "member_digests": members,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    result = verify(a.wheel)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
