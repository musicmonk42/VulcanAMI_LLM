"""Offline gates for local language-adapter artifact manifests."""
import hashlib
import json

import pytest

from vulcan.local_language import ReleaseRole, ReleaseVerificationError, verify_release


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _release(root, *, promotion="approved", candidate=2.0, baseline=1.0):
    contents = {"weights.bin": b"weights", "tokenizer.json": b"tokenizer", "config.json": b"config", "evaluation.json": b"report"}
    for name, value in contents.items():
        (root / name).write_bytes(value)
    artifacts = [
        {"name": "weights", "path": "weights.bin", "sha256": _sha(contents["weights.bin"])},
        {"name": "tokenizer", "path": "tokenizer.json", "sha256": _sha(contents["tokenizer.json"])},
        {"name": "config", "path": "config.json", "sha256": _sha(contents["config.json"])},
        {"name": "evaluation_report", "path": "evaluation.json", "sha256": _sha(contents["evaluation.json"])},
    ]
    manifest = {
        "schema_version": "local-language-release/1", "release_id": "adapter-01",
        "role": "input-language-adapter", "runtime_abi": "semantic-ingress/2", "license_identifier": "LicenseRef-reviewed",
        "promotion": {"state": promotion, "approval_id": "review-01"}, "artifacts": artifacts,
        "evaluation": {"report_sha256": artifacts[-1]["sha256"], "deterministic_baseline": "deterministic-parser/2", "baseline_score": baseline, "candidate_score": candidate, "zero_tolerance_passed": True},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_release_binds_complete_approved_role_specific_artifacts(tmp_path):
    _release(tmp_path)
    release = verify_release(tmp_path)
    assert release.role is ReleaseRole.INPUT
    assert release.release_id == "adapter-01"


@pytest.mark.parametrize("mutate", ["digest", "path", "unpromoted", "not_better"])
def test_verify_release_fails_closed_for_tampering_or_unjustified_promotion(tmp_path, mutate):
    _release(tmp_path, promotion="candidate" if mutate == "unpromoted" else "approved", candidate=1.0 if mutate == "not_better" else 2.0)
    if mutate == "digest":
        (tmp_path / "weights.bin").write_bytes(b"tampered")
    elif mutate == "path":
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["artifacts"][0]["path"] = "../outside"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReleaseVerificationError):
        verify_release(tmp_path)


def test_duplicate_manifest_keys_are_rejected(tmp_path):
    _release(tmp_path)
    (tmp_path / "manifest.json").write_text('{"schema_version":"local-language-release/1","schema_version":"local-language-release/1"}')
    with pytest.raises(ReleaseVerificationError, match="duplicate JSON key"):
        verify_release(tmp_path)
