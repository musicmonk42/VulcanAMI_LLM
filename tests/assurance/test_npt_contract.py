from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.check_production_imports import check
from vulcan.assurance.npt_contract import (
    DEFAULT_CONTRACT,
    DEFAULT_DIGEST,
    DIMENSIONS,
    NPTContractError,
    load_contract,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def mutable_contract() -> dict[str, object]:
    return json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def test_canonical_round_trip_and_exact_replay() -> None:
    contract = load_contract()
    replay = (
        json.dumps(
            dict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )
    assert replay == DEFAULT_CONTRACT.read_bytes()
    assert hashlib.sha256(replay).hexdigest() == DEFAULT_DIGEST.read_text().strip()
    assert tuple(item["id"] for item in contract["dimensions"]) == DIMENSIONS
    assert all(
        item["current_gate"]
        == {
            "gate_f": "open",
            "gate_g": "open",
            "evidence_disposition": "insufficient",
        }
        for item in contract["dimensions"]
    )


def test_one_byte_mutation_is_rejected(tmp_path: Path) -> None:
    raw = bytearray(DEFAULT_CONTRACT.read_bytes())
    raw[20] ^= 1
    mutated = tmp_path / "contract.json"
    mutated.write_bytes(raw)
    with pytest.raises(NPTContractError, match="digest mismatch"):
        load_contract(mutated, DEFAULT_DIGEST)


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [("conscious", True), ("consciousness", 0), ("is_conscious", "unknown")],
)
def test_scalar_and_boolean_consciousness_fields_are_rejected(
    bad_key: str, bad_value: object
) -> None:
    contract = mutable_contract()
    contract[bad_key] = bad_value
    with pytest.raises(NPTContractError, match="consciousness fields"):
        validate_contract(contract)


def test_forbidden_claims_are_closed_and_cannot_be_weakened() -> None:
    contract = mutable_contract()
    contract["forbidden_claims"]["product"].remove("consciousness_feature")
    with pytest.raises(NPTContractError, match="forbidden scientific/product claims"):
        validate_contract(contract)


def test_no_dimension_can_claim_gate_f_or_gate_g_is_closed() -> None:
    for gate in ("gate_f", "gate_g"):
        contract = mutable_contract()
        contract["dimensions"][0]["current_gate"][gate] = "closed"
        with pytest.raises(NPTContractError, match="may not close Gate F or G"):
            validate_contract(contract)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "eighth"])
def test_missing_duplicate_or_eighth_dimensions_cannot_be_averaged(
    mutation: str,
) -> None:
    contract = mutable_contract()
    dimensions = contract["dimensions"]
    if mutation == "missing":
        dimensions.pop()
    elif mutation == "duplicate":
        dimensions[-1] = copy.deepcopy(dimensions[0])
    else:
        dimensions.append(copy.deepcopy(dimensions[0]))
    with pytest.raises(
        NPTContractError, match="seven closure dimensions|seven canonical"
    ):
        validate_contract(contract)


def test_absent_rival_or_intervention_and_nonexistent_references_fail_closed() -> None:
    for field in ("rival_explanation", "intervention_family"):
        contract = mutable_contract()
        contract["dimensions"][0][field] = ""
        with pytest.raises(NPTContractError, match=field):
            validate_contract(contract)
    contract = mutable_contract()
    contract["dimensions"][0]["evidence_schema"]["invariant_references"] = ["NOT_REAL"]
    with pytest.raises(NPTContractError, match="nonexistent invariant"):
        validate_contract(contract)


def test_evidence_schema_cannot_drop_a_required_digest() -> None:
    contract = mutable_contract()
    contract["dimensions"][0]["evidence_schema"]["required_fields"].remove(
        "validator_sha256"
    )
    with pytest.raises(NPTContractError, match="evidence schema must be exact"):
        validate_contract(contract)


def test_evidence_commands_are_closed_and_nonfinite_values_are_rejected() -> None:
    contract = mutable_contract()
    contract["evidence"]["commands"].append("curl https://example.invalid")
    with pytest.raises(NPTContractError, match="exact governed set"):
        validate_contract(contract)
    contract = mutable_contract()
    contract["dimensions"][0]["observable_definition"] = float("nan")
    with pytest.raises(NPTContractError):
        validate_contract(contract)


def test_symlinked_evidence_is_rejected(tmp_path: Path) -> None:
    contract = mutable_contract()
    link = ROOT / f".npt-symlink-{tmp_path.name}.json"
    try:
        link.symlink_to(ROOT / "docs/research/npt-flagship-report.json")
        contract["dimensions"][0]["evidence_references"] = [link.name]
        with pytest.raises(NPTContractError, match="unsafe evidence reference"):
            validate_contract(contract)
    finally:
        link.unlink(missing_ok=True)
    contract = mutable_contract()
    contract["dimensions"][0]["evidence_references"] = ["not/real.json"]
    with pytest.raises(NPTContractError, match="nonexistent evidence"):
        validate_contract(contract)


def test_intermediate_symlink_and_release_inspection_behavior(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence.json").write_text("{}", encoding="utf-8")
    link = ROOT / f".npt-dir-{tmp_path.name}"
    try:
        link.symlink_to(outside, target_is_directory=True)
        contract = mutable_contract()
        contract["dimensions"][0]["evidence_references"] = [
            f"{link.name}/evidence.json"
        ]
        with pytest.raises(NPTContractError, match="unsafe evidence reference"):
            validate_contract(contract)
    finally:
        link.unlink(missing_ok=True)

    release_root = tmp_path / "release"
    (release_root / "config").mkdir(parents=True)
    (release_root / "docs/architecture").mkdir(parents=True)
    contract_path = release_root / "config/npt-engineering-contract.json"
    digest_path = release_root / "config/npt-engineering-contract.sha256"
    contract_path.write_bytes(DEFAULT_CONTRACT.read_bytes())
    digest_path.write_bytes(DEFAULT_DIGEST.read_bytes())
    (release_root / "docs/architecture/ami-invariants.yaml").write_bytes(
        (ROOT / "docs/architecture/ami-invariants.yaml").read_bytes()
    )
    inspected = load_contract(
        contract_path, digest_path, root=release_root, verify_artifacts=False
    )
    assert inspected["theory_status"] == "research_hypothesis"


def test_unknown_duplicate_fields_and_excess_maturity_are_rejected() -> None:
    contract = mutable_contract()
    contract["dimensions"][0]["surprise"] = "authority"
    with pytest.raises(NPTContractError, match="unknown"):
        validate_contract(contract)
    duplicate = DEFAULT_CONTRACT.read_text().replace(
        '"schema":"vulcan-npt-engineering-contract/v1"',
        '"schema":"vulcan-npt-engineering-contract/v1","schema":"duplicate"',
        1,
    )
    duplicate_path = ROOT / ".npt-duplicate-test.json"
    digest_path = ROOT / ".npt-duplicate-test.sha256"
    try:
        duplicate_path.write_text(duplicate)
        digest_path.write_text(hashlib.sha256(duplicate.encode()).hexdigest())
        with pytest.raises(NPTContractError, match="duplicate field"):
            load_contract(duplicate_path, digest_path)
    finally:
        duplicate_path.unlink(missing_ok=True)
        digest_path.unlink(missing_ok=True)
    contract = mutable_contract()
    contract["dimensions"][0]["maturity"] = "M3"
    with pytest.raises(NPTContractError, match="maturity exceeds"):
        validate_contract(contract)


def test_production_may_inspect_contract_but_research_imports_remain_denied() -> None:
    contract = load_contract()
    assert contract["authority"]["production_access"] == "inspect_contract_only"
    policy = json.loads((ROOT / "config/production-import-policy.json").read_text())
    assert "vulcan.assurance.npt_contract" in policy["allowed_prefixes"]
    assert "vulcan.research" in policy["denied_prefixes"]
    source = (ROOT / "src/vulcan/assurance/npt_contract.py").read_text()
    assert "vulcan.research" not in source
    wheel_manifest = json.loads(
        (ROOT / "config/wheel-inclusion-manifest.json").read_text()
    )
    release_paths = {item["path"] for item in wheel_manifest["resources"]}
    release_modules = {item["module"] for item in wheel_manifest["files"]}
    assert {
        "config/npt-engineering-contract.json",
        "config/npt-engineering-contract.sha256",
        "config/production-import-policy.json",
    } <= release_paths
    assert "vulcan.assurance.npt_contract" in release_modules
    _, errors = check()
    assert errors == []


def test_validator_is_enforced_under_optimized_python() -> None:
    completed = subprocess.run(
        [sys.executable, "-O", "scripts/ci/validate_npt_contract.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "theory_status=research_hypothesis" in completed.stdout
