from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from vulcan.assurance.evidence import EvidenceRecord, EvidenceType, sha256_hexdigest

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "docs" / "governance" / "controls.yaml"
RISKS = ROOT / "docs" / "governance" / "risk-register.yaml"
IMPACT = ROOT / "docs" / "governance" / "impact-assessment.yaml"
CROSSWALK = ROOT / "docs" / "governance" / "standards-crosswalk.md"
INVARIANTS = ROOT / "docs" / "architecture" / "ami-invariants.yaml"

EXPECTED_EVIDENCE_TYPES = {item.value for item in EvidenceType}
EXPECTED_STATUSES = {"planned", "implemented", "partial", "not_applicable", "retired"}
REQUIRED_RISKS = {
    "R-AUTHORITY-CONFUSION",
    "R-PROMPT-INJECTION",
    "R-MODEL-SEMANTIC-DRIFT",
    "R-FALSE-EPISTEMIC-COMMITMENT",
    "R-CSIU-MANIPULATION",
    "R-SYCOPHANCY",
    "R-CROSS-USER-CONTAMINATION",
    "R-CONSENT-FAILURE",
    "R-PROVIDER-LEAKAGE",
    "R-TRAINING-POISONING",
    "R-APPROVAL-FORGERY",
    "R-SOURCE-SELF-MODIFICATION",
    "R-AUDIT-EXHAUSTION",
    "R-SPLIT-BRAIN",
    "R-SUPPLY-CHAIN-COMPROMISE",
    "R-DELETION-OVERCLAIM",
}
REQUIRED_STANDARDS = [
    "NIST AI RMF 1.0",
    "NIST AI RMF GenAI Profile",
    "NIST SSDF 1.1",
    "NIST SP 800-218A",
    "NIST SSDF 1.2 draft tracking",
    "ISO/IEC 42001",
    "ISO/IEC 23894",
    "ISO/IEC 42005",
    "ISO/IEC 5338",
    "OWASP GenAI",
    "MITRE ATLAS",
    "SLSA",
    "Sigstore",
    "SPDX",
    "CycloneDX",
]


def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def load_catalog(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)


def invariant_ids() -> set[str]:
    text = INVARIANTS.read_text(encoding="utf-8")
    return {line.split(":", 1)[1].strip() for line in text.splitlines() if line.strip().startswith("- id:")}


def control_map() -> dict[str, dict[str, object]]:
    data = load_catalog(CONTROLS)
    return {control["id"]: control for control in data["controls"]}


def test_duplicate_keys_are_rejected_for_catalog_json_yaml_subset() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        json.loads('{"schema_version":1,"schema_version":2}', object_pairs_hook=reject_duplicate_pairs)


def test_controls_have_required_fields_and_known_values() -> None:
    data = load_catalog(CONTROLS)
    assert data["catalog_owner"] == "cognitive_microkernel"
    assert set(data["recognized_statuses"]) == EXPECTED_STATUSES
    assert set(data["recognized_evidence_types"]) == EXPECTED_EVIDENCE_TYPES
    assert "not certification" in data["certification_disclaimer"]

    known_invariants = invariant_ids()
    controls = data["controls"]
    by_id = {control["id"]: control for control in controls}
    assert len(by_id) == len(controls)

    for control in controls:
        assert control["id"].startswith("AMI-")
        assert control["owner"], control["id"]
        assert control["applicability"], control["id"]
        assert control["failure_disposition"] in {"fail_closed", "block_claim", "block_release", "shadow_only"}
        assert control["review_trigger"], control["id"]
        assert control["implementation_status"] in EXPECTED_STATUSES
        assert set(control["evidence_requirements"]).issubset(EXPECTED_EVIDENCE_TYPES)
        assert control["evidence_requirements"], control["id"]
        assert set(control["constitution_ids"]).issubset(known_invariants), control["id"]
        assert control["standards"], control["id"]
        for dependency in control["depends_on"]:
            assert dependency in by_id, f"{control['id']} depends on unknown {dependency}"


def test_control_dependencies_are_acyclic() -> None:
    controls = control_map()
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(control_id: str) -> None:
        if control_id in permanent:
            return
        if control_id in temporary:
            raise AssertionError(f"circular dependency at {control_id}")
        temporary.add(control_id)
        for dependency in controls[control_id]["depends_on"]:
            visit(dependency)
        temporary.remove(control_id)
        permanent.add(control_id)

    for control_id in controls:
        visit(control_id)


def test_standards_crosswalk_covers_required_frameworks_without_certification_claims() -> None:
    combined = CONTROLS.read_text(encoding="utf-8") + "\n" + CROSSWALK.read_text(encoding="utf-8")
    for standard in REQUIRED_STANDARDS:
        assert standard in combined
    forbidden_claims = ["certified", "certification achieved", "compliant with", "legally compliant"]
    normalized = combined.lower()
    for claim in forbidden_claims:
        assert claim not in normalized


def test_risk_register_has_required_high_critical_risks_and_all_control_kinds() -> None:
    risks = load_catalog(RISKS)["risks"]
    controls = control_map()
    by_id = {risk["id"]: risk for risk in risks}
    assert set(by_id) == REQUIRED_RISKS
    for risk in risks:
        assert risk["severity"] in {"high", "critical"}
        for key in ("preventive_controls", "detective_controls", "response_controls", "recovery_controls"):
            assert risk[key], f"{risk['id']} missing {key}"
            assert set(risk[key]).issubset(controls), f"{risk['id']} references unknown {key}"


def test_impact_assessment_is_machine_readable_and_constitution_bound() -> None:
    assessment = load_catalog(IMPACT)
    assert assessment["assessment_owner"] == "ImpactAssessmentOwner"
    assert set(assessment["minimum_evidence"]).issubset(EXPECTED_EVIDENCE_TYPES)
    assert set(assessment["constitutional_basis"]).issubset(invariant_ids())
    assert "not certification" in assessment["certification_disclaimer"]


def test_evidence_record_canonical_digest_and_validation() -> None:
    artifact_digest = sha256_hexdigest(b"pytest-result-artifact")
    record = EvidenceRecord(
        evidence_id="ev-control-catalog-pytest",
        evidence_type=EvidenceType.TEST_RESULT,
        control_id="AMI-EVID-001",
        artifact_uri="file://tests/assurance/test_control_catalog.py::test_evidence_record_canonical_digest_and_validation",
        artifact_sha256=artifact_digest,
        produced_at=datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc),
        producer="pytest",
        schema="ami-assurance-evidence/v1",
        metadata={"command": "python -m pytest -q tests/assurance/test_control_catalog.py"},
    )
    assert record.to_canonical_json() == record.to_canonical_json()
    assert record.record_sha256() == sha256_hexdigest(record.to_canonical_json())


def test_evidence_record_rejects_bad_inputs_and_test_path_without_artifact() -> None:
    base = {
        "evidence_id": "ev-good",
        "evidence_type": "source_digest",
        "control_id": "AMI-EVID-001",
        "artifact_uri": "git://repo/commit/abcdef",
        "artifact_sha256": "a" * 64,
        "produced_at": "2026-07-20T00:00:00Z",
        "producer": "unit-test",
        "schema": "ami-assurance-evidence/v1",
        "metadata": {"path": "src/vulcan/assurance/evidence.py"},
    }
    EvidenceRecord.from_json(json.dumps(base))

    bad_digest = dict(base, artifact_sha256="abc")
    with pytest.raises(ValueError, match="SHA-256"):
        EvidenceRecord.from_json(json.dumps(bad_digest))

    non_utc = dict(base, produced_at="2026-07-20T00:00:00+01:00")
    with pytest.raises(ValueError, match="UTC Z"):
        EvidenceRecord.from_json(json.dumps(non_utc))

    unknown_type = dict(base, evidence_type="test_path")
    with pytest.raises(ValueError):
        EvidenceRecord.from_json(json.dumps(unknown_type))

    secret_metadata = dict(base, metadata={"api_token": "do-not-store"})
    with pytest.raises(ValueError, match="secrets"):
        EvidenceRecord.from_json(json.dumps(secret_metadata))

    test_path_only = dict(base, artifact_uri="file://tests/test_fake.py")
    with pytest.raises(ValueError, match="test_result"):
        EvidenceRecord.from_json(json.dumps(test_path_only))
