from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from vulcan.assurance.capabilities import CapabilityRecord, CapabilityRegistry, CapabilityStatus, EvidenceArtifactRef
from vulcan.runtime.capabilities import composed_runtime_ports, public_capability_response

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "capabilities.yaml"
NOW = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)


def registry() -> CapabilityRegistry:
    return CapabilityRegistry.from_json_text(CONFIG.read_text(encoding="utf-8"), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())


def records() -> dict[str, CapabilityRecord]:
    return registry().records


def serialize(records_by_id: dict[str, CapabilityRecord]) -> str:
    def artifact(a: EvidenceArtifactRef) -> dict[str, str]:
        return {"evidence_type": a.evidence_type, "artifact_uri": a.artifact_uri, "artifact_sha256": a.artifact_sha256, "schema": a.schema}

    payload = {"capabilities": []}
    for record in records_by_id.values():
        payload["capabilities"].append({
            "capability_id": record.capability_id,
            "status": record.status.value,
            "owner": record.owner,
            "implementation_digest": record.implementation_digest,
            "release_digest": record.release_digest,
            "route_reachability": list(record.route_reachability),
            "port_reachability": list(record.port_reachability),
            "evaluation_artifact": artifact(record.evaluation_artifact),
            "safety_artifacts": [artifact(item) for item in record.safety_artifacts],
            "impact_artifacts": [artifact(item) for item in record.impact_artifacts],
            "active_policy_digest": record.active_policy_digest,
            "rollback_method": record.rollback_method,
            "limitations": list(record.limitations),
            "review_date": record.review_date.isoformat(),
            "expires_at": record.expires_at.isoformat().replace("+00:00", "Z"),
            "dependencies": list(record.dependencies),
        })
    return json.dumps(payload, sort_keys=True)


def test_registry_loads_current_config_and_public_projection_is_evidence_backed() -> None:
    loaded = registry()
    assert set(loaded.records) >= {"cap.bounded_arithmetic", "cap.broad_reasoning", "cap.internal_llm", "cap.learning", "cap.self_improvement"}
    public_ids = {item["capability_id"] for item in loaded.public_capabilities()}
    assert public_ids == {"cap.bounded_arithmetic"}
    assert loaded.records["cap.bounded_arithmetic"].evaluation_artifact.artifact_uri.startswith("file://tests/")
    assert loaded.records["cap.bounded_arithmetic"].port_reachability == ("POST /v1/chat",)


def test_runtime_public_capabilities_match_registry_active_projection() -> None:
    expected = {item["capability_id"] for item in registry().public_capabilities()}
    actual = {item["capability_id"] for item in public_capability_response(NOW)["capabilities"]}
    assert actual == expected
    assert "cap.broad_reasoning" not in actual
    assert "cap.internal_llm" not in actual
    assert "cap.learning" not in actual


def test_status_enum_rejects_unknown_statuses() -> None:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data["capabilities"][0]["status"] = "MARKETING_READY"
    with pytest.raises(ValueError):
        CapabilityRegistry.from_json_text(json.dumps(data), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())


def test_missing_owner_invalid_digest_expired_evidence_and_missing_port_fail_closed() -> None:
    base = records()
    bad_owner = dict(base)
    bad_owner["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], owner="")
    with pytest.raises(ValueError, match="owner"):
        CapabilityRegistry(list(bad_owner.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())

    bad_digest = dict(base)
    bad_digest["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], implementation_digest="abc")
    with pytest.raises(ValueError, match="implementation_digest"):
        CapabilityRegistry(list(bad_digest.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())

    expired = dict(base)
    expired["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="expired"):
        CapabilityRegistry(list(expired.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())

    with pytest.raises(ValueError, match="composed port"):
        CapabilityRegistry(list(base.values()), root=ROOT, now=NOW, composed_ports=set())


def test_digest_mismatch_and_test_path_without_artifact_do_not_satisfy_capability() -> None:
    base = records()
    artifact = base["cap.bounded_arithmetic"].evaluation_artifact
    mismatched = replace(artifact, artifact_sha256="0" * 64)
    changed = dict(base)
    changed["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], evaluation_artifact=mismatched)
    with pytest.raises(ValueError, match="digest mismatch"):
        CapabilityRegistry(list(changed.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())

    missing_file = replace(artifact, artifact_uri="file://tests/security/does_not_exist.py", artifact_sha256="0" * 64)
    changed["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], evaluation_artifact=missing_file)
    with pytest.raises(FileNotFoundError):
        CapabilityRegistry(list(changed.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())


def test_transitive_suspension_for_missing_or_suspended_dependencies() -> None:
    base = records()
    absent_verifier = dict(base)
    absent_verifier["cap.deterministic_language_fidelity_verifier"] = replace(base["cap.deterministic_language_fidelity_verifier"], status=CapabilityStatus.RESEARCH)
    loaded = CapabilityRegistry(list(absent_verifier.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())
    assert loaded.effective_statuses["cap.bounded_arithmetic"] is CapabilityStatus.SUSPENDED
    assert loaded.public_capabilities() == []

    suspended_learning = dict(base)
    suspended_learning["cap.learning"] = replace(base["cap.learning"], status=CapabilityStatus.SUSPENDED)
    loaded = CapabilityRegistry(list(suspended_learning.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())
    assert loaded.effective_statuses["cap.self_improvement"] is CapabilityStatus.SUSPENDED


def test_invalid_transition_properties_and_circular_dependencies() -> None:
    base = records()
    for status in (CapabilityStatus.ABSENT, CapabilityStatus.RESEARCH, CapabilityStatus.SHADOW, CapabilityStatus.EVALUATED, CapabilityStatus.ADMITTED, CapabilityStatus.SUSPENDED, CapabilityStatus.RETIRED):
        mutated = dict(base)
        mutated["cap.bounded_arithmetic"] = replace(base["cap.bounded_arithmetic"], status=status)
        loaded = CapabilityRegistry(list(mutated.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())
        assert all(item["capability_id"] != "cap.bounded_arithmetic" for item in loaded.public_capabilities())

    circular = dict(base)
    circular["cap.learning"] = replace(base["cap.learning"], dependencies=("cap.self_improvement",))
    with pytest.raises(ValueError, match="circular"):
        CapabilityRegistry(list(circular.values()), root=ROOT, now=NOW, composed_ports=composed_runtime_ports())


def test_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        CapabilityRegistry.from_json_text('{"capabilities": [], "capabilities": []}', root=ROOT, now=NOW, composed_ports=set())
