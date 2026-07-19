import ast
from pathlib import Path

from vulcan.learning_owner import LearningCapabilityStatus, LearningOwner

ROOT = Path(__file__).resolve().parents[1]


def matrix(owner):
    return {c.capability_id: c for c in owner.capability_matrix()}


def test_default_owner_advertises_no_active_learning_capabilities():
    owner = LearningOwner(isolated_test_owner=True)
    caps = matrix(owner)
    assert caps["tool-selection-bandit"].status is LearningCapabilityStatus.SHADOW
    assert caps["metacognition"].status is LearningCapabilityStatus.OBSERVE_ONLY
    assert caps["maml"].status is LearningCapabilityStatus.UNAVAILABLE
    assert caps["ppo"].status is LearningCapabilityStatus.UNAVAILABLE
    assert not [c for c in caps.values() if c.status is LearningCapabilityStatus.ACTIVE]
    for cap in caps.values():
        assert len(cap.implementation_digest) == 64
        assert cap.unavailability_reason


def test_unhealthy_owner_downgrades_dependent_capabilities():
    owner = LearningOwner(isolated_test_owner=True)
    owner.mark_worker_failed()
    assert {c.status for c in owner.capability_matrix()} == {LearningCapabilityStatus.UNHEALTHY}


def test_active_capabilities_require_proof_identifier():
    owner = LearningOwner(isolated_test_owner=True)
    for cap in owner.capability_matrix():
        if cap.status is LearningCapabilityStatus.ACTIVE:
            assert cap.proof_evaluation_id


def test_static_runtime_capabilities_are_owner_derived():
    source = (ROOT / "src/vulcan/runtime/container.py").read_text()
    tree = ast.parse(source)
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert "learning:active" not in constants
    assert "learning:shadow" not in constants
    assert "learning:{learning_status}" not in constants
    assert "learning_owner.capability.value" in source


def test_capability_matrix_document_marks_claims_without_active_language():
    doc = (ROOT / "docs/learning_capability_matrix.md").read_text().lower()
    prohibited = ["production-ready", "hallucination prevention | active", "ppo | active", "maml | active", "proto | active"]
    for phrase in prohibited:
        assert phrase not in doc
    for required in ("progressive continual learning", "rlhf shadow reward", "world-model planning", "unavailable", "experimental", "shadow-only"):
        assert required in doc
