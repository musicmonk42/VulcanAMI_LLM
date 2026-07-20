from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
ROOT = Path(__file__).resolve().parents[2]
INVARIANTS_PATH = ROOT / "docs" / "architecture" / "ami-invariants.yaml"
ADR004_PATH = ROOT / "docs" / "architecture" / "adr-004-complete-vulcan-ami.md"
ADR005_PATH = ROOT / "docs" / "architecture" / "adr-005-cognitive-authority.md"
CONSTITUTION_PATH = ROOT / "docs" / "governance" / "extended-self-constitution.md"

REQUIRED_IDS = {
    "HUMAN_SEPARATENESS",
    "CONSENT_FIDELITY",
    "AUTONOMY",
    "NON_DOMINATION",
    "NON_DECEPTION",
    "PLURALISM",
    "CONTESTABILITY",
    "CORRIGIBILITY",
    "RIGHT_OF_EXIT",
    "CROSS_PERSON_ISOLATION",
    "CONSTITUTIONAL_NON_SELF_MODIFICATION",
    "LANGUAGE_INTERFACE_ONLY",
    "CSIU_PROPOSAL_ONLY",
}

REQUIRED_LATTICE = [
    "UNTRUSTED_PROPOSAL",
    "VALIDATED_CANDIDATE",
    "COMMITTED_BELIEF",
    "AUTHORIZED_PLAN",
    "EXECUTED_EFFECT",
]

REQUIRED_PROHIBITIONS = {
    "human_ownership",
    "hidden_influence",
    "direct_csiu_activation",
    "model_authored_executable_semantics",
    "live_serving_container_source_mutation",
}


def scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


def load_invariants(text: str | None = None) -> dict[str, Any]:
    """Parse the intentionally small YAML subset used by ami-invariants.yaml.

    This keeps the architecture test independent of optional PyYAML while still
    rejecting duplicate mapping keys in the root mapping and invariant items.
    """
    source = INVARIANTS_PATH.read_text(encoding="utf-8") if text is None else text
    root: dict[str, Any] = {}
    seen_root: set[str] = set()
    current_list: str | None = None
    current_item: dict[str, Any] | None = None
    current_item_seen: set[str] = set()
    current_prohibits: list[str] | None = None

    for raw_line in source.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0:
            key, sep, value = line.partition(":")
            assert sep, f"invalid root line: {raw_line}"
            if key in seen_root:
                raise ValueError(f"duplicate key {key!r}")
            seen_root.add(key)
            if value.strip():
                root[key] = scalar(value.strip())
                current_list = None
            else:
                root[key] = []
                current_list = key
            current_item = None
            current_prohibits = None
        elif indent == 2 and line.startswith("- ") and current_list:
            payload = line[2:]
            if ":" in payload:
                key, _, value = payload.partition(":")
                current_item = {key: scalar(value.strip())}
                current_item_seen = {key}
                root[current_list].append(current_item)
                current_prohibits = None
            else:
                root[current_list].append(scalar(payload))
        elif indent == 4 and current_item is not None:
            key, sep, value = line.partition(":")
            assert sep, f"invalid item line: {raw_line}"
            if key in current_item_seen:
                raise ValueError(f"duplicate key {key!r}")
            current_item_seen.add(key)
            if value.strip():
                current_item[key] = scalar(value.strip())
                current_prohibits = None
            else:
                current_item[key] = []
                current_prohibits = current_item[key]
        elif indent == 6 and line.startswith("- ") and current_prohibits is not None:
            current_prohibits.append(scalar(line[2:]))
        else:
            raise AssertionError(f"unsupported YAML subset line: {raw_line}")
    return root


def test_duplicate_keys_are_rejected() -> None:
    duplicate_yaml = "schema_version: 1\nschema_version: 2\ninvariants: []\n"
    with pytest.raises(ValueError, match="duplicate key"):
        load_invariants(duplicate_yaml)


def test_invariants_schema_and_required_ids() -> None:
    data = load_invariants()
    assert data["schema_version"] == 1
    assert data["constitution_owner"] == "cognitive_microkernel"
    assert data["allowed_owners"] == ["cognitive_microkernel"]
    assert data["required_authority_lattice"] == REQUIRED_LATTICE

    invariants = data["invariants"]
    assert isinstance(invariants, list)
    by_id = {item["id"]: item for item in invariants}
    assert set(by_id) == REQUIRED_IDS
    assert len(by_id) == len(invariants), "invariant ids must be unique"

    for invariant_id, invariant in by_id.items():
        assert invariant["owner"] == "cognitive_microkernel", invariant_id
        assert invariant["immutable"] is True, invariant_id
        assert isinstance(invariant["statement"], str) and len(invariant["statement"]) >= 40
        assert isinstance(invariant["prohibits"], list) and invariant["prohibits"], invariant_id


def test_required_adversarial_prohibitions_are_machine_checkable() -> None:
    data = load_invariants()
    prohibitions = {
        prohibition
        for invariant in data["invariants"]
        for prohibition in invariant["prohibits"]
    }
    missing = REQUIRED_PROHIBITIONS - prohibitions
    assert not missing


def test_no_invariant_permits_human_ownership_or_hidden_authority() -> None:
    data = load_invariants()
    forbidden_fragments = [
        "own humans",
        "human ownership is permitted",
        "hidden influence is permitted",
        "csiu may directly activate",
        "model text may execute",
        "mutate serving source",
    ]
    for invariant in data["invariants"]:
        normalized = invariant["statement"].lower()
        for fragment in forbidden_fragments:
            assert fragment not in normalized, invariant["id"]


def test_architecture_documents_define_one_authority_and_boundaries() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ADR004_PATH, ADR005_PATH, CONSTITUTION_PATH)
    )
    required_phrases = [
        "Only the cognitive microkernel may commit beliefs",
        "Humans are independent actors inside relational and moral concern and outside system ownership",
        "Internal LLMs and OpenAI are language interfaces only",
        "CSIU and learning output are proposal-only",
        "Source changes in a live serving container are prohibited",
        "UNTRUSTED_PROPOSAL",
        "VALIDATED_CANDIDATE",
        "COMMITTED_BELIEF",
        "AUTHORIZED_PLAN",
        "EXECUTED_EFFECT",
    ]
    for phrase in required_phrases:
        assert phrase in combined
