from __future__ import annotations

import copy
import dataclasses
import pickle
from pathlib import Path

import pytest

from vulcan.microkernel._transition_permits import MutationPort, TransitionEdge
from vulcan.microkernel.principals import Principal, PrincipalKind

D = "d" * 64


def issue(port, **overrides):
    facts = {
        "actor_digest": "a" * 64,
        "episode_id": "episode-one",
        "policy_digest": "b" * 64,
        "snapshot_digest": "c" * 64,
        "validation_digest": "d" * 64,
        "verifier_digest": "e" * 64,
        "expected_prior_episode_digest": "f" * 64,
    }
    facts.update(overrides)
    return port.issue(edge=TransitionEdge.VALIDATION, **facts), facts


def test_live_permit_is_process_local_unserializable_and_single_use():
    port = MutationPort("1" * 64)
    permit, facts = issue(port)
    with pytest.raises(TypeError):
        pickle.dumps(permit)
    with pytest.raises(TypeError):
        copy.copy(permit)
    with pytest.raises(TypeError):
        copy.deepcopy(permit)
    with pytest.raises(TypeError):
        dataclasses.replace(permit)
    with pytest.raises(AttributeError, match="immutable"):
        permit._episode_id = "episode-other"
    receipt = port.consume(permit, edge=TransitionEdge.VALIDATION, **facts)
    assert receipt["edge"] == "validation"
    with pytest.raises(PermissionError, match="consumed"):
        port.consume(permit, edge=TransitionEdge.VALIDATION, **facts)


def test_cross_instance_edge_and_every_stable_binding_replay_fail():
    port = MutationPort("1" * 64)
    permit, facts = issue(port)
    with pytest.raises(PermissionError, match="issuer"):
        MutationPort("1" * 64).consume(permit, edge=TransitionEdge.VALIDATION, **facts)
    with pytest.raises(PermissionError, match="edge"):
        port.consume(permit, edge=TransitionEdge.PUBLICATION, **facts)
    for field in facts:
        candidate = dict(facts)
        candidate[field] = "9" * 64 if field != "episode_id" else "episode-other"
        with pytest.raises(PermissionError, match="binding mismatch"):
            port.consume(permit, edge=TransitionEdge.VALIDATION, **candidate)


def test_system_kernel_is_descriptive_identity_not_a_capability():
    principal = Principal(PrincipalKind.SYSTEM_KERNEL, "constructed", D)
    assert principal.is_kernel  # descriptive compatibility predicate only
    port = MutationPort("1" * 64)
    permit, facts = issue(port)
    with pytest.raises(PermissionError, match="live transition permit"):
        port.consume(principal, edge=TransitionEdge.VALIDATION, **facts)
    # Evidence/grant/command values likewise fail the exact live-capability type check.
    for value in (object(), dataclasses.make_dataclass("Grant", [])()):
        with pytest.raises(PermissionError, match="live transition permit"):
            port.consume(value, edge=TransitionEdge.VALIDATION, **facts)


def test_untrusted_packages_cannot_import_or_name_private_mutation_port():
    root = Path("src/vulcan")
    forbidden = (root / "graphix", root / "language", root / "reasoning")
    for directory in forbidden:
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            text = path.read_text()
            assert "_transition_permits" not in text, path
            assert "MutationPort" not in text, path
