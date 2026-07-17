import hashlib
import json

import pytest

from vulcan.local_language.governance import (
    DatasetSource, ExampleRole, GovernanceError, LanguageExample, ReleaseState,
    transition_release, validate_grouped_split,
)
from vulcan.local_language.tokenizer import load_tokenizer_contract
from vulcan.local_language.release import ReleaseVerificationError


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source():
    return DatasetSource("synthetic:arithmetic", "LicenseRef-project", "synthetic", "adapter-training", "non-sensitive", "no-retention", ("training",))


def _example(split="development"):
    return LanguageExample("example:1", ExampleRole.INPUT_PROPOSAL, "und", "bounded-arithmetic", "semantic-ingress/2", "formal-arithmetic/2", _digest("input"), _digest("target"), "interpretation_proposal", "synthetic:arithmetic", "generator:1", "family:1", split)


def test_dataset_is_default_deny_and_groups_cannot_cross_locked_splits():
    validate_grouped_split((_example(),), _source())
    with pytest.raises(GovernanceError, match="leaks"):
        validate_grouped_split((_example(), _example("promotion_test")), _source())
    with pytest.raises(GovernanceError):
        _example().validate(DatasetSource("synthetic:arithmetic", "", "synthetic", "training", "non-sensitive", "none", ("training",)))


def test_release_transition_requires_external_release_authority():
    assert transition_release(ReleaseState.EXPERIMENTAL, ReleaseState.EVALUATED, authority="evaluator") is ReleaseState.EVALUATED
    with pytest.raises(GovernanceError):
        transition_release(ReleaseState.EVALUATED, ReleaseState.APPROVED, authority="trainer")


def test_tokenizer_contract_is_immutable_and_rejects_duplicate_vocabulary(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"schema_version":"local-tokenizer/1", "normalization":"NFC", "vocabulary":["<pad>", "2", "+"], "special_tokens":["<pad>"], "max_length":32}))
    assert load_tokenizer_contract(path).vocabulary == ("<pad>", "2", "+")
    path.write_text(json.dumps({"schema_version":"local-tokenizer/1", "normalization":"NFC", "vocabulary":["<pad>", "2", "2"], "special_tokens":["<pad>"], "max_length":32}))
    with pytest.raises(ReleaseVerificationError):
        load_tokenizer_contract(path)
