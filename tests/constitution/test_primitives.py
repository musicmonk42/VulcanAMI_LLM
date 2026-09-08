from datetime import datetime, timedelta, timezone
import math

import pytest

from vulcan.constitution.primitives import (
    ArtifactId,
    AuthorityLevel,
    CommitId,
    Digest,
    EpisodeId,
    LineageId,
    PrincipalId,
    SnapshotId,
    canonical_json,
    canonical_json_loads,
    canonical_timestamp,
    parse_timestamp,
    require_utc,
)
from vulcan.graphix.core import AuthorityLevel as GraphixAuthorityLevel
from vulcan.microkernel.authority import AuthorityLevel as KernelAuthorityLevel


def test_digest_canonical_wire_and_named_legacy_adapter():
    digest = Digest.of_bytes(b"persisted")
    assert str(digest).startswith("sha256:")
    assert Digest.from_legacy_hex(digest.hex) == digest
    with pytest.raises(ValueError):
        Digest(digest.hex)
    with pytest.raises(ValueError):
        Digest("sha256:" + "A" * 64)


@pytest.mark.parametrize(
    "kind", [EpisodeId, ArtifactId, PrincipalId, SnapshotId, CommitId, LineageId]
)
def test_typed_identifier_constructors(kind):
    assert kind("case:123") == "case:123"
    with pytest.raises(ValueError):
        kind("x")


def test_authority_is_monotonic_and_incompatible_comparison_fails_closed():
    assert GraphixAuthorityLevel is AuthorityLevel
    assert KernelAuthorityLevel is AuthorityLevel
    assert AuthorityLevel.EXECUTED_EFFECT.dominates(AuthorityLevel.COMMITTED_BELIEF)
    assert not AuthorityLevel.UNTRUSTED_PROPOSAL.dominates(
        AuthorityLevel.VALIDATED_CANDIDATE
    )
    with pytest.raises(TypeError):
        AuthorityLevel.EXECUTED_EFFECT.dominates("COMMITTED_BELIEF")  # type: ignore[arg-type]
    assert AuthorityLevel.EXECUTED_EFFECT != "EXECUTED_EFFECT"


def test_canonical_json_rejects_ambiguous_or_unsupported_values():
    assert canonical_json({"b": 1, "a": [True]}) == b'{"a":[true],"b":1}'
    for value in (math.nan, math.inf, 2**53, object(), "e\u0301", "bad\ud800"):
        with pytest.raises((TypeError, ValueError)):
            canonical_json(value)
    with pytest.raises(ValueError):
        canonical_json_loads('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        canonical_json_loads('{"a":NaN}')


def test_utc_helper_is_canonical_and_rejects_naive_time():
    local = datetime(2026, 1, 1, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=1)))
    assert canonical_timestamp(local) == "2026-01-01T00:02:03.000Z"
    assert require_utc(local).tzinfo is timezone.utc
    assert parse_timestamp("2026-01-01T00:02:03.000Z") == datetime(
        2026, 1, 1, 0, 2, 3, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError):
        require_utc(datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        parse_timestamp("2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        parse_timestamp("2026-01-01T00:00:00.000001Z")
