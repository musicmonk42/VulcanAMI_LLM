from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

import pytest

from vulcan.core.canonical import canonical_digest, canonical_json
from vulcan.core.decisions import Decision, DecisionCategory, DecisionOutcome
from vulcan.core.faults import FaultInjected, FaultInjector
from vulcan.core.ids import IdKind, from_digest, from_slug, new_id, short_display, validate_id
from vulcan.core.time import DeterministicClock, SystemClock, format_utc, parse_utc


def test_canonical_id_constructors_and_validators() -> None:
    for kind in IdKind:
        issued = new_id(kind)
        assert validate_id(kind, issued.value) == issued
        slugged = from_slug(kind, "stable-name-01")
        assert validate_id(kind, slugged.value) == slugged
        digested = from_digest(kind, "a" * 64)
        assert validate_id(kind, digested.value) == digested

    legacy = validate_id(IdKind.CASE, "case-0123456789abcdef0123456789abcdef", allow_legacy_case=True)
    assert legacy.value.startswith("case-")
    assert new_id(IdKind.CASE).value.startswith("case_")


def test_invalid_ids_and_short_digests_are_rejected() -> None:
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        from_digest(IdKind.EVIDENCE, "a" * 12)
    with pytest.raises(ValueError, match="wrong prefix"):
        validate_id(IdKind.REQUEST, "case_0123456789abcdef0123456789abcdef")
    with pytest.raises(ValueError):
        validate_id(IdKind.CASE, "case-0123456789abcdef0123456789abcdef")
    assert short_display("x" * 64, 12) == "x" * 12 + "…"


def test_clock_utc_precision_and_boundaries() -> None:
    clock = DeterministicClock(datetime(2026, 7, 20, 1, 2, 3, 123456, tzinfo=timezone.utc))
    assert format_utc(clock.now()) == "2026-07-20T01:02:03.123Z"
    clock.advance(timedelta(milliseconds=2, microseconds=900))
    assert format_utc(clock.now()) == "2026-07-20T01:02:03.125Z"
    assert parse_utc("2026-07-20T01:02:03.125Z") == clock.now()
    assert SystemClock().now().tzinfo == timezone.utc
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(timedelta(seconds=-1))
    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc(datetime(2026, 7, 20))
    with pytest.raises(ValueError, match="milliseconds"):
        parse_utc("2026-07-20T01:02:03.123456Z")


def test_canonical_json_round_trips_unicode_normalization_and_digest_stability() -> None:
    left = {"b": [3, True], "é": "café"}
    right = {"e\u0301": "cafe\u0301", "b": [3, True]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_digest(left) == canonical_digest(right)
    assert canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}'


@dataclass(frozen=True)
class Sample:
    when: datetime
    status: DecisionOutcome


def test_canonical_json_typed_values_and_rejections() -> None:
    sample = Sample(datetime(2026, 7, 20, tzinfo=timezone.utc), DecisionOutcome.BLOCK)
    assert canonical_json(sample) == b'{"status":"BLOCK","when":"2026-07-20T00:00:00.000Z"}'
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"bad": value})
    with pytest.raises(ValueError, match="control"):
        canonical_json({"bad": "line\nbreak"})
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_json({1: "numeric key"})
    with pytest.raises(ValueError, match="key collision"):
        canonical_json({"é": 1, "e\u0301": 2})
    with pytest.raises(ValueError, match="depth"):
        value: object = "leaf"
        for _ in range(30):
            value = [value]
        canonical_json(value)


def test_closed_decisions_distinguish_denial_outcomes() -> None:
    allowed = Decision.allow(DecisionCategory.READINESS, "cap.bounded_arithmetic")
    assert allowed.allowed
    for outcome in (DecisionOutcome.BLOCK, DecisionOutcome.ERROR, DecisionOutcome.CANCELLED, DecisionOutcome.UNAVAILABLE, DecisionOutcome.STALE):
        denied = Decision.deny(DecisionCategory.SAFETY, outcome, "case_0123456789abcdef0123456789abcdef", outcome.value.lower())
        assert not denied.allowed
        assert denied.outcome is outcome
    with pytest.raises(ValueError, match="ALLOW"):
        Decision.deny(DecisionCategory.POLICY, DecisionOutcome.ALLOW, "policy_test", "bad")


def test_fault_points_are_test_only_and_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    disabled = FaultInjector.disabled()
    with pytest.raises(RuntimeError, match="disabled"):
        disabled.arm("txn.before_commit")
    disabled.check("txn.before_commit")

    with pytest.raises(RuntimeError, match="test-only"):
        FaultInjector.for_tests("wrong")
    monkeypatch.setenv("VULCAN_ENABLE_TEST_FAULTS", "VULCAN_TEST_FAULTS_ONLY")
    injector = FaultInjector.for_tests("VULCAN_TEST_FAULTS_ONLY")
    injector.arm("txn.before_commit")
    with pytest.raises(FaultInjected, match="txn.before_commit"):
        injector.check("txn.before_commit")
    injector.check("txn.before_commit")
    with pytest.raises(ValueError, match="lowercase"):
        injector.arm("Bad Name")
