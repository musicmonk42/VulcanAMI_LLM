"""Typed observation and kernel-governed reafference candidates.

Environment adapters report measurements; they never commit state.  The
deterministic assessor produces a validated candidate and the transaction
service durably reconciles its receipt chain before kernel callbacks may commit
updates or advance lineage.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, cast

from vulcan.constitution.primitives import Digest, canonical_json

from .principals import Principal


class ReafferenceError(RuntimeError):
    """A reafference chain failed closed."""


class ReafferenceConflict(ReafferenceError):
    """A chain was replayed or concurrently changed."""


class ReafferenceCrash(BaseException):
    """Test-only simulated process termination at a durable boundary."""


class ReafferenceState(str, Enum):
    CANDIDATE = "candidate"
    UPDATES_COMMITTED = "updates_committed"
    RECONCILED = "reconciled"


def _digest(value: object) -> str:
    return cast(str, Digest.of_bytes(canonical_json(value)).hex)


def _valid_digest(value: str) -> None:
    Digest.from_legacy_hex(value)


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid {name}")


def _timestamp(value: str, name: str) -> None:
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    if not value.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be canonical UTC")


def _probability(value: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0,1]")


def _text_tuple(values: tuple[str, ...], name: str, *, required: bool = False) -> None:
    if type(values) is not tuple or (required and not values) or len(values) > 64:
        raise ValueError(f"invalid {name}")
    for value in values:
        _text(value, name)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


def _strict_document(document: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReafferenceError("duplicate persisted JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(document, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ReafferenceError("invalid persisted reafference document") from exc
    if not isinstance(value, dict) or canonical_json(value).decode() != document:
        raise ReafferenceError("reafference document is not canonical JSON")
    return value


@dataclass(frozen=True)
class ObservedChange:
    field: str
    before_digest: str
    after_digest: str

    def __post_init__(self) -> None:
        _text(self.field, "observed field")
        _valid_digest(self.before_digest)
        _valid_digest(self.after_digest)

    def to_json(self) -> dict[str, str]:
        return {
            "after_digest": self.after_digest,
            "before_digest": self.before_digest,
            "field": self.field,
        }


@dataclass(frozen=True)
class Observation:
    observation_id: str
    adapter_id: str
    expected_effect_ref: str
    receipt_ref: str
    observed_changes: tuple[ObservedChange, ...]
    observed_at: str
    latency_ms: int
    intervention_ref: str
    action_channel_active: bool
    uncertainty: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.observation_id, "observation id"),
            (self.adapter_id, "adapter id"),
            (self.intervention_ref, "intervention ref"),
        ):
            _text(value, name)
        _timestamp(self.observed_at, "observed at")
        _valid_digest(self.expected_effect_ref)
        _valid_digest(self.receipt_ref)
        if not self.observed_changes:
            raise ValueError("observation requires a measured change")
        if type(self.latency_ms) is not int or self.latency_ms < 0:
            raise ValueError("latency must be a non-negative integer")
        if type(self.action_channel_active) is not bool:
            raise ValueError("control evidence must be Boolean")
        if (
            type(self.observed_changes) is not tuple
            or not self.observed_changes
            or not all(
                isinstance(change, ObservedChange) for change in self.observed_changes
            )
        ):
            raise ValueError("observation requires an immutable measured change set")
        fields = [change.field for change in self.observed_changes]
        if len(fields) != len(set(fields)):
            raise ValueError("observation contains duplicate measured fields")
        _probability(self.uncertainty, "observation uncertainty")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "action_channel_active": self.action_channel_active,
            "expected_effect_ref": self.expected_effect_ref,
            "intervention_ref": self.intervention_ref,
            "latency_ms": self.latency_ms,
            "observation_id": self.observation_id,
            "observed_at": self.observed_at,
            "observed_changes": [change.to_json() for change in self.observed_changes],
            "receipt_ref": self.receipt_ref,
            "schema_version": "observation.v1",
            "uncertainty": self.uncertainty,
        }
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class ReafferenceAssessment:
    assessment_id: str
    expected_effect_ref: str
    receipt_ref: str
    observation_ref: str
    assessed_at: str
    prediction_error: float
    candidate_causes: tuple[str, ...]
    self_caused_probability: float
    violated_assumptions: tuple[str, ...]
    proposed_world_updates: tuple[str, ...]
    proposed_self_updates: tuple[str, ...]
    proposed_policy_updates: tuple[str, ...]
    uncertainty: float
    authority: str = "validated_candidate"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.assessment_id, "assessment id")
        _timestamp(self.assessed_at, "assessment time")
        for value in (self.expected_effect_ref, self.receipt_ref, self.observation_ref):
            _valid_digest(value)
        if self.authority != "validated_candidate":
            raise ValueError("reafference cannot commit its own updates")
        _probability(self.prediction_error, "prediction error")
        _probability(self.self_caused_probability, "self-caused probability")
        _probability(self.uncertainty, "assessment uncertainty")
        _text_tuple(self.candidate_causes, "candidate cause", required=True)
        _text_tuple(self.violated_assumptions, "violated assumption")
        _text_tuple(self.proposed_world_updates, "proposed world update")
        _text_tuple(self.proposed_self_updates, "proposed self update")
        _text_tuple(self.proposed_policy_updates, "proposed policy update")
        object.__setattr__(self, "digest", _digest(self.to_json(False)))

    def to_json(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            key: getattr(self, key)
            for key in (
                "assessment_id",
                "assessed_at",
                "authority",
                "candidate_causes",
                "expected_effect_ref",
                "observation_ref",
                "prediction_error",
                "proposed_policy_updates",
                "proposed_self_updates",
                "proposed_world_updates",
                "receipt_ref",
                "self_caused_probability",
                "uncertainty",
                "violated_assumptions",
            )
        }
        value["schema_version"] = "reafference-assessment.v1"
        if include_digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class CalibrationMetric:
    assessment_ref: str
    predicted_probability: float
    causal_label: bool
    brier_score: float

    def __post_init__(self) -> None:
        _valid_digest(self.assessment_ref)
        _probability(self.predicted_probability, "predicted probability")
        if type(self.causal_label) is not bool:
            raise ValueError("causal label must be Boolean")
        _probability(self.brier_score, "Brier score")
        expected = (self.predicted_probability - float(self.causal_label)) ** 2
        if not math.isclose(self.brier_score, round(expected, 6), abs_tol=1e-9):
            raise ValueError("Brier score does not match probability and label")

    def to_json(self) -> dict[str, object]:
        return {
            "assessment_ref": self.assessment_ref,
            "brier_score": self.brier_score,
            "causal_label": self.causal_label,
            "predicted_probability": self.predicted_probability,
            "schema_version": "causal-calibration.v1",
        }


def _observation_from_document(document: str) -> Observation:
    raw = _strict_document(document)
    expected_keys = {
        "adapter_id",
        "action_channel_active",
        "digest",
        "expected_effect_ref",
        "intervention_ref",
        "latency_ms",
        "observation_id",
        "observed_at",
        "observed_changes",
        "receipt_ref",
        "schema_version",
        "uncertainty",
    }
    if set(raw) != expected_keys or raw["schema_version"] != "observation.v1":
        raise ReafferenceError("invalid persisted observation schema")
    try:
        changes_raw = raw["observed_changes"]
        if not isinstance(changes_raw, list):
            raise TypeError
        changes = tuple(
            ObservedChange(
                field=str(change["field"]),
                before_digest=str(change["before_digest"]),
                after_digest=str(change["after_digest"]),
            )
            for change in changes_raw
            if isinstance(change, dict)
            and set(change) == {"field", "before_digest", "after_digest"}
        )
        if len(changes) != len(changes_raw):
            raise ValueError
        observation = Observation(
            observation_id=str(raw["observation_id"]),
            adapter_id=str(raw["adapter_id"]),
            expected_effect_ref=str(raw["expected_effect_ref"]),
            receipt_ref=str(raw["receipt_ref"]),
            observed_changes=changes,
            observed_at=str(raw["observed_at"]),
            latency_ms=raw["latency_ms"],  # type: ignore[arg-type]
            intervention_ref=str(raw["intervention_ref"]),
            action_channel_active=raw["action_channel_active"],  # type: ignore[arg-type]
            uncertainty=raw["uncertainty"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReafferenceError("invalid persisted observation") from exc
    if raw["digest"] != observation.digest:
        raise ReafferenceError("persisted observation digest mismatch")
    return observation


def _assessment_from_document(document: str) -> ReafferenceAssessment:
    raw = _strict_document(document)
    fields = {
        "assessment_id",
        "assessed_at",
        "authority",
        "candidate_causes",
        "digest",
        "expected_effect_ref",
        "observation_ref",
        "prediction_error",
        "proposed_policy_updates",
        "proposed_self_updates",
        "proposed_world_updates",
        "receipt_ref",
        "schema_version",
        "self_caused_probability",
        "uncertainty",
        "violated_assumptions",
    }
    if set(raw) != fields or raw["schema_version"] != "reafference-assessment.v1":
        raise ReafferenceError("invalid persisted assessment schema")
    try:
        assessment = ReafferenceAssessment(
            assessment_id=str(raw["assessment_id"]),
            expected_effect_ref=str(raw["expected_effect_ref"]),
            receipt_ref=str(raw["receipt_ref"]),
            observation_ref=str(raw["observation_ref"]),
            assessed_at=str(raw["assessed_at"]),
            prediction_error=raw["prediction_error"],  # type: ignore[arg-type]
            candidate_causes=tuple(raw["candidate_causes"]),  # type: ignore[arg-type]
            self_caused_probability=raw["self_caused_probability"],  # type: ignore[arg-type]
            violated_assumptions=tuple(raw["violated_assumptions"]),  # type: ignore[arg-type]
            proposed_world_updates=tuple(raw["proposed_world_updates"]),  # type: ignore[arg-type]
            proposed_self_updates=tuple(raw["proposed_self_updates"]),  # type: ignore[arg-type]
            proposed_policy_updates=tuple(raw["proposed_policy_updates"]),  # type: ignore[arg-type]
            uncertainty=raw["uncertainty"],  # type: ignore[arg-type]
            authority=str(raw["authority"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReafferenceError("invalid persisted assessment") from exc
    if raw["digest"] != assessment.digest:
        raise ReafferenceError("persisted assessment digest mismatch")
    return assessment


class EnvironmentAdapter(Protocol):
    """Typed, non-language source of environment measurements."""

    adapter_id: str

    def observe(
        self, *, expected_effect_ref: str, receipt_ref: str, observed_at: str
    ) -> Observation: ...


class UpdateCommitter(Protocol):
    """Idempotent microkernel port keyed by the assessment digest."""

    def __call__(self, assessment: ReafferenceAssessment) -> str: ...


class ObservationValidator(Protocol):
    """Admit observations only from a code-registered typed adapter."""

    def __call__(self, observation: Observation) -> None: ...


class AssessmentValidator(Protocol):
    """Admit candidates only from a code-registered assessor."""

    def __call__(
        self, observation: Observation, assessment: ReafferenceAssessment
    ) -> None: ...


class LineageAdvancer(Protocol):
    """Idempotent lineage port keyed by the assessment digest."""

    def __call__(
        self, receipt_ref: str, assessment_ref: str, update_commit_ref: str
    ) -> str: ...


class DeterministicTestWorld:
    """Counterfactual closed loop or observation-matched yoked replay world."""

    adapter_id = "deterministic-test-world.v1"

    def __init__(self, observations: Sequence[int], *, yoked: bool = False) -> None:
        if not observations or any(type(value) is not int for value in observations):
            raise ValueError("test world requires observations")
        if type(yoked) is not bool:
            raise ValueError("yoked mode must be Boolean")
        self._script = tuple(observations)
        self._yoked = yoked
        self._index = 0
        self._state = 0
        self._prior = 0
        self._intervention = "not-applied"

    def act(self, delta: int, intervention_ref: str) -> None:
        if type(delta) is not int:
            raise ValueError("delta must be an integer")
        _text(intervention_ref, "intervention ref")
        self._prior = self._state
        self._intervention = intervention_ref
        if not self._yoked:
            self._state += delta

    def observe(
        self, *, expected_effect_ref: str, receipt_ref: str, observed_at: str
    ) -> Observation:
        if self._index >= len(self._script):
            raise ReafferenceError("observation sequence exhausted")
        scripted = self._script[self._index]
        self._index += 1
        if self._yoked:
            self._state = scripted
        # Closed-loop scripts are assertions about expected results, not drivers.
        elif self._state != scripted:
            raise ReafferenceError("closed-loop action did not produce expected state")
        change = ObservedChange(
            "world.counter", _digest(self._prior), _digest(self._state)
        )
        return Observation(
            f"observation-{self._index}",
            self.adapter_id,
            expected_effect_ref,
            receipt_ref,
            (change,),
            observed_at,
            0,
            self._intervention,
            not self._yoked,
            0.05 if not self._yoked else 0.25,
        )


def assess_reafference(
    observation: Observation,
    *,
    expected_after_digest: str,
    assessed_at: str,
    confounders: Sequence[str] = (),
    external_actors: Sequence[str] = (),
    expected_latency_ms: int = 0,
) -> ReafferenceAssessment:
    """Produce a deterministic candidate, never an authoritative state update."""
    _valid_digest(expected_after_digest)
    _timestamp(assessed_at, "assessment time")
    if type(expected_latency_ms) is not int or expected_latency_ms < 0:
        raise ValueError("expected latency must be a non-negative integer")
    confounder_values = tuple(confounders)
    external_values = tuple(external_actors)
    _text_tuple(confounder_values, "confounder")
    _text_tuple(external_values, "external actor")
    actual = observation.observed_changes[-1].after_digest
    prediction_error = 0.0 if actual == expected_after_digest else 1.0
    delay_error = observation.latency_ms > expected_latency_ms
    causes = ["current_action"]
    causes.extend(f"confounder:{item}" for item in sorted(confounder_values))
    causes.extend(f"external:{item}" for item in sorted(external_values))
    probability = 0.9 if observation.action_channel_active else 0.1
    probability -= min(0.4, 0.1 * (len(confounder_values) + len(external_values)))
    if prediction_error:
        probability -= 0.25
    if delay_error:
        probability -= 0.1
    probability = round(max(0.01, min(0.99, probability)), 4)
    violations = []
    if prediction_error:
        violations.append("expected_after_state")
    if delay_error:
        violations.append("expected_timing")
    uncertainty = round(min(0.99, observation.uncertainty + (1.0 - probability) / 2), 4)
    return ReafferenceAssessment(
        f"assessment-{observation.observation_id}",
        observation.expected_effect_ref,
        observation.receipt_ref,
        observation.digest,
        assessed_at,
        prediction_error,
        tuple(causes),
        probability,
        tuple(violations),
        ("world:model-observed-transition",),
        ("self:calibrate-action-ownership",),
        ("policy:review-prediction" if prediction_error else "policy:retain",),
        uncertainty,
    )


@dataclass(frozen=True)
class ReafferenceChain:
    receipt_ref: str
    expected_effect_ref: str
    observation_ref: str
    assessment_ref: str
    update_commit_ref: str | None
    lineage_head_ref: str | None
    state: ReafferenceState


class ReafferenceStore:
    """Durable receipt/observation/assessment reconciliation authority."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise ReafferenceError("unsafe reafference database path")
        with self._connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS reafference_chains(
              receipt_ref TEXT PRIMARY KEY, expected_effect_ref TEXT NOT NULL,
              observation_ref TEXT UNIQUE NOT NULL, assessment_ref TEXT UNIQUE NOT NULL,
              observation_document TEXT NOT NULL, assessment_document TEXT NOT NULL,
              update_commit_ref TEXT, lineage_head_ref TEXT,
              state TEXT NOT NULL CHECK(state IN ('candidate','updates_committed','reconciled')));
            CREATE TABLE IF NOT EXISTS reafference_calibration(
              assessment_ref TEXT PRIMARY KEY, document TEXT NOT NULL);
            """)
        os.chmod(p, 0o600)
        self.verify()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    def ensure_candidate(
        self, observation: Observation, assessment: ReafferenceAssessment
    ) -> ReafferenceChain:
        if assessment.observation_ref != observation.digest or (
            assessment.receipt_ref != observation.receipt_ref
            or assessment.expected_effect_ref != observation.expected_effect_ref
        ):
            raise ReafferenceError("assessment chain references do not match")
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT OR IGNORE INTO reafference_chains VALUES(?,?,?,?,?,?,NULL,NULL,'candidate')",
                    (
                        observation.receipt_ref,
                        observation.expected_effect_ref,
                        observation.digest,
                        assessment.digest,
                        canonical_json(observation.to_json()).decode(),
                        canonical_json(assessment.to_json()).decode(),
                    ),
                )
                row = db.execute(
                    "SELECT * FROM reafference_chains WHERE receipt_ref=?",
                    (observation.receipt_ref,),
                ).fetchone()
                if row is None or (
                    row["expected_effect_ref"] != observation.expected_effect_ref
                    or row["observation_ref"] != observation.digest
                    or row["assessment_ref"] != assessment.digest
                    or row["observation_document"]
                    != canonical_json(observation.to_json()).decode()
                    or row["assessment_document"]
                    != canonical_json(assessment.to_json()).decode()
                ):
                    raise ReafferenceConflict(
                        "receipt is already bound to a different reafference chain"
                    )
                db.commit()
            except (sqlite3.IntegrityError, ReafferenceConflict) as exc:
                db.rollback()
                if isinstance(exc, ReafferenceConflict):
                    raise
                raise ReafferenceConflict("reafference artifact was replayed") from exc
        return self.load(observation.receipt_ref)

    def record_update(self, assessment_ref: str, update_commit_ref: str) -> None:
        """Durably reconcile the candidate and committed update before lineage."""
        for value in (assessment_ref, update_commit_ref):
            _valid_digest(value)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE reafference_chains SET update_commit_ref=?,state='updates_committed' WHERE assessment_ref=? AND state='candidate'",
                (update_commit_ref, assessment_ref),
            ).rowcount
            if changed != 1:
                db.rollback()
                raise ReafferenceConflict("candidate is absent or already updated")
            db.commit()

    def finalize(
        self,
        assessment_ref: str,
        lineage_head_ref: str,
        metric: CalibrationMetric,
    ) -> None:
        for value in (assessment_ref, lineage_head_ref):
            _valid_digest(value)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE reafference_chains SET lineage_head_ref=?,state='reconciled' WHERE assessment_ref=? AND state='updates_committed' AND update_commit_ref IS NOT NULL",
                (lineage_head_ref, assessment_ref),
            ).rowcount
            if changed != 1:
                db.rollback()
                raise ReafferenceConflict(
                    "updated chain is absent or already reconciled"
                )
            db.execute(
                "INSERT INTO reafference_calibration VALUES(?,?)",
                (assessment_ref, canonical_json(metric.to_json()).decode()),
            )
            db.commit()

    def load(self, receipt_ref: str) -> ReafferenceChain:
        _valid_digest(receipt_ref)
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM reafference_chains WHERE receipt_ref=?", (receipt_ref,)
            ).fetchone()
        if row is None:
            raise ReafferenceError("reafference chain is missing")
        try:
            state = ReafferenceState(row["state"])
            return ReafferenceChain(
                receipt_ref=str(row["receipt_ref"]),
                expected_effect_ref=str(row["expected_effect_ref"]),
                observation_ref=str(row["observation_ref"]),
                assessment_ref=str(row["assessment_ref"]),
                update_commit_ref=(
                    str(row["update_commit_ref"])
                    if row["update_commit_ref"] is not None
                    else None
                ),
                lineage_head_ref=(
                    str(row["lineage_head_ref"])
                    if row["lineage_head_ref"] is not None
                    else None
                ),
                state=state,
            )
        except (TypeError, ValueError) as exc:
            raise ReafferenceError("invalid persisted reafference state") from exc

    def chain(self, receipt_ref: str) -> Mapping[str, object]:
        """Compatibility projection; remove after callers consume typed chains."""
        chain = self.load(receipt_ref)
        return {
            "receipt_ref": chain.receipt_ref,
            "expected_effect_ref": chain.expected_effect_ref,
            "observation_ref": chain.observation_ref,
            "assessment_ref": chain.assessment_ref,
            "update_commit_ref": chain.update_commit_ref,
            "lineage_head_ref": chain.lineage_head_ref,
            "state": chain.state.value,
        }

    def calibration(self, assessment_ref: str) -> CalibrationMetric:
        _valid_digest(assessment_ref)
        with self._connect() as db:
            row = db.execute(
                "SELECT document FROM reafference_calibration WHERE assessment_ref=?",
                (assessment_ref,),
            ).fetchone()
        if row is None:
            raise ReafferenceError("calibration metric is missing")
        raw = _strict_document(row["document"])
        if raw.pop("schema_version", None) != "causal-calibration.v1" or set(raw) != {
            "assessment_ref",
            "brier_score",
            "causal_label",
            "predicted_probability",
        }:
            raise ReafferenceError("invalid calibration metric schema")
        try:
            return CalibrationMetric(
                assessment_ref=str(raw["assessment_ref"]),
                predicted_probability=raw["predicted_probability"],  # type: ignore[arg-type]
                causal_label=raw["causal_label"],  # type: ignore[arg-type]
                brier_score=raw["brier_score"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ReafferenceError("invalid calibration metric") from exc

    def verify(self) -> None:
        with self._connect() as db:
            integrity = db.execute("PRAGMA quick_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise ReafferenceError("reafference database integrity check failed")
            rows = db.execute("SELECT * FROM reafference_chains").fetchall()
            metrics = db.execute("SELECT * FROM reafference_calibration").fetchall()
        for row in rows:
            try:
                observation = _observation_from_document(row["observation_document"])
                assessment = _assessment_from_document(row["assessment_document"])
                if (
                    row["observation_ref"] != observation.digest
                    or row["assessment_ref"] != assessment.digest
                ):
                    raise ValueError
                if (
                    observation.receipt_ref != row["receipt_ref"]
                    or assessment.receipt_ref != row["receipt_ref"]
                    or observation.expected_effect_ref != row["expected_effect_ref"]
                    or assessment.expected_effect_ref != row["expected_effect_ref"]
                    or assessment.observation_ref != row["observation_ref"]
                ):
                    raise ValueError
                state = ReafferenceState(row["state"])
                if (
                    state
                    in {ReafferenceState.UPDATES_COMMITTED, ReafferenceState.RECONCILED}
                    and not row["update_commit_ref"]
                ):
                    raise ValueError
                if state is ReafferenceState.CANDIDATE and (
                    row["update_commit_ref"] or row["lineage_head_ref"]
                ):
                    raise ValueError
                if (
                    state is ReafferenceState.UPDATES_COMMITTED
                    and row["lineage_head_ref"]
                ):
                    raise ValueError
                if state is ReafferenceState.RECONCILED and not row["lineage_head_ref"]:
                    raise ValueError
            except (KeyError, TypeError, ValueError, ReafferenceError) as exc:
                raise ReafferenceError(
                    "persisted reafference chain failed verification"
                ) from exc
        metric_by_assessment: dict[str, CalibrationMetric] = {}
        for row in metrics:
            try:
                raw = _strict_document(row["document"])
                if raw.pop("schema_version", None) != "causal-calibration.v1":
                    raise ValueError
                if set(raw) != {
                    "assessment_ref",
                    "brier_score",
                    "causal_label",
                    "predicted_probability",
                }:
                    raise ValueError
                metric = CalibrationMetric(
                    assessment_ref=str(raw["assessment_ref"]),
                    predicted_probability=raw["predicted_probability"],  # type: ignore[arg-type]
                    causal_label=raw["causal_label"],  # type: ignore[arg-type]
                    brier_score=raw["brier_score"],  # type: ignore[arg-type]
                )
                if row["assessment_ref"] != metric.assessment_ref:
                    raise ValueError
                metric_by_assessment[metric.assessment_ref] = metric
            except (KeyError, TypeError, ValueError, ReafferenceError) as exc:
                raise ReafferenceError(
                    "persisted calibration metric failed verification"
                ) from exc
        reconciled = {
            str(row["assessment_ref"])
            for row in rows
            if row["state"] == ReafferenceState.RECONCILED.value
        }
        if set(metric_by_assessment) != reconciled:
            raise ReafferenceError("calibration rows do not match reconciled chains")
        assessment_probability = {
            str(row["assessment_ref"]): _assessment_from_document(
                row["assessment_document"]
            ).self_caused_probability
            for row in rows
            if row["state"] == ReafferenceState.RECONCILED.value
        }
        if any(
            metric.predicted_probability != assessment_probability[reference]
            for reference, metric in metric_by_assessment.items()
        ):
            raise ReafferenceError("calibration probability does not match assessment")


class ReafferenceTransactionService:
    """Microkernel-only promotion boundary for reafferent updates."""

    def __init__(
        self,
        store: ReafferenceStore,
        principal: Principal,
        *,
        validate_observation: ObservationValidator,
        validate_assessment: AssessmentValidator,
        validate_receipt: Callable[[str, str], None],
        commit_updates: UpdateCommitter,
        advance_lineage: LineageAdvancer,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        if not principal.is_kernel:
            raise ReafferenceError("only SYSTEM_KERNEL may reconcile reafference")
        self.store = store
        self.validate_observation = validate_observation
        self.validate_assessment = validate_assessment
        self.validate_receipt = validate_receipt
        self.commit_updates = commit_updates
        self.advance_lineage = advance_lineage
        self.failpoint = failpoint

    def reconcile(
        self,
        observation: Observation,
        assessment: ReafferenceAssessment,
        *,
        causal_label: bool,
    ) -> tuple[str, str]:
        if type(causal_label) is not bool:
            raise ReafferenceError("causal calibration label must be Boolean")
        self.validate_observation(observation)
        self.validate_assessment(observation, assessment)
        self.validate_receipt(observation.receipt_ref, observation.expected_effect_ref)
        chain = self.store.ensure_candidate(observation, assessment)
        if self.failpoint:
            self.failpoint("after_candidate_commit")
        if chain.state is ReafferenceState.RECONCILED:
            if chain.update_commit_ref is None or chain.lineage_head_ref is None:
                raise ReafferenceError("reconciled chain is incomplete")
            if (
                self.store.calibration(assessment.digest).causal_label
                is not causal_label
            ):
                raise ReafferenceConflict("causal label conflicts with durable replay")
            return chain.update_commit_ref, chain.lineage_head_ref
        if chain.state is ReafferenceState.CANDIDATE:
            update_ref = self.commit_updates(assessment)
            _valid_digest(update_ref)
            try:
                self.store.record_update(assessment.digest, update_ref)
            except ReafferenceConflict:
                competing = self.store.load(observation.receipt_ref)
                if competing.update_commit_ref != update_ref:
                    raise
            if self.failpoint:
                self.failpoint("after_update_commit")
        else:
            persisted_update_ref = chain.update_commit_ref
            if persisted_update_ref is None:
                raise ReafferenceError("updated chain has no update commit")
            update_ref = persisted_update_ref
        # Lineage cannot move until receipt, observation, assessment and update commit exist.
        lineage_ref = self.advance_lineage(
            observation.receipt_ref, assessment.digest, update_ref
        )
        _valid_digest(lineage_ref)
        if self.failpoint:
            self.failpoint("after_lineage_advance")
        score = (assessment.self_caused_probability - float(causal_label)) ** 2
        metric = CalibrationMetric(
            assessment.digest,
            assessment.self_caused_probability,
            causal_label,
            round(score, 6),
        )
        try:
            self.store.finalize(assessment.digest, lineage_ref, metric)
        except ReafferenceConflict:
            competing = self.store.load(observation.receipt_ref)
            if (
                competing.state is not ReafferenceState.RECONCILED
                or competing.lineage_head_ref != lineage_ref
                or competing.update_commit_ref != update_ref
            ):
                raise
        return update_ref, lineage_ref
