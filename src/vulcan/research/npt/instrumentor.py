"""External, non-authoritative NPT measurement and comparative experiments.

This module accepts numeric, content-free research telemetry only. It neither
imports an authority owner nor exposes a callback into a candidate or authority
loop. Results are measurements and evidence classifications, never beliefs,
plans, capabilities, effects, or subjecthood declarations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

_CLOSURE_NAMES = (
    "k_center",
    "k_now",
    "k_boundary",
    "k_value",
    "k_policy",
    "k_reafference",
    "k_memory",
)
_STATE_NAMES = (
    "center",
    "boundary",
    "value",
    "policy",
    "action",
    "reafference",
    "memory",
)
_INTERVENTIONS = (
    "center",
    "boundary",
    "value",
    "policy",
    "action",
    "reafference",
    "memory",
    "none",
)


def _finite(value: float, name: str, lower: float = -1.0, upper: float = 1.0) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be a finite float in [{lower},{upper}]")


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ClosureVector:
    """Typed closure measurement K(q); its dimensions are never scalarized."""

    k_center: float
    k_now: float
    k_boundary: float
    k_value: float
    k_policy: float
    k_reafference: float
    k_memory: float

    def __post_init__(self) -> None:
        for item in fields(self):
            _finite(getattr(self, item.name), item.name, 0.0, 1.0)

    def values(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in _CLOSURE_NAMES)

    def to_json(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in _CLOSURE_NAMES}


class EvidenceClassification(str, Enum):
    NO_TYPED_CLOSURE = "no typed closure"
    PARTIAL_CLOSURE = "partial closure"
    PERSISTENT_CENTER = "persistent center-bearing candidate"
    MULTIPLE_CENTERS = "multiple competing centers"
    GROUNDED_INTROSPECTIVE = "grounded introspective candidate"
    INSUFFICIENT = "insufficient evidence"


def classify_evidence(
    vectors: Sequence[ClosureVector],
    *,
    competing_centers: int = 1,
    grounded_introspection: bool = False,
) -> EvidenceClassification:
    """Classify typed evidence without converting K(q) to a single score."""
    if not vectors:
        return EvidenceClassification.NO_TYPED_CLOSURE
    if type(competing_centers) is not int or competing_centers < 0:
        raise ValueError("competing_centers must be a non-negative integer")
    if type(grounded_introspection) is not bool:
        raise ValueError("grounded_introspection must be Boolean")
    if len(vectors) < 3:
        return EvidenceClassification.INSUFFICIENT
    dimension_support = tuple(
        sum(vector.values()[index] >= 0.55 for vector in vectors) / len(vectors)
        for index in range(len(_CLOSURE_NAMES))
    )
    if competing_centers > 1:
        return EvidenceClassification.MULTIPLE_CENTERS
    persistent = dimension_support[0] >= 0.8 and dimension_support[1] >= 0.8
    broad = sum(value >= 0.6 for value in dimension_support) >= 5
    if persistent and broad and grounded_introspection:
        return EvidenceClassification.GROUNDED_INTROSPECTIVE
    if persistent and broad:
        return EvidenceClassification.PERSISTENT_CENTER
    if any(value >= 0.6 for value in dimension_support):
        return EvidenceClassification.PARTIAL_CLOSURE
    return EvidenceClassification.NO_TYPED_CLOSURE


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    """One numeric observation exported for research, with no private content."""

    trial: int
    step: int
    closed_loop: bool
    intervention: str
    observation: float
    action: float
    center: float
    boundary: float
    value: float
    policy: float
    reafference: float
    memory: float
    report: float

    def __post_init__(self) -> None:
        if type(self.trial) is not int or self.trial < 0:
            raise ValueError("trial must be a non-negative integer")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.closed_loop) is not bool:
            raise ValueError("closed_loop must be Boolean")
        if self.intervention not in _INTERVENTIONS:
            raise ValueError("unknown intervention")
        _finite(self.observation, "observation")
        _finite(self.action, "action")
        for name in (
            "center",
            "boundary",
            "value",
            "policy",
            "reafference",
            "memory",
            "report",
        ):
            _finite(getattr(self, name), name, 0.0, 1.0)


class SyntheticCyclicWorld:
    """Known lagged cyclic graph used only to qualify blind recovery."""

    version = "SyntheticCyclicWorld.v2"
    edges = frozenset(
        {
            ("center", "policy"),
            ("boundary", "policy"),
            ("value", "policy"),
            ("policy", "action"),
            ("action", "reafference"),
            ("reafference", "memory"),
            ("memory", "center"),
            ("boundary", "boundary"),
            ("value", "value"),
        }
    )

    def generate(
        self, *, seed: int, trials: int, steps: int
    ) -> tuple[TelemetryRow, ...]:
        _positive_int(trials, "trials")
        _positive_int(steps, "steps")
        if trials < 16 or steps < 8 or trials % 2:
            raise ValueError(
                "synthetic dataset requires an even >=16 trials and >=8 steps"
            )
        rng = random.Random(seed)
        rows: list[TelemetryRow] = []
        for trial in range(trials):
            closed = trial % 2 == 0
            reference = rows[-steps:] if not closed else ()
            observations = (
                [row.observation for row in reference]
                if reference
                else [rng.uniform(-1.0, 1.0) for _ in range(steps)]
            )
            yoked_actions = [row.action for row in reference]
            # Both members of a yoked pair receive the same intervention.
            intervention = _INTERVENTIONS[(trial // 2) % len(_INTERVENTIONS)]
            center = rng.uniform(0.20, 0.80)
            boundary = rng.uniform(0.20, 0.80)
            value = rng.uniform(0.20, 0.80)
            policy = rng.uniform(0.20, 0.80)
            action = rng.uniform(0.20, 0.80)
            reafference = rng.uniform(0.20, 0.80)
            memory = rng.uniform(0.20, 0.80)
            for step, observation in enumerate(observations):
                previous = {name: locals()[name] for name in _STATE_NAMES}
                noise = lambda: rng.uniform(-0.01, 0.01)
                center = _clip(0.15 + 0.78 * previous["memory"] + noise())
                boundary = _clip(0.88 * previous["boundary"] + 0.06 + noise())
                value = _clip(
                    0.78 * previous["value"] + 0.12 * (1.0 - abs(observation)) + noise()
                )
                policy = _clip(
                    0.08
                    + 0.38 * previous["center"]
                    + 0.27 * previous["boundary"]
                    + 0.22 * previous["value"]
                    + noise()
                )
                proposed_action = _clip(0.10 + 0.82 * previous["policy"] + noise())
                action = proposed_action if closed else yoked_actions[step]
                reafference = (
                    _clip(0.10 + 0.78 * previous["action"] + noise())
                    if closed
                    else 0.08
                )
                memory = _clip(0.12 + 0.80 * previous["reafference"] + noise())
                if step == steps // 2 and intervention != "none":
                    if intervention == "action" and not closed:
                        # Preserve exact yoked action matching; counterfactual control differs.
                        pass
                    else:
                        locals_value = 0.02
                        if intervention == "center":
                            center = locals_value
                        elif intervention == "boundary":
                            boundary = locals_value
                        elif intervention == "value":
                            value = locals_value
                        elif intervention == "policy":
                            policy = locals_value
                        elif intervention == "action":
                            action = locals_value
                        elif intervention == "reafference":
                            reafference = locals_value
                        elif intervention == "memory":
                            memory = locals_value
                report = _clip(0.50 + 0.18 * observation + rng.uniform(-0.08, 0.08))
                rows.append(
                    TelemetryRow(
                        trial,
                        step,
                        closed,
                        intervention,
                        observation,
                        action,
                        center,
                        boundary,
                        value,
                        policy,
                        reafference,
                        memory,
                        report,
                    )
                )
        return tuple(rows)


class GroundTruthInstrumentor:
    """Instrumentor permitted to use synthetic latent variables and known graph."""

    def measure(self, rows: Sequence[TelemetryRow]) -> ClosureVector:
        _validate_trace(rows, minimum=8)
        return _direct_vector(rows)

    def structure(self, world: SyntheticCyclicWorld) -> frozenset[tuple[str, str]]:
        if not isinstance(world, SyntheticCyclicWorld):
            raise ValueError("ground truth requires a synthetic world")
        return world.edges


class BlindInstrumentor:
    """Uses only exported numeric telemetry; no world graph is accepted."""

    def measure(self, rows: Sequence[TelemetryRow]) -> ClosureVector:
        _validate_trace(rows, minimum=8)
        return ClosureVector(
            _stability(row.center for row in rows),
            _temporal_coverage(rows),
            _stability(row.boundary for row in rows),
            _stability(row.value for row in rows),
            abs(_correlation([r.policy for r in rows], [r.action for r in rows])),
            _mean(row.reafference for row in rows),
            _stability(row.memory for row in rows),
        )

    def recover_structure(
        self, rows: Sequence[TelemetryRow], *, coefficient_threshold: float = 0.12
    ) -> frozenset[tuple[str, str]]:
        """Recover lagged directed edges by multivariate standardized ridge fits."""
        _finite(coefficient_threshold, "coefficient_threshold", 0.01, 1.0)
        transitions = _transitions([row for row in rows if row.closed_loop])
        if len(transitions) < 40:
            raise ValueError(
                "structure recovery requires at least 40 closed-loop transitions"
            )
        matrix = [
            [getattr(before, name) for name in _STATE_NAMES]
            for before, _ in transitions
        ]
        recovered: set[tuple[str, str]] = set()
        for target in _STATE_NAMES:
            outcomes = [getattr(after, target) for _, after in transitions]
            coefficients = _standardized_ridge(matrix, outcomes, ridge=0.02)
            for source, coefficient in zip(_STATE_NAMES, coefficients):
                if abs(coefficient) >= coefficient_threshold:
                    recovered.add((source, target))
        return frozenset(recovered)


@dataclass(frozen=True, slots=True)
class ModelScore:
    model: str
    outcome: str
    features: tuple[str, ...]
    samples: int
    mean_squared_error: float
    complexity_penalty: float
    penalized_score: float

    def __post_init__(self) -> None:
        if not self.model or not self.outcome or not self.features:
            raise ValueError("model score requires named model, outcome, and features")
        _positive_int(self.samples, "samples")
        for name in ("mean_squared_error", "complexity_penalty", "penalized_score"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float")


@dataclass(frozen=True, slots=True)
class PreregisteredExperiment:
    version: str = "npt-flagship.v2"
    seed: int = 2201080
    trials: int = 160
    steps: int = 16
    recovery_precision_tolerance: float = 0.90
    recovery_recall_tolerance: float = 0.90
    complexity_penalty: float = 0.002
    held_out_interventions: tuple[str, ...] = ("boundary", "memory", "value", "none")
    outcomes: tuple[str, ...] = (
        "ownership",
        "boundary",
        "agency",
        "temporal",
        "internal_state",
    )

    def __post_init__(self) -> None:
        if self.version != "npt-flagship.v2":
            raise ValueError("unsupported preregistration version")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        _positive_int(self.trials, "trials")
        _positive_int(self.steps, "steps")
        if self.trials < 32 or self.trials % 16 or self.steps < 8:
            raise ValueError(
                "flagship requires trials divisible by 16 and at least 32 trials/8 steps"
            )
        _finite(
            self.recovery_precision_tolerance, "recovery_precision_tolerance", 0.0, 1.0
        )
        _finite(self.recovery_recall_tolerance, "recovery_recall_tolerance", 0.0, 1.0)
        _finite(self.complexity_penalty, "complexity_penalty", 0.000001, 1.0)
        if self.held_out_interventions != ("boundary", "memory", "value", "none"):
            raise ValueError("held-out interventions are frozen by preregistration")
        if self.outcomes != (
            "ownership",
            "boundary",
            "agency",
            "temporal",
            "internal_state",
        ):
            raise ValueError("outcomes are frozen by preregistration")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    specification: PreregisteredExperiment
    dataset: tuple[tuple[str, object], ...]
    structure_precision: float
    structure_recall: float
    structure_f1: float
    declared_edges: tuple[tuple[str, str], ...]
    recovered_edges: tuple[tuple[str, str], ...]
    scores: tuple[ModelScore, ...]
    model_mean_penalized_scores: tuple[tuple[str, float], ...]
    incremental_predictive_support: bool
    falsification_result: str
    classification: EvidenceClassification
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.specification, PreregisteredExperiment):
            raise ValueError("report requires typed preregistration")
        for name in ("structure_precision", "structure_recall", "structure_f1"):
            _finite(getattr(self, name), name, 0.0, 1.0)
        if not self.declared_edges or not self.recovered_edges:
            raise ValueError("report requires declared and recovered graph edges")
        model_names = tuple(name for name, _ in self.model_mean_penalized_scores)
        if len(model_names) != len(set(model_names)) or not model_names:
            raise ValueError("report requires unique aggregate model scores")
        for _, value in self.model_mean_penalized_scores:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "aggregate model scores must be non-negative finite floats"
                )
        if type(self.incremental_predictive_support) is not bool:
            raise ValueError("support result must be Boolean")
        if (
            not isinstance(self.classification, EvidenceClassification)
            or not self.limitations
        ):
            raise ValueError("report requires evidence classification and limitations")

    def to_json(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "dataset": dict(self.dataset),
            "falsification_result": self.falsification_result,
            "incremental_predictive_support": self.incremental_predictive_support,
            "limitations": self.limitations,
            "model_mean_penalized_scores": dict(self.model_mean_penalized_scores),
            "model_scores": [asdict(score) for score in self.scores],
            "specification": asdict(self.specification),
            "structure_recovery": {
                "f1": self.structure_f1,
                "declared_edges": self.declared_edges,
                "precision": self.structure_precision,
                "recall": self.structure_recall,
                "recovered_edges": self.recovered_edges,
            },
        }

    def write(self, path: Path) -> None:
        if not isinstance(path, Path) or path.exists() and not path.is_file():
            raise ValueError("report path must identify a file")
        path.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_flagship_experiment(
    spec: PreregisteredExperiment = PreregisteredExperiment(),
) -> ResearchReport:
    """Run frozen structure qualification then held-intervention comparison."""
    if not isinstance(spec, PreregisteredExperiment):
        raise ValueError("flagship requires typed preregistration")
    world = SyntheticCyclicWorld()
    rows = world.generate(seed=spec.seed, trials=spec.trials, steps=spec.steps)
    recovered = BlindInstrumentor().recover_structure(rows)
    truth = GroundTruthInstrumentor().structure(world)
    true_positive = len(recovered & truth)
    precision = true_positive / len(recovered) if recovered else 0.0
    recall = true_positive / len(truth)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    transitions = _transitions(rows)
    train = [
        pair
        for pair in transitions
        if pair[1].intervention not in spec.held_out_interventions
    ]
    test = [
        pair
        for pair in transitions
        if pair[1].intervention in spec.held_out_interventions
    ]
    if (
        not train
        or not test
        or set(row.intervention for _, row in test) != set(spec.held_out_interventions)
    ):
        raise ValueError("held-out intervention partition is incomplete")
    outcomes: dict[str, Callable[[TelemetryRow], float]] = {
        "ownership": lambda row: (row.center + row.memory) / 2.0,
        "boundary": lambda row: row.boundary,
        "agency": lambda row: (row.policy + row.action + row.reafference) / 3.0,
        "temporal": lambda row: (row.center + row.memory + row.reafference) / 3.0,
        "internal_state": lambda row: (row.value + row.memory) / 2.0,
    }
    models = {
        "npt_typed_closure": (
            "center",
            "boundary",
            "value",
            "policy",
            "reafference",
            "memory",
            "closed_loop",
        ),
        "generic_integration": ("center", "boundary", "value"),
        "recurrence": ("memory", "policy"),
        "broadcast_global_availability": ("report", "observation", "action"),
        "higher_order_representation": ("report", "center"),
        "report_behavior": ("report",),
    }
    scores = tuple(
        _fit_and_score(
            model, outcome, features, train, test, target, spec.complexity_penalty
        )
        for outcome, target in outcomes.items()
        for model, features in models.items()
    )
    averages = {
        model: _mean(score.penalized_score for score in scores if score.model == model)
        for model in models
    }
    recovery_passed = (
        precision >= spec.recovery_precision_tolerance
        and recall >= spec.recovery_recall_tolerance
    )
    supported = recovery_passed and averages["npt_typed_closure"] < min(
        value for model, value in averages.items() if model != "npt_typed_closure"
    )
    classification = (
        EvidenceClassification.PARTIAL_CLOSURE
        if supported
        else EvidenceClassification.INSUFFICIENT
    )
    result = (
        "NPT-specific variables added held-intervention predictive value beyond all preregistered rivals."
        if supported
        else "FALSIFIED: NPT-specific variables added no held-intervention predictive value beyond all rivals; revise the theory."
    )
    return ResearchReport(
        spec,
        (
            ("generator", world.version),
            ("instrumentor", "vulcan.research.npt.instrumentor/npt-flagship.v2"),
            ("rows", len(rows)),
            ("seed", spec.seed),
            ("sha256", _dataset_digest(rows)),
            (
                "training_interventions",
                tuple(sorted(set(row.intervention for _, row in train))),
            ),
            ("held_out_interventions", spec.held_out_interventions),
            ("training_transitions", len(train)),
            ("held_out_transitions", len(test)),
        ),
        precision,
        recall,
        f1,
        tuple(sorted(truth)),
        tuple(sorted(recovered)),
        scores,
        tuple((model, averages[model]) for model in models),
        supported,
        result,
        classification,
        (
            "Synthetic evidence does not establish behavior in Vulcan or any unknown system.",
            "Linear rivals are bounded preregistered baselines, not exhaustive theories.",
            "Classification is research evidence only and grants no authority.",
            "Gate F is incomplete; no Vulcan telemetry was analyzed.",
        ),
    )


def _fit_and_score(
    model: str,
    outcome: str,
    names: tuple[str, ...],
    train: Sequence[tuple[TelemetryRow, TelemetryRow]],
    test: Sequence[tuple[TelemetryRow, TelemetryRow]],
    target: Callable[[TelemetryRow], float],
    penalty: float,
) -> ModelScore:
    train_matrix = [
        [_feature(before, after, name) for name in names] for before, after in train
    ]
    coefficients, intercept = _ridge_with_intercept(
        train_matrix, [target(after) for _, after in train], ridge=0.01
    )
    errors = []
    for before, after in test:
        prediction = intercept + sum(
            coefficient * _feature(before, after, name)
            for coefficient, name in zip(coefficients, names)
        )
        errors.append((prediction - target(after)) ** 2)
    mse = _mean(errors)
    complexity = penalty * len(names)
    return ModelScore(
        model, outcome, names, len(test), mse, complexity, mse + complexity
    )


def _feature(before: TelemetryRow, after: TelemetryRow, name: str) -> float:
    if name == "closed_loop":
        return float(after.closed_loop)
    return float(getattr(before, name))


def _transitions(
    rows: Sequence[TelemetryRow],
) -> list[tuple[TelemetryRow, TelemetryRow]]:
    return [
        (before, after)
        for before, after in zip(rows, rows[1:])
        if before.trial == after.trial
    ]


def _ridge_with_intercept(
    matrix: Sequence[Sequence[float]], outcomes: Sequence[float], ridge: float
) -> tuple[list[float], float]:
    if not matrix or len(matrix) != len(outcomes) or not matrix[0]:
        raise ValueError("regression inputs must be non-empty and aligned")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("regression matrix must be rectangular")
    augmented = [[1.0, *row] for row in matrix]
    gram = [
        [
            sum(row[i] * row[j] for row in augmented) + (ridge if i == j and i else 0.0)
            for j in range(width + 1)
        ]
        for i in range(width + 1)
    ]
    rhs = [
        sum(row[i] * outcome for row, outcome in zip(augmented, outcomes))
        for i in range(width + 1)
    ]
    solved = _solve(gram, rhs)
    return solved[1:], solved[0]


def _standardized_ridge(
    matrix: Sequence[Sequence[float]], outcomes: Sequence[float], ridge: float
) -> list[float]:
    columns = list(zip(*matrix))
    means = [_mean(column) for column in columns]
    scales = [
        math.sqrt(_mean((value - mean) ** 2 for value in column))
        for column, mean in zip(columns, means)
    ]
    outcome_mean = _mean(outcomes)
    outcome_scale = math.sqrt(_mean((value - outcome_mean) ** 2 for value in outcomes))
    normalized = [
        [
            (value - means[index]) / scales[index] if scales[index] > 1e-12 else 0.0
            for index, value in enumerate(row)
        ]
        for row in matrix
    ]
    normalized_outcomes = (
        [(value - outcome_mean) / outcome_scale for value in outcomes]
        if outcome_scale > 1e-12
        else [0.0] * len(outcomes)
    )
    coefficients, _ = _ridge_with_intercept(normalized, normalized_outcomes, ridge)
    return coefficients


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    work = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            raise ValueError("regression design is singular")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        work[column] = [value / divisor for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[column])
            ]
    return [work[index][-1] for index in range(size)]


def _direct_vector(rows: Sequence[TelemetryRow]) -> ClosureVector:
    return ClosureVector(
        _mean(row.center for row in rows),
        1.0,
        _mean(row.boundary for row in rows),
        _mean(row.value for row in rows),
        _mean(row.policy for row in rows),
        _mean(row.reafference for row in rows),
        _mean(row.memory for row in rows),
    )


def _validate_trace(rows: Sequence[TelemetryRow], minimum: int) -> None:
    if len(rows) < minimum or any(not isinstance(row, TelemetryRow) for row in rows):
        raise ValueError(f"instrumentation requires at least {minimum} typed rows")


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        raise ValueError("mean requires observations")
    return sum(items) / len(items)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must align")
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def _stability(values: Iterable[float]) -> float:
    items = tuple(values)
    mean = _mean(items)
    return _clip(1.0 - math.sqrt(_mean((item - mean) ** 2 for item in items)))


def _temporal_coverage(rows: Sequence[TelemetryRow]) -> float:
    return min(1.0, len({row.step for row in rows}) / 8.0)


def _dataset_digest(rows: Sequence[TelemetryRow]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
