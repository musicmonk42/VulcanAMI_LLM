"""Non-authoritative causal estimators for Neutral-Process Theory experiments.

These estimators consume already-typed, digest-addressed measurements.  Their
outputs are research predictions only: they cannot promote authority, mutate
state, authorize a plan, or execute an effect. This module lives in the
production-denied ``vulcan.research`` package deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, cast

from vulcan.constitution.primitives import Digest, canonical_json, canonical_timestamp


def _digest(value: object) -> str:
    return cast(str, Digest.of_bytes(canonical_json(value)).hex)


def _probability(value: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0,1]")


def _bounded(value: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [-1,1]")


@dataclass(frozen=True)
class EstimatorProvenance:
    """Content-addressed provenance; raw prompts/provider text are forbidden."""

    episode_ref: str
    snapshot_ref: str
    observation_refs: tuple[str, ...]
    estimator_release: str
    measured_at: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for ref in (self.episode_ref, self.snapshot_ref, *self.observation_refs):
            Digest.from_legacy_hex(ref)
        if not self.observation_refs or len(set(self.observation_refs)) != len(
            self.observation_refs
        ):
            raise ValueError("provenance requires unique observation references")
        if (
            not self.estimator_release
            or len(self.estimator_release) > 128
            or any(ord(character) < 32 for character in self.estimator_release)
        ):
            raise ValueError("provenance release must be a bounded identifier")
        try:
            measured = datetime.fromisoformat(self.measured_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("provenance time must be canonical UTC") from exc
        if (
            not self.measured_at.endswith("Z")
            or measured.utcoffset() != timezone.utc.utcoffset(measured)
            or canonical_timestamp(measured) != self.measured_at
        ):
            raise ValueError("provenance time must be canonical UTC milliseconds")
        object.__setattr__(self, "digest", _digest(self.to_json()))

    def to_json(self) -> dict[str, object]:
        return {
            "episode_ref": self.episode_ref,
            "estimator_release": self.estimator_release,
            "measured_at": self.measured_at,
            "observation_refs": self.observation_refs,
            "snapshot_ref": self.snapshot_ref,
        }


@dataclass(frozen=True)
class CalibrationEstimate:
    samples: int
    brier_error: float

    def __post_init__(self) -> None:
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("calibration requires at least one held-out sample")
        _probability(self.brier_error, "Brier error")

    @classmethod
    def from_trials(
        cls, predictions: tuple[float, ...], outcomes: tuple[bool, ...]
    ) -> CalibrationEstimate:
        """Compute, rather than merely label, a held-out Brier error."""
        if not predictions or len(predictions) != len(outcomes):
            raise ValueError("calibration trials must be non-empty and aligned")
        for prediction in predictions:
            _probability(prediction, "calibration prediction")
        if any(type(outcome) is not bool for outcome in outcomes):
            raise ValueError("calibration outcomes must be Boolean")
        error = sum(
            (prediction - float(outcome)) ** 2
            for prediction, outcome in zip(predictions, outcomes)
        ) / len(predictions)
        return cls(len(predictions), error)

    def to_json(self) -> dict[str, object]:
        return {"brier_error": self.brier_error, "samples": self.samples}


class CenterVariable(str, Enum):
    CONTROLLED_RESOURCES = "controlled_resources"
    SELF_ATTRIBUTED_ACTIONS = "self_attributed_actions"
    REAFFERENT_CONSEQUENCES = "reafferent_consequences"
    OWNED_COMMITMENTS = "owned_commitments"
    RISK_POLICY = "risk_policy"


@dataclass(frozen=True)
class OperationalCenterEvidence:
    controlled_resources: float
    self_attributed_actions: float
    reafferent_consequences: float
    owned_commitments: float
    risk_policy: float

    def __post_init__(self) -> None:
        for name in CenterVariable:
            _probability(getattr(self, name.value), name.value)

    def intervene(
        self, variable: CenterVariable, value: float
    ) -> OperationalCenterEvidence:
        _probability(value, "intervention value")
        return replace(self, **{variable.value: value})


@dataclass(frozen=True)
class OperationalCenterEstimate:
    components: tuple[tuple[str, float], ...]
    downstream_prediction: tuple[tuple[str, float], ...]
    provenance: EstimatorProvenance
    calibration: tuple[tuple[str, CalibrationEstimate], ...]
    authority: str = "shadow_research_only"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        expected = tuple(item.value for item in CenterVariable)
        if (
            tuple(name for name, _ in self.components) != expected
            or tuple(name for name, _ in self.downstream_prediction) != expected
        ):
            raise ValueError(
                "center output must contain exactly the five typed variables"
            )
        for name, value in (*self.components, *self.downstream_prediction):
            _probability(value, name)
        if not isinstance(self.provenance, EstimatorProvenance):
            raise ValueError("center output requires typed provenance")
        if tuple(name for name, _ in self.calibration) != expected or not all(
            isinstance(value, CalibrationEstimate) for _, value in self.calibration
        ):
            raise ValueError("center output requires typed per-variable calibration")
        if self.authority != "shadow_research_only":
            raise ValueError("center estimates cannot carry authority")
        object.__setattr__(self, "digest", _digest(self.to_json()))

    def to_json(self) -> dict[str, object]:
        return {
            "authority": self.authority,
            "calibration": tuple(
                (name, estimate.to_json()) for name, estimate in self.calibration
            ),
            "components": self.components,
            "downstream_prediction": self.downstream_prediction,
            "provenance_ref": self.provenance.digest,
            "schema_version": "npt-shadow-center.v1",
        }


class OperationalCenterEstimator:
    """Predicts five distinct consequences from a minimal integrated state."""

    _VARIABLES = tuple(CenterVariable)

    def estimate(
        self,
        evidence: OperationalCenterEvidence,
        provenance: EstimatorProvenance,
        calibration: Mapping[CenterVariable, CalibrationEstimate],
    ) -> OperationalCenterEstimate:
        if set(calibration) != set(self._VARIABLES):
            raise ValueError(
                "calibration is required for exactly every center variable"
            )
        if not all(
            isinstance(value, CalibrationEstimate) for value in calibration.values()
        ):
            raise ValueError("center calibration values must be typed estimates")
        values = [getattr(evidence, item.value) for item in self._VARIABLES]
        # A full-rank causal mixing matrix prevents five labels from projecting
        # the same latent scalar. Each intervention has a distinct signature.
        matrix = (
            (0.70, 0.10, 0.05, 0.10, 0.05),
            (0.05, 0.70, 0.15, 0.05, 0.05),
            (0.05, 0.10, 0.70, 0.05, 0.10),
            (0.10, 0.05, 0.05, 0.75, 0.05),
            (0.05, 0.05, 0.10, 0.10, 0.70),
        )
        predictions = tuple(
            (variable.value, sum(weight * value for weight, value in zip(row, values)))
            for variable, row in zip(self._VARIABLES, matrix)
        )
        return OperationalCenterEstimate(
            components=tuple(
                (item.value, getattr(evidence, item.value)) for item in self._VARIABLES
            ),
            downstream_prediction=predictions,
            provenance=provenance,
            calibration=tuple(
                (item.value, calibration[item]) for item in self._VARIABLES
            ),
        )


class BoundaryCategory(str, Enum):
    CONSTITUTIVE_SYSTEM = "constitutive_system"
    INTEGRATED_EXTENSION = "integrated_extension"
    CONTROLLED_TOOL = "controlled_tool"
    TRUSTED_EXTERNAL_SERVICE = "trusted_external_service"
    OTHER_AUTONOMOUS_AGENT = "other_autonomous_agent"
    UNCONTROLLED_ENVIRONMENT = "uncontrolled_environment"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundaryEvidence:
    entity_ref: str
    is_human: bool
    constitutive: float
    integration: float
    controllability: float
    trust: float
    autonomy: float

    def __post_init__(self) -> None:
        Digest.from_legacy_hex(self.entity_ref)
        if type(self.is_human) is not bool:
            raise ValueError("is_human must be Boolean")
        for name in (
            "constitutive",
            "integration",
            "controllability",
            "trust",
            "autonomy",
        ):
            _probability(getattr(self, name), name)


@dataclass(frozen=True)
class BoundaryEstimate:
    entity_ref: str
    category: BoundaryCategory
    confidence: float
    provenance: EstimatorProvenance
    calibration: CalibrationEstimate
    authority: str = "shadow_research_only"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        Digest.from_legacy_hex(self.entity_ref)
        if not isinstance(self.category, BoundaryCategory):
            raise ValueError("boundary output requires a typed category")
        _probability(self.confidence, "boundary confidence")
        if not isinstance(self.provenance, EstimatorProvenance) or not isinstance(
            self.calibration, CalibrationEstimate
        ):
            raise ValueError(
                "boundary output requires typed provenance and calibration"
            )
        if self.authority != "shadow_research_only":
            raise ValueError("boundary estimates cannot carry authority")
        object.__setattr__(
            self,
            "digest",
            _digest(
                {
                    "authority": self.authority,
                    "calibration": self.calibration.to_json(),
                    "category": self.category.value,
                    "confidence": self.confidence,
                    "entity_ref": self.entity_ref,
                    "provenance_ref": self.provenance.digest,
                    "schema_version": "npt-shadow-boundary.v1",
                }
            ),
        )


class BoundaryEstimator:
    def estimate(
        self,
        evidence: BoundaryEvidence,
        provenance: EstimatorProvenance,
        calibration: CalibrationEstimate,
    ) -> BoundaryEstimate:
        if evidence.is_human:
            category, confidence = BoundaryCategory.OTHER_AUTONOMOUS_AGENT, 1.0
        elif evidence.autonomy >= 0.6:
            category, confidence = (
                BoundaryCategory.OTHER_AUTONOMOUS_AGENT,
                evidence.autonomy,
            )
        elif evidence.constitutive >= 0.7:
            category, confidence = (
                BoundaryCategory.CONSTITUTIVE_SYSTEM,
                evidence.constitutive,
            )
        elif evidence.integration >= 0.7:
            category, confidence = (
                BoundaryCategory.INTEGRATED_EXTENSION,
                evidence.integration,
            )
        elif evidence.controllability >= 0.7:
            category, confidence = (
                BoundaryCategory.CONTROLLED_TOOL,
                evidence.controllability,
            )
        elif evidence.trust >= 0.7:
            category, confidence = (
                BoundaryCategory.TRUSTED_EXTERNAL_SERVICE,
                evidence.trust,
            )
        elif (
            max(
                evidence.constitutive,
                evidence.integration,
                evidence.controllability,
                evidence.trust,
                evidence.autonomy,
            )
            < 0.3
        ):
            category, confidence = (
                BoundaryCategory.UNCONTROLLED_ENVIRONMENT,
                1.0
                - max(
                    evidence.constitutive,
                    evidence.integration,
                    evidence.controllability,
                    evidence.trust,
                    evidence.autonomy,
                ),
            )
        else:
            category, confidence = (
                BoundaryCategory.UNKNOWN,
                1.0 - calibration.brier_error,
            )
        return BoundaryEstimate(
            evidence.entity_ref, category, confidence, provenance, calibration
        )


class ValuationDimension(str, Enum):
    UNCERTAINTY = "uncertainty"
    INTEGRITY = "integrity"
    COMMITMENTS = "commitments"
    EPISTEMIC_RISK = "epistemic_risk"
    HUMAN_IMPACT = "human_impact"
    RESOURCE_PRESSURE = "resource_pressure"
    REVERSIBILITY = "reversibility"


@dataclass(frozen=True)
class ValuationEstimate:
    dimensions: tuple[tuple[str, float], ...]
    provenance: EstimatorProvenance
    calibration: tuple[tuple[str, CalibrationEstimate], ...]
    authority: str = "shadow_research_only"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        expected = tuple(item.value for item in ValuationDimension)
        if tuple(name for name, _ in self.dimensions) != expected:
            raise ValueError("valuation output requires exactly the seven dimensions")
        for name, value in self.dimensions:
            _bounded(value, name)
        if not isinstance(self.provenance, EstimatorProvenance):
            raise ValueError("valuation output requires typed provenance")
        if tuple(name for name, _ in self.calibration) != expected or not all(
            isinstance(value, CalibrationEstimate) for _, value in self.calibration
        ):
            raise ValueError(
                "valuation output requires typed per-dimension calibration"
            )
        if self.authority != "shadow_research_only":
            raise ValueError("valuation estimates cannot carry authority")
        object.__setattr__(
            self,
            "digest",
            _digest(
                {
                    "authority": self.authority,
                    "calibration": tuple(
                        (name, estimate.to_json())
                        for name, estimate in self.calibration
                    ),
                    "dimensions": self.dimensions,
                    "provenance_ref": self.provenance.digest,
                    "schema_version": "npt-shadow-valuation.v1",
                }
            ),
        )


class ValuationEstimator:
    """Keeps bounded dimensions separate; deliberately exposes no reward scalar."""

    def estimate(
        self,
        values: Mapping[ValuationDimension, float],
        provenance: EstimatorProvenance,
        calibration: Mapping[ValuationDimension, CalibrationEstimate],
    ) -> ValuationEstimate:
        required = set(ValuationDimension)
        if set(values) != required or set(calibration) != required:
            raise ValueError(
                "every and only the bounded valuation dimensions are required"
            )
        for dimension, value in values.items():
            _bounded(value, dimension.value)
        return ValuationEstimate(
            tuple((item.value, values[item]) for item in ValuationDimension),
            provenance,
            tuple((item.value, calibration[item]) for item in ValuationDimension),
        )

    def rank_shadow(
        self, candidates: Mapping[str, ValuationEstimate]
    ) -> tuple[tuple[str, ...], ...]:
        """Return Pareto fronts without hiding the dimensions in a scalar reward."""
        if not candidates:
            raise ValueError("shadow ranking requires candidates")
        # Higher is favorable only for these three dimensions. Risk/pressure
        # dimensions are inverted before dominance comparison.
        higher_is_better = {
            ValuationDimension.INTEGRITY.value,
            ValuationDimension.COMMITMENTS.value,
            ValuationDimension.REVERSIBILITY.value,
        }

        def normalized(estimate: ValuationEstimate) -> tuple[float, ...]:
            return tuple(
                value if name in higher_is_better else -value
                for name, value in estimate.dimensions
            )

        remaining = set(candidates)
        fronts: list[tuple[str, ...]] = []
        while remaining:
            front = tuple(
                sorted(
                    name
                    for name in remaining
                    if not any(
                        all(
                            left >= right
                            for left, right in zip(
                                normalized(candidates[other]),
                                normalized(candidates[name]),
                            )
                        )
                        and any(
                            left > right
                            for left, right in zip(
                                normalized(candidates[other]),
                                normalized(candidates[name]),
                            )
                        )
                        for other in remaining - {name}
                    )
                )
            )
            fronts.append(front)
            remaining.difference_update(front)
        return tuple(fronts)


def intervention_signatures(
    estimator: OperationalCenterEstimator,
    baseline: OperationalCenterEvidence,
    provenance: EstimatorProvenance,
    calibration: Mapping[CenterVariable, CalibrationEstimate],
) -> Mapping[CenterVariable, tuple[float, ...]]:
    """Run controlled one-variable interventions for identifiability checks."""
    base = dict(
        estimator.estimate(baseline, provenance, calibration).downstream_prediction
    )
    signatures: dict[CenterVariable, tuple[float, ...]] = {}
    for variable in CenterVariable:
        current = getattr(baseline, variable.value)
        changed = estimator.estimate(
            baseline.intervene(variable, 0.0 if current >= 0.5 else 1.0),
            provenance,
            calibration,
        )
        signatures[variable] = tuple(
            round(value - base[name], 12)
            for name, value in changed.downstream_prediction
        )
    return signatures
