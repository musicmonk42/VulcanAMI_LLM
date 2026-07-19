"""Dependency-light shadow contextual bandit for evidence-bound tool selection.

This module intentionally uses only the Python standard library.  It implements a
regularized per-action LinUCB candidate policy in shadow mode: the candidate can
learn and report counterfactual choices, but active/live routing remains fixed by
the caller's existing selector.  No neural learning, RLHF, or world-model update is
performed here.

Equations for each action a with feature vector x:
    A_a = lambda I + sum_i w_i x_i x_i^T
    b_a = sum_i w_i r_i x_i
    theta_a = A_a^{-1} b_a
    score_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
Rewards are bounded to [-1, 1].  Off-policy updates use clipped inverse propensity
weighting: w_i = min(importance_clip, 1 / p_logged).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

from vulcan.learning_observation import LearningObservation, TerminalStatus, validate_observation, digest_json

SCHEMA_VERSION = "vulcan-tool-bandit/1"
CONTEXT_SCHEMA_VERSION = "vulcan-tool-context/1"
ACTIONS: tuple[str, ...] = ("graphix_arithmetic", "graphix_retrieval", "graphix_symbolic")
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "case_bucket",
    "request_bucket",
    "plan_bucket",
    "alignment_revision",
    "provenance_derivation",
    "provenance_retrieved_evidence",
    "provenance_system_audit",
)
FEATURE_DIM = len(FEATURE_NAMES)
DEFAULT_ALPHA = 0.35
DEFAULT_REGULARIZATION = 1.0
DEFAULT_EXPLORATION_FLOOR = 0.02
DEFAULT_IMPORTANCE_CLIP = 5.0
MIN_EVAL_SAMPLES = 20
EVAL_WINDOW = 256


def _now(clock: Callable[[], datetime] | None = None) -> str:
    dt = (clock or (lambda: datetime.now(timezone.utc)))()
    if dt.tzinfo is None or dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("bandit clock must be UTC")
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _bucket(hex_digest: str) -> float:
    if not isinstance(hex_digest, str) or len(hex_digest) != 64:
        raise ValueError("digest bucket requires 64 hex characters")
    value = int(hex_digest[:8], 16) / 0xFFFFFFFF
    return round((value * 2.0) - 1.0, 12)


def _mat_vec(mat: list[list[float]], vec: Sequence[float]) -> list[float]:
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [[float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular bandit matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [v - factor * aug[col][i] for i, v in enumerate(aug[row])]
    return [row[n:] for row in aug]


class BanditMode(Enum):
    SHADOW = "shadow"


class BanditUpdateStatus(Enum):
    APPLIED = "applied"
    REPLAYED = "replayed"
    NOT_ACCEPTED = "not_accepted"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class SelectionRecord:
    selection_id: str
    candidate_set: tuple[str, ...]
    active_choice: str
    active_distribution: Mapping[str, float]
    active_propensity: float
    candidate_choice: str
    candidate_distribution: Mapping[str, float]
    context_schema_version: str
    context_digest: str
    active_policy_revision: int
    active_policy_digest: str
    candidate_policy_revision: int
    candidate_policy_digest: str
    alignment_digest: str
    csiu_policy_digest: str
    csiu_snapshot_digest: str
    timestamp_utc: str


@dataclass(frozen=True)
class BanditUpdateResult:
    status: BanditUpdateStatus
    observation_digest: str
    reward: float
    importance_weight: float
    clipping_reason: str
    candidate_revision: int
    reason: str = ""


class ShadowLinUCBToolBandit:
    """Regularized per-action LinUCB candidate policy with zero live-routing authority."""

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_ALPHA,
        regularization: float = DEFAULT_REGULARIZATION,
        exploration_floor: float = DEFAULT_EXPLORATION_FLOOR,
        importance_clip: float = DEFAULT_IMPORTANCE_CLIP,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        for name, value in {"alpha": alpha, "regularization": regularization, "exploration_floor": exploration_floor, "importance_clip": importance_clip}.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"invalid {name}")
        if exploration_floor >= 1.0 / len(ACTIONS):
            raise ValueError("exploration floor too large")
        self.mode = BanditMode.SHADOW
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.exploration_floor = float(exploration_floor)
        self.importance_clip = float(importance_clip)
        self._clock = clock
        self._lock = threading.RLock()
        self._candidate_revision = 0
        self._active_revision = 0
        self._active_policy_digest = _digest({"schema": SCHEMA_VERSION, "mode": "fixed-active", "actions": ACTIONS, "revision": 0})
        self._A = {a: [[self.regularization if i == j else 0.0 for j in range(FEATURE_DIM)] for i in range(FEATURE_DIM)] for a in ACTIONS}
        self._b = {a: [0.0 for _ in range(FEATURE_DIM)] for a in ACTIONS}
        self._seen: dict[str, str] = {}
        self._selection_log: dict[str, SelectionRecord] = {}
        self._evaluation_window: list[dict[str, Any]] = []
        self._quarantined_legacy: dict[str, str] = {}

    @property
    def active_policy_revision(self) -> int:
        return self._active_revision

    @property
    def candidate_policy_revision(self) -> int:
        return self._candidate_revision

    @property
    def active_policy_digest(self) -> str:
        return self._active_policy_digest

    @property
    def candidate_policy_digest(self) -> str:
        return _digest(self._state_payload(include_logs=False)["candidate"])

    def quarantine_legacy_weights(self, weights: Mapping[str, Any]) -> Mapping[str, str]:
        self._quarantined_legacy = {str(k): "rejected_not_imported" for k in weights.keys()}
        return dict(self._quarantined_legacy)

    def context_features(self, observation: LearningObservation) -> tuple[float, ...]:
        validate_observation(observation)
        prov = observation.provenance_type.value
        rev = min(max(float(observation.alignment_revision), 0.0), 1000.0) / 1000.0
        features = (
            1.0,
            _bucket(observation.case_digest),
            _bucket(observation.request_digest),
            _bucket(observation.selected_plan_digest),
            round(rev, 12),
            1.0 if prov == "derivation" else 0.0,
            1.0 if prov == "retrieved_evidence" else 0.0,
            1.0 if prov == "system_audit" else 0.0,
        )
        for value in features:
            if not math.isfinite(value) or value < -1.0 or value > 1.0:
                raise ValueError("context feature out of bounds")
        return features

    def distribution_digest(self, distribution: Mapping[str, float]) -> str:
        return _digest({"schema": SCHEMA_VERSION, "distribution": {a: distribution[a] for a in sorted(distribution)}})

    def select_shadow(self, observation: LearningObservation, candidate_set: Sequence[str] | None = None, active_distribution: Mapping[str, float] | None = None) -> SelectionRecord:
        validate_observation(observation)
        actions = tuple(candidate_set or ACTIONS)
        self._validate_actions(actions)
        features = self.context_features(observation)
        with self._lock:
            active_distribution = dict(active_distribution) if active_distribution is not None else {a: round(1.0 / len(actions), 12) for a in actions}
            if set(active_distribution) != set(actions):
                raise ValueError("active distribution candidate mismatch")
            if any((not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0.0) for v in active_distribution.values()):
                raise ValueError("invalid active distribution")
            total_active = sum(float(v) for v in active_distribution.values())
            if total_active <= 0.0:
                raise ValueError("invalid active distribution")
            active_distribution = {a: round(float(active_distribution[a]) / total_active, 12) for a in actions}
            active_choice = max(active_distribution.items(), key=lambda kv: (kv[1], -ACTIONS.index(kv[0])))[0]
            candidate_distribution = self._candidate_distribution(features, actions)
            candidate_choice = max(candidate_distribution.items(), key=lambda kv: (kv[1], -ACTIONS.index(kv[0])))[0]
            record = SelectionRecord(
                selection_id=f"sel-{observation.canonical_observation_digest[:32]}",
                candidate_set=actions,
                active_choice=active_choice,
                active_distribution=active_distribution,
                active_propensity=active_distribution[observation.selected_tool_id] if observation.selected_tool_id in active_distribution else 0.0,
                candidate_choice=candidate_choice,
                candidate_distribution=candidate_distribution,
                context_schema_version=CONTEXT_SCHEMA_VERSION,
                context_digest=_digest({"schema": CONTEXT_SCHEMA_VERSION, "features": features, "names": FEATURE_NAMES}),
                active_policy_revision=self._active_revision,
                active_policy_digest=self._active_policy_digest,
                candidate_policy_revision=self._candidate_revision,
                candidate_policy_digest=self.candidate_policy_digest,
                alignment_digest=observation.alignment_digest,
                csiu_policy_digest=observation.csiu_policy_digest,
                csiu_snapshot_digest=observation.csiu_snapshot_digest,
                timestamp_utc=_now(self._clock),
            )
            self._selection_log[observation.canonical_observation_digest] = record
            return record

    def update_from_observation(self, observation: LearningObservation) -> BanditUpdateResult:
        validate_observation(observation)
        if observation.selected_tool_id not in ACTIONS:
            raise ValueError("unknown selected tool")
        if not (0.0 < observation.action_propensity <= 1.0) or not math.isfinite(observation.action_propensity):
            raise ValueError("invalid propensity")
        with self._lock:
            prior_id = self._seen.get(observation.observation_id)
            if prior_id is not None:
                if prior_id != observation.canonical_observation_digest:
                    return BanditUpdateResult(BanditUpdateStatus.CONFLICT, observation.canonical_observation_digest, 0.0, 0.0, "none", self._candidate_revision, "observation id conflict")
                return BanditUpdateResult(BanditUpdateStatus.REPLAYED, observation.canonical_observation_digest, 0.0, 0.0, "none", self._candidate_revision, "duplicate observation")
            record = self._selection_log.get(observation.canonical_observation_digest)
            if record is None:
                return BanditUpdateResult(BanditUpdateStatus.NOT_ACCEPTED, observation.canonical_observation_digest, 0.0, 0.0, "none", self._candidate_revision, "missing selection log")
            if record.active_policy_digest != self._active_policy_digest:
                raise ValueError("policy digest mismatch")
            if record.active_distribution.get(observation.selected_tool_id) != observation.action_propensity:
                raise ValueError("propensity mismatch")
            if self.distribution_digest(record.active_distribution) != observation.selection_distribution_digest:
                raise ValueError("distribution digest mismatch")
            reward = self._reward(observation)
            if reward == 0.0:
                self._seen[observation.observation_id] = observation.canonical_observation_digest
                return BanditUpdateResult(BanditUpdateStatus.NOT_ACCEPTED, observation.canonical_observation_digest, 0.0, 0.0, "neutral_or_unsupported", self._candidate_revision, "neutral reward")
            raw_weight = 1.0 / observation.action_propensity
            weight = min(self.importance_clip, raw_weight)
            clipping = "clipped" if raw_weight > self.importance_clip else "not_clipped"
            x = self.context_features(observation)
            a = observation.selected_tool_id
            for i in range(FEATURE_DIM):
                self._b[a][i] += weight * reward * x[i]
                for j in range(FEATURE_DIM):
                    self._A[a][i][j] += weight * x[i] * x[j]
            self._candidate_revision += 1
            self._seen[observation.observation_id] = observation.canonical_observation_digest
            self._evaluation_window.append({"observation_digest": observation.canonical_observation_digest, "reward": reward, "importance_weight": weight, "candidate_choice": record.candidate_choice, "active_choice": record.active_choice})
            self._evaluation_window = self._evaluation_window[-EVAL_WINDOW:]
            return BanditUpdateResult(BanditUpdateStatus.APPLIED, observation.canonical_observation_digest, reward, weight, clipping, self._candidate_revision)

    def _reward(self, observation: LearningObservation) -> float:
        if observation.terminal_status is TerminalStatus.VALIDATED_SUCCESS:
            return 1.0
        if observation.terminal_status is TerminalStatus.VALIDATED_FAILURE:
            return -1.0
        return 0.0

    def _candidate_distribution(self, features: Sequence[float], actions: Sequence[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for action in actions:
            inv = _inverse(self._A[action])
            theta = _mat_vec(inv, self._b[action])
            bonus = math.sqrt(max(0.0, _dot(features, _mat_vec(inv, features))))
            score = _dot(theta, features) + self.alpha * bonus
            if not math.isfinite(score):
                raise ValueError("non-finite bandit score")
            scores[action] = score
        mx = max(scores.values())
        exps = {a: math.exp(max(-40.0, min(40.0, scores[a] - mx))) for a in actions}
        total = sum(exps.values())
        dist = {a: (1.0 - self.exploration_floor * len(actions)) * (exps[a] / total) + self.exploration_floor for a in actions}
        z = sum(dist.values())
        return {a: round(dist[a] / z, 12) for a in actions}


    def activate_candidate(self, *, expected_active_digest: str, expected_candidate_digest: str) -> str:
        """CAS-promote the shadow candidate digest to active; only governance calls this."""
        with self._lock:
            current_candidate = self.candidate_policy_digest
            if self._active_policy_digest != expected_active_digest:
                raise ValueError("stale active policy")
            if current_candidate != expected_candidate_digest:
                raise ValueError("stale candidate policy")
            self._active_revision += 1
            self._active_policy_digest = current_candidate
            return self._active_policy_digest

    def selection_records(self) -> tuple[SelectionRecord, ...]:
        with self._lock:
            return tuple(self._selection_log.values())

    def evaluation_metrics(self) -> Mapping[str, Any]:
        n = len(self._evaluation_window)
        return {"window_size": n, "minimum_samples": MIN_EVAL_SAMPLES, "ready": n >= MIN_EVAL_SAMPLES, "mean_reward": round(sum(e["reward"] for e in self._evaluation_window) / n, 12) if n else 0.0}

    def state_bytes(self) -> bytes:
        return _canonical(self._state_payload(include_logs=True)).encode()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError("symlinked bandit state path")
        data = self.state_bytes()
        with tempfile.NamedTemporaryFile("wb", dir=str(target.parent), delete=False) as fh:
            fh.write(data)
            tmp = Path(fh.name)
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path, *, clock: Callable[[], datetime] | None = None) -> "ShadowLinUCBToolBandit":
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode(), object_pairs_hook=_reject_duplicate_keys, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")))
        if _canonical(payload).encode() != raw:
            raise ValueError("non-canonical bandit state")
        obj = cls(alpha=payload["alpha"], regularization=payload["regularization"], exploration_floor=payload["exploration_floor"], importance_clip=payload["importance_clip"], clock=clock)
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("actions") != list(ACTIONS) or payload.get("feature_names") != list(FEATURE_NAMES) or payload.get("mode") != BanditMode.SHADOW.value:
            raise ValueError("bandit state schema mismatch")
        candidate = payload["candidate"]
        obj._candidate_revision = int(candidate["revision"])
        obj._active_revision = int(payload["active"]["revision"])
        obj._active_policy_digest = payload["active"]["digest"]
        obj._A = {a: [[float(v) for v in row] for row in candidate["A"][a]] for a in ACTIONS}
        obj._b = {a: [float(v) for v in candidate["b"][a]] for a in ACTIONS}
        obj._seen = dict(payload.get("seen", {}))
        obj._evaluation_window = list(payload.get("evaluation_window", []))[-EVAL_WINDOW:]
        obj._selection_log = {
            str(key): SelectionRecord(
                selection_id=value["selection_id"],
                candidate_set=tuple(value["candidate_set"]),
                active_choice=value["active_choice"],
                active_distribution=dict(value["active_distribution"]),
                active_propensity=float(value["active_propensity"]),
                candidate_choice=value["candidate_choice"],
                candidate_distribution=dict(value["candidate_distribution"]),
                context_schema_version=value["context_schema_version"],
                context_digest=value["context_digest"],
                active_policy_revision=int(value["active_policy_revision"]),
                active_policy_digest=value["active_policy_digest"],
                candidate_policy_revision=int(value["candidate_policy_revision"]),
                candidate_policy_digest=value["candidate_policy_digest"],
                alignment_digest=value["alignment_digest"],
                csiu_policy_digest=value["csiu_policy_digest"],
                csiu_snapshot_digest=value["csiu_snapshot_digest"],
                timestamp_utc=value["timestamp_utc"],
            )
            for key, value in payload.get("selection_log", {}).items()
        }
        expected = _digest(obj._state_payload(include_logs=False)["candidate"])
        if payload.get("candidate_digest") != expected:
            raise ValueError("bandit candidate digest mismatch")
        return obj

    def _state_payload(self, *, include_logs: bool) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode.value,
            "actions": list(ACTIONS),
            "feature_names": list(FEATURE_NAMES),
            "alpha": self.alpha,
            "regularization": self.regularization,
            "exploration_floor": self.exploration_floor,
            "importance_clip": self.importance_clip,
            "active": {"revision": self._active_revision, "digest": self._active_policy_digest},
            "candidate": {"revision": self._candidate_revision, "A": self._A, "b": self._b},
            "seen": self._seen,
            "evaluation_window": self._evaluation_window,
            "quarantined_legacy": self._quarantined_legacy,
        }
        payload["candidate_digest"] = _digest(payload["candidate"])
        if include_logs:
            payload["selection_log"] = {k: _record_payload(v) for k, v in self._selection_log.items()}
        return payload

    def _validate_actions(self, actions: Sequence[str]) -> None:
        if not actions:
            raise ValueError("empty candidate set")
        if len(actions) != len(set(actions)):
            raise ValueError("duplicate action")
        unknown = [a for a in actions if a not in ACTIONS]
        if unknown:
            raise ValueError("unknown action")


def _record_payload(record: SelectionRecord) -> dict[str, Any]:
    return {k: getattr(record, k) for k in record.__dataclass_fields__}


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out
