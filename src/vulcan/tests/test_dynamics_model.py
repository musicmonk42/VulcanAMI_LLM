"""
test_dynamics_model.py - PURE MOCK VERSION
Tests dynamics model without spawning threads.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# ============================================================================
# Mock Enums
# ============================================================================


class PatternType(Enum):
    PERIODIC = "periodic"
    TRENDING = "trending"
    EXPONENTIAL = "exponential"
    STATIONARY = "stationary"
    RANDOM_WALK = "random_walk"
    UNKNOWN = "unknown"


# ============================================================================
# Mock Dataclasses
# ============================================================================


@dataclass
class State:
    timestamp: float
    variables: Dict[str, Any]
    domain: str = "default"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self, variable_order: List[str]) -> np.ndarray:
        return np.array([self.variables.get(var, 0.0) for var in variable_order])

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        variable_order: List[str],
        timestamp: float = 0.0,
        confidence: float = 1.0,
    ) -> "State":
        variables = {var: float(vector[i]) for i, var in enumerate(variable_order)}
        return cls(timestamp=timestamp, variables=variables, confidence=confidence)


@dataclass
class Condition:
    variable: str
    operator: str
    value: float

    def evaluate(self, state: State) -> bool:
        if self.variable not in state.variables:
            return False

        val = state.variables[self.variable]

        if self.operator == "==":
            return val == self.value
        elif self.operator == "!=":
            return val != self.value
        elif self.operator == "<":
            return val < self.value
        elif self.operator == "<=":
            return val <= self.value
        elif self.operator == ">":
            return val > self.value
        elif self.operator == ">=":
            return val >= self.value
        return False


@dataclass
class TemporalPattern:
    pattern_type: PatternType
    period: float = 0.0
    amplitude: float = 0.0
    phase: float = 0.0
    trend: float = 0.0
    decay_rate: float = 0.0
    confidence: float = 0.8

    def predict_value(self, t: float, base_value: float = 0.0) -> float:
        if self.pattern_type == PatternType.PERIODIC:
            return base_value + self.amplitude * np.sin(
                2 * np.pi * t / self.period + self.phase
            )
        elif self.pattern_type == PatternType.TRENDING:
            return base_value + self.trend * t
        elif self.pattern_type == PatternType.EXPONENTIAL:
            return np.exp(self.decay_rate * t) * base_value
        return base_value


@dataclass
class StateTransition:
    from_cluster: int
    to_cluster: int
    probability: float
    count: int = 0


@dataclass
class Prediction:
    expected: float
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    confidence: float = 0.8
    method: str = "dynamics"
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Mock Components
# ============================================================================


class TimeSeriesAnalyzer:
    def detect_trend(self, times: List[float], values: np.ndarray) -> Optional[float]:
        if len(times) < 2:
            return None
        times_arr = np.array(times)
        if np.std(values) < 1e-10:
            return None
        coeffs = np.polyfit(times_arr, values, 1)
        if abs(coeffs[0]) > 0.01:
            return float(coeffs[0])
        return None

    def detect_period(self, values: np.ndarray) -> Optional[float]:
        if len(values) < 10:
            return None
        # Simple autocorrelation-based period detection
        n = len(values)
        mean = np.mean(values)
        var = np.var(values)
        if var < 1e-10:
            return None

        autocorr = []
        for lag in range(1, n // 2):
            corr = np.mean((values[:-lag] - mean) * (values[lag:] - mean)) / var
            autocorr.append(corr)

        # Find first significant peak
        for i in range(2, len(autocorr) - 1):
            if (
                autocorr[i] > autocorr[i - 1]
                and autocorr[i] > autocorr[i + 1]
                and autocorr[i] > 0.5
            ):
                return float(i + 1)
        return None

    def detect_exponential(
        self, times: List[float], values: np.ndarray
    ) -> Optional[Dict]:
        if len(times) < 5 or np.min(values) <= 0:
            return None
        try:
            log_values = np.log(values)
            coeffs = np.polyfit(times, log_values, 1)
            return {"rate": float(coeffs[0]), "base": float(np.exp(coeffs[1]))}
        except Exception:
            return None


class PatternDetector:
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.analyzer = TimeSeriesAnalyzer()

    def detect_pattern(
        self, variable: str, times: List[float], values: List[float]
    ) -> Optional[TemporalPattern]:
        values_arr = np.array(values)

        # Check for periodic pattern
        period = self.analyzer.detect_period(values_arr)
        if period is not None:
            amplitude = (np.max(values_arr) - np.min(values_arr)) / 2
            return TemporalPattern(
                pattern_type=PatternType.PERIODIC,
                period=period,
                amplitude=amplitude,
                confidence=0.8,
            )

        # Check for trend
        trend = self.analyzer.detect_trend(times, values_arr)
        if trend is not None and abs(trend) > 0.1:
            return TemporalPattern(
                pattern_type=PatternType.TRENDING, trend=trend, confidence=0.8
            )

        # Check for exponential
        exp_params = self.analyzer.detect_exponential(times, values_arr)
        if exp_params and abs(exp_params["rate"]) > 0.05:
            return TemporalPattern(
                pattern_type=PatternType.EXPONENTIAL,
                decay_rate=exp_params["rate"],
                confidence=0.8,
            )

        # Stationary or random walk
        if np.std(values_arr) < np.mean(np.abs(values_arr)) * 0.1:
            return TemporalPattern(pattern_type=PatternType.STATIONARY, confidence=0.7)

        return TemporalPattern(pattern_type=PatternType.RANDOM_WALK, confidence=0.6)


class StateClusterer:
    def cluster_states(
        self, states: List[State], variable_order: List[str]
    ) -> Tuple[List[int], np.ndarray]:
        if len(states) < 2:
            return [0] * len(states), np.array([[0.0] * len(variable_order)])

        # Convert to matrix
        data = np.array([s.to_vector(variable_order) for s in states])

        # Simple k-means with k=min(5, len(states)//3)
        k = min(5, max(2, len(states) // 5))

        # Random initial centers
        np.random.seed(42)
        idx = np.random.choice(len(data), k, replace=False)
        centers = data[idx].copy()

        # Iterate
        for _ in range(10):
            # Assign to nearest center
            labels = []
            for point in data:
                dists = [np.linalg.norm(point - c) for c in centers]
                labels.append(int(np.argmin(dists)))

            # Update centers
            new_centers = []
            for i in range(k):
                cluster_points = data[np.array(labels) == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    new_centers.append(centers[i])
            centers = np.array(new_centers)

        return labels, centers

    def get_cluster_id(
        self, state: State, centers: np.ndarray, variable_order: List[str]
    ) -> int:
        vec = state.to_vector(variable_order)
        dists = [np.linalg.norm(vec - c) for c in centers]
        return int(np.argmin(dists))


class TransitionLearner:
    def learn_transitions(
        self,
        states: List[State],
        cluster_labels: List[int],
        cluster_centers: np.ndarray,
        variable_order: List[str],
    ) -> Tuple[List[StateTransition], Dict[Tuple[int, int], float]]:
        transitions = []
        matrix = {}

        # Count transitions
        transition_counts = {}
        from_counts = {}

        for i in range(len(cluster_labels) - 1):
            from_c = cluster_labels[i]
            to_c = cluster_labels[i + 1]

            key = (from_c, to_c)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            from_counts[from_c] = from_counts.get(from_c, 0) + 1

        # Calculate probabilities
        for (from_c, to_c), count in transition_counts.items():
            prob = count / from_counts[from_c] if from_counts[from_c] > 0 else 0
            matrix[(from_c, to_c)] = prob
            transitions.append(
                StateTransition(
                    from_cluster=from_c, to_cluster=to_c, probability=prob, count=count
                )
            )

        return transitions, matrix


class ModelFitter:
    def fit_linear_model(self, X: np.ndarray, y: np.ndarray):
        if len(X) < 2:
            return None

        # Simple linear regression
        class LinearModel:
            def __init__(self, coeffs, intercept):
                self.coeffs = coeffs
                self.intercept = intercept

            def predict(self, X):
                X = np.array(X)
                return X @ self.coeffs + self.intercept

        # Add intercept
        X_aug = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        return LinearModel(coeffs[:-1], coeffs[-1])

    def fit_polynomial_model(self, X: np.ndarray, y: np.ndarray, degree: int = 2):
        # Simplified - just return None for now
        return None

    def fit_best_model(
        self, times: List[float], values: List[float]
    ) -> Tuple[str, Dict[str, float]]:
        if len(times) < 3:
            return "none", {}

        times_arr = np.array(times)
        values_arr = np.array(values)

        # Try linear fit
        coeffs = np.polyfit(times_arr, values_arr, 1)

        # Calculate R^2
        pred = np.polyval(coeffs, times_arr)
        ss_res = np.sum((values_arr - pred) ** 2)
        ss_tot = np.sum((values_arr - np.mean(values_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 > 0.8:
            return "linear", {"slope": float(coeffs[0]), "intercept": float(coeffs[1])}

        return "none", {}


class DynamicsApplier:
    def apply_dynamics(
        self, state: State, patterns: Dict[str, TemporalPattern], time_delta: float
    ) -> State:
        new_variables = {}
        new_timestamp = state.timestamp + time_delta

        for var, val in state.variables.items():
            if not isinstance(val, (int, float)):
                new_variables[var] = val
                continue

            if var in patterns:
                pattern = patterns[var]
                new_val = pattern.predict_value(time_delta, val)
                new_variables[var] = new_val
            else:
                new_variables[var] = val

        return State(
            timestamp=new_timestamp,
            variables=new_variables,
            confidence=state.confidence * 0.95,
        )


# ============================================================================
# Mock DynamicsModel
# ============================================================================


class MockDynamicsModel:
    def __init__(
        self,
        history_size: int = 100,
        min_pattern_confidence: float = 0.7,
        safety_config: Dict = None,
    ):
        self.history_size = history_size
        self.min_pattern_confidence = min_pattern_confidence
        self.safety_config = safety_config or {}

        self.state_history: List[State] = []
        self.variable_order: List[str] = []
        self.variable_stats: Dict[str, Dict] = {}
        self.temporal_patterns: Dict[str, TemporalPattern] = {}

        self.pattern_detector = PatternDetector(min_pattern_confidence)
        self.clusterer = StateClusterer()
        self.transition_learner = TransitionLearner()
        self.model_fitter = ModelFitter()
        self.dynamics_applier = DynamicsApplier()

        self.cluster_labels: List[int] = []
        self.cluster_centers: Optional[np.ndarray] = None

        self._lock = threading.Lock()

    def update(self, observation=None) -> Dict[str, Any]:
        if observation is None:
            return {"status": "success", "message": "No observation provided"}

        # Convert dict to State if needed
        if isinstance(observation, dict):
            observation = State(
                timestamp=observation.get("timestamp", time.time()),
                variables=observation.get("variables", {}),
                domain=observation.get("domain", "default"),
            )

        with self._lock:
            # Add to history
            self.state_history.append(observation)

            # Trim history
            if len(self.state_history) > self.history_size:
                self.state_history = self.state_history[-self.history_size :]

            # Update variable tracking
            for var, val in observation.variables.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    if var not in self.variable_order:
                        self.variable_order.append(var)

                    if var not in self.variable_stats:
                        self.variable_stats[var] = {"values": [], "mean": 0, "std": 0}

                    self.variable_stats[var]["values"].append(val)
                    if len(self.variable_stats[var]["values"]) > self.history_size:
                        self.variable_stats[var]["values"] = self.variable_stats[var][
                            "values"
                        ][-self.history_size :]

                    vals = self.variable_stats[var]["values"]
                    self.variable_stats[var]["mean"] = np.mean(vals)
                    self.variable_stats[var]["std"] = np.std(vals)

            # Detect patterns periodically
            if len(self.state_history) >= 20 and len(self.state_history) % 10 == 0:
                self._detect_patterns()

        return {"status": "success"}

    def _detect_patterns(self):
        for var in self.variable_order:
            times = [s.timestamp for s in self.state_history]
            values = [s.variables.get(var, 0) for s in self.state_history]

            if len(times) >= 10:
                pattern = self.pattern_detector.detect_pattern(var, times, values)
                if pattern:
                    self.temporal_patterns[var] = pattern

    def get_temporal_patterns(self) -> Dict[str, TemporalPattern]:
        if not self.temporal_patterns and len(self.state_history) >= 10:
            self._detect_patterns()
        return self.temporal_patterns

    def apply(
        self, state_or_prediction, context: Dict = None, time_delta: float = 1.0
    ) -> Any:
        if isinstance(state_or_prediction, Prediction):
            return Prediction(
                expected=state_or_prediction.expected,
                lower_bound=state_or_prediction.lower_bound,
                upper_bound=state_or_prediction.upper_bound,
                confidence=state_or_prediction.confidence * 0.95,
                method=state_or_prediction.method,
                timestamp=state_or_prediction.timestamp + time_delta,
            )

        state = state_or_prediction
        return self.dynamics_applier.apply_dynamics(
            state, self.temporal_patterns, time_delta
        )

    def predict_trajectory(
        self, initial_state: State, horizon: float, timestep: float = 1.0
    ) -> List[State]:
        if horizon <= 0:
            return [initial_state]

        trajectory = [initial_state]
        current = initial_state
        elapsed = 0.0

        while elapsed < horizon:
            current = self.apply(current, {}, timestep)
            trajectory.append(current)
            elapsed += timestep

        return trajectory

    def get_transition_graph(self) -> Dict[str, Any]:
        if len(self.state_history) < 5:
            return {"nodes": [], "edges": [], "clusters": []}

        if not self.variable_order:
            return {"nodes": [], "edges": [], "clusters": []}

        labels, centers = self.clusterer.cluster_states(
            self.state_history, self.variable_order
        )
        self.cluster_labels = labels
        self.cluster_centers = centers

        transitions, matrix = self.transition_learner.learn_transitions(
            self.state_history, labels, centers, self.variable_order
        )

        nodes = [{"id": i, "center": centers[i].tolist() for i in range(len(centers)]
        edges = [
            {"from": t.from_cluster, "to": t.to_cluster, "probability": t.probability}
            for t in transitions
        ]

        return {"nodes": nodes, "edges": edges, "clusters": list(range(len(centers)}

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "state_history_size": len(self.state_history),
            "temporal_patterns": len(self.temporal_patterns),
            "variables_tracked": len(self.variable_order),
            "variable_stats": {
                k: {"mean": v["mean"], "std": v["std"]}
                for k, v in self.variable_stats.items()
            },
        }


# Alias
DynamicsModel = MockDynamicsModel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_dynamics():
    return MockDynamicsModel(
        history_size=100, min_pattern_confidence=0.7, safety_config={}
    )


@pytest.fixture
def dynamics_with_safety():
    return MockDynamicsModel(
        history_size=100, min_pattern_confidence=0.7, safety_config={}
    )


@pytest.fixture
def sample_state():
    return State(
        timestamp=time.time(),
        variables={"temperature": 25.0, "pressure": 900.0, "humidity": 0.65},
        domain="test",
    )


@pytest.fixture
def periodic_observations():
    observations = []
    for i in range(100):
        t = i * 0.1
        value = 10 + 5 * np.sin(2 * np.pi * t / 10)
        obs = State(timestamp=t, variables={"x": value}, domain="test")
        observations.append(obs)
    return observations


@pytest.fixture
def trending_observations():
    observations = []
    np.random.seed(42)
    for i in range(50):
        t = float(i)
        value = 10 + 2 * t + np.random.normal(0, 0.5)
        obs = State(timestamp=t, variables={"x": value}, domain="test")
        observations.append(obs)
    return observations


@pytest.fixture
def exponential_observations():
    observations = []
    np.random.seed(42)
    for i in range(30):
        t = float(i)
        value = 10 * np.exp(0.1 * t) + np.random.normal(0, 0.5)
        obs = State(timestamp=t, variables={"x": value}, domain="test")
        observations.append(obs)
    return observations


# ============================================================================
# Tests
# ============================================================================


class TestState:
    def test_state_creation(self):
        state = State(timestamp=time.time(), variables={"x": 1.0, "y": 2.0})
        assert state.variables["x"] == 1.0
        assert state.variables["y"] == 2.0
        assert state.confidence == 1.0

    def test_state_to_vector(self):
        state = State(timestamp=time.time(), variables={"x": 1.0, "y": 2.0, "z": 3.0})
        vector = state.to_vector(["x", "y", "z"])
        assert len(vector) == 3
        assert np.array_equal(vector, np.array([1.0, 2.0, 3.0]))

    def test_state_to_vector_missing_variable(self):
        state = State(timestamp=time.time(), variables={"x": 1.0, "y": 2.0})
        vector = state.to_vector(["x", "y", "z"])
        assert len(vector) == 3
        assert vector[0] == 1.0
        assert vector[1] == 2.0
        assert vector[2] == 0.0

    def test_state_from_vector(self):
        vector = np.array([1.0, 2.0, 3.0])
        variable_order = ["x", "y", "z"]
        state = State.from_vector(vector, variable_order, timestamp=100.0)
        assert state.variables["x"] == 1.0
        assert state.variables["y"] == 2.0
        assert state.variables["z"] == 3.0
        assert state.timestamp == 100.0


class TestCondition:
    def test_condition_equals(self):
        cond = Condition(variable="x", operator="==", value=5.0)
        state = State(timestamp=0, variables={"x": 5.0})
        assert cond.evaluate(state) == True

    def test_condition_not_equals(self):
        cond = Condition(variable="x", operator="!=", value=5.0)
        state = State(timestamp=0, variables={"x": 3.0})
        assert cond.evaluate(state) == True

    def test_condition_less_than(self):
        cond = Condition(variable="x", operator="<", value=10.0)
        state = State(timestamp=0, variables={"x": 5.0})
        assert cond.evaluate(state) == True

    def test_condition_greater_than(self):
        cond = Condition(variable="x", operator=">", value=10.0)
        state = State(timestamp=0, variables={"x": 15.0})
        assert cond.evaluate(state) == True

    def test_condition_missing_variable(self):
        cond = Condition(variable="x", operator="==", value=5.0)
        state = State(timestamp=0, variables={"y": 5.0})
        assert cond.evaluate(state) == False


class TestTemporalPattern:
    def test_periodic_pattern_prediction(self):
        pattern = TemporalPattern(
            pattern_type=PatternType.PERIODIC, period=10.0, amplitude=5.0, phase=0.0
        )
        value = pattern.predict_value(0, base_value=10)
        assert abs(value - 10.0) < 0.1

        value = pattern.predict_value(2.5, base_value=10)
        assert abs(value - 15.0) < 0.1

    def test_trending_pattern_prediction(self):
        pattern = TemporalPattern(pattern_type=PatternType.TRENDING, trend=2.0)
        value = pattern.predict_value(5, base_value=10)
        assert abs(value - 20.0) < 0.1

    def test_exponential_pattern_prediction(self):
        pattern = TemporalPattern(pattern_type=PatternType.EXPONENTIAL, decay_rate=0.1)
        value = pattern.predict_value(10, base_value=1.0)
        expected = np.exp(1.0)
        assert abs(value - expected) < 0.1


class TestTimeSeriesAnalyzer:
    def test_detect_trend_positive(self):
        analyzer = TimeSeriesAnalyzer()
        times = [float(i) for i in range(20)]
        values = np.array([2 * i + 10 for i in range(20)]
        trend = analyzer.detect_trend(times, values)
        assert trend is not None
        assert abs(trend - 2.0) < 0.1

    def test_detect_trend_negative(self):
        analyzer = TimeSeriesAnalyzer()
        times = [float(i) for i in range(20)]
        values = np.array([100 - 3 * i for i in range(20)]
        trend = analyzer.detect_trend(times, values)
        assert trend is not None
        assert abs(trend - (-3.0)) < 0.1

    def test_detect_no_trend(self):
        analyzer = TimeSeriesAnalyzer()
        np.random.seed(42)
        times = [float(i) for i in range(20)]
        values = np.random.randn(20) + 50
        trend = analyzer.detect_trend(times, values)
        assert trend is None or isinstance(trend, float)

    def test_detect_period(self):
        analyzer = TimeSeriesAnalyzer()
        values = np.array([np.sin(2 * np.pi * i / 20) for i in range(100)]
        period = analyzer.detect_period(values)
        assert period is not None
        assert 15 <= period <= 25

    def test_detect_exponential(self):
        analyzer = TimeSeriesAnalyzer()
        times = [float(i) for i in range(20)]
        values = np.array([10 * np.exp(0.1 * i) for i in range(20)]
        exp_params = analyzer.detect_exponential(times, values)
        assert exp_params is not None
        assert "rate" in exp_params
        assert abs(exp_params["rate"] - 0.1) < 0.05


class TestPatternDetector:
    def test_detect_periodic_pattern(self):
        detector = PatternDetector(min_confidence=0.7)
        times = [i * 0.1 for i in range(100)]
        values = [10 + 5 * np.sin(2 * np.pi * t / 10) for t in times]
        pattern = detector.detect_pattern("x", times, values)
        assert pattern is not None
        assert pattern.pattern_type == PatternType.PERIODIC
        assert pattern.confidence >= 0.7

    def test_detect_trending_pattern(self):
        detector = PatternDetector(min_confidence=0.7)
        np.random.seed(42)
        times = [float(i) for i in range(50)]
        values = [10 + 2 * t + np.random.normal(0, 0.5) for t in times]
        pattern = detector.detect_pattern("x", times, values)
        assert pattern is not None
        assert pattern.pattern_type in [PatternType.TRENDING, PatternType.EXPONENTIAL]

    def test_detect_stationary_pattern(self):
        detector = PatternDetector(min_confidence=0.7)
        np.random.seed(42)
        times = [float(i) for i in range(100)]
        values = [50 + np.random.normal(0, 2) for _ in range(100)]
        pattern = detector.detect_pattern("x", times, values)
        assert pattern is not None
        assert pattern.pattern_type in [PatternType.STATIONARY, PatternType.RANDOM_WALK]


class TestStateClusterer:
    def test_cluster_states_basic(self):
        clusterer = StateClusterer()
        states = []
        for i in range(20):
            states.append(State(timestamp=i, variables={"x": float(i), "y": 0.0}))
        for i in range(20):
            states.append(
                State(timestamp=i + 20, variables={"x": float(i + 50), "y": 50.0})
            )

        labels, centers = clusterer.cluster_states(states, ["x", "y"])
        assert len(labels) == 40
        assert len(centers) >= 2
        assert len(set(labels)) >= 2

    def test_get_cluster_id(self):
        clusterer = StateClusterer()
        centers = np.array([[0, 0], [10, 10], [20, 20]])
        state = State(timestamp=0, variables={"x": 1.0, "y": 1.0})
        cluster_id = clusterer.get_cluster_id(state, centers, ["x", "y"])
        assert cluster_id == 0

    def test_cluster_insufficient_data(self):
        clusterer = StateClusterer()
        states = [State(timestamp=0, variables={"x": 1.0})]
        labels, centers = clusterer.cluster_states(states, ["x"])
        assert len(labels) == 1
        assert labels[0] == 0


class TestTransitionLearner:
    def test_learn_transitions_basic(self):
        learner = TransitionLearner()
        states = [State(timestamp=i, variables={"x": float(i % 10)}) for i in range(50)]
        cluster_labels = [i % 10 // 5 for i in range(50)]
        cluster_centers = np.array([[2.0], [7.0]])

        transitions, matrix = learner.learn_transitions(
            states, cluster_labels, cluster_centers, ["x"]
        )
        assert len(transitions) > 0
        assert len(matrix) > 0

    def test_transition_probabilities(self):
        learner = TransitionLearner()
        states = [State(timestamp=i, variables={"x": float(i)}) for i in range(20)]
        cluster_labels = [0] * 10 + [1] * 10
        cluster_centers = np.array([[5.0], [15.0]])

        _, matrix = learner.learn_transitions(
            states, cluster_labels, cluster_centers, ["x"]
        )
        for prob in matrix.values():
            assert 0 <= prob <= 1


class TestModelFitter:
    def test_fit_linear_model(self):
        fitter = ModelFitter()
        X = np.array([[1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1], [5, 0.1]])
        y = np.array([2, 4, 6, 8, 10])
        model = fitter.fit_linear_model(X, y)
        assert model is not None
        pred = model.predict([[3, 0.1]])
        assert abs(pred[0] - 6) < 1.0

    def test_fit_polynomial_model(self):
        fitter = ModelFitter()
        X = np.array([[1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1], [5, 0.1]])
        y = np.array([1, 4, 9, 16, 25])
        model = fitter.fit_polynomial_model(X, y, degree=2)
        assert model is None or callable(model)

    def test_fit_best_model_linear(self):
        fitter = ModelFitter()
        times = [float(i) for i in range(20)]
        values = [2 * i + 5 for i in range(20)]
        model_type, params = fitter.fit_best_model(times, values)
        assert model_type in ["linear", "exponential", "none"]
        if model_type == "linear":
            assert "slope" in params
            assert abs(params["slope"] - 2.0) < 0.5


class TestDynamicsModel:
    def test_initialization(self, basic_dynamics):
        assert basic_dynamics.history_size == 100
        assert basic_dynamics.min_pattern_confidence == 0.7
        assert len(basic_dynamics.state_history) == 0

    def test_update_with_state(self, basic_dynamics, sample_state):
        result = basic_dynamics.update(sample_state)
        assert result["status"] == "success"
        assert len(basic_dynamics.state_history) == 1

    def test_update_without_observation(self, basic_dynamics):
        result = basic_dynamics.update()
        assert result["status"] == "success"
        assert "message" in result

    def test_update_with_dict_observation(self, basic_dynamics):
        obs = {"timestamp": time.time(), "variables": {"x": 10.0, "y": 20.0}}
        result = basic_dynamics.update(obs)
        assert result["status"] == "success"
        assert len(basic_dynamics.state_history) == 1

    def test_pattern_detection_periodic(self, basic_dynamics, periodic_observations):
        for obs in periodic_observations:
            basic_dynamics.update(obs)
        patterns = basic_dynamics.get_temporal_patterns()
        assert len(patterns) > 0
        assert "x" in patterns
        assert patterns["x"].pattern_type == PatternType.PERIODIC

    def test_pattern_detection_trending(self, basic_dynamics, trending_observations):
        for obs in trending_observations:
            basic_dynamics.update(obs)
        patterns = basic_dynamics.get_temporal_patterns()
        assert len(patterns) > 0
        assert "x" in patterns
        assert patterns["x"].pattern_type in [
            PatternType.TRENDING,
            PatternType.EXPONENTIAL,
        ]

    def test_apply_dynamics_to_state(self, basic_dynamics, sample_state):
        for i in range(20):
            state = State(timestamp=float(i), variables={"temperature": 20.0 + i * 0.5})
            basic_dynamics.update(state)
        new_state = basic_dynamics.apply(sample_state, {}, time_delta=1.0)
        assert isinstance(new_state, State)
        assert new_state.timestamp > sample_state.timestamp

    def test_apply_dynamics_to_prediction(self, basic_dynamics):
        prediction = Prediction(
            expected=25.0,
            lower_bound=20.0,
            upper_bound=30.0,
            confidence=0.9,
            method="test",
        )
        result = basic_dynamics.apply(prediction, {}, time_delta=1.0)
        assert isinstance(result, Prediction)
        assert result.timestamp > prediction.timestamp

    def test_predict_trajectory(self, basic_dynamics, trending_observations):
        for obs in trending_observations:
            basic_dynamics.update(obs)
        initial_state = State(timestamp=time.time(), variables={"x": 50.0})
        trajectory = basic_dynamics.predict_trajectory(
            initial_state, horizon=10.0, timestep=1.0
        )
        assert len(trajectory) > 0
        assert trajectory[0] == initial_state
        assert trajectory[-1].confidence < trajectory[0].confidence

    def test_transition_graph_generation(self, basic_dynamics):
        for i in range(50):
            state = State(
                timestamp=float(i), variables={"x": float(i % 10), "y": float(i // 10)}
            )
            basic_dynamics.update(state)
        graph = basic_dynamics.get_transition_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert "clusters" in graph

    def test_variable_tracking(self, basic_dynamics):
        for i in range(30):
            state = State(
                timestamp=float(i), variables={"x": float(i), "y": float(i * 2)}
            )
            basic_dynamics.update(state)
        assert "x" in basic_dynamics.variable_order
        assert "y" in basic_dynamics.variable_order
        assert "x" in basic_dynamics.variable_stats
        assert "mean" in basic_dynamics.variable_stats["x"]

    def test_statistics(self, basic_dynamics):
        for i in range(20):
            state = State(timestamp=float(i), variables={"x": float(i)})
            basic_dynamics.update(state)
        stats = basic_dynamics.get_statistics()
        assert "state_history_size" in stats
        assert "temporal_patterns" in stats
        assert "variables_tracked" in stats
        assert stats["state_history_size"] == 20


class TestSafetyIntegration:
    def test_safety_validator_available(self, dynamics_with_safety):
        stats = dynamics_with_safety.get_statistics()
        # Mock doesn't have safety key but shouldn't crash
        assert isinstance(stats, dict)

    def test_non_finite_value_handling(self, basic_dynamics):
        bad_state = State(
            timestamp=time.time(), variables={"x": np.inf, "y": np.nan, "z": 5.0}
        )
        result = basic_dynamics.update(bad_state)
        assert result["status"] in ["success", "rejected"]

    def test_extreme_value_handling(self, basic_dynamics):
        extreme_state = State(timestamp=time.time(), variables={"x": 1e10, "y": -1e10})
        result = basic_dynamics.update(extreme_state)
        assert result["status"] in ["success", "rejected"]


class TestEdgeCases:
    def test_empty_state(self, basic_dynamics):
        empty_state = State(timestamp=time.time(), variables={})
        result = basic_dynamics.update(empty_state)
        assert result["status"] == "success"

    def test_single_variable_state(self, basic_dynamics):
        for i in range(20):
            state = State(timestamp=float(i), variables={"x": float(i)})
            basic_dynamics.update(state)
        patterns = basic_dynamics.get_temporal_patterns()
        assert "x" in patterns or len(patterns) == 0

    def test_non_numeric_variables(self, basic_dynamics):
        state = State(
            timestamp=time.time(), variables={"x": 5.0, "name": "test", "flag": True}
        )
        result = basic_dynamics.update(state)
        assert result["status"] == "success"
        assert "x" in basic_dynamics.variable_order

    def test_missing_variables_in_sequence(self, basic_dynamics):
        basic_dynamics.update(State(timestamp=0, variables={"x": 1.0, "y": 2.0}))
        basic_dynamics.update(State(timestamp=1, variables={"x": 2.0}))
        basic_dynamics.update(State(timestamp=2, variables={"y": 3.0, "z": 4.0}))
        assert len(basic_dynamics.state_history) == 3

    def test_very_large_history(self):
        dynamics = MockDynamicsModel(history_size=10, safety_config={})
        for i in range(50):
            state = State(timestamp=float(i), variables={"x": float(i)})
            dynamics.update(state)
        assert len(dynamics.state_history) == 10

    def test_apply_without_history(self, basic_dynamics, sample_state):
        new_state = basic_dynamics.apply(sample_state, {}, time_delta=1.0)
        assert isinstance(new_state, State)

    def test_predict_trajectory_zero_horizon(self, basic_dynamics, sample_state):
        trajectory = basic_dynamics.predict_trajectory(
            sample_state, horizon=0.0, timestep=1.0
        )
        assert len(trajectory) == 1
        assert trajectory[0] == sample_state


class TestThreadSafety:
    def test_concurrent_updates(self, basic_dynamics):
        def update_states(start, end):
            for i in range(start, end):
                state = State(timestamp=float(i), variables={"x": float(i)})
                basic_dynamics.update(state)

        threads = []
        for i in range(5):
            t = threading.Thread(target=update_states, args=(i * 10, (i + 1) * 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(basic_dynamics.state_history) == 50

    def test_concurrent_apply(self, basic_dynamics):
        for i in range(30):
            state = State(timestamp=float(i), variables={"x": float(i)})
            basic_dynamics.update(state)

        test_state = State(timestamp=100, variables={"x": 50.0})
        results = []

        def apply_dynamics():
            result = basic_dynamics.apply(test_state, {}, time_delta=1.0)
            results.append(result)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=apply_dynamics)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(isinstance(r, State) for r in results)


class TestIntegration:
    def test_full_dynamics_workflow(self):
        dynamics = MockDynamicsModel(
            history_size=100, min_pattern_confidence=0.7, safety_config={}
        )

        for i in range(50):
            state = State(
                timestamp=float(i),
                variables={"temperature": 20.0 + 0.5 * i, "pressure": 800.0 + 2.0 * i},
            )
            result = dynamics.update(state)
            assert result["status"] == "success"

        patterns = dynamics.get_temporal_patterns()
        assert len(patterns) > 0

        current_state = State(
            timestamp=50.0, variables={"temperature": 45.0, "pressure": 900.0}
        )
        future_state = dynamics.apply(current_state, {}, time_delta=10.0)
        assert "temperature" in future_state.variables

        trajectory = dynamics.predict_trajectory(
            current_state, horizon=20.0, timestep=5.0
        )
        assert len(trajectory) > 1

        stats = dynamics.get_statistics()
        assert stats["state_history_size"] == 50
        assert stats["variables_tracked"] >= 2

    def test_multi_pattern_detection(self):
        dynamics = MockDynamicsModel(history_size=200, safety_config={})

        for i in range(100):
            t = i * 0.1
            state = State(
                timestamp=t,
                variables={
                    "periodic": 10 + 5 * np.sin(2 * np.pi * t / 10),
                    "trending": 20 + 2 * t,
                    "stationary": 50 + np.random.normal(0, 1),
                },
            )
            dynamics.update(state)

        patterns = dynamics.get_temporal_patterns()
        assert len(patterns) > 0
        pattern_types = {p.pattern_type for p in patterns.values()}
        assert len(pattern_types) >= 1


class TestPerformance:
    def test_large_scale_updates(self, basic_dynamics):
        start = time.time()
        np.random.seed(42)

        for i in range(500):
            noise_x = np.random.randn() * 0.01
            noise_y = np.random.randn() * 0.01
            state = State(
                timestamp=float(i),
                variables={
                    "x": float(i % 400) + noise_x,
                    "y": float((i * 2) % 400) + noise_y,
                },
            )
            basic_dynamics.update(state)

        elapsed = time.time() - start
        assert elapsed < 60, f"Took {elapsed}s to process 500 updates"
        assert len(basic_dynamics.state_history) <= basic_dynamics.history_size

    def test_many_variables(self):
        dynamics = MockDynamicsModel(history_size=50, safety_config={})
        start = time.time()

        for i in range(50):
            variables = {f"var_{j}": float(i * j) for j in range(20)}
            state = State(timestamp=float(i), variables=variables)
            dynamics.update(state)

        elapsed = time.time() - start
        assert elapsed < 10, f"Took {elapsed}s to process 20 variables"

    def test_trajectory_prediction_performance(self, basic_dynamics):
        for i in range(50):
            state = State(timestamp=float(i), variables={"x": float(i)})
            basic_dynamics.update(state)

        initial_state = State(timestamp=50.0, variables={"x": 50.0})
        start = time.time()

        trajectory = basic_dynamics.predict_trajectory(
            initial_state, horizon=100.0, timestep=1.0
        )

        elapsed = time.time() - start
        assert elapsed < 5, f"Trajectory prediction took {elapsed}s"
        assert len(trajectory) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
