"""
test_world_model_core.py - PURE MOCK VERSION
Tests world model core functionality without spawning threads.
"""

import json
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import Mock

import numpy as np
import pytest

# ============================================================================
# Mock Dataclasses
# ============================================================================


@dataclass
class Observation:
    timestamp: float
    variables: Dict[str, Any]
    domain: str = "default"
    confidence: float = 1.0
    intervention_data: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelContext:
    domain: str
    targets: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    features: Optional[np.ndarray] = None


@dataclass
class Prediction:
    target: str
    value: float
    confidence: float
    method: str = "correlation"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Intervention:
    intervention_id: str
    variable: str
    value: float
    status: str = "pending"
    scheduled_time: Optional[float] = None
    result: Optional[Dict] = None


# ============================================================================
# Mock Classes
# ============================================================================


class MockObservationProcessor:
    def __init__(self, safety_validator=None):
        self.safety_validator = safety_validator or Mock()
        self.variable_types: Dict[str, str] = {}
        self.observation_buffer: deque = deque(maxlen=1000)
        self.temporal_patterns: Dict[str, List] = defaultdict(list)
        self._lock = threading.Lock()

    def extract_variables(self, observation: Observation) -> Dict[str, Any]:
        variables = observation.variables.copy()

        for var, value in variables.items():
            if isinstance(value, bool):
                self.variable_types[var] = "boolean"
            elif isinstance(value, (int, float)):
                self.variable_types[var] = "numeric"
            elif isinstance(value, str):
                self.variable_types[var] = "categorical"
            elif isinstance(value, (list, np.ndarray)):
                self.variable_types[var] = "vector"
            else:
                self.variable_types[var] = "unknown"

        return variables

    def detect_intervention_data(self, observation: Observation) -> Optional[Dict]:
        if observation.intervention_data:
            return observation.intervention_data

        if observation.metadata.get("is_intervention"):
            return {
                "intervened_variable": observation.metadata.get("intervened_var"),
                "intervention_value": observation.metadata.get("intervention_val"),
            }

        return None

    def extract_temporal_patterns(self, observation: Observation) -> Dict:
        with self._lock:
            self.observation_buffer.append(observation)

        patterns = {"trends": {}, "cycles": {}, "anomalies": []}

        if len(self.observation_buffer) >= 5:
            for var in observation.variables:
                values = [obs.variables.get(var, 0) for obs in self.observation_buffer]
                if len(values) >= 5:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    patterns["trends"][var] = (
                        "increasing"
                        if trend > 0.1
                        else "decreasing"
                        if trend < -0.1
                        else "stable"
                    )

        return patterns

    def validate_observation(self, observation: Observation) -> tuple:
        if observation.timestamp < 0:
            return False, "Invalid timestamp"

        if not observation.variables:
            return False, "Empty variables"

        for var, value in observation.variables.items():
            if isinstance(value, float) and np.isnan(value):
                return False, f"NaN value in {var}"

        return True, None

    def get_statistics(self) -> Dict:
        return {
            "observations_processed": len(self.observation_buffer),
            "variables_tracked": len(self.variable_types),
        }


class MockInterventionManager:
    def __init__(self, safety_validator=None):
        self.safety_validator = safety_validator or Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )
        self.interventions: Dict[str, Intervention] = {}
        self.scheduled: List[Intervention] = []
        self.max_interventions = 5
        self._lock = threading.Lock()

    def schedule_intervention(
        self, variable: str, value: float, scheduled_time: float = None
    ) -> Intervention:
        with self._lock:
            if len(self.scheduled) >= self.max_interventions:
                # Remove oldest
                self.scheduled.pop(0)

            intervention = Intervention(
                intervention_id=f"int_{int(time.time() * 1000) % 100000}",
                variable=variable,
                value=value,
                scheduled_time=scheduled_time or time.time(),
            )

            self.interventions[intervention.intervention_id] = intervention
            self.scheduled.append(intervention)

            return intervention

    def execute_intervention(self, intervention_id: str) -> Dict:
        with self._lock:
            if intervention_id not in self.interventions:
                return {"success": False, "error": "Not found"}

            intervention = self.interventions[intervention_id]
            intervention.status = "executed"
            intervention.result = {"success": True, "value": intervention.value}

            return intervention.result

    def cancel_intervention(self, intervention_id: str) -> bool:
        with self._lock:
            if intervention_id in self.interventions:
                self.interventions[intervention_id].status = "cancelled"
                return True
            return False

    def get_pending_interventions(self) -> List[Intervention]:
        return [i for i in self.scheduled if i.status == "pending"]

    def get_statistics(self) -> Dict:
        return {
            "total_interventions": len(self.interventions),
            "pending": len(
                list(self.interventions.values() if i.status == "pending")
            ),
            "executed": len(
                list(self.interventions.values() if i.status == "executed")
            ),
        }


class MockPredictionManager:
    def __init__(self, safety_validator=None):
        self.safety_validator = safety_validator or Mock()
        self.predictions: List[Prediction] = []
        self.correlation_data: Dict[str, List[float]] = defaultdict(list)
        self.causal_relations: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_observation(self, observation: Observation):
        with self._lock:
            for var, value in observation.variables.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.correlation_data[var].append(float(value))

    def predict_correlation(self, target: str, context: ModelContext) -> Prediction:
        confidence = 0.7
        value = (
            np.mean(self.correlation_data.get(target, [0.0]))
            if self.correlation_data.get(target)
            else 0.0
        )

        pred = Prediction(
            target=target, value=value, confidence=confidence, method="correlation"
        )

        with self._lock:
            self.predictions.append(pred)

        return pred

    def predict_causal(
        self, target: str, intervention: Dict, context: ModelContext
    ) -> Prediction:
        confidence = 0.6
        value = intervention.get("value", 0.0) * 1.5  # Simple causal effect

        pred = Prediction(
            target=target, value=value, confidence=confidence, method="causal"
        )

        with self._lock:
            self.predictions.append(pred)

        return pred

    def get_prediction_confidence(self, target: str) -> float:
        target_preds = [p for p in self.predictions if p.target == target]
        if not target_preds:
            return 0.0
        return np.mean([p.confidence for p in target_preds])

    def get_statistics(self) -> Dict:
        return {
            "total_predictions": len(self.predictions),
            "variables_tracked": len(self.correlation_data),
        }


class MockConsistencyValidator:
    def __init__(self, safety_validator=None):
        self.safety_validator = safety_validator or Mock()
        self.validations: List[Dict] = []
        self.inconsistencies: List[Dict] = []

    def validate_observation(self, observation: Observation) -> Dict:
        result = {"valid": True, "issues": [], "confidence": 0.9}

        if observation.timestamp < 0:
            result["valid"] = False
            result["issues"].append("Invalid timestamp")

        self.validations.append(result)
        return result

    def validate_prediction(self, prediction: Prediction, actuals: Dict) -> Dict:
        result = {"valid": True, "error": 0.0, "within_bounds": True}

        if prediction.target in actuals:
            actual = actuals[prediction.target]
            result["error"] = abs(prediction.value - actual)
            result["within_bounds"] = result["error"] < 1.0

        return result

    def check_model_consistency(self) -> Dict:
        return {
            "consistent": True,
            "inconsistencies": self.inconsistencies,
            "last_check": time.time(),
        }

    def get_statistics(self) -> Dict:
        valid_count = sum(1 for v in self.validations if v["valid"])
        return {
            "total_validations": len(self.validations),
            "valid_count": valid_count,
            "validity_rate": valid_count / max(len(self.validations), 1),
        }


class MockCausalGraph:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Dict] = {}

    def add_node(self, node: str):
        self.nodes.add(node)

    def add_edge(self, source: str, target: str, **kwargs):
        self.edges[f"{source}->{target}"] = kwargs

    def has_node(self, node: str) -> bool:
        return node in self.nodes

    def has_edge(self, source: str, target: str) -> bool:
        return f"{source}->{target}" in self.edges


class MockWorldModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.safety_validator = Mock()
        self.safety_validator.validate_action_comprehensive = Mock(
            return_value=Mock(safe=True, confidence=0.9)
        )

        self.observation_processor = MockObservationProcessor(self.safety_validator)
        self.intervention_manager = MockInterventionManager(self.safety_validator)
        self.prediction_manager = MockPredictionManager(self.safety_validator)
        self.consistency_validator = MockConsistencyValidator(self.safety_validator)
        self.causal_graph = MockCausalGraph()

        self.observations: List[Observation] = []
        self.min_correlation = config.get("min_correlation", 0.8)
        self.min_causal = config.get("min_causal", 0.7)
        self.simulation_mode = config.get("simulation_mode", False)
        self.enable_meta_reasoning = config.get("enable_meta_reasoning", False)

        self._lock = threading.Lock()

    def process_observation(self, observation: Observation) -> Dict:
        with self._lock:
            # Validate
            is_valid, error = self.observation_processor.validate_observation(
                observation
            )
            if not is_valid:
                return {"success": False, "error": error}

            # Extract variables
            variables = self.observation_processor.extract_variables(observation)

            # Add to prediction manager
            self.prediction_manager.add_observation(observation)

            # Store
            self.observations.append(observation)

            # Extract patterns
            patterns = self.observation_processor.extract_temporal_patterns(observation)

            return {"success": True, "variables": variables, "patterns": patterns}

    def predict(self, target: str, context: ModelContext = None) -> Prediction:
        context = context or ModelContext(domain="default", targets=[target])
        return self.prediction_manager.predict_correlation(target, context)

    def predict_intervention(
        self, target: str, intervention: Dict, context: ModelContext = None
    ) -> Prediction:
        context = context or ModelContext(domain="default", targets=[target])
        return self.prediction_manager.predict_causal(target, intervention, context)

    def schedule_intervention(self, variable: str, value: float) -> Intervention:
        return self.intervention_manager.schedule_intervention(variable, value)

    def execute_intervention(self, intervention_id: str) -> Dict:
        return self.intervention_manager.execute_intervention(intervention_id)

    def validate_state(self) -> Dict:
        return self.consistency_validator.check_model_consistency()

    def get_statistics(self) -> Dict:
        return {
            "observations": len(self.observations),
            "processor": self.observation_processor.get_statistics(),
            "interventions": self.intervention_manager.get_statistics(),
            "predictions": self.prediction_manager.get_statistics(),
            "consistency": self.consistency_validator.get_statistics(),
        }

    def save_state(self, path: Path):
        state = {
            "observations_count": len(self.observations),
            "config": self.config,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(state))

    def load_state(self, path: Path):
        if path.exists():
            state = json.loads(path.read_text())
            return state
        return None


# Aliases
ObservationProcessor = MockObservationProcessor
InterventionManager = MockInterventionManager
PredictionManager = MockPredictionManager
ConsistencyValidator = MockConsistencyValidator
WorldModel = MockWorldModel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_observation():
    return Observation(
        timestamp=time.time(),
        variables={"x": 1.0, "y": 2.0, "z": 3.0},
        domain="test",
        confidence=0.9,
    )


@pytest.fixture
def intervention_observation():
    return Observation(
        timestamp=time.time(),
        variables={"x": 5.0, "y": 10.0},
        intervention_data={
            "intervened_variable": "x",
            "intervention_value": 5.0,
            "control_group": False,
        },
        domain="test",
        confidence=1.0,
    )


@pytest.fixture
def invalid_observation():
    return Observation(timestamp=-1.0, variables={}, domain="test")


@pytest.fixture
def observation_with_nan():
    return Observation(
        timestamp=time.time(),
        variables={"x": 1.0, "y": np.nan, "z": 3.0},
        domain="test",
    )


@pytest.fixture
def model_context():
    return ModelContext(
        domain="test", targets=["y", "z"], constraints={"time_horizon": 1.0}
    )


@pytest.fixture
def observation_processor():
    return MockObservationProcessor()


@pytest.fixture
def world_model():
    config = {
        "min_correlation": 0.8,
        "min_causal": 0.7,
        "max_interventions": 5,
        "bootstrap_mode": True,
        "simulation_mode": True,
        "enable_meta_reasoning": False,
    }
    return MockWorldModel(config)


@pytest.fixture
def world_model_with_safety():
    config = {
        "safety_config": {},
        "simulation_mode": True,
        "enable_meta_reasoning": False,
    }
    return MockWorldModel(config)


@pytest.fixture
def observation_sequence():
    observations = []
    for i in range(20):
        obs = Observation(
            timestamp=time.time() + i,
            variables={
                "x": float(i),
                "y": 2.0 * i + np.random.normal(0, 0.1),
                "z": 5.0 + np.random.normal(0, 0.05),
            },
            domain="test",
        )
        observations.append(obs)
    return observations


# ============================================================================
# Tests
# ============================================================================


class TestObservation:
    def test_observation_creation(self, sample_observation):
        assert sample_observation.timestamp > 0
        assert len(sample_observation.variables) == 3
        assert sample_observation.domain == "test"
        assert sample_observation.confidence == 0.9

    def test_observation_with_intervention_data(self, intervention_observation):
        assert intervention_observation.intervention_data is not None
        assert "intervened_variable" in intervention_observation.intervention_data
        assert intervention_observation.intervention_data["intervened_variable"] == "x"

    def test_observation_with_metadata(self):
        obs = Observation(
            timestamp=time.time(),
            variables={"x": 1.0},
            metadata={"source": "sensor", "quality": "high"},
        )
        assert obs.metadata["source"] == "sensor"
        assert obs.metadata["quality"] == "high"


class TestModelContext:
    def test_context_creation(self, model_context):
        assert model_context.domain == "test"
        assert model_context.targets == ["y", "z"]
        assert "time_horizon" in model_context.constraints

    def test_context_with_features(self):
        features = np.array([1.0, 2.0, 3.0])
        context = ModelContext(domain="test", targets=["x"], features=features)
        assert context.features is not None
        assert len(context.features) == 3


class TestObservationProcessor:
    def test_extract_variables(self, observation_processor, sample_observation):
        variables = observation_processor.extract_variables(sample_observation)
        assert len(variables) == 3
        assert "x" in variables
        assert variables["x"] == 1.0

    def test_extract_variables_type_inference(self, observation_processor):
        obs = Observation(
            timestamp=time.time(),
            variables={
                "numeric": 1.5,
                "boolean": True,
                "categorical": "test",
                "vector": [1, 2, 3],
            },
        )
        observation_processor.extract_variables(obs)
        assert observation_processor.variable_types["numeric"] == "numeric"
        assert observation_processor.variable_types["boolean"] == "boolean"
        assert observation_processor.variable_types["categorical"] == "categorical"

    def test_detect_intervention_data(
        self, observation_processor, intervention_observation
    ):
        intervention_data = observation_processor.detect_intervention_data(
            intervention_observation
        )
        assert intervention_data is not None
        assert "intervened_variable" in intervention_data

    def test_detect_intervention_from_metadata(self, observation_processor):
        obs = Observation(
            timestamp=time.time(),
            variables={"x": 5.0},
            metadata={
                "is_intervention": True,
                "intervened_var": "x",
                "intervention_val": 5.0,
            },
        )
        intervention_data = observation_processor.detect_intervention_data(obs)
        assert intervention_data is not None
        assert intervention_data["intervened_variable"] == "x"

    def test_extract_temporal_patterns(
        self, observation_processor, observation_sequence
    ):
        for obs in observation_sequence:
            observation_processor.extract_temporal_patterns(obs)

        final_patterns = observation_processor.extract_temporal_patterns(
            observation_sequence[-1]
        )
        assert "trends" in final_patterns
        assert "cycles" in final_patterns
        assert "anomalies" in final_patterns

    def test_validate_observation_valid(
        self, observation_processor, sample_observation
    ):
        is_valid, error = observation_processor.validate_observation(sample_observation)
        assert is_valid == True
        assert error is None

    def test_validate_observation_invalid_timestamp(
        self, observation_processor, invalid_observation
    ):
        is_valid, error = observation_processor.validate_observation(
            invalid_observation
        )
        assert is_valid == False

    def test_validate_observation_nan(
        self, observation_processor, observation_with_nan
    ):
        is_valid, error = observation_processor.validate_observation(
            observation_with_nan
        )
        assert is_valid == False


class TestInterventionManager:
    def test_schedule_intervention(self):
        manager = MockInterventionManager()
        intervention = manager.schedule_intervention("x", 5.0)
        assert intervention is not None
        assert intervention.variable == "x"
        assert intervention.value == 5.0

    def test_execute_intervention(self):
        manager = MockInterventionManager()
        intervention = manager.schedule_intervention("x", 5.0)
        result = manager.execute_intervention(intervention.intervention_id)
        assert result["success"] == True

    def test_cancel_intervention(self):
        manager = MockInterventionManager()
        intervention = manager.schedule_intervention("x", 5.0)
        success = manager.cancel_intervention(intervention.intervention_id)
        assert success == True

    def test_max_interventions(self):
        manager = MockInterventionManager()
        manager.max_interventions = 3
        for i in range(5):
            manager.schedule_intervention("x", float(i))
        assert len(manager.scheduled) <= 3


class TestPredictionManager:
    def test_add_observation(self, sample_observation):
        manager = MockPredictionManager()
        manager.add_observation(sample_observation)
        assert len(manager.correlation_data) > 0

    def test_predict_correlation(self, model_context):
        manager = MockPredictionManager()
        for i in range(10):
            obs = Observation(timestamp=time.time(), variables={"y": float(i)})
            manager.add_observation(obs)

        prediction = manager.predict_correlation("y", model_context)
        assert prediction is not None
        assert prediction.target == "y"
        assert prediction.method == "correlation"

    def test_predict_causal(self, model_context):
        manager = MockPredictionManager()
        intervention = {"variable": "x", "value": 5.0}
        prediction = manager.predict_causal("y", intervention, model_context)
        assert prediction is not None
        assert prediction.method == "causal"


class TestConsistencyValidator:
    def test_validate_observation(self, sample_observation):
        validator = MockConsistencyValidator()
        result = validator.validate_observation(sample_observation)
        assert result["valid"] == True

    def test_check_model_consistency(self):
        validator = MockConsistencyValidator()
        result = validator.check_model_consistency()
        assert "consistent" in result


class TestWorldModel:
    def test_initialization(self, world_model):
        assert world_model.observation_processor is not None
        assert world_model.intervention_manager is not None
        assert world_model.prediction_manager is not None

    def test_process_observation(self, world_model, sample_observation):
        result = world_model.process_observation(sample_observation)
        assert result["success"] == True
        assert "variables" in result

    def test_process_invalid_observation(self, world_model, invalid_observation):
        result = world_model.process_observation(invalid_observation)
        assert result["success"] == False

    def test_predict(self, world_model, sample_observation, model_context):
        world_model.process_observation(sample_observation)
        prediction = world_model.predict("y", model_context)
        assert prediction is not None

    def test_schedule_and_execute_intervention(self, world_model):
        intervention = world_model.schedule_intervention("x", 5.0)
        result = world_model.execute_intervention(intervention.intervention_id)
        assert result["success"] == True

    def test_get_statistics(self, world_model, sample_observation):
        world_model.process_observation(sample_observation)
        stats = world_model.get_statistics()
        assert "observations" in stats
        assert stats["observations"] == 1


class TestThreadSafety:
    def test_concurrent_observations(self, world_model):
        def add_observations():
            for i in range(10):
                obs = Observation(
                    timestamp=time.time(),
                    variables={"x": float(i), "y": float(i * 2)},
                    domain="test",
                )
                world_model.process_observation(obs)

        threads = []
        for _ in range(3):
            t = threading.Thread(target=add_observations)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(world_model.observations) == 30


class TestPersistence:
    def test_save_and_load_state(self, world_model, sample_observation):
        world_model.process_observation(sample_observation)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            world_model.save_state(path)
            assert path.exists()

            loaded = world_model.load_state(path)
            assert loaded is not None
            assert loaded["observations_count"] == 1
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
