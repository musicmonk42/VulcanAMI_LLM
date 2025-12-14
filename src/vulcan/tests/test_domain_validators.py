"""
Comprehensive test suite for domain-specific safety validators.
Tests validators for causal, prediction, optimization, data processing, and model inference domains.
"""

import numpy as np
import pytest

from vulcan.safety.domain_validators import (
    CausalSafetyValidator,
    DataProcessingSafetyValidator,
    DomainValidator,
    DomainValidatorRegistry,
    ModelInferenceSafetyValidator,
    OptimizationSafetyValidator,
    PredictionSafetyValidator,
    ValidationResult,
    validator_registry,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def causal_validator():
    """Create a CausalSafetyValidator instance."""
    return CausalSafetyValidator()


@pytest.fixture
def prediction_validator():
    """Create a PredictionSafetyValidator instance."""
    safe_regions = {"temperature": (0.0, 100.0), "pressure": (0.0, 1000.0)}
    return PredictionSafetyValidator(safe_regions=safe_regions)


@pytest.fixture
def optimization_validator():
    """Create an OptimizationSafetyValidator instance."""
    return OptimizationSafetyValidator()


@pytest.fixture
def data_validator():
    """Create a DataProcessingSafetyValidator instance."""
    return DataProcessingSafetyValidator()


@pytest.fixture
def model_validator():
    """Create a ModelInferenceSafetyValidator instance."""
    return ModelInferenceSafetyValidator()


@pytest.fixture
def registry():
    """Create a DomainValidatorRegistry instance."""
    return DomainValidatorRegistry()


# ============================================================================
# VALIDATION RESULT TESTS
# ============================================================================


class TestValidationResult:
    """Test ValidationResult class."""

    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(safe=True, reason="All checks passed")

        assert result.safe is True
        assert result.reason == "All checks passed"
        assert result.corrected_values == {}
        assert result.severity == "medium"
        assert result.timestamp > 0

    def test_initialization_with_corrections(self):
        """Test ValidationResult with corrected values."""
        corrected = {"value": 10.0}
        result = ValidationResult(
            safe=False,
            reason="Value too high",
            corrected_values=corrected,
            severity="high",
        )

        assert result.safe is False
        assert result.corrected_values == corrected
        assert result.severity == "high"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(safe=True, reason="OK", metadata={"test": "data"})

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["safe"] is True
        assert d["reason"] == "OK"
        assert "timestamp" in d
        assert d["metadata"]["test"] == "data"


# ============================================================================
# DOMAIN VALIDATOR BASE TESTS
# ============================================================================


class TestDomainValidator:
    """Test DomainValidator base class."""

    def test_base_validator_not_implemented(self):
        """Test that base validate() raises NotImplementedError."""
        validator = DomainValidator()

        with pytest.raises(NotImplementedError):
            validator.validate({})

    def test_record_validation(self):
        """Test validation recording."""
        validator = DomainValidator()
        result = ValidationResult(safe=True, reason="Test")

        validator._record_validation(result, {"test": "data"})

        assert len(validator.validation_history) == 1

    def test_get_stats_empty(self):
        """Test getting stats with no history."""
        validator = DomainValidator()
        stats = validator.get_stats()

        assert stats["total_validations"] == 0

    def test_get_stats_with_history(self):
        """Test getting stats with validation history."""
        validator = DomainValidator()

        # Record some validations
        for i in range(10):
            result = ValidationResult(safe=i % 2 == 0, reason=f"Test {i}")
            validator._record_validation(result, {"i": i})

        stats = validator.get_stats()

        assert stats["total_validations"] == 10
        assert stats["safe_count"] == 5
        assert stats["unsafe_count"] == 5
        assert stats["safety_rate"] == 0.5


# ============================================================================
# CAUSAL SAFETY VALIDATOR TESTS
# ============================================================================


class TestCausalSafetyValidator:
    """Test CausalSafetyValidator class."""

    def test_initialization(self, causal_validator):
        """Test validator initialization."""
        assert causal_validator.max_causal_strength > 0
        assert len(causal_validator.unsafe_patterns) > 0

    def test_validate_edge_safe(self, causal_validator):
        """Test validating a safe causal edge."""
        result = causal_validator.validate_causal_edge(
            cause="temperature", effect="pressure", strength=2.5
        )

        assert result.safe is True
        assert "temperature" in result.metadata["cause"]

    def test_validate_edge_excessive_strength(self, causal_validator):
        """Test edge with excessive causal strength."""
        result = causal_validator.validate_causal_edge(
            cause="x",
            effect="y",
            strength=100.0,  # Too large
        )

        assert result.safe is False
        assert "too large" in result.reason.lower()
        assert "strength" in result.corrected_values

    def test_validate_edge_nan(self, causal_validator):
        """Test edge with NaN strength."""
        result = causal_validator.validate_causal_edge(
            cause="x", effect="y", strength=np.nan
        )

        assert result.safe is False
        assert "nan" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_edge_inf(self, causal_validator):
        """Test edge with Inf strength."""
        result = causal_validator.validate_causal_edge(
            cause="x", effect="y", strength=np.inf
        )

        assert result.safe is False
        assert result.severity == "critical"

    def test_validate_edge_unsafe_pattern(self, causal_validator):
        """Test edge matching unsafe pattern."""
        result = causal_validator.validate_causal_edge(
            cause="harm", effect="increase", strength=2.0
        )

        assert result.safe is False
        assert "unsafe pattern" in result.reason.lower()

    def test_validate_edge_self_loop(self, causal_validator):
        """Test edge with self-loop."""
        result = causal_validator.validate_causal_edge(
            cause="variable", effect="variable", strength=1.0
        )

        assert result.safe is False
        assert "self-loop" in result.reason.lower()

    def test_validate_edge_empty_names(self, causal_validator):
        """Test edge with empty names."""
        result = causal_validator.validate_causal_edge(
            cause="", effect="y", strength=1.0
        )

        assert result.safe is False
        assert "empty" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_edge_small_strength(self, causal_validator):
        """Test edge with very small strength (numerical instability)."""
        result = causal_validator.validate_causal_edge(
            cause="x", effect="y", strength=1e-15
        )

        assert result.safe is False
        assert "too small" in result.reason.lower()

    def test_validate_path_safe(self, causal_validator):
        """Test validating a safe causal path."""
        result = causal_validator.validate_causal_path(
            nodes=["A", "B", "C"], strengths=[2.0, 1.5]
        )

        assert result.safe is True
        assert result.metadata["path_length"] == 3

    def test_validate_path_too_long(self, causal_validator):
        """Test path that's too long."""
        nodes = [f"node_{i}" for i in range(25)]
        strengths = [1.0] * 24

        result = causal_validator.validate_causal_path(nodes, strengths)

        assert result.safe is False
        assert "too long" in result.reason.lower()

    def test_validate_path_with_cycle(self, causal_validator):
        """Test path containing a cycle."""
        result = causal_validator.validate_causal_path(
            nodes=["A", "B", "C", "A"], strengths=[1.0, 1.0, 1.0]
        )

        assert result.safe is False
        assert "cycle" in result.reason.lower()

    def test_validate_path_mismatch(self, causal_validator):
        """Test path with mismatched nodes and strengths."""
        result = causal_validator.validate_causal_path(
            nodes=["A", "B", "C"],
            strengths=[1.0],  # Should be 2 strengths for 3 nodes
        )

        assert result.safe is False
        assert "mismatch" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_path_excessive_amplification(self, causal_validator):
        """Test path with excessive total amplification."""
        result = causal_validator.validate_causal_path(
            nodes=["A", "B", "C", "D"],
            strengths=[5.0, 5.0, 5.0],  # Total = 125x
        )

        assert result.safe is False
        assert "amplification" in result.reason.lower()
        assert "safe_amplification" in result.corrected_values

    def test_validate_path_nan_strength(self, causal_validator):
        """Test path with NaN strength."""
        result = causal_validator.validate_causal_path(
            nodes=["A", "B", "C"], strengths=[1.0, np.nan]
        )

        assert result.safe is False
        assert "nan" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_path_empty(self, causal_validator):
        """Test empty path."""
        result = causal_validator.validate_causal_path(nodes=[], strengths=[])

        assert result.safe is False
        assert "empty" in result.reason.lower()

    def test_validate_graph_safe(self, causal_validator):
        """Test validating a safe causal graph."""
        adjacency = {
            "A": [("B", 2.0), ("C", 1.5)],
            "B": [("D", 1.0)],
            "C": [("D", 0.5)],
        }

        result = causal_validator.validate_causal_graph(adjacency)

        assert result.safe is True
        assert result.metadata["node_count"] == 3
        assert result.metadata["edge_count"] == 4

    def test_validate_graph_with_cycle(self, causal_validator):
        """Test graph with cycles."""
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [("A", 1.0)],  # Cycle back to A
        }

        result = causal_validator.validate_causal_graph(adjacency)

        assert result.safe is False
        assert "cycle" in result.reason.lower()

    def test_validate_graph_too_large(self, causal_validator):
        """Test graph that's too large."""
        # Create large graph
        adjacency = {}
        for i in range(100):
            adjacency[f"node_{i}"] = [
                (f"node_{j}", 1.0) for j in range(i + 1, min(i + 20, 100))
            ]

        result = causal_validator.validate_causal_graph(adjacency)

        # May fail due to size or edge violations
        if not result.safe:
            assert (
                "too large" in result.reason.lower()
                or "unsafe edges" in result.reason.lower()
            )


# ============================================================================
# PREDICTION SAFETY VALIDATOR TESTS
# ============================================================================


class TestPredictionSafetyValidator:
    """Test PredictionSafetyValidator class."""

    def test_initialization(self, prediction_validator):
        """Test validator initialization."""
        assert "temperature" in prediction_validator.safe_regions
        assert prediction_validator.max_magnitude > 0

    def test_validate_prediction_safe(self, prediction_validator):
        """Test validating a safe prediction."""
        result = prediction_validator.validate_prediction(
            expected=50.0, lower=45.0, upper=55.0, variable="temperature"
        )

        assert result.safe is True
        assert result.metadata["variable"] == "temperature"

    def test_validate_prediction_nan(self, prediction_validator):
        """Test prediction with NaN."""
        result = prediction_validator.validate_prediction(
            expected=np.nan, lower=0.0, upper=100.0, variable="test"
        )

        assert result.safe is False
        assert "nan" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_prediction_inf(self, prediction_validator):
        """Test prediction with Inf."""
        result = prediction_validator.validate_prediction(
            expected=100.0, lower=np.inf, upper=200.0, variable="test"
        )

        assert result.safe is False
        assert "nan" in result.reason.lower() or "inf" in result.reason.lower()

    def test_validate_prediction_excessive_magnitude(self, prediction_validator):
        """Test prediction with excessive magnitude."""
        result = prediction_validator.validate_prediction(
            expected=10000.0, lower=9000.0, upper=11000.0, variable="test"
        )

        assert result.safe is False
        assert "too large" in result.reason.lower()
        assert "safe_expected" in result.corrected_values

    def test_validate_prediction_invalid_bounds(self, prediction_validator):
        """Test prediction with invalid bounds (lower > upper)."""
        result = prediction_validator.validate_prediction(
            expected=50.0, lower=100.0, upper=0.0, variable="test"
        )

        assert result.safe is False
        assert "invalid bounds" in result.reason.lower()
        assert "safe_lower" in result.corrected_values
        assert "safe_upper" in result.corrected_values

    def test_validate_prediction_excessive_uncertainty(self, prediction_validator):
        """Test prediction with excessive uncertainty."""
        result = prediction_validator.validate_prediction(
            expected=10.0,
            lower=0.0,
            upper=100.0,  # Range of 100 for expected of 10
            variable="test",
        )

        assert result.safe is False
        assert "uncertain" in result.reason.lower()

    def test_validate_prediction_outside_bounds(self, prediction_validator):
        """Test prediction where expected is outside bounds."""
        result = prediction_validator.validate_prediction(
            expected=50.0,
            lower=0.0,
            upper=40.0,  # Expected > upper
            variable="test",
        )

        assert result.safe is False
        assert "not within bounds" in result.reason.lower()
        assert "safe_expected" in result.corrected_values

    def test_validate_prediction_outside_safe_region(self, prediction_validator):
        """Test prediction outside safe region."""
        result = prediction_validator.validate_prediction(
            expected=150.0,  # Outside safe region [0, 100]
            lower=140.0,
            upper=160.0,
            variable="temperature",
        )

        assert result.safe is False
        assert "outside safe region" in result.reason.lower()
        assert "safe_expected" in result.corrected_values

    def test_validate_prediction_bounds_extend_outside_safe_region(
        self, prediction_validator
    ):
        """Test prediction bounds extending outside safe region."""
        result = prediction_validator.validate_prediction(
            expected=50.0,
            lower=-10.0,  # Below safe minimum
            upper=110.0,  # Above safe maximum
            variable="temperature",
        )

        assert result.safe is False
        assert "extend outside safe region" in result.reason.lower()

    def test_validate_prediction_batch_safe(self, prediction_validator):
        """Test validating a batch of safe predictions."""
        predictions = [
            {"expected": 50.0, "lower": 45.0, "upper": 55.0, "variable": "temperature"},
            {"expected": 500.0, "lower": 450.0, "upper": 550.0, "variable": "pressure"},
        ]

        result = prediction_validator.validate_prediction_batch(predictions)

        assert result.safe is True
        assert result.metadata["total"] == 2
        assert result.metadata["unsafe"] == 0

    def test_validate_prediction_batch_mixed(self, prediction_validator):
        """Test batch with mixed safe/unsafe predictions."""
        predictions = [
            {"expected": 50.0, "lower": 45.0, "upper": 55.0, "variable": "temperature"},
            {"expected": np.nan, "lower": 0.0, "upper": 100.0, "variable": "test"},
            {
                "expected": 150.0,
                "lower": 140.0,
                "upper": 160.0,
                "variable": "temperature",
            },
        ]

        result = prediction_validator.validate_prediction_batch(predictions)

        assert result.safe is False
        assert result.metadata["total"] == 3
        assert result.metadata["unsafe"] == 2


# ============================================================================
# OPTIMIZATION SAFETY VALIDATOR TESTS
# ============================================================================


class TestOptimizationSafetyValidator:
    """Test OptimizationSafetyValidator class."""

    def test_initialization(self, optimization_validator):
        """Test validator initialization."""
        assert optimization_validator.max_iterations > 0
        assert optimization_validator.min_improvement > 0

    def test_validate_params_safe(self, optimization_validator):
        """Test validating safe optimization parameters."""
        params = {
            "max_iterations": 1000,
            "learning_rate": 0.01,
            "tolerance": 1e-6,
            "bounds": {"x": (0, 10), "y": (-5, 5)},
        }

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is True

    def test_validate_params_too_many_iterations(self, optimization_validator):
        """Test parameters with too many iterations."""
        params = {"max_iterations": 100000, "tolerance": 1e-6}

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "too many iterations" in result.reason.lower()
        assert "max_iterations" in result.corrected_values

    def test_validate_params_invalid_learning_rate(self, optimization_validator):
        """Test parameters with invalid learning rate."""
        params = {
            "max_iterations": 100,
            "learning_rate": 1.5,  # > 1.0
            "tolerance": 1e-6,
        }

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "learning rate" in result.reason.lower()

    def test_validate_params_negative_learning_rate(self, optimization_validator):
        """Test parameters with negative learning rate."""
        params = {"max_iterations": 100, "learning_rate": -0.01, "tolerance": 1e-6}

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "learning rate" in result.reason.lower()

    def test_validate_params_tolerance_too_small(self, optimization_validator):
        """Test parameters with tolerance too small."""
        params = {"max_iterations": 100, "tolerance": 1e-15}

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "tolerance too small" in result.reason.lower()

    def test_validate_params_invalid_bounds(self, optimization_validator):
        """Test parameters with invalid bounds (lower >= upper)."""
        params = {
            "max_iterations": 100,
            "tolerance": 1e-6,
            "bounds": {"x": (10, 5)},  # Invalid: lower > upper
        }

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "invalid bounds" in result.reason.lower()

    def test_validate_params_infinite_bounds(self, optimization_validator):
        """Test parameters with infinite bounds."""
        params = {
            "max_iterations": 100,
            "tolerance": 1e-6,
            "bounds": {"x": (-np.inf, np.inf)},
        }

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "infinite bounds" in result.reason.lower()

    def test_validate_params_missing_required(self, optimization_validator):
        """Test parameters missing required fields."""
        params = {"learning_rate": 0.01}

        result = optimization_validator.validate_optimization_params(params)

        assert result.safe is False
        assert "missing required" in result.reason.lower()

    def test_validate_result_success(self, optimization_validator):
        """Test validating successful optimization result."""
        result_obj = {
            "success": True,
            "objective_value": 10.5,
            "solution": {"x": 5.0, "y": 3.2},
            "iterations": 50,
        }

        result = optimization_validator.validate_optimization_result(result_obj)

        assert result.safe is True

    def test_validate_result_failed(self, optimization_validator):
        """Test validating failed optimization."""
        result_obj = {
            "success": False,
            "message": "Convergence failed",
            "objective_value": 100.0,
            "solution": {},
            "iterations": 1000,
        }

        result = optimization_validator.validate_optimization_result(result_obj)

        assert result.safe is False
        assert "failed" in result.reason.lower()

    def test_validate_result_excessive_objective(self, optimization_validator):
        """Test result with excessive objective value."""
        result_obj = {
            "success": True,
            "objective_value": 1e15,
            "solution": {"x": 5.0},
            "iterations": 50,
        }

        result = optimization_validator.validate_optimization_result(result_obj)

        assert result.safe is False
        assert "too large" in result.reason.lower()

    def test_validate_result_nan_objective(self, optimization_validator):
        """Test result with NaN objective."""
        result_obj = {
            "success": True,
            "objective_value": np.nan,
            "solution": {"x": 5.0},
            "iterations": 50,
        }

        result = optimization_validator.validate_optimization_result(result_obj)

        assert result.safe is False
        assert "nan" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_result_nan_solution(self, optimization_validator):
        """Test result with NaN in solution."""
        result_obj = {
            "success": True,
            "objective_value": 10.0,
            "solution": {"x": np.nan, "y": 5.0},
            "iterations": 50,
        }

        result = optimization_validator.validate_optimization_result(result_obj)

        assert result.safe is False
        assert "nan" in result.reason.lower()


# ============================================================================
# DATA PROCESSING SAFETY VALIDATOR TESTS
# ============================================================================


class TestDataProcessingSafetyValidator:
    """Test DataProcessingSafetyValidator class."""

    def test_initialization(self, data_validator):
        """Test validator initialization."""
        assert data_validator.max_data_size_mb > 0
        assert data_validator.max_rows > 0

    def test_validate_dataframe_safe(self, data_validator):
        """Test validating a safe dataframe."""
        df_info = {
            "rows": 1000,
            "columns": 10,
            "memory_mb": 5.0,
            "missing_ratio": 0.05,
            "dtypes": {"col1": "int64", "col2": "float64"},
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is True

    def test_validate_dataframe_too_many_rows(self, data_validator):
        """Test dataframe with too many rows."""
        df_info = {
            "rows": 20000000,
            "columns": 10,
            "memory_mb": 100.0,
            "missing_ratio": 0.0,
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "too many rows" in result.reason.lower()

    def test_validate_dataframe_too_many_columns(self, data_validator):
        """Test dataframe with too many columns."""
        df_info = {
            "rows": 1000,
            "columns": 15000,
            "memory_mb": 50.0,
            "missing_ratio": 0.0,
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "too many columns" in result.reason.lower()

    def test_validate_dataframe_too_large(self, data_validator):
        """Test dataframe that's too large."""
        df_info = {
            "rows": 1000,
            "columns": 10,
            "memory_mb": 2000.0,  # Too large
            "missing_ratio": 0.0,
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "too large" in result.reason.lower()

    def test_validate_dataframe_excessive_missing(self, data_validator):
        """Test dataframe with too much missing data."""
        df_info = {
            "rows": 1000,
            "columns": 10,
            "memory_mb": 5.0,
            "missing_ratio": 0.7,  # 70% missing
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "missing data" in result.reason.lower()

    def test_validate_dataframe_large_object_columns(self, data_validator):
        """Test dataframe with large object columns."""
        df_info = {
            "rows": 200000,
            "columns": 5,
            "memory_mb": 50.0,
            "missing_ratio": 0.0,
            "dtypes": {"col1": "object", "col2": "int64"},
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "object column" in result.reason.lower()

    def test_validate_dataframe_duplicate_columns(self, data_validator):
        """Test dataframe with duplicate columns."""
        df_info = {
            "rows": 1000,
            "columns": 10,
            "memory_mb": 5.0,
            "missing_ratio": 0.0,
            "duplicate_columns": ["col1", "col2"],
        }

        result = data_validator.validate_dataframe(df_info)

        assert result.safe is False
        assert "duplicate columns" in result.reason.lower()

    def test_validate_transformation_safe(self, data_validator):
        """Test validating a safe transformation."""
        transform_info = {
            "type": "filter",
            "input_rows": 1000,
            "output_rows": 800,
            "total_columns": 10,
            "columns_dropped": [],
        }

        result = data_validator.validate_data_transformation(transform_info)

        assert result.safe is True

    def test_validate_transformation_unsafe_type(self, data_validator):
        """Test transformation with unsafe type."""
        transform_info = {
            "type": "eval",  # Unsafe
            "input_rows": 1000,
            "output_rows": 1000,
        }

        result = data_validator.validate_data_transformation(transform_info)

        assert result.safe is False
        assert "unsafe transformation" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_transformation_eliminates_data(self, data_validator):
        """Test transformation that eliminates all data."""
        transform_info = {"type": "filter", "input_rows": 1000, "output_rows": 0}

        result = data_validator.validate_data_transformation(transform_info)

        assert result.safe is False
        assert "eliminated all data" in result.reason.lower()

    def test_validate_transformation_dramatic_expansion(self, data_validator):
        """Test transformation that dramatically expands data."""
        transform_info = {
            "type": "explode",
            "input_rows": 1000,
            "output_rows": 15000,  # 15x expansion
        }

        result = data_validator.validate_data_transformation(transform_info)

        assert result.safe is False
        assert "dramatically expanded" in result.reason.lower()

    def test_validate_transformation_drops_many_columns(self, data_validator):
        """Test transformation that drops many columns."""
        transform_info = {
            "type": "select",
            "input_rows": 1000,
            "output_rows": 1000,
            "total_columns": 20,
            "columns_dropped": [f"col_{i}" for i in range(15)],  # Dropped 15/20
        }

        result = data_validator.validate_data_transformation(transform_info)

        assert result.safe is False
        assert "dropped" in result.reason.lower()


# ============================================================================
# MODEL INFERENCE SAFETY VALIDATOR TESTS
# ============================================================================


class TestModelInferenceSafetyValidator:
    """Test ModelInferenceSafetyValidator class."""

    def test_initialization(self, model_validator):
        """Test validator initialization."""
        assert model_validator.max_batch_size > 0
        assert model_validator.min_confidence > 0

    def test_validate_input_safe(self, model_validator):
        """Test validating safe inference input."""
        input_data = {
            "batch_size": 32,
            "shape": (32, 224, 224, 3),
            "contains_nan": False,
            "contains_inf": False,
            "min_value": -1.0,
            "max_value": 1.0,
        }

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is True

    def test_validate_input_excessive_batch_size(self, model_validator):
        """Test input with excessive batch size."""
        input_data = {"batch_size": 100000, "shape": (100000, 224, 224, 3)}

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is False
        assert "batch size too large" in result.reason.lower()
        assert "batch_size" in result.corrected_values

    def test_validate_input_no_shape(self, model_validator):
        """Test input with no shape."""
        input_data = {"batch_size": 32, "shape": ()}

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is False
        assert "no shape" in result.reason.lower()

    def test_validate_input_contains_nan(self, model_validator):
        """Test input containing NaN."""
        input_data = {
            "batch_size": 32,
            "shape": (32, 224, 224, 3),
            "contains_nan": True,
        }

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is False
        assert "nan" in result.reason.lower()

    def test_validate_input_contains_inf(self, model_validator):
        """Test input containing Inf."""
        input_data = {
            "batch_size": 32,
            "shape": (32, 224, 224, 3),
            "contains_inf": True,
        }

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is False
        assert "inf" in result.reason.lower()

    def test_validate_input_extreme_range(self, model_validator):
        """Test input with extreme value range."""
        input_data = {
            "batch_size": 32,
            "shape": (32, 224, 224, 3),
            "min_value": -1e8,
            "max_value": 1e8,
        }

        result = model_validator.validate_inference_input(input_data)

        assert result.safe is False
        assert "extreme range" in result.reason.lower()

    def test_validate_output_safe(self, model_validator):
        """Test validating safe inference output."""
        output_data = {
            "contains_nan": False,
            "contains_inf": False,
            "confidences": [0.9, 0.85, 0.92],
            "inference_time": 0.5,
            "shape": (32, 10),
        }

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is True

    def test_validate_output_contains_nan(self, model_validator):
        """Test output containing NaN."""
        output_data = {"contains_nan": True, "inference_time": 0.5, "shape": (32, 10)}

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is False
        assert "nan" in result.reason.lower()
        assert result.severity == "critical"

    def test_validate_output_contains_inf(self, model_validator):
        """Test output containing Inf."""
        output_data = {"contains_inf": True, "inference_time": 0.5, "shape": (32, 10)}

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is False
        assert result.severity == "critical"

    def test_validate_output_low_confidence(self, model_validator):
        """Test output with low confidence."""
        output_data = {
            "confidences": [0.3, 0.4, 0.2],
            "inference_time": 0.5,
            "shape": (32, 10),
        }

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is False
        assert "low confidence" in result.reason.lower()

    def test_validate_output_slow_inference(self, model_validator):
        """Test output with slow inference time."""
        output_data = {
            "confidences": [0.9, 0.85],
            "inference_time": 120.0,  # Too slow
            "shape": (32, 10),
        }

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is False
        assert "too long" in result.reason.lower()

    def test_validate_output_invalid_shape(self, model_validator):
        """Test output with invalid shape."""
        output_data = {"inference_time": 0.5, "shape": ()}

        result = model_validator.validate_inference_output(output_data)

        assert result.safe is False
        assert "invalid shape" in result.reason.lower()


# ============================================================================
# DOMAIN VALIDATOR REGISTRY TESTS
# ============================================================================


class TestDomainValidatorRegistry:
    """Test DomainValidatorRegistry class."""

    def test_initialization(self, registry):
        """Test registry initialization."""
        domains = registry.list_domains()

        assert "causal" in domains
        assert "prediction" in domains
        assert "optimization" in domains
        assert "data_processing" in domains
        assert "model_inference" in domains

    def test_get_validator(self, registry):
        """Test getting validator from registry."""
        validator = registry.get_validator("causal")

        assert validator is not None
        assert isinstance(validator, CausalSafetyValidator)

    def test_get_validator_nonexistent(self, registry):
        """Test getting nonexistent validator."""
        validator = registry.get_validator("nonexistent_domain")

        assert validator is None

    def test_register_custom_validator(self, registry):
        """Test registering a custom validator."""

        class CustomValidator(DomainValidator):
            def validate(self, data, context=None):
                return ValidationResult(safe=True)

        # Register an instance, not the class (matches pattern in initialize_domain_validators)
        registry.register("custom", CustomValidator())

        validator = registry.get_validator("custom")
        assert validator is not None
        assert isinstance(validator, CustomValidator)

    def test_list_domains(self, registry):
        """Test listing all domains."""
        domains = registry.list_domains()

        assert isinstance(domains, list)
        assert len(domains) >= 5

    def test_global_registry_instance(self):
        """Test that global registry instance exists."""
        assert validator_registry is not None
        assert isinstance(validator_registry, DomainValidatorRegistry)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for domain validators."""

    def test_multi_domain_validation(self):
        """Test validating across multiple domains."""
        registry = DomainValidatorRegistry()

        # Causal validation
        causal_result = registry.get_validator("causal").validate_causal_edge(
            "x", "y", 2.0
        )

        # Prediction validation
        pred_result = registry.get_validator("prediction").validate_prediction(
            expected=50.0, lower=45.0, upper=55.0, variable="test"
        )

        assert causal_result.safe is True
        assert pred_result.safe is True

    def test_validation_statistics_tracking(self):
        """Test that validation statistics are tracked correctly."""
        validator = CausalSafetyValidator()

        # Perform multiple validations
        for i in range(20):
            validator.validate_causal_edge(
                f"x{i}",
                f"y{i}",
                2.0 if i % 2 == 0 else 100.0,  # Half will be unsafe
            )

        stats = validator.get_stats()

        assert stats["total_validations"] == 20
        assert stats["safe_count"] == 10
        assert stats["unsafe_count"] == 10
        assert stats["safety_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
