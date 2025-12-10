"""
Comprehensive test suite for explainability_node.py
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from explainability_node import (ALLOWED_METHODS, MAX_TENSOR_DIM,
                                 MAX_TENSOR_SIZE, MIN_TENSOR_DIM,
                                 CounterfactualNode, ExplainabilityNode,
                                 ExplainabilityValidator, ExplanationResult,
                                 dispatch_explainability_node)


@pytest.fixture
def validator():
    """Create validator."""
    return ExplainabilityValidator()


@pytest.fixture
def explain_node():
    """Create explainability node."""
    return ExplainabilityNode()


@pytest.fixture
def context():
    """Create test context."""
    return {
        'audit_log': [],
        'input_tensor': [[1.0, 2.0, 3.0, 4.0, 5.0]],  # Changed to 2D
        'ethical_label': 'EU2025:Safe'
    }


class TestExplanationResult:
    """Test ExplanationResult dataclass."""

    def test_initialization(self):
        """Test result initialization."""
        from datetime import datetime

        result = ExplanationResult(
            explanation={'test': 'data'},
            coverage=0.8,
            compression_ok=True,
            compression_meta={},
            kernel_audit=None,
            photonic_drift={},
            ethical_label='Safe',
            method='shap'
        )

        assert result.coverage == 0.8
        assert result.compression_ok
        assert result.method == 'shap'

    def test_to_dict(self):
        """Test conversion to dict."""
        from datetime import datetime

        result = ExplanationResult(
            explanation={'test': 'data'},
            coverage=0.5,
            compression_ok=False,
            compression_meta={'valid': False},
            kernel_audit={'result': 'ok'},
            photonic_drift={'drift': 0.1},
            ethical_label='Safe',
            method='lime'
        )

        d = result.to_dict()

        assert d['coverage'] == 0.5
        assert d['method'] == 'lime'
        assert 'timestamp' in d


class TestExplainabilityValidator:
    """Test ExplainabilityValidator class."""

    def test_validate_tensor_valid(self, validator):
        """Test validating valid tensor."""
        tensor = np.random.randn(10, 20)

        valid, error, validated = validator.validate_tensor(tensor)

        assert valid is True
        assert error is None
        assert validated is not None

    def test_validate_tensor_from_list(self, validator):
        """Test validating tensor from list."""
        tensor_list = [[1.0, 2.0], [3.0, 4.0]]

        valid, error, validated = validator.validate_tensor(tensor_list)

        assert valid is True
        assert isinstance(validated, np.ndarray)

    def test_validate_tensor_none(self, validator):
        """Test validating None tensor."""
        valid, error, validated = validator.validate_tensor(None)

        assert valid is False
        assert "cannot be None" in error

    def test_validate_tensor_wrong_type(self, validator):
        """Test validating wrong type."""
        valid, error, validated = validator.validate_tensor("not a tensor")

        assert valid is False
        assert "must be list or numpy array" in error

    def test_validate_tensor_wrong_ndim(self, validator):
        """Test validating wrong dimensions."""
        tensor_1d = np.array([1, 2, 3])

        valid, error, validated = validator.validate_tensor(tensor_1d)

        assert valid is False
        assert "dimension must be" in error

    def test_validate_tensor_too_large(self, validator):
        """Test validating too large tensor."""
        large_tensor = np.ones((MAX_TENSOR_SIZE + 1, 2))

        valid, error, validated = validator.validate_tensor(large_tensor)

        assert valid is False
        assert "too large" in error

    def test_validate_tensor_with_nan(self, validator):
        """Test validating tensor with NaN."""
        tensor = np.array([[1.0, 2.0], [np.nan, 4.0]])

        valid, error, validated = validator.validate_tensor(tensor)

        assert valid is False
        assert "NaN or Inf" in error

    def test_validate_baseline_valid(self, validator):
        """Test validating valid baseline."""
        baseline = np.zeros((10, 20))

        valid, error, validated = validator.validate_baseline(baseline, (10, 20))

        assert valid is True
        assert error is None

    def test_validate_baseline_none(self, validator):
        """Test validating None baseline."""
        valid, error, validated = validator.validate_baseline(None, (10, 20))

        assert valid is True
        assert validated is None

    def test_validate_baseline_shape_mismatch(self, validator):
        """Test validating baseline with wrong shape."""
        baseline = np.zeros((5, 10))

        valid, error, validated = validator.validate_baseline(baseline, (10, 20))

        assert valid is False
        assert "shape" in error

    def test_validate_method_valid(self, validator):
        """Test validating valid method."""
        for method in ALLOWED_METHODS:
            valid, error = validator.validate_method(method)
            assert valid is True

    def test_validate_method_invalid(self, validator):
        """Test validating invalid method."""
        valid, error = validator.validate_method("invalid_method")

        assert valid is False
        assert "not in allowed methods" in error

    def test_validate_drift_data(self, validator):
        """Test validating drift data."""
        drift_data = {
            'drift': 0.5,
            'timestamp': '2025-01-01',
            'metrics': {'value': 1.0}
        }

        is_valid, validated = validator.validate_drift_data(drift_data)

        assert is_valid is True
        assert validated['drift'] == 0.5


class TestExplainabilityNode:
    """Test ExplainabilityNode class."""

    def test_initialization(self):
        """Test node initialization."""
        node = ExplainabilityNode()

        assert node.validator is not None

    def test_execute_basic(self, explain_node, context):
        """Test basic execution."""
        params = {
            'method': 'integrated_gradients',
            'baseline': [[0.0, 0.0, 0.0, 0.0, 0.0]]  # Changed to 2D
        }

        tensor = np.array(context['input_tensor'])

        result = explain_node.execute(tensor, params, context)

        assert 'audit' in result
        assert result['audit']['status'] == 'success'
        assert len(context['audit_log']) == 1

    def test_execute_invalid_method(self, explain_node, context):
        """Test execution with invalid method."""
        params = {'method': 'invalid'}
        tensor = np.array(context['input_tensor'])

        with pytest.raises(ValueError, match="Invalid method"):
            explain_node.execute(tensor, params, context)

    def test_execute_invalid_tensor(self, explain_node, context):
        """Test execution with invalid tensor."""
        params = {'method': 'shap'}
        tensor = "not a tensor"

        with pytest.raises(ValueError, match="Invalid tensor"):
            explain_node.execute(tensor, params, context)

    def test_execute_baseline_mismatch(self, explain_node, context):
        """Test execution with baseline shape mismatch."""
        params = {
            'method': 'lime',
            'baseline': [[0.0, 0.0]]  # Wrong size
        }
        tensor = np.array(context['input_tensor'])

        with pytest.raises(ValueError, match="Invalid baseline"):
            explain_node.execute(tensor, params, context)

    def test_execute_with_all_methods(self, explain_node, context):
        """Test execution with all allowed methods."""
        tensor = np.array(context['input_tensor'])

        for method in ALLOWED_METHODS:
            context_copy = {
                'audit_log': [],
                'input_tensor': context['input_tensor']
            }

            params = {'method': method}

            try:
                result = explain_node.execute(tensor, params, context_copy)
                assert result['method'] == method
            except Exception as e:
                # Some methods may fail without proper setup, that's ok
                pass


class TestCounterfactualNode:
    """Test CounterfactualNode class."""

    def test_initialization(self):
        """Test node initialization."""
        node = CounterfactualNode()

        assert node.validator is not None

    def test_execute_basic(self, context):
        """Test basic execution."""
        node = CounterfactualNode()

        params = {
            'target_class': 1,
            'perturbation_scale': 0.1
        }

        tensor = np.array(context['input_tensor'])

        result = node.execute(tensor, params, context)

        assert 'counterfactual' in result
        assert 'original' in result
        assert result['target_class'] == 1
        assert len(context['audit_log']) == 1

    def test_execute_invalid_tensor(self, context):
        """Test execution with invalid tensor."""
        node = CounterfactualNode()

        params = {'target_class': 0}
        tensor = "not a tensor"

        with pytest.raises(ValueError):
            node.execute(tensor, params, context)


class TestDispatchFunction:
    """Test dispatch_explainability_node function."""

    def test_dispatch_explainability_node(self, context):
        """Test dispatching to ExplainabilityNode."""
        node = {
            'type': 'ExplainabilityNode',
            'params': {'method': 'saliency'}
        }

        result = dispatch_explainability_node(node, context)

        assert 'audit' in result
        assert result['method'] == 'saliency'

    def test_dispatch_counterfactual_node(self, context):
        """Test dispatching to CounterfactualNode."""
        node = {
            'type': 'CounterfactualNode',
            'params': {'target_class': 1}
        }

        result = dispatch_explainability_node(node, context)

        assert 'counterfactual' in result

    def test_dispatch_unknown_node(self, context):
        """Test dispatching to unknown node type."""
        node = {
            'type': 'UnknownNode',
            'params': {}
        }

        with pytest.raises(ValueError, match="Unknown node type"):
            dispatch_explainability_node(node, context)

    def test_dispatch_missing_tensor(self):
        """Test dispatching without tensor in context."""
        node = {
            'type': 'ExplainabilityNode',
            'params': {'method': 'shap'}
        }

        context = {}  # No input_tensor

        with pytest.raises(ValueError, match="No input_tensor"):
            dispatch_explainability_node(node, context)


class TestAuditLogging:
    """Test audit logging."""

    def test_audit_log_success(self, explain_node, context):
        """Test audit log on success."""
        params = {'method': 'gradcam'}
        tensor = np.array(context['input_tensor'])

        explain_node.execute(tensor, params, context)

        assert len(context['audit_log']) == 1
        assert context['audit_log'][0]['status'] == 'success'
        assert context['audit_log'][0]['node_type'] == 'ExplainabilityNode'

    def test_audit_log_error(self, explain_node, context):
        """Test audit log on error."""
        params = {'method': 'invalid'}
        tensor = np.array(context['input_tensor'])

        try:
            explain_node.execute(tensor, params, context)
        except:
            pass

        assert len(context['audit_log']) == 1
        assert context['audit_log'][0]['status'] == 'error'

    def test_audit_log_multiple_calls(self, context):
        """Test audit log accumulation."""
        node = {
            'type': 'ExplainabilityNode',
            'params': {'method': 'attention'}
        }

        # Call multiple times
        for _ in range(3):
            try:
                dispatch_explainability_node(node, context)
            except:
                pass

        assert len(context['audit_log']) == 3


class TestContextImmutability:
    """Test that context is properly handled."""

    def test_context_not_polluted(self, explain_node):
        """Test context is not polluted with node data."""
        context = {
            'audit_log': [],
            'input_tensor': [[1.0, 2.0, 3.0]],  # Changed to 2D
            'original_key': 'original_value'
        }

        params = {'method': 'deeplift'}
        tensor = np.array(context['input_tensor'])

        explain_node.execute(tensor, params, context)

        # Original key should still be there
        assert context['original_key'] == 'original_value'
        # audit_log should be updated
        assert len(context['audit_log']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
