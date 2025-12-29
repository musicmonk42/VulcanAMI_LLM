"""
Tests for MathematicalAccuracyIntegration.

Tests the integration between MathematicalVerificationEngine and Vulcan's
Learning System for providing feedback on mathematical reasoning accuracy.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Check if required modules are available
try:
    from vulcan.learning.mathematical_accuracy_integration import (
        MathematicalAccuracyIntegration,
        MathematicalFeedback,
        create_math_learning_integration,
        MATH_ERROR_PENALTIES,
        MATH_VERIFICATION_REWARD,
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    try:
        from src.vulcan.learning.mathematical_accuracy_integration import (
            MathematicalAccuracyIntegration,
            MathematicalFeedback,
            create_math_learning_integration,
            MATH_ERROR_PENALTIES,
            MATH_VERIFICATION_REWARD,
        )
        INTEGRATION_AVAILABLE = True
    except ImportError as e:
        INTEGRATION_AVAILABLE = False

try:
    from vulcan.reasoning.mathematical_verification import (
        BayesianProblem,
        MathErrorType,
        MathVerificationStatus,
        VerificationResult,
    )
    VERIFICATION_AVAILABLE = True
except ImportError:
    try:
        from src.vulcan.reasoning.mathematical_verification import (
            BayesianProblem,
            MathErrorType,
            MathVerificationStatus,
            VerificationResult,
        )
        VERIFICATION_AVAILABLE = True
    except ImportError:
        VERIFICATION_AVAILABLE = False


# Skip all tests if modules not available
pytestmark = pytest.mark.skipif(
    not INTEGRATION_AVAILABLE or not VERIFICATION_AVAILABLE,
    reason="Mathematical integration modules not available"
)


class TestMathematicalAccuracyIntegration:
    """Tests for MathematicalAccuracyIntegration class."""

    @pytest.fixture
    def integration(self):
        """Create a MathematicalAccuracyIntegration instance."""
        return MathematicalAccuracyIntegration()

    @pytest.fixture
    def mock_learning_system(self):
        """Create a mock learning system."""
        mock = MagicMock()
        mock.tool_weight_adjustments = {}
        mock._weight_lock = MagicMock()
        mock._weight_lock.__enter__ = MagicMock(return_value=None)
        mock._weight_lock.__exit__ = MagicMock(return_value=None)
        return mock

    def test_initialization(self, integration):
        """Test integration initializes correctly."""
        assert integration is not None
        assert integration.math_engine is not None
        assert integration.total_verifications == 0
        assert integration.total_errors == 0

    @pytest.mark.asyncio
    async def test_process_successful_verification(
        self, integration, mock_learning_system
    ):
        """Test processing a successful verification."""
        # Create a successful verification result
        result = VerificationResult(
            status=MathVerificationStatus.VERIFIED,
            confidence=0.99,
            errors=[],
            explanation="Correct calculation",
        )
        
        feedback = await integration.process_verification_result(
            result,
            tool_name="probabilistic",
            learning_system=mock_learning_system,
            problem_type="bayesian",
        )
        
        assert feedback.verified is True
        assert feedback.tool_name == "probabilistic"
        assert feedback.error_type is None
        assert integration.tool_success_counts["probabilistic"] == 1

    @pytest.mark.asyncio
    async def test_process_error_verification(
        self, integration, mock_learning_system
    ):
        """Test processing a verification with error."""
        # Create an error verification result
        result = VerificationResult(
            status=MathVerificationStatus.ERROR_DETECTED,
            confidence=0.95,
            errors=[MathErrorType.SPECIFICITY_CONFUSION],
            corrections={"correct_posterior": 0.167},
            explanation="Specificity confusion detected",
        )
        
        feedback = await integration.process_verification_result(
            result,
            tool_name="probabilistic",
            learning_system=mock_learning_system,
            problem_type="bayesian",
        )
        
        assert feedback.verified is False
        assert feedback.error_type == "specificity_confusion"
        assert integration.total_errors == 1
        assert integration.tool_error_counts["probabilistic"]["specificity_confusion"] == 1

    @pytest.mark.asyncio
    async def test_verify_and_learn_correct(
        self, integration, mock_learning_system
    ):
        """Test verify_and_learn with correct calculation."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,
        )
        
        # Calculate correct posterior
        fpr = 1 - 0.904
        p_evidence = 0.80 * 0.01 + fpr * 0.99
        correct_posterior = (0.80 * 0.01) / p_evidence
        
        result, feedback = await integration.verify_and_learn(
            problem,
            correct_posterior,
            "probabilistic",
            mock_learning_system,
        )
        
        assert result.status == MathVerificationStatus.VERIFIED
        assert feedback.verified is True

    @pytest.mark.asyncio
    async def test_verify_and_learn_error(
        self, integration, mock_learning_system
    ):
        """Test verify_and_learn with specificity confusion error."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,
        )
        
        # Calculate with wrong method (specificity confusion)
        wrong_p_evidence = 0.80 * 0.01 + 0.904 * 0.99
        wrong_posterior = (0.80 * 0.01) / wrong_p_evidence
        
        result, feedback = await integration.verify_and_learn(
            problem,
            wrong_posterior,
            "probabilistic",
            mock_learning_system,
        )
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert feedback.verified is False
        assert MathErrorType.SPECIFICITY_CONFUSION in result.errors

    def test_penalize_tool_sync(self, integration, mock_learning_system):
        """Test synchronous penalize_tool method."""
        integration.penalize_tool(
            "probabilistic",
            MathErrorType.SPECIFICITY_CONFUSION,
            mock_learning_system,
        )
        
        assert integration.total_errors == 1
        assert "specificity_confusion" in integration.tool_error_counts["probabilistic"]

    def test_reward_tool_sync(self, integration, mock_learning_system):
        """Test synchronous reward_tool method."""
        integration.reward_tool(
            "probabilistic",
            mock_learning_system,
        )
        
        assert integration.tool_success_counts["probabilistic"] == 1

    def test_get_statistics(self, integration):
        """Test statistics retrieval."""
        stats = integration.get_statistics()
        
        assert "total_verifications" in stats
        assert "total_errors" in stats
        assert "overall_accuracy" in stats
        assert "tool_accuracy" in stats
        assert "most_common_errors" in stats

    def test_get_tool_error_report(self, integration, mock_learning_system):
        """Test getting error report for specific tool."""
        # Add some errors
        integration.penalize_tool(
            "test_tool",
            MathErrorType.SPECIFICITY_CONFUSION,
            mock_learning_system,
        )
        integration.penalize_tool(
            "test_tool",
            MathErrorType.BASE_RATE_NEGLECT,
            mock_learning_system,
        )
        
        report = integration.get_tool_error_report("test_tool")
        
        assert report["tool_name"] == "test_tool"
        assert report["total_verifications"] == 2
        assert report["successes"] == 0
        assert len(report["errors"]) == 2


class TestMathErrorPenalties:
    """Tests for error penalty calculations."""

    def test_penalty_values_defined(self):
        """Test that penalty values are defined for error types."""
        assert MATH_ERROR_PENALTIES is not None
        assert len(MATH_ERROR_PENALTIES) > 0

    def test_specificity_confusion_penalty(self):
        """Test that specificity confusion has appropriate penalty."""
        if MathErrorType.SPECIFICITY_CONFUSION in MATH_ERROR_PENALTIES:
            penalty = MATH_ERROR_PENALTIES[MathErrorType.SPECIFICITY_CONFUSION]
            assert penalty < 0  # Should be negative (penalty)
            assert penalty >= -0.05  # Should not be too severe

    def test_reward_is_positive(self):
        """Test that verification reward is positive."""
        assert MATH_VERIFICATION_REWARD > 0


class TestCreateIntegration:
    """Tests for factory function."""

    def test_create_math_learning_integration(self):
        """Test factory function creates integration."""
        integration = create_math_learning_integration()
        
        assert integration is not None
        assert isinstance(integration, MathematicalAccuracyIntegration)


class TestSafeMathEvaluator:
    """Tests for the safe math evaluator used in verification."""

    def _import_safe_eval(self):
        """Import safe_math_eval from appropriate location."""
        try:
            from vulcan.reasoning.mathematical_verification import (
                SafeMathEvaluator,
                safe_math_eval,
            )
            return SafeMathEvaluator, safe_math_eval
        except ImportError:
            from src.vulcan.reasoning.mathematical_verification import (
                SafeMathEvaluator,
                safe_math_eval,
            )
            return SafeMathEvaluator, safe_math_eval

    def test_safe_evaluator_import(self):
        """Test that SafeMathEvaluator can be imported."""
        SafeMathEvaluator, safe_math_eval = self._import_safe_eval()
        
        evaluator = SafeMathEvaluator()
        assert evaluator is not None

    def test_safe_eval_basic(self):
        """Test basic arithmetic with safe evaluator."""
        _, safe_math_eval = self._import_safe_eval()
        
        assert safe_math_eval("2 + 3") == 5
        assert safe_math_eval("4 * 5") == 20
        assert safe_math_eval("10 / 2") == 5

    def test_safe_eval_functions(self):
        """Test mathematical functions with safe evaluator."""
        _, safe_math_eval = self._import_safe_eval()
        import math
        
        assert abs(safe_math_eval("sqrt(16)") - 4.0) < 1e-10
        assert abs(safe_math_eval("sin(0)") - 0.0) < 1e-10
        assert abs(safe_math_eval("cos(0)") - 1.0) < 1e-10

    def test_safe_eval_constants(self):
        """Test mathematical constants with safe evaluator."""
        _, safe_math_eval = self._import_safe_eval()
        import math
        
        assert abs(safe_math_eval("pi") - math.pi) < 1e-10
        assert abs(safe_math_eval("e") - math.e) < 1e-10

    def test_safe_eval_variables(self):
        """Test variable substitution with safe evaluator."""
        _, safe_math_eval = self._import_safe_eval()
        
        result = safe_math_eval("x + y * 2", {"x": 1, "y": 3})
        assert result == 7

    def test_safe_eval_rejects_builtins(self):
        """Test that safe evaluator rejects dangerous operations."""
        _, safe_math_eval = self._import_safe_eval()
        
        # These should raise ValueError, not execute
        with pytest.raises(ValueError):
            safe_math_eval("__import__('os')")
        
        with pytest.raises(ValueError):
            safe_math_eval("open('/etc/passwd')")

    def test_safe_eval_division_by_zero(self):
        """Test that division by zero is properly handled."""
        _, safe_math_eval = self._import_safe_eval()
        
        with pytest.raises(ZeroDivisionError):
            safe_math_eval("1 / 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
