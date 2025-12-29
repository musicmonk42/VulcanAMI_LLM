"""
Comprehensive tests for MathematicalVerificationEngine.

Tests cover:
- Bayesian calculation verification
- Specificity/sensitivity confusion detection
- Complement error detection
- Base rate neglect detection
- Probability axiom verification
- Arithmetic verification
- Concept relationship reasoning
- Meta-learning for error patterns

Critical focus areas:
- P(Test-|No Disease) vs P(Test+|No Disease) confusion
- Proper Bayesian inference validation
- Human-interpretable error explanations
"""

import math
import pytest

from src.vulcan.reasoning.mathematical_verification import (
    BayesianProblem,
    MathematicalToolOrchestrator,
    MathematicalVerificationEngine,
    MathErrorType,
    MathProblem,
    MathSolution,
    MathVerificationStatus,
    VerificationResult,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a MathematicalVerificationEngine instance."""
    return MathematicalVerificationEngine()


@pytest.fixture
def tool_orchestrator():
    """Create a MathematicalToolOrchestrator instance."""
    return MathematicalToolOrchestrator()


# ============================================================================
# BAYESIAN PROBLEM TESTS
# ============================================================================


class TestBayesianProblem:
    """Tests for BayesianProblem data structure."""

    def test_create_basic_problem(self):
        """Test creating a basic Bayesian problem."""
        problem = BayesianProblem(
            prior=0.01,
            likelihood=0.99,
            false_positive_rate=0.05,
        )
        
        assert problem.prior == 0.01
        assert problem.likelihood == 0.99
        assert problem.false_positive_rate == 0.05

    def test_create_diagnostic_problem(self):
        """Test creating a diagnostic test problem with sensitivity/specificity."""
        problem = BayesianProblem(
            prior=0.001,  # 0.1% disease prevalence
            sensitivity=0.99,  # 99% true positive rate
            specificity=0.95,  # 95% true negative rate
            description="Cancer screening test",
        )
        
        assert problem.prior == 0.001
        assert problem.sensitivity == 0.99
        assert problem.specificity == 0.95

    def test_validate_valid_problem(self):
        """Test validation of valid problem."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.99,
            specificity=0.95,
        )
        
        errors = problem.validate()
        assert len(errors) == 0

    def test_validate_invalid_prior(self):
        """Test validation catches invalid prior."""
        problem = BayesianProblem(
            prior=1.5,  # Invalid: > 1
            sensitivity=0.99,
            specificity=0.95,
        )
        
        errors = problem.validate()
        assert len(errors) > 0
        assert "prior" in errors[0].lower()

    def test_validate_invalid_sensitivity(self):
        """Test validation catches invalid sensitivity."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=-0.1,  # Invalid: < 0
            specificity=0.95,
        )
        
        errors = problem.validate()
        assert len(errors) > 0
        assert "sensitivity" in errors[0].lower()

    def test_validate_invalid_specificity(self):
        """Test validation catches invalid specificity."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.99,
            specificity=1.5,  # Invalid: > 1
        )
        
        errors = problem.validate()
        assert len(errors) > 0
        assert "specificity" in errors[0].lower()


# ============================================================================
# BAYESIAN VERIFICATION TESTS
# ============================================================================


class TestBayesianVerification:
    """Tests for Bayesian calculation verification."""

    def test_verify_correct_calculation(self, engine):
        """Test verification of correct Bayesian calculation."""
        # Classic mammogram example
        # Prior: 1% have disease
        # Sensitivity: 80% (test detects disease when present)
        # Specificity: 90.4% (test negative when no disease)
        
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,
        )
        
        # Correct calculation:
        # P(Test+|Disease) = 0.80
        # P(Test+|No Disease) = 1 - 0.904 = 0.096
        # P(Test+) = 0.80 * 0.01 + 0.096 * 0.99 = 0.008 + 0.09504 = 0.10304
        # P(Disease|Test+) = 0.80 * 0.01 / 0.10304 ≈ 0.0776
        
        correct_posterior = 0.80 * 0.01 / (0.80 * 0.01 + 0.096 * 0.99)
        
        result = engine.verify_bayesian_calculation(problem, correct_posterior)
        
        assert result.status == MathVerificationStatus.VERIFIED
        assert result.confidence > 0.9
        assert len(result.errors) == 0

    def test_detect_specificity_confusion(self, engine):
        """
        CRITICAL TEST: Detect when specificity is confused with false positive rate.
        
        This is a common error where P(Test-|No Disease) is used instead of
        P(Test+|No Disease) = 1 - specificity in the Bayes calculation.
        """
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,  # True negative rate
        )
        
        # INCORRECT calculation using specificity instead of FPR:
        # Uses 0.904 where it should use 0.096
        # P(Test+) = 0.80 * 0.01 + 0.904 * 0.99 (WRONG!)
        wrong_p_evidence = 0.80 * 0.01 + 0.904 * 0.99
        wrong_posterior = (0.80 * 0.01) / wrong_p_evidence
        
        result = engine.verify_bayesian_calculation(problem, wrong_posterior)
        
        # Should detect error
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.SPECIFICITY_CONFUSION in result.errors
        
        # Should provide correction
        assert "correct_posterior" in result.corrections
        
        # Explanation should mention specificity confusion
        assert "specificity" in result.explanation.lower()

    def test_detect_base_rate_neglect(self, engine):
        """
        Test detection of base rate neglect.
        
        A common error is to confuse P(Disease|Test+) with P(Test+|Disease),
        essentially ignoring the prior probability.
        """
        problem = BayesianProblem(
            prior=0.01,  # Very low base rate
            sensitivity=0.99,  # High sensitivity
            specificity=0.95,
        )
        
        # Wrong answer: just using sensitivity (ignoring base rate)
        wrong_posterior = 0.99
        
        result = engine.verify_bayesian_calculation(problem, wrong_posterior)
        
        # Should detect error
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.BASE_RATE_NEGLECT in result.errors
        
        # Correct posterior should be much lower than 0.99
        correct = result.corrections.get("correct_posterior")
        assert correct is not None
        assert correct < 0.5  # Much lower than the sensitivity

    def test_detect_complement_error(self, engine):
        """Test detection of complement errors."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.99,
            specificity=0.95,
        )
        
        # Calculate correct posterior
        fpr = 1 - 0.95  # 0.05
        p_evidence = 0.99 * 0.01 + fpr * 0.99
        correct_posterior = (0.99 * 0.01) / p_evidence
        
        # Use complement of correct answer
        wrong_posterior = 1 - correct_posterior
        
        result = engine.verify_bayesian_calculation(problem, wrong_posterior)
        
        # Should detect complement error
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.COMPLEMENT_ERROR in result.errors

    def test_verify_high_prior_case(self, engine):
        """Test verification with high prior probability."""
        # When disease is common, positive test is more meaningful
        problem = BayesianProblem(
            prior=0.5,  # 50% base rate
            sensitivity=0.99,
            specificity=0.95,
        )
        
        # Correct calculation
        fpr = 0.05
        p_evidence = 0.99 * 0.5 + fpr * 0.5
        correct_posterior = (0.99 * 0.5) / p_evidence
        
        result = engine.verify_bayesian_calculation(problem, correct_posterior)
        
        assert result.status == MathVerificationStatus.VERIFIED
        assert correct_posterior > 0.95  # High posterior with high prior

    def test_verify_perfect_test(self, engine):
        """Test verification with perfect sensitivity and specificity."""
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=1.0,  # Perfect sensitivity
            specificity=1.0,  # Perfect specificity
        )
        
        # With perfect test, positive result means disease
        result = engine.verify_bayesian_calculation(problem, 1.0)
        
        assert result.status == MathVerificationStatus.VERIFIED


# ============================================================================
# ARITHMETIC VERIFICATION TESTS
# ============================================================================


class TestArithmeticVerification:
    """Tests for arithmetic calculation verification."""

    def test_verify_simple_arithmetic(self, engine):
        """Test verification of simple arithmetic."""
        result = engine.verify_arithmetic("2 + 2", 4)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_verify_complex_expression(self, engine):
        """Test verification of complex expression."""
        result = engine.verify_arithmetic("sqrt(16) + 3 * 2", 10)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_verify_with_variables(self, engine):
        """Test verification with variable substitution."""
        result = engine.verify_arithmetic(
            "x + y * 2",
            7,
            variables={"x": 1, "y": 3},
        )
        assert result.status == MathVerificationStatus.VERIFIED

    def test_detect_arithmetic_error(self, engine):
        """Test detection of arithmetic error."""
        result = engine.verify_arithmetic("2 + 2", 5)
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.ARITHMETIC_ERROR in result.errors
        assert result.corrections.get("correct_result") == 4

    def test_detect_division_by_zero(self, engine):
        """Test detection of division by zero."""
        result = engine.verify_arithmetic("1 / 0", 0)
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.DIVISION_BY_ZERO in result.errors

    def test_verify_trigonometric(self, engine):
        """Test verification of trigonometric expressions."""
        result = engine.verify_arithmetic("sin(pi/2)", 1.0)
        assert result.status == MathVerificationStatus.VERIFIED
        
        result = engine.verify_arithmetic("cos(0)", 1.0)
        assert result.status == MathVerificationStatus.VERIFIED


# ============================================================================
# PROBABILITY DISTRIBUTION TESTS
# ============================================================================


class TestProbabilityDistributionVerification:
    """Tests for probability distribution verification."""

    def test_verify_valid_distribution(self, engine):
        """Test verification of valid probability distribution."""
        distribution = {"A": 0.3, "B": 0.5, "C": 0.2}
        
        result = engine.verify_probability_distribution(distribution)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_detect_sum_not_one(self, engine):
        """Test detection when probabilities don't sum to 1."""
        distribution = {"A": 0.3, "B": 0.5, "C": 0.3}  # Sum = 1.1
        
        result = engine.verify_probability_distribution(distribution)
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.PROBABILITY_AXIOM_VIOLATION in result.errors

    def test_detect_out_of_bounds(self, engine):
        """Test detection of probability out of [0, 1] bounds."""
        distribution = {"A": 1.2, "B": -0.1, "C": -0.1}
        
        result = engine.verify_probability_distribution(distribution)
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.PROBABILITY_AXIOM_VIOLATION in result.errors


# ============================================================================
# CONCEPT GRAPH TESTS
# ============================================================================


class TestConceptGraph:
    """Tests for mathematical concept relationship reasoning."""

    def test_get_related_concepts(self, engine):
        """Test getting related mathematical concepts."""
        related = engine.get_related_concepts("sensitivity")
        
        assert "true_positive_rate" in related or "recall" in related

    def test_get_concept_complement(self, engine):
        """Test getting complement of a concept."""
        complement = engine.get_concept_complement("sensitivity")
        assert complement == "false_negative_rate"
        
        complement = engine.get_concept_complement("specificity")
        assert complement == "false_positive_rate"

    def test_detect_conceptual_misuse(self, engine):
        """Test detection of conceptual misuse."""
        # Using sensitivity when specificity should be used
        error = engine.detect_conceptual_error(
            "sensitivity",
            context={"required_concepts": ["specificity"]},
        )
        
        assert error is not None
        assert error["error"] == MathErrorType.CONCEPTUAL_MISUSE
        assert error["should_use"] == "specificity"


# ============================================================================
# META-LEARNING TESTS
# ============================================================================


class TestMetaLearning:
    """Tests for meta-learning capabilities."""

    def test_get_likely_errors_bayesian(self, engine):
        """Test prediction of likely errors for Bayesian problems."""
        problem = BayesianProblem(
            prior=0.001,  # Very low prior
            sensitivity=0.99,
            specificity=0.95,
        )
        
        likely_errors = engine.get_likely_errors(problem)
        
        # Should predict base rate neglect for low prior
        assert MathErrorType.BASE_RATE_NEGLECT in likely_errors
        # Should predict specificity confusion when specificity is present
        assert MathErrorType.SPECIFICITY_CONFUSION in likely_errors

    def test_statistics_tracking(self, engine):
        """Test that verification statistics are tracked."""
        # Perform some verifications
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.99,
            specificity=0.95,
        )
        
        engine.verify_bayesian_calculation(problem, 0.99)  # Wrong
        engine.verify_bayesian_calculation(problem, 0.99)  # Wrong again
        
        stats = engine.get_statistics()
        
        assert stats["total_verifications"] >= 2
        assert len(stats["error_counts"]) > 0


# ============================================================================
# TOOL ORCHESTRATOR TESTS
# ============================================================================


class TestToolOrchestrator:
    """Tests for MathematicalToolOrchestrator."""

    def test_available_tools(self, tool_orchestrator):
        """Test that tools are detected."""
        # NumPy should always be available
        assert tool_orchestrator.tools_available.get("numpy", False)

    def test_verify_with_multiple_tools(self, tool_orchestrator):
        """Test cross-validation using multiple tools."""
        result = tool_orchestrator.verify_with_multiple_tools(
            "2 * 3 + 4",
            10,
        )
        
        assert result.status == MathVerificationStatus.VERIFIED
        assert result.confidence > 0.8

    def test_detect_error_with_multiple_tools(self, tool_orchestrator):
        """Test error detection using multiple tools."""
        result = tool_orchestrator.verify_with_multiple_tools(
            "2 * 3 + 4",
            15,  # Wrong answer
        )
        
        assert result.status == MathVerificationStatus.ERROR_DETECTED


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for mathematical verification."""

    def test_full_bayesian_workflow(self, engine):
        """Test complete Bayesian verification workflow."""
        # Real-world scenario: COVID-19 rapid test
        # Prevalence: 5%
        # Sensitivity: 85%
        # Specificity: 99%
        
        problem = BayesianProblem(
            prior=0.05,
            sensitivity=0.85,
            specificity=0.99,
            description="COVID-19 rapid test positive result",
        )
        
        # 1. Validate the problem specification
        validation_errors = problem.validate()
        assert len(validation_errors) == 0
        
        # 2. Calculate correct posterior
        fpr = 1 - problem.specificity  # 0.01
        p_positive = (
            problem.sensitivity * problem.prior +
            fpr * (1 - problem.prior)
        )
        correct_posterior = (
            problem.sensitivity * problem.prior
        ) / p_positive
        
        # 3. Verify the correct calculation
        result = engine.verify_bayesian_calculation(problem, correct_posterior)
        assert result.status == MathVerificationStatus.VERIFIED
        
        # 4. Get likely errors for this problem type
        likely_errors = engine.get_likely_errors(problem)
        assert isinstance(likely_errors, list)
        
        # 5. Verify statistics are updated
        stats = engine.get_statistics()
        assert stats["total_verifications"] >= 1

    def test_error_correction_pipeline(self, engine):
        """Test the complete error detection and correction pipeline."""
        # Problem with common specificity confusion error
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.95,
            specificity=0.90,
        )
        
        # Calculate with WRONG method (specificity confusion)
        wrong_p_evidence = 0.95 * 0.01 + 0.90 * 0.99  # Wrong: using spec instead of FPR
        wrong_posterior = (0.95 * 0.01) / wrong_p_evidence
        
        # Verify (should detect error)
        result = engine.verify_bayesian_calculation(problem, wrong_posterior)
        
        # Check error was detected
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.SPECIFICITY_CONFUSION in result.errors
        
        # Get the correction
        correct_posterior = result.corrections.get("correct_posterior")
        assert correct_posterior is not None
        
        # Verify the corrected value
        result2 = engine.verify_bayesian_calculation(problem, correct_posterior)
        assert result2.status == MathVerificationStatus.VERIFIED

    def test_concurrent_verifications(self, engine):
        """Test thread safety of concurrent verifications."""
        import threading
        
        results = []
        errors = []
        
        def verify_task(problem_id):
            try:
                problem = BayesianProblem(
                    prior=0.01 * (problem_id + 1),
                    sensitivity=0.95,
                    specificity=0.90,
                )
                
                # Calculate correct posterior
                fpr = 0.10
                p_evidence = 0.95 * problem.prior + fpr * (1 - problem.prior)
                correct = (0.95 * problem.prior) / p_evidence
                
                result = engine.verify_bayesian_calculation(problem, correct)
                results.append(result.status == MathVerificationStatus.VERIFIED)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent verifications
        threads = [
            threading.Thread(target=verify_task, args=(i,))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should succeed without errors
        assert len(errors) == 0
        assert all(results)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_prior(self, engine):
        """Test handling of zero prior probability."""
        problem = BayesianProblem(
            prior=0.0,
            sensitivity=0.99,
            specificity=0.95,
        )
        
        # With zero prior, posterior should also be zero
        result = engine.verify_bayesian_calculation(problem, 0.0)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_certainty_prior(self, engine):
        """Test handling of prior probability = 1."""
        problem = BayesianProblem(
            prior=1.0,
            sensitivity=0.99,
            specificity=0.95,
        )
        
        # With prior = 1, posterior should also be 1
        result = engine.verify_bayesian_calculation(problem, 1.0)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_very_small_numbers(self, engine):
        """Test handling of very small probabilities."""
        problem = BayesianProblem(
            prior=0.0001,  # 0.01%
            sensitivity=0.999,
            specificity=0.9999,
        )
        
        # Calculate correct value
        fpr = 0.0001
        p_evidence = 0.999 * 0.0001 + fpr * 0.9999
        correct = (0.999 * 0.0001) / p_evidence
        
        result = engine.verify_bayesian_calculation(problem, correct)
        assert result.status == MathVerificationStatus.VERIFIED

    def test_floating_point_precision(self, engine):
        """Test handling of floating point precision issues."""
        # Values that might cause precision issues
        result = engine.verify_arithmetic(
            "0.1 + 0.2",
            0.3,
        )
        
        # Should handle floating point comparison correctly
        assert result.status == MathVerificationStatus.VERIFIED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
