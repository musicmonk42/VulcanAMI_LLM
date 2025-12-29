"""
Tests for EnhancedMathematicalDecomposer.

Tests the integration between ProblemDecomposer and MathematicalVerificationEngine.
"""

import pytest

# Check if required modules are available
try:
    from src.vulcan.problem_decomposer.mathematical_decomposer import (
        EnhancedMathematicalDecomposer,
        MathematicalProblemContext,
        detect_mathematical_problem,
        MATH_VERIFICATION_AVAILABLE,
        DECOMPOSER_AVAILABLE,
    )
    MATHEMATICAL_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    MATHEMATICAL_DECOMPOSER_AVAILABLE = False

try:
    from src.vulcan.reasoning.mathematical_verification import (
        BayesianProblem,
        MathErrorType,
        MathVerificationStatus,
    )
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not MATHEMATICAL_DECOMPOSER_AVAILABLE,
    reason="Mathematical decomposer not available"
)


class TestMathematicalProblemDetection:
    """Tests for mathematical problem detection."""

    def test_detect_non_mathematical_problem(self):
        """Test detection of non-mathematical problem."""
        # Create a mock problem graph
        class MockProblemGraph:
            metadata = {
                "domain": "general",
                "type": "classification",
                "description": "Classify images",
            }
            nodes = {}
        
        context = detect_mathematical_problem(MockProblemGraph())
        assert not context.is_mathematical

    def test_detect_bayesian_problem(self):
        """Test detection of Bayesian problem."""
        class MockProblemGraph:
            metadata = {
                "domain": "medical",
                "type": "diagnosis",
                "description": "Calculate posterior probability given sensitivity and specificity",
            }
            nodes = {}
        
        context = detect_mathematical_problem(MockProblemGraph())
        assert context.is_mathematical
        assert context.has_bayesian
        assert context.has_probability

    def test_detect_probability_problem(self):
        """Test detection of probability problem."""
        class MockProblemGraph:
            metadata = {
                "domain": "statistics",
                "type": "inference",
                "description": "Calculate conditional probability distribution",
            }
            nodes = {}
        
        context = detect_mathematical_problem(MockProblemGraph())
        assert context.is_mathematical
        assert context.has_probability

    def test_detect_arithmetic_in_nodes(self):
        """Test detection of arithmetic in compute nodes."""
        class MockProblemGraph:
            metadata = {
                "domain": "engineering",
                "type": "calculation",
                "description": "Process data",
            }
            nodes = {
                "node1": {"type": "compute"},
                "node2": {"type": "calculate_sum"},
            }
        
        context = detect_mathematical_problem(MockProblemGraph())
        assert context.is_mathematical
        assert context.has_arithmetic


class TestEnhancedMathematicalDecomposer:
    """Tests for EnhancedMathematicalDecomposer."""

    @pytest.fixture
    def decomposer(self):
        """Create an EnhancedMathematicalDecomposer instance."""
        return EnhancedMathematicalDecomposer()

    def test_initialization(self, decomposer):
        """Test decomposer initializes correctly."""
        assert decomposer is not None
        if MATH_VERIFICATION_AVAILABLE:
            assert decomposer.math_engine is not None

    @pytest.mark.skipif(
        not VERIFICATION_AVAILABLE,
        reason="Verification module not available"
    )
    def test_verify_correct_bayesian(self, decomposer):
        """Test verification of correct Bayesian calculation."""
        if not decomposer.math_engine:
            pytest.skip("Math engine not available")
        
        # Create a Bayesian problem
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,
        )
        
        # Calculate correct posterior
        fpr = 1 - 0.904
        p_evidence = 0.80 * 0.01 + fpr * 0.99
        correct_posterior = (0.80 * 0.01) / p_evidence
        
        result = decomposer.verify_bayesian_calculation(problem, correct_posterior)
        
        assert result is not None
        assert result.status == MathVerificationStatus.VERIFIED

    @pytest.mark.skipif(
        not VERIFICATION_AVAILABLE,
        reason="Verification module not available"
    )
    def test_detect_specificity_confusion(self, decomposer):
        """Test detection of specificity confusion error."""
        if not decomposer.math_engine:
            pytest.skip("Math engine not available")
        
        problem = BayesianProblem(
            prior=0.01,
            sensitivity=0.80,
            specificity=0.904,
        )
        
        # Calculate with wrong method (specificity confusion)
        wrong_p_evidence = 0.80 * 0.01 + 0.904 * 0.99
        wrong_posterior = (0.80 * 0.01) / wrong_p_evidence
        
        result = decomposer.verify_bayesian_calculation(problem, wrong_posterior)
        
        assert result is not None
        assert result.status == MathVerificationStatus.ERROR_DETECTED
        assert MathErrorType.SPECIFICITY_CONFUSION in result.errors

    def test_get_statistics(self, decomposer):
        """Test statistics retrieval."""
        stats = decomposer.get_math_statistics()
        
        assert isinstance(stats, dict)
        assert "total_verifications" in stats
        assert "error_counts" in stats
        assert "engine_available" in stats

    @pytest.mark.skipif(
        not VERIFICATION_AVAILABLE,
        reason="Verification module not available"
    )
    def test_get_likely_errors(self, decomposer):
        """Test prediction of likely errors."""
        if not decomposer.math_engine:
            pytest.skip("Math engine not available")
        
        problem = BayesianProblem(
            prior=0.001,  # Very low prior
            sensitivity=0.99,
            specificity=0.95,
        )
        
        likely_errors = decomposer.get_likely_errors(problem)
        
        assert isinstance(likely_errors, list)
        # Should predict base rate neglect for low prior
        assert MathErrorType.BASE_RATE_NEGLECT in likely_errors


class TestIntegration:
    """Integration tests."""

    def test_imports_work(self):
        """Test that imports work correctly."""
        from src.vulcan.problem_decomposer import (
            MATHEMATICAL_DECOMPOSER_AVAILABLE,
            MATH_VERIFICATION_AVAILABLE,
        )
        
        # Just verify the flags are defined
        assert isinstance(MATHEMATICAL_DECOMPOSER_AVAILABLE, bool)
        assert isinstance(MATH_VERIFICATION_AVAILABLE, bool)

    @pytest.mark.skipif(
        not DECOMPOSER_AVAILABLE,
        reason="Base decomposer not available"
    )
    def test_with_base_decomposer(self):
        """Test that decomposer works when base decomposer is available."""
        decomposer = EnhancedMathematicalDecomposer()
        
        # Should have base decomposer
        assert decomposer.base_decomposer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
