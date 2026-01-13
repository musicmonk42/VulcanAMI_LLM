"""
Test suite for Mathematical Reasoning Platform Fixes (Jan 2026)

This test suite validates the fixes for systematic failures in mathematical
reasoning queries that were returning 0.10 confidence scores.

Root Causes Fixed:
1. MATHEMATICAL missing from UNKNOWN_TYPE_FALLBACK_ORDER
2. Mathematical notation patterns not detected (∑, ∫, ∂, P(X|Y))
3. Induction proof patterns not recognized
4. Bayesian vs pure mathematical probability notation confusion

Industry Standards Applied:
- Comprehensive test coverage for all fix scenarios
- Clear test names describing what is being tested
- Isolated unit tests for each component
- Integration tests for end-to-end flows
- Performance benchmarks for classification speed
- Type annotations throughout
- Docstrings with examples and expected behavior
"""

import pytest
import re
import time
from typing import Dict, Any

# Try to import the modules - handle missing dependencies gracefully
try:
    from vulcan.reasoning.unified.config import UNKNOWN_TYPE_FALLBACK_ORDER
    from vulcan.reasoning.unified.orchestrator import (
        MATH_SYMBOLS_PATTERN,
        PROBABILITY_NOTATION_PATTERN,
        INDUCTION_PATTERN,
        MATH_EXPRESSION_PATTERN,
        MATH_QUERY_PATTERN,
    )
    from vulcan.reasoning.reasoning_types import ReasoningType
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ==============================================================================
# TEST SUITE 1: Configuration Changes
# ==============================================================================

@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestFallbackOrderFix:
    """Test that UNKNOWN_TYPE_FALLBACK_ORDER includes MATHEMATICAL"""
    
    def test_mathematical_in_fallback_order(self):
        """MATHEMATICAL should be in fallback order to prevent 0.10 confidence failures"""
        assert "MATHEMATICAL" in UNKNOWN_TYPE_FALLBACK_ORDER, \
            "MATHEMATICAL must be in UNKNOWN_TYPE_FALLBACK_ORDER to handle unknown math queries"
    
    def test_multimodal_in_fallback_order(self):
        """MULTIMODAL should be in fallback order"""
        assert "MULTIMODAL" in UNKNOWN_TYPE_FALLBACK_ORDER, \
            "MULTIMODAL should be in fallback order"
    
    def test_abstract_in_fallback_order(self):
        """ABSTRACT should be in fallback order"""
        assert "ABSTRACT" in UNKNOWN_TYPE_FALLBACK_ORDER, \
            "ABSTRACT should be in fallback order"
    
    def test_fallback_order_priority(self):
        """MATHEMATICAL should come before SYMBOLIC for better coverage"""
        fallback_list = list(UNKNOWN_TYPE_FALLBACK_ORDER)
        if "MATHEMATICAL" in fallback_list and "SYMBOLIC" in fallback_list:
            math_idx = fallback_list.index("MATHEMATICAL")
            symbolic_idx = fallback_list.index("SYMBOLIC")
            assert math_idx < symbolic_idx, \
                "MATHEMATICAL should come before SYMBOLIC in fallback order"


# ==============================================================================
# TEST SUITE 2: Pattern Detection
# ==============================================================================

@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestMathematicalNotationPatterns:
    """Test that advanced mathematical notation is properly detected"""
    
    def test_summation_symbol_detected(self):
        """∑ (summation) symbol should be detected"""
        query = "Compute exactly: ∑_{k=1}^n (2k-1). Then verify by induction."
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "∑ (summation) symbol must be detected"
    
    def test_integral_symbol_detected(self):
        """∫ (integral) symbol should be detected"""
        query = "Calculate ∫ x^2 dx from 0 to 1"
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "∫ (integral) symbol must be detected"
    
    def test_partial_derivative_detected(self):
        """∂ (partial derivative) symbol should be detected"""
        query = "Find ∂f/∂x where f(x,y) = x^2 + y^2"
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "∂ (partial derivative) symbol must be detected"
    
    def test_latex_sum_command_detected(self):
        """\\sum LaTeX command should be detected"""
        query = "Compute \\sum_{i=1}^{10} i^2"
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "\\sum LaTeX command must be detected"
    
    def test_latex_int_command_detected(self):
        """\\int LaTeX command should be detected"""
        query = "Evaluate \\int_0^1 x^3 dx"
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "\\int LaTeX command must be detected"
    
    def test_english_sum_keyword_detected(self):
        """'sum' keyword should be detected"""
        query = "Calculate the sum of squares from 1 to n"
        assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
            "'sum' keyword must be detected"
    
    def test_set_theory_symbols_detected(self):
        """Set theory symbols (∈, ∪, ∩) should be detected"""
        queries = [
            "Is x ∈ A?",
            "Find A ∪ B",
            "Calculate A ∩ B",
        ]
        for query in queries:
            assert MATH_SYMBOLS_PATTERN.search(query) is not None, \
                f"Set theory symbols must be detected in: {query}"


@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestProbabilityNotationPatterns:
    """Test probability notation P(X), P(X|Y) detection"""
    
    def test_simple_probability_notation(self):
        """P(X) notation should be detected"""
        query = "What is P(Disease)?"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "P(X) notation must be detected"
    
    def test_conditional_probability_notation(self):
        """P(X|Y) notation should be detected"""
        query = "Compute P(X|+)"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "P(X|Y) notation must be detected"
    
    def test_conditional_probability_with_plus(self):
        """P(Disease|Test+) notation should be detected"""
        query = "Calculate P(Disease|Test+) given sensitivity and specificity"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "P(X|Y+) notation must be detected"
    
    def test_pr_notation(self):
        """Pr(X) alternative notation should be detected"""
        query = "Find Pr(Success)"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "Pr(X) notation must be detected"
    
    def test_expected_value_notation(self):
        """E[X] expected value notation should be detected"""
        query = "Calculate E[X] where X is a random variable"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "E[X] notation must be detected"
    
    def test_variance_notation(self):
        """Var(X) variance notation should be detected"""
        query = "What is Var(X)?"
        assert PROBABILITY_NOTATION_PATTERN.search(query) is not None, \
            "Var(X) notation must be detected"


@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestInductionPatterns:
    """Test mathematical induction pattern detection"""
    
    def test_prove_by_induction(self):
        """'prove by induction' phrase should be detected"""
        query = "Prove by induction that ∑_{k=1}^n (2k-1) = n^2"
        assert INDUCTION_PATTERN.search(query) is not None, \
            "'prove by induction' must be detected"
    
    def test_verify_by_induction(self):
        """'verify by induction' phrase should be detected"""
        query = "Verify by induction that the formula holds"
        assert INDUCTION_PATTERN.search(query) is not None, \
            "'verify by induction' must be detected"
    
    def test_base_case(self):
        """'base case' phrase should be detected"""
        query = "First, check the base case where n=1"
        assert INDUCTION_PATTERN.search(query) is not None, \
            "'base case' must be detected"
    
    def test_inductive_step(self):
        """'inductive step' phrase should be detected"""
        query = "Now perform the inductive step"
        assert INDUCTION_PATTERN.search(query) is not None, \
            "'inductive step' must be detected"
    
    def test_inductive_hypothesis(self):
        """'inductive hypothesis' phrase should be detected"""
        query = "Assume the inductive hypothesis holds for n=k"
        assert INDUCTION_PATTERN.search(query) is not None, \
            "'inductive hypothesis' must be detected"


# ==============================================================================
# TEST SUITE 3: End-to-End Integration Tests
# ==============================================================================

@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestMathematicalQueryClassification:
    """Integration tests for mathematical query classification"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create UnifiedReasoner instance for testing"""
        try:
            from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
            return UnifiedReasoner()
        except Exception as e:
            pytest.skip(f"Could not create UnifiedReasoner: {e}")
    
    def test_summation_query_classification(self, orchestrator):
        """Summation query should classify as MATHEMATICAL"""
        query = "Compute exactly: ∑_{k=1}^n (2k-1). Then verify by induction."
        reasoning_type = orchestrator._determine_reasoning_type(query, {})
        assert reasoning_type == ReasoningType.MATHEMATICAL, \
            f"Summation query should classify as MATHEMATICAL, got {reasoning_type}"
    
    def test_bayesian_query_classification(self, orchestrator):
        """Bayesian query should classify as PROBABILISTIC"""
        query = "A test with Sensitivity: 0.99, Specificity: 0.95, Prevalence: 0.01. Compute P(X|+)"
        reasoning_type = orchestrator._determine_reasoning_type(query, {})
        assert reasoning_type == ReasoningType.PROBABILISTIC, \
            f"Bayesian query should classify as PROBABILISTIC, got {reasoning_type}"
    
    def test_sat_query_classification(self, orchestrator):
        """SAT query should classify as SYMBOLIC"""
        query = "Propositions A,B,C with A→B, B→C, ¬C, A∨B. Is satisfiable?"
        reasoning_type = orchestrator._determine_reasoning_type(query, {})
        assert reasoning_type == ReasoningType.SYMBOLIC, \
            f"SAT query should classify as SYMBOLIC, got {reasoning_type}"


# ==============================================================================
# TEST SUITE 4: Performance Tests
# ==============================================================================

@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestPatternPerformance:
    """Test that pattern matching is efficient"""
    
    def test_pattern_compilation(self):
        """Patterns should be pre-compiled for performance"""
        # All patterns should be compiled regex objects
        patterns_to_check = [
            (MATH_SYMBOLS_PATTERN, "MATH_SYMBOLS_PATTERN"),
            (PROBABILITY_NOTATION_PATTERN, "PROBABILITY_NOTATION_PATTERN"),
            (INDUCTION_PATTERN, "INDUCTION_PATTERN"),
            (MATH_EXPRESSION_PATTERN, "MATH_EXPRESSION_PATTERN"),
            (MATH_QUERY_PATTERN, "MATH_QUERY_PATTERN"),
        ]
        
        for pattern, name in patterns_to_check:
            assert isinstance(pattern, re.Pattern), f"{name} should be compiled"
    
    @pytest.mark.benchmark
    def test_pattern_matching_speed(self):
        """Pattern matching should be fast (< 1ms per query)"""
        queries = [
            "Compute exactly: ∑_{k=1}^n (2k-1). Then verify by induction.",
            "Calculate ∫ x^2 dx from 0 to 1",
            "A test with Sensitivity: 0.99, Specificity: 0.95, Prevalence: 0.01. Compute P(X|+)",
            "Propositions A,B,C with A→B, B→C, ¬C, A∨B. Is satisfiable?",
        ]
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            for query in queries:
                MATH_SYMBOLS_PATTERN.search(query)
                PROBABILITY_NOTATION_PATTERN.search(query)
                INDUCTION_PATTERN.search(query)
        
        elapsed_time = time.time() - start_time
        avg_time_per_query = (elapsed_time / (iterations * len(queries))) * 1000  # in ms
        
        assert avg_time_per_query < 1.0, \
            f"Pattern matching should be < 1ms per query, got {avg_time_per_query:.2f}ms"


# ==============================================================================
# TEST SUITE 5: Regression Tests
# ==============================================================================

@pytest.mark.skipif(not MODULES_AVAILABLE, reason=f"Modules not available: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}")
class TestRegressionPrevention:
    """Ensure fixes don't break existing functionality"""
    
    def test_basic_arithmetic_still_works(self):
        """Basic arithmetic (2+2) should still be detected"""
        query = "What is 2+2?"
        assert MATH_EXPRESSION_PATTERN.search(query) is not None, \
            "Basic arithmetic must still be detected"
    
    def test_simple_probability_queries_still_work(self):
        """Simple probability queries should still work
        
        Note: This test accesses the private method _is_probability_query() directly
        to ensure regression prevention for the gate check logic, which is critical
        for routing queries correctly. The public interface (reason()) would require
        full system setup and wouldn't isolate this specific functionality.
        """
        from vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
        reasoner = ProbabilisticReasoner()
        assert reasoner._is_probability_query("What is the probability of heads?"), \
            "Simple probability queries must still be detected"
    
    def test_fallback_order_maintains_probabilistic_first(self):
        """PROBABILISTIC should remain first in fallback order"""
        assert UNKNOWN_TYPE_FALLBACK_ORDER[0] == "PROBABILISTIC", \
            "PROBABILISTIC should remain first in fallback order for general queries"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
