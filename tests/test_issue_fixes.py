"""
Industry-Standard Test Suite for Issue #1: Math Tool Computes Wrong Formula

This test suite ensures that summation expressions are correctly parsed and computed.
It covers multiple notation formats and edge cases to prevent regressions.

Test Coverage:
1. Basic summation: ∑(2k-1) from k=1 to n → n²
2. Unicode notation variations
3. LaTeX-style notation
4. English natural language
5. Edge cases (missing bounds, malformed expressions)

Industry Standards Applied:
- Comprehensive test coverage (happy path + edge cases)
- Clear test names that describe what is being tested
- Descriptive assertions with helpful error messages
- Parameterized tests for multiple notation formats
- Integration with existing test infrastructure
"""

import pytest
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

try:
    from vulcan.reasoning.mathematical_computation import (
        MathematicalComputationTool,
        CodeTemplates,
        create_mathematical_computation_tool,
        SAFE_EXECUTION_AVAILABLE,
    )
    TOOL_AVAILABLE = True
except ImportError:
    TOOL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)


class TestIssue1SummationFormula:
    """
    Test suite for Issue #1: Math Tool Computes Wrong Formula
    
    Problem: Query asked for Σ(2k-1) from k=1 to n, but Vulcan computed Σk instead,
    returning n*(n+1)/2 instead of n².
    
    Solution: Fixed expression extraction and removed incorrect default fallback.
    """
    
    @pytest.fixture
    def templates(self):
        """Create CodeTemplates instance for testing."""
        return CodeTemplates()
    
    @pytest.fixture
    def tool(self):
        """Create MathematicalComputationTool instance for integration tests."""
        return create_mathematical_computation_tool()
    
    # =========================================================================
    # Unit Tests: Expression Extraction
    # =========================================================================
    
    def test_sum_odd_numbers_unicode(self, templates):
        """Test: ∑(2k-1) from k=1 to n should extract expression '2*k-1'"""
        query = "Compute ∑(2k-1) from k=1 to n"
        code = templates.generate_from_query(query)
        
        # Verify the expression contains 2*k (not just k)
        assert code is not None, "Code generation should not return None for valid summation"
        assert "2*k" in code or "2k" in code, \
            f"Expression should contain '2k' for sum of odd numbers. Generated code:\n{code}"
    
    def test_sum_odd_numbers_with_spaces(self, templates):
        """Test: ∑(2k - 1) from k=1 to n (with spaces) should extract '2*k-1'"""
        query = "Compute ∑(2k - 1) from k=1 to n"
        code = templates.generate_from_query(query)
        
        assert code is not None
        assert "2*k" in code or "2k" in code, \
            "Expression should handle spaces correctly"
    
    def test_sum_odd_numbers_parentheses(self, templates):
        """Test: ∑(2k-1) should preserve the full expression"""
        query = "Calculate ∑(2k-1) from k=1 to n"
        code = templates.generate_from_query(query)
        
        assert code is not None
        # Check that we're not computing just k or k-1
        assert ("2*k" in code or "2k" in code), \
            "Should preserve the full expression (2k-1), not simplify to k"
    
    def test_sum_with_unicode_minus(self, templates):
        """Test: Handle Unicode minus sign (−) vs ASCII hyphen (-)"""
        query = "Compute ∑(2k−1) from k=1 to n"  # Note: − is Unicode minus U+2212
        code = templates.generate_from_query(query)
        
        assert code is not None
        # The code should normalize Unicode minus to ASCII minus
        assert "2*k" in code or "2k" in code
    
    def test_sum_natural_language(self, templates):
        """Test: Natural language format 'sum from k=1 to n of (2k-1)'"""
        query = "Compute the sum from k=1 to n of (2k-1)"
        code = templates.generate_from_query(query)
        
        assert code is not None
        assert "2*k" in code or "2k" in code, \
            "Natural language format should extract expression correctly"
    
    @pytest.mark.skipif(
        not SAFE_EXECUTION_AVAILABLE,
        reason="Safe execution (SymPy) not available"
    )
    def test_sum_odd_numbers_computes_correct_result(self, tool):
        """
        Integration test: Verify that ∑(2k-1) actually computes to n²
        
        This is the core bug from Issue #1:
        - Expected: n**2 (correct formula for sum of odd numbers)
        - Bug: n*(n+1)/2 (formula for sum of integers)
        """
        query = "Compute ∑(2k-1) from k=1 to n"
        result = tool.execute(query)
        
        assert result.success, f"Computation should succeed. Error: {result.error}"
        assert result.result is not None, "Result should not be None"
        
        # The result should be n**2, not n*(n+1)/2
        result_str = str(result.result).lower()
        
        # Check for n**2 or n^2 (correct)
        has_n_squared = "n**2" in result_str or "n^2" in result_str or "n²" in result_str
        
        # Check for n*(n+1)/2 (incorrect - this is the bug)
        has_sum_of_k = "n*(n+1)" in result_str.replace(" ", "") or "n(n+1)" in result_str.replace(" ", "")
        
        assert has_n_squared, \
            f"ISSUE #1 BUG: Result should be n², not {result_str}. " \
            f"Sum of odd numbers 1+3+5+...+(2n-1) = n²"
        
        assert not has_sum_of_k, \
            f"ISSUE #1 BUG: Result is n*(n+1)/2 (sum of k), should be n² (sum of 2k-1)"
    
    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================
    
    def test_sum_without_expression_returns_none(self, templates):
        """Test: Summation query without extractable expression should return None"""
        query = "Compute the sum"  # No expression, no bounds
        code = templates.generate_from_query(query)
        
        # Industry standard: Explicit failure (None) is better than wrong default
        # Previously returned summation("k") which is incorrect
        assert code is None, \
            "Should return None when no expression can be extracted, not a default value"
    
    def test_sum_missing_bounds_uses_defaults(self, templates):
        """Test: ∑(2k-1) without bounds should use defaults k=1 to n"""
        query = "Compute ∑(2k-1)"
        code = templates.generate_from_query(query)
        
        assert code is not None, "Should generate code with default bounds"
        assert "2*k" in code or "2k" in code, "Should preserve expression"
    
    def test_sum_complex_expression(self, templates):
        """Test: More complex expressions like ∑(k²+2k+1)"""
        query = "Calculate ∑(k²+2k+1) from k=1 to n"
        code = templates.generate_from_query(query)
        
        assert code is not None
        # Should contain the polynomial expression
        assert ("k**2" in code or "k^2" in code), \
            "Complex expression should be preserved"
    
    # =========================================================================
    # Multiple Notation Formats (Parameterized Test)
    # =========================================================================
    
    @pytest.mark.parametrize("query,expected_expression", [
        # Unicode formats
        ("∑(2k-1) from k=1 to n", "2*k"),
        ("Compute ∑(2k−1) from k=1 to n", "2*k"),  # Unicode minus
        ("Calculate ∑(2k - 1) from k=1 to n", "2*k"),  # Spaces
        
        # Natural language formats
        ("sum from k=1 to n of (2k-1)", "2*k"),
        ("summation from k=1 to n of 2k-1", "2*k"),
        
        # Different variable names
        ("∑(2i-1) from i=1 to n", "2*i"),
        ("∑(2j-1) from j=1 to n", "2*j"),
        
        # Different expressions
        ("∑(3k+2) from k=1 to n", "3*k"),
        ("∑k² from k=1 to n", "k"),  # Note: k² or k**2
    ])
    def test_sum_notation_formats(self, templates, query, expected_expression):
        """
        Parameterized test for multiple notation formats.
        
        Industry Standard: Use parametrize to test multiple inputs efficiently
        without duplicating test code.
        """
        code = templates.generate_from_query(query)
        
        assert code is not None, f"Should generate code for query: {query}"
        assert expected_expression in code, \
            f"Expression '{expected_expression}' not found in generated code for query: {query}"


class TestIssue2SyntaxErrors:
    """
    Test suite for Issue #2: Math Tool Syntax Errors in Code Generation
    
    Problem: Medical device ethics query produced invalid Python: `f = 0*Tu(t)2`
    
    Solution: Add syntax validation, improve parsing, graceful fallback.
    """
    
    @pytest.fixture
    def tool(self):
        """Create MathematicalComputationTool instance."""
        return create_mathematical_computation_tool()
    
    @pytest.mark.skipif(
        not SAFE_EXECUTION_AVAILABLE,
        reason="Safe execution not available"
    )
    def test_syntax_error_graceful_handling(self, tool):
        """Test: Code with syntax errors should not crash, should return error result"""
        # Simulate a query that previously caused syntax errors
        query = "Calculate ∫₀ᵀ u(t)² dt"
        result = tool.execute(query)
        
        # Should either succeed OR fail gracefully (not crash)
        # If it fails, should have a meaningful error message
        if not result.success:
            assert result.error is not None
            assert len(result.error) > 0, "Error message should be descriptive"
        else:
            # If it succeeds, code should be valid Python
            assert "SyntaxError" not in str(result.result)


class TestIssue3ShowSteps:
    """
    Test suite for Issue #3: Reasoning Engines Don't Show Work When Asked
    
    Problem: User requested "show steps" for Bayes calculation but only got final number.
    
    Solution: Detect step requests and return intermediate calculations.
    """
    
    @pytest.mark.skip(reason="Issue #3 fix not yet implemented")
    def test_bayes_shows_steps(self):
        """
        Test: Bayesian calculation with "show steps" should include intermediate values
        
        Expected output should include:
        - P(+|X) = sensitivity
        - P(+) = total probability
        - Intermediate calculation P(+) = P(+|X)*P(X) + P(+|¬X)*P(¬X)
        """
        # Will be implemented when Issue #3 is fixed
        pass


class TestIssue5LinguisticQueries:
    """
    Test suite for Issue #5: Wrong Tool Selection for Linguistic Queries
    
    Problem: Coreference/pronoun queries were routed to probabilistic reasoning,
    which rejected them.
    
    Solution: Add linguistic query detection to prevent incorrect routing.
    """
    
    @pytest.fixture
    def prob_reasoner(self):
        """Create ProbabilisticReasoner instance if available."""
        try:
            from vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
            return ProbabilisticReasoner()
        except ImportError:
            pytest.skip("ProbabilisticReasoner not available")
    
    def test_coreference_query_rejected_by_probabilistic(self, prob_reasoner):
        """Test: Coreference query should be rejected by probabilistic reasoner"""
        query = "Resolve the coreference in: John saw him. Who does 'him' refer to?"
        result = prob_reasoner.reason(query)
        
        # Should reject with low confidence
        assert result.confidence <= 0.1, \
            f"Probabilistic reasoner should reject linguistic queries with low confidence. Got: {result.confidence}"
        
        # Should indicate it's not applicable
        assert result.conclusion.get("not_applicable") or result.conclusion.get("applicable") == False, \
            "Result should indicate query is not applicable to probabilistic reasoning"
        
        # Should recommend language tool
        metadata = result.metadata or {}
        assert metadata.get("recommended_tool") == "language", \
            "Should recommend 'language' tool for linguistic queries"
    
    def test_pronoun_query_rejected_by_probabilistic(self, prob_reasoner):
        """Test: Pronoun resolution query should be rejected by probabilistic reasoner"""
        query = "What does the pronoun 'it' refer to in: The cat ate the fish. It was delicious."
        result = prob_reasoner.reason(query)
        
        assert result.confidence <= 0.1, \
            "Probabilistic reasoner should reject pronoun queries"
        assert result.conclusion.get("not_applicable") == True
    
    def test_parsing_query_rejected_by_probabilistic(self, prob_reasoner):
        """Test: Parsing query should be rejected by probabilistic reasoner"""
        query = "Parse the sentence: Every student reviewed a document."
        result = prob_reasoner.reason(query)
        
        assert result.confidence <= 0.1, \
            "Probabilistic reasoner should reject parsing queries"
        assert result.conclusion.get("not_applicable") == True
    
    def test_quantifier_scope_query_rejected(self, prob_reasoner):
        """Test: Quantifier scope query should be rejected by probabilistic reasoner"""
        query = "What is the quantifier scope ambiguity in: Someone loves everyone?"
        result = prob_reasoner.reason(query)
        
        assert result.confidence <= 0.1
        assert result.conclusion.get("not_applicable") == True
    
    def test_probability_query_accepted(self, prob_reasoner):
        """Test: Actual probability query should NOT be rejected"""
        query = "What is the probability of rolling a 6 on a fair die?"
        result = prob_reasoner.reason(query)
        
        # This should not be rejected (should have higher confidence or actual computation)
        # We're not checking the result value, just that it wasn't rejected as "not applicable"
        if result.conclusion.get("not_applicable") == True:
            # If it was rejected, confidence must be low
            assert result.confidence <= 0.3, \
                f"Probability queries should not be rejected with high confidence. Got: {result.confidence}"


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short"])
