"""
Tests for reasoning complexity indicators in query_router.py.

This test suite validates that queries with reasoning-related terms get a complexity
boost, ensuring they don't hit the fast-path in reasoning_integration.py and bypass
the ToolSelector.

Run with:
    pytest tests/test_reasoning_complexity_indicators.py -v
"""

import pytest


class TestReasoningComplexityIndicators:
    """Tests for REASONING_COMPLEXITY_INDICATORS and complexity boost logic."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_causal_reasoning_indicators_boost_complexity(self, query_analyzer):
        """Test that causal reasoning keywords boost complexity above 0.3 threshold."""
        # The butterfly query with "causal" explicitly mentioned
        query = "Explain the causal relationship between events"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Should be above FAST_PATH_COMPLEXITY_THRESHOLD (0.3) due to "causal" keyword
        assert complexity >= 0.15, (
            f"Causal query should have complexity >= 0.15, got {complexity}. "
            "This query should receive a reasoning boost."
        )

    def test_effect_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'effect' keyword boosts complexity."""
        query = "What is the effect of climate change on polar ice?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'effect' should have boosted complexity, got {complexity}"
        )

    def test_intervention_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'intervention' keyword boosts complexity."""
        query = "What intervention would prevent this outcome?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'intervention' should have boosted complexity, got {complexity}"
        )

    def test_counterfactual_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'counterfactual' keyword boosts complexity."""
        query = "Consider the counterfactual scenario where X never happened"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'counterfactual' should have boosted complexity, got {complexity}"
        )

    def test_probabilistic_reasoning_indicators(self, query_analyzer):
        """Test that probabilistic reasoning keywords boost complexity."""
        query = "What is the probability that this event occurs given prior evidence?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Should have boost from "probability" and "prior"
        assert complexity >= 0.3, (
            f"Probabilistic query should have complexity >= 0.3, got {complexity}"
        )

    def test_bayesian_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'bayesian' keyword boosts complexity."""
        query = "Use bayesian inference to update our beliefs"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'bayesian' should have boosted complexity, got {complexity}"
        )

    def test_likelihood_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'likelihood' keyword boosts complexity."""
        query = "What is the likelihood of success?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'likelihood' should have boosted complexity, got {complexity}"
        )

    def test_analogical_reasoning_indicators(self, query_analyzer):
        """Test that analogical reasoning keywords boost complexity."""
        query = "This is similar to how a computer processes data"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Should have boost from "similar to"
        assert complexity >= 0.15, (
            f"Analogical query should have boosted complexity, got {complexity}"
        )

    def test_analogy_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'analogy' keyword boosts complexity."""
        query = "Draw an analogy between these two concepts"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'analogy' should have boosted complexity, got {complexity}"
        )

    def test_symbolic_reasoning_indicators(self, query_analyzer):
        """Test that symbolic reasoning keywords boost complexity."""
        query = "Prove this theorem using formal logic"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Should have boost from "prove", "theorem", "logic"
        assert complexity >= 0.3, (
            f"Symbolic reasoning query should have complexity >= 0.3, got {complexity}"
        )

    def test_deduce_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'deduce' keyword boosts complexity."""
        query = "From these premises, deduce the conclusion"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'deduce' should have boosted complexity, got {complexity}"
        )

    def test_therefore_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'therefore' keyword boosts complexity."""
        query = "All men are mortal, Socrates is a man, therefore Socrates is mortal"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'therefore' should have boosted complexity, got {complexity}"
        )

    def test_implies_keyword_boosts_complexity(self, query_analyzer):
        """Test that 'implies' keyword boosts complexity."""
        query = "This condition implies that X must be true"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.15, (
            f"Query with 'implies' should have boosted complexity, got {complexity}"
        )

    def test_multiple_reasoning_indicators_cumulative_boost(self, query_analyzer):
        """Test that multiple reasoning indicators provide cumulative boost."""
        query = "What is the probability of cause and effect given bayesian inference?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Should have boost from "probability", "cause", "effect", "bayesian", "inference"
        # 5 indicators * 0.15 = 0.75, capped at 0.4
        assert complexity >= 0.4, (
            f"Multiple reasoning indicators should provide significant boost, got {complexity}"
        )

    def test_simple_query_still_has_low_complexity(self, query_analyzer):
        """Test that simple queries without reasoning terms still have low complexity."""
        query = "What is the weather today?"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        # Simple query should have low complexity (word count only)
        assert complexity < 0.3, (
            f"Simple query should have complexity < 0.3, got {complexity}"
        )

    def test_greeting_has_zero_complexity(self, query_analyzer):
        """Test that simple greetings have minimal complexity."""
        query = "hello"
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity == 0.0, (
            f"Simple greeting should have complexity 0.0, got {complexity}"
        )


class TestReasoningComplexityIntegration:
    """Integration tests for reasoning complexity with route_query."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_causal_query_above_fast_path_threshold(self, query_analyzer):
        """Test that causal queries with multiple indicators have complexity above 0.3 fast-path threshold."""
        query = "What causes the relationship between X and Y? Explain the causal chain."
        plan = query_analyzer.route_query(query, source="user")
        
        # From reasoning_integration.py: FAST_PATH_COMPLEXITY_THRESHOLD = 0.3
        # This query has "causes" and "causal" indicators, so should exceed 0.3
        assert plan.complexity_score >= 0.3, (
            f"Causal query should have complexity >= 0.3 to avoid fast-path bypass. "
            f"Got {plan.complexity_score}"
        )

    def test_butterfly_causal_query(self, query_analyzer):
        """Test the specific butterfly query from the problem statement."""
        query = (
            "A butterfly flaps its wings in Brazil, causing a tornado in Texas. "
            "Explain the causal chain."
        )
        plan = query_analyzer.route_query(query, source="user")
        
        # This query should now have complexity >= 0.3 due to:
        # - "cause" indicator (+0.15)
        # - "causal" indicator (+0.15)
        # - word count boost
        assert plan.complexity_score >= 0.3, (
            f"Butterfly causal query should NOT hit fast-path. "
            f"Complexity: {plan.complexity_score}, expected >= 0.3"
        )

    def test_probabilistic_query_above_threshold(self, query_analyzer):
        """Test that probabilistic queries have sufficient complexity."""
        query = "Given prior evidence, calculate the posterior probability"
        plan = query_analyzer.route_query(query, source="user")
        
        assert plan.complexity_score >= 0.3, (
            f"Probabilistic query should have complexity >= 0.3. "
            f"Got {plan.complexity_score}"
        )

    def test_symbolic_reasoning_query_above_threshold(self, query_analyzer):
        """Test that symbolic reasoning queries have sufficient complexity."""
        query = "Prove this theorem using deductive logic"
        plan = query_analyzer.route_query(query, source="user")
        
        assert plan.complexity_score >= 0.3, (
            f"Symbolic reasoning query should have complexity >= 0.3. "
            f"Got {plan.complexity_score}"
        )


class TestReasoningComplexityIndicatorsConstant:
    """Tests to validate the REASONING_COMPLEXITY_INDICATORS constant is properly defined."""

    def test_reasoning_indicators_tuple_exists(self):
        """Test that REASONING_COMPLEXITY_INDICATORS is defined."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        assert REASONING_COMPLEXITY_INDICATORS is not None
        assert isinstance(REASONING_COMPLEXITY_INDICATORS, tuple)

    def test_reasoning_indicators_not_empty(self):
        """Test that REASONING_COMPLEXITY_INDICATORS has entries."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        assert len(REASONING_COMPLEXITY_INDICATORS) > 0

    def test_causal_indicators_present(self):
        """Test that causal reasoning indicators are present."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        causal_terms = {"causal", "cause", "causes", "caused", "causing", "effect", "effects", "intervention", "counterfactual"}
        present_terms = set(REASONING_COMPLEXITY_INDICATORS) & causal_terms
        
        assert len(present_terms) >= 5, (
            f"Expected at least 5 causal terms (including verb forms), found: {present_terms}"
        )

    def test_probabilistic_indicators_present(self):
        """Test that probabilistic reasoning indicators are present."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        prob_terms = {"probability", "bayesian", "likelihood", "posterior", "prior"}
        present_terms = set(REASONING_COMPLEXITY_INDICATORS) & prob_terms
        
        assert len(present_terms) >= 3, (
            f"Expected at least 3 probabilistic terms, found: {present_terms}"
        )

    def test_analogical_indicators_present(self):
        """Test that analogical reasoning indicators are present."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        analog_terms = {"analogy", "analogous", "similar to", "mapping"}
        present_terms = set(REASONING_COMPLEXITY_INDICATORS) & analog_terms
        
        assert len(present_terms) >= 2, (
            f"Expected at least 2 analogical terms, found: {present_terms}"
        )

    def test_symbolic_indicators_present(self):
        """Test that symbolic reasoning indicators are present."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        symbolic_terms = {"prove", "theorem", "logic", "deduce", "axiom"}
        present_terms = set(REASONING_COMPLEXITY_INDICATORS) & symbolic_terms
        
        assert len(present_terms) >= 3, (
            f"Expected at least 3 symbolic terms, found: {present_terms}"
        )

    def test_all_indicators_are_lowercase(self):
        """Test that all indicators are lowercase for consistent matching."""
        from vulcan.routing.query_router import REASONING_COMPLEXITY_INDICATORS
        
        for indicator in REASONING_COMPLEXITY_INDICATORS:
            assert indicator == indicator.lower(), (
                f"Indicator '{indicator}' should be lowercase"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
