"""
Tests for complex physics detection in query_router.py.

This test suite validates that PhD-level physics/control theory problems like
triple-inverted pendulum Lagrangian mechanics are correctly detected and routed
with appropriate complexity scores and timeouts, rather than being incorrectly
sent to the MATH-FAST-PATH.

Run with:
    pytest tests/test_complex_physics_detection.py -v
"""

import pytest


def matches_any_pattern(query: str, patterns) -> bool:
    """Helper function to check if query matches any pattern in the tuple.
    
    Args:
        query: The query string to test
        patterns: Tuple of compiled regex patterns
        
    Returns:
        True if any pattern matches the query
    """
    return any(pattern.search(query) for pattern in patterns)


class TestComplexPhysicsKeywords:
    """Tests for COMPLEX_PHYSICS_KEYWORDS constant."""

    def test_complex_physics_keywords_exists(self):
        """Test that COMPLEX_PHYSICS_KEYWORDS is defined."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_KEYWORDS
        
        assert COMPLEX_PHYSICS_KEYWORDS is not None
        assert isinstance(COMPLEX_PHYSICS_KEYWORDS, tuple)
        assert len(COMPLEX_PHYSICS_KEYWORDS) > 0

    def test_control_theory_keywords_present(self):
        """Test that control theory keywords are present."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_KEYWORDS
        
        control_terms = {
            "controllability", "observability", "state space", "linearize",
            "lyapunov", "stability analysis"
        }
        present_terms = {kw for kw in COMPLEX_PHYSICS_KEYWORDS if kw in control_terms}
        
        assert len(present_terms) >= 3, (
            f"Expected at least 3 control theory terms, found: {present_terms}"
        )

    def test_pendulum_keywords_present(self):
        """Test that pendulum system keywords are present."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_KEYWORDS
        
        pendulum_terms = {
            "inverted pendulum", "double pendulum", "triple pendulum",
            "coupled pendulum", "cart-pole"
        }
        present_terms = {kw for kw in COMPLEX_PHYSICS_KEYWORDS if kw in pendulum_terms}
        
        assert len(present_terms) >= 3, (
            f"Expected at least 3 pendulum terms, found: {present_terms}"
        )

    def test_lagrangian_mechanics_keywords_present(self):
        """Test that Lagrangian mechanics keywords are present."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_KEYWORDS
        
        lagrangian_terms = {
            "euler-lagrange", "euler lagrange", "hamilton's equations",
            "hamiltonian mechanics", "lagrangian mechanics", "generalized coordinates",
            "canonical transformation", "action principle"
        }
        present_terms = {kw for kw in COMPLEX_PHYSICS_KEYWORDS if kw in lagrangian_terms}
        
        assert len(present_terms) >= 3, (
            f"Expected at least 3 Lagrangian mechanics terms, found: {present_terms}"
        )

    def test_all_keywords_are_lowercase(self):
        """Test that all keywords are lowercase for consistent matching."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_KEYWORDS
        
        for keyword in COMPLEX_PHYSICS_KEYWORDS:
            assert keyword == keyword.lower(), (
                f"Keyword '{keyword}' should be lowercase"
            )


class TestForceFullMathPatterns:
    """Tests for FORCE_FULL_MATH_PATTERNS regex patterns."""

    def test_patterns_exist(self):
        """Test that FORCE_FULL_MATH_PATTERNS is defined."""
        from vulcan.routing.query_router import FORCE_FULL_MATH_PATTERNS
        
        assert FORCE_FULL_MATH_PATTERNS is not None
        assert isinstance(FORCE_FULL_MATH_PATTERNS, tuple)
        assert len(FORCE_FULL_MATH_PATTERNS) > 0

    def test_derive_equations_of_motion_pattern(self):
        """Test pattern for 'derive equations of motion'."""
        from vulcan.routing.query_router import FORCE_FULL_MATH_PATTERNS
        
        test_queries = [
            "Derive the equations of motion for a triple pendulum",
            "derive equation of motion",
            "Derive the dynamics of the system",
        ]
        
        for query in test_queries:
            assert matches_any_pattern(query, FORCE_FULL_MATH_PATTERNS), (
                f"Pattern should match: '{query}'"
            )

    def test_controllability_proof_pattern(self):
        """Test pattern for controllability proofs."""
        from vulcan.routing.query_router import FORCE_FULL_MATH_PATTERNS
        
        test_queries = [
            "Prove controllability of the linearized system",
            "Show controllability of the state space model",
            "Demonstrate controllability for this system",
        ]
        
        for query in test_queries:
            assert matches_any_pattern(query, FORCE_FULL_MATH_PATTERNS), (
                f"Pattern should match: '{query}'"
            )

    def test_lagrangian_formula_pattern(self):
        """Test pattern for L = T - V formula."""
        from vulcan.routing.query_router import FORCE_FULL_MATH_PATTERNS
        
        test_queries = [
            "The Lagrangian is L = T - V",
            "Given L=T-V, derive the equations",
            "Using L = T - V",
        ]
        
        for query in test_queries:
            assert matches_any_pattern(query, FORCE_FULL_MATH_PATTERNS), (
                f"Pattern should match: '{query}'"
            )

    def test_eigenvalue_analysis_pattern(self):
        """Test pattern for eigenvalue analysis."""
        from vulcan.routing.query_router import FORCE_FULL_MATH_PATTERNS
        
        test_queries = [
            "Find the eigenvalues of the state matrix",
            "Calculate the eigenvectors",
            "Compute eigenvalues for the linearized system",
        ]
        
        for query in test_queries:
            assert matches_any_pattern(query, FORCE_FULL_MATH_PATTERNS), (
                f"Pattern should match: '{query}'"
            )


class TestComplexPhysicsDetection:
    """Tests for _is_complex_physics_query method."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_triple_pendulum_query_detected(self, query_analyzer):
        """Test that triple pendulum query is detected as complex physics."""
        query = (
            "Derive the equations of motion for a triple inverted pendulum "
            "using Lagrangian mechanics"
        )
        
        assert query_analyzer._is_complex_physics_query(query), (
            "Triple pendulum Lagrangian query should be detected as complex physics"
        )

    def test_control_theory_query_detected(self, query_analyzer):
        """Test that control theory query is detected as complex physics."""
        query = "Prove controllability of the linearized system around the equilibrium point"
        
        assert query_analyzer._is_complex_physics_query(query), (
            "Control theory query should be detected as complex physics"
        )

    def test_eigenvalue_analysis_detected(self, query_analyzer):
        """Test that eigenvalue analysis is detected as complex physics."""
        query = "Analyze the stability of the system by computing the eigenvalues of the state matrix"
        
        assert query_analyzer._is_complex_physics_query(query), (
            "Eigenvalue analysis query should be detected as complex physics"
        )

    def test_lyapunov_stability_detected(self, query_analyzer):
        """Test that Lyapunov stability analysis is detected."""
        query = "Use Lyapunov's method to prove stability of the nonlinear system"
        
        assert query_analyzer._is_complex_physics_query(query), (
            "Lyapunov stability query should be detected as complex physics"
        )

    def test_inverted_pendulum_with_derive_detected(self, query_analyzer):
        """Test that inverted pendulum with derivation is detected."""
        query = "Derive the state space model for an inverted pendulum on a cart"
        
        assert query_analyzer._is_complex_physics_query(query), (
            "Inverted pendulum state space query should be detected as complex physics"
        )

    def test_simple_pendulum_not_detected(self, query_analyzer):
        """Test that simple pendulum (without advanced analysis) is NOT detected."""
        query = "What is the period of a simple pendulum?"
        
        # This should NOT be complex physics - it's basic mechanics
        is_complex = query_analyzer._is_complex_physics_query(query)
        assert not is_complex, (
            "Simple pendulum period question should NOT be detected as complex physics"
        )

    def test_simple_probability_not_detected(self, query_analyzer):
        """Test that simple probability query is NOT detected as complex physics."""
        query = "What is the probability of rolling a 6 on a fair die?"
        
        is_complex = query_analyzer._is_complex_physics_query(query)
        assert not is_complex, (
            "Simple probability question should NOT be detected as complex physics"
        )


class TestMathFastPathExclusion:
    """Tests that complex physics queries are excluded from math fast-path."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_complex_physics_excluded_from_math_fast_path(self, query_analyzer):
        """Test that complex physics queries return False from _is_mathematical_query."""
        query = (
            "Derive the equations of motion for a triple inverted pendulum "
            "using Lagrangian mechanics"
        )
        
        # This query contains "lagrangian" which is in MATHEMATICAL_KEYWORDS,
        # but should be excluded because it's complex physics
        is_math = query_analyzer._is_mathematical_query(query)
        
        assert not is_math, (
            "Complex physics query should NOT use math fast-path. "
            "_is_mathematical_query should return False."
        )

    def test_simple_bayesian_uses_math_fast_path(self, query_analyzer):
        """Test that simple Bayesian queries still use math fast-path."""
        query = (
            "What is the probability that someone has a disease given a "
            "positive test result with 95% sensitivity and 1% prevalence?"
        )
        
        is_math = query_analyzer._is_mathematical_query(query)
        
        assert is_math, (
            "Simple Bayesian probability query SHOULD use math fast-path"
        )


class TestComplexPhysicsComplexityScoring:
    """Tests for complexity scoring with complex physics queries."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_complex_physics_high_complexity_score(self, query_analyzer):
        """Test that complex physics queries get complexity >= 0.80."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_MIN_COMPLEXITY
        
        query = (
            "Derive the equations of motion for a triple inverted pendulum "
            "using Lagrangian mechanics and linearize around the equilibrium"
        )
        
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= COMPLEX_PHYSICS_MIN_COMPLEXITY, (
            f"Complex physics query should have complexity >= {COMPLEX_PHYSICS_MIN_COMPLEXITY}, "
            f"got {complexity}"
        )

    def test_controllability_high_complexity_score(self, query_analyzer):
        """Test that controllability analysis gets high complexity."""
        query = "Prove controllability of the linearized state space model"
        
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.5, (
            f"Controllability query should have high complexity, got {complexity}"
        )

    def test_eigenvalue_analysis_high_complexity_score(self, query_analyzer):
        """Test that eigenvalue analysis gets high complexity."""
        query = "Compute the eigenvalues of the 8x8 state matrix for stability analysis"
        
        complexity = query_analyzer._calculate_complexity(query.lower())
        
        assert complexity >= 0.5, (
            f"Eigenvalue analysis query should have high complexity, got {complexity}"
        )


class TestComplexPhysicsRouting:
    """Integration tests for complex physics routing."""

    @pytest.fixture
    def query_analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        from vulcan.routing.query_router import QueryAnalyzer
        return QueryAnalyzer(enable_safety_validation=False)

    def test_triple_pendulum_route_not_math_fast_path(self, query_analyzer):
        """Test that triple pendulum query does NOT use MATH-FAST-PATH."""
        query = (
            "Derive the equations of motion for a triple inverted pendulum "
            "using Lagrangian mechanics. Include the mass matrix and prove controllability."
        )
        
        plan = query_analyzer.route_query(query, source="user")
        
        # Check that this is NOT the math fast-path
        # Math fast-path would have complexity=0.30 and short timeout
        assert plan.complexity_score >= 0.5, (
            f"Complex physics query should have complexity >= 0.5, got {plan.complexity_score}"
        )
        
        # Check telemetry data for path type
        assert plan.telemetry_data.get("math_fast_path") != True, (
            "Complex physics query should NOT be on math_fast_path"
        )

    def test_complex_physics_extended_timeout(self, query_analyzer):
        """Test that complex physics queries get extended timeout."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_TIMEOUT_SECONDS
        
        query = "Derive the equations of motion for a triple inverted pendulum using Lagrangian"
        
        plan = query_analyzer.route_query(query, source="user")
        
        # Check for complex physics path indicators
        if plan.telemetry_data.get("complex_physics_path"):
            # Should have extended timeout
            timeout = plan.telemetry_data.get("timeout_seconds", 0)
            assert timeout >= COMPLEX_PHYSICS_TIMEOUT_SECONDS, (
                f"Complex physics should have timeout >= {COMPLEX_PHYSICS_TIMEOUT_SECONDS}s, "
                f"got {timeout}s"
            )

    def test_complex_physics_activates_all_tools(self, query_analyzer):
        """Test that complex physics queries activate all mathematical tools."""
        query = "Derive the state space model and prove controllability for a double pendulum"
        
        plan = query_analyzer.route_query(query, source="user")
        
        # Check if complex physics path was used
        if plan.telemetry_data.get("complex_physics_path"):
            selected_tools = plan.telemetry_data.get("selected_tools", [])
            
            expected_tools = {"symbolic", "mathematical", "probabilistic", "causal", "analogical"}
            actual_tools = set(selected_tools)
            
            assert expected_tools.issubset(actual_tools), (
                f"Complex physics should activate all tools. Expected {expected_tools}, "
                f"got {actual_tools}"
            )


class TestComplexPhysicsConstants:
    """Tests for complex physics constants."""

    def test_complex_physics_timeout_is_120_plus(self):
        """Test that COMPLEX_PHYSICS_TIMEOUT_SECONDS is at least 120."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_TIMEOUT_SECONDS
        
        assert COMPLEX_PHYSICS_TIMEOUT_SECONDS >= 120.0, (
            f"Complex physics timeout should be >= 120s, got {COMPLEX_PHYSICS_TIMEOUT_SECONDS}s"
        )

    def test_complex_physics_min_complexity_is_080_plus(self):
        """Test that COMPLEX_PHYSICS_MIN_COMPLEXITY is at least 0.80."""
        from vulcan.routing.query_router import COMPLEX_PHYSICS_MIN_COMPLEXITY
        
        assert COMPLEX_PHYSICS_MIN_COMPLEXITY >= 0.80, (
            f"Complex physics min complexity should be >= 0.80, got {COMPLEX_PHYSICS_MIN_COMPLEXITY}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
