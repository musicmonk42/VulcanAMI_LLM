"""
Tests for LLM-based physics query routing.

Updated to work with LLM-based routing (removed regex constants and methods).
"""

import pytest


class TestComplexPhysicsRouting:
    """Tests for LLM-based complex physics query routing."""

    def test_lagrangian_mechanics_routes_to_mathematical(self):
        """Test that Lagrangian mechanics query routes to mathematical engine."""
        from vulcan.routing.llm_router import get_llm_router
        
        router = get_llm_router()
        query = "Derive Euler-Lagrange equations for triple inverted pendulum"
        
        decision = router.route(query)
        assert decision.destination == "reasoning_engine"
        assert decision.engine in ["mathematical", "symbolic"]

    def test_control_theory_routes_correctly(self):
        """Test that control theory query routes to appropriate engine."""
        from vulcan.routing.llm_router import get_llm_router
        
        router = get_llm_router()
        query = "Analyze controllability matrix for state space system"
        
        decision = router.route(query)
        assert decision.destination == "reasoning_engine"
        assert decision.engine in ["mathematical", "symbolic", "causal"]

    def test_pendulum_dynamics_routes_to_mathematical(self):
        """Test that pendulum dynamics query routes to mathematical engine."""
        from vulcan.routing.llm_router import get_llm_router
        
        router = get_llm_router()
        query = "What are the equations of motion for a double pendulum system?"
        
        decision = router.route(query)
        assert decision.destination == "reasoning_engine"
        assert decision.engine in ["mathematical", "symbolic"]


class TestLLMPhysicsClassification:
    """Test that LLM correctly classifies physics queries."""
    
    def test_complex_physics_not_simple_math(self):
        """Complex physics should route to reasoning engines."""
        from vulcan.routing.llm_router import get_llm_router
        
        router = get_llm_router()
        query = "Derive Lagrangian for coupled oscillators"
        
        decision = router.route(query)
        assert decision.destination == "reasoning_engine"
        assert decision.confidence > 0.5
