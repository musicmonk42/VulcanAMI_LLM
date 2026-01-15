"""
Test suite for reasoning query routing fixes.

Tests the three fixes implemented to ensure reasoning queries 
route correctly and don't return boilerplate responses.

Industry Standard: Comprehensive test coverage with clear pass/fail criteria.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


class TestFix1QueryRouterReasoningDomainDetection:
    """
    Test Fix 1: Query router correctly identifies reasoning domain queries
    and doesn't override them with philosophical routing.
    """
    
    def test_sat_query_not_philosophical(self):
        """SAT query should NOT be classified as philosophical."""
        from vulcan.routing.query_router import QueryRouter
        
        router = QueryRouter()
        query = "Is A→B, B→C, ¬C, A∨B satisfiable?"
        
        # Should NOT be classified as philosophical
        is_philosophical = router._is_philosophical_query(query)
        assert not is_philosophical, (
            f"SAT query incorrectly classified as philosophical. "
            f"Expected False, got {is_philosophical}"
        )
    
    def test_bayesian_query_not_philosophical(self):
        """Bayesian query should NOT be classified as philosophical."""
        from vulcan.routing.query_router import QueryRouter
        
        router = QueryRouter()
        query = "P(disease|positive test) with sensitivity=0.99, specificity=0.95, prevalence=0.01"
        
        is_philosophical = router._is_philosophical_query(query)
        assert not is_philosophical, (
            f"Bayesian query incorrectly classified as philosophical. "
            f"Expected False, got {is_philosophical}"
        )
    
    def test_causal_query_not_philosophical(self):
        """Causal inference query should NOT be classified as philosophical."""
        from vulcan.routing.query_router import QueryRouter
        
        router = QueryRouter()
        query = "Does conditioning on B induce correlation between A and C in graph A→B←C?"
        
        is_philosophical = router._is_philosophical_query(query)
        assert not is_philosophical, (
            f"Causal query incorrectly classified as philosophical. "
            f"Expected False, got {is_philosophical}"
        )
    
    def test_fol_query_not_philosophical(self):
        """First-order logic query should NOT be classified as philosophical."""
        from vulcan.routing.query_router import QueryRouter
        
        router = QueryRouter()
        query = "∀X (human(X) → mortal(X)), human(socrates) ⊢ mortal(socrates)?"
        
        is_philosophical = router._is_philosophical_query(query)
        assert not is_philosophical, (
            f"FOL query incorrectly classified as philosophical. "
            f"Expected False, got {is_philosophical}"
        )
    
    def test_actual_philosophical_query_is_philosophical(self):
        """Actual philosophical query should still be classified correctly."""
        from vulcan.routing.query_router import QueryRouter
        
        router = QueryRouter()
        query = "What is the meaning of life?"
        
        is_philosophical = router._is_philosophical_query(query)
        assert is_philosophical, (
            f"Genuine philosophical query not classified as philosophical. "
            f"Expected True, got {is_philosophical}"
        )


class TestFix2WorldModelMetaReasoningIntegration:
    """
    Test Fix 2: World model uses actual meta-reasoning components
    instead of returning templates.
    """
    
    def test_philosophical_reasoning_method_exists(self):
        """_philosophical_reasoning method should exist and be callable."""
        from vulcan.world_model.world_model_core import WorldModel
        
        # Check method exists
        assert hasattr(WorldModel, '_philosophical_reasoning'), (
            "_philosophical_reasoning method not found"
        )
        
        # Check it's callable
        assert callable(getattr(WorldModel, '_philosophical_reasoning')), (
            "_philosophical_reasoning is not callable"
        )
    
    def test_philosophical_reasoning_has_helper_methods(self):
        """Helper methods for ethical parsing should exist."""
        from vulcan.world_model.world_model_core import WorldModel
        
        required_helpers = [
            '_parse_ethical_query_structure',
            '_run_ethical_boundary_analysis',
            '_detect_goal_conflicts_in_query',
            '_analyze_option_counterfactuals',
            '_synthesize_ethical_response',
            '_generate_philosophical_template'
        ]
        
        for helper in required_helpers:
            assert hasattr(WorldModel, helper), (
                f"Required helper method {helper} not found"
            )
    
    def test_philosophical_reasoning_returns_correct_structure(self):
        """_philosophical_reasoning should return dict with required keys."""
        from vulcan.world_model.world_model_core import WorldModel, check_component_availability
        
        # Initialize component availability
        check_component_availability()
        
        # Create world model (minimal config)
        wm = WorldModel(config={})
        
        # Call with trolley problem
        result = wm._philosophical_reasoning(
            "Should you pull the lever to save 5 people but kill 1?"
        )
        
        # Check structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'response' in result, "Result missing 'response' key"
        assert 'confidence' in result, "Result missing 'confidence' key"
        assert 'reasoning_trace' in result, "Result missing 'reasoning_trace' key"
        
        # Check confidence is reasonable
        assert 0.0 <= result['confidence'] <= 1.0, (
            f"Confidence should be between 0 and 1, got {result['confidence']}"
        )
        
        # Check response is not empty
        assert len(result['response']) > 0, "Response should not be empty"


class TestFix3AgentPoolTemplateDetection:
    """
    Test Fix 3: Agent pool correctly detects template responses
    and allows fallback to specialized reasoning engines.
    """
    
    def test_template_detection_for_boilerplate(self):
        """Boilerplate responses should be detected as templates."""
        from vulcan.orchestrator.agent_pool import _is_privileged_result
        
        # Simulate world_model result with boilerplate
        result = {
            'selected_tools': ['world_model'],
            'response': "This is a philosophical question requiring reasoned analysis. I'll analyze this using multiple ethical frameworks: Consequentialist: What outcomes matter?",
            'metadata': {},
            'reasoning_strategy': ''
        }
        
        is_privileged = _is_privileged_result(result)
        assert not is_privileged, (
            "Boilerplate response incorrectly marked as privileged. "
            "Expected False to allow fallback, got True"
        )
    
    def test_substantive_response_is_privileged(self):
        """Substantive world_model responses should be privileged."""
        from vulcan.orchestrator.agent_pool import _is_privileged_result
        
        # Simulate world_model result with substantive content
        result = {
            'selected_tools': ['world_model'],
            'response': "Based on goal conflict analysis between minimize_deaths and avoid_direct_harm, the utilitarian framework suggests action X while deontological analysis indicates Y. The ethical boundary check reveals...",
            'metadata': {},
            'reasoning_strategy': ''
        }
        
        is_privileged = _is_privileged_result(result)
        assert is_privileged, (
            "Substantive world_model response incorrectly marked as non-privileged. "
            "Expected True, got False"
        )
    
    def test_self_introspection_always_privileged(self):
        """Self-introspection queries should always be privileged."""
        from vulcan.orchestrator.agent_pool import _is_privileged_result
        
        result = {
            'selected_tools': ['world_model'],
            'response': "Any response",  # Content doesn't matter
            'metadata': {'is_self_introspection': True},
            'reasoning_strategy': ''
        }
        
        is_privileged = _is_privileged_result(result)
        assert is_privileged, (
            "Self-introspection query should be privileged regardless of content"
        )


class TestEndToEndIntegration:
    """
    Test end-to-end integration: SAT query should route to symbolic reasoner
    and return actual answer, not boilerplate.
    """
    
    @pytest.mark.skip(reason="Requires full system setup - manual verification only")
    def test_sat_query_returns_actual_answer(self):
        """
        SAT query should return UNSAT with proof, not boilerplate.
        
        This is an integration test requiring full system initialization.
        It's marked for manual verification only.
        """
        # This would require:
        # 1. Full QueryRouter initialization
        # 2. UnifiedReasoner with SymbolicReasoner
        # 3. Agent pool setup
        # 4. End-to-end query processing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
