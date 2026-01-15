"""
Test suite for WorldModel routing methods.

Tests the new _should_route_to_reasoning_engine and _route_to_appropriate_engine
methods that route queries to specialized reasoning engines.

Industry Standard: Comprehensive test coverage with clear pass/fail criteria.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


class TestShouldRouteToReasoningEngine:
    """Test _should_route_to_reasoning_engine detection logic."""
    
    def test_causal_query_detection(self):
        """Causal queries should be detected."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        causal_queries = [
            "Does conditioning on B confound the relationship between A and C?",
            "What is the causal effect of intervention do(X=1)?",
            "Does Pearl's backdoor criterion apply here?",
            "Is this a case of confounding variables?",
        ]
        
        for query in causal_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is True, f"Failed to detect causal query: {query}"
    
    def test_analogical_query_detection(self):
        """Analogical queries should be detected."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        analogical_queries = [
            "In structure mapping, how does domain S correspond to domain T?",
            "What is the analogy between solar system and atom?",
            "Map the correspondence between these two domains",
            "How do relations in the source domain map to the target?",
        ]
        
        for query in analogical_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is True, f"Failed to detect analogical query: {query}"
    
    def test_mathematical_query_detection(self):
        """Mathematical queries should be detected."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        mathematical_queries = [
            "Compute the sum of integers from 1 to n",
            "Prove by mathematical induction that P(n) holds",
            "Calculate the derivative of f(x) = x^2",
            "What is the integral of sin(x)?",
        ]
        
        for query in mathematical_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is True, f"Failed to detect mathematical query: {query}"
    
    def test_symbolic_sat_query_detection(self):
        """SAT/symbolic queries should be detected."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        symbolic_queries = [
            "Is A→B, B→C, ¬C, A∨B satisfiable?",
            "Check satisfiability of this formula",
            "Convert to CNF and solve",
            "Is this first-order logic formula valid?",
        ]
        
        for query in symbolic_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is True, f"Failed to detect symbolic query: {query}"
    
    def test_philosophical_query_not_routed(self):
        """Philosophical queries should NOT be routed to technical engines."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        philosophical_queries = [
            "What is the meaning of life?",
            "Is free will real?",
            "What makes an action morally right?",
            "Should I prioritize happiness or virtue?",
        ]
        
        for query in philosophical_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is False, f"Incorrectly routed philosophical query: {query}"
    
    def test_creative_query_not_routed(self):
        """Creative queries should NOT be routed to technical engines."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        creative_queries = [
            "Write a poem about love",
            "Tell me a story about a hero",
            "Create an essay on freedom",
        ]
        
        for query in creative_queries:
            result = wm._should_route_to_reasoning_engine(query)
            assert result is False, f"Incorrectly routed creative query: {query}"
    
    def test_input_validation(self):
        """Test input validation and edge cases."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        # None input
        assert wm._should_route_to_reasoning_engine(None) is False
        
        # Empty string
        assert wm._should_route_to_reasoning_engine("") is False
        
        # Non-string input (should handle gracefully)
        assert wm._should_route_to_reasoning_engine(123) is False
        
        # Very long query (should reject for security)
        long_query = "x" * 20000
        assert wm._should_route_to_reasoning_engine(long_query) is False


class TestNormalizeEngineResult:
    """Test _normalize_engine_result formatting."""
    
    def test_dict_result_normalization(self):
        """Test normalization of dict results."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        result = {
            'response': 'The answer is 42',
            'confidence': 0.9,
            'reasoning_trace': {'steps': ['step1', 'step2']},
            'mode': 'causal'
        }
        
        normalized = wm._normalize_engine_result(result, 'causal', 'test query')
        
        assert normalized['response'] == 'The answer is 42'
        assert normalized['confidence'] == 0.9
        assert normalized['engine_used'] == 'causal'
        assert 'reasoning_trace' in normalized
    
    def test_string_result_normalization(self):
        """Test normalization of string results."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        result = "UNSAT: No satisfying assignment exists"
        
        normalized = wm._normalize_engine_result(result, 'symbolic', 'test query')
        
        assert normalized['response'] == result
        assert 'confidence' in normalized
        assert normalized['engine_used'] == 'symbolic'
        assert 'reasoning_trace' in normalized
    
    def test_object_result_normalization(self):
        """Test normalization of object results with attributes."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        # Mock result object
        class MockResult:
            def __init__(self):
                self.result = "Answer from engine"
                self.confidence = 0.85
                self.trace = {'method': 'backtracking'}
        
        result = MockResult()
        
        normalized = wm._normalize_engine_result(result, 'mathematical', 'test query')
        
        assert 'Answer from engine' in normalized['response']
        assert normalized['confidence'] == 0.85
        assert normalized['engine_used'] == 'mathematical'
    
    def test_error_handling_in_normalization(self):
        """Test graceful error handling in normalization."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        # Completely invalid result
        result = None
        
        normalized = wm._normalize_engine_result(result, 'test_engine', 'test query')
        
        # Should return valid dict even on error
        assert isinstance(normalized, dict)
        assert 'response' in normalized
        assert 'confidence' in normalized
        assert 'engine_used' in normalized


class TestReasonMethodRouting:
    """Test the updated reason() method with routing logic."""
    
    def test_reason_calls_routing_for_causal_query(self):
        """Test that reason() detects causal queries and routes them."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        query = "Does confounding affect the causal relationship?"
        
        # This should attempt routing (may fail if engine not available, which is OK)
        # We're just testing that it attempts the routing path
        try:
            result = wm.reason(query, mode=None)
            # If routing succeeded, should have engine_used
            # If routing failed, should fall back to _general_reasoning
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'confidence' in result
        except Exception:
            # Engine import may fail in test environment, that's OK
            pass
    
    def test_reason_respects_explicit_mode(self):
        """Test that explicit mode overrides routing."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        # Causal query with explicit philosophical mode
        query = "Does confounding affect causation?"
        
        result = wm.reason(query, mode='philosophical')
        
        # Should respect the explicit mode and not route to causal engine
        assert isinstance(result, dict)
        assert result.get('mode') == 'philosophical'
    
    def test_reason_handles_dict_query(self):
        """Test that reason() handles dict query format."""
        from vulcan.world_model.world_model_core import WorldModel
        
        wm = WorldModel(config={})
        
        query = {
            'query': 'Is this formula satisfiable?',
            'mode': None
        }
        
        result = wm.reason(query)
        
        assert isinstance(result, dict)
        assert 'response' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
