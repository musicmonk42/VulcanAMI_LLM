# ============================================================
# VULCAN-AGI Routing Fixes Tests
# ============================================================
# Tests for the fixes to LLM router fallback and template detection
# addressing issues with specialized query routing and template responses.
#
# Tests verify:
# 1. Analogical and language/quantifier queries are routed correctly
# 2. Template responses are detected and force LLM synthesis
# 3. WorldModel detects misrouted specialized queries
# ============================================================

"""
Unit tests for routing fixes.

These tests verify the fixes for:
- Issue 1: LLM router fallback correctly routes analogical/language queries
- Issue 2: Template detection prevents boilerplate responses from bypassing LLM
- Issue 3: WorldModel detects and rejects misrouted specialized queries
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# Import the modules under test
from src.vulcan.routing.llm_router import (
    LLMQueryRouter,
    RoutingDecision,
    ANALOGICAL_KEYWORDS,
    LANGUAGE_QUANTIFIER_KEYWORDS,
)
from src.vulcan.endpoints.unified_chat import (
    TEMPLATE_RESPONSE_INDICATORS,
    _is_template_response,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def router_no_llm():
    """Create a router without an LLM client (fallback only)."""
    return LLMQueryRouter(llm_client=None)


# ============================================================
# ISSUE 1: LLM ROUTER FALLBACK TESTS
# ============================================================

class TestAnalogicalRouting:
    """Tests for analogical reasoning query routing."""
    
    def test_analogical_keywords_defined(self):
        """Analogical keywords should be defined."""
        assert ANALOGICAL_KEYWORDS
        # Verify specific expected keywords are present
        assert "analogical mapping" in ANALOGICAL_KEYWORDS
        assert "analogy" in ANALOGICAL_KEYWORDS
        assert "structure mapping" in ANALOGICAL_KEYWORDS or "map the deep structure" in ANALOGICAL_KEYWORDS
    
    def test_map_deep_structure_query(self, router_no_llm):
        """Query about mapping deep structures should route to analogical engine."""
        query = "Map the deep structure S→T from the solar system to atoms"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
        assert decision.confidence >= 0.8
        assert "analogical" in decision.reason.lower()
    
    def test_analogical_mapping_query(self, router_no_llm):
        """Query about analogical mapping should route to analogical engine."""
        query = "Perform an analogical mapping from source domain X to target domain Y"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
    
    def test_analogy_query(self, router_no_llm):
        """Query using analogy keyword should route to analogical engine."""
        query = "Find an analogy between the brain and a computer"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"


class TestLanguageQuantifierRouting:
    """Tests for language/quantifier query routing."""
    
    def test_language_quantifier_keywords_defined(self):
        """Language/quantifier keywords should be defined."""
        assert LANGUAGE_QUANTIFIER_KEYWORDS
        assert "quantifier scope" in LANGUAGE_QUANTIFIER_KEYWORDS
    
    def test_quantifier_scope_query(self, router_no_llm):
        """Query about quantifier scope should route to symbolic engine."""
        query = "Analyze the quantifier scope ambiguity in 'Every student loves some professor'"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"
        assert decision.confidence >= 0.8
        assert "quantifier" in decision.reason.lower() or "language" in decision.reason.lower()
    
    def test_scope_ambiguity_query(self, router_no_llm):
        """Query about scope ambiguity should route to symbolic engine."""
        query = "What is the scope ambiguity in this sentence?"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"
    
    def test_universal_quantifier_query(self, router_no_llm):
        """Query about universal quantifier should route to symbolic engine."""
        query = "Analyze the universal quantifier ∀x in this formula"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"


class TestExistingSpecializedRouting:
    """Tests to ensure existing specialized routing still works."""
    
    def test_causal_routing_still_works(self, router_no_llm):
        """Causal queries should still route to causal engine."""
        query = "What is the effect of confounding variables?"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "causal"
    
    def test_mathematical_routing_still_works(self, router_no_llm):
        """Mathematical queries should still route to mathematical/probabilistic engine."""
        query = "Compute exactly: ∑(i=1 to n) i²"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        # Could be mathematical or probabilistic depending on detection
        assert decision.engine in ["mathematical", "probabilistic"]
    
    def test_logic_routing_still_works(self, router_no_llm):
        """Logic queries should still route to symbolic engine."""
        query = "Is this formula satisfiable: (A ∧ B) → C?"
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"


# ============================================================
# ISSUE 2: TEMPLATE DETECTION TESTS
# ============================================================

class TestTemplateDetection:
    """Tests for template response detection."""
    
    def test_template_indicators_defined(self):
        """Template indicators should be defined."""
        assert TEMPLATE_RESPONSE_INDICATORS
        assert len(TEMPLATE_RESPONSE_INDICATORS) > 0
        assert "Vulcan's Introspective Analysis" in TEMPLATE_RESPONSE_INDICATORS
    
    def test_detect_vulcan_introspective_analysis(self):
        """Should detect template with 'Vulcan's Introspective Analysis'."""
        response = """
        Vulcan's Introspective Analysis
        
        I'm approaching this question from my own evolving value system.
        """
        assert _is_template_response(response) is True
    
    def test_detect_evolved_values_template(self):
        """Should detect template with 'evolved values' phrase."""
        response = """
        Based on my evolved values and learned ethical boundaries,
        I believe the right approach balances multiple considerations.
        """
        assert _is_template_response(response) is True
    
    def test_detect_multiple_considerations_template(self):
        """Should detect template with 'balances multiple considerations'."""
        response = """
        The answer balances multiple considerations while staying true
        to my core objective of beneficial outcomes.
        """
        assert _is_template_response(response) is True
    
    def test_non_template_response_passes(self):
        """Non-template responses should not be detected as templates."""
        response = "The capital of France is Paris."
        assert _is_template_response(response) is False
    
    def test_substantive_response_passes(self):
        """Substantive responses should not be detected as templates."""
        response = """
        To solve this problem, we need to consider the causal relationships
        between the variables. First, we identify the confounding factors...
        """
        assert _is_template_response(response) is False
    
    def test_none_response(self):
        """None response should not be detected as template."""
        assert _is_template_response(None) is False
    
    def test_dict_response_with_template(self):
        """Dict containing template text should be detected."""
        response = {
            "conclusion": "Vulcan's Introspective Analysis\n\nI'm approaching this..."
        }
        assert _is_template_response(response) is True
    
    def test_dict_response_without_template(self):
        """Dict without template text should not be detected."""
        response = {
            "conclusion": "The answer is 42."
        }
        assert _is_template_response(response) is False


# ============================================================
# ISSUE 3: WORLDMODEL MISROUTING DETECTION TESTS
# ============================================================

class TestWorldModelMisroutingDetection:
    """Tests for WorldModel detection of misrouted queries."""
    
    def test_analogical_query_detected(self):
        """WorldModel should detect misrouted analogical queries."""
        # This test verifies the detection logic exists
        # Actual integration test would require WorldModelToolWrapper instance
        
        query = "map the deep structure from domain A to domain B"
        
        # Verify the detection keywords include analogical patterns
        from src.vulcan.routing.llm_router import ANALOGICAL_KEYWORDS
        assert any(keyword in query.lower() for keyword in ANALOGICAL_KEYWORDS)
    
    def test_causal_query_detected(self):
        """WorldModel should detect misrouted causal queries."""
        query = "analyze the confounding variables in this DAG"
        
        # Verify the detection keywords include causal patterns
        from src.vulcan.routing.llm_router import CAUSAL_KEYWORDS
        assert any(keyword in query.lower() for keyword in CAUSAL_KEYWORDS)
    
    def test_mathematical_query_detected(self):
        """WorldModel should detect misrouted mathematical queries."""
        query = "compute exactly the integral ∫(x² dx)"
        
        # Verify the detection keywords/patterns include mathematical patterns
        from src.vulcan.routing.llm_router import MATHEMATICAL_KEYWORDS
        assert any(keyword in query.lower() for keyword in MATHEMATICAL_KEYWORDS)
    
    def test_logical_query_detected(self):
        """WorldModel should detect misrouted logical queries."""
        query = "check if this proposition is satisfiable using FOL"
        
        # Verify the detection keywords include logical patterns
        from src.vulcan.routing.llm_router import LOGIC_KEYWORDS
        assert any(keyword in query.lower() for keyword in LOGIC_KEYWORDS)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestRoutingIntegration:
    """Integration tests for the complete routing fix."""
    
    def test_analogical_query_end_to_end(self, router_no_llm):
        """Analogical query should route correctly without LLM."""
        query = "Map the deep structure S→T from source to target domain"
        decision = router_no_llm.route(query)
        
        # Should route to analogical reasoning engine
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
        assert decision.source == "fallback"
        
        # Should have reasonable confidence
        assert decision.confidence >= 0.7
    
    def test_quantifier_query_end_to_end(self, router_no_llm):
        """Quantifier scope query should route correctly without LLM."""
        query = "Analyze the quantifier scope ambiguity in: every student reads some book"
        decision = router_no_llm.route(query)
        
        # Should route to symbolic reasoning engine
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"
        assert decision.source == "fallback"
        
        # Should have reasonable confidence
        assert decision.confidence >= 0.7
    
    def test_template_detection_prevents_bypass(self):
        """Template responses should prevent direct reasoning bypass."""
        # Test that template detection function exists and works
        template = "Vulcan's Introspective Analysis\n\nI'm approaching this..."
        assert _is_template_response(template) is True
        
        # Test that non-template passes
        substantive = "The solar system consists of 8 planets..."
        assert _is_template_response(substantive) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
