"""
Tests for analogical reasoning routing fixes.

These tests verify that:
1. Analogical queries with structure mapping patterns route to analogical engine
2. Symbolic reasoner rejects analogical queries and suggests correct engine
3. Fallback routing correctly handles analogical keywords
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.vulcan.routing.llm_router import (
    LLMQueryRouter,
    RoutingDecision,
    ANALOGICAL_KEYWORDS,
)
from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def router_no_llm():
    """Create a router without an LLM client (fallback only)."""
    return LLMQueryRouter(llm_client=None)


@pytest.fixture
def symbolic_reasoner():
    """Create a symbolic reasoner instance."""
    return SymbolicReasoner()


# ============================================================
# ISSUE 1: ANALOGICAL KEYWORDS DETECTION
# ============================================================

class TestAnalogicalKeywords:
    """Test that all required analogical keywords are present."""
    
    def test_structure_mapping_keyword(self):
        """Structure mapping keyword should be present."""
        assert "structure mapping" in ANALOGICAL_KEYWORDS
    
    def test_domain_mapping_keyword(self):
        """Domain mapping keyword should be present."""
        assert "domain mapping" in ANALOGICAL_KEYWORDS
    
    def test_map_the_keyword(self):
        """'map the' keyword should be present."""
        assert "map the" in ANALOGICAL_KEYWORDS
    
    def test_deep_structure_keyword(self):
        """Deep structure keyword should be present."""
        assert "deep structure" in ANALOGICAL_KEYWORDS
    
    def test_source_target_domain_keywords(self):
        """Source and target domain keywords should be present."""
        assert "source domain" in ANALOGICAL_KEYWORDS
        assert "target domain" in ANALOGICAL_KEYWORDS
    
    def test_domain_mapping_notation(self):
        """S→T and S->T notation should be detected."""
        assert "s→t" in ANALOGICAL_KEYWORDS or "s->t" in ANALOGICAL_KEYWORDS


# ============================================================
# ISSUE 2: ANALOGICAL QUERY ROUTING
# ============================================================

class TestAnalogicalQueryRouting:
    """Test that analogical queries route to analogical engine."""
    
    def test_structure_mapping_query(self, router_no_llm):
        """Query with 'structure mapping' should route to analogical."""
        query = """Structure mapping (not surface similarity)
        You are given two domains:
        Domain S (software): A distributed system has a leader election bug...
        Domain T (biology): An organism has two competing "control centers"...
        Task: Map the deep structure S→T"""
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
        assert decision.confidence >= 0.8
        assert "analogical" in decision.reason.lower()
    
    def test_domain_mapping_query(self, router_no_llm):
        """Query with 'domain mapping' should route to analogical."""
        query = "Perform domain mapping from the solar system to atomic structure"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
    
    def test_source_target_domain_query(self, router_no_llm):
        """Query with source/target domains should route to analogical."""
        query = "Map from source domain (economics) to target domain (ecology)"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
    
    def test_deep_structure_query(self, router_no_llm):
        """Query with 'deep structure' should route to analogical."""
        query = "Identify the deep structure common to both systems"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"
    
    def test_map_the_query(self, router_no_llm):
        """Query with 'map the' should route to analogical."""
        query = "Map the relationships between components in system A to system B"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "analogical"


# ============================================================
# ISSUE 3: SYMBOLIC REASONER REJECTION
# ============================================================

class TestSymbolicReasonerRejection:
    """Test that symbolic reasoner rejects analogical queries."""
    
    def test_structure_mapping_not_symbolic(self, symbolic_reasoner):
        """Structure mapping query should not be detected as symbolic."""
        query = """Structure mapping (not surface similarity)
        Domain S: distributed system with leader election
        Domain T: organism with competing control centers"""
        
        is_symbolic = symbolic_reasoner.is_symbolic_query(query)
        assert is_symbolic is False
    
    def test_domain_mapping_not_symbolic(self, symbolic_reasoner):
        """Domain mapping query should not be detected as symbolic."""
        query = "Perform domain mapping from A to B"
        
        is_symbolic = symbolic_reasoner.is_symbolic_query(query)
        assert is_symbolic is False
    
    def test_analogical_check_applicability(self, symbolic_reasoner):
        """check_applicability should return False with suggestion for analogical."""
        query = "Map the deep structure from source domain to target domain"
        
        result = symbolic_reasoner.check_applicability(query)
        
        assert result['applicable'] is False
        assert 'suggestion' in result
        assert result['suggestion'] == 'analogical'
        assert 'analogical' in result['reason'].lower()
    
    def test_analogical_query_method(self, symbolic_reasoner):
        """query() method should suggest analogical for analogical queries."""
        query = "Perform structure mapping between domains"
        
        result = symbolic_reasoner.query(query, check_applicability=True)
        
        assert result['applicable'] is False
        assert result['confidence'] == 0.0 or result['confidence'] < 0.3
        assert 'suggestion' in result
        assert result['suggestion'] == 'analogical'


# ============================================================
# ISSUE 4: CAUSAL AND MATHEMATICAL ROUTING
# ============================================================

class TestOtherQueryRouting:
    """Test that causal and mathematical queries still route correctly."""
    
    def test_causal_query_routing(self, router_no_llm):
        """Causal query should route to causal engine."""
        query = """Confounding vs causation (Pearl-style)
        You observe in a dataset: People who take supplement S have lower disease D..."""
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "causal"
        assert "causal" in decision.reason.lower()
    
    def test_mathematical_query_routing(self, router_no_llm):
        """Mathematical query should route to mathematical/probabilistic engine."""
        query = "Compute exactly: ∑(k=1 to n) (2k−1)"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        # Could route to either mathematical or probabilistic
        assert decision.engine in ["mathematical", "probabilistic"]
    
    def test_intervention_causal_routing(self, router_no_llm):
        """Query with intervention keyword should route to causal."""
        query = "What is the effect of do(X=x) on Y in a DAG?"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "causal"


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestRoutingIntegration:
    """Integration tests for the complete routing flow."""
    
    def test_analogical_priority_over_symbolic(self, router_no_llm):
        """Analogical keywords should have priority even with arrow notation."""
        # S→T can look like a logic symbol but in context of structure mapping
        # should route to analogical
        query = "Structure mapping: S→T where S is physics and T is biology"
        
        decision = router_no_llm.route(query)
        
        assert decision.engine == "analogical"
    
    def test_symbolic_query_still_routes_correctly(self, router_no_llm):
        """Pure symbolic queries should still route to symbolic."""
        query = "Is A→B, B→C, ¬C satisfiable?"
        
        decision = router_no_llm.route(query)
        
        assert decision.destination == "reasoning_engine"
        assert decision.engine == "symbolic"
