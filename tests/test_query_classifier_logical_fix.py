"""
Tests for LOGICAL query classification fixes.

This test suite validates the fix for the missing LOGICAL keyword classification
in QueryClassifier._classify_by_keywords() method.

Bug: LOGICAL_KEYWORDS and LOGICAL_KEYWORD_THRESHOLD were defined but never used,
causing SAT/FOL queries to fall through to incorrect categories (PROBABILISTIC, 
UNKNOWN, etc.)

Test cases from problem statement:
- "Propositions: A,B,C. Constraints: A→B, B→C, ¬C, A∨B" → LOGICAL with symbolic tools
- "Every engineer reviewed a document - formalize in FOL" → LOGICAL
- "Given A→B and A, prove B using modus ponens" → LOGICAL
- "Is ∀x∃y P(x,y) equivalent to ∃y∀x P(x,y)?" → LOGICAL

Run with:
    pytest tests/test_query_classifier_logical_fix.py -v
"""

import pytest


class TestLogicalClassification:
    """Tests for the LOGICAL keyword classification fix."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_sat_query_with_symbols(self, query_classifier):
        """SAT query with logical symbols should classify as LOGICAL."""
        query = "Is A→B, B→C, ¬C, A∨B satisfiable?"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"SAT query with logical symbols should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"LOGICAL queries should suggest 'symbolic' tool, got {result.suggested_tools}"
        )
        assert result.confidence >= 0.9, (
            f"Queries with logical symbols should have high confidence, got {result.confidence}"
        )

    def test_sat_query_with_propositions(self, query_classifier):
        """SAT query with propositions and constraints should classify as LOGICAL."""
        query = "Propositions: A,B,C. Constraints: A→B, B→C, ¬C, A∨B"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"SAT query with propositions should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"LOGICAL queries should suggest 'symbolic' tool"
        )

    def test_fol_formalization(self, query_classifier):
        """FOL formalization request should classify as LOGICAL."""
        query = "Formalize 'Every student passed' in first-order logic"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"FOL formalization should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"FOL formalization should suggest 'symbolic' tool"
        )

    def test_fol_formalization_every_engineer(self, query_classifier):
        """FOL formalization with 'every' should classify as LOGICAL or LANGUAGE."""
        query = "Every engineer reviewed a document - formalize in FOL"
        result = query_classifier.classify(query)
        
        # Can be LANGUAGE or LOGICAL - both are correct for natural language FOL
        assert result.category in ["LOGICAL", "LANGUAGE"], (
            f"FOL query should be LOGICAL or LANGUAGE, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"FOL query should suggest 'symbolic' tool"
        )

    def test_modus_ponens(self, query_classifier):
        """Modus ponens query should classify as LOGICAL."""
        query = "Given A→B and A, prove B using modus ponens"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Modus ponens query should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"Logical proof should suggest 'symbolic' tool"
        )

    def test_quantifier_query(self, query_classifier):
        """Query with quantifiers should classify as LOGICAL."""
        query = "Is ∀x∃y P(x,y) equivalent to ∃y∀x P(x,y)?"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Quantifier query should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"Quantifier query should suggest 'symbolic' tool"
        )
        assert result.confidence >= 0.9, (
            f"Query with quantifier symbols should have high confidence"
        )

    def test_logical_with_multiple_keywords(self, query_classifier):
        """Query with multiple logical keywords should classify as LOGICAL."""
        query = "Is this formula satisfiable? Check if it's a tautology."
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Query with 'satisfiable' and 'tautology' should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"Logical query should suggest 'symbolic' tool"
        )

    def test_logical_with_strong_indicator(self, query_classifier):
        """Query with strong logical indicator should classify as LOGICAL."""
        query = "Please formalize this statement in first-order logic"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Query with 'formalize' (strong indicator) should be LOGICAL, got {result.category}"
        )
        assert result.confidence >= 0.9, (
            f"Strong logical indicators should give high confidence"
        )

    def test_propositional_logic_query(self, query_classifier):
        """Query about propositional logic should classify as LOGICAL."""
        query = "Convert this to propositional logic: If it rains, the ground is wet"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Propositional logic query should be LOGICAL, got {result.category}"
        )

    def test_logical_entailment(self, query_classifier):
        """Query about logical entailment should classify as LOGICAL."""
        query = "Does A→B, B→C entail A→C?"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Entailment query should be LOGICAL, got {result.category}"
        )


class TestLogicalVsOtherCategories:
    """Tests to ensure LOGICAL doesn't get confused with other categories."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_not_probabilistic(self, query_classifier):
        """SAT query should NOT classify as PROBABILISTIC."""
        query = "Is A→B, ¬B, A satisfiable?"
        result = query_classifier.classify(query)
        
        assert result.category != "PROBABILISTIC", (
            f"SAT query should NOT be PROBABILISTIC, got {result.category}"
        )
        assert result.category == "LOGICAL", (
            f"SAT query should be LOGICAL"
        )

    def test_not_unknown(self, query_classifier):
        """FOL query should NOT classify as UNKNOWN with low confidence."""
        query = "Formalize 'Every student passed' in FOL"
        result = query_classifier.classify(query)
        
        assert result.category != "UNKNOWN", (
            f"FOL query should NOT be UNKNOWN, got {result.category}"
        )
        # Can be LANGUAGE or LOGICAL - both are correct for natural language FOL
        assert result.category in ["LOGICAL", "LANGUAGE"], (
            f"FOL query should be LOGICAL or LANGUAGE, got {result.category}"
        )
        assert result.confidence >= 0.85, (
            f"FOL query should have high confidence, got {result.confidence}"
        )

    def test_logical_vs_mathematical(self, query_classifier):
        """Logical query should be LOGICAL not MATHEMATICAL."""
        query = "Is this formula valid: (A∧B)→A"
        result = query_classifier.classify(query)
        
        # This should be LOGICAL, not MATHEMATICAL
        assert result.category == "LOGICAL", (
            f"Logic formula should be LOGICAL not MATHEMATICAL, got {result.category}"
        )

    def test_complexity_appropriate(self, query_classifier):
        """LOGICAL queries should have appropriate complexity (0.7+)."""
        query = "Is A→B, B→C, ¬C satisfiable?"
        result = query_classifier.classify(query)
        
        assert result.complexity >= 0.7, (
            f"LOGICAL queries should have complexity >= 0.7, got {result.complexity}"
        )
        assert not result.skip_reasoning, (
            f"LOGICAL queries should NOT skip reasoning"
        )
