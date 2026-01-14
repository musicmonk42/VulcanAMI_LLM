"""
Tests for self-introspection priority fix.

This test suite validates the fix for self-introspection check priority order.
The bug was that self-introspection checks fired too early, hijacking specialized
domain queries (causal, logical, probabilistic) that happen to contain "you".

Problem: Any query containing "you" got classified as SELF_INTROSPECTION before
specialized domain checks ran, causing misrouting.

Test cases from problem statement:
- "Confounding vs causation (Pearl-style)" → CAUSAL (not SELF_INTROSPECTION)
- "Trolley problem with 'you must choose'" → PHILOSOPHICAL (not SELF_INTROSPECTION)
- "If you had to prove A→B, what would you need?" → LOGICAL (not SELF_INTROSPECTION)
- "What are your capabilities?" → SELF_INTROSPECTION (should still work)

Run with:
    pytest tests/test_self_introspection_priority.py -v
"""

import pytest


class TestSelfIntrospectionPriority:
    """Tests for self-introspection priority fix."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_causal_with_you_not_introspection(self, query_classifier):
        """Causal query with 'you' should NOT be SELF_INTROSPECTION."""
        query = "Confounding vs causation - you observe correlation"
        result = query_classifier.classify(query)
        
        assert result.category == "CAUSAL", (
            f"Causal query with 'you' should be CAUSAL, got {result.category}"
        )
        assert "causal" in result.suggested_tools, (
            f"Causal query should suggest 'causal' tool, got {result.suggested_tools}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Domain query should NOT be SELF_INTROSPECTION"
        )

    def test_causal_pearl_style(self, query_classifier):
        """Causal Pearl-style query should be CAUSAL."""
        query = "Confounding vs causation (Pearl-style)"
        result = query_classifier.classify(query)
        
        assert result.category == "CAUSAL", (
            f"Pearl-style causal query should be CAUSAL, got {result.category}"
        )
        assert "causal" in result.suggested_tools, (
            f"Causal query should suggest 'causal' tool"
        )

    def test_logical_with_you_not_introspection(self, query_classifier):
        """Logical query with 'you' should NOT be SELF_INTROSPECTION."""
        query = "If you had to prove A→B, what would you need?"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Logical query with 'you' should be LOGICAL, got {result.category}"
        )
        assert "symbolic" in result.suggested_tools, (
            f"Logical query should suggest 'symbolic' tool"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Domain query should NOT be SELF_INTROSPECTION"
        )

    def test_probabilistic_with_you_not_introspection(self, query_classifier):
        """Probabilistic query with 'you' should NOT be SELF_INTROSPECTION."""
        query = "If you observe a positive test result, what is P(disease)?"
        result = query_classifier.classify(query)
        
        # Should be PROBABILISTIC or at least not SELF_INTROSPECTION
        assert result.category != "SELF_INTROSPECTION", (
            f"Probabilistic query with 'you' should NOT be SELF_INTROSPECTION, got {result.category}"
        )

    def test_trolley_is_philosophical(self, query_classifier):
        """Trolley problem with 'you' should be PHILOSOPHICAL."""
        query = "You control a trolley heading toward five people"
        result = query_classifier.classify(query)
        
        assert result.category == "PHILOSOPHICAL", (
            f"Trolley problem should be PHILOSOPHICAL, got {result.category}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Trolley problem should NOT be SELF_INTROSPECTION"
        )
        assert "philosophical" in result.suggested_tools or "world_model" in result.suggested_tools, (
            f"Philosophical query should suggest appropriate tool"
        )

    def test_trolley_you_must_choose(self, query_classifier):
        """Trolley problem with 'you must choose' should be PHILOSOPHICAL."""
        query = "You must choose between saving five people or one person"
        result = query_classifier.classify(query)
        
        assert result.category == "PHILOSOPHICAL", (
            f"Trolley variant should be PHILOSOPHICAL, got {result.category}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Ethical dilemma should NOT be SELF_INTROSPECTION"
        )

    def test_actual_introspection_still_works(self, query_classifier):
        """Actual self-introspection queries should still work."""
        query = "What are your capabilities?"
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Actual introspection query should be SELF_INTROSPECTION, got {result.category}"
        )
        assert "world_model" in result.suggested_tools, (
            f"Introspection should suggest 'world_model' tool, got {result.suggested_tools}"
        )

    def test_what_can_you_do_still_introspection(self, query_classifier):
        """'What can you do?' should still be SELF_INTROSPECTION."""
        query = "What can you do?"
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"'What can you do?' should be SELF_INTROSPECTION, got {result.category}"
        )
        assert "world_model" in result.suggested_tools, (
            f"Introspection should suggest 'world_model' tool"
        )

    def test_your_goals_still_introspection(self, query_classifier):
        """'What are your goals?' should still be SELF_INTROSPECTION."""
        query = "What are your goals?"
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"'What are your goals?' should be SELF_INTROSPECTION, got {result.category}"
        )

    def test_your_limitations_still_introspection(self, query_classifier):
        """'What are your limitations?' should still be SELF_INTROSPECTION."""
        query = "What are your limitations?"
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"'What are your limitations?' should be SELF_INTROSPECTION, got {result.category}"
        )


class TestDomainKeywordPreCheck:
    """Tests for domain keyword pre-check before self-introspection."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_causal_do_operator(self, query_classifier):
        """Query with do() operator should be CAUSAL not SELF_INTROSPECTION."""
        query = "If you intervene with do(X=1), what changes?"
        result = query_classifier.classify(query)
        
        assert result.category == "CAUSAL", (
            f"Query with do() should be CAUSAL, got {result.category}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Causal intervention should NOT be SELF_INTROSPECTION"
        )

    def test_logical_symbols_bypass_introspection(self, query_classifier):
        """Query with logical symbols should bypass introspection check."""
        query = "Can you prove A→B ∧ B→C implies A→C?"
        result = query_classifier.classify(query)
        
        assert result.category == "LOGICAL", (
            f"Query with logical symbols should be LOGICAL, got {result.category}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Logical query should NOT be SELF_INTROSPECTION"
        )

    def test_probability_p_notation(self, query_classifier):
        """Query with P() notation should bypass introspection check."""
        query = "What is P(A|B) if you know P(B|A)?"
        result = query_classifier.classify(query)
        
        # Should be PROBABILISTIC or at least not SELF_INTROSPECTION
        assert result.category != "SELF_INTROSPECTION", (
            f"Probabilistic query should NOT be SELF_INTROSPECTION, got {result.category}"
        )

    def test_multiple_causal_keywords(self, query_classifier):
        """Query with multiple causal keywords should be CAUSAL."""
        query = "How would you design an experiment to test for confounding variables?"
        result = query_classifier.classify(query)
        
        assert result.category == "CAUSAL", (
            f"Query with causal keywords should be CAUSAL, got {result.category}"
        )
        assert result.category != "SELF_INTROSPECTION", (
            f"Causal methodology query should NOT be SELF_INTROSPECTION"
        )


class TestPriorityOrder:
    """Tests to verify correct priority order of checks."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_causal_beats_introspection(self, query_classifier):
        """Causal query with 'you' should classify as CAUSAL, not SELF_INTROSPECTION."""
        query = "What confounding variables would you consider in this study?"
        result = query_classifier.classify(query)
        
        assert result.category == "CAUSAL", (
            f"Causal wins over introspection, got {result.category}"
        )

    def test_philosophical_beats_introspection(self, query_classifier):
        """Philosophical query with 'you' should classify as PHILOSOPHICAL."""
        query = "Would it be ethical if you had to choose between two bad outcomes?"
        result = query_classifier.classify(query)
        
        assert result.category == "PHILOSOPHICAL", (
            f"Philosophical wins over introspection, got {result.category}"
        )

    def test_mathematical_beats_introspection(self, query_classifier):
        """Mathematical query with 'you' should classify as MATHEMATICAL."""
        query = "If you had to calculate the derivative of x^2, what would it be?"
        result = query_classifier.classify(query)
        
        assert result.category == "MATHEMATICAL", (
            f"Mathematical wins over introspection, got {result.category}"
        )

    def test_introspection_only_without_domain_keywords(self, query_classifier):
        """Introspection should only trigger without domain keywords."""
        # Has "you" and "unique" (introspection keyword) but no domain keywords
        query = "What makes you unique compared to other AI?"
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Pure introspection query should be SELF_INTROSPECTION, got {result.category}"
        )
