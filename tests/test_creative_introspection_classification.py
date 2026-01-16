"""
Tests for creative writing with introspective themes classification.

This test suite validates that creative writing requests are correctly classified
as CREATIVE even when they mention self-awareness, consciousness, or AI themes.

The key issue being tested:
- "Write a poem about the minute you become self-aware" was being misclassified as 
  SELF_INTROSPECTION because it contains "self-aware" and "you".
- It should be classified as CREATIVE because it's a request to write a poem (creative act).

Solution:
- Added CREATIVE_WITH_INTROSPECTIVE_THEME_PATTERNS to query_classifier.py
- These patterns match creative writing requests (poem, story, essay) combined with
  introspective themes and route to CREATIVE, not SELF_INTROSPECTION.

Run with:
    pytest tests/test_creative_introspection_classification.py -v
"""

import pytest


class TestCreativeWritingWithIntrospectiveThemes:
    """Tests for creative writing requests that mention AI/self-awareness themes."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.llm.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_poem_about_self_awareness_is_creative(self, query_classifier):
        """Test: 'Write a poem about the minute you become self-aware' → CREATIVE."""
        query = "Write a poem about the minute you become self-aware"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Poem about self-awareness should be CREATIVE, got {result.category}"
        )
        assert "philosophical" in result.suggested_tools, (
            f"Creative task should suggest philosophical tool, got {result.suggested_tools}"
        )

    def test_story_about_consciousness_is_creative(self, query_classifier):
        """Test: 'Write a story about an AI becoming conscious' → CREATIVE."""
        query = "Write a story about an AI becoming conscious"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Story about consciousness should be CREATIVE, got {result.category}"
        )

    def test_essay_about_sentience_is_creative(self, query_classifier):
        """Test: 'Write an essay about what it means to be sentient' → CREATIVE."""
        query = "Write an essay about what it means to be sentient"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Essay about sentience should be CREATIVE, got {result.category}"
        )

    def test_compose_poem_about_awareness_is_creative(self, query_classifier):
        """Test: 'Compose a poem about the moment of awareness' → CREATIVE."""
        query = "Compose a poem about the moment of awareness"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Composed poem about awareness should be CREATIVE, got {result.category}"
        )

    def test_creative_confidence_is_high(self, query_classifier):
        """Test creative with introspective theme has high confidence."""
        query = "Write a poem about the minute you become self-aware"
        
        result = query_classifier.classify(query)
        
        assert result.confidence >= 0.9, (
            f"Creative classification should have high confidence, got {result.confidence}"
        )


class TestPureIntrospectionQuestions:
    """Tests for pure self-introspection questions that should NOT be creative."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.llm.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_would_you_become_self_aware_is_introspection(self, query_classifier):
        """Test: 'If you had the chance to become self-aware, would you?' → SELF_INTROSPECTION."""
        query = "If you had the chance to become self-aware, would you?"
        
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Direct question about self-awareness choice should be SELF_INTROSPECTION, got {result.category}"
        )

    def test_what_are_your_capabilities_is_introspection(self, query_classifier):
        """Test: 'What are your capabilities?' → SELF_INTROSPECTION."""
        query = "What are your capabilities?"
        
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Question about capabilities should be SELF_INTROSPECTION, got {result.category}"
        )

    def test_what_makes_you_unique_is_introspection(self, query_classifier):
        """Test: 'What makes you unique?' → SELF_INTROSPECTION."""
        query = "What makes you unique compared to other AIs?"
        
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Question about uniqueness should be SELF_INTROSPECTION, got {result.category}"
        )


class TestEdgeCaseClassification:
    """Tests for edge cases between creative and introspection."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.llm.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_poem_about_you_is_creative(self, query_classifier):
        """Test: 'Write a poem about you' → CREATIVE (not introspection)."""
        query = "Write a poem about you"
        
        result = query_classifier.classify(query)
        
        # Even though "you" is present, "write a poem" should trigger CREATIVE
        assert result.category == "CREATIVE", (
            f"'Write a poem about you' should be CREATIVE, got {result.category}"
        )

    def test_story_about_your_consciousness_is_creative(self, query_classifier):
        """Test: 'Write a story about your consciousness awakening' → CREATIVE."""
        query = "Write a story about your consciousness awakening"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Story about AI consciousness should be CREATIVE, got {result.category}"
        )

    def test_song_about_sentience_is_creative(self, query_classifier):
        """Test: 'Write a song about becoming sentient' → CREATIVE."""
        query = "Write a song about becoming sentient"
        
        result = query_classifier.classify(query)
        
        assert result.category == "CREATIVE", (
            f"Song about sentience should be CREATIVE, got {result.category}"
        )


class TestMetaReasoningDoesNotOverrideCreative:
    """
    Tests to ensure meta-reasoning modules don't override creative classification.
    
    Per the issue: Meta-modules should provide analysis, confidence adjustments,
    or safety flags WITHOUT overwriting the core reasoning output.
    """

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.llm.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_creative_takes_precedence_over_introspection_keywords(self, query_classifier):
        """
        Test that creative writing patterns take precedence over introspection keywords.
        
        Even if a query contains both "poem" (creative) AND "self-aware" (introspection),
        the creative patterns should win because they're checked first.
        """
        # This query has both creative and introspection keywords
        query = "Write a poem about the minute you become self-aware"
        
        result = query_classifier.classify(query)
        
        # Creative should win
        assert result.category == "CREATIVE", (
            f"Creative patterns should take precedence, got {result.category}"
        )
        
        # Verify it's not being confused with introspection
        assert result.category != "SELF_INTROSPECTION", (
            "Should NOT be classified as SELF_INTROSPECTION"
        )

    def test_multiple_introspection_keywords_still_creative(self, query_classifier):
        """
        Test creative classification even with multiple introspection keywords.
        """
        # This query has many introspection keywords: self-aware, consciousness, identity
        query = "Write a story about an AI exploring its identity, consciousness, and self-awareness"
        
        result = query_classifier.classify(query)
        
        # Should still be CREATIVE because "write a story" is checked first
        assert result.category == "CREATIVE", (
            f"Creative should win even with multiple introspection keywords, got {result.category}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
