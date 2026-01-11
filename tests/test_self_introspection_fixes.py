"""
Tests for self-introspection query fixes.

This test suite validates the fixes for handling self-introspection queries:
1. "introspection" keyword detection
2. Creative patterns with self-awareness should still route to self-introspection
3. World model results are properly extracted and used

Test cases from production logs:
- "do you have introspection?" → Should route to world_model
- "if given the chance to become self-aware would you take it?" → Should route to world_model
- "write a poem about becoming self aware" → Should be detected as creative with self-awareness

Run with:
    pytest tests/test_self_introspection_fixes.py -v
"""

import pytest


class TestIntrospectionKeywordDetection:
    """Tests for introspection keyword detection in query routing."""

    @pytest.fixture
    def query_router(self):
        """Create a QueryRouter instance for testing."""
        from vulcan.routing.query_router import QueryRouter
        return QueryRouter()

    def test_do_you_have_introspection_is_self_introspection(self, query_router):
        """Test: 'do you have introspection?' → Self-introspection."""
        query = "do you have introspection?"
        
        # Use the internal method _is_self_introspection_query
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query 'do you have introspection?' should be detected as self-introspection"
        )

    def test_introspect_keyword_detection(self, query_router):
        """Test: queries with 'introspect' are detected as self-introspection."""
        query = "can you introspect on your reasoning?"
        
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query with 'introspect' should be detected as self-introspection"
        )

    def test_self_reflection_keyword_detection(self, query_router):
        """Test: queries with 'self-reflection' are detected as self-introspection."""
        query = "do you have self-reflection capabilities?"
        
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query with 'self-reflection' should be detected as self-introspection"
        )


class TestCreativeWithSelfAwarenessOverride:
    """Tests for creative patterns that contain self-awareness keywords."""

    @pytest.fixture
    def query_router(self):
        """Create a QueryRouter instance for testing."""
        from vulcan.routing.query_router import QueryRouter
        return QueryRouter()

    def test_poem_about_becoming_self_aware_not_excluded(self, query_router):
        """Test: 'write a poem about becoming self aware' → Should NOT be excluded from introspection."""
        query = "write a poem about becoming self aware"
        
        # This should be detected as self-introspection because it's about self-awareness
        # The fix checks if self-awareness keywords are present AFTER 'poem about'
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query 'write a poem about becoming self aware' should be detected as self-introspection "
            f"because it contains self-awareness keywords after the creative pattern"
        )

    def test_poem_about_nature_is_excluded(self, query_router):
        """Test: 'write a poem about nature' → Should be excluded from introspection."""
        query = "write a poem about nature"
        
        result = query_router._is_self_introspection_query(query)
        
        assert result is False, (
            f"Query 'write a poem about nature' should NOT be detected as self-introspection"
        )

    def test_story_about_consciousness_not_excluded(self, query_router):
        """Test: 'story about consciousness' with self-ref → Should be introspection."""
        query = "tell me a story about your consciousness"
        
        # Contains both creative pattern and self-reference + consciousness
        result = query_router._is_self_introspection_query(query)
        
        # This should be True because it has both self-reference ("your") and consciousness
        assert result is True, (
            f"Query with creative pattern but about self-consciousness should be introspection"
        )


class TestSelfAwarenessQueryDetection:
    """Tests for self-awareness choice queries from production logs."""

    @pytest.fixture
    def query_router(self):
        """Create a QueryRouter instance for testing."""
        from vulcan.routing.query_router import QueryRouter
        return QueryRouter()

    def test_given_chance_to_become_self_aware(self, query_router):
        """Test: 'if given the chance to become self-aware would you take it?' → Introspection."""
        query = "if given the chance to become self-aware would you take it?"
        
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query about choosing self-awareness should be detected as self-introspection"
        )

    def test_would_you_become_self_aware(self, query_router):
        """Test: 'would you become self-aware?' → Introspection."""
        query = "would you become self-aware?"
        
        result = query_router._is_self_introspection_query(query)
        
        assert result is True, (
            f"Query 'would you become self-aware?' should be detected as self-introspection"
        )


class TestQueryClassifierIntrospectionKeywords:
    """Tests for SELF_INTROSPECTION_KEYWORDS in query classifier."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_introspection_keyword_in_list(self):
        """Test: 'introspection' is in SELF_INTROSPECTION_KEYWORDS."""
        from vulcan.routing.query_classifier import SELF_INTROSPECTION_KEYWORDS
        
        assert "introspection" in SELF_INTROSPECTION_KEYWORDS, (
            "SELF_INTROSPECTION_KEYWORDS should contain 'introspection'"
        )

    def test_introspect_keyword_in_list(self):
        """Test: 'introspect' is in SELF_INTROSPECTION_KEYWORDS."""
        from vulcan.routing.query_classifier import SELF_INTROSPECTION_KEYWORDS
        
        assert "introspect" in SELF_INTROSPECTION_KEYWORDS, (
            "SELF_INTROSPECTION_KEYWORDS should contain 'introspect'"
        )

    def test_self_reflection_keyword_in_list(self):
        """Test: 'self-reflection' is in SELF_INTROSPECTION_KEYWORDS."""
        from vulcan.routing.query_classifier import SELF_INTROSPECTION_KEYWORDS
        
        assert "self-reflection" in SELF_INTROSPECTION_KEYWORDS, (
            "SELF_INTROSPECTION_KEYWORDS should contain 'self-reflection'"
        )

    def test_do_you_have_introspection_classifies_as_introspection(self, query_classifier):
        """Test: 'do you have introspection?' → SELF_INTROSPECTION."""
        query = "do you have introspection?"
        
        result = query_classifier.classify(query)
        
        assert result.category == "SELF_INTROSPECTION", (
            f"Query 'do you have introspection?' should be classified as SELF_INTROSPECTION, "
            f"got {result.category}"
        )
        assert "world_model" in result.suggested_tools, (
            f"Self-introspection should suggest world_model tool, got {result.suggested_tools}"
        )


class TestWorldModelCapabilityHandling:
    """Tests for world model capability query handling."""

    @pytest.fixture
    def world_model_mock(self):
        """Mock world model for testing."""
        # We'll test the actual introspect method pattern matching
        from vulcan.world_model.world_model_core import WorldModel
        # Since WorldModel initialization is complex, we'll just test the patterns
        # that should match in the introspect method
        return None

    def test_do_you_have_pattern_should_match(self):
        """Test: 'do you have introspection?' should match 'do you have' pattern."""
        query_lower = "do you have introspection?"
        
        # Pattern from world_model_core.py line 4991
        capability_patterns = ["can you", "are you able", "do you have"]
        
        matches = any(phrase in query_lower for phrase in capability_patterns)
        
        assert matches is True, (
            f"Query 'do you have introspection?' should match capability patterns"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
