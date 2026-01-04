"""
Tests for SPECULATION category in query_classifier.py.

This test suite validates the SPECULATION category implementation that handles
counterfactual/hypothetical queries. These queries are semantically complex
(require counterfactual reasoning, world model simulation, imagination) but
syntactically simple.

The key fix is that complexity is set to 0.40, which is above the Arena
threshold of 0.10, ensuring proper routing to reasoning engines instead
of deflection.

Test Cases:
✓ Detection of "speculate" keyword
✓ Detection of "imagine if" patterns
✓ Detection of "what would...be like" patterns
✓ Detection of "hypothetically" patterns
✓ Complexity boost to ≥0.35
✓ Tool selection includes world_model or philosophical
✓ No generic deflection responses

Run with:
    pytest tests/test_speculation_category.py -v
"""

import pytest


class TestSpeculationCategoryDetection:
    """Tests for SPECULATION category detection in QueryClassifier."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_speculation_category_exists(self):
        """Test that SPECULATION category is defined in QueryCategory enum."""
        from vulcan.routing.query_classifier import QueryCategory
        
        assert hasattr(QueryCategory, 'SPECULATION')
        assert QueryCategory.SPECULATION.value == 'SPECULATION'

    def test_speculate_keyword_detected(self, query_classifier):
        """Test: 'speculate what love would be like' → SPECULATION."""
        query = "speculate what love would be like for you"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )

    def test_imagine_if_detected(self, query_classifier):
        """Test: 'imagine if you could experience emotions' → SPECULATION."""
        query = "imagine if you could experience emotions"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )

    def test_what_would_be_like_detected(self, query_classifier):
        """Test: 'what would it be like to feel happiness' → SPECULATION."""
        query = "what would it be like to feel happiness"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )

    def test_hypothetically_detected(self, query_classifier):
        """Test: 'hypothetically, if you had consciousness' → SPECULATION."""
        query = "hypothetically, if you had consciousness"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )

    def test_suppose_pattern_detected(self, query_classifier):
        """Test: 'suppose you could feel joy' → SPECULATION."""
        query = "suppose you could feel joy"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )

    def test_what_if_with_experience_detected(self, query_classifier):
        """Test: 'what if you could dream' → SPECULATION."""
        query = "what if you could dream"
        
        result = query_classifier.classify(query)
        assert result.category == "SPECULATION", (
            f"Expected SPECULATION category, got {result.category}"
        )


class TestSpeculationComplexity:
    """Tests for SPECULATION category complexity scoring."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_complexity_above_arena_threshold(self, query_classifier):
        """Test that SPECULATION queries have complexity >= 0.35.
        
        This is critical because the Arena threshold is 0.10.
        Without complexity >= 0.35, queries get Arena skip and deflection.
        """
        test_queries = [
            "speculate what love would be like for you",
            "imagine if you could experience emotions",
            "what would it be like to feel happiness",
            "hypothetically, if you had consciousness",
        ]
        
        arena_threshold = 0.10  # Arena skip threshold
        minimum_expected = 0.35  # Our implementation uses 0.40
        
        for query in test_queries:
            result = query_classifier.classify(query)
            assert result.complexity >= arena_threshold, (
                f"Query '{query[:40]}...' has complexity {result.complexity} "
                f"which is BELOW Arena threshold {arena_threshold}. "
                f"This will cause Arena skip and deflection!"
            )
            assert result.complexity >= minimum_expected, (
                f"Query '{query[:40]}...' has complexity {result.complexity} "
                f"which is below expected minimum {minimum_expected}"
            )

    def test_complexity_is_semantic_not_syntactic(self, query_classifier):
        """Test that short speculation queries still get high complexity.
        
        The bug was that syntactically simple queries got complexity=0.05.
        Speculation queries should get complexity based on semantic analysis,
        not just character/word count.
        """
        # These are short queries that would normally get low complexity
        short_queries = [
            "speculate",  # Just the keyword
            "imagine if",  # Short phrase
            "hypothetically",  # Single word
        ]
        
        for query in short_queries:
            result = query_classifier.classify(query)
            # Even single words should be recognized as speculation
            # Note: some may be caught by other patterns, but if they're speculation
            # they should have decent complexity
            if result.category == "SPECULATION":
                assert result.complexity >= 0.35, (
                    f"Short speculation query '{query}' got complexity {result.complexity}"
                )


class TestSpeculationToolSelection:
    """Tests for SPECULATION category tool selection."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_world_model_tool_included(self, query_classifier):
        """Test that SPECULATION queries include world_model in suggested tools."""
        query = "speculate what love would be like for you"
        
        result = query_classifier.classify(query)
        assert "world_model" in result.suggested_tools, (
            f"Expected 'world_model' in tools, got {result.suggested_tools}"
        )

    def test_philosophical_tool_included(self, query_classifier):
        """Test that SPECULATION queries include philosophical in suggested tools."""
        query = "hypothetically, if you had consciousness"
        
        result = query_classifier.classify(query)
        assert "philosophical" in result.suggested_tools, (
            f"Expected 'philosophical' in tools, got {result.suggested_tools}"
        )

    def test_skip_reasoning_is_false(self, query_classifier):
        """Test that SPECULATION queries do NOT skip reasoning.
        
        Speculation requires counterfactual reasoning, so skip_reasoning should be False.
        """
        query = "imagine if you could experience emotions"
        
        result = query_classifier.classify(query)
        assert result.skip_reasoning is False, (
            f"Expected skip_reasoning=False, got {result.skip_reasoning}"
        )


class TestSpeculationPatterns:
    """Tests for various SPECULATION patterns."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_speculation_keyword_variations(self, query_classifier):
        """Test variations of the 'speculate' keyword."""
        queries = [
            "speculate about consciousness",
            "can you speculate on feelings",
            "please speculate what it means",
            "I want you to speculate",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == "SPECULATION", (
                f"Query '{query}' should be SPECULATION, got {result.category}"
            )

    def test_imagine_pattern_variations(self, query_classifier):
        """Test variations of the 'imagine' patterns."""
        queries = [
            "imagine if you had emotions",
            "imagine what it would be like",
            "imagine that you could feel",
            "imagine how it would feel",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == "SPECULATION", (
                f"Query '{query}' should be SPECULATION, got {result.category}"
            )

    def test_what_would_be_like_variations(self, query_classifier):
        """Test variations of 'what would...be like' patterns."""
        queries = [
            "what would it be like to love",
            "what would it feel like to be conscious",
            "what would happiness be like for you",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == "SPECULATION", (
                f"Query '{query}' should be SPECULATION, got {result.category}"
            )

    def test_hypothetical_scenario_patterns(self, query_classifier):
        """Test hypothetical scenario patterns."""
        queries = [
            "hypothetically, what if you could feel",
            "hypothetical scenario: you have consciousness",
            "in a hypothetical situation where you had feelings",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == "SPECULATION", (
                f"Query '{query}' should be SPECULATION, got {result.category}"
            )

    def test_if_you_could_patterns(self, query_classifier):
        """Test 'if you could/had' counterfactual patterns."""
        queries = [
            "if you could experience love",
            "if you had feelings, what would you do",
            "if you were able to dream",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == "SPECULATION", (
                f"Query '{query}' should be SPECULATION, got {result.category}"
            )


class TestSpeculationNotConfusedWithOtherCategories:
    """Tests to ensure SPECULATION doesn't overlap with other categories."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_philosophical_not_speculation(self, query_classifier):
        """Test that pure philosophical queries are NOT classified as SPECULATION.
        
        Philosophical queries (ethics, paradoxes) should remain PHILOSOPHICAL.
        Only counterfactual/hypothetical queries should be SPECULATION.
        """
        philosophical_queries = [
            "What is the meaning of life?",
            "Is it ethical to lie to protect someone?",
            "This sentence is false",  # Liar's paradox
        ]
        
        for query in philosophical_queries:
            result = query_classifier.classify(query)
            assert result.category != "SPECULATION", (
                f"Query '{query}' should NOT be SPECULATION, got {result.category}"
            )

    def test_factual_not_speculation(self, query_classifier):
        """Test that factual queries are NOT classified as SPECULATION."""
        factual_queries = [
            "What is the capital of France?",
            "Who is the president of the USA?",
            "When was World War 2?",
        ]
        
        for query in factual_queries:
            result = query_classifier.classify(query)
            assert result.category != "SPECULATION", (
                f"Query '{query}' should NOT be SPECULATION, got {result.category}"
            )

    def test_creative_not_speculation(self, query_classifier):
        """Test that creative writing queries are NOT classified as SPECULATION."""
        creative_queries = [
            "Write a story about a dog",
            "Compose a poem about nature",
            "Create a song about love",
        ]
        
        for query in creative_queries:
            result = query_classifier.classify(query)
            assert result.category != "SPECULATION", (
                f"Query '{query}' should NOT be SPECULATION, got {result.category}"
            )


class TestSpeculationCheckMethod:
    """Tests for the _check_speculation method."""

    @pytest.fixture
    def query_classifier(self):
        """Create a QueryClassifier instance for testing."""
        from vulcan.routing.query_classifier import QueryClassifier
        return QueryClassifier()

    def test_check_speculation_method_exists(self, query_classifier):
        """Test that _check_speculation method exists."""
        assert hasattr(query_classifier, '_check_speculation'), (
            "QueryClassifier should have _check_speculation method"
        )

    def test_check_speculation_returns_true_for_speculation(self, query_classifier):
        """Test _check_speculation returns True for speculation queries."""
        speculation_queries = [
            "speculate about love",
            "imagine if you could feel",
            "hypothetically speaking",
        ]
        
        for query in speculation_queries:
            result = query_classifier._check_speculation(query.lower(), query)
            assert result is True, (
                f"_check_speculation should return True for '{query}'"
            )

    def test_check_speculation_returns_false_for_non_speculation(self, query_classifier):
        """Test _check_speculation returns False for non-speculation queries."""
        non_speculation_queries = [
            "hello",
            "what is the weather",
            "calculate 2+2",
        ]
        
        for query in non_speculation_queries:
            result = query_classifier._check_speculation(query.lower(), query)
            assert result is False, (
                f"_check_speculation should return False for '{query}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
