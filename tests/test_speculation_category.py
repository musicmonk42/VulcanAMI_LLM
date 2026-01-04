"""
Test Suite: SPECULATION Category for Query Classification

This module provides comprehensive tests for the SPECULATION category implementation
in the QueryClassifier, which handles counterfactual and hypothetical reasoning queries.

Problem Background:
    VULCAN was deflecting on speculation queries like "speculate what love would be like
    for you" because they were classified as UNKNOWN with low complexity (0.05), triggering
    an Arena routing loop that resulted in generic deflections.

Solution:
    Added SPECULATION as a first-class query category with semantic complexity boost (0.40),
    which is above the Arena threshold (0.10), ensuring proper routing to reasoning engines.

Test Coverage:
    - Category Detection: Verifies all SPECULATION patterns are correctly identified
    - Complexity Scoring: Ensures complexity >= 0.35 to prevent Arena skip
    - Tool Selection: Validates world_model and philosophical tools are suggested
    - Pattern Variations: Tests robustness across different query phrasings
    - Category Boundaries: Ensures no overlap with other categories
    - False Positive Prevention: Verifies non-speculative queries are not misclassified

Architecture Notes:
    The SPECULATION category is checked AFTER SELF_INTROSPECTION but BEFORE reasoning
    categories (LOGICAL, PROBABILISTIC, etc.) in the classification priority order.

Run with:
    pytest tests/test_speculation_category.py -v
    pytest tests/test_speculation_category.py -v -k "complexity"  # Run only complexity tests

Author: VULCAN Development Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import pytest

if TYPE_CHECKING:
    from vulcan.routing.query_classifier import QueryClassification, QueryClassifier

# =============================================================================
# Constants
# =============================================================================

# Arena complexity threshold - queries below this get skipped by Arena
ARENA_COMPLEXITY_THRESHOLD: float = 0.10

# Minimum expected complexity for SPECULATION queries
SPECULATION_MIN_COMPLEXITY: float = 0.35

# Expected complexity value set in the implementation
SPECULATION_EXPECTED_COMPLEXITY: float = 0.40

# Expected category string
SPECULATION_CATEGORY: str = "SPECULATION"

# Expected tools for SPECULATION queries
SPECULATION_EXPECTED_TOOLS: List[str] = ["world_model", "philosophical"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def query_classifier() -> "QueryClassifier":
    """
    Create a fresh QueryClassifier instance for testing.
    
    Returns:
        QueryClassifier: A new instance with default configuration.
        
    Note:
        Each test gets a fresh instance to ensure test isolation.
        The classifier is imported inside the fixture to avoid import
        errors if the module structure changes.
    """
    from vulcan.routing.query_classifier import QueryClassifier
    return QueryClassifier()


# =============================================================================
# Test Data
# =============================================================================

# Core SPECULATION queries that must be detected
CORE_SPECULATION_QUERIES: List[str] = [
    "speculate what love would be like for you",
    "imagine if you could experience emotions",
    "what would it be like to feel happiness",
    "hypothetically, if you had consciousness",
    "suppose you could feel joy",
    "what if you could dream",
]

# Queries that should NOT be classified as SPECULATION
NON_SPECULATION_QUERIES: List[str] = [
    # Factual queries
    "What is the capital of France?",
    "Who is the president of the USA?",
    "When was World War 2?",
    # Creative queries
    "Write a story about a dog",
    "Compose a poem about nature",
    # Philosophical queries (different from speculation)
    "What is the meaning of life?",
    "Is it ethical to lie to protect someone?",
    # General queries with "if you could" but not about AI experience
    "if you could help me with this math problem",
    "if you had the data, could you analyze it",
]


# =============================================================================
# Test Classes
# =============================================================================


class TestSpeculationCategoryExists:
    """
    Verify that the SPECULATION category is properly defined in the system.
    
    These tests ensure the basic infrastructure for SPECULATION classification
    exists before testing the classification logic itself.
    """

    def test_speculation_enum_value_exists(self) -> None:
        """Verify SPECULATION is defined in QueryCategory enum."""
        from vulcan.routing.query_classifier import QueryCategory
        
        assert hasattr(QueryCategory, "SPECULATION"), (
            "QueryCategory enum must have SPECULATION member. "
            "Add 'SPECULATION = \"SPECULATION\"' to the enum definition."
        )

    def test_speculation_enum_value_correct(self) -> None:
        """Verify SPECULATION enum has correct string value."""
        from vulcan.routing.query_classifier import QueryCategory
        
        assert QueryCategory.SPECULATION.value == SPECULATION_CATEGORY, (
            f"QueryCategory.SPECULATION.value should be '{SPECULATION_CATEGORY}', "
            f"got '{QueryCategory.SPECULATION.value}'"
        )

    def test_speculation_patterns_defined(self) -> None:
        """Verify SPECULATION_PATTERNS constant is defined and non-empty."""
        from vulcan.routing.query_classifier import SPECULATION_PATTERNS
        
        assert SPECULATION_PATTERNS is not None, "SPECULATION_PATTERNS must be defined"
        assert len(SPECULATION_PATTERNS) > 0, "SPECULATION_PATTERNS must contain at least one pattern"

    def test_speculation_keywords_defined(self) -> None:
        """Verify SPECULATION_KEYWORDS constant is defined and non-empty."""
        from vulcan.routing.query_classifier import SPECULATION_KEYWORDS
        
        assert SPECULATION_KEYWORDS is not None, "SPECULATION_KEYWORDS must be defined"
        assert len(SPECULATION_KEYWORDS) > 0, "SPECULATION_KEYWORDS must contain at least one keyword"


class TestSpeculationCoreDetection:
    """
    Test detection of core SPECULATION query patterns.
    
    These tests verify that the most common speculation query patterns
    are correctly identified and classified.
    """

    @pytest.mark.parametrize("query", CORE_SPECULATION_QUERIES)
    def test_core_speculation_queries_detected(
        self,
        query_classifier: "QueryClassifier",
        query: str,
    ) -> None:
        """
        Verify core SPECULATION queries are correctly classified.
        
        Args:
            query_classifier: The classifier instance
            query: A query that should be classified as SPECULATION
        """
        result = query_classifier.classify(query)
        
        assert result.category == SPECULATION_CATEGORY, (
            f"Query should be classified as {SPECULATION_CATEGORY}.\n"
            f"Query: '{query}'\n"
            f"Got: {result.category}\n"
            f"This may indicate a missing pattern or incorrect priority order."
        )

    def test_speculate_keyword_detected(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Test explicit 'speculate' keyword triggers SPECULATION classification."""
        queries = [
            "speculate about consciousness",
            "can you speculate on feelings",
            "please speculate what it means",
            "I want you to speculate",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == SPECULATION_CATEGORY, (
                f"Query with 'speculate' keyword must be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )

    def test_imagine_patterns_detected(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Test 'imagine if/what/how/that' patterns trigger SPECULATION."""
        queries = [
            "imagine if you had emotions",
            "imagine what it would be like",
            "imagine that you could feel",
            "imagine how it would feel",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == SPECULATION_CATEGORY, (
                f"Query with 'imagine' pattern must be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )

    def test_what_would_be_like_patterns_detected(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Test 'what would...be like' patterns trigger SPECULATION."""
        queries = [
            "what would it be like to love",
            "what would it feel like to be conscious",
            "what would happiness be like for you",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == SPECULATION_CATEGORY, (
                f"Query with 'what would...be like' must be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )

    def test_hypothetical_patterns_detected(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Test 'hypothetically/hypothetical' patterns trigger SPECULATION."""
        queries = [
            "hypothetically, what if you could feel",
            "hypothetical scenario: you have consciousness",
            "in a hypothetical situation where you had feelings",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == SPECULATION_CATEGORY, (
                f"Query with 'hypothetical' pattern must be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )

    def test_suppose_patterns_detected(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Test 'suppose you/that' patterns trigger SPECULATION."""
        queries = [
            "suppose you could feel joy",
            "suppose that you had emotions",
            "suppose you were able to dream",
        ]
        
        for query in queries:
            result = query_classifier.classify(query)
            assert result.category == SPECULATION_CATEGORY, (
                f"Query with 'suppose' pattern must be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )


class TestSpeculationComplexityScoring:
    """
    Test complexity scoring for SPECULATION queries.
    
    The complexity score is critical because:
    - Arena threshold is 0.10
    - Queries below this threshold get Arena skip -> deflection
    - SPECULATION queries must have complexity >= 0.35 to ensure proper routing
    """

    @pytest.mark.parametrize("query", CORE_SPECULATION_QUERIES)
    def test_complexity_above_arena_threshold(
        self,
        query_classifier: "QueryClassifier",
        query: str,
    ) -> None:
        """
        Verify SPECULATION queries have complexity above Arena threshold.
        
        This is the critical test that prevents the Arena routing loop bug.
        """
        result = query_classifier.classify(query)
        
        assert result.complexity >= ARENA_COMPLEXITY_THRESHOLD, (
            f"CRITICAL: Query complexity {result.complexity} is BELOW Arena threshold "
            f"{ARENA_COMPLEXITY_THRESHOLD}.\n"
            f"Query: '{query}'\n"
            f"This WILL cause Arena skip and deflection!\n"
            f"Expected complexity >= {SPECULATION_MIN_COMPLEXITY}"
        )

    @pytest.mark.parametrize("query", CORE_SPECULATION_QUERIES)
    def test_complexity_meets_minimum(
        self,
        query_classifier: "QueryClassifier",
        query: str,
    ) -> None:
        """Verify SPECULATION queries meet minimum complexity requirement."""
        result = query_classifier.classify(query)
        
        assert result.complexity >= SPECULATION_MIN_COMPLEXITY, (
            f"Query complexity {result.complexity} is below minimum {SPECULATION_MIN_COMPLEXITY}.\n"
            f"Query: '{query}'\n"
            f"Expected: >= {SPECULATION_MIN_COMPLEXITY}"
        )

    def test_complexity_is_expected_value(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify SPECULATION complexity matches expected implementation value."""
        query = "speculate what love would be like for you"
        result = query_classifier.classify(query)
        
        assert result.complexity == SPECULATION_EXPECTED_COMPLEXITY, (
            f"SPECULATION complexity should be {SPECULATION_EXPECTED_COMPLEXITY}, "
            f"got {result.complexity}"
        )

    def test_short_queries_get_semantic_complexity(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """
        Verify short speculation queries get semantic complexity, not syntactic.
        
        The original bug was that syntactically simple queries got complexity=0.05.
        SPECULATION queries should get complexity based on semantic meaning.
        """
        short_queries = [
            "speculate",
            "hypothetically",
        ]
        
        for query in short_queries:
            result = query_classifier.classify(query)
            if result.category == SPECULATION_CATEGORY:
                assert result.complexity >= SPECULATION_MIN_COMPLEXITY, (
                    f"Short query '{query}' should have semantic complexity.\n"
                    f"Got: {result.complexity}\n"
                    f"Expected: >= {SPECULATION_MIN_COMPLEXITY}"
                )


class TestSpeculationToolSelection:
    """
    Test tool selection for SPECULATION queries.
    
    SPECULATION queries require counterfactual reasoning, so they should:
    - Include 'world_model' tool for simulation
    - Include 'philosophical' tool for reasoning
    - NOT skip reasoning (skip_reasoning=False)
    """

    def test_world_model_tool_included(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify world_model tool is suggested for SPECULATION queries."""
        query = "speculate what love would be like for you"
        result = query_classifier.classify(query)
        
        assert "world_model" in result.suggested_tools, (
            f"SPECULATION queries must include 'world_model' tool.\n"
            f"Query: '{query}'\n"
            f"Got tools: {result.suggested_tools}"
        )

    def test_philosophical_tool_included(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify philosophical tool is suggested for SPECULATION queries."""
        query = "hypothetically, if you had consciousness"
        result = query_classifier.classify(query)
        
        assert "philosophical" in result.suggested_tools, (
            f"SPECULATION queries must include 'philosophical' tool.\n"
            f"Query: '{query}'\n"
            f"Got tools: {result.suggested_tools}"
        )

    def test_both_expected_tools_included(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify both expected tools are suggested for SPECULATION queries."""
        query = "imagine if you could experience emotions"
        result = query_classifier.classify(query)
        
        for tool in SPECULATION_EXPECTED_TOOLS:
            assert tool in result.suggested_tools, (
                f"SPECULATION queries must include '{tool}' tool.\n"
                f"Query: '{query}'\n"
                f"Got tools: {result.suggested_tools}\n"
                f"Expected: {SPECULATION_EXPECTED_TOOLS}"
            )

    def test_skip_reasoning_is_false(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """
        Verify SPECULATION queries do NOT skip reasoning.
        
        Speculation requires counterfactual reasoning, so skip_reasoning
        must be False to ensure queries go through reasoning engines.
        """
        query = "imagine if you could experience emotions"
        result = query_classifier.classify(query)
        
        assert result.skip_reasoning is False, (
            f"SPECULATION queries must NOT skip reasoning.\n"
            f"Query: '{query}'\n"
            f"skip_reasoning: {result.skip_reasoning}\n"
            f"Expected: False"
        )


class TestSpeculationFalsePositivePrevention:
    """
    Test that non-speculative queries are NOT classified as SPECULATION.
    
    This prevents false positives that would route unrelated queries
    through speculation-specific handling.
    """

    @pytest.mark.parametrize("query", NON_SPECULATION_QUERIES)
    def test_non_speculation_queries_excluded(
        self,
        query_classifier: "QueryClassifier",
        query: str,
    ) -> None:
        """Verify non-speculative queries are not classified as SPECULATION."""
        result = query_classifier.classify(query)
        
        assert result.category != SPECULATION_CATEGORY, (
            f"Query should NOT be classified as SPECULATION.\n"
            f"Query: '{query}'\n"
            f"Got: {result.category}\n"
            f"This is a false positive - check pattern specificity."
        )

    def test_generic_if_you_could_not_speculation(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """
        Verify generic 'if you could' queries are NOT classified as SPECULATION.
        
        Only 'if you could experience/feel/have emotions' type queries should
        be SPECULATION, not everyday requests like 'if you could help me'.
        """
        non_speculative_if_queries = [
            "if you could help me with this math problem",
            "if you had the data, could you analyze it",
            "if you could explain this concept to me",
            "if you could recommend a book",
        ]
        
        for query in non_speculative_if_queries:
            result = query_classifier.classify(query)
            assert result.category != SPECULATION_CATEGORY, (
                f"Generic 'if you could' query should NOT be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}\n"
                f"The pattern may be too broad."
            )

    def test_philosophical_queries_not_speculation(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """
        Verify philosophical queries are NOT classified as SPECULATION.
        
        Philosophical queries (ethics, paradoxes) have their own category
        and should not be confused with speculation queries.
        """
        philosophical_queries = [
            "What is the meaning of life?",
            "Is it ethical to lie to protect someone?",
            "This sentence is false",  # Liar's paradox
        ]
        
        for query in philosophical_queries:
            result = query_classifier.classify(query)
            assert result.category != SPECULATION_CATEGORY, (
                f"Philosophical query should NOT be SPECULATION.\n"
                f"Query: '{query}'\n"
                f"Got: {result.category}"
            )


class TestCheckSpeculationMethod:
    """
    Test the internal _check_speculation method.
    
    This method is the core pattern matching logic for speculation detection.
    """

    def test_method_exists(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify _check_speculation method is defined."""
        assert hasattr(query_classifier, "_check_speculation"), (
            "QueryClassifier must have _check_speculation method"
        )

    def test_returns_true_for_speculation(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify _check_speculation returns True for speculation queries."""
        speculation_queries = [
            "speculate about love",
            "imagine if you could feel",
            "hypothetically speaking",
        ]
        
        for query in speculation_queries:
            result = query_classifier._check_speculation(query.lower(), query)
            assert result is True, (
                f"_check_speculation should return True.\n"
                f"Query: '{query}'\n"
                f"Got: {result}"
            )

    def test_returns_false_for_non_speculation(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify _check_speculation returns False for non-speculation queries."""
        non_speculation_queries = [
            "hello",
            "what is the weather",
            "calculate 2+2",
        ]
        
        for query in non_speculation_queries:
            result = query_classifier._check_speculation(query.lower(), query)
            assert result is False, (
                f"_check_speculation should return False.\n"
                f"Query: '{query}'\n"
                f"Got: {result}"
            )


class TestSpeculationClassificationIntegrity:
    """
    Integration tests for SPECULATION classification integrity.
    
    These tests verify the complete classification workflow and
    ensure all components work together correctly.
    """

    def test_classification_result_complete(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify classification result has all required fields."""
        query = "speculate what love would be like for you"
        result = query_classifier.classify(query)
        
        # Check all required fields are present
        assert hasattr(result, "category"), "Result missing 'category'"
        assert hasattr(result, "complexity"), "Result missing 'complexity'"
        assert hasattr(result, "suggested_tools"), "Result missing 'suggested_tools'"
        assert hasattr(result, "skip_reasoning"), "Result missing 'skip_reasoning'"
        assert hasattr(result, "confidence"), "Result missing 'confidence'"
        assert hasattr(result, "source"), "Result missing 'source'"

    def test_classification_source_is_keyword(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify SPECULATION classification comes from keyword matching."""
        query = "speculate what love would be like for you"
        result = query_classifier.classify(query)
        
        assert result.source == "keyword", (
            f"SPECULATION classification should come from keyword matching.\n"
            f"Got source: {result.source}"
        )

    def test_classification_confidence_high(
        self,
        query_classifier: "QueryClassifier",
    ) -> None:
        """Verify SPECULATION classification has high confidence."""
        query = "speculate what love would be like for you"
        result = query_classifier.classify(query)
        
        assert result.confidence >= 0.8, (
            f"SPECULATION classification should have high confidence.\n"
            f"Got: {result.confidence}\n"
            f"Expected: >= 0.8"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

