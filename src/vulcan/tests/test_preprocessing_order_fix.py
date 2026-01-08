"""
Unit tests for the preprocessing order fix.

This test file verifies that query preprocessing (header stripping) happens
BEFORE classification, fixing the issue where headers like "A1" in 
"Analogical Reasoning A1 — Structure mapping..." would trigger incorrect
CRYPTOGRAPHIC classification.

Problem Statement:
    Classification happens BEFORE preprocessing:
    1. QueryClassifier runs ← Sees "A1" → CRYPTOGRAPHIC ❌
    2. QueryRouter routes based on classification
    3. QueryPreprocessor strips headers ← TOO LATE!
    4. Reasoning engines receive cleaned query but routing already wrong

Solution:
    Move preprocessing BEFORE classification:
    1. preprocess_query(raw_query) → strips headers
    2. classify_query(preprocessed_query) → correct classification
    3. route_to_engine() → correct engine
"""

import pytest
from typing import Optional


class TestHeaderStrippingFunction:
    """Tests for the strip_query_headers function."""

    @pytest.fixture
    def strip_headers(self):
        """Get the strip_query_headers function."""
        from vulcan.routing.query_classifier import strip_query_headers
        return strip_query_headers

    def test_strip_analogical_reasoning_header(self, strip_headers):
        """Test stripping 'Analogical Reasoning A1 — Structure mapping...' header.
        
        This is the exact case from the problem statement where "A1" was
        incorrectly triggering CRYPTOGRAPHIC classification.
        """
        query = "Analogical Reasoning A1 — Structure mapping: Given the analogy..."
        result = strip_headers(query)
        
        # Should strip the header, leaving the actual content
        assert "A1" not in result
        assert "Analogical Reasoning" not in result
        # The actual content should remain
        assert "mapping" in result.lower() or "analogy" in result.lower()

    def test_strip_causal_reasoning_header(self, strip_headers):
        """Test stripping 'Causal Reasoning C1 — Confounding...' header."""
        query = "Causal Reasoning C1 — Confounding: A study shows that..."
        result = strip_headers(query)
        
        assert "C1" not in result
        assert "Causal Reasoning" not in result
        # The actual content should remain
        assert "study" in result.lower() or "confounding" in result.lower()

    def test_strip_mathematical_header(self, strip_headers):
        """Test stripping 'M1 — Proof check...' header."""
        query = "M1 — Proof check: Prove that the sum of first n odd numbers..."
        result = strip_headers(query)
        
        assert "M1" not in result
        # The actual content should remain
        assert "prove" in result.lower() or "sum" in result.lower()

    def test_strip_numeric_verification_header(self, strip_headers):
        """Test stripping 'Numeric Verification (∑(2k-1))' header.
        
        This is the exact case from problem statement #5 where "verification"
        triggered CRYPTOGRAPHIC classification.
        """
        query = "Numeric Verification (∑(2k-1)): Calculate the sum of the first..."
        result = strip_headers(query)
        
        # "Numeric Verification" should be stripped to prevent "verification"
        # from triggering CRYPTOGRAPHIC classification
        # But the mathematical content should remain
        assert "∑" in result or "calculate" in result.lower() or "sum" in result.lower()

    def test_strip_rule_chaining_header(self, strip_headers):
        """Test stripping 'Rule Chaining (Different Query)' header."""
        query = "Rule Chaining (Different Query): If A implies B and B implies C..."
        result = strip_headers(query)
        
        # The actual logic content should remain
        assert "implies" in result.lower() or "if" in result.lower()

    def test_strip_quantifier_scope_header(self, strip_headers):
        """Test stripping 'Quantifier Scope:' header."""
        query = "Quantifier Scope: An engineer reviewed every report..."
        result = strip_headers(query)
        
        # The actual content should remain
        assert "engineer" in result.lower() or "report" in result.lower()

    def test_strip_self_description_header(self, strip_headers):
        """Test stripping 'Self-Description Queries' header."""
        query = "Self-Description Queries: what makes you different from other ai systems?"
        result = strip_headers(query)
        
        # The actual question should remain
        assert "different" in result.lower() or "ai" in result.lower()

    def test_preserve_content_without_headers(self, strip_headers):
        """Test that queries without headers are preserved."""
        query = "What is the probability of rain tomorrow given the current forecast?"
        result = strip_headers(query)
        
        assert result == query

    def test_empty_query(self, strip_headers):
        """Test handling of empty query."""
        assert strip_headers("") == ""
        assert strip_headers(None) is None

    def test_strip_task_prefix(self, strip_headers):
        """Test stripping 'Task:' prefix."""
        query = "Task: Analyze the following data..."
        result = strip_headers(query)
        
        # "Task:" should be stripped
        assert not result.startswith("Task:")
        assert "analyze" in result.lower()


class TestClassificationAfterStripping:
    """Tests verifying that classification works correctly after header stripping."""

    @pytest.fixture
    def classifier(self):
        """Get the classify_query function."""
        from vulcan.routing.query_classifier import classify_query
        return classify_query

    @pytest.fixture
    def strip_headers(self):
        """Get the strip_query_headers function."""
        from vulcan.routing.query_classifier import strip_query_headers
        return strip_headers

    def test_a1_header_not_cryptographic(self, classifier):
        """Test that 'A1' in header doesn't trigger CRYPTOGRAPHIC classification.
        
        This directly tests the fix from the problem statement.
        """
        query = "Analogical Reasoning A1 — Structure mapping: Given the analogy between..."
        result = classifier(query)
        
        # Should NOT be classified as CRYPTOGRAPHIC
        assert result.category != "CRYPTOGRAPHIC", (
            f"Query with 'A1' header should not be CRYPTOGRAPHIC, got {result.category}"
        )

    def test_numeric_verification_not_cryptographic(self, classifier):
        """Test that 'Numeric Verification' doesn't trigger CRYPTOGRAPHIC.
        
        This tests problem statement #5.
        """
        query = "Numeric Verification (∑(2k-1)): Calculate the sum of the first 10 odd numbers..."
        result = classifier(query)
        
        # Should NOT be classified as CRYPTOGRAPHIC just because of "verification"
        # Should be MATHEMATICAL since it's about calculating sums
        assert result.category != "CRYPTOGRAPHIC", (
            f"Mathematical query should not be CRYPTOGRAPHIC, got {result.category}"
        )

    def test_causal_reasoning_not_cryptographic(self, classifier):
        """Test that 'C1' in header doesn't trigger CRYPTOGRAPHIC."""
        query = "Causal Reasoning C1 — Confounding: A study shows correlation between X and Y..."
        result = classifier(query)
        
        # Should NOT be classified as CRYPTOGRAPHIC
        # Should be CAUSAL since it's about causal reasoning
        assert result.category != "CRYPTOGRAPHIC", (
            f"Causal query should not be CRYPTOGRAPHIC, got {result.category}"
        )

    def test_actual_cryptographic_query(self, classifier):
        """Test that actual crypto queries are still classified correctly."""
        query = "What is the SHA-256 hash of 'hello world'?"
        result = classifier(query)
        
        # This SHOULD be classified as CRYPTOGRAPHIC because it's asking
        # for an actual hash computation
        assert result.category == "CRYPTOGRAPHIC", (
            f"Actual crypto query should be CRYPTOGRAPHIC, got {result.category}"
        )


class TestRouterPreprocessingOrder:
    """Tests verifying that QueryRouter applies preprocessing BEFORE classification."""

    @pytest.fixture
    def route_query(self):
        """Get the route_query function from query_router."""
        from vulcan.routing.query_router import route_query
        return route_query

    def test_router_strips_headers_before_classification(self, route_query):
        """Test that route_query strips headers before any classification.
        
        This is the core test for the fix - headers should be stripped
        BEFORE the query is processed.
        """
        # This query has a header that would otherwise trigger wrong routing
        query = "Analogical Reasoning A1 — Structure mapping between atoms and planets"
        
        plan = route_query(query, source="user")
        
        # The original_query should store the original (with headers) for audit
        assert plan.original_query == query, "original_query should preserve the raw query"
        
        # The routing should NOT be based on "A1" cryptographic pattern
        # Check that detected_patterns don't include crypto-related patterns
        crypto_patterns = [p for p in plan.detected_patterns if "crypto" in p.lower()]
        assert not crypto_patterns, (
            f"Headers should be stripped before routing, but got crypto patterns: {crypto_patterns}"
        )

    def test_router_preserves_original_query_for_audit(self, route_query):
        """Test that ProcessingPlan.original_query contains the unmodified query."""
        query = "M1 — Proof check: Prove that 1+1=2"
        
        plan = route_query(query, source="user")
        
        # original_query should have the raw query with headers
        assert plan.original_query == query

    def test_simple_query_not_affected(self, route_query):
        """Test that queries without headers are not affected by stripping."""
        query = "What is the capital of France?"
        
        plan = route_query(query, source="user")
        
        assert plan.original_query == query


class TestEdgeCases:
    """Edge case tests for header stripping."""

    @pytest.fixture
    def strip_headers(self):
        """Get the strip_query_headers function."""
        from vulcan.routing.query_classifier import strip_query_headers
        return strip_query_headers

    def test_multiple_headers_in_query(self, strip_headers):
        """Test handling of multiple headers in one query."""
        query = "M1 — Task: Mathematical Reasoning — Prove that..."
        result = strip_headers(query)
        
        # Multiple header patterns should all be stripped
        assert "M1" not in result
        # The core content should remain
        assert "prove" in result.lower()

    def test_header_in_middle_of_query(self, strip_headers):
        """Test that headers are only stripped from the beginning."""
        query = "Calculate hash(A1) where A1 is a variable"
        result = strip_headers(query)
        
        # "A1" in the middle of the query should NOT be stripped
        # It's part of the query content, not a header
        assert "A1" in result

    def test_newlines_with_headers(self, strip_headers):
        """Test handling of multi-line queries with headers."""
        query = """Causal Reasoning C1 — Confounding

A study shows correlation between coffee and cancer.
Is this a causal relationship?"""
        
        result = strip_headers(query)
        
        # Header should be stripped
        assert "C1" not in result
        # Content should remain
        assert "coffee" in result.lower()
        assert "cancer" in result.lower()
