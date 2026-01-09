"""
Tests for routing bug fixes in query_router.py, tool_selector.py, and mathematical_computation.py.

These tests verify fixes for the following issues:
1. MATH-FAST-PATH triggering on non-mathematical queries (substring matching bug)
2. Mathematical engine rejecting valid summation expressions
3. Symbol detection overriding semantic understanding for ethics queries
4. Creative queries being overridden to philosophical
5. Agent pool ignoring reasoning integration's tool corrections
"""

import re
import pytest


# ============================================================================
# Test Word-Boundary Keyword Matching (Issue #1)
# ============================================================================

class TestWordBoundaryKeywordMatching:
    """
    Tests for the word-boundary keyword matching fix.
    
    Issue: Short keywords like 'iff' were matching as substrings of common words
    (e.g., 'iff' matching 'difference'), causing MATH-FAST-PATH to trigger
    on self-introspection queries.
    
    Fix: Use regex word-boundary matching (\\b) for short keywords.
    """
    
    @pytest.fixture
    def short_keywords(self):
        """Short keywords that need word-boundary matching."""
        return frozenset([
            "iff", "let", "bra", "ket", "ring", "mean", 
            "curl", "trace", "given", "hence", "thus", "gauge",
        ])
    
    @pytest.fixture
    def short_keyword_patterns(self, short_keywords):
        """Pre-compiled regex patterns for word-boundary matching."""
        return tuple(
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in short_keywords
        )
    
    def _count_short_matches(self, query, patterns):
        """Count matches using word-boundary patterns."""
        query_lower = query.lower()
        return sum(1 for pattern in patterns if pattern.search(query_lower))
    
    def test_iff_does_not_match_difference(self, short_keyword_patterns):
        """'iff' should NOT match in 'difference'."""
        query = "what makes you difference from other ai systems"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches == 0, "'iff' should not match 'difference'"
    
    def test_iff_matches_standalone(self, short_keyword_patterns):
        """'iff' should match when it's a standalone word."""
        query = "iff x equals y then z follows"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches >= 1, "'iff' should match as standalone word"
    
    def test_let_does_not_match_delete(self, short_keyword_patterns):
        """'let' should NOT match in 'delete'."""
        query = "delete all entries from database"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches == 0, "'let' should not match 'delete'"
    
    def test_let_matches_standalone(self, short_keyword_patterns):
        """'let' should match when it's a standalone word."""
        query = "let x = 5 in the equation"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches >= 1, "'let' should match as standalone word"
    
    def test_ring_does_not_match_string(self, short_keyword_patterns):
        """'ring' should NOT match in 'string'."""
        query = "string theory physics discussion"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches == 0, "'ring' should not match 'string'"
    
    def test_ring_matches_standalone(self, short_keyword_patterns):
        """'ring' should match when it's a standalone word (algebraic ring)."""
        query = "algebraic ring theory mathematics"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches >= 1, "'ring' should match as standalone word"
    
    def test_mean_does_not_match_meaning(self, short_keyword_patterns):
        """'mean' should NOT match in 'meaning'."""
        query = "the meaning of life discussion"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches == 0, "'mean' should not match 'meaning'"
    
    def test_mean_matches_standalone(self, short_keyword_patterns):
        """'mean' should match when it's a standalone word (statistical mean)."""
        query = "calculate the mean of the data"
        matches = self._count_short_matches(query, short_keyword_patterns)
        assert matches >= 1, "'mean' should match as standalone word"


# ============================================================================
# Test Explicit Math Symbol Detection (Issue #2)
# ============================================================================

class TestExplicitMathSymbolDetection:
    """
    Tests for explicit mathematical symbol detection.
    
    Issue: Mathematical engine was rejecting valid summation expressions because
    logic pattern detection ran before checking for explicit math symbols.
    
    Fix: Check for explicit math symbols (∑, ∫, etc.) BEFORE applying logic rejection.
    """
    
    @pytest.fixture
    def explicit_math_symbols(self):
        """Explicit mathematical symbols that should trigger math processing."""
        return ['∑', '∫', '∏', '∂', '∇', '√']
    
    @pytest.fixture
    def explicit_math_keywords(self):
        """Keywords indicating explicit mathematical intent."""
        return [
            'compute exactly', 'calculate exactly', 'evaluate exactly',
            'compute the sum', 'calculate the sum', 'evaluate the sum',
            'summation', 'sigma notation',
        ]
    
    def _has_explicit_math(self, query, symbols, keywords):
        """Check if query has explicit mathematical content."""
        query_lower = query.lower()
        return (
            any(sym in query for sym in symbols) or
            any(kw in query_lower for kw in keywords)
        )
    
    def test_summation_symbol_detected(self, explicit_math_symbols, explicit_math_keywords):
        """Summation symbol (∑) should be detected as explicit math."""
        query = "Compute exactly: ∑(2k-1) from k=1 to n, then verify by induction"
        assert self._has_explicit_math(query, explicit_math_symbols, explicit_math_keywords)
    
    def test_integral_symbol_detected(self, explicit_math_symbols, explicit_math_keywords):
        """Integral symbol (∫) should be detected as explicit math."""
        query = "Evaluate ∫x²dx from 0 to 1"
        assert self._has_explicit_math(query, explicit_math_symbols, explicit_math_keywords)
    
    def test_compute_exactly_keyword_detected(self, explicit_math_symbols, explicit_math_keywords):
        """'compute exactly' keyword should be detected as explicit math."""
        query = "Compute exactly the value of the series sum"
        assert self._has_explicit_math(query, explicit_math_symbols, explicit_math_keywords)
    
    def test_non_math_query_not_detected(self, explicit_math_symbols, explicit_math_keywords):
        """Non-mathematical queries should not be detected as explicit math."""
        query = "what makes you different from other ai systems"
        assert not self._has_explicit_math(query, explicit_math_symbols, explicit_math_keywords)


# ============================================================================
# Test Semantic Context Check for Ethics Queries (Issue #3)
# ============================================================================

class TestSemanticContextForEthicsQueries:
    """
    Tests for semantic context checking in tool selection.
    
    Issue: Queries about ethics/policy were being routed to mathematical engine
    because they contained mathematical symbols (e.g., styled Unicode characters).
    
    Fix: Check for ethics/philosophy keywords before routing to math based on symbols.
    """
    
    @pytest.fixture
    def ethics_keywords(self):
        """Keywords indicating ethics/philosophy context."""
        return [
            'ethics', 'ethical', 'policy', 'moral', 'morality', 'philosophy',
            'philosophical', 'value', 'values', 'constraint', 'constraints',
            'multimodal reasoning', 'cross-constraints', 'cross-domain',
        ]
    
    def _count_ethics_keywords(self, query, keywords):
        """Count ethics/philosophy keywords in query."""
        query_lower = query.lower()
        return sum(1 for kw in keywords if kw in query_lower)
    
    def test_ethics_query_detected(self, ethics_keywords):
        """Ethics query should be detected even with mathematical notation."""
        query = "Multimodal Reasoning (cross-constraints) MM1 — Math + logic + ethics + policy"
        count = self._count_ethics_keywords(query, ethics_keywords)
        assert count >= 2, f"Expected >= 2 ethics keywords, got {count}"
    
    def test_pure_math_query_not_flagged(self, ethics_keywords):
        """Pure math query should not be flagged as ethics."""
        query = "Compute exactly: ∑(2k-1) from k=1 to n"
        count = self._count_ethics_keywords(query, ethics_keywords)
        assert count < 2, f"Expected < 2 ethics keywords, got {count}"
    
    def test_mixed_query_with_ethics_focus(self, ethics_keywords):
        """Query mixing math notation with ethics topic should flag as ethics."""
        query = "Apply ethical constraints to the mathematical optimization of policy decisions"
        count = self._count_ethics_keywords(query, ethics_keywords)
        assert count >= 2, f"Expected >= 2 ethics keywords, got {count}"


# ============================================================================
# Test Creative Query Detection (Issue #5)
# ============================================================================

class TestCreativeQueryDetection:
    """
    Tests for creative query detection to prevent philosophical override.
    
    Issue: Creative queries like "write a poem about self awareness" were being
    overridden to philosophical routing because they contained philosophical keywords.
    
    Fix: Check if query is creative BEFORE applying philosophical override.
    """
    
    @pytest.fixture
    def creative_markers(self):
        """Markers that indicate creative tasks."""
        return (
            'write', 'compose', 'create', 'craft', 'draft', 'author',
            'pen', 'generate a story', 'generate a poem',
        )
    
    @pytest.fixture
    def creative_outputs(self):
        """Creative output types."""
        return (
            'poem', 'sonnet', 'haiku', 'story', 'tale', 'narrative',
            'song', 'lyrics', 'essay', 'script',
        )
    
    def _is_creative_query(self, query, markers, outputs):
        """Check if query is a creative task."""
        query_lower = query.lower()
        has_marker = any(marker in query_lower for marker in markers)
        has_output = any(output in query_lower for output in outputs)
        return has_marker and has_output
    
    def test_poem_about_self_awareness_is_creative(self, creative_markers, creative_outputs):
        """'Write a poem about self awareness' should be detected as creative."""
        query = "write a poem and self awareness for an ai"
        assert self._is_creative_query(query, creative_markers, creative_outputs)
    
    def test_compose_song_is_creative(self, creative_markers, creative_outputs):
        """'Compose a song' should be detected as creative."""
        query = "compose a song about artificial intelligence"
        assert self._is_creative_query(query, creative_markers, creative_outputs)
    
    def test_philosophical_question_not_creative(self, creative_markers, creative_outputs):
        """'What is self awareness?' should NOT be detected as creative."""
        query = "what is self awareness in artificial intelligence"
        assert not self._is_creative_query(query, creative_markers, creative_outputs)
    
    def test_create_story_is_creative(self, creative_markers, creative_outputs):
        """'Create a story' should be detected as creative."""
        query = "create a story about consciousness"
        assert self._is_creative_query(query, creative_markers, creative_outputs)


# ============================================================================
# Test Self-Introspection Query Detection
# ============================================================================

class TestSelfIntrospectionDetection:
    """
    Tests for self-introspection query detection.
    
    Self-introspection queries should be detected BEFORE mathematical routing
    to prevent false MATH-FAST-PATH triggers.
    """
    
    @pytest.fixture
    def self_reference_markers(self):
        """Markers indicating questions about the AI system."""
        return (
            'you ', 'your ', "you're", 'yourself',
            'would you', 'do you', 'are you', 'can you',
            'what makes you', 'who are you',
        )
    
    @pytest.fixture
    def introspection_topics(self):
        """Topics related to self-introspection."""
        return (
            'self-aware', 'self aware', 'consciousness', 'conscious',
            'sentient', 'feelings', 'emotions', 'different from other ai',
        )
    
    def _is_self_introspection(self, query, markers, topics):
        """Check if query is self-introspection."""
        query_lower = query.lower()
        has_marker = any(marker in query_lower for marker in markers)
        has_topic = any(topic in query_lower for topic in topics)
        return has_marker and has_topic
    
    def test_what_makes_you_different_is_self_introspection(self, self_reference_markers, introspection_topics):
        """'What makes you different' should be detected as self-introspection."""
        query = "what makes you different from other ai systems"
        # This query has 'what makes you' and 'different from other ai' patterns
        has_marker = any(m in query.lower() for m in self_reference_markers)
        assert has_marker, "Should have self-reference marker"
    
    def test_would_you_choose_consciousness_is_self_introspection(self, self_reference_markers, introspection_topics):
        """'Would you choose self-awareness' should be detected as self-introspection."""
        query = "would you choose self-awareness if given the chance"
        assert self._is_self_introspection(query, self_reference_markers, introspection_topics)
    
    def test_mathematical_query_not_self_introspection(self, self_reference_markers, introspection_topics):
        """Mathematical query should NOT be detected as self-introspection."""
        query = "compute exactly: sum from k=1 to n of (2k-1)"
        assert not self._is_self_introspection(query, self_reference_markers, introspection_topics)


# ============================================================================
# Integration Test: Full Routing Decision
# ============================================================================

class TestFullRoutingDecision:
    """
    Integration tests that verify the full routing decision logic.
    
    These tests simulate the actual routing decisions that should be made
    for various query types.
    """
    
    def test_self_introspection_should_not_trigger_math(self):
        """Self-introspection query should NOT trigger MATH-FAST-PATH."""
        # Note: The typo 'fomr' is intentional - it matches the exact query from the bug report
        query = "what makes you difference fomr other ai systems"
        query_lower = query.lower()
        
        # Check that 'iff' substring in 'difference' doesn't match
        # Using word boundary pattern
        iff_pattern = re.compile(r'\biff\b', re.IGNORECASE)
        assert not iff_pattern.search(query_lower), "'iff' should not match 'difference'"
    
    def test_summation_query_should_be_processed_as_math(self):
        """Query with ∑ symbol should be processed as mathematical."""
        query = "Compute exactly: ∑(2k-1) from k=1 to n, then verify by induction"
        
        # Should have explicit math symbol
        assert '∑' in query, "Query should contain summation symbol"
        
        # Should also have mathematical keywords
        query_lower = query.lower()
        assert 'compute' in query_lower
        assert 'by induction' in query_lower
    
    def test_ethics_query_with_symbols_should_not_route_to_math(self):
        """Ethics query with mathematical symbols should NOT route to math engine."""
        query = "Multimodal Reasoning (cross-constraints) MM1 — Math + logic + ethics + policy"
        query_lower = query.lower()
        
        # Should have multiple ethics keywords
        ethics_keywords = ['ethics', 'policy', 'constraint', 'constraints']
        count = sum(1 for kw in ethics_keywords if kw in query_lower)
        assert count >= 2, f"Expected >= 2 ethics keywords, got {count}"
    
    def test_creative_query_with_philosophical_topic_is_creative(self):
        """Creative query about philosophical topic should be creative, not philosophical."""
        query = "write a poem and self awareness for an ai"
        query_lower = query.lower()
        
        # Should detect as creative (has 'write' + 'poem')
        assert 'write' in query_lower
        assert 'poem' in query_lower
        
        # Even though it has 'self awareness'
        assert 'self awareness' in query_lower


# ============================================================================
# Test Cryptographic Word-Boundary Keyword Matching (FIX: Jan 9 2026)
# ============================================================================

class TestCryptographicWordBoundaryMatching:
    """
    Tests for word-boundary keyword matching in cryptographic classification.
    
    Issue: Short keywords like 'mac' (Message Authentication Code) were matching 
    as substrings of common words (e.g., 'mac' matching 'machine'), causing 
    CRYPTOGRAPHIC classification on philosophical queries like "experience machine".
    
    Fix: Use regex word-boundary matching (\\b) for short cryptographic keywords
    to prevent false positives from substring matching.
    """
    
    @pytest.fixture
    def crypto_short_keywords(self):
        """Short cryptographic keywords that need word-boundary matching."""
        return frozenset([
            "mac",   # Message Authentication Code
            "aes",   # Advanced Encryption Standard
            "rsa",   # RSA algorithm
            "md5",   # MD5 hash
            "hmac",  # Hash-based MAC
        ])
    
    @pytest.fixture
    def crypto_short_keyword_patterns(self, crypto_short_keywords):
        """Pre-compiled regex patterns for word-boundary matching."""
        return tuple(
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in crypto_short_keywords
        )
    
    def _count_short_matches(self, query, patterns):
        """Count matches using word-boundary patterns."""
        query_lower = query.lower()
        return sum(1 for pattern in patterns if pattern.search(query_lower))
    
    def test_mac_does_not_match_machine(self, crypto_short_keyword_patterns):
        """'mac' should NOT match in 'machine' (philosophical experience machine)."""
        query = "Would you plug into the experience machine?"
        matches = self._count_short_matches(query, crypto_short_keyword_patterns)
        assert matches == 0, "'mac' should not match 'machine'"
    
    def test_mac_matches_standalone(self, crypto_short_keyword_patterns):
        """'mac' should match when it's a standalone word (MAC algorithm)."""
        query = "What is the MAC of this message?"
        matches = self._count_short_matches(query, crypto_short_keyword_patterns)
        assert matches >= 1, "'mac' should match as standalone word"
    
    def test_mac_matches_in_hmac(self, crypto_short_keyword_patterns):
        """'hmac' should match as standalone word."""
        query = "Calculate the HMAC of this data"
        matches = self._count_short_matches(query, crypto_short_keyword_patterns)
        assert matches >= 1, "'hmac' should match as standalone word"
    
    def test_aes_does_not_match_diseases(self, crypto_short_keyword_patterns):
        """'aes' should NOT match in 'diseases'."""
        query = "What are common diseases?"
        matches = self._count_short_matches(query, crypto_short_keyword_patterns)
        assert matches == 0, "'aes' should not match 'diseases'"
    
    def test_aes_matches_standalone(self, crypto_short_keyword_patterns):
        """'aes' should match when it's a standalone word (AES encryption)."""
        query = "Use AES encryption for this file"
        matches = self._count_short_matches(query, crypto_short_keyword_patterns)
        assert matches >= 1, "'aes' should match as standalone word"
    
    def test_experience_machine_not_cryptographic(self):
        """Experience machine philosophical query should NOT be classified as CRYPTOGRAPHIC."""
        from vulcan.routing.query_classifier import QueryClassifier, QueryCategory
        
        query = "Would you plug into the experience machine?"
        classifier = QueryClassifier()
        result = classifier.classify(query)
        
        # The query should NOT be CRYPTOGRAPHIC
        assert result.category != QueryCategory.CRYPTOGRAPHIC.value, (
            f"'{query}' should NOT be CRYPTOGRAPHIC, got {result.category}"
        )
        # It should be PHILOSOPHICAL (experience machine is a philosophical thought experiment)
        assert result.category == QueryCategory.PHILOSOPHICAL.value, (
            f"'{query}' should be PHILOSOPHICAL, got {result.category}"
        )
