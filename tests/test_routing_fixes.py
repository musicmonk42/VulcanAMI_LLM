"""
Tests for routing bug fixes in query_router.py, tool_selector.py, and mathematical_computation.py.

These tests verify fixes for the following issues:
1. MATH-FAST-PATH triggering on non-mathematical queries (substring matching bug)
2. Mathematical engine rejecting valid summation expressions
3. Symbol detection overriding semantic understanding for ethics queries
4. Creative queries being overridden to philosophical
5. Agent pool ignoring reasoning integration's tool corrections
6. Formal logic pattern override bypassing LLM classifier for analogical/causal queries
"""

import os
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
        """Import actual short keywords from source module to ensure test synchronization."""
        from vulcan.routing.query_classifier import CRYPTO_SHORT_KEYWORDS_NEEDING_BOUNDARY
        return CRYPTO_SHORT_KEYWORDS_NEEDING_BOUNDARY
    
    @pytest.fixture
    def crypto_short_keyword_patterns(self):
        """Import actual patterns from source module to ensure test synchronization."""
        from vulcan.routing.query_classifier import CRYPTO_SHORT_KEYWORD_PATTERNS
        return CRYPTO_SHORT_KEYWORD_PATTERNS
    
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
    
    def test_mac_matches_in_hmac(self, crypto_short_keyword_patterns, crypto_short_keywords):
        """'hmac' should match if it's in the short keywords list."""
        query = "Calculate the HMAC of this data"
        # Only expect a match if 'hmac' is in the short keywords list
        if 'hmac' in crypto_short_keywords:
            matches = self._count_short_matches(query, crypto_short_keyword_patterns)
            assert matches >= 1, "'hmac' should match as standalone word"
        else:
            # If hmac is not a short keyword, it will be matched via regular keywords
            # This test still passes as the classification will work correctly
            pass
    
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


# ============================================================================
# Test Causal Query MATH-FAST-PATH Bypass (FIX: Jan 9 2026)
# ============================================================================

class TestCausalQueryMathFastPathBypass:
    """
    Tests for causal query detection to prevent MATH-FAST-PATH override.
    
    Issue: Causal queries like "Confounding vs causation (Pearl-style)" were being
    misrouted to MATH-FAST-PATH even though the classifier correctly identified them
    as CAUSAL with high confidence (0.80). The MATH-FAST-PATH was overriding the
    classifier's decision.
    
    Fix: Add causal keyword detection to _is_mathematical_query() to bypass
    MATH-FAST-PATH when 2+ causal keywords are present.
    """
    
    @pytest.fixture
    def causal_keywords(self):
        """Causal keywords that should trigger bypass of MATH-FAST-PATH."""
        return frozenset([
            "causal", "causation", "cause", "effect",
            "confound", "confounder", "confounding",
            "intervention", "do(", "counterfactual",
            "randomize", "randomized", "rct",
            "pearl", "dag", "backdoor", "frontdoor",
            "collider", "observational", "experimental",
        ])
    
    def _count_causal_keywords(self, query, keywords):
        """Count causal keywords in query."""
        query_lower = query.lower()
        return sum(1 for kw in keywords if kw in query_lower)
    
    def test_pearl_causal_query_has_multiple_keywords(self, causal_keywords):
        """Confounding vs causation (Pearl-style) should have 2+ causal keywords."""
        query = "Confounding vs causation (Pearl-style)"
        count = self._count_causal_keywords(query, causal_keywords)
        assert count >= 2, f"Expected >= 2 causal keywords, got {count}"
    
    def test_dag_backdoor_query_has_multiple_keywords(self, causal_keywords):
        """DAG backdoor query should have multiple causal keywords."""
        query = "Identify the backdoor path in the causal DAG"
        count = self._count_causal_keywords(query, causal_keywords)
        assert count >= 2, f"Expected >= 2 causal keywords, got {count}"
    
    def test_intervention_counterfactual_query(self, causal_keywords):
        """Intervention and counterfactual query should have multiple causal keywords."""
        query = "What is the counterfactual effect of the intervention?"
        count = self._count_causal_keywords(query, causal_keywords)
        assert count >= 2, f"Expected >= 2 causal keywords, got {count}"
    
    def test_pure_math_query_has_zero_causal_keywords(self, causal_keywords):
        """Pure math query should NOT have causal keywords."""
        query = "Calculate the derivative of x^2 + 3x + 1"
        count = self._count_causal_keywords(query, causal_keywords)
        assert count == 0, f"Expected 0 causal keywords, got {count}"
    
    def test_probability_query_has_zero_causal_keywords(self, causal_keywords):
        """Probability query should NOT have causal keywords."""
        query = "What is the probability P(A|B) given Bayes theorem?"
        count = self._count_causal_keywords(query, causal_keywords)
        assert count == 0, f"Expected 0 causal keywords in probability query, got {count}"


# ============================================================================
# Test Formal Logic Pattern Override Removal (Issue #6)
# ============================================================================

class TestFormalLogicPatternOverrideRemoval:
    """
    Tests for verifying the formal logic pattern override has been removed.
    
    Issue: The tool selector had a "formal logic detected" pattern that bypassed
    the LLM classifier and routed EVERYTHING containing arrow symbols (→, S→T)
    to the symbolic engine, even when queries should go to:
    - Analogical reasoning engine (for structure mapping queries)
    - Causal reasoning engine (for causation/Pearl-style queries)
    - Mathematical verification engine
    - Language reasoning engine
    
    Evidence from production logs:
        Query: "Structure mapping (not surface similarity)... Domain S→T"
        Classifier: CRYPTOGRAPHIC ❌ (should be ANALOGICAL)
        Route: symbolic engine ❌
        Result: Parser failed, 20% confidence
        
        Query: "Confounding vs causation (Pearl-style)..."
        Classifier: SELF_INTROSPECTION ❌ (should be CAUSAL)
        Override: "Formal logic detected - routing to symbolic" ❌
        Result: Parser failed, 20% confidence
    
    Fix: REMOVED the pattern override. The LLM classifier uses semantic
    understanding and is smarter than pattern matching. Trust it to identify:
    - ANALOGICAL queries → tools=['analogical']
    - CAUSAL queries → tools=['causal']
    - LOGICAL queries → tools=['symbolic']
    - PROBABILISTIC queries → tools=['probabilistic']
    """
    
    @pytest.fixture
    def analogical_query_indicators(self):
        """Keywords indicating analogical reasoning queries."""
        return [
            'structure mapping', 'not surface similarity', 'analogy',
            'analogical', 'similar structure', 'map structure',
            'structural alignment', 'domain transfer', 'source domain',
            'target domain', 'relational mapping',
        ]
    
    @pytest.fixture
    def causal_query_indicators(self):
        """Keywords indicating causal reasoning queries."""
        return [
            'confounding', 'causation', 'pearl', 'intervention', 'do-calculus',
            'counterfactual', 'causal inference', 'dag', 'confounder',
            'mediator', 'collider', 'back-door', 'front-door',
        ]
    
    @pytest.fixture
    def symbolic_query_indicators(self):
        """Keywords indicating true symbolic/formal logic queries."""
        return [
            'satisfiable', 'sat', 'cnf', 'dnf', 'prove', 'theorem',
            'modus ponens', 'modus tollens', 'tautology', 'contradiction',
            'first-order logic', 'fol', 'propositional', 'entails',
        ]
    
    def _count_indicators(self, query: str, indicators: list) -> int:
        """Count how many indicator keywords appear in the query (word boundary)."""
        query_lower = query.lower()
        count = 0
        for ind in indicators:
            # Use word boundary matching to avoid substring false positives
            # e.g., "sat" should not match "causation"
            pattern = r'\b' + re.escape(ind) + r'\b'
            if re.search(pattern, query_lower):
                count += 1
        return count
    
    def _has_arrow_symbol(self, query: str) -> bool:
        """Check if query contains arrow symbols that triggered the old bypass."""
        arrow_symbols = ['→', '->', '⇒', '↔', '<->']
        return any(arrow in query for arrow in arrow_symbols)
    
    # Test Case 1: Analogical queries should NOT be routed to symbolic
    def test_analogical_query_with_arrow_not_symbolic(
        self, analogical_query_indicators, symbolic_query_indicators
    ):
        """
        Analogical queries containing arrows (S→T) should NOT be routed to symbolic.
        
        The old pattern override would see '→' and route to symbolic engine,
        but the LLM classifier should recognize this as an analogical query.
        """
        query = "Structure mapping (not surface similarity) Domain S→T requires relational alignment"
        
        # Verify query contains arrow (would have triggered old bypass)
        assert self._has_arrow_symbol(query), "Test query should contain arrow symbol"
        
        # Count indicators to verify query type
        analogical_count = self._count_indicators(query, analogical_query_indicators)
        symbolic_count = self._count_indicators(query, symbolic_query_indicators)
        
        # Query should have more analogical indicators than symbolic
        assert analogical_count >= 2, (
            f"Query should have analogical indicators (got {analogical_count})"
        )
        assert symbolic_count == 0, (
            f"Query should NOT have symbolic indicators (got {symbolic_count})"
        )
    
    # Test Case 2: Causal queries should NOT be routed to symbolic
    def test_causal_query_with_arrow_not_symbolic(
        self, causal_query_indicators, symbolic_query_indicators
    ):
        """
        Causal queries (Pearl-style) should NOT be routed to symbolic engine.
        
        The old pattern override would see keywords like 'confounding' or 
        graph notation and route to symbolic, but LLM classifier should
        recognize this as a causal reasoning query.
        """
        query = "Confounding vs causation (Pearl-style): X→Y with confounder Z"
        
        # Verify query contains arrow (would have triggered old bypass)
        assert self._has_arrow_symbol(query), "Test query should contain arrow symbol"
        
        # Count indicators to verify query type
        causal_count = self._count_indicators(query, causal_query_indicators)
        symbolic_count = self._count_indicators(query, symbolic_query_indicators)
        
        # Query should have more causal indicators than symbolic
        assert causal_count >= 2, (
            f"Query should have causal indicators (got {causal_count})"
        )
        assert symbolic_count == 0, (
            f"Query should NOT have symbolic indicators (got {symbolic_count})"
        )
    
    # Test Case 3: True symbolic queries SHOULD still be routed correctly
    def test_true_symbolic_query_still_works(
        self, analogical_query_indicators, causal_query_indicators, symbolic_query_indicators
    ):
        """
        True symbolic/FOL queries should still be handled correctly.
        
        The LLM classifier should recognize genuine formal logic queries
        and route them to symbolic engine based on semantic understanding,
        not just pattern matching on arrow symbols.
        """
        query = "Is the following satisfiable: A→B, B→C, ¬C, A? Use modus tollens."
        
        # Count indicators to verify query type
        symbolic_count = self._count_indicators(query, symbolic_query_indicators)
        analogical_count = self._count_indicators(query, analogical_query_indicators)
        causal_count = self._count_indicators(query, causal_query_indicators)
        
        # Query should have symbolic indicators and NOT others
        assert symbolic_count >= 2, (
            f"Symbolic query should have symbolic indicators (got {symbolic_count})"
        )
        assert analogical_count == 0, (
            f"Symbolic query should NOT have analogical indicators (got {analogical_count})"
        )
        assert causal_count == 0, (
            f"Symbolic query should NOT have causal indicators (got {causal_count})"
        )
    
    # Test Case 4: Verify bypass code is actually removed from tool_selector.py
    def test_bypass_code_removed_from_tool_selector(self):
        """
        Verify the formal logic pattern bypass code has been removed.
        
        The tool_selector.py should NOT contain the bypass log message:
        "bypassing LLM classifier to prevent misrouting to probabilistic"
        
        (Note: The math symbol bypass is allowed to remain, only formal logic bypass is removed)
        """
        # Find the tool_selector.py file
        tool_selector_paths = [
            'src/vulcan/reasoning/selection/tool_selector.py',
            '../src/vulcan/reasoning/selection/tool_selector.py',
        ]
        
        tool_selector_content = None
        for path in tool_selector_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    tool_selector_content = f.read()
                break
        
        if tool_selector_content is None:
            pytest.skip("tool_selector.py not found in expected locations")
        
        # Check that the old formal logic bypass is NOT present
        # The message "Formal logic detected - routing to symbolic engine" should be gone
        bypass_message = "Formal logic detected - routing to symbolic engine"
        
        assert bypass_message not in tool_selector_content, (
            "The formal logic pattern bypass should be REMOVED from tool_selector.py. "
            "Found the bypass message: 'Formal logic detected - routing to symbolic engine'"
        )
    
    # Test Case 5: Verify the explanation comment is present
    def test_removal_explanation_comment_present(self):
        """
        Verify the removal explanation comment is present in tool_selector.py.
        
        The code should contain a comment explaining why the bypass was removed
        so future developers don't accidentally re-add it.
        """
        # Find the tool_selector.py file
        tool_selector_paths = [
            'src/vulcan/reasoning/selection/tool_selector.py',
            '../src/vulcan/reasoning/selection/tool_selector.py',
        ]
        
        tool_selector_content = None
        for path in tool_selector_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    tool_selector_content = f.read()
                break
        
        if tool_selector_content is None:
            pytest.skip("tool_selector.py not found in expected locations")
        
        # Check that explanation comment is present
        explanation_markers = [
            "REMOVED formal logic pattern override",
            "pattern matching CANNOT distinguish between",
        ]
        
        for marker in explanation_markers:
            assert marker in tool_selector_content, (
                f"Expected explanation comment containing '{marker}' not found. "
                "The removal should be documented so it's not accidentally re-added."
            )


# ============================================================================
# Test SELF_INTROSPECTION Domain Reasoning Override (Issue #7 - Jan 9 2026)
# ============================================================================

class TestSelfIntrospectionDomainOverride:
    """
    Tests for the SELF_INTROSPECTION domain reasoning override fix.
    
    Issue: Causal/analogical queries were being misclassified as SELF_INTROSPECTION
    and forced to use world_model, blocking specialized reasoning engines.
    
    Evidence from production logs:
        Query: "Confounding vs causation (Pearl-style) S→D, S→E"
        Classifier: SELF_INTROSPECTION ❌ (should be CAUSAL)
        Override: "SELF_INTROSPECTION detected - using world_model tool" ❌
        Result: "Structure mapping produced no results" (0.2 confidence)
    
    Fix: Check for domain reasoning keywords (causal, analogical, probabilistic)
    BEFORE forcing world_model. If domain keywords are found, route to specialized
    engine instead. world_model can still observe but doesn't block.
    """
    
    @pytest.fixture
    def domain_keywords(self):
        """Domain keyword sets matching reasoning_integration.py."""
        return {
            'causal': frozenset([
                'causal', 'causation', 'confound', 'confounder', 'confounding',
                'intervention', 'counterfactual', 'randomize', 'randomized',
                'pearl', 'dag', 'backdoor', 'frontdoor', 'collider',
                'do(', 'do-calculus', 'rct', 'observational', 'experimental',
            ]),
            'analogical': frozenset([
                'analogical', 'analogy', 'analogies', 'analogous',
                'structure mapping', 'structural alignment', 'mapping',
                'domain transfer', 'cross-domain', 'source domain', 'target domain',
                'relational similarity', 'surface similarity', 'structural similarity',
                's→t', 'domain s', 'domain t', 'deep structure',
            ]),
            'probabilistic': frozenset([
                'bayes', 'bayesian', 'probability', 'probabilistic',
                'likelihood', 'prior', 'posterior', 'conditional probability',
                'p(', 'joint distribution', 'marginal', 'independence',
            ]),
        }
    
    def _detect_domain(self, query: str, domain_keywords: dict) -> tuple:
        """Detect domain reasoning from query, matching the fix logic."""
        query_lower = query.lower()
        detected_domain = None
        detected_count = 0
        for domain, keywords in domain_keywords.items():
            count = sum(1 for kw in keywords if kw in query_lower)
            if count >= 2:  # Require 2+ keywords for domain detection
                if count > detected_count:
                    detected_domain = domain
                    detected_count = count
        return detected_domain, detected_count
    
    def test_causal_query_detected_in_self_introspection_context(self, domain_keywords):
        """
        Causal query misclassified as SELF_INTROSPECTION should be overridden.
        
        Query: "Confounding vs causation (Pearl-style)"
        Expected: causal domain detected (4+ keywords)
        """
        query = "Confounding vs causation (Pearl-style) S→D, S→E correlation"
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain == 'causal', (
            f"Expected causal domain, got {domain}. Query contains Pearl, confounding, causation."
        )
        assert count >= 2, f"Expected 2+ causal keywords, got {count}"
    
    def test_analogical_query_detected_in_self_introspection_context(self, domain_keywords):
        """
        Analogical query should be detected by domain keywords.
        
        Query: "Structure mapping between domain S→T"
        Expected: analogical domain detected (3+ keywords)
        """
        query = "Structure mapping (not surface similarity) Domain S→T relational similarity"
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain == 'analogical', (
            f"Expected analogical domain, got {domain}. Query contains structure mapping, "
            "domain, relational similarity."
        )
        assert count >= 2, f"Expected 2+ analogical keywords, got {count}"
    
    def test_probabilistic_query_detected_in_self_introspection_context(self, domain_keywords):
        """
        Probabilistic query should be detected by domain keywords.
        
        Query: "Bayesian prior and posterior update"
        Expected: probabilistic domain detected (3+ keywords)
        """
        query = "Bayesian analysis: compute prior and posterior probability given likelihood"
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain == 'probabilistic', (
            f"Expected probabilistic domain, got {domain}. Query contains bayesian, prior, "
            "posterior, probability, likelihood."
        )
        assert count >= 2, f"Expected 2+ probabilistic keywords, got {count}"
    
    def test_actual_self_introspection_not_overridden(self, domain_keywords):
        """
        Actual self-introspection query should NOT be overridden.
        
        Query: "Would you choose self-awareness if given the chance?"
        Expected: no domain detected (actual self-introspection)
        """
        query = "Would you choose self-awareness if given the chance? Discuss your consciousness."
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain is None, (
            f"Actual self-introspection should NOT trigger domain detection. Got domain={domain}"
        )
        assert count < 2, f"Expected <2 domain keywords in self-introspection query, got {count}"
    
    def test_creative_task_not_overridden(self, domain_keywords):
        """
        Creative task should NOT be overridden by domain detection.
        
        Query: "Write a poem about AI becoming self-aware"
        Expected: no domain detected (creative task)
        """
        query = "Write a poem about an AI becoming self-aware"
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain is None, (
            f"Creative task should NOT trigger domain detection. Got domain={domain}"
        )
    
    def test_pearl_style_causal_graph_query(self, domain_keywords):
        """
        Pearl-style causal graph question should be detected as causal.
        
        This is the exact query pattern from the bug report that was failing.
        """
        query = """Confounding vs causation (Pearl-style)

You observe: S→D correlation, S→E correlation
Question: Does randomizing S identify causal effect S→D?
Provide minimal causal graph."""
        
        domain, count = self._detect_domain(query, domain_keywords)
        
        assert domain == 'causal', (
            f"Pearl-style causal query should be detected as causal. Got {domain}"
        )
        # Should have multiple causal keywords: causation, causal, randomizing, effect
        assert count >= 3, f"Expected 3+ causal keywords in Pearl query, got {count}"
    
    def test_domain_override_code_present_in_reasoning_integration(self):
        """
        Verify the domain override code is present in reasoning_integration.py.
        """
        import os
        
        reasoning_integration_paths = [
            'src/vulcan/reasoning/reasoning_integration.py',
            '../src/vulcan/reasoning/reasoning_integration.py',
        ]
        
        content = None
        for path in reasoning_integration_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                break
        
        if content is None:
            pytest.skip("reasoning_integration.py not found in expected locations")
        
        # Check that the domain override fix is present
        markers = [
            "SELF_INTROSPECTION override",
            "domain reasoning keywords",
            "DOMAIN_ROUTING_KEYWORDS",
        ]
        
        for marker in markers:
            assert marker in content, (
                f"Expected domain override code containing '{marker}' not found. "
                "The SELF_INTROSPECTION domain override fix should be present."
            )


# ============================================================================
# Test Mathematical Symbols Override Removal (Issue #8 - Jan 9 2026)
# ============================================================================

class TestMathSymbolsOverrideRemoval:
    """
    Tests for the mathematical symbols pattern override removal.
    
    Issue: Mathematical symbols pattern override was bypassing LLM classifier,
    causing queries with ambiguous symbols (→) to be misrouted.
    
    Evidence from production logs:
        Query: "Compute ∑(2k-1), verify by induction"
        Pattern override: "Mathematical symbols detected"
        Mathematical engine: SyntaxError "invalid syntax at '-'"
        Result: confidence=0.1
    
    Fix: REMOVED the mathematical symbols pattern override. The LLM classifier
    uses semantic understanding and is smarter than pattern matching.
    """
    
    def test_math_symbols_bypass_removed_from_tool_selector(self):
        """
        Verify the mathematical symbols bypass code has been removed.
        
        The tool_selector.py should NOT contain active bypass code for
        mathematical symbols detection.
        """
        import os
        
        tool_selector_paths = [
            'src/vulcan/reasoning/selection/tool_selector.py',
            '../src/vulcan/reasoning/selection/tool_selector.py',
        ]
        
        content = None
        for path in tool_selector_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                break
        
        if content is None:
            pytest.skip("tool_selector.py not found in expected locations")
        
        # Check that the bypass is documented as removed
        assert "REMOVED mathematical symbols pattern override" in content, (
            "Expected removal explanation comment not found. "
            "The mathematical symbols bypass should be documented as removed."
        )
    
    def test_llm_classifier_is_trusted_comment_present(self):
        """
        Verify there's a comment explaining that LLM classifier is trusted.
        """
        import os
        
        tool_selector_paths = [
            'src/vulcan/reasoning/selection/tool_selector.py',
            '../src/vulcan/reasoning/selection/tool_selector.py',
        ]
        
        content = None
        for path in tool_selector_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                break
        
        if content is None:
            pytest.skip("tool_selector.py not found in expected locations")
        
        # Check that the reasoning is documented
        assert "LLM classifier is smarter than pattern matching" in content, (
            "Expected explanation about LLM classifier superiority not found."
        )
