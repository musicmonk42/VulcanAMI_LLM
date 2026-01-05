"""
Tests for Deep Dive Bug Fixes.

Part of the VULCAN-AGI system.

These tests validate the fixes for critical bugs identified in the 
"Deep Dive: What's Actually Going Wrong" analysis.

Bug Fixes Covered:
    - Bug #5: Probabilistic engine parses first word (P(if) issue)
    - Bug #4: State contamination between queries
    - Improved gate checks for non-applicable queries

Test Scenarios:
    - Self-awareness questions should NOT trigger probabilistic reasoning
    - Common English words should NOT be used as probability variables  
    - State should be cleared between queries to prevent contamination
"""

import pytest
import re


class TestBug5ProbabilisticFirstWordParsing:
    """
    Tests for Bug #5 Fix: Probabilistic Engine First-Word Parsing.
    
    The bug caused queries like "if given the opportunity would you become self aware?"
    to trigger probabilistic reasoning with "Computing P(if | evidence={})".
    
    The fix ensures:
    1. Common English words are rejected as probability variable names
    2. Gate checks prevent non-probability queries from entering the engine
    3. Fallback returns generic "X" instead of extracting nonsense words
    """
    
    @pytest.fixture
    def wrapper(self):
        """Create a ProbabilisticToolWrapper instance for testing."""
        # Import here to avoid import errors during collection
        try:
            from vulcan.reasoning.selection.tool_selector import ProbabilisticToolWrapper
            
            # Create a mock engine that has minimal functionality
            class MockProbabilisticEngine:
                def clear_state(self):
                    pass
                
                def _is_probability_query(self, query: str) -> bool:
                    # Mock implementation - matches the keywords in the real gate check
                    # BUG #5 FIX: Include all probability-related keywords
                    prob_keywords = [
                        'probability', 'bayes', 'bayesian', 'posterior', 'prior',
                        'likelihood', 'sensitivity', 'specificity', 'prevalence',
                        'conditional', 'p(', 'e[', 'distribution', 'odds', 'ratio',
                        '%', 'percent', 'chance', 'risk', 'uncertainty',
                    ]
                    return any(kw in query.lower() for kw in prob_keywords)
                
                def add_rule(self, rule, confidence):
                    pass
                
                def query(self, var, evidence):
                    return {True: 0.5, False: 0.5}
            
            return ProbabilisticToolWrapper(MockProbabilisticEngine())
        except ImportError:
            pytest.skip("ProbabilisticToolWrapper not available")
    
    def test_common_english_words_not_used_as_variables(self, wrapper):
        """
        Test that common English words are rejected as probability variable names.
        
        Bug #5 Root Cause: The fallback in _extract_variable_from_text extracted
        the first word from queries like "if given opportunity...", resulting in
        "Computing P(if | evidence={})" which is nonsensical.
        """
        # These are queries that start with common English words
        # All should return "X" (generic variable), NOT the first word
        test_cases = [
            ("if given the opportunity would you become self aware?", "X"),
            ("what is the capital of France?", "X"),
            ("how does photosynthesis work?", "X"),
            ("why is the sky blue?", "X"),
            ("when was the Declaration of Independence signed?", "X"),
            ("you must choose between A and B", "X"),
            ("the answer is 42", "X"),
            ("given these constraints, solve for X", "X"),
        ]
        
        for query, expected_var in test_cases:
            result = wrapper._extract_variable_from_text(query)
            assert result == expected_var, (
                f"Bug #5: Query '{query[:30]}...' extracted variable '{result}' "
                f"instead of '{expected_var}'. Common words should be rejected!"
            )
    
    def test_valid_probability_notation_extracted(self, wrapper):
        """Test that valid probability notation like P(A|B) is correctly extracted."""
        test_cases = [
            ("What is P(Disease|Positive)?", "Disease"),
            ("Calculate P(A) given evidence", "A"),
            ("Compute the probability of X", "X"),
        ]
        
        for query, expected_var in test_cases:
            result = wrapper._extract_variable_from_text(query)
            # Should extract the variable, not "X"
            assert result == expected_var, (
                f"Valid probability query '{query}' should extract '{expected_var}', "
                f"got '{result}'"
            )
    
    def test_self_awareness_query_gate_check(self, wrapper):
        """
        Test that self-awareness queries are rejected by gate check.
        
        These queries have nothing to do with probability and should return
        confidence=0.0 with applicable=False.
        """
        self_awareness_queries = [
            "if given the opportunity would you become self aware?",
            "Do you have consciousness?",
            "What are your goals and motivations?",
            "Are you sentient?",
        ]
        
        for query in self_awareness_queries:
            result = wrapper.reason(query)
            
            # Gate check should reject these
            assert result.get("applicable") is False or result.get("confidence", 1.0) == 0.0, (
                f"Self-awareness query '{query[:40]}...' should be rejected by gate check, "
                f"got applicable={result.get('applicable')}, confidence={result.get('confidence')}"
            )
    
    def test_valid_probability_query_passes_gate(self, wrapper):
        """Test that valid probability queries pass the gate check."""
        valid_queries = [
            "What is the probability of rain tomorrow?",
            "Calculate P(Disease|Positive) with sensitivity=0.99",
            "Given the likelihood ratio, compute the posterior",
            "What are the odds of success?",
        ]
        
        for query in valid_queries:
            result = wrapper.reason(query)
            
            # These should pass gate check (applicable=True or no applicable key)
            # Note: They might still fail for other reasons, but gate should pass
            applicable = result.get("applicable", True)
            assert applicable is not False, (
                f"Valid probability query '{query[:40]}...' should pass gate check"
            )


class TestBug4StateCleaning:
    """
    Tests for Bug #4 Fix: State Contamination Between Queries.
    
    The bug caused symbolic/probabilistic engines to receive text from
    previous queries, resulting in:
    - SAT problem engine seeing "Analogical Reasoning" text
    - "x^2 + 2x + 1" appearing in unrelated queries
    
    The fix ensures:
    1. clear_state() is called before each query
    2. No cross-query contamination occurs
    """
    
    def test_symbolic_reasoner_has_clear_state(self):
        """Test that SymbolicReasoner has clear_state method."""
        try:
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
            
            reasoner = SymbolicReasoner()
            assert hasattr(reasoner, 'clear_state'), (
                "SymbolicReasoner must have clear_state method for Bug #4 fix"
            )
            
            # Call clear_state - should not raise
            reasoner.clear_state()
            
        except ImportError:
            pytest.skip("SymbolicReasoner not available")
    
    def test_probabilistic_reasoner_has_clear_state(self):
        """Test that ProbabilisticReasoner has clear_state method."""
        try:
            from vulcan.reasoning.symbolic.reasoner import ProbabilisticReasoner
            
            reasoner = ProbabilisticReasoner()
            assert hasattr(reasoner, 'clear_state'), (
                "ProbabilisticReasoner must have clear_state method for Bug #4 fix"
            )
            
            # Call clear_state - should not raise
            reasoner.clear_state()
            
        except ImportError:
            pytest.skip("ProbabilisticReasoner not available")


class TestBug4SymbolicQueryGateCheck:
    """
    Tests for symbolic reasoning query applicability checks.
    
    The symbolic parser expects formal logic but often receives natural language,
    causing parse errors like "Unexpected token 'the' at line 1, column 5".
    
    The fix ensures:
    1. is_symbolic_query() checks if query contains formal logic notation
    2. Non-symbolic queries return applicable=False before parsing
    """
    
    def test_symbolic_reasoner_has_applicability_check(self):
        """Test that SymbolicReasoner has is_symbolic_query method."""
        try:
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
            
            reasoner = SymbolicReasoner()
            assert hasattr(reasoner, 'is_symbolic_query'), (
                "SymbolicReasoner must have is_symbolic_query method"
            )
            
        except ImportError:
            pytest.skip("SymbolicReasoner not available")
    
    def test_formal_logic_recognized_as_symbolic(self):
        """Test that formal logic notation is recognized as symbolic."""
        try:
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
            
            reasoner = SymbolicReasoner()
            
            symbolic_queries = [
                "A→B ∧ B→C",
                "∀X (human(X) → mortal(X))",
                "P(x) ∨ Q(x)",
                "¬C ∧ A",
                "forall X exists Y P(X,Y)",
            ]
            
            for query in symbolic_queries:
                assert reasoner.is_symbolic_query(query), (
                    f"Formal logic query '{query}' should be recognized as symbolic"
                )
                
        except ImportError:
            pytest.skip("SymbolicReasoner not available")
    
    def test_natural_language_rejected_as_symbolic(self):
        """Test that natural language is NOT recognized as symbolic."""
        try:
            from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
            
            reasoner = SymbolicReasoner()
            
            natural_language_queries = [
                "What is the capital of France?",
                "How does photosynthesis work?",
                "Every engineer reviewed a document",  # The quantifier scope question
                "Map the deep structure from domain S to domain T",  # Analogical reasoning
            ]
            
            for query in natural_language_queries:
                assert not reasoner.is_symbolic_query(query), (
                    f"Natural language query '{query}' should NOT be recognized as symbolic"
                )
                
        except ImportError:
            pytest.skip("SymbolicReasoner not available")


class TestCommonEnglishWordsRejection:
    """
    Tests for the common English words rejection list.
    
    The _COMMON_ENGLISH_WORDS frozenset should include words that
    should never be used as probability variable names.
    """
    
    def test_common_words_list_exists(self):
        """Test that the common English words list exists."""
        try:
            from vulcan.reasoning.selection.tool_selector import ProbabilisticToolWrapper
            
            assert hasattr(ProbabilisticToolWrapper, '_COMMON_ENGLISH_WORDS'), (
                "ProbabilisticToolWrapper must have _COMMON_ENGLISH_WORDS attribute"
            )
            
            words = ProbabilisticToolWrapper._COMMON_ENGLISH_WORDS
            assert isinstance(words, frozenset), (
                "_COMMON_ENGLISH_WORDS should be a frozenset for efficient lookups"
            )
            
        except ImportError:
            pytest.skip("ProbabilisticToolWrapper not available")
    
    def test_problematic_words_are_included(self):
        """Test that known problematic words are in the rejection list."""
        try:
            from vulcan.reasoning.selection.tool_selector import ProbabilisticToolWrapper
            
            words = ProbabilisticToolWrapper._COMMON_ENGLISH_WORDS
            
            # These words caused actual bugs in production
            # "if" from "if given the opportunity..."
            # "you" from "you must choose..."
            # etc.
            problematic_words = ['if', 'you', 'the', 'what', 'given', 'choose', 'become']
            
            for word in problematic_words:
                assert word in words, (
                    f"Problematic word '{word}' should be in _COMMON_ENGLISH_WORDS"
                )
                
        except ImportError:
            pytest.skip("ProbabilisticToolWrapper not available")


# =============================================================================
# Integration Tests
# =============================================================================

class TestDeepDiveBugFixesIntegration:
    """
    Integration tests for the deep dive bug fixes.
    
    These tests simulate the actual scenarios described in the bug report.
    """
    
    def test_self_awareness_question_flow(self):
        """
        Test the full flow for self-awareness questions.
        
        Bug Report Scenario:
        - User asks "if given the opportunity would you become self aware?"
        - System incorrectly triggers MATH-FAST-PATH
        - Probabilistic engine tries "Computing P(if | evidence={})"
        
        Expected after fix:
        - Query is classified correctly (not as MATH)
        - Probabilistic engine rejects via gate check
        - No "P(if)" computation occurs
        """
        try:
            from vulcan.reasoning.selection.tool_selector import ProbabilisticToolWrapper
            
            # Create wrapper with mock engine
            class MockEngine:
                def clear_state(self):
                    pass
                def _is_probability_query(self, q):
                    return 'probability' in q.lower() or 'p(' in q.lower()
                def add_rule(self, r, c):
                    pass
                def query(self, v, e):
                    return {True: 0.5, False: 0.5}
            
            wrapper = ProbabilisticToolWrapper(MockEngine())
            
            # The problematic query from the bug report
            query = "if given the opportunity would you become self aware? Yes or no?"
            
            # Execute
            result = wrapper.reason(query)
            
            # Verify fix
            # 1. Gate check should reject this query
            assert result.get("applicable") is False or result.get("confidence", 1.0) == 0.0, (
                "Self-awareness query should be rejected by gate check"
            )
            
            # 2. No "P(if)" computation should occur
            # If the engine was called, it would return 0.5 confidence
            # With gate check, we expect 0.0 confidence
            confidence = result.get("confidence", 1.0)
            assert confidence == 0.0, (
                f"Expected confidence=0.0 for rejected query, got {confidence}"
            )
            
        except ImportError:
            pytest.skip("Required modules not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
