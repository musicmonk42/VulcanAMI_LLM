"""
Tests for Natural Language to Logic Converter.

BUG #5 FIX: Tests to verify that natural language sentences are correctly
converted to first-order logic notation.

Test cases:
- Universal quantifiers ("Every X does Y")
- Existential quantifiers ("Some X does Y")
- Implications ("If X then Y")
- Negation ("No X does Y", "Not X")
- Simple predicates ("X is Y", "X verbs Y")
- Already formal logic (should pass through unchanged)
- Verb normalization utility functions
"""

import pytest

from src.vulcan.reasoning.symbolic.nl_converter import (
    NaturalLanguageToLogicConverter,
    convert_nl_to_logic,
    normalize_verb,
    PatternConfig,
    FORMAL_LOGIC_SYMBOLS,
)


class TestNaturalLanguageToLogicConverter:
    """Test cases for NL to Logic conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = NaturalLanguageToLogicConverter()
    
    # =========================================================================
    # Universal Quantifier Tests
    # =========================================================================
    
    def test_universal_quantifier_reviewed(self):
        """Test: Every engineer reviewed a document -> ∀e ∃d Reviewed(e, d)"""
        result = self.converter.convert("Every engineer reviewed a document")
        assert result is not None
        # Should contain universal and existential quantifiers
        assert "∀" in result
        assert "∃" in result
        # Should have variables
        assert "e" in result or "E" in result
        assert "d" in result or "D" in result
    
    def test_universal_quantifier_is(self):
        """Test: Every human is mortal -> ∀h Mortal(h)"""
        result = self.converter.convert("Every human is mortal")
        assert result is not None
        assert "∀" in result
        assert "h" in result or "H" in result
    
    def test_universal_quantifier_all(self):
        """Test: All students are learners -> ∀s Learners(s)"""
        result = self.converter.convert("All students are learners")
        assert result is not None
        assert "∀" in result
    
    def test_universal_with_article_object(self):
        """Test: Every cat has a tail -> ∀c ∃t Has(c, t)"""
        result = self.converter.convert("Every cat has a tail")
        assert result is not None
        assert "∀" in result
        assert "∃" in result
    
    # =========================================================================
    # Existential Quantifier Tests
    # =========================================================================
    
    def test_existential_quantifier_some(self):
        """Test: Some students passed the exam -> ∃s Passed(s)"""
        result = self.converter.convert("Some students passed the exam")
        assert result is not None
        assert "∃" in result
    
    def test_existential_there_exists(self):
        """Test: There exists a solution that works -> ∃s Works(s)"""
        result = self.converter.convert("There exists a solution that works")
        assert result is not None
        assert "∃" in result
    
    # =========================================================================
    # Implication Tests
    # =========================================================================
    
    def test_implication_if_then(self):
        """Test: If it rains then the ground is wet -> Rain → Wet"""
        result = self.converter.convert("If it rains then the ground is wet")
        assert result is not None
        assert "→" in result
    
    def test_implication_implies(self):
        """Test: Smoke implies fire -> Smoke → Fire"""
        result = self.converter.convert("Smoke implies fire")
        assert result is not None
        assert "→" in result
    
    # =========================================================================
    # Biconditional Tests
    # =========================================================================
    
    def test_biconditional(self):
        """Test: A if and only if B -> A ↔ B"""
        result = self.converter.convert("A if and only if B")
        assert result is not None
        assert "↔" in result
    
    # =========================================================================
    # Conjunction Tests
    # =========================================================================
    
    def test_conjunction_simple(self):
        """Test: A and B -> (A ∧ B)"""
        result = self.converter.convert("Rain and snow")
        assert result is not None
        assert "∧" in result
    
    def test_conjunction_predicates(self):
        """Test: Socrates is wise and mortal -> conjunction"""
        result = self.converter.convert("Socrates is wise and mortal")
        assert result is not None
        assert "∧" in result
    
    def test_both_and(self):
        """Test: Both dogs and cats are animals -> Animals(dogs) ∧ Animals(cats)"""
        result = self.converter.convert("Both dogs and cats are animals")
        assert result is not None
        assert "∧" in result
    
    # =========================================================================
    # Disjunction Tests
    # =========================================================================
    
    def test_disjunction_simple(self):
        """Test: A or B -> (A ∨ B)"""
        result = self.converter.convert("Rain or shine")
        assert result is not None
        assert "∨" in result
    
    def test_either_or(self):
        """Test: Either John or Mary is correct -> Correct(john) ∨ Correct(mary)"""
        result = self.converter.convert("Either John or Mary is correct")
        assert result is not None
        assert "∨" in result
    
    def test_neither_nor(self):
        """Test: Neither rain nor snow -> (¬Rain ∧ ¬Snow)"""
        result = self.converter.convert("Neither rain nor snow")
        assert result is not None
        assert "¬" in result
        assert "∧" in result
    
    # =========================================================================
    # Negation Tests
    # =========================================================================
    
    def test_negation_no(self):
        """Test: No dogs are cats -> ∀d ¬Cats(d)"""
        result = self.converter.convert("No dogs are cats")
        assert result is not None
        assert "∀" in result
        assert "¬" in result
    
    def test_negation_not(self):
        """Test: Not all birds can fly -> ¬All(birds, fly)"""
        result = self.converter.convert("Not all birds can fly")
        assert result is not None
        assert "¬" in result
    
    # =========================================================================
    # Simple Predicate Tests
    # =========================================================================
    
    def test_simple_predicate_is(self):
        """Test: Socrates is mortal -> Mortal(socrates)"""
        result = self.converter.convert("Socrates is mortal")
        assert result is not None
        assert "socrates" in result.lower()
    
    def test_binary_predicate(self):
        """Test: John loves Mary -> Loves(john, mary)"""
        result = self.converter.convert("John loves Mary")
        assert result is not None
        assert "john" in result.lower()
        assert "mary" in result.lower()
    
    # =========================================================================
    # Already Formal Logic Tests
    # =========================================================================
    
    def test_already_formal_logic_symbols(self):
        """Test: Already formal logic with symbols should pass through."""
        formal = "∀x (P(x) → Q(x))"
        result = self.converter.convert(formal)
        # Should return the input unchanged
        assert result == formal
    
    def test_already_formal_logic_ascii(self):
        """Test: Already formal logic with ASCII should pass through."""
        formal = "P(x) -> Q(x)"
        result = self.converter.convert(formal)
        # Should return the input unchanged
        assert result == formal
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_empty_string(self):
        """Test: Empty string returns None."""
        result = self.converter.convert("")
        assert result is None
    
    def test_whitespace_only(self):
        """Test: Whitespace-only string returns None."""
        result = self.converter.convert("   \n\t   ")
        assert result is None
    
    def test_unrecognized_sentence(self):
        """Test: Unrecognized sentences may return None or simple extraction."""
        result = self.converter.convert("This is a random sentence without clear structure")
        # May return None or attempt extraction - should not raise
        assert result is None or isinstance(result, str)


class TestConvenienceFunction:
    """Test the convenience function convert_nl_to_logic."""
    
    def test_basic_conversion(self):
        """Test basic conversion with convenience function."""
        result = convert_nl_to_logic("Every student passed the exam")
        assert result is not None
        assert "∀" in result or "∃" in result or "→" in result or "(" in result
    
    def test_empty_input(self):
        """Test empty input with convenience function."""
        result = convert_nl_to_logic("")
        assert result is None


class TestIntegrationWithSymbolicReasoner:
    """
    Test that the NL converter integrates correctly with SymbolicReasoner.
    
    BUG #5 FIX: These tests verify that natural language queries can be
    processed by the symbolic reasoner without parse errors.
    """
    
    def test_reasoner_with_nl_query(self):
        """Test that SymbolicReasoner can handle NL queries via NL converter."""
        # Import here to avoid circular imports during test collection
        from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
        
        reasoner = SymbolicReasoner()
        
        # Add a fact using natural language (should be converted)
        # Note: This tests the parse_formula path which now includes NL conversion
        try:
            # Try to parse natural language - should not raise
            clause = reasoner.parse_formula("Socrates is a human")
            assert clause is not None
        except Exception as e:
            # Should not fail with parse error on natural language
            pytest.fail(f"Natural language parsing failed: {e}")
    
    def test_reasoner_has_nl_converter(self):
        """Test that SymbolicReasoner has NL converter initialized."""
        from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
        
        reasoner = SymbolicReasoner()
        assert hasattr(reasoner, 'nl_converter')
        assert reasoner.nl_converter is not None
        assert isinstance(reasoner.nl_converter, NaturalLanguageToLogicConverter)
    
    def test_reasoner_clear_state_resets_nl_converter(self):
        """Test that clear_state also resets the NL converter."""
        from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
        
        reasoner = SymbolicReasoner()
        old_converter = reasoner.nl_converter
        
        reasoner.clear_state()
        
        # Should have a new converter instance
        assert reasoner.nl_converter is not old_converter


class TestVerbNormalization:
    """Tests for the verb normalization utility function."""
    
    def test_normalize_past_tense_ed(self):
        """Test: passed -> pass"""
        assert normalize_verb('passed') == 'pass'
    
    def test_normalize_past_tense_ied(self):
        """Test: studied -> study"""
        assert normalize_verb('studied') == 'study'
    
    def test_normalize_third_person_s(self):
        """Test: loves -> love"""
        assert normalize_verb('loves') == 'love'
    
    def test_normalize_third_person_es(self):
        """Test: watches -> watch"""
        assert normalize_verb('watches') == 'watch'
    
    def test_normalize_double_s(self):
        """Test: pass -> pass (not normalized, double s)"""
        result = normalize_verb('pass')
        assert result == 'pass'
    
    def test_normalize_empty(self):
        """Test: empty string returns empty"""
        assert normalize_verb('') == ''
    
    def test_normalize_already_base(self):
        """Test: base verbs stay unchanged"""
        assert normalize_verb('love') == 'love'
        assert normalize_verb('run') == 'run'


class TestPatternConfig:
    """Tests for the PatternConfig dataclass."""
    
    def test_pattern_config_immutable(self):
        """Test that PatternConfig is frozen (immutable)."""
        import re
        config = PatternConfig(
            pattern=re.compile(r'test'),
            handler_name='_handle_test',
            description='Test pattern'
        )
        
        # Should raise FrozenInstanceError when trying to modify
        with pytest.raises(Exception):  # dataclass.FrozenInstanceError
            config.handler_name = 'new_value'
    
    def test_pattern_config_attributes(self):
        """Test PatternConfig has correct attributes."""
        import re
        pattern = re.compile(r'test')
        config = PatternConfig(
            pattern=pattern,
            handler_name='_handle_test',
            description='Test pattern'
        )
        
        assert config.pattern == pattern
        assert config.handler_name == '_handle_test'
        assert config.description == 'Test pattern'
    
    def test_pattern_config_priority(self):
        """Test PatternConfig has priority attribute."""
        import re
        config = PatternConfig(
            pattern=re.compile(r'test'),
            handler_name='_handle_test',
            description='Test pattern',
            priority=75
        )
        
        assert config.priority == 75
    
    def test_pattern_config_default_priority(self):
        """Test PatternConfig has default priority of 50."""
        import re
        config = PatternConfig(
            pattern=re.compile(r'test'),
            handler_name='_handle_test',
            description='Test pattern'
        )
        
        assert config.priority == 50


class TestPatternPriority:
    """Tests for pattern priority sorting."""
    
    def test_patterns_sorted_by_priority(self):
        """Test that patterns are sorted by priority (highest first)."""
        converter = NaturalLanguageToLogicConverter()
        priorities = [p.priority for p in converter.patterns]
        
        # Should be sorted in descending order
        assert priorities == sorted(priorities, reverse=True)
    
    def test_biconditional_has_highest_priority(self):
        """Test that biconditional pattern has highest priority."""
        converter = NaturalLanguageToLogicConverter()
        
        # Find biconditional pattern
        biconditional_priority = None
        for p in converter.patterns:
            if 'biconditional' in p.description.lower():
                biconditional_priority = p.priority
                break
        
        assert biconditional_priority is not None
        assert biconditional_priority >= 100
    
    def test_binary_predicate_has_lowest_priority(self):
        """Test that binary predicate (fallback) has lowest priority."""
        converter = NaturalLanguageToLogicConverter()
        
        # Last pattern should be binary predicate (lowest priority)
        last_pattern = converter.patterns[-1]
        assert 'binary predicate' in last_pattern.description.lower()
        assert last_pattern.priority <= 25


class TestConverterThreadSafety:
    """Tests related to thread safety of the converter."""
    
    def test_patterns_are_immutable_tuple(self):
        """Test that patterns are stored as immutable tuple."""
        converter = NaturalLanguageToLogicConverter()
        assert isinstance(converter.patterns, tuple)
    
    def test_patterns_contain_pattern_configs(self):
        """Test that patterns tuple contains PatternConfig objects."""
        converter = NaturalLanguageToLogicConverter()
        for pattern in converter.patterns:
            assert isinstance(pattern, PatternConfig)
    
    def test_multiple_converters_independent(self):
        """Test that multiple converter instances are independent."""
        c1 = NaturalLanguageToLogicConverter()
        c2 = NaturalLanguageToLogicConverter()
        
        # They should have separate pattern tuples (though identical content)
        assert c1.patterns is not c2.patterns
        # But same number of patterns
        assert len(c1.patterns) == len(c2.patterns)


class TestFormalLogicDetection:
    """Tests for formal logic symbol detection."""
    
    def test_all_formal_symbols_detected(self):
        """Test that all formal logic symbols are recognized."""
        converter = NaturalLanguageToLogicConverter()
        
        for symbol in FORMAL_LOGIC_SYMBOLS:
            text = f"P(x) {symbol} Q(x)"
            result = converter.convert(text)
            # Should return unchanged (pass through)
            assert result == text, f"Symbol {symbol} not detected"
    
    def test_mixed_formal_and_nl_returns_formal(self):
        """Test: Text with formal symbols passes through unchanged."""
        converter = NaturalLanguageToLogicConverter()
        formal = "∀x (Human(x) → Mortal(x))"
        result = converter.convert(formal)
        assert result == formal


class TestEdgeCases:
    """Additional edge case tests for robustness."""
    
    def test_very_long_input(self):
        """Test handling of very long input strings."""
        converter = NaturalLanguageToLogicConverter()
        long_text = "Every " + "very " * 100 + "long sentence"
        # Should not raise, may return None or converted form
        result = converter.convert(long_text)
        assert result is None or isinstance(result, str)
    
    def test_special_characters(self):
        """Test handling of special characters in input."""
        converter = NaturalLanguageToLogicConverter()
        result = converter.convert("Every user@domain.com is valid")
        # Should not raise
        assert result is None or isinstance(result, str)
    
    def test_unicode_input(self):
        """Test handling of unicode input."""
        converter = NaturalLanguageToLogicConverter()
        result = converter.convert("Every café is cozy")
        assert result is None or isinstance(result, str)
    
    def test_numeric_entities(self):
        """Test handling of numeric entities."""
        converter = NaturalLanguageToLogicConverter()
        result = converter.convert("123 is greater than 100")
        assert result is None or isinstance(result, str)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
