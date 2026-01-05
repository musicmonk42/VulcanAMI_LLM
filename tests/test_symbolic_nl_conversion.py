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
"""

import pytest

from src.vulcan.reasoning.symbolic.nl_converter import (
    NaturalLanguageToLogicConverter,
    convert_nl_to_logic,
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


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
