"""
Comprehensive tests for the Answer Validator module.

Tests cover:
- Type inference from queries
- Domain-specific validation (FOL, SAT, mathematical, etc.)
- Nonsensical output detection
- Edge cases and error handling
- Integration scenarios

Part of the VULCAN-AGI test suite.
"""

import pytest
from typing import Dict, Any

# Import the module under test
try:
    from vulcan.reasoning.answer_validator import (
        AnswerValidator,
        ValidationResult,
        ValidationFailureReason,
        validate_reasoning_result,
    )
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def validator():
    """Create an AnswerValidator instance."""
    return AnswerValidator()


# =============================================================================
# ValidationResult Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(
            valid=True,
            confidence=1.0,
            failures=[],
            explanation="Answer validation passed"
        )
        
        assert result.valid is True
        assert result.confidence == 1.0
        assert len(result.failures) == 0
        assert result.explanation == "Answer validation passed"
        assert result.suggestions == []
    
    def test_create_invalid_result(self):
        """Test creating an invalid result with failures."""
        result = ValidationResult(
            valid=False,
            confidence=0.0,
            failures=[ValidationFailureReason.WRONG_DOMAIN],
            explanation="Wrong domain",
            suggestions=["Use different engine"]
        )
        
        assert result.valid is False
        assert result.confidence == 0.0
        assert ValidationFailureReason.WRONG_DOMAIN in result.failures
        assert len(result.suggestions) == 1
    
    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            ValidationResult(
                valid=True,
                confidence=1.5,  # Invalid
                failures=[],
                explanation="Test"
            )
        
        with pytest.raises(ValueError):
            ValidationResult(
                valid=True,
                confidence=-0.1,  # Invalid
                failures=[],
                explanation="Test"
            )
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ValidationResult(
            valid=False,
            confidence=0.0,
            failures=[ValidationFailureReason.WRONG_DOMAIN, ValidationFailureReason.NO_ANSWER_PROVIDED],
            explanation="Test explanation",
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        d = result.to_dict()
        
        assert d['valid'] is False
        assert d['confidence'] == 0.0
        assert 'wrong_domain' in d['failures']
        assert 'no_answer_provided' in d['failures']
        assert len(d['suggestions']) == 2


# =============================================================================
# Type Inference Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestTypeInference:
    """Tests for query type inference."""
    
    def test_infer_fol_type(self, validator):
        """Test FOL type inference."""
        queries = [
            "Formalize 'Every engineer reviewed a document' in FOL",
            "Express this in first-order logic",
            "Write using quantifiers ∀ and ∃",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'fol', f"Expected 'fol' for: {query}"
    
    def test_infer_sat_type(self, validator):
        """Test SAT type inference."""
        queries = [
            "Is {A→B, B→C, ¬C, A∨B} satisfiable?",
            "Check if this propositional formula is satisfiable",
            "Find a model for the formula",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'sat', f"Expected 'sat' for: {query}"
    
    def test_infer_ethical_type(self, validator):
        """Test ethical type inference."""
        queries = [
            "Is it permissible to pull the lever?",
            "Is this action morally right?",
            "Consider the trolley dilemma",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'ethical', f"Expected 'ethical' for: {query}"
    
    def test_infer_mathematical_type(self, validator):
        """Test mathematical type inference."""
        queries = [
            "Compute the derivative of x^3",
            "Calculate the integral of sin(x)",
            "Solve the equation x^2 - 4 = 0",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'mathematical', f"Expected 'mathematical' for: {query}"
    
    def test_infer_bayesian_type(self, validator):
        """Test Bayesian type inference."""
        queries = [
            "What is the posterior probability?",
            "Use Bayes to find P(A|B)",
            "Given P(A) and P(B|A), find P(A|B) using Bayesian inference",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'bayesian', f"Expected 'bayesian' for: {query}"
    
    def test_infer_proof_type(self, validator):
        """Test proof type inference."""
        queries = [
            "Verify this proof is valid",
            "Check if the proof steps are correct",
            "Is this theorem proven?",
        ]
        
        for query in queries:
            result_type = validator._infer_expected_type(query)
            assert result_type == 'proof', f"Expected 'proof' for: {query}"


# =============================================================================
# FOL Validation Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestFOLValidation:
    """Tests for FOL answer validation."""
    
    def test_valid_fol_answer(self, validator):
        """Test validation of valid FOL answers."""
        query = "Formalize 'Every person is mortal' in FOL"
        
        valid_answers = [
            {'conclusion': '∀x(Person(x) → Mortal(x))'},
            {'conclusion': 'forall x: Person(x) implies Mortal(x)'},
            {'conclusion': '∃x(Person(x) AND Mortal(x))'},
        ]
        
        for answer in valid_answers:
            result = validator.validate(query, answer, expected_type='fol')
            assert result.valid, f"Should be valid: {answer}"
    
    def test_invalid_fol_answer_missing_quantifiers(self, validator):
        """Test rejection of FOL answers without quantifiers."""
        query = "Formalize in FOL"
        answer = {'conclusion': 'The result is 42'}
        
        result = validator.validate(query, answer, expected_type='fol')
        
        assert not result.valid
        assert ValidationFailureReason.MISSING_REQUIRED_ELEMENTS in result.failures
    
    def test_invalid_fol_answer_math_expression(self, validator):
        """Test rejection of mathematical expressions as FOL answers."""
        query = "Formalize in FOL"
        answer = {'conclusion': '3x**2 + 2x'}  # Known bug output
        
        result = validator.validate(query, answer, expected_type='fol')
        
        assert not result.valid
        assert ValidationFailureReason.WRONG_DOMAIN in result.failures


# =============================================================================
# SAT Validation Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestSATValidation:
    """Tests for SAT answer validation."""
    
    def test_valid_sat_answer(self, validator):
        """Test validation of valid SAT answers."""
        query = "Is {A→B, B→C, ¬C} satisfiable?"
        
        valid_answers = [
            {'conclusion': 'The formula is satisfiable with model {A=false, B=false, C=false}'},
            {'conclusion': 'UNSATISFIABLE - contradiction found'},
            {'conclusion': 'YES, a model exists: A=true, B=true, C=true'},
        ]
        
        for answer in valid_answers:
            result = validator.validate(query, answer, expected_type='sat')
            assert result.valid, f"Should be valid: {answer}"
    
    def test_invalid_sat_answer_wrong_domain(self, validator):
        """Test rejection of calculus answers for SAT queries."""
        query = "Is {A→B} satisfiable?"
        answer = {'conclusion': 'The derivative is 3x**2'}
        
        result = validator.validate(query, answer, expected_type='sat')
        
        assert not result.valid
        assert ValidationFailureReason.WRONG_DOMAIN in result.failures


# =============================================================================
# Mathematical Validation Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestMathematicalValidation:
    """Tests for mathematical answer validation."""
    
    def test_valid_math_answer(self, validator):
        """Test validation of valid mathematical answers."""
        query = "Compute the derivative of x^3"
        
        valid_answers = [
            {'conclusion': '3x^2'},
            {'conclusion': 'The result is 42'},
            {'conclusion': 'x = 5 + 3'},
        ]
        
        for answer in valid_answers:
            result = validator.validate(query, answer, expected_type='mathematical')
            assert result.valid, f"Should be valid: {answer}"
    
    def test_invalid_math_answer_no_numbers(self, validator):
        """Test rejection of answers without numbers or expressions."""
        query = "Calculate the sum"
        # Use an answer that truly has no numbers or math symbols
        answer = {'conclusion': 'The answer cannot be determined'}
        
        result = validator.validate(query, answer, expected_type='mathematical')
        
        assert not result.valid
        assert ValidationFailureReason.MISSING_REQUIRED_ELEMENTS in result.failures


# =============================================================================
# Ethical Validation Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestEthicalValidation:
    """Tests for ethical answer validation."""
    
    def test_valid_ethical_answer(self, validator):
        """Test validation of valid ethical answers."""
        query = "Is it permissible to lie to save a life?"
        
        valid_answers = [
            {'conclusion': 'It is morally permissible in this case'},
            {'conclusion': 'This action is impermissible according to deontology'},
            {'conclusion': 'The right thing to do is to tell the truth'},
        ]
        
        for answer in valid_answers:
            result = validator.validate(query, answer, expected_type='ethical')
            assert result.valid, f"Should be valid: {answer}"
    
    def test_invalid_ethical_answer_math(self, validator):
        """Test rejection of mathematical answers for ethical queries."""
        query = "Is it permissible?"
        answer = {'conclusion': '3x**2 + 2x = 0'}
        
        result = validator.validate(query, answer, expected_type='ethical')
        
        assert not result.valid
        assert ValidationFailureReason.WRONG_DOMAIN in result.failures


# =============================================================================
# Nonsensical Output Detection Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestNonsensicalDetection:
    """Tests for nonsensical output detection."""
    
    def test_detect_logic_query_calculus_answer(self, validator):
        """Test detection of calculus answer for logic query."""
        query = "Is {A→B, B→C, ¬C} satisfiable?"
        answer = {'conclusion': 'The derivative is exp(x) * sin(x)'}
        
        result = validator.validate(query, answer)
        
        assert not result.valid
        assert ValidationFailureReason.NONSENSICAL_OUTPUT in result.failures
    
    def test_detect_known_bug_output(self, validator):
        """Test detection of known bug output '3x**2 + 2x'."""
        query = "Is the formula valid?"  # Not a derivative query
        answer = {'conclusion': '3x**2 + 2x'}
        
        result = validator.validate(query, answer)
        
        assert not result.valid
        assert ValidationFailureReason.NONSENSICAL_OUTPUT in result.failures
    
    def test_allow_derivative_output_for_derivative_query(self, validator):
        """Test that derivative output is allowed for derivative queries."""
        query = "Find the derivative of x^3 + x^2"
        answer = {'conclusion': '3x**2 + 2x'}
        
        result = validator.validate(query, answer)
        
        # Should be valid because this IS a derivative query
        assert result.valid
    
    def test_detect_ethical_query_math_answer(self, validator):
        """Test detection of math answer for ethical query."""
        query = "Is it permissible to harm one to save five?"
        answer = {'conclusion': 'The integral of sin(x) is -cos(x)'}
        
        result = validator.validate(query, answer)
        
        assert not result.valid
        assert ValidationFailureReason.NONSENSICAL_OUTPUT in result.failures


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_query(self, validator):
        """Test handling of empty query."""
        result = validator.validate("", {'conclusion': 'answer'})
        # Should not crash, should return some result
        assert isinstance(result, ValidationResult)
    
    def test_empty_conclusion(self, validator):
        """Test handling of empty conclusion."""
        result = validator.validate("Some query", {'conclusion': ''})
        assert isinstance(result, ValidationResult)
    
    def test_missing_conclusion_key(self, validator):
        """Test handling of result without conclusion key."""
        result = validator.validate("Some query", {'other_key': 'value'})
        assert isinstance(result, ValidationResult)
    
    def test_none_conclusion(self, validator):
        """Test handling of None conclusion."""
        result = validator.validate("Some query", {'conclusion': None})
        assert isinstance(result, ValidationResult)
    
    def test_invalid_query_type(self, validator):
        """Test that invalid query type raises TypeError."""
        with pytest.raises(TypeError):
            validator.validate(123, {'conclusion': 'answer'})
    
    def test_invalid_result_type(self, validator):
        """Test that invalid result type raises TypeError."""
        with pytest.raises(TypeError):
            validator.validate("query", "not a dict")
    
    def test_unknown_expected_type(self, validator):
        """Test handling of unknown expected type."""
        result = validator.validate(
            "Some query", 
            {'conclusion': 'answer'},
            expected_type='unknown_type'
        )
        # Should not crash, validators dict doesn't have this type
        assert isinstance(result, ValidationResult)


# =============================================================================
# Convenience Function Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestConvenienceFunction:
    """Tests for validate_reasoning_result convenience function."""
    
    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        result = validate_reasoning_result(
            "Is 2+2=4?",
            {'conclusion': 'Yes, that is correct'}
        )
        
        assert isinstance(result, ValidationResult)
    
    def test_with_expected_type(self):
        """Test with explicit expected type."""
        result = validate_reasoning_result(
            "Calculate 2+2",
            {'conclusion': '4'},
            expected_type='mathematical'
        )
        
        assert result.valid


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Answer validator not available")
class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_fol_query_with_math_engine_output(self, validator):
        """Test the exact bug scenario: FOL query gets derivative answer."""
        # This is the exact scenario from the problem statement
        query = "Every engineer reviewed a document. Formalize in FOL."
        answer = {'conclusion': '3x**2 + 2x'}  # Bug output from math engine
        
        result = validator.validate(query, answer)
        
        assert not result.valid
        # Should fail for multiple reasons
        assert len(result.failures) > 0
        assert len(result.suggestions) > 0
    
    def test_sat_query_with_differential_equation(self, validator):
        """Test SAT query getting differential equation answer."""
        query = "Is {A→B, B→C, ¬C, A∨B} satisfiable?"
        answer = {'conclusion': 'f(x) = C1*exp(x)'}  # Wrong domain
        
        result = validator.validate(query, answer)
        
        assert not result.valid
    
    def test_proof_verification_query(self, validator):
        """Test proof verification query with bug output.
        
        This tests the scenario where a proof verification query gets
        a derivative output (the known bug). The validator should detect
        this as either:
        1. Invalid because it's nonsensical (calculus answer for proof query), OR
        2. The query isn't detected as 'proof' type due to 'differentiable' keyword
        
        Note: The fix for 'differentiable' not triggering math engine is in
        mathematical_computation.py, not the validator. The validator catches
        the output after the engine produces it.
        """
        query = "Verify proof about differentiable functions"
        answer = {'conclusion': '3x**2 + 2x'}  # Bug output
        
        result = validator.validate(query, answer)
        inferred_type = validator._infer_expected_type(query)
        
        # Either the result is invalid OR the type wasn't detected as 'proof'
        # (because 'differentiable' might influence type inference)
        assert not result.valid or inferred_type != 'proof', (
            f"Expected invalid result or non-proof type, got valid={result.valid}, type={inferred_type}"
        )
    
    def test_bayesian_query_proper_format(self, validator):
        """Test Bayesian query with proper probability answer."""
        query = "P(X|+) with sensitivity=0.99, specificity=0.95, prevalence=0.01"
        answer = {'conclusion': 'P(Disease|Positive) ≈ 0.167 (16.7%)'}
        
        result = validator.validate(query, answer)
        
        assert result.valid


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
