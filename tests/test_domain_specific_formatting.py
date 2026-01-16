"""
Test suite for domain-specific reasoning engine output formatting.

Industry Standards Applied:
- Comprehensive coverage: Tests all formatting functions
- Edge cases: Handles None, empty, and malformed data
- Security: Tests UTF-8 handling and truncation
- Maintainability: Clear test names and documentation
- Isolation: Each test is independent and focused

Tests the fix for the critical issue where domain-specific structured outputs
from reasoning engines were being discarded by the generic formatter.
"""

import pytest
from vulcan.endpoints.chat_helpers import (
    _format_fol_formalization,
    _format_causal_reasoning,
    _format_probabilistic_reasoning,
    _format_analogical_reasoning,
    _format_mathematical_reasoning,
    _format_engine_result_dict,
    format_reasoning_results,
)


class TestFOLFormalizationFormatting:
    """Test FOL formalization formatting (symbolic reasoning)."""
    
    def test_fol_formalization_with_both_readings(self):
        """Test formatting FOL with both narrow and wide scope readings."""
        result = {
            "fol_formalization": {
                "original_sentence": "Every engineer reviewed a document.",
                "reading_a": {
                    "fol": "∃d.(∀e.Reviewed(e,d))",
                    "interpretation": "Narrow scope existential",
                    "english_rewrite": "There is a specific document that every engineer reviewed."
                },
                "reading_b": {
                    "fol": "∀e.(∃d.Reviewed(e,d))",
                    "interpretation": "Wide scope universal",
                    "english_rewrite": "Every engineer reviewed some document (possibly different ones)."
                },
                "ambiguity_type": "quantifier_scope"
            }
        }
        
        formatted = _format_fol_formalization(result)
        
        # Verify all key components are present
        assert "Every engineer reviewed a document" in formatted
        assert "quantifier_scope" in formatted
        assert "Reading A" in formatted
        assert "Reading B" in formatted
        assert "∃d.(∀e.Reviewed(e,d))" in formatted
        assert "∀e.(∃d.Reviewed(e,d))" in formatted
        assert "Narrow scope" in formatted
        assert "Wide scope" in formatted
    
    def test_fol_formalization_missing_readings(self):
        """Test handling of incomplete FOL formalization data."""
        result = {
            "fol_formalization": {
                "original_sentence": "Test sentence."
            }
        }
        
        formatted = _format_fol_formalization(result)
        
        # Should still format the original sentence
        assert "Test sentence" in formatted
    
    def test_fol_formalization_empty(self):
        """Test handling of empty FOL formalization."""
        result = {}
        formatted = _format_fol_formalization(result)
        assert formatted == ""
    
    def test_fol_formalization_none(self):
        """Test handling of None FOL formalization."""
        result = {"fol_formalization": None}
        formatted = _format_fol_formalization(result)
        assert formatted == ""
    
    def test_fol_formalization_invalid_type(self):
        """Test handling of invalid type for FOL formalization."""
        result = {"fol_formalization": "not a dict"}
        formatted = _format_fol_formalization(result)
        assert formatted == ""


class TestCausalReasoningFormatting:
    """Test causal reasoning formatting."""
    
    def test_causal_graph_formatting(self):
        """Test formatting of causal graph structure."""
        result = {
            "causal_graph": {
                "smoking": {
                    "lung_cancer": {"strength": 0.85, "confidence": 0.92}
                },
                "exercise": {
                    "health": {"strength": 0.70, "confidence": 0.88}
                }
            }
        }
        
        formatted = _format_causal_reasoning(result)
        
        assert "Causal Graph" in formatted
        assert "smoking" in formatted
        assert "lung_cancer" in formatted
        assert "0.85" in formatted
        assert "92%" in formatted
    
    def test_confounders_formatting(self):
        """Test formatting of confounder list."""
        result = {
            "confounders": ["age", "genetics", "environment"]
        }
        
        formatted = _format_causal_reasoning(result)
        
        assert "Confounders" in formatted
        assert "age" in formatted
        assert "genetics" in formatted
    
    def test_intervention_formatting(self):
        """Test formatting of intervention recommendations."""
        result = {
            "intervention": "Increase exercise frequency to reduce health risk"
        }
        
        formatted = _format_causal_reasoning(result)
        
        assert "Intervention" in formatted
        assert "exercise" in formatted
    
    def test_causal_empty(self):
        """Test handling of empty causal reasoning data."""
        result = {}
        formatted = _format_causal_reasoning(result)
        assert formatted == ""


class TestProbabilisticReasoningFormatting:
    """Test probabilistic reasoning formatting."""
    
    def test_posterior_distribution_dict(self):
        """Test formatting of posterior distribution as dict."""
        result = {
            "posterior": {
                "mu": 0.5234,
                "sigma": 0.1567
            }
        }
        
        formatted = _format_probabilistic_reasoning(result)
        
        assert "Posterior Distribution" in formatted
        assert "mu" in formatted
        assert "0.5234" in formatted
    
    def test_posterior_single_value(self):
        """Test formatting of posterior as single probability."""
        result = {
            "posterior": 0.7845
        }
        
        formatted = _format_probabilistic_reasoning(result)
        
        assert "Posterior" in formatted
        assert "0.7845" in formatted
    
    def test_parameters_formatting(self):
        """Test formatting of model parameters."""
        result = {
            "parameters": {
                "alpha": 2.5,
                "beta": 1.3
            }
        }
        
        formatted = _format_probabilistic_reasoning(result)
        
        assert "Parameters" in formatted
        assert "alpha" in formatted
        assert "2.5" in formatted
    
    def test_prior_and_posterior(self):
        """Test formatting of both prior and posterior."""
        result = {
            "prior": 0.3,
            "posterior": 0.7
        }
        
        formatted = _format_probabilistic_reasoning(result)
        
        assert "Prior" in formatted
        assert "Posterior" in formatted
        assert "0.3" in formatted
        assert "0.7" in formatted
    
    def test_probabilistic_empty(self):
        """Test handling of empty probabilistic data."""
        result = {}
        formatted = _format_probabilistic_reasoning(result)
        assert formatted == ""


class TestAnalogicalReasoningFormatting:
    """Test analogical reasoning formatting."""
    
    def test_entity_mappings(self):
        """Test formatting of entity mappings."""
        result = {
            "entity_mappings": {
                "atom": "solar_system",
                "electron": "planet",
                "nucleus": "sun"
            }
        }
        
        formatted = _format_analogical_reasoning(result)
        
        assert "Entity Mappings" in formatted
        assert "atom" in formatted
        assert "solar_system" in formatted
        assert "→" in formatted
    
    def test_inferences_list(self):
        """Test formatting of inferences as list."""
        result = {
            "inferences": [
                "Electrons orbit nucleus like planets orbit sun",
                "Nuclear forces similar to gravitational forces"
            ]
        }
        
        formatted = _format_analogical_reasoning(result)
        
        assert "Inferences" in formatted
        assert "Electrons orbit" in formatted
        assert "Nuclear forces" in formatted
    
    def test_source_and_target_domains(self):
        """Test formatting of domain information."""
        result = {
            "source_domain": "atomic physics",
            "target_domain": "solar system"
        }
        
        formatted = _format_analogical_reasoning(result)
        
        assert "Source Domain" in formatted
        assert "Target Domain" in formatted
        assert "atomic physics" in formatted
    
    def test_analogical_empty(self):
        """Test handling of empty analogical data."""
        result = {}
        formatted = _format_analogical_reasoning(result)
        assert formatted == ""


class TestMathematicalReasoningFormatting:
    """Test mathematical reasoning formatting."""
    
    def test_closed_form_solution(self):
        """Test formatting of closed-form solutions."""
        result = {
            "closed_form": "x = (-b ± √(b²-4ac)) / 2a"
        }
        
        formatted = _format_mathematical_reasoning(result)
        
        assert "Closed-Form Solution" in formatted
        assert "(-b ± √(b²-4ac))" in formatted
    
    def test_proof_steps(self):
        """Test formatting of proof steps."""
        result = {
            "proof_steps": [
                "Assume P(n) holds for n=k",
                "Show P(k+1) follows from P(k)",
                "By induction, P(n) holds for all n"
            ]
        }
        
        formatted = _format_mathematical_reasoning(result)
        
        assert "Proof Steps" in formatted
        assert "Assume P(n)" in formatted
        assert "induction" in formatted
    
    def test_verification_status(self):
        """Test formatting of verification results."""
        result_verified = {"verification": True}
        result_not_verified = {"verification": False}
        
        formatted_verified = _format_mathematical_reasoning(result_verified)
        formatted_not_verified = _format_mathematical_reasoning(result_not_verified)
        
        assert "Verified ✓" in formatted_verified
        assert "Not Verified ✗" in formatted_not_verified
    
    def test_mathematical_empty(self):
        """Test handling of empty mathematical data."""
        result = {}
        formatted = _format_mathematical_reasoning(result)
        assert formatted == ""


class TestEngineResultDictIntegration:
    """Test the main _format_engine_result_dict function with domain-specific data."""
    
    def test_symbolic_engine_with_fol(self):
        """Test formatting symbolic engine output with FOL formalization."""
        result = {
            "confidence": 0.90,
            "fol_formalization": {
                "original_sentence": "Every engineer reviewed a document.",
                "reading_a": {
                    "fol": "∃d.(∀e.Reviewed(e,d))",
                    "interpretation": "Narrow scope"
                }
            }
        }
        
        formatted = _format_engine_result_dict("symbolic", result)
        
        # Should include both FOL formalization and generic confidence
        assert "Symbolic:" in formatted
        assert "Every engineer reviewed a document" in formatted
        assert "∃d.(∀e.Reviewed(e,d))" in formatted
        assert "90%" in formatted
    
    def test_causal_engine_with_graph(self):
        """Test formatting causal engine output."""
        result = {
            "confidence": 0.85,
            "causal_graph": {
                "X": {"Y": {"strength": 0.75}}
            }
        }
        
        formatted = _format_engine_result_dict("causal", result)
        
        assert "Causal:" in formatted
        assert "Causal Graph" in formatted
        assert "X" in formatted
        assert "85%" in formatted
    
    def test_probabilistic_engine_with_posterior(self):
        """Test formatting probabilistic engine output."""
        result = {
            "confidence": 0.88,
            "posterior": 0.65,
            "prior": 0.35
        }
        
        formatted = _format_engine_result_dict("probabilistic", result)
        
        assert "Probabilistic:" in formatted
        assert "Posterior" in formatted
        assert "Prior" in formatted
        assert "88%" in formatted
    
    def test_generic_fallback(self):
        """Test that generic fields still work for engines without specific formatters."""
        result = {
            "conclusion": "The answer is 42",
            "confidence": 0.95,
            "explanation": "Based on analysis"
        }
        
        formatted = _format_engine_result_dict("generic_engine", result)
        
        assert "Generic Engine:" in formatted
        assert "The answer is 42" in formatted
        assert "95%" in formatted
        assert "Based on analysis" in formatted
    
    def test_empty_result(self):
        """Test handling of empty result dictionary."""
        result = {}
        formatted = _format_engine_result_dict("empty", result)
        assert formatted == ""


class TestFormatReasoningResultsIntegration:
    """Test the main format_reasoning_results function with multiple engines."""
    
    def test_multiple_engines_with_domain_specific_data(self):
        """Test formatting output from multiple reasoning engines."""
        reasoning_results = {
            "symbolic": {
                "confidence": 0.90,
                "fol_formalization": {
                    "original_sentence": "Every engineer reviewed a document.",
                    "reading_a": {"fol": "∃d.(∀e.Reviewed(e,d))"}
                }
            },
            "probabilistic": {
                "confidence": 0.85,
                "posterior": 0.72
            }
        }
        
        formatted = format_reasoning_results(reasoning_results)
        
        assert "Reasoning Analysis:" in formatted
        assert "Symbolic:" in formatted
        assert "Probabilistic:" in formatted
        assert "∃d.(∀e.Reviewed(e,d))" in formatted
        assert "0.72" in formatted
    
    def test_empty_reasoning_results(self):
        """Test handling of empty reasoning results."""
        formatted = format_reasoning_results({})
        assert formatted == ""
    
    def test_none_reasoning_results(self):
        """Test handling of None reasoning results."""
        formatted = format_reasoning_results(None)
        assert formatted == ""


class TestUTF8SafetyAndTruncation:
    """Test UTF-8 safety and truncation in formatting."""
    
    def test_unicode_characters_in_fol(self):
        """Test that Unicode characters (FOL symbols) are handled correctly."""
        result = {
            "fol_formalization": {
                "original_sentence": "Test with symbols: ∀∃∧∨¬→↔",
                "reading_a": {
                    "fol": "∀x.(P(x) → ∃y.Q(x,y))"
                }
            }
        }
        
        formatted = _format_fol_formalization(result)
        
        # Unicode symbols should be preserved
        assert "∀∃∧∨¬→↔" in formatted
        assert "∀x.(P(x) → ∃y.Q(x,y))" in formatted
    
    def test_very_long_strings_are_truncated(self):
        """Test that very long strings are safely truncated."""
        long_string = "A" * 10000  # Very long string
        result = {
            "fol_formalization": {
                "original_sentence": long_string
            }
        }
        
        formatted = _format_fol_formalization(result)
        
        # Should be truncated to MAX_REASONING_RESULT_LENGTH
        assert len(formatted) < len(long_string)
        assert "A" in formatted  # Some content should be there


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
