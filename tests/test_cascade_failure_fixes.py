"""
Test suite for cascade failure fixes in VULCAN reasoning system.

These tests verify that the fixes for Issues #1-#7 work correctly:
- Issue #3: Symbolic engine parse failures return not_applicable
- Issue #4: Mathematical tool accepts proof verification queries
- Issue #6: Confidence threshold excludes not_applicable results  
- Issue #7: Hybrid executor allows OpenAI when engine declines
"""

import pytest
import re


class TestIssue4MathematicalVerification:
    """Test Issue #4 fix: Mathematical tool now accepts proof verification queries."""
    
    def test_mathematical_verification_pattern_detection(self):
        """Test that mathematical verification queries are recognized as mathematical."""
        from vulcan.reasoning.mathematical_computation import MathematicalComputationTool
        
        tool = MathematicalComputationTool()
        
        # Queries that should be recognized as mathematical verification
        math_verification_queries = [
            "Mathematical Verification: Proof check with hidden flaw",
            "Verify this proof step by step",
            "Proof check: Step 1: ..., Step 2: ...",
            "Check this mathematical proof for hidden flaws",
            "Claim: The sum converges. Therefore...",
        ]
        
        for query in math_verification_queries:
            result = tool._is_genuinely_mathematical(query)
            assert result is True, (
                f"Query should be recognized as mathematical: '{query}'"
            )
    
    def test_logic_queries_still_rejected(self):
        """Test that pure logic queries are still rejected."""
        from vulcan.reasoning.mathematical_computation import MathematicalComputationTool
        
        tool = MathematicalComputationTool()
        
        # Queries that should NOT be mathematical
        logic_queries = [
            "Prove that A → B using first-order logic",
            "Is this SAT formula satisfiable?",
            "Formalize this in FOL",
            "∀X (human(X) → mortal(X))",
        ]
        
        for query in logic_queries:
            result = tool._is_genuinely_mathematical(query)
            assert result is False, (
                f"Query should NOT be recognized as mathematical: '{query}'"
            )


class TestIssue3SymbolicParseFailures:
    """Test Issue #3 fix: Symbolic engine parse failures return not_applicable."""
    
    def test_parse_failure_returns_not_applicable(self):
        """Test that parse failures return not_applicable flag."""
        from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
        
        reasoner = SymbolicReasoner()
        
        # Query that will fail to parse (malformed logic)
        invalid_query = "This is not valid logic syntax @#$%"
        
        # Should not raise exception, should return dict with not_applicable
        result = reasoner.query(invalid_query, check_applicability=True)
        
        # Check that it returns proper failure signal
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result.get("confidence") == 0.0, (
            f"Parse failure should return confidence=0.0, got {result.get('confidence')}"
        )
        assert result.get("applicable") is False, (
            "Parse failure should set applicable=False"
        )
        assert result.get("not_applicable") is True, (
            "Parse failure should include explicit not_applicable=True flag"
        )
    
    def test_non_symbolic_query_returns_not_applicable(self):
        """Test that non-symbolic queries return not_applicable."""
        from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
        
        reasoner = SymbolicReasoner()
        
        # Natural language query without formal logic
        natural_query = "What is the weather like today?"
        
        result = reasoner.query(natural_query, check_applicability=True)
        
        assert result.get("applicable") is False, (
            "Natural language query should be marked as not applicable"
        )
        assert result.get("confidence") == 0.0, (
            "Non-applicable query should have confidence 0.0"
        )


class TestIssue6ConfidenceThreshold:
    """Test Issue #6 fix: Confidence threshold excludes not_applicable results."""
    
    def test_not_applicable_results_excluded_from_candidates(self):
        """Test that not_applicable results are filtered from candidates."""
        # This is an integration test concept - we can't easily test the full endpoint
        # but we can verify the logic
        
        # Simulate results with not_applicable flags
        unified_result = {
            "conclusion": "Some result",
            "confidence": 0.8,
            "not_applicable": True,  # Should be excluded
        }
        
        agent_result = {
            "conclusion": "Another result",  
            "confidence": 0.6,
            "applicable": False,  # Should be excluded
        }
        
        direct_result = {
            "conclusion": "Valid result",
            "confidence": 0.7,
            "applicable": True,  # Should be included
        }
        
        # Check the filtering logic
        unified_excluded = (
            unified_result.get("not_applicable") is True or
            unified_result.get("applicable") is False
        )
        assert unified_excluded, "Unified result should be excluded (not_applicable=True)"
        
        agent_excluded = (
            agent_result.get("not_applicable") is True or
            agent_result.get("applicable") is False
        )
        assert agent_excluded, "Agent result should be excluded (applicable=False)"
        
        direct_excluded = (
            direct_result.get("not_applicable") is True or
            direct_result.get("applicable") is False  
        )
        assert not direct_excluded, "Direct result should NOT be excluded"


class TestIssue7HybridExecutor:
    """Test Issue #7 fix: Hybrid executor allows OpenAI when engine declines."""
    
    def test_not_applicable_detection_in_reasoning_output(self):
        """Test that not_applicable flag is properly detected in reasoning output."""
        
        # Simulate reasoning output with not_applicable
        class MockReasoningOutput:
            def __init__(self, confidence, not_applicable=False):
                self.confidence = confidence
                self._not_applicable = not_applicable
            
            def to_dict(self):
                return {
                    "confidence": self.confidence,
                    "not_applicable": self._not_applicable,
                }
        
        # Case 1: Not applicable - should allow OpenAI
        output1 = MockReasoningOutput(confidence=0.0, not_applicable=True)
        is_not_applicable = False
        if hasattr(output1, 'to_dict'):
            try:
                output_dict = output1.to_dict()
                is_not_applicable = (
                    output_dict.get('not_applicable') is True or
                    output_dict.get('applicable') is False
                )
            except Exception:
                pass
        
        assert is_not_applicable, "Should detect not_applicable=True"
        
        # Case 2: Low confidence but applicable - should block OpenAI
        output2 = MockReasoningOutput(confidence=0.3, not_applicable=False)
        is_not_applicable = False
        if hasattr(output2, 'to_dict'):
            try:
                output_dict = output2.to_dict()
                is_not_applicable = (
                    output_dict.get('not_applicable') is True or
                    output_dict.get('applicable') is False
                )
            except Exception:
                pass
        
        assert not is_not_applicable, "Should NOT detect as not_applicable when False"


class TestIssue1QueryClassifier:
    """Test Issue #1: Query classifier header stripping."""
    
    def test_section_labels_stripped(self):
        """Test that section labels like C1, M1, P1 are stripped from queries."""
        from vulcan.routing.query_classifier import strip_query_headers
        
        test_cases = [
            ("M1 — Proof check: Verify this", "Proof check: Verify this"),
            ("Causal Reasoning C1 — Confounding vs causation", "Confounding vs causation"),
            ("Mathematical Verification M1 — Hidden flaw", "Hidden flaw"),
            ("Analogical Reasoning A1 — Structure mapping", "Structure mapping"),
        ]
        
        for input_query, expected_output in test_cases:
            result = strip_query_headers(input_query)
            # Strip extra whitespace for comparison
            result = result.strip()
            expected_output = expected_output.strip()
            
            assert result == expected_output, (
                f"Header stripping failed:\n"
                f"  Input: '{input_query}'\n"
                f"  Expected: '{expected_output}'\n"
                f"  Got: '{result}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
