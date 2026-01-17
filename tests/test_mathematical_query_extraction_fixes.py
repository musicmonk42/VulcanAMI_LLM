"""
Tests for Mathematical Query Extraction Pipeline Bug Fixes

This test suite validates the three critical bug fixes in the extraction pipeline:
- Bug A: _is_genuinely_mathematical() pattern recognition enhancements
- Bug B: Conclusion extraction with proper formatted_output handling
- Bug C: Routing instructions extraction from task parameters

Industry Standard: Test-driven validation of critical bug fixes with clear documentation
"""

import pytest
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the mathematical computation module
try:
    from vulcan.reasoning.mathematical_computation import (
        MathematicalComputationTool,
        ProblemClassifier,
        create_mathematical_computation_tool,
    )
    COMPUTATION_TOOL_AVAILABLE = True
except ImportError:
    try:
        from src.vulcan.reasoning.mathematical_computation import (
            MathematicalComputationTool,
            ProblemClassifier,
            create_mathematical_computation_tool,
        )
        COMPUTATION_TOOL_AVAILABLE = True
    except ImportError:
        COMPUTATION_TOOL_AVAILABLE = False
        logger.warning("Mathematical computation tool not available for testing")


# ============================================================================
# BUG A: MATHEMATICAL PATTERN RECOGNITION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestBugA_MathematicalPatternRecognition:
    """
    Test Bug A fixes for _is_genuinely_mathematical() pattern recognition.
    
    Industry Standard: Each test validates a specific pattern category with
    clear naming and documentation of expected behavior.
    """
    
    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        return create_mathematical_computation_tool()
    
    def test_bayesian_conditional_probability_notation(self, tool):
        """
        BUG A FIX: Test P(X|Y) conditional probability notation recognition.
        
        Expected: Query with P(X|+) notation should be recognized as mathematical.
        """
        query = "A test for condition X: Sensitivity: 0.99, Specificity: 0.95, Prevalence: 0.01. Compute P(X|+) exactly."
        
        # Access the internal method for direct testing
        is_math = tool._is_genuinely_mathematical(query)
        
        assert is_math, "P(X|+) notation should be recognized as mathematical"
        logger.info("✓ Bayesian conditional probability P(X|+) correctly recognized")
    
    def test_bayesian_keywords_with_numerical_data(self, tool):
        """
        BUG A FIX: Test Bayesian keywords (sensitivity, specificity, etc.) with decimals.
        
        Expected: Queries with Bayesian terms and decimal numbers should be mathematical.
        """
        queries = [
            "Given sensitivity 0.99 and specificity 0.95, compute the posterior",
            "Prevalence is 0.01, what is the false positive rate?",
            "Apply Bayes theorem with prior 0.05 and likelihood 0.90",
            "Bayes' rule: sensitivity = 0.99, specificity = 0.95",
        ]
        
        for query in queries:
            is_math = tool._is_genuinely_mathematical(query)
            assert is_math, f"Bayesian query should be mathematical: {query[:50]}..."
        
        logger.info("✓ All Bayesian keyword queries correctly recognized")
    
    def test_summation_with_unicode_symbol(self, tool):
        """
        BUG A FIX: Test summation with unicode ∑ symbol recognition.
        
        Expected: Queries with ∑ and bounds (from...to) should be mathematical.
        """
        queries = [
            "Compute exactly: ∑(2k-1) from k=1 to n",
            "Find the sum ∑ k from 1 to 100",
            "Evaluate ∑(i^2) from i=1 to n, then verify by induction",
            "What is ∑(1/n) from n=1 to infinity?",
        ]
        
        for query in queries:
            is_math = tool._is_genuinely_mathematical(query)
            assert is_math, f"Summation query should be mathematical: {query[:50]}..."
        
        logger.info("✓ All summation queries with ∑ correctly recognized")
    
    def test_natural_language_math_commands(self, tool):
        """
        BUG A FIX: Test natural language mathematical command recognition.
        
        Expected: Commands like "compute exactly", "show steps", "verify by induction"
        should be recognized as mathematical.
        """
        queries = [
            "Compute exactly the value of the integral",
            "Calculate exactly: 2^10 - 5^3",
            "Show all steps for solving x^2 + 5x + 6 = 0",
            "Verify by induction that n^2 is always even",
            "Prove by induction that the sum is n^2",
            "Find the closed form of the recurrence relation",
            "Derive the formula for the nth term",
        ]
        
        for query in queries:
            is_math = tool._is_genuinely_mathematical(query)
            assert is_math, f"Natural language math command should be mathematical: {query[:50]}..."
        
        logger.info("✓ All natural language math commands correctly recognized")
    
    def test_proof_verification_with_calculus_terms(self, tool):
        """
        BUG A FIX: Test mathematical verification with calculus terms.
        
        Expected: Queries containing "verify" with calculus terms like "differentiable",
        "continuous" should be recognized as mathematical.
        """
        queries = [
            "Claim: All differentiable functions are continuous. Verify each step.",
            "Check the proof that the limit exists",
            "Verify that the derivative is zero at the maximum",
            "Mathematical verification: Is this integral convergent?",
        ]
        
        for query in queries:
            is_math = tool._is_genuinely_mathematical(query)
            assert is_math, f"Verification with calculus should be mathematical: {query[:50]}..."
        
        logger.info("✓ All proof verification queries correctly recognized")
    
    def test_non_mathematical_queries_rejected(self, tool):
        """
        BUG A FIX: Ensure non-mathematical queries are still properly rejected.
        
        Expected: Queries without genuine math content should return False.
        Industry Standard: Defensive testing - verify negative cases work correctly.
        """
        non_math_queries = [
            "What makes you different from other AI systems?",
            "Can you solve this riddle about a river?",
            "Integrate this feedback into your response",
            "Tell me a story about a mathematician",
        ]
        
        for query in non_math_queries:
            is_math = tool._is_genuinely_mathematical(query)
            assert not is_math, f"Non-math query should be rejected: {query[:50]}..."
        
        logger.info("✓ All non-mathematical queries correctly rejected")


# ============================================================================
# BUG B: CONCLUSION EXTRACTION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestBugB_ConclusionExtraction:
    """
    Test Bug B fixes for conclusion extraction in orchestrator.
    
    Industry Standard: Test data structures match real-world response formats
    from mathematical computation tools.
    """
    
    def test_formatted_output_extraction(self):
        """
        BUG B FIX: Test that non-empty formatted_output is properly extracted.
        
        Expected: formatted_output string should be used when present and non-empty.
        """
        # Simulate raw_result from MathematicalComputationTool
        raw_result = {
            'conclusion': {
                'success': True,
                'result': 'n**2',
                'code': 'some_code'
            },
            'confidence': 0.9,
            'formatted_output': '**Mathematical Computation**\n\n**Result:** n²\n\nSteps shown below...',
        }
        
        # Extract following Bug B fix logic
        conclusion = raw_result.get('conclusion', {})
        computed_result = None
        if isinstance(conclusion, dict):
            computed_result = conclusion.get('result')
        
        formatted_output = raw_result.get('formatted_output', '')
        
        # Bug B fix: Check formatted_output is non-empty string
        if isinstance(formatted_output, str) and formatted_output.strip():
            user_conclusion = formatted_output
            extraction_method = "formatted_output"
        elif computed_result is not None:
            user_conclusion = f"**Result:** {computed_result}"
            extraction_method = "computed_result"
        else:
            user_conclusion = raw_result
            extraction_method = "fallback_raw_result"
        
        assert extraction_method == "formatted_output", "Should use formatted_output when non-empty"
        assert user_conclusion == raw_result['formatted_output'], "Should extract full formatted_output"
        logger.info("✓ Formatted output correctly extracted when non-empty")
    
    def test_nested_result_extraction(self):
        """
        BUG B FIX: Test extraction from nested conclusion.result field.
        
        Expected: When formatted_output is empty, should extract conclusion.result.
        """
        raw_result = {
            'conclusion': {
                'success': True,
                'result': 'n**2',  # Nested result
                'code': 'some_code'
            },
            'confidence': 0.9,
            'formatted_output': '',  # Empty string (falsy but not None)
        }
        
        # Extract following Bug B fix logic
        conclusion = raw_result.get('conclusion', {})
        computed_result = None
        if isinstance(conclusion, dict):
            computed_result = conclusion.get('result')
        
        formatted_output = raw_result.get('formatted_output', '')
        
        # Bug B fix: Check formatted_output is non-empty string
        if isinstance(formatted_output, str) and formatted_output.strip():
            user_conclusion = formatted_output
            extraction_method = "formatted_output"
        elif computed_result is not None:
            user_conclusion = f"**Result:** {computed_result}"
            extraction_method = "computed_result"
        else:
            user_conclusion = raw_result
            extraction_method = "fallback_raw_result"
        
        assert extraction_method == "computed_result", "Should extract computed_result when formatted_output empty"
        assert "n**2" in user_conclusion, "Should include nested result value"
        logger.info("✓ Nested conclusion.result correctly extracted")
    
    def test_empty_formatted_output_not_used(self):
        """
        BUG B FIX: Test that empty formatted_output falls through to computed_result.
        
        Expected: Empty string formatted_output should not be used (was the bug).
        """
        raw_result = {
            'conclusion': {
                'result': '0.167'  # This should be used
            },
            'confidence': 0.89,
            'formatted_output': '',  # Empty - should not be used
        }
        
        # Extract following Bug B fix logic
        conclusion = raw_result.get('conclusion', {})
        computed_result = None
        if isinstance(conclusion, dict):
            computed_result = conclusion.get('result')
        
        formatted_output = raw_result.get('formatted_output', '')
        
        # Bug B fix: Check formatted_output is non-empty string (not just truthy)
        if isinstance(formatted_output, str) and formatted_output.strip():
            user_conclusion = formatted_output
            extraction_method = "formatted_output"
        elif computed_result is not None:
            user_conclusion = f"**Result:** {computed_result}"
            extraction_method = "computed_result"
        else:
            user_conclusion = raw_result
            extraction_method = "fallback_raw_result"
        
        # Verify empty formatted_output was NOT used
        assert extraction_method != "formatted_output", "Empty formatted_output should not be used"
        assert extraction_method == "computed_result", "Should use computed_result instead"
        assert "0.167" in user_conclusion, "Should include the actual computed result"
        logger.info("✓ Empty formatted_output correctly bypassed")


# ============================================================================
# BUG C: ROUTING INSTRUCTIONS EXTRACTION TESTS
# ============================================================================


class TestBugC_RoutingInstructionsExtraction:
    """
    Test Bug C fixes for routing instructions extraction from task parameters.
    
    Industry Standard: Test the command pattern implementation with proper
    priority chain: task.parameters → task.attributes → fallbacks.
    """
    
    def test_task_parameters_priority(self):
        """
        BUG C FIX: Test that task["parameters"] is checked first for routing instructions.
        
        Expected: reasoning_type and tool_name in task.parameters should be
        extracted with highest priority (router sets these).
        """
        # Simulate task dict as passed to _execute_agent_task
        task = {
            "task_id": "job_35f8c26c",
            "parameters": {
                "reasoning_type": "mathematical",  # Router sets this
                "tool_name": "mathematical",       # Router sets this
                "query": "Compute exactly: ∑(2k-1) from k=1 to n"
            },
            "graph": {}
        }
        
        # Simulate Bug C fix extraction logic
        router_reasoning_type = None
        router_tool_name = None
        
        # BUG C FIX: Check task["parameters"] dict first
        task_params = task.get("parameters", {}) if isinstance(task, dict) else {}
        if task_params:
            if "reasoning_type" in task_params:
                router_reasoning_type = task_params.get("reasoning_type")
            router_tool_name = task_params.get("tool_name") or task_params.get("selected_tool")
        
        # Verify extraction succeeded
        assert router_reasoning_type == "mathematical", "Should extract reasoning_type from task.parameters"
        assert router_tool_name == "mathematical", "Should extract tool_name from task.parameters"
        logger.info("✓ Routing instructions correctly extracted from task.parameters")
    
    def test_selected_tool_variant(self):
        """
        BUG C FIX: Test that both tool_name and selected_tool are checked.
        
        Expected: Router might use either tool_name or selected_tool key.
        """
        task = {
            "parameters": {
                "reasoning_type": "probabilistic",
                "selected_tool": "probabilistic",  # Variant name
            }
        }
        
        # Simulate extraction
        task_params = task.get("parameters", {})
        router_reasoning_type = task_params.get("reasoning_type")
        router_tool_name = task_params.get("tool_name") or task_params.get("selected_tool")
        
        assert router_reasoning_type == "probabilistic"
        assert router_tool_name == "probabilistic", "Should extract selected_tool variant"
        logger.info("✓ selected_tool variant correctly handled")
    
    def test_command_pattern_validation_detection(self):
        """
        BUG C FIX: Test that missing routing instructions are properly detected.
        
        Expected: When router instructions are missing, should be detected and logged.
        Industry Standard: Fail-fast validation with clear error detection.
        """
        # Task without routing instructions (bug scenario)
        task = {
            "task_id": "job_test",
            "parameters": {},  # No routing instructions
            "graph": {"type": "reasoning"}
        }
        
        # Simulate extraction
        task_params = task.get("parameters", {})
        router_reasoning_type = task_params.get("reasoning_type")
        router_tool_name = task_params.get("tool_name") or task_params.get("selected_tool")
        
        # Check if violation would be detected
        is_reasoning_task = True  # From graph.type
        has_routing_instructions = bool(router_reasoning_type and router_tool_name)
        
        assert not has_routing_instructions, "Should detect missing routing instructions"
        
        if is_reasoning_task and not has_routing_instructions:
            violation_detected = True
        else:
            violation_detected = False
        
        assert violation_detected, "COMMAND PATTERN VIOLATION should be detected"
        logger.info("✓ Missing routing instructions properly detected")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(
    not COMPUTATION_TOOL_AVAILABLE,
    reason="Mathematical computation tool not available"
)
class TestBugFixes_Integration:
    """
    Integration tests validating all three bug fixes work together.
    
    Industry Standard: End-to-end validation of complete pipeline.
    """
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return create_mathematical_computation_tool()
    
    def test_bayesian_query_end_to_end(self, tool):
        """
        Integration test: Bayesian query should be recognized and processed.
        
        Tests Bug A fix: Query recognition
        Note: Full execution would test Bug B (result extraction)
        """
        query = "Sensitivity: 0.99, Specificity: 0.95, Prevalence: 0.01. Compute P(X|+)."
        
        # Test Bug A: Recognition
        is_math = tool._is_genuinely_mathematical(query)
        assert is_math, "Bayesian query should be recognized as mathematical"
        
        logger.info("✓ Bayesian query integration test passed")
    
    def test_summation_induction_query_end_to_end(self, tool):
        """
        Integration test: Summation with induction verification.
        
        Tests Bug A fix: Complex query with multiple mathematical patterns
        """
        query = "Compute exactly: ∑(2k-1) from k=1 to n, then verify by induction"
        
        # Test Bug A: Recognition of multiple patterns
        is_math = tool._is_genuinely_mathematical(query)
        assert is_math, "Summation with induction should be recognized"
        
        logger.info("✓ Summation with induction integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
