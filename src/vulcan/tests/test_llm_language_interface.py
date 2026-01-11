"""
Tests for LLM Language Interface Architecture

Tests that LLMs are used correctly as language interfaces (parsing input,
formatting output) rather than as reasoning engines.

Test Strategy:
- Test query parsing (Language IN)
- Test 3-step flow (Parse → Compute → Format)
- Test that LLMs never answer directly when reasoning fails
- Test proper enum handling and error cases
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

# Import the modules we're testing
from vulcan.llm.query_parser import (
    QueryIntent,
    QueryDomain,
    StructuredQuery,
)
from vulcan.llm.hybrid_executor import (
    HybridLLMExecutor,
    VulcanReasoningOutput,
)


class TestQueryParser:
    """Test the query parser module."""
    
    def test_query_intent_enum_values(self):
        """Test that QueryIntent enum has correct values."""
        assert QueryIntent.COMPUTE.value == "compute"
        assert QueryIntent.EXPLAIN.value == "explain"
        assert QueryIntent.SEARCH.value == "search"
        assert QueryIntent.ANALYZE.value == "analyze"
        assert QueryIntent.PLAN.value == "plan"
        assert QueryIntent.COMPARE.value == "compare"
        assert QueryIntent.UNKNOWN.value == "unknown"
    
    def test_query_domain_enum_values(self):
        """Test that QueryDomain enum has correct values."""
        assert QueryDomain.MATH.value == "math"
        assert QueryDomain.LOGIC.value == "logic"
        assert QueryDomain.CAUSAL.value == "causal"
        assert QueryDomain.GENERAL.value == "general"
        assert QueryDomain.CODE.value == "code"
    
    def test_structured_query_from_json_valid(self):
        """Test parsing valid JSON into StructuredQuery."""
        json_str = json.dumps({
            "intent": "compute",
            "domain": "math",
            "parameters": {"operation": "add", "operands": [2, 2]},
            "confidence": 0.95
        })
        
        query = StructuredQuery.from_json(json_str, original_text="What's 2 plus 2?")
        
        assert query.intent == QueryIntent.COMPUTE
        assert query.domain == QueryDomain.MATH
        assert query.parameters["operation"] == "add"
        assert query.parameters["operands"] == [2, 2]
        assert query.confidence == 0.95
        assert query.original_text == "What's 2 plus 2?"
    
    def test_structured_query_from_json_missing_fields(self):
        """Test parsing JSON with missing optional fields."""
        json_str = json.dumps({
            "intent": "explain",
            "domain": "general"
        })
        
        query = StructuredQuery.from_json(json_str, original_text="Explain AI")
        
        assert query.intent == QueryIntent.EXPLAIN
        assert query.domain == QueryDomain.GENERAL
        assert query.parameters == {}
        assert query.confidence == 0.5  # Default confidence
    
    def test_structured_query_from_json_invalid_intent(self):
        """Test handling of invalid intent value."""
        json_str = json.dumps({
            "intent": "invalid_intent",
            "domain": "math",
            "confidence": 0.8
        })
        
        query = StructuredQuery.from_json(json_str, original_text="Test")
        
        # Should default to UNKNOWN for invalid intent
        assert query.intent == QueryIntent.UNKNOWN
        assert query.domain == QueryDomain.MATH
    
    def test_structured_query_from_json_invalid_domain(self):
        """Test handling of invalid domain value."""
        json_str = json.dumps({
            "intent": "compute",
            "domain": "invalid_domain",
            "confidence": 0.8
        })
        
        query = StructuredQuery.from_json(json_str, original_text="Test")
        
        # Should default to GENERAL for invalid domain
        assert query.intent == QueryIntent.COMPUTE
        assert query.domain == QueryDomain.GENERAL
    
    def test_structured_query_from_json_invalid_json(self):
        """Test handling of malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            StructuredQuery.from_json("not valid json", original_text="Test")
    
    def test_structured_query_to_dict(self):
        """Test converting StructuredQuery to dictionary."""
        query = StructuredQuery(
            intent=QueryIntent.COMPUTE,
            domain=QueryDomain.MATH,
            parameters={"operation": "multiply", "operands": [3, 4]},
            original_text="What's 3 times 4?",
            confidence=0.9
        )
        
        result = query.to_dict()
        
        assert result["intent"] == "compute"
        assert result["domain"] == "math"
        assert result["parameters"]["operation"] == "multiply"
        assert result["confidence"] == 0.9
        assert result["original_text"] == "What's 3 times 4?"
    
    def test_structured_query_repr(self):
        """Test string representation of StructuredQuery."""
        query = StructuredQuery(
            intent=QueryIntent.ANALYZE,
            domain=QueryDomain.LOGIC,
            confidence=0.75
        )
        
        repr_str = repr(query)
        
        assert "analyze" in repr_str
        assert "logic" in repr_str
        assert "0.75" in repr_str
    
    def test_structured_query_validation(self):
        """Test validation of StructuredQuery."""
        query = StructuredQuery(
            intent=QueryIntent.COMPUTE,
            domain=QueryDomain.MATH,
            parameters={"operation": "add"},
            confidence=0.9
        )
        
        assert query.validate() is True
    
    def test_structured_query_is_high_confidence(self):
        """Test high confidence check."""
        high_conf_query = StructuredQuery(
            intent=QueryIntent.COMPUTE,
            domain=QueryDomain.MATH,
            confidence=0.85
        )
        
        low_conf_query = StructuredQuery(
            intent=QueryIntent.UNKNOWN,
            domain=QueryDomain.GENERAL,
            confidence=0.3
        )
        
        assert high_conf_query.is_high_confidence() is True
        assert low_conf_query.is_high_confidence() is False
        assert low_conf_query.is_high_confidence(threshold=0.2) is True
    
    def test_structured_query_confidence_validation(self):
        """Test that invalid confidence values are rejected."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            StructuredQuery(
                intent=QueryIntent.COMPUTE,
                domain=QueryDomain.MATH,
                confidence=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            StructuredQuery(
                intent=QueryIntent.COMPUTE,
                domain=QueryDomain.MATH,
                confidence=-0.1  # Invalid: < 0.0
            )
    
    def test_structured_query_confidence_type_validation(self):
        """Test that non-numeric confidence values are rejected."""
        with pytest.raises(TypeError, match="confidence must be a number"):
            StructuredQuery(
                intent=QueryIntent.COMPUTE,
                domain=QueryDomain.MATH,
                confidence="high"  # Invalid: not a number
            )


class TestLanguageInterfaceMethods:
    """Test the language interface methods in HybridExecutor."""
    
    @pytest.mark.asyncio
    async def test_parse_natural_language_query_success(self):
        """Test successful parsing of natural language query."""
        # Create executor with mocked OpenAI
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        # Mock the OpenAI call to return valid JSON
        mock_json = json.dumps({
            "intent": "compute",
            "domain": "math",
            "parameters": {"operation": "add", "operands": [2, 2]},
            "confidence": 0.95
        })
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_json
            
            query = await executor.parse_natural_language_query("What's 2 plus 2?")
            
            assert query.intent == QueryIntent.COMPUTE
            assert query.domain == QueryDomain.MATH
            assert query.parameters["operation"] == "add"
            assert query.confidence == 0.95
            assert query.original_text == "What's 2 plus 2?"
    
    @pytest.mark.asyncio
    async def test_parse_natural_language_query_fallback(self):
        """Test fallback when LLM returns invalid JSON."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        # Mock the OpenAI call to return invalid JSON
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "not valid json"
            
            query = await executor.parse_natural_language_query("Test query")
            
            # Should return UNKNOWN query as fallback
            assert query.intent == QueryIntent.UNKNOWN
            assert query.domain == QueryDomain.GENERAL
            assert query.confidence == 0.0
            assert "raw_text" in query.parameters
    
    @pytest.mark.asyncio
    async def test_parse_natural_language_query_null_response(self):
        """Test fallback when LLM returns None."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = None
            
            query = await executor.parse_natural_language_query("Test query")
            
            # Should return UNKNOWN query as fallback
            assert query.intent == QueryIntent.UNKNOWN
            assert query.domain == QueryDomain.GENERAL
            assert query.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_with_language_interface_full_flow(self):
        """Test complete 3-step flow: Parse → Compute → Format."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        # Mock the parsing step
        parse_json = json.dumps({
            "intent": "compute",
            "domain": "math",
            "parameters": {"operation": "add", "operands": [2, 2]},
            "confidence": 0.95
        })
        
        # Mock reasoning function
        async def mock_reasoning(query: StructuredQuery) -> VulcanReasoningOutput:
            return VulcanReasoningOutput(
                query_id="test123",
                success=True,
                result=4,
                result_type="mathematical",
                method_used="arithmetic",
                confidence=1.0,
                reasoning_trace=["2 + 2 = 4"]
            )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            # First call is for parsing, second is for formatting
            mock_call.side_effect = [
                parse_json,  # Parse step
                "The answer is 4."  # Format step
            ]
            
            result = await executor.execute_with_language_interface(
                user_text="What's 2 plus 2?",
                vulcan_reasoning_fn=mock_reasoning
            )
            
            # Verify all three steps were used
            assert "llm_input_parsing" in result["systems_used"]
            assert "arithmetic" in result["systems_used"]
            assert "llm_output_formatting" in result["systems_used"]
            
            # Verify structured query is in result
            assert result["structured_query"]["intent"] == "compute"
            assert result["structured_query"]["domain"] == "math"
            
            # Verify reasoning output is in result
            assert result["reasoning_output"]["success"] is True
            assert result["reasoning_output"]["result"] == 4
            
            # Verify source
            assert result["source"] == "vulcan_language_interface"
    
    @pytest.mark.asyncio
    async def test_execute_with_language_interface_no_reasoning_fn(self):
        """Test behavior when no reasoning function is provided."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        parse_json = json.dumps({
            "intent": "compute",
            "domain": "math",
            "confidence": 0.95
        })
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [
                parse_json,  # Parse step
                "I cannot process this request without a reasoning engine."  # Format step
            ]
            
            result = await executor.execute_with_language_interface(
                user_text="What's 2 plus 2?",
                vulcan_reasoning_fn=None  # No reasoning function
            )
            
            # Should still complete all steps
            assert "llm_input_parsing" in result["systems_used"]
            assert "no_reasoning_configured" in result["systems_used"]
            assert "llm_output_formatting" in result["systems_used"]
            
            # Reasoning output should indicate failure
            assert result["reasoning_output"]["success"] is False
            assert "No reasoning function configured" in result["reasoning_output"]["error"]
    
    @pytest.mark.asyncio
    async def test_execute_with_language_interface_reasoning_error(self):
        """Test handling when VULCAN reasoning fails."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        parse_json = json.dumps({
            "intent": "compute",
            "domain": "math",
            "confidence": 0.95
        })
        
        # Mock reasoning function that fails
        async def failing_reasoning(query: StructuredQuery) -> VulcanReasoningOutput:
            raise ValueError("Reasoning engine error")
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [
                parse_json,  # Parse step
                "I encountered an error processing your request."  # Format step
            ]
            
            result = await executor.execute_with_language_interface(
                user_text="What's 2 plus 2?",
                vulcan_reasoning_fn=failing_reasoning
            )
            
            # Should handle error gracefully
            assert "llm_input_parsing" in result["systems_used"]
            assert "llm_output_formatting" in result["systems_used"]
            
            # Reasoning output should indicate error
            assert result["reasoning_output"]["success"] is False
            assert "Reasoning engine error" in result["reasoning_output"]["error"]


class TestDeprecationWarnings:
    """Test that deprecated methods emit warnings."""
    
    @pytest.mark.asyncio
    async def test_execute_openai_first_deprecation_warning(self):
        """Test that _execute_openai_first emits deprecation warning."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Test response"
            
            with pytest.warns(DeprecationWarning, match="Use execute_with_language_interface"):
                loop = asyncio.get_running_loop()
                await executor._execute_openai_first(
                    loop=loop,
                    prompt="Test",
                    max_tokens=100,
                    temperature=0.7,
                    system_prompt="System"
                )
    
    @pytest.mark.asyncio
    async def test_execute_openai_only_deprecation_warning(self):
        """Test that _execute_openai_only emits deprecation warning."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Test response"
            
            with pytest.warns(DeprecationWarning, match="Use execute_with_language_interface"):
                loop = asyncio.get_running_loop()
                await executor._execute_openai_only(
                    loop=loop,
                    prompt="Test",
                    max_tokens=100,
                    temperature=0.7,
                    system_prompt="System"
                )


class TestNoDirectLLMAnswering:
    """Test that LLMs never answer directly when reasoning fails."""
    
    @pytest.mark.asyncio
    async def test_llm_does_not_solve_when_reasoning_fails(self):
        """Test that LLM formats failure messages, not attempts to solve."""
        executor = HybridLLMExecutor(
            local_llm=None,
            mode="openai_only"
        )
        
        # Mock reasoning function that returns failure
        async def failing_reasoning(query: StructuredQuery) -> VulcanReasoningOutput:
            return VulcanReasoningOutput(
                query_id="test123",
                success=False,
                result=None,
                error="Cannot solve this problem",
                method_used="symbolic_failed",
                confidence=0.0
            )
        
        parse_json = json.dumps({
            "intent": "compute",
            "domain": "math",
            "confidence": 0.95
        })
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            # LLM should receive the reasoning failure and format it appropriately
            mock_call.side_effect = [
                parse_json,  # Parse step
                "I was unable to solve this problem. The reasoning engine reported: Cannot solve this problem"
            ]
            
            result = await executor.execute_with_language_interface(
                user_text="Solve the halting problem",
                vulcan_reasoning_fn=failing_reasoning
            )
            
            # The response should acknowledge failure, not contain an attempt to solve
            response_text = result["text"].lower()
            
            # Should contain failure acknowledgment
            assert "unable" in response_text or "cannot" in response_text or "error" in response_text
            
            # Should NOT contain the LLM trying to explain the halting problem
            # (which would indicate it's answering instead of just formatting)
            assert result["reasoning_output"]["success"] is False


class TestModuleExports:
    """Test that module exports are correct."""
    
    def test_query_parser_exports(self):
        """Test that query_parser exports all required components."""
        from vulcan.llm.query_parser import __all__
        
        assert "QueryIntent" in __all__
        assert "QueryDomain" in __all__
        assert "StructuredQuery" in __all__
    
    def test_llm_module_exports(self):
        """Test that llm module exports query parser components."""
        from vulcan.llm import __all__
        
        assert "StructuredQuery" in __all__
        assert "QueryIntent" in __all__
        assert "QueryDomain" in __all__
    
    def test_llm_module_imports_query_parser(self):
        """Test that components can be imported from vulcan.llm."""
        from vulcan.llm import StructuredQuery, QueryIntent, QueryDomain
        
        # Should not raise ImportError
        assert StructuredQuery is not None
        assert QueryIntent is not None
        assert QueryDomain is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
