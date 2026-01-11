"""
Integration tests for LLM Language Interface Architecture

Tests that the new language interface components properly integrate with:
- HybridLLMExecutor
- VulcanReasoningOutput
- Existing reasoning systems
- Module exports

These tests verify end-to-end functionality and compatibility.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from vulcan.llm import (
    HybridLLMExecutor,
    VulcanReasoningOutput,
    StructuredQuery,
    QueryIntent,
    QueryDomain,
    get_or_create_hybrid_executor,
)


class TestModuleIntegration:
    """Test that all components are properly exported and importable."""
    
    def test_all_exports_available(self):
        """Test that all new exports are available from vulcan.llm."""
        from vulcan.llm import __all__
        
        # Check new exports are in __all__
        required_exports = ["StructuredQuery", "QueryIntent", "QueryDomain"]
        for export in required_exports:
            assert export in __all__, f"{export} not in __all__"
    
    def test_direct_imports_work(self):
        """Test that components can be imported directly."""
        # Should not raise ImportError
        from vulcan.llm import StructuredQuery, QueryIntent, QueryDomain
        from vulcan.llm.query_parser import StructuredQuery as DirectStructuredQuery
        
        # Should be the same class
        assert StructuredQuery is DirectStructuredQuery
    
    def test_backward_compatibility(self):
        """Test that existing exports still work."""
        from vulcan.llm import (
            HybridLLMExecutor,
            VulcanReasoningOutput,
            get_or_create_hybrid_executor,
        )
        
        # Should not raise ImportError or AttributeError
        assert HybridLLMExecutor is not None
        assert VulcanReasoningOutput is not None
        assert get_or_create_hybrid_executor is not None


class TestHybridExecutorIntegration:
    """Test integration of new methods with HybridExecutor."""
    
    def test_hybrid_executor_has_new_methods(self):
        """Test that HybridExecutor has the new language interface methods."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        assert hasattr(executor, "parse_natural_language_query")
        assert hasattr(executor, "execute_with_language_interface")
        assert callable(executor.parse_natural_language_query)
        assert callable(executor.execute_with_language_interface)
    
    def test_new_methods_are_async(self):
        """Test that new methods are properly async."""
        import inspect
        
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        assert asyncio.iscoroutinefunction(executor.parse_natural_language_query)
        assert asyncio.iscoroutinefunction(executor.execute_with_language_interface)
    
    def test_existing_methods_still_work(self):
        """Test that existing HybridExecutor methods are not broken."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        # Check existing methods still exist
        assert hasattr(executor, "execute")
        assert hasattr(executor, "format_output_for_user")
        assert hasattr(executor, "get_stats")
        assert callable(executor.execute)
        assert callable(executor.format_output_for_user)


class TestStructuredQueryIntegration:
    """Test StructuredQuery integration with other components."""
    
    def test_structured_query_with_vulcan_reasoning_output(self):
        """Test that StructuredQuery works with VulcanReasoningOutput."""
        # Create a query
        query = StructuredQuery(
            intent=QueryIntent.COMPUTE,
            domain=QueryDomain.MATH,
            parameters={"operation": "add", "operands": [2, 2]},
            original_text="What's 2 + 2?",
            confidence=0.95
        )
        
        # Create reasoning output
        reasoning = VulcanReasoningOutput(
            query_id="test123",
            success=True,
            result=4,
            result_type="mathematical",
            method_used="arithmetic",
            confidence=1.0
        )
        
        # Should be able to convert both to dict
        query_dict = query.to_dict()
        reasoning_dict = reasoning.to_dict()
        
        assert "intent" in query_dict
        assert "result" in reasoning_dict
        
        # Should be JSON serializable
        json.dumps(query_dict)
        json.dumps(reasoning_dict)
    
    def test_structured_query_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = StructuredQuery(
            intent=QueryIntent.ANALYZE,
            domain=QueryDomain.LOGIC,
            parameters={"topic": "reasoning"},
            original_text="Analyze this",
            confidence=0.85
        )
        
        # Convert to dict, then to JSON, then parse back
        query_dict = original.to_dict()
        json_str = json.dumps(query_dict)
        parsed = StructuredQuery.from_json(json_str, original_text="Analyze this")
        
        assert parsed.intent == original.intent
        assert parsed.domain == original.domain
        assert parsed.confidence == original.confidence
        assert parsed.parameters == original.parameters


class TestEndToEndIntegration:
    """Test end-to-end integration of the language interface."""
    
    @pytest.mark.asyncio
    async def test_full_language_interface_flow(self):
        """Test complete flow: Parse → Compute → Format."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        # Mock the LLM calls
        parse_response = json.dumps({
            "intent": "compute",
            "domain": "math",
            "parameters": {"operation": "multiply", "operands": [6, 7]},
            "confidence": 0.95
        })
        
        format_response = "The answer is 42."
        
        # Mock reasoning function
        async def mock_reasoning(query: StructuredQuery) -> VulcanReasoningOutput:
            assert query.intent == QueryIntent.COMPUTE
            assert query.domain == QueryDomain.MATH
            return VulcanReasoningOutput(
                query_id="test123",
                success=True,
                result=42,
                result_type="mathematical",
                method_used="multiplication",
                confidence=1.0
            )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [parse_response, format_response]
            
            result = await executor.execute_with_language_interface(
                user_text="What's 6 times 7?",
                vulcan_reasoning_fn=mock_reasoning
            )
            
            # Verify result structure
            assert "text" in result
            assert "source" in result
            assert "systems_used" in result
            assert "structured_query" in result
            assert "reasoning_output" in result
            
            # Verify all steps were executed
            assert "llm_input_parsing" in result["systems_used"]
            assert "multiplication" in result["systems_used"]
            assert "llm_output_formatting" in result["systems_used"]
            
            # Verify structured query
            assert result["structured_query"]["intent"] == "compute"
            assert result["structured_query"]["domain"] == "math"
            
            # Verify reasoning output
            assert result["reasoning_output"]["success"] is True
            assert result["reasoning_output"]["result"] == 42
    
    @pytest.mark.asyncio
    async def test_integration_with_format_output_for_user(self):
        """Test that new methods integrate with existing format_output_for_user."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        # Create reasoning output
        reasoning_output = VulcanReasoningOutput(
            query_id="test123",
            success=True,
            result="The sky is blue due to Rayleigh scattering",
            result_type="factual",
            method_used="knowledge_retrieval",
            confidence=0.9
        )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "The sky appears blue because of Rayleigh scattering."
            
            result = await executor.format_output_for_user(
                reasoning_output=reasoning_output.to_dict(),
                original_prompt="Why is the sky blue?",
                max_tokens=500
            )
            
            # Verify result structure
            assert "text" in result
            assert "source" in result
            assert "systems_used" in result
            
            # Verify OpenAI was used only for formatting
            assert "openai_formatting" in result["source"]


class TestSingletonIntegration:
    """Test singleton pattern integration with new functionality."""
    
    def test_singleton_has_new_methods(self):
        """Test that singleton executor has new methods."""
        executor = get_or_create_hybrid_executor(
            local_llm=None,
            mode="openai_only",
            force_new=True  # Force new instance for test isolation
        )
        
        assert hasattr(executor, "parse_natural_language_query")
        assert hasattr(executor, "execute_with_language_interface")
    
    @pytest.mark.asyncio
    async def test_singleton_can_use_new_methods(self):
        """Test that singleton can actually call new methods."""
        executor = get_or_create_hybrid_executor(
            local_llm=None,
            mode="openai_only",
            force_new=True
        )
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = json.dumps({
                "intent": "search",
                "domain": "general",
                "parameters": {"query": "test"},
                "confidence": 0.8
            })
            
            query = await executor.parse_natural_language_query("Find test data")
            
            assert isinstance(query, StructuredQuery)
            assert query.intent == QueryIntent.SEARCH


class TestDeprecationIntegration:
    """Test that deprecation warnings are properly integrated."""
    
    @pytest.mark.asyncio
    async def test_deprecated_methods_still_work_with_warning(self):
        """Test that deprecated methods still function but warn."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Test response"
            
            # Should work but emit warning
            with pytest.warns(DeprecationWarning):
                loop = asyncio.get_running_loop()
                result = await executor._execute_openai_first(
                    loop=loop,
                    prompt="Test",
                    max_tokens=100,
                    temperature=0.7,
                    system_prompt="System"
                )
            
            assert result["text"] == "Test response"
    
    def test_execute_method_still_uses_old_modes(self):
        """Test that execute() method still works for backward compatibility."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        # Should have execute method
        assert hasattr(executor, "execute")
        assert callable(executor.execute)


class TestErrorHandlingIntegration:
    """Test error handling integration."""
    
    @pytest.mark.asyncio
    async def test_parse_error_handling(self):
        """Test that parse errors are handled gracefully."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            # Return invalid JSON
            mock_call.return_value = "not valid json"
            
            query = await executor.parse_natural_language_query("Test")
            
            # Should return UNKNOWN query, not crash
            assert query.intent == QueryIntent.UNKNOWN
            assert query.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_reasoning_error_handling(self):
        """Test that reasoning errors are handled gracefully."""
        executor = HybridLLMExecutor(local_llm=None, mode="openai_only")
        
        async def failing_reasoning(query: StructuredQuery) -> VulcanReasoningOutput:
            raise RuntimeError("Reasoning failed")
        
        parse_response = json.dumps({
            "intent": "compute",
            "domain": "math",
            "confidence": 0.9
        })
        
        with patch.object(executor, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [
                parse_response,
                "I encountered an error."
            ]
            
            result = await executor.execute_with_language_interface(
                user_text="Test",
                vulcan_reasoning_fn=failing_reasoning
            )
            
            # Should handle error gracefully
            assert result["reasoning_output"]["success"] is False
            assert "Reasoning failed" in result["reasoning_output"]["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
