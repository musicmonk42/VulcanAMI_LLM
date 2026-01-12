"""
Tests for Issue #ROUTING-001: Query Routing Bypass Fixes

These tests verify that queries route correctly to VULCAN's reasoning engines
instead of bypassing them and going directly to OpenAI.

Fixes tested:
1. Self-awareness queries route to WorldModel
2. Analogical reasoning queries route to AnalogyEngine
3. Fallback detection catches queries that slip through pattern matching
4. Content preservation prevents OpenAI from replacing VULCAN's responses
"""

import pytest
import re


class TestSelfAwarenessPatterns:
    """Test that self-awareness queries are correctly detected."""
    
    def test_self_awareness_pattern_in_classifier(self):
        """Verify self-awareness patterns exist in query_classifier.py"""
        from src.vulcan.routing.query_classifier import SELF_INTROSPECTION_PATTERNS
        
        # Test queries that should match
        test_queries = [
            "if you have the chance to become self aware would you take it",
            "would you choose to be self-aware",
            "given the opportunity to become self aware",
            "would you take it if you could become self-aware",
        ]
        
        for query in test_queries:
            matched = any(pattern.search(query) for pattern in SELF_INTROSPECTION_PATTERNS)
            assert matched, f"Pattern should match query: {query}"
    
    def test_analogical_keywords_in_classifier(self):
        """Verify analogical reasoning keywords exist in query_classifier.py"""
        from src.vulcan.routing.query_classifier import ANALOGICAL_KEYWORDS
        
        # Test keywords that should be present
        required_keywords = [
            "analogical reasoning",
            "structure mapping",
            "domain s",
            "domain t",
        ]
        
        for keyword in required_keywords:
            assert keyword in ANALOGICAL_KEYWORDS, f"Missing keyword: {keyword}"


class TestFallbackDetection:
    """Test fallback detection when pattern matching fails."""
    
    def test_fallback_detection_method_exists(self):
        """Verify fallback detection method exists in QueryAnalyzer."""
        from src.vulcan.routing.query_router import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        assert hasattr(analyzer, '_detect_query_type_fallback'), \
            "QueryAnalyzer should have _detect_query_type_fallback method"
    
    def test_fallback_detects_self_awareness(self):
        """Test fallback detection catches self-awareness queries."""
        from src.vulcan.routing.query_router import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        
        # Query with self-awareness keywords
        query = "if you have the chance to become self aware would you take it"
        result = analyzer._detect_query_type_fallback(query)
        
        assert result == 'self_introspection', \
            f"Fallback should detect self_introspection, got: {result}"
    
    def test_fallback_detects_analogical(self):
        """Test fallback detection catches analogical reasoning queries."""
        from src.vulcan.routing.query_router import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        
        # Query with analogical keywords
        query = "map the deep structure from domain s to domain t"
        result = analyzer._detect_query_type_fallback(query)
        
        assert result == 'analogical', \
            f"Fallback should detect analogical, got: {result}"


class TestContentPreservation:
    """Test content preservation prompt exists and is used."""
    
    def test_content_preservation_prompt_exists(self):
        """Verify VULCAN_CONTENT_PRESERVATION_PROMPT constant exists."""
        from src.vulcan.llm.hybrid_executor import HybridLLMExecutor
        
        assert hasattr(HybridLLMExecutor, 'VULCAN_CONTENT_PRESERVATION_PROMPT'), \
            "HybridLLMExecutor should have VULCAN_CONTENT_PRESERVATION_PROMPT constant"
        
        prompt = HybridLLMExecutor.VULCAN_CONTENT_PRESERVATION_PROMPT
        
        # Verify prompt contains key preservation rules
        assert "PRESERVE all factual claims" in prompt
        assert "NEVER replace VULCAN's answer" in prompt
        assert "NEVER say 'As an AI assistant'" in prompt


class TestWorldModelPhraseMatching:
    """Test WorldModel expanded phrase matching."""
    
    def test_world_model_phrase_list_expanded(self):
        """Verify WorldModel has expanded self-awareness phrase matching."""
        # We can't easily test the private method, but we can verify
        # the file contains the expanded phrases
        import os
        world_model_file = os.path.join(
            os.path.dirname(__file__), 
            '../src/vulcan/world_model/world_model_core.py'
        )
        
        if os.path.exists(world_model_file):
            with open(world_model_file, 'r') as f:
                content = f.read()
                
            # Check for expanded phrases
            assert '"take it"' in content or "'take it'" in content, \
                "WorldModel should have 'take it' phrase"
            assert '"choose it"' in content or "'choose it'" in content, \
                "WorldModel should have 'choose it' phrase"
            assert 'is_introspection' in content, \
                "WorldModel should mark introspection responses"


class TestTracingLogs:
    """Test that tracing logs are present in the code."""
    
    def test_trace_logs_in_chat_endpoint(self):
        """Verify VULCAN-TRACE logs exist in chat.py"""
        import os
        chat_file = os.path.join(
            os.path.dirname(__file__),
            '../src/vulcan/endpoints/chat.py'
        )
        
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                content = f.read()
            
            # Check for trace log markers
            assert '[VULCAN-TRACE] Query received:' in content
            assert '[VULCAN-TRACE] Classified as:' in content
            assert '[VULCAN-TRACE] Selected tools:' in content
            assert '[VULCAN-TRACE] WorldModel response' in content
            assert '[VULCAN-TRACE] Final response source:' in content


class TestAgentPoolContentFlags:
    """Test that agent_pool marks WorldModel responses for preservation."""
    
    def test_content_preservation_flags_in_agent_pool(self):
        """Verify agent_pool sets preserve_content flags."""
        import os
        agent_pool_file = os.path.join(
            os.path.dirname(__file__),
            '../src/vulcan/orchestrator/agent_pool.py'
        )
        
        if os.path.exists(agent_pool_file):
            with open(agent_pool_file, 'r') as f:
                content = f.read()
            
            # Check for content preservation logic
            assert 'preserve_content' in content, \
                "agent_pool should set preserve_content flag"
            assert 'no_openai_replacement' in content, \
                "agent_pool should set no_openai_replacement flag"
            assert 'Marked WorldModel response for content preservation' in content, \
                "agent_pool should log content preservation"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
