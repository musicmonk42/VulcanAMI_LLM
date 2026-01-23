"""
Test suite for LLM Router markdown fence parsing edge cases.

This test suite specifically validates robust handling of markdown code fences
in JSON responses, following industry best practices for parser testing.

Tests cover:
- Standard markdown fences with language specifier
- Inline fences (no newlines)
- Fences with extra whitespace
- Mixed formats
- Malformed fences
- Edge cases that caused production bugs
"""

import pytest
from src.vulcan.routing.llm_router import LLMQueryRouter


class TestMarkdownFenceParsing:
    """Comprehensive tests for markdown code fence handling."""
    
    @pytest.fixture
    def router(self):
        """Create router without LLM for testing parser directly."""
        return LLMQueryRouter(llm_client=None)
    
    def test_standard_json_fence(self, router):
        """Standard ```json fence should be parsed."""
        response = """```json
{
  "destination": "reasoning_engine",
  "engine": "mathematical",
  "confidence": 0.95
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "mathematical"
    
    def test_plain_fence(self, router):
        """Plain ``` fence without language should be parsed."""
        response = """```
{
  "destination": "world_model",
  "confidence": 0.8
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "world_model"
    
    def test_inline_fence_no_newline(self, router):
        """Inline fence without newlines should be parsed."""
        response = '```{"destination": "skip", "confidence": 0.9}```'
        result = router._parse_json_response(response)
        assert result["destination"] == "skip"
    
    def test_inline_json_fence_no_newline(self, router):
        """Inline ```json fence without newlines should be parsed."""
        response = '```json{"destination": "reasoning_engine", "engine": "causal"}```'
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "causal"
    
    def test_fence_with_space_after_language(self, router):
        """Fence with space after language specifier should be parsed."""
        response = """```json 
{
  "destination": "reasoning_engine",
  "engine": "symbolic"
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "symbolic"
    
    def test_fence_missing_closing(self, router):
        """JSON with missing closing fence should still extract JSON."""
        response = """```json
{
  "destination": "reasoning_engine",
  "engine": "probabilistic",
  "confidence": 0.85
}"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "probabilistic"
    
    def test_fence_with_text_before(self, router):
        """Text before fence should not interfere with JSON extraction."""
        response = """Here is my classification:
```json
{
  "destination": "reasoning_engine",
  "engine": "mathematical"
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "mathematical"
    
    def test_fence_with_text_after(self, router):
        """Text after fence should not interfere with JSON extraction."""
        response = """```json
{
  "destination": "world_model",
  "confidence": 0.75
}
```
This is a philosophical query."""
        result = router._parse_json_response(response)
        assert result["destination"] == "world_model"
    
    def test_multiple_newlines_in_fence(self, router):
        """Multiple newlines inside fence should be handled."""
        response = """```json


{
  "destination": "skip",
  "confidence": 0.99
}


```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "skip"
    
    def test_tabs_and_spaces(self, router):
        """Tabs and spaces in fence should be handled."""
        response = """\t```json\t
\t{
\t  "destination": "reasoning_engine",
\t  "engine": "causal"
\t}
\t```\t"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
    
    def test_windows_line_endings(self, router):
        """Windows (CRLF) line endings should be handled."""
        response = "```json\r\n{\r\n  \"destination\": \"skip\"\r\n}\r\n```"
        result = router._parse_json_response(response)
        assert result["destination"] == "skip"
    
    def test_no_fence_pure_json(self, router):
        """Pure JSON without any fence should work (backward compat)."""
        response = """{
  "destination": "world_model",
  "confidence": 0.7
}"""
        result = router._parse_json_response(response)
        assert result["destination"] == "world_model"
    
    def test_nested_json_with_braces_in_strings(self, router):
        """JSON with braces in strings and fence should be parsed."""
        response = """```json
{
  "destination": "reasoning_engine",
  "engine": "symbolic",
  "reason": "Contains {variable} syntax"
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert "{variable}" in result["reason"]
    
    def test_escaped_quotes_in_fenced_json(self, router):
        """Escaped quotes in fenced JSON should be handled."""
        response = '''```json
{
  "destination": "world_model",
  "reason": "Query asks \\"What is consciousness?\\""
}
```'''
        result = router._parse_json_response(response)
        assert result["destination"] == "world_model"
        assert "consciousness" in result["reason"]
    
    def test_malformed_json_returns_defaults(self, router):
        """Malformed JSON should return safe defaults, not crash."""
        # Missing comma between "destination" and "engine" makes this invalid JSON
        response = """```json
{
  "destination": "reasoning_engine"
  "engine": "causal"
}
```"""
        result = router._parse_json_response(response)
        # Should return defaults, not crash
        assert "destination" in result
        assert result["destination"] == "world_model"  # default fallback
        assert result["engine"] is None  # default fallback
        assert result["confidence"] == 0.5  # low confidence for parse failure
    
    def test_empty_response(self, router):
        """Empty response should return safe defaults."""
        result = router._parse_json_response("")
        assert result["destination"] == "world_model"
    
    def test_only_whitespace(self, router):
        """Whitespace-only response should return safe defaults."""
        result = router._parse_json_response("   \n\t  \n   ")
        assert result["destination"] == "world_model"
    
    def test_only_fence_markers(self, router):
        """Only fence markers without JSON should return defaults."""
        result = router._parse_json_response("```json\n```")
        assert result["destination"] == "world_model"
    
    def test_production_bug_format(self, router):
        """
        Test the exact format from production logs that caused the bug.
        
        Production error: "JSON extraction failed: Expecting property name 
        enclosed in double quotes: line 1 column 2 (char 1)"
        
        This error indicated that the line-based fence stripping was leaving
        invalid characters at the start of the JSON string. The regex-based
        approach fixes this by properly extracting the JSON content.
        """
        # This is the format from the error logs showing "line 1 column 2"
        response = """```json
{
  "destination": "reasoning_engine",
  "engine": "mathematical",
  "confidence": 0.95,
  "reason": "Mathematical computation"
}
```"""
        result = router._parse_json_response(response)
        assert result["destination"] == "reasoning_engine"
        assert result["engine"] == "mathematical"
        assert result["confidence"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
