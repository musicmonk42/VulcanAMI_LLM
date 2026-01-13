"""
Tests for ToolSelector + QueryClassifier integration.

Verifies that:
1. ToolSelector uses LLM classification when confidence is high
2. ToolSelector falls back to existing methods when LLM confidence is low
3. ToolSelector handles LLM failures gracefully
4. SemanticToolMatcher can defer to QueryClassifier
5. Configuration flags work correctly
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test data classes
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MockQueryClassification:
    """Mock QueryClassification for testing."""
    category: str
    complexity: float
    suggested_tools: List[str]
    skip_reasoning: bool = False
    confidence: float = 1.0
    source: str = "llm"


class TestToolSelectorLLMIntegration:
    """Test ToolSelector integration with QueryClassifier."""
    
    @pytest.fixture
    def mock_tool_selector(self):
        """Create a mock ToolSelector instance."""
        # Import the required modules
        try:
            from vulcan.reasoning.selection.tool_selector import ToolSelector
            from vulcan.reasoning.selection import SelectionRequest, SelectionMode
            
            # Create a basic tool selector
            selector = ToolSelector(
                tool_names=["symbolic", "probabilistic", "causal", "analogical", "mathematical"],
                config={"safety_enabled": False}
            )
            return selector
        except ImportError as e:
            pytest.skip(f"ToolSelector not available: {e}")
    
    def test_uses_llm_classification_when_high_confidence(self, mock_tool_selector):
        """ToolSelector should use LLM-suggested tools when confidence >= 0.8."""
        # Mock the classify_query function to return high-confidence result
        mock_classification = MockQueryClassification(
            category="LOGICAL",
            complexity=0.7,
            suggested_tools=["symbolic"],
            confidence=0.9
        )
        
        with patch('vulcan.reasoning.selection.tool_selector.classify_query', return_value=mock_classification):
            with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                # Call _get_llm_classification
                safe_tools = ["symbolic", "probabilistic", "causal"]
                result = mock_tool_selector._get_llm_classification("Is A→B satisfiable?", safe_tools)
                
                assert result is not None
                assert "symbolic" in result
                assert len(result) <= 2  # CANDIDATE_MAX_COUNT
    
    def test_falls_back_when_low_confidence(self, mock_tool_selector):
        """ToolSelector should use fallback methods when LLM confidence < 0.8."""
        # Mock the classify_query function to return low-confidence result
        mock_classification = MockQueryClassification(
            category="UNKNOWN",
            complexity=0.5,
            suggested_tools=["general"],
            confidence=0.6  # Below threshold
        )
        
        with patch('vulcan.reasoning.selection.tool_selector.classify_query', return_value=mock_classification):
            with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                # Call _get_llm_classification
                safe_tools = ["symbolic", "probabilistic", "causal"]
                result = mock_tool_selector._get_llm_classification("What is X?", safe_tools)
                
                # Should return None because confidence is too low
                assert result is None
    
    def test_falls_back_when_llm_fails(self, mock_tool_selector):
        """ToolSelector should gracefully handle LLM failures."""
        # Mock classify_query to raise an exception
        with patch('vulcan.reasoning.selection.tool_selector.classify_query', side_effect=Exception("LLM error")):
            with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                # Call _get_llm_classification
                safe_tools = ["symbolic", "probabilistic", "causal"]
                result = mock_tool_selector._get_llm_classification("Test query", safe_tools)
                
                # Should return None on exception
                assert result is None
    
    def test_filters_unsafe_tools(self, mock_tool_selector):
        """LLM-suggested tools must be filtered through safety checks."""
        # Mock classification that suggests a tool not in safe_tools
        mock_classification = MockQueryClassification(
            category="LOGICAL",
            complexity=0.7,
            suggested_tools=["symbolic", "unsafe_tool", "probabilistic"],
            confidence=0.9
        )
        
        with patch('vulcan.reasoning.selection.tool_selector.classify_query', return_value=mock_classification):
            with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                # Only symbolic and probabilistic are safe
                safe_tools = ["symbolic", "probabilistic", "causal"]
                result = mock_tool_selector._get_llm_classification("Test query", safe_tools)
                
                # Should only return tools that are in safe_tools
                assert result is not None
                assert "unsafe_tool" not in result
                assert all(tool in safe_tools for tool in result)
    
    def test_respects_feature_flag(self, mock_tool_selector):
        """LLM classification should be skipped when flag is disabled."""
        with patch('vulcan.reasoning.selection.tool_selector.LLM_CLASSIFICATION_ENABLED', False):
            with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                # Call _get_llm_classification
                safe_tools = ["symbolic", "probabilistic", "causal"]
                result = mock_tool_selector._get_llm_classification("Test query", safe_tools)
                
                # Should return None because feature is disabled
                assert result is None
    
    def test_extract_query_text_from_string(self, mock_tool_selector):
        """Test extracting query text from a string problem."""
        query_str = "Is A→B satisfiable?"
        result = mock_tool_selector._extract_query_text(query_str)
        assert result == query_str
    
    def test_extract_query_text_from_dict(self, mock_tool_selector):
        """Test extracting query text from a dict problem."""
        problem_dict = {"query": "Test query", "context": "Some context"}
        result = mock_tool_selector._extract_query_text(problem_dict)
        assert result == "Test query"
    
    def test_extract_query_text_from_object(self, mock_tool_selector):
        """Test extracting query text from an object with attributes."""
        class Problem:
            def __init__(self):
                self.query = "Test query from object"
        
        problem_obj = Problem()
        result = mock_tool_selector._extract_query_text(problem_obj)
        assert result == "Test query from object"


class TestSemanticToolMatcherLLMIntegration:
    """Test SemanticToolMatcher integration with QueryClassifier."""
    
    @pytest.fixture
    def mock_semantic_matcher(self):
        """Create a mock SemanticToolMatcher instance."""
        try:
            from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
            
            matcher = SemanticToolMatcher()
            return matcher
        except ImportError as e:
            pytest.skip(f"SemanticToolMatcher not available: {e}")
    
    def test_uses_llm_when_enabled(self, mock_semantic_matcher):
        """SemanticToolMatcher should use LLM classification when enabled."""
        # Mock high-confidence LLM classification
        mock_classification = MockQueryClassification(
            category="LOGICAL",
            complexity=0.7,
            suggested_tools=["symbolic"],
            confidence=0.9
        )
        
        with patch('vulcan.routing.query_classifier.classify_query', return_value=mock_classification):
            result = mock_semantic_matcher.match_query(
                "Is A→B satisfiable?",
                available_tools=["symbolic", "probabilistic"],
                use_llm_classification=True
            )
            
            # Should return LLM-based result
            assert "symbolic" in result
            assert result["symbolic"].similarity_score >= 0.8
    
    def test_uses_embeddings_when_llm_disabled(self, mock_semantic_matcher):
        """SemanticToolMatcher should use embeddings when LLM disabled."""
        with patch('vulcan.routing.query_classifier.classify_query') as mock_classify:
            result = mock_semantic_matcher.match_query(
                "Test query",
                available_tools=["symbolic", "probabilistic"],
                use_llm_classification=False  # Explicitly disable
            )
            
            # Should NOT call classify_query
            mock_classify.assert_not_called()
            
            # Should return embedding-based results
            assert isinstance(result, dict)
    
    def test_converts_classification_to_semantic_match(self, mock_semantic_matcher):
        """LLM classification should be converted to SemanticMatch format."""
        # Mock high-confidence LLM classification
        mock_classification = MockQueryClassification(
            category="CAUSAL",
            complexity=0.6,
            suggested_tools=["causal", "probabilistic"],
            confidence=0.85
        )
        
        with patch('vulcan.routing.query_classifier.classify_query', return_value=mock_classification):
            result = mock_semantic_matcher.match_query(
                "What is the causal effect?",
                available_tools=["causal", "probabilistic", "symbolic"],
                use_llm_classification=True
            )
            
            # Should return properly formatted SemanticMatch objects
            assert "causal" in result
            assert hasattr(result["causal"], "tool_name")
            assert hasattr(result["causal"], "similarity_score")
            assert hasattr(result["causal"], "combined_score")
            
            # First tool should have higher score
            if "probabilistic" in result:
                assert result["causal"].combined_score >= result["probabilistic"].combined_score
    
    def test_llm_failure_falls_back_to_embeddings(self, mock_semantic_matcher):
        """When LLM fails, should fall back to embedding-based matching."""
        # Mock LLM to raise exception
        with patch('vulcan.routing.query_classifier.classify_query', side_effect=Exception("LLM error")):
            result = mock_semantic_matcher.match_query(
                "Test query",
                available_tools=["symbolic", "probabilistic"],
                use_llm_classification=True
            )
            
            # Should still return results (from embedding fallback)
            assert isinstance(result, dict)
            # At least one tool should be matched
            assert len(result) > 0


class TestConfigurationIntegration:
    """Test configuration settings integration."""
    
    def test_settings_have_correct_defaults(self):
        """Test that settings.py has correct default values."""
        try:
            from vulcan.settings import settings
            
            # Check LLM classification settings exist
            assert hasattr(settings, 'llm_first_classification')
            assert hasattr(settings, 'classification_llm_timeout')
            assert hasattr(settings, 'tool_selector_use_llm_classification')
            assert hasattr(settings, 'tool_selector_llm_confidence_threshold')
            
            # Check default values
            assert settings.tool_selector_llm_confidence_threshold == 0.8
            
        except ImportError:
            pytest.skip("Settings module not available")
    
    def test_tool_selector_constants_match_settings(self):
        """Test that tool_selector.py constants can be overridden by settings."""
        try:
            from vulcan.reasoning.selection import tool_selector
            
            # Check constants exist
            assert hasattr(tool_selector, 'LLM_CLASSIFICATION_ENABLED')
            assert hasattr(tool_selector, 'LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD')
            assert hasattr(tool_selector, 'LLM_CLASSIFICATION_TIMEOUT')
            
            # Check they have reasonable values
            assert isinstance(tool_selector.LLM_CLASSIFICATION_ENABLED, bool)
            assert 0.0 <= tool_selector.LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD <= 1.0
            assert tool_selector.LLM_CLASSIFICATION_TIMEOUT > 0
            
        except ImportError:
            pytest.skip("tool_selector module not available")


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_llm_classification(self):
        """Test the full pipeline from query to candidate generation."""
        try:
            from vulcan.reasoning.selection.tool_selector import ToolSelector
            from vulcan.reasoning.selection import SelectionRequest, SelectionMode
            import numpy as np
            
            # Create tool selector
            selector = ToolSelector(
                tool_names=["symbolic", "probabilistic", "causal"],
                config={"safety_enabled": False}
            )
            
            # Mock high-confidence classification
            mock_classification = MockQueryClassification(
                category="LOGICAL",
                complexity=0.7,
                suggested_tools=["symbolic"],
                confidence=0.9
            )
            
            # Create mock request
            request = SelectionRequest(
                problem="Is A→B satisfiable?",
                mode=SelectionMode.SINGLE,
                constraints={"time_budget_ms": 5000, "energy_budget_mj": 1000}
            )
            
            with patch('vulcan.reasoning.selection.tool_selector.classify_query', return_value=mock_classification):
                with patch('vulcan.reasoning.selection.tool_selector.QUERY_CLASSIFIER_AVAILABLE', True):
                    # Create mock features and prior
                    features = np.zeros(10)
                    
                    class MockPrior:
                        tool_probs = {"symbolic": 0.5, "probabilistic": 0.3, "causal": 0.2}
                    
                    prior = MockPrior()
                    safe_tools = ["symbolic", "probabilistic", "causal"]
                    
                    # Generate candidates
                    candidates = selector._generate_candidates(request, features, safe_tools, prior)
                    
                    # Should return LLM-based candidates
                    assert len(candidates) > 0
                    assert any(c["tool"] == "symbolic" for c in candidates)
                    # LLM-classified tools should have high quality scores
                    symbolic_candidate = next(c for c in candidates if c["tool"] == "symbolic")
                    assert symbolic_candidate.get("source") == "llm_classification"
                    
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
