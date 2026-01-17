"""
Unit tests for collision bug fixes in reasoning architecture.

Tests for:
- Bug 1: Tool Priority Collision (The Hijack)
- Bug 2: Silent Success - Dictionary Conclusion Formatting (The Mute)

Note: Bug 3 (Philosophical Query Routing) tests removed as philosophical_router.py
has been deprecated and removed. Philosophical reasoning now routes to World Model.

Industry Standard: Comprehensive test coverage with edge cases
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vulcan.endpoints.chat_helpers import ConclusionFormatter


class TestConclusionFormatter:
    """Test Bug 2 fix: Dictionary conclusion handling."""
    
    def test_string_conclusion(self):
        """Test that string conclusions are returned as-is."""
        assert ConclusionFormatter.format("YES") == "YES"
        assert ConclusionFormatter.format("The answer is 42") == "The answer is 42"
    
    def test_dict_conclusion_with_result(self):
        """Test dict with 'result' key extraction."""
        conclusion = {"result": "NO", "proof": "contradiction found"}
        formatted = ConclusionFormatter.format(conclusion)
        assert formatted == "NO"
    
    def test_dict_conclusion_with_answer(self):
        """Test dict with 'answer' key extraction."""
        conclusion = {"answer": "TRUE", "steps": ["step1", "step2"]}
        formatted = ConclusionFormatter.format(conclusion)
        assert formatted == "TRUE"
    
    def test_dict_conclusion_with_conclusion(self):
        """Test dict with 'conclusion' key extraction."""
        conclusion = {"conclusion": "VALID", "confidence": 0.95}
        formatted = ConclusionFormatter.format(conclusion)
        assert formatted == "VALID"
    
    def test_dict_conclusion_full_json(self):
        """Test dict without standard keys returns JSON."""
        conclusion = {"satisfiable": False, "model": None}
        formatted = ConclusionFormatter.format(conclusion)
        assert "satisfiable" in formatted
        assert "false" in formatted.lower()
    
    def test_none_conclusion(self):
        """Test that None returns None."""
        assert ConclusionFormatter.format(None) is None
    
    def test_empty_string(self):
        """Test that empty/whitespace strings return None."""
        assert ConclusionFormatter.format("   ") is None
        assert ConclusionFormatter.format("") is None
    
    def test_list_conclusion_single_item(self):
        """Test list with single item."""
        conclusion = ["The answer is A"]
        formatted = ConclusionFormatter.format(conclusion)
        assert formatted == "The answer is A"
    
    def test_list_conclusion_multiple_items(self):
        """Test list with multiple items."""
        conclusion = ["First step", "Second step", "Third step"]
        formatted = ConclusionFormatter.format(conclusion)
        assert "1. First step" in formatted
        assert "2. Second step" in formatted
        assert "3. Third step" in formatted
    
    def test_empty_list(self):
        """Test that empty list returns None."""
        assert ConclusionFormatter.format([]) is None
    
    def test_nested_dict_in_conclusion(self):
        """Test nested dict extraction."""
        conclusion = {"data": {"result": "PASS"}}
        formatted = ConclusionFormatter.format(conclusion)
        # Should extract the nested structure
        assert formatted is not None
    
    def test_object_with_to_dict(self):
        """Test object with to_dict method."""
        class MockResult:
            def to_dict(self):
                return {"conclusion": "SUCCESS"}
        
        obj = MockResult()
        formatted = ConclusionFormatter.format(obj)
        assert formatted == "SUCCESS"
    
    def test_object_with_dict(self):
        """Test regular object with __dict__."""
        class MockResult:
            def __init__(self):
                self.result = "VALID"
                self.confidence = 0.9
        
        obj = MockResult()
        formatted = ConclusionFormatter.format(obj)
        assert formatted == "VALID"


class TestToolPriority:
    """Test Bug 1 fix: Tool priority ordering."""
    
    def test_causal_before_probabilistic(self):
        """Test causal has higher priority than probabilistic."""
        from vulcan.orchestrator.agent_pool import TOOL_SELECTION_PRIORITY_ORDER
        causal_idx = TOOL_SELECTION_PRIORITY_ORDER.index("causal")
        prob_idx = TOOL_SELECTION_PRIORITY_ORDER.index("probabilistic")
        assert causal_idx < prob_idx, "Causal must have higher priority than probabilistic"
    
    def test_analogical_before_symbolic(self):
        """Test analogical has higher priority than symbolic."""
        from vulcan.orchestrator.agent_pool import TOOL_SELECTION_PRIORITY_ORDER
        analog_idx = TOOL_SELECTION_PRIORITY_ORDER.index("analogical")
        symb_idx = TOOL_SELECTION_PRIORITY_ORDER.index("symbolic")
        assert analog_idx < symb_idx, "Analogical must have higher priority than symbolic"
    
    def test_mathematical_before_general(self):
        """Test mathematical has higher priority than general."""
        from vulcan.orchestrator.agent_pool import TOOL_SELECTION_PRIORITY_ORDER
        math_idx = TOOL_SELECTION_PRIORITY_ORDER.index("mathematical")
        gen_idx = TOOL_SELECTION_PRIORITY_ORDER.index("general")
        assert math_idx < gen_idx, "Mathematical must have higher priority than general"
    
    def test_philosophical_before_world_model(self):
        """Test philosophical has higher priority than world_model."""
        from vulcan.orchestrator.agent_pool import TOOL_SELECTION_PRIORITY_ORDER
        phil_idx = TOOL_SELECTION_PRIORITY_ORDER.index("philosophical")
        wm_idx = TOOL_SELECTION_PRIORITY_ORDER.index("world_model")
        assert phil_idx < wm_idx, "Philosophical must have higher priority than world_model"
    
    def test_multimodal_in_tier1(self):
        """Test multimodal is in tier 1 (specialized tools)."""
        from vulcan.orchestrator.agent_pool import TOOL_SELECTION_PRIORITY_ORDER
        multimodal_idx = TOOL_SELECTION_PRIORITY_ORDER.index("multimodal")
        symbolic_idx = TOOL_SELECTION_PRIORITY_ORDER.index("symbolic")
        assert multimodal_idx < symbolic_idx, "Multimodal should be checked before symbolic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
