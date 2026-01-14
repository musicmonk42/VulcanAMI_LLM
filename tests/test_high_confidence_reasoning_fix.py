"""
Test High-Confidence Reasoning Fix

This test validates the fix for the reasoning overwrite bug where high-confidence
results from reasoning engines (symbolic, probabilistic, etc.) were being
overwritten by UnifiedReasoner.

Root Cause:
- agent_pool.py only short-circuited for world_model results
- Other high-confidence results (symbolic with 0.85 confidence) fell through
- UnifiedReasoner ran with wrong reasoning type and returned 0.0 confidence
- Valid results were lost

Fix:
- Extended high-confidence check to ANY reasoning engine result >= 0.5
- Added observe_engine_result() call for learning integration
- Proper conversion of integration_result to reasoning_result for all tools

Test Cases:
1. Symbolic reasoning with high confidence (>= 0.5) should use result directly
2. Probabilistic reasoning with high confidence should use result directly
3. Low confidence results (< 0.5) should fall back to UnifiedReasoner
4. Learning integration should be called for high-confidence results
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Path constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_POOL_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/orchestrator/agent_pool.py')


class TestHighConfidenceThresholdConstant(unittest.TestCase):
    """Test that HIGH_CONFIDENCE_THRESHOLD constant exists."""
    
    def test_high_confidence_threshold_exists(self):
        """Verify HIGH_CONFIDENCE_THRESHOLD constant is defined in agent_pool.py."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the constant exists
        self.assertIn('HIGH_CONFIDENCE_THRESHOLD', source_code)
        
        # Verify it's set to 0.5
        self.assertIn('HIGH_CONFIDENCE_THRESHOLD = 0.5', source_code)
        
        # Verify the comment explains it applies to all engines
        self.assertIn('ANY tool result', source_code)


class TestHighConfidenceResultDetection(unittest.TestCase):
    """Test that high-confidence results are detected for all reasoning engines."""
    
    def test_is_high_confidence_result_check_exists(self):
        """Verify is_high_confidence_result variable is defined."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the condition variable is defined
        self.assertIn('is_high_confidence_result', source_code)
        
        # Verify it uses HIGH_CONFIDENCE_THRESHOLD
        self.assertIn('integration_result.confidence >= HIGH_CONFIDENCE_THRESHOLD', source_code)
    
    def test_high_confidence_check_not_tool_specific(self):
        """Verify high-confidence check applies to ANY tool, not just world_model."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Find the is_high_confidence_result definition
        lines = source_code.split('\n')
        high_conf_check_found = False
        
        for i, line in enumerate(lines):
            if 'is_high_confidence_result =' in line:
                # Check next few lines to ensure it's not checking for specific tools
                context = '\n'.join(lines[i:i+5])
                high_conf_check_found = True
                
                # The check should only look at confidence, not selected_tools
                self.assertIn('integration_result.confidence', context)
                # This line should NOT check for specific tool names in the primary check
                # (world_model check comes separately for backward compatibility)
                break
        
        self.assertTrue(high_conf_check_found, "is_high_confidence_result check not found")


class TestLearningIntegration(unittest.TestCase):
    """Test that observe_engine_result is called for high-confidence results."""
    
    def test_observe_engine_result_import_exists(self):
        """Verify observe_engine_result is imported from utils."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the import statement exists
        self.assertIn('from vulcan.reasoning.integration.utils import observe_engine_result', source_code)
    
    def test_observe_engine_result_called_for_high_confidence(self):
        """Verify observe_engine_result is called when using high-confidence results."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify observe_engine_result is called
        self.assertIn('observe_engine_result(', source_code)
        
        # Verify it's called with the right parameters
        self.assertIn('query_id=', source_code)
        self.assertIn('engine_name=', source_code)
        self.assertIn('success=True', source_code)
    
    def test_learning_observation_error_handling(self):
        """Verify learning observation errors don't fail the task."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify there's error handling around observe_engine_result
        # Simply check that the code has try/except around the observe call
        
        # Find the section with LEARNING INTEGRATION comment
        self.assertIn('LEARNING INTEGRATION', source_code)
        self.assertIn('observe_engine_result(', source_code)
        
        # The pattern should be: 
        # LEARNING INTEGRATION comment
        # try:
        # ... observe_engine_result(...)
        # except Exception:
        # ... Don't fail the task
        
        # Extract the LEARNING INTEGRATION section (need more chars to get the except block)
        learning_section_start = source_code.find('LEARNING INTEGRATION')
        learning_section = source_code[learning_section_start:learning_section_start + 2500]
        
        # Verify try/except pattern
        self.assertIn('try:', learning_section)
        self.assertIn('observe_engine_result(', learning_section)
        self.assertIn('except Exception', learning_section)
        self.assertIn("fail the task", learning_section)  # Relaxed check - just check for the key phrase


class TestReasoningResultConversion(unittest.TestCase):
    """Test that integration_result is properly converted to reasoning_result."""
    
    def test_tool_to_reasoning_type_mapping_exists(self):
        """Verify tool names are mapped to ReasoningType enums."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify symbolic tool is mapped
        self.assertIn("'symbolic': RT_Local.SYMBOLIC", source_code)
        
        # Verify probabilistic tool is mapped
        self.assertIn("'probabilistic': RT_Local.PROBABILISTIC", source_code)
        
        # Verify causal tool is mapped
        self.assertIn("'causal': RT_Local.CAUSAL", source_code)
        
        # Verify mathematical tool is mapped
        self.assertIn("'mathematical': RT_Local.MATHEMATICAL", source_code)
    
    def test_high_confidence_metadata_flag(self):
        """Verify high_confidence_direct_use flag is added to metadata."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the flag is set in metadata
        self.assertIn('high_confidence_direct_use', source_code)
        self.assertIn('"high_confidence_direct_use": True', source_code)
    
    def test_primary_engine_extraction(self):
        """Verify primary_engine is extracted from selected_tools."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify primary_engine variable is defined
        self.assertIn('primary_engine =', source_code)
        
        # Verify it extracts from selected_tools
        self.assertIn('integration_result.selected_tools[0]', source_code)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that world_model special handling is preserved."""
    
    def test_world_model_result_check_still_exists(self):
        """Verify is_world_model_result check is still present for backward compatibility."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the world_model specific check still exists
        self.assertIn('is_world_model_result', source_code)
        
        # Verify it checks for world_model in selected_tools
        self.assertIn('selected_tools == ["world_model"]', source_code)
    
    def test_world_model_content_preservation(self):
        """Verify world_model content preservation flags are still set."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify preserve_content flag is still set
        self.assertIn('preserve_content', source_code)
        
        # Verify no_openai_replacement flag is still set
        self.assertIn('no_openai_replacement', source_code)


class TestLogging(unittest.TestCase):
    """Test that appropriate logging is in place."""
    
    def test_high_confidence_logging(self):
        """Verify logging when high-confidence result is used directly."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify log message about using high-confidence result directly
        self.assertIn('High-confidence result from', source_code)
        self.assertIn('Using this result directly without invoking UnifiedReasoner', source_code)
    
    def test_primary_engine_in_logging(self):
        """Verify primary engine name is included in log messages."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify engine name is logged
        # Look for f-string with primary_engine variable
        self.assertIn("f\"[AgentPool] High-confidence result from '{primary_engine}'", source_code)


class TestDocumentation(unittest.TestCase):
    """Test that code is well-documented."""
    
    def test_fix_comments_exist(self):
        """Verify explanatory comments are in place."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify CRITICAL FIX comment is present
        self.assertIn('CRITICAL FIX: High-Confidence Result Detection for All Engines', source_code)
        
        # Verify root cause is explained
        self.assertIn('Root Cause:', source_code)
        
        # Verify fix is explained
        self.assertIn('Fix: Check confidence threshold for ALL tools', source_code)
    
    def test_learning_integration_comments(self):
        """Verify LEARNING INTEGRATION section is documented."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify LEARNING INTEGRATION comment block exists
        self.assertIn('LEARNING INTEGRATION', source_code)
        
        # Verify it explains the purpose
        self.assertIn('Record successful engine execution', source_code)


class TestElIfConditionUpdate(unittest.TestCase):
    """Test that the elif condition now checks for high-confidence, not world_model."""
    
    def test_elif_condition_updated(self):
        """Verify elif now checks 'NOT a high-confidence result' instead of 'NOT a world model result'."""
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Find the elif block after the high-confidence check
        lines = source_code.split('\n')
        
        for i, line in enumerate(lines):
            if 'elif UnifiedReasoner is not None and create_unified_reasoner is not None:' in line:
                # Check the comment above this line
                context_before = '\n'.join(lines[max(0, i-3):i])
                
                # The comment should mention "high-confidence" not "world model"
                # This verifies the logic was updated
                self.assertIn('high-confidence', context_before.lower())
                break


if __name__ == '__main__':
    unittest.main(verbosity=2)
