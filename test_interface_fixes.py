#!/usr/bin/env python3
"""
Test script to verify interface fixes for AnalogicalReasoner and MultimodalReasoner.

This validates that the interface methods required by ToolWrappers are present and working.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_analogical_reasoner_interface():
    """Test that AnalogicalReasoner has the required interface methods."""
    logger.info("=" * 70)
    logger.info("Testing AnalogicalReasoner interface...")
    logger.info("=" * 70)
    
    try:
        from src.vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
        
        # Create reasoner instance
        reasoner = AnalogicalReasoner(enable_caching=False, enable_learning=False)
        logger.info("✓ AnalogicalReasoner instantiated successfully")
        
        # Test find_analogies method exists
        assert hasattr(reasoner, 'find_analogies'), "Missing find_analogies() method"
        logger.info("✓ find_analogies() method exists")
        
        # Test reason method exists
        assert hasattr(reasoner, 'reason'), "Missing reason() method"
        logger.info("✓ reason() method exists")
        
        # Test that find_analogies is callable
        result = reasoner.find_analogies("test query", k=1)
        assert isinstance(result, dict), "find_analogies should return a dict"
        assert "analogies" in result, "Result should have 'analogies' key"
        assert "query" in result, "Result should have 'query' key"
        logger.info("✓ find_analogies() is callable and returns correct format")
        
        # Test that reason is callable
        result = reasoner.reason("test problem")
        assert isinstance(result, dict), "reason should return a dict"
        assert "found" in result, "Result should have 'found' key"
        logger.info("✓ reason() is callable and returns correct format")
        
        logger.info("✅ AnalogicalReasoner interface test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ AnalogicalReasoner interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_reasoner_interface():
    """Test that MultimodalReasoner has the required interface methods."""
    logger.info("=" * 70)
    logger.info("Testing MultimodalReasoner interface...")
    logger.info("=" * 70)
    
    try:
        from src.vulcan.reasoning.multimodal_reasoning import MultimodalReasoner
        
        # Create reasoner instance
        reasoner = MultimodalReasoner()
        logger.info("✓ MultimodalReasoner instantiated successfully")
        
        # Test process method exists
        assert hasattr(reasoner, 'process'), "Missing process() method"
        logger.info("✓ process() method exists")
        
        # Test reason method exists
        assert hasattr(reasoner, 'reason'), "Missing reason() method"
        logger.info("✓ reason() method exists")
        
        # Test that process is callable
        result = reasoner.process({"text": "test input"})
        assert isinstance(result, dict), "process should return a dict"
        assert "result" in result, "Result should have 'result' key"
        assert "success" in result, "Result should have 'success' key"
        logger.info("✓ process() is callable and returns correct format")
        
        # Test that reason is callable
        result = reasoner.reason({"text": "test problem"})
        assert isinstance(result, dict), "reason should return a dict"
        logger.info("✓ reason() is callable and returns correct format")
        
        logger.info("✅ MultimodalReasoner interface test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MultimodalReasoner interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_wrapper_initialization():
    """Test that ToolWrappers can initialize with the reasoners."""
    logger.info("=" * 70)
    logger.info("Testing ToolWrapper initialization...")
    logger.info("=" * 70)
    
    try:
        from src.vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
        from src.vulcan.reasoning.multimodal_reasoning import MultimodalReasoner
        from src.vulcan.reasoning.selection.tool_selector import (
            AnalogicalToolWrapper,
            MultimodalToolWrapper
        )
        
        # Test AnalogicalToolWrapper
        logger.info("Testing AnalogicalToolWrapper...")
        analogical_reasoner = AnalogicalReasoner(enable_caching=False, enable_learning=False)
        analogical_wrapper = AnalogicalToolWrapper(analogical_reasoner)
        logger.info("✓ AnalogicalToolWrapper instantiated successfully")
        
        # Test that it can reason
        result = analogical_wrapper.reason("test problem")
        assert isinstance(result, dict), "Wrapper should return dict"
        logger.info("✓ AnalogicalToolWrapper.reason() works")
        
        # Test MultimodalToolWrapper
        logger.info("Testing MultimodalToolWrapper...")
        multimodal_reasoner = MultimodalReasoner()
        multimodal_wrapper = MultimodalToolWrapper(multimodal_reasoner)
        logger.info("✓ MultimodalToolWrapper instantiated successfully")
        
        # Test that it can reason
        result = multimodal_wrapper.reason("test problem")
        assert isinstance(result, dict), "Wrapper should return dict"
        logger.info("✓ MultimodalToolWrapper.reason() works")
        
        logger.info("✅ ToolWrapper initialization test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ToolWrapper initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all interface tests."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("INTERFACE FIX VALIDATION TEST SUITE")
    logger.info("=" * 70)
    logger.info("\n")
    
    all_passed = True
    
    # Test AnalogicalReasoner interface
    if not test_analogical_reasoner_interface():
        all_passed = False
    logger.info("\n")
    
    # Test MultimodalReasoner interface
    if not test_multimodal_reasoner_interface():
        all_passed = False
    logger.info("\n")
    
    # Test ToolWrapper initialization
    if not test_tool_wrapper_initialization():
        all_passed = False
    logger.info("\n")
    
    # Summary
    logger.info("=" * 70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        logger.info("=" * 70)
        return 0
    else:
        logger.info("❌ SOME TESTS FAILED")
        logger.info("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
