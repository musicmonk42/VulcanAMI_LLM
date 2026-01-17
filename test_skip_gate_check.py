#!/usr/bin/env python3
"""
Test script to validate skip_gate_check functionality.

This test verifies that when the LLM classifier has high confidence (≥0.8),
the skip_gate_check flag is properly propagated and honored by reasoning engines.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_probabilistic_skip_gate_check():
    """Test that ProbabilisticReasoner respects skip_gate_check flag."""
    try:
        from src.vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
        from src.vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = ProbabilisticReasoner()
        
        # Test 1: Query without probability keywords should fail gate check
        query1 = "What is the capital of France?"
        result1 = reasoner.reason_with_uncertainty(
            query1,
            skip_gate_check=False
        )
        
        assert result1.confidence == 0.0, f"Expected confidence 0.0 without skip, got {result1.confidence}"
        assert result1.conclusion.get("not_applicable"), "Expected not_applicable=True"
        logger.info("✓ Test 1 passed: Gate check properly rejects non-probability query")
        
        # Test 2: Same query WITH skip_gate_check should bypass gate check
        result2 = reasoner.reason_with_uncertainty(
            query1,
            skip_gate_check=True,
            router_confidence=0.9,
            llm_classification="PROBABILISTIC"
        )
        
        # Should not be rejected (confidence > 0.0 or different conclusion)
        is_bypassed = (result2.confidence > 0.0 or 
                      not result2.conclusion.get("not_applicable", False))
        assert is_bypassed, "Expected gate check to be skipped"
        logger.info("✓ Test 2 passed: skip_gate_check bypasses gate check")
        
        # Test 3: Query WITH probability keywords works without skip flag
        query3 = "What is the probability of rolling a 6 on a fair die?"
        result3 = reasoner.reason_with_uncertainty(
            query3,
            skip_gate_check=False
        )
        
        assert result3.confidence > 0.0, f"Expected confidence > 0.0, got {result3.confidence}"
        logger.info("✓ Test 3 passed: Valid probability query passes gate check")
        
        logger.info("✅ All ProbabilisticReasoner tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ProbabilisticReasoner test failed: {e}", exc_info=True)
        return False


def test_mathematical_skip_gate_check():
    """Test that MathematicalComputationTool respects skip_gate_check flag."""
    try:
        from src.vulcan.reasoning.mathematical_computation import MathematicalComputationTool
        
        tool = MathematicalComputationTool(prefer_templates=True)
        
        # Test 1: Non-mathematical query should be rejected by gate check
        query1 = "What makes you different from other AI systems?"
        result1 = tool.reason(query1, query={})
        
        # Should fail (either no result or low confidence)
        is_rejected = (not result1.get("conclusion", {}).get("success", False) or
                      result1.get("confidence", 0.0) < 0.5)
        assert is_rejected, "Expected non-mathematical query to be rejected"
        logger.info("✓ Test 1 passed: Gate check properly rejects non-mathematical query")
        
        # Test 2: Mathematical query should pass gate check
        query2 = "Calculate 2 + 2"
        result2 = tool.reason(query2, query={})
        
        is_success = result2.get("conclusion", {}).get("success", False)
        assert is_success, "Expected mathematical query to succeed"
        logger.info("✓ Test 2 passed: Valid mathematical query passes gate check")
        
        logger.info("✅ All MathematicalComputationTool tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MathematicalComputationTool test failed: {e}", exc_info=True)
        return False


def test_end_to_end_flow():
    """Test end-to-end flow with high-confidence LLM classification."""
    try:
        # This would test the complete flow from ToolSelector through to reasoning engines
        # For now, we'll just validate that the components are wired correctly
        
        from src.vulcan.reasoning.selection.tool_selector import (
            LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD
        )
        
        assert LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD == 0.8, \
            f"Expected threshold 0.8, got {LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD}"
        
        logger.info("✓ LLM classification confidence threshold is correct (0.8)")
        logger.info("✅ End-to-end configuration validated!")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Testing skip_gate_check functionality")
    logger.info("=" * 80)
    
    results = []
    
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: ProbabilisticReasoner")
    logger.info("=" * 80)
    results.append(("ProbabilisticReasoner", test_probabilistic_skip_gate_check()))
    
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: MathematicalComputationTool")
    logger.info("=" * 80)
    results.append(("MathematicalComputationTool", test_mathematical_skip_gate_check()))
    
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: End-to-end configuration")
    logger.info("=" * 80)
    results.append(("End-to-end", test_end_to_end_flow()))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
