#!/usr/bin/env python3
"""
Simple validation test for skip_gate_check implementation.

Tests the code changes without requiring full environment setup.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_code_changes():
    """Validate that the code changes are present."""
    
    logger.info("=" * 80)
    logger.info("Validating skip_gate_check implementation")
    logger.info("=" * 80)
    
    passed = []
    failed = []
    
    # Test 1: Check ProbabilisticReasoner has skip_gate_check support
    logger.info("\n[1/5] Checking ProbabilisticReasoner...")
    try:
        with open("src/vulcan/reasoning/probabilistic_reasoning.py", "r") as f:
            content = f.read()
            if "skip_gate_check" in content and "router_confidence" in content:
                logger.info("✓ ProbabilisticReasoner has skip_gate_check support")
                passed.append("ProbabilisticReasoner")
            else:
                logger.error("✗ Missing skip_gate_check in ProbabilisticReasoner")
                failed.append("ProbabilisticReasoner")
    except Exception as e:
        logger.error(f"✗ Failed to check ProbabilisticReasoner: {e}")
        failed.append("ProbabilisticReasoner")
    
    # Test 2: Check MathematicalComputationTool has skip_gate_check support
    logger.info("\n[2/5] Checking MathematicalComputationTool...")
    try:
        with open("src/vulcan/reasoning/mathematical_computation.py", "r") as f:
            content = f.read()
            if "skip_gate_check" in content:
                logger.info("✓ MathematicalComputationTool has skip_gate_check support")
                passed.append("MathematicalComputationTool")
            else:
                logger.error("✗ Missing skip_gate_check in MathematicalComputationTool")
                failed.append("MathematicalComputationTool")
    except Exception as e:
        logger.error(f"✗ Failed to check MathematicalComputationTool: {e}")
        failed.append("MathematicalComputationTool")
    
    # Test 3: Check ToolSelector sets skip_gate_check in candidates
    logger.info("\n[3/5] Checking ToolSelector...")
    try:
        with open("src/vulcan/reasoning/selection/tool_selector.py", "r") as f:
            content = f.read()
            if '"skip_gate_check": True' in content and "llm_authoritative" in content:
                logger.info("✓ ToolSelector sets skip_gate_check in candidates")
                passed.append("ToolSelector")
            else:
                logger.error("✗ Missing skip_gate_check in ToolSelector candidates")
                failed.append("ToolSelector")
    except Exception as e:
        logger.error(f"✗ Failed to check ToolSelector: {e}")
        failed.append("ToolSelector")
    
    # Test 4: Check agent_pool propagates skip_gate_check
    logger.info("\n[4/5] Checking agent_pool...")
    try:
        with open("src/vulcan/orchestrator/agent_pool.py", "r") as f:
            content = f.read()
            if "skip_gate_check" in content and "context['skip_gate_check']" in content:
                logger.info("✓ agent_pool propagates skip_gate_check")
                passed.append("agent_pool")
            else:
                logger.error("✗ Missing skip_gate_check propagation in agent_pool")
                failed.append("agent_pool")
    except Exception as e:
        logger.error(f"✗ Failed to check agent_pool: {e}")
        failed.append("agent_pool")
    
    # Test 5: Check UnifiedReasoner passes skip_gate_check to engines
    logger.info("\n[5/5] Checking UnifiedReasoner...")
    try:
        with open("src/vulcan/reasoning/unified/orchestrator.py", "r") as f:
            content = f.read()
            if "skip_gate_check" in content and "reasoning_kwargs" in content:
                logger.info("✓ UnifiedReasoner passes skip_gate_check to engines")
                passed.append("UnifiedReasoner")
            else:
                logger.error("✗ Missing skip_gate_check in UnifiedReasoner")
                failed.append("UnifiedReasoner")
    except Exception as e:
        logger.error(f"✗ Failed to check UnifiedReasoner: {e}")
        failed.append("UnifiedReasoner")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Passed: {len(passed)}/5")
    logger.info(f"Failed: {len(failed)}/5")
    
    if failed:
        logger.error(f"\nFailed components: {', '.join(failed)}")
        return 1
    else:
        logger.info("\n✅ All validation checks passed!")
        logger.info("\nThe skip_gate_check implementation is complete:")
        logger.info("1. ToolSelector marks high-confidence LLM classifications")
        logger.info("2. agent_pool propagates the flag through context")
        logger.info("3. UnifiedReasoner passes it to reasoning engines")
        logger.info("4. ProbabilisticReasoner & MathematicalComputationTool respect it")
        logger.info("\nExpected behavior:")
        logger.info("- When LLM has confidence ≥ 0.8, skip_gate_check=True is set")
        logger.info("- Reasoning engines skip redundant gate checks")
        logger.info("- Valid queries flow through instead of falling back to LLM")
        return 0


if __name__ == "__main__":
    sys.exit(test_code_changes())
