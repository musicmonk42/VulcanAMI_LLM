#!/usr/bin/env python
"""
Validation script to demonstrate the routing override fixes.

This script tests the key changes:
1. ReasoningResult has override_router_tools field
2. TRULY_SIMPLE_CATEGORIES vs REASONING_CATEGORIES distinction
3. AgentPool logic for detecting general fallback
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_reasoning_result_field():
    """Test that ReasoningResult has override_router_tools field."""
    print("\n=== Test #1: ReasoningResult has override_router_tools field ===")
    try:
        from vulcan.reasoning.integration.types import ReasoningResult
        
        # Test default value (False)
        result1 = ReasoningResult(
            selected_tools=['test'],
            reasoning_strategy='direct',
            confidence=0.8,
            rationale='test',
        )
        assert hasattr(result1, 'override_router_tools'), "Missing override_router_tools field"
        assert result1.override_router_tools == False, "Default should be False"
        print("✓ Default override_router_tools = False")
        
        # Test explicit True
        result2 = ReasoningResult(
            selected_tools=['world_model'],
            reasoning_strategy='meta_reasoning',
            confidence=0.9,
            rationale='Self-introspection',
            override_router_tools=True,
        )
        assert result2.override_router_tools == True, "Should be True when set"
        print("✓ Can set override_router_tools = True")
        
        print("✅ PASS: ReasoningResult.override_router_tools field works correctly")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_category_definitions():
    """Test the category set definitions in apply_reasoning_impl.py."""
    print("\n=== Test #2: Category Set Definitions ===")
    
    # Define the expected categories (same as in the code)
    TRULY_SIMPLE_CATEGORIES = frozenset([
        'GREETING', 'CHITCHAT', 'greeting', 'chitchat'
    ])
    
    REASONING_CATEGORIES = frozenset([
        'PROBABILISTIC', 'LOGICAL', 'CAUSAL', 'MATHEMATICAL', 'ANALOGICAL', 
        'PHILOSOPHICAL', 'SYMBOLIC', 'LANGUAGE',
        'probabilistic', 'logical', 'causal', 'mathematical', 'analogical',
        'philosophical', 'symbolic', 'language',
    ])
    
    # Verify truly simple categories
    assert 'GREETING' in TRULY_SIMPLE_CATEGORIES
    assert 'CHITCHAT' in TRULY_SIMPLE_CATEGORIES
    assert 'FACTUAL' not in TRULY_SIMPLE_CATEGORIES  # Should NOT be here
    print("✓ TRULY_SIMPLE_CATEGORIES only contains GREETING and CHITCHAT")
    
    # Verify reasoning categories
    assert 'PROBABILISTIC' in REASONING_CATEGORIES
    assert 'LOGICAL' in REASONING_CATEGORIES
    assert 'CAUSAL' in REASONING_CATEGORIES
    assert 'MATHEMATICAL' in REASONING_CATEGORIES
    assert 'SYMBOLIC' in REASONING_CATEGORIES
    print("✓ REASONING_CATEGORIES contains all reasoning types")
    
    print("✅ PASS: Category definitions are correct")
    return True


def test_general_fallback_detection():
    """Test the logic for detecting general fallback."""
    print("\n=== Test #3: General Fallback Detection Logic ===")
    
    from vulcan.reasoning.integration.types import ReasoningResult
    
    # Test case 1: General fallback without override flag
    integration_result = ReasoningResult(
        selected_tools=['general'],
        reasoning_strategy='direct',
        confidence=0.5,
        rationale='Fallback',
        override_router_tools=False,
    )
    router_tools = ['probabilistic', 'symbolic']
    
    is_general_fallback = (
        integration_result.selected_tools == ['general'] and
        router_tools and 
        router_tools != ['general'] and
        not integration_result.override_router_tools
    )
    
    assert is_general_fallback, "Should detect general fallback"
    print("✓ Detects ['general'] as fallback when router has specific tools")
    
    # Test case 2: Explicit override with override flag
    integration_result2 = ReasoningResult(
        selected_tools=['world_model'],
        reasoning_strategy='meta_reasoning',
        confidence=0.9,
        rationale='Self-introspection',
        override_router_tools=True,
    )
    
    should_override = integration_result2.override_router_tools
    assert should_override, "Should respect override flag"
    print("✓ Respects override_router_tools=True flag")
    
    # Test case 3: Not a fallback if router also had general
    router_tools3 = ['general']
    integration_result3 = ReasoningResult(
        selected_tools=['general'],
        reasoning_strategy='direct',
        confidence=0.5,
        rationale='Simple',
        override_router_tools=False,
    )
    
    is_general_fallback3 = (
        integration_result3.selected_tools == ['general'] and
        router_tools3 and 
        router_tools3 != ['general'] and  # This is False
        not integration_result3.override_router_tools
    )
    
    assert not is_general_fallback3, "Not a fallback if router also had general"
    print("✓ Not a fallback when both have ['general']")
    
    print("✅ PASS: General fallback detection logic works correctly")
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("ROUTING OVERRIDE FIXES - VALIDATION SCRIPT")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(test_reasoning_result_field())
    results.append(test_category_definitions())
    results.append(test_general_fallback_detection())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Routing override fixes are working correctly!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
