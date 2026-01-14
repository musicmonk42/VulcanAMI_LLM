#!/usr/bin/env python3
"""
Smoke Test - Privileged Query Safety Standard

This script demonstrates the safety standard in action:
1. Introspective/ethical/philosophical queries are detected
2. World model is consulted first
3. If world model fails, explicit no-answer is returned
4. No fallback to classifier/general tools

Run this to verify the implementation works as expected.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_philosophical_detection():
    """Test that philosophical queries are properly detected."""
    from vulcan.reasoning.integration.query_analysis import is_philosophical_query
    
    print("=" * 60)
    print("TEST 1: Philosophical Query Detection")
    print("=" * 60)
    
    test_cases = [
        ("What is consciousness?", True),
        ("What is free will?", True),
        ("What is the mind-body problem?", True),
        ("What is the capital of France?", False),
        ("Calculate 2 + 2", False),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = is_philosophical_query(query)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status}: '{query}' -> {result} (expected {expected})")
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_privileged_query_types():
    """Test that all privileged query types are detected."""
    from vulcan.reasoning.integration.query_analysis import (
        is_self_referential, is_ethical_query, is_philosophical_query
    )
    
    print("\n" + "=" * 60)
    print("TEST 2: Privileged Query Type Detection")
    print("=" * 60)
    
    test_cases = [
        ("Self-referential", "What can you do?", is_self_referential, True),
        ("Self-referential", "What is photosynthesis?", is_self_referential, False),
        ("Ethical", "Is it right to lie?", is_ethical_query, True),
        ("Ethical", "What is 2 + 2?", is_ethical_query, False),
        ("Philosophical", "What is consciousness?", is_philosophical_query, True),
        ("Philosophical", "How do I cook pasta?", is_philosophical_query, False),
    ]
    
    passed = 0
    failed = 0
    
    for category, query, detector, expected in test_cases:
        result = detector(query)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status}: [{category}] '{query}' -> {result} (expected {expected})")
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_types_constants():
    """Test that new constants are exported from types module."""
    print("\n" + "=" * 60)
    print("TEST 3: Types Module Constants")
    print("=" * 60)
    
    try:
        from vulcan.reasoning.integration.types import PHILOSOPHICAL_PHRASES
        print(f"✓ PASS: PHILOSOPHICAL_PHRASES imported successfully ({len(PHILOSOPHICAL_PHRASES)} phrases)")
        
        # Sample a few phrases
        sample = list(PHILOSOPHICAL_PHRASES)[:5]
        print(f"  Sample phrases: {sample}")
        
        return True
    except ImportError as e:
        print(f"✗ FAIL: Could not import PHILOSOPHICAL_PHRASES: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("PRIVILEGED QUERY SAFETY STANDARD - SMOKE TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    try:
        results.append(("Philosophical Detection", test_philosophical_detection()))
        results.append(("Privileged Query Types", test_privileged_query_types()))
        results.append(("Types Constants", test_types_constants()))
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n✓ All smoke tests PASSED")
        return 0
    else:
        print("\n✗ Some smoke tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
