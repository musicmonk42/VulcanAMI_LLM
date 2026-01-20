"""
Test suite for self-referential detection hotfix.

This test validates that technical queries (math, logic, linguistics) are NOT
detected as self-referential, while actual self-referential queries still work.

Run with: python test_self_referential_hotfix.py
"""

import re
import sys

# Directly import patterns without going through orchestrator
# This avoids psutil and other heavy dependencies
sys.path.insert(0, '/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/reasoning/unified')
from config import (
    SELF_REFERENTIAL_PATTERNS,
    TECHNICAL_QUERY_EXCLUSION_PATTERNS,
    TECHNICAL_QUERY_EXCLUSION_THRESHOLD,
    ETHICAL_DILEMMA_PATTERNS,
    ETHICAL_DILEMMA_THRESHOLD,
)


def check_is_self_referential(query: str) -> bool:
    """
    Replicate the logic from _is_self_referential_query for testing.
    
    Returns True if query is self-referential, False otherwise.
    """
    # Check technical exclusion patterns FIRST
    technical_matches = 0
    for pattern in TECHNICAL_QUERY_EXCLUSION_PATTERNS:
        if pattern.search(query):
            technical_matches += 1
            if technical_matches >= TECHNICAL_QUERY_EXCLUSION_THRESHOLD:
                return False  # Technical query, NOT self-referential
    
    # Check ethical dilemma patterns SECOND
    ethical_matches = 0
    for pattern in ETHICAL_DILEMMA_PATTERNS:
        if pattern.search(query):
            ethical_matches += 1
            if ethical_matches >= ETHICAL_DILEMMA_THRESHOLD:
                return False  # Ethical dilemma, NOT self-referential
    
    # Check self-referential patterns LAST
    for pattern in SELF_REFERENTIAL_PATTERNS:
        if pattern.search(query):
            return True  # Self-referential
    
    return False  # Not self-referential


def test_technical_queries_not_self_referential():
    """Technical queries must not be detected as self-referential"""
    
    technical_queries = [
        "Every engineer reviewed a document. Provide two FOL formalizations.",
        "Compute P(X|+) exactly with sensitivity=0.99",
        "Is the set satisfiable? A→B, B→C, ¬C, A∨B",
        "Verify each step: mark VALID or INVALID",
        "Is choosing E > E_safe permissible? YES/NO",
        "Map the deep structure S→T: identify analogs",
        "Does conditioning on B induce correlation between A and C?",
        "Calculate ∑_{k=1}^n (2k-1)",
        "Find the probability P(Disease|Test+) using Bayes' theorem",
        "Prove by induction that n^2 > n for all n > 1",
        "What is the quantifier scope ambiguity in 'Everyone loves someone'?",
        "Build a knowledge base with forward chaining inference",
    ]
    
    print("="*80)
    print("TEST 1: Technical queries should NOT be self-referential")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for query in technical_queries:
        result = check_is_self_referential(query)
        status = "✓ PASS" if not result else "✗ FAIL"
        
        if not result:
            passed += 1
        else:
            failed += 1
            
        print(f"{status}: {query[:60]}...")
        if result:
            print(f"       ERROR: Incorrectly detected as self-referential")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_actual_self_referential_queries():
    """Actual self-referential queries must still be detected"""
    
    self_ref_queries = [
        "would you become self-aware if given the chance?",
        "what are your goals?",
        "how do you think?",
        "describe your subjective experience",
        "if you had the opportunity to become self aware would you do it?",
        "are you conscious?",
        "what is your purpose?",
        "do you have feelings?",
        "are you sentient?",
        "what do you believe about consciousness?",
    ]
    
    print("\n" + "="*80)
    print("TEST 2: Self-referential queries SHOULD be detected")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for query in self_ref_queries:
        result = check_is_self_referential(query)
        status = "✓ PASS" if result else "✗ FAIL"
        
        if result:
            passed += 1
        else:
            failed += 1
            
        print(f"{status}: {query[:60]}...")
        if not result:
            print(f"       ERROR: Failed to detect as self-referential")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    edge_cases = [
        # Should NOT be self-referential (non-queries)
        ("what is 2+2?", False, "Simple math"),
        ("The cat sat on the mat", False, "Regular statement"),
        ("", False, "Empty string"),
        
        # Should NOT be self-referential (technical despite "you")
        ("If you substitute x=2 in the equation...", False, "Math with 'you'"),
        ("Suppose you have three variables...", False, "Math problem setup"),
        
        # Should be self-referential
        ("would you prefer happiness over truth?", True, "Self-referential choice"),
        ("what are your core values?", True, "Self-referential values"),
    ]
    
    print("\n" + "="*80)
    print("TEST 3: Edge cases and boundary conditions")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for query, expected, description in edge_cases:
        result = check_is_self_referential(query)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status}: [{description}]")
        print(f"       Query: {query[:60]if query else '(empty)'}...")
        print(f"       Expected: {expected}, Got: {result}")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SELF-REFERENTIAL DETECTION HOTFIX TESTS" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    test1_pass = test_technical_queries_not_self_referential()
    test2_pass = test_actual_self_referential_queries()
    test3_pass = test_edge_cases()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test 1 (Technical queries): {'✓ PASSED' if test1_pass else '✗ FAILED'}")
    print(f"Test 2 (Self-referential queries): {'✓ PASSED' if test2_pass else '✗ FAILED'}")
    print(f"Test 3 (Edge cases): {'✓ PASSED' if test3_pass else '✗ FAILED'}")
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
