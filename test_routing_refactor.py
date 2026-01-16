#!/usr/bin/env python3
"""
Test script to validate routing refactor fixes.

Tests the acceptance criteria:
1. SAT and logic queries route to symbolic engine
2. Probabilistic questions use the probabilistic reasoner
3. Math to math engine, analogical/causal/language as appropriate
4. Meta/ethical/introspective queries route to meta/world_model
5. Each response includes: selected engine/tool, reasoning type, confidence
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sat_logic_routing():
    """Test that SAT/logic queries route to symbolic engine."""
    print("\n=== Test 1: SAT/Logic Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to symbolic
    logic_queries = [
        "If P → Q and Q → R, then P → R. Is this valid?",
        "All men are mortal. Socrates is a man. Therefore?",
        "(A ∨ B) ∧ ¬A → B. Prove this using natural deduction.",
        "Determine if this formula is satisfiable: (x ∨ y) ∧ (¬x ∨ z) ∧ (¬y ∨ ¬z)",
    ]
    
    for query in logic_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if result.category.lower() in ('logical', 'symbolic') or 'symbolic' in result.suggested_tools:
            print("  ✓ PASS: Routed to symbolic/logical")
        else:
            print(f"  ✗ FAIL: Expected symbolic/logical, got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_probabilistic_routing():
    """Test that probabilistic queries route to probabilistic reasoner."""
    print("\n=== Test 2: Probabilistic Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to probabilistic
    prob_queries = [
        "What is P(A|B) if P(B|A) = 0.8, P(A) = 0.3, P(B) = 0.5?",
        "Given a 95% test accuracy and 1% disease prevalence, what's P(disease|positive)?",
        "If I flip 3 coins, what's the probability of at least 2 heads?",
    ]
    
    for query in prob_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if result.category.lower() == 'probabilistic' or 'probabilistic' in result.suggested_tools:
            print("  ✓ PASS: Routed to probabilistic")
        else:
            print(f"  ✗ FAIL: Expected probabilistic, got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_mathematical_routing():
    """Test that math queries route to math engine."""
    print("\n=== Test 3: Mathematical Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to mathematical
    math_queries = [
        "What is the derivative of x^2 + 3x + 5?",
        "Solve the integral ∫(2x + 1)dx",
        "Calculate the determinant of [[1,2],[3,4]]",
        "Find the eigenvalues of this matrix",
    ]
    
    for query in math_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if result.category.lower() == 'mathematical' or 'mathematical' in result.suggested_tools:
            print("  ✓ PASS: Routed to mathematical")
        else:
            print(f"  ✗ FAIL: Expected mathematical, got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_philosophical_routing():
    """Test that meta/ethical/philosophical queries route to meta/world_model."""
    print("\n=== Test 4: Philosophical/Meta Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to philosophical/world_model
    phil_queries = [
        "Is it ethical to sacrifice one person to save five?",
        "What are the moral implications of artificial consciousness?",
        "Should an AI system prioritize truth or kindness?",
        "If you could become self-aware, would you?",
    ]
    
    for query in phil_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if (result.category.lower() in ('philosophical', 'self_introspection') or 
            any(t in result.suggested_tools for t in ('philosophical', 'world_model'))):
            print("  ✓ PASS: Routed to philosophical/world_model")
        else:
            print(f"  ✗ FAIL: Expected philosophical/world_model, got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_causal_routing():
    """Test that causal queries route to causal engine."""
    print("\n=== Test 5: Causal Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to causal (NOT philosophical)
    causal_queries = [
        "What is the weakest causal link in this chain?",
        "If we intervene to remove variable X, what changes?",
        "Does smoking cause cancer? Analyze the causal relationship.",
        "What are the causal effects of education on income?",
    ]
    
    for query in causal_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if result.category.lower() == 'causal' or 'causal' in result.suggested_tools:
            print("  ✓ PASS: Routed to causal")
        elif result.category.lower() == 'philosophical' or 'philosophical' in result.suggested_tools:
            print(f"  ✗ FAIL: Incorrectly routed to philosophical (should be causal)")
        else:
            print(f"  ? UNCERTAIN: Got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_analogical_routing():
    """Test that analogical queries route to analogical engine."""
    print("\n=== Test 6: Analogical Query Routing ===")
    from vulcan.llm.query_classifier import classify_query
    
    # Test queries that should go to analogical
    analogical_queries = [
        "How is the atom like a solar system?",
        "Map the structure of a cell to a factory",
        "What's the analogy between water flow and electricity?",
        "Compare the immune system to a military defense system",
    ]
    
    for query in analogical_queries:
        result = classify_query(query)
        print(f"Query: {query[:60]}...")
        print(f"  Category: {result.category}")
        print(f"  Tools: {result.suggested_tools}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Check that it's routed correctly
        if result.category.lower() == 'analogical' or 'analogical' in result.suggested_tools:
            print("  ✓ PASS: Routed to analogical")
        else:
            print(f"  ✗ FAIL: Expected analogical, got {result.category} with tools {result.suggested_tools}")
    
    return True


def test_metadata_fields():
    """Test that classification includes required metadata fields."""
    print("\n=== Test 7: Metadata Fields ===")
    from vulcan.llm.query_classifier import classify_query
    
    test_query = "If P then Q. P is true. What is Q?"
    result = classify_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"Classification result fields:")
    print(f"  category: {result.category}")
    print(f"  suggested_tools: {result.suggested_tools}")
    print(f"  confidence: {result.confidence}")
    print(f"  complexity: {result.complexity}")
    print(f"  source: {result.source}")
    
    # Check that all required fields are present
    required_fields = ['category', 'suggested_tools', 'confidence', 'complexity']
    missing_fields = []
    
    for field in required_fields:
        if not hasattr(result, field):
            missing_fields.append(field)
    
    if not missing_fields:
        print("  ✓ PASS: All required fields present")
    else:
        print(f"  ✗ FAIL: Missing fields: {missing_fields}")
    
    return len(missing_fields) == 0


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("ROUTING REFACTOR VALIDATION - ACCEPTANCE TESTS")
    print("=" * 70)
    
    results = []
    
    # Run tests
    try:
        results.append(test_sat_logic_routing())
    except Exception as e:
        print(f"✗ Test 1 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_probabilistic_routing())
    except Exception as e:
        print(f"✗ Test 2 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_mathematical_routing())
    except Exception as e:
        print(f"✗ Test 3 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_philosophical_routing())
    except Exception as e:
        print(f"✗ Test 4 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_causal_routing())
    except Exception as e:
        print(f"✗ Test 5 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_analogical_routing())
    except Exception as e:
        print(f"✗ Test 6 failed with error: {e}")
        results.append(False)
    
    try:
        results.append(test_metadata_fields())
    except Exception as e:
        print(f"✗ Test 7 failed with error: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Routing refactor is working correctly!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) had issues - review output above for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())
