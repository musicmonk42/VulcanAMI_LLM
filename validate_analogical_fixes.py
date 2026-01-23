#!/usr/bin/env python3
"""
Validation script for analogical routing fixes.

This script validates the fixes without requiring the full test environment.
"""

import sys
from pathlib import Path

# Get repository root
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.vulcan.routing.llm_router import (
    LLMQueryRouter,
    ANALOGICAL_KEYWORDS,
    CAUSAL_KEYWORDS,
)
from src.vulcan.reasoning.symbolic.reasoner import SymbolicReasoner


def test_analogical_keywords():
    """Test that all required keywords are present."""
    print("=" * 80)
    print("TEST 1: Checking Analogical Keywords")
    print("=" * 80)
    
    required_keywords = [
        "structure mapping",
        "domain mapping", 
        "map the",
        "deep structure",
        "source domain",
        "target domain",
    ]
    
    for keyword in required_keywords:
        if keyword in ANALOGICAL_KEYWORDS:
            print(f"✓ '{keyword}' found in ANALOGICAL_KEYWORDS")
        else:
            print(f"✗ '{keyword}' NOT FOUND in ANALOGICAL_KEYWORDS")
            return False
    
    # Check domain mapping notation
    if "s→t" in ANALOGICAL_KEYWORDS or "s->t" in ANALOGICAL_KEYWORDS:
        print(f"✓ Domain mapping notation (S→T/S->T) found")
    else:
        print(f"✗ Domain mapping notation NOT FOUND")
        return False
    
    print("\n✓ All required keywords present\n")
    return True


def test_analogical_routing():
    """Test that analogical queries route correctly."""
    print("=" * 80)
    print("TEST 2: Analogical Query Routing")
    print("=" * 80)
    
    # Create router without LLM (fallback only)
    router = LLMQueryRouter(llm_client=None)
    
    test_queries = [
        ("Structure mapping (not surface similarity)\nDomain S: distributed system\nDomain T: organism\nTask: Map the deep structure S→T", "analogical"),
        ("Perform domain mapping from economics to ecology", "analogical"),
        ("Map the relationships between source domain and target domain", "analogical"),
        ("Identify the deep structure common to both systems", "analogical"),
        ("Confounding vs causation (Pearl-style)\nPeople who take supplement S have lower disease D", "causal"),
        ("Compute exactly: ∑(k=1 to n) (2k−1)", "probabilistic or mathematical"),
    ]
    
    all_passed = True
    for query, expected_engine in test_queries:
        decision = router.route(query)
        
        # For mathematical queries, accept either probabilistic or mathematical
        if "or" in expected_engine:
            engines = expected_engine.split(" or ")
            if decision.engine in engines:
                print(f"✓ Query routed to {decision.engine} (expected: {expected_engine})")
                print(f"  Query: {query[:60]}...")
            else:
                print(f"✗ Query routed to {decision.engine} (expected: {expected_engine})")
                print(f"  Query: {query[:60]}...")
                all_passed = False
        else:
            if decision.engine == expected_engine:
                print(f"✓ Query routed to {decision.engine}")
                print(f"  Query: {query[:60]}...")
            else:
                print(f"✗ Query routed to {decision.engine} (expected: {expected_engine})")
                print(f"  Query: {query[:60]}...")
                all_passed = False
        
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reason: {decision.reason}")
        print()
    
    if all_passed:
        print("✓ All routing tests passed\n")
    else:
        print("✗ Some routing tests failed\n")
    
    return all_passed


def test_symbolic_rejection():
    """Test that symbolic reasoner rejects analogical queries."""
    print("=" * 80)
    print("TEST 3: Symbolic Reasoner Rejection of Analogical Queries")
    print("=" * 80)
    
    reasoner = SymbolicReasoner()
    
    test_queries = [
        "Structure mapping from domain S to domain T",
        "Perform domain mapping between systems",
        "Map the deep structure between source and target",
    ]
    
    all_passed = True
    for query in test_queries:
        is_symbolic = reasoner.is_symbolic_query(query)
        
        if not is_symbolic:
            print(f"✓ Query correctly rejected as non-symbolic")
            print(f"  Query: {query[:60]}...")
        else:
            print(f"✗ Query incorrectly accepted as symbolic")
            print(f"  Query: {query[:60]}...")
            all_passed = False
        
        # Check applicability
        result = reasoner.check_applicability(query)
        if not result['applicable'] and result.get('suggestion') == 'analogical':
            print(f"✓ check_applicability suggests 'analogical'")
        elif not result['applicable']:
            print(f"⚠ check_applicability rejected but didn't suggest 'analogical'")
            print(f"  Suggestion: {result.get('suggestion', 'none')}")
        else:
            print(f"✗ check_applicability incorrectly said applicable")
            all_passed = False
        
        print()
    
    if all_passed:
        print("✓ All symbolic rejection tests passed\n")
    else:
        print("✗ Some symbolic rejection tests failed\n")
    
    return all_passed


def test_symbolic_still_works():
    """Test that pure symbolic queries still route correctly."""
    print("=" * 80)
    print("TEST 4: Pure Symbolic Queries Still Route to Symbolic")
    print("=" * 80)
    
    reasoner = SymbolicReasoner()
    router = LLMQueryRouter(llm_client=None)
    
    symbolic_queries = [
        "Is A→B, B→C, ¬C satisfiable?",
        "∀x (P(x) → Q(x))",
        "Prove A ∧ B → C",
    ]
    
    all_passed = True
    for query in symbolic_queries:
        is_symbolic = reasoner.is_symbolic_query(query)
        decision = router.route(query)
        
        if is_symbolic:
            print(f"✓ Query correctly identified as symbolic")
        else:
            print(f"✗ Query incorrectly rejected as non-symbolic")
            all_passed = False
        
        if decision.engine == "symbolic":
            print(f"✓ Query routed to symbolic engine")
        else:
            print(f"✗ Query routed to {decision.engine} (expected: symbolic)")
            all_passed = False
        
        print(f"  Query: {query[:60]}...")
        print()
    
    if all_passed:
        print("✓ All pure symbolic tests passed\n")
    else:
        print("✗ Some pure symbolic tests failed\n")
    
    return all_passed


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("VALIDATING ANALOGICAL ROUTING FIXES")
    print("=" * 80 + "\n")
    
    results = []
    
    results.append(("Analogical Keywords", test_analogical_keywords()))
    results.append(("Analogical Routing", test_analogical_routing()))
    results.append(("Symbolic Rejection", test_symbolic_rejection()))
    results.append(("Symbolic Still Works", test_symbolic_still_works()))
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
