#!/usr/bin/env python3
"""
Simple validation script for analogical routing keywords.
Tests the changes without requiring full module imports.
"""

import sys
import re

def test_llm_router_keywords():
    """Test that ANALOGICAL_KEYWORDS in llm_router.py has the required keywords."""
    print("=" * 80)
    print("TEST: Checking ANALOGICAL_KEYWORDS in llm_router.py")
    print("=" * 80)
    
    # Read the file
    with open('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/routing/llm_router.py', 'r') as f:
        content = f.read()
    
    # Find the ANALOGICAL_KEYWORDS section
    match = re.search(r'ANALOGICAL_KEYWORDS.*?frozenset\(\[(.*?)\]\)', content, re.DOTALL)
    if not match:
        print("✗ Could not find ANALOGICAL_KEYWORDS definition")
        return False
    
    keywords_text = match.group(1)
    
    required_keywords = [
        "structure mapping",
        "domain mapping",
        "map the",
        "deep structure",
        "source domain",
        "target domain",
        "analogical",
        "analogy",
    ]
    
    all_present = True
    for keyword in required_keywords:
        if f'"{keyword}"' in keywords_text or f"'{keyword}'" in keywords_text:
            print(f"✓ '{keyword}' found")
        else:
            print(f"✗ '{keyword}' NOT FOUND")
            all_present = False
    
    # Check for domain mapping notation
    if 's→t' in keywords_text.lower() or 's->t' in keywords_text.lower():
        print(f"✓ Domain mapping notation (S→T/S->T) found")
    else:
        print(f"✗ Domain mapping notation NOT FOUND")
        all_present = False
    
    print()
    return all_present


def test_symbolic_reasoner_rejection():
    """Test that symbolic reasoner has analogical detection."""
    print("=" * 80)
    print("TEST: Checking Symbolic Reasoner Analogical Detection")
    print("=" * 80)
    
    # Read the file
    with open('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/reasoning/symbolic/reasoner.py', 'r') as f:
        content = f.read()
    
    # Check for analogical keyword detection in is_symbolic_query
    if 'analogical_keywords' in content and 'structure mapping' in content:
        print("✓ Analogical keyword detection found in is_symbolic_query")
    else:
        print("✗ Analogical keyword detection NOT FOUND in is_symbolic_query")
        return False
    
    # Check for early return on analogical patterns
    if 'has_analogical_keyword' in content:
        print("✓ Early return logic for analogical patterns found")
    else:
        print("✗ Early return logic NOT FOUND")
        return False
    
    # Check check_applicability method has suggestion
    if "'suggestion': 'analogical'" in content or '"suggestion": "analogical"' in content:
        print("✓ check_applicability suggests 'analogical' for analogical queries")
    else:
        print("✗ check_applicability does NOT suggest 'analogical'")
        return False
    
    # Check query method has analogical suggestion
    match = re.search(r'def query\(.*?\):', content, re.DOTALL)
    if match:
        # Find the query method and check for analogical suggestion
        query_method_start = match.start()
        # Find next method definition
        next_method = re.search(r'\n    def ', content[query_method_start + 100:])
        if next_method:
            query_method = content[query_method_start:query_method_start + 100 + next_method.start()]
        else:
            query_method = content[query_method_start:]
        
        if 'analogical' in query_method:
            print("✓ query() method mentions analogical")
        else:
            print("⚠ query() method may not mention analogical (check manually)")
    
    print()
    return True


def test_orchestrator_aliases():
    """Test that orchestrator has analogical tool aliases."""
    print("=" * 80)
    print("TEST: Checking Orchestrator Tool Aliases")
    print("=" * 80)
    
    # Read the file
    with open('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/reasoning/unified/orchestrator.py', 'r') as f:
        content = f.read()
    
    # Check for analogical tool mappings
    if "'analogical': ReasoningType.ANALOGICAL" in content or '"analogical": ReasoningType.ANALOGICAL' in content:
        print("✓ 'analogical' → ReasoningType.ANALOGICAL mapping found")
    else:
        print("✗ 'analogical' mapping NOT FOUND")
        return False
    
    if "'analogy': ReasoningType.ANALOGICAL" in content or '"analogy": ReasoningType.ANALOGICAL' in content:
        print("✓ 'analogy' → ReasoningType.ANALOGICAL mapping found")
    else:
        print("✗ 'analogy' mapping NOT FOUND")
        return False
    
    print()
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("VALIDATING ANALOGICAL ROUTING FIXES (Code-Level)")
    print("=" * 80 + "\n")
    
    results = []
    
    results.append(("LLM Router Keywords", test_llm_router_keywords()))
    results.append(("Symbolic Reasoner Rejection", test_symbolic_reasoner_rejection()))
    results.append(("Orchestrator Tool Aliases", test_orchestrator_aliases()))
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ ALL CODE-LEVEL TESTS PASSED")
        print("\nThe code changes are in place correctly.")
        print("Runtime tests require numpy and other dependencies to be installed.")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
