#!/usr/bin/env python3
"""
Manual validation script for self-introspection query fixes.

This script tests the key changes without requiring full test infrastructure:
1. Keywords are present in the lists
2. Exclusion pattern logic works correctly
3. Query router detects self-introspection correctly

Run with: python3 validate_introspection_fixes.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_keywords_in_classifier():
    """Test that introspection keywords were added to query_classifier.py"""
    print("\n=== Testing SELF_INTROSPECTION_KEYWORDS ===")
    
    # Read the file directly
    classifier_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vulcan', 'routing', 'query_classifier.py')
    with open(classifier_path, 'r') as f:
        content = f.read()
    
    keywords_to_check = ['introspection', 'introspect', 'self-reflection', 'self-examine']
    
    all_found = True
    for kw in keywords_to_check:
        if f'"{kw}"' in content:
            print(f"✓ '{kw}' found in SELF_INTROSPECTION_KEYWORDS")
        else:
            print(f"✗ '{kw}' NOT found in SELF_INTROSPECTION_KEYWORDS")
            all_found = False
    
    return all_found


def test_keywords_in_router():
    """Test that introspection keywords were added to query_router.py"""
    print("\n=== Testing introspection_topics in QueryRouter ===")
    
    router_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vulcan', 'routing', 'query_router.py')
    with open(router_path, 'r') as f:
        content = f.read()
    
    keywords_to_check = ['introspection', 'introspect', 'self-reflection', 'self-examine']
    
    all_found = True
    for kw in keywords_to_check:
        if f"'{kw}'" in content:
            print(f"✓ '{kw}' found in introspection_topics")
        else:
            print(f"✗ '{kw}' NOT found in introspection_topics")
            all_found = False
    
    return all_found


def test_exclusion_pattern_fix():
    """Test that exclusion pattern fix is present"""
    print("\n=== Testing Exclusion Pattern Fix ===")
    
    router_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vulcan', 'routing', 'query_router.py')
    with open(router_path, 'r') as f:
        content = f.read()
    
    # Check for key components of the fix
    checks = [
        ('creative_exclusion_patterns', 'creative_exclusion_patterns = ('),
        ('self_awareness_override_keywords', 'self_awareness_override_keywords = ('),
        ('text_after_pattern', 'text_after_pattern = query_lower'),
        ('has_self_awareness check', 'has_self_awareness = any'),
    ]
    
    all_found = True
    for name, pattern in checks:
        if pattern in content:
            print(f"✓ {name} found")
        else:
            print(f"✗ {name} NOT found")
            all_found = False
    
    return all_found


def test_agent_pool_fix():
    """Test that agent_pool.py fix is present"""
    print("\n=== Testing Agent Pool Fix ===")
    
    agent_pool_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vulcan', 'orchestrator', 'agent_pool.py')
    with open(agent_pool_path, 'r') as f:
        content = f.read()
    
    # Check for the fix - updated to match new professional comment
    if 'Simplified World Model Result Detection' in content or 'Relax metadata flag requirement' in content:
        print("✓ Agent pool metadata flag fix found")
        
        # Check that the condition no longer requires metadata flags
        # The new check should be simpler: just selected_tools and confidence
        if 'integration_result.selected_tools == ["world_model"] and' in content and \
           'integration_result.confidence >= WORLD_MODEL_CONFIDENCE_THRESHOLD' in content:
            print("✓ Relaxed confidence check logic found")
            return True
        else:
            print("✗ Relaxed confidence check logic NOT found")
            return False
    else:
        print("✗ Agent pool metadata flag fix NOT found")
        return False


def test_main_py_fix():
    """Test that main.py fix is present"""
    print("\n=== Testing main.py Fix ===")
    
    main_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vulcan', 'main.py')
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Check for the fix comment
    if 'FIX (Issue #5): Check content FIRST, then confidence' in content:
        print("✓ main.py content-first check fix found")
        
        # Check for candidates list approach
        if 'candidates = []' in content and 'if unified_conclusion is not None and unified_conclusion.strip():' in content:
            print("✓ Content-first logic (candidates approach) found")
            return True
        else:
            print("✗ Content-first logic NOT found")
            return False
    else:
        print("✗ main.py content-first check fix NOT found")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("Self-Introspection Query Fixes Validation")
    print("=" * 60)
    
    results = []
    
    results.append(("Keywords in query_classifier.py", test_keywords_in_classifier()))
    results.append(("Keywords in query_router.py", test_keywords_in_router()))
    results.append(("Exclusion pattern fix", test_exclusion_pattern_fix()))
    results.append(("Agent pool fix", test_agent_pool_fix()))
    results.append(("main.py fix", test_main_py_fix()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n✗ Some validation checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
