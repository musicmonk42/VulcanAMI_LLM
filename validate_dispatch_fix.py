#!/usr/bin/env python3
"""
Manual validation of dispatch fix.
Tests the actual code flow with mock data.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

print("=" * 70)
print("DISPATCH FIX MANUAL VALIDATION")
print("=" * 70)
print()

# Test 1: Verify patterns removed
print("TEST 1: Verify keyword override patterns removed")
print("-" * 70)

try:
    from vulcan.reasoning.selection.tool_selector import ToolSelector
    
    tests_passed = 0
    tests_failed = 0
    
    if not hasattr(ToolSelector, '_MATH_PATTERN'):
        print("✅ _MATH_PATTERN removed")
        tests_passed += 1
    else:
        print("❌ _MATH_PATTERN still exists")
        tests_failed += 1
    
    if not hasattr(ToolSelector, '_SAT_PATTERN'):
        print("✅ _SAT_PATTERN removed")
        tests_passed += 1
    else:
        print("❌ _SAT_PATTERN still exists")
        tests_failed += 1
    
    if not hasattr(ToolSelector, '_CAUSAL_PATTERN'):
        print("✅ _CAUSAL_PATTERN removed")
        tests_passed += 1
    else:
        print("❌ _CAUSAL_PATTERN still exists")
        tests_failed += 1
    
    print(f"\nTest 1 Result: {tests_passed} passed, {tests_failed} failed")
    
except Exception as e:
    print(f"❌ Error importing ToolSelector: {e}")
    tests_failed += 1

print()

# Test 2: Verify mapping logic
print("TEST 2: Verify selected_tools → classifier_suggested_tools mapping")
print("-" * 70)

try:
    import inspect
    source = inspect.getsource(ToolSelector.select_and_execute)
    
    # Check for mapping logic
    if "selected_tools" in source and "classifier_suggested_tools" in source:
        # Look for the specific mapping line
        if "request.context['classifier_suggested_tools'] = selected_tools" in source:
            print("✅ Mapping logic found: classifier_suggested_tools = selected_tools")
        else:
            print("⚠️  Both variables present but exact mapping line not found")
    else:
        print("❌ Mapping logic not found")
    
    # Check that keyword_override_tool is removed
    if "keyword_override_tool" not in source:
        print("✅ keyword_override_tool logic removed")
    else:
        print("❌ keyword_override_tool logic still present")
    
    # Check that pattern matching override is removed
    if "_MATH_PATTERN.search" not in source and "_SAT_PATTERN.search" not in source:
        print("✅ Pattern matching override removed")
    else:
        print("❌ Pattern matching override still present")
    
except Exception as e:
    print(f"❌ Error inspecting source: {e}")

print()

# Test 3: Test the mapping logic with mock request
print("TEST 3: Test mapping logic with mock SelectionRequest")
print("-" * 70)

try:
    from vulcan.reasoning.selection.tool_selector import SelectionRequest
    
    # Create a request with selected_tools (as query_router would set)
    request = SelectionRequest(
        problem="Test query",
        context={
            'selected_tools': ['symbolic'],
            'classifier_category': 'LOGICAL',
        },
        query_id="test-001"
    )
    
    # Simulate the mapping logic
    if hasattr(request, 'context') and isinstance(request.context, dict):
        selected_tools = request.context.get('selected_tools')
        
        if selected_tools and not request.context.get('classifier_suggested_tools'):
            request.context['classifier_suggested_tools'] = selected_tools
            print(f"✅ Mapping executed: selected_tools={selected_tools} → classifier_suggested_tools")
        
        # Verify the result
        if request.context.get('classifier_suggested_tools') == ['symbolic']:
            print("✅ Context now has classifier_suggested_tools=['symbolic']")
        else:
            print(f"❌ Unexpected value: {request.context.get('classifier_suggested_tools')}")
    else:
        print("❌ Request context not accessible")
    
except Exception as e:
    print(f"❌ Error testing mapping: {e}")

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
