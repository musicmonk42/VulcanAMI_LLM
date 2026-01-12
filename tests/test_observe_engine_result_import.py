#!/usr/bin/env python3
"""
Test for observe_engine_result import in unified_chat.py

This test verifies that the observe_engine_result function is properly imported
to fix the NameError that occurred on line 1451.

Related Issue: Fix for /vulcan/v1/chat returning 500 error due to missing observe_engine_result import
"""

import ast
from pathlib import Path


def test_observe_engine_result_imported():
    """
    Verify that observe_engine_result is imported in unified_chat.py.
    
    This test verifies the fix for a critical production bug where the function
    observe_engine_result() was called on line 1451 but not imported, causing
    NameError: name 'observe_engine_result' is not defined and 500 errors.
    
    This test uses AST parsing to check for the presence of observe_engine_result
    in the imports from vulcan.reasoning.integration.utils.
    """
    # Read the unified_chat.py file
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Check for 'from vulcan.reasoning.integration.utils import ..., observe_engine_result'
    has_observe_engine_result_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'vulcan.reasoning.integration.utils':
                for alias in node.names:
                    if alias.name == 'observe_engine_result':
                        has_observe_engine_result_import = True
                        break
        if has_observe_engine_result_import:
            break
    
    assert has_observe_engine_result_import, \
        "observe_engine_result must be imported from vulcan.reasoning.integration.utils"
    
    print("✓ observe_engine_result is correctly imported in unified_chat.py")
    return True


def test_observe_engine_result_usage():
    """
    Verify that observe_engine_result is actually used in the file.
    
    This ensures the usage pattern that requires the import is still present.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for observe_engine_result usage
    assert 'observe_engine_result(' in content, \
        "Expected to find observe_engine_result() usage in the file"
    
    print("✓ observe_engine_result() is used in unified_chat.py")
    return True


def test_all_observe_functions_imported():
    """
    Verify that all three observe functions are imported together.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Find the import statement for vulcan.reasoning.integration.utils
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'vulcan.reasoning.integration.utils':
                for alias in node.names:
                    imported_names.add(alias.name)
    
    required_functions = {'observe_query_start', 'observe_outcome', 'observe_engine_result'}
    missing_functions = required_functions - imported_names
    
    assert not missing_functions, \
        f"Missing required imports: {missing_functions}"
    
    print("✓ All three observe functions are imported: observe_query_start, observe_outcome, observe_engine_result")
    return True


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: observe_engine_result import test for unified_chat.py")
    print("=" * 70)
    print()
    
    tests = [
        test_observe_engine_result_imported,
        test_observe_engine_result_usage,
        test_all_observe_functions_imported,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"Running: {test.__name__}")
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total:  {len(tests)} tests")
    print(f"Passed: {passed} tests")
    print(f"Failed: {failed} tests")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print(f"\n✗ {failed}/{len(tests)} tests failed")
        exit(1)
