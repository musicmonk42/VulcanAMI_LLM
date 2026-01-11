#!/usr/bin/env python3
"""
Test for os module import in unified_chat.py

This test verifies that the os module is properly imported to fix the NameError
that occurred when accessing os.environ.get() on line 1464.

Related Issue: Fix for /vulcan/v1/chat returning 500 error due to missing os import
"""

import ast
from pathlib import Path


def test_os_module_imported():
    """
    Verify that the os module is imported in unified_chat.py.
    
    This test uses AST parsing to check for the presence of 'import os'
    in the imports section of the file.
    """
    # Read the unified_chat.py file
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Check for 'import os' statement
    has_os_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'os':
                    has_os_import = True
                    break
        if has_os_import:
            break
    
    assert has_os_import, "The 'os' module must be imported to use os.environ.get()"
    
    print("✓ os module is correctly imported in unified_chat.py")
    return True


def test_os_environ_usage():
    """
    Verify that os.environ.get() is used in the file (confirming the need for os import).
    
    This ensures that the usage pattern that requires the os import is still present.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for os.environ.get usage
    assert 'os.environ.get(' in content, \
        "Expected to find os.environ.get() usage in the file"
    
    # Specifically check for the VULCAN_MIN_REASONING_CONFIDENCE variable usage
    assert 'os.environ.get("VULCAN_MIN_REASONING_CONFIDENCE"' in content, \
        "Expected to find VULCAN_MIN_REASONING_CONFIDENCE environment variable usage"
    
    print("✓ os.environ.get() is used in unified_chat.py (confirming need for os import)")
    return True


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: os module import test for unified_chat.py")
    print("=" * 70)
    print()
    
    tests = [
        test_os_module_imported,
        test_os_environ_usage,
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
