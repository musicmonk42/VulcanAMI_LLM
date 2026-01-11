#!/usr/bin/env python3
"""
Test for secrets and observe_query_start imports in unified_chat.py

This test verifies that the secrets module and observe_query_start function
are properly imported to fix the NameErrors that occurred when:
- Using secrets.token_urlsafe() on lines 2169 and 2170
- Calling observe_query_start() on line 180

Related Issue: Fix for /vulcan/v1/chat returning 500 error due to missing imports
"""

import ast
from pathlib import Path


def test_secrets_module_imported():
    """
    Verify that the secrets module is imported in unified_chat.py.

    This test uses AST parsing to check for the presence of 'import secrets'
    in the imports section of the file.
    """
    # Read the unified_chat.py file
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Check for 'import secrets' statement
    has_secrets_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'secrets':
                    has_secrets_import = True
                    break
        if has_secrets_import:
            break
    
    assert has_secrets_import, "The 'secrets' module must be imported to use secrets.token_urlsafe()"
    
    print("✓ secrets module is correctly imported in unified_chat.py")
    return True


def test_secrets_token_urlsafe_usage():
    """
    Verify that secrets.token_urlsafe() is used in the file (confirming the need for secrets import).

    This ensures that the usage pattern that requires the secrets import is still present.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for secrets.token_urlsafe usage
    assert 'secrets.token_urlsafe(' in content, \
        "Expected to find secrets.token_urlsafe() usage in the file"
    
    # Specifically check for the response_id generation
    assert 'secrets.token_urlsafe(16)' in content, \
        "Expected to find secrets.token_urlsafe(16) for response_id generation"
    
    print("✓ secrets.token_urlsafe() is used in unified_chat.py (confirming need for secrets import)")
    return True


def test_observe_query_start_imported():
    """
    Verify that observe_query_start is imported in unified_chat.py.

    This test uses AST parsing to check for the import of observe_query_start
    from vulcan.reasoning.integration.utils.
    """
    # Read the unified_chat.py file
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Check for 'from vulcan.reasoning.integration.utils import observe_query_start' statement
    has_observe_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and 'vulcan.reasoning.integration' in node.module:
                for alias in node.names:
                    if alias.name == 'observe_query_start':
                        has_observe_import = True
                        break
        if has_observe_import:
            break
    
    assert has_observe_import, \
        "The 'observe_query_start' function must be imported from vulcan.reasoning.integration"
    
    print("✓ observe_query_start is correctly imported in unified_chat.py")
    return True


def test_observe_query_start_usage():
    """
    Verify that observe_query_start() is called in the file (confirming the need for the import).

    This ensures that the usage pattern that requires the observe_query_start import is still present.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for observe_query_start usage
    assert 'observe_query_start(' in content, \
        "Expected to find observe_query_start() usage in the file"
    
    print("✓ observe_query_start() is used in unified_chat.py (confirming need for the import)")
    return True


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: secrets and observe_query_start import tests")
    print("=" * 70)
    print()
    
    tests = [
        test_secrets_module_imported,
        test_secrets_token_urlsafe_usage,
        test_observe_query_start_imported,
        test_observe_query_start_usage,
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
