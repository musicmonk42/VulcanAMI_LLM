#!/usr/bin/env python3
"""
Static analysis test for unified_chat signature fix.

This test uses AST parsing to verify the function signature without
requiring runtime imports of FastAPI or Pydantic.

Related Issue: Fix for /vulcan/v1/chat returning 500 error
"""

import ast
from pathlib import Path


def test_unified_chat_signature_static():
    """
    Verify unified_chat has correct signature using AST parsing.
    
    Validates:
        - Function has 'request' parameter
        - Function has 'body' parameter
        - Function is async
    """
    # Read the unified_chat.py file
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(unified_chat_file))
    
    # Find the unified_chat function
    unified_chat_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "unified_chat":
            unified_chat_func = node
            break
    
    assert unified_chat_func is not None, "unified_chat function not found"
    
    # Check it's an async function
    assert isinstance(unified_chat_func, ast.AsyncFunctionDef), \
        "unified_chat should be an async function"
    
    # Get parameter names
    param_names = [arg.arg for arg in unified_chat_func.args.args]
    
    # Check parameters exist
    assert 'request' in param_names, "unified_chat should have 'request' parameter"
    assert 'body' in param_names, "unified_chat should have 'body' parameter"
    
    # Check parameter order
    request_idx = param_names.index('request')
    body_idx = param_names.index('body')
    assert request_idx < body_idx, "'request' parameter should come before 'body'"
    
    print(f"✓ unified_chat signature: async def unified_chat({', '.join(param_names[:2])}, ...)")
    return True


def test_body_references_in_unified_chat():
    """
    Verify that body.* references are used instead of request.* for Pydantic fields.
    
    Validates:
        - Uses body.message, not request.message
        - Uses body.history, not request.history
        - Uses body.max_tokens, not request.max_tokens
        - Uses body.enable_*, not request.enable_*
        - Uses body.conversation_id, not request.conversation_id
        - Still uses request.app (FastAPI Request attribute)
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check that request.app is still present (FastAPI Request access)
    assert 'request.app' in content, "Should still access request.app from FastAPI Request"
    
    # Check that problematic request.* patterns are NOT present
    problematic_patterns = [
        'request.message',
        'request.history',
        'request.max_tokens',
        'request.enable_safety',
        'request.enable_memory',
        'request.enable_reasoning',
        'request.enable_causal',
        'request.enable_planning',
        'request.conversation_id',
    ]
    
    for pattern in problematic_patterns:
        assert pattern not in content, \
            f"Found problematic pattern '{pattern}' - should use 'body.' instead"
    
    # Check that correct body.* patterns ARE present
    correct_patterns = [
        'body.message',
        'body.history',
        'body.max_tokens',
        'body.enable_safety',
        'body.enable_memory',
        'body.enable_reasoning',
        'body.enable_causal',
        'body.enable_planning',
        'body.conversation_id',
    ]
    
    for pattern in correct_patterns:
        assert pattern in content, \
            f"Missing correct pattern '{pattern}' - should use this instead of 'request.{pattern[5:]}'"
    
    print("✓ All body.* references are correct (no request.<field> patterns found)")
    return True


def test_truncate_history_call():
    """
    Verify that truncate_history is called (not _truncate_history).
    
    Validates:
        - Uses truncate_history(body.history), not _truncate_history(request.history)
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check that the typo _truncate_history is not present
    assert '_truncate_history' not in content, \
        "Found typo '_truncate_history' - should be 'truncate_history'"
    
    # Check that correct function call is present
    assert 'truncate_history(body.history)' in content, \
        "Should call truncate_history(body.history)"
    
    print("✓ truncate_history(body.history) is used correctly")
    return True


def test_full_platform_proxy_call():
    """
    Verify that full_platform.py proxy passes both request and chat_request.
    
    Validates:
        - Calls unified_chat(request, chat_request)
    """
    full_platform_file = Path("src/full_platform.py")
    assert full_platform_file.exists(), f"File not found: {full_platform_file}"
    
    with open(full_platform_file, 'r') as f:
        content = f.read()
    
    # Check for the correct call pattern
    assert 'await unified_chat(request, chat_request)' in content, \
        "full_platform.py should call: await unified_chat(request, chat_request)"
    
    # Check that the old incorrect pattern is NOT present
    assert 'await unified_chat(chat_request)' not in content or \
           'await unified_chat(request, chat_request)' in content, \
        "Should not have old pattern: await unified_chat(chat_request)"
    
    print("✓ full_platform.py proxy calls unified_chat(request, chat_request)")
    return True


def test_unified_chat_request_import():
    """
    Verify that UnifiedChatRequest is imported in unified_chat.py.
    
    Validates:
        - Imports UnifiedChatRequest from vulcan.api.models
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for the import
    assert 'from vulcan.api.models import UnifiedChatRequest' in content, \
        "Should import UnifiedChatRequest from vulcan.api.models"
    
    print("✓ UnifiedChatRequest is imported from vulcan.api.models")
    return True


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: unified_chat Static Analysis Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_unified_chat_signature_static,
        test_body_references_in_unified_chat,
        test_truncate_history_call,
        test_full_platform_proxy_call,
        test_unified_chat_request_import,
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
