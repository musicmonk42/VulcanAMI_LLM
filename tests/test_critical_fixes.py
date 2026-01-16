#!/usr/bin/env python3
"""
Test for critical bug fixes in unified_chat.py

This test verifies:
1. chat.py has been deleted (split brain fix)
2. Orphaned job bug fix (increased polling timeout)
3. Crash bug fix (safe attribute access)
"""

import ast
from pathlib import Path


def test_chat_py_deleted():
    """Verify that src/vulcan/endpoints/chat.py has been deleted."""
    chat_file = Path("src/vulcan/endpoints/chat.py")
    assert not chat_file.exists(), \
        "chat.py should be deleted to fix split brain issue"
    print("✓ chat.py has been deleted (split brain fixed)")


def test_chat_router_removed_from_init():
    """Verify that chat_router has been removed from __init__.py."""
    init_file = Path("src/vulcan/endpoints/__init__.py")
    assert init_file.exists(), f"File not found: {init_file}"
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check that chat_router import is removed
    assert 'from vulcan.endpoints.chat import' not in content, \
        "chat_router import should be removed from __init__.py"
    
    # Check that chat_router is not in __all__
    assert '"chat_router"' not in content and "'chat_router'" not in content, \
        "chat_router should not be in __all__ list"
    
    print("✓ chat_router removed from __init__.py")


def test_orphaned_job_fix():
    """
    Verify that orphaned job bug is fixed by checking increased timeout values.
    
    The bug: MAX_POLL_ATTEMPTS=5, MAX_POLL_DELAY=1.0 gave only ~2.5s total wait
    The fix: MAX_POLL_ATTEMPTS=8, MAX_POLL_DELAY=2.0 gives ~12.7s total wait
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
        tree = ast.parse(content, filename=str(unified_chat_file))
    
    # Find the MAX_POLL_ATTEMPTS and MAX_POLL_DELAY assignments
    found_max_poll_attempts = False
    found_max_poll_delay = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == 'MAX_POLL_ATTEMPTS':
                        if isinstance(node.value, ast.Constant):
                            value = node.value.value
                            assert value >= 8, \
                                f"MAX_POLL_ATTEMPTS should be >= 8 (found {value})"
                            found_max_poll_attempts = True
                            print(f"  ✓ MAX_POLL_ATTEMPTS = {value}")
                    
                    elif target.id == 'MAX_POLL_DELAY':
                        if isinstance(node.value, ast.Constant):
                            value = node.value.value
                            assert value >= 2.0, \
                                f"MAX_POLL_DELAY should be >= 2.0 (found {value})"
                            found_max_poll_delay = True
                            print(f"  ✓ MAX_POLL_DELAY = {value}")
    
    assert found_max_poll_attempts, "MAX_POLL_ATTEMPTS not found in code"
    assert found_max_poll_delay, "MAX_POLL_DELAY not found in code"
    
    print("✓ Orphaned job bug fixed (increased polling timeout)")


def test_crash_bug_fix():
    """
    Verify that crash bug is fixed by checking for safe attribute access.
    
    The bug: integration_result.selected_tools causing AttributeError
    The fix: getattr(integration_result, 'selected_tools', 'unknown')
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
        tree = ast.parse(content, filename=str(unified_chat_file))
    
    # Industry Standard: Use AST parsing to robustly check for getattr patterns
    # Look for getattr calls with integration_result as first argument
    found_safe_access = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a getattr call
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                if len(node.args) >= 2:
                    # Check if first argument is integration_result (attribute access)
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Name) and first_arg.id == 'integration_result':
                        # Check if second argument is 'selected_tools' or 'metadata' or 'confidence'
                        if isinstance(node.args[1], ast.Constant):
                            attr_name = node.args[1].value
                            if attr_name in ('selected_tools', 'metadata', 'confidence', 'reasoning_strategy', 'rationale'):
                                found_safe_access = True
                                print(f"  ✓ Found safe access: getattr(integration_result, '{attr_name}')")
    
    assert found_safe_access, \
        "integration_result attributes should be accessed with getattr() for safety"
    
    print("✓ Crash bug fixed (safe attribute access with getattr)")


def test_defensive_metadata_extraction():
    """
    Verify that metadata is extracted defensively at the beginning.
    
    Industry standard: Extract once with safe defaults to prevent AttributeError.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), f"File not found: {unified_chat_file}"
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
        tree = ast.parse(content, filename=str(unified_chat_file))
    
    # Industry Standard: Use AST parsing to check for defensive variable assignments
    # Look for patterns like: metadata = getattr(integration_result, 'metadata', None) or {}
    found_metadata_extraction = False
    found_confidence_extraction = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Check for metadata assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == 'metadata':
                        # Check if it uses getattr with fallback
                        # Pattern 1: getattr(...) or {} (BoolOp)
                        # Pattern 2: just getattr(...) with default
                        if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                            # Pattern: getattr(...) or {}
                            if isinstance(node.value.values[0], ast.Call):
                                call = node.value.values[0]
                                if isinstance(call.func, ast.Name) and call.func.id == 'getattr':
                                    found_metadata_extraction = True
                                    print("  ✓ Found defensive metadata extraction (with or fallback)")
                        elif isinstance(node.value, ast.Call):
                            # Pattern: just getattr(...) with default
                            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'getattr':
                                if len(node.value.args) >= 3:  # has default value
                                    found_metadata_extraction = True
                                    print("  ✓ Found defensive metadata extraction (with default)")
                    
                    elif target.id == 'confidence':
                        # Check if it uses getattr
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'getattr':
                                found_confidence_extraction = True
                                print("  ✓ Found defensive confidence extraction")
    
    assert found_metadata_extraction, \
        "metadata should be extracted defensively with getattr() and fallback"
    
    assert found_confidence_extraction, \
        "confidence should be extracted defensively with getattr()"
    
    print("✓ Defensive metadata extraction implemented (industry standard)")


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: Critical Bug Fixes Verification")
    print("=" * 70)
    print()
    
    tests = [
        test_chat_py_deleted,
        test_chat_router_removed_from_init,
        test_orphaned_job_fix,
        test_crash_bug_fix,
        test_defensive_metadata_extraction,
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
        print("\n✓ All critical bug fixes verified!")
        exit(0)
    else:
        print(f"\n✗ {failed}/{len(tests)} tests failed")
        exit(1)
