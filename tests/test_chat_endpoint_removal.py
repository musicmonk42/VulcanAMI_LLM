#!/usr/bin/env python3
"""
Test for redundant chat endpoint removal

This test verifies that the redundant chat_endpoint.py file and its configuration
have been properly removed while maintaining the VULCAN chat functionality.

Related Issue: Remove redundant /chat/v1/chat endpoint
"""

import ast
from pathlib import Path


def test_chat_endpoint_file_removed():
    """
    Verify that src/chat_endpoint.py has been removed.
    """
    chat_endpoint_file = Path("src/chat_endpoint.py")
    assert not chat_endpoint_file.exists(), \
        "src/chat_endpoint.py should be removed as it's redundant"
    
    print("✓ src/chat_endpoint.py has been removed")
    return True


def test_full_platform_chat_config_removed():
    """
    Verify that chat endpoint configuration has been removed from full_platform.py.
    """
    full_platform_file = Path("src/full_platform.py")
    assert full_platform_file.exists(), f"File not found: {full_platform_file}"
    
    with open(full_platform_file, 'r') as f:
        content = f.read()
    
    # Check that chat endpoint config variables are removed
    assert 'enable_chat_endpoint' not in content, \
        "enable_chat_endpoint should be removed from UnifiedPlatformSettings"
    assert 'chat_mount: str = "/chat"' not in content, \
        "chat_mount configuration should be removed"
    assert 'chat_module: str = "src.chat_endpoint"' not in content, \
        "chat_module configuration should be removed"
    assert 'chat_attr: str = "app"' not in content, \
        "chat_attr configuration should be removed"
    
    # Check that mounting block is removed
    assert 'Import and mount Chat Endpoint' not in content, \
        "Chat Endpoint mounting block should be removed"
    assert '→ Chat API available at /chat/v1/chat' not in content, \
        "Chat endpoint logging should be removed"
    
    print("✓ Chat endpoint configuration removed from full_platform.py")
    return True


def test_unified_chat_still_exists():
    """
    Verify that the main unified_chat.py endpoint still exists.
    """
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), \
        "unified_chat.py must exist as the primary chat endpoint"
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Verify it has the observe_engine_result import
    assert 'observe_engine_result' in content, \
        "unified_chat.py should have observe_engine_result imported"
    
    print("✓ src/vulcan/endpoints/unified_chat.py exists and is functional")
    return True


def test_verify_deployment_updated():
    """
    Verify that scripts/verify_deployment.py no longer checks /chat/health.
    """
    verify_script = Path("scripts/verify_deployment.py")
    assert verify_script.exists(), f"File not found: {verify_script}"
    
    with open(verify_script, 'r') as f:
        content = f.read()
    
    # Check that Chat Health check is removed
    assert '"/chat/health"' not in content, \
        "Chat Health check should be removed from verify_deployment.py"
    
    # Verify VULCAN endpoints are still there
    assert '"/vulcan/health"' in content, \
        "VULCAN Health check should still be present"
    assert '"/vulcan/v1/chat"' in content, \
        "VULCAN Chat check should still be present"
    
    print("✓ scripts/verify_deployment.py updated correctly")
    return True


def test_start_script_updated():
    """
    Verify that start_chat_interface.sh no longer mentions /chat/v1/chat.
    """
    start_script = Path("start_chat_interface.sh")
    assert start_script.exists(), f"File not found: {start_script}"
    
    with open(start_script, 'r') as f:
        content = f.read()
    
    # Check that /chat/v1/chat reference is removed
    assert 'http://localhost:8080/chat/v1/chat' not in content, \
        "Chat endpoint URL should be removed from start_chat_interface.sh"
    
    # Verify VULCAN endpoint is still there
    assert 'http://localhost:8080/vulcan/v1/chat' in content, \
        "VULCAN Chat URL should still be present"
    
    print("✓ start_chat_interface.sh updated correctly")
    return True


def test_readme_chat_updated():
    """
    Verify that docs/README_CHAT.md has been updated.
    """
    readme_file = Path("docs/README_CHAT.md")
    assert readme_file.exists(), f"File not found: {readme_file}"
    
    with open(readme_file, 'r') as f:
        content = f.read()
    
    # Check for removed references
    assert '/chat/v1/chat' not in content, \
        "/chat/v1/chat references should be removed from README_CHAT.md"
    assert '`src/chat_endpoint.py`' not in content, \
        "chat_endpoint.py should not be referenced in the files table"
    
    # Verify VULCAN endpoints are documented
    assert '/vulcan/v1/chat' in content, \
        "VULCAN chat endpoint should be documented"
    assert 'unified_chat.py' in content, \
        "unified_chat.py should be referenced as the chat implementation"
    
    print("✓ docs/README_CHAT.md updated correctly")
    return True


def test_frontend_uses_correct_endpoint():
    """
    Verify that vulcan_chat.html doesn't use /chat/v1/chat.
    """
    frontend_file = Path("vulcan_chat.html")
    if not frontend_file.exists():
        print("⊘ vulcan_chat.html not found, skipping frontend check")
        return True
    
    with open(frontend_file, 'r') as f:
        content = f.read()
    
    # Verify frontend doesn't use the removed endpoint
    assert '/chat/v1/chat' not in content, \
        "Frontend should not use /chat/v1/chat endpoint"
    
    # Verify it uses VULCAN endpoints
    assert '/vulcan/v1/chat' in content or '/v1/chat' in content, \
        "Frontend should use /vulcan/v1/chat or /v1/chat"
    
    print("✓ vulcan_chat.html uses correct endpoints")
    return True


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 70)
    print("VULCAN-AGI: Redundant chat endpoint removal verification")
    print("=" * 70)
    print()
    
    tests = [
        test_chat_endpoint_file_removed,
        test_full_platform_chat_config_removed,
        test_unified_chat_still_exists,
        test_verify_deployment_updated,
        test_start_script_updated,
        test_readme_chat_updated,
        test_frontend_uses_correct_endpoint,
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
