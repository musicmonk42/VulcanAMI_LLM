#!/usr/bin/env python3
"""
Test script to verify the HybridLLMExecutor initialization fix.

This script verifies that:
1. The correct import paths are being used
2. The imports can be successfully loaded
3. The functions are properly available

Run with: python3 test_initialization_fix.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_correct_imports():
    """Test that the corrected import paths work."""
    print("Testing corrected import paths...")
    
    try:
        # These are the correct imports that we fixed
        from vulcan.llm.hybrid_executor import (
            get_or_create_hybrid_executor,
            verify_hybrid_executor_setup
        )
        from vulcan.llm.openai_client import get_openai_client, log_openai_status
        
        print("✓ All imports successful from correct paths")
        print(f"  - get_or_create_hybrid_executor: {get_or_create_hybrid_executor}")
        print(f"  - verify_hybrid_executor_setup: {verify_hybrid_executor_setup}")
        print(f"  - get_openai_client: {get_openai_client}")
        print(f"  - log_openai_status: {log_openai_status}")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_old_imports_fail():
    """Verify that the old incorrect import paths are not used."""
    print("\nVerifying old incorrect paths are not being used...")
    
    # Try the old incorrect import paths - they should fail
    old_paths_correct = True
    
    try:
        from vulcan.distillation.hybrid_executor import get_or_create_hybrid_executor
        print("✗ WARNING: Old incorrect import path 'vulcan.distillation.hybrid_executor' still works!")
        old_paths_correct = False
    except (ImportError, ModuleNotFoundError):
        print("✓ Old incorrect path 'vulcan.distillation.hybrid_executor' properly fails")
    
    try:
        from vulcan.utils_main.openai_client import get_openai_client
        print("✗ WARNING: Old incorrect import path 'vulcan.utils_main.openai_client' still works!")
        old_paths_correct = False
    except (ImportError, ModuleNotFoundError):
        print("✓ Old incorrect path 'vulcan.utils_main.openai_client' properly fails")
    
    return old_paths_correct


def test_openai_status_function():
    """Test that log_openai_status function works correctly."""
    print("\nTesting log_openai_status function...")
    
    try:
        from vulcan.llm.openai_client import log_openai_status
        
        # Call the function - it should not raise an exception
        log_openai_status()
        
        print("✓ log_openai_status() executed successfully")
        return True
    except Exception as e:
        print(f"✗ log_openai_status() failed: {e}")
        return False


def test_verification_function():
    """Test that verify_hybrid_executor_setup function works correctly."""
    print("\nTesting verify_hybrid_executor_setup function...")
    
    try:
        from vulcan.llm.hybrid_executor import verify_hybrid_executor_setup
        
        # Call the function - it should return a dict with status info
        result = verify_hybrid_executor_setup()
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dict"
        assert "status" in result, "Result should have 'status' key"
        assert "message" in result, "Result should have 'message' key"
        assert "has_internal_llm" in result, "Result should have 'has_internal_llm' key"
        
        print(f"✓ verify_hybrid_executor_setup() returned: status={result['status']}")
        print(f"  Message: {result['message']}")
        
        return True
    except Exception as e:
        print(f"✗ verify_hybrid_executor_setup() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("HybridLLMExecutor Initialization Fix - Verification Tests")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Correct imports work
    results.append(("Correct imports", test_correct_imports()))
    
    # Test 2: Old imports fail (as they should)
    results.append(("Old imports fail", test_old_imports_fail()))
    
    # Test 3: log_openai_status function works
    results.append(("log_openai_status", test_openai_status_function()))
    
    # Test 4: verify_hybrid_executor_setup function works
    results.append(("verify_hybrid_executor_setup", test_verification_function()))
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests PASSED! The initialization fix is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
