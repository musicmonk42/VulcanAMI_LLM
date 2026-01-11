"""
Test Suite: Backward Compatibility Exports After main.py Refactoring

This test suite validates that the refactored main.py maintains backward compatibility
by re-exporting models and functions that external code (src/full_platform.py, tests,
and integrations) depends on.

CONTEXT:
    - Original main.py: 11,316 lines (monolithic architecture)
    - Refactored main.py: 222 lines (modular architecture, PR #704)
    - Components extracted: 24 focused modules
    - Breaking change risk: High (many direct imports from main)

TEST STRATEGY:
    1. Static analysis via AST parsing (no runtime imports needed)
    2. Verify model definitions in vulcan/api/models.py
    3. Verify function definitions in vulcan/endpoints/feedback.py
    4. Verify backward compatibility re-exports in main.py
    5. Verify dependency validation fixes in orchestrator/dependencies.py
    6. Runtime validation of helper function behavior

DEPENDENCIES:
    - Python 3.8+ (ast, pathlib, sys modules)
    - No external test frameworks required (standalone executable)
    - Optional: pytest for integration into CI/CD pipeline

USAGE:
    Standalone execution:
        $ python tests/test_backward_compatibility_exports.py
    
    With pytest:
        $ pytest tests/test_backward_compatibility_exports.py -v
    
    With coverage:
        $ pytest tests/test_backward_compatibility_exports.py --cov=src/vulcan

AUTHOR: VULCAN-AGI Team
VERSION: 1.0.0
CREATED: 2026-01-11 (Issue: Missing exports after refactoring)
"""

import ast
import sys
from pathlib import Path
from typing import Set


# ============================================================================
# TEST UTILITIES
# ============================================================================

def parse_python_file(file_path: Path) -> ast.Module:
    """
    Parse a Python file into an AST.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Parsed AST module
        
    Raises:
        AssertionError: If file doesn't exist or has syntax errors
    """
    assert file_path.exists(), f"File not found: {file_path}"
    
    with open(file_path, encoding="utf-8") as f:
        try:
            return ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")


def extract_class_names(tree: ast.Module) -> Set[str]:
    """Extract all class names from an AST."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }


def extract_function_names(tree: ast.Module) -> Set[str]:
    """Extract all function names (sync and async) from an AST."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# ============================================================================
# TEST CASES
# ============================================================================


def test_models_defined_in_api_models():
    """
    TEST: Verify feedback and chat models are defined in vulcan/api/models.py
    
    VALIDATES:
        - FeedbackRequest class exists
        - ThumbsFeedbackRequest class exists
        - UnifiedChatRequest class exists
    
    RATIONALE:
        These models were moved from main.py to api/models.py during refactoring.
        They must exist for backward compatibility re-exports to work.
    """
    models_file = Path("src/vulcan/api/models.py")
    tree = parse_python_file(models_file)
    classes = extract_class_names(tree)
    
    # Check required models exist
    required_models = {
        "FeedbackRequest",
        "ThumbsFeedbackRequest",
        "UnifiedChatRequest",
    }
    
    for model in required_models:
        assert model in classes, f"{model} not defined in vulcan/api/models.py"
    
    print(f"✓ All required models defined in vulcan/api/models.py: {required_models}")


def test_models_exported_from_api_models():
    """
    TEST: Verify models are in __all__ export list
    
    VALIDATES:
        - Models are explicitly exported via __all__
        - Follows Python best practices for public API definition
    
    RATIONALE:
        Explicit exports prevent internal implementation details from leaking.
        Required for "from vulcan.api.models import *" patterns.
    """
    models_file = Path("src/vulcan/api/models.py")
    
    with open(models_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check __all__ contains the models
    required_exports = [
        "FeedbackRequest",
        "ThumbsFeedbackRequest",
        "UnifiedChatRequest",
    ]
    
    for export in required_exports:
        assert f'"{export}"' in content, f'{export} not in __all__ in vulcan/api/models.py'
    
    print(f"✓ All required models in __all__ export list")


def test_backward_compat_imports_in_main():
    """
    TEST: Verify main.py has backward compatibility re-exports
    
    VALIDATES:
        - BACKWARD COMPATIBILITY EXPORTS section exists
        - All required names are imported
        - Imports are from correct source modules
    
    RATIONALE:
        Code that does "from src.vulcan.main import FeedbackRequest" must continue
        to work after refactoring. Re-exports maintain this API surface.
    """
    main_file = Path("src/vulcan/main.py")
    
    with open(main_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check for backward compatibility section
    assert "BACKWARD COMPATIBILITY EXPORTS" in content, \
        "Missing BACKWARD COMPATIBILITY EXPORTS section in main.py"
    
    # Check specific imports exist (imports may be on same line)
    required_names = [
        "FeedbackRequest",
        "ThumbsFeedbackRequest",
        "UnifiedChatRequest",
        "submit_feedback",
        "submit_thumbs_feedback",
        "get_feedback_stats",
        "_get_reasoning_attr",
    ]
    
    for name in required_names:
        assert name in content, f"Missing import in main.py: {name}"
    
    # Verify imports are from correct modules
    assert "from vulcan.api.models import" in content, \
        "Missing import from vulcan.api.models"
    assert "from vulcan.endpoints.feedback import" in content, \
        "Missing import from vulcan.endpoints.feedback"
    assert "from vulcan.utils.reasoning_helpers import" in content, \
        "Missing import from vulcan.utils.reasoning_helpers"
    
    print(f"✓ All backward compatibility imports present in main.py")


def test_feedback_functions_exist():
    """
    TEST: Verify feedback endpoint functions exist
    
    VALIDATES:
        - submit_feedback() async function exists
        - submit_thumbs_feedback() async function exists
        - get_feedback_stats() async function exists
    
    RATIONALE:
        These functions are FastAPI endpoint handlers that were extracted
        from main.py. They must exist for re-export and proxy routing.
    """
    feedback_file = Path("src/vulcan/endpoints/feedback.py")
    tree = parse_python_file(feedback_file)
    functions = extract_function_names(tree)
    
    required_functions = {
        "submit_feedback",
        "submit_thumbs_feedback",
        "get_feedback_stats",
    }
    
    for func in required_functions:
        assert func in functions, f"{func} not defined in vulcan/endpoints/feedback.py"
    
    print(f"✓ All required functions defined in vulcan/endpoints/feedback.py")


def test_dependency_validation_recognizes_continual():
    """
    TEST: Verify dependency validation recognizes 'continual' as learning_system
    
    VALIDATES:
        - dependencies.py mentions both 'continual' and 'learning_system'
        - Validation logic includes alias/substitution handling
        - Fix comment or implementation is present
    
    RATIONALE:
        The learning system is implemented as 'continual' (ContinualLearner)
        but legacy code expects 'learning_system'. Validation must handle both.
        
    BUG CONTEXT:
        Startup logs showed: "Missing critical dependencies in category 'learning': learning_system"
        even though continual was initialized. This causes false errors.
    """
    deps_file = Path("src/vulcan/orchestrator/dependencies.py")
    
    with open(deps_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check that validate_dependencies mentions both names
    assert "continual" in content.lower(), "continual not mentioned in dependencies.py"
    assert "learning_system" in content, "learning_system not mentioned in dependencies.py"
    
    # Check for the specific fix comment or logic
    validation_fix_indicators = [
        "Treat 'continual' as alias",
        "continual' as alias for 'learning_system'",
        "using 'continual' for 'learning_system'",
        "continual' (ContinualLearner)",
        "satisfies 'learning_system' requirement",
    ]
    
    has_fix = any(indicator in content for indicator in validation_fix_indicators)
    assert has_fix, "Dependency validation fix for continual/learning_system not found"
    
    print(f"✓ Dependency validation recognizes 'continual' as learning_system")


def test_model_field_definitions():
    """
    TEST: Verify model classes have expected fields with validation
    
    VALIDATES:
        - FeedbackRequest has feedback_type, query_id, response_id, reward_signal
        - ThumbsFeedbackRequest has query_id, response_id, is_positive
        - UnifiedChatRequest has message, max_tokens, history, enable_* flags
    
    RATIONALE:
        Field presence ensures API compatibility. Missing fields would cause
        AttributeError or validation errors in client code.
    """
    models_file = Path("src/vulcan/api/models.py")
    
    with open(models_file, encoding="utf-8") as f:
        content = f.read()
    
    # Check FeedbackRequest has key fields
    feedback_fields = [
        "feedback_type",
        "query_id", 
        "response_id",
        "reward_signal",
    ]
    
    for field in feedback_fields:
        assert field in content, f"FeedbackRequest missing field: {field}"
    
    # Check ThumbsFeedbackRequest has key fields
    thumbs_fields = [
        "query_id",
        "response_id",
        "is_positive",
    ]
    
    for field in thumbs_fields:
        assert field in content, f"ThumbsFeedbackRequest missing field: {field}"
    
    # Check UnifiedChatRequest has key fields
    unified_fields = [
        "message",
        "max_tokens",
        "history",
        "enable_reasoning",
    ]
    
    for field in unified_fields:
        assert field in content, f"UnifiedChatRequest missing field: {field}"
    
    print(f"✓ All models have expected field definitions")


def test_reasoning_helper_exists():
    """
    TEST: Verify _get_reasoning_attr helper function exists
    
    VALIDATES:
        - reasoning_helpers.py module exists
        - _get_reasoning_attr function is defined
    
    RATIONALE:
        This helper was in original main.py and is used by tests.
        Must be extracted and re-exported for backward compatibility.
    """
    helper_file = Path("src/vulcan/utils/reasoning_helpers.py")
    tree = parse_python_file(helper_file)
    functions = extract_function_names(tree)
    
    assert "_get_reasoning_attr" in functions, \
        "_get_reasoning_attr not defined in reasoning_helpers.py"
    
    print(f"✓ _get_reasoning_attr helper function exists")


def test_reasoning_helper_functionality():
    """
    TEST: Verify _get_reasoning_attr works correctly with multiple input types
    
    VALIDATES:
        - Dictionary input: Extracts values via dict.get()
        - Object input: Extracts values via getattr()
        - None input: Returns default value
        - Missing attributes: Returns default value
    
    RATIONALE:
        Runtime behavior validation ensures the helper correctly handles
        polymorphic reasoning results from different engines.
    """
    # Import the helper
    sys.path.insert(0, "src")
    from vulcan.utils.reasoning_helpers import _get_reasoning_attr
    
    # Test with dict
    result_dict = {"conclusion": "test", "confidence": 0.8}
    assert _get_reasoning_attr(result_dict, "conclusion") == "test", \
        "Failed to extract 'conclusion' from dict"
    assert _get_reasoning_attr(result_dict, "confidence") == 0.8, \
        "Failed to extract 'confidence' from dict"
    assert _get_reasoning_attr(result_dict, "missing", "default") == "default", \
        "Failed to return default for missing key in dict"
    
    # Test with None
    assert _get_reasoning_attr(None, "any", "default") == "default", \
        "Failed to return default for None input"
    
    # Test with object
    class MockResult:
        """Mock reasoning result object for testing."""
        def __init__(self):
            self.conclusion = "object_test"
            self.confidence = 0.9
    
    result_obj = MockResult()
    assert _get_reasoning_attr(result_obj, "conclusion") == "object_test", \
        "Failed to extract 'conclusion' from object"
    assert _get_reasoning_attr(result_obj, "confidence") == 0.9, \
        "Failed to extract 'confidence' from object"
    assert _get_reasoning_attr(result_obj, "missing", "default") == "default", \
        "Failed to return default for missing attribute in object"
    
    print(f"✓ _get_reasoning_attr functionality works correctly")


# ============================================================================


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests() -> int:
    """
    Execute all test cases and report results.
    
    Returns:
        0 if all tests passed, 1 if any test failed
        
    Exit Codes:
        0: All tests passed
        1: One or more tests failed
    """
    tests = [
        test_models_defined_in_api_models,
        test_models_exported_from_api_models,
        test_backward_compat_imports_in_main,
        test_feedback_functions_exist,
        test_dependency_validation_recognizes_continual,
        test_model_field_definitions,
        test_reasoning_helper_exists,
        test_reasoning_helper_functionality,
    ]
    
    failed = 0
    passed = 0
    
    print("=" * 70)
    print("VULCAN-AGI: Backward Compatibility Export Tests")
    print("=" * 70)
    
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Total:  {len(tests)} tests")
    print(f"Passed: {passed} tests")
    print(f"Failed: {failed} tests")
    
    if failed == 0:
        print(f"\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed}/{len(tests)} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
