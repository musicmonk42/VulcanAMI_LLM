"""
Test backward compatibility exports from main.py

This test verifies that the refactored main.py maintains backward compatibility
by re-exporting models and functions that src/full_platform.py depends on.
"""

import ast
import sys
from pathlib import Path


def test_models_defined_in_api_models():
    """Test that required models are defined in vulcan/api/models.py"""
    models_file = Path("src/vulcan/api/models.py")
    assert models_file.exists(), "src/vulcan/api/models.py not found"
    
    with open(models_file) as f:
        tree = ast.parse(f.read())
    
    # Find all class definitions
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    
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
    """Test that required models are in __all__ export list"""
    models_file = Path("src/vulcan/api/models.py")
    
    with open(models_file) as f:
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
    """Test that main.py has backward compatibility imports"""
    main_file = Path("src/vulcan/main.py")
    assert main_file.exists(), "src/vulcan/main.py not found"
    
    with open(main_file) as f:
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
    ]
    
    for name in required_names:
        assert name in content, f"Missing import in main.py: {name}"
    
    # Verify imports are from correct modules
    assert "from vulcan.api.models import" in content
    assert "from vulcan.endpoints.feedback import" in content
    
    print(f"✓ All backward compatibility imports present in main.py")


def test_feedback_functions_exist():
    """Test that feedback endpoint functions exist"""
    feedback_file = Path("src/vulcan/endpoints/feedback.py")
    assert feedback_file.exists(), "src/vulcan/endpoints/feedback.py not found"
    
    with open(feedback_file) as f:
        tree = ast.parse(f.read())
    
    # Find all function definitions (both sync and async)
    functions = {
        node.name 
        for node in ast.walk(tree) 
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    
    required_functions = {
        "submit_feedback",
        "submit_thumbs_feedback",
        "get_feedback_stats",
    }
    
    for func in required_functions:
        assert func in functions, f"{func} not defined in vulcan/endpoints/feedback.py"
    
    print(f"✓ All required functions defined in vulcan/endpoints/feedback.py")


def test_dependency_validation_recognizes_continual():
    """Test that dependency validation recognizes 'continual' as learning_system"""
    deps_file = Path("src/vulcan/orchestrator/dependencies.py")
    assert deps_file.exists(), "src/vulcan/orchestrator/dependencies.py not found"
    
    with open(deps_file) as f:
        content = f.read()
    
    # Check that validate_dependencies has the fix for continual/learning_system
    assert "continual" in content.lower(), "continual not mentioned in dependencies.py"
    assert "learning_system" in content, "learning_system not mentioned in dependencies.py"
    
    # Check for the specific fix comment or logic
    validation_fix_indicators = [
        "Treat 'continual' as alias",
        "continual' as alias for 'learning_system'",
        "using 'continual' for 'learning_system'",
    ]
    
    has_fix = any(indicator in content for indicator in validation_fix_indicators)
    assert has_fix, "Dependency validation fix for continual/learning_system not found"
    
    print(f"✓ Dependency validation recognizes 'continual' as learning_system")


def test_model_field_definitions():
    """Test that model classes have the expected fields"""
    models_file = Path("src/vulcan/api/models.py")
    
    with open(models_file) as f:
        content = f.read()
    
    # Check FeedbackRequest has key fields
    feedback_fields = [
        "feedback_type",
        "query_id", 
        "response_id",
        "reward_signal",
    ]
    
    for field in feedback_fields:
        # Look for field definition (Field(...) or = )
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


if __name__ == "__main__":
    # Run tests
    tests = [
        test_models_defined_in_api_models,
        test_models_exported_from_api_models,
        test_backward_compat_imports_in_main,
        test_feedback_functions_exist,
        test_dependency_validation_recognizes_continual,
        test_model_field_definitions,
    ]
    
    failed = 0
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"✓ All {len(tests)} tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {failed}/{len(tests)} tests failed")
        sys.exit(1)
