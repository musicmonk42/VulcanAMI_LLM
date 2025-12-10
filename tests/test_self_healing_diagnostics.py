"""
Test for self-healing diagnostics and verification

This test verifies that the self-healing diagnostic functions work correctly
and that required methods are present in the WorldModel class.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_worldmodel_has_required_methods():
    """Test that WorldModel has all required self-healing methods"""
    from vulcan.world_model.world_model_core import WorldModel
    
    required_methods = [
        '_handle_improvement_alert',
        '_check_improvement_approval',
        'start_autonomous_improvement',
        'stop_autonomous_improvement',
        'get_improvement_status',
        'report_error',
        'update_performance_metric',
        '_execute_improvement',
        '_build_improvement_context'
    ]
    
    for method_name in required_methods:
        assert hasattr(WorldModel, method_name), \
            f"WorldModel missing required method: {method_name}"
        
        method = getattr(WorldModel, method_name)
        assert callable(method), \
            f"{method_name} exists but is not callable"
    
    print("✓ All required methods exist and are callable")


def test_validate_self_healing_setup():
    """Test the validate_self_healing_setup diagnostic function"""
    from vulcan.world_model.world_model_core import validate_self_healing_setup
    
    is_working, issues = validate_self_healing_setup()
    
    # The setup should be working (all methods exist)
    assert is_working, f"Self-healing setup validation failed with issues: {issues}"
    assert len(issues) == 0, f"Unexpected issues found: {issues}"
    
    print("✓ validate_self_healing_setup() passed")


def test_diagnostic_functions_callable():
    """Test that diagnostic functions are callable"""
    from vulcan.world_model.world_model_core import (
        print_diagnostics, print_self_healing_diagnostics,
        validate_component_installation, validate_self_healing_setup)

    # Test validate_self_healing_setup
    is_working, issues = validate_self_healing_setup()
    assert isinstance(is_working, bool)
    assert isinstance(issues, list)
    
    # Test that print functions don't raise exceptions
    # (we don't capture output, just verify they run)
    try:
        print_self_healing_diagnostics()
        print("✓ print_self_healing_diagnostics() executed successfully")
    except Exception as e:
        raise AssertionError(f"print_self_healing_diagnostics() failed: {e}")
    
    try:
        print_diagnostics()
        print("✓ print_diagnostics() executed successfully")
    except Exception as e:
        raise AssertionError(f"print_diagnostics() failed: {e}")
    
    # Test validate_component_installation
    all_available, missing = validate_component_installation()
    assert isinstance(all_available, bool)
    assert isinstance(missing, list)
    
    print("✓ All diagnostic functions are callable")


def test_method_signatures():
    """Test that required methods have correct signatures"""
    import inspect

    from vulcan.world_model.world_model_core import WorldModel

    # Test _handle_improvement_alert signature
    method = getattr(WorldModel, '_handle_improvement_alert')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    
    # Should have: self, severity, alert_data
    assert 'self' in params, "_handle_improvement_alert missing 'self' parameter"
    assert 'severity' in params, "_handle_improvement_alert missing 'severity' parameter"
    assert 'alert_data' in params, "_handle_improvement_alert missing 'alert_data' parameter"
    
    # Test _check_improvement_approval signature
    method = getattr(WorldModel, '_check_improvement_approval')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    
    # Should have: self, approval_id
    assert 'self' in params, "_check_improvement_approval missing 'self' parameter"
    assert 'approval_id' in params, "_check_improvement_approval missing 'approval_id' parameter"
    
    print("✓ Method signatures are correct")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Self-Healing Diagnostics Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("WorldModel has required methods", test_worldmodel_has_required_methods),
        ("validate_self_healing_setup works", test_validate_self_healing_setup),
        ("Diagnostic functions are callable", test_diagnostic_functions_callable),
        ("Method signatures are correct", test_method_signatures),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
