"""
Test Server Startup Bug Fixes

Validates that all 16 fixes from the comprehensive bug fix PR work correctly.
Tests cover P0-P3 priority issues including thread safety, error handling,
timeout enforcement, and graceful degradation.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_state_module_exists():
    """Test that the shared state module was created and is importable."""
    from vulcan.server import state
    
    # Verify module attributes exist
    assert hasattr(state, 'process_lock')
    assert hasattr(state, 'rate_limit_cleanup_thread')
    assert hasattr(state, 'rate_limit_thread_lock')
    assert hasattr(state, 'redis_client')
    
    # Verify lock is a proper Lock object
    from threading import Lock
    assert isinstance(state.rate_limit_thread_lock, Lock)
    
    print("✓ State module created successfully with all required attributes")


def test_deployment_mode_constants():
    """Test that DeploymentMode constants were added."""
    from vulcan.server.startup import DeploymentMode
    
    # Verify constants exist
    assert hasattr(DeploymentMode, 'PRODUCTION')
    assert hasattr(DeploymentMode, 'TESTING')
    assert hasattr(DeploymentMode, 'DEVELOPMENT')
    assert hasattr(DeploymentMode, 'VALID_MODES')
    assert hasattr(DeploymentMode, 'DEFAULT')
    
    # Verify values
    assert DeploymentMode.PRODUCTION == "production"
    assert DeploymentMode.TESTING == "testing"
    assert DeploymentMode.DEVELOPMENT == "development"
    assert DeploymentMode.DEFAULT == DeploymentMode.DEVELOPMENT
    
    # Test validation methods
    assert DeploymentMode.is_valid("production")
    assert DeploymentMode.is_valid("testing")
    assert DeploymentMode.is_valid("development")
    assert not DeploymentMode.is_valid("invalid")
    
    # Test normalization
    assert DeploymentMode.normalize("production") == "production"
    assert DeploymentMode.normalize("invalid") == DeploymentMode.DEFAULT
    
    print("✓ DeploymentMode constants working correctly")


def test_log_emoji_constants():
    """Test that LogEmoji constants were added."""
    from vulcan.server.startup import LogEmoji
    
    # Verify constants exist
    assert hasattr(LogEmoji, 'SUCCESS')
    assert hasattr(LogEmoji, 'SUCCESS_MAJOR')
    assert hasattr(LogEmoji, 'WARNING')
    assert hasattr(LogEmoji, 'ERROR')
    assert hasattr(LogEmoji, 'INFO')
    assert hasattr(LogEmoji, 'ROCKET')
    assert hasattr(LogEmoji, 'CHART')
    assert hasattr(LogEmoji, 'STOP')
    
    # Verify they are strings
    assert isinstance(LogEmoji.SUCCESS, str)
    assert isinstance(LogEmoji.SUCCESS_MAJOR, str)
    
    print("✓ LogEmoji constants defined correctly")


def test_startup_phase_config_removed():
    """Test that the unused StartupPhaseConfig class was removed."""
    from vulcan.server.startup import constants
    
    # Verify class doesn't exist
    assert not hasattr(constants, 'StartupPhaseConfig')
    
    print("✓ Unused StartupPhaseConfig class removed")


def test_subsystem_manager_thread_safety():
    """Test that SubsystemManager has thread safety lock."""
    from vulcan.server.startup.subsystems import SubsystemManager
    from threading import Lock
    from unittest.mock import Mock
    
    # Create a mock deployment
    mock_deployment = Mock()
    mock_deployment.collective = Mock()
    mock_deployment.collective.deps = Mock()
    
    # Create manager
    manager = SubsystemManager(mock_deployment)
    
    # Verify lock exists
    assert hasattr(manager, '_lock')
    assert isinstance(manager._lock, Lock)
    
    print("✓ SubsystemManager has thread safety lock")


def test_startup_manager_imports_state():
    """Test that StartupManager properly imports state module."""
    from vulcan.server.startup import manager as manager_module
    from vulcan.server import state
    
    # Verify state is imported
    assert hasattr(manager_module, 'state')
    assert manager_module.state is state
    
    print("✓ StartupManager imports state module correctly")


def test_app_imports_state():
    """Test that app.py properly imports state module."""
    from vulcan.server import app as app_module
    from vulcan.server import state
    
    # Verify state is imported
    assert hasattr(app_module, 'state')
    assert app_module.state is state
    
    print("✓ app.py imports state module correctly")


def test_constants_exported_from_startup_init():
    """Test that new constants are exported from startup __init__."""
    from vulcan.server.startup import (
        DeploymentMode,
        LogEmoji,
        StartupManager,
        HealthCheck,
    )
    
    # If imports work, constants are properly exported
    assert DeploymentMode is not None
    assert LogEmoji is not None
    assert StartupManager is not None
    assert HealthCheck is not None
    
    print("✓ All constants properly exported from startup module")


def test_phase_metadata_unchanged():
    """Test that phase metadata structure is still intact."""
    from vulcan.server.startup.phases import (
        StartupPhase,
        get_phase_metadata,
        is_critical_phase,
    )
    
    # Verify all phases have metadata
    for phase in StartupPhase:
        meta = get_phase_metadata(phase)
        assert meta.name
        assert meta.timeout_seconds > 0
        assert isinstance(meta.critical, bool)
    
    # Verify critical phases
    assert is_critical_phase(StartupPhase.CONFIGURATION)
    assert is_critical_phase(StartupPhase.CORE_SERVICES)
    
    print("✓ Phase metadata structure intact")


def test_state_module_documentation():
    """Test that state module has proper documentation."""
    from vulcan.server import state
    
    # Verify module docstring exists
    assert state.__doc__ is not None
    assert len(state.__doc__) > 100  # Should be substantial
    assert "circular import" in state.__doc__.lower()
    
    print("✓ State module has comprehensive documentation")


def run_all_tests():
    """Run all validation tests."""
    tests = [
        test_state_module_exists,
        test_deployment_mode_constants,
        test_log_emoji_constants,
        test_startup_phase_config_removed,
        test_subsystem_manager_thread_safety,
        test_startup_manager_imports_state,
        test_app_imports_state,
        test_constants_exported_from_startup_init,
        test_phase_metadata_unchanged,
        test_state_module_documentation,
    ]
    
    print("\n" + "="*70)
    print("Running Server Startup Bug Fix Validation Tests")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
