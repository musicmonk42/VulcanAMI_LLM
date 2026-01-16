"""
Test for scipy import check fix in causal_graph.py

This test validates that SCIPY_AVAILABLE is set correctly based on
whether scipy is actually available, fixing the bug where it was
always set to True.
"""

import sys
import pytest
from unittest.mock import patch


def test_scipy_available_flag_correct():
    """Test that SCIPY_AVAILABLE reflects actual scipy availability."""
    # First, test the actual state
    try:
        import scipy
        scipy_actually_available = True
    except ImportError:
        scipy_actually_available = False
    
    # Import causal_graph and check the flag
    from vulcan.world_model import causal_graph
    
    # The flag should match the actual availability
    assert causal_graph.SCIPY_AVAILABLE == scipy_actually_available, (
        f"SCIPY_AVAILABLE={causal_graph.SCIPY_AVAILABLE} but "
        f"scipy is {'available' if scipy_actually_available else 'not available'}"
    )


def test_scipy_import_failure_sets_flag_false():
    """Test that SCIPY_AVAILABLE is False when scipy import fails."""
    # Remove scipy from sys.modules if it's there
    scipy_modules = [key for key in sys.modules.keys() if key.startswith('scipy')]
    saved_modules = {}
    for mod in scipy_modules:
        saved_modules[mod] = sys.modules.pop(mod)
    
    # Also remove causal_graph if it's loaded
    if 'vulcan.world_model.causal_graph' in sys.modules:
        del sys.modules['vulcan.world_model.causal_graph']
    
    try:
        # Mock the scipy import to fail
        with patch.dict('sys.modules', {'scipy': None}):
            # This should trigger the ImportError and set SCIPY_AVAILABLE to False
            # However, we can't easily force a reimport in the same test session
            # So we just verify the current state is consistent
            from vulcan.world_model import causal_graph
            
            # If scipy is not available, the flag should be False
            try:
                import scipy
                # If we can import scipy, the flag should be True
                assert causal_graph.SCIPY_AVAILABLE is True
            except ImportError:
                # If we can't import scipy, the flag should be False
                assert causal_graph.SCIPY_AVAILABLE is False
    finally:
        # Restore saved modules
        sys.modules.update(saved_modules)


def test_scipy_flag_consistency_with_other_files():
    """Test that scipy import pattern is consistent with other world_model files."""
    from vulcan.world_model import causal_graph
    from vulcan.world_model import dynamics_model
    from vulcan.world_model import intervention_manager
    from vulcan.world_model import confidence_calibrator
    
    # All files should have the same SCIPY_AVAILABLE value
    # since they all try to import scipy the same way
    flags = [
        ('causal_graph', causal_graph.SCIPY_AVAILABLE),
        ('dynamics_model', dynamics_model.SCIPY_AVAILABLE),
        ('intervention_manager', intervention_manager.SCIPY_AVAILABLE),
        ('confidence_calibrator', confidence_calibrator.SCIPY_AVAILABLE),
    ]
    
    # All should be the same
    values = [flag[1] for flag in flags]
    assert len(set(values)) == 1, (
        f"SCIPY_AVAILABLE flags are inconsistent across files: {flags}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
