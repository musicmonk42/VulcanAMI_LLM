"""
Pytest configuration for VULCAN tests.

This conftest.py ensures the src directory is in the Python path
so that `from vulcan.xxx import yyy` style imports work correctly.
"""

import sys
import pathlib
import pytest
import asyncio
import warnings
import threading
import gc
from unittest.mock import Mock

# Add src directory to Python path
ROOT = pathlib.Path(__file__).resolve().parents[3]  # Go up to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Thread leak tolerance - Allow up to 2 extra threads after test
# This accounts for:
# - Temporary daemon threads that may not have exited yet
# - Background cleanup threads from pytest plugins
THREAD_LEAK_TOLERANCE = 2


@pytest.fixture(scope="function", autouse=True)
def cleanup_resources():
    """
    Cleanup resources after each test to prevent leaks and crashes.
    
    This fixture ensures:
    1. Each test starts with a clean event loop state
    2. Any pending tasks are properly cancelled after test
    3. Event loop is properly closed to prevent resource leaks
    4. Orphaned threads are tracked (but not forcefully killed)
    5. Garbage collection runs to free memory
    
    This prevents tests from stopping/crashing when run together.
    """
    # Track threads before test
    initial_thread_count = threading.active_count()
    
    yield
    
    # Cleanup after test
    # 1. Clean up asyncio event loop with timeout to prevent hanging
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            # Cancel all pending tasks
            try:
                pending = asyncio.all_tasks(loop)
            except RuntimeError:
                # If we can't get tasks, that's okay
                pending = []
            
            for task in pending:
                task.cancel()
            
            # Give tasks a moment to cancel - only if loop is not running
            # Use a short timeout to prevent hanging
            if pending and not loop.is_running():
                try:
                    # Use wait_for with timeout to prevent hanging indefinitely
                    loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=2.0
                        )
                    )
                except (RuntimeError, asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    # Loop might be closed, running, tasks failed, or timeout - that's okay in cleanup
                    pass
    except RuntimeError:
        # No event loop in current thread - that's fine
        pass
    except Exception as e:
        # Log but don't fail the test
        warnings.warn(f"Event loop cleanup error: {e}")
    
    # 2. Force garbage collection to free resources
    gc.collect()
    
    # 3. Check for thread leaks (warn only, don't fail)
    final_thread_count = threading.active_count()
    if final_thread_count > initial_thread_count + THREAD_LEAK_TOLERANCE:
        warnings.warn(
            f"Possible thread leak: {initial_thread_count} -> {final_thread_count} threads"
        )


@pytest.fixture(scope="session", autouse=True)
def mock_correlation_tracker_safety_validator():
    """
    Mock EnhancedSafetyValidator for correlation_tracker to prevent spawning 70+ background threads.
    
    This is a session-scoped fixture that runs BEFORE any test module imports,
    ensuring the mock is in place before correlation_tracker can lazy-load the real validator.
    
    Without this mock, each CorrelationTracker instance creates:
    - 50+ rollback_audit rotation_worker threads
    - 10+ rollback_audit cleanup_worker threads  
    - 2+ distributed.py monitor_loop threads
    
    CRITICAL: This must run at session scope in conftest.py to ensure it happens
    before pytest imports test modules (which would trigger lazy imports).
    """
    # Import the module early
    try:
        import vulcan.world_model.correlation_tracker as ct_module
    except ImportError:
        # Module not available, skip mocking
        yield None
        return
    
    # Save whatever is currently there (likely None, but could be already loaded)
    original_validator = ct_module.EnhancedSafetyValidator
    original_config = ct_module.SafetyConfig
    
    # Create mock - DO NOT import the real validator!
    mock_validator_instance = Mock()
    mock_validator_instance.analyze_observation_safety.return_value = {"safe": True}
    mock_validator_instance.validate_state_vector.return_value = {"safe": True}
    mock_validator_instance.clamp_to_safe_region.side_effect = lambda state_vec, *args, **kwargs: state_vec
    
    mock_validator_class = Mock(return_value=mock_validator_instance)
    mock_config_class = Mock()
    
    # Replace with mock BEFORE any lazy import can happen
    # This prevents the lazy import from ever running because
    # the check `if EnhancedSafetyValidator is None` will be False
    ct_module.EnhancedSafetyValidator = mock_validator_class
    ct_module.SafetyConfig = mock_config_class
    
    yield mock_validator_class
    
    # Restore originals after all tests in session
    ct_module.EnhancedSafetyValidator = original_validator
    ct_module.SafetyConfig = original_config


@pytest.fixture(scope="session", autouse=True)
def cleanup_session_resources():
    """
    Cleanup resources at session end to prevent hanging after all tests complete.
    
    This fixture ensures:
    1. All background threads are given time to complete
    2. Event loops are properly closed
    3. Resources are freed before pytest exits
    4. Monitoring threads are stopped
    5. Multiprocessing child processes are terminated
    
    This prevents the test runner from hanging after all tests pass.
    """
    yield
    
    # Cleanup at session end with timeout
    import time
    import multiprocessing
    
    # 1. Terminate any multiprocessing child processes first
    # This is critical to prevent hanging after tests complete
    active_children = multiprocessing.active_children()
    if active_children:
        warnings.warn(
            f"Found {len(active_children)} active child processes during cleanup, terminating..."
        )
        for child in active_children:
            try:
                child.terminate()
            except Exception:
                pass
        
        # Give processes a moment to terminate
        time.sleep(0.2)
        
        # Force kill any that didn't terminate
        remaining = multiprocessing.active_children()
        for child in remaining:
            try:
                child.kill()
                child.join(timeout=0.5)
            except Exception:
                pass
    
    # 2. Stop any EnhancedResourceMonitor threads
    try:
        from vulcan import planning
        # Find and stop all monitor instances
        for obj in gc.get_objects():
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'EnhancedResourceMonitor':
                try:
                    if hasattr(obj, 'stop_monitoring'):
                        obj.stop_monitoring.set()
                    if hasattr(obj, 'cleanup'):
                        obj.cleanup()
                except Exception:
                    pass  # Best effort cleanup
    except Exception:
        pass  # Module might not be loaded
    
    # 3. Force garbage collection to trigger cleanup finalizers
    gc.collect()
    
    # 4. Give background threads a moment to clean up (but don't wait forever)
    time.sleep(0.5)
    
    # 5. Clean up any remaining event loops
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            # Cancel all pending tasks with timeout
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                if pending and not loop.is_running():
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=1.0
                            )
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError, RuntimeError):
                        pass
            except RuntimeError:
                pass
    except RuntimeError:
        pass
    except Exception:
        pass
    
    # 6. Final garbage collection
    gc.collect()
