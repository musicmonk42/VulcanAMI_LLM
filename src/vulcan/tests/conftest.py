"""
Pytest configuration for VULCAN tests.

This conftest.py ensures the src directory is in the Python path
so that `from vulcan.xxx import yyy` style imports work correctly.
"""

import sys
import os
import pathlib
import pytest
import asyncio
import warnings
import threading
import gc
import uuid
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


@pytest.fixture(autouse=True)
def reset_environment_state(tmp_path, monkeypatch):
    """
    Reset environment variables before each test to prevent state contamination.
    
    This fixes:
    - CSIU state contamination where _csiu_regs_enabled is False when it should be True
    - Environment variables persisting between tests
    - Database locking by ensuring unique storage paths per test
    """
    # Remove CSIU environment variables to ensure clean state
    # monkeypatch automatically restores original values after test
    env_vars_to_reset = [
        'INTRINSIC_CSIU_OFF',
        'INTRINSIC_CSIU_REGS_OFF',
        'INTRINSIC_CSIU_CALC_OFF',
    ]
    
    for var in env_vars_to_reset:
        monkeypatch.delenv(var, raising=False)
    
    # Set unique storage path for this test to prevent database conflicts
    # This ensures each test gets its own isolated database
    test_storage_path = tmp_path / f"test_storage_{uuid.uuid4().hex}"
    test_storage_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('VULCAN_STORAGE_PATH', str(test_storage_path))
    
    yield
    # monkeypatch automatically restores environment on cleanup


@pytest.fixture
def isolated_db_path(tmp_path):
    """
    Create isolated temporary database path for each test.
    
    This fixes:
    - SQLite database locking issues from concurrent access
    - "database is locked" errors during teardown/setup overlap
    
    Each test gets its own unique database file to prevent contention.
    """
    db_file = tmp_path / f"test_{uuid.uuid4().hex}.db"
    return str(db_file)


@pytest.fixture
def fresh_pytorch_model():
    """
    Create a fresh PyTorch model in training mode for each test.
    
    This fixes:
    - PyTorch gradient state contamination
    - "element 0 of tensors does not require grad" errors
    - Models stuck in eval() mode from previous tests
    
    The model is explicitly set to train() mode and is NOT shared across tests.
    """
    try:
        import torch
        import torch.nn as nn
        
        class FreshTestModel(nn.Module):
            def __init__(self, input_dim=512, hidden_dim=256):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, input_dim)
            
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        model = FreshTestModel()
        model.train()  # Explicitly set to train mode
        return model
    except ImportError:
        # PyTorch not available, return None
        return None


@pytest.fixture
def fresh_tensors():
    """
    Create fresh PyTorch tensors with gradients enabled for each test.
    
    This fixes:
    - Tensor gradient state contamination
    - Reused tensors without requires_grad
    - Shared tensor instances across tests
    
    Each test gets NEW tensors with requires_grad=True.
    """
    try:
        import torch
        
        def create_tensor(shape, requires_grad=True):
            """Create a new tensor with specified shape and gradient tracking"""
            return torch.randn(*shape, requires_grad=requires_grad)
        
        return create_tensor
    except ImportError:
        # PyTorch not available, return None
        return None


@pytest.fixture(autouse=True)
def reset_pytorch_state():
    """
    Reset PyTorch state before and after each test.
    
    This fixes:
    - Models left in eval() mode by previous tests
    - Tensors with .detach() called on shared instances
    - Gradient state contamination from torch.no_grad() or .eval() calls
    - Global gradient state being disabled
    
    This is applied automatically to all tests.
    """
    try:
        import torch
        
        # CRITICAL: Explicitly enable gradients before each test
        # This prevents "element 0 of tensors does not require grad" errors
        # when tests run together (test pollution from previous tests)
        torch.set_grad_enabled(True)
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set default dtype to float32 to ensure consistency
        torch.set_default_dtype(torch.float32)
        
        yield
        
        # CRITICAL: Re-enable gradients after each test as well
        # This ensures the next test starts with a clean gradient state
        torch.set_grad_enabled(True)
        
        # Cleanup after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except ImportError:
        # PyTorch not available, skip
        yield


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
