"""
CRITICAL FIX for test_unified_runtime_integration.py

The problem: Tests create UnifiedRuntime which internally creates services
(RollbackManager, AuditLogger, HardwareDispatcher, EnhancedResourceMonitor)
with default worker intervals (10s), but tests timeout at 60s.

Solution: Use environment variables to configure fast intervals globally for tests.
"""

import os
import sys

# ============================================================================
# CRITICAL: Set fast worker intervals BEFORE importing any modules
# This must be at the TOP of the test file, before any other imports
# ============================================================================

# Set environment variable for test mode - services will check this
os.environ['VULCAN_TEST_MODE'] = '1'
os.environ['WORKER_CHECK_INTERVAL'] = '1.0'
os.environ['HEALTH_CHECK_INTERVAL'] = '1.0'
os.environ['SAMPLING_INTERVAL'] = '0.5'

# Now proceed with regular imports
import asyncio
import pytest
import time
import json
import logging
import threading  # Add this for cleanup verification
from typing import Any, Dict
from pathlib import Path

from unified_runtime.unified_runtime_core import (
    UnifiedRuntime,
    RuntimeConfig,
    get_runtime,
    execute_graph,
    execute_batch,
    async_cleanup,
)
from unified_runtime.execution_engine import ExecutionStatus, ExecutionMode
from unified_runtime.graph_validator import ValidationResult, ValidationError
from unified_runtime.node_handlers import AI_ERRORS

logger = logging.getLogger(__name__)


# ============================================================================
# PYTEST FIXTURES FOR PROPER CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_threads():
    """
    Fixture that runs for every test to ensure proper thread cleanup.
    This helps prevent thread leaks between tests.
    """
    # Record threads at start
    threads_before = threading.active_count()
    
    yield  # Run the test
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Give threads time to cleanup
    time.sleep(0.5)
    
    # Check for thread leaks
    threads_after = threading.active_count()
    if threads_after > threads_before + 2:  # Allow 2 extra threads
        logger.warning(
            f"Potential thread leak: {threads_before} threads before test, "
            f"{threads_after} after. Difference: {threads_after - threads_before}"
        )
        # Log which threads are still alive
        for t in threading.enumerate():
            if t != threading.main_thread():
                logger.warning(f"  Active thread: {t.name} (daemon={t.daemon})")


@pytest.fixture
def fast_runtime_config():
    """
    Create RuntimeConfig optimized for fast test execution.
    """
    return RuntimeConfig(
        execution_timeout_seconds=5.0,  # Reasonable timeout for tests
        learned_subgraphs_dir="learned_subgraphs",
        enable_hardware_dispatch=True,
        enable_metrics=True,
        enable_evolution=False,
    )


# ============================================================================
# REST OF YOUR ORIGINAL TEST FILE GOES HERE
# Just replace the imports section with the above
# ============================================================================

# Your helper functions (_graph_add_only, etc.) go here unchanged
# Your test functions go here, but with modifications:

# MODIFICATION 1: Add explicit cleanup to test_timeout_handling
async def test_timeout_handling():
    """Test execution timeout with proper cleanup."""
    cfg = RuntimeConfig(
        execution_timeout_seconds=0.1,
        learned_subgraphs_dir="learned_subgraphs",
    )
    runtime = UnifiedRuntime(cfg)

    async def slow_node_handler(node, context, inputs):
        await asyncio.sleep(0.5)
        return {"result": "slept"}

    runtime.register_node_type("SLOW_NODE", slow_node_handler)

    g_slow = {
        "nodes": [
            {"id": "slow", "type": "SLOW_NODE"},
            {"id": "out", "type": "OUTPUT"},
        ],
        "edges": [
            {"from": "slow", "to": {"node": "out", "port": "input"}},
        ],
    }

    try:
        result_dict = await runtime.execute_graph(g_slow)
        assert result_dict["status"] == ExecutionStatus.TIMEOUT.value
        assert "_graph" in result_dict.get("errors", {})
        graph_error = result_dict.get("errors", {}).get("_graph", "").lower()
        assert "timed out" in graph_error or "timeout" in graph_error
    finally:
        # CRITICAL: Enhanced cleanup
        runtime.cleanup()
        
        # Give threads time to exit
        await asyncio.sleep(0.2)
        
        # Force any remaining cleanup
        import gc
        gc.collect()


# MODIFICATION 2: Add cleanup verification at end of test suite
@pytest.mark.asyncio
async def test_zzz_final_cleanup_verification():
    """
    Final test (runs last due to 'zzz' prefix) to verify no thread leaks.
    This ensures all previous tests cleaned up properly.
    """
    import gc
    
    # Force cleanup
    gc.collect()
    await asyncio.sleep(1.0)
    
    # Check thread count
    active_threads = threading.active_count()
    thread_list = threading.enumerate()
    
    logger.info(f"Final thread count: {active_threads}")
    for t in thread_list:
        logger.info(f"  - {t.name}: daemon={t.daemon}, alive={t.is_alive()}")
    
    # Should only have main thread + maybe 1-2 daemon threads
    assert active_threads <= 3, (
        f"Thread leak detected: {active_threads} threads still active. "
        f"Threads: {[t.name for t in thread_list]}"
    )
