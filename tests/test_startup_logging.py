"""
Test that startup logging sequence works correctly.
Verifies that all expected INFO logs appear during module import.
"""

import sys
import logging
from io import StringIO
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_startup_logging_sequence():
    """Test that all expected startup logs appear in the correct order."""
    
    # Capture logs
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    try:
        # Import modules in order (this should trigger logging)
        import persistant_memory_v46
        from vulcan.memory import retrieval
        from vulcan import orchestrator
        
        # Get captured logs
        log_output = log_capture.getvalue()
        
        # Verify expected log messages appear
        expected_logs = [
            "persistant_memory_v46",
            "Vulcan Persistent Memory v46.0.0 loaded",
            "vulcan.memory.retrieval",
            "FAISS loaded successfully",
            "vulcan.orchestrator.agent_lifecycle",
            "Agent lifecycle state machine validated successfully",
            "vulcan.orchestrator",
            "ExperimentGenerator components loaded successfully",
            "ProblemExecutor components loaded successfully",
            "Self-improvement system fully available",
            "VULCAN-AGI Orchestrator module loaded successfully"
        ]
        
        for expected in expected_logs:
            assert expected in log_output, f"Expected log message not found: {expected}"
        
        print("✓ All expected startup logs verified")
        return True
        
    finally:
        # Clean up
        root_logger.removeHandler(handler)


if __name__ == "__main__":
    # Run test
    success = test_startup_logging_sequence()
    if success:
        print("SUCCESS: Startup logging test passed")
        sys.exit(0)
    else:
        print("FAILED: Startup logging test failed")
        sys.exit(1)
