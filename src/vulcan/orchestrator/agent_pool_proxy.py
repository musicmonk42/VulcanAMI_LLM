# ============================================================
# VULCAN-AGI Orchestrator - Agent Pool Proxy & Worker Functions
# Extracted from agent_pool.py for modularity
# Contains: is_main_process, _standalone_agent_worker, AgentPoolProxy
# ============================================================

import logging
import multiprocessing
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ============================================================
# ISSUE 2: Windows Multiprocessing Helper Functions
# ============================================================

def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    ISSUE 2 FIX: On Windows, multiprocessing uses 'spawn' mode which creates
    fresh worker processes. This helper distinguishes the main process from
    worker processes to prevent singleton pattern failures.

    Returns:
        True if this is the main process, False if worker process
    """
    return multiprocessing.current_process().name == 'MainProcess'


# ============================================================
# STANDALONE WORKER FUNCTION (MUST BE AT MODULE LEVEL FOR PICKLING)
# ============================================================

def _standalone_agent_worker(agent_id: str):
    """
    Standalone agent worker function - runs in separate process
    FIXED: Must be at module level to be picklable on Windows

    This is a minimal stub that just runs without accessing parent state.
    In a production system, this would communicate via IPC (queues, pipes, etc.)

    Args:
        agent_id: Agent identifier
    """
    worker_logger = logging.getLogger(__name__)
    worker_logger.info(f"Agent {agent_id} worker started (standalone)")

    try:
        # This is intentionally minimal to avoid Windows pickling issues
        # In production, implement proper IPC here (multiprocessing.Queue, etc.)
        while True:
            # Short, non-blocking sleep is fine here since it's just a placeholder loop
            time.sleep(0.1)
            # TODO: Poll for tasks via IPC mechanism
            # TODO: Execute tasks
            # TODO: Report results back via IPC
    except KeyboardInterrupt:
        worker_logger.info("KeyboardInterrupt - graceful shutdown")
    except Exception as e:
        worker_logger.error(f"Agent {agent_id} worker error: {e}")

    worker_logger.info(f"Agent {agent_id} worker stopped")


# ============================================================
# AGENT POOL PROXY FOR WORKER PROCESSES
# ============================================================

class AgentPoolProxy:
    """
    Lightweight proxy for read-only access to AgentPool status from worker processes.

    ISSUE 2 FIX: On Windows with 'spawn' multiprocessing, worker processes get fresh
    copies of modules with empty singleton dictionaries. This proxy provides safe
    read-only access without attempting to instantiate the full AgentPoolManager.

    Industry Standard: Worker processes should never modify shared state or create
    new pool instances. This proxy enforces that constraint by:
    - Providing read-only status queries via IPC (queues/pipes)
    - Raising clear errors if workers try to spawn agents or submit jobs
    - Documenting the main-process-only requirement

    Attributes:
        _main_process_pid: PID of the main process (for validation)
    """

    def __init__(self):
        """
        Initialize proxy for worker process.

        Raises:
            RuntimeError: If called from main process (use AgentPoolManager instead)
        """
        if is_main_process():
            raise RuntimeError(
                "AgentPoolProxy should only be used in worker processes. "
                "Use AgentPoolManager.get_instance() in the main process."
            )

        self._main_process_pid = os.getppid()  # Parent process ID
        logger.info(
            f"[AgentPoolProxy] Initialized in worker process "
            f"(PID={os.getpid()}, parent PID={self._main_process_pid})"
        )

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get read-only pool status.

        Note: This is a stub implementation. In production, this would query
        the main process via IPC (multiprocessing.Queue, Pipe, or shared memory).

        Returns:
            Dictionary with pool status (limited info available to workers)
        """
        logger.warning(
            "[AgentPoolProxy] get_pool_status() called from worker - "
            "returning stub data. Implement IPC for production use."
        )
        return {
            "error": "AgentPoolProxy does not have access to full pool state",
            "worker_pid": os.getpid(),
            "main_process_pid": self._main_process_pid,
            "note": "Workers should communicate via IPC, not access pool directly",
        }

    def spawn_agent(self, *args, **kwargs):
        """
        Spawn agent - NOT ALLOWED from worker processes.

        Raises:
            RuntimeError: Always, as workers cannot spawn agents
        """
        raise RuntimeError(
            "Cannot spawn agents from worker processes. "
            "Agent spawning must be done in the main process via AgentPoolManager."
        )

    def submit_job(self, *args, **kwargs):
        """
        Submit job - NOT ALLOWED from worker processes.

        Raises:
            RuntimeError: Always, as workers cannot submit jobs
        """
        raise RuntimeError(
            "Cannot submit jobs from worker processes. "
            "Job submission must be done in the main process via AgentPoolManager."
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "is_main_process",
    "AgentPoolProxy",
    "_standalone_agent_worker",
]
