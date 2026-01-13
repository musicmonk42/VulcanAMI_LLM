"""
curiosity_driver.py - Active driver for CuriosityEngine with process isolation
Part of the VULCAN-AGI system

Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern

This module provides the CuriosityDriver class that:
1. Makes the CuriosityEngine active via an async heartbeat loop
2. Offloads computationally expensive learning cycles to a separate process
   to avoid blocking the main event loop and causing 100% CPU freeze

The driver uses ProcessPoolExecutor (not ThreadPoolExecutor) to bypass the GIL
and ensure true parallel execution for the CPU-intensive run_learning_cycle method.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class DriverState(Enum):
    """Driver operational states"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class CycleOutcome(Enum):
    """Outcome types for learning cycles"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CuriosityDriverConfig:
    """Configuration for CuriosityDriver"""

    # Heartbeat interval in seconds
    heartbeat_interval: float = 30.0

    # Minimum budget required to run a learning cycle
    min_budget_threshold: float = 10.0

    # Maximum experiments per learning cycle
    max_experiments_per_cycle: int = 5

    # Sleep duration when budget is low (seconds)
    low_budget_sleep: float = 60.0

    # Timeout for each learning cycle (seconds)
    cycle_timeout: float = 300.0

    # Number of worker processes (should be 1 for isolation)
    max_workers: int = 1

    # Maximum history entries to keep
    max_history: int = 1000

    # Cache TTL for statistics (seconds)
    stats_cache_ttl: float = 5.0

    # PERFORMANCE FIX: Idle detection and backoff settings
    # Maximum consecutive empty cycles before entering dormant mode
    # FIX MINOR-2: Increased from 3 to 10 to prevent premature dormant mode
    # FIX OPERATIONAL: Increased from 10 to 30 to allow more exploration before dormant
    max_empty_cycles: int = 30

    # Progressive backoff intervals in seconds when no work is found
    # Applied after consecutive empty cycles: 1st=5s, 2nd=15s, 3rd=60s, 4th+=300s
    backoff_intervals: tuple = (5, 15, 60, 300)

    # Dormant mode check interval (how often to check for new work when dormant)
    dormant_check_interval: float = 300.0  # 5 minutes

    # Minimum time between cycles to prevent CPU thrashing (seconds)
    min_cycle_interval: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "heartbeat_interval": self.heartbeat_interval,
            "min_budget_threshold": self.min_budget_threshold,
            "max_experiments_per_cycle": self.max_experiments_per_cycle,
            "low_budget_sleep": self.low_budget_sleep,
            "cycle_timeout": self.cycle_timeout,
            "max_workers": self.max_workers,
            "max_history": self.max_history,
            "stats_cache_ttl": self.stats_cache_ttl,
            "max_empty_cycles": self.max_empty_cycles,
            "backoff_intervals": list(self.backoff_intervals),
            "dormant_check_interval": self.dormant_check_interval,
            "min_cycle_interval": self.min_cycle_interval,
        }


@dataclass
class CycleResult:
    """Result from a single learning cycle - SEPARATED CONCERN"""

    cycle_id: int
    outcome: CycleOutcome
    experiments_run: int
    successful_experiments: int
    success_rate: float
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    subprocess_pid: Optional[int] = None
    error: Optional[str] = None
    strategy_used: Optional[str] = None
    budget_before: float = 0.0
    budget_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cycle_id": self.cycle_id,
            "outcome": self.outcome.value,
            "experiments_run": self.experiments_run,
            "successful_experiments": self.successful_experiments,
            "success_rate": self.success_rate,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "subprocess_pid": self.subprocess_pid,
            "error": self.error,
            "strategy_used": self.strategy_used,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "metadata": self.metadata,
        }


# =============================================================================
# Statistics Tracking - SEPARATED CONCERN
# =============================================================================


class CycleStatisticsTracker:
    """Tracks learning cycle statistics - SEPARATED CONCERN"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.cycle_history: deque = deque(maxlen=max_history)
        self.outcome_counts: Dict[CycleOutcome, int] = {
            outcome: 0 for outcome in CycleOutcome
        }
        self.total_experiments = 0
        self.total_successes = 0
        self.total_execution_time = 0.0

        # Performance tracking
        self.execution_times: deque = deque(maxlen=100)
        self.success_rates: deque = deque(maxlen=100)

        # Thread safety
        self.lock = threading.RLock()

    def record_cycle(self, result: CycleResult) -> None:
        """Record a cycle result - APPLY pattern"""
        with self.lock:
            try:
                # APPLY: Update counters
                self.cycle_history.append(result)
                self.outcome_counts[result.outcome] += 1
                self.total_experiments += result.experiments_run
                self.total_successes += result.successful_experiments
                self.total_execution_time += result.execution_time

                # REMEMBER: Track performance metrics
                self.execution_times.append(result.execution_time)
                if result.experiments_run > 0:
                    self.success_rates.append(result.success_rate)

            except Exception as e:
                logger.error("Error recording cycle: %s", e)

    def get_average_execution_time(self) -> float:
        """Get average execution time for recent cycles"""
        with self.lock:
            try:
                if self.execution_times:
                    return sum(self.execution_times) / len(self.execution_times)
                return 0.0
            except Exception as e:
                logger.warning("Error calculating average execution time: %s", e)
                return 0.0

    def get_average_success_rate(self) -> float:
        """Get average success rate for recent cycles"""
        with self.lock:
            try:
                if self.success_rates:
                    return sum(self.success_rates) / len(self.success_rates)
                return 0.0
            except Exception as e:
                logger.warning("Error calculating average success rate: %s", e)
                return 0.0

    def get_overall_success_rate(self) -> float:
        """Get overall success rate across all cycles"""
        with self.lock:
            try:
                if self.total_experiments > 0:
                    return self.total_successes / self.total_experiments
                return 0.0
            except Exception as e:
                logger.warning("Error calculating overall success rate: %s", e)
                return 0.0

    def get_recent_history(self, count: int = 10) -> List[CycleResult]:
        """Get recent cycle history"""
        with self.lock:
            try:
                # Return copy to prevent external modification
                # Using slice on list for efficiency with small counts
                history_list = list(self.cycle_history)
                return history_list[-count:] if count < len(history_list) else history_list
            except Exception as e:
                logger.warning("Error getting recent history: %s", e)
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics - EXAMINE pattern"""
        with self.lock:
            try:
                return {
                    "total_cycles": len(self.cycle_history),
                    "total_experiments": self.total_experiments,
                    "total_successes": self.total_successes,
                    "total_execution_time": self.total_execution_time,
                    "outcome_counts": {
                        k.value: v for k, v in self.outcome_counts.items()
                    },
                    "average_execution_time": self.get_average_execution_time(),
                    "average_success_rate": self.get_average_success_rate(),
                    "overall_success_rate": self.get_overall_success_rate(),
                }
            except Exception as e:
                logger.error("Error getting statistics: %s", e)
                return {}

    def reset(self) -> None:
        """Reset all statistics"""
        with self.lock:
            self.cycle_history.clear()
            self.outcome_counts = {outcome: 0 for outcome in CycleOutcome}
            self.total_experiments = 0
            self.total_successes = 0
            self.total_execution_time = 0.0
            self.execution_times.clear()
            self.success_rates.clear()


# =============================================================================
# Process Pool Manager - SEPARATED CONCERN
# =============================================================================


class ProcessPoolManager:
    """Manages process pool lifecycle - SEPARATED CONCERN"""

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.lock = threading.RLock()
        self._active_tasks = 0

    def initialize(self) -> bool:
        """Initialize the process pool - APPLY pattern"""
        with self.lock:
            try:
                if self.process_pool is not None:
                    # Expected behavior for idempotent initialization - use DEBUG level
                    logger.debug("Process pool already initialized")
                    return True

                # Create process pool with specified workers
                # Using max_workers=1 ensures sequential execution and prevents
                # multiple heavy learning cycles from competing for resources
                self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

                logger.info(
                    "Process pool initialized (max_workers=%d)", self.max_workers
                )
                return True

            except Exception as e:
                logger.error("Failed to initialize process pool: %s", e)
                return False

    def shutdown(self, wait: bool = True) -> bool:
        """Shutdown the process pool gracefully"""
        with self.lock:
            try:
                if self.process_pool is None:
                    logger.debug("Process pool not initialized, nothing to shutdown")
                    return True

                logger.info(
                    "Shutting down process pool (wait=%s, active_tasks=%d)",
                    wait,
                    self._active_tasks,
                )

                self.process_pool.shutdown(wait=wait)
                self.process_pool = None
                self._active_tasks = 0

                logger.info("Process pool shutdown complete")
                return True

            except Exception as e:
                logger.error("Error shutting down process pool: %s", e)
                return False

    def is_available(self) -> bool:
        """Check if process pool is available"""
        with self.lock:
            return self.process_pool is not None

    def get_pool(self) -> Optional[ProcessPoolExecutor]:
        """Get the process pool instance"""
        with self.lock:
            return self.process_pool

    def increment_active_tasks(self) -> None:
        """Increment active task counter"""
        with self.lock:
            self._active_tasks += 1

    def decrement_active_tasks(self) -> None:
        """Decrement active task counter"""
        with self.lock:
            self._active_tasks = max(0, self._active_tasks - 1)

    def get_active_task_count(self) -> int:
        """Get number of active tasks"""
        with self.lock:
            return self._active_tasks


# =============================================================================
# Process-safe wrapper for learning cycle execution
# =============================================================================


def _run_cycle_wrapper(engine_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Picklable wrapper function for running learning cycle in subprocess.

    This function is designed to be called in a separate process via
    ProcessPoolExecutor. It re-initializes logging for the subprocess
    and executes the learning cycle.

    Args:
        engine_state: Dictionary containing engine configuration and state
            - max_experiments: Maximum experiments to run
            - base_allocation: Budget base allocation
            - enable_recovery: Whether budget recovery is enabled
            - cycle_id: Identifier for this cycle

    Returns:
        Dictionary containing learning cycle results or error information
    """
    # Re-initialize logging in subprocess (loggers are not picklable)
    import logging
    import sys

    subprocess_logger = logging.getLogger(__name__)
    
    # Note: Subprocess Logging Duplication - prevent propagation to parent loggers
    # This prevents the same log from appearing twice (once from subprocess handler, once from root)
    subprocess_logger.propagate = False

    # Note: Configure subprocess logging to use stdout instead of stderr
    # Previously using default StreamHandler (stderr) caused subprocess logs to appear as errors
    handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - [SUBPROCESS] - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Note: Clear existing handlers and add fresh one to avoid duplication
    subprocess_logger.handlers.clear()
    subprocess_logger.addHandler(handler)
    subprocess_logger.setLevel(logging.INFO)

    cycle_id = engine_state.get("cycle_id", 0)
    subprocess_logger.info(
        "Starting learning cycle %d in subprocess (PID: %d)", cycle_id, os.getpid()
    )

    try:
        # Import here to avoid issues with multiprocessing
        from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine

        # Create fresh engine instance for the subprocess
        # Note: We create a new instance because CuriosityEngine contains
        # non-picklable objects (locks, thread-local storage, etc.)
        engine = CuriosityEngine()

        # APPLY: Apply configuration from engine_state
        if "base_allocation" in engine_state:
            engine.exploration_budget.base_allocation = engine_state["base_allocation"]
        if "enable_recovery" in engine_state:
            engine.exploration_budget.enable_recovery = engine_state["enable_recovery"]

        # Extract parameters
        max_experiments = engine_state.get("max_experiments", 5)

        # Run the computationally expensive learning cycle
        start_time = time.time()
        result = engine.run_learning_cycle(max_experiments=max_experiments)
        execution_time = time.time() - start_time

        subprocess_logger.info(
            "Learning cycle %d completed in %.2fs (experiments: %d, success_rate: %.2f)",
            cycle_id,
            execution_time,
            result.get("experiments_run", 0),
            result.get("success_rate", 0),
        )

        # REMEMBER: Add subprocess metadata
        result["subprocess_pid"] = os.getpid()
        result["execution_time_total"] = execution_time
        result["cycle_id"] = cycle_id

        return result

    except Exception as e:
        subprocess_logger.error(
            "Learning cycle %d failed in subprocess: %s", cycle_id, e
        )
        return {
            "error": str(e),
            "subprocess_pid": os.getpid(),
            "strategy_used": "error",
            "experiments_run": 0,
            "successful_experiments": 0,
            "success_rate": 0.0,
            "cycle_id": cycle_id,
        }


# =============================================================================
# CuriosityDriver class - REFACTORED
# =============================================================================


class CuriosityDriver:
    """
    Active driver for CuriosityEngine with process isolation.

    Refactored to follow EXAMINE → SELECT → APPLY → REMEMBER pattern.

    This class provides:
    1. An async heartbeat loop that periodically triggers learning cycles
    2. Process isolation for CPU-intensive learning cycles via ProcessPoolExecutor
    3. Resource-aware execution that checks budget before running cycles
    4. Comprehensive statistics tracking and history
    5. Graceful shutdown handling with state management

    Usage:
        driver = CuriosityDriver(engine)
        await driver.start()
        # ... application runs ...
        await driver.stop()

    Or as async context manager:
        async with CuriosityDriver(engine) as driver:
            # ... application runs ...

    The driver uses max_workers=1 to ensure that only one learning cycle
    runs at a time, preventing resource contention.
    """

    def __init__(
        self,
        engine: CuriosityEngine,
        config: Optional[CuriosityDriverConfig] = None,
    ):
        """
        Initialize CuriosityDriver.

        Args:
            engine: CuriosityEngine instance to drive
            config: Optional configuration (defaults will be used if not provided)
        """
        self.engine = engine
        self.config = config or CuriosityDriverConfig()

        # Components - SEPARATED CONCERNS
        self._pool_manager = ProcessPoolManager(max_workers=self.config.max_workers)
        self._statistics = CycleStatisticsTracker(max_history=self.config.max_history)

        # Background task handle
        self._heartbeat_task: Optional[asyncio.Task] = None

        # State tracking with thread safety
        self._state = DriverState.STOPPED
        self._state_lock = threading.RLock()
        self._cycle_count = 0
        self._last_cycle_time: Optional[float] = None
        self._last_cycle_result: Optional[CycleResult] = None
        self._consecutive_failures = 0

        # PERFORMANCE FIX: Idle detection and backoff tracking
        # Tracks consecutive cycles that found 0 knowledge gaps
        self._consecutive_empty_cycles = 0
        # Whether the driver is in dormant mode (no work found for extended period)
        self._is_dormant = False
        # Track last time we checked for work (for dormant mode)
        self._last_work_check_time = 0.0
        # Track when we last found actual work
        self._last_work_found_time = 0.0

        # Caching for stats
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_cache_time = 0.0

        # Note: Keep reference to process pool for backward compatibility
        self.process_pool: Optional[ProcessPoolExecutor] = None

        logger.info(
            "CuriosityDriver initialized (heartbeat_interval=%.1fs, "
            "min_budget=%.1f, max_workers=%d, max_empty_cycles=%d)",
            self.config.heartbeat_interval,
            self.config.min_budget_threshold,
            self.config.max_workers,
            self.config.max_empty_cycles,
        )

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, new_state: DriverState) -> None:
        """Set driver state with thread safety"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            logger.debug(
                "State transition: %s -> %s", old_state.value, new_state.value
            )

    def _get_state(self) -> DriverState:
        """Get current driver state"""
        with self._state_lock:
            return self._state

    @property
    def _running(self) -> bool:
        """Check if driver is running (backward compatibility)"""
        return self._get_state() == DriverState.RUNNING

    @_running.setter
    def _running(self, value: bool) -> None:
        """Set running state (backward compatibility)"""
        if value:
            self._set_state(DriverState.RUNNING)
        else:
            self._set_state(DriverState.STOPPED)

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """
        Start the curiosity driver.

        This method:
        1. Validates current state
        2. Initializes the ProcessPoolExecutor
        3. Spawns the background heartbeat loop task

        Raises:
            RuntimeError: If driver is already running or in error state
        """
        current_state = self._get_state()

        # EXAMINE: Check if can start
        if current_state == DriverState.RUNNING:
            logger.warning("CuriosityDriver already running, ignoring start()")
            return

        if current_state == DriverState.STARTING:
            logger.warning("CuriosityDriver already starting, ignoring start()")
            return

        logger.info("Starting CuriosityDriver...")
        self._set_state(DriverState.STARTING)

        try:
            # SELECT & APPLY: Initialize process pool
            if not self._pool_manager.initialize():
                raise RuntimeError("Failed to initialize process pool")

            # Note: Keep reference for backward compatibility
            self.process_pool = self._pool_manager.get_pool()

            # Mark as running before spawning task
            self._set_state(DriverState.RUNNING)

            # APPLY: Spawn heartbeat loop
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="curiosity_heartbeat"
            )

            # REMEMBER: Log successful start
            logger.info("CuriosityDriver started successfully")

        except Exception as e:
            logger.error("Failed to start CuriosityDriver: %s", e)
            self._set_state(DriverState.ERROR)
            raise

    async def stop(self) -> None:
        """
        Stop the curiosity driver gracefully.

        This method:
        1. Cancels the background heartbeat task
        2. Shuts down the process pool (waiting for any running tasks)
        3. Logs final statistics
        """
        current_state = self._get_state()

        # EXAMINE: Check if can stop
        if current_state == DriverState.STOPPED:
            logger.warning("CuriosityDriver not running, ignoring stop()")
            return

        if current_state == DriverState.STOPPING:
            logger.warning("CuriosityDriver already stopping, ignoring stop()")
            return

        logger.info("Stopping CuriosityDriver...")
        self._set_state(DriverState.STOPPING)

        try:
            # SELECT: Cancel heartbeat task
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    logger.debug("Heartbeat task cancelled")

            # APPLY: Shutdown process pool gracefully
            self._pool_manager.shutdown(wait=True)
            self.process_pool = None

            # REMEMBER: Log final statistics
            stats = self._statistics.get_statistics()
            logger.info(
                "CuriosityDriver stopped (cycles=%d, total_experiments=%d, "
                "avg_success_rate=%.2f, total_time=%.2fs)",
                stats.get("total_cycles", 0),
                stats.get("total_experiments", 0),
                stats.get("overall_success_rate", 0.0),
                stats.get("total_execution_time", 0.0),
            )

            self._set_state(DriverState.STOPPED)

        except Exception as e:
            logger.error("Error during CuriosityDriver stop: %s", e)
            self._set_state(DriverState.ERROR)
            raise

    # =========================================================================
    # Heartbeat Loop
    # =========================================================================

    def _should_run_cycle(self) -> tuple:
        """
        Determine if a learning cycle should run.
        
        PERFORMANCE FIX: Implements intelligent cycle management to prevent
        running 320+ empty cycles that waste CPU. This method checks:
        1. Whether we're in dormant mode (no work for extended period)
        2. Progressive backoff based on consecutive empty cycles
        3. Minimum time between cycles to prevent thrashing
        
        Returns:
            Tuple of (should_run: bool, reason: str, sleep_time: float)
            If should_run is False, sleep_time indicates how long to wait
        """
        current_time = time.time()
        
        # Check minimum cycle interval to prevent CPU thrashing
        if self._last_cycle_time:
            time_since_last = current_time - self._last_cycle_time
            if time_since_last < self.config.min_cycle_interval:
                wait_time = self.config.min_cycle_interval - time_since_last
                return (False, "min_interval_not_reached", wait_time)
        
        # Check dormant mode
        if self._is_dormant:
            # In dormant mode, only check periodically for new work
            time_since_check = current_time - self._last_work_check_time
            if time_since_check < self.config.dormant_check_interval:
                wait_time = self.config.dormant_check_interval - time_since_check
                return (False, "dormant_mode", wait_time)
            else:
                # Time to check for new work - will exit dormant if work found
                self._last_work_check_time = current_time
                logger.debug(
                    "Dormant mode: periodic check for new work "
                    f"(empty_cycles={self._consecutive_empty_cycles})"
                )
        
        # Apply progressive backoff based on consecutive empty cycles
        if self._consecutive_empty_cycles > 0:
            backoff_index = min(
                self._consecutive_empty_cycles - 1,
                len(self.config.backoff_intervals) - 1
            )
            backoff_time = self.config.backoff_intervals[backoff_index]
            
            time_since_last = current_time - (self._last_cycle_time or 0)
            if time_since_last < backoff_time:
                wait_time = backoff_time - time_since_last
                return (False, "backoff", wait_time)
        
        return (True, "ok", 0.0)

    def _update_empty_cycle_tracking(self, gaps_found: int, experiments_run: int) -> None:
        """
        Update tracking for empty cycles and manage dormant mode.
        
        PERFORMANCE FIX: Tracks consecutive empty cycles (0 gaps found) and
        enters dormant mode after max_empty_cycles to prevent continuous
        subprocess spawning when there's no work.
        
        Args:
            gaps_found: Number of knowledge gaps identified
            experiments_run: Number of experiments actually run
        """
        current_time = time.time()
        
        if gaps_found == 0 and experiments_run == 0:
            # Empty cycle - increment counter
            self._consecutive_empty_cycles += 1
            
            # FIX OPERATIONAL: Inject synthetic gaps before entering dormant mode
            # This keeps the system learning even when no natural gaps are found
            if (self._consecutive_empty_cycles >= self.config.max_empty_cycles - 5 and 
                not self._is_dormant):
                logger.info(
                    f"[CuriosityDriver] FIX OPERATIONAL: Approaching dormant mode "
                    f"({self._consecutive_empty_cycles}/{self.config.max_empty_cycles} empty cycles). "
                    f"Injecting synthetic gaps to maintain learning activity."
                )
                # Note: Synthetic gap injection happens in the engine, not here
                # This log serves as a diagnostic indicator
            
            # Check if we should enter dormant mode
            if self._consecutive_empty_cycles >= self.config.max_empty_cycles:
                if not self._is_dormant:
                    self._is_dormant = True
                    self._last_work_check_time = current_time
                    logger.info(
                        f"Entering dormant mode after {self._consecutive_empty_cycles} "
                        f"consecutive empty cycles (0 gaps found). "
                        f"Will check for work every {self.config.dormant_check_interval}s"
                    )
        else:
            # Found work - reset tracking and exit dormant mode
            if self._is_dormant:
                logger.info(
                    f"Exiting dormant mode - found {gaps_found} gaps, "
                    f"ran {experiments_run} experiments"
                )
            self._consecutive_empty_cycles = 0
            self._is_dormant = False
            self._last_work_found_time = current_time

    async def _heartbeat_loop(self) -> None:
        """
        Background heartbeat loop that periodically runs learning cycles.

        PERFORMANCE FIX: Now implements intelligent cycle management:
        1. Checks if cycle should run (prevents 320+ empty cycles)
        2. Progressive backoff on consecutive empty cycles
        3. Dormant mode when no work for extended period
        4. Only spawns subprocess when actual work is available

        Follows EXAMINE → SELECT → APPLY → REMEMBER pattern:
        1. EXAMINE: Check budget, state, and work availability
        2. SELECT: Decide whether to run cycle, sleep, or enter dormant
        3. APPLY: Execute learning cycle in subprocess (only if work exists)
        4. REMEMBER: Record results and update statistics
        """
        logger.info("Heartbeat loop started")

        while self._running:
            try:
                # EXAMINE: Check if we should run a cycle
                should_run, reason, wait_time = self._should_run_cycle()
                
                if not should_run:
                    if reason == "dormant_mode":
                        # In dormant mode - long sleep, don't log every iteration
                        pass
                    elif reason == "backoff":
                        logger.debug(
                            f"Backoff active (empty_cycles={self._consecutive_empty_cycles}), "
                            f"waiting {wait_time:.1f}s"
                        )
                    await asyncio.sleep(min(wait_time, self.config.heartbeat_interval))
                    continue
                
                # EXAMINE: Check resource budget
                available_budget = self.engine.exploration_budget.get_available()

                # SELECT: Decide action based on budget
                if available_budget <= self.config.min_budget_threshold:
                    logger.info(
                        "Budget low (%.2f <= %.2f), sleeping for %.1fs",
                        available_budget,
                        self.config.min_budget_threshold,
                        self.config.low_budget_sleep,
                    )

                    # REMEMBER: Record skipped cycle
                    self._record_skipped_cycle(available_budget)

                    await asyncio.sleep(self.config.low_budget_sleep)
                    continue

                # APPLY: Run learning cycle
                logger.debug(
                    "Starting learning cycle (budget=%.2f, cycle=%d, "
                    "empty_cycles=%d, dormant=%s)",
                    available_budget,
                    self._cycle_count + 1,
                    self._consecutive_empty_cycles,
                    self._is_dormant,
                )

                result = await self._run_cycle_async()

                # REMEMBER: Update statistics and empty cycle tracking
                self._process_cycle_result(result, available_budget)
                
                # PERFORMANCE FIX: Update empty cycle tracking
                gaps_found = result.get("gaps_identified", 0)
                experiments_run = result.get("experiments_run", 0)
                self._update_empty_cycle_tracking(gaps_found, experiments_run)

                # Sleep until next heartbeat (modified by backoff if empty)
                if self._consecutive_empty_cycles > 0:
                    # Don't sleep full heartbeat interval if we're backing off
                    # The backoff will be applied on next iteration
                    await asyncio.sleep(self.config.min_cycle_interval)
                else:
                    await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error("Error in heartbeat loop: %s", e, exc_info=True)
                self._consecutive_failures += 1

                # Note: Increase sleep time on consecutive failures
                sleep_time = min(
                    self.config.heartbeat_interval * (1 + self._consecutive_failures),
                    self.config.cycle_timeout,
                )
                await asyncio.sleep(sleep_time)

        logger.info("Heartbeat loop stopped")

    def _record_skipped_cycle(self, available_budget: float) -> None:
        """Record a skipped cycle due to low budget"""
        try:
            result = CycleResult(
                cycle_id=self._cycle_count,
                outcome=CycleOutcome.SKIPPED,
                experiments_run=0,
                successful_experiments=0,
                success_rate=0.0,
                execution_time=0.0,
                budget_before=available_budget,
                budget_after=available_budget,
                metadata={"reason": "low_budget"},
            )
            self._statistics.record_cycle(result)
        except Exception as e:
            logger.warning("Error recording skipped cycle: %s", e)

    def _process_cycle_result(
        self, raw_result: Dict[str, Any], budget_before: float
    ) -> None:
        """Process and record a cycle result - REMEMBER pattern"""
        try:
            self._cycle_count += 1
            self._last_cycle_time = time.time()

            # Determine outcome
            if "error" in raw_result:
                outcome = CycleOutcome.FAILED
                self._consecutive_failures += 1
            elif raw_result.get("experiments_run", 0) == 0:
                outcome = CycleOutcome.PARTIAL
                self._consecutive_failures = 0
            else:
                outcome = CycleOutcome.SUCCESS
                self._consecutive_failures = 0

            # Create structured result
            cycle_result = CycleResult(
                cycle_id=self._cycle_count,
                outcome=outcome,
                experiments_run=raw_result.get("experiments_run", 0),
                successful_experiments=raw_result.get("successful_experiments", 0),
                success_rate=raw_result.get("success_rate", 0.0),
                execution_time=raw_result.get("execution_time_total", 0.0),
                subprocess_pid=raw_result.get("subprocess_pid"),
                error=raw_result.get("error"),
                strategy_used=raw_result.get("strategy_used"),
                budget_before=budget_before,
                budget_after=self.engine.exploration_budget.get_available(),
                metadata=raw_result.get("metadata", {}),
            )

            # Record in statistics tracker
            self._statistics.record_cycle(cycle_result)
            self._last_cycle_result = cycle_result

            # Invalidate stats cache
            self._stats_cache = None

            # Log result - only INFO if something meaningful happened (success_rate > 0 or experiments > 0)
            if outcome == CycleOutcome.SUCCESS:
                if cycle_result.success_rate > 0 or cycle_result.experiments_run > 0:
                    logger.info(
                        "Learning cycle %d completed: experiments=%d, "
                        "success_rate=%.2f, time=%.2fs",
                        self._cycle_count,
                        cycle_result.experiments_run,
                        cycle_result.success_rate,
                        cycle_result.execution_time,
                    )
                else:
                    # Stay silent at INFO level if nothing happened (0 gaps)
                    logger.debug(
                        "Learning cycle %d completed: no experiments (0 gaps), time=%.2fs",
                        self._cycle_count,
                        cycle_result.execution_time,
                    )
            elif outcome == CycleOutcome.FAILED:
                logger.error(
                    "Learning cycle %d failed: %s",
                    self._cycle_count,
                    cycle_result.error,
                )

        except Exception as e:
            logger.error("Error processing cycle result: %s", e)

    # =========================================================================
    # Cycle Execution
    # =========================================================================

    async def _run_cycle_async(self) -> Dict[str, Any]:
        """
        Run learning cycle asynchronously in subprocess.

        This method uses asyncio.get_running_loop().run_in_executor() to
        offload the heavy computation to the process pool without blocking
        the main event loop.

        Returns:
            Dictionary containing learning cycle results
        """
        # EXAMINE: Validate state
        if not self._pool_manager.is_available():
            logger.error("Process pool not initialized")
            return {"error": "Process pool not initialized"}

        pool = self._pool_manager.get_pool()
        if pool is None:
            logger.error("Process pool is None")
            return {"error": "Process pool is None"}

        # SELECT: Prepare engine state for subprocess
        engine_state = {
            "max_experiments": self.config.max_experiments_per_cycle,
            "base_allocation": self.engine.exploration_budget.base_allocation,
            "enable_recovery": self.engine.exploration_budget.enable_recovery,
            "cycle_id": self._cycle_count + 1,
        }

        try:
            # APPLY: Run in process pool
            self._pool_manager.increment_active_tasks()

            loop = asyncio.get_running_loop()

            # Run the wrapper function in the process pool
            # This is non-blocking and will not freeze the main thread
            result = await asyncio.wait_for(
                loop.run_in_executor(pool, _run_cycle_wrapper, engine_state),
                timeout=self.config.cycle_timeout,
            )

            return result

        except asyncio.TimeoutError:
            logger.error(
                "Learning cycle timed out after %.1fs", self.config.cycle_timeout
            )
            return {
                "error": f"Timeout after {self.config.cycle_timeout}s",
                "cycle_id": engine_state["cycle_id"],
            }
        except Exception as e:
            logger.error("Error running learning cycle: %s", e)
            return {"error": str(e), "cycle_id": engine_state["cycle_id"]}
        finally:
            self._pool_manager.decrement_active_tasks()

    # =========================================================================
    # Async context manager support
    # =========================================================================

    async def __aenter__(self) -> "CuriosityDriver":
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.stop()

    # =========================================================================
    # Status and Statistics
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if driver is currently running."""
        return self._running

    @property
    def state(self) -> DriverState:
        """Get current driver state."""
        return self._get_state()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get driver statistics with caching.

        Returns:
            Dictionary containing:
            - running: Whether driver is active
            - state: Current driver state
            - cycle_count: Number of completed cycles
            - total_experiments: Total experiments across all cycles
            - total_successes: Total successful experiments
            - average_success_rate: Overall success rate
            - average_cycle_time: Average time per cycle
            - last_cycle_time: Timestamp of last cycle
            - last_cycle_result: Result of most recent cycle
            - consecutive_failures: Number of consecutive failed cycles
            - consecutive_empty_cycles: Number of cycles with 0 gaps found
            - is_dormant: Whether driver is in dormant mode
            - config: Driver configuration
            - statistics: Detailed statistics from tracker
        """
        # EXAMINE: Check cache
        current_time = time.time()
        if (
            self._stats_cache is not None
            and current_time - self._stats_cache_time < self.config.stats_cache_ttl
        ):
            return self._stats_cache

        # SELECT & APPLY: Build stats
        try:
            detailed_stats = self._statistics.get_statistics()

            stats = {
                "running": self._running,
                "state": self._get_state().value,
                "cycle_count": self._cycle_count,
                "total_experiments": detailed_stats.get("total_experiments", 0),
                "total_successes": detailed_stats.get("total_successes", 0),
                "average_success_rate": detailed_stats.get("overall_success_rate", 0.0),
                "average_cycle_time": detailed_stats.get("average_execution_time", 0.0),
                "last_cycle_time": self._last_cycle_time,
                "last_cycle_result": (
                    self._last_cycle_result.to_dict()
                    if self._last_cycle_result
                    else None
                ),
                "consecutive_failures": self._consecutive_failures,
                # PERFORMANCE FIX: Include idle detection metrics
                "consecutive_empty_cycles": self._consecutive_empty_cycles,
                "is_dormant": self._is_dormant,
                "last_work_found_time": self._last_work_found_time,
                "active_tasks": self._pool_manager.get_active_task_count(),
                "config": self.config.to_dict(),
                "statistics": detailed_stats,
            }

            # REMEMBER: Cache result
            self._stats_cache = stats
            self._stats_cache_time = current_time

            return stats

        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return {"error": str(e), "running": self._running}

    def get_recent_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycle history as dictionaries"""
        try:
            history = self._statistics.get_recent_history(count)
            return [r.to_dict() for r in history]
        except Exception as e:
            logger.error("Error getting recent history: %s", e)
            return []

    def reset_statistics(self) -> None:
        """Reset all statistics (useful for testing or fresh start)"""
        try:
            self._statistics.reset()
            self._cycle_count = 0
            self._consecutive_failures = 0
            self._last_cycle_result = None
            self._stats_cache = None
            # PERFORMANCE FIX: Also reset idle detection tracking
            self._consecutive_empty_cycles = 0
            self._is_dormant = False
            self._last_work_check_time = 0.0
            self._last_work_found_time = 0.0
            logger.info("Statistics reset (including idle detection state)")
        except Exception as e:
            logger.error("Error resetting statistics: %s", e)
    
    def wake_from_dormant(self) -> None:
        """
        Wake the driver from dormant mode.
        
        PERFORMANCE FIX: Allows external components to wake the driver
        when new work is available, without waiting for the dormant check
        interval. Call this when new queries/data arrive.
        """
        if self._is_dormant:
            logger.info("Driver woken from dormant mode by external trigger")
            self._is_dormant = False
            self._consecutive_empty_cycles = 0
            self._last_work_check_time = 0.0  # Allow immediate cycle
    
    @property
    def is_dormant(self) -> bool:
        """Check if driver is in dormant mode."""
        return self._is_dormant
    
    @property
    def consecutive_empty_cycles(self) -> int:
        """Get number of consecutive empty cycles."""
        return self._consecutive_empty_cycles


# =============================================================================
# Factory function for creating configured driver
# =============================================================================


def create_curiosity_driver(
    engine: CuriosityEngine,
    heartbeat_interval: float = 30.0,
    min_budget_threshold: float = 10.0,
    max_experiments_per_cycle: int = 5,
) -> CuriosityDriver:
    """
    Factory function to create a configured CuriosityDriver.

    Args:
        engine: CuriosityEngine instance to drive
        heartbeat_interval: Seconds between heartbeats
        min_budget_threshold: Minimum budget to run a cycle
        max_experiments_per_cycle: Max experiments per cycle

    Returns:
        Configured CuriosityDriver instance
    """
    config = CuriosityDriverConfig(
        heartbeat_interval=heartbeat_interval,
        min_budget_threshold=min_budget_threshold,
        max_experiments_per_cycle=max_experiments_per_cycle,
    )
    return CuriosityDriver(engine, config)
