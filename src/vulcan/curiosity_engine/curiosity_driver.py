"""
curiosity_driver.py - Active driver for CuriosityEngine with process isolation

Part of the VULCAN-AGI system

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
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine

logger = logging.getLogger(__name__)


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

    Returns:
        Dictionary containing learning cycle results or error information
    """
    # Re-initialize logging in subprocess (loggers are not picklable)
    import logging

    subprocess_logger = logging.getLogger(__name__)

    # Configure subprocess logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - [SUBPROCESS] - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers
    if not subprocess_logger.handlers:
        subprocess_logger.addHandler(handler)
        subprocess_logger.setLevel(logging.INFO)

    subprocess_logger.info(
        "Starting learning cycle in subprocess (PID: %d)", os.getpid()
    )

    try:
        # Import here to avoid issues with multiprocessing
        from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine

        # Create fresh engine instance for the subprocess
        # Note: We create a new instance because CuriosityEngine contains
        # non-picklable objects (locks, thread-local storage, etc.)
        engine = CuriosityEngine()

        # Apply any configuration from engine_state
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
            "Learning cycle completed in %.2fs (experiments: %d, success_rate: %.2f)",
            execution_time,
            result.get("experiments_run", 0),
            result.get("success_rate", 0),
        )

        # Add subprocess metadata
        result["subprocess_pid"] = os.getpid()
        result["execution_time_total"] = execution_time

        return result

    except Exception as e:
        subprocess_logger.error("Learning cycle failed in subprocess: %s", e)
        return {
            "error": str(e),
            "subprocess_pid": os.getpid(),
            "strategy_used": "error",
            "experiments_run": 0,
            "successful_experiments": 0,
            "success_rate": 0.0,
        }


# =============================================================================
# CuriosityDriver class
# =============================================================================


class CuriosityDriver:
    """
    Active driver for CuriosityEngine with process isolation.

    This class provides:
    1. An async heartbeat loop that periodically triggers learning cycles
    2. Process isolation for CPU-intensive learning cycles via ProcessPoolExecutor
    3. Resource-aware execution that checks budget before running cycles
    4. Graceful shutdown handling

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

        # Process pool for CPU offloading
        # Using max_workers=1 ensures sequential execution and prevents
        # multiple heavy learning cycles from competing for resources
        self.process_pool: Optional[ProcessPoolExecutor] = None

        # Background task handle
        self._heartbeat_task: Optional[asyncio.Task] = None

        # State tracking
        self._running = False
        self._cycle_count = 0
        self._last_cycle_time: Optional[float] = None
        self._last_cycle_result: Optional[Dict[str, Any]] = None

        # Statistics
        self._total_experiments = 0
        self._total_successes = 0
        self._total_cycle_time = 0.0

        logger.info(
            "CuriosityDriver initialized (heartbeat_interval=%.1fs, "
            "min_budget=%.1f, max_workers=%d)",
            self.config.heartbeat_interval,
            self.config.min_budget_threshold,
            self.config.max_workers,
        )

    async def start(self) -> None:
        """
        Start the curiosity driver.

        This method:
        1. Creates the ProcessPoolExecutor
        2. Spawns the background heartbeat loop task
        """
        if self._running:
            logger.warning("CuriosityDriver already running, ignoring start()")
            return

        logger.info("Starting CuriosityDriver...")

        # Create process pool
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.max_workers,
            # Use spawn method for cleaner process isolation
            # mp_context is not available on all platforms, so we skip it
        )

        # Mark as running before spawning task
        self._running = True

        # Spawn heartbeat loop
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="curiosity_heartbeat"
        )

        logger.info("CuriosityDriver started successfully")

    async def stop(self) -> None:
        """
        Stop the curiosity driver gracefully.

        This method:
        1. Cancels the background heartbeat task
        2. Shuts down the process pool (waiting for any running tasks)
        """
        if not self._running:
            logger.warning("CuriosityDriver not running, ignoring stop()")
            return

        logger.info("Stopping CuriosityDriver...")

        # Mark as not running to signal the loop to stop
        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.debug("Heartbeat task cancelled")

        # Shutdown process pool gracefully
        if self.process_pool is not None:
            logger.info("Shutting down process pool (waiting for pending tasks)...")
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

        logger.info(
            "CuriosityDriver stopped (cycles=%d, total_experiments=%d, "
            "avg_success_rate=%.2f)",
            self._cycle_count,
            self._total_experiments,
            self._total_successes / max(1, self._total_experiments),
        )

    async def _heartbeat_loop(self) -> None:
        """
        Background heartbeat loop that periodically runs learning cycles.

        This loop:
        1. Checks if the driver is still running
        2. Checks available budget
        3. If budget is sufficient, runs a learning cycle in subprocess
        4. Sleeps for the configured interval
        """
        logger.info("Heartbeat loop started")

        while self._running:
            try:
                # Check resource budget
                available_budget = self.engine.exploration_budget.get_available()

                if available_budget <= self.config.min_budget_threshold:
                    logger.info(
                        "Budget low (%.2f <= %.2f), sleeping for %.1fs",
                        available_budget,
                        self.config.min_budget_threshold,
                        self.config.low_budget_sleep,
                    )
                    await asyncio.sleep(self.config.low_budget_sleep)
                    continue

                # Run learning cycle
                logger.info(
                    "Running learning cycle (budget=%.2f, cycle=%d)",
                    available_budget,
                    self._cycle_count + 1,
                )

                result = await self._run_cycle_async()

                # Update statistics
                self._cycle_count += 1
                self._last_cycle_time = time.time()
                self._last_cycle_result = result

                if "error" not in result:
                    self._total_experiments += result.get("experiments_run", 0)
                    self._total_successes += result.get("successful_experiments", 0)
                    self._total_cycle_time += result.get("execution_time_total", 0)

                    logger.info(
                        "Learning cycle %d completed: experiments=%d, "
                        "success_rate=%.2f, time=%.2fs",
                        self._cycle_count,
                        result.get("experiments_run", 0),
                        result.get("success_rate", 0),
                        result.get("execution_time_total", 0),
                    )
                else:
                    logger.error(
                        "Learning cycle %d failed: %s",
                        self._cycle_count,
                        result.get("error"),
                    )

                # Sleep until next heartbeat
                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error("Error in heartbeat loop: %s", e, exc_info=True)
                # Sleep on error to avoid tight loop
                await asyncio.sleep(self.config.heartbeat_interval)

        logger.info("Heartbeat loop stopped")

    async def _run_cycle_async(self) -> Dict[str, Any]:
        """
        Run learning cycle asynchronously in subprocess.

        This method uses asyncio.get_running_loop().run_in_executor() to
        offload the heavy computation to the process pool without blocking
        the main event loop.

        Returns:
            Dictionary containing learning cycle results
        """
        if self.process_pool is None:
            logger.error("Process pool not initialized")
            return {"error": "Process pool not initialized"}

        # Prepare engine state for subprocess
        engine_state = {
            "max_experiments": self.config.max_experiments_per_cycle,
            "base_allocation": self.engine.exploration_budget.base_allocation,
            "enable_recovery": self.engine.exploration_budget.enable_recovery,
        }

        try:
            # Get the current event loop
            loop = asyncio.get_running_loop()

            # Run the wrapper function in the process pool
            # This is non-blocking and will not freeze the main thread
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.process_pool, _run_cycle_wrapper, engine_state
                ),
                timeout=self.config.cycle_timeout,
            )

            return result

        except asyncio.TimeoutError:
            logger.error(
                "Learning cycle timed out after %.1fs", self.config.cycle_timeout
            )
            return {"error": f"Timeout after {self.config.cycle_timeout}s"}
        except Exception as e:
            logger.error("Error running learning cycle: %s", e)
            return {"error": str(e)}

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
    # Status and statistics
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if driver is currently running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """
        Get driver statistics.

        Returns:
            Dictionary containing:
            - running: Whether driver is active
            - cycle_count: Number of completed cycles
            - total_experiments: Total experiments across all cycles
            - total_successes: Total successful experiments
            - average_success_rate: Overall success rate
            - average_cycle_time: Average time per cycle
            - last_cycle_time: Timestamp of last cycle
            - last_cycle_result: Result of most recent cycle
        """
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "total_experiments": self._total_experiments,
            "total_successes": self._total_successes,
            "average_success_rate": (
                self._total_successes / max(1, self._total_experiments)
            ),
            "average_cycle_time": (
                self._total_cycle_time / max(1, self._cycle_count)
            ),
            "last_cycle_time": self._last_cycle_time,
            "last_cycle_result": self._last_cycle_result,
            "config": {
                "heartbeat_interval": self.config.heartbeat_interval,
                "min_budget_threshold": self.config.min_budget_threshold,
                "max_experiments_per_cycle": self.config.max_experiments_per_cycle,
            },
        }


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
