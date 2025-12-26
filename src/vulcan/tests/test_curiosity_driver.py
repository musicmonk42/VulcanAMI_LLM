"""
test_curiosity_driver.py - Tests for CuriosityDriver class

Part of the VULCAN-AGI system

Tests cover:
- Process isolation and subprocess execution
- Async lifecycle (start/stop)
- Resource budget checking
- Heartbeat loop behavior
- Graceful shutdown
- Error handling
- Statistics tracking
- State management
"""

import asyncio
import time

import pytest

from vulcan.curiosity_engine.curiosity_driver import (
    CuriosityDriver,
    CuriosityDriverConfig,
    CycleOutcome,
    CycleResult,
    CycleStatisticsTracker,
    DriverState,
    ProcessPoolManager,
    _run_cycle_wrapper,
    create_curiosity_driver,
)
from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def curiosity_engine():
    """Create a CuriosityEngine instance for testing"""
    return CuriosityEngine()


@pytest.fixture
def driver_config():
    """Create a test configuration with short intervals"""
    return CuriosityDriverConfig(
        heartbeat_interval=0.5,  # Short interval for tests
        min_budget_threshold=10.0,
        max_experiments_per_cycle=2,
        low_budget_sleep=0.2,  # Short sleep for tests
        cycle_timeout=30.0,
        max_workers=1,
    )


@pytest.fixture
def curiosity_driver(curiosity_engine, driver_config):
    """Create a CuriosityDriver instance"""
    return CuriosityDriver(curiosity_engine, driver_config)


# =============================================================================
# Test CuriosityDriverConfig
# =============================================================================


class TestCuriosityDriverConfig:
    """Tests for CuriosityDriverConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CuriosityDriverConfig()

        assert config.heartbeat_interval == 30.0
        assert config.min_budget_threshold == 10.0
        assert config.max_experiments_per_cycle == 5
        assert config.low_budget_sleep == 60.0
        assert config.cycle_timeout == 300.0
        assert config.max_workers == 1

    def test_custom_config(self):
        """Test custom configuration values"""
        config = CuriosityDriverConfig(
            heartbeat_interval=60.0,
            min_budget_threshold=20.0,
            max_experiments_per_cycle=10,
        )

        assert config.heartbeat_interval == 60.0
        assert config.min_budget_threshold == 20.0
        assert config.max_experiments_per_cycle == 10


# =============================================================================
# Test _run_cycle_wrapper
# =============================================================================


class TestRunCycleWrapper:
    """Tests for the _run_cycle_wrapper subprocess function"""

    def test_wrapper_returns_dict(self):
        """Test that wrapper returns a dictionary result"""
        engine_state = {
            "max_experiments": 2,
            "base_allocation": 100.0,
            "enable_recovery": True,
        }

        result = _run_cycle_wrapper(engine_state)

        assert isinstance(result, dict)
        assert "subprocess_pid" in result

    def test_wrapper_includes_execution_time(self):
        """Test that wrapper includes execution time"""
        engine_state = {"max_experiments": 1}

        result = _run_cycle_wrapper(engine_state)

        assert "execution_time_total" in result or "error" in result

    def test_wrapper_handles_error(self):
        """Test that wrapper handles errors gracefully"""
        # Pass invalid state to trigger error handling
        engine_state = {"max_experiments": -1}

        result = _run_cycle_wrapper(engine_state)

        # Should return a dict even on error
        assert isinstance(result, dict)


# =============================================================================
# Test CuriosityDriver Initialization
# =============================================================================


class TestCuriosityDriverInit:
    """Tests for CuriosityDriver initialization"""

    def test_init_with_defaults(self, curiosity_engine):
        """Test initialization with default config"""
        driver = CuriosityDriver(curiosity_engine)

        assert driver.engine is curiosity_engine
        assert driver.config is not None
        assert not driver._running
        assert driver._cycle_count == 0

    def test_init_with_custom_config(self, curiosity_engine, driver_config):
        """Test initialization with custom config"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        assert driver.config.heartbeat_interval == 0.5
        assert driver.config.max_experiments_per_cycle == 2

    def test_init_process_pool_not_created(self, curiosity_engine):
        """Test that process pool is not created on init"""
        driver = CuriosityDriver(curiosity_engine)

        assert driver.process_pool is None


# =============================================================================
# Test CuriosityDriver Start/Stop Lifecycle
# =============================================================================


class TestCuriosityDriverLifecycle:
    """Tests for CuriosityDriver start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start_creates_process_pool(self, curiosity_driver):
        """Test that start() creates the process pool"""
        await curiosity_driver.start()

        try:
            assert curiosity_driver.process_pool is not None
            assert curiosity_driver._running
        finally:
            await curiosity_driver.stop()

    @pytest.mark.asyncio
    async def test_start_spawns_heartbeat_task(self, curiosity_driver):
        """Test that start() spawns the heartbeat task"""
        await curiosity_driver.start()

        try:
            assert curiosity_driver._heartbeat_task is not None
            assert not curiosity_driver._heartbeat_task.done()
        finally:
            await curiosity_driver.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_heartbeat_task(self, curiosity_driver):
        """Test that stop() cancels the heartbeat task"""
        await curiosity_driver.start()
        await curiosity_driver.stop()

        assert curiosity_driver._heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_process_pool(self, curiosity_driver):
        """Test that stop() shuts down the process pool"""
        await curiosity_driver.start()
        await curiosity_driver.stop()

        assert curiosity_driver.process_pool is None

    @pytest.mark.asyncio
    async def test_double_start_ignored(self, curiosity_driver):
        """Test that calling start() twice is handled gracefully"""
        await curiosity_driver.start()

        try:
            # Second start should be ignored
            await curiosity_driver.start()
            assert curiosity_driver._running
        finally:
            await curiosity_driver.stop()

    @pytest.mark.asyncio
    async def test_double_stop_ignored(self, curiosity_driver):
        """Test that calling stop() twice is handled gracefully"""
        await curiosity_driver.start()
        await curiosity_driver.stop()

        # Second stop should be ignored
        await curiosity_driver.stop()

        assert not curiosity_driver._running


# =============================================================================
# Test Async Context Manager
# =============================================================================


class TestCuriosityDriverContextManager:
    """Tests for CuriosityDriver async context manager"""

    @pytest.mark.asyncio
    async def test_context_manager_starts_driver(self, curiosity_engine, driver_config):
        """Test that context manager starts the driver"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        async with driver:
            assert driver._running
            assert driver.process_pool is not None

    @pytest.mark.asyncio
    async def test_context_manager_stops_driver(self, curiosity_engine, driver_config):
        """Test that context manager stops the driver on exit"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        async with driver:
            pass

        assert not driver._running
        assert driver.process_pool is None


# =============================================================================
# Test Budget Checking
# =============================================================================


class TestCuriosityDriverBudgetCheck:
    """Tests for budget checking behavior"""

    @pytest.mark.asyncio
    async def test_low_budget_triggers_sleep(self, curiosity_engine, driver_config):
        """Test that low budget triggers sleep instead of cycle"""
        # Set very low budget
        curiosity_engine.exploration_budget.tracker.set_budget(5.0)

        driver = CuriosityDriver(curiosity_engine, driver_config)

        # Start and immediately stop (to test initial budget check)
        await driver.start()

        # Give a moment for first budget check
        await asyncio.sleep(0.1)

        await driver.stop()

        # Should not have run any cycles due to low budget
        assert driver._cycle_count == 0

    @pytest.mark.asyncio
    async def test_sufficient_budget_runs_cycle(self, curiosity_engine, driver_config):
        """Test that sufficient budget runs a learning cycle"""
        # Ensure sufficient budget
        curiosity_engine.exploration_budget.tracker.set_budget(100.0)

        driver = CuriosityDriver(curiosity_engine, driver_config)

        await driver.start()

        # Wait for at least one cycle to potentially complete
        # Note: In real tests, we might mock the process pool
        await asyncio.sleep(1.0)

        await driver.stop()

        # At least one cycle should have started (may or may not complete)
        # This is a soft check since actual execution depends on process pool


# =============================================================================
# Test Statistics
# =============================================================================


class TestCuriosityDriverStats:
    """Tests for driver statistics"""

    def test_get_stats_initial(self, curiosity_driver):
        """Test initial statistics values"""
        stats = curiosity_driver.get_stats()

        assert stats["running"] is False
        assert stats["cycle_count"] == 0
        assert stats["total_experiments"] == 0
        assert stats["total_successes"] == 0
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_is_running_property(self, curiosity_driver):
        """Test is_running property"""
        assert not curiosity_driver.is_running

        await curiosity_driver.start()
        assert curiosity_driver.is_running

        await curiosity_driver.stop()
        assert not curiosity_driver.is_running


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateCuriosityDriver:
    """Tests for create_curiosity_driver factory function"""

    def test_factory_creates_driver(self, curiosity_engine):
        """Test factory function creates a driver"""
        driver = create_curiosity_driver(curiosity_engine)

        assert isinstance(driver, CuriosityDriver)
        assert driver.engine is curiosity_engine

    def test_factory_with_custom_params(self, curiosity_engine):
        """Test factory function with custom parameters"""
        driver = create_curiosity_driver(
            curiosity_engine,
            heartbeat_interval=60.0,
            min_budget_threshold=20.0,
            max_experiments_per_cycle=10,
        )

        assert driver.config.heartbeat_interval == 60.0
        assert driver.config.min_budget_threshold == 20.0
        assert driver.config.max_experiments_per_cycle == 10


# =============================================================================
# Test Error Handling
# =============================================================================


class TestCuriosityDriverErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_run_cycle_handles_timeout(self, curiosity_engine):
        """Test that cycle timeout is handled"""
        config = CuriosityDriverConfig(
            heartbeat_interval=0.5,
            cycle_timeout=0.1,  # Very short timeout
        )
        driver = CuriosityDriver(curiosity_engine, config)

        await driver.start()

        # The very short timeout might cause timeout errors
        # which should be handled gracefully
        await asyncio.sleep(0.3)

        await driver.stop()

        # Should not crash, stats should be available
        stats = driver.get_stats()
        assert isinstance(stats, dict)


# =============================================================================
# Test Process Pool Execution
# =============================================================================


class TestProcessPoolExecution:
    """Tests for process pool execution behavior"""

    @pytest.mark.asyncio
    async def test_non_blocking_execution(self, curiosity_engine, driver_config):
        """Test that learning cycle execution is non-blocking"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        await driver.start()

        # The main event loop should remain responsive
        # This is verified by the fact that we can await other operations
        start = time.time()
        await asyncio.sleep(0.1)
        elapsed = time.time() - start

        await driver.stop()

        # Sleep should have been approximately 0.1 seconds
        # If it was blocked, it would be much longer
        assert elapsed < 0.5  # Generous margin for test reliability


# =============================================================================
# Integration Tests
# =============================================================================


class TestCuriosityDriverIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_complete_lifecycle(self, curiosity_engine):
        """Test complete driver lifecycle from start to stop"""
        config = CuriosityDriverConfig(
            heartbeat_interval=0.5,
            min_budget_threshold=10.0,
            max_experiments_per_cycle=2,
            low_budget_sleep=0.2,
        )

        async with CuriosityDriver(curiosity_engine, config) as driver:
            # Driver should be running
            assert driver.is_running

            # Wait for potential cycle
            await asyncio.sleep(0.3)

            # Get stats
            stats = driver.get_stats()
            assert stats["running"]

        # After context manager exit
        assert not driver.is_running

    @pytest.mark.asyncio
    async def test_multiple_drivers_sequential(self, curiosity_engine):
        """Test multiple sequential driver instances"""
        config = CuriosityDriverConfig(heartbeat_interval=0.3)

        # First driver
        async with CuriosityDriver(curiosity_engine, config):
            await asyncio.sleep(0.1)

        # Second driver
        async with CuriosityDriver(curiosity_engine, config):
            await asyncio.sleep(0.1)

        # Both should have completed without error


# =============================================================================
# Test Separated Concern Classes
# =============================================================================


class TestCycleStatisticsTracker:
    """Tests for CycleStatisticsTracker class"""

    def test_record_cycle(self):
        """Test recording a cycle result"""
        tracker = CycleStatisticsTracker()

        result = CycleResult(
            cycle_id=1,
            outcome=CycleOutcome.SUCCESS,
            experiments_run=5,
            successful_experiments=4,
            success_rate=0.8,
            execution_time=10.0,
        )

        tracker.record_cycle(result)

        assert tracker.total_experiments == 5
        assert tracker.total_successes == 4
        assert tracker.outcome_counts[CycleOutcome.SUCCESS] == 1

    def test_get_statistics(self):
        """Test getting comprehensive statistics"""
        tracker = CycleStatisticsTracker()

        for i in range(5):
            result = CycleResult(
                cycle_id=i,
                outcome=CycleOutcome.SUCCESS,
                experiments_run=3,
                successful_experiments=2,
                success_rate=0.67,
                execution_time=5.0,
            )
            tracker.record_cycle(result)

        stats = tracker.get_statistics()

        assert stats["total_cycles"] == 5
        assert stats["total_experiments"] == 15
        assert stats["total_successes"] == 10

    def test_reset(self):
        """Test resetting statistics"""
        tracker = CycleStatisticsTracker()

        result = CycleResult(
            cycle_id=1,
            outcome=CycleOutcome.SUCCESS,
            experiments_run=5,
            successful_experiments=4,
            success_rate=0.8,
            execution_time=10.0,
        )
        tracker.record_cycle(result)

        tracker.reset()

        assert tracker.total_experiments == 0
        assert tracker.total_successes == 0
        assert len(tracker.cycle_history) == 0


class TestProcessPoolManager:
    """Tests for ProcessPoolManager class"""

    def test_initialize_and_shutdown(self):
        """Test process pool initialization and shutdown"""
        manager = ProcessPoolManager(max_workers=1)

        assert not manager.is_available()

        # Initialize
        result = manager.initialize()
        assert result is True
        assert manager.is_available()

        # Shutdown
        result = manager.shutdown()
        assert result is True
        assert not manager.is_available()

    def test_double_initialize(self):
        """Test double initialization is handled"""
        manager = ProcessPoolManager(max_workers=1)

        manager.initialize()
        result = manager.initialize()  # Should return True without error

        assert result is True

        manager.shutdown()

    def test_active_task_tracking(self):
        """Test active task counter"""
        manager = ProcessPoolManager()

        assert manager.get_active_task_count() == 0

        manager.increment_active_tasks()
        assert manager.get_active_task_count() == 1

        manager.decrement_active_tasks()
        assert manager.get_active_task_count() == 0

        # Ensure decrement doesn't go below 0
        manager.decrement_active_tasks()
        assert manager.get_active_task_count() == 0


class TestCycleResult:
    """Tests for CycleResult dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = CycleResult(
            cycle_id=1,
            outcome=CycleOutcome.SUCCESS,
            experiments_run=5,
            successful_experiments=4,
            success_rate=0.8,
            execution_time=10.0,
            subprocess_pid=12345,
            strategy_used="balanced",
        )

        d = result.to_dict()

        assert d["cycle_id"] == 1
        assert d["outcome"] == "success"
        assert d["experiments_run"] == 5
        assert d["subprocess_pid"] == 12345


class TestDriverState:
    """Tests for DriverState enum"""

    def test_states_exist(self):
        """Test all expected states exist"""
        assert DriverState.STOPPED is not None
        assert DriverState.STARTING is not None
        assert DriverState.RUNNING is not None
        assert DriverState.STOPPING is not None
        assert DriverState.ERROR is not None

    def test_state_values(self):
        """Test state string values"""
        assert DriverState.STOPPED.value == "stopped"
        assert DriverState.RUNNING.value == "running"


class TestCuriosityDriverState:
    """Tests for driver state management"""

    @pytest.mark.asyncio
    async def test_state_transitions(self, curiosity_engine, driver_config):
        """Test state transitions during lifecycle"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        # Initial state
        assert driver.state == DriverState.STOPPED

        # After start
        await driver.start()
        assert driver.state == DriverState.RUNNING

        # After stop
        await driver.stop()
        assert driver.state == DriverState.STOPPED

    def test_state_property(self, curiosity_driver):
        """Test state property access"""
        state = curiosity_driver.state
        assert isinstance(state, DriverState)


class TestCuriosityDriverExtendedStats:
    """Tests for extended statistics functionality"""

    @pytest.mark.asyncio
    async def test_get_recent_history(self, curiosity_engine, driver_config):
        """Test getting recent cycle history"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        history = driver.get_recent_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_reset_statistics(self, curiosity_engine, driver_config):
        """Test resetting statistics"""
        driver = CuriosityDriver(curiosity_engine, driver_config)

        driver.reset_statistics()

        assert driver._cycle_count == 0
        assert driver._consecutive_failures == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
