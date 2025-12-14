"""
Comprehensive test suite for scheduler_node.py
"""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from scheduler_node import (
    SchedulerNode,
    TaskManager,
    _check_async_context,
    async_dispatch_scheduler_node,
    dispatch_scheduler_node,
)


@pytest.fixture
def task_manager():
    """Create TaskManager instance."""
    return TaskManager()


@pytest.fixture
def scheduler_node():
    """Create SchedulerNode instance."""
    return SchedulerNode()


@pytest.fixture
def context():
    """Create test context."""
    return {"audit_log": [], "running": True, "tasks": {}}


class TestTaskManager:
    """Test TaskManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test task manager initialization."""
        manager = TaskManager()

        assert manager.tasks == {}
        assert manager.task_metadata == {}

    @pytest.mark.asyncio
    async def test_register_task(self, task_manager):
        """Test registering a task."""

        async def dummy_task():
            await asyncio.sleep(0.1)
            return "done"

        task = asyncio.create_task(dummy_task())
        await task_manager.register_task("test_task", task, {"meta": "data"})

        assert "test_task" in task_manager.tasks
        assert task_manager.task_metadata["test_task"]["meta"] == "data"

        # Wait for task to complete
        await task

    @pytest.mark.asyncio
    async def test_register_duplicate_task(self, task_manager):
        """Test registering duplicate task cancels old one."""

        async def dummy_task():
            await asyncio.sleep(1)

        task1 = asyncio.create_task(dummy_task())
        await task_manager.register_task("dup_task", task1)

        task2 = asyncio.create_task(dummy_task())
        await task_manager.register_task("dup_task", task2)

        # Old task should be cancelled
        assert task1.cancelled() or task1.done()

        # Clean up
        await task_manager.cancel_all_tasks()

    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager):
        """Test cancelling a task."""

        async def long_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(long_task())
        await task_manager.register_task("cancel_me", task)

        result = await task_manager.cancel_task("cancel_me")

        assert result is True
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, task_manager):
        """Test cancelling non-existent task."""
        result = await task_manager.cancel_task("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_tasks(self, task_manager):
        """Test cancelling all tasks."""

        async def dummy_task():
            await asyncio.sleep(10)

        for i in range(3):
            task = asyncio.create_task(dummy_task())
            await task_manager.register_task(f"task_{i}", task)

        await task_manager.cancel_all_tasks()

        assert len(task_manager.tasks) == 0

    @pytest.mark.asyncio
    async def test_get_task_status(self, task_manager):
        """Test getting task status."""

        async def dummy_task():
            await asyncio.sleep(0.1)

        task = asyncio.create_task(dummy_task())
        await task_manager.register_task("status_task", task, {"key": "value"})

        status = task_manager.get_task_status("status_task")

        assert status["task_id"] == "status_task"
        assert "done" in status
        assert status["metadata"]["key"] == "value"

        await task

    @pytest.mark.asyncio
    async def test_get_all_tasks_status(self, task_manager):
        """Test getting all tasks status."""

        async def dummy_task():
            await asyncio.sleep(0.1)

        for i in range(2):
            task = asyncio.create_task(dummy_task())
            await task_manager.register_task(f"task_{i}", task)

        statuses = task_manager.get_all_tasks_status()

        assert len(statuses) == 2

        # Clean up
        await task_manager.cancel_all_tasks()


class TestSchedulerNode:
    """Test SchedulerNode class."""

    def test_initialization(self):
        """Test scheduler node initialization."""
        node = SchedulerNode()

        assert node.task_manager is not None

    @pytest.mark.asyncio
    async def test_execute_periodic_task(self, scheduler_node, context):
        """Test executing periodic task."""
        params = {
            "trigger": "periodic",
            "interval_ms": 100,
            "task_id": "periodic_test",
            "max_iterations": 2,
        }

        result = await scheduler_node.execute(params, context)

        assert result["status"] == "scheduled"
        assert result["task_id"] == "periodic_test"

        # Wait for task to complete
        await asyncio.sleep(0.3)

        # Clean up
        await scheduler_node.shutdown()

    @pytest.mark.asyncio
    async def test_execute_event_task(self, scheduler_node, context):
        """Test executing event-triggered task."""
        context["event_signal"] = "test_event"

        params = {"trigger": "event", "task_id": "event_test"}

        result = await scheduler_node.execute(params, context)

        assert result["status"] == "scheduled"
        assert result["task_id"] == "event_test"

        # Wait for task
        await asyncio.sleep(0.1)

        await scheduler_node.shutdown()

    @pytest.mark.asyncio
    async def test_execute_invalid_trigger(self, scheduler_node, context):
        """Test executing with invalid trigger."""
        params = {"trigger": "invalid_trigger", "task_id": "test"}

        with pytest.raises(ValueError):
            await scheduler_node.execute(params, context)

    @pytest.mark.asyncio
    async def test_execute_event_without_signal(self, scheduler_node, context):
        """Test event trigger without event signal."""
        params = {"trigger": "event", "task_id": "test"}

        # Remove event_signal from context
        context.pop("event_signal", None)

        with pytest.raises(ValueError):
            await scheduler_node.execute(params, context)

    @pytest.mark.asyncio
    async def test_cancel_task(self, scheduler_node, context):
        """Test cancelling a scheduled task."""
        params = {"trigger": "periodic", "interval_ms": 100, "task_id": "cancel_test"}

        await scheduler_node.execute(params, context)

        result = await scheduler_node.cancel_task("cancel_test")

        assert result is True

        await scheduler_node.shutdown()

    @pytest.mark.asyncio
    async def test_get_task_status(self, scheduler_node, context):
        """Test getting task status."""
        params = {
            "trigger": "periodic",
            "interval_ms": 100,
            "task_id": "status_test",
            "max_iterations": 2,
        }

        await scheduler_node.execute(params, context)

        status = scheduler_node.get_task_status("status_test")

        assert status is not None
        assert status["task_id"] == "status_test"

        await scheduler_node.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, scheduler_node, context):
        """Test graceful shutdown."""
        # Start multiple tasks
        for i in range(3):
            params = {"trigger": "periodic", "interval_ms": 100, "task_id": f"task_{i}"}
            await scheduler_node.execute(params, context)

        # Shutdown should cancel all
        await scheduler_node.shutdown()

        assert len(scheduler_node.task_manager.tasks) == 0


class TestDispatchFunctions:
    """Test dispatch functions."""

    @pytest.mark.asyncio
    async def test_async_dispatch_scheduler_node(self, context):
        """Test async dispatch."""
        node = {
            "type": "SchedulerNode",
            "params": {
                "trigger": "periodic",
                "interval_ms": 100,
                "task_id": "async_test",
                "max_iterations": 1,
            },
        }

        result = await async_dispatch_scheduler_node(node, context)

        assert result["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_async_dispatch_invalid_type(self, context):
        """Test async dispatch with invalid node type."""
        node = {"type": "InvalidNode", "params": {}}

        with pytest.raises(ValueError):
            await async_dispatch_scheduler_node(node, context)

    def test_dispatch_scheduler_node(self, context):
        """Test sync dispatch."""
        node = {
            "type": "SchedulerNode",
            "params": {
                "trigger": "periodic",
                "interval_ms": 100,
                "task_id": "sync_test",
                "max_iterations": 1,
            },
        }

        result = dispatch_scheduler_node(node, context)

        assert result["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_dispatch_from_async_context_raises(self, context):
        """Test that sync dispatch raises from async context."""
        node = {
            "type": "SchedulerNode",
            "params": {"trigger": "periodic", "task_id": "test"},
        }

        # Should raise when called from async context
        with pytest.raises(RuntimeError):
            dispatch_scheduler_node(node, context)


class TestCheckAsyncContext:
    """Test async context checking."""

    def test_outside_async_context(self):
        """Test checking outside async context."""
        result = _check_async_context()

        assert result is False

    @pytest.mark.asyncio
    async def test_inside_async_context(self):
        """Test checking inside async context."""
        result = _check_async_context()

        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
