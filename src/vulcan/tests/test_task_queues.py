# ============================================================
# VULCAN-AGI Orchestrator - Task Queues Tests
# Comprehensive test suite for task_queues.py
# FIXED: Added concrete TestTaskQueue implementation for testing base interface
# ============================================================

import unittest
import sys
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from collections import defaultdict

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import components to test
from vulcan.orchestrator.task_queues import (
    TaskStatus,
    QueueType,
    TaskMetadata,
    TaskQueueInterface,
    RayTaskQueue,
    CeleryTaskQueue,
    CustomTaskQueue,
    create_task_queue,
    RAY_AVAILABLE,
    CELERY_AVAILABLE,
    ZMQ_AVAILABLE
)


# ============================================================
# CONCRETE TEST IMPLEMENTATION
# ============================================================

class TestTaskQueue(TaskQueueInterface):
    """
    Concrete implementation of TaskQueueInterface for testing
    This allows us to test the base class functionality without hitting NotImplementedError
    """
    
    def __init__(self):
        super().__init__()
        self.submitted_tasks = []
        self.cancelled_tasks = []
    
    def submit_task(self, task: dict, priority: int = 0, 
                   timeout: float = None, max_retries: int = 3) -> str:
        """Mock submit_task implementation"""
        import uuid
        task_id = str(uuid.uuid4())
        
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.PENDING,
            submitted_at=time.time(),
            priority=priority,
            timeout_seconds=timeout,
            max_retries=max_retries
        )
        
        with self._lock:
            self.pending_tasks[task_id] = metadata
            self.submitted_tasks.append(task_id)
        
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None):
        """Mock get_result implementation"""
        with self._lock:
            if task_id in self.pending_tasks:
                metadata = self.pending_tasks[task_id]
                return metadata.result
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Mock cancel_task implementation"""
        with self._lock:
            if task_id in self.pending_tasks:
                metadata = self.pending_tasks[task_id]
                metadata.status = TaskStatus.CANCELLED
                metadata.completed_at = time.time()
                self.cancelled_tasks.append(task_id)
                return True
        return False
    
    def get_queue_status(self) -> dict:
        """Mock get_queue_status implementation"""
        with self._lock:
            status_counts = defaultdict(int)
            for metadata in self.pending_tasks.values():
                status_counts[metadata.status.value] += 1
            
            return {
                'queue_type': 'test',
                'total_tasks': len(self.pending_tasks),
                'pending': status_counts[TaskStatus.PENDING.value],
                'running': status_counts[TaskStatus.RUNNING.value],
                'completed': status_counts[TaskStatus.COMPLETED.value],
                'failed': status_counts[TaskStatus.FAILED.value],
                'cancelled': status_counts[TaskStatus.CANCELLED.value]
            }
    
    def cleanup(self):
        """Mock cleanup implementation"""
        with self._lock:
            self.pending_tasks.clear()
            self.submitted_tasks.clear()
            self.cancelled_tasks.clear()


# ============================================================
# TEST: ENUMS
# ============================================================

class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_task_status_enum(self):
        """Test TaskStatus enum values"""
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.CANCELLED.value, "cancelled")
        self.assertEqual(TaskStatus.TIMEOUT.value, "timeout")
    
    def test_queue_type_enum(self):
        """Test QueueType enum values"""
        self.assertEqual(QueueType.RAY.value, "ray")
        self.assertEqual(QueueType.CELERY.value, "celery")
        self.assertEqual(QueueType.ZMQ.value, "zmq")
        self.assertEqual(QueueType.CUSTOM.value, "custom")


# ============================================================
# TEST: TASK METADATA
# ============================================================

class TestTaskMetadata(unittest.TestCase):
    """Test TaskMetadata dataclass"""
    
    def test_task_metadata_creation(self):
        """Test creating TaskMetadata"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.PENDING,
            submitted_at=time.time()
        )
        
        self.assertEqual(metadata.task_id, "test_123")
        self.assertEqual(metadata.status, TaskStatus.PENDING)
        self.assertIsNotNone(metadata.submitted_at)
    
    def test_task_metadata_with_defaults(self):
        """Test TaskMetadata default values"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.PENDING,
            submitted_at=time.time()
        )
        
        self.assertIsNone(metadata.started_at)
        self.assertIsNone(metadata.completed_at)
        self.assertEqual(metadata.priority, 0)
        self.assertEqual(metadata.retry_count, 0)
        self.assertEqual(metadata.max_retries, 3)
        self.assertIsNone(metadata.timeout_seconds)
        self.assertIsNone(metadata.error_message)
        self.assertIsNone(metadata.result)
    
    def test_get_duration(self):
        """Test get_duration method"""
        current_time = time.time()
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.COMPLETED,
            submitted_at=current_time,
            started_at=current_time + 1,
            completed_at=current_time + 3
        )
        
        duration = metadata.get_duration()
        self.assertAlmostEqual(duration, 2.0, places=1)
    
    def test_get_duration_incomplete_task(self):
        """Test get_duration with incomplete task"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.RUNNING,
            submitted_at=time.time()
        )
        
        duration = metadata.get_duration()
        self.assertIsNone(duration)
    
    def test_get_wait_time_started(self):
        """Test get_wait_time for started task"""
        current_time = time.time()
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.RUNNING,
            submitted_at=current_time,
            started_at=current_time + 2
        )
        
        wait_time = metadata.get_wait_time()
        self.assertAlmostEqual(wait_time, 2.0, places=1)
    
    def test_get_wait_time_not_started(self):
        """Test get_wait_time for not started task"""
        current_time = time.time()
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.PENDING,
            submitted_at=current_time - 1
        )
        
        wait_time = metadata.get_wait_time()
        self.assertGreater(wait_time, 0.9)
    
    def test_should_retry_failed_task(self):
        """Test should_retry with failed task"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.FAILED,
            submitted_at=time.time(),
            retry_count=1,
            max_retries=3
        )
        
        self.assertTrue(metadata.should_retry())
    
    def test_should_retry_max_retries_reached(self):
        """Test should_retry when max retries reached"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.FAILED,
            submitted_at=time.time(),
            retry_count=3,
            max_retries=3
        )
        
        self.assertFalse(metadata.should_retry())
    
    def test_should_retry_completed_task(self):
        """Test should_retry with completed task"""
        metadata = TaskMetadata(
            task_id="test_123",
            status=TaskStatus.COMPLETED,
            submitted_at=time.time()
        )
        
        self.assertFalse(metadata.should_retry())


# ============================================================
# TEST: BASE TASK QUEUE INTERFACE
# ============================================================

class TestTaskQueueInterface(unittest.TestCase):
    """
    Test TaskQueueInterface base class
    FIXED: Using TestTaskQueue concrete implementation instead of abstract interface
    """
    
    def test_interface_initialization(self):
        """Test TaskQueueInterface initialization"""
        queue = TestTaskQueue()
        
        self.assertIsNotNone(queue.pending_tasks)
        self.assertIsNotNone(queue._lock)
        # FIXED: Check the *state* of the event, not the event object
        self.assertFalse(queue._shutdown.is_set())
        self.assertIsNotNone(queue._monitor_thread)
        
        # Cleanup
        queue.shutdown()
    
    def test_monitor_thread_started(self):
        """Test that monitor thread is started"""
        queue = TestTaskQueue()
        
        time.sleep(0.2)  # Give thread time to start
        self.assertTrue(queue._monitor_thread.is_alive())
        
        # Cleanup
        queue.shutdown()
    
    def test_get_task_status_existing(self):
        """Test getting status of existing task"""
        queue = TestTaskQueue()
        
        # Add a task using the concrete implementation
        task_id = queue.submit_task({"test": "data"})
        
        status = queue.get_task_status(task_id)
        self.assertEqual(status, TaskStatus.PENDING)
        
        # Cleanup
        queue.shutdown()
    
    def test_get_task_status_nonexistent(self):
        """Test getting status of nonexistent task"""
        queue = TestTaskQueue()
        
        status = queue.get_task_status("nonexistent")
        self.assertIsNone(status)
        
        # Cleanup
        queue.shutdown()
    
    def test_cleanup_completed_tasks(self):
        """Test cleanup of old completed tasks"""
        queue = TestTaskQueue()
        
        # Add old completed task
        old_time = time.time() - 7200  # 2 hours ago
        task_id = "old_task"
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            submitted_at=old_time,
            completed_at=old_time
        )
        queue.pending_tasks[task_id] = metadata
        
        # Add recent task
        recent_id = queue.submit_task({"test": "data"})
        queue.pending_tasks[recent_id].status = TaskStatus.COMPLETED
        queue.pending_tasks[recent_id].completed_at = time.time()
        
        # Run cleanup
        queue._cleanup_completed_tasks()
        
        # Old task should be removed, recent task should remain
        self.assertNotIn(task_id, queue.pending_tasks)
        self.assertIn(recent_id, queue.pending_tasks)
        
        # Cleanup
        queue.shutdown()
    
    def test_check_task_timeouts(self):
        """Test checking for task timeouts"""
        queue = TestTaskQueue()
        
        # Add task that has timed out
        old_time = time.time() - 100
        task_id = queue.submit_task({"test": "data"}, timeout=10)
        metadata = queue.pending_tasks[task_id]
        metadata.status = TaskStatus.RUNNING
        metadata.started_at = old_time
        
        # Check timeouts
        queue._check_task_timeouts()
        
        # Task should be marked as timed out
        self.assertEqual(metadata.status, TaskStatus.TIMEOUT)
        self.assertIsNotNone(metadata.completed_at)
        
        # Cleanup
        queue.shutdown()
    
    def test_shutdown(self):
        """Test shutdown method"""
        queue = TestTaskQueue()
        
        # Verify thread is running
        self.assertTrue(queue._monitor_thread.is_alive())
        
        # Shutdown
        queue.shutdown()
        
        # Verify shutdown flag is set
        # FIXED: Check the *state* of the event
        self.assertTrue(queue._shutdown.is_set())
    
    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract base class methods raise NotImplementedError"""
        # Create actual abstract interface to test NotImplementedError
        queue = TaskQueueInterface()
        
        with self.assertRaises(NotImplementedError):
            queue.submit_task({})
        
        with self.assertRaises(NotImplementedError):
            queue.get_result("test")
        
        with self.assertRaises(NotImplementedError):
            queue.cancel_task("test")
        
        with self.assertRaises(NotImplementedError):
            queue.get_queue_status()
        
        with self.assertRaises(NotImplementedError):
            queue.cleanup()
        
        # Manually stop the monitor thread and set shutdown flag
        # FIXED: Use the correct method to signal the event
        queue._shutdown.set()
        if queue._monitor_thread and queue._monitor_thread.is_alive():
            queue._monitor_thread.join(timeout=1)


# ============================================================
# TEST: RAY TASK QUEUE
# ============================================================

@unittest.skipIf(not RAY_AVAILABLE, "Ray not available")
class TestRayTaskQueue(unittest.TestCase):
    """Test RayTaskQueue implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not RAY_AVAILABLE:
            self.skipTest("Ray not available")
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_initialization(self, mock_ray):
        """Test RayTaskQueue initialization"""
        mock_ray.is_initialized.return_value = False
        mock_ray.init.return_value = None
        
        queue = RayTaskQueue()
        
        self.assertIsNotNone(queue)
        mock_ray.init.assert_called_once()
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_initialization_already_initialized(self, mock_ray):
        """Test initialization when Ray is already initialized"""
        mock_ray.is_initialized.return_value = True
        
        queue = RayTaskQueue()
        
        self.assertIsNotNone(queue)
        mock_ray.init.assert_not_called()
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_submit_task(self, mock_ray):
        """Test submitting task to Ray"""
        mock_ray.is_initialized.return_value = True
        mock_remote = MagicMock()
        mock_ray.remote.return_value = mock_remote
        mock_future = MagicMock()
        mock_remote.remote.return_value = mock_future
        
        queue = RayTaskQueue()
        
        task = {"job_id": "test_job"}
        task_id = queue.submit_task(task)
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, queue.pending_tasks)
        self.assertIn(task_id, queue.ray_futures)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_get_result(self, mock_ray):
        """Test getting result from Ray"""
        mock_ray.is_initialized.return_value = True
        mock_ray.get.return_value = {"status": "completed", "result": "success"}
        
        queue = RayTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        mock_future = MagicMock()
        queue.ray_futures[task_id] = mock_future
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.PENDING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        result = queue.get_result(task_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(metadata.status, TaskStatus.COMPLETED)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_cancel_task(self, mock_ray):
        """Test cancelling Ray task"""
        mock_ray.is_initialized.return_value = True
        mock_ray.cancel.return_value = None
        
        queue = RayTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        mock_future = MagicMock()
        queue.ray_futures[task_id] = mock_future
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        success = queue.cancel_task(task_id)
        
        self.assertTrue(success)
        self.assertEqual(metadata.status, TaskStatus.CANCELLED)
        self.assertNotIn(task_id, queue.ray_futures)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_get_queue_status(self, mock_ray):
        """Test getting Ray queue status"""
        mock_ray.is_initialized.return_value = True
        mock_ray.cluster_resources.return_value = {"CPU": 4}
        mock_ray.available_resources.return_value = {"CPU": 2}
        
        queue = RayTaskQueue()
        
        status = queue.get_queue_status()
        
        self.assertIn('queue_type', status)
        self.assertEqual(status['queue_type'], 'ray')
        self.assertIn('total_tasks', status)
        
        # Cleanup
        queue.shutdown()


# ============================================================
# TEST: CELERY TASK QUEUE
# ============================================================

@unittest.skipIf(not CELERY_AVAILABLE, "Celery not available")
class TestCeleryTaskQueue(unittest.TestCase):
    """Test CeleryTaskQueue implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CELERY_AVAILABLE:
            self.skipTest("Celery not available")
    
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_initialization(self, mock_celery_class):
        """Test CeleryTaskQueue initialization"""
        mock_app = MagicMock()
        mock_celery_class.return_value = mock_app
        
        queue = CeleryTaskQueue()
        
        self.assertIsNotNone(queue)
        self.assertIsNotNone(queue.app)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_submit_task(self, mock_celery_class):
        """Test submitting task to Celery"""
        mock_app = MagicMock()
        mock_result = MagicMock()
        mock_app.send_task.return_value = mock_result
        mock_celery_class.return_value = mock_app
        
        queue = CeleryTaskQueue()
        
        task = {"job_id": "test_job"}
        task_id = queue.submit_task(task)
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, queue.pending_tasks)
        self.assertIn(task_id, queue.celery_results)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_get_result(self, mock_celery_class):
        """Test getting result from Celery"""
        mock_app = MagicMock()
        mock_celery_class.return_value = mock_app
        
        queue = CeleryTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        mock_result = MagicMock()
        mock_result.get.return_value = {"status": "completed"}
        queue.celery_results[task_id] = mock_result
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.PENDING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        result = queue.get_result(task_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(metadata.status, TaskStatus.COMPLETED)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_cancel_task(self, mock_celery_class):
        """Test cancelling Celery task"""
        mock_app = MagicMock()
        mock_celery_class.return_value = mock_app
        
        queue = CeleryTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        mock_result = MagicMock()
        queue.celery_results[task_id] = mock_result
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        success = queue.cancel_task(task_id)
        
        self.assertTrue(success)
        self.assertEqual(metadata.status, TaskStatus.CANCELLED)
        self.assertNotIn(task_id, queue.celery_results)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_get_queue_status(self, mock_celery_class):
        """Test getting Celery queue status"""
        mock_app = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {}
        mock_inspect.scheduled.return_value = {}
        mock_inspect.reserved.return_value = {}
        mock_app.control.inspect.return_value = mock_inspect
        mock_celery_class.return_value = mock_app
        
        queue = CeleryTaskQueue()
        
        status = queue.get_queue_status()
        
        self.assertIn('queue_type', status)
        self.assertEqual(status['queue_type'], 'celery')
        self.assertIn('total_tasks', status)
        
        # Cleanup
        queue.shutdown()


# ============================================================
# TEST: CUSTOM ZMQ TASK QUEUE
# ============================================================

@unittest.skipIf(not ZMQ_AVAILABLE, "ZMQ not available")
class TestCustomTaskQueue(unittest.TestCase):
    """Test CustomTaskQueue implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not ZMQ_AVAILABLE:
            self.skipTest("ZMQ not available")
    
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_initialization(self, mock_context_class):
        """Test CustomTaskQueue initialization"""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_context.socket.return_value = mock_socket
        mock_context_class.return_value = mock_context
        
        queue = CustomTaskQueue()
        
        self.assertIsNotNone(queue)
        self.assertIsNotNone(queue.context)
        self.assertIsNotNone(queue.socket)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_submit_task(self, mock_context_class):
        """Test submitting task via ZMQ"""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_json.return_value = {"status": "accepted"}
        mock_context.socket.return_value = mock_socket
        mock_context_class.return_value = mock_context
        
        queue = CustomTaskQueue()
        
        task = {"job_id": "test_job"}
        task_id = queue.submit_task(task)
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, queue.pending_tasks)
        mock_socket.send_json.assert_called()
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_get_result(self, mock_context_class):
        """Test getting result via ZMQ"""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_json.return_value = {
            "status": "completed",
            "result": {"data": "success"}
        }
        mock_socket.getsockopt.return_value = 5000
        mock_context.socket.return_value = mock_socket
        mock_context_class.return_value = mock_context
        
        queue = CustomTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.PENDING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        result = queue.get_result(task_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(metadata.status, TaskStatus.COMPLETED)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_cancel_task(self, mock_context_class):
        """Test cancelling task via ZMQ"""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_json.return_value = {"status": "cancelled"}
        mock_context.socket.return_value = mock_socket
        mock_context_class.return_value = mock_context
        
        queue = CustomTaskQueue()
        
        # Add task manually
        task_id = "test_123"
        metadata = TaskMetadata(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            submitted_at=time.time()
        )
        queue.pending_tasks[task_id] = metadata
        
        success = queue.cancel_task(task_id)
        
        self.assertTrue(success)
        self.assertEqual(metadata.status, TaskStatus.CANCELLED)
        
        # Cleanup
        queue.shutdown()
    
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_get_queue_status(self, mock_context_class):
        """Test getting ZMQ queue status"""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_json.return_value = {"workers": 5}
        # Add mock for poller
        mock_poller = MagicMock()
        mock_poller.poll.return_value = True
        mock_zmq = MagicMock()
        mock_zmq.Poller.return_value = mock_poller
        with patch('vulcan.orchestrator.task_queues.zmq.Poller', mock_poller):
            mock_context.socket.return_value = mock_socket
            mock_context_class.return_value = mock_context
            
            queue = CustomTaskQueue()
            
            status = queue.get_queue_status()
            
            self.assertIn('queue_type', status)
            self.assertEqual(status['queue_type'], 'zmq')
            self.assertIn('total_tasks', status)
        
        # Cleanup
        queue.shutdown()


# ============================================================
# TEST: FACTORY FUNCTION
# ============================================================

class TestFactoryFunction(unittest.TestCase):
    """Test create_task_queue factory function"""
    
    @unittest.skipIf(not RAY_AVAILABLE, "Ray not available")
    @patch('vulcan.orchestrator.task_queues.ray')
    def test_create_ray_queue(self, mock_ray):
        """Test creating Ray queue"""
        mock_ray.is_initialized.return_value = True
        
        queue = create_task_queue("ray")
        
        self.assertIsInstance(queue, RayTaskQueue)
        
        # Cleanup
        queue.shutdown()
    
    @unittest.skipIf(not CELERY_AVAILABLE, "Celery not available")
    @patch('vulcan.orchestrator.task_queues.Celery')
    def test_create_celery_queue(self, mock_celery):
        """Test creating Celery queue"""
        mock_app = MagicMock()
        mock_celery.return_value = mock_app
        
        queue = create_task_queue("celery")
        
        self.assertIsInstance(queue, CeleryTaskQueue)
        
        # Cleanup
        queue.shutdown()
    
    @unittest.skipIf(not ZMQ_AVAILABLE, "ZMQ not available")
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_create_custom_queue(self, mock_context):
        """Test creating custom ZMQ queue"""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_context.return_value = mock_ctx
        
        queue = create_task_queue("custom")
        
        self.assertIsInstance(queue, CustomTaskQueue)
        
        # Cleanup
        queue.shutdown()
    
    @unittest.skipIf(not ZMQ_AVAILABLE, "ZMQ not available")
    @patch('vulcan.orchestrator.task_queues.zmq.Context')
    def test_create_zmq_queue(self, mock_context):
        """Test creating ZMQ queue via 'zmq' name"""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_context.return_value = mock_ctx
        
        queue = create_task_queue("zmq")
        
        self.assertIsInstance(queue, CustomTaskQueue)
        
        # Cleanup
        queue.shutdown()
    
    def test_create_invalid_queue_type(self):
        """Test creating queue with invalid type"""
        with self.assertRaises(ValueError):
            create_task_queue("invalid_type")
    
    @unittest.skipIf(RAY_AVAILABLE, "Test requires Ray to be unavailable")
    def test_create_ray_queue_not_available(self):
        """Test creating Ray queue when not available"""
        with self.assertRaises(ImportError):
            create_task_queue("ray")
    
    @unittest.skipIf(CELERY_AVAILABLE, "Test requires Celery to be unavailable")
    def test_create_celery_queue_not_available(self):
        """Test creating Celery queue when not available"""
        with self.assertRaises(ImportError):
            create_task_queue("celery")
    
    @unittest.skipIf(ZMQ_AVAILABLE, "Test requires ZMQ to be unavailable")
    def test_create_zmq_queue_not_available(self):
        """Test creating ZMQ queue when not available"""
        with self.assertRaises(ImportError):
            create_task_queue("zmq")


# ============================================================
# TEST: AVAILABILITY FLAGS
# ============================================================

class TestAvailabilityFlags(unittest.TestCase):
    """Test availability flags"""
    
    def test_availability_flags_are_boolean(self):
        """Test that availability flags are boolean"""
        self.assertIsInstance(RAY_AVAILABLE, bool)
        self.assertIsInstance(CELERY_AVAILABLE, bool)
        self.assertIsInstance(ZMQ_AVAILABLE, bool)


# ============================================================
# TEST: INTEGRATION
# ============================================================

class TestIntegration(unittest.TestCase):
    """
    Integration tests
    FIXED: Using TestTaskQueue concrete implementation
    """
    
    def test_task_lifecycle_with_interface(self):
        """Test complete task lifecycle with concrete implementation"""
        queue = TestTaskQueue()
        
        # Submit task
        task_id = queue.submit_task({"test": "data"})
        
        # Check initial status
        status = queue.get_task_status(task_id)
        self.assertEqual(status, TaskStatus.PENDING)
        
        # Update to running
        metadata = queue.pending_tasks[task_id]
        metadata.status = TaskStatus.RUNNING
        metadata.started_at = time.time()
        
        status = queue.get_task_status(task_id)
        self.assertEqual(status, TaskStatus.RUNNING)
        
        # Complete task
        metadata.status = TaskStatus.COMPLETED
        metadata.completed_at = time.time()
        
        status = queue.get_task_status(task_id)
        self.assertEqual(status, TaskStatus.COMPLETED)
        
        # Cleanup
        queue.shutdown()
    
    def test_concurrent_task_status_checks(self):
        """Test concurrent task status checks"""
        queue = TestTaskQueue()
        
        # Add multiple tasks
        task_ids = []
        for i in range(10):
            task_id = queue.submit_task({"index": i})
            task_ids.append(task_id)
        
        # Check statuses concurrently
        results = []
        
        def check_status(tid):
            status = queue.get_task_status(tid)
            results.append((tid, status))
        
        threads = []
        for task_id in task_ids:
            t = threading.Thread(target=check_status, args=(task_id,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all checks succeeded
        self.assertEqual(len(results), 10)
        for tid, status in results:
            self.assertEqual(status, TaskStatus.PENDING)
        
        # Cleanup
        queue.shutdown()


# ============================================================
# TEST SUITE RUNNER
# ============================================================

def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestEnums))
    test_suite.addTest(unittest.makeSuite(TestTaskMetadata))
    test_suite.addTest(unittest.makeSuite(TestTaskQueueInterface))
    
    if RAY_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestRayTaskQueue))
    
    if CELERY_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestCeleryTaskQueue))
    
    if ZMQ_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestCustomTaskQueue))
    
    test_suite.addTest(unittest.makeSuite(TestFactoryFunction))
    test_suite.addTest(unittest.makeSuite(TestAvailabilityFlags))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    return test_suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())