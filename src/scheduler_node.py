import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
import weakref

# 2025: Photonic/LLM/energy integration, EU ethical_label, ITU F.748.53 compliance
try:
    from src.hardware_dispatcher import HardwareDispatcher
except ImportError:
    HardwareDispatcher = None

try:
    from src.llm_compressor import LLMCompressor
except ImportError:
    LLMCompressor = None

logger = logging.getLogger("SchedulerNode")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TaskManager:
    """
    Manages scheduled tasks with proper lifecycle control.
    
    Features:
    - Task registration and tracking
    - Graceful cancellation
    - Resource cleanup
    - Error handling for task failures
    """
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def _cancel_task_internal(self, task_id: str) -> bool:
        """
        Internal cancel method that assumes lock is already held.
        MUST be called with self._lock already acquired.
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        task = self.tasks[task_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {task_id} cancelled successfully")
            except Exception as e:
                logger.error(f"Error cancelling task {task_id}: {e}")
        
        # The done_callback will handle the actual removal, but we can try here too
        # to make cancellation appear more synchronous if needed.
        if self.tasks.pop(task_id, None):
            self.task_metadata.pop(task_id, None)
        
        return True
    
    async def register_task(self, task_id: str, task: asyncio.Task, metadata: Dict[str, Any] = None):
        """Register a task for lifecycle management and automatic cleanup."""
        
        manager_ref = weakref.ref(self)

        def _cleanup_task(fut: asyncio.Future):
            """Callback to remove the task from tracking upon completion."""
            manager = manager_ref()
            if not manager:
                # The TaskManager instance was garbage collected
                return

            try:
                # Log exceptions to prevent "Task exception was never retrieved" warnings
                if not fut.cancelled() and fut.exception():
                    exc = fut.exception()
                    logger.error(f"Task {task_id} finished with an unhandled exception: {exc}", exc_info=exc)

                # Use pop with a default to prevent errors if already removed (e.g., by cancel_task)
                if manager.tasks.pop(task_id, None):
                    manager.task_metadata.pop(task_id, None)
                    logger.info(f"Cleaned up completed/cancelled task: {task_id}")
            except Exception as e:
                logger.error(f"Error during automatic task cleanup for {task_id}: {e}")

        async with self._lock:
            if task_id in self.tasks:
                # Cancel and clean up existing task with the same ID
                # Use internal method since we already hold the lock
                await self._cancel_task_internal(task_id)
            
            self.tasks[task_id] = task
            self.task_metadata[task_id] = metadata or {}
            task.add_done_callback(_cleanup_task)
            logger.info(f"Registered task: {task_id}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        async with self._lock:
            return await self._cancel_task_internal(task_id)
    
    async def cancel_all_tasks(self):
        """Cancel all registered tasks."""
        async with self._lock:
            task_ids = list(self.tasks.keys())
        
        for task_id in task_ids:
            await self.cancel_task(task_id)
        
        logger.info("All tasks cancelled")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "task_id": task_id,
            "done": task.done(),
            "cancelled": task.cancelled(),
            "metadata": self.task_metadata.get(task_id, {})
        }
    
    def get_all_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks."""
        return {
            task_id: self.get_task_status(task_id)
            for task_id in self.tasks.keys()
        }


class SchedulerNode:
    """
    Schedules tasks based on time-based or event-based triggers for reactive workflows.
    Supports periodic execution and trigger edges for async/event-driven flows.
    2025: Integrates photonic/LLM/energy metrics, EU Code 2025, ITU F.748.53 compliance.
    
    FIXES APPLIED:
    - Proper task lifecycle management with cancellation
    - Task manager for tracking and cleanup
    - Fixed control flow (no unreachable exception handling)
    - Event triggers now properly schedule async tasks
    - Resource leak prevention
    - Context manager support for cleanup
    """
    
    def __init__(self):
        self.hardware = HardwareDispatcher() if HardwareDispatcher is not None else None
        self.compressor = LLMCompressor() if LLMCompressor is not None else None
        self.task_manager = TaskManager()
    
    async def _run_photonic_mvm(self, tensor: Any) -> Dict[str, Any]:
        """Run photonic MVM with error handling."""
        photonic_meta = {}
        energy_nj = None
        
        if self.hardware and tensor is not None:
            try:
                photonic_meta = self.hardware.run_photonic_mvm(tensor)
                energy_nj = photonic_meta.get('energy_nj', None)
            except Exception as e:
                logger.warning(f"Photonic MVM failed: {e}")
        
        return {"photonic_meta": photonic_meta, "energy_nj": energy_nj}
    
    async def _run_compression(self, tensor: Any) -> Dict[str, Any]:
        """Run compression validation with error handling."""
        compression_ok = True
        compression_meta = {}
        
        if self.compressor and tensor is not None:
            try:
                compressed = self.compressor.compress_tensor(tensor)
                compression_ok = self.compressor.validate_compression(compressed)
                compression_meta = {"compression": "ITU F.748.53", "valid": compression_ok}
                
                if not compression_ok:
                    raise ValueError("ITU F.748.53 compression validation failed")
            except Exception as e:
                logger.warning(f"Compression error: {e}")
                compression_ok = False
                compression_meta = {"compression": "ITU F.748.53", "valid": False, "error": str(e)}
        
        return {"compression_ok": compression_ok, "compression_meta": compression_meta}

    async def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SchedulerNode to schedule a task based on trigger or interval.

        Args:
            params: Dict with 'trigger' (e.g., 'periodic', 'event'), 'interval_ms' (for periodic),
                    'task_id' (optional identifier for task), 'tensor' (for photonic/LLM),
                    'max_iterations' (optional limit for periodic tasks).
            context: Runtime context for state, task registration, and auditing.

        Returns:
            Dict with scheduling status, energy/photonic/ethical/compression metadata, and audit info.
        """
        trigger = params.get('trigger', 'periodic')
        interval_ms = params.get('interval_ms', 1000)  # Default: 1 second
        task_id = params.get('task_id', 'default_task')
        tensor = params.get('tensor', None)
        max_iterations = params.get('max_iterations', None)  # Optional limit
        
        logger.info(f"Executing SchedulerNode with trigger={trigger}, interval_ms={interval_ms}, task_id={task_id}")

        # 2025: Prepare for photonic/LLM/energy/ITU metrics
        photonic_meta = {}
        energy_nj = None
        compression_ok = True
        compression_meta = {}
        ethical_label = params.get('ethical_label', None)

        try:
            if trigger == 'periodic':
                # FIXED: Proper task with cancellation support
                async def periodic_task():
                    iteration = 0
                    try:
                        while context.get('running', True):
                            # Check iteration limit
                            if max_iterations is not None and iteration >= max_iterations:
                                logger.info(f"Task {task_id} reached max iterations: {max_iterations}")
                                break
                            
                            # Update task status
                            context['tasks'] = context.get('tasks', {})
                            context['tasks'][task_id] = {
                                'status': 'running',
                                'last_run': datetime.utcnow().isoformat(),
                                'iteration': iteration
                            }
                            logger.info(f"Running task {task_id} iteration {iteration} at {context['tasks'][task_id]['last_run']}")
                            
                            # 2025: Hardware/photonic/energy per run
                            photonic_result = await self._run_photonic_mvm(tensor)
                            
                            # 2025: Compression audit
                            compression_result = await self._run_compression(tensor)
                            
                            # Store results in context
                            context['tasks'][task_id].update({
                                'energy_nj': photonic_result['energy_nj'],
                                'compression_ok': compression_result['compression_ok']
                            })
                            
                            iteration += 1
                            await asyncio.sleep(interval_ms / 1000.0)
                        
                        # Task completed normally
                        context['tasks'][task_id]['status'] = 'completed'
                        return {'status': 'completed', 'iterations': iteration}
                    
                    except asyncio.CancelledError:
                        # Task was cancelled
                        logger.info(f"Task {task_id} was cancelled after {iteration} iterations")
                        context['tasks'][task_id]['status'] = 'cancelled'
                        raise
                    
                    except Exception as e:
                        # Task error
                        logger.error(f"Task {task_id} error: {e}")
                        context['tasks'][task_id]['status'] = 'error'
                        context['tasks'][task_id]['error'] = str(e)
                        raise

                # FIXED: Properly register task with manager
                task = asyncio.create_task(periodic_task())
                await self.task_manager.register_task(task_id, task, {
                    'trigger': 'periodic',
                    'interval_ms': interval_ms,
                    'started_at': datetime.utcnow().isoformat()
                })
                
                # Store task reference in context for external management
                context['scheduled_tasks'] = context.get('scheduled_tasks', [])
                context['scheduled_tasks'].append(task)
                
                # Run initial photonic/compression for the response
                photonic_result = await self._run_photonic_mvm(tensor)
                photonic_meta = photonic_result['photonic_meta']
                energy_nj = photonic_result['energy_nj']
                
                compression_result = await self._run_compression(tensor)
                compression_ok = compression_result['compression_ok']
                compression_meta = compression_result['compression_meta']

            elif trigger == 'event':
                # FIXED: Event trigger now properly schedules async work
                if 'event_signal' not in context:
                    raise ValueError("Event trigger requires 'event_signal' in context")
                
                async def event_task():
                    try:
                        # Update task status
                        context['tasks'] = context.get('tasks', {})
                        context['tasks'][task_id] = {
                            'status': 'triggered',
                            'event': context['event_signal'],
                            'triggered_at': datetime.utcnow().isoformat()
                        }
                        logger.info(f"Event task {task_id} triggered by {context['event_signal']}")
                        
                        # Run photonic/compression processing
                        photonic_result = await self._run_photonic_mvm(tensor)
                        compression_result = await self._run_compression(tensor)
                        
                        # Store results
                        context['tasks'][task_id].update({
                            'energy_nj': photonic_result['energy_nj'],
                            'compression_ok': compression_result['compression_ok'],
                            'status': 'completed'
                        })
                        
                        return {
                            'status': 'completed',
                            'event': context['event_signal'],
                            **photonic_result,
                            **compression_result
                        }
                    
                    except asyncio.CancelledError:
                        logger.info(f"Event task {task_id} was cancelled")
                        context['tasks'][task_id]['status'] = 'cancelled'
                        raise
                    
                    except Exception as e:
                        logger.error(f"Event task {task_id} error: {e}")
                        context['tasks'][task_id]['status'] = 'error'
                        context['tasks'][task_id]['error'] = str(e)
                        raise
                
                # Create and register event task
                task = asyncio.create_task(event_task())
                await self.task_manager.register_task(task_id, task, {
                    'trigger': 'event',
                    'event_signal': context['event_signal'],
                    'started_at': datetime.utcnow().isoformat()
                })
                
                # Store task reference
                context['scheduled_tasks'] = context.get('scheduled_tasks', [])
                context['scheduled_tasks'].append(task)
                
                # Run initial photonic/compression for the response
                photonic_result = await self._run_photonic_mvm(tensor)
                photonic_meta = photonic_result['photonic_meta']
                energy_nj = photonic_result['energy_nj']
                
                compression_result = await self._run_compression(tensor)
                compression_ok = compression_result['compression_ok']
                compression_meta = compression_result['compression_meta']

            else:
                raise ValueError(f"Unsupported trigger type: {trigger}")

            # FIXED: Construct result without unreachable code
            result = {
                'status': 'scheduled',
                'task_id': task_id,
                'energy_nj': energy_nj,
                'photonic_meta': photonic_meta,
                'compression_ok': compression_ok,
                'compression_meta': compression_meta,
                'ethical_label': ethical_label,
                'audit': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'node_type': 'SchedulerNode',
                    'params': params,
                    'status': 'success',
                    'energy_nj': energy_nj,
                    'compression_ok': compression_ok,
                    'ethical_label': ethical_label
                }
            }
            
            context['audit_log'] = context.get('audit_log', [])
            context['audit_log'].append(result['audit'])
            logger.info(f"SchedulerNode success: {result['audit']}")
            
            return result

        except Exception as e:
            # FIXED: Now reachable - exception handling works properly
            logger.error(f"SchedulerNode error: {str(e)}")
            error_result = {
                'status': 'error',
                'task_id': task_id,
                'energy_nj': energy_nj,
                'photonic_meta': photonic_meta,
                'compression_ok': compression_ok,
                'compression_meta': compression_meta,
                'ethical_label': ethical_label,
                'audit': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'node_type': 'SchedulerNode',
                    'params': params,
                    'status': 'error',
                    'error': str(e)
                }
            }
            context['audit_log'] = context.get('audit_log', [])
            context['audit_log'].append(error_result['audit'])
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific scheduled task."""
        return await self.task_manager.cancel_task(task_id)
    
    async def cancel_all_tasks(self):
        """Cancel all scheduled tasks."""
        await self.task_manager.cancel_all_tasks()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        return self.task_manager.get_task_status(task_id)
    
    def get_all_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks."""
        return self.task_manager.get_all_tasks_status()
    
    async def shutdown(self):
        """Gracefully shutdown all tasks."""
        logger.info("Shutting down SchedulerNode...")
        await self.cancel_all_tasks()
        logger.info("SchedulerNode shutdown complete")


def _check_async_context() -> bool:
    """Check if we're already in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


async def async_dispatch_scheduler_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async dispatch function for SchedulerNode.

    Args:
        node: Dict with node type and params.
        context: Runtime context for state and auditing.

    Returns:
        Result of node execution.
    """
    node_type = node.get('type')
    params = node.get('params', {})

    for k in ('tensor', 'ethical_label'):
        if k in node:
            params[k] = node[k]

    if node_type == 'SchedulerNode':
        scheduler = SchedulerNode()
        return await scheduler.execute(params, context)
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def dispatch_scheduler_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Sync dispatch function for SchedulerNode with proper event loop handling.

    Args:
        node: Dict with node type and params.
        context: Runtime context for state and auditing.

    Returns:
        Result of node execution.
    """
    # FIXED: Check if already in async context
    if _check_async_context():
        raise RuntimeError(
            "dispatch_scheduler_node() cannot be called from an async context. "
            "Use async_dispatch_scheduler_node() instead."
        )
    
    return asyncio.run(async_dispatch_scheduler_node(node, context))


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SchedulerNode Production Demo")
    print("="*70 + "\n")
    
    # Demo usage with photonic/energy/compression/ethics extensions
    context = {'audit_log': [], 'running': True}

    # Example 1: Periodic task with iteration limit
    print("--- Example 1: Periodic Task (Limited Iterations) ---")
    periodic_node = {
        'type': 'SchedulerNode',
        'params': {
            'trigger': 'periodic',
            'interval_ms': 500,
            'task_id': 'periodic_1',
            'max_iterations': 3,  # Stop after 3 iterations
            'tensor': [[0.1, 0.2], [0.3, 0.4]],
            'ethical_label': 'EU2025:Safe'
        }
    }
    
    result = dispatch_scheduler_node(periodic_node, context)
    print(f"Periodic task scheduled: {result['status']}")
    print(f"Task ID: {result['task_id']}")
    print(f"Energy: {result['energy_nj']} nJ")
    print(f"Compression OK: {result['compression_ok']}")
    
    # Wait for task to complete
    import time
    print("\nWaiting for periodic task to complete...")
    time.sleep(2)
    
    # Example 2: Event-driven task
    print("\n--- Example 2: Event-Driven Task ---")
    context['event_signal'] = 'data_ready'
    event_node = {
        'type': 'SchedulerNode',
        'params': {
            'trigger': 'event',
            'task_id': 'event_1',
            'tensor': [[0.5, 0.6], [0.7, 0.8]],
            'ethical_label': 'EU2025:Safe'
        }
    }
    
    result = dispatch_scheduler_node(event_node, context)
    print(f"Event task scheduled: {result['status']}")
    print(f"Task ID: {result['task_id']}")
    print(f"Triggered by: {context['event_signal']}")
    
    # Wait for event task to complete
    print("\nWaiting for event task to complete...")
    time.sleep(1)
    
    # Example 3: Demonstrate task cancellation
    print("\n--- Example 3: Task Cancellation ---")
    
    async def cancellation_demo():
        scheduler = SchedulerNode()
        cancel_context = {'audit_log': [], 'running': True}
        
        # Start a long-running periodic task
        cancel_node = {
            'type': 'SchedulerNode',
            'params': {
                'trigger': 'periodic',
                'interval_ms': 500,
                'task_id': 'cancellable_task',
                'tensor': [[1.0, 2.0], [3.0, 4.0]],
                'ethical_label': 'EU2025:Safe'
            }
        }
        
        result = await scheduler.execute(cancel_node['params'], cancel_context)
        print(f"Started cancellable task: {result['task_id']}")
        
        # Let it run for a bit
        await asyncio.sleep(1)
        
        # Check status
        status = scheduler.get_task_status('cancellable_task')
        print(f"Task status: {status}")
        
        # Cancel the task
        cancelled = await scheduler.cancel_task('cancellable_task')
        print(f"Task cancelled: {cancelled}")
        
        # Verify cancellation
        final_status = scheduler.get_task_status('cancellable_task')
        print(f"Final task status: {final_status}")
    
    asyncio.run(cancellation_demo())
    
    # Print audit log
    print("\n--- Audit Log ---")
    for i, entry in enumerate(context['audit_log'], 1):
        print(f"\nEntry {i}:")
        print(f"  Timestamp: {entry['timestamp']}")
        print(f"  Status: {entry['status']}")
        if 'error' in entry:
            print(f"  Error: {entry['error']}")
        else:
            print(f"  Energy: {entry.get('energy_nj')} nJ")
            print(f"  Compression OK: {entry.get('compression_ok')}")
    
    print("\n" + "="*70)
    print("Demo Complete")
    print("="*70 + "\n")