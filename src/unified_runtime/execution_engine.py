"""
Execution Engine Module for Graphix IR
Core graph execution logic with parallel processing and context management
"""

import asyncio
import time
import json
import traceback
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
from datetime import datetime
import math # Import math

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

# Import metrics types carefully
try:
    from .execution_metrics import ExecutionMetrics, NodeExecutionStats
except ImportError:
    ExecutionMetrics = None
    NodeExecutionStats = None


logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION MODELS
# ============================================================================

class ExecutionStatus(Enum):
    """Execution status codes"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ExecutionMode(Enum):
    """Execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class ExecutionContext:
    """
    Execution context that flows through the graph
    """
    graph: Dict[str, Any]
    node_map: Dict[str, Dict[str, Any]]
    runtime: Any # Reference to UnifiedRuntime
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict) # Graph-level inputs
    recursion_depth: int = 0
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    parent_context: Optional['ExecutionContext'] = None
    start_time: float = field(default_factory=time.time)
    execution_id: str = ""

    def __post_init__(self):
        if not self.execution_id:
            # Generate unique execution ID
            exec_data = f"{id(self.graph)}_{time.time()}"
            self.execution_id = hashlib.md5(exec_data.encode()).hexdigest()[:16]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        return self.node_map.get(node_id)

    def set_output(self, node_id: str, output: Any):
        """Set output for a node"""
        self.outputs[node_id] = output

    def get_output(self, node_id: str) -> Any:
        """Get output for a node"""
        return self.outputs.get(node_id)

    def record_error(self, node_id: str, error: str):
        """Record error for a node"""
        self.errors[node_id] = error

    def add_audit_entry(self, entry: Dict[str, Any]):
        """Add entry to audit log"""
        entry['timestamp'] = datetime.now().isoformat()
        entry['execution_id'] = self.execution_id
        self.audit_log.append(entry)

    def create_child_context(self, graph: Dict[str, Any]) -> 'ExecutionContext':
        """Create child context for nested execution"""
        child = ExecutionContext(
            graph=graph,
            node_map={n['id']: n for n in graph.get('nodes', [])},
            runtime=self.runtime, # Propagate runtime
            metadata=self.metadata.copy(),
            recursion_depth=self.recursion_depth + 1,
            parent_context=self,
            # Inherit inputs? Or should they be explicitly mapped? For now, don't inherit.
        )
        return child

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary (primarily for node handlers)"""
        return {
            'execution_id': self.execution_id,
            'recursion_depth': self.recursion_depth,
            # Exclude outputs/errors from context passed to nodes
            'metadata': self.metadata,
            'runtime': self.runtime, # Pass runtime reference
            'graph': self.graph, # Pass graph reference
            'node_map': self.node_map, # Pass node_map reference
            'audit_log': self.audit_log, # Pass audit_log reference (mutable!)
            'inputs': self.inputs # Pass graph-level inputs
        }


@dataclass
class NodeExecutionResult:
    """Result from node execution"""
    node_id: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphExecutionResult:
    """Result from graph execution"""
    status: ExecutionStatus
    inputs: Dict[str, Any] = field(default_factory=dict) # Added inputs
    output: Dict[str, Any] = field(default_factory=dict) # Changed outputs -> output
    errors: Dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0
    nodes_executed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[ExecutionMetrics] = None # Added metrics

    def to_dict(self) -> Dict[str, Any]:
        # Convert status enum to string value for serialization
        status_value = self.status.value if isinstance(self.status, Enum) else str(self.status)

        return {
            'status': status_value,
            'inputs': self.inputs,
            'output': self.output,
            'errors': self.errors,
            'duration_ms': self.duration_ms,
            'nodes_executed': self.nodes_executed,
            'metadata': self.metadata,
            'metrics': self.metrics.to_dict() if self.metrics and hasattr(self.metrics, 'to_dict') else None
            # Audit log is often large, optionally exclude from simple dict conversion
            # 'audit_log': self.audit_log
        }


# ============================================================================
# EXECUTION SCHEDULER
# ============================================================================

class ExecutionScheduler:
    """
    Schedules and manages graph execution order
    """

    def __init__(self, graph: Dict[str, Any]):
        self.graph = graph
        self.node_map = {n['id']: n for n in graph.get('nodes', [])}
        self.edges = graph.get('edges', [])

        # Build dependency graph
        self.dependencies = self._build_dependencies()
        self.dependents = self._build_dependents()

        # Execution state
        self.executed = set()
        self.executing = set()
        self.failed = set()

        # Detect cycles
        self.has_cycles = self._detect_cycles()

    def _build_dependencies(self) -> Dict[str, Set[str]]:
        """Build dependency map (node -> set of nodes it depends on)"""
        deps = defaultdict(set)

        for edge in self.edges:
            from_node = self._get_edge_node(edge, 'from')
            to_node = self._get_edge_node(edge, 'to')

            if from_node and to_node and from_node in self.node_map and to_node in self.node_map: # Check nodes exist
                deps[to_node].add(from_node)

        # Add all nodes to ensure they're in the map
        for node_id in self.node_map:
            if node_id not in deps:
                deps[node_id] = set()

        return dict(deps)

    def _build_dependents(self) -> Dict[str, List[str]]:
        """Build dependent map (node -> list of nodes that depend on it)"""
        deps = defaultdict(list)

        for edge in self.edges:
            from_node = self._get_edge_node(edge, 'from')
            to_node = self._get_edge_node(edge, 'to')

            if from_node and to_node and from_node in self.node_map and to_node in self.node_map: # Check nodes exist
                deps[from_node].append(to_node)

        return dict(deps)

    def _get_edge_node(self, edge: Dict[str, Any], key: str) -> Optional[str]:
        """Extract node ID from edge"""
        value = edge.get(key)
        if isinstance(value, dict):
            node_id = value.get('node')
            # Ensure node_id exists in the graph
            return node_id if node_id in self.node_map else None
        # Ensure node_id exists in the graph
        return value if value in self.node_map else None


    def _detect_cycles(self) -> bool:
        """Detect if graph has cycles"""
        if not NETWORKX_AVAILABLE:
            # Simple DFS-based cycle detection
            visited = set()
            rec_stack = set()

            def has_cycle(node_id: str) -> bool:
                visited.add(node_id)
                rec_stack.add(node_id)

                for dep in self.dependents.get(node_id, []):
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

                rec_stack.remove(node_id)
                return False

            for node_id in self.node_map:
                if node_id not in visited:
                    if has_cycle(node_id):
                        logger.warning(f"Cycle detected involving node {node_id} (using DFS)")
                        return True

            return False
        else:
            # Use NetworkX for cycle detection
            try:
                G = nx.DiGraph()
                G.add_nodes_from(self.node_map.keys())

                for edge in self.edges:
                    from_node = self._get_edge_node(edge, 'from')
                    to_node = self._get_edge_node(edge, 'to')
                    if from_node and to_node:
                        G.add_edge(from_node, to_node)

                is_dag = nx.is_directed_acyclic_graph(G)
                if not is_dag:
                    cycles = list(nx.simple_cycles(G))
                    logger.warning(f"Cycle detected (using NetworkX): {cycles[0] if cycles else 'unknown'}")
                return not is_dag
            except Exception as e:
                 logger.error(f"NetworkX cycle detection failed: {e}. Falling back to DFS.")
                 # Fallback to DFS if NetworkX fails unexpectedly
                 return self._detect_cycles() # Call the DFS version recursively

    def get_ready_nodes(self) -> List[str]:
        """Get nodes that are ready to execute"""
        ready = []

        for node_id in self.node_map:
            # Skip if already done or currently running
            if node_id in self.executed or node_id in self.executing or node_id in self.failed:
                continue

            # Check if all dependencies are satisfied (i.e., executed successfully)
            deps = self.dependencies.get(node_id, set())
            if deps.issubset(self.executed):
                ready.append(node_id)
            # Log why a node isn't ready (optional, can be verbose)
            # else:
            #    unmet_deps = deps - self.executed
            #    if unmet_deps:
            #        logger.debug(f"Node {node_id} not ready, waiting for: {unmet_deps}")

        return ready

    def mark_executed(self, node_id: str):
        """Mark node as successfully executed"""
        self.executing.discard(node_id)
        self.executed.add(node_id)
        self.failed.discard(node_id) # Ensure it's not marked as failed

    def mark_failed(self, node_id: str, reason: str):
        """Mark node as failed"""
        self.executing.discard(node_id)
        self.failed.add(node_id)
        # We might still add to executed to prevent re-trying,
        # but failed set is the primary indicator. Let's keep it simple:
        # self.executed.add(node_id)

    def mark_executing(self, node_id: str):
        """Mark node as currently executing"""
        self.executing.add(node_id)

    def get_execution_layers(self) -> List[List[str]]:
        """Get execution layers for parallel processing"""
        if self.has_cycles:
            return []

        layers = []
        temp_executed = set()
        nodes_in_layers = set()

        while len(nodes_in_layers) < len(self.node_map):
            layer = []

            for node_id in self.node_map:
                # Skip if already assigned to a layer or already processed conceptually
                if node_id in nodes_in_layers or node_id in temp_executed:
                    continue

                deps = self.dependencies.get(node_id, set())
                # Ready if all dependencies are in conceptually processed set
                if deps.issubset(temp_executed):
                    layer.append(node_id)

            if not layer:
                # No progress possible - might have unreachable nodes or an issue
                unreachable = set(self.node_map.keys()) - nodes_in_layers
                if unreachable:
                     logger.warning(f"Could not schedule all nodes. Unreachable/Cyclic?: {unreachable}")
                break

            layers.append(layer)
            # Add nodes in this layer to the set of nodes assigned to layers
            nodes_in_layers.update(layer)
            # Update the conceptually processed set for the next iteration
            temp_executed.update(layer)

        return layers

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of nodes"""
        if self.has_cycles:
            return []

        if NETWORKX_AVAILABLE:
            try:
                G = nx.DiGraph()
                G.add_nodes_from(self.node_map.keys())

                for edge in self.edges:
                    from_node = self._get_edge_node(edge, 'from')
                    to_node = self._get_edge_node(edge, 'to')
                    if from_node and to_node:
                        G.add_edge(from_node, to_node)

                return list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible: # Catch cycle error specifically
                logger.warning("NetworkX detected cycle during topological sort.")
                return []
            except Exception as e:
                 logger.error(f"NetworkX topological sort failed: {e}. Falling back to manual.")
                 # Fallback to manual if NetworkX fails

        # Manual topological sort (Kahn's algorithm)
        in_degree = defaultdict(int)
        # Initialize in-degree for all nodes
        for node_id in self.node_map:
            in_degree[node_id] = 0 # Start with 0

        # Calculate actual in-degrees
        for node_id in self.node_map:
            for dependent in self.dependents.get(node_id, []):
                 in_degree[dependent] += 1

        queue = deque([n for n in self.node_map if in_degree[n] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for dependent in self.dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) == len(self.node_map):
            return result
        else:
            # This indicates a cycle if the graph was fully connected
            logger.warning("Manual topological sort failed - graph may contain cycles or be disconnected.")
            return []


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """
    Core execution engine for graph processing
    """

    def __init__(self,
                 runtime: Any, # Pass runtime reference
                 max_parallel: int = 10,
                 timeout_seconds: float = 300, # Default timeout from config
                 enable_metrics: bool = True,
                 enable_streaming: bool = True, # Add streaming/batch flags
                 enable_batch: bool = True):

        self.runtime = runtime
        self.max_parallel = max_parallel
        # <<< --- FIX for timeout --- >>>
        # Use the timeout_seconds passed during initialization
        self.timeout_seconds = timeout_seconds
        # <<< --- END FIX --- >>>
        self.enable_metrics = enable_metrics
        self.enable_streaming = enable_streaming
        self.enable_batch = enable_batch
        self._shutdown_event = asyncio.Event() # For graceful shutdown

        # Thread pool primarily for running synchronous node handlers without blocking asyncio loop
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)

        # Execution cache (simple dict for now)
        self.execution_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock() # Lock for cache access
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"ExecutionEngine initialized with max_parallel={max_parallel}, timeout={self.timeout_seconds}s") # Log timeout

    async def run_graph(self,
                        context: ExecutionContext,
                        mode: ExecutionMode,
                        metrics: Optional[ExecutionMetrics] = None
                       ) -> Union[GraphExecutionResult, AsyncIterator[GraphExecutionResult]]:
        """
        Main graph execution entry point, replacing execute_graph.
        Supports PARALLEL, SEQUENTIAL, and STREAMING modes.
        """
        start_time_ns = time.time_ns()
        scheduler = ExecutionScheduler(context.graph)

        if scheduler.has_cycles:
             logger.error("Graph execution aborted due to cycle detection.")
             return GraphExecutionResult(
                status=ExecutionStatus.FAILED,
                errors={'_graph': 'Graph contains cycles'},
                duration_ms=0,
                nodes_executed=0,
                metadata={"recursion_depth": context.recursion_depth},
                audit_log=context.audit_log,
                metrics=metrics,
                inputs=context.inputs
            )

        context.add_audit_entry({'event': 'run_graph_start', 'mode': mode.value if mode else 'unknown'})

        # <<< --- START Timeout Fix --- >>>
        try:
            if mode == ExecutionMode.STREAMING and self.enable_streaming:
                # Streaming needs careful timeout handling within the generator itself,
                # or a wrapper that manages the timeout externally.
                # For now, let's skip the main timeout wrapper for streaming.
                # TODO: Implement timeout for streaming if required.
                logger.debug("Timeout wrapper skipped for STREAMING mode.")
                result_or_iterator = self._run_streaming(context, scheduler, metrics)
            elif mode == ExecutionMode.SEQUENTIAL:
                 result_or_iterator = await asyncio.wait_for(
                     self._run_sequential(context, scheduler, metrics),
                     timeout=self.timeout_seconds
                 )
            else: # Default to PARALLEL
                 result_or_iterator = await asyncio.wait_for(
                     self._run_parallel(context, scheduler, metrics),
                     timeout=self.timeout_seconds
                 )

        except asyncio.TimeoutError:
            logger.error(f"Graph execution timed out after {self.timeout_seconds}s.")
            self._shutdown_event.set() # Signal engine components to stop
            duration_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
            # Return a specific TIMEOUT result
            final_result = GraphExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                errors={'_graph': f'Execution timed out after {self.timeout_seconds}s'},
                duration_ms=duration_ms,
                nodes_executed=len(scheduler.executed) + len(scheduler.failed), # Count nodes attempted
                metadata={"recursion_depth": context.recursion_depth, "timeout_triggered": True},
                audit_log=context.audit_log,
                metrics=metrics,
                inputs=context.inputs
            )
            context.add_audit_entry({
                'event': 'run_graph_end',
                'status': ExecutionStatus.TIMEOUT.value,
                'duration_ms': duration_ms
            })
            # Ensure metrics are finalized even on timeout
            if metrics: metrics.finalize_graph()
            return final_result # Return the timeout result directly
        except Exception as e:
             # Handle other potential exceptions during execution
             logger.error(f"Graph execution failed with unexpected error: {e}", exc_info=True)
             duration_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
             final_result = GraphExecutionResult(
                 status=ExecutionStatus.FAILED,
                 errors={'_graph': f'Unexpected execution error: {str(e)}'},
                 duration_ms=duration_ms,
                 nodes_executed=len(scheduler.executed) + len(scheduler.failed),
                 metadata={"recursion_depth": context.recursion_depth},
                 audit_log=context.audit_log,
                 metrics=metrics,
                 inputs=context.inputs
             )
             context.add_audit_entry({
                'event': 'run_graph_end',
                'status': ExecutionStatus.FAILED.value,
                'error': str(e),
                'duration_ms': duration_ms
             })
             if metrics: metrics.finalize_graph()
             return final_result # Return the failure result
        # <<< --- END Timeout Fix --- >>>


        # Finalize non-streaming result (if no timeout/error occurred)
        # Note: Streaming returns the iterator directly before this block
        if isinstance(result_or_iterator, GraphExecutionResult):
            result = result_or_iterator # Assign the result if it's not the iterator
            duration_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
            result.duration_ms = duration_ms
            result.audit_log = context.audit_log
            result.metrics = metrics # Attach metrics object
            result.inputs = context.inputs # Attach original inputs

            context.add_audit_entry({
                'event': 'run_graph_end',
                'status': result.status.value,
                'duration_ms': duration_ms
            })
            if metrics: metrics.finalize_graph() # Finalize metrics here too
            return result
        else: # It was the streaming iterator
             # The iterator handles its own finalization yield
             return result_or_iterator # Return the iterator


    async def _run_parallel(self,
                            context: ExecutionContext,
                            scheduler: ExecutionScheduler,
                            metrics: Optional[ExecutionMetrics]
                           ) -> GraphExecutionResult:
        """Executes the graph layer by layer in parallel."""
        tasks: Dict[str, asyncio.Task] = {}
        nodes_executed = 0
        nodes_succeeded = 0

        while len(scheduler.executed) + len(scheduler.failed) < len(context.node_map):
            if self._shutdown_event.is_set():
                 logger.warning("Shutdown requested during parallel execution.")
                 break

            ready_nodes = scheduler.get_ready_nodes()

            # Limit concurrency
            num_can_run = self.max_parallel - len(scheduler.executing)
            nodes_to_run = ready_nodes[:num_can_run]

            if not nodes_to_run and not scheduler.executing:
                # No nodes are ready and none are running - graph might be stuck or finished
                unprocessed_nodes = set(context.node_map.keys()) - scheduler.executed - scheduler.failed
                if unprocessed_nodes:
                    logger.warning(f"Execution stuck? Ready: {ready_nodes}, Executing: {scheduler.executing}, Unprocessed: {unprocessed_nodes}")
                break # Exit loop

            for node_id in nodes_to_run:
                if node_id not in tasks:
                    scheduler.mark_executing(node_id)
                    task = asyncio.create_task(self._run_single_node(node_id, context, scheduler, metrics))
                    tasks[node_id] = task

            if not tasks: # No tasks running or ready to start
                 await asyncio.sleep(0.01) # Avoid busy-waiting
                 continue

            # Wait for any task to complete
            done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)

            # Process completed tasks
            completed_ids = []
            for task in done:
                # Find node_id associated with this task
                node_id = None
                for nid, t in tasks.items():
                    if t == task:
                        node_id = nid
                        break

                if node_id:
                    completed_ids.append(node_id)
                    try:
                        result: NodeExecutionResult = await task # Get result/exception
                        nodes_executed += 1
                        if result.status == ExecutionStatus.SUCCESS:
                            scheduler.mark_executed(node_id)
                            nodes_succeeded += 1
                        else:
                            scheduler.mark_failed(node_id, result.error or "Unknown node error")
                            context.record_error(node_id, result.error or "Unknown node error")
                            # Check for critical failure after marking
                            if self._is_critical_node(node_id, context):
                                 logger.warning(f"Critical node {node_id} failed. Stopping parallel execution.")
                                 self._shutdown_event.set() # Signal shutdown

                    except Exception as e:
                        # Task raised an unexpected exception
                        nodes_executed += 1
                        error_msg = f"Node {node_id} raised unexpected error: {e}"
                        logger.error(error_msg, exc_info=True)
                        scheduler.mark_failed(node_id, error_msg)
                        context.record_error(node_id, error_msg)
                        if self._is_critical_node(node_id, context):
                            logger.warning(f"Critical node {node_id} failed unexpectedly. Stopping parallel execution.")
                            self._shutdown_event.set() # Signal shutdown

            # Remove completed tasks from the tracking dict
            for node_id in completed_ids:
                tasks.pop(node_id, None)

        # --- Finalization ---
        # Cancel any remaining tasks if shutdown was requested
        if self._shutdown_event.is_set():
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True) # Wait for cancellations

        status = ExecutionStatus.SUCCESS if not scheduler.failed and not self._shutdown_event.is_set() else ExecutionStatus.FAILED
        if self._shutdown_event.is_set(): status = ExecutionStatus.CANCELLED # Or FAILED? Let's use CANCELLED

        output_nodes_map = self._get_output_nodes(context)
        final_outputs = {
            out_id: context.outputs.get(out_id)
            for out_id in output_nodes_map if out_id in context.outputs
        }
        # Simplify single output node case if desired (matching previous logic)
        final_output_payload = final_outputs
        # if len(final_outputs) == 1:
        #    single_output_val = list(final_outputs.values())[0]
        #    # If the output node handler returned a dict with 'result' or 'output'
        #    if isinstance(single_output_val, dict):
        #         final_output_payload = single_output_val.get('result', single_output_val.get('output', single_output_val))
        #    else:
        #         final_output_payload = single_output_val


        return GraphExecutionResult(
            status=status,
            output=final_output_payload,
            errors=context.errors,
            nodes_executed=nodes_executed, # Count nodes attempted
            metadata={"recursion_depth": context.recursion_depth}
        )

    async def _run_sequential(self,
                             context: ExecutionContext,
                             scheduler: ExecutionScheduler,
                             metrics: Optional[ExecutionMetrics]
                            ) -> GraphExecutionResult:
        """Executes the graph node by node following topological order."""
        exec_order = scheduler.get_topological_order()
        nodes_executed = 0
        nodes_succeeded = 0

        if not exec_order and context.node_map:
             # This indicates a cycle or disconnected graph wasn't handled earlier
             logger.error("Sequential execution failed: Could not determine valid execution order (cycle?).")
             return GraphExecutionResult(
                status=ExecutionStatus.FAILED,
                errors={'_graph': 'Could not determine valid execution order (cycle?).'},
                nodes_executed=0,
                metadata={"recursion_depth": context.recursion_depth}
             )

        for node_id in exec_order:
             if self._shutdown_event.is_set():
                  logger.warning("Shutdown requested during sequential execution.")
                  break

             scheduler.mark_executing(node_id)
             try:
                result = await self._run_single_node(node_id, context, scheduler, metrics)
                nodes_executed += 1
                if result.status == ExecutionStatus.SUCCESS:
                    scheduler.mark_executed(node_id)
                    nodes_succeeded += 1
                else:
                    scheduler.mark_failed(node_id, result.error or "Unknown node error")
                    context.record_error(node_id, result.error or "Unknown node error")
                    if self._is_critical_node(node_id, context):
                         logger.warning(f"Critical node {node_id} failed. Stopping sequential execution.")
                         break # Stop processing

             except Exception as e:
                # Unexpected exception during node run
                nodes_executed += 1
                error_msg = f"Node {node_id} raised unexpected error: {e}"
                logger.error(error_msg, exc_info=True)
                scheduler.mark_failed(node_id, error_msg)
                context.record_error(node_id, error_msg)
                if self._is_critical_node(node_id, context):
                    logger.warning(f"Critical node {node_id} failed unexpectedly. Stopping sequential execution.")
                    break # Stop processing

        # --- Finalization ---
        status = ExecutionStatus.SUCCESS if not scheduler.failed and not self._shutdown_event.is_set() else ExecutionStatus.FAILED
        if self._shutdown_event.is_set(): status = ExecutionStatus.CANCELLED

        output_nodes_map = self._get_output_nodes(context)
        final_outputs = {
            out_id: context.outputs.get(out_id)
            for out_id in output_nodes_map if out_id in context.outputs
        }
        final_output_payload = final_outputs
        # if len(final_outputs) == 1:
        #    single_output_val = list(final_outputs.values())[0]
        #    if isinstance(single_output_val, dict):
        #         final_output_payload = single_output_val.get('result', single_output_val.get('output', single_output_val))
        #    else:
        #         final_output_payload = single_output_val

        return GraphExecutionResult(
            status=status,
            output=final_output_payload,
            errors=context.errors,
            nodes_executed=nodes_executed,
            metadata={"recursion_depth": context.recursion_depth}
        )

    async def _run_streaming(self,
                           context: ExecutionContext,
                           scheduler: ExecutionScheduler,
                           metrics: Optional[ExecutionMetrics]
                          ) -> AsyncIterator[GraphExecutionResult]:
        """Executes the graph in streaming mode, yielding results as they become available."""
        # Similar logic to parallel, but yields partial results
        tasks: Dict[str, asyncio.Task] = {}
        nodes_executed = 0
        nodes_succeeded = 0
        last_yield_time = time.time()
        yield_interval = 0.1 # Yield at least every 100ms if something changed

        while len(scheduler.executed) + len(scheduler.failed) < len(context.node_map):
            if self._shutdown_event.is_set():
                 logger.warning("Shutdown requested during streaming execution.")
                 break

            ready_nodes = scheduler.get_ready_nodes()
            num_can_run = self.max_parallel - len(scheduler.executing)
            nodes_to_run = ready_nodes[:num_can_run]

            if not nodes_to_run and not scheduler.executing:
                 unprocessed_nodes = set(context.node_map.keys()) - scheduler.executed - scheduler.failed
                 if unprocessed_nodes:
                     logger.warning(f"Streaming execution stuck? Unprocessed: {unprocessed_nodes}")
                 break

            for node_id in nodes_to_run:
                if node_id not in tasks:
                    scheduler.mark_executing(node_id)
                    task = asyncio.create_task(self._run_single_node(node_id, context, scheduler, metrics))
                    tasks[node_id] = task

            if not tasks:
                 await asyncio.sleep(0.01)
                 continue

            done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED, timeout=yield_interval)

            processed_task = False
            completed_ids = []
            for task in done:
                processed_task = True
                node_id = None
                for nid, t in tasks.items():
                    if t == task:
                        node_id = nid
                        break

                if node_id:
                    completed_ids.append(node_id)
                    try:
                        result: NodeExecutionResult = await task
                        nodes_executed += 1
                        if result.status == ExecutionStatus.SUCCESS:
                            scheduler.mark_executed(node_id)
                            nodes_succeeded += 1
                        else:
                            scheduler.mark_failed(node_id, result.error or "Unknown node error")
                            context.record_error(node_id, result.error or "Unknown node error")
                            if self._is_critical_node(node_id, context):
                                 self._shutdown_event.set()

                    except Exception as e:
                        nodes_executed += 1
                        error_msg = f"Node {node_id} raised unexpected error: {e}"
                        logger.error(error_msg, exc_info=True)
                        scheduler.mark_failed(node_id, error_msg)
                        context.record_error(node_id, error_msg)
                        if self._is_critical_node(node_id, context):
                            self._shutdown_event.set()

            for node_id in completed_ids:
                tasks.pop(node_id, None)

            # Yield intermediate result if a task finished or interval passed
            current_time = time.time()
            if processed_task or (current_time - last_yield_time > yield_interval):
                status = ExecutionStatus.RUNNING if not self._shutdown_event.is_set() else ExecutionStatus.CANCELLED
                if scheduler.failed: status = ExecutionStatus.FAILED # If any failed, mark as failed

                output_nodes_map = self._get_output_nodes(context)
                current_outputs = {
                    out_id: context.outputs.get(out_id)
                    for out_id in output_nodes_map if out_id in context.outputs
                }
                # simplified_output = current_outputs
                # if len(current_outputs) == 1:
                #     single_output_val = list(current_outputs.values())[0]
                #     if isinstance(single_output_val, dict):
                #          simplified_output = single_output_val.get('result', single_output_val.get('output', single_output_val))
                #     else:
                #          simplified_output = single_output_val

                yield GraphExecutionResult(
                    status=status,
                    output=current_outputs,
                    errors=context.errors.copy(), # Yield copy
                    nodes_executed=nodes_executed,
                    metadata={"intermediate": True, "recursion_depth": context.recursion_depth},
                    metrics=metrics # Yield current metrics state
                )
                last_yield_time = current_time

        # --- Final Yield ---
        if self._shutdown_event.is_set():
             for task in tasks.values(): task.cancel()
             await asyncio.gather(*tasks.values(), return_exceptions=True)

        final_status = ExecutionStatus.SUCCESS if not scheduler.failed and not self._shutdown_event.is_set() else ExecutionStatus.FAILED
        if self._shutdown_event.is_set(): final_status = ExecutionStatus.CANCELLED

        output_nodes_map = self._get_output_nodes(context)
        final_outputs = {
            out_id: context.outputs.get(out_id)
            for out_id in output_nodes_map if out_id in context.outputs
        }
        # simplified_output = final_outputs
        # if len(final_outputs) == 1:
        #     single_output_val = list(final_outputs.values())[0]
        #     if isinstance(single_output_val, dict):
        #          simplified_output = single_output_val.get('result', single_output_val.get('output', single_output_val))
        #     else:
        #          simplified_output = single_output_val


        yield GraphExecutionResult(
            status=final_status,
            output=final_outputs,
            errors=context.errors,
            nodes_executed=nodes_executed,
            metadata={"final": True, "recursion_depth": context.recursion_depth},
            metrics=metrics # Yield final metrics state
        )
        context.add_audit_entry({'event': 'run_graph_end', 'status': final_status.value})


    async def _run_single_node(self,
                             node_id: str,
                             context: ExecutionContext,
                             scheduler: ExecutionScheduler, # Pass scheduler for state updates
                             metrics: Optional[ExecutionMetrics]
                            ) -> NodeExecutionResult:
        """
        Executes a single node, handles inputs, handler lookup, execution, metrics, and output storage.
        """
        node = context.get_node(node_id)
        if not node:
             return NodeExecutionResult(node_id=node_id, status=ExecutionStatus.FAILED, error="Node definition not found")

        node_type = node.get("type")
        cache_key = self._compute_cache_key(node, context)
        is_cached = False
        stats: Optional[NodeExecutionStats] = None

        # Check cache
        if cache_key:
             with self._cache_lock:
                  if cache_key in self.execution_cache:
                       cached_output = self.execution_cache[cache_key]
                       is_cached = True
                       self.cache_hits += 1
                       # <<< --- FIX for metrics call --- >>>
                       # Correctly call record_node_start which returns the stats object
                       if metrics and NodeExecutionStats:
                           stats = metrics.record_node_start(node_id=node_id, node_type=node_type, cache_hit=True)
                       # <<< --- END FIX --- >>>
                       context.set_output(node_id, cached_output)
                       logger.debug(f"Cache hit for node {node_id}")
                       # Record metrics for cache hit
                       if stats and metrics:
                           # <<< --- FIX for metrics call --- >>>
                           # Remove the invalid cache_hit argument
                           metrics.record_node_end(stats, status="success")
                           # <<< --- END FIX --- >>>
                       return NodeExecutionResult(
                            node_id=node_id,
                            status=ExecutionStatus.SUCCESS,
                            output=cached_output,
                            duration_ms=0, # Cache hits are instant for this calculation
                            metadata={'cached': True}
                       )
             # Only count miss if caching was attempted and missed
             self.cache_misses += 1

        # Record node start metric if not cached
        # <<< --- FIX for metrics call --- >>>
        # Correctly call record_node_start which returns the stats object
        if metrics and NodeExecutionStats and not is_cached:
             stats = metrics.record_node_start(node_id=node_id, node_type=node_type, cache_hit=False)
        # <<< --- END FIX --- >>>

        t0_ns = time.time_ns()
        output = None
        result_status = ExecutionStatus.FAILED # Default to failed
        err_msg = None

        try:
            # Gather inputs
            inputs_dict = await self._gather_inputs(node_id, context)

            # Lookup handler
            handler = self._get_node_executor(node, context)
            if not handler:
                raise ValueError(f"No executor found for node type: {node_type}")

            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                 output = await handler(node, context.to_dict(), inputs_dict)
            else:
                 # Run sync handler in thread pool executor
                 loop = asyncio.get_running_loop()
                 output = await loop.run_in_executor(
                     self.executor,
                     handler,
                     node,
                     context.to_dict(),
                     inputs_dict
                 )

            # Success
            # Ensure output is JSON serializable (node handlers *should* do this)
            try:
                json.dumps(output, default=str) # Test serialization
            except (TypeError, OverflowError) as json_err:
                 logger.warning(f"Output of node {node_id} ({node_type}) is not JSON serializable: {json_err}. Storing raw.")
                 # Decide whether to raise error or store raw: storing raw for now
                 pass # Store the raw output

            context.set_output(node_id, output)
            result_status = ExecutionStatus.SUCCESS
            context.add_audit_entry({'event': 'node_success', 'node_id': node_id, 'node_type': node_type})

            # Cache output if applicable
            if cache_key and self._is_deterministic_node(node):
                 with self._cache_lock:
                      self.execution_cache[cache_key] = output


        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Node {node_id} ({node_type}) failed: {err_msg}", exc_info=True)
            context.record_error(node_id, err_msg)
            result_status = ExecutionStatus.FAILED
            context.add_audit_entry({'event': 'node_failure', 'node_id': node_id, 'node_type': node_type, 'error': err_msg})
            output = None # Ensure output is None on failure

        duration_ms = (time.time_ns() - t0_ns) / 1_000_000.0

        # Record node end metric
        if stats and metrics:
             metrics.record_node_end(stats, status=result_status.value, error_message=err_msg)

        return NodeExecutionResult(
            node_id=node_id,
            status=result_status,
            output=output,
            error=err_msg,
            duration_ms=duration_ms,
            metadata={'cached': is_cached}
        )

    async def _gather_inputs(self, node_id: str, context: ExecutionContext) -> Dict[str, Any]:
        """
        Gather inputs for a node from connected edges and context.
        """
        inputs = {}
        processed_edges = set() # Track edges to handle multi-input ports correctly

        # Process incoming edges
        for edge in context.graph.get('edges', []):
            edge_id = edge.get('id', f"{edge.get('from')}_{edge.get('to')}") # Create a unique ID for the edge if none exists
            to_node_spec = edge.get('to')
            to_node_id = None
            to_port = 'input' # Default port name

            if isinstance(to_node_spec, dict):
                to_node_id = to_node_spec.get('node')
                to_port = to_node_spec.get('port', 'input')
            elif isinstance(to_node_spec, str):
                to_node_id = to_node_spec

            # If this edge targets the current node
            if to_node_id == node_id and edge_id not in processed_edges:
                from_node_spec = edge.get('from')
                from_node_id = None
                from_port = 'output' # Default source port

                if isinstance(from_node_spec, dict):
                    from_node_id = from_node_spec.get('node')
                    from_port = from_node_spec.get('port', 'output')
                elif isinstance(from_node_spec, str):
                    from_node_id = from_node_spec

                if from_node_id and from_node_id in context.outputs:
                    source_output = context.outputs[from_node_id]
                    value_to_pass = source_output # Default to passing the whole output

                    # Try to extract specific port value if source output is a dict
                    if isinstance(source_output, dict):
                         # Prioritize the specified from_port
                         if from_port in source_output:
                              value_to_pass = source_output[from_port]
                         # Fallbacks for common output keys if port doesn't match
                         elif 'output' in source_output:
                              value_to_pass = source_output['output']
                         elif 'result' in source_output:
                              value_to_pass = source_output['result']
                         elif 'value' in source_output:
                              value_to_pass = source_output['value']
                         # Add more common keys if needed
                         # else: keep value_to_pass as the whole dict

                    # Assign value to the target port
                    inputs[to_port] = value_to_pass
                    processed_edges.add(edge_id)
                elif from_node_id:
                     logger.warning(f"Input edge for {node_id} from {from_node_id} has no output data yet.")
                     # Decide handling: maybe raise error, pass None, or let node handle missing input
                     inputs[to_port] = None # Pass None for missing optional inputs

        # Inject graph-level inputs if node is an InputNode or similar
        node = context.get_node(node_id)
        if node and node.get("type") in ("INPUT", "InputNode"): # Match common names
            input_key = node.get("params", {}).get("key", "default_input")
            if input_key in context.inputs:
                 # Input nodes typically output directly, overriding edge inputs if name matches
                 inputs['output'] = context.inputs[input_key] # Assuming 'output' is the standard port
                 logger.debug(f"Injected graph input '{input_key}' into InputNode {node_id}")
            elif 'input' in context.inputs and input_key == 'default_input': # Handle simple 'input' key
                 inputs['output'] = context.inputs['input']
                 logger.debug(f"Injected graph input 'input' into InputNode {node_id}")


        return inputs

    def _get_edge_node(self, edge: Dict[str, Any], key: str) -> Optional[str]:
        """Extract node ID from edge"""
        value = edge.get(key)
        if isinstance(value, dict):
            return value.get('node')
        return value if isinstance(value, str) else None

    def _get_node_executor(self, node: Dict[str, Any], context: ExecutionContext) -> Optional[Callable]:
        """
        Get executor function for node type from runtime.
        """
        node_type = node.get('type')

        # Use the get_node_executor method from the runtime, as specified by the updated API
        if context.runtime and hasattr(context.runtime, 'get_node_executor'):
            executor = context.runtime.get_node_executor(node_type)
            if executor:
                return executor # Found it!

        # Fallback for old handlers (if any)
        if context.runtime and hasattr(context.runtime, 'node_handlers'):
            return context.runtime.node_handlers.get(node_type)

        logger.warning(f"Could not find node executor for type '{node_type}' in runtime.")
        return None # Indicate handler not found


    def _compute_cache_key(self, node: Dict[str, Any], context: ExecutionContext) -> str:
        """Compute cache key for node result"""
        if not self._is_deterministic_node(node):
            return ""

        input_values_for_key = {}
        scheduler = ExecutionScheduler(context.graph) # Need scheduler to access dependencies

        # Iterate through dependencies (nodes that feed into this one)
        for dep_id in scheduler.dependencies.get(node.get('id'), set()):
            if dep_id in context.outputs:
                 input_val = context.outputs[dep_id]
                 # Simple representation: hash the JSON string
                 try:
                      input_repr = hashlib.md5(json.dumps(input_val, sort_keys=True, default=str).encode()).hexdigest()[:8]
                 except (TypeError, ValueError) as e:
                      # If value can't be hashed, use type representation
                      logger.debug(f"Could not hash input value: {e}")
                      input_repr = f"unhashable_{type(input_val).__name__}"
                 input_values_for_key[dep_id] = input_repr
            # else: Dependency output not ready - shouldn't happen if called correctly

        key_data = {
            'type': node.get('type'),
            'params': node.get('params', {}),
            'inputs_hashed': sorted(input_values_for_key.items())
        }

        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except TypeError:
             # Fallback if params are non-serializable
             params_repr = {k: type(v).__name__ for k, v in node.get('params', {}).items()}
             key_data['params'] = params_repr
             key_str = json.dumps(key_data, sort_keys=True, default=str)

        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_deterministic_node(self, node: Dict[str, Any]) -> bool:
        """Check if node produces deterministic output"""
        non_deterministic_types = [
            'RandomNode', 'GenerativeNode', 'SchedulerNode',
            'SearchNode', 'ExternalAPINode', # Common examples
            'INPUT', 'InputNode' # Input nodes depend on external state
        ]
        if node.get("params", {}).get("is_deterministic") is False:
             return False
        return node.get('type') not in non_deterministic_types

    def _is_critical_node(self, node_id: str, context: ExecutionContext) -> bool:
        """Check if node failure should stop execution"""
        node = context.get_node(node_id)
        if node:
            if node.get('critical') is True: # Explicit critical flag
                return True
            # Output nodes are implicitly critical
            if node.get('type') in ('OUTPUT', 'OutputNode'):
                return True
        return False

    def _should_stop_execution(self, context: ExecutionContext, scheduler: ExecutionScheduler) -> bool:
        """Determine if execution should stop based on critical failures."""
        # Only stop if a CRITICAL node failed. Let non-critical failures continue.
        for failed_id in scheduler.failed:
            if self._is_critical_node(failed_id, context):
                logger.warning(f"Stopping execution because critical node '{failed_id}' failed.")
                return True
        return False

    def _get_output_nodes(self, context: ExecutionContext) -> List[str]:
        """Get IDs of output nodes or sink nodes if no explicit outputs."""
        output_nodes = [
            node_id for node_id, node in context.node_map.items()
            if node.get('type') in ('OUTPUT', 'OutputNode')
        ]

        if not output_nodes:
            # Fallback to sink nodes (nodes with no outgoing edges)
            all_nodes = set(context.node_map.keys())
            source_nodes = set()
            for edge in context.graph.get('edges', []):
                from_node = self._get_edge_node(edge, 'from')
                if from_node:
                    source_nodes.add(from_node)

            output_nodes = list(all_nodes - source_nodes)

            # If still empty (e.g., single node graph), consider all nodes potential outputs?
            # Or handle based on context? Let's return the single node if it's the only one.
            if not output_nodes and len(all_nodes) == 1:
                 output_nodes = list(all_nodes)


        return output_nodes

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution engine specific metrics"""
        with self._cache_lock:
            cache_size = len(self.execution_cache)
            hits = self.cache_hits
            misses = self.cache_misses
        total_lookups = hits + misses
        hit_rate = hits / max(1, total_lookups)

        return {
            'cache_hits': hits,
            'cache_misses': misses,
            'cache_hit_rate': hit_rate,
            'cache_size': cache_size
        }

    async def shutdown(self):
        """Gracefully shut down the execution engine."""
        logger.info("Shutting down ExecutionEngine...")
        self._shutdown_event.set() # Signal loops to stop

        # Shutdown the thread pool executor
        # Use wait=True to ensure threads finish cleanly if possible
        self.executor.shutdown(wait=True, cancel_futures=False) # Allow running tasks to finish
        logger.info("ExecutionEngine thread pool shut down.")

    def cleanup(self):
        """Synchronous cleanup (calls async shutdown)."""
        logger.info("Cleaning up ExecutionEngine...")
        if not self._shutdown_event.is_set():
             try:
                 # Try to run the async shutdown cleanly
                 asyncio.run(self.shutdown())
             except RuntimeError as e:
                 logger.warning(f"Could not run async shutdown cleanly (event loop closed?): {e}")
                 # Force shutdown executor if async failed
                 if hasattr(self, 'executor'):
                      self.executor.shutdown(wait=False, cancel_futures=True)
             except Exception as e:
                 logger.error(f"Error during cleanup's async shutdown: {e}")
                 if hasattr(self, 'executor'):
                      self.executor.shutdown(wait=False, cancel_futures=True)

        with self._cache_lock:
            self.execution_cache.clear()
        logger.info("ExecutionEngine cleanup complete.")


# ============================================================================
# MODULE-LEVEL FUNCTIONS (DEPRECATED - Use runtime methods)
# ============================================================================

_global_engine: Optional[ExecutionEngine] = None

def get_global_engine() -> ExecutionEngine:
    """DEPRECATED: Get or create global execution engine"""
    global _global_engine
    logger.warning("get_global_engine is deprecated. Obtain engine via UnifiedRuntime.")
    # Corrected the typo below: _global_g_engine -> _global_engine
    if _global_engine is None:
        # Cannot initialize properly without runtime reference
        raise RuntimeError("Global engine cannot be initialized directly. Use UnifiedRuntime.")
    return _global_engine


async def execute_graph(*args, **kwargs):
    """DEPRECATED: Module-level graph execution function"""
    logger.warning("Module-level execute_graph is deprecated. Use runtime.execute_graph().")
    # Cannot function correctly without runtime context
    raise NotImplementedError("Use runtime.execute_graph() instead.")


async def execute_node(*args, **kwargs):
    """DEPRECATED: Module-level node execution function"""
    logger.warning("Module-level execute_node is deprecated.")
    raise NotImplementedError("Node execution is handled internally by the engine.")
