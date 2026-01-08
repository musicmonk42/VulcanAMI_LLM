"""
Unified Runtime Core Module for Graphix IR
Main orchestrator that integrates all runtime components
Enhanced with VULCAN AGI integration for meta-reasoning and safety
FIXED: Proper async/await handling for coroutine shutdown methods
"""

import asyncio
import inspect
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

# Import all modules (FIXED: relative imports)
# NOTE: Assuming sibling imports from sibling modules within the 'src.unified_runtime' package
# If any module imports fail, it means the dependency is missing, which is handled below.
try:
    from .execution_metrics import ExecutionMetrics, MetricsAggregator
except ImportError:
    ExecutionMetrics = None
    MetricsAggregator = None

try:
    from .node_handlers import AI_ERRORS, get_node_handlers
except ImportError:
    # Fallback/Mock for testing when dependencies aren't fully installed
    def get_node_handlers():
        return {}

    AI_ERRORS = Exception

try:
    from .hardware_dispatcher_integration import HardwareDispatcherIntegration
except ImportError:
    HardwareDispatcherIntegration = None

try:
    from .graph_validator import GraphValidator, ResourceLimits, ValidationResult
except ImportError:
    GraphValidator = None
    ResourceLimits = None
    ValidationResult = None

try:
    from .ai_runtime_integration import AIContract, AIRuntime, AITask
except ImportError:
    AIRuntime = None
    AITask = None
    AIContract = None

try:
    from .execution_engine import (
        ExecutionContext,
        ExecutionEngine,
        ExecutionMode,
        ExecutionStatus,
    )
except ImportError:
    ExecutionEngine = None
    ExecutionContext = None
    ExecutionMode = None
    ExecutionStatus = None

try:
    from .runtime_extensions import RuntimeExtensions
except ImportError:
    RuntimeExtensions = None


# Optional component imports
try:
    # These imports are generally expected to be top-level or from a different package structure
    from stdio_policy import StdioPolicy

    STDIO_POLICY_AVAILABLE = True
except ImportError:
    StdioPolicy = None
    STDIO_POLICY_AVAILABLE = False

try:
    from distributed_sharder import DistributedSharder

    SHARDER_AVAILABLE = True
except ImportError:
    DistributedSharder = None
    SHARDER_AVAILABLE = False

try:
    from observability_manager import ObservabilityManager

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    ObservabilityManager = None
    OBSERVABILITY_AVAILABLE = False

try:
    from multimodal_processor import MultimodalProcessor

    MULTIMODAL_AVAILABLE = True
except ImportError:
    MultimodalProcessor = None
    MULTIMODAL_AVAILABLE = False

try:
    from cross_modal_reasoner import CrossModalReasoner

    REASONER_AVAILABLE = True
except ImportError:
    CrossModalReasoner = None
    REASONER_AVAILABLE = False

try:
    from continual_learner import ContinualLearner

    LEARNER_AVAILABLE = True
except ImportError:
    ContinualLearner = None
    LEARNER_AVAILABLE = False

try:
    from vulcan.safety.safety_validator import SafetyValidator

    SAFETY_AVAILABLE = True
except ImportError:
    SafetyValidator = None
    SAFETY_AVAILABLE = False

try:
    from autobiographical_memory import AutobiographicalMemory

    MEMORY_AVAILABLE = True
except ImportError:
    AutobiographicalMemory = None
    MEMORY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("UnifiedRuntime")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class RuntimeConfig:
    """Configuration for the unified runtime"""

    manifest_path: Optional[str] = None
    ontology_path: Optional[str] = None  # Added ontology_path
    learned_subgraphs_dir: Optional[str] = None
    enable_hardware_dispatch: bool = True
    enable_streaming: bool = True
    enable_batch: bool = True
    enable_metrics: bool = True
    enable_governed_io: bool = True
    enable_evolution: bool = True
    enable_explainability: bool = True
    enable_distributed: bool = True
    enable_vulcan_integration: bool = True
    enable_validation: bool = True
    max_parallel_tasks: int = 10
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    validation_timeout_seconds: float = 30.0
    execution_timeout_seconds: float = 300.0
    batch_size: int = 10  # <<< ADDED BATCH SIZE
    max_memory_mb: int = 8000  # Added for _initialize_components
    max_node_count: int = 10000  # Added for _initialize_components
    max_edge_count: int = 50000  # Added for _initialize_components
    max_recursion_depth: int = 20  # Added for _initialize_components
    # cache_size: int = 1000 # This is a duplicate, max_cache_size is used
    max_execution_time_s: int = 300  # Added for _initialize_components
    enable_vulcan_agi: bool = False  # Added for _initialize_components
    # Added enable_autonomous based on RuntimeExtensions constructor
    enable_autonomous: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MAIN RUNTIME
# ============================================================================


class UnifiedRuntime:
    """Main orchestrator integrating all runtime components"""

    # Note Issue #1: Track singleton instance to prevent __del__ cleanup
    # when multiple instances are created. Only the singleton should cleanup.
    _singleton_instance: Optional["UnifiedRuntime"] = None

    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.node_handlers = get_node_handlers()
        self._lock = threading.Lock()
        self._cache = {}
        # Use self._metrics_aggregator consistently
        self._metrics_aggregator: Optional[MetricsAggregator] = None
        self.stats = defaultdict(int)
        self.node_executors = self.node_handlers  # Use node_handlers for executors
        self.subgraph_definitions = {}
        self.audit_log = deque(maxlen=1000)
        self.io_verbosity = 0.5
        self.io_count = 0
        self._execution_lock = threading.Lock()
        self._cleanup_in_progress = False
        # Note Issue #1: Track if this instance is the singleton
        self._is_singleton = False

        # Determine paths relative to this file's location if possible
        base_path = Path(__file__).resolve().parent.parent.parent

        # Manifest Path Handling
        if self.config.manifest_path:
            manifest_path_obj = Path(self.config.manifest_path)
            if manifest_path_obj.is_absolute() or manifest_path_obj.exists():
                self.manifest_path = manifest_path_obj
            else:
                self.manifest_path = base_path / self.config.manifest_path
        else:
            self.manifest_path = base_path / "configs" / "graphix_core_manifest.json"

        # Learned Subgraphs Path Handling
        if self.config.learned_subgraphs_dir:
            learned_path_obj = Path(self.config.learned_subgraphs_dir)
            if learned_path_obj.is_absolute() or learned_path_obj.exists():
                self.learned_subgraphs_path = learned_path_obj
            else:
                self.learned_subgraphs_path = (
                    base_path / self.config.learned_subgraphs_dir
                )
        else:
            self.learned_subgraphs_path = base_path / "learned_subgraphs"

        # Load manifest
        self.manifest = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                    logger.info(f"Manifest loaded from {self.manifest_path}")
            except Exception as e:
                logger.error(f"Failed to load manifest from {self.manifest_path}: {e}")
        else:
            logger.warning(f"Manifest file not found at {self.manifest_path}")

        # Initialize components using the _initialize_components method logic
        self._initialize_components()

        # Load schema (using manifest content) and learned subgraphs
        self.schema = self.manifest
        self._load_learned_subgraphs()

        # Initialize VULCAN integration
        if self.config.enable_vulcan_integration:
            try:
                from .vulcan_integration import enable_vulcan_integration

                self.vulcan_bridge = enable_vulcan_integration(self)
                logger.info("✅ VULCAN integration enabled")
            except Exception as e:
                logger.error(f"❌ Failed to initialize VULCAN integration: {e}")
                self.vulcan_bridge = None
        else:
            self.vulcan_bridge = None

        logger.info(
            f"Runtime initialized. Grammar {self.schema.get('version', 'unknown')} loaded"
        )
        logger.info(
            f"Components: Hardware={bool(self.hardware_dispatcher)}, "
            f"Streaming={self.config.enable_streaming}, "
            f"Batch={self.config.enable_batch}, "
            f"Metrics={bool(self._metrics_aggregator)}, "
            f"VULCAN={bool(self.vulcan_bridge)}"
        )

    def _initialize_components(self):
        """Initialize all runtime components"""

        # Metrics - Assign to self._metrics_aggregator
        if self.config.enable_metrics and MetricsAggregator:
            self._metrics_aggregator = MetricsAggregator()
        else:
            self._metrics_aggregator = None

        # AI Runtime - Note Issue #28: Use singleton to prevent duplicate provider registration
        if AIRuntime:
            try:
                from vulcan.reasoning.singletons import get_ai_runtime
                self.ai_runtime = get_ai_runtime()
                if self.ai_runtime is None:
                    # Fallback to direct instantiation if singleton fails
                    self.ai_runtime = AIRuntime()
            except ImportError:
                self.ai_runtime = AIRuntime()
        else:
            self.ai_runtime = None

        # Execution engine
        if ExecutionEngine:
            # <<< --- START Timeout Fix --- >>>
            self.execution_engine = ExecutionEngine(
                runtime=self,  # Pass runtime self
                max_parallel=self.config.max_parallel_tasks,
                # --- Pass timeout from config ---
                timeout_seconds=self.config.execution_timeout_seconds,
                # --- End Pass timeout ---
                enable_streaming=self.config.enable_streaming,
                enable_batch=self.config.enable_batch,
            )
            # <<< --- END Timeout Fix --- >>>
        else:
            self.execution_engine = None

        # Validation
        if self.config.enable_validation and GraphValidator:
            self.validator = GraphValidator(
                ontology_path=self.config.ontology_path,
                manifest_node_types=self.manifest.get("nodes", {}),
                max_memory_mb=self.config.max_memory_mb,
                max_node_count=self.config.max_node_count,
                max_edge_count=self.config.max_edge_count,
                max_recursion_depth=self.config.max_recursion_depth,
                enable_cycle_detection=True,
                enable_resource_checking=True,
                enable_security_validation=True,
            )
        elif not self.config.enable_validation:
            logger.info("Graph validation disabled by config.")
            self.validator = None
        else:
            logger.warning("GraphValidator not available, disabling graph validation.")
            self.validator = None

        # Hardware dispatch
        if self.config.enable_hardware_dispatch and HardwareDispatcherIntegration:
            self.hardware_dispatcher = HardwareDispatcherIntegration()
        else:
            self.hardware_dispatcher = None

        # Extensions
        # <<< --- START CORRECTION --- >>>
        if RuntimeExtensions:
            # Check if enable_autonomous exists in config, default to True if not
            enable_autonomous_flag = getattr(
                self.config, "enable_autonomous", True
            )  # Get flag safely
            self.extensions = RuntimeExtensions(
                # Remove runtime=self
                learned_subgraphs_dir=str(self.learned_subgraphs_path),
                # Pass enable_autonomous instead of evolution/explainability
                enable_autonomous=enable_autonomous_flag,
            )
        else:
            self.extensions = None
        # <<< --- END CORRECTION --- >>>

        # Governed I/O
        if self.config.enable_governed_io and STDIO_POLICY_AVAILABLE and StdioPolicy:
            self.stdio_policy = StdioPolicy()
            self.stdio_policy.set_audit_callback(self._audit_io_operation)
            self.stdio_policy.set_max_operations(10000)
        else:
            self.stdio_policy = None

        # Distributed components
        if self.config.enable_distributed:
            self.sharder = (
                DistributedSharder()
                if SHARDER_AVAILABLE and DistributedSharder
                else None
            )
            self.obs_manager = (
                ObservabilityManager()
                if OBSERVABILITY_AVAILABLE and ObservabilityManager
                else None
            )
        else:
            self.sharder = None
            self.obs_manager = None

        # VULCAN-AGI components
        if hasattr(self.config, "enable_vulcan_agi") and self.config.enable_vulcan_agi:
            self.multimodal_processor = (
                MultimodalProcessor()
                if MULTIMODAL_AVAILABLE and MultimodalProcessor
                else None
            )
            self.cross_modal_reasoner = (
                CrossModalReasoner()
                if REASONER_AVAILABLE and CrossModalReasoner
                else None
            )
            self.continual_learner = (
                ContinualLearner() if LEARNER_AVAILABLE and ContinualLearner else None
            )
            self.safety_validator = (
                SafetyValidator() if SAFETY_AVAILABLE and SafetyValidator else None
            )
            self.autobiographical_memory = (
                AutobiographicalMemory()
                if MEMORY_AVAILABLE and AutobiographicalMemory
                else None
            )
        else:
            self.multimodal_processor = None
            self.cross_modal_reasoner = None
            self.continual_learner = None
            self.safety_validator = None
            self.autobiographical_memory = None

    def _load_schema_from_manifest(self) -> Dict[str, Any]:
        """Load schema from manifest file"""
        return self.manifest

    def _load_learned_subgraphs(self):
        """Load learned subgraph definitions"""
        self.subgraph_definitions = {}
        try:
            if self.learned_subgraphs_path.exists():
                for f in self.learned_subgraphs_path.glob("*.json"):
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        # Use 'name' or 'pattern_id' as key
                        key = data.get("name", data.get("pattern_id"))
                        if key:
                            self.subgraph_definitions[key] = data.get(
                                "graph_definition", data
                            )

            # Also load from extensions
            if self.extensions:
                subgraphs = self.extensions.load_learned_subgraphs()
                self.subgraph_definitions.update(subgraphs)
                logger.info(
                    f"Loaded {len(self.subgraph_definitions)} total learned subgraphs"
                )
        except Exception as e:
            logger.error(f"Failed to load learned subgraphs: {e}")

    def load_learned_subgraphs(self) -> List[Dict[str, Any]]:
        """Public method to load learned subgraphs, returns list of graph definitions."""
        if not self.subgraph_definitions:
            self._load_learned_subgraphs()
        return list(self.subgraph_definitions.values())

    # ========================================================================
    # VULCAN INTEGRATION
    # ========================================================================

    def enable_vulcan_integration(self, config=None):
        """
        Enable VULCAN integration for this runtime
        """
        try:
            from .vulcan_integration import enable_vulcan_integration as enable_vulcan

            self.vulcan_bridge = enable_vulcan(self, config)
            logger.info("VULCAN integration enabled")
            return self.vulcan_bridge

        except ImportError as e:
            logger.warning(f"Cannot enable VULCAN integration: {e}")
            self.vulcan_bridge = None
            return None

    # ========================================================================
    # I/O GOVERNANCE
    # ========================================================================

    def safe_print(self, *args, **kwargs):
        """Thread-safe governed print"""
        if self.stdio_policy:
            return self.stdio_policy.print(*args, **kwargs)
        elif self.io_verbosity > 0.5 and self.io_count < 100000:
            with self._execution_lock:
                print(*args, **kwargs)
                self.io_count += 1

    def json_print(self, obj):
        """Thread-safe governed JSON print"""
        if self.stdio_policy:
            return self.stdio_policy.json_print(obj)
        elif self.io_verbosity > 0.5 and self.io_count < 100000:
            with self._execution_lock:
                print(json.dumps(obj, indent=2, default=str))
                self.io_count += 1

    def set_verbosity(self, level: float):
        """Set I/O verbosity level"""
        level = (
            float(level) if not isinstance(level, bool) and level is not None else 0.0
        )
        self.io_verbosity = max(0.0, min(1.0, level))
        if self.stdio_policy:
            self.stdio_policy.set_verbosity(self.io_verbosity)

    def get_io_count(self) -> int:
        """Get total I/O operation count"""
        if self.stdio_policy:
            return self.stdio_policy.get_operation_count()
        return self.io_count

    def _audit_io_operation(self, operation_type: str, content: Any):
        """Audit callback for I/O operations"""
        # Note: self._metrics is not a class attribute, it's created per-run
        # We can't audit to a single-run metric here.
        # This should log to the runtime's main audit_log
        self.audit_log.append(
            {
                "type": "io_operation",
                "operation": operation_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # ========================================================================
    # GRAPH EXECUTION
    # ========================================================================

    async def execute_graph(
        self, graph: Dict[str, Any], mode: Optional["ExecutionMode"] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a graph with all safety checks and optimizations
        """
        mode = mode or (ExecutionMode.PARALLEL if ExecutionMode else "parallel")
        recursion_depth = kwargs.get("recursion_depth", 0)

        # 1. validation (GraphValidator)
        if self.validator:
            timeout_s = getattr(self.config, "validation_timeout_seconds", 30.0)
            if hasattr(self.validator, "validate_with_timeout"):
                validation = self.validator.validate_with_timeout(
                    graph,
                    manifest_node_types=self.manifest.get("nodes", {}),
                    timeout_s=timeout_s,
                )
            else:
                validation = self.validator.validate_graph(
                    graph, manifest_node_types=self.manifest.get("nodes", {})
                )

            if not validation.is_valid:
                # Build failed result
                return {
                    "status": "FAILED_VALIDATION",
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "metadata": {"validation": validation.metadata},
                }

        # 2. VULCAN pre-check
        if getattr(self, "vulcan_bridge", None):
            # call bridge pre-exec validation if exposed (e.g. self.vulcan_bridge.validate_graph_request)
            try:
                if hasattr(self.vulcan_bridge, "validate_graph_request"):
                    vulcan_resp = await self.vulcan_bridge.validate_graph_request(graph)
                    if not vulcan_resp.success:
                        return {
                            "status": "BLOCKED_BY_VULCAN",
                            "errors": [vulcan_resp.reason],
                            "metadata": {"vulcan": vulcan_resp.to_dict()},
                        }
            except Exception as e:
                logger.error(f"VULCAN pre-check failed: {e}")

        # 3. Build execution context
        node_map = {n["id"]: n for n in graph.get("nodes", [])}
        context = ExecutionContext(
            graph=graph,
            node_map=node_map,
            runtime=self,
            inputs=kwargs.get("inputs", {}),
            recursion_depth=recursion_depth,
        )

        # 4. Run engine
        exec_metrics = (
            ExecutionMetrics()
            if ExecutionMetrics and self.config.enable_metrics
            else None
        )

        if not self.execution_engine:
            return {
                "status": "failed",
                "error": "ExecutionEngine is not initialized. Check dependencies.",
            }

        graph_result = await self.execution_engine.run_graph(
            context=context, mode=mode, metrics=exec_metrics
        )

        # 5. Aggregate metrics
        if exec_metrics:
            # finalize_graph() is now called within run_graph in the engine
            # exec_metrics.finalize_graph() # Removed call here
            if hasattr(exec_metrics, "metadata") and isinstance(
                exec_metrics.metadata, dict
            ):  # Check if metadata exists
                exec_metrics.metadata["hardware"] = (
                    self.hardware_dispatcher.get_health_snapshot()
                    if self.hardware_dispatcher
                    and hasattr(self.hardware_dispatcher, "get_health_snapshot")
                    else {}
                )
            if self._metrics_aggregator:
                self._metrics_aggregator.record_metrics(
                    exec_metrics
                )  # Corrected method name

        # 6. Runtime extensions hook
        ext_meta = {}
        if self.extensions:
            try:
                # Extensions hook might not exist, check first
                if hasattr(self.extensions, "on_run_complete") and callable(
                    self.extensions.on_run_complete
                ):
                    ext_meta = self.extensions.on_run_complete(
                        graph=graph,
                        exec_result=graph_result,
                        metrics=exec_metrics,
                        audit_log=context.audit_log,
                    )
            except Exception as e:
                logger.error(f"RuntimeExtensions.on_run_complete error: {e}")

        # 7. Merge audit back to runtime
        for entry in context.audit_log:
            self.audit_log.append(entry)

        # 8. VULCAN post-run hook
        if getattr(self, "vulcan_bridge", None):
            try:
                if hasattr(self.vulcan_bridge, "on_run_complete"):
                    await self.vulcan_bridge.on_run_complete(
                        graph=graph,
                        result=graph_result,
                        metrics=exec_metrics,
                        extension_meta=ext_meta,
                    )
            except Exception as e:
                logger.error(f"VULCAN post-run hook failed: {e}")

        # Ensure graph_result is a dict before returning
        if isinstance(graph_result, dict):
            return graph_result
        elif hasattr(graph_result, "to_dict"):
            return graph_result.to_dict()
        else:
            # Fallback if result is unexpected type
            logger.error(
                f"Execution engine returned unexpected type: {type(graph_result)}"
            )
            return {
                "status": "failed",
                "errors": {
                    "_graph": "Internal error: Unexpected execution result type"
                },
            }

    async def _trigger_autonomous_optimization(
        self, graph: Dict[str, Any], metrics: Dict[str, Any], result: Dict[str, Any]
    ):
        """Trigger autonomous optimization cycle"""
        if not self.extensions or not self.config.enable_evolution:
            return

        try:
            # Check if trigger_autonomous_cycle exists
            if hasattr(self.extensions, "trigger_autonomous_cycle"):
                report = await self.extensions.trigger_autonomous_cycle(
                    graph, metrics, self
                )

                if report.get("optimizations_applied"):
                    logger.info(
                        f"Applied {len(report['optimizations_applied'])} optimizations"
                    )

        except Exception as e:
            logger.error(f"Autonomous optimization failed: {e}", exc_info=True)

    # ========================================================================
    # STREAMING EXECUTION
    # ========================================================================

    async def execute_stream(
        self, graph: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute a single graph in streaming mode, yielding intermediate results.
        """
        if not self.config.enable_streaming:
            yield {"status": "failed", "errors": ["Streaming mode is not enabled"]}
            return

        # Validate
        if self.validator:
            manifest_node_types = self.manifest.get("nodes", {})
            # Use validate_with_timeout if available
            timeout_s = getattr(self.config, "validation_timeout_seconds", 30.0)
            if hasattr(self.validator, "validate_with_timeout"):
                validation = self.validator.validate_with_timeout(
                    graph, manifest_node_types=manifest_node_types, timeout_s=timeout_s
                )
            else:
                validation = self.validator.validate_graph(
                    graph, manifest_node_types=manifest_node_types
                )
        else:
            validation = (
                ValidationResult(is_valid=True)
                if ValidationResult
                else {"valid": True, "errors": [], "warnings": []}
            )

        is_valid = (
            getattr(validation, "is_valid", validation.get("valid", True))
            if validation
            else True
        )
        errors = (
            getattr(
                validation,
                "errors",
                validation.get("errors", ["Unknown validation error"]),
            )
            if validation
            else []
        )

        if not is_valid:
            yield {"status": "failed", "errors": errors}
            return

        if not self.execution_engine:
            yield {"status": "failed", "errors": ["ExecutionEngine not initialized"]}
            return

        context = ExecutionContext(
            graph=graph,
            node_map={n["id"]: n for n in graph.get("nodes", [])},
            runtime=self,
        )

        exec_metrics = (
            ExecutionMetrics()
            if ExecutionMetrics and self.config.enable_metrics
            else None
        )

        try:
            exec_mode = ExecutionMode.STREAMING if ExecutionMode else "streaming"
            # Use run_graph as requested
            async for result in self.execution_engine.run_graph(
                context, exec_mode, exec_metrics
            ):
                yield result.to_dict()
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}", exc_info=True)
            yield {"status": "failed", "errors": [f"Streaming execution failed: {e}"]}
        finally:
            # Finalize and record metrics for the stream
            if exec_metrics:
                # finalize_graph() is called within run_graph now
                # exec_metrics.finalize_graph() # Removed call here
                if self._metrics_aggregator:
                    self._metrics_aggregator.record_metrics(exec_metrics)

            # Cleanup for the engine related to this stream is implicitly handled
            # by the completion of the run_graph generator. Explicit shutdown call removed.
            # if self.execution_engine:
            #     try:
            #         if hasattr(self.execution_engine, 'shutdown') and inspect.iscoroutinefunction(self.execution_engine.shutdown):
            #             await self.execution_engine.shutdown()
            #     except Exception as e:
            #          logger.error(f"Error during streaming execution engine cleanup: {e}", exc_info=True)
            pass  # Keep finally block for structure if needed later

    # ========================================================================
    # BATCH EXECUTION
    # ========================================================================

    async def execute_batch(self, graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute graphs in batch mode with optimized resource usage
        """
        if not self.config.enable_batch:
            raise RuntimeError("Batch mode is not enabled")

        results = []
        batch_size = self.config.batch_size  # Use config value

        for i in range(0, len(graphs), batch_size):
            batch = graphs[i : i + batch_size]

            total_nodes = sum(len(g.get("nodes", [])) for g in batch)
            max_nodes = self.config.max_node_count

            if total_nodes > max_nodes:
                mid = len(batch) // 2
                if mid == 0:
                    logger.warning(
                        f"Single graph exceeds node limit ({total_nodes} > {max_nodes}). Skipping."
                    )
                    results.append(
                        {
                            "status": "failed",
                            "error": "Graph exceeds node limit",
                            "batch_index": i // batch_size,
                            "item_index": 0,
                        }
                    )
                    continue

                batch1_results = await self.execute_batch(batch[:mid])
                batch2_results = await self.execute_batch(batch[mid:])
                results.extend(batch1_results)
                results.extend(batch2_results)
                continue

            # Use the standard execute_graph method for each item
            batch_tasks = [self.execute_graph(graph) for graph in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(
                        {
                            "status": "failed",
                            "error": str(result),
                            "batch_index": i // batch_size,
                            "item_index": j,
                        }
                    )
                else:
                    # Ensure result is a dictionary before adding keys
                    if isinstance(result, dict):
                        result["batch_index"] = i // batch_size
                        result["item_index"] = j
                        results.append(result)
                    else:  # Fallback if execute_graph didn't return a dict
                        logger.error(
                            f"Batch execution got unexpected result type {type(result)}"
                        )
                        results.append(
                            {
                                "status": "failed",
                                "error": f"Internal error: Unexpected result type {type(result)}",
                                "batch_index": i // batch_size,
                                "item_index": j,
                            }
                        )

        return results

    # ========================================================================
    # NODE REGISTRATION
    # ========================================================================

    def register_node_type(self, node_type: str, executor: Callable):
        """Register a new node type"""
        if not callable(executor):
            logger.error(f"Executor for {node_type} must be a callable function.")
            return False

        self.node_executors[node_type] = executor
        logger.info(f"Registered node type: {node_type}")
        return True

    def learn_subgraph(self, graph_definition: Dict[str, Any]) -> bool:
        """Learn a new subgraph pattern"""
        if not self.extensions:
            logger.error("Runtime extensions not available")
            return False

        # Ensure subgraph_learner exists
        if not hasattr(self.extensions, "subgraph_learner"):
            logger.error("Extensions object missing subgraph_learner component.")
            return False

        success, msg = self.extensions.subgraph_learner.learn_subgraph(
            subgraph_type=graph_definition.get("id", "custom"),
            graph_definition=graph_definition,
        )
        if success:
            nodes_info = graph_definition.get("nodes", [])
            if isinstance(nodes_info, list):
                node_ids = [n.get("id", "unknown") for n in nodes_info]
                logger.info(f"Learned subgraph involving nodes: {node_ids}")
            else:
                logger.info(
                    f"Learned subgraph (details unavailable): {graph_definition.get('id', 'unknown')}"
                )
        return success

    # ========================================================================
    # HARDWARE DISPATCH
    # ========================================================================

    async def _dispatch_to_hardware(self, operation: str, *args, **kwargs) -> Any:
        """Dispatch operation to hardware"""
        if not self.hardware_dispatcher:
            raise RuntimeError("Hardware dispatch not enabled")

        # This function is called by node handlers, which expect run_tensor_op
        if hasattr(self.hardware_dispatcher, "run_tensor_op"):
            # Create a closure for the operation
            def op_closure():
                # This is a generic placeholder; real nodes should pass closures
                logger.warning(
                    "Generic _dispatch_to_hardware closure used. Node should provide its own."
                )
                if operation == "photonic_mvm":
                    return np.dot(args[0], args[1])
                return f"executed_{operation}"

            # This is a guess, node handlers should ideally pass their own closure
            dispatch_result = await self.hardware_dispatcher.run_tensor_op(
                op=op_closure
            )
            return dispatch_result.result
        else:
            # Fallback to old method if new one not present
            result = await self.hardware_dispatcher.dispatch_to_hardware(
                operation, *args, **kwargs
            )
            return getattr(result, "result", result)

    # ========================================================================
    # INTROSPECTION & METRICS
    # ========================================================================

    def introspect(self) -> Dict[str, Any]:
        """Get runtime introspection data"""
        info = {
            "config": self.config.to_dict() if self.config else {},
            "grammar_version": self.schema.get("version", "unknown"),  # Use self.schema
            "node_types": {
                "core": sorted(
                    [
                        k
                        for k in self.node_executors.keys()  # Use self.node_executors
                        if not k.startswith("_")
                    ]
                ),
                "learned": sorted(list(self.subgraph_definitions.keys())),
            },
            "components": {
                "hardware_dispatch": self.hardware_dispatcher is not None,
                "ai_runtime": self.ai_runtime is not None,
                "metrics": self._metrics_aggregator
                is not None,  # Use self._metrics_aggregator
                "sharder": self.sharder is not None,
                "vulcan_agi": {
                    "multimodal": self.multimodal_processor is not None,
                    "reasoner": self.cross_modal_reasoner is not None,
                    "learner": self.continual_learner is not None,
                    "safety": self.safety_validator is not None,
                    "memory": self.autobiographical_memory is not None,
                },
                "vulcan_integration": self.vulcan_bridge is not None,
            },
        }

        if self._metrics_aggregator and hasattr(
            self._metrics_aggregator, "get_summary"
        ):
            info["metrics"] = self._metrics_aggregator.get_summary()

        if self.extensions and hasattr(self.extensions, "get_statistics"):
            info["extensions"] = self.extensions.get_statistics()

        if self.vulcan_bridge:
            try:
                get_stats = getattr(
                    self.vulcan_bridge, "get_integration_statistics", None
                )
                if get_stats and callable(get_stats):
                    info["vulcan_integration_stats"] = get_stats()
            except Exception as e:
                logger.debug(f"Could not get VULCAN stats: {e}")

        # Add audit log (limited size)
        info["audit_log"] = list(self.audit_log)[-50:]  # Last 50 entries

        return info

    def get_hardware_metrics(self) -> Dict[str, Any]:
        """Get hardware dispatcher metrics"""
        if self.hardware_dispatcher and hasattr(
            self.hardware_dispatcher, "get_metrics_summary"
        ):
            return self.hardware_dispatcher.get_metrics_summary()
        return {"enabled": False}

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        metrics = {}

        # No 'current' metrics, only aggregated

        if self._metrics_aggregator and hasattr(
            self._metrics_aggregator, "get_summary"
        ):
            metrics["aggregated"] = self._metrics_aggregator.get_summary()

        if self.execution_engine and hasattr(self.execution_engine, "get_metrics"):
            metrics["engine"] = self.execution_engine.get_metrics()

        return metrics

    # <<< ADDED METHOD >>>
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution metrics."""
        if self._metrics_aggregator and hasattr(
            self._metrics_aggregator, "get_summary"
        ):  # Check aggregator
            return self._metrics_aggregator.get_summary()
        # Fallback if metrics are disabled or the object/method is missing
        return {
            "execution_count_total": 0,
            "cache_hits_total": 0,
            "cache_misses_total": 0,
            "nodes_executed_total": 0,
            "avg_latency_ms_per_execution": 0.0,
        }

    def validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a graph structure"""
        if self.validator:
            manifest_node_types = self.manifest.get(
                "node_types", {}
            )  # Should probably be manifest.get('nodes', {})
            if not manifest_node_types:
                manifest_node_types = self.manifest.get("nodes", {})  # Try 'nodes' key

            # Use validate_with_timeout if available
            timeout_s = getattr(self.config, "validation_timeout_seconds", 30.0)
            if hasattr(self.validator, "validate_with_timeout"):
                result = self.validator.validate_with_timeout(
                    graph, manifest_node_types=manifest_node_types, timeout_s=timeout_s
                )
            else:
                result = self.validator.validate_graph(
                    graph, manifest_node_types=manifest_node_types
                )

            if hasattr(result, "to_dict"):
                return result.to_dict()
            else:
                logger.error(f"Validator returned unexpected type: {type(result)}")
                return {
                    "valid": False,
                    "errors": ["Validation returned unexpected result type"],
                    "warnings": [],
                }

        else:
            return {
                "valid": False,
                "errors": ["Graph validation disabled (Validator not initialized)"],
                "warnings": [],
            }

    # ========================================================================
    # CLEANUP - REPLACED with prompt's version
    # ========================================================================

    async def shutdown(self):
        """Async cleanup for all resources"""
        if self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        logger.info("Shutting down UnifiedRuntime (async)...")
        try:
            # stop engine if it has shutdown
            if hasattr(
                self.execution_engine, "shutdown"
            ) and inspect.iscoroutinefunction(self.execution_engine.shutdown):
                await self.execution_engine.shutdown()
            elif hasattr(self.execution_engine, "shutdown"):
                self.execution_engine.shutdown()

            # ai runtime cleanup
            if (
                hasattr(self, "ai_runtime")
                and self.ai_runtime
                and hasattr(self.ai_runtime, "shutdown")
            ):
                if inspect.iscoroutinefunction(self.ai_runtime.shutdown):
                    await self.ai_runtime.shutdown()
                else:
                    self.ai_runtime.shutdown()

            # hardware dispatcher threadpool cleanup
            if hasattr(self, "hardware_dispatcher") and self.hardware_dispatcher:
                # Ensure executor exists before calling shutdown
                if (
                    hasattr(self.hardware_dispatcher, "executor")
                    and self.hardware_dispatcher.executor
                ):  # Corrected attribute name
                    self.hardware_dispatcher.executor.shutdown(
                        wait=False
                    )  # Corrected attribute name
                # Call cleanup on the integration layer too if it exists
                if hasattr(self.hardware_dispatcher, "cleanup"):
                    self.hardware_dispatcher.cleanup()

            # persist extensions / learned subgraphs, if needed
            if self.extensions and hasattr(self.extensions, "persist"):
                try:
                    self.extensions.persist()
                except Exception as e:
                    logger.error(f"Failed to persist extensions: {e}")

            # vulcan bridge cleanup
            if getattr(self, "vulcan_bridge", None):
                if hasattr(self.vulcan_bridge, "shutdown"):
                    if inspect.iscoroutinefunction(self.vulcan_bridge.shutdown):
                        await self.vulcan_bridge.shutdown()
                    else:
                        self.vulcan_bridge.shutdown()

        finally:
            self._cleanup_in_progress = False
            logger.info("UnifiedRuntime shutdown complete (async)")

    def cleanup(self):
        """
        Synchronous cleanup wrapper for all resources
        """
        if self._cleanup_in_progress:
            logger.debug("Cleanup already in progress")
            return

        logger.info("Shutting down UnifiedRuntime...")

        try:
            loop = asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop_policy().get_event_loop()
                is_running = loop.is_running()
            except RuntimeError:
                loop = None
                is_running = False

        if loop and is_running:
            logger.debug("Event loop running, creating cleanup task...")
            # Can't block here, so this is best-effort
            asyncio.create_task(self.shutdown())
        elif loop and not is_running:
            logger.debug("Running async shutdown in existing event loop...")
            loop.run_until_complete(self.shutdown())
        else:
            logger.debug(
                "No event loop found, running async shutdown via asyncio.run()..."
            )
            try:
                asyncio.run(self.shutdown())
            except Exception as e:
                logger.error(f"Error during asyncio.run(shutdown): {e}")

        logger.info("UnifiedRuntime shutdown complete")

    def __del__(self):
        """
        Destructor to ensure cleanup.
        
        Note Issue #1: Only cleanup if this is the singleton instance.
        Non-singleton instances created as fallbacks should NOT trigger
        cleanup, as that destroys shared state and caches for all queries.
        This was causing the "UnifiedRuntime shutdown complete" log after
        every query followed by full reinitialization on the next query.
        """
        try:
            # Note Issue #1: Skip cleanup for non-singleton instances
            # This prevents the per-query reinitialization problem
            if not getattr(self, '_is_singleton', False):
                if logger:
                    logger.debug(
                        "Skipping cleanup for non-singleton UnifiedRuntime instance"
                    )
                return
            
            if not self._cleanup_in_progress:
                self.cleanup()
        except Exception as e:
            # Avoid logging errors during interpreter shutdown if logger is gone
            if logger:
                logger.debug(f"Error in destructor: {e}")


# ============================================================================
# MODULE-LEVEL INTERFACE
# ============================================================================

_global_runtime: Optional[UnifiedRuntime] = None


def get_runtime(config: Optional[RuntimeConfig] = None) -> UnifiedRuntime:
    """Get or create global runtime instance.
    
    Note Issue #1: Marks the instance as singleton so __del__ knows
    to perform cleanup only for this instance, not transient fallback instances.
    """
    global _global_runtime
    if _global_runtime is None:
        _global_runtime = UnifiedRuntime(config)
        # Note Issue #1: Mark as singleton for __del__ check
        _global_runtime._is_singleton = True
        UnifiedRuntime._singleton_instance = _global_runtime
    # If a config is passed later, should we update the global one?
    # For now, only initialize if None. Re-initializing might have side effects.
    # elif config is not None and _global_runtime.config != config:
    #     logger.warning("Requesting runtime with different config, but global instance already exists. Returning existing instance.")
    return _global_runtime


async def execute_graph(
    graph: Dict[str, Any], mode: Optional["ExecutionMode"] = None
) -> Dict[str, Any]:
    """Execute a graph using global runtime"""
    runtime = get_runtime()
    return await runtime.execute_graph(graph, mode=mode)


async def execute_batch(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute batch of graphs"""
    runtime = get_runtime()
    return await runtime.execute_batch(graphs)


def introspect() -> Dict[str, Any]:
    """Get runtime introspection"""
    runtime = get_runtime()
    return runtime.introspect()


async def async_cleanup():
    """
    Async cleanup for global runtime
    """
    global _global_runtime
    if _global_runtime:
        await _global_runtime.shutdown()  # Use new shutdown method
        _global_runtime = None


def cleanup():
    """
    Synchronous cleanup for global runtime
    """
    global _global_runtime
    if _global_runtime:
        _global_runtime.cleanup()
        _global_runtime = None


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main():
    """Example usage and testing"""

    config = RuntimeConfig(
        enable_hardware_dispatch=True,
        enable_metrics=True,
        enable_evolution=True,
        enable_vulcan_integration=True,
    )

    runtime = UnifiedRuntime(config)

    print("\n--- Test 1: Simple Addition Graph ---")
    simple_graph = {
        "nodes": [
            {"id": "c1", "type": "CONST", "params": {"value": 10}},
            {"id": "c2", "type": "CONST", "params": {"value": 20}},
            {"id": "add", "type": "ADD"},
            {"id": "out", "type": "OUTPUT"},  # Use OUTPUT
        ],
        "edges": [
            {"from": "c1", "to": {"node": "add", "port": "val1"}},
            {"from": "c2", "to": {"node": "add", "port": "val2"}},
            {"from": "add", "to": {"node": "out", "port": "input"}},  # Use 'input'
        ],
    }

    result = await runtime.execute_graph(simple_graph)
    print(f"Result: {result}")

    print("\n--- Test 2: Batch Execution ---")
    batch_graphs = [
        {
            "nodes": [
                {"id": "c", "type": "CONST", "params": {"value": i}},
                {"id": "add", "type": "ADD"},
                {"id": "out", "type": "OUTPUT"},  # Use OUTPUT
            ],
            "edges": [
                {"from": "c", "to": {"node": "add", "port": "val1"}},
                {"from": "c", "to": {"node": "add", "port": "val2"}},
                {"from": "add", "to": {"node": "out", "port": "input"}},  # Use 'input'
            ],
        }
        for i in range(5)
    ]

    batch_results = await runtime.execute_batch(batch_graphs)
    print(f"Batch results: {len(batch_results)} graphs executed")

    print("\n--- Test 3: Introspection ---")
    info = runtime.introspect()
    print(json.dumps(info, indent=2))

    await runtime.shutdown()  # Use new shutdown method
    print("\n--- Tests Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
