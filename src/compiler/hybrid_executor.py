"""
Hybrid Executor for Graphix IR
Intelligently routes between compiled and interpreted execution paths
with profiling, caching, and automatic optimization selection
"""

import atexit
import ctypes
import hashlib
import json
import logging
import os
import pickle
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .security_fixes import safe_pickle_load

# Initialize logger
logger = logging.getLogger(__name__)

# Performance tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ExecutionMode(Enum):
    """Execution modes"""

    INTERPRETED = "interpreted"
    COMPILED = "compiled"
    HYBRID = "hybrid"
    AUTO = "auto"


class OptimizationLevel(Enum):
    """Optimization levels for compilation"""

    O0 = 0  # No optimization
    O1 = 1  # Basic optimization
    O2 = 2  # Standard optimization
    O3 = 3  # Aggressive optimization
    Os = 4  # Size optimization


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution"""

    mode: ExecutionMode
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    cache_hits: int
    cache_misses: int
    compilation_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "compilation_time_ms": self.compilation_time_ms,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


@dataclass
class GraphProfile:
    """Performance profile for a graph"""

    graph_id: str
    graph_hash: str
    interpreted_metrics: List[ExecutionMetrics] = field(default_factory=list)
    compiled_metrics: List[ExecutionMetrics] = field(default_factory=list)
    best_mode: ExecutionMode = ExecutionMode.INTERPRETED
    speedup: float = 1.0
    compilation_success: bool = False
    compiled_binary: Optional[bytes] = None
    last_updated: float = field(default_factory=time.time)
    execution_count: int = 0

    def update_best_mode(self):
        """Update best execution mode based on metrics"""
        if not self.compiled_metrics or not self.interpreted_metrics:
            return

        # Calculate average times
        avg_interpreted = np.mean(
            [m.duration_ms for m in self.interpreted_metrics[-10:]]
        )
        avg_compiled = np.mean([m.duration_ms for m in self.compiled_metrics[-10:]])

        # Include compilation time amortized over executions
        if self.compiled_metrics:
            total_compilation_time = sum(
                m.compilation_time_ms for m in self.compiled_metrics
            )
            amortized_compilation = total_compilation_time / max(
                1, len(self.compiled_metrics)
            )
            avg_compiled += amortized_compilation * 0.1  # Weight compilation cost

        self.speedup = avg_interpreted / avg_compiled if avg_compiled > 0 else 1.0

        # Choose compiled if >20% faster
        if self.speedup > 1.2:
            self.best_mode = ExecutionMode.COMPILED
        else:
            self.best_mode = ExecutionMode.INTERPRETED


class CompiledBinaryCache:
    """Cache for compiled binaries - simplified without memory mapping"""

    def __init__(self, cache_dir: str = ".graphix_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index = self._load_index()

    def _load_index(self) -> Dict:
        """Load cache index"""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        """Save cache index"""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def get(self, graph_hash: str) -> Optional[bytes]:
        """Get compiled binary from cache"""
        if graph_hash not in self.cache_index:
            return None

        cache_file = self.cache_dir / f"{graph_hash}.bin"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read cache file {graph_hash}: {e}")
            return None

    def put(self, graph_hash: str, binary: bytes, metadata: Dict = None):
        """Store compiled binary in cache"""
        cache_file = self.cache_dir / f"{graph_hash}.bin"

        try:
            with open(cache_file, "wb") as f:
                f.write(binary)

            self.cache_index[graph_hash] = {
                "size": len(binary),
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
            self._save_index()
        except Exception as e:
            logger.debug(f"Operation failed: {e}")

    def cleanup(self, max_age_days: int = 7):
        """Clean old cache entries"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        for graph_hash, info in list(self.cache_index.items()):
            if current_time - info["timestamp"] > max_age_seconds:
                cache_file = self.cache_dir / f"{graph_hash}.bin"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.debug(f"Operation failed: {e}")
                del self.cache_index[graph_hash]

        self._save_index()


class HybridExecutor:
    """
    Hybrid execution engine that intelligently routes between
    compiled and interpreted execution paths
    """

    def __init__(
        self,
        runtime,
        compiler=None,
        optimization_level: OptimizationLevel = OptimizationLevel.O2,
        profile_window: int = 10,
        cache_dir: str = ".graphix_cache",
        enable_profiling: bool = True,
        enable_compilation: bool = True,
        max_compilation_attempts: int = 3,
        compilation_timeout: float = 30.0,
        executor_threads: int = 4,
    ):
        self.runtime = runtime
        self.compiler = compiler or self._init_compiler()
        self.optimization_level = optimization_level
        self.profile_window = profile_window
        self.enable_profiling = enable_profiling
        self.enable_compilation = enable_compilation
        self.max_compilation_attempts = max_compilation_attempts
        self.compilation_timeout = compilation_timeout

        # Caches
        self.binary_cache = CompiledBinaryCache(cache_dir)
        self.profiles: Dict[str, GraphProfile] = {}
        self.execution_history = deque(maxlen=1000)

        # Thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=executor_threads)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        self._executors_shutdown = False

        # Metrics
        self.total_executions = 0
        self.compilation_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Locks for thread safety
        self.profile_lock = threading.RLock()
        self.compilation_lock = threading.Lock()

        # Logger
        self.logger = logging.getLogger(__name__)

        # Temp directory tracking
        self.temp_dir = str(self.binary_cache.cache_dir)

        # Load persisted profiles
        self._load_profiles()

        # Register cleanup on exit
        atexit.register(self._atexit_cleanup)

    def _init_compiler(self):
        """Initialize compiler if not provided"""
        try:
            from src.compiler.graph_compiler import GraphCompiler

            return GraphCompiler()
        except ImportError:
            self.logger.warning("GraphCompiler not available, compilation disabled")
            self.enable_compilation = False
            return None

    def _load_profiles(self):
        """Load persisted profiles from disk"""
        profile_file = Path(self.binary_cache.cache_dir) / "profiles.pkl"
        if profile_file.exists():
            try:
                with open(profile_file, "rb") as f:
                    self.profiles = safe_pickle_load(f)
                self.logger.info(f"Loaded {len(self.profiles)} profiles from disk")
            except Exception as e:
                self.logger.error(f"Failed to load profiles: {e}")

    def _save_profiles(self):
        """Persist profiles to disk"""
        # Check if directory still exists (might be cleaned up in tests)
        if not os.path.exists(self.temp_dir):
            return

        profile_file = os.path.join(self.temp_dir, "profiles.pkl")

        try:
            with open(profile_file, "wb") as f:
                pickle.dump(self.profiles, f)
        except Exception as e:
            # Silently fail if logging is shut down
            try:
                self.logger.error(f"Failed to save profiles: {e}")
            except (ValueError, AttributeError):
                # Logging already shut down, ignore
                pass

    def _compute_graph_hash(self, graph: Dict[str, Any]) -> str:
        """Compute deterministic hash for graph"""
        # Remove non-deterministic fields
        graph_copy = graph.copy()
        for field in ["timestamp", "execution_id", "metadata"]:
            graph_copy.pop(field, None)

        # Sort for determinism
        graph_str = json.dumps(graph_copy, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _measure_resources(self) -> Tuple[float, float]:
        """Measure current resource usage"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent(interval=0.01)
            except Exception:
                memory_mb = 0.0
                cpu_percent = 0.0
        else:
            memory_mb = 0.0
            cpu_percent = 0.0

        return memory_mb, cpu_percent

    async def _execute_interpreted(
        self, graph: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], ExecutionMetrics]:
        """Execute graph in interpreted mode"""
        start_time = time.perf_counter()
        start_mem, start_cpu = self._measure_resources()

        try:
            # Use runtime's standard graph execution method
            result = await self.runtime.execute_graph(graph)

            # Measure metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            end_mem, end_cpu = self._measure_resources()

            metrics = ExecutionMetrics(
                mode=ExecutionMode.INTERPRETED,
                duration_ms=duration_ms,
                memory_mb=max(0, end_mem - start_mem),
                cpu_percent=end_cpu,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
            )

            return result, metrics

        except Exception as e:
            self.logger.error(f"Interpreted execution failed: {e}")
            metrics = ExecutionMetrics(
                mode=ExecutionMode.INTERPRETED,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                memory_mb=0,
                cpu_percent=0,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                errors=[str(e)],
            )
            raise

    def _compile_graph(self, graph: Dict[str, Any], graph_hash: str) -> Optional[bytes]:
        """Compile graph to native code"""
        if not self.enable_compilation or not self.compiler:
            return None

        # Check if compilation is feasible
        if not self.compiler.can_compile(graph):
            self.logger.debug(f"Graph {graph_hash} not compilable")
            return None

        # Try to compile with timeout
        compilation_start = time.perf_counter()

        try:
            # Compile in separate thread to allow timeout
            future = self.thread_executor.submit(self.compiler.compile_graph, graph)
            binary = future.result(timeout=self.compilation_timeout)

            compilation_time = (time.perf_counter() - compilation_start) * 1000
            self.logger.info(f"Compiled graph {graph_hash} in {compilation_time:.2f}ms")

            # Cache the binary
            if binary:
                self.binary_cache.put(
                    graph_hash,
                    binary,
                    {
                        "compilation_time_ms": compilation_time,
                        "optimization_level": self.optimization_level.value,
                    },
                )

            return binary

        except Exception as e:
            self.logger.error(f"Compilation failed for {graph_hash}: {e}")
            self.compilation_failures += 1
            return None

    def _execute_compiled(
        self, graph: Dict[str, Any], binary: bytes, context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], ExecutionMetrics]:
        """Execute compiled binary"""
        start_time = time.perf_counter()
        start_mem, start_cpu = self._measure_resources()
        lib_path = None

        try:
            # Create temporary shared library
            with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
                f.write(binary)
                lib_path = f.name

            # Load and execute
            lib = ctypes.CDLL(lib_path)

            # Prepare input data based on graph
            inputs = self._prepare_inputs(graph, context)
            outputs = self._allocate_outputs(graph)

            # Call main entry point
            entry_point = lib.graphix_main
            entry_point.restype = ctypes.c_int

            # Execute
            result_code = entry_point(
                ctypes.byref(inputs),
                ctypes.byref(outputs),
                ctypes.c_int(len(graph.get("nodes", []))),
            )

            if result_code != 0:
                raise RuntimeError(f"Compiled execution failed with code {result_code}")

            # Extract results
            result = self._extract_outputs(outputs, graph)

            # Measure metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            end_mem, end_cpu = self._measure_resources()

            metrics = ExecutionMetrics(
                mode=ExecutionMode.COMPILED,
                duration_ms=duration_ms,
                memory_mb=max(0, end_mem - start_mem),
                cpu_percent=end_cpu,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
            )

            return result, metrics

        except Exception as e:
            self.logger.error(f"Compiled execution failed: {e}")

            metrics = ExecutionMetrics(
                mode=ExecutionMode.COMPILED,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                memory_mb=0,
                cpu_percent=0,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                errors=[str(e)],
            )
            raise
        finally:
            # Ensure cleanup happens
            if lib_path and os.path.exists(lib_path):
                try:
                    os.unlink(lib_path)
                except Exception as cleanup_e:
                    self.logger.error(
                        f"Failed to clean up temporary file {lib_path}: {cleanup_e}"
                    )

    def _prepare_inputs(self, graph: Dict[str, Any], context: Dict[str, Any] = None):
        """Prepare input data structure for compiled execution"""

        # Create ctypes structure based on graph inputs
        class Inputs(ctypes.Structure):
            _fields_ = []

        # Add fields based on input nodes
        for node in graph.get("nodes", []):
            if node.get("type") == "InputNode":
                # Determine type and add field
                value = node.get("params", {}).get("value", 0)
                if isinstance(value, float):
                    field_type = ctypes.c_double
                elif isinstance(value, int):
                    field_type = ctypes.c_int
                elif isinstance(value, (list, np.ndarray)):
                    # Array type
                    arr = np.asarray(value, dtype=np.float32)
                    field_type = ctypes.POINTER(ctypes.c_float)
                else:
                    field_type = ctypes.c_void_p

                Inputs._fields_.append((node["id"], field_type))

        # Handle empty inputs case
        if not Inputs._fields_:
            Inputs._fields_.append(("dummy", ctypes.c_int))

        # Create and populate instance
        inputs = Inputs()
        for node in graph.get("nodes", []):
            if node.get("type") == "InputNode":
                value = node.get("params", {}).get("value", 0)
                if isinstance(value, (list, np.ndarray)):
                    arr = np.asarray(value, dtype=np.float32)
                    setattr(
                        inputs,
                        node["id"],
                        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )
                else:
                    setattr(inputs, node["id"], value)

        return inputs

    def _allocate_outputs(self, graph: Dict[str, Any]):
        """Allocate output buffer for compiled execution"""
        output_nodes = [
            n for n in graph.get("nodes", []) if n.get("type") == "OutputNode"
        ]

        if not output_nodes:
            output_nodes = [{"id": "default", "params": {"size": 1}}]

        # Create output structure
        class Outputs(ctypes.Structure):
            _fields_ = []

        for node in output_nodes:
            size = node.get("params", {}).get("size", 1024)
            Outputs._fields_.append((node["id"], ctypes.POINTER(ctypes.c_double)))

        outputs = Outputs()

        # Allocate buffers
        for node in output_nodes:
            size = node.get("params", {}).get("size", 1024)
            buffer = (ctypes.c_double * size)()
            setattr(
                outputs,
                node["id"],
                ctypes.cast(buffer, ctypes.POINTER(ctypes.c_double)),
            )

        return outputs

    def _extract_outputs(self, outputs, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Extract results from output buffer"""
        result = {"outputs": {}}

        output_nodes = [
            n for n in graph.get("nodes", []) if n.get("type") == "OutputNode"
        ]

        for node in output_nodes:
            # Extract values from buffer
            size = node.get("params", {}).get("size", 10)
            values = []
            output_ptr = getattr(outputs, node["id"])
            for j in range(size):
                values.append(output_ptr[j])

            result["outputs"][node["id"]] = values

        result["status"] = "success"
        result["execution_mode"] = "compiled"

        return result

    async def execute_with_profiling(
        self,
        graph: Dict[str, Any],
        context: Dict[str, Any] = None,
        force_mode: Optional[ExecutionMode] = None,
    ) -> Dict[str, Any]:
        """
        Execute graph with profiling and automatic mode selection
        """
        self.total_executions += 1
        graph_id = graph.get("id", "unknown")
        graph_hash = self._compute_graph_hash(graph)

        # Get or create profile
        with self.profile_lock:
            if graph_hash not in self.profiles:
                self.profiles[graph_hash] = GraphProfile(
                    graph_id=graph_id, graph_hash=graph_hash
                )
            profile = self.profiles[graph_hash]
            profile.execution_count += 1

        # Determine execution mode
        if force_mode:
            mode = force_mode
        elif not self.enable_profiling:
            mode = ExecutionMode.INTERPRETED
        elif profile.execution_count <= 3:
            # Profile both modes initially
            mode = (
                ExecutionMode.INTERPRETED
                if profile.execution_count % 2 == 1
                else ExecutionMode.COMPILED
            )
        else:
            # Use best known mode
            mode = profile.best_mode

        # Execute based on mode
        result = None
        metrics = None

        if mode == ExecutionMode.COMPILED:
            # Try compiled execution
            binary = None

            # Check cache first
            binary = self.binary_cache.get(graph_hash)
            if binary:
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for graph {graph_hash}")
            else:
                self.cache_misses += 1

                # Compile if needed
                with self.compilation_lock:
                    # Double-check after acquiring lock
                    binary = self.binary_cache.get(graph_hash)
                    if not binary and profile.compilation_success != False:
                        binary = self._compile_graph(graph, graph_hash)
                        if binary:
                            profile.compilation_success = True
                        else:
                            profile.compilation_success = False

            if binary:
                try:
                    result, metrics = self._execute_compiled(graph, binary, context)
                    profile.compiled_metrics.append(metrics)
                except Exception as e:
                    self.logger.warning(f"Compiled execution failed, falling back: {e}")
                    mode = ExecutionMode.INTERPRETED

        # Fall back to or use interpreted mode
        if mode == ExecutionMode.INTERPRETED or result is None:
            result, metrics = await self._execute_interpreted(graph, context)
            profile.interpreted_metrics.append(metrics)

        # Update profile
        with self.profile_lock:
            profile.update_best_mode()

            # Add to history
            self.execution_history.append(
                {
                    "graph_hash": graph_hash,
                    "mode": mode.value,
                    "duration_ms": metrics.duration_ms,
                    "timestamp": time.time(),
                }
            )

        # Periodically save profiles
        if self.total_executions % 100 == 0:
            self._save_profiles()

        # Add execution metadata
        result["execution_metrics"] = metrics.to_dict()
        result["execution_profile"] = {
            "best_mode": profile.best_mode.value,
            "speedup": profile.speedup,
            "execution_count": profile.execution_count,
        }

        return result

    async def benchmark_graph(
        self,
        graph: Dict[str, Any],
        iterations: int = 10,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark graph in both execution modes
        """
        results = {
            "graph_id": graph.get("id", "unknown"),
            "graph_hash": self._compute_graph_hash(graph),
            "iterations": iterations,
            "interpreted": [],
            "compiled": [],
            "compilation_time_ms": 0,
        }

        # Benchmark interpreted mode
        for _ in range(iterations):
            _, metrics = await self._execute_interpreted(graph, context)
            results["interpreted"].append(metrics.duration_ms)

        # Compile graph
        graph_hash = results["graph_hash"]
        compilation_start = time.perf_counter()
        binary = self._compile_graph(graph, graph_hash)
        results["compilation_time_ms"] = (
            time.perf_counter() - compilation_start
        ) * 1000

        # Benchmark compiled mode if successful
        if binary:
            for _ in range(iterations):
                _, metrics = self._execute_compiled(graph, binary, context)
                results["compiled"].append(metrics.duration_ms)

        # Calculate statistics
        if results["interpreted"]:
            results["interpreted_stats"] = {
                "mean": np.mean(results["interpreted"]),
                "std": np.std(results["interpreted"]),
                "min": np.min(results["interpreted"]),
                "max": np.max(results["interpreted"]),
            }

        if results["compiled"]:
            results["compiled_stats"] = {
                "mean": np.mean(results["compiled"]),
                "std": np.std(results["compiled"]),
                "min": np.min(results["compiled"]),
                "max": np.max(results["compiled"]),
            }

            # Calculate speedup
            results["speedup"] = (
                results["interpreted_stats"]["mean"] / results["compiled_stats"]["mean"]
            )

            # Break-even point
            if results["compilation_time_ms"] > 0:
                time_saved_per_execution = (
                    results["interpreted_stats"]["mean"]
                    - results["compiled_stats"]["mean"]
                )
                if time_saved_per_execution > 0:
                    results["break_even_executions"] = int(
                        results["compilation_time_ms"] / time_saved_per_execution
                    )
                else:
                    results["break_even_executions"] = float("inf")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = {
            "total_executions": self.total_executions,
            "compilation_failures": self.compilation_failures,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
            "profiles_count": len(self.profiles),
            "compiled_graphs": sum(
                1 for p in self.profiles.values() if p.compilation_success
            ),
            "average_speedup": (
                np.mean([p.speedup for p in self.profiles.values()])
                if self.profiles
                else 1.0
            ),
        }

        # Mode distribution
        mode_counts = defaultdict(int)
        for entry in self.execution_history:
            mode_counts[entry["mode"]] += 1
        stats["mode_distribution"] = dict(mode_counts)

        # Recent performance
        if self.execution_history:
            recent = list(self.execution_history)[-100:]
            stats["recent_avg_duration_ms"] = np.mean(
                [e["duration_ms"] for e in recent]
            )

        return stats

    def cleanup(self):
        """Cleanup resources"""
        if self._executors_shutdown:
            return

        self._executors_shutdown = True

        # Save profiles
        self._save_profiles()

        # Clean cache
        self.binary_cache.cleanup()

        # Shutdown executors properly
        self.thread_executor.shutdown(wait=True, cancel_futures=True)
        self.process_executor.shutdown(wait=True, cancel_futures=True)

    def _atexit_cleanup(self):
        """Cleanup called by atexit"""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Cleanup on exit failed: {e}")

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Cleanup in destructor failed: {e}")
