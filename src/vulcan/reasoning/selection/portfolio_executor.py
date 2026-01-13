"""
Portfolio Executor for Tool Selection System

Implements sophisticated multi-tool execution strategies including speculative
parallelism, sequential refinement, committee consensus, and adaptive mixing.

Fixed version with proper resource management and timeout handling.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Reasoning Engine Imports
# ============================================================================
# Import all reasoning engines for default initialization.
# Note: These imports use explicit module paths to avoid ambiguity:
# - ProbabilisticReasoner: probabilistic_reasoning.py (GP-based), NOT symbolic/reasoner.py (Bayesian network)
# - MathematicalComputationTool: mathematical_computation.py (SOTA tool), NOT mathematical_verification.py
# - SymbolicReasoner: symbolic/reasoner.py (FOL theorem prover)

# Tracking which engines are available (graceful degradation if imports fail)
_AVAILABLE_ENGINES: Dict[str, type] = {}

try:
    from vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoner
    _AVAILABLE_ENGINES['probabilistic'] = ProbabilisticReasoner
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] ProbabilisticReasoner not available: {e}")

try:
    from vulcan.reasoning.symbolic.reasoner import SymbolicReasoner
    _AVAILABLE_ENGINES['symbolic'] = SymbolicReasoner
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] SymbolicReasoner not available: {e}")

try:
    from vulcan.reasoning.causal_reasoning import CausalReasoner
    _AVAILABLE_ENGINES['causal'] = CausalReasoner
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] CausalReasoner not available: {e}")

try:
    from vulcan.reasoning.mathematical_computation import MathematicalComputationTool
    _AVAILABLE_ENGINES['mathematical'] = MathematicalComputationTool
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] MathematicalComputationTool not available: {e}")

# PHILOSOPHICAL REASONER REMOVED: Ethical reasoning now handled by World Model
# The World Model has full meta-reasoning machinery for ethical reasoning.
# Route 'philosophical' tool requests to 'world_model' with mode='philosophical'
logger.info("[PortfolioExecutor] Philosophical reasoning: Routed to World Model")

try:
    from vulcan.reasoning.analogical import AnalogicalReasoningEngine
    _AVAILABLE_ENGINES['analogical'] = AnalogicalReasoningEngine
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] AnalogicalReasoningEngine not available: {e}")

try:
    from vulcan.reasoning.multimodal_reasoning import MultimodalReasoner
    _AVAILABLE_ENGINES['multimodal'] = MultimodalReasoner
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] MultimodalReasoner not available: {e}")

# Pattern 1 FIX: Add CryptographicEngine to available engines
# The 'cryptographic' tool was in the tools list but not imported/initialized
try:
    from vulcan.reasoning.cryptographic_engine import CryptographicEngine
    _AVAILABLE_ENGINES['cryptographic'] = CryptographicEngine
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] CryptographicEngine not available: {e}")

# Pattern 1 FIX: Add WorldModel adapter to available engines
# This allows world_model to be used as a reasoning tool
try:
    from vulcan.world_model.world_model_core import WorldModel
    _AVAILABLE_ENGINES['world_model'] = WorldModel
except ImportError as e:
    logger.warning(f"[PortfolioExecutor] WorldModel not available: {e}")


class ExecutionStrategy(Enum):
    """Portfolio execution strategies"""

    SINGLE = "single"
    SPECULATIVE_PARALLEL = "speculative_parallel"
    SEQUENTIAL_REFINEMENT = "sequential_refinement"
    COMMITTEE_CONSENSUS = "committee_consensus"
    CASCADE = "cascade"
    TOURNAMENT = "tournament"
    ADAPTIVE_MIX = "adaptive_mix"


class ExecutionStatus(Enum):
    """Execution status for tools"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ToolExecution:
    """Single tool execution context"""

    tool_name: str
    tool_instance: Any
    problem: Any
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    future: Optional[Future] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioResult:
    """Result from portfolio execution"""

    strategy: ExecutionStrategy
    primary_result: Any
    all_results: Dict[str, Any]
    execution_time: float
    energy_used: float
    tools_used: List[str]
    consensus_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionMonitor:
    """Monitors tool execution for timeouts and quality thresholds"""

    def __init__(
        self,
        time_budget_ms: float,
        energy_budget_mj: float,
        min_confidence: float = 0.5,
    ):
        self.time_budget = time_budget_ms / 1000.0  # Convert to seconds
        self.energy_budget = energy_budget_mj
        self.min_confidence = min_confidence

        self.start_time = time.time()
        self.energy_consumed = 0.0
        self.executions = []

        # CRITICAL FIX: Use RLock for thread safety
        self.lock = threading.RLock()

    def time_remaining(self) -> float:
        """Get remaining time in seconds"""
        elapsed = time.time() - self.start_time
        return max(0, self.time_budget - elapsed)

    def energy_remaining(self) -> float:
        """Get remaining energy budget"""
        with self.lock:
            return max(0, self.energy_budget - self.energy_consumed)

    def is_timeout(self) -> bool:
        """Check if we've exceeded time budget"""
        return self.time_remaining() <= 0

    def is_energy_exceeded(self) -> bool:
        """Check if we've exceeded energy budget"""
        return self.energy_remaining() <= 0

    def should_continue(self, partial_result: Optional[Any] = None) -> bool:
        """Check if execution should continue"""
        if self.is_timeout() or self.is_energy_exceeded():
            return False

        # Check partial result confidence if available
        if partial_result and hasattr(partial_result, "confidence"):
            if partial_result.confidence >= self.min_confidence:
                return True  # Good enough, can stop

        return True

    def record_execution(self, tool_name: str, time_ms: float, energy_mj: float):
        """Record tool execution metrics"""
        with self.lock:
            self.energy_consumed += energy_mj
            self.executions.append(
                {
                    "tool": tool_name,
                    "time_ms": time_ms,
                    "energy_mj": energy_mj,
                    "timestamp": time.time(),
                }
            )


class PortfolioExecutor:
    """
    Executes multiple tools with various strategies
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None, max_workers: int = 4):
        """
        Args:
            tools: Optional dictionary of tool_name -> tool_instance
                   If not provided, will initialize all default reasoning engines
            max_workers: Maximum parallel executions
        """
        # Initialize reasoning engines if not provided
        if tools is None:
            tools = self._initialize_default_engines()
        
        self.tools = tools
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # CRITICAL FIX: Add locks for thread safety
        self._stats_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

        # Execution statistics
        self.execution_history = deque(maxlen=1000)
        self.strategy_performance = defaultdict(
            lambda: {"count": 0, "successes": 0, "avg_time": 0.0, "avg_energy": 0.0}
        )

        # Strategy implementations
        self.strategies = {
            ExecutionStrategy.SINGLE: self._execute_single,
            ExecutionStrategy.SPECULATIVE_PARALLEL: self._execute_speculative_parallel,
            ExecutionStrategy.SEQUENTIAL_REFINEMENT: self._execute_sequential_refinement,
            ExecutionStrategy.COMMITTEE_CONSENSUS: self._execute_committee_consensus,
            ExecutionStrategy.CASCADE: self._execute_cascade,
            ExecutionStrategy.TOURNAMENT: self._execute_tournament,
            ExecutionStrategy.ADAPTIVE_MIX: self._execute_adaptive_mix,
        }

    def _initialize_default_engines(self) -> Dict[str, Any]:
        """
        Initialize all available reasoning engines.
        
        Uses the _AVAILABLE_ENGINES module-level dict which contains only
        engines that were successfully imported. Each engine is instantiated
        with its default (parameter-less) constructor.
        
        Note: All reasoning engines in this codebase support parameter-less
        initialization with sensible defaults. If an engine requires configuration,
        users should pass a pre-configured tools dict to __init__ instead.

        Returns:
            Dictionary mapping tool names to initialized engine instances
        """
        engines = {}
        total_engines = len(_AVAILABLE_ENGINES)

        for name, engine_class in _AVAILABLE_ENGINES.items():
            try:
                engines[name] = engine_class()
                logger.info(f"[PortfolioExecutor] ✓ Initialized {name} engine")
            except Exception as e:
                logger.warning(f"[PortfolioExecutor] ✗ Failed to initialize {name}: {e}")

        logger.info(
            f"[PortfolioExecutor] Successfully registered {len(engines)}/{total_engines} "
            f"reasoning engines: {list(engines.keys())}"
        )

        return engines

    # =============================================================================
    # Note (Jan 7 2026): Tool Fallback Mapping
    # =============================================================================
    # If a requested tool isn't available, map it to an equivalent tool.
    # 
    # PHILOSOPHICAL REASONER REMOVED: Ethical reasoning now handled by World Model
    # Route 'philosophical' requests to 'world_model' with mode='philosophical'
    #
    # COMPREHENSIVE FIX (Jan 7 2026): Added 'creative' and 'general' tool mappings
    # to ensure all query types route through VULCAN reasoning, not OpenAI directly.
    TOOL_FALLBACKS: Dict[str, List[str]] = {
        'philosophical': ['world_model', 'causal', 'analogical'],  # Philosophical → world_model (mode='philosophical')
        # Creative should fallback to world_model (has _creative_reasoning method)
        'creative': ['world_model', 'symbolic'],  # Creative → world_model (mode='creative')
        'cryptographic': ['symbolic', 'mathematical'],  # Crypto falls back to symbolic
        'analogical': ['symbolic', 'probabilistic'],  # Analogical falls back to symbolic
        'causal': ['probabilistic', 'symbolic'],  # Causal falls back to probabilistic
        'mathematical': ['symbolic'],  # Mathematical falls back to symbolic
        'multimodal': ['symbolic'],  # Multimodal falls back to symbolic
        'general': ['world_model', 'symbolic'],  # General queries → world_model
    }

    def _get_fallback_tools(self, tool_name: str) -> List[str]:
        """
        Get fallback tools for an unavailable tool.
        
        Args:
            tool_name: The requested tool that isn't available
            
        Returns:
            List of fallback tool names that could substitute
        """
        fallbacks = self.TOOL_FALLBACKS.get(tool_name, [])
        available_fallbacks = [t for t in fallbacks if t in self.tools]
        return available_fallbacks

    def execute(
        self,
        strategy: ExecutionStrategy,
        tool_names: List[str],
        problem: Any,
        constraints: Dict[str, float],
        monitor: Optional[ExecutionMonitor] = None,
    ) -> PortfolioResult:
        """
        Execute tools with specified strategy

        Args:
            strategy: Execution strategy to use
            tool_names: List of tool names to execute
            problem: Problem to solve
            constraints: Execution constraints (time_budget_ms, energy_budget_mj, min_confidence)
            monitor: Optional execution monitor

        Returns:
            PortfolioResult with execution results
        """

        # CRITICAL FIX: Check if shutdown
        with self._shutdown_lock:
            if self._is_shutdown:
                logger.error("Cannot execute: system is shutdown")
                return self._error_result(tool_names, "System is shutdown")

        try:
            if not monitor:
                # BUG #6 FIX: Increase timeout for philosophical tools
                # Philosophical reasoning takes longer (15-20s) than the default 5s timeout
                time_budget = constraints.get("time_budget_ms", 5000)
                if any(tool in ['philosophical', 'world_model'] for tool in tool_names):
                    # Use higher timeout for philosophical/world_model tools
                    time_budget = max(time_budget, 30000)  # BUG #6 FIX: 30 second timeout
                    logger.info(
                        f"[PortfolioExecutor] BUG #6 FIX: Using extended timeout "
                        f"({time_budget}ms) for philosophical reasoning"
                    )
                
                monitor = ExecutionMonitor(
                    time_budget,
                    constraints.get("energy_budget_mj", 1000),
                    constraints.get("min_confidence", 0.5),
                )

            # Validate tools exist, with fallback support
            valid_tools = [t for t in tool_names if t in self.tools]
            
            # Note (Jan 7 2026): Try fallbacks for unavailable tools
            if not valid_tools:
                for requested_tool in tool_names:
                    if requested_tool not in self.tools:
                        fallbacks = self._get_fallback_tools(requested_tool)
                        if fallbacks:
                            logger.warning(
                                f"[PortfolioExecutor] Note: Tool '{requested_tool}' not available, "
                                f"using fallback: {fallbacks[0]}"
                            )
                            valid_tools.append(fallbacks[0])
                            break
            
            if not valid_tools:
                available = list(self.tools.keys())
                logger.error(f"No valid tools found. Requested: {tool_names}, Available: {available}")
                raise ValueError(f"No valid tools found in {tool_names}. Available tools: {available}")

            # Execute strategy
            strategy_func = self.strategies.get(strategy, self._execute_single)
            result = strategy_func(valid_tools, problem, monitor)

            # Update statistics
            self._update_statistics(strategy, result)

            return result
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return self._error_result(tool_names, str(e))

    def _execute_single(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """Execute single tool with proper resource management"""

        tool_name = tool_names[0]
        tool = self.tools[tool_name]

        start_time = time.time()
        future = None

        try:
            # Execute tool with timeout
            future = self.executor.submit(self._run_tool, tool, problem)

            # CRITICAL FIX: Use proper timeout
            timeout = max(0.1, monitor.time_remaining())
            result = future.result(timeout=timeout)

            execution_time = time.time() - start_time

            # Estimate energy (simplified)
            energy = self._estimate_energy(tool_name, execution_time)

            return PortfolioResult(
                strategy=ExecutionStrategy.SINGLE,
                primary_result=result,
                all_results={tool_name: result},
                execution_time=execution_time,
                energy_used=energy,
                tools_used=[tool_name],
            )

        except TimeoutError:
            logger.warning(f"Tool {tool_name} timed out")
            return self._timeout_result([tool_name])
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return self._error_result([tool_name], str(e))
        finally:
            # CRITICAL FIX: Ensure future is cancelled if not done
            if future is not None and not future.done():
                future.cancel()

    # CRITICAL FIX: Complete rewrite with proper resource management
    def _execute_speculative_parallel(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Run multiple tools in parallel, take first good result
        CRITICAL: Proper future tracking and cleanup
        """

        if len(tool_names) < 2:
            return self._execute_single(tool_names, problem, monitor)

        # Sort tools by expected speed (fast first)
        sorted_tools = sorted(tool_names, key=lambda t: self._get_tool_speed_rank(t))

        # CRITICAL FIX: Track all futures for cleanup
        executions = []
        futures = []

        try:
            # Start executions with staggered delays
            delay = 0

            for tool_name in sorted_tools[:3]:  # Max 3 parallel
                tool = self.tools[tool_name]

                # Create execution
                execution = ToolExecution(
                    tool_name=tool_name, tool_instance=tool, problem=problem
                )

                # Submit with delay
                if delay > 0:
                    time.sleep(delay / 1000.0)  # Convert ms to seconds

                future = self.executor.submit(self._run_tool, tool, problem)
                execution.future = future
                execution.status = ExecutionStatus.RUNNING
                executions.append(execution)
                futures.append(future)  # CRITICAL: Track for cleanup

                # Increase delay for next tool (give fast tools head start)
                delay += 50  # 50ms additional delay

            # Monitor executions
            start_time = time.time()
            all_results = {}
            primary_result = None
            tools_used = []

            while executions and not monitor.is_timeout():
                for execution in list(executions):
                    if execution.future.done():
                        try:
                            # CRITICAL FIX: Use timeout even for done futures
                            result = execution.future.result(timeout=0.1)
                            execution.status = ExecutionStatus.SUCCESS
                            execution.result = result
                            execution.end_time = time.time()

                            all_results[execution.tool_name] = result
                            tools_used.append(execution.tool_name)

                            # Check if result is good enough
                            if self._is_acceptable_result(
                                result, monitor.min_confidence
                            ):
                                primary_result = result

                                # Cancel remaining executions
                                for other in executions:
                                    if other != execution and other.future:
                                        other.future.cancel()
                                        other.status = ExecutionStatus.CANCELLED

                                # Return early with good result
                                execution_time = time.time() - start_time
                                energy = sum(
                                    self._estimate_energy(
                                        t, execution_time / len(tools_used)
                                    )
                                    for t in tools_used
                                )

                                return PortfolioResult(
                                    strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
                                    primary_result=primary_result,
                                    all_results=all_results,
                                    execution_time=execution_time,
                                    energy_used=energy,
                                    tools_used=tools_used,
                                    metadata={"early_termination": True},
                                )

                            executions.remove(execution)

                        except TimeoutError:
                            logger.warning(f"Tool {execution.tool_name} result timeout")
                            execution.status = ExecutionStatus.TIMEOUT
                            executions.remove(execution)
                        except Exception as e:
                            logger.error(f"Tool {execution.tool_name} failed: {e}")
                            execution.status = ExecutionStatus.FAILED
                            execution.error = e
                            executions.remove(execution)

                time.sleep(0.01)  # Small sleep to prevent busy waiting

            # No early termination, return best result
            if all_results:
                primary_result = self._select_best_result(all_results)

            execution_time = time.time() - start_time
            energy = (
                sum(self._estimate_energy(t, execution_time) for t in tools_used)
                if tools_used
                else 0
            )

            return PortfolioResult(
                strategy=ExecutionStrategy.SPECULATIVE_PARALLEL,
                primary_result=primary_result,
                all_results=all_results,
                execution_time=execution_time,
                energy_used=energy,
                tools_used=tools_used,
            )

        finally:
            # CRITICAL FIX: Ensure all futures are cancelled/completed
            for future in futures:
                if future is not None and not future.done():
                    future.cancel()

    def _execute_sequential_refinement(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Use cheap tool to narrow space, then expensive tool to refine
        CRITICAL: Proper future cleanup
        """

        if len(tool_names) < 2:
            return self._execute_single(tool_names, problem, monitor)

        # Phase 1: Fast exploration (usually analogical)
        explorer_name = self._get_fastest_tool(tool_names)
        explorer = self.tools[explorer_name]

        start_time = time.time()
        all_results = {}
        tools_used = []

        # CRITICAL FIX: Track futures for cleanup
        futures = []

        try:
            # Get candidates from explorer
            future = self.executor.submit(self._run_tool, explorer, problem)
            futures.append(future)

            timeout = max(0.1, monitor.time_remaining() * 0.3)
            exploration_result = future.result(timeout=timeout)

            all_results[explorer_name] = exploration_result
            tools_used.append(explorer_name)

            # Extract candidates (tool-specific logic)
            candidates = self._extract_candidates(exploration_result)

            if not candidates:
                # Fall back to full analysis with most accurate tool
                refiner_name = self._get_most_accurate_tool(tool_names)
                refiner = self.tools[refiner_name]

                future = self.executor.submit(self._run_tool, refiner, problem)
                futures.append(future)

                timeout = max(0.1, monitor.time_remaining())
                refined_result = future.result(timeout=timeout)

                all_results[refiner_name] = refined_result
                tools_used.append(refiner_name)
                primary_result = refined_result
            else:
                # Phase 2: Refine candidates
                refiner_name = self._get_most_accurate_tool(tool_names)
                refiner = self.tools[refiner_name]

                best_result = None
                best_score = -float("inf")

                for candidate in candidates[:3]:  # Refine top 3 candidates
                    if monitor.is_timeout():
                        break

                    # Validate/refine candidate
                    refined_problem = self._create_refined_problem(problem, candidate)
                    future = self.executor.submit(
                        self._run_tool, refiner, refined_problem
                    )
                    futures.append(future)

                    try:
                        remaining = monitor.time_remaining()
                        timeout = max(0.1, remaining / len(candidates))
                        result = future.result(timeout=timeout)
                        score = self._score_result(result)

                        if score > best_score:
                            best_score = score
                            best_result = result
                    except Exception as e:
                        logger.warning(f"Refinement failed for candidate: {e}")
                        continue

                all_results[refiner_name] = best_result
                tools_used.append(refiner_name)
                primary_result = best_result

            execution_time = time.time() - start_time
            energy = (
                sum(
                    self._estimate_energy(t, execution_time / len(tools_used))
                    for t in tools_used
                )
                if tools_used
                else 0
            )

            return PortfolioResult(
                strategy=ExecutionStrategy.SEQUENTIAL_REFINEMENT,
                primary_result=primary_result,
                all_results=all_results,
                execution_time=execution_time,
                energy_used=energy,
                tools_used=tools_used,
                metadata={"candidates_explored": len(candidates) if candidates else 0},
            )

        except TimeoutError:
            logger.warning("Sequential refinement timed out")
            return self._timeout_result(tool_names)
        except Exception as e:
            logger.error(f"Sequential refinement failed: {e}")
            return self._error_result(tool_names, str(e))
        finally:
            # CRITICAL FIX: Cancel all futures
            for future in futures:
                if future is not None and not future.done():
                    future.cancel()

    def _execute_committee_consensus(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Run multiple tools and combine results through voting/consensus
        CRITICAL: Proper future cleanup
        """

        # CRITICAL FIX: Track all futures
        executions = []
        futures = []

        try:
            # Run all tools in parallel
            for tool_name in tool_names[:5]:  # Max 5 for committee
                tool = self.tools[tool_name]
                execution = ToolExecution(
                    tool_name=tool_name, tool_instance=tool, problem=problem
                )
                future = self.executor.submit(self._run_tool, tool, problem)
                execution.future = future
                execution.status = ExecutionStatus.RUNNING
                executions.append(execution)
                futures.append(future)

            # Collect results
            start_time = time.time()
            all_results = {}
            tools_used = []

            # Wait for all or timeout
            remaining_time = monitor.time_remaining()
            timeout_per_tool = (
                max(0.1, remaining_time / len(executions)) if executions else 0.1
            )

            for execution in executions:
                try:
                    result = execution.future.result(timeout=timeout_per_tool)
                    all_results[execution.tool_name] = result
                    tools_used.append(execution.tool_name)
                    execution.status = ExecutionStatus.SUCCESS
                except TimeoutError:
                    logger.warning(f"Committee tool {execution.tool_name} timed out")
                    execution.status = ExecutionStatus.TIMEOUT
                except Exception as e:
                    logger.warning(f"Committee tool {execution.tool_name} failed: {e}")
                    execution.status = ExecutionStatus.FAILED
                    execution.error = e

            if not all_results:
                return self._error_result(tool_names, "All tools failed")

            # Compute consensus
            consensus_result, consensus_confidence = self._compute_consensus(
                all_results
            )

            execution_time = time.time() - start_time
            energy = (
                sum(self._estimate_energy(t, execution_time) for t in tools_used)
                if tools_used
                else 0
            )

            return PortfolioResult(
                strategy=ExecutionStrategy.COMMITTEE_CONSENSUS,
                primary_result=consensus_result,
                all_results=all_results,
                execution_time=execution_time,
                energy_used=energy,
                tools_used=tools_used,
                consensus_confidence=consensus_confidence,
                metadata={"committee_size": len(all_results)},
            )

        finally:
            # CRITICAL FIX: Cancel all futures
            for future in futures:
                if future is not None and not future.done():
                    future.cancel()

    def _execute_cascade(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Execute tools in cascade, each refining previous result
        CRITICAL: Proper future cleanup
        """

        all_results = {}
        tools_used = []
        current_result = None
        start_time = time.time()

        # CRITICAL FIX: Track futures
        futures = []

        try:
            for tool_name in tool_names:
                if monitor.is_timeout():
                    break

                tool = self.tools[tool_name]

                # Adapt problem based on previous result
                if current_result:
                    adapted_problem = self._adapt_problem_with_result(
                        problem, current_result
                    )
                else:
                    adapted_problem = problem

                try:
                    future = self.executor.submit(self._run_tool, tool, adapted_problem)
                    futures.append(future)

                    timeout = max(0.1, monitor.time_remaining())
                    result = future.result(timeout=timeout)

                    all_results[tool_name] = result
                    tools_used.append(tool_name)
                    current_result = result

                    # Check if we can stop early
                    if self._is_acceptable_result(result, monitor.min_confidence * 1.2):
                        break

                except TimeoutError:
                    logger.warning(f"Tool {tool_name} timed out in cascade")
                    break
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed in cascade: {e}")
                    continue

            execution_time = time.time() - start_time
            energy = (
                sum(
                    self._estimate_energy(t, execution_time / len(tools_used))
                    for t in tools_used
                )
                if tools_used
                else 0
            )

            return PortfolioResult(
                strategy=ExecutionStrategy.CASCADE,
                primary_result=current_result,
                all_results=all_results,
                execution_time=execution_time,
                energy_used=energy,
                tools_used=tools_used,
            )

        finally:
            # CRITICAL FIX: Cancel all futures
            for future in futures:
                if future is not None and not future.done():
                    future.cancel()

    def _execute_tournament(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Tournament-style elimination of tools
        CRITICAL: Proper future cleanup
        """

        if len(tool_names) < 2:
            return self._execute_single(tool_names, problem, monitor)

        # CRITICAL FIX: Track all futures
        all_futures = []

        try:
            # Run initial round
            contestants = tool_names.copy()
            all_results = {}
            tools_used = []
            start_time = time.time()

            while len(contestants) > 1 and not monitor.is_timeout():
                # Pair up contestants
                pairs = []
                for i in range(0, len(contestants), 2):
                    if i + 1 < len(contestants):
                        pairs.append((contestants[i], contestants[i + 1]))
                    else:
                        pairs.append((contestants[i],))  # Bye

                # Run matches
                winners = []
                for pair in pairs:
                    if len(pair) == 1:
                        winners.append(pair[0])
                        continue

                    # Run both tools
                    results = {}
                    pair_futures = []

                    for tool_name in pair:
                        tool = self.tools[tool_name]
                        try:
                            future = self.executor.submit(self._run_tool, tool, problem)
                            pair_futures.append(future)
                            all_futures.append(future)

                            timeout = max(
                                0.1, monitor.time_remaining() / (2 * len(pairs))
                            )
                            result = future.result(timeout=timeout)

                            results[tool_name] = result
                            all_results[tool_name] = result
                            tools_used.append(tool_name)
                        except TimeoutError:
                            logger.warning(f"Tool {tool_name} timed out in tournament")
                        except Exception as e:
                            logger.warning(
                                f"Tool {tool_name} failed in tournament: {e}"
                            )

                    # Select winner
                    if results:
                        winner = max(
                            results.keys(), key=lambda k: self._score_result(results[k])
                        )
                        winners.append(winner)

                contestants = winners

            # Final winner
            primary_result = all_results.get(contestants[0]) if contestants else None

            execution_time = time.time() - start_time
            energy = (
                sum(
                    self._estimate_energy(t, execution_time / len(tools_used))
                    for t in tools_used
                )
                if tools_used
                else 0
            )

            return PortfolioResult(
                strategy=ExecutionStrategy.TOURNAMENT,
                primary_result=primary_result,
                all_results=all_results,
                execution_time=execution_time,
                energy_used=energy,
                tools_used=tools_used,
                metadata={"winner": contestants[0] if contestants else None},
            )

        finally:
            # CRITICAL FIX: Cancel all futures
            for future in all_futures:
                if future is not None and not future.done():
                    future.cancel()

    def _execute_adaptive_mix(
        self, tool_names: List[str], problem: Any, monitor: ExecutionMonitor
    ) -> PortfolioResult:
        """
        Adaptively mix strategies based on problem characteristics
        """

        try:
            # Analyze problem to select strategy
            problem_complexity = self._estimate_problem_complexity(problem)
            available_time = monitor.time_remaining()

            if problem_complexity < 0.3 and available_time > 5:
                # Simple problem with time: single tool
                result = self._execute_single(tool_names[:1], problem, monitor)
            elif problem_complexity < 0.6 and len(tool_names) >= 2:
                # Medium complexity: sequential refinement
                result = self._execute_sequential_refinement(
                    tool_names, problem, monitor
                )
            elif available_time < 2:
                # Low time: speculative parallel
                result = self._execute_speculative_parallel(
                    tool_names, problem, monitor
                )
            else:
                # Complex problem: committee consensus
                result = self._execute_committee_consensus(tool_names, problem, monitor)

            # Note: Ensure the final result's strategy is correctly reported as ADAPTIVE_MIX
            result.strategy = ExecutionStrategy.ADAPTIVE_MIX
            return result
        except Exception as e:
            logger.error(f"Adaptive mix failed: {e}")
            return self._error_result(tool_names, str(e))

    # Helper methods

    def _run_tool(self, tool: Any, problem: Any) -> Any:
        """Run a tool on a problem"""

        try:
            # Handle different tool interfaces
            if hasattr(tool, "reason"):
                return tool.reason(problem)
            elif hasattr(tool, "solve"):
                return tool.solve(problem)
            elif hasattr(tool, "execute"):
                return tool.execute(problem)
            elif callable(tool):
                return tool(problem)
            else:
                raise ValueError(f"Tool {tool} has no known execution method")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise

    def _is_acceptable_result(self, result: Any, min_confidence: float) -> bool:
        """Check if result meets minimum requirements"""

        if result is None:
            return False

        try:
            if hasattr(result, "confidence"):
                return result.confidence >= min_confidence

            if hasattr(result, "score"):
                return result.score >= min_confidence

            # Default: accept non-None results
            return True
        except Exception as e:
            logger.warning(f"Result validation failed: {e}")
            return False

    def _select_best_result(self, results: Dict[str, Any]) -> Any:
        """Select best result from multiple tools"""

        if not results:
            return None

        try:
            # Score each result
            scored = []
            for tool_name, result in results.items():
                score = self._score_result(result)
                scored.append((score, result))

            # Return highest scoring
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[0][1] if scored else None
        except Exception as e:
            logger.error(f"Result selection failed: {e}")
            return list(results.values())[0] if results else None

    def _score_result(self, result: Any) -> float:
        """Score a result for comparison
        
        BUG #4 FIX: Prioritize successful results over failures.
        When a specialized tool (like ProbabilisticEngine) returns a successful
        result with reasonable confidence, it should not be overridden by a
        generic tool's failure with low confidence.
        """

        if result is None:
            return 0.0

        try:
            score = 0.5  # Base score

            # BUG #4 FIX: Handle dict results (most common from reasoning engines)
            if isinstance(result, dict):
                # Check if this is an error/failure result
                has_error = result.get("error") is not None
                has_result = "result" in result or "answer" in result or "conclusion" in result
                
                # If result has an error and no actual answer, penalize heavily
                if has_error and not has_result:
                    score = result.get("confidence", 0.1) * 0.5  # Reduce failed result score
                else:
                    # Get confidence from dict
                    score = result.get("confidence", result.get("score", result.get("probability", 0.5)))
                    
                    # Bonus for successful results with actual content
                    if has_result and score > 0.0:
                        score = min(score * 1.1, 1.0)  # Small boost for results with content
            elif hasattr(result, "confidence"):
                score = result.confidence
            elif hasattr(result, "score"):
                score = result.score
            elif hasattr(result, "probability"):
                score = result.probability

            # CRITICAL FIX: Clamp score to valid range
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Result scoring failed: {e}")
            return 0.5

    def _compute_consensus(self, results: Dict[str, Any]) -> Tuple[Any, float]:
        """Compute consensus from multiple results"""

        if not results:
            return None, 0.0

        try:
            # For now, return highest confidence result
            # In practice, this would be more sophisticated
            best_result = self._select_best_result(results)

            # Compute consensus confidence
            scores = [self._score_result(r) for r in results.values()]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # High consensus if low variance
            # CRITICAL FIX: Handle edge cases
            consensus_confidence = mean_score * (1 - min(std_score, 1.0))
            consensus_confidence = float(np.clip(consensus_confidence, 0.0, 1.0))

            return best_result, consensus_confidence
        except Exception as e:
            logger.error(f"Consensus computation failed: {e}")
            return list(results.values())[0] if results else None, 0.0

    def _extract_candidates(self, exploration_result: Any) -> List[Any]:
        """Extract candidates from exploration result"""

        try:
            if hasattr(exploration_result, "candidates"):
                return exploration_result.candidates

            if isinstance(exploration_result, list):
                return exploration_result

            if (
                isinstance(exploration_result, dict)
                and "candidates" in exploration_result
            ):
                return exploration_result["candidates"]

            return []
        except Exception as e:
            logger.warning(f"Candidate extraction failed: {e}")
            return []

    def _create_refined_problem(self, original_problem: Any, candidate: Any) -> Any:
        """Create refined problem focusing on specific candidate"""

        try:
            if isinstance(original_problem, dict):
                return {**original_problem, "focus": candidate}
            else:
                return {"original": original_problem, "focus": candidate}
        except Exception as e:
            logger.warning(f"Problem refinement failed: {e}")
            return original_problem

    def _adapt_problem_with_result(self, problem: Any, previous_result: Any) -> Any:
        """Adapt problem based on previous result"""

        try:
            if isinstance(problem, dict):
                return {**problem, "previous_result": previous_result}
            else:
                return {"problem": problem, "previous_result": previous_result}
        except Exception as e:
            logger.warning(f"Problem adaptation failed: {e}")
            return problem

    def _get_tool_speed_rank(self, tool_name: str) -> int:
        """Get speed ranking for tool (lower is faster)"""

        speed_ranks = {
            "probabilistic": 1,
            "analogical": 2,
            "symbolic": 3,
            "causal": 4,
            "multimodal": 5,
        }

        return speed_ranks.get(tool_name, 10)

    def _get_fastest_tool(self, tool_names: List[str]) -> str:
        """Get fastest tool from list"""

        if not tool_names:
            return ""

        return min(tool_names, key=lambda t: self._get_tool_speed_rank(t))

    def _get_most_accurate_tool(self, tool_names: List[str]) -> str:
        """Get most accurate tool from list"""

        if not tool_names:
            return ""

        accuracy_ranks = {
            "causal": 1,
            "symbolic": 2,
            "multimodal": 3,
            "probabilistic": 4,
            "analogical": 5,
        }

        return min(tool_names, key=lambda t: accuracy_ranks.get(t, 10))

    def _estimate_energy(self, tool_name: str, execution_time: float) -> float:
        """Estimate energy consumption for tool"""

        base_energy = {
            "symbolic": 50,
            "probabilistic": 100,
            "causal": 200,
            "analogical": 80,
            "multimodal": 300,
        }

        base = base_energy.get(tool_name, 100)
        # CRITICAL FIX: Ensure non-negative
        return max(0, base + execution_time * 10)  # Simple linear model

    def _estimate_problem_complexity(self, problem: Any) -> float:
        """Estimate problem complexity (0-1)"""

        try:
            # Simple heuristics
            if isinstance(problem, str):
                return min(1.0, len(problem) / 1000)
            elif isinstance(problem, dict):
                return min(1.0, len(str(problem)) / 5000)
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"Complexity estimation failed: {e}")
            return 0.5

    def _timeout_result(self, tool_names: List[str]) -> PortfolioResult:
        """Create timeout result"""

        return PortfolioResult(
            strategy=ExecutionStrategy.SINGLE,
            primary_result=None,
            all_results={},
            execution_time=0,
            energy_used=0,
            tools_used=tool_names,
            metadata={"error": "timeout"},
        )

    def _error_result(self, tool_names: List[str], error: str) -> PortfolioResult:
        """Create error result"""

        return PortfolioResult(
            strategy=ExecutionStrategy.SINGLE,
            primary_result=None,
            all_results={},
            execution_time=0,
            energy_used=0,
            tools_used=tool_names,
            metadata={"error": error},
        )

    # CRITICAL FIX: Thread-safe statistics update
    def _update_statistics(self, strategy: ExecutionStrategy, result: PortfolioResult):
        """Update execution statistics"""

        with self._stats_lock:
            try:
                stats = self.strategy_performance[strategy.value]
                stats["count"] += 1

                if result.primary_result is not None:
                    stats["successes"] += 1

                # Update running averages
                alpha = 0.1  # Exponential moving average
                stats["avg_time"] = (1 - alpha) * stats[
                    "avg_time"
                ] + alpha * result.execution_time
                stats["avg_energy"] = (1 - alpha) * stats[
                    "avg_energy"
                ] + alpha * result.energy_used

                # Add to history
                self.execution_history.append(
                    {
                        "strategy": strategy.value,
                        "success": result.primary_result is not None,
                        "execution_time": result.execution_time,
                        "energy_used": result.energy_used,
                        "tools_used": result.tools_used,
                        "timestamp": time.time(),
                    }
                )
            except Exception as e:
                logger.error(f"Statistics update failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""

        with self._stats_lock:
            try:
                return {
                    "strategy_performance": dict(self.strategy_performance),
                    "total_executions": len(self.execution_history),
                    "recent_executions": list(self.execution_history)[-10:],
                }
            except Exception as e:
                logger.error(f"Statistics retrieval failed: {e}")
                return {}

    # CRITICAL FIX: Proper shutdown with timeout
    def shutdown(self, timeout: float = 10.0):
        """Shutdown executor with proper cleanup"""

        with self._shutdown_lock:
            if self._is_shutdown:
                logger.warning("Executor already shutdown")
                return

            self._is_shutdown = True

        logger.info("Shutting down portfolio executor")

        try:
            # Note: The 'timeout' parameter for shutdown was added in Python 3.9.
            # Removing it ensures compatibility with older versions and fixes the logged error.
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Executor shutdown failed: {e}")

        logger.info("Portfolio executor shutdown complete")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if not self._is_shutdown:
                self.shutdown(timeout=5.0)
        except Exception as e:
            logger.debug(f"Operation failed: {e}")
