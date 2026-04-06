"""
Statistics, metrics, and shutdown for unified reasoning orchestration.

Provides methods for updating metrics, recording history and audit
entries, gathering statistics, clearing caches, and graceful shutdown
with proper resource cleanup.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import inspect
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

from .types import ReasoningTask
from ..reasoning_types import ReasoningResult, ReasoningStrategy, ReasoningType

logger = logging.getLogger(__name__)


def update_metrics(
    reasoner: Any,
    result: ReasoningResult,
    elapsed_time: float,
    strategy: ReasoningStrategy,
) -> None:
    """Update performance metrics thread-safely."""
    with reasoner._stats_lock:
        if result and result.reasoning_type:
            reasoner.performance_metrics["type_usage"][
                result.reasoning_type
            ] += 1
        reasoner.performance_metrics["strategy_usage"][strategy] += 1

        n = reasoner.performance_metrics["total_reasonings"]

        if n > 0:
            old_avg_conf = reasoner.performance_metrics["average_confidence"]
            reasoner.performance_metrics["average_confidence"] = (
                old_avg_conf * (n - 1) + result.confidence
            ) / n

            old_avg_time = reasoner.performance_metrics["average_time"]
            reasoner.performance_metrics["average_time"] = (
                old_avg_time * (n - 1) + elapsed_time
            ) / n

        if result.confidence >= reasoner.confidence_threshold:
            reasoner.performance_metrics["successful_reasonings"] += 1


def add_to_history(
    reasoner: Any,
    task: ReasoningTask,
    result: ReasoningResult,
    elapsed_time: float,
) -> None:
    """Add reasoning to history."""
    try:
        history_entry = {
            "task_id": task.task_id,
            "reasoning_type": result.reasoning_type,
            "confidence": result.confidence,
            "elapsed_time": elapsed_time,
            "timestamp": time.time(),
            "success": result.confidence >= reasoner.confidence_threshold,
        }
        reasoner.reasoning_history.append(history_entry)
    except Exception as e:
        logger.warning(f"History update failed: {e}")


def add_audit_entry(
    reasoner: Any,
    task: ReasoningTask,
    result: ReasoningResult,
    strategy: ReasoningStrategy,
    elapsed_time: float,
) -> None:
    """Add entry to audit trail."""
    try:
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "strategy": strategy.value,
            "confidence": result.confidence,
            "elapsed_time": elapsed_time,
            "conclusion_type": type(result.conclusion).__name__,
            "safety_applied": reasoner.enable_safety,
            "utility_context": (
                task.utility_context.mode.value
                if (
                    task.utility_context
                    and hasattr(task.utility_context, "mode")
                )
                else None
            ),
        }
        reasoner.audit_trail.append(audit_entry)
    except Exception as e:
        logger.warning(f"Audit entry failed: {e}")


def get_statistics(reasoner: Any) -> Dict[str, Any]:
    """Get comprehensive statistics."""
    with reasoner._stats_lock:
        stats = {
            "performance": reasoner.performance_metrics.copy(),
            "cache_stats": {
                "result_cache_size": len(reasoner.result_cache),
                "plan_cache_size": len(reasoner.plan_cache),
            },
            "task_stats": {
                "completed_tasks": len(reasoner.completed_tasks),
                "active_tasks": len(reasoner.active_tasks),
                "queued_tasks": len(reasoner.task_queue),
            },
            "history_size": len(reasoner.reasoning_history),
            "audit_trail_size": len(reasoner.audit_trail),
            "execution_count": reasoner.execution_count,
        }

    try:
        if (
            reasoner.tool_selector
            and hasattr(reasoner.tool_selector, "get_statistics")
        ):
            stats["tool_selector_stats"] = (
                reasoner.tool_selector.get_statistics()
            )
        if (
            reasoner.tool_monitor
            and hasattr(reasoner.tool_monitor, "get_statistics")
        ):
            stats["monitor_stats"] = (
                reasoner.tool_monitor.get_statistics()
            )
        if (
            reasoner.voi_gate
            and hasattr(reasoner.voi_gate, "get_statistics")
        ):
            stats["voi_stats"] = reasoner.voi_gate.get_statistics()
    except Exception as e:
        logger.warning(f"Component statistics failed: {e}")

    for reasoning_type, eng in reasoner.reasoners.items():
        if hasattr(eng, "get_statistics"):
            try:
                stats[f"{reasoning_type.value}_stats"] = (
                    eng.get_statistics()
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get stats for {reasoning_type.value}: {e}"
                )

    return stats


def clear_caches(reasoner: Any) -> None:
    """Clear all caches."""
    with reasoner._cache_lock:
        reasoner.result_cache.clear()
        reasoner.plan_cache.clear()

        if reasoner.cache:
            try:
                if hasattr(reasoner.cache, "feature_cache"):
                    reasoner.cache.feature_cache.l1.clear()
                if hasattr(reasoner.cache, "selection_cache"):
                    reasoner.cache.selection_cache.l1.clear()
            except Exception as e:
                logger.warning(f"Cache clearing failed: {e}")

    logger.info("All caches cleared")


def shutdown_component(component: Any, name: str) -> None:
    """Shutdown a single component with proper parameter handling."""
    try:
        sig = inspect.signature(component.shutdown)
        if "timeout" in sig.parameters:
            component.shutdown(timeout=1.0)
        else:
            component.shutdown()
    except Exception as e:
        logger.warning(f"Component {name} shutdown raised: {e}")
        raise


def shutdown(
    reasoner: Any, timeout: float = 5.0, skip_save: bool = False
) -> None:
    """
    Shutdown unified reasoner with proper cleanup and timeout.

    Args:
        reasoner: UnifiedReasoner instance.
        timeout: Maximum time to wait for complete shutdown.
        skip_save: Skip auto-save during shutdown.
    """
    with reasoner._shutdown_lock:
        if reasoner._is_shutdown:
            logger.debug("System already shutdown")
            return
        reasoner._is_shutdown = True

    logger.info("Shutting down unified reasoning system")
    start_time = time.time()

    if not skip_save:
        try:
            reasoner.save_state("auto_save")
        except Exception as e:
            logger.error(f"Auto-save failed during shutdown: {e}")

    if hasattr(reasoner, "executor") and reasoner.executor:
        try:
            logger.debug("Shutting down main executor")
            reasoner.executor.shutdown(wait=True, cancel_futures=True)
            reasoner.executor = None
        except Exception as e:
            logger.error(f"Executor shutdown failed: {e}")

    components_to_shutdown = [
        ("cache", reasoner.cache),
        ("warm_pool", reasoner.warm_pool),
        ("tool_selector", reasoner.tool_selector),
        ("portfolio_executor", reasoner.portfolio_executor),
        ("safety_governor", reasoner.safety_governor),
        ("tool_monitor", reasoner.tool_monitor),
        ("processor", reasoner.processor),
        ("runtime", reasoner.runtime),
    ]

    for name, component in components_to_shutdown:
        if component and hasattr(component, "shutdown"):
            if time.time() - start_time >= timeout:
                logger.warning(
                    "Overall timeout reached, forcing remaining shutdowns"
                )
                break

            try:
                logger.debug(f"Shutting down {name}")
                sig = inspect.signature(component.shutdown)
                if "timeout" in sig.parameters:
                    component.shutdown(timeout=1.0)
                else:
                    component.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {name}: {e}")

    try:
        from vulcan.routing.governance_logger import (
            shutdown_buffered_governance_logger,
            shutdown_governance_logger,
        )
        shutdown_buffered_governance_logger()
        shutdown_governance_logger()
        logger.debug("Governance loggers shut down")
    except Exception as e:
        logger.warning(f"Error shutting down governance loggers: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Shutdown complete in {elapsed:.2f}s")


def getstate(reasoner: Any) -> Dict[str, Any]:
    """Prepare state for pickling (multiprocessing)."""
    state = reasoner.__dict__.copy()

    non_picklable = [
        'executor', '_cache_lock', '_stats_lock', '_shutdown_lock',
        'tool_selector', 'portfolio_executor', 'warm_pool', 'cache',
        'tool_monitor', 'safety_governor',
    ]

    for attr in non_picklable:
        if attr in state:
            state[attr] = None

    state['_reinit_max_workers'] = getattr(reasoner, 'max_workers', 4)
    return state


def setstate(reasoner: Any, state: Dict[str, Any]) -> None:
    """Restore state after unpickling (in child process)."""
    reasoner.__dict__.update(state)

    max_workers = state.get('_reinit_max_workers', 4)
    reasoner.executor = ThreadPoolExecutor(max_workers=max_workers)

    reasoner._cache_lock = threading.RLock()
    reasoner._stats_lock = threading.RLock()
    reasoner._shutdown_lock = threading.Lock()

    reasoner._is_shutdown = False
