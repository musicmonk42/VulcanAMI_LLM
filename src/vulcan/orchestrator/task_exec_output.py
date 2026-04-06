# ============================================================
# VULCAN-AGI Orchestrator - Task Execution Output Module
# Extracted from task_execution_core.py for modularity
# Contains: conclusion extraction, validation, output formatting
# ============================================================

import logging
import time
from typing import Any, Dict, Optional

from .agent_pool_types import (
    CONCLUSION_EXTRACTION_KEYS as _CONCLUSION_EXTRACTION_KEYS,
    MAX_CONCLUSION_EXTRACTION_DEPTH as _MAX_CONCLUSION_EXTRACTION_DEPTH,
)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _safe_get_attr(obj, attr_name, default=None):
    """Safely get attribute from result object, with fallback."""
    return getattr(obj, attr_name, default)


def _safe_get_selected_tools(result):
    """Safely extract selected_tools from result."""
    tools = _safe_get_attr(result, 'selected_tools', None)
    return tools if tools is not None else []


def _safe_get_reasoning_strategy(result):
    """Safely extract reasoning_strategy from result."""
    return _safe_get_attr(result, 'reasoning_strategy', 'unknown')


def _safe_get_metadata(result):
    """Safely extract metadata from result."""
    meta = _safe_get_attr(result, 'metadata', None)
    return meta if meta is not None else {}


def extract_conclusion_from_dict(
    data_dict: Dict[str, Any],
    _depth: int = 0
) -> Optional[Any]:
    """
    Extract conclusion from a dictionary with multiple fallback keys.

    Enhanced extraction logic with recursive support and depth limiting.

    Args:
        data_dict: Dictionary that may contain conclusion data
        _depth: Internal parameter for recursion depth tracking

    Returns:
        Conclusion value if found and valid, None otherwise
    """
    if not isinstance(data_dict, dict):
        return None

    if _depth >= _MAX_CONCLUSION_EXTRACTION_DEPTH:
        logger.warning(
            f"[AgentPool] Max conclusion extraction depth ({_MAX_CONCLUSION_EXTRACTION_DEPTH}) "
            f"reached - possible circular reference"
        )
        return None

    for key in _CONCLUSION_EXTRACTION_KEYS:
        value = data_dict.get(key)

        if value is None:
            continue
        if isinstance(value, str):
            if not value.strip() or value.strip().lower() == "none":
                continue

        if isinstance(value, dict):
            nested_conclusion = extract_conclusion_from_dict(value, _depth=_depth + 1)
            if nested_conclusion is not None:
                return nested_conclusion
            if len(value) > 0:
                return value

        return value

    return None


def is_valid_conclusion(conclusion: Any) -> bool:
    """
    Check if a conclusion is valid (not None or string "None").

    Args:
        conclusion: Conclusion value to check

    Returns:
        True if conclusion is valid, False otherwise
    """
    if conclusion is None:
        return False

    if isinstance(conclusion, str):
        stripped = conclusion.strip()
        if not stripped:
            return False
        if stripped.lower() == "none":
            return False
        return True

    if isinstance(conclusion, (dict, list, tuple)):
        return len(conclusion) > 0

    if hasattr(conclusion, 'conclusion'):
        inner = getattr(conclusion, 'conclusion', None)
        return is_valid_conclusion(inner)

    return True


def _extract_reasoning_output(manager, reasoning_result, task_id):
    """Extract reasoning output from reasoning_result object."""
    conclusion = None
    confidence = None
    reasoning_type_str = "unknown"
    explanation = None

    if hasattr(reasoning_result, "conclusion"):
        conclusion = getattr(reasoning_result, "conclusion", None)
        confidence = getattr(reasoning_result, "confidence", None)
        reasoning_type_obj = getattr(reasoning_result, "reasoning_type", None)
        reasoning_type_str = str(reasoning_type_obj) if reasoning_type_obj else "unknown"
        explanation = getattr(reasoning_result, "explanation", None)

        if hasattr(reasoning_result, "metadata") and reasoning_result.metadata:
            metadata_conclusion = extract_conclusion_from_dict(reasoning_result.metadata)
            if metadata_conclusion and not is_valid_conclusion(conclusion):
                conclusion = metadata_conclusion

            if explanation is None:
                explanation = reasoning_result.metadata.get("explanation")

    elif isinstance(reasoning_result, dict):
        conclusion = extract_conclusion_from_dict(reasoning_result)
        confidence = reasoning_result.get("confidence")
        reasoning_type_str = str(reasoning_result.get("reasoning_type", "unknown"))
        explanation = reasoning_result.get("explanation")

        if not is_valid_conclusion(conclusion):
            meta = reasoning_result.get("metadata", {})
            metadata_conclusion = extract_conclusion_from_dict(meta)
            if metadata_conclusion:
                conclusion = metadata_conclusion

    if is_valid_conclusion(conclusion):
        conclusion_preview = str(conclusion)[:100]
    else:
        conclusion_preview = "<no conclusion extracted>"

    logger.info(
        f"[AgentPool] Reasoning output extracted: "
        f"has_conclusion={is_valid_conclusion(conclusion)}, "
        f"conclusion_preview='{conclusion_preview}', "
        f"confidence={confidence}, "
        f"type={reasoning_type_str}"
    )

    # Warn if high confidence but no valid conclusion
    if confidence is not None and confidence >= 0.5 and not is_valid_conclusion(conclusion):
        logger.warning(
            f"[AgentPool] BUG DETECTED: Reasoning has high confidence ({confidence:.2f}) "
            f"but conclusion is None or 'None' string! task={task_id}"
        )

    return {
        "conclusion": conclusion,
        "confidence": confidence,
        "reasoning_type": reasoning_type_str,
        "explanation": explanation,
    }


def _finalize_task_result(
    manager, agent_id, task_id, graph_id, task_type, nodes,
    node_results, parameters, metadata, reasoning_was_invoked,
    selected_tools, reasoning_result, provenance, start_time,
):
    """Build final result dict, update metadata/provenance/stats, and return result."""
    duration = time.time() - start_time
    result = {
        "status": "completed", "outcome": "success", "agent_id": agent_id,
        "task_id": task_id, "graph_id": graph_id, "task_type": task_type,
        "execution_time": duration, "timestamp": time.time(),
        "nodes_processed": len(nodes), "node_results": node_results,
        "parameters_used": list(parameters.keys()) if parameters else [],
        "capability": metadata.capability.value,
        "reasoning_invoked": reasoning_was_invoked,
        "selected_tools": selected_tools if selected_tools else [],
    }

    if reasoning_result is not None:
        result["reasoning_output"] = _extract_reasoning_output(manager, reasoning_result, task_id)

    metadata.record_task_completion(success=True, duration_s=duration)
    if provenance:
        provenance.complete("success", result=result)
        rc = {"cpu_seconds": duration}
        if PSUTIL_AVAILABLE:
            try:
                rc["memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
            except Exception:
                pass
        provenance.update_resource_consumption(rc)

    with manager.stats_lock:
        manager.stats["total_jobs_completed"] += 1
        current_stats = dict(manager.stats)

    logger.info(
        f"[AgentPool] Job completed: task={task_id}, agent={agent_id}. "
        f"Stats: submitted={current_stats['total_jobs_submitted']}, "
        f"completed={current_stats['total_jobs_completed']}, "
        f"failed={current_stats['total_jobs_failed']}"
    )
    manager._persist_state_to_redis()
    logger.info(f"Agent {agent_id} completed task {task_id} in {duration:.3f}s")
    return result


def _handle_task_failure(manager, agent_id, task_id, metadata, provenance, start_time, error):
    """Record failure in metadata/provenance/stats and re-raise."""
    duration = time.time() - start_time
    logger.error(f"Agent {agent_id} task {task_id} failed: {error}")
    metadata.record_task_completion(success=False, duration_s=duration)
    metadata.record_error(error, {"task_id": task_id, "phase": "execution"})
    if provenance:
        provenance.complete("failed", error=str(error))
        provenance.update_resource_consumption({"cpu_seconds": duration})
    with manager.stats_lock:
        manager.stats["total_jobs_failed"] += 1
    manager._persist_state_to_redis()
