# VULCAN-AGI Orchestrator - Fallback reasoning + high-confidence result handling
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_pool_types import REASONING_IMPORT_PATHS, TOOL_SELECTION_PRIORITY_ORDER
from .task_exec_output import _safe_get_attr, extract_conclusion_from_dict, is_valid_conclusion

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


def _handle_high_confidence_result(
    manager, integration_result, result_selected_tools,
    result_reasoning_strategy, result_metadata, nodes, task_id,
    selected_tools, is_world_model_result, context
):
    """Handle high-confidence results from any engine."""
    primary_engine = result_selected_tools[0] if result_selected_tools else "general"

    logger.info(
        f"[AgentPool] High-confidence result from '{primary_engine}' engine "
        f"(confidence={integration_result.confidence:.2f}). Using directly."
    )

    # Learning integration
    try:
        from vulcan.reasoning import observe_engine_result
        query_id = context.get("conversation_id", f"query_{task_id}")
        result_dict = {
            "confidence": integration_result.confidence,
            "selected_tools": result_selected_tools,
            "strategy": result_reasoning_strategy,
        }
        observe_engine_result(
            query_id=query_id,
            engine_name=primary_engine,
            result=result_dict,
            success=True,
            execution_time_ms=0.0
        )
    except Exception:
        pass

    # Content preservation for WorldModel
    is_introspection = integration_result.metadata.get("is_introspection", False)
    if is_world_model_result or is_introspection or 'world_model' in selected_tools:
        integration_result.metadata['preserve_content'] = True
        integration_result.metadata['no_openai_replacement'] = True

    # Convert to reasoning_result format
    conclusion = _safe_get_attr(integration_result, "conclusion", None)
    if not is_valid_conclusion(conclusion):
        conclusion = extract_conclusion_from_dict(
            getattr(integration_result, 'metadata', {})
        )
    if not is_valid_conclusion(conclusion):
        conclusion = _safe_get_attr(integration_result, "rationale", "")

    explanation = result_metadata.get("explanation") or _safe_get_attr(integration_result, 'rationale', '')
    try:
        from vulcan.reasoning.reasoning_types import (
            ReasoningResult as UR_RR, ReasoningType as RT_Local,
        )
        if is_world_model_result:
            sel_rt, source_name = RT_Local.HYBRID, "world_model"
        else:
            _rt_map = {
                'symbolic': RT_Local.SYMBOLIC, 'probabilistic': RT_Local.PROBABILISTIC,
                'causal': RT_Local.CAUSAL, 'analogical': RT_Local.ANALOGICAL,
                'mathematical': RT_Local.MATHEMATICAL, 'philosophical': RT_Local.PHILOSOPHICAL,
                'multimodal': RT_Local.MULTIMODAL,
            }
            sel_rt = _rt_map.get(primary_engine.lower(), RT_Local.HYBRID)
            source_name = primary_engine
        meta = {
            "source": source_name, "selected_tools": result_selected_tools,
            "strategy": result_reasoning_strategy,
            "self_referential": result_metadata.get("self_referential", False),
            "preserve_content": result_metadata.get("preserve_content", False),
            "no_openai_replacement": result_metadata.get("no_openai_replacement", False),
            "is_introspection": result_metadata.get("is_introspection", False),
            "high_confidence_direct_use": True,
        }
        reasoning_result = UR_RR(
            conclusion=conclusion, confidence=integration_result.confidence,
            reasoning_type=sel_rt, explanation=explanation, metadata=meta,
        )
    except ImportError:
        class _HCRR:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        rt_string = 'hybrid' if is_world_model_result else primary_engine.lower()
        reasoning_result = _HCRR(
            conclusion=conclusion, confidence=integration_result.confidence,
            reasoning_type=rt_string, explanation=explanation,
            metadata={"source": primary_engine, "high_confidence_direct_use": True},
        )

    node_results = {}
    for i, node in enumerate(nodes):
        node_id = node.get("id", f"node_{i}")
        node_type = node.get("type", "unknown")
        node_results[node_id] = {
            "status": "completed",
            "node_type": node_type,
            "reasoning_applied": True,
            "selected_tools": result_selected_tools,
            "reasoning_strategy": result_reasoning_strategy,
        }

    return reasoning_result, node_results, True


def _execute_fallback_reasoning(
    manager, agent_id, task_id, task_type, parameters, selected_tools,
    nodes, node_results, provenance, start_time, metadata
):
    """Execute fallback reasoning when selected_tools present but not yet invoked."""
    logger.info(f"[REASONING] INVOKING engines for task {task_id}, tools={selected_tools}")
    try:
        ReasoningType_local = None
        ReasoningStrategy_local = None

        for path_prefix in REASONING_IMPORT_PATHS:
            try:
                rt_module = __import__(f'{path_prefix}.reasoning.reasoning_types', fromlist=['ReasoningType'])
                ReasoningType_local = getattr(rt_module, 'ReasoningType', None)
                ReasoningStrategy_local = getattr(rt_module, 'ReasoningStrategy', None)
                if ReasoningType_local and ReasoningStrategy_local:
                    break
            except ImportError:
                continue

        # Get singleton UnifiedReasoner
        reasoning = None
        for path_prefix in REASONING_IMPORT_PATHS:
            try:
                singleton_module = __import__(f'{path_prefix}.reasoning.singletons', fromlist=['get_unified_reasoner'])
                get_unified_reasoner_func = getattr(singleton_module, 'get_unified_reasoner')
                reasoning = get_unified_reasoner_func()
                if reasoning is not None:
                    break
            except ImportError:
                continue

        if reasoning is None:
            for path_prefix in REASONING_IMPORT_PATHS:
                try:
                    ur_module = __import__(f'{path_prefix}.reasoning.unified', fromlist=['UnifiedReasoner'])
                    DirectUnifiedReasoner = getattr(ur_module, 'UnifiedReasoner')
                    reasoning = DirectUnifiedReasoner()
                    break
                except ImportError:
                    continue
            if reasoning is None:
                raise ImportError(f"Could not import UnifiedReasoner from any of: {REASONING_IMPORT_PATHS}")

        query_text = parameters.get("prompt", "") or parameters.get("query", "")

        reasoning_type = None
        if selected_tools and len(selected_tools) > 0 and ReasoningType_local is not None:
            tool_name = selected_tools[0].upper()
            try:
                reasoning_type = ReasoningType_local[tool_name]
            except KeyError:
                for rt in ReasoningType_local:
                    if rt.value == selected_tools[0].lower():
                        reasoning_type = rt
                        break

        strategy = None
        if ReasoningStrategy_local is not None:
            strategy = ReasoningStrategy_local.ADAPTIVE
            if selected_tools and len(selected_tools) > 1:
                strategy = ReasoningStrategy_local.ENSEMBLE

        reasoning_result = reasoning.reason(
            input_data=query_text,
            query=parameters,
            reasoning_type=reasoning_type,
            strategy=strategy
        )

        # Build node results
        local_node_results = {}
        for i, node in enumerate(nodes):
            node_id = node.get("id", f"node_{i}")
            node_type = node.get("type", "unknown")
            local_node_results[node_id] = {
                "status": "completed",
                "node_type": node_type,
                "reasoning_applied": True,
                "selected_tools": selected_tools,
            }

        duration = time.time() - start_time

        reasoning_output_dict = {
            "conclusion": getattr(reasoning_result, "conclusion", None),
            "confidence": getattr(reasoning_result, "confidence", None),
            "reasoning_type": str(getattr(reasoning_result, "reasoning_type", "unknown")),
            "explanation": getattr(reasoning_result, "explanation", None),
        }

        result = {
            "status": "completed",
            "reasoning_invoked": True,
            "reasoning_output": reasoning_output_dict,
            "tools_used": selected_tools,
            "execution_time": duration,
            "agent_id": agent_id,
            "task_id": task_id,
            "nodes_processed": len(nodes),
            "node_results": local_node_results,
        }

        metadata.record_task_completion(success=True, duration_s=duration)

        if provenance:
            provenance.complete("success", result=result)
            resource_consumption = {"cpu_seconds": duration}
            if PSUTIL_AVAILABLE:
                try:
                    resource_consumption["memory_mb"] = (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )
                except Exception:
                    pass
            provenance.update_resource_consumption(resource_consumption)

        with manager.stats_lock:
            manager.stats["total_jobs_completed"] += 1

        manager._persist_state_to_redis()

        return result
    except Exception as e:
        logger.error(f"[REASONING] Invocation FAILED: {e}", exc_info=True)
        return None
