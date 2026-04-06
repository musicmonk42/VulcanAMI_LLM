# VULCAN-AGI Orchestrator - Reasoning task detection, query extraction, privileged results
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_pool_types import MIN_REASONING_QUERY_LENGTH, LONG_QUERY_REASONING_THRESHOLD
from .task_exec_output import _safe_get_attr, extract_conclusion_from_dict, is_valid_conclusion

logger = logging.getLogger(__name__)

REASONING_TASK_TYPES = {
    "reasoning", "causal", "symbolic", "analogical", "probabilistic",
    "counterfactual", "multimodal", "deductive", "inductive", "abductive",
    "philosophical", "mathematical", "hybrid",
    "self_introspection", "meta_reasoning", "world_model",
}


def _determine_reasoning_task(
    normalized_task_type, task_type, parameters, graph, metadata,
    selected_tools, manager=None,
):
    """Determine if this is a reasoning task. Returns (is_reasoning_task, selected_tools)."""
    from .agent_lifecycle import AgentCapability

    is_reasoning_task = normalized_task_type in REASONING_TASK_TYPES

    # Check selected_tools from QueryRouter
    if not is_reasoning_task and selected_tools:
        if any(tool in REASONING_TASK_TYPES for tool in selected_tools):
            is_reasoning_task = True
            logger.info(
                f"Reasoning triggered by selected_tools={selected_tools}"
            )

    if metadata.capability == AgentCapability.REASONING:
        is_reasoning_task = True

    query_text = parameters.get("query") or parameters.get("prompt") or ""
    is_creative_task = (
        normalized_task_type == "creative" or task_type == "creative_task"
        or "creative" in (graph.get("detected_patterns", []) or [])
    )

    if isinstance(query_text, str) and not is_reasoning_task and not is_creative_task:
        query_lower = query_text.lower()
        introspection_kws = [
            "would you", "do you", "are you", "can you", "self-aware",
            "self aware", "consciousness", "sentient", "would vulcan",
            "your capabilities", "your limitations",
        ]
        if any(kw in query_lower for kw in introspection_kws):
            is_reasoning_task = True
            if not selected_tools or selected_tools == ["general"]:
                selected_tools = ["world_model"]
            logger.info(f"Self-introspection detected, tools={selected_tools}")
    elif is_creative_task:
        complexity = graph.get("complexity", 0.0)
        if not complexity and graph and manager is not None:
            try:
                complexity = manager._calculate_task_complexity(graph, parameters)
            except (AttributeError, KeyError, TypeError):
                complexity = 0.5
        complexity = complexity or 0.5
        if complexity > 0.5:
            is_reasoning_task = True
            logger.info(f"Complex creative task (complexity={complexity:.2f}), invoking reasoning")

    if not is_reasoning_task and parameters.get("is_philosophical"):
        is_reasoning_task = True
        selected_tools = ["symbolic", "causal"]
        logger.info(f"Philosophical query detected, tools={selected_tools}")

    return is_reasoning_task, selected_tools


def _extract_query_and_context(parameters, nodes, is_reasoning_task, selected_tools, task_id):
    """Extract query/input_data/context; handle long query detection and truncation checks."""
    query = next(
        (v for v in [
            parameters.get("prompt"),
            parameters.get("query"),
            parameters.get("original_prompt"),
            parameters.get("user_query")
        ] if v),
        ""
    )
    input_data = parameters.get("input_data") or parameters.get("input", "")
    context = parameters.get("context", {})

    # Propagate gate/auth flags into context
    if "skip_gate_checks" in parameters:
        context["skip_gate_checks"] = context["skip_gate_check"] = parameters["skip_gate_checks"]
    for key in ("llm_authoritative", "router_confidence", "llm_classification"):
        if key in parameters:
            context[key] = parameters[key]

    reasoning_context = parameters.get("reasoning_context", {})
    if reasoning_context:
        context = {**context, **reasoning_context}
    if selected_tools and 'router_tools' not in context:
        context['router_tools'] = selected_tools

    if not query and nodes:
        for node in nodes:
            if node.get("type") in ("input", "query", "InputNode"):
                query = node.get("data", {}).get("value", "") or node.get("params", {}).get("query", "")
                input_data = input_data or node.get("data", {}).get("input", "")
                break

    query_len = len(query) if query else 0
    if not is_reasoning_task and query_len > LONG_QUERY_REASONING_THRESHOLD:
        is_reasoning_task = True
        if not selected_tools or selected_tools == ["general"]:
            selected_tools = ["hybrid"]
        logger.info(f"[AgentPool] task {task_id}: long query ({query_len} chars), tools={selected_tools}")

    if query_len < MIN_REASONING_QUERY_LENGTH and is_reasoning_task:
        qtc = query or ""
        ends_mid = len(qtc) > 10 and qtc[-1].isalnum() and " " not in qtc[-10:]
        if qtc.endswith("...") or qtc.endswith("-") or ends_mid:
            logger.warning(f"[AgentPool] Potentially truncated query: {query_len} chars")

    return query, input_data, context, selected_tools, is_reasoning_task


def _handle_privileged_result(
    manager, integration_result, result_selected_tools,
    result_reasoning_strategy, result_metadata, nodes, task_id
):
    """Handle privileged results (world_model, meta-reasoning, etc.)."""
    privileged_type = "UNKNOWN"

    if 'world_model' in result_selected_tools:
        privileged_type = "world_model"
    elif result_reasoning_strategy in ('meta_reasoning', 'philosophical_reasoning'):
        privileged_type = result_reasoning_strategy
    elif result_metadata.get('is_self_introspection'):
        privileged_type = "self_introspection"
    elif result_metadata.get('self_referential'):
        privileged_type = "self_referential"

    logger.info(
        f"[AgentPool] PRIVILEGED RESULT DETECTED: {privileged_type} - "
        f"IMMEDIATELY returning without fallback/consensus/blending."
    )

    if not integration_result.metadata:
        integration_result.metadata = {}
    integration_result.metadata['privileged_result'] = True
    integration_result.metadata['privileged_type'] = privileged_type
    integration_result.metadata['bypassed_fallback'] = True

    conclusion = _safe_get_attr(integration_result, "conclusion", None)
    if not is_valid_conclusion(conclusion):
        conclusion = extract_conclusion_from_dict(
            getattr(integration_result, 'metadata', {})
        )
    if not is_valid_conclusion(conclusion):
        conclusion = _safe_get_attr(integration_result, "rationale", "")

    explanation = result_metadata.get("explanation") or _safe_get_attr(integration_result, 'rationale', '')
    meta = {**result_metadata, "source": "privileged_path", "selected_tools": result_selected_tools,
            "strategy": result_reasoning_strategy}
    try:
        from vulcan.reasoning.reasoning_types import ReasoningResult as UR_ReasoningResult
        reasoning_result = UR_ReasoningResult(
            conclusion=conclusion, confidence=integration_result.confidence,
            reasoning_type=result_reasoning_strategy, explanation=explanation, metadata=meta,
        )
    except ImportError:
        class _PRR:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        reasoning_result = _PRR(
            conclusion=conclusion, confidence=integration_result.confidence,
            reasoning_type=result_reasoning_strategy, explanation=explanation,
            metadata={**result_metadata, "source": "privileged_path"},
        )

    node_results = {}
    for i, node in enumerate(nodes):
        node_id = node.get("id", f"node_{i}")
        node_type = node.get("type", "unknown")
        node_results[node_id] = {
            "status": "completed",
            "node_type": node_type,
            "reasoning_applied": True,
            "privileged_result": True,
            "selected_tools": result_selected_tools,
            "reasoning_strategy": result_reasoning_strategy,
        }

    return reasoning_result, node_results, True
