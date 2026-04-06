# VULCAN-AGI Orchestrator - Task Execution Core (entry point + dispatch)
import logging
import time
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentMetadata
from .agent_pool_types import (
    REASONING_TOOL_NAMES, MIN_REASONING_QUERY_LENGTH,
    HIGH_CONFIDENCE_THRESHOLD, WORLD_MODEL_CONFIDENCE_THRESHOLD,
    is_privileged_result as _is_privileged_result,
)
# Re-export public API from sub-modules for backward compatibility
from .task_exec_output import (  # noqa: F401
    extract_conclusion_from_dict, is_valid_conclusion, _extract_reasoning_output,
    _safe_get_selected_tools, _safe_get_reasoning_strategy, _safe_get_metadata,
    _finalize_task_result, _handle_task_failure,
)
from .task_exec_reasoning import (
    _determine_reasoning_task, _extract_query_and_context, _handle_privileged_result,
)
from .task_exec_fallback import _handle_high_confidence_result, _execute_fallback_reasoning
from .task_exec_tools import (
    _extract_router_command, _update_selected_tools_from_integration,
    _resolve_reasoning_type, _resolve_tool_reasoning_type, ensure_type_conversion,
)

logger = logging.getLogger(__name__)


def execute_agent_task(
    manager: "AgentPoolManager",
    agent_id: str,
    task: Dict[str, Any],
    metadata: AgentMetadata
) -> Any:
    """Execute task on agent with reasoning engine invocation."""
    from . import agent_pool as _ap
    _ap._lazy_import_reasoning()
    REASONING_AVAILABLE = _ap.REASONING_AVAILABLE
    UnifiedReasoner = _ap.UnifiedReasoner
    ReasoningType = _ap.ReasoningType
    create_unified_reasoner = _ap.create_unified_reasoner
    apply_reasoning = _ap.apply_reasoning

    start_time = time.time()
    task_id = task.get("task_id")
    provenance = task.get("provenance")
    graph = task.get("graph", {})
    parameters = task.get("parameters", {})

    try:
        logger.info(f"Agent {agent_id} executing task {task_id}")
        graph_id = graph.get("id", "unknown")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        task_type = graph.get("type", "general").lower()

        # Normalize task_type
        normalized_task_type = task_type
        for suffix in ("_task", "_support"):
            if normalized_task_type.endswith(suffix):
                normalized_task_type = normalized_task_type[:-len(suffix)]
                break

        selected_tools = parameters.get("selected_tools", []) or parameters.get("tools", []) or []

        # Phase 1: Determine if reasoning task
        is_reasoning_task, selected_tools = _determine_reasoning_task(
            normalized_task_type, task_type, parameters, graph, metadata, selected_tools, manager
        )

        # Phase 2: Extract query, context
        query, input_data, context, selected_tools, is_reasoning_task = _extract_query_and_context(
            parameters, nodes, is_reasoning_task, selected_tools, task_id
        )
        query_len = len(query) if query else 0

        # Phase 3: Reasoning execution
        reasoning_result = None
        node_results = {}
        reasoning_was_invoked = False

        if is_reasoning_task:
            logger.info(
                f"[AgentPool] Task {task_id} starting reasoning: "
                f"query_len={query_len}, tools={selected_tools}, type={task_type}"
            )

        # Phase 4: Extract router command
        router_reasoning_type, router_tool_name = _extract_router_command(
            task, graph, parameters, selected_tools, is_reasoning_task, task_id
        )

        _ap._lazy_import_reasoning()

        if is_reasoning_task and REASONING_AVAILABLE:
            logger.info(
                f"Agent {agent_id} invoking reasoning for task {task_id} "
                f"(ROUTER: type={router_reasoning_type}, tool={router_tool_name})"
            )
            if query_len < MIN_REASONING_QUERY_LENGTH:
                logger.warning(f"[AgentPool] Task {task_id}: Query too short ({query_len} chars)")

            try:
                reasoning_type = _resolve_reasoning_type(
                    router_reasoning_type, selected_tools, task_type, manager, task_id, ReasoningType
                )
                complexity = manager._calculate_task_complexity(graph, parameters)
                skip_tool_selection = bool(selected_tools and selected_tools != ["general"])

                skip_gate_check = context.get('skip_gate_check', False) or context.get('llm_authoritative', False)
                router_confidence = context.get('router_confidence', 0.0) or context.get('llm_confidence', 0.0)
                if not skip_gate_check:
                    classifier_auth = context.get('classifier_is_authoritative', False)
                    classifier_conf = context.get('classifier_confidence', 0.0)
                    if classifier_auth and classifier_conf >= 0.8:
                        skip_gate_check = True
                        router_confidence = classifier_conf
                if skip_gate_check:
                    context['skip_gate_check'] = True
                    context['router_confidence'] = router_confidence
                    context['llm_classification'] = context.get('classifier_category', 'unknown')

                integration_result = apply_reasoning(
                    query=query or str(input_data) or f"Process task {task_id}",
                    query_type=task_type, complexity=complexity, context=context,
                    selected_tools=selected_tools if skip_tool_selection else None,
                    skip_tool_selection=skip_tool_selection,
                )

                result_selected_tools = _safe_get_selected_tools(integration_result)
                result_reasoning_strategy = _safe_get_reasoning_strategy(integration_result)
                result_metadata = _safe_get_metadata(integration_result)

                integration_result = ensure_type_conversion(integration_result, "agent_pool")
                if hasattr(integration_result, 'metadata') and integration_result.metadata:
                    integration_result.metadata = ensure_type_conversion(
                        integration_result.metadata, "agent_pool:metadata"
                    )

                best_result = integration_result
                best_confidence = integration_result.confidence
                best_source = result_selected_tools[0] if result_selected_tools else "unknown"

                if integration_result.confidence < 0.3:
                    logger.warning(f"[AgentPool] Task {task_id}: LOW CONFIDENCE ({integration_result.confidence:.2f})")

                # Privileged result check
                if _is_privileged_result(integration_result):
                    reasoning_result, node_results, reasoning_was_invoked = _handle_privileged_result(
                        manager, integration_result, result_selected_tools,
                        result_reasoning_strategy, result_metadata, nodes, task_id
                    )

                # High-confidence result check
                is_privileged = _is_privileged_result(integration_result)
                is_high_conf = integration_result.confidence >= HIGH_CONFIDENCE_THRESHOLD and not is_privileged
                is_wm = (result_selected_tools == ["world_model"] and
                         integration_result.confidence >= WORLD_MODEL_CONFIDENCE_THRESHOLD and not is_privileged)

                if is_high_conf or (is_wm and not is_privileged):
                    reasoning_result, node_results, reasoning_was_invoked = _handle_high_confidence_result(
                        manager, integration_result, result_selected_tools,
                        result_reasoning_strategy, result_metadata, nodes, task_id,
                        selected_tools, is_wm, context
                    )

                # UnifiedReasoner for non-high-confidence results
                elif UnifiedReasoner is not None and create_unified_reasoner is not None and not reasoning_was_invoked:
                    try:
                        reasoner = create_unified_reasoner(enable_learning=True, enable_safety=True)
                        if reasoner is not None:
                            sel_rt = _resolve_tool_reasoning_type(result_selected_tools, reasoning_type, ReasoningType)
                            reasoning_result = reasoner.reason(
                                input_data=input_data or query,
                                query={"query": query, "context": context, "task_type": task_type},
                                reasoning_type=sel_rt,
                            )
                            reasoning_result = ensure_type_conversion(reasoning_result, "agent_pool:reasoner")
                            rc = reasoning_result.get('confidence', 0.0) if isinstance(reasoning_result, dict) else getattr(reasoning_result, 'confidence', 0.0)
                            if rc > best_confidence:
                                best_confidence = rc
                                best_source = str(sel_rt) if sel_rt else "unified_reasoner"
                    except Exception as reasoning_error:
                        logger.warning(f"Agent {agent_id} UnifiedReasoner failed: {reasoning_error}")

                if not node_results:
                    for i, node in enumerate(nodes):
                        node_id = node.get("id", f"node_{i}")
                        node_results[node_id] = {
                            "status": "completed", "node_type": node.get("type", "unknown"),
                            "reasoning_applied": True, "selected_tools": result_selected_tools,
                            "reasoning_strategy": result_reasoning_strategy,
                        }

                reasoning_was_invoked = True
                _update_selected_tools_from_integration(integration_result, result_selected_tools, selected_tools, task_id)

                if hasattr(integration_result, 'metadata') and integration_result.metadata:
                    uqt = integration_result.metadata.get('query_type')
                    is_si = (integration_result.metadata.get('self_referential', False) or
                             integration_result.metadata.get('is_self_introspection', False))
                    if uqt and uqt != task_type:
                        task_type = uqt
                    if is_si and task_type != 'self_introspection':
                        task_type = 'self_introspection'

            except Exception as reasoning_error:
                logger.warning(f"Agent {agent_id} reasoning failed: {reasoning_error}. Falling back.")
                is_reasoning_task = False

        # Fallback reasoning
        should_fallback = (
            selected_tools and not node_results and not reasoning_was_invoked and
            (is_reasoning_task or any(t.lower() in REASONING_TOOL_NAMES for t in selected_tools))
        )
        if should_fallback:
            result = _execute_fallback_reasoning(
                manager, agent_id, task_id, task_type, parameters, selected_tools,
                nodes, node_results, provenance, start_time, metadata
            )
            if result is not None:
                return result

        # Graph-based execution fallback
        if not is_reasoning_task or not node_results:
            for node in nodes:
                nid = node.get("id", "unknown")
                node_results[nid] = {
                    "status": "completed", "node_type": node.get("type", "unknown"),
                    "params_processed": list(node.get("params", {}).keys()), "reasoning_applied": False,
                }

        return _finalize_task_result(
            manager, agent_id, task_id, graph_id, task_type, nodes,
            node_results, parameters, metadata, reasoning_was_invoked,
            selected_tools, reasoning_result, provenance, start_time,
        )

    except Exception as e:
        _handle_task_failure(manager, agent_id, task_id, metadata, provenance, start_time, e)
        raise
