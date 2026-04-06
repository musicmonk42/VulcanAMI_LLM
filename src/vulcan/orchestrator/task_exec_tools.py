# VULCAN-AGI Orchestrator - Tool integration, selection updates, reasoning type resolution
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_pool_types import TOOL_SELECTION_PRIORITY_ORDER

logger = logging.getLogger(__name__)

try:
    from vulcan.reasoning import ensure_reasoning_type_enum
    TYPE_CONVERSION_AVAILABLE = True
except ImportError:
    ensure_reasoning_type_enum = None
    TYPE_CONVERSION_AVAILABLE = False


def _extract_router_command(task, graph, parameters, selected_tools, is_reasoning_task, task_id):
    """Extract Router's reasoning_type and tool_name from task/graph/parameters."""
    router_reasoning_type = None
    router_tool_name = None

    if task is not None:
        task_params = task.get("parameters", {}) if isinstance(task, dict) else {}
        if task_params:
            if not router_reasoning_type and "reasoning_type" in task_params:
                router_reasoning_type = task_params.get("reasoning_type")
            if not router_tool_name:
                router_tool_name = task_params.get("tool_name") or task_params.get("selected_tool")

        if hasattr(task, 'reasoning_type') and task.reasoning_type and not router_reasoning_type:
            router_reasoning_type = task.reasoning_type
        if hasattr(task, 'tool_name') and task.tool_name and not router_tool_name:
            router_tool_name = task.tool_name

    if not router_reasoning_type or not router_tool_name:
        if isinstance(graph, dict):
            if not router_reasoning_type:
                router_reasoning_type = graph.get("reasoning_type")
            if not router_tool_name:
                router_tool_name = graph.get("tool_name")

    if not router_reasoning_type:
        router_reasoning_type = parameters.get("reasoning_type")
    if not router_tool_name:
        router_tool_name = parameters.get("tool_name")

    selected_tools_lower = [t.lower() for t in selected_tools] if selected_tools else []
    if not router_tool_name and selected_tools_lower:
        for priority_tool in TOOL_SELECTION_PRIORITY_ORDER:
            if priority_tool in selected_tools_lower:
                router_tool_name = priority_tool
                break
        if not router_tool_name:
            router_tool_name = selected_tools[0] if selected_tools else None

    if is_reasoning_task and not (router_reasoning_type and router_tool_name):
        if 'routing_metadata' not in parameters:
            parameters['routing_metadata'] = {}
        parameters['routing_metadata']['inferred_from_selected_tools'] = True
        parameters['routing_metadata']['routing_method'] = 'tool_inference'

    return router_reasoning_type, router_tool_name


def _update_selected_tools_from_integration(
    integration_result, result_selected_tools, selected_tools, task_id
):
    """Update selected_tools based on integration result (mutates selected_tools in-place via reference)."""
    if not (hasattr(integration_result, 'selected_tools') and result_selected_tools):
        return

    should_override = (
        (hasattr(integration_result, 'override_router_tools') and
         integration_result.override_router_tools) or
        (hasattr(integration_result, 'metadata') and
         integration_result.metadata and
         integration_result.metadata.get('classifier_is_authoritative', False)) or
        (hasattr(integration_result, 'metadata') and
         integration_result.metadata and
         integration_result.metadata.get('is_self_introspection', False))
    )

    is_general_fallback = (
        result_selected_tools == ['general'] and
        selected_tools and
        selected_tools != ['general'] and
        not should_override
    )

    if is_general_fallback:
        logger.info(
            f"[AgentPool] PRESERVING router tools '{selected_tools}' - "
            f"integration returned ['general'] as fallback"
        )
    elif should_override:
        old_tools = selected_tools[:]
        selected_tools.clear()
        selected_tools.extend(result_selected_tools)
        logger.info(
            f"[AgentPool] AUTHORITATIVE override: Updated selected_tools from "
            f"'{old_tools}' to '{selected_tools}'"
        )


def _resolve_reasoning_type(
    router_reasoning_type, selected_tools, task_type, manager, task_id, ReasoningType,
):
    """Resolve reasoning type from router instruction, selected tools, or task type."""
    reasoning_type = None

    # PRIORITY 0: Router's explicit instruction
    if router_reasoning_type and isinstance(router_reasoning_type, str) and ReasoningType is not None:
        reasoning_type_map = {
            "cryptographic": ReasoningType.SYMBOLIC,
            "mathematical": ReasoningType.MATHEMATICAL,
            "philosophical": ReasoningType.PHILOSOPHICAL,
            "symbolic": ReasoningType.SYMBOLIC,
            "probabilistic": ReasoningType.PROBABILISTIC,
            "causal": ReasoningType.CAUSAL,
            "analogical": ReasoningType.ANALOGICAL,
            "multimodal": ReasoningType.MULTIMODAL,
            "hybrid": ReasoningType.HYBRID,
            "general": ReasoningType.SYMBOLIC,
        }
        reasoning_type = reasoning_type_map.get(router_reasoning_type.lower(), ReasoningType.HYBRID)
        logger.info(
            f"[AgentPool] Task {task_id}: EXECUTING ROUTER INSTRUCTION: "
            f"reasoning_type={reasoning_type} (from router={router_reasoning_type})"
        )

    # PRIORITY 1: Check selected_tools
    elif selected_tools and ReasoningType is not None:
        selected_tools_lower = [t.lower() for t in selected_tools]
        tool_to_reasoning_type = {
            'symbolic': ReasoningType.SYMBOLIC,
            'probabilistic': ReasoningType.PROBABILISTIC,
            'causal': ReasoningType.CAUSAL,
            'analogical': ReasoningType.ANALOGICAL,
            'mathematical': ReasoningType.MATHEMATICAL,
            'philosophical': ReasoningType.PHILOSOPHICAL,
            'world_model': ReasoningType.PHILOSOPHICAL,
            'general': ReasoningType.SYMBOLIC,
            'multimodal': ReasoningType.MULTIMODAL,
            'cryptographic': ReasoningType.SYMBOLIC,
        }

        primary_tool = None
        for priority_tool in TOOL_SELECTION_PRIORITY_ORDER:
            if priority_tool in selected_tools_lower:
                primary_tool = priority_tool
                break

        if not primary_tool and selected_tools:
            primary_tool = selected_tools[0].lower()

        if primary_tool:
            mapped_reasoning_type = tool_to_reasoning_type.get(primary_tool)
            if mapped_reasoning_type:
                reasoning_type = mapped_reasoning_type
                logger.info(
                    f"[AgentPool] Task {task_id}: Using reasoning type {reasoning_type} "
                    f"from selected_tools={selected_tools}"
                )

    # PRIORITY 2: Fall back to task_type mapping
    if reasoning_type is None:
        reasoning_type = manager._map_task_to_reasoning_type(task_type)

    return reasoning_type


def _resolve_tool_reasoning_type(result_selected_tools, default_reasoning_type, ReasoningType):
    """Resolve reasoning type from integration result's selected tools."""
    selected_tool_reasoning_type = default_reasoning_type

    if result_selected_tools and ReasoningType is not None:
        tool_to_reasoning_type_map = {
            'symbolic': ReasoningType.SYMBOLIC,
            'probabilistic': ReasoningType.PROBABILISTIC,
            'causal': ReasoningType.CAUSAL,
            'analogical': ReasoningType.ANALOGICAL,
            'mathematical': ReasoningType.MATHEMATICAL,
            'philosophical': ReasoningType.PHILOSOPHICAL,
            'world_model': ReasoningType.PHILOSOPHICAL,
            'general': ReasoningType.SYMBOLIC,
            'multimodal': ReasoningType.MULTIMODAL,
            'cryptographic': ReasoningType.SYMBOLIC,
        }

        primary_tool = None
        for priority_tool in TOOL_SELECTION_PRIORITY_ORDER:
            if priority_tool in [t.lower() for t in result_selected_tools]:
                primary_tool = priority_tool
                break

        if not primary_tool and result_selected_tools:
            primary_tool = result_selected_tools[0].lower()

        if primary_tool:
            mapped_type = tool_to_reasoning_type_map.get(primary_tool)
            if mapped_type is not None:
                selected_tool_reasoning_type = mapped_type

    return selected_tool_reasoning_type


def ensure_type_conversion(obj, source_label: str):
    """Apply reasoning type enum conversion if available."""
    if TYPE_CONVERSION_AVAILABLE and ensure_reasoning_type_enum is not None:
        return ensure_reasoning_type_enum(obj, source_label)
    return obj
