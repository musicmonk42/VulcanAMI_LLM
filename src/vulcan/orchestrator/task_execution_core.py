# ============================================================
# VULCAN-AGI Orchestrator - Task Execution Core Module
# Extracted from agent_pool.py for modularity
# Contains the main _execute_agent_task logic and helpers
# ============================================================

import gc
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_pool import AgentPoolManager

from .agent_lifecycle import AgentCapability, AgentMetadata, AgentState
from .agent_pool_types import (
    REASONING_IMPORT_PATHS,
    REASONING_TOOL_NAMES,
    TOOL_SELECTION_PRIORITY_ORDER,
    MIN_REASONING_QUERY_LENGTH,
    LONG_QUERY_REASONING_THRESHOLD,
    WORLD_MODEL_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    CONCLUSION_EXTRACTION_KEYS as _CONCLUSION_EXTRACTION_KEYS,
    MAX_CONCLUSION_EXTRACTION_DEPTH as _MAX_CONCLUSION_EXTRACTION_DEPTH,
    is_privileged_result as _is_privileged_result,
)

# Lazy-loaded globals imported from parent module at call time
# These are set by _get_reasoning_globals() from agent_pool module state

logger = logging.getLogger(__name__)


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from vulcan.reasoning import ensure_reasoning_type_enum
    TYPE_CONVERSION_AVAILABLE = True
except ImportError:
    ensure_reasoning_type_enum = None
    TYPE_CONVERSION_AVAILABLE = False


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


def execute_agent_task(
    manager: "AgentPoolManager",
    agent_id: str,
    task: Dict[str, Any],
    metadata: AgentMetadata
) -> Any:
    """
    Execute task on agent with ACTUAL reasoning engine invocation.

    CRITICAL FIX: This function now properly invokes UnifiedReasoning.reason()
    for reasoning tasks instead of just creating placeholder results.

    Args:
        manager: AgentPoolManager instance
        agent_id: Agent identifier
        task: Task dictionary containing task_id, graph, parameters, provenance
        metadata: Agent metadata

    Returns:
        Task result dictionary with actual reasoning output

    Raises:
        Exception: If task execution fails
    """
    # Import reasoning globals from the parent module
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

        # Extract task information from graph
        graph_id = graph.get("id", "unknown")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        task_type = graph.get("type", "general").lower()

        # Normalize task_type by stripping common suffixes
        normalized_task_type = task_type
        for suffix in ("_task", "_support"):
            if normalized_task_type.endswith(suffix):
                normalized_task_type = normalized_task_type[:-len(suffix)]
                break

        # Determine if this is a reasoning task
        reasoning_task_types = {
            "reasoning", "causal", "symbolic", "analogical", "probabilistic",
            "counterfactual", "multimodal", "deductive", "inductive", "abductive",
            "philosophical", "mathematical", "hybrid",
            "self_introspection", "meta_reasoning", "world_model",
        }
        is_reasoning_task = normalized_task_type in reasoning_task_types

        # Check selected_tools from QueryRouter
        selected_tools = parameters.get("selected_tools", []) or parameters.get("tools", []) or []
        if not is_reasoning_task and selected_tools:
            if any(tool in reasoning_task_types for tool in selected_tools):
                is_reasoning_task = True
                logger.info(
                    f"Agent {agent_id} task {task_id}: reasoning triggered by selected_tools={selected_tools}"
                )

        if metadata.capability == AgentCapability.REASONING:
            is_reasoning_task = True

        # TASK 6 FIX: Check if query is self-introspection
        query_text = parameters.get("query") or parameters.get("prompt") or ""
        is_creative_task = (
            normalized_task_type == "creative" or
            task_type == "creative_task" or
            "creative" in (graph.get("detected_patterns", []) or [])
        )

        if isinstance(query_text, str) and not is_reasoning_task and not is_creative_task:
            query_lower = query_text.lower()
            self_introspection_keywords = [
                "would you", "do you", "are you", "can you",
                "self-aware", "self aware", "consciousness", "sentient",
                "would vulcan", "your capabilities", "your limitations",
            ]
            if any(kw in query_lower for kw in self_introspection_keywords):
                is_reasoning_task = True
                if not selected_tools or selected_tools == ["general"]:
                    selected_tools = ["world_model"]
                logger.info(
                    f"[AgentPool] Task {task_id}: Self-introspection query detected, "
                    f"forcing reasoning with tools={selected_tools}"
                )
        elif is_creative_task:
            complexity = graph.get("complexity", 0.0)
            if not complexity and graph:
                try:
                    complexity = manager._calculate_task_complexity(graph, parameters)
                except (AttributeError, KeyError, TypeError) as e:
                    logger.warning(
                        f"[AgentPool] Task {task_id}: Failed to calculate complexity "
                        f"for creative task: {e}. Using default complexity=0.5"
                    )
                    complexity = 0.5

            if complexity > 0.5:
                is_reasoning_task = True
                logger.info(
                    f"[AgentPool] Task {task_id}: Complex creative task detected "
                    f"(complexity={complexity:.2f}), invoking reasoning for context "
                    f"(keeping tools={selected_tools})"
                )
            else:
                logger.info(
                    f"[AgentPool] Task {task_id}: Simple creative task detected "
                    f"(complexity={complexity:.2f}), NOT forcing world_model "
                    f"(keeping tools={selected_tools})"
                )

        # FIX TASK 2: Invoke reasoning for philosophical queries
        if not is_reasoning_task and parameters.get("is_philosophical"):
            is_reasoning_task = True
            selected_tools = ["symbolic", "causal"]
            logger.info(
                f"[AgentPool] task {task_id}: philosophical query detected, "
                f"forcing reasoning with tools={selected_tools}"
            )

        # FIX TASK 1: Extract FULL query context
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

        # MULTI-LAYER GATE CHECK FIX: Extract flags from task parameters
        if "skip_gate_checks" in parameters:
            context["skip_gate_checks"] = parameters["skip_gate_checks"]
            context["skip_gate_check"] = parameters["skip_gate_checks"]
            logger.debug(f"[AgentPool] Extracted skip_gate_checks={parameters['skip_gate_checks']} from parameters")

        if "llm_authoritative" in parameters:
            context["llm_authoritative"] = parameters["llm_authoritative"]

        if "router_confidence" in parameters:
            context["router_confidence"] = parameters["router_confidence"]

        if "llm_classification" in parameters:
            context["llm_classification"] = parameters["llm_classification"]

        # Preserve full reasoning context
        reasoning_context = parameters.get("reasoning_context", {})
        if reasoning_context:
            context = {**context, **reasoning_context}

        # FIX #6: Ensure router_tools are in the context for ToolSelector
        if selected_tools and 'router_tools' not in context:
            context['router_tools'] = selected_tools

        # Try to extract query from nodes if not in parameters
        if not query and nodes:
            for node in nodes:
                if node.get("type") in ("input", "query", "InputNode"):
                    query = node.get("data", {}).get("value", "") or node.get("params", {}).get("query", "")
                    input_data = input_data or node.get("data", {}).get("input", "")
                    break

        # FIX TASK 2: Trigger reasoning for long complex queries
        query_len = len(query) if query else 0
        if not is_reasoning_task and query_len > LONG_QUERY_REASONING_THRESHOLD:
            is_reasoning_task = True
            if not selected_tools or selected_tools == ["general"]:
                selected_tools = ["hybrid"]
            logger.info(
                f"[AgentPool] task {task_id}: long query ({query_len} chars > {LONG_QUERY_REASONING_THRESHOLD}) detected, "
                f"forcing reasoning with tools={selected_tools}"
            )

        # FIX TASK 1: Check for query truncation
        if query_len < MIN_REASONING_QUERY_LENGTH and is_reasoning_task:
            query_text_check = query if query else ""
            ends_mid_word = False
            text_len = len(query_text_check)
            if text_len > 10:
                ends_mid_word = query_text_check[-1].isalnum() and " " not in query_text_check[-10:]
            appears_truncated = (
                query_text_check.endswith("...") or
                query_text_check.endswith("-") or
                ends_mid_word
            )
            if appears_truncated:
                logger.warning(
                    f"[AgentPool] Potentially truncated query for reasoning task: "
                    f"query_len={query_len} chars - query may be incomplete. "
                    f"Check parameters keys: {list(parameters.keys())}"
                )
            else:
                logger.debug(
                    f"[AgentPool] Short query ({query_len} chars) for reasoning task - "
                    f"this is acceptable for simple requests"
                )

        # ============================================================
        # REASONING TASK EXECUTION
        # ============================================================
        reasoning_result = None
        node_results = {}
        reasoning_was_invoked = False

        if is_reasoning_task:
            logger.info(
                f"[AgentPool] Task {task_id} starting reasoning invocation: "
                f"query_len={query_len}, tools={selected_tools}, "
                f"task_type={task_type}, agent={agent_id}"
            )

        # ===============================================================================
        # COMMAND PATTERN: Extract Router's instruction
        # ===============================================================================
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
            logger.debug(
                f"[AgentPool] Task {task_id}: Router provided selected_tools without explicit reasoning_type. "
                f"Will infer reasoning_type from selected_tools={selected_tools}."
            )
            if 'routing_metadata' not in parameters:
                parameters['routing_metadata'] = {}
            parameters['routing_metadata']['inferred_from_selected_tools'] = True
            parameters['routing_metadata']['routing_method'] = 'tool_inference'

        # Trigger lazy import of reasoning components
        _ap._lazy_import_reasoning()

        if is_reasoning_task and REASONING_AVAILABLE:
            logger.info(
                f"Agent {agent_id} invoking reasoning engine for task {task_id} "
                f"(type={task_type}, capability={metadata.capability.value}, "
                f"ROUTER INSTRUCTION: reasoning_type={router_reasoning_type}, tool={router_tool_name})"
            )

            if query_len < MIN_REASONING_QUERY_LENGTH:
                logger.warning(
                    f"[AgentPool] Task {task_id}: Query too short ({query_len} chars < {MIN_REASONING_QUERY_LENGTH}) - "
                    f"reasoning may produce poor results."
                )

            try:
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
                elif selected_tools_lower and ReasoningType is not None:
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
                        if priority_tool in [t.lower() for t in selected_tools]:
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

                # Calculate complexity
                complexity = manager._calculate_task_complexity(graph, parameters)

                # SINGLE AUTHORITY PATTERN
                skip_tool_selection = bool(selected_tools and selected_tools != ["general"])

                # Propagate skip_gate_check when LLM authoritative
                skip_gate_check = context.get('skip_gate_check', False) or context.get('llm_authoritative', False)
                router_confidence = context.get('router_confidence', 0.0) or context.get('llm_confidence', 0.0)

                if not skip_gate_check:
                    classifier_is_authoritative = context.get('classifier_is_authoritative', False)
                    classifier_confidence = context.get('classifier_confidence', 0.0)
                    if classifier_is_authoritative and classifier_confidence >= 0.8:
                        skip_gate_check = True
                        router_confidence = classifier_confidence

                if skip_gate_check:
                    context['skip_gate_check'] = True
                    context['router_confidence'] = router_confidence
                    context['llm_classification'] = context.get('classifier_category', 'unknown')

                logger.info(
                    f"[AgentPool] Calling apply_reasoning: query_len={query_len}, "
                    f"query_type={task_type}, complexity={complexity:.2f}, "
                    f"selected_tools={selected_tools}, skip_tool_selection={skip_tool_selection}, "
                    f"skip_gate_check={skip_gate_check}"
                )

                # Apply reasoning via the integration layer
                integration_result = apply_reasoning(
                    query=query or str(input_data) or f"Process task {task_id}",
                    query_type=task_type,
                    complexity=complexity,
                    context=context,
                    selected_tools=selected_tools if skip_tool_selection else None,
                    skip_tool_selection=skip_tool_selection,
                )

                result_selected_tools = _safe_get_selected_tools(integration_result)
                result_reasoning_strategy = _safe_get_reasoning_strategy(integration_result)
                result_metadata = _safe_get_metadata(integration_result)

                # BUG FIX: Convert reasoning_type to enum if needed
                if TYPE_CONVERSION_AVAILABLE:
                    integration_result = ensure_reasoning_type_enum(integration_result, "agent_pool")
                    if hasattr(integration_result, 'metadata') and integration_result.metadata:
                        integration_result.metadata = ensure_reasoning_type_enum(
                            integration_result.metadata,
                            "agent_pool:metadata"
                        )

                # Track best result across engine attempts
                best_result = integration_result
                best_confidence = integration_result.confidence
                best_source = result_selected_tools[0] if result_selected_tools else "unknown"

                logger.info(
                    f"Agent {agent_id} reasoning selection complete: "
                    f"tools={result_selected_tools}, "
                    f"strategy={result_reasoning_strategy}, "
                    f"confidence={integration_result.confidence:.2f}"
                )

                if integration_result.confidence < 0.3:
                    logger.warning(
                        f"[AgentPool] Task {task_id}: LOW CONFIDENCE result ({integration_result.confidence:.2f})."
                    )

                # PRIVILEGED RESULT CHECK
                if _is_privileged_result(integration_result):
                    reasoning_result, node_results, reasoning_was_invoked = _handle_privileged_result(
                        manager, integration_result, result_selected_tools,
                        result_reasoning_strategy, result_metadata, nodes, task_id
                    )

                # HIGH-CONFIDENCE RESULT CHECK
                is_privileged = _is_privileged_result(integration_result)
                is_high_confidence_result = (
                    integration_result.confidence >= HIGH_CONFIDENCE_THRESHOLD and
                    not is_privileged
                )
                is_world_model_result = (
                    result_selected_tools == ["world_model"] and
                    integration_result.confidence >= WORLD_MODEL_CONFIDENCE_THRESHOLD and
                    not is_privileged
                )

                if is_high_confidence_result or (is_world_model_result and not is_privileged):
                    reasoning_result, node_results, reasoning_was_invoked = _handle_high_confidence_result(
                        manager, integration_result, result_selected_tools,
                        result_reasoning_strategy, result_metadata, nodes, task_id,
                        selected_tools, is_world_model_result, context
                    )

                # UnifiedReasoner invocation for non-high-confidence results
                elif UnifiedReasoner is not None and create_unified_reasoner is not None and not reasoning_was_invoked:
                    try:
                        reasoner = create_unified_reasoner(
                            enable_learning=True,
                            enable_safety=True,
                        )

                        if reasoner is not None:
                            selected_tool_reasoning_type = reasoning_type

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

                            reasoning_result = reasoner.reason(
                                input_data=input_data or query,
                                query={"query": query, "context": context, "task_type": task_type},
                                reasoning_type=selected_tool_reasoning_type,
                            )

                            if TYPE_CONVERSION_AVAILABLE:
                                reasoning_result = ensure_reasoning_type_enum(reasoning_result, "agent_pool:reasoner")

                            if isinstance(reasoning_result, dict):
                                result_confidence = reasoning_result.get('confidence', 0.0)
                            else:
                                result_confidence = getattr(reasoning_result, 'confidence', 0.0)

                            if result_confidence > best_confidence:
                                best_confidence = result_confidence
                                best_source = str(selected_tool_reasoning_type) if selected_tool_reasoning_type else "unified_reasoner"

                    except Exception as reasoning_error:
                        logger.warning(
                            f"Agent {agent_id} UnifiedReasoner invocation failed: {reasoning_error}. "
                            f"Using integration result only."
                        )

                # Build node results from reasoning (if not already built)
                if not node_results:
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

                reasoning_was_invoked = True

                # Update selected_tools from integration result
                _update_selected_tools_from_integration(
                    integration_result, result_selected_tools, selected_tools, task_id
                )

                # Update task_type based on integration result
                if hasattr(integration_result, 'metadata') and integration_result.metadata:
                    updated_query_type = integration_result.metadata.get('query_type')
                    is_self_introspection = (
                        integration_result.metadata.get('self_referential', False) or
                        integration_result.metadata.get('is_self_introspection', False)
                    )

                    if updated_query_type and updated_query_type != task_type:
                        task_type = updated_query_type

                    if is_self_introspection and task_type != 'self_introspection':
                        task_type = 'self_introspection'

            except Exception as reasoning_error:
                logger.warning(
                    f"Agent {agent_id} reasoning integration failed: {reasoning_error}. "
                    f"Falling back to graph execution."
                )
                is_reasoning_task = False

        # FALLBACK REASONING when selected_tools present but not yet invoked
        should_invoke_fallback_reasoning = (
            selected_tools and
            not node_results and
            not reasoning_was_invoked and
            (is_reasoning_task or any(tool.lower() in REASONING_TOOL_NAMES for tool in selected_tools))
        )

        if should_invoke_fallback_reasoning:
            result = _execute_fallback_reasoning(
                manager, agent_id, task_id, task_type, parameters, selected_tools,
                nodes, node_results, provenance, start_time, metadata
            )
            if result is not None:
                return result

        # GRAPH-BASED EXECUTION - For non-reasoning tasks or fallback
        if not is_reasoning_task or not node_results:
            logger.debug(f"Agent {agent_id} using graph-based execution for task {task_id}")
            for node in nodes:
                node_id = node.get("id", "unknown")
                node_type = node.get("type", "unknown")
                node_params = node.get("params", {})

                node_results[node_id] = {
                    "status": "completed",
                    "node_type": node_type,
                    "params_processed": list(node_params.keys()),
                    "reasoning_applied": False,
                }

        # Create comprehensive result
        duration = time.time() - start_time
        result = {
            "status": "completed",
            "outcome": "success",
            "agent_id": agent_id,
            "task_id": task_id,
            "graph_id": graph_id,
            "task_type": task_type,
            "execution_time": duration,
            "timestamp": time.time(),
            "nodes_processed": len(nodes),
            "node_results": node_results,
            "parameters_used": list(parameters.keys()) if parameters else [],
            "capability": metadata.capability.value,
            "reasoning_invoked": reasoning_was_invoked,
            "selected_tools": selected_tools if selected_tools else [],
        }

        # Add reasoning-specific results if available
        if reasoning_result is not None:
            result["reasoning_output"] = _extract_reasoning_output(
                manager, reasoning_result, task_id
            )

        # Update agent metadata
        metadata.record_task_completion(success=True, duration_s=duration)

        # Update provenance
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

        # Update statistics
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

        logger.info(
            f"Agent {agent_id} completed task {task_id} in {duration:.3f}s "
            f"(processed {len(nodes)} nodes, reasoning_invoked={result['reasoning_invoked']})"
        )
        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Agent {agent_id} task {task_id} failed: {e}")

        metadata.record_task_completion(success=False, duration_s=duration)
        metadata.record_error(e, {"task_id": task_id, "phase": "execution"})

        if provenance:
            provenance.complete("failed", error=str(e))
            provenance.update_resource_consumption({"cpu_seconds": duration})

        with manager.stats_lock:
            manager.stats["total_jobs_failed"] += 1

        manager._persist_state_to_redis()

        raise


# ============================================================
# HELPER FUNCTIONS for execute_agent_task
# ============================================================

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

    # Build reasoning_result
    conclusion = _safe_get_attr(integration_result, "conclusion", None)
    if not is_valid_conclusion(conclusion):
        conclusion = extract_conclusion_from_dict(
            getattr(integration_result, 'metadata', {})
        )
    if not is_valid_conclusion(conclusion):
        conclusion = _safe_get_attr(integration_result, "rationale", "")

    try:
        from vulcan.reasoning.reasoning_types import ReasoningResult as UR_ReasoningResult
        reasoning_result = UR_ReasoningResult(
            conclusion=conclusion,
            confidence=integration_result.confidence,
            reasoning_type=result_reasoning_strategy,
            explanation=result_metadata.get("explanation") or _safe_get_attr(integration_result, 'rationale', ''),
            metadata={
                **result_metadata,
                "source": "privileged_path",
                "selected_tools": result_selected_tools,
                "strategy": result_reasoning_strategy,
            }
        )
    except ImportError:
        class PrivilegedReasoningResult:
            def __init__(self, conclusion, confidence, reasoning_type, explanation, metadata):
                self.conclusion = conclusion
                self.confidence = confidence
                self.reasoning_type = reasoning_type
                self.explanation = explanation
                self.metadata = metadata

        reasoning_result = PrivilegedReasoningResult(
            conclusion=conclusion,
            confidence=integration_result.confidence,
            reasoning_type=result_reasoning_strategy,
            explanation=result_metadata.get("explanation") or _safe_get_attr(integration_result, 'rationale', ''),
            metadata={**result_metadata, "source": "privileged_path"}
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

    try:
        from vulcan.reasoning.reasoning_types import ReasoningResult as UR_ReasoningResult, ReasoningType as RT_Local

        if is_world_model_result:
            selected_reasoning_type = RT_Local.HYBRID
            source_name = "world_model"
        else:
            tool_to_rt_map = {
                'symbolic': RT_Local.SYMBOLIC,
                'probabilistic': RT_Local.PROBABILISTIC,
                'causal': RT_Local.CAUSAL,
                'analogical': RT_Local.ANALOGICAL,
                'mathematical': RT_Local.MATHEMATICAL,
                'philosophical': RT_Local.PHILOSOPHICAL,
                'multimodal': RT_Local.MULTIMODAL,
            }
            selected_reasoning_type = tool_to_rt_map.get(primary_engine.lower(), RT_Local.HYBRID)
            source_name = primary_engine

        reasoning_result = UR_ReasoningResult(
            conclusion=conclusion,
            confidence=integration_result.confidence,
            reasoning_type=selected_reasoning_type,
            explanation=result_metadata.get("explanation") or _safe_get_attr(integration_result, 'rationale', ''),
            metadata={
                "source": source_name,
                "selected_tools": result_selected_tools,
                "strategy": result_reasoning_strategy,
                "self_referential": result_metadata.get("self_referential", False),
                "preserve_content": result_metadata.get("preserve_content", False),
                "no_openai_replacement": result_metadata.get("no_openai_replacement", False),
                "is_introspection": result_metadata.get("is_introspection", False),
                "high_confidence_direct_use": True,
            }
        )
    except ImportError:
        class HighConfidenceReasoningResult:
            def __init__(self, conclusion, confidence, reasoning_type, explanation, metadata=None):
                self.conclusion = conclusion
                self.confidence = confidence
                self.reasoning_type = reasoning_type
                self.explanation = explanation
                self.metadata = metadata or {}

        rt_string = 'hybrid' if is_world_model_result else primary_engine.lower()
        reasoning_result = HighConfidenceReasoningResult(
            conclusion=conclusion,
            confidence=integration_result.confidence,
            reasoning_type=rt_string,
            explanation=result_metadata.get("explanation", _safe_get_attr(integration_result, "rationale", "")),
            metadata={"source": primary_engine, "high_confidence_direct_use": True}
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
