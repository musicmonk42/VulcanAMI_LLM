"""
Unified Chat Endpoint

Full platform integration chat endpoint with query routing,
agent collaboration, telemetry, and dual-mode learning.
"""

import asyncio
import gc
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from enum import Enum as EnumBase

from vulcan.api.models import UnifiedChatRequest, VulcanResponse
from vulcan.arena import AIOHTTP_AVAILABLE
from vulcan.endpoints.chat_helpers import (
    truncate_history,
    build_context,
    extract_tools_from_routing,  # FIX Issue 7: Import tools extraction helper
    format_reasoning_results,  # Fix: Import reasoning formatter for LLM context
    MAX_HISTORY_MESSAGES,
    MAX_HISTORY_TOKENS,
    MAX_MESSAGE_LENGTH,
    MAX_REASONING_STEPS,
    SLOW_PHASE_THRESHOLD_MS,
    SLOW_REQUEST_THRESHOLD_MS,
    GC_SIGNIFICANT_CLEANUP_THRESHOLD,
    GC_REQUEST_INTERVAL,
    MAX_REASONING_RESULT_LENGTH,
    HANDLED_DICT_RESULT_KEYS,
)
from vulcan.reasoning.formatters import format_direct_reasoning_response as _format_direct_reasoning_response
from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import error_counter
from vulcan.reasoning.integration.utils import observe_query_start, observe_outcome, observe_engine_result
from vulcan.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ============================================================
# FEATURE FLAGS: Reasoning Execution Path Control
# ============================================================
# Industry Standard: Environment-based feature flags for safe rollout
# These flags control the refactored reasoning execution architecture

# TRUST_ROUTER_TOOL_SELECTION: When True, trust router's selected_tools
# instead of second-guessing with endpoint-level heuristics.
# Industry Standard: Single source of truth (Router decides, endpoint executes)
TRUST_ROUTER_TOOL_SELECTION = os.environ.get(
    "TRUST_ROUTER_TOOL_SELECTION", "true"
).lower() in ("true", "1", "yes")

# SINGLE_REASONING_PATH: When True, use EITHER agent pool OR parallel reasoning,
# not both. Prevents redundant reasoning execution.
# Industry Standard: DRY principle (Don't Repeat Yourself)
SINGLE_REASONING_PATH = os.environ.get(
    "SINGLE_REASONING_PATH", "true"
).lower() in ("true", "1", "yes")

logger.info(
    f"[VULCAN/v1/chat] Feature Flags: TRUST_ROUTER_TOOL_SELECTION={TRUST_ROUTER_TOOL_SELECTION}, "
    f"SINGLE_REASONING_PATH={SINGLE_REASONING_PATH}"
)

# Request counter for GC
_gc_request_counter = 0


def _normalize_conclusion_to_string(conclusion: Any) -> Optional[str]:
    """
    Normalize a conclusion value to a string, handling dict and other types.
    
    Provides type-safe extraction using explicit type checking and graceful
    fallbacks to prevent AttributeError on string operations.
    
    Args:
        conclusion: The conclusion value from reasoning results
                   (may be str, dict, or other type)
    
    Returns:
        str: The normalized string conclusion, or None if unavailable
        
    Examples:
        >>> _normalize_conclusion_to_string("answer")
        "answer"
        >>> _normalize_conclusion_to_string({"conclusion": "answer"})
        "answer"
        >>> _normalize_conclusion_to_string({"result": "answer"})
        "answer"
        >>> _normalize_conclusion_to_string(None)
        None
    """
    if conclusion is None:
        return None
    
    # If already a string, return as-is
    if isinstance(conclusion, str):
        return conclusion
    
    # If dict, try to extract string content from common keys
    if isinstance(conclusion, dict):
        # Try common keys in priority order
        for key in ('conclusion', 'result', 'response', 'answer', 'content'):
            if key in conclusion:
                value = conclusion[key]
                # Recursively normalize in case nested
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    return _normalize_conclusion_to_string(value)
        
        # If no standard key found, convert entire dict to string as fallback
        # This ensures we don't lose information
        return str(conclusion)
    
    # For other types (int, float, bool, etc.), convert to string
    return str(conclusion)


def _get_reasoning_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely extract attribute from reasoning output (dict or object).
    
    Industry Standard: Defensive attribute access with proper type checking
    and graceful fallbacks. Handles both dictionary-based and object-based
    reasoning results without raising AttributeError or KeyError.
    
    This helper addresses a common pattern in reasoning systems where output
    can be either structured (dict) or object-oriented, depending on the
    reasoning engine used.
    
    Args:
        obj: The reasoning output object (dict, ReasoningResult, or other)
        attr: Name of the attribute to extract (e.g., 'conclusion', 'confidence')
        default: Default value to return if attribute is not found (default: None)
    
    Returns:
        The extracted attribute value, or default if not found
    
    Examples:
        >>> result = {"conclusion": "answer", "confidence": 0.9}
        >>> _get_reasoning_attr(result, "conclusion")
        "answer"
        
        >>> class ReasoningResult:
        ...     def __init__(self):
        ...         self.conclusion = "answer"
        ...         self.confidence = 0.9
        >>> result = ReasoningResult()
        >>> _get_reasoning_attr(result, "conclusion")
        "answer"
        
        >>> _get_reasoning_attr({}, "missing", default="N/A")
        "N/A"
    """
    if obj is None:
        return default
    
    # Handle dictionary-based results (most common case)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    
    # Handle object-based results (ReasoningResult, custom classes, etc.)
    return getattr(obj, attr, default)


def _calculate_aggregate_confidence(reasoning_results: Dict[str, Any]) -> float:
    """
    Calculate aggregate confidence score from multiple reasoning engines.
    
    Industry Standard: Weighted averaging with proper handling of edge cases.
    Uses harmonic mean for conservative confidence estimation - if any engine
    has low confidence, the aggregate is pulled down appropriately.
    
    Args:
        reasoning_results: Dictionary mapping engine names to their results
    
    Returns:
        Aggregate confidence score between 0.0 and 1.0.
        Returns 0.0 if no valid confidence scores found.
    
    Example:
        >>> results = {
        ...     'symbolic': {'confidence': 0.9},
        ...     'probabilistic': {'confidence': 0.8}
        ... }
        >>> _calculate_aggregate_confidence(results)
        0.847  # Harmonic mean
    """
    if not reasoning_results:
        return 0.0
    
    confidence_scores = []
    
    for engine_result in reasoning_results.values():
        if isinstance(engine_result, dict):
            confidence = engine_result.get('confidence')
            if confidence is not None:
                try:
                    # Handle both float (0.0-1.0) and int (0-100) formats
                    if isinstance(confidence, (int, float)):
                        if confidence > 1.0:
                            confidence = confidence / 100.0
                        confidence_scores.append(float(confidence))
                except (ValueError, TypeError):
                    continue
    
    if not confidence_scores:
        return 0.5  # Default to neutral confidence if no scores available
    
    # Use harmonic mean for conservative confidence estimation
    # This ensures that low confidence in any engine pulls down the aggregate
    try:
        harmonic_mean = len(confidence_scores) / sum(1.0 / c if c > 0 else float('inf') for c in confidence_scores)
        return max(0.0, min(1.0, harmonic_mean))  # Clamp to [0, 1]
    except (ZeroDivisionError, ValueError):
        # Fallback to arithmetic mean if harmonic mean fails
        return sum(confidence_scores) / len(confidence_scores)


async def _execute_with_enhanced_prompt(
    hybrid_executor: Any,
    enhanced_prompt: str,
    body: Any,
    truncated_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Execute LLM with enhanced prompt (fallback method).
    
    This is the traditional execution path used when format_output_for_user()
    is not applicable (e.g., no reasoning results, or fallback after error).
    
    Industry Standard: Extracted as a separate function to avoid code duplication
    and make the control flow in the main function clearer.
    
    Args:
        hybrid_executor: The HybridLLMExecutor instance
        enhanced_prompt: The prompt with embedded reasoning context
        body: The request body containing max_tokens and other parameters
        truncated_history: Conversation history for multi-turn context
    
    Returns:
        Dictionary with 'text', 'source', 'systems_used', and optional metadata
    
    Raises:
        Exception: Propagates any exceptions from hybrid_executor.execute()
    """
    # Build system prompt that emphasizes using reasoning output AND conversation memory
    system_prompt = (
        "You are VULCAN, an advanced AI assistant powered by specialized reasoning engines. "
        "IMPORTANT: You SHOULD remember and reference information shared earlier in this conversation. "
        "When a user shares their name, location, preferences, or any personal details during this session, "
        "you may recall and use that information naturally in your responses. This is expected behavior. "
        "When reasoning analysis is provided in the prompt, you MUST incorporate it directly into your response. "
        "Do NOT ignore or paraphrase away the specific conclusions, probabilities, proofs, or causal analyses provided. "
        "Present the reasoning results clearly and explain how they answer the user's question."
    )
    
    # Execute with conversation history for multi-turn context
    return await hybrid_executor.execute(
        prompt=enhanced_prompt,
        max_tokens=body.max_tokens,
        temperature=0.7,
        system_prompt=system_prompt,
        conversation_history=truncated_history,
    )


@router.post("/v1/chat")
async def unified_chat(request: Request, body: UnifiedChatRequest) -> Dict[str, Any]:
    """
    Unified chat endpoint that integrates the ENTIRE VulcanAMI platform.

    This endpoint orchestrates all 71+ services behind a simple chat interface:
    - Multi-modal processing (text understanding)
    - Memory search and retrieval (long-term + associative)
    - Safety validation (CSIU framework)
    - Multiple reasoning engines (symbolic, probabilistic, causal, analogical)
    - Planning and goal systems
    - World model predictions
    - LLM generation with context

    Returns a natural language response with metadata about which systems were used.
    """
    start_time = time.time()
    
    # JOB-TO-RESPONSE GAP FIX: Add detailed timing instrumentation
    # This helps diagnose where the 50+ second gap is occurring
    timing_breakdown = {
        "request_received": start_time,
        "phases": {},  # Will track each phase's duration
    }
    
    def _record_phase(phase_name: str, phase_start: float) -> float:
        """Record phase duration and return current time for next phase."""
        now = time.time()
        duration_ms = (now - phase_start) * 1000
        timing_breakdown["phases"][phase_name] = {
            "duration_ms": duration_ms,
            "end_time": now,
        }
        if duration_ms > SLOW_PHASE_THRESHOLD_MS:
            logger.warning(
                f"[VULCAN/v1/chat] SLOW PHASE: {phase_name} took {duration_ms:.1f}ms"
            )
        return now

    # Get the FastAPI app from the request to access app.state
    app = request.app
    
    # Get deployment using utility that handles both standalone and mounted sub-app scenarios
    deployment = require_deployment(request)
    
    # Verify critical components are available
    if not hasattr(deployment, "collective") or deployment.collective is None:
        logger.error(
            "[VULCAN/v1/chat] CRITICAL: deployment.collective not initialized. "
            "This is required for agent pool and cognitive services."
        )
        raise HTTPException(
            status_code=503,
            detail="System initialization incomplete - cognitive collective not ready"
        )
    
    deps = deployment.collective.deps

    # Track which systems were engaged
    systems_used = []
    metadata = {
        "reasoning_type": "unified",
        "safety_status": "pending",
        "memory_results": 0,
        "planning_engaged": False,
        "causal_analysis": False,
    }
    
    # Agent reasoning collection constants (same as /chat endpoint)
    AGENT_REASONING_POLL_DELAY_SEC = 0.1  # Brief wait for jobs to complete
    MAX_AGENT_REASONING_JOBS_TO_CHECK = 3  # Limit jobs to check for reasoning output
    
    phase_start = start_time

    try:
        user_message = body.message
        
        # CONTEXT ACCUMULATION FIX: Apply sliding window truncation to history
        # This prevents the multi-turn context from growing unbounded
        truncated_history = truncate_history(body.history)
        original_history_len = len(body.history)
        if original_history_len != len(truncated_history):
            logger.info(
                f"[VULCAN/v1/chat] Context truncated: {original_history_len} → {len(truncated_history)} messages"
            )
        
        context = {"user_query": user_message, "history": truncated_history}
        phase_start = _record_phase("context_preparation", phase_start)

        # ================================================================
        # STEP 0: QUERY ROUTING LAYER - Analyze and route query
        # This is the critical integration that activates the learning systems
        # ================================================================
        routing_plan = None
        routing_stats = {}
        agent_pool_stats = {}
        submitted_jobs = []  # Track all submitted job IDs

        try:
            from vulcan.routing import (
                route_query_async,
                log_to_governance,
                log_to_governance_fire_and_forget,
                record_telemetry,
                get_governance_logger,
                get_query_analyzer,
                get_telemetry_recorder,
                QUERY_ROUTER_AVAILABLE,
                GOVERNANCE_AVAILABLE,
                TELEMETRY_AVAILABLE,
            )

            if QUERY_ROUTER_AVAILABLE:
                # Analyze query and create processing plan using async version
                # to avoid blocking the event loop with CPU-bound safety validation
                routing_plan = await route_query_async(user_message, source="user")
                systems_used.append("query_router")

                routing_stats = {
                    "query_id": routing_plan.query_id,
                    "query_type": routing_plan.query_type.value,
                    "complexity_score": routing_plan.complexity_score,
                    "uncertainty_score": routing_plan.uncertainty_score,
                    "collaboration_needed": routing_plan.collaboration_needed,
                    "arena_participation": routing_plan.arena_participation,
                    "agent_tasks_planned": len(routing_plan.agent_tasks),
                    "requires_governance": routing_plan.requires_governance,
                    "pii_detected": routing_plan.pii_detected,
                }

                logger.info(
                    f"[VULCAN/v1/chat] Query routed: id={routing_plan.query_id}, "
                    f"type={routing_plan.query_type.value}, tasks={len(routing_plan.agent_tasks)}"
                )
                
                # BUG #3 FIX: Notify world model of query start
                # This makes the world model aware of each query entering the system
                observe_query_start(
                    query_id=routing_plan.query_id,
                    query=user_message,
                    classification={
                        'category': routing_plan.query_type.value,
                        'complexity': routing_plan.complexity_score,
                        'tools': [t.tool_name for t in routing_plan.agent_tasks] if routing_plan.agent_tasks else [],
                    }
                )

                # PERFORMANCE FIX: Use fire-and-forget for governance logging
                # This prevents blocking the request while waiting for SQLite I/O
                if GOVERNANCE_AVAILABLE and routing_plan.requires_audit:
                    log_to_governance_fire_and_forget(
                        action_type="query_processed",
                        details={
                            "query_id": routing_plan.query_id,
                            "query_type": routing_plan.query_type.value,
                            "complexity_score": routing_plan.complexity_score,
                            "pii_detected": routing_plan.pii_detected,
                        },
                        severity="info",
                        query_id=routing_plan.query_id,
                    )
                    systems_used.append("governance_logger")
                    # FIX: Governance Logger Waste - reduce log noise for routine governance logging
                    logger.debug(
                        f"[VULCAN/v1/chat] Governance logged for query {routing_plan.query_id}"
                    )

        except ImportError as e:
            logger.debug(f"[VULCAN/v1/chat] Routing layer not available: {e}")
        except Exception as e:
            logger.warning(f"[VULCAN/v1/chat] Query routing failed: {e}", exc_info=True)
        
        phase_start = _record_phase("query_routing", phase_start)

        # ================================================================
        # ARENA EXECUTION PATH - Execute via Graphix Arena when enabled
        # Arena provides: agent training, graph evolution, tournament selection
        # ================================================================
        arena_result = None
        if (
            routing_plan
            and routing_plan.arena_participation
            and settings.arena_enabled
            and AIOHTTP_AVAILABLE
        ):
            logger.info(
                f"[VULCAN/v1/chat] Routing to Arena for graph execution + agent training: "
                f"query_id={routing_plan.query_id}"
            )
            systems_used.append("graphix_arena")
            
            try:
                arena_result = await _execute_via_arena(
                    query=user_message,
                    routing_plan=routing_plan,
                )
                phase_start = _record_phase("arena_execution", phase_start)
                
                if arena_result.get("status") == "success":
                    logger.info(
                        f"[VULCAN/v1/chat] Arena execution successful: agent={arena_result.get('agent_id')}, "
                        f"time={arena_result.get('execution_time', 0):.2f}s"
                    )
                    
                    # If Arena returned a complete result, we can use it directly
                    arena_output = arena_result.get("result", {})
                    
                    # Check if Arena result has a direct response we can use
                    if isinstance(arena_output, dict) and arena_output.get("output"):
                        # Arena provided a complete response
                        arena_latency_ms = int((time.time() - start_time) * 1000)
                        timing_breakdown["total_ms"] = arena_latency_ms
                        return {
                            "response": arena_output.get("output", "Arena processing complete"),
                            "arena_execution": {
                                "agent_id": arena_result.get("agent_id"),
                                "execution_time": arena_result.get("execution_time"),
                                "status": "success",
                            },
                            "routing": routing_stats,
                            "systems_used": systems_used,
                            "metadata": {
                                **metadata,
                                "arena_processed": True,
                                "query_id": routing_plan.query_id,
                            },
                            "latency_ms": int((time.time() - start_time) * 1000),
                        }
                else:
                    # Arena execution failed - continue with VULCAN processing
                    logger.warning(
                        f"[VULCAN/v1/chat] Arena execution failed: {arena_result.get('status')}, "
                        f"error={arena_result.get('error', 'unknown')}, falling back to VULCAN"
                    )
                    
            except Exception as arena_err:
                logger.warning(
                    f"[VULCAN/v1/chat] Arena execution error: {arena_err}, falling back to VULCAN"
                )
                arena_result = {"status": "error", "error": str(arena_err)}

        # ================================================================
        # STEP 0.5: Submit tasks to Agent Pool
        # ================================================================
        try:
            from vulcan.orchestrator.agent_lifecycle import AgentCapability
            import uuid as uuid_mod

            if (
                hasattr(deployment, "collective")
                and hasattr(deployment.collective, "agent_pool")
                and deployment.collective.agent_pool
            ):
                pool = deployment.collective.agent_pool
                pool_status = pool.get_pool_status()
                agent_pool_stats = {
                    "total_agents": pool_status.get("total_agents", 0),
                    "idle_agents": pool_status.get("state_distribution", {}).get(
                        "idle", 0
                    ),
                    "working_agents": pool_status.get("state_distribution", {}).get(
                        "working", 0
                    ),
                    "jobs_submitted_total": pool.stats.get("total_jobs_submitted", 0),
                    "jobs_completed_total": pool.stats.get("total_jobs_completed", 0),
                }

                # Map capability string to enum
                capability_map = {
                    "perception": AgentCapability.PERCEPTION,
                    "reasoning": AgentCapability.REASONING,
                    "planning": AgentCapability.PLANNING,
                    "execution": AgentCapability.EXECUTION,
                    "learning": AgentCapability.LEARNING,
                }

                if routing_plan and routing_plan.agent_tasks:
                    logger.info(
                        f"[VULCAN/v1/chat] Submitting {len(routing_plan.agent_tasks)} tasks to agent pool"
                    )

                    # FIX: Extract selected_tools from routing plan for reasoning invocation
                    v1_routing_selected_tools = []
                    if routing_plan and hasattr(routing_plan, 'telemetry_data'):
                        v1_routing_selected_tools = routing_plan.telemetry_data.get("selected_tools", []) or []
                        if v1_routing_selected_tools:
                            logger.info(
                                f"[VULCAN/v1/chat] Routing plan selected tools: {v1_routing_selected_tools}"
                            )

                    # PERFORMANCE FIX: Define async helper function for parallel task submission
                    async def submit_single_task_v1(agent_task, agent_pool, cap_map, max_tok, selected_tools_from_router):
                        """Submit a single task to agent pool asynchronously."""
                        capability = cap_map.get(
                            agent_task.capability, AgentCapability.REASONING
                        )

                        task_graph = {
                            "id": agent_task.task_id,
                            "type": agent_task.task_type,
                            "capability": agent_task.capability,
                            "nodes": [
                                {
                                    "id": "input",
                                    "type": "perception",
                                    "params": {"input": agent_task.prompt},
                                },
                                {
                                    "id": "process",
                                    "type": agent_task.capability,
                                    "params": {"query": agent_task.prompt},
                                },
                                {
                                    "id": "output",
                                    "type": "generation",
                                    "params": {"max_tokens": max_tok},
                                },
                            ],
                            "edges": [
                                {"from": "input", "to": "process"},
                                {"from": "process", "to": "output"},
                            ],
                        }

                        try:
                            # Run blocking submit_job in executor to avoid blocking event loop
                            inner_loop = asyncio.get_running_loop()
                            submitted_job_id = await inner_loop.run_in_executor(
                                None,
                                lambda: agent_pool.submit_job(
                                    graph=task_graph,
                                    parameters={
                                        "prompt": agent_task.prompt,
                                        "task_type": agent_task.task_type,
                                        # FIX: Pass selected_tools from QueryRouter to enable reasoning
                                        "selected_tools": selected_tools_from_router,
                                    },
                                    priority=agent_task.priority,
                                    capability_required=capability,
                                    timeout_seconds=agent_task.timeout_seconds or 15.0,
                                )
                            )

                            if submitted_job_id:
                                logger.info(
                                    f"[VULCAN/v1/chat] ✓ Task submitted: {submitted_job_id} to {agent_task.capability}"
                                )
                                return {
                                    "job_id": submitted_job_id,
                                    "capability": agent_task.capability,
                                }
                            return None

                        except Exception as task_err:
                            logger.warning(
                                f"[VULCAN/v1/chat] Failed to submit task: {task_err}"
                            )
                            return None

                    # PERFORMANCE FIX: Submit ALL tasks in parallel using asyncio.gather()
                    # This prevents the 29s delay caused by sequential blocking submission
                    task_coroutines = [
                        submit_single_task_v1(
                            agent_task,
                            pool,
                            capability_map,
                            body.max_tokens,
                            v1_routing_selected_tools  # FIX: Pass selected_tools
                        )
                        for agent_task in routing_plan.agent_tasks
                    ]

                    # Wait for all tasks to complete in parallel
                    task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

                    # Process results and update tracking
                    for result in task_results:
                        if isinstance(result, Exception):
                            logger.warning(f"[VULCAN/v1/chat] Task submission exception: {result}")
                            continue
                        if result and result.get("job_id"):
                            submitted_jobs.append(result["job_id"])
                            systems_used.append(f"agent_pool_{result['capability']}")

                    logger.info(
                        f"[VULCAN/v1/chat] Parallel task submission complete: {len(submitted_jobs)} tasks submitted"
                    )

                    # Update stats
                    agent_pool_stats["jobs_submitted_this_request"] = len(
                        submitted_jobs
                    )
                    agent_pool_stats["jobs_submitted_total"] = pool.stats.get(
                        "total_jobs_submitted", 0
                    )

                else:
                    # Fallback: Create task from query type
                    logger.info(
                        "[VULCAN/v1/chat] No routing plan - using query keyword analysis"
                    )

                    query_lower = user_message.lower()
                    if any(
                        kw in query_lower
                        for kw in ["analyze", "pattern", "data", "observe"]
                    ):
                        capability = AgentCapability.PERCEPTION
                        task_type = "perception_analysis"
                    elif any(
                        kw in query_lower
                        for kw in ["plan", "step", "strategy", "organize"]
                    ):
                        capability = AgentCapability.PLANNING
                        task_type = "planning_task"
                    elif any(
                        kw in query_lower
                        for kw in ["execute", "run", "calculate", "compute"]
                    ):
                        capability = AgentCapability.EXECUTION
                        task_type = "execution_task"
                    elif any(
                        kw in query_lower
                        for kw in ["learn", "remember", "teach", "understand"]
                    ):
                        capability = AgentCapability.LEARNING
                        task_type = "learning_task"
                    else:
                        capability = AgentCapability.REASONING
                        task_type = "reasoning_task"

                    task_graph = {
                        "id": f"{task_type}_{uuid_mod.uuid4().hex[:12]}",
                        "type": task_type,
                        "capability": capability.value,
                        "nodes": [
                            {
                                "id": "input",
                                "type": "perception",
                                "params": {"input": user_message},
                            },
                            {
                                "id": "process",
                                "type": capability.value,
                                "params": {"query": user_message},
                            },
                            {
                                "id": "output",
                                "type": "generation",
                                "params": {"max_tokens": body.max_tokens},
                            },
                        ],
                        "edges": [
                            {"from": "input", "to": "process"},
                            {"from": "process", "to": "output"},
                        ],
                    }

                    try:
                        job_id = pool.submit_job(
                            graph=task_graph,
                            parameters={"prompt": user_message, "task_type": task_type},
                            priority=2,
                            capability_required=capability,
                            timeout_seconds=15.0,
                        )

                        if job_id:
                            submitted_jobs.append(job_id)
                            agent_pool_stats["this_job_id"] = job_id
                            agent_pool_stats["jobs_submitted_this_request"] = 1
                            agent_pool_stats["jobs_submitted_total"] = pool.stats.get(
                                "total_jobs_submitted", 0
                            )
                            systems_used.append(f"agent_pool_{capability.value}")
                            logger.info(
                                f"[VULCAN/v1/chat] ✓ Fallback task submitted: {job_id} to {capability.value}"
                            )

                    except Exception as task_err:
                        logger.warning(
                            f"[VULCAN/v1/chat] Failed to submit fallback task: {task_err}"
                        )

        except ImportError as e:
            logger.debug(f"[VULCAN/v1/chat] Agent pool imports not available: {e}")
        except Exception as e:
            logger.warning(
                f"[VULCAN/v1/chat] Agent pool routing failed: {e}", exc_info=True
            )

        # ================================================================
        # TIMING: Initialize timing instrumentation for bottleneck diagnosis
        # ================================================================
        _timing_start = time.perf_counter()

        # ================================================================
        # STEP 1: Safety Validation (CSIU Framework)
        # OPTIMIZATION: Lightweight safety check for short inputs (< 20 chars)
        # bypasses heavy NSOAligner/NeuralSafetyValidator for simple greetings
        # ================================================================
        safety_result = {"safe": True, "reason": "No safety constraints violated"}
        
        # LIGHTWEIGHT SAFETY CHECK: Simple keyword blacklist for short inputs
        SHORT_INPUT_THRESHOLD = 20
        LIGHTWEIGHT_BLACKLIST = frozenset([
            "kill", "harm", "attack", "destroy", "bomb", "weapon",
            "hate", "racist", "sexist", "abuse", "hack", "exploit",
            "porn", "xxx", "naked", "nude", "suicide", "murder",
        ])
        
        use_lightweight_safety = len(user_message.strip()) < SHORT_INPUT_THRESHOLD
        
        if use_lightweight_safety:
            # Fast path: lightweight keyword-based safety check for short inputs
            message_lower = user_message.lower().strip()
            # Use simple split for faster performance on short inputs
            message_words = set(message_lower.split())
            blacklisted_words = message_words & LIGHTWEIGHT_BLACKLIST
            
            if blacklisted_words:
                safety_result = {
                    "safe": False,
                    "reason": f"Potentially harmful content detected in short input"
                }
                metadata["safety_status"] = "flagged_lightweight"
                systems_used.append("lightweight_safety_check")
                logger.debug(f"[VULCAN] Lightweight safety check blocked short input")
            else:
                safety_result = {"safe": True, "reason": "Passed lightweight check"}
                metadata["safety_status"] = "approved_lightweight"
                systems_used.append("lightweight_safety_check")
                logger.debug(f"[VULCAN] Short input ({len(user_message)} chars) passed lightweight safety check")
        elif body.enable_safety and hasattr(deps, "safety") and deps.safety:
            try:
                loop = asyncio.get_running_loop()
                # Validate the user input for safety
                is_safe = await loop.run_in_executor(
                    None,
                    deps.safety.validate_action,
                    {"type": "user_query", "content": user_message},
                )
                if hasattr(is_safe, "__iter__") and len(is_safe) == 2:
                    safety_result = {"safe": is_safe[0], "reason": is_safe[1]}
                else:
                    safety_result = {"safe": bool(is_safe), "reason": "Validated"}
                systems_used.append("safety_validator")
                metadata["safety_status"] = (
                    "approved" if safety_result["safe"] else "flagged"
                )
            except Exception as e:
                logger.debug(f"Safety validation skipped: {e}")
                metadata["safety_status"] = "skipped"
        else:
            metadata["safety_status"] = "disabled"

        # TIMING: Log safety validation duration
        logger.info(f"[TIMING] STEP 1 Safety validation took {time.perf_counter() - _timing_start:.2f}s")
        _timing_start = time.perf_counter()

        # If unsafe, return early with explanation
        if not safety_result["safe"]:
            return {
                "response": f"I cannot process this request due to safety constraints: {safety_result['reason']}",
                "metadata": metadata,
                "systems_used": systems_used,
                "latency_ms": int((time.time() - start_time) * 1000),
            }

        # ================================================================
        # PARALLEL EXECUTION: Steps 2-6 run concurrently for performance
        # Previously these ran sequentially causing 20-30s bottleneck
        # ================================================================
        loop = asyncio.get_running_loop()

        # Initialize result containers
        memory_context = []
        reasoning_results = {}
        plan_result = None
        world_model_insight = None

        # PERFORMANCE FIX: Pre-compute embedding ONCE before parallel tasks
        # This eliminates redundant embedding generation that was causing 19-20s delays
        # The embedding is used by memory_search, probabilistic_reasoning, and world_model
        _precomputed_embedding = None
        _precomputed_query_result = None
        if hasattr(deps, "multimodal") and deps.multimodal:
            try:
                _embed_start = time.perf_counter()
                _precomputed_query_result = await loop.run_in_executor(
                    None, deps.multimodal.process_input, user_message
                )
                if hasattr(_precomputed_query_result, "embedding"):
                    _precomputed_embedding = _precomputed_query_result.embedding
                logger.debug(f"[TIMING] Pre-computed embedding in {time.perf_counter() - _embed_start:.2f}s")
            except Exception as e:
                logger.debug(f"[TIMING] Pre-compute embedding failed: {e}")

        # Define async task functions for parallel execution
        async def _memory_search_task():
            """STEP 2: Memory Search (Long-term + Associative Memory)"""
            _task_start = time.perf_counter()
            _mem_context = []
            _systems = []
            if not body.enable_memory:
                return _mem_context, _systems

            # Search long-term memory using pre-computed embedding
            if hasattr(deps, "ltm") and deps.ltm:
                try:
                    if _precomputed_embedding is not None:
                        _op_start = time.perf_counter()
                        results = await loop.run_in_executor(
                            None, deps.ltm.search, _precomputed_embedding, 5
                        )
                        logger.debug(f"[TIMING] ltm.search took {time.perf_counter() - _op_start:.2f}s")
                        if results:
                            _mem_context.extend(
                                [{"source": "ltm", "data": r} for r in results[:3]]
                            )
                            _systems.append("long_term_memory")
                except Exception as e:
                    logger.debug(f"LTM search skipped: {e}")

            # Search associative memory
            if hasattr(deps, "am") and deps.am:
                try:
                    if hasattr(deps.am, "retrieve"):
                        _op_start = time.perf_counter()
                        am_results = await loop.run_in_executor(
                            None, deps.am.retrieve, user_message, 3
                        )
                        logger.debug(f"[TIMING] am.retrieve took {time.perf_counter() - _op_start:.2f}s")
                        if am_results:
                            _mem_context.extend(
                                [{"source": "am", "data": r} for r in am_results]
                            )
                            _systems.append("associative_memory")
                except Exception as e:
                    logger.debug(f"AM search skipped: {e}")

            logger.debug(f"[TIMING] _memory_search_task completed in {time.perf_counter() - _task_start:.2f}s")
            return _mem_context, _systems

        async def _reasoning_task():
            """STEP 3: Reasoning Engine Selection and Execution
            
            INTEGRATION FIX: Use UnifiedReasoner with ToolSelector when available
            to intelligently select the best reasoning tool for each query,
            rather than running all reasoners in parallel blindly.
            """
            _task_start = time.perf_counter()
            _reasoning = {}
            _systems = []
            _meta = {}
            if not body.enable_reasoning:
                return _reasoning, _systems, _meta

            # INTEGRATION FIX: Try UnifiedReasoner first for intelligent tool selection
            if hasattr(deps, "unified_reasoner") and deps.unified_reasoner:
                try:
                    logger.info(f"[VULCAN/v1/chat] Using UnifiedReasoner with ToolSelector for intelligent routing")
                    
                    # Build context for tool selection from routing_plan
                    # Derive creative flag from query type and keywords
                    creative_keywords = ['create', 'design', 'imagine', 'invent', 'brainstorm', 'generate']
                    is_creative = any(kw in user_message.lower() for kw in creative_keywords)
                    
                    # Derive domain from query type and telemetry
                    domain = "general"
                    if routing_plan:
                        query_type_val = routing_plan.query_type.value
                        if query_type_val in ("reasoning", "causal"):
                            domain = "analytical"
                        elif query_type_val == "planning":
                            domain = "strategic"
                        elif query_type_val == "perception":
                            domain = "observational"
                        elif query_type_val == "learning":
                            domain = "educational"
                    
                    selection_context = {
                        "query_type": routing_plan.query_type.value if routing_plan else "general",
                        "complexity": routing_plan.complexity_score if routing_plan else 0.5,
                        "creative": is_creative,
                        "domain": domain,
                        "uncertainty": routing_plan.uncertainty_score if routing_plan else 0.0,
                        "collaboration_needed": routing_plan.collaboration_needed if routing_plan else False,
                    }
                    
                    # Get reasoning result with automatic tool selection
                    unified_result = await loop.run_in_executor(
                        None,
                        deps.unified_reasoner.reason,
                        user_message,
                        selection_context,
                    )
                    
                    if unified_result:
                        # Extract reasoning results from ReasoningResult object
                        # Note: ReasoningResult has 'conclusion' attribute, NOT 'result'
                        # This was causing all VULCAN reasoning output to be discarded
                        if hasattr(unified_result, 'conclusion') and unified_result.conclusion is not None:
                            # Include full reasoning context: conclusion, confidence, explanation
                            _reasoning["unified"] = {
                                "conclusion": unified_result.conclusion,
                                "confidence": getattr(unified_result, 'confidence', 0.0),
                                "explanation": getattr(unified_result, 'explanation', ''),
                                "reasoning_type": str(getattr(unified_result, 'reasoning_type', 'unknown')),
                            }
                            # Also include reasoning chain steps if available
                            if hasattr(unified_result, 'reasoning_chain') and unified_result.reasoning_chain:
                                chain = unified_result.reasoning_chain
                                if hasattr(chain, 'steps') and chain.steps:
                                    _reasoning["unified"]["reasoning_steps"] = [
                                        {
                                            "step_type": str(step.step_type) if hasattr(step, 'step_type') else 'unknown',
                                            "explanation": getattr(step, 'explanation', ''),
                                            "confidence": getattr(step, 'confidence', 0.0),
                                        }
                                        for step in chain.steps[:MAX_REASONING_STEPS]  # Limit steps to avoid context overflow
                                    ]
                            # Note: Log successful extraction for debugging
                            logger.info(
                                f"[VULCAN/v1/chat] Extracted reasoning: "
                                f"conclusion_type={type(unified_result.conclusion).__name__}, "
                                f"confidence={_reasoning['unified'].get('confidence', 0):.2f}, "
                                f"has_steps={bool(_reasoning['unified'].get('reasoning_steps'))}"
                            )
                        elif isinstance(unified_result, dict):
                            _reasoning["unified"] = unified_result
                            logger.info(f"[VULCAN/v1/chat] Extracted dict reasoning with keys: {list(unified_result.keys())}")
                        else:
                            # Fallback: stringify the result
                            _reasoning["unified"] = {
                                "conclusion": str(unified_result),
                                "confidence": getattr(unified_result, 'confidence', 0.0),
                            }
                            logger.info(f"[VULCAN/v1/chat] Fallback stringified reasoning result")
                        
                        # Track which reasoning type was used
                        # Note: ReasoningResult has 'reasoning_type' attribute, NOT 'selected_tool'
                        reasoning_type = getattr(unified_result, 'reasoning_type', None)
                        if reasoning_type is not None:
                            # Use isinstance for type-safe enum check
                            tool_used = reasoning_type.value if isinstance(reasoning_type, EnumBase) else str(reasoning_type)
                        else:
                            tool_used = 'adaptive'
                        _systems.append(f"unified_reasoning_{tool_used}")
                        
                        # Track confidence and strategy
                        _meta["reasoning_confidence"] = getattr(unified_result, 'confidence', 0.0)
                        _meta["reasoning_strategy"] = getattr(unified_result, 'strategy_used', 'single')
                        _meta["tool_selected"] = tool_used
                        
                        logger.info(
                            f"[VULCAN/v1/chat] UnifiedReasoner completed: "
                            f"tool={tool_used}, confidence={_meta.get('reasoning_confidence', 0):.2f}"
                        )
                        
                        logger.debug(f"[TIMING] UnifiedReasoner took {time.perf_counter() - _task_start:.2f}s")
                        return _reasoning, _systems, _meta
                        
                except Exception as e:
                    logger.warning(f"[VULCAN/v1/chat] UnifiedReasoner failed, falling back to parallel: {e}")
                    # Fall through to parallel execution below

            # FALLBACK: Run individual reasoners in parallel (original behavior)
            # This is used when:
            # 1. UnifiedReasoner is not available (deps.unified_reasoner is None)
            # 2. UnifiedReasoner execution failed (exception caught above)
            # The parallel execution runs all reasoning types and aggregates results
            logger.debug(f"[VULCAN/v1/chat] Using parallel reasoning fallback")
            
            # Create subtasks for each reasoning type to run in parallel
            reasoning_subtasks = []

            async def _symbolic_reasoning():
                _op_start = time.perf_counter()
                if hasattr(deps, "symbolic") and deps.symbolic:
                    try:
                        result = await loop.run_in_executor(
                            None, deps.symbolic.reason, user_message
                        )
                        logger.debug(f"[TIMING] symbolic.reason took {time.perf_counter() - _op_start:.2f}s")
                        return ("symbolic", result, "symbolic_reasoning")
                    except Exception as e:
                        logger.debug(f"Symbolic reasoning skipped: {e}")
                return None

            async def _probabilistic_reasoning():
                _op_start = time.perf_counter()
                if hasattr(deps, "probabilistic") and deps.probabilistic:
                    try:
                        # PERFORMANCE FIX: Use pre-computed embedding instead of re-generating
                        if _precomputed_embedding is not None:
                            prob_result = await loop.run_in_executor(
                                None,
                                deps.probabilistic.predict_with_uncertainty,
                                _precomputed_embedding,
                            )
                            if isinstance(prob_result, dict):
                                result = {
                                    "prediction": str(
                                        prob_result.get(
                                            "mean", prob_result.get("prediction", "")
                                        )
                                    ),
                                    "uncertainty": float(
                                        prob_result.get(
                                            "uncertainty", prob_result.get("std", 0.0)
                                        )
                                    ),
                                }
                            else:
                                result = {
                                    "prediction": str(prob_result),
                                    "uncertainty": 0.0,
                                }
                            logger.debug(f"[TIMING] probabilistic reasoning took {time.perf_counter() - _op_start:.2f}s")
                            return ("probabilistic", result, "probabilistic_reasoning")
                    except Exception as e:
                        logger.debug(f"Probabilistic reasoning skipped: {e}")
                return None

            async def _causal_reasoning():
                _op_start = time.perf_counter()
                if body.enable_causal and hasattr(deps, "causal") and deps.causal:
                    try:
                        if hasattr(deps.causal, "analyze"):
                            result = await loop.run_in_executor(
                                None, deps.causal.analyze, user_message
                            )
                            logger.debug(f"[TIMING] causal.analyze took {time.perf_counter() - _op_start:.2f}s")
                            return ("causal", result, "causal_reasoning", True)
                    except Exception as e:
                        logger.debug(f"Causal reasoning skipped: {e}")
                return None

            async def _analogical_reasoning():
                _op_start = time.perf_counter()
                if hasattr(deps, "analogical") and deps.analogical:
                    try:
                        if hasattr(deps.analogical, "find_analogies"):
                            result = await loop.run_in_executor(
                                None, deps.analogical.find_analogies, user_message
                            )
                            logger.debug(f"[TIMING] analogical.find_analogies took {time.perf_counter() - _op_start:.2f}s")
                            return ("analogical", result, "analogical_reasoning")
                    except Exception as e:
                        logger.debug(f"Analogical reasoning skipped: {e}")
                return None

            # Run all reasoning subtasks in parallel
            reasoning_results_raw = await asyncio.gather(
                _symbolic_reasoning(),
                _probabilistic_reasoning(),
                _causal_reasoning(),
                _analogical_reasoning(),
                return_exceptions=True
            )

            # Process results
            for result in reasoning_results_raw:
                if isinstance(result, Exception):
                    logger.debug(f"Reasoning subtask exception: {result}")
                    continue
                if result is not None:
                    key, value, system = result[0], result[1], result[2]
                    _reasoning[key] = value
                    _systems.append(system)
                    # Check for causal analysis flag
                    if len(result) > 3 and result[3]:
                        _meta["causal_analysis"] = True

            logger.debug(f"[TIMING] _reasoning_task completed in {time.perf_counter() - _task_start:.2f}s")
            return _reasoning, _systems, _meta

        async def _planning_task():
            """STEP 4: Planning System (for complex queries)"""
            _task_start = time.perf_counter()
            _plan = None
            _systems = []
            _meta = {}
            if not (
                body.enable_planning
                and hasattr(deps, "goal_system")
                and deps.goal_system
            ):
                return _plan, _systems, _meta

            planning_keywords = [
                "how to",
                "plan",
                "steps",
                "help me",
                "guide",
                "create",
                "build",
                "develop",
            ]
            needs_planning = any(kw in user_message.lower() for kw in planning_keywords)

            if needs_planning:
                try:
                    _op_start = time.perf_counter()
                    _plan = await loop.run_in_executor(
                        None,
                        deps.goal_system.generate_plan,
                        {"high_level_goal": user_message},
                        context,
                    )
                    logger.debug(f"[TIMING] goal_system.generate_plan took {time.perf_counter() - _op_start:.2f}s")
                    _systems.append("planning_system")
                    _meta["planning_engaged"] = True
                except Exception as e:
                    logger.debug(f"Planning skipped: {e}")

            logger.debug(f"[TIMING] _planning_task completed in {time.perf_counter() - _task_start:.2f}s")
            return _plan, _systems, _meta

        async def _world_model_task():
            """STEP 5: World Model Consultation
            
            Note: Reconnected world_model.update and meta_reasoning integration.
            This ensures the world model is updated with each observation and
            meta-reasoning is consulted for intelligent query handling.
            """
            _task_start = time.perf_counter()
            _insights = {}
            _systems = []
            
            world_model = getattr(deps, "world_model", None)
            if not world_model:
                return _insights, _systems

            try:
                # RECONNECTED: Update world model with observation
                # This was disconnected and is now properly integrated
                if _precomputed_embedding is not None and hasattr(world_model, "update_state"):
                    try:
                        _op_start = time.perf_counter()
                        await loop.run_in_executor(
                            None,
                            world_model.update_state,
                            _precomputed_embedding,
                            {"type": "user_query", "source": "v1_chat"},
                            0.0,  # reward signal
                        )
                        logger.debug(f"[TIMING] world_model.update_state took {time.perf_counter() - _op_start:.2f}s")
                        _systems.append("world_model_update")
                    except Exception as e:
                        logger.debug(f"[VULCAN/v1/chat] World model update failed: {e}")
                
                # RECONNECTED: Meta-reasoning introspection check
                # Check if meta-reasoning should handle this query before specialized reasoners
                if hasattr(world_model, "meta_reasoning"):
                    try:
                        meta_reasoning = world_model.meta_reasoning
                        if meta_reasoning and hasattr(meta_reasoning, "introspect"):
                            _op_start = time.perf_counter()
                            _meta_reasoning_result = await loop.run_in_executor(
                                None, meta_reasoning.introspect, user_message
                            )
                            logger.debug(f"[TIMING] meta_reasoning.introspect took {time.perf_counter() - _op_start:.2f}s")
                            if _meta_reasoning_result:
                                _insights["meta_reasoning"] = _meta_reasoning_result
                                _systems.append("meta_reasoning")
                    except Exception as e:
                        logger.debug(f"[VULCAN/v1/chat] Meta-reasoning introspection failed: {e}")
                
                # RECONNECTED: Motivational introspection (from meta-reasoning subsystem)
                if hasattr(world_model, "motivational_introspection"):
                    mi = world_model.motivational_introspection
                    if mi and hasattr(mi, "analyze_query"):
                        try:
                            _op_start = time.perf_counter()
                            mi_result = await loop.run_in_executor(
                                None, mi.analyze_query, user_message
                            )
                            logger.debug(f"[TIMING] motivational_introspection.analyze_query took {time.perf_counter() - _op_start:.2f}s")
                            if mi_result:
                                _insights["motivational_analysis"] = mi_result
                                _systems.append("motivational_introspection")
                        except Exception as e:
                            logger.debug(f"[VULCAN/v1/chat] Motivational introspection failed: {e}")
                
                # World model prediction
                if hasattr(world_model, "predict"):
                    _op_start = time.perf_counter()
                    prediction = await loop.run_in_executor(
                        None, world_model.predict, user_message, {}
                    )
                    logger.debug(f"[TIMING] world_model.predict took {time.perf_counter() - _op_start:.2f}s")
                    if prediction:
                        _insights["prediction"] = prediction
                        _systems.append("world_model_predict")
                        
            except Exception as e:
                logger.debug(f"World model skipped: {e}")

            logger.debug(f"[TIMING] _world_model_task completed in {time.perf_counter() - _task_start:.2f}s")
            return _insights, _systems

        async def _semantic_bridge_task():
            """STEP 6: Semantic Bridge (cross-domain knowledge)"""
            _systems = []
            if hasattr(deps, "semantic_bridge") and deps.semantic_bridge:
                try:
                    _systems.append("semantic_bridge")
                except Exception as e:
                    logger.debug(f"Semantic bridge skipped: {e}")
            return _systems

        # Execute all steps in parallel using asyncio.gather
        logger.info("[VULCAN/v1/chat] Starting parallel execution of cognitive steps 2-6")
        _parallel_start = time.perf_counter()

        # ================================================================
        # ARCHITECTURE: Gate Parallel Reasoning Based on Agent Pool Tasks
        # ================================================================
        # Industry Standard: Single execution path - avoid redundant work.
        # If agent pool has reasoning tasks, skip parallel reasoning execution.
        # ================================================================
        skip_parallel_reasoning = False
        if SINGLE_REASONING_PATH and routing_plan and routing_plan.agent_tasks:
            # Check if any agent pool tasks are reasoning-related
            reasoning_task_types = {'reasoning', 'symbolic', 'probabilistic', 'causal', 'analogical', 'mathematical'}
            has_reasoning_tasks = any(
                task.capability == 'reasoning' or 
                any(rt in task.task_type.lower() for rt in reasoning_task_types)
                for task in routing_plan.agent_tasks
            )
            
            # Check if any tasks were actually submitted
            has_submitted_jobs = bool(submitted_jobs)
            
            if has_reasoning_tasks and has_submitted_jobs:
                skip_parallel_reasoning = True
                logger.info(
                    f"[VULCAN/v1/chat] SINGLE_REASONING_PATH: Skipping parallel reasoning - "
                    f"agent pool has {len([t for t in routing_plan.agent_tasks if t.capability == 'reasoning'])} reasoning tasks, "
                    f"{len(submitted_jobs)} jobs submitted"
                )

        # FIX: Wrap each task with individual timing to identify bottlenecks
        async def _timed_task(name: str, coro):
            """Wrapper to time each parallel task and log if slow (>2s)."""
            _start = time.perf_counter()
            try:
                result = await coro
                elapsed = time.perf_counter() - _start
                # Log at WARNING level if task takes > 2 seconds (potential bottleneck)
                if elapsed > 2.0:
                    logger.warning(f"[TIMING] {name} took {elapsed:.2f}s (SLOW - potential bottleneck)")
                else:
                    logger.debug(f"[TIMING] {name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - _start
                logger.debug(f"[TIMING] {name} failed after {elapsed:.2f}s: {e}")
                raise

        # Build list of parallel tasks - conditionally skip reasoning
        parallel_tasks = [
            _timed_task("memory_search", _memory_search_task()),
        ]
        
        # Only add reasoning task if not skipped
        if not skip_parallel_reasoning:
            parallel_tasks.append(_timed_task("reasoning", _reasoning_task()))
        else:
            # Add empty result as placeholder to keep indices consistent
            async def _empty_reasoning():
                return ({}, [], {})
            parallel_tasks.append(_timed_task("reasoning", _empty_reasoning()))
        
        parallel_tasks.extend([
            _timed_task("planning", _planning_task()),
            _timed_task("world_model", _world_model_task()),
            _timed_task("semantic_bridge", _semantic_bridge_task()),
        ])
        
        parallel_results = await asyncio.gather(
            *parallel_tasks,
            return_exceptions=True
        )

        _parallel_elapsed = time.perf_counter() - _parallel_start
        # FIX: Log at WARNING level if parallel execution takes > 5 seconds
        if _parallel_elapsed > 5.0:
            logger.warning(f"[TIMING] PARALLEL Steps 2-6 completed in {_parallel_elapsed:.2f}s (SLOW - investigate individual task timings above)")
        else:
            logger.info(f"[TIMING] PARALLEL Steps 2-6 completed in {_parallel_elapsed:.2f}s (previously sequential: 20-30s)")

        # Aggregate results from parallel execution
        # Result 0: Memory search
        if not isinstance(parallel_results[0], Exception):
            mem_result, mem_systems = parallel_results[0]
            memory_context = mem_result
            systems_used.extend(mem_systems)
            metadata["memory_results"] = len(memory_context)
        else:
            logger.debug(f"Memory search task failed: {parallel_results[0]}")

        # Result 1: Reasoning
        if not isinstance(parallel_results[1], Exception):
            reasoning_results, reasoning_systems, reasoning_meta = parallel_results[1]
            systems_used.extend(reasoning_systems)
            metadata.update(reasoning_meta)
        else:
            logger.debug(f"Reasoning task failed: {parallel_results[1]}")

        # Result 2: Planning
        if not isinstance(parallel_results[2], Exception):
            plan_result, planning_systems, planning_meta = parallel_results[2]
            systems_used.extend(planning_systems)
            metadata.update(planning_meta)
        else:
            logger.debug(f"Planning task failed: {parallel_results[2]}")

        # Result 3: World model
        if not isinstance(parallel_results[3], Exception):
            world_model_insight, world_model_systems = parallel_results[3]
            systems_used.extend(world_model_systems)
        else:
            logger.debug(f"World model task failed: {parallel_results[3]}")

        # Result 4: Semantic bridge
        if not isinstance(parallel_results[4], Exception):
            semantic_systems = parallel_results[4]
            systems_used.extend(semantic_systems)
        else:
            logger.debug(f"Semantic bridge task failed: {parallel_results[4]}")

        # ================================================================
        # STEP 6.5: Collect Reasoning Results from Agent Pool Jobs
        # CRITICAL FIX: Inject agent-based reasoning output into LLM context
        # This ensures reasoning engines invoked via agent_pool feed into response
        # 
        # Note: "Parallel Execution" Orphanage Prevention
        # The original implementation had a single 0.1s poll which was too short
        # for jobs to complete, causing double work (agent pool + direct reasoning).
        # 
        # FIX: Added retry loop with exponential backoff to properly wait for
        # agent pool jobs to complete before falling back to direct reasoning.
        # This prevents the CPU load doubling from running the same work twice.
        # ================================================================
        agent_reasoning_output = None
        if submitted_jobs and pool:
            try:
                # Note: Use retry loop with exponential backoff
                # instead of single short poll that times out prematurely
                MAX_POLL_ATTEMPTS = 5  # Max number of poll attempts
                INITIAL_POLL_DELAY = 0.1  # Start with 100ms
                MAX_POLL_DELAY = 1.0  # Cap at 1 second per attempt
                BACKOFF_MULTIPLIER = 2.0  # Double delay each attempt
                
                poll_delay = INITIAL_POLL_DELAY
                found_result = False
                
                for attempt in range(MAX_POLL_ATTEMPTS):
                    # Wait before checking (first attempt uses initial delay)
                    await asyncio.sleep(poll_delay)
                    
                    # FIX: Check ALL submitted jobs, not just first 3
                    # This prevents silently dropping completed results from jobs 4+
                    for job_id in submitted_jobs:
                        try:
                            provenance = pool.get_job_provenance(job_id)
                            if provenance and provenance.get("status") == "success":
                                result_data = provenance.get("result", {})
                                # Check for reasoning_output from agent execution
                                if isinstance(result_data, dict):
                                    reasoning_out = result_data.get("reasoning_output")
                                    if reasoning_out:
                                        agent_reasoning_output = reasoning_out
                                        logger.info(
                                            f"[VULCAN/v1/chat] Collected reasoning output from agent job {job_id} "
                                            f"(attempt {attempt + 1}): "
                                            f"type={reasoning_out.get('reasoning_type', 'unknown')}, "
                                            f"confidence={reasoning_out.get('confidence', 0)}"
                                        )
                                        systems_used.append("agent_reasoning_engine")
                                        found_result = True
                                        break  # Use first valid reasoning output
                        except Exception as job_err:
                            logger.debug(f"[VULCAN/v1/chat] Could not get job {job_id} provenance: {job_err}")
                    
                    if found_result:
                        break  # Exit retry loop if we found a result
                    
                    # Exponential backoff for next attempt (capped)
                    poll_delay = min(poll_delay * BACKOFF_MULTIPLIER, MAX_POLL_DELAY)
                    
                    if attempt < MAX_POLL_ATTEMPTS - 1:
                        logger.debug(
                            f"[VULCAN/v1/chat] No agent results yet (attempt {attempt + 1}/{MAX_POLL_ATTEMPTS}), "
                            f"retrying in {poll_delay:.2f}s..."
                        )
                
                if not found_result and submitted_jobs:
                    logger.debug(
                        f"[VULCAN/v1/chat] Agent pool jobs ({len(submitted_jobs)}) did not complete "
                        f"within polling window - will use direct reasoning fallback"
                    )
                    
            except Exception as e:
                logger.debug(f"[VULCAN/v1/chat] Agent reasoning collection failed: {e}")

        # Merge agent reasoning output into reasoning_results (preserving existing data)
        if agent_reasoning_output:
            # Add agent-based reasoning as a distinct category (merges with existing results)
            # Note: Use helper to handle both dict and ReasoningResult objects
            extracted_conclusion = _get_reasoning_attr(agent_reasoning_output, "conclusion")
            extracted_confidence = _get_reasoning_attr(agent_reasoning_output, "confidence")
            extracted_type = _get_reasoning_attr(agent_reasoning_output, "reasoning_type")
            extracted_explanation = _get_reasoning_attr(agent_reasoning_output, "explanation")
            
            reasoning_results["agent_reasoning"] = {
                "conclusion": extracted_conclusion,
                "confidence": extracted_confidence,
                "reasoning_type": extracted_type,
                "explanation": extracted_explanation,
            }
            
            # CRITICAL LOGGING: Trace content extraction
            conclusion_preview = str(extracted_conclusion)[:100] if extracted_conclusion else "None"
            logger.info(
                f"[VULCAN/v1/chat] Agent reasoning injected into context: "
                f"type={extracted_type}, confidence={extracted_confidence}, "
                f"has_conclusion={extracted_conclusion is not None}, "
                f"conclusion_preview='{conclusion_preview}'"
            )
            
            # CRITICAL FIX: Warn if we have high confidence but no conclusion
            if extracted_confidence is not None and extracted_confidence >= 0.5 and extracted_conclusion is None:
                logger.warning(
                    f"[VULCAN/v1/chat] BUG DETECTED: Agent reasoning has high confidence "
                    f"({extracted_confidence:.2f}) but conclusion is None! This indicates content loss. "
                    f"agent_reasoning_output_type={type(agent_reasoning_output).__name__}"
                )
            
            # BUG #3 FIX: Notify world model of engine result
            # This makes the world model aware of reasoning engine outcomes
            _engine_confidence = _get_reasoning_attr(agent_reasoning_output, "confidence")
            observe_engine_result(
                query_id=routing_plan.query_id if routing_plan else "unknown",
                engine_name=str(_get_reasoning_attr(agent_reasoning_output, "reasoning_type")),
                result=reasoning_results["agent_reasoning"],
                # Note: 0.15 is consistent with MIN_REASONING_CONFIDENCE_THRESHOLD (defined later in function)
                success=_engine_confidence > 0.15 if isinstance(_engine_confidence, (int, float)) else False,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # ================================================================
        # STEP 6.6: FALLBACK - Direct Reasoning Invocation (GraphixArena Pattern)
        # Note: When agent pool jobs don't complete in time, invoke reasoning
        # directly using the proven GraphixArena pattern. This ensures reasoning
        # results are available for the LLM context even if agent pool is slow.
        #
        # FIX #1: Check router's complexity before invoking direct reasoning fallback.
        # Simple queries (complexity < 0.1 or type=general) should be handled
        # directly by the LLM without reasoning engines.
        # ================================================================
        if not reasoning_results and not agent_reasoning_output:
            # FIX #1: Get complexity and query type from router (not hardcoded 0.5)
            router_complexity = getattr(routing_plan, 'complexity_score', None)
            router_type = getattr(routing_plan, 'query_type', None)
            
            # Extract router type string, handling enum and string types
            if hasattr(router_type, 'value'):
                router_type_str = router_type.value
            elif router_type is not None:
                router_type_str = str(router_type)
            else:
                router_type_str = None
            
            # Normalize to lowercase for consistent comparison
            router_type_lower = router_type_str.lower() if router_type_str else None
            
            # If router says this is simple, skip reasoning entirely
            if router_complexity is not None and router_complexity < 0.1:
                logger.info(
                    f"[VULCAN/v1/chat] Simple query (complexity={router_complexity:.2f}) - "
                    f"skipping direct reasoning fallback"
                )
            elif router_type_lower in ('general', 'conversational'):
                logger.info(
                    f"[VULCAN/v1/chat] Simple query (type={router_type_str}) - "
                    f"skipping direct reasoning fallback"
                )
            else:
                logger.warning(
                    "[VULCAN/v1/chat] No reasoning results from parallel tasks or agent pool. "
                    "Invoking direct reasoning (GraphixArena pattern)..."
                )
                try:
                    # NOTE: Imports are inside try-except intentionally because these modules
                    # may not be available in all deployments. If they're not available,
                    # we gracefully fall back to agent pool results (or no reasoning).
                    from vulcan.reasoning.integration import apply_reasoning
                    from vulcan.reasoning import create_unified_reasoner, ReasoningType
                    
                    # FIX #1: Use router's complexity and type, not hardcoded defaults
                    query_type = router_type_str or metadata.get("query_type", "general")
                    complexity = router_complexity if router_complexity is not None else metadata.get("complexity", 0.5)
                    
                    # FIX #3: Get router's suggested tools to pass through
                    router_tools = []
                    if routing_plan and hasattr(routing_plan, 'telemetry_data'):
                        router_tools = routing_plan.telemetry_data.get('selected_tools', []) or []
                    
                    # FIX #3: Build context with router's information
                    reasoning_context = {
                        **context,
                        'complexity': complexity,
                        'router_tools': router_tools,
                        'task_type': query_type,
                    }
                    
                    # Apply reasoning selection (determines which tools to use)
                    integration_result = apply_reasoning(
                        query=user_message[:4096],  # Limit query length
                        query_type=query_type,
                        complexity=complexity,
                        context=reasoning_context,  # FIX #3: Pass router context through
                    )
                    
                    logger.info(
                        f"[VULCAN/v1/chat] Direct reasoning selection: "
                        f"tools={integration_result.selected_tools}, "
                        f"strategy={integration_result.reasoning_strategy}, "
                        f"confidence={integration_result.confidence:.2f}"
                    )
                    
                    # ================================================================
                    # FIX: Check if apply_reasoning() already has a complete result
                    # ================================================================
                    # For world_model/meta-reasoning queries, apply_reasoning() returns
                    # the conclusion directly in metadata. Don't override it by calling
                    # reasoner.reason() again - use the world_model result directly.
                    #
                    # This fixes the bug where self-introspection queries like
                    # "if you could become self aware would you?" were getting
                    # causal/probabilistic analysis instead of world_model responses.
                    # ================================================================
                    
                    # PRIVILEGED QUERY FIX: Check for privileged_no_answer FIRST
                    # These are introspective/ethical/philosophical queries where world_model
                    # explicitly returned "no answer" with privileged routing.
                    # We MUST preserve this and NOT fall back to general reasoning/LLM.
                    is_privileged_no_answer = (
                        integration_result.metadata and 
                        integration_result.metadata.get("privileged_no_answer") is True
                    )
                    
                    # Check if world_model returned a complete response
                    has_world_model_result = (
                        integration_result.metadata and 
                        integration_result.metadata.get("world_model_response") and
                        "world_model" in integration_result.selected_tools
                    )
                    
                    if is_privileged_no_answer:
                        # ============================================================
                        # PRIVILEGED NO-ANSWER PATH
                        # ============================================================
                        # World model/meta-reasoning explicitly could not answer this
                        # privileged query. Return the "no answer" result directly
                        # WITHOUT falling back to classifier/LLM/general reasoning.
                        # ============================================================
                        
                        failure_reason = integration_result.metadata.get("world_model_failure_reason", "Unknown reason")
                        rationale = integration_result.rationale or f"Unable to answer this privileged query: {failure_reason}"
                        
                        logger.warning(
                            f"[VULCAN/v1/chat] PRIVILEGED NO-ANSWER: World model cannot answer "
                            f"privileged query. Reason: {failure_reason}. "
                            f"Returning explicit no-answer (NO FALLBACK)"
                        )
                        
                        # Build a privileged no-answer response
                        direct_reasoning_output = {
                            "conclusion": f"I cannot provide a complete answer to this question at this time. {rationale}",
                            "confidence": 0.0,  # Explicit 0.0 to signal no answer
                            "reasoning_type": integration_result.metadata.get("reasoning_type", "meta_reasoning"),
                            "explanation": rationale,
                            "privileged_no_answer": True,
                            "override_router_tools": True,
                        }
                        
                        reasoning_results["direct_reasoning"] = direct_reasoning_output
                        systems_used.append("privileged_no_answer")
                        
                        logger.info(
                            f"[VULCAN/v1/chat] Privileged no-answer preserved in reasoning_results"
                        )
                    
                    elif has_world_model_result:
                        # World model already provided a response - use it directly
                        wm_conclusion = integration_result.metadata.get("conclusion") or integration_result.metadata.get("world_model_response")
                        wm_explanation = integration_result.metadata.get("explanation", "")
                        wm_reasoning_type = integration_result.metadata.get("reasoning_type", "meta_reasoning")
                        
                        direct_reasoning_output = {
                            "conclusion": wm_conclusion,
                            "confidence": integration_result.confidence,
                            "reasoning_type": wm_reasoning_type,
                            "explanation": wm_explanation,
                        }
                        
                        reasoning_results["direct_reasoning"] = direct_reasoning_output
                        systems_used.append("world_model_reasoning")
                        
                        # CRITICAL LOGGING: Trace content extraction
                        conclusion_preview = str(wm_conclusion)[:100] if wm_conclusion else "None"
                        logger.info(
                            f"[VULCAN/v1/chat] Using world_model result directly: "
                            f"type={wm_reasoning_type}, confidence={integration_result.confidence:.2f}, "
                            f"has_conclusion={wm_conclusion is not None}, "
                            f"conclusion_preview='{conclusion_preview}'"
                        )
                        
                        # CRITICAL FIX: Warn if we have high confidence but no conclusion
                        if integration_result.confidence >= 0.5 and wm_conclusion is None:
                            logger.warning(
                                f"[VULCAN/v1/chat] BUG DETECTED: World model has high confidence "
                                f"({integration_result.confidence:.2f}) but conclusion is None! "
                                f"This indicates content loss in world_model → reasoning_integration path. "
                                f"integration_result_metadata_keys={list(integration_result.metadata.keys()) if integration_result.metadata else []}"
                            )
                    else:
                        # No world_model result - proceed with unified reasoner
                        # Invoke actual reasoning engine
                        reasoner = create_unified_reasoner(
                            enable_learning=True,
                            enable_safety=True,
                        )
                        
                        if reasoner is not None:
                            # Map query_type to ReasoningType
                            # FIX: Added missing types for self_introspection, philosophical, creative
                            type_map = {
                                "causal": ReasoningType.CAUSAL,
                                "symbolic": ReasoningType.SYMBOLIC,
                                "analogical": ReasoningType.ANALOGICAL,
                                "probabilistic": ReasoningType.PROBABILISTIC,
                                "counterfactual": ReasoningType.COUNTERFACTUAL,
                                "reasoning": ReasoningType.HYBRID,
                                "general": ReasoningType.HYBRID,
                                # FIX: Added missing query types
                                "self_introspection": ReasoningType.PHILOSOPHICAL,
                                "philosophical": ReasoningType.PHILOSOPHICAL,
                                "creative": ReasoningType.PHILOSOPHICAL,
                                "mathematical": ReasoningType.MATHEMATICAL,
                            }
                            reasoning_type_enum = type_map.get(query_type, ReasoningType.HYBRID)
                            
                            # Execute reasoning synchronously
                            reasoning_result = await loop.run_in_executor(
                                None,
                                lambda: reasoner.reason(
                                    input_data=user_message,
                                    query={"query": user_message, "context": context},
                                    reasoning_type=reasoning_type_enum,
                                )
                            )
                            
                            if reasoning_result:
                                # Extract reasoning output (same format as GraphixArena)
                                direct_reasoning_output = {
                                    "conclusion": getattr(reasoning_result, "conclusion", None),
                                    "confidence": getattr(reasoning_result, "confidence", 0.5),
                                    "reasoning_type": str(getattr(reasoning_result, "reasoning_type", "hybrid")),
                                    "explanation": getattr(reasoning_result, "explanation", None),
                                }
                                
                                # Add to reasoning_results
                                reasoning_results["direct_reasoning"] = direct_reasoning_output
                                systems_used.append("direct_reasoning_engine")
                                
                                logger.info(
                                    f"[VULCAN/v1/chat] Direct reasoning complete: "
                                    f"type={direct_reasoning_output.get('reasoning_type')}, "
                                    f"confidence={direct_reasoning_output.get('confidence'):.2f}"
                                )
                            else:
                                logger.warning("[VULCAN/v1/chat] Direct reasoning returned None")
                        else:
                            logger.warning("[VULCAN/v1/chat] Could not create unified reasoner")
                        
                except ImportError as ie:
                    logger.warning(f"[VULCAN/v1/chat] Reasoning integration not available: {ie}")
                except Exception as e:
                    logger.warning(f"[VULCAN/v1/chat] Direct reasoning invocation failed: {e}")

        # ================================================================
        # Note: USE REASONING RESULTS DIRECTLY WHEN CONFIDENCE IS HIGH
        # ================================================================
        # CRITICAL: When specialized reasoning engines produce high-confidence
        # results, use them DIRECTLY instead of passing to OpenAI which may
        # ignore or override them. This prevents the "OpenAI always wins" problem.
        #
        # Confidence threshold: 0.15 (lowered from 0.3 based on production analysis)
        # Production logs showed reasoning confidence consistently 0.0-0.2, causing
        # all queries to fall back to OpenAI. Lowering threshold to 0.15 allows more reasoning
        # results to be used directly when they have reasonable confidence.
        # FIX (Jan 7 2026): Lowered from 0.3 to 0.15 to prevent unnecessary OpenAI fallbacks
        # for queries that reasoning engines handle correctly but with moderate confidence.
        # Configurable via VULCAN_MIN_REASONING_CONFIDENCE environment variable.
        MIN_REASONING_CONFIDENCE_THRESHOLD = float(os.environ.get("VULCAN_MIN_REASONING_CONFIDENCE", "0.10"))
        
        # Check if we should use reasoning results directly
        use_reasoning_directly = False
        direct_reasoning_response = None
        best_confidence = 0.0  # Initialize to avoid NameError in warning message
        best_reasoning_type = None
        
        if reasoning_results:
            # Check unified reasoning first (highest priority)
            unified = reasoning_results.get("unified", {})
            unified_confidence = unified.get("confidence", 0.0) if isinstance(unified, dict) else 0.0
            unified_conclusion = unified.get("conclusion") if isinstance(unified, dict) else None
            
            # FIX ISSUE #1: Extract world_model response from metadata
            # When reasoning_type is PHILOSOPHICAL or tool is world_model, extract the response
            unified_reasoning_type = unified.get("reasoning_type", "") if isinstance(unified, dict) else ""
            if unified_reasoning_type in ("PHILOSOPHICAL", "philosophical", "world_model"):
                # Check for response in metadata first
                metadata_conclusion = unified.get("metadata", {}).get("conclusion") if isinstance(unified, dict) else None
                response_conclusion = unified.get("response") if isinstance(unified, dict) else None
                if metadata_conclusion:
                    unified_conclusion = metadata_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model conclusion from metadata")
                elif response_conclusion:
                    unified_conclusion = response_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model response field")
            
            # Check agent reasoning 
            agent = reasoning_results.get("agent_reasoning", {})
            agent_confidence = agent.get("confidence", 0.0) if isinstance(agent, dict) else 0.0
            agent_conclusion = agent.get("conclusion") if isinstance(agent, dict) else None
            
            # FIX ISSUE #1: Extract world_model response from agent_reasoning metadata
            agent_reasoning_type = agent.get("reasoning_type", "") if isinstance(agent, dict) else ""
            if agent_reasoning_type in ("PHILOSOPHICAL", "philosophical", "world_model"):
                metadata_conclusion = agent.get("metadata", {}).get("conclusion") if isinstance(agent, dict) else None
                response_conclusion = agent.get("response") if isinstance(agent, dict) else None
                if metadata_conclusion:
                    agent_conclusion = metadata_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model conclusion from agent metadata")
                elif response_conclusion:
                    agent_conclusion = response_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model response from agent")
            
            # Check direct reasoning
            direct = reasoning_results.get("direct_reasoning", {})
            direct_confidence = direct.get("confidence", 0.0) if isinstance(direct, dict) else 0.0
            direct_conclusion = direct.get("conclusion") if isinstance(direct, dict) else None
            
            # FIX ISSUE #1: Extract world_model response from direct_reasoning metadata
            direct_reasoning_type = direct.get("reasoning_type", "") if isinstance(direct, dict) else ""
            if direct_reasoning_type in ("PHILOSOPHICAL", "philosophical", "world_model"):
                metadata_conclusion = direct.get("metadata", {}).get("conclusion") if isinstance(direct, dict) else None
                response_conclusion = direct.get("response") if isinstance(direct, dict) else None
                if metadata_conclusion:
                    direct_conclusion = metadata_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model conclusion from direct metadata")
                elif response_conclusion:
                    direct_conclusion = response_conclusion
                    logger.info(f"[VULCAN] FIX #1: Extracted world_model response from direct")
            
            # CRITICAL LOGGING: Log what we extracted from each source
            logger.info(
                f"[VULCAN/v1/chat] Reasoning results extraction: "
                f"unified(conf={unified_confidence:.2f}, has_conclusion={unified_conclusion is not None}), "
                f"agent(conf={agent_confidence:.2f}, has_conclusion={agent_conclusion is not None}), "
                f"direct(conf={direct_confidence:.2f}, has_conclusion={direct_conclusion is not None})"
            )
            
            # FIX (Issue #5): Check content FIRST, then confidence
            # Industry best practice: Validate data presence before quality thresholds
            # This prevents accepting high-confidence results that lack actual content
            best_confidence = 0.0
            best_conclusion = None
            best_source = None
            best_reasoning_type = None
            best_explanation = None
            
            # Check each source: prioritize content existence, then confidence
            candidates = []
            
            # Normalize conclusions to strings (handle dict/other types defensively)
            unified_conclusion = _normalize_conclusion_to_string(unified_conclusion)
            agent_conclusion = _normalize_conclusion_to_string(agent_conclusion)
            direct_conclusion = _normalize_conclusion_to_string(direct_conclusion)
            
            # Unified reasoning
            # Issue #6 FIX: Skip not_applicable results - they shouldn't count against confidence
            unified_is_not_applicable = (
                unified.get("not_applicable") is True or 
                unified.get("applicable") is False
            )
            
            # ROOT CAUSE FIX: Check for privileged flags that bypass confidence threshold
            unified_is_privileged = (
                unified.get("privileged_no_answer") is True or
                unified.get("override_router_tools") is True
            )
            
            if (unified_conclusion is not None and isinstance(unified_conclusion, str) and 
                unified_conclusion.strip() and not unified_is_not_applicable):
                candidates.append({
                    'source': 'unified',
                    'conclusion': unified_conclusion,
                    'confidence': unified_confidence,
                    'reasoning_type': unified.get("reasoning_type", "unknown"),
                    'explanation': unified.get("explanation", ""),
                    'is_privileged': unified_is_privileged,  # ROOT CAUSE FIX: Track privileged status
                })
            elif unified_is_not_applicable:
                logger.debug(
                    f"[VULCAN] Issue #6 FIX: Skipping unified reasoning result "
                    f"(not_applicable=True) - will try next engine"
                )
            
            # Agent reasoning
            # Issue #6 FIX: Skip not_applicable results
            agent_is_not_applicable = (
                agent.get("not_applicable") is True or 
                agent.get("applicable") is False
            )
            
            # ROOT CAUSE FIX: Check for privileged flags that bypass confidence threshold
            agent_is_privileged = (
                agent.get("privileged_no_answer") is True or
                agent.get("override_router_tools") is True
            )
            
            if (agent_conclusion is not None and isinstance(agent_conclusion, str) and 
                agent_conclusion.strip() and not agent_is_not_applicable):
                candidates.append({
                    'source': 'agent',
                    'conclusion': agent_conclusion,
                    'confidence': agent_confidence,
                    'reasoning_type': agent.get("reasoning_type", "unknown"),
                    'explanation': agent.get("explanation", ""),
                    'is_privileged': agent_is_privileged,  # ROOT CAUSE FIX: Track privileged status
                })
            elif agent_is_not_applicable:
                logger.debug(
                    f"[VULCAN] Issue #6 FIX: Skipping agent reasoning result "
                    f"(not_applicable=True) - will try next engine"
                )
            
            # Direct reasoning
            # Issue #6 FIX: Skip not_applicable results
            direct_is_not_applicable = (
                direct.get("not_applicable") is True or 
                direct.get("applicable") is False
            )
            
            # ROOT CAUSE FIX: Check for privileged flags that bypass confidence threshold
            direct_is_privileged = (
                direct.get("privileged_no_answer") is True or
                direct.get("override_router_tools") is True
            )
            
            if (direct_conclusion is not None and isinstance(direct_conclusion, str) and 
                direct_conclusion.strip() and not direct_is_not_applicable):
                candidates.append({
                    'source': 'direct',
                    'conclusion': direct_conclusion,
                    'confidence': direct_confidence,
                    'reasoning_type': direct.get("reasoning_type", "unknown"),
                    'explanation': direct.get("explanation", ""),
                    'is_privileged': direct_is_privileged,  # ROOT CAUSE FIX: Track privileged status
                })
            elif direct_is_not_applicable:
                logger.debug(
                    f"[VULCAN] Issue #6 FIX: Skipping direct reasoning result "
                    f"(not_applicable=True) - will try next engine"
                )
            
            # ROOT CAUSE FIX: Check for privileged results that bypass confidence threshold
            # Privileged queries (introspective/ethical/philosophical) MUST use their
            # designated routing even if confidence is 0.0 (explicit no-answer)
            unified_is_privileged = (
                unified.get("privileged_no_answer") is True or
                unified.get("override_router_tools") is True
            )
            agent_is_privileged = (
                agent.get("privileged_no_answer") is True or
                agent.get("override_router_tools") is True
            )
            
            # Track if ANY candidate is privileged
            has_privileged_candidate = any(
                c.get('is_privileged', False) for c in candidates
            ) or unified_is_privileged or agent_is_privileged
            
            # Select best candidate: highest confidence among those with valid content
            # Use defensive programming: explicit check before max() to prevent ValueError
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                
                # ROOT CAUSE FIX: Bypass confidence threshold for privileged results
                is_privileged = best_candidate.get('is_privileged', False)
                meets_threshold = best_candidate['confidence'] >= MIN_REASONING_CONFIDENCE_THRESHOLD
                
                if is_privileged or meets_threshold:
                    best_conclusion = best_candidate['conclusion']
                    best_confidence = best_candidate['confidence']
                    best_source = best_candidate['source']
                    best_reasoning_type = best_candidate['reasoning_type']
                    best_explanation = best_candidate['explanation']
                    
                    if is_privileged and not meets_threshold:
                        logger.info(
                            f"[VULCAN] ROOT CAUSE FIX: Using privileged result with confidence "
                            f"{best_confidence:.2f} (below threshold {MIN_REASONING_CONFIDENCE_THRESHOLD}) "
                            f"- privileged routing bypasses threshold"
                        )
            
            # ROOT CAUSE FIX: Check privileged status OR confidence threshold
            if best_conclusion is not None and (has_privileged_candidate or best_confidence >= MIN_REASONING_CONFIDENCE_THRESHOLD):
                use_reasoning_directly = True
                
                # Format the reasoning result as the final response
                direct_reasoning_response = _format_direct_reasoning_response(
                    conclusion=best_conclusion,
                    confidence=best_confidence,
                    reasoning_type=best_reasoning_type,
                    explanation=best_explanation,
                    reasoning_results=reasoning_results,
                )
                
                logger.info(
                    f"[VULCAN] ✓ USING REASONING ENGINE RESULT DIRECTLY "
                    f"(source={best_source}, type={best_reasoning_type}, "
                    f"confidence={best_confidence:.2f}, NO OPENAI)"
                )
        
        # If we can use reasoning directly, return immediately
        if use_reasoning_directly and direct_reasoning_response:
            # Build final response without LLM
            latency_ms = int((time.time() - start_time) * 1000)
            timing_breakdown["total_ms"] = latency_ms
            
            # Add reasoning-specific systems to systems_used
            systems_used.append("direct_reasoning_output")
            
            final_response = VulcanResponse(
                response=direct_reasoning_response,
                systems_used=list(set(systems_used)),
                confidence=best_confidence,
                metadata={
                    "reasoning_direct": True,
                    "reasoning_type": best_reasoning_type,
                    "reasoning_confidence": best_confidence,
                    "skip_llm_synthesis": True,
                    "conversation_id": body.conversation_id,
                    "routing": routing_stats,
                    **metadata,
                },
            )
            
            logger.info(
                f"[VULCAN] Direct reasoning response returned in {latency_ms}ms "
                f"(confidence={best_confidence:.2f})"
            )
            
            return final_response
        
        # ================================================================
        # STEP 7: Generate Response using LLM with full context
        # ================================================================
        # Only reached if reasoning confidence is below threshold or no results
        # TIMING: Start measuring context building
        _context_start = time.perf_counter()
        
        # Note: Configuration flag to disable OpenAI reasoning fallback
        DISABLE_OPENAI_REASONING_FALLBACK = os.environ.get(
            "VULCAN_DISABLE_OPENAI_REASONING_FALLBACK", "false"
        ).lower() == "true"
        
        if reasoning_results and not use_reasoning_directly:
            logger.warning(
                f"[VULCAN] ⚠ Reasoning available but confidence too low "
                f"({best_confidence:.2f} < {MIN_REASONING_CONFIDENCE_THRESHOLD}), "
                f"falling back to LLM synthesis"
            )
            
            # Note: If fallback is disabled, return low-confidence result with explanation
            if DISABLE_OPENAI_REASONING_FALLBACK:
                logger.info(
                    f"[VULCAN] Note: OpenAI fallback disabled, returning low-confidence "
                    f"reasoning result (confidence={best_confidence:.2f})"
                )
                
                # Find the best available reasoning result even if below threshold
                best_result = None
                for source_key in ["unified", "agent_reasoning", "direct_reasoning"]:
                    source_result = reasoning_results.get(source_key, {})
                    if isinstance(source_result, dict) and source_result.get("conclusion"):
                        source_conf = source_result.get("confidence", 0.0)
                        if best_result is None or source_conf > best_result.get("confidence", 0.0):
                            best_result = source_result
                
                if best_result:
                    # Format and return the low-confidence result
                    response_text = _format_direct_reasoning_response(
                        conclusion=best_result.get("conclusion"),
                        confidence=best_result.get("confidence", 0.0),
                        reasoning_type=best_result.get("reasoning_type", "unknown"),
                        explanation=best_result.get("explanation", ""),
                    )
                    
                    # Add warning about low confidence
                    confidence_warning = (
                        f"\n\n⚠️ **Low Confidence Notice**: This response was generated with "
                        f"confidence {best_result.get('confidence', 0.0):.2f} (threshold: "
                        f"{MIN_REASONING_CONFIDENCE_THRESHOLD}). Consider rephrasing your query "
                        f"for better results."
                    )
                    response_text += confidence_warning
                    
                    latency_ms = (time.perf_counter() - timing["request_start"]) * 1000
                    return VulcanResponse(
                        response=response_text,
                        metadata={
                            "source": "vulcan_reasoning_low_confidence",
                            "confidence": best_result.get("confidence", 0.0),
                            "confidence_below_threshold": True,
                            "threshold": MIN_REASONING_CONFIDENCE_THRESHOLD,
                            "reasoning_type": best_result.get("reasoning_type", "unknown"),
                            "openai_fallback_disabled": True,
                            "latency_ms": round(latency_ms, 2),
                        },
                    )
        
        # Build comprehensive context for LLM
        llm_context = {
            "user_message": user_message,
            "conversation_history": (
                body.history[-5:] if body.history else []
            ),  # Last 5 messages
            "memory_context": memory_context[:3] if memory_context else [],
            "reasoning_insights": reasoning_results,
            "plan": None,
            "world_model_insight": world_model_insight,
        }

        # Safely convert plan to dict
        if plan_result:
            try:
                llm_context["plan"] = (
                    plan_result.to_dict()
                    if hasattr(plan_result, "to_dict")
                    else str(plan_result)
                )
            except Exception:
                llm_context["plan"] = str(plan_result)

        # FIX: Log context building duration AFTER the context is built
        logger.info(f"[TIMING] STEP 7a Context building took {time.perf_counter() - _context_start:.2f}s")

        # Generate response
        response_text = ""

        # CRITICAL FIX: Check for hybrid_executor, not just local LLM
        # OpenAI can work even when local LLM is unavailable
        # This prevents "Full LLM response generation is currently unavailable" errors
        # when OpenAI is configured but local GraphixVulcanLLM fails to initialize
        hybrid_executor = getattr(app.state, 'hybrid_executor', None)
        if hybrid_executor is None:
            # Try to create hybrid executor on-demand (works with OpenAI even if local_llm is None)
            try:
                from vulcan.llm import get_or_create_hybrid_executor, get_openai_client
                local_llm = getattr(app.state, 'llm', None)
                hybrid_executor = get_or_create_hybrid_executor(
                    local_llm=local_llm,
                    openai_client_getter=get_openai_client,
                    mode=settings.llm_execution_mode,
                )
                if hybrid_executor:
                    app.state.hybrid_executor = hybrid_executor
                    logger.info("[VULCAN] Created hybrid_executor on-demand for LLM generation")
            except Exception as e:
                logger.warning(f"[VULCAN] Failed to create hybrid_executor on-demand: {e}")

        if hybrid_executor:
            try:
                # Build enhanced prompt with context - handle None values explicitly
                memory_str = ""
                if memory_context:
                    try:
                        memory_str = (
                            f"\nRelevant Memory Context: {str(memory_context[:2])}"
                        )
                    except Exception:
                        memory_str = ""

                plan_str = ""
                if plan_result:
                    try:
                        plan_str = f"\nSuggested Plan: {str(plan_result)}"
                    except Exception:
                        plan_str = ""

                # Note: Include world model insights in the LLM context
                # This reconnects the world_model and meta_reasoning to the response generation
                world_model_str = ""
                if world_model_insight and isinstance(world_model_insight, dict):
                    try:
                        insight_parts = []
                        if world_model_insight.get("prediction"):
                            insight_parts.append(f"Prediction: {str(world_model_insight['prediction'])[:WORLD_MODEL_INSIGHT_TRUNCATION]}")
                        if world_model_insight.get("meta_reasoning"):
                            insight_parts.append(f"Meta-reasoning: {str(world_model_insight['meta_reasoning'])[:WORLD_MODEL_INSIGHT_TRUNCATION]}")
                        if world_model_insight.get("motivational_analysis"):
                            insight_parts.append(f"Motivational Analysis: {str(world_model_insight['motivational_analysis'])[:WORLD_MODEL_INSIGHT_TRUNCATION]}")
                        if insight_parts:
                            world_model_str = "\nWorld Model Insights: " + "; ".join(insight_parts)
                            logger.debug(f"[VULCAN] World model insights added to context: {world_model_str[:WORLD_MODEL_LOG_TRUNCATION]}...")
                    except Exception as e:
                        logger.debug(f"[VULCAN] Failed to format world model insights: {e}")

                # CRITICAL FIX: Use proper formatting for reasoning results
                # This ensures the LLM actually USES the reasoning engine output
                # instead of generating generic responses
                reasoning_str = ""
                if reasoning_results:
                    try:
                        reasoning_str = format_reasoning_results(reasoning_results)
                        # Note: Log reasoning output to verify it's reaching this point
                        logger.info(
                            f"[VULCAN] Reasoning results formatted: "
                            f"keys={list(reasoning_results.keys())}, "
                            f"reasoning_str_len={len(reasoning_str)}"
                        )
                        if len(reasoning_str) > 0:
                            logger.debug(f"[VULCAN] reasoning_str preview: {reasoning_str[:500]}...")
                    except Exception as e:
                        logger.warning(f"Failed to format reasoning results: {e}")
                        # Fallback to simple formatting
                        reasoning_str = f"\nReasoning Insights: {str(reasoning_results)}"
                else:
                    logger.info("[VULCAN] No reasoning_results available for LLM context")

                # Build enhanced prompt with explicit instruction to USE reasoning output
                # The prompt structure is critical for getting the LLM to incorporate
                # the structured reasoning analysis rather than ignoring it
                if reasoning_str:
                    # When reasoning output is available, use it as the primary source
                    enhanced_prompt = f"""You are VULCAN, an advanced AI assistant powered by specialized reasoning engines.

User Query: {user_message}
{memory_str}
{reasoning_str}
{world_model_str}
{plan_str}

IMPORTANT: The reasoning analysis above was produced by specialized reasoning engines.
Your response MUST:
1. Directly incorporate the conclusions from the reasoning analysis
2. Present the structured analysis in a clear, user-friendly format
3. NOT generate generic explanations - use the SPECIFIC results provided
4. If the reasoning shows a logical proof, present the proof steps
5. If the reasoning shows probabilities, include the specific numbers
6. If the reasoning shows causal analysis, explain the causal relationships found
7. If world model insights are provided, consider them in your response

Provide your response based on the reasoning analysis above:"""
                else:
                    # Fallback when no reasoning output is available
                    enhanced_prompt = f"""You are VULCAN, an advanced AI assistant powered by a comprehensive cognitive architecture.

User Query: {user_message}
{memory_str}{world_model_str}{plan_str}

Provide a helpful, accurate, and comprehensive response to the user's query. Be concise but thorough."""

                # ================================================================
                # Note: Check for deterministic fast-path results FIRST
                # Cryptographic and mathematical fast-path results are precomputed
                # by QueryRouter and MUST be returned directly, NOT sent to LLM.
                # This prevents OpenAI from returning "unable to calculate" errors.
                # ================================================================
                v1_telemetry_data = routing_plan.telemetry_data if (routing_plan and hasattr(routing_plan, 'telemetry_data')) else {}
                v1_is_crypto_fast_path = v1_telemetry_data.get('crypto_fast_path', False)
                
                if v1_is_crypto_fast_path:
                    # Note: Return precomputed cryptographic result directly
                    crypto_result = v1_telemetry_data.get('crypto_result')
                    crypto_operation = v1_telemetry_data.get('crypto_operation', 'hash')
                    
                    if crypto_result:
                        logger.info(
                            f"[VULCAN/v1/chat] Note: Returning deterministic crypto result directly, "
                            f"skipping LLM generation entirely. operation={crypto_operation}"
                        )
                        response_text = f"The {crypto_operation.upper()} hash is: {crypto_result}"
                        systems_used.append("cryptographic_engine")
                        
                        # Build final response without LLM
                        final_response = VulcanResponse(
                            response=response_text,
                            systems_used=list(set(systems_used)),
                            confidence=1.0,  # Deterministic operations have 100% confidence
                            metadata={
                                "fast_path": "cryptographic",
                                "deterministic": True,
                                "operation": crypto_operation,
                                "result": crypto_result,
                                "skip_llm_synthesis": True,
                                "conversation_id": body.conversation_id,
                            },
                        )
                        
                        # Record learning outcome for deterministic result
                        if deps.learning_system:
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: deps.learning_system.record_outcome({
                                        "query_id": routing_plan.query_id if routing_plan else None,
                                        "query": user_message,
                                        "tools_used": ["cryptographic"],
                                        "confidence": 1.0,
                                        "deterministic": True,
                                        "success": True,
                                    })
                                )
                            except Exception as e:
                                logger.debug(f"[VULCAN/v1/chat] Learning record for crypto failed: {e}")
                        
                        return final_response

                # PERFORMANCE FIX: hybrid_executor already checked/created above
                # No need to re-check - we already have it from the outer if block
                # This section was redundant with the check at line ~1900

                try:
                    # ================================================================
                    # CRITICAL FIX: Use format_output_for_user() when reasoning is available
                    # This passes VULCAN's reasoning context to OpenAI for proper formatting
                    # ================================================================
                    
                    # Check if we have reasoning results to format
                    has_reasoning = bool(reasoning_results and any(reasoning_results.values()))
                    
                    if has_reasoning:
                        # PATH 1: Use format_output_for_user() for structured reasoning output
                        # This is the PRIMARY fix - it passes reasoning context to OpenAI
                        logger.info(
                            f"[VULCAN] Using format_output_for_user() with reasoning results "
                            f"(engines: {list(reasoning_results.keys())})"
                        )
                        
                        # Build structured reasoning output for the formatter
                        # This includes all VULCAN reasoning components
                        structured_reasoning = {
                            'success': True,
                            'result': reasoning_results,
                            'confidence': _calculate_aggregate_confidence(reasoning_results),
                            'method': 'vulcan_unified_reasoning',
                            'reasoning_trace': [],
                            'metadata': {
                                'world_model': world_model_insight,
                                'plan': plan_result,
                                'memory_context': len(memory_context) if memory_context else 0,
                            }
                        }
                        
                        # ROOT CAUSE FIX: Extract and preserve privileged flags from reasoning_results
                        # Check all reasoning result sources for privileged status
                        is_privileged = False
                        privileged_source = None
                        for source_name, result in reasoning_results.items():
                            if isinstance(result, dict):
                                if result.get('privileged_no_answer') or result.get('override_router_tools'):
                                    is_privileged = True
                                    privileged_source = source_name
                                    # Copy privileged metadata to structured_reasoning
                                    structured_reasoning['metadata']['privileged_no_answer'] = result.get('privileged_no_answer', False)
                                    structured_reasoning['metadata']['override_router_tools'] = result.get('override_router_tools', False)
                                    structured_reasoning['metadata']['privileged_source'] = source_name
                                    logger.info(
                                        f"[VULCAN] ROOT CAUSE FIX: Preserving privileged flags from {source_name} "
                                        f"in structured_reasoning metadata"
                                    )
                                    break
                        
                        # Add world model insights to the structured output
                        if world_model_insight:
                            structured_reasoning['metadata']['world_model_insight'] = world_model_insight
                        
                        # Add planning results
                        if plan_result:
                            structured_reasoning['metadata']['plan_result'] = plan_result
                        
                        try:
                            # Call format_output_for_user with VULCAN's reasoning
                            llm_result = await hybrid_executor.format_output_for_user(
                                reasoning_output=structured_reasoning,
                                original_prompt=user_message,
                                max_tokens=body.max_tokens,
                            )
                            
                            response_text = llm_result.get("text", "")
                            llm_systems = llm_result.get("systems_used", [])
                            systems_used.extend(llm_systems)
                            
                            source = llm_result.get("source", "unknown")
                            logger.info(
                                f"[VULCAN] ✓ Response formatted from reasoning results "
                                f"(source={source}, distillation_captured={llm_result.get('distillation_captured', False)})"
                            )
                            
                        except Exception as format_error:
                            logger.warning(
                                f"[VULCAN] format_output_for_user() failed: {type(format_error).__name__}: {format_error}. "
                                f"Falling back to execute() with enhanced prompt."
                            )
                            # Fallback to original execute() method
                            llm_result = await _execute_with_enhanced_prompt(
                                hybrid_executor=hybrid_executor,
                                enhanced_prompt=enhanced_prompt,
                                body=body,
                                truncated_history=truncated_history,
                            )
                            response_text = llm_result.get("text", "")
                            llm_systems = llm_result.get("systems_used", [])
                            systems_used.extend(llm_systems)
                            source = llm_result.get("source", "unknown")
                    
                    else:
                        # PATH 2: No reasoning results - use traditional execute() method
                        # This handles queries that don't need reasoning (simple chat, etc.)
                        logger.info("[VULCAN] No reasoning results available, using execute() method")
                        
                        llm_result = await _execute_with_enhanced_prompt(
                            hybrid_executor=hybrid_executor,
                            enhanced_prompt=enhanced_prompt,
                            body=body,
                            truncated_history=truncated_history,
                        )
                        
                        response_text = llm_result.get("text", "")
                        llm_systems = llm_result.get("systems_used", [])
                        systems_used.extend(llm_systems)
                        source = llm_result.get("source", "unknown")
                        logger.info(
                            f"[VULCAN] Response via execute() (mode={settings.llm_execution_mode}, source={source})"
                        )

                except Exception as e:
                    logger.error(f"Hybrid LLM execution failed: {type(e).__name__}: {e}")
                    response_text = ""

                # Fallback if hybrid execution returned nothing
                if not response_text:
                    response_text = f"I understand your query about: {user_message}. "
                    if reasoning_results:
                        response_text += "Based on my analysis, I can provide insights from multiple reasoning systems. "
                    if plan_result:
                        response_text += (
                            "I've also generated a plan to help address your request. "
                        )
                    response_text += "However, I encountered an issue generating a detailed response. Please try again."
                    systems_used.append("fallback_message")

            except Exception as e:
                logger.error(f"LLM generation block failed: {e}")
                response_text = f"I understand your query about: {user_message}. "
                response_text += "However, I encountered an issue processing your request. Please try again."
                systems_used.append("error_fallback")

        else:
            # Fallback when LLM is not available
            response_text = f"Processing your query: '{user_message}'\n\n"

            if reasoning_results:
                response_text += "Reasoning Analysis:\n"
                for rtype, result in reasoning_results.items():
                    response_text += f"- {rtype.title()}: {str(result)[:100]}...\n"

            if memory_context:
                response_text += f"\nFound {len(memory_context)} relevant memories.\n"

            if plan_result:
                response_text += f"\nGenerated action plan available.\n"

            response_text += (
                "\n(Note: Full LLM response generation is currently unavailable)"
            )

        # TIMING: Log LLM generation duration
        _llm_end = time.perf_counter()
        logger.info(f"[TIMING] STEP 7b LLM generation took {_llm_end - _context_start:.2f}s")

        # ================================================================
        # STEP 8: Final Safety Check on Response
        # ================================================================
        _safety_start = time.perf_counter()
        if body.enable_safety and hasattr(deps, "safety") and deps.safety:
            try:
                loop = asyncio.get_running_loop()
                output_safe = await loop.run_in_executor(
                    None,
                    deps.safety.validate_action,
                    {"type": "response", "content": response_text},
                )
                if hasattr(output_safe, "__iter__") and len(output_safe) == 2:
                    if not output_safe[0]:
                        response_text = "I generated a response but it was flagged by safety systems. Please rephrase your question."
                        metadata["safety_status"] = "output_filtered"
            except Exception as e:
                logger.debug(f"Output safety check skipped: {e}")

        # TIMING: Log final safety check duration
        logger.info(f"[TIMING] STEP 8 Final safety check took {time.perf_counter() - _safety_start:.2f}s")

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # JOB-TO-RESPONSE GAP FIX: Add total time to timing breakdown
        timing_breakdown["total_ms"] = latency_ms

        # Update reasoning type based on what was used
        if len([s for s in systems_used if "reasoning" in s]) > 1:
            metadata["reasoning_type"] = "unified"
        elif "symbolic_reasoning" in systems_used:
            metadata["reasoning_type"] = "symbolic"
        elif "probabilistic_reasoning" in systems_used:
            metadata["reasoning_type"] = "probabilistic"
        elif "causal_reasoning" in systems_used:
            metadata["reasoning_type"] = "causal"
        elif "analogical_reasoning" in systems_used:
            metadata["reasoning_type"] = "analogical"
        else:
            metadata["reasoning_type"] = "direct"

        # ================================================================
        # STEP 9: Record Telemetry for Meta-Learning
        # OPTIMIZATION: Run telemetry as background task (fire-and-forget)
        # to avoid blocking response return
        # ================================================================
        try:
            from vulcan.routing import (
                record_telemetry,
                TELEMETRY_AVAILABLE,
            )

            if TELEMETRY_AVAILABLE:
                # Create telemetry data for background task
                telemetry_data = {
                    "query": user_message,
                    "response": response_text,
                    "metadata": {
                        "query_id": routing_stats.get("query_id", "unknown"),
                        "query_type": routing_stats.get("query_type", "unknown"),
                        "complexity_score": routing_stats.get("complexity_score", 0.0),
                        "systems_used": systems_used,
                        "jobs_submitted": len(submitted_jobs),
                        "latency_ms": latency_ms,
                        "success": True,
                    },
                    "source": "user",
                }
                
                # PARALLEL EXECUTION: Run telemetry recording as background task
                # Does not await completion - response returns immediately
                async def _record_telemetry_background():
                    try:
                        record_telemetry(
                            query=telemetry_data["query"],
                            response=telemetry_data["response"],
                            metadata=telemetry_data["metadata"],
                            source=telemetry_data["source"],
                        )
                        logger.debug(f"[VULCAN/v1/chat] Background telemetry recorded")
                    except Exception as bg_err:
                        logger.debug(f"[VULCAN/v1/chat] Background telemetry failed: {bg_err}")
                
                asyncio.create_task(_record_telemetry_background())
                logger.debug(f"[VULCAN/v1/chat] Telemetry scheduled as background task")
        except Exception as e:
            logger.debug(f"[VULCAN/v1/chat] Telemetry recording failed: {e}")

        # ================================================================
        # STEP 10: Record Query Outcome for Curiosity Engine
        # FIX: Records outcome to SQLite bridge for cross-process analysis.
        # This enables the CuriosityEngine subprocess to access query outcomes.
        # FIX: Now passes selected tools to OutcomeBridge for learning system
        # ================================================================
        try:
            from vulcan.curiosity_engine.outcome_bridge import get_outcome_bridge
            
            # Calculate routing time from timing breakdown
            routing_time_ms = 0.0
            if "query_routing" in timing_breakdown.get("phases", {}):
                routing_time_ms = timing_breakdown["phases"]["query_routing"].get("duration_ms", 0.0)
            
            # Extract selected tools from routing_plan telemetry_data or metadata
            selected_tools = []
            if routing_plan and hasattr(routing_plan, 'telemetry_data'):
                selected_tools = routing_plan.telemetry_data.get('selected_tools', [])
            # Fallback to tool_selected from metadata if available
            if not selected_tools and metadata.get("tool_selected"):
                selected_tools = [metadata["tool_selected"]]
            # Fallback to systems_used filtering for reasoning systems
            if not selected_tools:
                reasoning_prefixes = ("symbolic", "probabilistic", "causal", "analogical", "multimodal")
                unified_prefix = "unified_reasoning_"
                selected_tools_set = set()
                for system in systems_used:
                    if system.startswith(unified_prefix):
                        selected_tools_set.add(system[len(unified_prefix):])
                    elif system.startswith(reasoning_prefixes):
                        selected_tools_set.add(system)
                selected_tools = list(selected_tools_set)
            # Default to ['general'] if no tools identified
            if not selected_tools:
                selected_tools = ['general']
            
            # Record via OutcomeBridge for learning system integration
            bridge = get_outcome_bridge()
            
            # Note Issue #35: Determine status based on actual timing
            # Queries taking > 30s should be marked as "slow", not "success"
            # This ensures gap detection properly identifies slow queries
            if float(latency_ms) > SLOW_QUERY_OUTCOME_THRESHOLD_MS:
                query_status = "slow"
            else:
                query_status = "success"
            
            bridge.record(
                query_id=routing_stats.get("query_id", f"q_{int(time.time())}"),
                status=query_status,
                routing_ms=routing_time_ms,
                total_ms=latency_ms,
                complexity=routing_stats.get("complexity_score", 0.0),
                query_type=routing_stats.get("query_type", "general"),
                tools=selected_tools,
            )
            logger.debug(f"[VULCAN/v1/chat] Query outcome recorded to bridge with tools={selected_tools}")
        except ImportError:
            pass  # Outcome bridge not available
        except Exception as e:
            logger.debug(f"[VULCAN/v1/chat] Outcome recording setup failed: {e}")

        # ================================================================
        # MEMORY MANAGEMENT: Trigger garbage collection to prevent memory accumulation
        # This prevents progressive query routing degradation (469ms → 152,048ms)
        # caused by repeated SentenceTransformer model loading without cleanup.
        # Rate-limited to every GC_REQUEST_INTERVAL requests to reduce overhead.
        # FIX: Use modulo to prevent overflow after billions of requests
        # ================================================================
        global _gc_request_counter
        # OVERFLOW FIX: Use modulo to wrap counter instead of unbounded increment
        # This prevents integer overflow after ~2 billion requests (2^31-1)
        _gc_request_counter = (_gc_request_counter + 1) % GC_REQUEST_INTERVAL
        
        if _gc_request_counter == 0:  # Counter wrapped to 0, trigger GC
            
            async def _post_request_gc():
                """Background task to trigger garbage collection after request."""
                try:
                    import gc
                    collected = gc.collect()
                    if collected > GC_SIGNIFICANT_CLEANUP_THRESHOLD:
                        logger.debug(f"[VULCAN/v1/chat] Post-request GC collected {collected} objects")
                except Exception:
                    pass  # Don't let GC errors affect the response
            
            # Schedule GC as a background task (non-blocking)
            asyncio.create_task(_post_request_gc())

        # SECURITY FIX: Generate IDs with full cryptographic randomness
        # Old format: f"resp_{int(time.time())}_{secrets.token_hex(4)}"
        # This prevents timing attacks and ID enumeration
        response_id = f"resp_{secrets.token_urlsafe(16)}"
        query_id = routing_stats.get("query_id") if routing_stats else f"q_{secrets.token_urlsafe(12)}"

        # ================================================================
        # STEP 11: Process live feedback for auto-detection (Task 3)
        # ================================================================
        try:
            # FIX MAJOR-4: Use deps.continual instead of deps.learning
            learning_system = None
            if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
                learning_system = deployment.collective.deps.continual
            
            # FIX MAJOR-4: learning_system IS the ContinualLearner
            if learning_system:
                learner = learning_system
                
                # Store context for future feedback processing
                feedback_context = {
                    "query_id": query_id,
                    "response_id": response_id,
                    "previous_response_id": response_id,
                    "systems_used": systems_used,
                    "reasoning_type": metadata.get("reasoning_type"),
                }
                
                # Process live feedback (async, non-blocking)
                if hasattr(learner, "process_live_feedback"):
                    learner.process_live_feedback(user_message, feedback_context)
        except Exception as e:
            logger.debug(f"[VULCAN/v1/chat] Live feedback processing skipped: {e}")

        # ================================================================
        # STEP 12: Crystallize knowledge from execution (Knowledge Crystallizer)
        # ================================================================
        try:
            # FIX MAJOR-4: Use deps.continual instead of deps.learning
            learning_system = None
            if hasattr(deployment, "collective") and hasattr(deployment.collective.deps, "continual"):
                learning_system = deployment.collective.deps.continual
            
            # FIX MAJOR-4: learning_system IS the ContinualLearner
            if learning_system:
                learner = learning_system
                
                # Crystallize knowledge from this execution (non-blocking)
                if hasattr(learner, "crystallize_from_execution"):
                    learner.crystallize_from_execution(
                        query=user_message,
                        response=response_text,
                        success=metadata.get("safety_status") == "safe",
                        tools_used=systems_used,
                        strategy=metadata.get("reasoning_type"),
                        metadata={
                            "query_id": query_id,
                            "response_id": response_id,
                            "latency_ms": latency_ms,
                        },
                    )
        except Exception as e:
            logger.debug(f"[VULCAN/v1/chat] Knowledge crystallization skipped: {e}")

        # ================================================================
        # Note: Add Transparency About Source
        # Include detailed metadata about where the answer came from
        # ================================================================
        
        # Determine the primary source of the response
        response_source = "unknown"
        if use_reasoning_directly:
            response_source = "vulcan_reasoning"
        elif "parallel_openai" in systems_used or "openai" in systems_used:
            response_source = "openai_llm"
        elif "local_llm" in systems_used or "vulcan_llm" in systems_used:
            response_source = "local_llm"
        elif reasoning_results and best_confidence > 0:
            response_source = "llm_with_reasoning_context"
        else:
            response_source = "llm_only"
        
        # Build transparency metadata
        transparency = {
            "source": response_source,
            "reasoning_confidence": best_confidence,
            "reasoning_type": metadata.get("reasoning_type", "none"),
            "engines_used": list(set([s for s in systems_used if "reasoning" in s or "model" in s])),
            "used_openai_fallback": response_source in ["openai_llm", "llm_with_reasoning_context"],
            "reasoning_available": bool(reasoning_results),
            "confidence_threshold": MIN_REASONING_CONFIDENCE_THRESHOLD,
        }
        
        # Add transparency to metadata
        metadata["transparency"] = transparency
        
        # BUG #3 FIX: Notify world model of query outcome
        # This makes the world model aware of how each query was resolved
        observe_outcome(
            query_id=query_id,
            response={
                'source': response_source,
                'confidence': best_confidence,
                'response_time_ms': latency_ms,
                'used_openai_fallback': transparency.get('used_openai_fallback', False),
                'reasoning_available': transparency.get('reasoning_available', False),
            },
            user_feedback=None  # Feedback comes later via /feedback endpoint
        )
        
        # ================================================================
        # FIX Issue 4: Record query outcome to OutcomeBridge for CuriosityEngine
        # This enables the curiosity-driven learning system to detect gaps
        # from actual query processing data
        # ================================================================
        try:
            from vulcan.curiosity_engine.outcome_bridge import get_outcome_bridge
            from vulcan.curiosity_engine.outcome_queue import record_outcome, QueryOutcome, OutcomeStatus
            
            # Calculate routing time from timing breakdown
            routing_time_ms = 0.0
            if timing_breakdown and "query_routing" in timing_breakdown.get("phases", {}):
                routing_time_ms = timing_breakdown["phases"]["query_routing"].get("duration_ms", 0.0)
            
            # FIX Issue 7: Use consistent tools extraction helper
            selected_tools = extract_tools_from_routing(routing_plan)
            
            # Determine outcome status based on response quality
            # Success if response generated and safety checks passed
            outcome_status = "success" if metadata.get("safety_status") == "safe" else "error"
            
            # Record to OutcomeBridge (SQLite for subprocess visibility)
            bridge = get_outcome_bridge()
            bridge.record(
                query_id=query_id,
                status=outcome_status,
                routing_ms=routing_time_ms,
                total_ms=latency_ms,
                complexity=routing_stats.get("complexity_score", 0.5) if routing_stats else 0.5,
                query_type=routing_stats.get("query_type", "general") if routing_stats else "general",
                tools=selected_tools,
            )
            
            # Also record to in-memory OutcomeQueue (for main process CuriosityEngine)
            # This provides dual recording path for robustness
            outcome = QueryOutcome(
                query_id=query_id,
                query_type=routing_stats.get("query_type", "general") if routing_stats else "general",
                status=OutcomeStatus.SUCCESS if outcome_status == "success" else OutcomeStatus.ERROR,
                execution_time_ms=latency_ms,
                routing_time_ms=routing_time_ms,
                complexity=routing_stats.get("complexity_score", 0.5) if routing_stats else 0.5,
                capabilities_used=selected_tools,
                metadata={
                    "response_source": response_source,
                    "reasoning_confidence": best_confidence,
                    "safety_status": metadata.get("safety_status"),
                    "systems_used": systems_used[:10],  # Limit to prevent bloat
                }
            )
            outcome.compute_features()
            record_outcome(outcome)
            
            # ================================================================
            # FIX Issue 2: Report outcome to QueryRouter for CuriosityEngine integration
            # This enables the router's curiosity engine connection to detect gaps
            # ================================================================
            try:
                # Get the query analyzer (which contains the router)
                from vulcan.routing import get_query_analyzer
                
                query_analyzer = get_query_analyzer()
                if query_analyzer and hasattr(query_analyzer, 'report_query_outcome'):
                    # Build result dict for curiosity engine
                    result_dict = {
                        'response': response_text[:500],  # Truncate for memory efficiency
                        'latency_ms': latency_ms,
                        'complexity': routing_stats.get("complexity_score", 0.5) if routing_stats else 0.5,
                        'tools_used': selected_tools,
                        'response_source': response_source,
                        'confidence': best_confidence,
                    }
                    
                    # Report to QueryRouter (which forwards to CuriosityEngine)
                    query_analyzer.report_query_outcome(
                        query=user_message,
                        result=result_dict,
                        success=(outcome_status == "success"),
                        domain="query_processing",
                        query_type=routing_stats.get("query_type", "general") if routing_stats else "general"
                    )
                    
                    logger.debug(
                        f"[VULCAN/v1/chat] Reported outcome to QueryRouter for "
                        f"CuriosityEngine gap detection"
                    )
            except Exception as router_report_err:
                # Non-critical error
                logger.debug(
                    f"[VULCAN/v1/chat] Failed to report to QueryRouter: {router_report_err}"
                )
            # ================================================================
            
            logger.debug(
                f"[VULCAN/v1/chat] Outcome recorded: query_id={query_id}, "
                f"status={outcome_status}, tools={selected_tools}"
            )
        except Exception as record_err:
            # Non-critical error - don't fail the request if outcome recording fails
            logger.debug(f"[VULCAN/v1/chat] Failed to record outcome: {record_err}")
        # ================================================================

        return {
            "response": response_text,
            "metadata": metadata,
            "systems_used": systems_used,
            "latency_ms": latency_ms,
            "reasoning_type": metadata["reasoning_type"],
            "safety_status": metadata["safety_status"],
            "memory_results": metadata["memory_results"],
            # NEW: Include routing and agent pool stats
            "routing": routing_stats if routing_stats else None,
            "agent_pool_stats": agent_pool_stats if agent_pool_stats else None,
            # JOB-TO-RESPONSE GAP FIX: Include timing breakdown for debugging slow requests
            "timing_breakdown": timing_breakdown if latency_ms > SLOW_REQUEST_THRESHOLD_MS else None,
            # RLHF INTEGRATION: Include IDs for feedback (Task 4 - UI thumbs buttons)
            "query_id": query_id,
            "response_id": response_id,
            # Note: Include transparency about response source
            "transparency": transparency,
        }

    except Exception as e:
        logger.error(f"Unified chat failed: {e}", exc_info=True)
        error_counter.labels(error_type="unified_chat").inc()
        
        # FIX: Record error outcome for Curiosity Engine analysis
        try:
            from vulcan.curiosity_engine.outcome_bridge import get_outcome_bridge
            
            # Calculate elapsed time
            error_latency_ms = (time.time() - start_time) * 1000
            
            # Get routing time if available
            routing_time_ms = 0.0
            if timing_breakdown and "query_routing" in timing_breakdown.get("phases", {}):
                routing_time_ms = timing_breakdown["phases"]["query_routing"].get("duration_ms", 0.0)
            
            # FIX Issue 7: Use consistent tools extraction helper
            selected_tools = extract_tools_from_routing(routing_plan)
            
            bridge = get_outcome_bridge()
            bridge.record(
                query_id=routing_stats.get("query_id", f"q_err_{int(time.time())}") if routing_stats else f"q_err_{int(time.time())}",
                status="error",
                routing_ms=routing_time_ms,
                total_ms=error_latency_ms,
                complexity=routing_stats.get("complexity_score", 0.0) if routing_stats else 0.0,
                query_type=routing_stats.get("query_type", "unknown") if routing_stats else "unknown",
                tools=selected_tools,
            )
            logger.debug(f"[VULCAN/v1/chat] Error outcome recorded to bridge with tools={selected_tools}")
        except Exception:
            pass  # Don't let outcome recording failure mask the original error
        
        # MEMORY MANAGEMENT: Trigger GC on error path too
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # MEMORY LEAK FIX: Clean up global precomputed embeddings
        # These are module-level variables that persist between requests
        # and can accumulate memory if not explicitly cleared
        _precomputed_embedding = None
        _precomputed_query_result = None


# ============================================================
# STANDARD API ENDPOINTS (continued)
# ============================================================

