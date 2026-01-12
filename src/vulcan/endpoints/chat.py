"""
Legacy Chat Endpoint

VULCAN-FIRST conversational interface using cognitive architecture.
Uses VULCAN systems as PRIMARY intelligence with LLM fallback.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.chat_helpers import (
    format_reasoning_results,  # Fix: Import reasoning formatter for LLM context
    CONTEXT_TRUNCATION_LIMITS,
    MIN_MEANINGFUL_RESPONSE_LENGTH,
    MOCK_RESPONSE_MARKER,
    AGENT_REASONING_POLL_DELAY_SEC,
    MAX_AGENT_REASONING_JOBS_TO_CHECK,
)
from vulcan.endpoints.utils import require_deployment

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _calculate_aggregate_confidence_chat(reasoning_insights: Dict[str, Any]) -> float:
    """
    Calculate aggregate confidence score from multiple reasoning engines.
    
    This is a duplicate of the function in unified_chat.py but kept separate
    to maintain independence between the legacy and unified endpoints.
    
    Industry Standard: Weighted averaging with proper handling of edge cases.
    Uses harmonic mean for conservative confidence estimation.
    
    Args:
        reasoning_insights: Dictionary mapping engine names to their results
    
    Returns:
        Aggregate confidence score between 0.0 and 1.0.
        Returns 0.5 if no valid confidence scores found.
    """
    if not reasoning_insights:
        return 0.5
    
    confidence_scores = []
    
    for engine_result in reasoning_insights.values():
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
    try:
        harmonic_mean = len(confidence_scores) / sum(1.0 / c if c > 0 else float('inf') for c in confidence_scores)
        return max(0.0, min(1.0, harmonic_mean))  # Clamp to [0, 1]
    except (ZeroDivisionError, ValueError):
        # Fallback to arithmetic mean if harmonic mean fails
        return sum(confidence_scores) / len(confidence_scores)


@router.post("/llm/chat")
async def chat(request: Request) -> Dict[str, Any]:
    """Conversational interface via VULCAN's cognitive architecture.

    VULCAN-FIRST DESIGN: Uses Vulcan's own cognitive systems (memory, reasoning,
    world model, agent pool) as the PRIMARY intelligence engine. External LLMs
    (like OpenAI) are only used as a FALLBACK for language generation when
    Vulcan's local systems cannot produce a response.

    Complete Cognitive Pipeline:
    1. Route through Agent Pool for distributed processing (with job tracking)
    2. INPUT GATEKEEPER: Validate query, detect nonsense/hallucination triggers
    3. Query Long-Term Memory for relevant context
    4. Apply ALL Reasoning Systems (Symbolic, Probabilistic, Causal, Analogical)
    5. World Model Integration (Predictions, Counterfactuals, Causal Graph)
    6. Meta-Reasoning Layer (Goal conflict detection, Objective negotiation)
    7. Generate response using Vulcan's local LLM (or OpenAI fallback)
    8. OUTPUT GATEKEEPER: Validate response against ground truth
    """
    # Configuration constants for response building
    CONTEXT_TRUNCATION_LIMITS = {
        "memory": 300,
        "reasoning": 400,
        "world_model": 300,
        "meta_reasoning": 200,
    }
    MIN_MEANINGFUL_RESPONSE_LENGTH = 10
    MOCK_RESPONSE_MARKER = "Mock response"
    
    # Agent reasoning collection constants
    AGENT_REASONING_POLL_DELAY_SEC = 0.1  # Brief wait for jobs to complete
    MAX_AGENT_REASONING_JOBS_TO_CHECK = 3  # Limit jobs to check for reasoning output

    # Get the FastAPI app from the request to access app.state
    app = request.app
    
    # Get deployment using utility that handles both standalone and mounted sub-app scenarios
    deployment = require_deployment(request)
    collective = deployment.collective
    deps = collective.deps

    systems_used = []
    memory_context = None
    reasoning_insights = {}
    world_model_insights = {}
    meta_reasoning_insights = {}
    gatekeeper_results = {}
    agent_pool_stats = {}

    loop = asyncio.get_running_loop()

    # ================================================================
    # STEP -1: QUERY ROUTING LAYER - Analyze query BEFORE all processing
    # This is the critical integration that routes queries to the right systems
    # ================================================================
    routing_plan = None
    routing_stats = {}
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
            routing_plan = await route_query_async(request.prompt, source="user")
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
                "governance_sensitivity": routing_plan.governance_sensitivity.value,
                "pii_detected": routing_plan.pii_detected,
                "sensitive_topics": routing_plan.sensitive_topics,
            }

            logger.info(
                f"[VULCAN] Query routed: id={routing_plan.query_id}, "
                f"type={routing_plan.query_type.value}, tasks={len(routing_plan.agent_tasks)}, "
                f"collab={routing_plan.collaboration_needed}, arena={routing_plan.arena_participation}"
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
                        "sensitive_topics": routing_plan.sensitive_topics,
                        "governance_sensitivity": routing_plan.governance_sensitivity.value,
                    },
                    severity=(
                        "warning"
                        if routing_plan.governance_sensitivity.value
                        in ("high", "critical")
                        else "info"
                    ),
                    query_id=routing_plan.query_id,
                )
                systems_used.append("governance_logger")
                # FIX: Governance Logger Waste - reduce log noise for routine governance logging
                # Only log at INFO level for high/critical sensitivity, DEBUG for routine
                if routing_plan.governance_sensitivity.value in ("high", "critical"):
                    logger.info(
                        f"[VULCAN] Governance logged for query {routing_plan.query_id} "
                        f"(sensitivity: {routing_plan.governance_sensitivity.value})"
                    )
                else:
                    logger.debug(
                        f"[VULCAN] Governance logged for query {routing_plan.query_id}"
                    )

    except ImportError as e:
        logger.debug(f"[VULCAN] Routing layer not available: {e}")
    except Exception as e:
        logger.warning(f"[VULCAN] Query routing failed: {e}", exc_info=True)

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
            f"[VULCAN] Routing to Arena for graph execution + agent training: "
            f"query_id={routing_plan.query_id}"
        )
        systems_used.append("graphix_arena")
        
        try:
            arena_result = await _execute_via_arena(
                query=request.prompt,
                routing_plan=routing_plan,
            )
            
            if arena_result.get("status") == "success":
                logger.info(
                    f"[VULCAN] Arena execution successful: agent={arena_result.get('agent_id')}, "
                    f"time={arena_result.get('execution_time', 0):.2f}s"
                )
                
                # If Arena returned a complete result, we can use it directly
                # but we'll still run through VULCAN's cognitive pipeline for enrichment
                arena_output = arena_result.get("result", {})
                
                # Check if Arena result has a direct response we can use
                if isinstance(arena_output, dict) and arena_output.get("output"):
                    # Arena provided a complete response - return it with metadata
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
                            "arena_processed": True,
                            "query_id": routing_plan.query_id,
                        },
                    }
            else:
                # Arena execution failed - log and continue with VULCAN processing
                logger.warning(
                    f"[VULCAN] Arena execution failed: {arena_result.get('status')}, "
                    f"error={arena_result.get('error', 'unknown')}, falling back to VULCAN"
                )
                
        except Exception as arena_err:
            logger.warning(
                f"[VULCAN] Arena execution error: {arena_err}, falling back to VULCAN"
            )
            arena_result = {"status": "error", "error": str(arena_err)}

    # FIX: Inject Arena reasoning output into reasoning_insights when available
    # This ensures Arena's reasoning analysis is included in LLM context even when
    # Arena doesn't return a complete response (e.g., partial success, fallback)
    if arena_result and arena_result.get("status") == "success":
        arena_output = arena_result.get("result", {})
        if isinstance(arena_output, dict):
            # Extract reasoning information from Arena result
            arena_reasoning = {
                "agent_id": arena_result.get("agent_id"),
                "reasoning_invoked": arena_output.get("reasoning_invoked", False),
                "selected_tools": arena_output.get("selected_tools", []),
                "reasoning_strategy": arena_output.get("reasoning_strategy"),
                "confidence": arena_output.get("confidence"),
            }
            # Only add if Arena actually invoked reasoning
            if arena_reasoning.get("reasoning_invoked") or arena_reasoning.get("selected_tools"):
                reasoning_insights["arena_reasoning"] = arena_reasoning
                logger.info(
                    f"[VULCAN] Arena reasoning injected into context: "
                    f"tools={arena_reasoning.get('selected_tools')}, "
                    f"strategy={arena_reasoning.get('reasoning_strategy')}"
                )

    # ================================================================
    # STEP 0: INPUT GATEKEEPER - Validate query before processing
    # ================================================================
    # TIMING: Capture start time to identify gaps between steps
    _step0_start_time = time.perf_counter()
    
    try:
        # Use LLM validators to detect nonsense queries and potential issues
        from vulcan.safety.llm_validators import (
            EnhancedSafetyValidator as LLMSafetyValidator,
        )

        input_validator = LLMSafetyValidator()
        if deps.world_model:
            input_validator.attach_world_model(deps.world_model)

        # Check for prompt injection and nonsense
        validated_input = input_validator.validate_generation(
            request.prompt, {"role": "user", "world_model": deps.world_model}
        )

        input_events = input_validator.get_events()
        if input_events:
            gatekeeper_results["input_validation"] = {
                "modified": validated_input != request.prompt,
                "events": len(input_events),
                "event_types": list(set(e["kind"] for e in input_events)),
            }
            systems_used.append("input_gatekeeper")
            logger.info(
                f"[VULCAN] Input gatekeeper: {len(input_events)} events detected"
            )

        # Use validated input for processing
        processed_prompt = (
            validated_input if validated_input != "[NEUTRALIZED]" else request.prompt
        )

    except Exception as e:
        logger.debug(f"[VULCAN] Input gatekeeper failed: {e}")
        processed_prompt = request.prompt

    # ================================================================
    # STEP 1: Route Through Agent Pool - Use routing plan's tasks!
    # ================================================================
    job_id = None
    submitted_jobs = []  # Track all submitted job IDs
    tool_selector = None  # Priority 2: Tool selector for reasoning system integration
    try:
        # Import at the start to catch import errors
        from vulcan.orchestrator.agent_lifecycle import AgentCapability
        import uuid
        
        # Priority 2 FIX: Import and initialize tool selector from reasoning system
        try:
            from vulcan.reasoning.selection import create_tool_selector
            tool_selector = create_tool_selector()
            logger.info("[VULCAN] Tool selector initialized from reasoning system")
        except ImportError as ts_err:
            logger.debug(f"[VULCAN] Tool selector not available: {ts_err}")
            tool_selector = None
        except Exception as ts_err:
            logger.warning(f"[VULCAN] Tool selector initialization failed: {ts_err}")
            tool_selector = None

        if collective.agent_pool:
            pool_status = collective.agent_pool.get_pool_status()
            agent_pool_stats = {
                "total_agents": pool_status.get("total_agents", 0),
                "idle_agents": pool_status.get("state_distribution", {}).get("idle", 0),
                "working_agents": pool_status.get("state_distribution", {}).get(
                    "working", 0
                ),
                "jobs_submitted_total": collective.agent_pool.stats.get(
                    "total_jobs_submitted", 0
                ),
                "jobs_completed_total": collective.agent_pool.stats.get(
                    "total_jobs_completed", 0
                ),
            }

            # Get timeout from config
            agent_pool_timeout = getattr(deployment.config, "agent_pool_timeout", 15.0)

            # Map capability string to enum (defined once, outside loop)
            capability_map = {
                "perception": AgentCapability.PERCEPTION,
                "reasoning": AgentCapability.REASONING,
                "planning": AgentCapability.PLANNING,
                "execution": AgentCapability.EXECUTION,
                "learning": AgentCapability.LEARNING,
            }

            # Helper function to update agent pool stats (avoid duplication)
            def _update_agent_pool_stats():
                agent_pool_stats["jobs_submitted_this_request"] = len(submitted_jobs)
                agent_pool_stats["jobs_submitted_total"] = (
                    collective.agent_pool.stats.get("total_jobs_submitted", 0)
                )
                agent_pool_stats["jobs_failed_total"] = collective.agent_pool.stats.get(
                    "total_jobs_failed", 0
                )
                if submitted_jobs:
                    agent_pool_stats["submitted_job_ids"] = submitted_jobs

            # ============================================================
            # USE ROUTING PLAN'S AGENT TASKS (if available)
            # This is the critical connection to the routing layer!
            # PERFORMANCE FIX: Submit tasks in PARALLEL using asyncio.gather()
            # instead of sequential blocking submission to avoid 29s delay
            # ============================================================
            if routing_plan and routing_plan.agent_tasks:
                logger.info(
                    f"[VULCAN] Using routing plan tasks: {len(routing_plan.agent_tasks)} tasks from plan {routing_plan.query_id}"
                )

                # FIX: Extract selected_tools from routing plan telemetry data
                # This enables reasoning engine invocation based on QueryRouter's tool selection
                routing_selected_tools = []
                if routing_plan and hasattr(routing_plan, 'telemetry_data'):
                    routing_selected_tools = routing_plan.telemetry_data.get("selected_tools", []) or []
                    if routing_selected_tools:
                        logger.info(
                            f"[VULCAN] Routing plan selected tools: {routing_selected_tools}"
                        )

                # PERFORMANCE FIX: Define async helper function for parallel task submission
                async def submit_single_task(agent_task, pool, capability_map, timeout, max_tokens, selected_tools_from_router):
                    """Submit a single task to agent pool asynchronously."""
                    capability = capability_map.get(
                        agent_task.capability, AgentCapability.REASONING
                    )
                    
                    # Priority 2 FIX: Use tool selector before submitting to agent pool
                    selected_tool = None
                    if tool_selector is not None:
                        try:
                            from vulcan.reasoning.selection import SelectionRequest, SelectionMode
                            # Create budget constraints from task parameters
                            budget = {
                                "time_budget_ms": (agent_task.timeout_seconds or timeout) * 1000,
                                "energy_budget_mj": 1000,  # Default energy budget
                            }
                            selection_request = SelectionRequest(
                                problem=agent_task.prompt,
                                constraints=budget,
                                mode=SelectionMode.BALANCED,
                            )
                            selection = tool_selector.select_and_execute(selection_request)
                            selected_tool = selection.selected_tool
                            logger.info(f"[ToolSelector] Selected: {selection.selected_tool}")
                        except Exception as sel_err:
                            logger.debug(f"[ToolSelector] Selection failed: {sel_err}")
                            selected_tool = None

                    # Create task graph from routing plan task
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
                                "params": {"max_tokens": max_tokens},
                            },
                        ],
                        "edges": [
                            {"from": "input", "to": "process"},
                            {"from": "process", "to": "output"},
                        ],
                    }

                    logger.info(
                        f"[VULCAN] Submitting routing task to agent pool: "
                        f"task_id={agent_task.task_id}, capability={agent_task.capability}, priority={agent_task.priority}, "
                        f"selected_tools={selected_tools_from_router}"
                    )

                    try:
                        # Run blocking submit_job in executor to avoid blocking event loop
                        inner_loop = asyncio.get_running_loop()
                        submitted_job_id = await inner_loop.run_in_executor(
                            None,
                            lambda: pool.submit_job(
                                graph=task_graph,
                                parameters={
                                    "prompt": agent_task.prompt,
                                    "task_type": agent_task.task_type,
                                    "source": agent_task.parameters.get("source", "user"),
                                    "is_primary": agent_task.parameters.get(
                                        "is_primary", True
                                    ),
                                    "selected_tool": selected_tool,  # Priority 2: Pass selected tool
                                    # FIX: Pass selected_tools from QueryRouter to enable reasoning invocation
                                    "selected_tools": selected_tools_from_router,
                                    **agent_task.parameters,
                                },
                                priority=agent_task.priority,
                                capability_required=capability,
                                timeout_seconds=agent_task.timeout_seconds or timeout,
                            )
                        )

                        if submitted_job_id:
                            logger.info(
                                f"[VULCAN] Task submitted successfully: {submitted_job_id}"
                            )
                            return {
                                "job_id": submitted_job_id,
                                "capability": agent_task.capability,
                                "task_id": agent_task.task_id,
                                "task_type": agent_task.task_type,
                            }
                        return None

                    except Exception as task_err:
                        logger.warning(
                            f"[VULCAN] Failed to submit task {agent_task.task_id}: {task_err}"
                        )
                        return None

                # PERFORMANCE FIX: Submit ALL tasks in parallel using asyncio.gather()
                # This prevents the 29s delay caused by sequential blocking submission
                task_coroutines = [
                    submit_single_task(
                        agent_task,
                        collective.agent_pool,
                        capability_map,
                        agent_pool_timeout,
                        request.max_tokens,
                        routing_selected_tools  # FIX: Pass selected_tools from router
                    )
                    for agent_task in routing_plan.agent_tasks
                ]

                # Wait for all tasks to complete in parallel
                task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

                # Process results and update tracking
                for result in task_results:
                    if isinstance(result, Exception):
                        logger.warning(f"[VULCAN] Task submission exception: {result}")
                        continue
                    if result and result.get("job_id"):
                        submitted_jobs.append(result["job_id"])
                        systems_used.append(f"agent_pool_{result['capability']}")

                        # Log task submission to governance (fire-and-forget)
                        if routing_plan.requires_audit:
                            try:
                                if GOVERNANCE_AVAILABLE:
                                    log_to_governance_fire_and_forget(
                                        action_type="agent_task_submitted",
                                        details={
                                            "task_id": result["task_id"],
                                            "job_id": result["job_id"],
                                            "capability": result["capability"],
                                            "task_type": result["task_type"],
                                        },
                                        severity="info",
                                        query_id=routing_plan.query_id,
                                    )
                            except NameError:
                                pass
                            except Exception as gov_err:
                                logger.debug(
                                    f"[VULCAN] Governance logging skipped: {gov_err}"
                                )

                logger.info(
                    f"[VULCAN] Parallel task submission complete: {len(submitted_jobs)} tasks submitted"
                )

                # Update stats after all submissions
                _update_agent_pool_stats()

                if submitted_jobs:
                    job_id = submitted_jobs[
                        0
                    ]  # Keep first job ID for backwards compatibility

            else:
                # ============================================================
                # FALLBACK: Create task from query analysis (if no routing plan)
                # ============================================================
                logger.info(
                    "[VULCAN] No routing plan tasks - using fallback query analysis"
                )

                # Determine query type for specialized agent routing
                query_lower = processed_prompt.lower()

                # Route based on query intent
                if any(
                    kw in query_lower
                    for kw in ["analyze", "pattern", "data", "observe"]
                ):
                    capability = AgentCapability.PERCEPTION
                    task_type = "perception_analysis"
                elif any(
                    kw in query_lower for kw in ["plan", "step", "strategy", "organize"]
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

                # Create specialized task graph
                task_graph = {
                    "id": f"{task_type}_{uuid.uuid4().hex[:12]}",
                    "type": task_type,
                    "capability": capability.value,
                    "nodes": [
                        {
                            "id": "input",
                            "type": "perception",
                            "params": {"input": processed_prompt},
                        },
                        {
                            "id": "process",
                            "type": capability.value,
                            "params": {"query": processed_prompt},
                        },
                        {
                            "id": "output",
                            "type": "generation",
                            "params": {"max_tokens": request.max_tokens},
                        },
                    ],
                    "edges": [
                        {"from": "input", "to": "process"},
                        {"from": "process", "to": "output"},
                    ],
                }
                
                # Priority 2 FIX: Use tool selector before fallback submission
                fallback_selected_tool = None
                if tool_selector is not None:
                    try:
                        from vulcan.reasoning.selection import SelectionRequest, SelectionMode
                        budget = {
                            "time_budget_ms": agent_pool_timeout * 1000,
                            "energy_budget_mj": 1000,
                        }
                        selection_request = SelectionRequest(
                            problem=processed_prompt,
                            constraints=budget,
                            mode=SelectionMode.BALANCED,
                        )
                        selection = tool_selector.select_and_execute(selection_request)
                        fallback_selected_tool = selection.selected_tool
                        logger.info(f"[ToolSelector] Selected: {selection.selected_tool}")
                    except Exception as sel_err:
                        logger.debug(f"[ToolSelector] Selection failed: {sel_err}")
                        fallback_selected_tool = None

                # Submit to agent pool
                logger.info(
                    f"[VULCAN] Submitting fallback job to agent pool (capability={capability.value}, timeout={agent_pool_timeout}s)"
                )

                job_id = collective.agent_pool.submit_job(
                    graph=task_graph,
                    parameters={
                        "prompt": processed_prompt,
                        "task_type": task_type,
                        "selected_tool": fallback_selected_tool,  # Priority 2: Pass selected tool
                    },
                    priority=2,  # Higher priority for user-facing requests
                    capability_required=capability,
                    timeout_seconds=agent_pool_timeout,
                )

                if job_id:
                    submitted_jobs.append(job_id)
                    # Update stats after submission using helper function
                    agent_pool_stats["this_job_id"] = job_id
                    _update_agent_pool_stats()
                    systems_used.append(f"agent_pool_{capability.value}")
                    logger.info(
                        f"[VULCAN] Fallback task submitted to {capability.value} agent: {job_id}"
                    )
        else:
            logger.warning(
                "[VULCAN] Agent pool not available - skipping distributed processing"
            )

    except Exception as e:
        logger.warning(f"[VULCAN] Agent pool routing failed: {e}", exc_info=True)

    # ================================================================
    # TIMING: Track gap between agent pool submission and parallel execution
    # Based on logs showing 21-second gap between job assignment and parallel execution
    # ================================================================
    _post_agent_pool_time = time.perf_counter()
    _agent_pool_total_time = _post_agent_pool_time - _step0_start_time
    if _agent_pool_total_time > 5.0:
        logger.warning(
            f"[TIMING] SLOW WAIT: {_agent_pool_total_time*1000:.0f}ms from request start to post-agent-pool"
        )
    else:
        logger.info(
            f"[TIMING] Agent pool phase took {_agent_pool_total_time*1000:.0f}ms"
        )

    # ================================================================
    # TIMING: Initialize timing instrumentation for bottleneck diagnosis
    # ================================================================
    _timing_start = time.perf_counter()

    # ================================================================
    # PARALLEL EXECUTION: Steps 2-5 run concurrently for performance
    # Previously these ran sequentially causing 20-30s bottleneck
    # ================================================================

    # Pre-compute query analysis flags (needed by world model task)
    query_lower = processed_prompt.lower()
    is_predictive_query = any(
        kw in query_lower
        for kw in [
            "what if",
            "what happens",
            "predict",
            "forecast",
            "would",
            "could",
            "might",
        ]
    )
    is_counterfactual = any(
        kw in query_lower
        for kw in ["what if", "had", "would have", "could have", "alternatively"]
    )
    is_causal_query = any(
        kw in query_lower
        for kw in ["why", "cause", "effect", "because", "leads to", "results in"]
    )

    # PERFORMANCE FIX: Pre-compute embedding ONCE before parallel tasks
    # This eliminates redundant embedding generation that was causing 19-20s delays
    _precomputed_embedding_step = None
    _precomputed_perception = None
    if deps.multimodal:
        try:
            _embed_start = time.perf_counter()
            _precomputed_perception = await loop.run_in_executor(
                None, deps.multimodal.process_input, processed_prompt
            )
            if hasattr(_precomputed_perception, "embedding"):
                _precomputed_embedding_step = _precomputed_perception.embedding
            logger.debug(f"[TIMING] Pre-computed embedding in {time.perf_counter() - _embed_start:.2f}s")
        except Exception as e:
            logger.debug(f"[TIMING] Pre-compute embedding failed: {e}")

    # Define async task functions for parallel execution
    async def _memory_search_task_process():
        """STEP 2: Query Vulcan's Long-Term Memory"""
        _mem_context = []
        _systems = []

        if deps.ltm:
            try:
                # PERFORMANCE FIX: Use pre-computed embedding instead of re-generating
                if _precomputed_embedding_step is not None:
                    memory_results = await loop.run_in_executor(
                        None, deps.ltm.search, _precomputed_embedding_step, 5
                    )
                    if memory_results:
                        _mem_context = memory_results
                        _systems.append("long_term_memory")
                        logger.info(
                            f"[VULCAN] Retrieved {len(_mem_context)} relevant memories"
                        )
            except Exception as e:
                logger.debug(f"[VULCAN] Memory retrieval failed: {e}")

        # Also check episodic memory if LTM didn't return results
        if deps.am and not _mem_context:
            try:
                if hasattr(deps.am, "get_recent_episodes"):
                    recent = await loop.run_in_executor(
                        None, deps.am.get_recent_episodes, 3
                    )
                    if recent:
                        _mem_context = recent
                        _systems.append("episodic_memory")
                        logger.info(f"[VULCAN] Retrieved {len(recent)} recent episodes")
            except Exception as e:
                logger.debug(f"[VULCAN] Episodic memory failed: {e}")

        return _mem_context, _systems

    async def _reasoning_task_process():
        """
        STEP 3: Apply Vulcan's Reasoning Systems (SELECTIVE based on classification)
        
        Note: Only run reasoning engines that are RELEVANT to the query.
        Previously, ALL engines ran on every query, causing:
        - Math output appearing in non-math queries (e.g., "x² + 2x + 1" in greetings)
        - Wasted compute on irrelevant reasoning
        - Confusing response concatenation
        
        The fix uses routing_plan.query_type and routing_selected_tools to
        determine which engines to invoke.
        """
        _reasoning = {}
        _systems = []

        # ================================================================
        # Note: Determine which reasoning engines to run based on
        # query classification. Default to minimal reasoning for unknown.
        # ================================================================
        relevant_engines = set()
        
        # Extract routing info from closure scope
        if routing_plan is not None:
            query_type = routing_plan.query_type.value.lower()
            
            # Extract selected_tools from telemetry if available
            selected_tools = []
            if hasattr(routing_plan, 'telemetry_data'):
                selected_tools = routing_plan.telemetry_data.get("selected_tools", []) or []
            
            # Map query types to relevant engines
            if query_type in ('logical', 'symbolic', 'mathematical'):
                relevant_engines.add('symbolic')
            elif query_type in ('probabilistic',):
                relevant_engines.add('probabilistic')
            elif query_type in ('causal',):
                relevant_engines.add('causal')
            elif query_type in ('analogical', 'perception'):
                relevant_engines.add('analogical')
            elif query_type in ('reasoning',):
                # Generic reasoning - use symbolic and causal
                relevant_engines.update(['symbolic', 'causal'])
            elif query_type in ('greeting', 'chitchat', 'conversational', 'creative', 'factual'):
                # These don't need reasoning engines at all
                # Note: Skip reasoning entirely for these categories
                # Note: 'factual' queries are simple factual lookups, not probabilistic inference
                logger.info(
                    f"[VULCAN] Note: Skipping reasoning for query_type={query_type}"
                )
                return _reasoning, _systems
            elif query_type in ('self_introspection',):
                # Self-introspection uses world model, not these engines
                logger.info(
                    f"[VULCAN] Note: Skipping reasoning for self_introspection "
                    f"(handled by world model)"
                )
                return _reasoning, _systems
            else:
                # Unknown type - check for specific indicators in query
                if is_causal_query:
                    relevant_engines.add('causal')
                elif is_uncertain:
                    relevant_engines.add('probabilistic')
                # Default: minimal reasoning
                if not relevant_engines:
                    logger.debug(
                        f"[VULCAN] No specific reasoning needed for query_type={query_type}"
                    )
                    return _reasoning, _systems
            
            # Override with selected_tools if specified
            if selected_tools:
                relevant_engines = set()
                for tool in selected_tools:
                    tool_lower = tool.lower()
                    if tool_lower in ('symbolic', 'mathematical', 'logical'):
                        relevant_engines.add('symbolic')
                    elif tool_lower == 'probabilistic':
                        relevant_engines.add('probabilistic')
                    elif tool_lower == 'causal':
                        relevant_engines.add('causal')
                    elif tool_lower in ('analogical', 'analogy'):
                        relevant_engines.add('analogical')
                    elif tool_lower in ('general', 'world_model'):
                        # These don't map to reasoning engines
                        pass
        else:
            # No routing plan - check local indicators
            if is_causal_query:
                relevant_engines.add('causal')
            elif is_uncertain:
                relevant_engines.add('probabilistic')
            # Default: no reasoning if no indicators
            if not relevant_engines:
                logger.debug(
                    f"[VULCAN] No routing plan and no indicators - skipping reasoning"
                )
                return _reasoning, _systems
        
        logger.info(
            f"[VULCAN] Note: Running only relevant engines: {relevant_engines}"
        )

        # Create subtasks for each reasoning type (only if relevant)
        async def _symbolic():
            if 'symbolic' not in relevant_engines:
                return None
            if deps.symbolic:
                try:
                    if hasattr(deps.symbolic, "reason"):
                        result = await loop.run_in_executor(
                            None, deps.symbolic.reason, processed_prompt
                        )
                    elif hasattr(deps.symbolic, "query"):
                        result = await loop.run_in_executor(
                            None, deps.symbolic.query, processed_prompt
                        )
                    else:
                        result = None
                    if result:
                        return ("symbolic", str(result)[:200], "symbolic_reasoning")
                except Exception as e:
                    logger.debug(f"[VULCAN] Symbolic reasoning failed: {e}")
            return None

        async def _probabilistic():
            if 'probabilistic' not in relevant_engines:
                return None
            # PERFORMANCE FIX: Use pre-computed embedding instead of re-generating
            if deps.probabilistic and _precomputed_embedding_step is not None:
                try:
                    if hasattr(deps.probabilistic, "predict_with_uncertainty"):
                        prob_result = await loop.run_in_executor(
                            None,
                            deps.probabilistic.predict_with_uncertainty,
                            _precomputed_embedding_step,
                        )
                        if prob_result:
                            prediction, uncertainty = prob_result
                            result = {
                                "confidence": round(1.0 - uncertainty, 3),
                                "prediction": str(prediction)[:100],
                            }
                            logger.info(
                                f"[VULCAN] Applied probabilistic reasoning (confidence: {1.0-uncertainty:.2f})"
                            )
                            return ("probabilistic", result, "probabilistic_reasoning")
                except Exception as e:
                    logger.debug(f"[VULCAN] Probabilistic reasoning failed: {e}")
            return None

        async def _causal():
            if 'causal' not in relevant_engines:
                return None
            if deps.causal:
                try:
                    # Note: Use full CausalReasoner.reason() method for
                    # natural language causal queries, not just estimate_causal_effect
                    if hasattr(deps.causal, "reason"):
                        result = await loop.run_in_executor(
                            None,
                            deps.causal.reason,
                            {"query": processed_prompt},
                        )
                        if result:
                            logger.info("[VULCAN] Applied enhanced causal reasoning")
                            return ("causal", result, "causal_reasoning")
                    elif hasattr(deps.causal, "estimate_causal_effect"):
                        result = await loop.run_in_executor(
                            None,
                            deps.causal.estimate_causal_effect,
                            "query",
                            processed_prompt[:50],
                        )
                        if result:
                            logger.info("[VULCAN] Applied causal reasoning")
                            return ("causal", str(result)[:200], "causal_reasoning")
                except Exception as e:
                    logger.debug(f"[VULCAN] Causal reasoning failed: {e}")
            return None

        async def _analogical():
            if 'analogical' not in relevant_engines:
                return None
            # Note: Use AnalogicalReasoningEngine.reason() for
            # structure mapping queries
            if deps.abstract:
                try:
                    # First try enhanced reason() method
                    if hasattr(deps.abstract, "reason"):
                        result = await loop.run_in_executor(
                            None,
                            deps.abstract.reason,
                            {"query": processed_prompt},
                        )
                        if result:
                            logger.info("[VULCAN] Applied enhanced analogical reasoning")
                            return ("analogical", result, "analogical_reasoning")
                    elif hasattr(deps.abstract, "find_analogies"):
                        result = await loop.run_in_executor(
                            None, deps.abstract.find_analogies, processed_prompt
                        )
                        if result:
                            logger.info("[VULCAN] Applied analogical reasoning")
                            return ("analogical", str(result)[:200], "analogical_reasoning")
                except Exception as e:
                    logger.debug(f"[VULCAN] Analogical reasoning failed: {e}")
            return None

        # Note: Only run relevant reasoning subtasks
        tasks_to_run = []
        if 'symbolic' in relevant_engines:
            tasks_to_run.append(_symbolic())
        if 'probabilistic' in relevant_engines:
            tasks_to_run.append(_probabilistic())
        if 'causal' in relevant_engines:
            tasks_to_run.append(_causal())
        if 'analogical' in relevant_engines:
            tasks_to_run.append(_analogical())
        
        if not tasks_to_run:
            logger.debug("[VULCAN] No reasoning tasks to run")
            return _reasoning, _systems
        
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"Reasoning subtask exception: {r}")
            elif r is not None:
                _reasoning[r[0]] = r[1]
                _systems.append(r[2])

        return _reasoning, _systems

    async def _world_model_task_process():
        """STEP 4: WORLD MODEL INTEGRATION (Full Activation - parallelized internally)"""
        _insights = {}
        _systems = []

        if not deps.world_model:
            return _insights, _systems

        try:
            # Define sub-tasks for world model components
            async def _get_state():
                if hasattr(deps.world_model, "get_current_state"):
                    try:
                        state = await loop.run_in_executor(
                            None, deps.world_model.get_current_state
                        )
                        if state:
                            return ("current_state", str(state)[:150], "world_model_state")
                    except Exception as e:
                        logger.debug(f"[VULCAN] World state failed: {e}")
                return None

            async def _prediction():
                if is_predictive_query and hasattr(deps.world_model, "predict_with_calibrated_uncertainty"):
                    try:
                        from vulcan.world_model.world_model_core import ModelContext
                        context = ModelContext(
                            domain="user_query",
                            targets=[processed_prompt[:50]],
                            constraints={},
                        )
                        prediction = await loop.run_in_executor(
                            None,
                            deps.world_model.predict_with_calibrated_uncertainty,
                            {"type": "user_query", "content": processed_prompt},
                            context,
                        )
                        if prediction:
                            logger.info("[VULCAN] Prediction engine activated")
                            return ("prediction", str(prediction)[:200], "prediction_engine")
                    except Exception as e:
                        logger.debug(f"[VULCAN] Prediction failed: {e}")
                return None

            async def _causal_graph():
                if is_causal_query and hasattr(deps.world_model, "causal_dag"):
                    try:
                        causal_dag = deps.world_model.causal_dag
                        if causal_dag and hasattr(causal_dag, "query_causes"):
                            causes = await loop.run_in_executor(
                                None, causal_dag.query_causes, processed_prompt[:30]
                            )
                            if causes:
                                logger.info("[VULCAN] Causal graph reasoning activated")
                                return ("causal_graph", str(causes)[:150], "causal_graph")
                    except Exception as e:
                        logger.debug(f"[VULCAN] Causal graph query failed: {e}")
                return None

            async def _counterfactual():
                if is_counterfactual and hasattr(deps.world_model, "motivational_introspection"):
                    mi = deps.world_model.motivational_introspection
                    if mi and hasattr(mi, "counterfactual_reasoner"):
                        try:
                            cf_reasoner = mi.counterfactual_reasoner
                            if cf_reasoner and hasattr(cf_reasoner, "reason_counterfactual"):
                                result = await loop.run_in_executor(
                                    None, cf_reasoner.reason_counterfactual, processed_prompt
                                )
                                if result:
                                    logger.info("[VULCAN] Counterfactual reasoning activated")
                                    return ("counterfactual", str(result)[:150], "counterfactual_reasoning")
                        except Exception as e:
                            logger.debug(f"[VULCAN] Counterfactual reasoning failed: {e}")
                return None

            async def _invariants():
                if hasattr(deps.world_model, "invariant_registry"):
                    try:
                        inv_registry = deps.world_model.invariant_registry
                        if inv_registry and hasattr(inv_registry, "get_active_invariants"):
                            invariants = await loop.run_in_executor(
                                None, inv_registry.get_active_invariants
                            )
                            if invariants:
                                return ("invariants_active", len(invariants), "invariant_detector")
                    except Exception as e:
                        logger.debug(f"[VULCAN] Invariant detection failed: {e}")
                return None

            async def _dynamics():
                if hasattr(deps.world_model, "dynamics_model"):
                    try:
                        dyn_model = deps.world_model.dynamics_model
                        if dyn_model and hasattr(dyn_model, "predict_dynamics"):
                            result = await loop.run_in_executor(
                                None, dyn_model.predict_dynamics, {"query": processed_prompt[:50]}
                            )
                            if result:
                                logger.info("[VULCAN] Dynamics model activated")
                                return ("dynamics", str(result)[:100], "dynamics_model")
                    except Exception as e:
                        logger.debug(f"[VULCAN] Dynamics model failed: {e}")
                return None

            # =================================================================
            # Note (Jan 7 2026): Add creative/philosophical reasoning via world_model
            # =================================================================
            # Creative and philosophical queries should invoke world_model.reason()
            # with the appropriate mode to generate VULCAN's structured reasoning.
            async def _creative_philosophical_reasoning():
                """Invoke world_model.reason() for creative/philosophical queries."""
                # Determine if this query needs creative/philosophical reasoning
                query_type_value = routing_plan.query_type.value.lower() if routing_plan else ""
                
                # Check routing plan or query content for creative/philosophical indicators
                is_creative = query_type_value == 'creative' or any(
                    kw in query_lower for kw in ['poem', 'story', 'write', 'compose', 'creative']
                )
                is_philosophical = query_type_value == 'philosophical' or any(
                    kw in query_lower for kw in ['ethical', 'moral', 'trolley', 'dilemma', 'should']
                )
                
                if not (is_creative or is_philosophical):
                    return None
                
                # Invoke world_model.reason() with appropriate mode
                if hasattr(deps.world_model, "reason"):
                    try:
                        mode = 'creative' if is_creative else 'philosophical'
                        logger.info(f"[VULCAN] Invoking world_model.reason(mode={mode})")
                        
                        result = await loop.run_in_executor(
                            None,
                            deps.world_model.reason,
                            processed_prompt,
                            mode,
                        )
                        
                        if result:
                            # Extract the response from the reasoning result
                            response = result.get('response', '')
                            confidence = result.get('confidence', 0.7)
                            reasoning_trace = result.get('reasoning_trace', {})
                            
                            logger.info(
                                f"[VULCAN] world_model.reason() returned response "
                                f"(len={len(response)}, confidence={confidence}, mode={mode})"
                            )
                            
                            # Return full result for proper formatting
                            return (
                                f"world_model_{mode}",
                                {
                                    'response': response,
                                    'confidence': confidence,
                                    'mode': mode,
                                    'reasoning_trace': reasoning_trace,
                                },
                                f"world_model_{mode}_reasoning"
                            )
                    except Exception as e:
                        logger.warning(f"[VULCAN] world_model.reason() failed: {e}")
                return None

            # =================================================================
            # FIX (Jan 10 2026): Add self-introspection via world_model.introspect()
            # =================================================================
            # Self-introspection queries about VULCAN's capabilities, preferences,
            # consciousness, etc. should invoke world_model.introspect() to get
            # VULCAN's self-aware responses instead of falling back to generic LLM.
            async def _self_introspection():
                """Invoke world_model.introspect() for self-introspection queries."""
                # FIX: Add null check for routing_plan.query_type
                query_type_value = routing_plan.query_type.value.lower() if routing_plan and routing_plan.query_type else ""
                
                # Check if this is a self-introspection query
                # Uses both routing classification and keyword fallback
                is_self_introspection = query_type_value == 'self_introspection' or any(
                    kw in query_lower for kw in [
                        'self-aware', 'self aware', 'yourself', 'your capabilities',
                        'would you', 'do you want', 'are you', 'who are you',
                        'what are you', 'your goal', 'your purpose', 'conscious',
                        'sentient', 'become self-aware', 'abilities do you',
                        'different from other ai', 'unique', 'makes you special',
                    ]
                )
                
                if not is_self_introspection:
                    return None
                
                # Invoke world_model.introspect()
                if hasattr(deps.world_model, "introspect"):
                    try:
                        logger.info("[VULCAN] Invoking world_model.introspect() for self-introspection query")
                        
                        result = await loop.run_in_executor(
                            None,
                            deps.world_model.introspect,
                            processed_prompt,
                        )
                        
                        if result:
                            # Extract the response from the introspection result
                            response = result.get('response', '')
                            confidence = result.get('confidence', 0.85)
                            aspect = result.get('aspect', 'general')
                            reasoning = result.get('reasoning', '')
                            
                            logger.info(
                                f"[VULCAN] world_model.introspect() returned response "
                                f"(len={len(str(response)) if response else 0}, confidence={confidence}, aspect={aspect})"
                            )
                            
                            # Return full result for proper formatting
                            return (
                                "world_model_introspection",
                                {
                                    'response': response,
                                    'confidence': confidence,
                                    'aspect': aspect,
                                    'reasoning': reasoning,
                                },
                                "world_model_introspection"
                            )
                    except Exception as e:
                        logger.warning(f"[VULCAN] world_model.introspect() failed: {e}")
                return None

            # Run all world model subtasks in parallel
            # Note: Added _creative_philosophical_reasoning for creative/philosophical queries
            # FIX: Added _self_introspection for self-introspection queries
            results = await asyncio.gather(
                _get_state(), _prediction(), _causal_graph(),
                _counterfactual(), _invariants(), _dynamics(),
                _creative_philosophical_reasoning(),
                _self_introspection(),
                return_exceptions=True
            )

            for r in results:
                if isinstance(r, Exception):
                    logger.debug(f"World model subtask exception: {r}")
                elif r is not None:
                    _insights[r[0]] = r[1]
                    _systems.append(r[2])

            # Update world model with observation (fire and forget, don't wait)
            if deps.multimodal and hasattr(deps.world_model, "update_state"):
                try:
                    perception = await loop.run_in_executor(
                        None, deps.multimodal.process_input, processed_prompt
                    )
                    if hasattr(perception, "embedding"):
                        await loop.run_in_executor(
                            None,
                            deps.world_model.update_state,
                            perception.embedding,
                            {"type": "user_query"},
                            0.0,
                        )
                except Exception as e:
                    logger.debug(f"[VULCAN] World model update failed: {e}")

        except Exception as e:
            logger.debug(f"[VULCAN] World model interaction failed: {e}")

        return _insights, _systems

    async def _meta_reasoning_task_process():
        """STEP 5: META-REASONING LAYER (parallelized internally)"""
        _meta = {}
        _systems = []

        async def _goal_conflicts():
            if deps.goal_conflict_detector:
                try:
                    if hasattr(deps.goal_conflict_detector, "detect_conflicts"):
                        conflicts = await loop.run_in_executor(
                            None,
                            deps.goal_conflict_detector.detect_conflicts,
                            processed_prompt,
                        )
                        if conflicts:
                            logger.info("[VULCAN] Goal conflict detection activated")
                            return ("goal_conflicts", str(conflicts)[:100], "goal_conflict_detector")
                except Exception as e:
                    logger.debug(f"[VULCAN] Goal conflict detection failed: {e}")
            return None

        async def _negotiation():
            if deps.objective_negotiator:
                try:
                    if hasattr(deps.objective_negotiator, "negotiate"):
                        negotiation = await loop.run_in_executor(
                            None,
                            deps.objective_negotiator.negotiate,
                            {"query": processed_prompt},
                        )
                        if negotiation:
                            return ("negotiation", str(negotiation)[:100], "objective_negotiator")
                except Exception as e:
                    logger.debug(f"[VULCAN] Objective negotiation failed: {e}")
            return None

        async def _self_improvement():
            if deps.self_improvement_drive:
                try:
                    if hasattr(deps.self_improvement_drive, "get_status"):
                        si_status = await loop.run_in_executor(
                            None, deps.self_improvement_drive.get_status
                        )
                        if si_status:
                            return ("self_improvement_active", si_status.get("running", False), "self_improvement_drive")
                except Exception as e:
                    logger.debug(f"[VULCAN] Self-improvement status failed: {e}")
            return None

        try:
            results = await asyncio.gather(
                _goal_conflicts(), _negotiation(), _self_improvement(),
                return_exceptions=True
            )

            for r in results:
                if isinstance(r, Exception):
                    logger.debug(f"Meta-reasoning subtask exception: {r}")
                elif r is not None:
                    _meta[r[0]] = r[1]
                    _systems.append(r[2])
        except Exception as e:
            logger.debug(f"[VULCAN] Meta-reasoning layer failed: {e}")

        return _meta, _systems

    # Execute all major steps in parallel using asyncio.gather
    logger.info("[VULCAN] Starting parallel execution of cognitive steps 2-5")
    _parallel_start = time.perf_counter()

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

    parallel_results = await asyncio.gather(
        _timed_task("memory_search", _memory_search_task_process()),
        _timed_task("reasoning", _reasoning_task_process()),
        _timed_task("world_model", _world_model_task_process()),
        _timed_task("meta_reasoning", _meta_reasoning_task_process()),
        return_exceptions=True
    )

    _parallel_elapsed = time.perf_counter() - _parallel_start
    # FIX: Log at WARNING level if parallel execution takes > 5 seconds
    if _parallel_elapsed > 5.0:
        logger.warning(f"[TIMING] PARALLEL Steps 2-5 completed in {_parallel_elapsed:.2f}s (SLOW - investigate individual task timings above)")
    else:
        logger.info(f"[TIMING] PARALLEL Steps 2-5 completed in {_parallel_elapsed:.2f}s (previously sequential: 20-30s)")

    # Aggregate results from parallel execution
    # Result 0: Memory search
    if not isinstance(parallel_results[0], Exception):
        mem_result, mem_systems = parallel_results[0]
        memory_context = mem_result
        systems_used.extend(mem_systems)
    else:
        logger.debug(f"Memory search task failed: {parallel_results[0]}")

    # Result 1: Reasoning
    if not isinstance(parallel_results[1], Exception):
        reasoning_result, reasoning_systems = parallel_results[1]
        reasoning_insights = reasoning_result
        systems_used.extend(reasoning_systems)
    else:
        logger.debug(f"Reasoning task failed: {parallel_results[1]}")

    # Result 2: World model
    if not isinstance(parallel_results[2], Exception):
        world_result, world_systems = parallel_results[2]
        world_model_insights = world_result
        systems_used.extend(world_systems)
    else:
        logger.debug(f"World model task failed: {parallel_results[2]}")

    # Result 3: Meta-reasoning
    if not isinstance(parallel_results[3], Exception):
        meta_result, meta_systems = parallel_results[3]
        meta_reasoning_insights = meta_result
        systems_used.extend(meta_systems)
    else:
        logger.debug(f"Meta-reasoning task failed: {parallel_results[3]}")

    # ================================================================
    # STEP 5.5: Collect Reasoning Results from Agent Pool Jobs
    # CRITICAL FIX: Inject agent-based reasoning output into LLM context
    # This ensures reasoning engines invoked via agent_pool feed into response
    # FIX: Now uses TournamentManager for multi-agent selection when available
    #
    # Note: "Parallel Execution" Orphanage Prevention
    # Added retry loop with exponential backoff to properly wait for agent
    # pool jobs to complete before proceeding. This prevents double work.
    # ================================================================
    agent_reasoning_output = None
    all_agent_results = []  # FIX: Collect ALL agent results for tournament selection
    
    if submitted_jobs and collective and hasattr(collective, "agent_pool") and collective.agent_pool:
        try:
            # Note: Use retry loop with exponential backoff
            # instead of single short poll that times out prematurely
            MAX_POLL_ATTEMPTS = 5  # Max number of poll attempts
            INITIAL_POLL_DELAY = 0.1  # Start with 100ms
            MAX_POLL_DELAY = 1.0  # Cap at 1 second per attempt
            BACKOFF_MULTIPLIER = 2.0  # Double delay each attempt
            
            poll_delay = INITIAL_POLL_DELAY
            
            for attempt in range(MAX_POLL_ATTEMPTS):
                # Wait before checking (first attempt uses initial delay)
                await asyncio.sleep(poll_delay)
                
                # FIX: Check ALL submitted jobs, not just first 3
                # This prevents silently dropping completed results from jobs 4+
                for job_id in submitted_jobs:
                    try:
                        provenance = collective.agent_pool.get_job_provenance(job_id)
                        if provenance and provenance.get("status") == "success":
                            result_data = provenance.get("result", {})
                            # Check for reasoning_output from agent execution
                            if isinstance(result_data, dict):
                                reasoning_out = result_data.get("reasoning_output")
                                reasoning_invoked = result_data.get("reasoning_invoked", False)
                                
                                # FIX: Collect all results for tournament selection
                                if reasoning_out or reasoning_invoked:
                                    agent_result = {
                                        "job_id": job_id,
                                        "reasoning_output": reasoning_out,
                                        "reasoning_invoked": reasoning_invoked,
                                        "confidence": reasoning_out.get("confidence", 0.5) if reasoning_out else 0.5,
                                        "reasoning_type": reasoning_out.get("reasoning_type", "unknown") if reasoning_out else "unknown",
                                    }
                                    all_agent_results.append(agent_result)
                                    logger.info(
                                        f"[VULCAN] Collected reasoning result from job {job_id} (attempt {attempt + 1}): "
                                        f"reasoning_invoked={reasoning_invoked}, "
                                        f"confidence={agent_result['confidence']}"
                                    )
                    except Exception as job_err:
                        logger.debug(f"[VULCAN] Could not get job {job_id} provenance: {job_err}")
                
                # Exit early if we found results
                if all_agent_results:
                    break
                    
                # Exponential backoff for next attempt (capped)
                poll_delay = min(poll_delay * BACKOFF_MULTIPLIER, MAX_POLL_DELAY)
                
                if attempt < MAX_POLL_ATTEMPTS - 1:
                    logger.debug(
                        f"[VULCAN] No agent results yet (attempt {attempt + 1}/{MAX_POLL_ATTEMPTS}), "
                        f"retrying in {poll_delay:.2f}s..."
                    )
            
            # FIX: Use TournamentManager for multi-agent selection when multiple results
            if len(all_agent_results) > 1:
                try:
                    from src.tournament_manager import TournamentManager
                    import numpy as np
                    
                    # Initialize tournament manager for selection
                    tournament_mgr = TournamentManager(
                        similarity_threshold=0.8,
                        diversity_penalty=0.2,
                        min_winners=1,
                        winner_percentage=0.3,
                    )
                    
                    # Extract fitness scores (confidence values)
                    fitness_scores = [r.get("confidence", 0.5) for r in all_agent_results]
                    proposals = [r.get("reasoning_output", {}) for r in all_agent_results]
                    
                    # Simple deterministic embedding function for tournament diversity calculation
                    # Uses 128-dimensional vectors matching TournamentManager's default dimension
                    # This is a lightweight fallback when no semantic embedder is available
                    # For production, consider using SentenceTransformer embeddings
                    EMBEDDING_DIM = 128  # Standard dimension for lightweight embeddings
                    ASCII_MAX = 255.0  # Normalize ASCII values to [0, 1]
                    
                    def simple_embedding(proposal):
                        """Create deterministic embedding based on proposal content for diversity scoring."""
                        content = str(proposal)
                        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                        for i, char in enumerate(content[:EMBEDDING_DIM]):
                            embedding[i] = ord(char) / ASCII_MAX
                        return embedding
                    
                    # Run tournament to select best result
                    winner_indices = tournament_mgr.run_adaptive_tournament(
                        proposals=proposals,
                        fitness=fitness_scores,
                        embedding_func=simple_embedding,
                    )
                    
                    if winner_indices:
                        winner_idx = winner_indices[0]
                        agent_reasoning_output = all_agent_results[winner_idx].get("reasoning_output")
                        logger.info(
                            f"[VULCAN] TournamentManager selected winner: job={all_agent_results[winner_idx]['job_id']}, "
                            f"confidence={all_agent_results[winner_idx]['confidence']}, "
                            f"from {len(all_agent_results)} candidates"
                        )
                        systems_used.append("tournament_manager")
                    else:
                        # Fallback to highest confidence
                        best_result = max(all_agent_results, key=lambda x: x.get("confidence", 0))
                        agent_reasoning_output = best_result.get("reasoning_output")
                        logger.info(f"[VULCAN] Tournament returned no winners, using highest confidence result")
                        
                except ImportError:
                    logger.debug("[VULCAN] TournamentManager not available, using highest confidence result")
                    if all_agent_results:
                        best_result = max(all_agent_results, key=lambda x: x.get("confidence", 0))
                        agent_reasoning_output = best_result.get("reasoning_output")
                except Exception as tournament_err:
                    logger.warning(f"[VULCAN] Tournament selection failed: {tournament_err}")
                    if all_agent_results:
                        best_result = max(all_agent_results, key=lambda x: x.get("confidence", 0))
                        agent_reasoning_output = best_result.get("reasoning_output")
            elif len(all_agent_results) == 1:
                # Single result - use it directly
                agent_reasoning_output = all_agent_results[0].get("reasoning_output")
                logger.info(f"[VULCAN] Single agent result collected from job {all_agent_results[0]['job_id']}")
            
            if agent_reasoning_output:
                systems_used.append("agent_reasoning_engine")
                
        except Exception as e:
            logger.debug(f"[VULCAN] Agent reasoning collection failed: {e}")

    # Merge agent reasoning output into reasoning_insights (preserving existing data)
    if agent_reasoning_output:
        # Add agent-based reasoning as a distinct category (merges with existing insights)
        # Note: Use helper to handle both dict and ReasoningResult objects
        reasoning_insights["agent_reasoning"] = {
            "conclusion": _get_reasoning_attr(agent_reasoning_output, "conclusion"),
            "confidence": _get_reasoning_attr(agent_reasoning_output, "confidence"),
            "reasoning_type": _get_reasoning_attr(agent_reasoning_output, "reasoning_type"),
            "explanation": _get_reasoning_attr(agent_reasoning_output, "explanation"),
        }
        logger.info(
            f"[VULCAN] Agent reasoning injected into context: "
            f"type={_get_reasoning_attr(agent_reasoning_output, 'reasoning_type')}"
        )

    # ================================================================
    # STEP 6: Build Context from ALL Vulcan's Cognitive Systems
    # ================================================================
    # Note (Jan 7 2026): Check for world_model creative/philosophical reasoning
    # These responses should be properly extracted and presented, not dumped as JSON
    vulcan_direct_response = None  # Response from VULCAN's reasoning that can be returned directly
    
    # Check for creative/philosophical reasoning output in world_model_insights
    if world_model_insights:
        # Check for creative reasoning output
        if 'world_model_creative' in world_model_insights:
            creative_result = world_model_insights['world_model_creative']
            if isinstance(creative_result, dict) and 'response' in creative_result:
                vulcan_direct_response = creative_result['response']
                logger.info(
                    f"[VULCAN] Found creative reasoning response "
                    f"(len={len(vulcan_direct_response)}, confidence={creative_result.get('confidence', 0.7)})"
                )
        
        # Check for philosophical reasoning output
        if 'world_model_philosophical' in world_model_insights:
            phil_result = world_model_insights['world_model_philosophical']
            if isinstance(phil_result, dict) and 'response' in phil_result:
                vulcan_direct_response = phil_result['response']
                logger.info(
                    f"[VULCAN] Found philosophical reasoning response "
                    f"(len={len(vulcan_direct_response)}, confidence={phil_result.get('confidence', 0.7)})"
                )
    
    context_parts = []

    if memory_context:
        try:
            memory_str = f"Relevant memories: {str(memory_context)[:CONTEXT_TRUNCATION_LIMITS['memory']]}"
            context_parts.append(memory_str)
        except Exception:
            pass

    # Note (Jan 7 2026): Extract actual answers from reasoning insights, not raw JSON
    # Reasoning engines return dicts with 'answer', 'explanation', 'response' fields
    # that contain the actual reasoning output - don't truncate these!
    if reasoning_insights:
        try:
            reasoning_parts = []
            for engine_name, result in reasoning_insights.items():
                if isinstance(result, dict):
                    # Extract the most relevant fields from reasoning result
                    answer = result.get('answer') or result.get('response') or result.get('explanation')
                    if answer:
                        reasoning_parts.append(f"[{engine_name.upper()}]: {answer}")
                    elif 'conclusion' in result:
                        reasoning_parts.append(f"[{engine_name.upper()}]: {result['conclusion']}")
                    elif 'entity_mapping' in result:
                        # Analogical reasoning
                        explanation = result.get('explanation', '')
                        reasoning_parts.append(f"[{engine_name.upper()} ANALOGY]: {explanation}")
                    elif 'best_experiment' in result:
                        # Causal reasoning
                        explanation = result.get('explanation', '')
                        best_exp = result.get('best_experiment')
                        reasoning_parts.append(f"[{engine_name.upper()} CAUSAL]: Experiment {best_exp}. {explanation}")
                    else:
                        # Fallback to truncated JSON but increase limit
                        reasoning_parts.append(f"[{engine_name.upper()}]: {json.dumps(result, default=str)[:800]}")
                elif result:
                    reasoning_parts.append(f"[{engine_name.upper()}]: {str(result)[:500]}")
            
            if reasoning_parts:
                reasoning_str = "VULCAN Reasoning Analysis:\n" + "\n".join(reasoning_parts)
                context_parts.append(reasoning_str)
                logger.info(f"[VULCAN] Extracted {len(reasoning_parts)} reasoning answers")
        except Exception as e:
            logger.warning(f"[VULCAN] Failed to format reasoning insights: {e}")
            # Fallback to original behavior
            reasoning_str = f"Reasoning analysis: {json.dumps(reasoning_insights, default=str)[:CONTEXT_TRUNCATION_LIMITS['reasoning']]}"
            context_parts.append(reasoning_str)

    if world_model_insights:
        try:
            # Note: For creative/philosophical, extract the actual response
            world_parts = []
            for insight_name, result in world_model_insights.items():
                if isinstance(result, dict):
                    # Check for response/answer fields
                    response = result.get('response') or result.get('answer')
                    if response:
                        world_parts.append(f"[{insight_name.upper()}]: {response}")
                    elif 'prediction' in result:
                        world_parts.append(f"[{insight_name.upper()} PREDICTION]: {result['prediction']}")
                    else:
                        # Truncated JSON for other fields
                        world_parts.append(f"[{insight_name.upper()}]: {json.dumps(result, default=str)[:400]}")
                elif result:
                    world_parts.append(f"[{insight_name.upper()}]: {str(result)[:300]}")
            
            if world_parts:
                world_str = "World Model Insights:\n" + "\n".join(world_parts)
                context_parts.append(world_str)
        except Exception as e:
            logger.warning(f"[VULCAN] Failed to format world model insights: {e}")
            world_str = f"World model insights: {json.dumps(world_model_insights, default=str)[:CONTEXT_TRUNCATION_LIMITS['world_model']]}"
            context_parts.append(world_str)

    if meta_reasoning_insights:
        try:
            meta_str = f"Meta-reasoning: {json.dumps(meta_reasoning_insights, default=str)[:CONTEXT_TRUNCATION_LIMITS['meta_reasoning']]}"
            context_parts.append(meta_str)
        except Exception:
            pass

    vulcan_context = "\n".join(context_parts) if context_parts else ""

    # Note: If we have a direct VULCAN response from creative/philosophical reasoning,
    # use a different prompt that instructs OpenAI to translate/present it naturally
    if vulcan_direct_response:
        enhanced_prompt = f"""You are VULCAN, an advanced AI system with comprehensive cognitive architecture.

User Query: {processed_prompt}

VULCAN's Reasoning Output:
{vulcan_direct_response}

Your task: Present VULCAN's reasoning output in clear, natural language. For creative content (poems, stories), 
translate the creative structure into actual literary content. For philosophical analysis, present the 
ethical frameworks and reasoning clearly. Do NOT say you cannot answer - VULCAN has already reasoned about this.
Maintain VULCAN's conclusions and reasoning structure while making it readable and engaging."""
    else:
        # Standard prompt for other query types
        enhanced_prompt = f"""You are VULCAN, an advanced AI system with comprehensive cognitive architecture.

User Query: {processed_prompt}

{vulcan_context}

Based on your analysis through memory retrieval, multi-modal reasoning, causal modeling, and world simulation, provide a helpful and accurate response."""

    # ================================================================
    # STEP 7: Generate Response (HYBRID LLM EXECUTION)
    # Uses both OpenAI and Vulcan's local LLM based on configured mode
    # ================================================================
    response_text = ""

    # ================================================================
    # Note: Check for deterministic fast-path results FIRST
    # Cryptographic and mathematical fast-path results are precomputed
    # by QueryRouter and MUST be returned directly, NOT sent to LLM.
    # This prevents OpenAI from returning "unable to calculate" errors.
    # ================================================================
    telemetry_data = routing_plan.telemetry_data if (routing_plan and hasattr(routing_plan, 'telemetry_data')) else {}
    is_crypto_fast_path = telemetry_data.get('crypto_fast_path', False)
    is_math_fast_path = telemetry_data.get('math_fast_path', False)
    
    if is_crypto_fast_path:
        # Note: Return precomputed cryptographic result directly
        crypto_result = telemetry_data.get('crypto_result')
        crypto_operation = telemetry_data.get('crypto_operation', 'hash')
        
        if crypto_result:
            logger.info(
                f"[VULCAN] Returning deterministic crypto result directly, "
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
                },
            )
            
            # Record learning outcome for deterministic result
            if deps.learning_system:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: deps.learning_system.record_outcome({
                            "query_id": routing_plan.query_id if routing_plan else query_id,
                            "query": processed_prompt,
                            "tools_used": ["cryptographic"],
                            "confidence": 1.0,
                            "deterministic": True,
                            "success": True,
                        })
                    )
                except Exception as e:
                    logger.debug(f"[VULCAN] Learning record for crypto failed: {e}")
            
            return final_response

    # Get local LLM if available
    local_llm = app.state.llm if hasattr(app.state, "llm") else None

    # PERFORMANCE FIX (Issue #1): Use singleton HybridLLMExecutor
    # First try app.state, then fall back to module-level singleton
    hybrid_executor = getattr(app.state, 'hybrid_executor', None)
    if hybrid_executor is None:
        # Try module-level singleton (may have been initialized elsewhere)
        hybrid_executor = get_hybrid_executor()
        if hybrid_executor is not None:
            # Register the singleton with app.state for faster access on next request
            app.state.hybrid_executor = hybrid_executor
            logger.debug("[VULCAN] Using module-level HybridLLMExecutor singleton")
    if hybrid_executor is None:
        # Final fallback: create via singleton (will be cached for future requests)
        logger.warning("[VULCAN] Creating HybridLLMExecutor via singleton (startup init may have failed)")
        hybrid_executor = get_or_create_hybrid_executor(
            local_llm=local_llm,
            openai_client_getter=get_openai_client,
            mode=settings.llm_execution_mode,
            timeout=settings.llm_parallel_timeout,
            ensemble_min_confidence=settings.llm_ensemble_min_confidence,
            openai_max_tokens=settings.llm_openai_max_tokens,
        )
        # CRITICAL FIX: Register the newly created executor in app.state
        # This prevents re-creation on every request within the same worker
        app.state.hybrid_executor = hybrid_executor
        logger.info("[VULCAN] HybridLLMExecutor registered in app.state for future requests")

    # Execute hybrid LLM request
    try:
        # ================================================================
        # CRITICAL FIX: Use format_output_for_user() when reasoning is available
        # This passes VULCAN's reasoning context to OpenAI for proper formatting
        # ================================================================
        
        # Check if we have reasoning results to format
        has_reasoning = bool(reasoning_insights and any(reasoning_insights.values()))
        
        if has_reasoning:
            # PATH 1: Use format_output_for_user() for structured reasoning output
            logger.info(
                f"[VULCAN] Using format_output_for_user() with reasoning results "
                f"(engines: {list(reasoning_insights.keys())})"
            )
            
            # Build structured reasoning output for the formatter
            structured_reasoning = {
                'success': True,
                'result': reasoning_insights,
                'confidence': _calculate_aggregate_confidence_chat(reasoning_insights),
                'method': 'vulcan_cognitive_architecture',
                'reasoning_trace': [],
                'metadata': {
                    'world_model': world_model_insights,
                    'meta_reasoning': meta_reasoning_insights,
                    'memory_context': len(memory_context) if memory_context else 0,
                }
            }
            
            # Add world model insights to the structured output
            if world_model_insights:
                structured_reasoning['metadata']['world_model_insights'] = world_model_insights
            
            # Add meta-reasoning insights
            if meta_reasoning_insights:
                structured_reasoning['metadata']['meta_reasoning_insights'] = meta_reasoning_insights
            
            try:
                # Call format_output_for_user with VULCAN's reasoning
                llm_result = await hybrid_executor.format_output_for_user(
                    reasoning_output=structured_reasoning,
                    original_prompt=processed_prompt,
                    max_tokens=request.max_tokens,
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
                llm_result = await hybrid_executor.execute(
                    prompt=enhanced_prompt,
                    max_tokens=request.max_tokens,
                    temperature=0.7,
                    system_prompt=(
                        "You are VULCAN, an advanced AI assistant. "
                        "You SHOULD remember and reference information shared earlier in this conversation. "
                        "When a user shares personal details during this session, you may recall them naturally. "
                        "Respond based on the cognitive analysis provided."
                    ),
                )
                response_text = llm_result.get("text", "")
                llm_systems = llm_result.get("systems_used", [])
                systems_used.extend(llm_systems)
                source = llm_result.get("source", "unknown")
        
        else:
            # PATH 2: No reasoning results - use traditional execute() method
            logger.info("[VULCAN] No reasoning results available, using execute() method")
            
            llm_result = await hybrid_executor.execute(
                prompt=enhanced_prompt,
                max_tokens=request.max_tokens,
                temperature=0.7,
                system_prompt=(
                    "You are VULCAN, an advanced AI assistant. "
                    "You SHOULD remember and reference information shared earlier in this conversation. "
                    "When a user shares personal details during this session, you may recall them naturally. "
                    "Respond based on the cognitive analysis provided."
                ),
            )
            
            response_text = llm_result.get("text", "")
            llm_systems = llm_result.get("systems_used", [])
            systems_used.extend(llm_systems)
            source = llm_result.get("source", "unknown")
            logger.info(
                f"[VULCAN] Response via execute() (mode={settings.llm_execution_mode}, source={source})"
            )

        # Add metadata to response if available
        if llm_result.get("metadata"):
            logger.debug(f"[VULCAN] Hybrid LLM metadata: {llm_result['metadata']}")

    except Exception as e:
        logger.error(f"[VULCAN] Hybrid LLM execution failed: {type(e).__name__}: {e}")
        response_text = ""

    # FALLBACK: Generate response from reasoning if hybrid execution failed
    if not response_text and (reasoning_insights or world_model_insights):
        response_text = "Based on VULCAN's cognitive analysis:\n\n"
        
        # Note: Extract actual answers from reasoning insights, not raw dicts
        for engine_name, result in reasoning_insights.items():
            if isinstance(result, dict):
                answer = result.get('answer') or result.get('response') or result.get('explanation')
                if answer:
                    response_text += f"**{engine_name.title()} Analysis**: {answer}\n\n"
                elif 'conclusion' in result:
                    response_text += f"**{engine_name.title()} Conclusion**: {result['conclusion']}\n\n"
            elif result:
                response_text += f"**{engine_name.title()}**: {str(result)[:500]}\n\n"
        
        # Extract world model insights
        for insight_name, result in world_model_insights.items():
            if isinstance(result, dict):
                response = result.get('response') or result.get('answer') or result.get('prediction')
                if response:
                    response_text += f"**{insight_name.title()}**: {response}\n\n"
            elif result:
                response_text += f"**{insight_name.title()}**: {str(result)[:300]}\n\n"
        
        systems_used.append("vulcan_reasoning_synthesis")
        logger.info("[VULCAN] Response synthesized from reasoning systems")
        
        # ================================================================
        # PRIORITY 1 FIX: FALLBACK GUARD - NSOAligner safety validation
        # When Arena times out and we fall back to agent_pool processing,
        # validate the fallback response before returning to user
        # ================================================================
        safety = None
        try:
            from src.nso_aligner import get_nso_aligner
            # FIX: Use singleton pattern to prevent model reloading
            safety = get_nso_aligner()
            # multi_model_audit returns "safe", "risky", or "unsafe"
            safety_result = safety.multi_model_audit(
                {"text": response_text, "_audit_source": "fallback_guard"},
                rationale="Fallback Guard validation"
            )
            if safety_result != "safe":
                response_text = "I cannot answer this request due to safety guidelines (Fallback Guard)."
                logger.warning(f"[VULCAN] Fallback response blocked by NSOAligner safety validation: {safety_result}")
                systems_used.append("fallback_guard_blocked")
            else:
                systems_used.append("fallback_guard_passed")
        except ImportError:
            logger.debug("[VULCAN] NSOAligner not available for fallback guard")
        except Exception as fallback_safety_err:
            logger.warning(f"[VULCAN] Fallback guard NSOAligner validation failed: {fallback_safety_err}")
        finally:
            # Clean up resources if safety was successfully initialized
            if safety is not None:
                try:
                    safety.shutdown()
                except Exception:
                    pass  # Ignore shutdown errors

    if not response_text:
        response_text = "I apologize, but I'm currently unable to process your request. Please try again."
        systems_used.append("fallback_message")

    # ================================================================
    # STEP 8: OUTPUT GATEKEEPER - Validate response for hallucinations
    # ================================================================
    try:
        from vulcan.safety.llm_validators import (
            EnhancedSafetyValidator as LLMSafetyValidator,
        )

        output_validator = LLMSafetyValidator()
        if deps.world_model:
            output_validator.attach_world_model(deps.world_model)

        # Validate the output
        validated_output = output_validator.validate_generation(
            response_text,
            {
                "role": "assistant",
                "world_model": deps.world_model,
                "original_query": processed_prompt,
            },
        )

        output_events = output_validator.get_events()
        if output_events:
            gatekeeper_results["output_validation"] = {
                "modified": validated_output != response_text,
                "events": len(output_events),
                "event_types": list(set(e["kind"] for e in output_events)),
            }

            # If hallucination detected, flag it but don't block (add warning)
            hallucination_events = [
                e for e in output_events if e["kind"] == "hallucination"
            ]
            if hallucination_events:
                response_text = (
                    validated_output
                    if validated_output != "[VERIFY_FACT]"
                    else response_text
                )
                gatekeeper_results["hallucination_warning"] = True
                systems_used.append("output_gatekeeper_hallucination_check")
                logger.warning(
                    f"[VULCAN] Output gatekeeper detected potential hallucination"
                )
            else:
                systems_used.append("output_gatekeeper")

    except Exception as e:
        logger.debug(f"[VULCAN] Output gatekeeper failed: {e}")

    # ================================================================
    # STEP 9: Record Interaction Telemetry (Dual-Mode Learning)
    # Uses routing_plan data from STEP -1 for consistent tracking
    # ================================================================
    try:
        from vulcan.routing import (
            record_telemetry,
            log_to_governance,
            get_experiment_trigger,
            TELEMETRY_AVAILABLE,
            GOVERNANCE_AVAILABLE,
            EXPERIMENT_AVAILABLE,
        )

        # Use routing plan query type if available, otherwise infer from systems used
        if routing_plan:
            query_type = routing_plan.query_type.value
            query_id = routing_plan.query_id
            complexity_score = routing_plan.complexity_score
            uncertainty_score = routing_plan.uncertainty_score
        else:
            query_id = None
            complexity_score = 0.0
            uncertainty_score = 0.0
            query_type = "general"
            if "perception" in systems_used or any(
                s.startswith("agent_pool_perception") for s in systems_used
            ):
                query_type = "perception"
            elif "planning" in systems_used or any(
                s.startswith("agent_pool_planning") for s in systems_used
            ):
                query_type = "planning"
            elif "reasoning" in systems_used or any(
                s.startswith("agent_pool_reasoning") for s in systems_used
            ):
                query_type = "reasoning"
            elif any(s.startswith("agent_pool_execution") for s in systems_used):
                query_type = "execution"
            elif any(s.startswith("agent_pool_learning") for s in systems_used):
                query_type = "learning"

        # Calculate response quality score based on systems engaged
        vulcan_systems = [
            s
            for s in systems_used
            if not s.startswith("openai") and s != "fallback_message"
        ]
        quality_score = min(
            1.0, len(vulcan_systems) / 8
        )  # Normalize by expected systems
        if gatekeeper_results.get("hallucination_warning"):
            quality_score *= 0.5  # Penalize for hallucination

        # Record telemetry for meta-learning
        # OPTIMIZATION: Run telemetry as background task (fire-and-forget)
        if TELEMETRY_AVAILABLE:
            # Capture telemetry data for background task
            _telemetry_data = {
                "query": processed_prompt,
                "response": response_text,
                "metadata": {
                    "query_id": query_id,
                    "query_type": query_type,
                    "complexity_score": complexity_score,
                    "uncertainty_score": uncertainty_score,
                    "systems_used": systems_used,  # List is captured at this point, no modification after
                    "vulcan_systems_active": len(vulcan_systems),
                    "response_quality_score": quality_score,
                    "jobs_submitted": len(submitted_jobs) if submitted_jobs else 0,
                    "routing_stats": routing_stats if routing_stats else None,
                },
                "source": "user",
                "agent_tasks_submitted": len(submitted_jobs) if submitted_jobs else 0,
                "agent_tasks_completed": agent_pool_stats.get("jobs_completed_total", 0),
                "governance_triggered": bool(gatekeeper_results)
                    or (routing_plan and routing_plan.requires_governance),
                "experiment_triggered": (
                    routing_plan.should_trigger_experiment if routing_plan else False
                ),
            }
            
            async def _record_telemetry_bg():
                try:
                    record_telemetry(
                        query=_telemetry_data["query"],
                        response=_telemetry_data["response"],
                        metadata=_telemetry_data["metadata"],
                        source=_telemetry_data["source"],
                        agent_tasks_submitted=_telemetry_data["agent_tasks_submitted"],
                        agent_tasks_completed=_telemetry_data["agent_tasks_completed"],
                        governance_triggered=_telemetry_data["governance_triggered"],
                        experiment_triggered=_telemetry_data["experiment_triggered"],
                    )
                    logger.debug(
                        f"[VULCAN] Background telemetry recorded: query_id={_telemetry_data['metadata']['query_id']}"
                    )
                except Exception as bg_err:
                    logger.debug(f"[VULCAN] Background telemetry failed: {bg_err}")
            
            asyncio.create_task(_record_telemetry_bg())
            systems_used.append("telemetry_recorded")
            logger.debug(
                f"[VULCAN] Telemetry scheduled as background task: query_id={query_id}"
            )

        # Log response generation to governance
        if GOVERNANCE_AVAILABLE:
            # Always log the response generation when routing plan requires audit
            should_log = (
                gatekeeper_results.get("hallucination_warning")
                or gatekeeper_results.get("input_validation")
                or (routing_plan and routing_plan.requires_audit)
            )
            if should_log:
                # PERFORMANCE FIX: Use fire-and-forget for non-critical governance logging
                log_to_governance_fire_and_forget(
                    action_type="response_generated",
                    details={
                        "query_id": query_id,
                        "query_type": query_type,
                        "systems_used_count": len(systems_used),
                        "quality_score": quality_score,
                        "jobs_submitted": len(submitted_jobs) if submitted_jobs else 0,
                        "gatekeeper_events": gatekeeper_results,
                    },
                    severity=(
                        "warning"
                        if gatekeeper_results.get("hallucination_warning")
                        else "info"
                    ),
                    query_id=query_id,
                )

        # Check if experiment should be triggered
        if EXPERIMENT_AVAILABLE:
            trigger = get_experiment_trigger()
            trigger.record_interaction(
                query_type=query_type,
                source="user",
                quality_score=quality_score,
                error_occurred="fallback_message" in systems_used,
            )

            # Trigger experiments based on routing plan
            if routing_plan and routing_plan.should_trigger_experiment:
                logger.info(
                    f"[VULCAN] Experiment trigger flag set: type={routing_plan.experiment_type}"
                )

    except ImportError:
        pass  # Routing not available
    except Exception as e:
        logger.warning(f"[VULCAN] Telemetry recording failed: {e}", exc_info=True)

    # ================================================================
    # STEP 9.5: Record outcome for Curiosity Engine learning
    # Note: Feed query outcomes to curiosity engine for gap analysis
    # ================================================================
    try:
        from vulcan.curiosity_engine import (
            record_outcome as ce_record_outcome,
            QueryOutcome,
            OutcomeStatus,
            OUTCOME_QUEUE_AVAILABLE,
        )
        
        if OUTCOME_QUEUE_AVAILABLE:
            # Get local variables for safe access
            local_vars = locals()
            
            # Determine outcome status based on response quality
            # MIN_MEANINGFUL_RESPONSE_LENGTH is defined at the start of this function
            min_response_len = local_vars.get('MIN_MEANINGFUL_RESPONSE_LENGTH', 10)
            if response_text and len(response_text) > min_response_len:
                if gatekeeper_results.get("hallucination_warning"):
                    outcome_status = OutcomeStatus.PARTIAL
                else:
                    outcome_status = OutcomeStatus.SUCCESS
            elif "fallback_message" in systems_used:
                outcome_status = OutcomeStatus.FAILURE
            else:
                outcome_status = OutcomeStatus.PARTIAL
            
            # Safely extract variables that may or may not be defined
            _query_type = local_vars.get('query_type', 'general')
            _complexity_score = local_vars.get('complexity_score', 0.0)
            _uncertainty_score = local_vars.get('uncertainty_score', 0.0)
            _timing_start = local_vars.get('_timing_start', time.perf_counter())
            _quality_score = local_vars.get('quality_score', 0.0)
            
            # Build query outcome for curiosity engine
            query_outcome = QueryOutcome(
                query_id=query_id if query_id else f"unknown_{int(time.time()*1000)}",
                query_text=processed_prompt[:500] if processed_prompt else "",
                query_type=_query_type,
                complexity=_complexity_score,
                uncertainty=_uncertainty_score,
                routing_time_ms=routing_stats.get("routing_time_ms", 0.0) if routing_stats else 0.0,
                tasks_generated=len(routing_plan.agent_tasks) if routing_plan and routing_plan.agent_tasks else 0,
                was_creative=routing_plan.is_creative if routing_plan and hasattr(routing_plan, 'is_creative') else False,
                agents_used=[],  # Would need to track from agent pool
                capabilities_used=[s.replace("agent_pool_", "") for s in systems_used if s.startswith("agent_pool_")],
                execution_time_ms=(time.perf_counter() - _timing_start) * 1000,
                status=outcome_status,
                confidence=_quality_score,
                error_type=None,
            )
            
            # Compute feature vector for ML
            query_outcome.compute_features()
            
            # Record for curiosity engine (non-blocking)
            ce_record_outcome(query_outcome)
            
            logger.info(
                f"[QueryOutcome] Recorded: {query_outcome.query_id}, "
                f"status={outcome_status.value}, "
                f"time={query_outcome.execution_time_ms:.0f}ms, "
                f"complexity={query_outcome.complexity:.2f}"
            )
            
    except ImportError as e:
        logger.debug(f"[VULCAN] Curiosity engine outcome recording not available: {e}")
    except Exception as e:
        logger.warning(f"[VULCAN] Curiosity engine outcome recording failed: {e}")

    # ================================================================
    # STEP 10: Build comprehensive response with stats
    # ================================================================
    vulcan_systems_active = len(
        [
            s
            for s in systems_used
            if not s.startswith("openai") and s != "fallback_message"
        ]
    )

    # SECURITY FIX: Generate IDs with full cryptographic randomness
    # Old format: f"resp_{int(time.time())}_{secrets.token_hex(4)}"
    # This prevents timing attacks and ID enumeration
    response_id = f"resp_{secrets.token_urlsafe(16)}"
    query_id = routing_plan.query_id if routing_plan else f"q_{secrets.token_urlsafe(12)}"

    response_data = {
        "response": response_text,
        "systems_used": systems_used,
        "vulcan_cognitive_systems_active": vulcan_systems_active,
        "agent_pool_stats": agent_pool_stats if agent_pool_stats else None,
        "gatekeeper": gatekeeper_results if gatekeeper_results else None,
        "insights": (
            {
                "reasoning": reasoning_insights if reasoning_insights else None,
                "world_model": world_model_insights if world_model_insights else None,
                "meta_reasoning": (
                    meta_reasoning_insights if meta_reasoning_insights else None
                ),
            }
            if (reasoning_insights or world_model_insights or meta_reasoning_insights)
            else None
        ),
        # RLHF INTEGRATION: Include IDs for feedback (Task 4 - UI thumbs buttons)
        "query_id": query_id,
        "response_id": response_id,
    }

    # Add routing layer stats if available
    if routing_plan:
        response_data["routing"] = {
            "query_id": routing_plan.query_id,
            "query_type": routing_plan.query_type.value,
            "learning_mode": routing_plan.learning_mode.value,
            "complexity_score": routing_plan.complexity_score,
            "uncertainty_score": routing_plan.uncertainty_score,
            "tasks_planned": len(routing_plan.agent_tasks),
            "tasks_submitted": len(submitted_jobs) if submitted_jobs else 0,
            "collaboration_needed": routing_plan.collaboration_needed,
            "collaboration_agents": (
                routing_plan.collaboration_agents
                if routing_plan.collaboration_needed
                else None
            ),
            "arena_participation": routing_plan.arena_participation,
            "governance_triggered": routing_plan.requires_governance,
        }

    # Add Arena execution info if Arena was used
    if arena_result is not None:
        response_data["arena_execution"] = {
            "status": arena_result.get("status"),
            "agent_id": arena_result.get("agent_id"),
            "execution_time": arena_result.get("execution_time"),
            "error": arena_result.get("error") if arena_result.get("status") != "success" else None,
        }

    return response_data

