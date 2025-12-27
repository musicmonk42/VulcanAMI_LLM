# ============================================================
# VULCAN-AGI Arena Client Module
# Client functions for Graphix Arena API integration
# ============================================================
#
# Arena is the training and execution environment for:
#     1. Agent training - Agents compete in tournaments, winners improve
#     2. Graph language runtime - Graphix IR graphs are executed and evolved
#     3. Language evolution - Successful patterns stored in Registry
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS
# ============================================================

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False
    logger.info("aiohttp not available - Arena API calls disabled")

try:
    from vulcan.settings import settings
except ImportError:
    settings = None
    logger.warning("Settings module not available - using defaults")

try:
    from vulcan.arena.http_session import get_http_session
except ImportError:
    get_http_session = None
    logger.warning("HTTP session module not available")

try:
    from vulcan.utils_main.sanitize import sanitize_payload, deep_sanitize_for_json
except ImportError:
    # Fallback implementations
    def sanitize_payload(data):
        if isinstance(data, dict):
            return {str(k): sanitize_payload(v) for k, v in data.items() if k is not None}
        elif isinstance(data, list):
            return [sanitize_payload(item) for item in data]
        return data
    
    def deep_sanitize_for_json(data, _depth=0):
        return sanitize_payload(data)


# ============================================================
# AGENT SELECTION
# ============================================================


def select_arena_agent(routing_plan) -> str:
    """
    Map query type to appropriate Arena agent.
    
    Arena agents:
    - generator: Creates new graphs from specs (creative/generative tasks)
    - evolver: Mutates existing graphs (optimization/evolution tasks)
    - visualizer: Renders graphs (explanation/visualization tasks)
    - photonic_optimizer: Hardware optimization (photonic/analog tasks)
    - automl_optimizer: Model tuning (hyperparameter/automl tasks)
    
    Args:
        routing_plan: The routing plan from QueryRouter
        
    Returns:
        Agent ID string
    """
    query_type = routing_plan.query_type.value if hasattr(routing_plan.query_type, 'value') else str(routing_plan.query_type)
    
    agent_mapping = {
        # Creative/generative → generator
        "GENERATIVE": "generator",
        "generative": "generator",
        "generation": "generator",
        "creative": "generator",
        "design": "generator",
        
        # Optimization → evolver
        "OPTIMIZATION": "evolver",
        "optimization": "evolver",
        "evolution": "evolver",
        "improve": "evolver",
        
        # Explanation → visualizer
        "PERCEPTION": "visualizer",
        "perception": "visualizer",
        "visualization": "visualizer",
        "explain": "visualizer",
        
        # Reasoning can use generator for graph-based reasoning
        "REASONING": "generator",
        "reasoning": "generator",
        
        # Planning uses generator for planning graphs
        "PLANNING": "generator",
        "planning": "generator",
        
        # Hardware → photonic_optimizer
        "photonic": "photonic_optimizer",
        "hardware": "photonic_optimizer",
        "analog": "photonic_optimizer",
        
        # Model tuning → automl_optimizer
        "automl": "automl_optimizer",
        "hyperparameter": "automl_optimizer",
        "tune": "automl_optimizer",
    }
    
    return agent_mapping.get(query_type, "generator")


def build_arena_payload(query: str, routing_plan, agent_id: str) -> dict:
    """
    Build payload for Arena API based on agent type.
    
    Generator expects GraphSpec format, others expect GraphixIRGraph format.
    
    Args:
        query: The user query
        routing_plan: The routing plan from QueryRouter
        agent_id: The target agent ID
        
    Returns:
        Payload dictionary for Arena API
    """
    query_id = routing_plan.query_id if hasattr(routing_plan, 'query_id') else f"q_{int(time.time() * 1000)}"
    query_type = routing_plan.query_type.value if hasattr(routing_plan.query_type, 'value') else str(routing_plan.query_type)
    complexity = routing_plan.complexity_score if hasattr(routing_plan, 'complexity_score') else 0.5
    
    # CRITICAL FIX: Extract selected_tools from routing_plan for reasoning invocation
    # This enables GraphixArena to invoke reasoning engines when selected_tools are present
    selected_tools = getattr(routing_plan, 'selected_tools', []) or []
    
    if agent_id == "generator":
        # Generator expects GraphSpec format
        return {
            "spec_id": f"query_{query_id}",
            "parameters": {
                "goal": query,
                "query_type": query_type,
                "complexity": complexity,
                "source": "vulcan",
                "timestamp": time.time(),
                "selected_tools": selected_tools,  # Pass reasoning tools to Arena
            }
        }
    else:
        # Evolver, visualizer, etc. expect GraphixIRGraph format
        return {
            "graph_id": f"g_{query_id}",
            "nodes": [
                {
                    "id": "root",
                    "label": "query_input",
                    "properties": {
                        "text": query,
                        "query_type": query_type,
                    }
                }
            ],
            "edges": [],
            "properties": {
                "source": "vulcan",
                "query_id": query_id,
                "query_type": query_type,
                "complexity": complexity,
                "timestamp": time.time(),
                "selected_tools": selected_tools,  # Pass reasoning tools to Arena
            }
        }


# ============================================================
# ARENA API FUNCTIONS
# ============================================================


async def execute_via_arena(query: str, routing_plan, arena_base_url: str = None) -> dict:
    """
    Execute query through Graphix Arena for training + graph execution.
    
    Arena handles:
    - Agent execution (generator/evolver/visualizer/etc)
    - Tournament selection among proposals
    - Feedback integration (RLHF)
    - Governance enforcement
    
    Args:
        query: The user query to process
        routing_plan: The routing plan from QueryRouter
        arena_base_url: Base URL for Arena API (defaults to settings)
        
    Returns:
        dict with Arena execution result including:
        - result: The agent's output
        - agent_id: Which agent was used
        - execution_time: How long it took
        - metrics: Any performance metrics from Arena
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning("[ARENA] aiohttp not available, falling back to VULCAN-only processing")
        return {
            "status": "fallback",
            "reason": "aiohttp not available for Arena API calls",
            "result": None,
        }
    
    if get_http_session is None:
        logger.warning("[ARENA] HTTP session not available")
        return {
            "status": "error",
            "reason": "HTTP session module not available",
            "result": None,
        }
    
    # Get Arena configuration
    base_url = arena_base_url
    api_key = None
    timeout = 120.0  # FIX #6: Increased from 90s to 120s to account for high CPU load
    complexity_threshold = 0.3  # PERFORMANCE FIX: Default fast-path threshold
    
    if settings is not None:
        base_url = base_url or settings.arena_base_url
        api_key = settings.arena_api_key
        timeout = settings.arena_timeout
        complexity_threshold = getattr(settings, 'arena_complexity_threshold', 0.3)
    else:
        base_url = base_url or "http://localhost:8080/arena"
    
    # FIX: Improved Arena threshold logic
    # Previously defaulted complexity to 0.0 which always skipped Arena
    # Now: If complexity_score is explicitly set AND below threshold, skip
    #      If complexity_score is not set, proceed with Arena (let it decide)
    complexity = getattr(routing_plan, 'complexity_score', None)
    
    if complexity is not None:
        if complexity < complexity_threshold:
            logger.info(f"[ARENA] Fast-path skip: complexity {complexity:.2f} < threshold {complexity_threshold:.2f}")
            return {
                "status": "skipped",
                "reason": f"Query complexity ({complexity:.2f}) below threshold ({complexity_threshold:.2f})",
                "result": None,
                "execution_time": 0,
            }
        else:
            logger.info(f"[ARENA] Proceeding: complexity {complexity:.2f} >= threshold {complexity_threshold:.2f}")
    else:
        # No complexity score provided - check if arena_participation flag is set
        arena_flag = getattr(routing_plan, 'arena_participation', False)
        if not arena_flag:
            logger.info("[ARENA] Skipping: no complexity_score and arena_participation=False")
            return {
                "status": "skipped",
                "reason": "No complexity_score provided and arena_participation not enabled",
                "result": None,
                "execution_time": 0,
            }
        logger.info("[ARENA] Proceeding: arena_participation=True (no complexity_score)")
    
    # Select appropriate Arena agent
    agent_id = select_arena_agent(routing_plan)
    
    # Build payload for Arena
    payload = build_arena_payload(query, routing_plan, agent_id)
    
    # CRITICAL FIX: Sanitize payload to remove None keys that cause serialization failures
    payload = sanitize_payload(payload)
    
    # CRITICAL FIX: Pre-serialize to JSON to catch serialization errors early
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as json_err:
        logger.error(f"[ARENA] JSON serialization failed: {json_err}")
        # Attempt deep sanitization as fallback
        payload = deep_sanitize_for_json(payload)
        try:
            payload_json = json.dumps(payload)
            logger.info("[ARENA] Deep sanitization succeeded, retrying serialization")
        except Exception as retry_err:
            logger.error(f"[ARENA] Deep sanitization also failed: {retry_err}")
            return {
                "status": "error",
                "agent_id": agent_id,
                "execution_time": 0,
                "error": f"JSON serialization failed (original: {json_err}, after deep sanitize: {retry_err})",
            }
    
    # Construct Arena API URL
    url = f"{base_url}/api/run/{agent_id}"
    
    # Warn if API key is not configured
    if not api_key:
        logger.warning("[ARENA] API key not configured - Arena request may fail authentication")
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    logger.info(f"[ARENA] Executing via {agent_id}: {url}")
    t0 = time.perf_counter()
    
    try:
        session = await get_http_session()
        async with session.post(
            url,
            data=payload_json,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            elapsed = time.perf_counter() - t0
            
            if resp.status == 200:
                result = await resp.json()
                logger.info(f"[ARENA] {agent_id} completed in {elapsed:.2f}s")
                
                return {
                    "status": "success",
                    "agent_id": agent_id,
                    "execution_time": elapsed,
                    "result": result,
                    "arena_url": url,
                }
            else:
                error_text = await resp.text()
                logger.error(f"[ARENA] {agent_id} failed: {resp.status} - {error_text}")
                
                return {
                    "status": "error",
                    "agent_id": agent_id,
                    "execution_time": elapsed,
                    "error": error_text,
                    "status_code": resp.status,
                }
                    
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        logger.error(f"[ARENA] {agent_id} timeout after {elapsed:.2f}s")
        return {
            "status": "timeout",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": f"Arena request timed out after {timeout}s",
        }
    except aiohttp.ClientError as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[ARENA] {agent_id} connection error: {e}")
        return {
            "status": "connection_error",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": str(e),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[ARENA] {agent_id} unexpected error: {e}")
        return {
            "status": "error",
            "agent_id": agent_id,
            "execution_time": elapsed,
            "error": str(e),
        }


async def submit_arena_feedback(
    proposal_id: str,
    score: float,
    rationale: str,
    arena_base_url: str = None
) -> dict:
    """
    Submit feedback to Arena for RLHF (Reinforcement Learning from Human Feedback).
    
    This enables the evolution loop where:
    - User feedback influences agent training
    - Successful patterns are reinforced
    - Losers get diversity penalty applied
    
    Args:
        proposal_id: ID of the proposal/graph to provide feedback on
        score: Feedback score (typically -1.0 to 1.0)
        rationale: Human-readable explanation of the feedback
        arena_base_url: Base URL for Arena API
        
    Returns:
        dict with feedback submission result
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning("[ARENA] aiohttp not available, cannot submit feedback")
        return {"status": "error", "reason": "aiohttp not available"}
    
    if get_http_session is None:
        return {"status": "error", "reason": "HTTP session module not available"}
    
    # Get configuration
    base_url = arena_base_url
    api_key = None
    
    if settings is not None:
        base_url = base_url or settings.arena_base_url
        api_key = settings.arena_api_key
    else:
        base_url = base_url or "http://localhost:8080/arena"
    
    url = f"{base_url}/api/feedback"
    
    payload = {
        "graph_id": proposal_id,
        "agent_id": "vulcan",
        "score": score,
        "rationale": rationale,
    }
    
    # Sanitize and serialize
    payload = sanitize_payload(payload)
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as json_err:
        logger.error(f"[ARENA] Feedback JSON serialization failed: {json_err}")
        payload = deep_sanitize_for_json(payload)
        try:
            payload_json = json.dumps(payload)
        except Exception as retry_err:
            return {"status": "error", "error": f"JSON serialization failed"}
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        session = await get_http_session()
        async with session.post(
            url,
            data=payload_json,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(f"[ARENA] Feedback submitted for {proposal_id}: score={score}")
                return {"status": "success", "result": result}
            else:
                error_text = await resp.text()
                logger.error(f"[ARENA] Feedback submission failed: {resp.status}")
                return {"status": "error", "error": error_text}
                    
    except Exception as e:
        logger.error(f"[ARENA] Feedback submission error: {e}")
        return {"status": "error", "error": str(e)}


# Backward compatibility aliases
_select_arena_agent = select_arena_agent
_build_arena_payload = build_arena_payload
_execute_via_arena = execute_via_arena
_submit_arena_feedback = submit_arena_feedback


__all__ = [
    "execute_via_arena",
    "submit_arena_feedback",
    "select_arena_agent",
    "build_arena_payload",
    "AIOHTTP_AVAILABLE",
    # Backward compatibility
    "_select_arena_agent",
    "_build_arena_payload",
    "_execute_via_arena",
    "_submit_arena_feedback",
]
