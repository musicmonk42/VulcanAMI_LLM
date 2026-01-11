"""
System Status Endpoints

This module provides comprehensive status endpoints for monitoring VULCAN's
health, cognitive subsystems, LLM availability, and routing layer.

Endpoints:
    GET /v1/status             - Overall system status
    GET /v1/cognitive/status   - Cognitive subsystems status
    GET /v1/llm/status        - LLM availability and configuration
    GET /v1/routing/status    - Query routing and dual-mode learning
    POST /v1/checkpoint       - Trigger manual checkpoint save
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict
from unittest.mock import MagicMock

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import error_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["status"])


@router.get("/v1/status")
async def system_status(request: Request) -> Dict[str, Any]:
    """
    Get detailed system status.
    
    Provides comprehensive overview of VULCAN's operational state including:
    - Deployment configuration and uptime
    - Total steps executed
    - Worker ID
    - Self-improvement status (if enabled)
    - LLM initialization status
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing system status with nested health metrics
    
    Raises:
        HTTPException: 503 if system not initialized
        HTTPException: 500 if status retrieval fails
    """
    app = request.app
    
    deployment = require_deployment(request)
    settings = getattr(app.state, "settings", None)

    try:
        status = deployment.get_status()

        uptime = (
            time.time() - app.state.startup_time
            if hasattr(app.state, "startup_time")
            else 0
        )

        status["deployment"] = {
            "mode": getattr(settings, "deployment_mode", "unknown") if settings else "unknown",
            "api_version": getattr(settings, "api_version", "unknown") if settings else "unknown",
            "uptime_seconds": uptime,
            "total_steps": status.get("step", 0),
            "worker_id": getattr(app.state, "worker_id", "unknown"),
        }

        # Add self-improvement status
        try:
            world_model = deployment.collective.deps.world_model
            if world_model and hasattr(world_model, "self_improvement_enabled"):
                status["self_improvement"] = {
                    "enabled": world_model.self_improvement_enabled,
                    "running": getattr(world_model, "improvement_running", False),
                }
        except Exception as e:
            logger.debug(f"Could not get self-improvement status: {e}")

        # Add LLM status
        status["llm"] = {
            "initialized": hasattr(app.state, "llm")
            and not isinstance(app.state.llm, MagicMock),
            "mocked": (
                isinstance(app.state.llm, MagicMock)
                if hasattr(app.state, "llm")
                else False
            ),
        }

        return status

    except Exception as e:
        error_counter.labels(error_type="status").inc()
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/cognitive/status")
async def cognitive_status(request: Request) -> Dict[str, Any]:
    """
    Get detailed status of VULCAN's cognitive subsystems.
    
    Shows which cognitive systems are active and their current state:
    - Agent Pool (distributed processing with detailed metrics)
    - Reasoning Systems (symbolic, probabilistic, causal, analogical, cross-modal)
    - Memory Systems (long-term, episodic, compressed)
    - Processing Systems (multimodal)
    - World Model (predictive modeling with meta-reasoning)
    - Learning Systems (continual, meta-cognitive, compositional)
    - Planning Systems (goal system, resource-aware compute)
    - Safety Systems (validator, governance, NSO aligner)
    - Self-Improvement Systems (drive, experiment generator, problem executor)
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - vulcan_cognitive_systems: Detailed status of each subsystem
            - summary: Aggregate statistics and OpenAI availability
            - timestamp: Current Unix timestamp
    
    Raises:
        HTTPException: 503 if VULCAN deployment not initialized
    """
    app = request.app
    
    deployment = require_deployment(request)
    deps = deployment.collective.deps

    cognitive_systems = {
        "agent_pool": {
            "active": hasattr(deployment.collective, "agent_pool")
            and deployment.collective.agent_pool is not None,
            "status": None,
        },
        "reasoning": {
            "symbolic": deps.symbolic is not None,
            "probabilistic": deps.probabilistic is not None,
            "causal": deps.causal is not None,
            "analogical": deps.abstract is not None,
            "cross_modal": deps.cross_modal is not None,
        },
        "memory": {
            "long_term": deps.ltm is not None,
            "episodic": deps.am is not None,
            "compressed": deps.compressed_memory is not None,
        },
        "processing": {
            "multimodal": deps.multimodal is not None,
        },
        "world_model": {
            "active": deps.world_model is not None,
            "meta_reasoning_enabled": False,
            "self_improvement_enabled": False,
        },
        "learning": {
            "continual": deps.continual is not None,
            "meta_cognitive": deps.meta_cognitive is not None,
            "compositional": deps.compositional is not None,
        },
        "planning": {
            "goal_system": deps.goal_system is not None,
            "resource_compute": deps.resource_compute is not None,
        },
        "safety": {
            "validator": deps.safety_validator is not None,
            "governance": deps.governance is not None,
            "nso_aligner": deps.nso_aligner is not None,
        },
        "self_improvement": {
            "drive_active": deps.self_improvement_drive is not None,
            "experiment_generator": deps.experiment_generator is not None,
            "problem_executor": deps.problem_executor is not None,
        },
    }

    # Get detailed agent pool status
    if cognitive_systems["agent_pool"]["active"]:
        try:
            pool_status = deployment.collective.agent_pool.get_pool_status()
            cognitive_systems["agent_pool"]["status"] = {
                "total_agents": pool_status.get("total_agents", 0),
                "idle": pool_status.get("state_distribution", {}).get("idle", 0),
                "working": pool_status.get("state_distribution", {}).get("working", 0),
                "max_agents": deployment.collective.agent_pool.max_agents,
                "min_agents": deployment.collective.agent_pool.min_agents,
            }
        except Exception as e:
            cognitive_systems["agent_pool"]["status"] = {"error": str(e)}

    # Check world model meta-reasoning
    if deps.world_model:
        try:
            if hasattr(deps.world_model, "motivational_introspection"):
                cognitive_systems["world_model"]["meta_reasoning_enabled"] = (
                    deps.world_model.motivational_introspection is not None
                )
            if hasattr(deps.world_model, "self_improvement_enabled"):
                cognitive_systems["world_model"][
                    "self_improvement_enabled"
                ] = deps.world_model.self_improvement_enabled
        except Exception:
            pass

    # Calculate summary statistics
    total_systems = 0
    active_systems = 0

    def count_systems(obj):
        nonlocal total_systems, active_systems
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, bool):
                    total_systems += 1
                    if value:
                        active_systems += 1
                elif isinstance(value, dict):
                    count_systems(value)

    count_systems(cognitive_systems)
    
    # Check OpenAI availability
    openai_available = False
    try:
        from vulcan.llm import OPENAI_AVAILABLE, get_openai_client
        openai_available = OPENAI_AVAILABLE and get_openai_client() is not None
    except ImportError:
        pass

    return {
        "vulcan_cognitive_systems": cognitive_systems,
        "summary": {
            "total_subsystems": total_systems,
            "active_subsystems": active_systems,
            "activation_percentage": round(
                (active_systems / total_systems * 100) if total_systems > 0 else 0, 1
            ),
            "openai_fallback_available": openai_available,
            "vulcan_primary": True,  # VULCAN systems are always primary
        },
        "timestamp": time.time(),
    }


@router.get("/v1/llm/status")
async def llm_status(request: Request) -> Dict[str, Any]:
    """
    Diagnostic endpoint to check LLM availability and configuration.
    
    Use this to debug OpenAI API issues on Railway or other deployments.
    Provides detailed information about:
    - OpenAI package availability
    - API key configuration (length for debugging truncated keys)
    - Client initialization status
    - Local LLM availability
    - Fallback chain for request handling
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - openai: OpenAI client status and configuration
            - local_llm: Local LLM availability
            - fallback_chain: Order of LLM fallback attempts
            - recommendation: Actionable advice on configuration
            - timestamp: Current Unix timestamp
    """
    app = request.app
    
    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    openai_key_length = len(os.getenv("OPENAI_API_KEY", "")) if openai_key_set else 0
    
    # Check OpenAI client status
    openai_available = False
    openai_client = None
    init_error = None
    
    try:
        from vulcan.llm import OPENAI_AVAILABLE, get_openai_client, get_openai_init_error
        openai_available = OPENAI_AVAILABLE
        openai_client = get_openai_client()
        init_error = get_openai_init_error()
    except ImportError:
        init_error = "vulcan.llm module not available"
    
    return {
        "openai": {
            "package_available": openai_available,
            "api_key_set": openai_key_set,
            "api_key_length": openai_key_length,  # For debugging truncated keys
            "client_initialized": openai_client is not None,
            "initialization_error": init_error,
        },
        "local_llm": {
            "available": hasattr(app.state, "llm") and app.state.llm is not None,
            "type": type(getattr(app.state, "llm", None)).__name__ if hasattr(app.state, "llm") else None,
        },
        "fallback_chain": [
            "1. VULCAN Local LLM (if available)",
            "2. OpenAI API (gpt-3.5-turbo)",
            "3. Reasoning-based response generation",
            "4. Static fallback message",
        ],
        "recommendation": (
            "OpenAI ready" if openai_client else
            f"OpenAI not available: {init_error or 'Unknown reason'}"
        ),
        "timestamp": time.time(),
    }


@router.get("/v1/routing/status")
async def routing_status() -> Dict[str, Any]:
    """
    Get detailed status of VULCAN's Query Routing and Dual-Mode Learning Integration.
    
    Shows:
    - Query Router status (query classification, complexity scoring)
    - Agent Collaboration status (multi-agent sessions)
    - Telemetry status (user/AI interaction counts, memory populations)
    - Governance status (audit logs, compliance checks, quarantine)
    - Experiment Trigger status (conditions, proposals)
    
    Returns:
        Dict containing:
            - routing_layer: Initialization status and components
            - dual_mode_learning: Interaction counts and triggered events
            - memory_populations: Memory system populations
            - governance: Audit and compliance statistics
            - telemetry_stats: Detailed telemetry metrics
            - collaboration_stats: Collaboration session metrics
            - governance_stats: Detailed governance metrics
            - experiment_stats: Experiment trigger metrics
            - query_router_stats: Query analysis statistics
            - timestamp: Current Unix timestamp
    """
    status = {
        "routing_layer": {"initialized": False, "components": {}},
        "dual_mode_learning": {
            "user_interactions": 0,
            "ai_interactions": 0,
            "total_collaborations": 0,
            "tournaments_triggered": 0,
            "experiments_triggered": 0,
        },
        "memory_populations": {},
        "governance": {
            "audit_logs": 0,
            "compliance_checks": 0,
            "quarantine_logs": 0,
        },
        "timestamp": time.time(),
    }

    try:
        from vulcan.routing import (
            get_routing_status,
            get_telemetry_recorder,
            get_governance_logger,
            get_experiment_trigger,
            get_collaboration_manager,
            QUERY_ROUTER_AVAILABLE,
            COLLABORATION_AVAILABLE,
            TELEMETRY_AVAILABLE,
            GOVERNANCE_AVAILABLE,
            EXPERIMENT_AVAILABLE,
        )

        # Get comprehensive routing status
        routing_info = get_routing_status()
        status["routing_layer"]["initialized"] = routing_info.get("initialized", False)
        status["routing_layer"]["components"] = routing_info.get("components", {})

        # Get telemetry stats
        if TELEMETRY_AVAILABLE:
            try:
                recorder = get_telemetry_recorder()
                telemetry_stats = recorder.get_stats()
                status["dual_mode_learning"]["user_interactions"] = telemetry_stats.get(
                    "user_interactions", 0
                )
                status["dual_mode_learning"]["ai_interactions"] = telemetry_stats.get(
                    "ai_interactions", 0
                )
                status["dual_mode_learning"]["agent_collaborations"] = (
                    telemetry_stats.get("agent_collaborations", 0)
                )
                status["dual_mode_learning"]["tournaments_triggered"] = (
                    telemetry_stats.get("tournaments", 0)
                )
                status["telemetry_stats"] = telemetry_stats
            except Exception as e:
                status["telemetry_stats"] = {"error": str(e)}

        # Get collaboration stats
        if COLLABORATION_AVAILABLE:
            try:
                collab_manager = get_collaboration_manager()
                collab_stats = collab_manager.get_stats()
                status["dual_mode_learning"]["total_collaborations"] = collab_stats.get(
                    "total_collaborations", 0
                )
                status["collaboration_stats"] = collab_stats
            except Exception as e:
                status["collaboration_stats"] = {"error": str(e)}

        # Get governance stats
        if GOVERNANCE_AVAILABLE:
            try:
                gov_logger = get_governance_logger()
                gov_stats = gov_logger.get_stats()
                status["governance"]["audit_logs"] = gov_stats.get("audit_log_count", 0)
                status["governance"]["compliance_checks"] = gov_stats.get(
                    "compliance_check_count", 0
                )
                status["governance"]["quarantine_logs"] = gov_stats.get(
                    "quarantine_count", 0
                )
                status["governance_stats"] = gov_stats
            except Exception as e:
                status["governance_stats"] = {"error": str(e)}

        # Get experiment stats
        if EXPERIMENT_AVAILABLE:
            try:
                trigger = get_experiment_trigger()
                exp_stats = trigger.get_stats()
                status["dual_mode_learning"]["experiments_triggered"] = exp_stats.get(
                    "experiments_triggered", 0
                )
                status["experiment_stats"] = exp_stats
            except Exception as e:
                status["experiment_stats"] = {"error": str(e)}

        # Get query router stats
        if QUERY_ROUTER_AVAILABLE:
            try:
                from vulcan.routing import get_query_analyzer

                analyzer = get_query_analyzer()
                router_stats = analyzer.get_stats()
                status["query_router_stats"] = router_stats
            except Exception as e:
                status["query_router_stats"] = {"error": str(e)}

    except ImportError:
        status["routing_layer"]["error"] = "Routing module not available"
    except Exception as e:
        status["routing_layer"]["error"] = str(e)

    return status


@router.post("/v1/checkpoint")
async def save_checkpoint(request: Request) -> Dict[str, str]:
    """
    Manually trigger checkpoint save.
    
    Saves the current state of the VULCAN deployment to a checkpoint file
    for later restoration. Useful for preserving learned state or creating
    backups before major operations.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        Dict containing:
            - status: "saved" on success
            - path: Path to the saved checkpoint file
    
    Raises:
        HTTPException: 503 if system not initialized
        HTTPException: 500 if checkpoint save fails
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        checkpoint_path = f"manual_checkpoint_{int(time.time())}.pkl"
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None, deployment.save_checkpoint, checkpoint_path
        )

        if success:
            return {"status": "saved", "path": checkpoint_path}
        else:
            raise HTTPException(status_code=500, detail="Checkpoint save failed")

    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="checkpoint").inc()
        logger.error(f"Checkpoint save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
