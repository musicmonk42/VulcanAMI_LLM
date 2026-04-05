"""
Execution and Streaming Endpoints

This module provides endpoints for executing cognitive steps and streaming
continuous execution with resource monitoring.

Endpoints:
    POST /v1/step   - Execute single cognitive step with timeout
    GET /v1/stream  - Stream continuous execution with resource monitoring
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import step_counter, step_duration, active_requests, error_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["execution"])


@router.post("/v1/step", response_model=None)
async def execute_step(request: Request) -> dict:
    """
    Execute single cognitive step with timeout and resource limits.
    
    Runs a single step of the VULCAN cognitive engine with monitoring
    and timeout protection. Each step processes the history and context
    to produce a reasoning result.
    
    Args:
        request: FastAPI request with StepRequest body containing:
            - history: List of previous interaction steps
            - context: Context dictionary for the step
            - timeout: Optional timeout override in seconds
    
    Returns:
        Dict containing the step execution result from deployment
    
    Raises:
        HTTPException: 503 if system not initialized
        HTTPException: 504 if execution timeout exceeded
        HTTPException: 500 if execution fails
    
    Note:
        This endpoint uses Prometheus metrics for monitoring:
        - active_requests: Gauge tracking concurrent requests
        - step_duration: Histogram of step execution times
        - step_counter: Counter of total steps executed
        - error_counter: Counter of errors by type
    """
    app = request.app
    
    deployment = require_deployment(request)
    if deployment is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Get settings
    settings = getattr(app.state, "settings", None)

    active_requests.inc()

    try:
        # Get request body
        from vulcan.api.models import StepRequest
        body = await request.json()
        step_request = StepRequest(**body)
        
        timeout = step_request.timeout or (getattr(settings, "max_execution_time_s", 30.0) if settings else 30.0)

        loop = asyncio.get_running_loop()

        # Time the execution with metrics
        with step_duration.time():
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    deployment.step_with_monitoring,
                    step_request.history,
                    step_request.context,
                ),
                timeout=timeout,
            )

        step_counter.inc()
        return result

    except asyncio.TimeoutError:
        error_counter.labels(error_type="timeout").inc()
        if 'step_request' in locals():
            timeout_val = step_request.timeout
        elif settings:
            timeout_val = getattr(settings, "max_execution_time_s", 30.0)
        else:
            timeout_val = 30.0
        logger.error(f"Step execution timeout after {timeout_val}s")
        raise HTTPException(
            status_code=504, detail=f"Execution timeout after {timeout_val}s"
        )

    except Exception as e:
        error_counter.labels(error_type="execution").inc()
        logger.error(f"Step execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        active_requests.dec()


@router.get("/v1/stream")
async def stream_execution(request: Request) -> StreamingResponse:
    """
    Stream continuous execution with resource monitoring.
    
    Provides a Server-Sent Events (SSE) stream of continuous cognitive
    execution. Each iteration runs a step and yields the result as JSON.
    Includes safety limits on iterations, duration, and memory usage.
    
    Args:
        request: FastAPI request object for accessing app state
    
    Returns:
        StreamingResponse with text/event-stream media type
    
    Raises:
        HTTPException: 503 if system not initialized
    
    Note:
        The stream automatically terminates if:
        - Maximum iterations reached (1000)
        - Maximum duration exceeded (300 seconds)
        - Memory limit exceeded
        - Critical error occurs
        
        Each event is formatted as:
        data: {"result": ..., "iteration": ...}
        
        Errors are yielded as:
        data: {"error": "error message"}
    """
    app = request.app
    
    deployment = require_deployment(request)
    settings = getattr(app.state, "settings", None)
    max_memory_mb = getattr(settings, "max_memory_mb", 2000) if settings else 2000

    async def generate() -> AsyncGenerator[str, None]:
        """Generate stream of execution steps."""
        iteration = 0
        max_iterations = 1000
        start_time = time.time()
        max_duration = 300

        try:
            while iteration < max_iterations:
                if time.time() - start_time > max_duration:
                    yield f'data: {{"error": "Maximum stream duration exceeded"}}\n\n'
                    break

                try:
                    # Check status before running the step
                    status = deployment.get_status()
                    if status["health"]["memory_usage_mb"] > max_memory_mb:
                        yield f'data: {{"error": "Memory limit exceeded"}}\n\n'
                        break

                    loop = asyncio.get_running_loop()
                    # Use a short timeout for stability in stream
                    step_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            deployment.step_with_monitoring,
                            [],
                            {"high_level_goal": "explore", "iteration": iteration},
                        ),
                        timeout=5.0,
                    )

                    # Ensure the result is serializable before yielding
                    yield f"data: {json.dumps(step_result, default=str)}\n\n"
                    iteration += 1
                    await asyncio.sleep(0.1)

                except asyncio.TimeoutError:
                    logger.warning("Stream step execution timeout, continuing stream...")
                    yield f'data: {{"warning": "Step timeout, continuing stream"}}\n\n'
                    iteration += 1
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Stream execution error: {e}")
                    yield f'data: {{"error": "{str(e)}"}}\n\n'
                    break

        except asyncio.CancelledError:
            logger.info("Stream cancelled by client")
            yield f'data: {{"status": "cancelled"}}\n\n'
        except Exception as e:
            logger.critical(f"Unexpected stream generator error: {e}", exc_info=True)
            yield f'data: {{"error": "Critical internal stream error: {str(e)}"}}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
