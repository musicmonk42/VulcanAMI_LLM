"""
self_improvement_engine.py - Extracted self-improvement loop functions from WorldModel.

Contains the autonomous improvement loop, context building, file loading,
and LLM prompt building for improvements.
Phase 1 of WorldModel decomposition.
"""

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def start_autonomous_improvement(wm) -> None:
    """Start the autonomous self-improvement background thread."""
    import threading

    if not wm.self_improvement_enabled:
        logger.warning(
            "Cannot start autonomous improvement - self-improvement drive not initialized"
        )
        return

    if wm.improvement_thread and wm.improvement_thread.is_alive():
        logger.warning("Self-improvement thread already running")
        return

    wm.improvement_running = True
    wm.improvement_thread = threading.Thread(
        target=lambda: _autonomous_improvement_loop(wm),
        name="VulcanSelfImprovement",
        daemon=True,
    )
    wm.improvement_thread.start()

    logger.info("Autonomous self-improvement drive started")


def stop_autonomous_improvement(wm) -> None:
    """Stop the autonomous self-improvement drive."""
    wm.improvement_running = False

    if wm.improvement_thread:
        wm.improvement_thread.join(timeout=5.0)

    logger.info("Autonomous self-improvement drive stopped")


def _autonomous_improvement_loop(wm) -> None:
    """Main loop for autonomous self-improvement."""
    logger.info("Autonomous improvement loop starting")

    check_interval = int(os.getenv("SELF_IMPROVEMENT_INTERVAL", "86400"))
    logger.info(f"Self-improvement check interval: {check_interval} seconds")

    kill_switch_env = os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", "1").lower()
    if kill_switch_env in ("0", "false", "no", "off"):
        logger.warning(
            "Self-improvement disabled via VULCAN_ENABLE_SELF_IMPROVEMENT=0. "
            "Exiting autonomous improvement loop."
        )
        wm.improvement_running = False
        return

    while wm.improvement_running:
        try:
            kill_switch_env = os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", "1").lower()
            if kill_switch_env in ("0", "false", "no", "off"):
                logger.warning(
                    "Self-improvement disabled via VULCAN_ENABLE_SELF_IMPROVEMENT=0. "
                    "Stopping autonomous improvement loop."
                )
                wm.improvement_running = False
                break

            context = _build_improvement_context(wm)

            if wm.self_improvement_drive.should_trigger(context):
                logger.info("Self-improvement drive triggered!")

                improvement_action = wm.self_improvement_drive.step(context)

                if improvement_action:
                    if improvement_action.get("_wait_for_approval"):
                        approval_id = improvement_action["_pending_approval"]
                        logger.info(f"Waiting for approval: {approval_id}")
                    else:
                        from .self_improvement_apply import _execute_improvement
                        _execute_improvement(wm, improvement_action)

            time.sleep(check_interval)

        except Exception as e:
            logger.error(
                f"Error in autonomous improvement loop: {e}", exc_info=True
            )
            time.sleep(check_interval)

    logger.info("Autonomous improvement loop stopped")


def _build_improvement_context(wm) -> Dict[str, Any]:
    """Build context for self-improvement decisions."""
    with wm.lock:
        is_startup = (
            time.time() - wm.system_state["session_start"]
        ) < 300

        current_time = time.time()
        window = 3600
        recent_errors = [
            e
            for e in wm.system_state["errors_in_window"]
            if e["timestamp"] > current_time - window
        ]

        cpu_percent = wm._get_cpu_usage()
        memory_mb = wm._get_memory_usage()

        context = {
            "is_startup": is_startup,
            "error_detected": len(recent_errors) > 0,
            "error_count": len(recent_errors),
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "low_activity_duration_minutes": wm._get_low_activity_duration(),
            },
            "performance_metrics": wm.system_state["performance_metrics"].copy(),
            "other_drives_total_priority": 0.0,
        }

        return context


def _load_file(wm, file_path: str) -> str:
    """Load a file relative to repo root."""
    try:
        full_path = wm.repo_root / file_path
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        logger.warning(f"Failed to load file {file_path}: {e}")
        raise


def _build_llm_prompt_for_improvement(wm, action: Dict[str, Any]) -> str:
    """Build the detailed prompt for the LLM based on improvement objective."""
    objective_type = action.get("_drive_metadata", {}).get(
        "objective_type", "System Improvement"
    )
    goal = action.get("high_level_goal", "Perform general system hygiene.")
    raw_obs = json.dumps(action.get("raw_observation", {}), indent=2)

    prompt = f"""
        Objective: {objective_type}
        High Level Goal: {goal}

        Please provide the necessary code changes to achieve this goal. Focus only on one Python file for this single step.

        You must specify the target file path and the complete, updated content of that file.

        Format your response exactly as follows:
        FILE: <path/relative/to/repo/root/file.py>
        ```python
        <complete updated file content here>
        ```
        Task Details:
        {raw_obs}
        """
    return prompt
