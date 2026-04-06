"""
state_metrics.py - Error reporting and performance metrics for World Model.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def report_error(wm, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Report an error to the world model (triggers self-improvement)"""

    with wm.lock:
        wm.system_state["error_count"] += 1
        wm.system_state["errors_in_window"].append(
            {
                "timestamp": time.time(),
                "error": str(error),
                "type": type(error).__name__,
                "context": context or {},
            }
        )

        logger.warning(f"Error reported to world model: {error}")


def update_performance_metric(wm, metric: str, value: float):
    """Update performance metric (feeds into self-improvement triggers)"""

    with wm.lock:
        old_value = wm.system_state["performance_metrics"].get(metric)
        wm.system_state["performance_metrics"][metric] = value

        # Calculate degradation if we have baseline
        if old_value is not None and old_value > 0:
            degradation = ((value - old_value) / old_value) * 100
            wm.system_state["performance_metrics"][
                f"{metric}_degradation_percent"
            ] = degradation


def get_improvement_status(wm) -> Dict[str, Any]:
    """Get current self-improvement status"""

    if not wm.self_improvement_enabled:
        return {
            "enabled": False,
            "reason": "Self-improvement drive not initialized",
        }

    status = wm.self_improvement_drive.get_status()

    # Add meta-reasoning stats if available
    if wm.meta_reasoning_enabled:
        meta_stats = wm.motivational_introspection.get_statistics()
        status["meta_reasoning"] = meta_stats

    status["system_state"] = wm.system_state.copy()

    return status


def handle_improvement_alert(wm, severity: str, alert_data: Dict[str, Any]):
    """Handle a self-improvement alert."""
    log_level = logging.WARNING if severity == "warning" else logging.INFO
    logger.log(
        log_level,
        f"Self-improvement alert [{severity}]: {alert_data.get('message', str(alert_data))}",
    )


def check_improvement_approval(wm, approval_id: str) -> Optional[str]:
    """Check approval status (integrate with your approval system)"""
    # TODO: Integrate with actual approval system
    return None
