# ============================================================
# VULCAN-AGI Metrics Package
# Prometheus metrics definitions for VULCAN-AGI
# ============================================================
#
# This package provides:
#     - Prometheus metric definitions
#     - Safe metric registration (handles re-imports)
#     - Standard metrics for requests, errors, and performance
#     - Self-improvement metrics
#
# USAGE:
#     from vulcan.metrics import step_counter, step_duration, error_counter
#     
#     step_counter.inc()
#     with step_duration.time():
#         # do work
#         pass
#     error_counter.labels(error_type="validation").inc()
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
from typing import Any, Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# PROMETHEUS IMPORTS
# ============================================================

try:
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None
    Gauge = None
    Histogram = None
    REGISTRY = None
    generate_latest = None
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not available - metrics disabled")


# ============================================================
# METRIC REGISTRATION HELPER
# ============================================================


def get_or_create_metric(metric_class, name: str, description: str, labelnames: List[str] = None):
    """
    Safely get or create a Prometheus metric (handles module re-imports).
    
    This function prevents the "Metric already exists" error that occurs
    when the module is re-imported (e.g., during testing or hot reloading).
    
    Args:
        metric_class: The Prometheus metric class (Counter, Gauge, Histogram)
        name: The metric name
        description: The metric description
        labelnames: Optional list of label names
        
    Returns:
        The metric instance (existing or newly created)
    """
    if not PROMETHEUS_AVAILABLE or REGISTRY is None:
        # Return a mock metric that does nothing
        return MockMetric(name)
    
    # Check if already registered by name
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    
    try:
        if labelnames:
            return metric_class(name, description, labelnames)
        return metric_class(name, description)
    except ValueError:
        # Race condition fallback - another import registered it
        return REGISTRY._names_to_collectors.get(name)


class MockMetric:
    """Mock metric for when prometheus_client is not available."""
    
    def __init__(self, name: str):
        self.name = name
    
    def inc(self, amount: float = 1):
        pass
    
    def dec(self, amount: float = 1):
        pass
    
    def set(self, value: float):
        pass
    
    def observe(self, amount: float):
        pass
    
    def labels(self, **kwargs):
        return self
    
    def time(self):
        return MockTimer()


class MockTimer:
    """Mock timer context manager."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# ============================================================
# STANDARD METRICS
# ============================================================

# Request metrics
step_counter = get_or_create_metric(
    Counter, "vulcan_steps_total", "Total steps executed"
)
step_duration = get_or_create_metric(
    Histogram, "vulcan_step_duration_seconds", "Step execution time"
)
active_requests = get_or_create_metric(
    Gauge, "vulcan_active_requests", "Number of active requests"
)

# Error metrics
error_counter = get_or_create_metric(
    Counter, "vulcan_errors_total", "Total errors", ["error_type"]
)
auth_failures = get_or_create_metric(
    Counter, "vulcan_auth_failures_total", "Authentication failures"
)

# ============================================================
# SELF-IMPROVEMENT METRICS
# ============================================================

improvement_attempts = get_or_create_metric(
    Counter,
    "vulcan_improvement_attempts_total",
    "Total improvement attempts",
    ["objective_type"],
)
improvement_successes = get_or_create_metric(
    Counter,
    "vulcan_improvement_successes_total",
    "Successful improvements",
    ["objective_type"],
)
improvement_failures = get_or_create_metric(
    Counter,
    "vulcan_improvement_failures_total",
    "Failed improvements",
    ["objective_type"],
)
improvement_cost = get_or_create_metric(
    Counter, "vulcan_improvement_cost_usd_total", "Total improvement cost in USD"
)
improvement_approvals_pending = get_or_create_metric(
    Gauge, "vulcan_improvement_approvals_pending", "Number of pending approvals"
)


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Helper
    "get_or_create_metric",
    "MockMetric",
    # Standard metrics
    "step_counter",
    "step_duration",
    "active_requests",
    "error_counter",
    "auth_failures",
    # Self-improvement metrics
    "improvement_attempts",
    "improvement_successes",
    "improvement_failures",
    "improvement_cost",
    "improvement_approvals_pending",
    # Prometheus exports
    "Counter",
    "Gauge",
    "Histogram",
    "REGISTRY",
    "generate_latest",
]


# Backward compatibility alias
_get_or_create_metric = get_or_create_metric

# Log module initialization
if PROMETHEUS_AVAILABLE:
    logger.debug(f"Metrics module v{__version__} loaded with Prometheus support")
else:
    logger.debug(f"Metrics module v{__version__} loaded (Prometheus not available)")
