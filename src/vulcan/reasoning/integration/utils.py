"""
Utility functions and conveniences for reasoning integration.

This module contains:
- Convenience functions (apply_reasoning, run_portfolio_reasoning, etc.)
- Observer integration functions
- Statistics tracking
- Singleton management
- Module-level utilities

Module: vulcan.reasoning.integration.utils
Author: Vulcan AI Team
"""

# NOTE: This is a STUB module during refactoring
# Utility functions are still at module level in parent reasoning_integration.py
# This will be properly extracted in subsequent commits

from typing import Any, Dict, List, Optional

# Re-export from parent module temporarily during refactoring
from ..reasoning_integration import (
    apply_reasoning,
    run_portfolio_reasoning,
    get_reasoning_integration,
    get_reasoning_statistics,
    shutdown_reasoning,
    observe_query_start,
    observe_engine_result,
    observe_outcome,
    observe_validation_failure,
    observe_error,
)

__all__ = [
    "apply_reasoning",
    "run_portfolio_reasoning",
    "get_reasoning_integration",
    "get_reasoning_statistics",
    "shutdown_reasoning",
    "observe_query_start",
    "observe_engine_result",
    "observe_outcome",
    "observe_validation_failure",
    "observe_error",
]
