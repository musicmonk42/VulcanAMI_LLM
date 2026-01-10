"""
ReasoningIntegration orchestrator - Main class for reasoning integration.

This module contains the main ReasoningIntegration class that orchestrates
tool selection, strategy determination, and reasoning execution.

Module: vulcan.reasoning.integration.orchestrator
Author: Vulcan AI Team
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

# Import from integration subpackage
from .types import (
    LOG_PREFIX,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIME_BUDGET_MS,
    DEFAULT_ENERGY_BUDGET_MJ,
    DEFAULT_MIN_CONFIDENCE,
    MAX_FALLBACK_ATTEMPTS,
    MIN_CONFIDENCE_FLOOR,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_GOOD_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    FAST_PATH_COMPLEXITY_THRESHOLD,
    LOW_COMPLEXITY_THRESHOLD,
    HIGH_COMPLEXITY_THRESHOLD,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
    CAUSAL_REASONING_THRESHOLD,
    PROBABILISTIC_REASONING_THRESHOLD,
    MAX_TIMING_SAMPLES,
    ANALYSIS_INDICATORS,
    ACTION_VERBS,
    ETHICAL_ANALYSIS_INDICATORS,
    PURE_ETHICAL_PHRASES,
    ReasoningStrategyType,
    QUERY_TYPE_STRATEGY_MAP,
    ROUTE_TO_REASONING_TYPE,
    RoutingDecision,
    ReasoningResult,
    IntegrationStatistics,
)

from .safety_checker import (
    is_false_positive_safety_block,
    is_result_safety_filtered,
    get_safety_filtered_fallback_tools,
)

from .query_router import get_reasoning_type_from_route

logger = logging.getLogger(__name__)

# Temporarily import the actual class from parent during migration
# This will be replaced with the actual implementation
from ..reasoning_integration import ReasoningIntegration

__all__ = ["ReasoningIntegration"]
