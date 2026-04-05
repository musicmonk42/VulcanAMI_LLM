"""
Selection Types - Data classes and enums for the tool selection system.

Extracted from tool_selector.py to reduce module size and provide
clean, importable type definitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np

# Conditional imports for types from sibling modules.
# These may not be available in all environments, so we provide fallbacks.
try:
    from .admission_control import RequestPriority
except ImportError:
    RequestPriority = None

try:
    from .safety_governor import SafetyLevel
except ImportError:
    SafetyLevel = None

try:
    from .portfolio_executor import ExecutionStrategy
except ImportError:
    ExecutionStrategy = None


class SelectionMode(Enum):
    """Tool selection modes"""

    FAST = "fast"  # Optimize for speed
    ACCURATE = "accurate"  # Optimize for accuracy
    EFFICIENT = "efficient"  # Optimize for energy
    BALANCED = "balanced"  # Balance all factors
    SAFE = "safe"  # Maximum safety checks


@dataclass
class SelectionRequest:
    """Request for tool selection"""

    problem: Any
    features: Optional[np.ndarray] = None
    constraints: Dict[str, float] = field(default_factory=dict)
    mode: SelectionMode = SelectionMode.BALANCED
    priority: "RequestPriority" = None  # type: ignore[assignment]
    safety_level: "SafetyLevel" = None  # type: ignore[assignment]
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

    def __post_init__(self):
        # Apply defaults if the types are available
        if self.priority is None and RequestPriority is not None:
            self.priority = RequestPriority.NORMAL
        if self.safety_level is None and SafetyLevel is not None:
            self.safety_level = SafetyLevel.MEDIUM


@dataclass
class SelectionResult:
    """Result of tool selection and execution"""

    selected_tool: str
    execution_result: Any
    confidence: float
    calibrated_confidence: float
    execution_time_ms: float
    energy_used_mj: float
    strategy_used: "ExecutionStrategy"  # type: ignore[type-arg]
    all_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
