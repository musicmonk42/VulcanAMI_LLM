"""
observation_types.py - Core data types and null object implementations for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Contains:
- Observation: Single observation from the environment
- ModelContext: Context for predictions and updates
- ComponentIntegrationError: Exception for component integration failures
- NullObjectiveHierarchy: Null object for ObjectiveHierarchy
- NullMotivationalIntrospection: Null object for MotivationalIntrospection
- NullMetaReasoningComponent: Generic null object for meta-reasoning components
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ComponentIntegrationError(Exception):
    """Raised when critical component integration fails"""


@dataclass
class Observation:
    """Single observation from the environment"""

    timestamp: float
    variables: Dict[str, Any]
    intervention_data: Optional[Dict[str, Any]] = None
    domain: str = "unknown"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelContext:
    """Context for predictions and updates"""

    domain: str
    targets: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    features: Optional[np.ndarray] = None

    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility"""
        return getattr(self, key, default)

    def keys(self):
        """Dict-like keys method for backward compatibility"""
        return self.__dataclass_fields__.keys()

    def values(self):
        """Dict-like values method for backward compatibility"""
        return [getattr(self, k) for k in self.__dataclass_fields__.keys()]

    def items(self):
        """Dict-like items method for backward compatibility"""
        return [(k, getattr(self, k)) for k in self.__dataclass_fields__.keys()]


class NullObjectiveHierarchy:
    """
    Null object implementation for ObjectiveHierarchy.
    Used when meta-reasoning components fail to import.

    This provides explicit warnings instead of silently swallowing method calls.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "Using NullObjectiveHierarchy - meta-reasoning not available. "
            "Objectives will not be tracked. This is a fallback implementation."
        )
        self.objectives = {}

    def add_objective(self, *args, **kwargs):
        logger.debug("NullObjectiveHierarchy.add_objective called - no-op")
        return False

    def get_priority_order(self, *args, **kwargs):
        logger.debug("NullObjectiveHierarchy.get_priority_order called - returning empty list")
        return []

    def get_dependencies(self, *args, **kwargs):
        logger.debug("NullObjectiveHierarchy.get_dependencies called - returning empty set")
        return set()

    def check_consistency(self, *args, **kwargs):
        logger.debug("NullObjectiveHierarchy.check_consistency called - returning empty dict")
        return {"conflicts": [], "cycles": []}


class NullMotivationalIntrospection:
    """
    Null object implementation for MotivationalIntrospection.
    Used when meta-reasoning components fail to import.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "Using NullMotivationalIntrospection - meta-reasoning not available. "
            "Motivational drives will not be tracked. This is a fallback implementation."
        )

    def __getattr__(self, name):
        logger.debug(f"NullMotivationalIntrospection.{name} called - no-op")
        return lambda *args, **kwargs: None


class NullMetaReasoningComponent:
    """
    Generic null object for other meta-reasoning components.
    """

    def __init__(self, component_name="Unknown", *args, **kwargs):
        self.component_name = component_name
        logger.warning(
            f"Using Null{component_name} - meta-reasoning not available. "
            f"This is a fallback implementation."
        )

    def __getattr__(self, name):
        logger.debug(f"Null{self.component_name}.{name} called - no-op")
        return lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        return None
