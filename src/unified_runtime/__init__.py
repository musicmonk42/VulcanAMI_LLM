# Expose core classes and functions
from .unified_runtime_core import (
    UnifiedRuntime,
    RuntimeConfig,
    get_runtime,
    execute_graph,
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionContext,
    ExecutionMode,
    ExecutionStatus,
)
from .graph_validator import GraphValidator, ValidationResult, ResourceLimits
from .hardware_dispatcher_integration import (
    HardwareDispatcherIntegration,
    HardwareBackend,
    DispatchStrategy,
)
from .ai_runtime_integration import AIRuntime, AITask, AIContract
from .node_handlers import get_node_handlers
from .runtime_extensions import RuntimeExtensions, LearningMode, ExplanationType

# Handle potential import error for execution_metrics
# Use try-except to avoid crashing if execution_metrics is missing
try:
    from .execution_metrics import ExecutionMetrics, MetricsAggregator
except ImportError:
    ExecutionMetrics = None
    MetricsAggregator = None

from .vulcan_integration import (
    VulcanGraphixBridge,
    VulcanIntegrationConfig,
    enable_vulcan_integration,
)

# Define __all__ to control what gets imported with 'from . import *'
__all__ = [
    "UnifiedRuntime",
    "RuntimeConfig",
    "get_runtime",
    "execute_graph",
    "ExecutionEngine",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionStatus",
    "GraphValidator",
    "ValidationResult",
    "ResourceLimits",
    "HardwareDispatcherIntegration",
    "HardwareBackend",
    "DispatchStrategy",
    "AIRuntime",
    "AITask",
    "AIContract",
    "get_node_handlers",
    "RuntimeExtensions",
    "LearningMode",
    "ExplanationType",
    "ExecutionMetrics",
    "MetricsAggregator",  # These might be None if import failed
    "VulcanGraphixBridge",
    "VulcanIntegrationConfig",
    "enable_vulcan_integration",
]
