"""Utility modules for VulcanAMI_LLM"""

from .cpu_capabilities import (
    CPUCapabilities,
    detect_cpu_capabilities,
    format_capability_warning,
    get_capability_summary,
    get_cpu_capabilities,
)
from .faiss_config import (
    FAISS_AVAILABLE,
    get_faiss,
    get_faiss_config_info,
    get_faiss_instruction_set,
    initialize_faiss,
    is_faiss_available,
)
from .performance_instrumentation import (
    GenerationPerformanceMetrics,
    GenerationPerformanceTracker,
    TimingContext,
    get_generation_performance_tracker,
    get_step_logger,
    get_token_logger,
    timed,
    timed_async,
)
from .performance_metrics import (
    PerformanceMetric,
    PerformanceTimer,
    PerformanceTracker,
    get_performance_tracker,
    log_performance_summary,
    track_analogical_reasoning,
    track_faiss_search,
    track_zk_proof_generation,
)
from .url_validator import (
    ALLOWED_SCHEMES,
    URLValidationError,
    safe_urlopen,
    validate_url_scheme,
)

__all__ = [
    # CPU Capabilities
    "CPUCapabilities",
    "detect_cpu_capabilities",
    "get_cpu_capabilities",
    "format_capability_warning",
    "get_capability_summary",
    # FAISS Configuration
    "FAISS_AVAILABLE",
    "initialize_faiss",
    "get_faiss",
    "is_faiss_available",
    "get_faiss_instruction_set",
    "get_faiss_config_info",
    # Performance Metrics
    "PerformanceMetric",
    "PerformanceTracker",
    "PerformanceTimer",
    "get_performance_tracker",
    "log_performance_summary",
    "track_zk_proof_generation",
    "track_analogical_reasoning",
    "track_faiss_search",
    # Performance Instrumentation
    "GenerationPerformanceTracker",
    "GenerationPerformanceMetrics",
    "TimingContext",
    "get_generation_performance_tracker",
    "get_step_logger",
    "get_token_logger",
    "timed",
    "timed_async",
    # URL Security
    "URLValidationError",
    "validate_url_scheme",
    "safe_urlopen",
    "ALLOWED_SCHEMES",
]
