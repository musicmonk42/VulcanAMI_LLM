# ============================================================
# VULCAN-AGI Distillation Package
# Knowledge distillation from OpenAI to local LLM
# ============================================================
#
# This package provides comprehensive knowledge distillation capabilities:
#     - PII and secrets redaction for privacy
#     - Governance sensitivity checking
#     - Example quality validation
#     - JSONL storage with optional encryption
#     - Promotion gate for trained weights
#     - Shadow model evaluation
#
# ARCHITECTURE:
#     The capture → train flow is:
#     
#         main.py (inference)
#             └─ OpenAI response
#             └─ capture_response() → Distillation Store (JSONL)
#                                         ↓
#         [Async/Batched - via GovernedTrainer]
#             └─ GovernedTrainer reads from Distillation Store
#             └─ Proposes weight updates
#             └─ ConsensusEngine approves/rejects
#             └─ SelfImprovingTraining evaluates
#             └─ Promotion or rollback
#
# USAGE:
#     from vulcan.distillation import (
#         get_knowledge_distiller,
#         initialize_knowledge_distiller,
#         OpenAIKnowledgeDistiller,
#     )
#     
#     # Initialize the distiller
#     distiller = initialize_knowledge_distiller(local_llm=my_llm)
#     
#     # Capture responses for training
#     distiller.capture_response(
#         prompt="What is AI?",
#         openai_response="AI is...",
#         session_opted_in=True,
#     )
#
# VERSION HISTORY:
#     1.0.0 - Initial extraction from main.py
# ============================================================

import logging
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CORE IMPORTS WITH AVAILABILITY TRACKING
# ============================================================

_imports_successful = True

try:
    from vulcan.distillation.models import DistillationExample
except ImportError as e:
    logger.warning(f"models module not available: {e}")
    _imports_successful = False
    DistillationExample = None

try:
    from vulcan.distillation.pii_redactor import PIIRedactor
except ImportError as e:
    logger.warning(f"pii_redactor module not available: {e}")
    _imports_successful = False
    PIIRedactor = None

try:
    from vulcan.distillation.governance_checker import GovernanceSensitivityChecker
except ImportError as e:
    logger.warning(f"governance_checker module not available: {e}")
    _imports_successful = False
    GovernanceSensitivityChecker = None

try:
    from vulcan.distillation.quality_validator import ExampleQualityValidator
except ImportError as e:
    logger.warning(f"quality_validator module not available: {e}")
    _imports_successful = False
    ExampleQualityValidator = None

try:
    from vulcan.distillation.storage import DistillationStorageBackend
except ImportError as e:
    logger.warning(f"storage module not available: {e}")
    _imports_successful = False
    DistillationStorageBackend = None

try:
    from vulcan.distillation.promotion_gate import PromotionGate
except ImportError as e:
    logger.warning(f"promotion_gate module not available: {e}")
    _imports_successful = False
    PromotionGate = None

try:
    from vulcan.distillation.evaluator import ShadowModelEvaluator
except ImportError as e:
    logger.warning(f"evaluator module not available: {e}")
    _imports_successful = False
    ShadowModelEvaluator = None

try:
    from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
except ImportError as e:
    logger.warning(f"distiller module not available: {e}")
    _imports_successful = False
    OpenAIKnowledgeDistiller = None


# ============================================================
# GLOBAL DISTILLER MANAGEMENT
# ============================================================

# Global knowledge distiller instance (initialized later with local LLM)
_knowledge_distiller: Optional["OpenAIKnowledgeDistiller"] = None
_distiller_lock = threading.Lock()


def get_knowledge_distiller() -> Optional["OpenAIKnowledgeDistiller"]:
    """
    Get the global knowledge distiller instance.
    
    Returns:
        The global OpenAIKnowledgeDistiller instance, or None if not initialized
    """
    return _knowledge_distiller


def initialize_knowledge_distiller(
    local_llm: Optional[Any] = None,
    **kwargs,
) -> "OpenAIKnowledgeDistiller":
    """
    Initialize the global knowledge distiller with thread-safe singleton pattern.
    
    Uses double-checked locking for thread safety without unnecessary lock contention.
    
    Args:
        local_llm: Reference to Vulcan's local LLM (optional)
        **kwargs: Additional arguments passed to OpenAIKnowledgeDistiller
        
    Returns:
        The initialized OpenAIKnowledgeDistiller instance
    """
    global _knowledge_distiller
    
    # First check without lock (fast path)
    if _knowledge_distiller is not None:
        logger.warning("Knowledge distiller already initialized, returning existing instance")
        return _knowledge_distiller
    
    # Acquire lock for initialization
    with _distiller_lock:
        # Double-check after acquiring lock (prevents race condition)
        if _knowledge_distiller is not None:
            logger.warning("Knowledge distiller already initialized, returning existing instance")
            return _knowledge_distiller
        
        if OpenAIKnowledgeDistiller is None:
            raise ImportError("OpenAIKnowledgeDistiller is not available")
        
        _knowledge_distiller = OpenAIKnowledgeDistiller(local_llm=local_llm, **kwargs)
        logger.info("Knowledge distiller initialized")
        return _knowledge_distiller


def reset_knowledge_distiller() -> None:
    """Reset the global knowledge distiller (thread-safe)."""
    global _knowledge_distiller
    
    with _distiller_lock:
        _knowledge_distiller = None
        logger.debug("Knowledge distiller reset")


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Global management
    "get_knowledge_distiller",
    "initialize_knowledge_distiller",
    "reset_knowledge_distiller",
    # Models
    "DistillationExample",
    # Components
    "PIIRedactor",
    "GovernanceSensitivityChecker",
    "ExampleQualityValidator",
    "DistillationStorageBackend",
    "PromotionGate",
    "ShadowModelEvaluator",
    "OpenAIKnowledgeDistiller",
    # Module utilities
    "get_module_info",
    "validate_distillation_module",
]


# ============================================================
# MODULE UTILITIES
# ============================================================


def get_module_info() -> Dict[str, Any]:
    """
    Get information about the distillation module.
    
    Returns:
        Dictionary with module information and availability status
    """
    return {
        "version": __version__,
        "author": __author__,
        "imports_successful": _imports_successful,
        "python_version": sys.version,
        "components": {
            "models": DistillationExample is not None,
            "pii_redactor": PIIRedactor is not None,
            "governance_checker": GovernanceSensitivityChecker is not None,
            "quality_validator": ExampleQualityValidator is not None,
            "storage": DistillationStorageBackend is not None,
            "promotion_gate": PromotionGate is not None,
            "evaluator": ShadowModelEvaluator is not None,
            "distiller": OpenAIKnowledgeDistiller is not None,
        },
        "distiller_initialized": _knowledge_distiller is not None,
    }


def validate_distillation_module() -> bool:
    """
    Validate that the distillation module is properly loaded.
    
    Returns:
        True if module is functional, False otherwise
    """
    try:
        info = get_module_info()
        
        # All core components must be available
        all_available = all(info["components"].values())
        
        if all_available:
            logger.info("Distillation module validated successfully")
        else:
            missing = [k for k, v in info["components"].items() if not v]
            logger.warning(f"Some distillation components unavailable: {missing}")
        
        return all_available
        
    except Exception as e:
        logger.error(f"Distillation module validation failed: {e}")
        return False


# ============================================================
# MODULE INITIALIZATION
# ============================================================

if _imports_successful:
    logger.info(f"VULCAN-AGI Distillation module v{__version__} loaded successfully")
else:
    logger.warning(
        f"VULCAN-AGI Distillation module v{__version__} loaded with some components unavailable"
    )
