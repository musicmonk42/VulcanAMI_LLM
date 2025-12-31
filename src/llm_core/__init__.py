# ============================================================
# VULCAN LLM Core Package
# Core transformer and execution components
# ============================================================
#
# MODULES:
#     graphix_transformer  - Graphix-based transformer implementation
#     graphix_executor     - Execution engine for transformer graphs
#     persistant_context   - Persistent context management
#     ir_attention         - Intermediate representation for attention
#     ir_embeddings        - Intermediate representation for embeddings
#     ir_feedforward       - Intermediate representation for feedforward
#     ir_layer_norm        - Intermediate representation for layer normalization
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Core transformer
try:
    from src.llm_core.graphix_transformer import (
        GraphixTransformer,
        GraphixTransformerConfig,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphixTransformer not available: {e}")
    TRANSFORMER_AVAILABLE = False
    GraphixTransformer = None
    GraphixTransformerConfig = None

# Executor
try:
    from src.llm_core.graphix_executor import GraphixExecutor
    EXECUTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphixExecutor not available: {e}")
    EXECUTOR_AVAILABLE = False
    GraphixExecutor = None

# Persistent context
try:
    from src.llm_core.persistant_context import PersistentContext
    PERSISTENT_CONTEXT_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PersistentContext not available: {e}")
    PERSISTENT_CONTEXT_AVAILABLE = False
    PersistentContext = None

__all__ = [
    "__version__",
    "__author__",
    # Transformer
    "GraphixTransformer",
    "GraphixTransformerConfig",
    "TRANSFORMER_AVAILABLE",
    # Executor
    "GraphixExecutor",
    "EXECUTOR_AVAILABLE",
    # Context
    "PersistentContext",
    "PERSISTENT_CONTEXT_AVAILABLE",
]

logger.debug(f"VULCAN LLM Core package v{__version__} loaded")
