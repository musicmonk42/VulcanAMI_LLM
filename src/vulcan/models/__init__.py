"""
Global model registry for expensive ML models.
Ensures models are loaded ONCE per process and shared across all components.

This module provides thread-safe model caching with LRU eviction, rate limiting,
and comprehensive error handling. See model_registry.py for full documentation.
"""

from .model_registry import (
    # Core functions
    get_sentence_transformer,
    get_bert_model,
    preload_all_models,
    
    # Monitoring and observability
    get_cache_stats,
    get_model_info,
    health_check,
    
    # Cache management
    clear_cache,
    
    # Exception classes
    ModelLoadError,
    ModelNotAvailableError,
    ModelLoadFailedError,
    RateLimitError,
)

__all__ = [
    # Core functions
    "get_sentence_transformer",
    "get_bert_model",
    "preload_all_models",
    
    # Monitoring and observability
    "get_cache_stats",
    "get_model_info",
    "health_check",
    
    # Cache management
    "clear_cache",
    
    # Exception classes
    "ModelLoadError",
    "ModelNotAvailableError",
    "ModelLoadFailedError",
    "RateLimitError",
]
