"""
Global model registry for expensive ML models.
Ensures models are loaded ONCE per process and shared across all components.
"""

from .model_registry import (
    get_sentence_transformer,
    get_bert_model,
    preload_all_models,
    get_cache_stats,
)

__all__ = [
    "get_sentence_transformer",
    "get_bert_model",
    "preload_all_models",
    "get_cache_stats",
]
