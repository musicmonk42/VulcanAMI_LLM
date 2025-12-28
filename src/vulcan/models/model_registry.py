"""
Global model registry for expensive ML models.
Ensures models are loaded ONCE per process and shared across all components.

This module provides thread-safe singleton pattern for expensive ML models
like SentenceTransformer, preventing multiple loads that cause:
- 7+ model loads in 10 minutes (2-4 seconds each)
- Query routing times of 6,000-58,000ms instead of <2,000ms
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level storage - persists across all instances in process
_model_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()


def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> Optional[Any]:
    """
    Get or create a SentenceTransformer model.
    Thread-safe singleton pattern at module level.
    
    Args:
        model_name: Name of the SentenceTransformer model to load.
        
    Returns:
        SentenceTransformer model instance, or None if unavailable.
    """
    cache_key = f"sentence_transformer:{model_name}"
    
    if cache_key in _model_cache:
        logger.debug(f"[ModelRegistry] Cache HIT for {model_name}")
        return _model_cache[cache_key]
    
    with _model_lock:
        # Double-check after acquiring lock
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        logger.info(f"[ModelRegistry] Loading SentenceTransformer: {model_name} (ONCE)")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            _model_cache[cache_key] = model
            logger.info(f"[ModelRegistry] ✓ {model_name} loaded and cached")
            return model
        except ImportError as e:
            logger.warning(f"[ModelRegistry] sentence-transformers not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load {model_name}: {e}")
            return None


def get_bert_model() -> Optional[Any]:
    """
    Get or create the BERT/GraphixTransformer model.
    
    Returns:
        GraphixTransformer model instance, or None if unavailable.
    """
    cache_key = "bert:graphix_transformer"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        logger.info("[ModelRegistry] Loading BERT model (ONCE)")
        try:
            from vulcan.processing import GraphixTransformer
            model = GraphixTransformer.get_instance()
            _model_cache[cache_key] = model
            logger.info("[ModelRegistry] ✓ BERT model loaded and cached")
            return model
        except ImportError as e:
            logger.warning(f"[ModelRegistry] GraphixTransformer not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"[ModelRegistry] BERT model load failed: {e}")
            return None


def preload_all_models() -> Dict[str, bool]:
    """
    Pre-load all expensive models at startup.
    Call this ONCE during platform initialization.
    
    Returns:
        Dictionary of model names to load success status.
    """
    logger.info("[ModelRegistry] Pre-loading all models...")
    results = {}
    
    # Load SentenceTransformer
    st_model = get_sentence_transformer('all-MiniLM-L6-v2')
    results['sentence_transformer'] = st_model is not None
    
    # Load BERT (optional, may not be available)
    try:
        bert_model = get_bert_model()
        results['bert'] = bert_model is not None
    except Exception as e:
        logger.warning(f"[ModelRegistry] BERT pre-load failed: {e}")
        results['bert'] = False
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"[ModelRegistry] ✓ Pre-loaded {success_count}/{len(results)} models")
    
    return results


def get_cache_stats() -> Dict[str, Any]:
    """Return current cache statistics."""
    return {
        'models_cached': len(_model_cache),
        'model_keys': list(_model_cache.keys())
    }


def clear_cache() -> None:
    """
    Clear the model cache. 
    Use with caution - this will force models to be reloaded on next access.
    """
    global _model_cache
    with _model_lock:
        _model_cache.clear()
        logger.info("[ModelRegistry] Cache cleared")
