"""
Global model registry for expensive ML models.

This module provides thread-safe model caching with LRU eviction, preventing:
- Memory leaks from unbounded model accumulation in long-running servers
- Race conditions from non-atomic cache access
- Lock contention from global locks blocking all model access

## Features
- **LRU Cache**: Automatically evicts least-recently-used models when cache is full
- **Per-Model Locks**: Only threads loading the same model wait for each other
- **Rate Limiting**: Prevents DoS attacks via rapid model loading
- **Health Monitoring**: Detects deadlocks and reports cache status
- **Custom Exceptions**: Distinguish permanent vs transient failures
- **Configurable**: All settings via environment variables

## Configuration (Environment Variables)
- `VULCAN_MAX_MODELS_CACHE`: Maximum models in cache (default: 5)
- `VULCAN_SENTENCE_TRANSFORMER_MODEL`: Default sentence transformer model
- `VULCAN_PRELOAD_MODELS`: Comma-separated list of models to preload
- `VULCAN_MAX_LOADS_PER_MINUTE`: Rate limit for model loading (default: 10)

## Usage Examples

### Basic Usage
```python
from vulcan.models import get_sentence_transformer

# Get a model (raises exception on failure)
try:
    model = get_sentence_transformer('all-MiniLM-L6-v2')
    embeddings = model.encode(['Hello world'])
except ModelNotAvailableError:
    # Permanent failure - dependency not installed
    print("sentence-transformers not installed")
except ModelLoadFailedError:
    # Transient failure - may retry
    print("Model load failed, retrying...")
```

### Health Monitoring
```python
from vulcan.models import health_check

status = health_check()
print(f"Status: {status['status']}")  # 'healthy', 'degraded', or 'unhealthy'
print(f"Models cached: {status['models_cached']}")
```

### Get Model Info
```python
from vulcan.models import get_model_info

info = get_model_info('sentence_transformer:all-MiniLM-L6-v2')
print(f"Access count: {info['access_count']}")
print(f"Last accessed: {info['last_access_time']}")
```

## Thread Safety
All functions are thread-safe. Multiple threads can:
- Load different models concurrently (per-model locks)
- Access cached models without blocking (atomic get operations)
- Call health_check() and get_model_info() safely

## Memory Management
- Cache uses LRU eviction when full (configurable max size)
- Models with cleanup() method are cleaned up on eviction
- Manual cleanup via clear_cache() when needed
"""

import logging
import os
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ModelLoadError(Exception):
    """Base class for model loading errors."""
    pass


class ModelNotAvailableError(ModelLoadError):
    """Model dependencies not installed (permanent failure)."""
    pass


class ModelLoadFailedError(ModelLoadError):
    """Model loading failed (may be transient)."""
    pass


class RateLimitError(ModelLoadError):
    """Rate limit exceeded for model loading."""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable with fallback."""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable with fallback."""
    return os.getenv(key, default)


def _get_env_list(key: str, default: List[str]) -> List[str]:
    """Get comma-separated list from environment variable."""
    value = os.getenv(key, '')
    if not value:
        return default
    return [item.strip() for item in value.split(',') if item.strip()]


# Configuration with environment variable support
MAX_MODELS_CACHE = _get_env_int('VULCAN_MAX_MODELS_CACHE', 5)
DEFAULT_SENTENCE_TRANSFORMER_MODEL = _get_env_str(
    'VULCAN_SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2'
)
PRELOAD_MODELS = _get_env_list('VULCAN_PRELOAD_MODELS', [])
MAX_LOADS_PER_MINUTE = _get_env_int('VULCAN_MAX_LOADS_PER_MINUTE', 10)


# ============================================================================
# LRU MODEL CACHE
# ============================================================================

class ModelCache:
    """
    Thread-safe LRU cache for ML models.
    
    Features:
    - Configurable maximum size
    - LRU eviction policy
    - Cleanup support for evicted models
    - Access tracking for observability
    """
    
    def __init__(self, max_size: int = MAX_MODELS_CACHE):
        """Initialize the cache with given max size."""
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_counts: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get a model from cache (thread-safe, atomic operation).
        Updates LRU order and access tracking.
        
        Returns:
            Model if found, None otherwise.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                self._access_times[key] = time.time()
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
    
    def put(self, key: str, model: Any) -> None:
        """
        Put a model in cache (thread-safe).
        Evicts LRU model if cache is full.
        """
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = model
            else:
                # Evict LRU if at capacity
                if len(self._cache) >= self.max_size:
                    self._evict_lru()
                
                # Add new model
                self._cache[key] = model
            
            # Update tracking
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used model (must be called with lock held)."""
        if not self._cache:
            return
        
        # Get LRU key (first item in OrderedDict)
        lru_key = next(iter(self._cache))
        lru_model = self._cache.pop(lru_key)
        
        # Cleanup if supported
        if hasattr(lru_model, 'cleanup') and callable(lru_model.cleanup):
            try:
                lru_model.cleanup()
            except Exception as e:
                logger.warning(f"[ModelCache] Cleanup failed for {lru_key}: {e}")
        
        # Remove tracking data
        self._access_counts.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
        
        self._evictions += 1
        logger.info(f"[ModelCache] Evicted LRU model: {lru_key}")
    
    def clear(self) -> None:
        """Clear all models from cache (thread-safe)."""
        with self._lock:
            # Cleanup all models that support it
            for key, model in self._cache.items():
                if hasattr(model, 'cleanup') and callable(model.cleanup):
                    try:
                        model.cleanup()
                    except Exception as e:
                        logger.warning(f"[ModelCache] Cleanup failed for {key}: {e}")
            
            self._cache.clear()
            self._access_counts.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics (thread-safe)."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'keys': list(self._cache.keys()),
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': self._hits / max(self._hits + self._misses, 1),
            }
    
    def get_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific cached model."""
        with self._lock:
            if key not in self._cache:
                return None
            
            return {
                'key': key,
                'access_count': self._access_counts.get(key, 0),
                'last_access_time': self._access_times.get(key, 0),
                'cached': True,
            }


# ============================================================================
# MODULE-LEVEL STATE
# ============================================================================

# Global cache instance
_model_cache = ModelCache(max_size=MAX_MODELS_CACHE)

# Per-model locks (only threads loading the same model wait)
_model_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

# Rate limiting state
_load_timestamps: List[float] = []
_rate_limit_lock = threading.Lock()

# Metrics
_load_attempts = {'success': 0, 'failure': 0}
_load_durations: List[float] = []
_metrics_lock = threading.Lock()



# ============================================================================
# RATE LIMITING
# ============================================================================

def _check_rate_limit() -> None:
    """
    Check if rate limit is exceeded.
    
    Raises:
        RateLimitError: If too many model loads in the last minute.
    """
    with _rate_limit_lock:
        now = time.time()
        cutoff = now - 60.0  # 1 minute window
        
        # Remove timestamps older than 1 minute
        _load_timestamps[:] = [ts for ts in _load_timestamps if ts > cutoff]
        
        # Check rate limit
        if len(_load_timestamps) >= MAX_LOADS_PER_MINUTE:
            raise RateLimitError(
                f"Rate limit exceeded: {len(_load_timestamps)} loads in last minute "
                f"(max: {MAX_LOADS_PER_MINUTE})"
            )
        
        # Record this load attempt
        _load_timestamps.append(now)


def _record_load_attempt(success: bool, duration: float) -> None:
    """Record metrics for a model load attempt."""
    with _metrics_lock:
        if success:
            _load_attempts['success'] += 1
        else:
            _load_attempts['failure'] += 1
        
        _load_durations.append(duration)
        
        # Keep only last 100 durations for memory efficiency
        if len(_load_durations) > 100:
            _load_durations.pop(0)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def get_sentence_transformer(model_name: Optional[str] = None) -> Any:
    """
    Get or create a SentenceTransformer model (thread-safe with LRU cache).
    
    This function uses per-model locks, so only threads loading the SAME model
    wait for each other. Different models can load concurrently.
    
    Args:
        model_name: Name of the SentenceTransformer model to load.
                   Defaults to VULCAN_SENTENCE_TRANSFORMER_MODEL env var
                   or 'all-MiniLM-L6-v2'.
    
    Returns:
        SentenceTransformer model instance.
    
    Raises:
        ModelNotAvailableError: If sentence-transformers package not installed.
        ModelLoadFailedError: If model loading fails (may be transient).
        RateLimitError: If rate limit exceeded.
    
    Example:
        >>> model = get_sentence_transformer('all-MiniLM-L6-v2')
        >>> embeddings = model.encode(['Hello world'])
    """
    if model_name is None:
        model_name = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    
    cache_key = f"sentence_transformer:{model_name}"
    
    # Fast path: check cache without lock (atomic operation)
    cached_model = _model_cache.get(cache_key)
    if cached_model is not None:
        logger.debug(f"[ModelRegistry] Cache HIT for {model_name}")
        return cached_model
    
    # Cache miss - need to load model
    logger.info(f"[ModelRegistry] Cache MISS for {model_name}, loading...")
    
    # Check rate limit before expensive operation
    _check_rate_limit()
    
    # Use per-model lock (only threads loading THIS model wait)
    with _model_locks[cache_key]:
        # Double-check after acquiring lock (another thread may have loaded it)
        cached_model = _model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        # Load the model
        start_time = time.time()
        logger.info(f"[ModelRegistry] Loading SentenceTransformer: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            _model_cache.put(cache_key, model)
            
            duration = time.time() - start_time
            _record_load_attempt(success=True, duration=duration)
            
            logger.info(
                f"[ModelRegistry] ✓ {model_name} loaded and cached "
                f"(took {duration:.2f}s)"
            )
            return model
            
        except ImportError as e:
            duration = time.time() - start_time
            _record_load_attempt(success=False, duration=duration)
            
            logger.error(
                f"[ModelRegistry] sentence-transformers not available: {e}"
            )
            raise ModelNotAvailableError(
                f"sentence-transformers package not installed: {e}"
            ) from e
            
        except Exception as e:
            duration = time.time() - start_time
            _record_load_attempt(success=False, duration=duration)
            
            logger.error(f"[ModelRegistry] Failed to load {model_name}: {e}")
            raise ModelLoadFailedError(
                f"Failed to load SentenceTransformer model '{model_name}': {e}"
            ) from e




def get_bert_model() -> Any:
    """
    Get or create the BERT/GraphixTransformer model (thread-safe with LRU cache).
    
    Returns:
        GraphixTransformer model instance.
    
    Raises:
        ModelNotAvailableError: If GraphixTransformer not available.
        ModelLoadFailedError: If model loading fails (may be transient).
        RateLimitError: If rate limit exceeded.
    
    Example:
        >>> model = get_bert_model()
        >>> result = model.process(text)
    """
    cache_key = "bert:graphix_transformer"
    
    # Fast path: atomic cache check
    cached_model = _model_cache.get(cache_key)
    if cached_model is not None:
        logger.debug("[ModelRegistry] Cache HIT for BERT model")
        return cached_model
    
    # Cache miss - need to load
    logger.info("[ModelRegistry] Cache MISS for BERT model, loading...")
    
    # Check rate limit
    _check_rate_limit()
    
    # Use per-model lock
    with _model_locks[cache_key]:
        # Double-check after lock
        cached_model = _model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        start_time = time.time()
        logger.info("[ModelRegistry] Loading BERT model")
        
        try:
            from vulcan.processing import GraphixTransformer
            model = GraphixTransformer.get_instance()
            _model_cache.put(cache_key, model)
            
            duration = time.time() - start_time
            _record_load_attempt(success=True, duration=duration)
            
            logger.info(
                f"[ModelRegistry] ✓ BERT model loaded and cached "
                f"(took {duration:.2f}s)"
            )
            return model
            
        except ImportError as e:
            duration = time.time() - start_time
            _record_load_attempt(success=False, duration=duration)
            
            logger.error(f"[ModelRegistry] GraphixTransformer not available: {e}")
            raise ModelNotAvailableError(
                f"GraphixTransformer not available: {e}"
            ) from e
            
        except Exception as e:
            duration = time.time() - start_time
            _record_load_attempt(success=False, duration=duration)
            
            logger.error(f"[ModelRegistry] BERT model load failed: {e}")
            raise ModelLoadFailedError(
                f"Failed to load BERT model: {e}"
            ) from e


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preload_all_models() -> Dict[str, bool]:
    """
    Pre-load all expensive models at startup.
    Call this during platform initialization to avoid cold-start latency.
    
    Respects VULCAN_PRELOAD_MODELS environment variable for custom list.
    
    Returns:
        Dictionary mapping model names to load success status.
    
    Example:
        >>> results = preload_all_models()
        >>> print(f"Loaded {sum(results.values())} models")
    """
    logger.info("[ModelRegistry] Pre-loading models...")
    results = {}
    
    # Build list of models to preload
    models_to_preload = PRELOAD_MODELS if PRELOAD_MODELS else [
        DEFAULT_SENTENCE_TRANSFORMER_MODEL
    ]
    
    # Load SentenceTransformers
    for model_name in models_to_preload:
        if model_name.startswith('bert:'):
            # Skip bert models in sentence transformer section
            continue
        
        try:
            get_sentence_transformer(model_name)
            results[f'sentence_transformer:{model_name}'] = True
        except ModelNotAvailableError:
            logger.warning(
                f"[ModelRegistry] Cannot preload {model_name}: "
                "sentence-transformers not available"
            )
            results[f'sentence_transformer:{model_name}'] = False
        except ModelLoadFailedError as e:
            logger.warning(f"[ModelRegistry] Failed to preload {model_name}: {e}")
            results[f'sentence_transformer:{model_name}'] = False
        except Exception as e:
            logger.error(f"[ModelRegistry] Unexpected error preloading {model_name}: {e}")
            results[f'sentence_transformer:{model_name}'] = False
    
    # Load BERT if requested
    if 'bert:graphix_transformer' in models_to_preload or not PRELOAD_MODELS:
        try:
            get_bert_model()
            results['bert:graphix_transformer'] = True
        except ModelNotAvailableError:
            logger.info("[ModelRegistry] BERT model not available (optional)")
            results['bert:graphix_transformer'] = False
        except ModelLoadFailedError as e:
            logger.warning(f"[ModelRegistry] BERT pre-load failed: {e}")
            results['bert:graphix_transformer'] = False
        except Exception as e:
            logger.error(f"[ModelRegistry] Unexpected BERT error: {e}")
            results['bert:graphix_transformer'] = False
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(
        f"[ModelRegistry] ✓ Pre-loaded {success_count}/{len(results)} models"
    )
    
    return results


def get_cache_stats() -> Dict[str, Any]:
    """
    Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache metrics including:
        - size: Current number of cached models
        - max_size: Maximum cache capacity
        - keys: List of cached model keys
        - hits: Total cache hits
        - misses: Total cache misses
        - evictions: Total evictions
        - hit_rate: Cache hit rate (0.0 to 1.0)
        - load_attempts: Success/failure counts
        - avg_load_duration: Average model load time
    
    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """
    cache_stats = _model_cache.stats()
    
    with _metrics_lock:
        avg_duration = (
            sum(_load_durations) / len(_load_durations)
            if _load_durations else 0.0
        )
        
        cache_stats.update({
            'load_attempts': dict(_load_attempts),
            'avg_load_duration': avg_duration,
            'recent_load_durations': list(_load_durations[-10:]),  # Last 10
        })
    
    return cache_stats


def get_model_info(model_key: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific cached model.
    
    Args:
        model_key: Cache key for the model (e.g., 'sentence_transformer:all-MiniLM-L6-v2')
    
    Returns:
        Dictionary with model info, or None if not cached.
        Includes:
        - key: The model cache key
        - access_count: Number of times accessed
        - last_access_time: Timestamp of last access
        - cached: Always True if returned
    
    Example:
        >>> info = get_model_info('sentence_transformer:all-MiniLM-L6-v2')
        >>> if info:
        ...     print(f"Accessed {info['access_count']} times")
    """
    return _model_cache.get_info(model_key)


def clear_cache() -> None:
    """
    Clear the model cache.
    
    This will:
    - Call cleanup() on models that support it
    - Remove all cached models
    - Clear access tracking data
    
    Use with caution - models will need to be reloaded on next access.
    
    Example:
        >>> clear_cache()  # Forces all models to reload
    """
    _model_cache.clear()
    logger.info("[ModelRegistry] Cache cleared")


def health_check() -> Dict[str, Any]:
    """
    Perform health check on the model registry.
    
    Checks for:
    - Lock acquisition (detects deadlocks)
    - Cache status
    - Recent errors
    
    Returns:
        Dictionary with health status:
        - status: 'healthy', 'degraded', or 'unhealthy'
        - models_cached: Number of models in cache
        - cache_keys: List of cached model keys
        - hit_rate: Cache hit rate
        - load_success_rate: Model load success rate
        - errors: List of any errors detected
        - timestamp: When check was performed
    
    Example:
        >>> status = health_check()
        >>> if status['status'] != 'healthy':
        ...     print(f"Issues: {status['errors']}")
    """
    errors = []
    status = 'healthy'
    
    # Test lock acquisition (detect deadlocks)
    try:
        lock_acquired = _model_cache._lock.acquire(timeout=5.0)
        if not lock_acquired:
            errors.append("Failed to acquire cache lock (possible deadlock)")
            status = 'unhealthy'
        else:
            _model_cache._lock.release()
    except Exception as e:
        errors.append(f"Lock acquisition error: {e}")
        status = 'unhealthy'
    
    # Get cache statistics
    cache_stats = get_cache_stats()
    
    # Check load success rate
    with _metrics_lock:
        total_loads = _load_attempts['success'] + _load_attempts['failure']
        success_rate = (
            _load_attempts['success'] / total_loads if total_loads > 0 else 1.0
        )
    
    # Determine overall status
    if success_rate < 0.5 and total_loads > 0:
        errors.append(f"Low model load success rate: {success_rate:.1%}")
        status = 'degraded' if status == 'healthy' else status
    
    if cache_stats['size'] == cache_stats['max_size']:
        # Cache is full - not an error but worth noting
        errors.append("Cache is at maximum capacity (LRU eviction active)")
        if status == 'healthy':
            status = 'degraded'
    
    return {
        'status': status,
        'models_cached': cache_stats['size'],
        'max_cache_size': cache_stats['max_size'],
        'cache_keys': cache_stats['keys'],
        'hit_rate': cache_stats['hit_rate'],
        'load_success_rate': success_rate,
        'total_loads': total_loads,
        'errors': errors,
        'timestamp': time.time(),
    }

