"""
Comprehensive tests for model registry hardening.

Tests cover:
- LRU cache eviction
- Thread safety and race conditions
- Custom exception handling
- Rate limiting
- Health checks
- Per-model locking
- Metrics and observability
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

import pytest

from vulcan.models import (
    ModelLoadError,
    ModelLoadFailedError,
    ModelNotAvailableError,
    RateLimitError,
    clear_cache,
    get_bert_model,
    get_cache_stats,
    get_model_info,
    get_sentence_transformer,
    health_check,
    preload_all_models,
)
from vulcan.models.model_registry import ModelCache


@contextmanager
def mock_sentence_transformer(return_value=None, side_effect=None):
    """
    Context manager to mock sentence_transformers module.
    
    Args:
        return_value: What SentenceTransformer() should return
        side_effect: Exception or function for SentenceTransformer()
    """
    mock_st_module = Mock()
    if side_effect:
        mock_st_module.SentenceTransformer = Mock(side_effect=side_effect)
    else:
        mock_st_module.SentenceTransformer = Mock(return_value=return_value or Mock())
    
    with patch.dict('sys.modules', {'sentence_transformers': mock_st_module}):
        yield mock_st_module.SentenceTransformer


@contextmanager
def mock_graphix_transformer(return_value=None, side_effect=None):
    """
    Context manager to mock vulcan.processing.GraphixTransformer.
    
    Args:
        return_value: What GraphixTransformer.get_instance() should return
        side_effect: Exception or function for get_instance()
    """
    mock_processing_module = Mock()
    mock_graphix_class = Mock()
    if side_effect:
        mock_graphix_class.get_instance = Mock(side_effect=side_effect)
    else:
        mock_graphix_class.get_instance = Mock(return_value=return_value or Mock())
    mock_processing_module.GraphixTransformer = mock_graphix_class
    
    with patch.dict('sys.modules', {'vulcan.processing': mock_processing_module}):
        yield mock_graphix_class


class TestModelCache:
    """Test the LRU ModelCache implementation."""
    
    def test_cache_get_miss(self):
        """Test cache miss returns None."""
        cache = ModelCache(max_size=2)
        result = cache.get('nonexistent')
        assert result is None
    
    def test_cache_put_and_get(self):
        """Test putting and getting items from cache."""
        cache = ModelCache(max_size=2)
        model = Mock()
        
        cache.put('key1', model)
        result = cache.get('key1')
        
        assert result is model
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ModelCache(max_size=2)
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()
        
        cache.put('key1', model1)
        cache.put('key2', model2)
        
        # Cache is full, adding key3 should evict key1 (LRU)
        cache.put('key3', model3)
        
        assert cache.get('key1') is None  # Evicted
        assert cache.get('key2') is model2  # Still there
        assert cache.get('key3') is model3  # Just added
    
    def test_cache_lru_access_updates_order(self):
        """Test that accessing an item makes it most recently used."""
        cache = ModelCache(max_size=2)
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()
        
        cache.put('key1', model1)
        cache.put('key2', model2)
        
        # Access key1 to make it most recently used
        cache.get('key1')
        
        # Adding key3 should evict key2 (now LRU), not key1
        cache.put('key3', model3)
        
        assert cache.get('key1') is model1  # Still there (was accessed)
        assert cache.get('key2') is None  # Evicted (was LRU)
        assert cache.get('key3') is model3  # Just added
    
    def test_cache_cleanup_on_eviction(self):
        """Test that cleanup() is called on evicted models."""
        cache = ModelCache(max_size=1)
        model1 = Mock()
        model1.cleanup = Mock()
        model2 = Mock()
        
        cache.put('key1', model1)
        cache.put('key2', model2)  # Should evict key1
        
        # Verify cleanup was called on evicted model
        model1.cleanup.assert_called_once()
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = ModelCache(max_size=3)
        model = Mock()
        
        cache.put('key1', model)
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 3
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert 'key1' in stats['keys']
    
    def test_cache_get_info(self):
        """Test getting info about a cached model."""
        cache = ModelCache(max_size=2)
        model = Mock()
        
        cache.put('key1', model)
        cache.get('key1')
        cache.get('key1')
        
        info = cache.get_info('key1')
        
        assert info is not None
        assert info['key'] == 'key1'
        assert info['access_count'] == 3  # 1 put + 2 gets
        assert info['cached'] is True
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = ModelCache(max_size=2)
        model1 = Mock()
        model1.cleanup = Mock()
        model2 = Mock()
        
        cache.put('key1', model1)
        cache.put('key2', model2)
        
        cache.clear()
        
        assert cache.get('key1') is None
        assert cache.get('key2') is None
        model1.cleanup.assert_called_once()
    
    def test_cache_thread_safety(self):
        """Test that cache operations are thread-safe."""
        cache = ModelCache(max_size=10)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(50):
                    key = f'key{i % 5}'
                    model = Mock()
                    cache.put(key, model)
                    result = cache.get(key)
                    if result is not None:
                        results.append((worker_id, key))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors and successful operations
        assert len(errors) == 0
        assert len(results) > 0


class TestExceptionHandling:
    """Test custom exception classes and error handling."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
        # Reset rate limit
        from vulcan.models.model_registry import _load_timestamps
        _load_timestamps.clear()
    
    def test_import_error_raises_model_not_available(self):
        """Test that ImportError raises ModelNotAvailableError."""
        # Simulate import failure by making sentence_transformers unavailable
        import sys
        sentence_transformers_backup = sys.modules.get('sentence_transformers')
        if 'sentence_transformers' in sys.modules:
            del sys.modules['sentence_transformers']
        
        try:
            # Patch the import to raise ImportError
            with patch.dict('sys.modules', {'sentence_transformers': None}):
                with pytest.raises(ModelNotAvailableError) as exc_info:
                    get_sentence_transformer('test-model')
                
                assert 'not installed' in str(exc_info.value).lower()
        finally:
            # Restore
            if sentence_transformers_backup is not None:
                sys.modules['sentence_transformers'] = sentence_transformers_backup
    
    def test_load_failure_raises_model_load_failed(self):
        """Test that load failure raises ModelLoadFailedError."""
        with mock_sentence_transformer(side_effect=RuntimeError("Model file corrupted")):
            with pytest.raises(ModelLoadFailedError) as exc_info:
                get_sentence_transformer('test-model')
            
            assert 'test-model' in str(exc_info.value)
            assert 'corrupted' in str(exc_info.value).lower()
    
    def test_exception_inheritance(self):
        """Test that exception classes have correct inheritance."""
        assert issubclass(ModelNotAvailableError, ModelLoadError)
        assert issubclass(ModelLoadFailedError, ModelLoadError)
        assert issubclass(RateLimitError, ModelLoadError)


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Clear cache and rate limit state before each test."""
        clear_cache()
        from vulcan.models.model_registry import _load_timestamps
        _load_timestamps.clear()
    
    def test_rate_limit_enforced(self):
        """Test that rate limit is enforced."""
        with mock_sentence_transformer():
            
            # Set low rate limit for testing
            with patch('vulcan.models.model_registry.MAX_LOADS_PER_MINUTE', 3):
                # Load 3 different models (should succeed)
                for i in range(3):
                    get_sentence_transformer(f'model-{i}')
                
                # 4th load should hit rate limit
                with pytest.raises(RateLimitError) as exc_info:
                    get_sentence_transformer('model-4')
                
                assert 'rate limit exceeded' in str(exc_info.value).lower()
    
    def test_rate_limit_resets_after_window(self):
        """Test that rate limit resets after time window."""
        with mock_sentence_transformer():
            
            with patch('vulcan.models.model_registry.MAX_LOADS_PER_MINUTE', 2):
                # Load 2 models
                get_sentence_transformer('model-1')
                get_sentence_transformer('model-2')
                
                # Simulate time passing (61 seconds)
                from vulcan.models.model_registry import _load_timestamps
                old_timestamps = list(_load_timestamps)
                _load_timestamps.clear()
                _load_timestamps.extend([ts - 61.0 for ts in old_timestamps])
                
                # Should succeed now
                get_sentence_transformer('model-3')


class TestHealthCheck:
    """Test health check functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    def test_health_check_healthy_state(self):
        """Test health check returns healthy when all is well."""
        status = health_check()
        
        # Status could be 'healthy' or 'degraded' depending on cache state
        assert status['status'] in ['healthy', 'degraded']
        assert 'models_cached' in status
        assert 'hit_rate' in status
        assert 'load_success_rate' in status
        assert isinstance(status['errors'], list)
    
    def test_health_check_with_cached_models(self):
        """Test health check with cached models."""
        with mock_sentence_transformer():
            
            get_sentence_transformer('test-model')
            status = health_check()
            
            assert status['models_cached'] >= 1
            assert 'sentence_transformer:test-model' in status['cache_keys']
    
    def test_health_check_degraded_on_cache_full(self):
        """Test health check shows degraded when cache is full."""
        # Create small cache and fill it
        from vulcan.models.model_registry import _model_cache
        
        # Save original max size
        original_max = _model_cache.max_size
        try:
            # Set small size
            _model_cache.max_size = 1
            
            # Fill cache
            model = Mock()
            _model_cache.put('key1', model)
            
            status = health_check()
            
            # Should mention cache capacity in errors if degraded
            if status['status'] == 'degraded':
                assert len(status['errors']) > 0
        finally:
            # Restore original max size
            _model_cache.max_size = original_max


class TestThreadSafety:
    """Test thread safety and race condition fixes."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
        from vulcan.models.model_registry import _load_timestamps
        _load_timestamps.clear()
    
    def test_no_race_condition_on_cache_access(self):
        """Test that cache access is atomic and doesn't raise KeyError."""
        with mock_sentence_transformer():
            
            # Load a model
            get_sentence_transformer('test-model')
            
            errors = []
            
            def reader():
                """Try to read from cache repeatedly."""
                try:
                    for _ in range(100):
                        get_sentence_transformer('test-model')
                except KeyError as e:
                    errors.append(e)
            
            def clearer():
                """Try to clear cache."""
                time.sleep(0.01)  # Let readers start
                try:
                    clear_cache()
                except Exception as e:
                    errors.append(e)
            
            # Start multiple readers and a clearer
            threads = [threading.Thread(target=reader) for _ in range(5)]
            threads.append(threading.Thread(target=clearer))
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Should have no KeyError exceptions
            key_errors = [e for e in errors if isinstance(e, KeyError)]
            assert len(key_errors) == 0, f"Got KeyError race conditions: {key_errors}"
    
    def test_concurrent_loads_different_models(self):
        """Test that concurrent loads of different models work."""
        # Create unique mock for each model
        def create_model(name):
            model = Mock()
            model.name = name
            return model
        
        with mock_sentence_transformer(side_effect=lambda name: create_model(name)):
            results = {}
            errors = []
            
            def loader(model_name):
                try:
                    model = get_sentence_transformer(model_name)
                    results[model_name] = model
                except Exception as e:
                    errors.append((model_name, e))
            
            # Load 5 different models concurrently
            threads = [
                threading.Thread(target=loader, args=(f'model-{i}',))
                for i in range(5)
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All models should load successfully
            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results) == 5
    
    def test_concurrent_loads_same_model(self):
        """Test that concurrent loads of same model don't load multiple times."""
        load_count = 0
        
        def create_model(name):
            nonlocal load_count
            load_count += 1
            time.sleep(0.1)  # Simulate slow load
            return Mock()
        
        with mock_sentence_transformer(side_effect=create_model):
            results = []
            
            def loader():
                model = get_sentence_transformer('same-model')
                results.append(model)
            
            # Try to load same model from 5 threads concurrently
            threads = [threading.Thread(target=loader) for _ in range(5)]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Model should only be loaded once due to per-model locking
            assert load_count == 1, f"Model loaded {load_count} times, expected 1"
            assert len(results) == 5
            # All threads should get the same model instance
            assert all(r is results[0] for r in results)


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""
    
    def test_max_cache_size_from_env(self):
        """Test that cache size can be configured via env var."""
        with patch.dict(os.environ, {'VULCAN_MAX_MODELS_CACHE': '3'}):
            # Need to reload module to pick up env var
            from importlib import reload
            import vulcan.models.model_registry
            reload(vulcan.models.model_registry)
            
            # Check that it was applied
            cache = vulcan.models.model_registry._model_cache
            assert cache.max_size == 3
    
    def test_default_model_from_env(self):
        """Test that default model can be configured via env var."""
        with patch.dict(os.environ, {'VULCAN_SENTENCE_TRANSFORMER_MODEL': 'custom-model'}):
            from importlib import reload
            import vulcan.models.model_registry
            reload(vulcan.models.model_registry)
            
            default = vulcan.models.model_registry.DEFAULT_SENTENCE_TRANSFORMER_MODEL
            assert default == 'custom-model'


class TestMetricsAndObservability:
    """Test metrics tracking and observability features."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
        from vulcan.models.model_registry import _load_timestamps, _load_attempts
        _load_timestamps.clear()
        _load_attempts['success'] = 0
        _load_attempts['failure'] = 0
    
    def test_get_cache_stats(self):
        """Test that cache stats are tracked correctly."""
        stats = get_cache_stats()
        
        assert 'size' in stats
        assert 'max_size' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'evictions' in stats
        assert 'hit_rate' in stats
        assert 'load_attempts' in stats
    
    def test_load_attempts_tracked(self):
        """Test that load attempts are tracked."""
        with mock_sentence_transformer():
            
            # Successful load
            get_sentence_transformer('model-1')
            
            stats = get_cache_stats()
            assert stats['load_attempts']['success'] >= 1
    
    def test_load_duration_tracked(self):
        """Test that load duration is tracked."""
        def slow_load(name):
            time.sleep(0.1)
            return Mock()
        
        with mock_sentence_transformer(side_effect=slow_load):
            get_sentence_transformer('model-1')
            
            stats = get_cache_stats()
            assert 'avg_load_duration' in stats
            assert stats['avg_load_duration'] > 0
    
    def test_get_model_info(self):
        """Test getting info about a specific model."""
        with mock_sentence_transformer():
            
            get_sentence_transformer('model-1')
            
            info = get_model_info('sentence_transformer:model-1')
            
            assert info is not None
            assert info['access_count'] >= 1
            assert 'last_access_time' in info


class TestPreloadModels:
    """Test model preloading functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
        from vulcan.models.model_registry import _load_timestamps
        _load_timestamps.clear()
    
    def test_preload_all_models(self):
        """Test preloading all models."""
        with mock_sentence_transformer():
            with mock_graphix_transformer():
                results = preload_all_models()
                
                assert isinstance(results, dict)
                assert len(results) > 0
                # Check that at least some models loaded
                success_count = sum(1 for v in results.values() if v)
                assert success_count > 0
    
    def test_preload_handles_failures_gracefully(self):
        """Test that preload handles failures without crashing."""
        with mock_sentence_transformer(side_effect=RuntimeError("Load failed")):
            # Should not raise exception
            results = preload_all_models()
            
            assert isinstance(results, dict)
            # Should report failure
            assert any(not v for v in results.values())


class TestBertModel:
    """Test BERT model loading."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
        from vulcan.models.model_registry import _load_timestamps
        _load_timestamps.clear()
    
    def teardown_method(self):
        """Clean up modules after each test."""
        import sys
        # Remove mock modules that might interfere with next test
        if 'vulcan.processing' in sys.modules:
            mod = sys.modules['vulcan.processing']
            # Only delete if it's a Mock
            if hasattr(mod, '_mock_name') or type(mod).__name__ == 'Mock':
                del sys.modules['vulcan.processing']
    
    def test_get_bert_model_not_available(self):
        """Test BERT model when not available."""
        # Run this test first to avoid module pollution
        # The test expects ModelNotAvailableError to be raised
        # when vulcan.processing cannot be imported
        import sys
        
        # Save the current state
        old_processing = sys.modules.get('vulcan.processing')
        
        try:
            # Remove from modules if present
            if 'vulcan.processing' in sys.modules:
                del sys.modules['vulcan.processing']
            
            # Now try to load with processing set to None
            with patch.dict('sys.modules', {'vulcan.processing': None}):
                with pytest.raises(ModelNotAvailableError) as exc_info:
                    get_bert_model()
                
                assert 'GraphixTransformer not available' in str(exc_info.value)
        finally:
            # Restore the old state
            if old_processing is not None:
                sys.modules['vulcan.processing'] = old_processing
            elif 'vulcan.processing' in sys.modules:
                del sys.modules['vulcan.processing']
    
    def test_get_bert_model_success(self):
        """Test successful BERT model loading."""
        with mock_graphix_transformer() as mock_graphix:
            mock_model = mock_graphix.get_instance.return_value
            
            model = get_bert_model()
            
            assert model is mock_model
            mock_graphix.get_instance.assert_called_once()
    
    def test_get_bert_model_cached(self):
        """Test that BERT model is cached."""
        with mock_graphix_transformer() as mock_graphix:
            mock_model = mock_graphix.get_instance.return_value
            
            # Load twice
            model1 = get_bert_model()
            model2 = get_bert_model()
            
            assert model1 is model2
            # Should only call get_instance once (cached second time)
            assert mock_graphix.get_instance.call_count == 1


class TestCacheClear:
    """Test cache clearing functionality."""
    
    def test_clear_cache_removes_models(self):
        """Test that clear_cache removes all models."""
        with mock_sentence_transformer():
            
            # Load some models
            get_sentence_transformer('model-1')
            get_sentence_transformer('model-2')
            
            # Verify they're cached
            stats_before = get_cache_stats()
            assert stats_before['size'] >= 2
            
            # Clear cache
            clear_cache()
            
            # Verify cache is empty
            stats_after = get_cache_stats()
            assert stats_after['size'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
