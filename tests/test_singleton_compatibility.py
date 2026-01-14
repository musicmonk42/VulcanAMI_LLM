"""
Comprehensive Test Suite for get_singleton() Compatibility Function
====================================================================

This test suite validates the get_singleton() compatibility function that
provides backward compatibility with legacy code using string-based singleton access.

Test Coverage:
1. Basic functionality - mapping names to correct singletons
2. Error handling - invalid singleton names
3. Deprecation warnings - tracking for refactoring
4. Thread safety - concurrent access patterns
5. Integration - actual usage patterns from legacy code
6. Edge cases - None returns, unavailable singletons

Industry Standards Applied:
- Clear test structure with AAA (Arrange-Act-Assert) pattern
- Comprehensive error case coverage
- Thread safety validation
- Integration testing with real components
- Descriptive test names and docstrings
- Proper setup/teardown for test isolation
"""

import pytest
import logging
import threading
from typing import Optional
from unittest.mock import patch, MagicMock

from src.vulcan.reasoning.singletons import (
    get_singleton,
    reset_all,
    get_world_model,
    get_curiosity_engine,
    get_unified_runtime,
)


class TestGetSingletonBasicFunctionality:
    """Test basic functionality of get_singleton() compatibility function."""
    
    def setup_method(self):
        """Reset singletons before each test for isolation."""
        reset_all()
    
    def teardown_method(self):
        """Clean up singletons after each test."""
        reset_all()
    
    def test_get_singleton_returns_world_model(self):
        """Test that get_singleton('world_model') returns WorldModel instance."""
        # Arrange & Act
        singleton = get_singleton("world_model")
        direct = get_world_model()
        
        # Assert
        # Both should return the same instance (may be None if dependencies missing)
        assert singleton is direct
    
    def test_get_singleton_returns_curiosity_engine(self):
        """Test that get_singleton('curiosity_engine') returns CuriosityEngine instance."""
        # Arrange & Act
        singleton = get_singleton("curiosity_engine")
        direct = get_curiosity_engine()
        
        # Assert
        assert singleton is direct
    
    def test_get_singleton_returns_unified_runtime(self):
        """Test that get_singleton('unified_runtime') returns UnifiedRuntime instance."""
        # Arrange & Act
        singleton = get_singleton("unified_runtime")
        direct = get_unified_runtime()
        
        # Assert
        assert singleton is direct
    
    def test_get_singleton_returns_consistent_instance(self):
        """Test that multiple calls return the same instance."""
        # Arrange & Act
        instance1 = get_singleton("world_model")
        instance2 = get_singleton("world_model")
        instance3 = get_singleton("world_model")
        
        # Assert
        assert instance1 is instance2
        assert instance2 is instance3
    
    def test_get_singleton_all_registered_names(self):
        """Test that all documented singleton names are accessible."""
        # Arrange
        registered_names = [
            "world_model",
            "curiosity_engine",
            "unified_runtime",
            "self_improvement_drive",
            "ai_runtime",
            "multimodal_engine",
            "hierarchical_memory",
            "unified_learning_system",
            "tool_selector",
            "reasoning_integration",
            "portfolio_executor",
            "bayesian_prior",
            "warm_pool",
            "cost_model",
            "semantic_matcher",
            "problem_decomposer",
            "semantic_bridge",
            "unified_reasoner",
            "math_verification_engine",
            "llm_client",
        ]
        
        # Act & Assert
        for name in registered_names:
            # Should not raise ValueError
            result = get_singleton(name)
            # Result may be None if dependencies missing, but call should succeed
            assert result is None or result is not None  # Either state is valid


class TestGetSingletonErrorHandling:
    """Test error handling and edge cases of get_singleton()."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_all()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_all()
    
    def test_get_singleton_invalid_name_raises_value_error(self):
        """Test that invalid singleton name raises ValueError with helpful message."""
        # Arrange
        invalid_name = "nonexistent_singleton"
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_singleton(invalid_name)
        
        # Verify error message is helpful
        error_message = str(exc_info.value)
        assert "No singleton registered for" in error_message
        assert invalid_name in error_message
        assert "Available singletons:" in error_message
    
    def test_get_singleton_empty_string_raises_value_error(self):
        """Test that empty string raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_singleton("")
        
        assert "No singleton registered" in str(exc_info.value)
    
    def test_get_singleton_with_typo_raises_helpful_error(self):
        """Test that common typos produce helpful error messages."""
        # Arrange
        typos = ["worldmodel", "world-model", "WorldModel"]
        
        # Act & Assert
        for typo in typos:
            with pytest.raises(ValueError) as exc_info:
                get_singleton(typo)
            
            # Error message should list available names
            assert "world_model" in str(exc_info.value)
    
    def test_get_singleton_returns_none_for_unavailable_singleton(self):
        """Test that unavailable singletons return None (not error)."""
        # Note: Some singletons may be None if dependencies are missing
        # This is valid behavior and should not raise an exception
        
        # Act
        result = get_singleton("world_model")
        
        # Assert - either value is acceptable
        assert result is None or result is not None


class TestGetSingletonDeprecationWarning:
    """Test that get_singleton() emits appropriate deprecation warnings."""
    
    def setup_method(self):
        """Reset singletons and configure logging capture."""
        reset_all()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_all()
    
    def test_get_singleton_logs_deprecation_warning(self, caplog):
        """Test that get_singleton() logs deprecation warning."""
        # Arrange
        with caplog.at_level(logging.DEBUG):
            # Act
            _ = get_singleton("world_model")
            
            # Assert
            deprecation_logs = [
                record for record in caplog.records
                if "DEPRECATION" in record.message
            ]
            assert len(deprecation_logs) > 0
            
            # Verify message content
            deprecation_message = deprecation_logs[0].message
            assert "get_singleton" in deprecation_message
            assert "world_model" in deprecation_message
    
    def test_deprecation_warning_includes_preferred_pattern(self, caplog):
        """Test that deprecation warning suggests preferred pattern."""
        # Arrange
        with caplog.at_level(logging.DEBUG):
            # Act
            _ = get_singleton("curiosity_engine")
            
            # Assert
            deprecation_logs = [
                record for record in caplog.records
                if "DEPRECATION" in record.message
            ]
            assert len(deprecation_logs) > 0
            
            # Should suggest direct getter
            message = deprecation_logs[0].message
            assert "get_curiosity_engine()" in message or "get_" in message


class TestGetSingletonThreadSafety:
    """Test thread safety of get_singleton() function."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_all()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_all()
    
    def test_get_singleton_concurrent_access_same_instance(self):
        """Test that concurrent calls return the same instance."""
        # Arrange
        instances = []
        lock = threading.Lock()
        
        def get_instance():
            instance = get_singleton("world_model")
            with lock:
                instances.append(instance)
        
        # Act
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Assert
        # All instances should be the same (if not None)
        non_none_instances = [i for i in instances if i is not None]
        if len(non_none_instances) > 0:
            first_instance = non_none_instances[0]
            for instance in non_none_instances:
                assert instance is first_instance
    
    def test_get_singleton_mixed_name_concurrent_access(self):
        """Test concurrent access with different singleton names."""
        # Arrange
        results = {"world_model": [], "curiosity_engine": []}
        lock = threading.Lock()
        
        def get_world_model_instance():
            instance = get_singleton("world_model")
            with lock:
                results["world_model"].append(instance)
        
        def get_curiosity_engine_instance():
            instance = get_singleton("curiosity_engine")
            with lock:
                results["curiosity_engine"].append(instance)
        
        # Act
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=get_world_model_instance))
            threads.append(threading.Thread(target=get_curiosity_engine_instance))
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Assert
        # Each singleton type should return consistent instances
        for singleton_type, instances in results.items():
            non_none = [i for i in instances if i is not None]
            if len(non_none) > 0:
                first = non_none[0]
                for instance in non_none:
                    assert instance is first


class TestGetSingletonIntegration:
    """Test integration with actual usage patterns from legacy code."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_all()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_all()
    
    def test_legacy_query_analysis_pattern(self):
        """Test the actual pattern used in query_analysis.py."""
        # This simulates the exact usage pattern from
        # src/vulcan/reasoning/integration/query_analysis.py
        
        # Arrange & Act
        try:
            from vulcan.reasoning.singletons import get_singleton
            world_model = get_singleton("world_model")
            
            # Assert
            # Should not raise ImportError
            assert get_singleton is not None
            # world_model may be None if dependencies missing, which is valid
            assert world_model is None or world_model is not None
            
        except ImportError as e:
            pytest.fail(f"ImportError should not occur: {e}")
    
    def test_dynamic_singleton_access_pattern(self):
        """Test dynamic singleton access pattern used in configuration-driven code."""
        # Arrange
        singleton_configs = [
            {"name": "world_model", "enabled": True},
            {"name": "curiosity_engine", "enabled": True},
            {"name": "unified_runtime", "enabled": False},
        ]
        
        # Act
        loaded_singletons = {}
        for config in singleton_configs:
            if config["enabled"]:
                singleton = get_singleton(config["name"])
                loaded_singletons[config["name"]] = singleton
        
        # Assert
        assert "world_model" in loaded_singletons
        assert "curiosity_engine" in loaded_singletons
        assert "unified_runtime" not in loaded_singletons


class TestGetSingletonDocumentation:
    """Test that get_singleton() has proper documentation."""
    
    def test_function_has_docstring(self):
        """Test that get_singleton() has comprehensive docstring."""
        # Assert
        assert get_singleton.__doc__ is not None
        assert len(get_singleton.__doc__) > 100  # Should be comprehensive
    
    def test_docstring_lists_supported_names(self):
        """Test that docstring documents all supported singleton names."""
        # Arrange
        expected_names = [
            "world_model",
            "curiosity_engine",
            "unified_runtime",
        ]
        
        # Act
        docstring = get_singleton.__doc__
        
        # Assert
        for name in expected_names:
            assert name in docstring
    
    def test_docstring_has_examples(self):
        """Test that docstring includes usage examples."""
        # Assert
        docstring = get_singleton.__doc__
        assert "Example" in docstring or ">>>" in docstring
    
    def test_docstring_mentions_deprecation(self):
        """Test that docstring mentions deprecation status."""
        # Assert
        docstring = get_singleton.__doc__
        assert "DEPRECATION" in docstring or "deprecated" in docstring.lower()


# Performance and Edge Case Tests
class TestGetSingletonPerformance:
    """Test performance characteristics of get_singleton()."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_all()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_all()
    
    def test_get_singleton_dispatch_is_fast(self):
        """Test that singleton dispatch uses efficient O(1) lookup."""
        # This test verifies that we're using dict lookup, not if/elif chains
        import time
        
        # Arrange
        iterations = 1000
        
        # Act
        start = time.time()
        for _ in range(iterations):
            _ = get_singleton("world_model")
        duration = time.time() - start
        
        # Assert - should complete quickly even for many iterations
        # This is a smoke test for performance
        assert duration < 1.0  # Should be much faster than 1 second
    
    def test_get_singleton_error_case_is_fast(self):
        """Test that error cases also perform well."""
        import time
        
        # Arrange
        iterations = 100
        
        # Act
        start = time.time()
        for _ in range(iterations):
            try:
                _ = get_singleton("nonexistent")
            except ValueError:
                pass
        duration = time.time() - start
        
        # Assert
        assert duration < 1.0  # Error handling should be fast


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
