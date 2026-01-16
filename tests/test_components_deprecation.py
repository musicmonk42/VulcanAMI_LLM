"""
Tests for the deprecated components module.

Verifies:
1. Deprecation warnings are properly issued
2. Functions redirect correctly to singletons
3. Backward compatibility is maintained
4. All exported functions work as expected

Following industry standards for deprecation testing.
"""

import warnings
from unittest.mock import Mock, patch

import pytest


class TestComponentsDeprecation:
    """Test that components module is properly deprecated."""
    
    def test_import_issues_deprecation_warning(self):
        """Verify importing components module issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            # Import the module
            import vulcan.utils_main.components
            
            # Should have at least one deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            
            # Check message content
            assert any(
                "vulcan.utils_main.components is deprecated" in str(warning.message)
                for warning in deprecation_warnings
            )
    
    def test_initialize_component_issues_warning(self):
        """Verify initialize_component issues deprecation warning."""
        from vulcan.utils_main.components import initialize_component
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            # Try to call the function (will fail due to missing singletons, but that's OK)
            try:
                initialize_component("test", lambda: "value")
            except Exception:
                pass  # Expected to fail, we just want the warning
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert any(
                "initialize_component() is deprecated" in str(warning.message)
                for warning in deprecation_warnings
            )
    
    def test_get_initialized_components_issues_warning(self):
        """Verify get_initialized_components issues deprecation warning."""
        from vulcan.utils_main.components import get_initialized_components
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            result = get_initialized_components()
            
            # Should return empty dict
            assert result == {}
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
    
    def test_set_component_issues_warning(self):
        """Verify set_component issues deprecation warning."""
        from vulcan.utils_main.components import set_component
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            set_component("test", "value")
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert any(
                "set_component() is deprecated" in str(warning.message)
                for warning in deprecation_warnings
            )
    
    def test_get_component_issues_warning(self):
        """Verify get_component issues deprecation warning."""
        from vulcan.utils_main.components import get_component
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            result = get_component("test", default="default_value")
            
            # Should return default value
            assert result == "default_value"
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
    
    def test_clear_components_issues_warning(self):
        """Verify clear_components issues deprecation warning."""
        from vulcan.utils_main.components import clear_components
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            result = clear_components()
            
            # Should return 0
            assert result == 0
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0


class TestComponentsBackwardCompatibility:
    """Test backward compatibility of deprecated components module."""
    
    def test_all_functions_exported(self):
        """Verify all expected functions are exported."""
        from vulcan.utils_main.components import (
            initialize_component,
            get_initialized_components,
            set_component,
            get_component,
            clear_components,
            has_component,
            remove_component,
            list_components,
            shutdown_components,
        )
        
        # All functions should be callable
        assert callable(initialize_component)
        assert callable(get_initialized_components)
        assert callable(set_component)
        assert callable(get_component)
        assert callable(clear_components)
        assert callable(has_component)
        assert callable(remove_component)
        assert callable(list_components)
        assert callable(shutdown_components)
    
    def test_has_component_returns_false(self):
        """Verify has_component returns False (deprecated behavior)."""
        from vulcan.utils_main.components import has_component
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = has_component("test")
            assert result is False
    
    def test_remove_component_returns_none(self):
        """Verify remove_component returns None (deprecated behavior)."""
        from vulcan.utils_main.components import remove_component
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = remove_component("test")
            assert result is None
    
    def test_list_components_returns_empty_list(self):
        """Verify list_components returns empty list (deprecated behavior)."""
        from vulcan.utils_main.components import list_components
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = list_components()
            assert result == []
    
    def test_shutdown_components_does_nothing(self):
        """Verify shutdown_components does nothing (deprecated behavior)."""
        from vulcan.utils_main.components import shutdown_components
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Should not raise any exception
            shutdown_components()


class TestComponentsRedirectToSingletons:
    """Test that components functions correctly redirect to singletons."""
    
    def test_initialize_component_redirects_to_get_or_create(self):
        """Verify initialize_component redirects to singletons.get_or_create."""
        from vulcan.utils_main.components import initialize_component
        
        # Mock the singletons module
        with patch('vulcan.reasoning.singletons.get_or_create') as mock_get_or_create:
            mock_get_or_create.return_value = "test_value"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                result = initialize_component("test", lambda: "value")
                
                # Should have called get_or_create
                mock_get_or_create.assert_called_once_with("test", lambda: "value")
                assert result == "test_value"
    
    def test_get_component_redirects_to_get_singleton(self):
        """Verify get_component redirects to singletons.get_singleton."""
        from vulcan.utils_main.components import get_component
        
        # Mock the singletons module
        with patch('vulcan.reasoning.singletons.get_singleton') as mock_get_singleton:
            mock_get_singleton.return_value = "test_value"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                result = get_component("test")
                
                # Should have called get_singleton
                mock_get_singleton.assert_called_once_with("test")
                assert result == "test_value"
    
    def test_get_component_handles_import_error(self):
        """Verify get_component returns default on import error."""
        from vulcan.utils_main.components import get_component
        
        # Mock the singletons module to raise ImportError
        with patch('vulcan.reasoning.singletons.get_singleton', side_effect=ImportError("Test error")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                result = get_component("test", default="default_value")
                
                # Should return default value on ImportError
                assert result == "default_value"
    
    def test_get_component_handles_value_error(self):
        """Verify get_component returns default on ValueError."""
        from vulcan.utils_main.components import get_component
        
        # Mock the singletons module to raise ValueError
        with patch('vulcan.reasoning.singletons.get_singleton', side_effect=ValueError("Test error")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                result = get_component("test", default="default_value")
                
                # Should return default value on ValueError
                assert result == "default_value"
    
    def test_clear_components_redirects_to_reset_all(self):
        """Verify clear_components redirects to singletons.reset_all."""
        from vulcan.utils_main.components import clear_components
        
        # Mock the singletons module
        with patch('vulcan.reasoning.singletons.reset_all') as mock_reset_all:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                result = clear_components()
                
                # Should have called reset_all
                mock_reset_all.assert_called_once()
                # Should return 0 for backward compatibility
                assert result == 0


class TestComponentsMetadata:
    """Test components module metadata."""
    
    def test_module_version(self):
        """Verify components module has version 2.0.0."""
        from vulcan.utils_main import components
        
        assert hasattr(components, '__version__')
        assert components.__version__ == "2.0.0"
    
    def test_module_author(self):
        """Verify components module has author."""
        from vulcan.utils_main import components
        
        assert hasattr(components, '__author__')
        assert components.__author__ == "VULCAN-AGI Team"
    
    def test_module_docstring(self):
        """Verify components module has informative docstring."""
        from vulcan.utils_main import components
        
        assert components.__doc__ is not None
        assert "DEPRECATED" in components.__doc__
        assert "singletons" in components.__doc__


class TestComponentsExportsFromUtilsMain:
    """Test that components are exportable from utils_main package."""
    
    def test_components_functions_exported_from_utils_main(self):
        """Verify components functions can be imported from utils_main."""
        from vulcan.utils_main import (
            initialize_component,
            get_initialized_components,
            set_component,
            get_component,
            clear_components,
        )
        
        # All should be callable
        assert callable(initialize_component)
        assert callable(get_initialized_components)
        assert callable(set_component)
        assert callable(get_component)
        assert callable(clear_components)


class TestDeprecationMessageQuality:
    """Test the quality of deprecation messages."""
    
    def test_deprecation_messages_provide_migration_path(self):
        """Verify deprecation messages tell users what to use instead."""
        from vulcan.utils_main.components import initialize_component
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            
            try:
                initialize_component("test", lambda: "value")
            except Exception:
                pass
            
            # Check that the warning mentions the replacement
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            
            assert any(
                "singletons" in str(warning.message).lower()
                for warning in deprecation_warnings
            )
    
    def test_all_functions_have_deprecation_warnings(self):
        """Verify all public functions issue deprecation warnings."""
        from vulcan.utils_main.components import (
            initialize_component,
            get_initialized_components,
            set_component,
            get_component,
            clear_components,
            has_component,
            remove_component,
            list_components,
            shutdown_components,
        )
        
        functions_to_test = [
            (initialize_component, ("test", lambda: "value")),
            (get_initialized_components, ()),
            (set_component, ("test", "value")),
            (get_component, ("test",)),
            (clear_components, ()),
            (has_component, ("test",)),
            (remove_component, ("test",)),
            (list_components, ()),
            (shutdown_components, ()),
        ]
        
        for func, args in functions_to_test:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", DeprecationWarning)
                
                try:
                    func(*args)
                except Exception:
                    pass  # We only care about warnings, not functionality
                
                deprecation_warnings = [
                    warning for warning in w
                    if issubclass(warning.category, DeprecationWarning)
                ]
                
                assert len(deprecation_warnings) > 0, f"{func.__name__} should issue deprecation warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
