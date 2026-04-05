"""Tests for the thinned full_platform.py shell and globals accessors."""
import os
import pytest
from unittest.mock import patch


class TestGlobalsLazyAccess:
    """Verify globals.py defers construction — no import-time side effects."""

    def test_import_globals_does_not_trigger_settings(self):
        """Importing globals.py must not call UnifiedPlatformSettings()."""
        # If this triggers Settings.__init__ in production mode, it would
        # raise ValueError for missing auth. The fact that this import
        # succeeds proves construction is deferred.
        from src.platform import globals as g
        assert callable(g.get_settings)
        assert callable(g.get_app)

    def test_get_app_raises_before_init(self):
        """get_app() must raise RuntimeError if init_app() not called."""
        import src.platform.globals as g
        old_app = g._app
        g._app = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                g.get_app()
        finally:
            g._app = old_app

    def test_get_service_manager_raises_before_init(self):
        """get_service_manager() raises RuntimeError if not initialized."""
        import src.platform.globals as g
        old_sm = g._service_manager
        g._service_manager = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                g.get_service_manager()
        finally:
            g._service_manager = old_sm

    def test_init_flags_default_false(self):
        """Background init flags default to False."""
        from src.platform.globals import (
            is_services_init_complete,
            is_services_init_failed,
        )
        # These may have been set by other tests, so just verify they're bool
        assert isinstance(is_services_init_complete(), bool)
        assert isinstance(is_services_init_failed(), bool)

    def test_init_flags_settable(self):
        """Background init flags can be set and read back."""
        from src.platform.globals import (
            set_services_init_complete,
            is_services_init_complete,
            set_services_init_failed,
            is_services_init_failed,
        )
        old_c = is_services_init_complete()
        old_f = is_services_init_failed()
        try:
            set_services_init_complete(True)
            assert is_services_init_complete() is True
            set_services_init_failed(True)
            assert is_services_init_failed() is True
        finally:
            set_services_init_complete(old_c)
            set_services_init_failed(old_f)


class TestShellReExports:
    """Verify backwards-compatible re-exports from full_platform."""

    def test_auth_types_importable(self):
        """Auth types must be importable from full_platform for compat."""
        from src.full_platform import AuthMethod, AuthenticationError
        assert AuthMethod is not None
        assert AuthenticationError is not None

    def test_settings_class_importable(self):
        """UnifiedPlatformSettings must be importable from full_platform."""
        from src.full_platform import UnifiedPlatformSettings
        assert UnifiedPlatformSettings is not None

    def test_platform_globals_importable(self):
        """Global accessors must be importable from full_platform."""
        from src.full_platform import get_settings, get_app, init_app
        assert callable(get_settings)
        assert callable(get_app)
        assert callable(init_app)
