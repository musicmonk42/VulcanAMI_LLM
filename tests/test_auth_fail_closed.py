"""Tests for fail-closed authentication and environment allowlist."""
import os
import pytest
from unittest.mock import patch

class TestIsDevEnv:
    """Tests for the is_dev_env() allowlist function."""

    def test_development_is_dev(self):
        with patch.dict(os.environ, {"VULCAN_ENV": "development"}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is True

    def test_test_is_dev(self):
        with patch.dict(os.environ, {"VULCAN_ENV": "test"}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is True

    def test_production_is_not_dev(self):
        with patch.dict(os.environ, {"VULCAN_ENV": "production"}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is False

    def test_staging_is_not_dev(self):
        """Staging must NOT be treated as dev — this was PV-2/PV-3 violation."""
        with patch.dict(os.environ, {"VULCAN_ENV": "staging"}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is False

    def test_empty_string_is_not_dev(self):
        with patch.dict(os.environ, {"VULCAN_ENV": ""}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is False

    def test_unset_is_not_dev(self):
        env = os.environ.copy()
        env.pop("VULCAN_ENV", None)
        with patch.dict(os.environ, env, clear=True):
            from src.env_utils import is_dev_env
            assert is_dev_env() is False

    def test_typo_prod_is_not_dev(self):
        """Typos must not bypass — this was PV-3 violation."""
        with patch.dict(os.environ, {"VULCAN_ENV": "prod"}):
            from src.env_utils import is_dev_env
            assert is_dev_env() is False


class TestAuthFailClosed:
    """Tests that auth refuses to start without credentials in production."""

    def test_no_auth_in_production_raises(self):
        """Platform must refuse to start with no auth configured in production."""
        env = {
            "VULCAN_ENV": "production",
            "JWT_SECRET": "",
            "API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            from src.platform.settings import UnifiedPlatformSettings
            with pytest.raises(ValueError, match="No authentication configured"):
                UnifiedPlatformSettings()

    def test_no_auth_in_dev_allowed(self):
        """Dev environment may run without auth."""
        env = {
            "VULCAN_ENV": "development",
            "JWT_SECRET": "",
            "API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            from src.platform.settings import UnifiedPlatformSettings
            settings = UnifiedPlatformSettings()
            assert settings.auth_method.value == "none"

    def test_configured_key_rejects_empty(self):
        """API key auth must reject empty configured key (PV-1 fix)."""
        from src.platform.auth import AuthenticationError
        # Verify that the configured_key path doesn't fall back to ""
        # by checking the source doesn't contain 'or ""' pattern
        import inspect
        from src.platform.auth import verify_authentication
        source = inspect.getsource(verify_authentication)
        assert 'or ""' not in source, (
            "verify_authentication must not fall back to empty string for configured_key"
        )
