"""
Comprehensive test suite for setup_agent.py
"""

import logging  # Import logging for level constants
import sys
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest

# Assuming setup_agent.py is in the same directory or accessible via python path
# If setup_agent.py is in src/, and tests are run from the root, use:
# from src.setup_agent import (...)
try:
    # Attempt import assuming tests run relative to src/
    import setup_agent  # Import the module itself to patch its logger
    from setup_agent import (
        VALID_ROLES,
        SetupError,
        ValidationError,
        setup,
        validate_agent_id,
        validate_roles,
    )
except ModuleNotFoundError:
    # Fallback assuming tests run from root and src is in PYTHONPATH
    import src.setup_agent as setup_agent  # Import the module itself
    from src.setup_agent import (
        VALID_ROLES,
        SetupError,
        ValidationError,
        setup,
        validate_agent_id,
        validate_roles,
    )


class TestValidateAgentId:
    """Test agent_id validation."""

    def test_validate_empty(self):
        """Test validation with empty agent_id."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_agent_id("")

    def test_validate_not_string(self):
        """Test validation with non-string."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_agent_id(123)

    def test_validate_too_short(self):
        """Test validation with too short agent_id."""
        with pytest.raises(ValidationError, match="at least 3 characters"):
            validate_agent_id("ab")

    def test_validate_too_long(self):
        """Test validation with too long agent_id."""
        long_id = "a" * 65
        with pytest.raises(ValidationError, match="at most 64 characters"):
            validate_agent_id(long_id)

    def test_validate_invalid_characters(self):
        """Test validation with invalid characters."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_agent_id("test@agent")

    def test_validate_starts_with_hyphen(self):
        """Test validation with leading hyphen."""
        with pytest.raises(ValidationError, match="cannot start with"):
            validate_agent_id("-test")

    def test_validate_starts_with_underscore(self):
        """Test validation with leading underscore."""
        with pytest.raises(ValidationError, match="cannot start with"):
            validate_agent_id("_test")

    def test_validate_valid(self):
        """Test validation with valid agent_id."""
        # Should not raise
        validate_agent_id("test_agent")
        validate_agent_id("agent-123")
        validate_agent_id("MyAgent456")


class TestValidateRoles:
    """Test role validation."""

    def test_validate_empty_list(self):
        """Test validation with empty list."""
        with pytest.raises(ValidationError, match="At least one role"):
            validate_roles([])

    def test_validate_not_list(self):
        """Test validation with non-list."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_roles("executor")

    def test_validate_role_not_string(self):
        """Test validation with non-string role."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_roles([123])

    def test_validate_empty_role(self):
        """Test validation with empty role."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_roles([""])

    def test_validate_whitespace_role(self):
        """Test validation with whitespace-only role."""
        with pytest.raises(ValidationError, match="cannot be whitespace"):
            validate_roles(["   "])

    def test_validate_invalid_characters(self):
        """Test validation with invalid characters in role."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_roles(["role@test"])

    def test_validate_normalizes_case(self):
        """Test that roles are normalized to lowercase."""
        result = validate_roles(["EXECUTOR", "Validator"])

        assert result == ["executor", "validator"]

    def test_validate_removes_duplicates(self):
        """Test that duplicate roles are removed."""
        result = validate_roles(["executor", "EXECUTOR", "validator"])

        assert len(result) == 2
        assert "executor" in result
        assert "validator" in result

    def test_validate_valid_roles(self):
        """Test validation with valid roles."""
        result = validate_roles(["executor", "validator", "admin"])

        assert len(result) == 3
        assert all(role in result for role in ["executor", "validator", "admin"])


class TestSetup:
    """Test setup function."""

    @patch("setup_agent.AgentRegistry")
    def test_setup_success(self, mock_registry_class):
        """Test successful setup."""
        mock_registry = MagicMock()
        mock_registry.register_agent.return_value = {
            "agent_id": "test_agent",
            "public_key": "mock_public_key",
            "private_key": "mock_private_key_long_enough_to_be_split_for_testing_output"
            * 3,
        }
        mock_registry_class.return_value = mock_registry

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("test_agent", ["executor", "validator"])

        assert result is True
        mock_registry.register_agent.assert_called_once_with(
            "test_agent", "test_agent", ["executor", "validator"]
        )
        output = mock_stdout.getvalue()
        assert "AGENT CREDENTIALS" in output
        assert "-----BEGIN PRIVATE KEY-----" in output
        assert "mock_private_key" in output
        assert "-----END PRIVATE KEY-----" in output
        assert "SUCCESS: 'test_agent'" in output

    @patch("setup_agent.AgentRegistry")
    def test_setup_already_registered(self, mock_registry_class):
        """Test setup with already registered agent."""
        mock_registry = MagicMock()
        mock_registry.register_agent.side_effect = ValueError("Agent already exists")
        mock_registry_class.return_value = mock_registry

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("existing_agent", ["executor"])

        assert result is True
        mock_registry.register_agent.assert_called_once_with(
            "existing_agent", "existing_agent", ["executor"]
        )
        output = mock_stdout.getvalue()
        assert "INFO: Agent 'existing_agent' was already registered" in output

    @patch("setup_agent.AgentRegistry")
    def test_setup_invalid_agent_id(self, mock_registry_class):
        """Test setup with invalid agent_id."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("ab", ["executor"])

        assert result is False
        mock_registry_class.assert_not_called()
        output = mock_stdout.getvalue()
        assert "ERROR: Agent ID must be at least 3 characters" in output

    @patch("setup_agent.AgentRegistry")
    def test_setup_invalid_roles(self, mock_registry_class):
        """Test setup with invalid roles."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("test_agent", [])

        assert result is False
        mock_registry_class.assert_not_called()
        output = mock_stdout.getvalue()
        assert "ERROR: At least one role must be specified" in output

    @patch("setup_agent.AgentRegistry")
    def test_setup_registry_init_fails(self, mock_registry_class):
        """Test setup when registry initialization fails."""
        mock_registry_class.side_effect = Exception("Init failed")

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("test_agent", ["executor"])

        assert result is False
        output = mock_stdout.getvalue()
        assert "ERROR: AgentRegistry initialization failed: Init failed" in output

    @patch("setup_agent.AgentRegistry")
    def test_setup_registration_fails_other_value_error(self, mock_registry_class):
        """Test setup when registration fails with a non-'already exists' ValueError."""
        mock_registry = MagicMock()
        mock_registry.register_agent.side_effect = ValueError(
            "Some other registration error"
        )
        mock_registry_class.return_value = mock_registry

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("test_agent", ["executor"])

        assert result is False
        mock_registry.register_agent.assert_called_once()
        output = mock_stdout.getvalue()
        assert (
            "ERROR: Agent registration failed: Some other registration error" in output
        )

    @patch("setup_agent.AgentRegistry")
    def test_setup_registration_fails_generic_exception(self, mock_registry_class):
        """Test setup when registration fails with a generic Exception."""
        mock_registry = MagicMock()
        mock_registry.register_agent.side_effect = Exception(
            "Generic registration failure"
        )
        mock_registry_class.return_value = mock_registry

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = setup("test_agent", ["executor"])

        assert result is False
        mock_registry.register_agent.assert_called_once()
        output = mock_stdout.getvalue()
        assert (
            "ERROR: Agent registration failed: Generic registration failure" in output
        )


class TestExceptions:
    """Test custom exceptions."""

    def test_setup_error(self):
        """Test SetupError."""
        error = SetupError("test error")

        assert str(error) == "test error"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("validation failed")

        assert str(error) == "validation failed"


class TestMain:
    """Test main function."""

    @patch("setup_agent.setup")
    @patch.object(
        sys, "argv", ["setup_agent.py", "test_agent", "executor", "validator"]
    )
    def test_main_success(self, mock_setup):
        """Test main with successful setup."""
        mock_setup.return_value = True

        from setup_agent import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_setup.assert_called_once_with("test_agent", ["executor", "validator"])

    @patch("setup_agent.setup")
    @patch.object(sys, "argv", ["setup_agent.py", "test_agent", "executor"])
    def test_main_failure(self, mock_setup):
        """Test main with failed setup."""
        mock_setup.return_value = False

        from setup_agent import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_setup.assert_called_once_with("test_agent", ["executor"])

    # --- FIX START ---
    @patch("setup_agent.setup")
    @patch("logging.getLogger")  # Mock the function that gets the root logger
    @patch.object(
        setup_agent, "logger", new_callable=MagicMock
    )  # Mock the module's logger instance
    @patch.object(sys, "argv", ["setup_agent.py", "verbose_agent", "reader", "-v"])
    def test_main_verbose(self, mock_module_logger, mock_get_root_logger, mock_setup):
        """Test main with verbose flag."""
        mock_root_logger = MagicMock()
        mock_get_root_logger.return_value = mock_root_logger  # getLogger() returns this

        mock_setup.return_value = True
        from setup_agent import main

        with pytest.raises(SystemExit):
            main()

        # Check if setLevel was called with DEBUG on both loggers
        mock_root_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_module_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_setup.assert_called_once_with("verbose_agent", ["reader"])

    @patch("setup_agent.setup")
    @patch("logging.getLogger")  # Mock the function that gets the root logger
    @patch.object(
        setup_agent, "logger", new_callable=MagicMock
    )  # Mock the module's logger instance
    @patch.object(sys, "argv", ["setup_agent.py", "quiet_agent", "monitor", "-q"])
    def test_main_quiet(self, mock_module_logger, mock_get_root_logger, mock_setup):
        """Test main with quiet flag."""
        mock_root_logger = MagicMock()
        mock_get_root_logger.return_value = mock_root_logger  # getLogger() returns this

        mock_setup.return_value = True
        from setup_agent import main

        with pytest.raises(SystemExit):
            main()

        # Check if setLevel was called with ERROR on both loggers
        mock_root_logger.setLevel.assert_called_with(logging.ERROR)
        mock_module_logger.setLevel.assert_called_with(logging.ERROR)
        mock_setup.assert_called_once_with("quiet_agent", ["monitor"])

    # --- FIX END ---


class TestConstants:
    """Test module constants."""

    def test_valid_roles_exist(self):
        """Test that VALID_ROLES is defined."""
        assert isinstance(VALID_ROLES, set)
        assert len(VALID_ROLES) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
