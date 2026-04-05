"""Tests for centralized logging configuration."""
import logging
import pytest
from unittest.mock import patch
import src.logging_config as lc


class TestLoggingConfig:
    def setup_method(self):
        """Reset configured state between tests."""
        lc._CONFIGURED = False

    def test_configure_sets_level(self):
        lc.configure(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_is_idempotent(self):
        lc.configure(level="DEBUG")
        lc.configure(level="WARNING")  # should be no-op
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_adds_stderr_handler(self):
        lc.configure()
        root = logging.getLogger()
        handler_types = [type(h) for h in root.handlers]
        assert logging.StreamHandler in handler_types
