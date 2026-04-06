"""Tests for extracted WorldModel request handling functions."""
import pytest
from unittest.mock import MagicMock, patch


class TestRequestHandlingImports:
    """Verify all extracted functions are importable."""

    def test_process_request_importable(self):
        from src.vulcan.world_model.request_handling import process_request
        assert callable(process_request)

    def test_request_dispatch_importable(self):
        from src.vulcan.world_model.request_dispatch import (
            _handle_self_referential_request,
            _handle_introspection_request,
            _handle_conversational_request,
        )
        assert callable(_handle_self_referential_request)

    def test_request_formatting_importable(self):
        from src.vulcan.world_model.request_formatting import (
            _invoke_reasoning_engine,
            _format_with_llm,
            _fallback_format,
        )
        assert callable(_invoke_reasoning_engine)


class TestSelfImprovementImports:
    """Verify self-improvement functions are importable."""

    def test_engine_importable(self):
        from src.vulcan.world_model.self_improvement_engine import (
            start_autonomous_improvement,
            stop_autonomous_improvement,
        )
        assert callable(start_autonomous_improvement)

    def test_apply_importable(self):
        from src.vulcan.world_model.self_improvement_apply import (
            _execute_improvement,
            _validate_code_ast,
        )
        assert callable(_execute_improvement)
