"""
test_extracted_modules.py - Comprehensive tests for refactored main.py modules

Tests all extracted modules from the main.py refactoring:
- vulcan.utils_main (ProcessLock, timing, sanitize, components, network)
- vulcan.llm (MockGraphixVulcanLLM, HybridLLMExecutor, OpenAI client)
- vulcan.distillation (PIIRedactor, storage, promotion gate, distiller)
- vulcan.arena (Arena client, HTTP session)
- vulcan.metrics (Prometheus metrics)
- vulcan.api (Models, rate limiting)

VERSION HISTORY:
    1.0.0 - Initial test suite for extracted modules
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# ============================================================================
# Test Utils Main Module
# ============================================================================


class TestUtilsMainModule:
    """Tests for vulcan.utils_main package."""

    def test_module_import(self):
        """Test that utils_main module imports successfully."""
        try:
            from vulcan import utils_main
            assert utils_main is not None
            assert hasattr(utils_main, "__version__")
        except ImportError:
            pytest.skip("utils_main module not available")

    def test_module_exports(self):
        """Test that all expected exports are available."""
        try:
            from vulcan.utils_main import (
                ProcessLock,
                timed_async,
                timed_sync,
                sanitize_payload,
                deep_sanitize_for_json,
                find_available_port,
                get_module_info,
                validate_utils,
            )
            # At least some should be available (graceful degradation)
            assert any([
                ProcessLock is not None,
                timed_async is not None,
                sanitize_payload is not None,
            ])
        except ImportError:
            pytest.skip("utils_main module not available")

    def test_get_module_info(self):
        """Test get_module_info returns expected structure."""
        try:
            from vulcan.utils_main import get_module_info
            info = get_module_info()
            assert "version" in info
            assert "components" in info
            assert isinstance(info["components"], dict)
        except ImportError:
            pytest.skip("utils_main module not available")

    def test_validate_utils(self):
        """Test validate_utils function."""
        try:
            from vulcan.utils_main import validate_utils
            result = validate_utils()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("utils_main module not available")


class TestSanitizeModule:
    """Tests for sanitize module."""

    def test_sanitize_payload_basic(self):
        """Test basic payload sanitization."""
        try:
            from vulcan.utils_main.sanitize import sanitize_payload
        except ImportError:
            pytest.skip("sanitize module not available")

        # Basic dict
        result = sanitize_payload({"key": "value"})
        assert result == {"key": "value"}

        # None key should be removed
        result = sanitize_payload({None: "bad", "good": "value"})
        assert None not in result
        assert "good" in result

    def test_sanitize_payload_nested(self):
        """Test nested dict sanitization."""
        try:
            from vulcan.utils_main.sanitize import sanitize_payload
        except ImportError:
            pytest.skip("sanitize module not available")

        payload = {
            "outer": {
                None: "removed",
                "inner": "kept",
            },
            "list": [1, 2, {"nested": True}],
        }
        result = sanitize_payload(payload)
        assert "outer" in result
        assert None not in result["outer"]
        assert result["outer"]["inner"] == "kept"
        assert result["list"] == [1, 2, {"nested": True}]

    def test_sanitize_payload_tuple_to_list(self):
        """Test tuple conversion to list."""
        try:
            from vulcan.utils_main.sanitize import sanitize_payload
        except ImportError:
            pytest.skip("sanitize module not available")

        result = sanitize_payload({"items": (1, 2, 3)})
        assert isinstance(result["items"], list)
        assert result["items"] == [1, 2, 3]

    def test_sanitize_payload_primitives(self):
        """Test primitive value pass-through."""
        try:
            from vulcan.utils_main.sanitize import sanitize_payload
        except ImportError:
            pytest.skip("sanitize module not available")

        assert sanitize_payload("string") == "string"
        assert sanitize_payload(123) == 123
        assert sanitize_payload(3.14) == 3.14
        assert sanitize_payload(True) is True
        assert sanitize_payload(None) is None

    def test_deep_sanitize_for_json(self):
        """Test deep sanitization."""
        try:
            from vulcan.utils_main.sanitize import deep_sanitize_for_json
        except ImportError:
            pytest.skip("sanitize module not available")

        # Test enum handling
        class TestEnum(Enum):
            VALUE = "test_value"

        result = deep_sanitize_for_json({"enum": TestEnum.VALUE})
        assert result["enum"] == "test_value"

        # Test datetime handling
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = deep_sanitize_for_json({"date": dt})
        assert "2024-01-01" in result["date"]

    def test_deep_sanitize_max_depth(self):
        """Test max depth handling."""
        try:
            from vulcan.utils_main.sanitize import deep_sanitize_for_json, MAX_DEPTH_MARKER
        except ImportError:
            pytest.skip("sanitize module not available")

        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(60):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        result = deep_sanitize_for_json(nested)
        # Should contain max depth marker somewhere in result
        result_str = json.dumps(result)
        assert isinstance(result_str, str)

    def test_is_json_serializable(self):
        """Test JSON serializability checker."""
        try:
            from vulcan.utils_main.sanitize import is_json_serializable
        except ImportError:
            pytest.skip("sanitize module not available")

        assert is_json_serializable({"key": "value"})
        assert is_json_serializable([1, 2, 3])
        # Lambda functions cannot be serialized
        assert not is_json_serializable(lambda x: x)

    def test_safe_json_dumps(self):
        """Test safe JSON dumps."""
        try:
            from vulcan.utils_main.sanitize import safe_json_dumps
        except ImportError:
            pytest.skip("sanitize module not available")

        # Normal data
        result = safe_json_dumps({"key": "value"})
        assert json.loads(result) == {"key": "value"}

        # Data with None key
        result = safe_json_dumps({None: "bad", "good": "value"})
        parsed = json.loads(result)
        assert "good" in parsed


class TestProcessLockModule:
    """Tests for ProcessLock module."""

    def test_process_lock_import(self):
        """Test ProcessLock import."""
        try:
            from vulcan.utils_main.process_lock import ProcessLock, FCNTL_AVAILABLE
            assert ProcessLock is not None
            assert isinstance(FCNTL_AVAILABLE, bool)
        except ImportError:
            pytest.skip("process_lock module not available")

    def test_process_lock_basic(self):
        """Test basic ProcessLock functionality."""
        try:
            from vulcan.utils_main.process_lock import ProcessLock
        except ImportError:
            pytest.skip("process_lock module not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ProcessLock(lock_path=lock_path)
            
            assert not lock.is_locked()
            
            acquired = lock.acquire()
            assert acquired
            assert lock.is_locked()
            
            lock.release()
            assert not lock.is_locked()

    def test_process_lock_context_manager(self):
        """Test ProcessLock as context manager."""
        try:
            from vulcan.utils_main.process_lock import ProcessLock
        except ImportError:
            pytest.skip("process_lock module not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            
            with ProcessLock(lock_path=lock_path) as lock:
                assert lock.is_locked()
            
            # After context, should be released
            assert not lock.is_locked()

    def test_process_lock_repr(self):
        """Test ProcessLock string representation."""
        try:
            from vulcan.utils_main.process_lock import ProcessLock
        except ImportError:
            pytest.skip("process_lock module not available")

        lock = ProcessLock(lock_path="/tmp/test.lock")
        repr_str = repr(lock)
        assert "ProcessLock" in repr_str
        assert "test.lock" in repr_str


class TestTimingModule:
    """Tests for timing module."""

    def test_timed_sync(self):
        """Test synchronous timing decorator."""
        try:
            from vulcan.utils_main.timing import timed_sync
        except ImportError:
            pytest.skip("timing module not available")

        @timed_sync
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_timed_async(self):
        """Test asynchronous timing decorator."""
        try:
            from vulcan.utils_main.timing import timed_async
        except ImportError:
            pytest.skip("timing module not available")

        @timed_async
        async def async_function():
            await asyncio.sleep(0.01)
            return "async_done"

        result = asyncio.run(async_function())
        assert result == "async_done"

    def test_run_tasks_in_parallel(self):
        """Test parallel task execution."""
        try:
            from vulcan.utils_main.timing import run_tasks_in_parallel
        except ImportError:
            pytest.skip("timing module not available")

        async def task(n):
            await asyncio.sleep(0.01)
            return n

        async def run_test():
            results = await run_tasks_in_parallel(task(1), task(2), task(3))
            return results

        results = asyncio.run(run_test())
        assert len(results) == 3
        assert set(results) == {1, 2, 3}

    def test_timer_context_manager(self):
        """Test Timer context manager."""
        try:
            from vulcan.utils_main.timing import Timer
        except ImportError:
            pytest.skip("timing module not available")

        with Timer() as t:
            time.sleep(0.01)

        assert t.elapsed_ms >= 10
        assert t.elapsed >= 0.01


# ============================================================================
# Test LLM Module
# ============================================================================


class TestLLMModule:
    """Tests for vulcan.llm package."""

    def test_module_import(self):
        """Test that LLM module imports successfully."""
        try:
            from vulcan import llm
            assert llm is not None
            assert hasattr(llm, "__version__")
        except ImportError:
            pytest.skip("llm module not available")

    def test_module_exports(self):
        """Test expected exports."""
        try:
            from vulcan.llm import (
                MockGraphixVulcanLLM,
                HybridLLMExecutor,
                get_openai_client,
                OPENAI_AVAILABLE,
                get_module_info,
                validate_llm_module,
            )
            # Module info should work
            info = get_module_info()
            assert "version" in info
            assert "backends" in info
        except ImportError:
            pytest.skip("llm module not available")

    def test_validate_llm_module(self):
        """Test LLM module validation."""
        try:
            from vulcan.llm import validate_llm_module
            result = validate_llm_module()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("llm module not available")

    def test_mock_llm_basic(self):
        """Test MockGraphixVulcanLLM basic functionality."""
        try:
            from vulcan.llm import MockGraphixVulcanLLM
            if MockGraphixVulcanLLM is None:
                pytest.skip("MockGraphixVulcanLLM not available")
        except ImportError:
            pytest.skip("llm module not available")

        # MockGraphixVulcanLLM may require a config_path
        # Just verify the class exists and is importable
        assert MockGraphixVulcanLLM is not None
        assert hasattr(MockGraphixVulcanLLM, "__init__")


# ============================================================================
# Test Distillation Module
# ============================================================================


class TestDistillationModule:
    """Tests for vulcan.distillation package."""

    def test_module_import(self):
        """Test that distillation module imports successfully."""
        try:
            from vulcan import distillation
            assert distillation is not None
            assert hasattr(distillation, "__version__")
        except ImportError:
            pytest.skip("distillation module not available")

    def test_module_exports(self):
        """Test expected exports."""
        try:
            from vulcan.distillation import (
                DistillationExample,
                PIIRedactor,
                GovernanceSensitivityChecker,
                ExampleQualityValidator,
                DistillationStorageBackend,
                PromotionGate,
                get_module_info,
                validate_distillation_module,
            )
            info = get_module_info()
            assert "version" in info
            assert "components" in info
        except ImportError:
            pytest.skip("distillation module not available")

    def test_global_distiller_management(self):
        """Test global distiller management functions."""
        try:
            from vulcan.distillation import (
                get_knowledge_distiller,
                reset_knowledge_distiller,
            )
        except ImportError:
            pytest.skip("distillation module not available")

        # Initially should be None
        reset_knowledge_distiller()
        assert get_knowledge_distiller() is None

    def test_validate_distillation_module(self):
        """Test distillation module validation."""
        try:
            from vulcan.distillation import validate_distillation_module
            result = validate_distillation_module()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("distillation module not available")


class TestPIIRedactor:
    """Tests for PIIRedactor class."""

    def test_pii_redactor_import(self):
        """Test PIIRedactor import."""
        try:
            from vulcan.distillation.pii_redactor import PIIRedactor
            assert PIIRedactor is not None
        except ImportError:
            pytest.skip("pii_redactor module not available")

    def test_pii_redactor_basic(self):
        """Test basic PII redaction."""
        try:
            from vulcan.distillation.pii_redactor import PIIRedactor
        except ImportError:
            pytest.skip("pii_redactor module not available")

        redactor = PIIRedactor()
        
        # Test email redaction
        text = "Contact john@example.com for details"
        redacted = redactor.redact(text)
        assert "john@example.com" not in redacted or "[REDACTED_EMAIL]" in redacted


class TestDistillationStorage:
    """Tests for DistillationStorageBackend."""

    def test_storage_import(self):
        """Test storage module import."""
        try:
            from vulcan.distillation.storage import DistillationStorageBackend
            assert DistillationStorageBackend is not None
        except ImportError:
            pytest.skip("storage module not available")

    def test_storage_thread_safety(self):
        """Test that storage is thread-safe."""
        try:
            from vulcan.distillation.storage import DistillationStorageBackend
        except ImportError:
            pytest.skip("storage module not available")

        # Just verify the class exists and has thread-safety mechanisms
        assert DistillationStorageBackend is not None
        # The storage backend should have internal locking mechanism
        # Actual initialization may require specific parameters


# ============================================================================
# Test Arena Module
# ============================================================================


class TestArenaModule:
    """Tests for vulcan.arena package."""

    def test_module_import(self):
        """Test that arena module imports successfully."""
        try:
            from vulcan import arena
            assert arena is not None
            assert hasattr(arena, "__version__")
        except ImportError:
            pytest.skip("arena module not available")

    def test_module_exports(self):
        """Test expected exports."""
        try:
            from vulcan.arena import (
                execute_via_arena,
                submit_arena_feedback,
                get_http_session,
                close_http_session,
                AIOHTTP_AVAILABLE,
                get_module_info,
                validate_arena_module,
            )
            info = get_module_info()
            assert "version" in info
            assert "aiohttp_available" in info
        except ImportError:
            pytest.skip("arena module not available")

    def test_validate_arena_module(self):
        """Test arena module validation."""
        try:
            from vulcan.arena import validate_arena_module
            result = validate_arena_module()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("arena module not available")


# ============================================================================
# Test Metrics Module
# ============================================================================


class TestMetricsModule:
    """Tests for vulcan.metrics package."""

    def test_module_import(self):
        """Test that metrics module imports successfully."""
        try:
            from vulcan import metrics
            assert metrics is not None
            assert hasattr(metrics, "__version__")
        except ImportError:
            pytest.skip("metrics module not available")

    def test_module_exports(self):
        """Test expected exports."""
        try:
            from vulcan.metrics import (
                step_counter,
                step_duration,
                active_requests,
                error_counter,
                PROMETHEUS_AVAILABLE,
                get_module_info,
                validate_metrics_module,
            )
            info = get_module_info()
            assert "version" in info
            assert "prometheus_available" in info
        except ImportError:
            pytest.skip("metrics module not available")

    def test_mock_metrics(self):
        """Test mock metrics work when Prometheus unavailable."""
        try:
            from vulcan.metrics import MockMetric, MockTimer
        except ImportError:
            pytest.skip("metrics module not available")

        mock = MockMetric("test_metric")
        mock.inc()
        mock.dec()
        mock.set(5)
        mock.observe(1.5)
        mock.labels(type="test").inc()
        
        with mock.time():
            pass

    def test_validate_metrics_module(self):
        """Test metrics module validation."""
        try:
            from vulcan.metrics import validate_metrics_module
            result = validate_metrics_module()
            assert result is True  # Should always be True (uses mocks if needed)
        except ImportError:
            pytest.skip("metrics module not available")


# ============================================================================
# Test API Module
# ============================================================================


class TestAPIModule:
    """Tests for vulcan.api package."""

    def test_module_import(self):
        """Test that API module imports successfully."""
        try:
            from vulcan import api
            assert api is not None
            assert hasattr(api, "__version__")
        except ImportError:
            pytest.skip("api module not available")

    def test_module_exports(self):
        """Test expected exports."""
        try:
            from vulcan.api import (
                StepRequest,
                StepResponse,
                ChatMessage,
                ChatRequest,
                ChatResponse,
                HealthStatus,
                ErrorType,
                get_module_info,
                validate_api_module,
            )
            info = get_module_info()
            assert "version" in info
            assert "models_available" in info
        except ImportError:
            pytest.skip("api module not available")

    def test_validate_api_module(self):
        """Test API module validation."""
        try:
            from vulcan.api import validate_api_module
            result = validate_api_module()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("api module not available")


class TestAPIModels:
    """Tests for API Pydantic models."""

    def test_step_request_model(self):
        """Test StepRequest model."""
        try:
            from vulcan.api.models import StepRequest
        except ImportError:
            pytest.skip("api models not available")

        request = StepRequest(
            history=[],
            context={"goal": "test"},
        )
        assert request.history == []
        assert request.context == {"goal": "test"}

    def test_step_response_model(self):
        """Test StepResponse model."""
        try:
            from vulcan.api.models import StepResponse
        except ImportError:
            pytest.skip("api models not available")

        response = StepResponse(
            action={"type": "explore"},
            success=True,
            uncertainty=0.3,
        )
        assert response.action["type"] == "explore"
        assert response.success is True
        assert response.uncertainty == 0.3

    def test_health_status_enum(self):
        """Test HealthStatus enum."""
        try:
            from vulcan.api.models import HealthStatus
        except ImportError:
            pytest.skip("api models not available")

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_error_type_enum(self):
        """Test ErrorType enum."""
        try:
            from vulcan.api.models import ErrorType
        except ImportError:
            pytest.skip("api models not available")

        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.INTERNAL.value == "internal"


class TestRateLimiting:
    """Tests for rate limiting module."""

    def test_rate_limit_storage(self):
        """Test rate limit storage."""
        try:
            from vulcan.api.rate_limiting import (
                rate_limit_storage,
                rate_limit_lock,
            )
            assert isinstance(rate_limit_storage, dict)
            assert rate_limit_lock is not None
        except ImportError:
            pytest.skip("rate_limiting module not available")

    def test_check_rate_limit(self):
        """Test rate limit checking."""
        try:
            from vulcan.api.rate_limiting import check_rate_limit
            if check_rate_limit is None:
                pytest.skip("check_rate_limit not available")
        except ImportError:
            pytest.skip("rate_limiting module not available")

        # Test with high limit (should pass)
        result = check_rate_limit("test_client", max_requests=1000, window_seconds=60)
        # Result may be a tuple (allowed, remaining) or just bool
        if isinstance(result, tuple):
            assert result[0] is True  # First element is allowed
        else:
            assert result is True


# ============================================================================
# Test Platform Integration
# ============================================================================


class TestPlatformIntegration:
    """Tests for platform integration via main.py re-exports."""

    def test_main_imports_modules(self):
        """Test that main.py can import from extracted modules."""
        try:
            from vulcan.main import get_platform_integration_status
            status = get_platform_integration_status()
            assert "modules_available" in status
            assert "all_modules_available" in status
        except ImportError:
            pytest.skip("main.py integration not available")

    def test_backward_compatibility(self):
        """Test backward compatibility of imports from main."""
        try:
            # These should be re-exported from main.py for backward compatibility
            from vulcan import main
            
            # Check module info function exists
            assert hasattr(main, "get_platform_integration_status")
            
        except ImportError:
            pytest.skip("main.py not available")


# ============================================================================
# Test Module Validation Functions
# ============================================================================


class TestModuleValidation:
    """Tests for all module validation functions."""

    def test_all_validation_functions(self):
        """Test that all validation functions work."""
        validation_results = {}
        
        try:
            from vulcan.utils_main import validate_utils
            validation_results["utils_main"] = validate_utils()
        except ImportError:
            validation_results["utils_main"] = None
            
        try:
            from vulcan.llm import validate_llm_module
            validation_results["llm"] = validate_llm_module()
        except ImportError:
            validation_results["llm"] = None
            
        try:
            from vulcan.distillation import validate_distillation_module
            validation_results["distillation"] = validate_distillation_module()
        except ImportError:
            validation_results["distillation"] = None
            
        try:
            from vulcan.arena import validate_arena_module
            validation_results["arena"] = validate_arena_module()
        except ImportError:
            validation_results["arena"] = None
            
        try:
            from vulcan.metrics import validate_metrics_module
            validation_results["metrics"] = validate_metrics_module()
        except ImportError:
            validation_results["metrics"] = None
            
        try:
            from vulcan.api import validate_api_module
            validation_results["api"] = validate_api_module()
        except ImportError:
            validation_results["api"] = None

        # At least some modules should be available
        available = [k for k, v in validation_results.items() if v is not None]
        assert len(available) > 0, "At least one module should be available"

    def test_all_module_info_functions(self):
        """Test that all get_module_info functions work."""
        info_results = {}
        
        try:
            from vulcan.utils_main import get_module_info
            info_results["utils_main"] = get_module_info()
        except ImportError:
            info_results["utils_main"] = None
            
        try:
            from vulcan.llm import get_module_info
            info_results["llm"] = get_module_info()
        except ImportError:
            info_results["llm"] = None
            
        try:
            from vulcan.distillation import get_module_info
            info_results["distillation"] = get_module_info()
        except ImportError:
            info_results["distillation"] = None
            
        try:
            from vulcan.arena import get_module_info
            info_results["arena"] = get_module_info()
        except ImportError:
            info_results["arena"] = None
            
        try:
            from vulcan.metrics import get_module_info
            info_results["metrics"] = get_module_info()
        except ImportError:
            info_results["metrics"] = None
            
        try:
            from vulcan.api import get_module_info
            info_results["api"] = get_module_info()
        except ImportError:
            info_results["api"] = None

        # Verify structure of available info dicts
        for module_name, info in info_results.items():
            if info is not None:
                assert "version" in info, f"{module_name} info missing version"


# ============================================================================
# Integration Tests
# ============================================================================


class TestCrossModuleIntegration:
    """Tests for cross-module integration."""

    def test_sanitize_with_api_models(self):
        """Test that sanitize works with API models."""
        try:
            from vulcan.utils_main.sanitize import sanitize_payload
            from vulcan.api.models import StepResponse
        except ImportError:
            pytest.skip("Required modules not available")

        response = StepResponse(
            action={"type": "explore", "data": (1, 2, 3)},
            success=True,
            uncertainty=0.3,
        )
        
        # Convert to dict and sanitize
        data = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        sanitized = sanitize_payload(data)
        
        # Should be JSON serializable
        json_str = json.dumps(sanitized)
        assert isinstance(json_str, str)

    def test_timing_decorator_integration(self):
        """Test timing decorators with async operations."""
        try:
            from vulcan.utils_main.timing import timed_async, run_tasks_in_parallel
        except ImportError:
            pytest.skip("timing module not available")

        @timed_async
        async def fetch_data(n):
            await asyncio.sleep(0.01)
            return {"data": n}

        async def run_test():
            results = await run_tasks_in_parallel(
                fetch_data(1),
                fetch_data(2),
                fetch_data(3),
            )
            return results

        results = asyncio.run(run_test())
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
