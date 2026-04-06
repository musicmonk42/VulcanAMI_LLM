"""
Tests for vulcan.safety.safety_status_endpoint module.

Covers: imports, router registration, GET /status endpoint behaviour
(healthy, uninitialised, degraded domain validators, internal error),
POST /initialize endpoint behaviour, and edge cases.

Uses unittest.mock to avoid importing real safety_validator / domain_validators.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# -------------------------------------------------------------------
# 1. Import Verification
# -------------------------------------------------------------------


class TestImports:
    def test_module_imports(self):
        import vulcan.safety.safety_status_endpoint  # noqa: F401

    def test_router_importable(self):
        from vulcan.safety.safety_status_endpoint import router
        assert router is not None

    def test_endpoint_functions_importable(self):
        from vulcan.safety.safety_status_endpoint import (
            get_safety_status,
            initialize_safety,
        )
        assert callable(get_safety_status)
        assert callable(initialize_safety)


# -------------------------------------------------------------------
# 2. Router Configuration
# -------------------------------------------------------------------


class TestRouterConfiguration:
    def test_router_has_status_route(self):
        from vulcan.safety.safety_status_endpoint import router

        paths = [route.path for route in router.routes]
        assert "/status" in paths

    def test_router_has_initialize_route(self):
        from vulcan.safety.safety_status_endpoint import router

        paths = [route.path for route in router.routes]
        assert "/initialize" in paths


# -------------------------------------------------------------------
# 3. GET /status -- Fully Initialised
# -------------------------------------------------------------------


class TestGetSafetyStatusHealthy:
    """Simulate a fully initialised safety system."""

    @pytest.fixture()
    def mock_validator(self):
        v = MagicMock()
        v._dedup_constraints = {"c1", "c2"}
        v._dedup_properties = {"p1"}
        v._dedup_invariants = {"i1", "i2", "i3"}
        return v

    @pytest.fixture()
    def mock_registry(self):
        reg = MagicMock()
        reg.list_domains.return_value = ["mythological", "scientific"]
        return reg

    @pytest.mark.asyncio
    async def test_returns_initialized_true(self, mock_validator, mock_registry):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        with patch(
            "vulcan.safety.safety_status_endpoint.get_safety_status.__module__",
            new="vulcan.safety.safety_status_endpoint",
        ):
            # We need to patch the lazy imports inside the function
            mock_domain_mod = MagicMock()
            mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = True
            mock_domain_mod.validator_registry = mock_registry

            mock_safety_mod = MagicMock()
            mock_safety_mod._SAFETY_SINGLETON_BUNDLE = mock_validator
            mock_safety_mod._SAFETY_SINGLETON_READY = True

            with patch.dict(
                "sys.modules",
                {
                    "vulcan.safety.domain_validators": mock_domain_mod,
                    "vulcan.safety.safety_validator": mock_safety_mod,
                },
            ):
                status = await get_safety_status()

        assert status["initialized"] is True
        assert status["constraints_count"] == 2
        assert status["properties_count"] == 1
        assert status["invariants_count"] == 3
        assert status["domains_count"] == 2
        assert "mythological" in status["domains_registered"]

    @pytest.mark.asyncio
    async def test_constraints_listed(self, mock_validator, mock_registry):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = True
        mock_domain_mod.validator_registry = mock_registry

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = mock_validator
        mock_safety_mod._SAFETY_SINGLETON_READY = True

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        assert isinstance(status["constraints"], list)
        assert set(status["constraints"]) == {"c1", "c2"}


# -------------------------------------------------------------------
# 4. GET /status -- Uninitialised (no validator)
# -------------------------------------------------------------------


class TestGetSafetyStatusUninitialised:
    @pytest.mark.asyncio
    async def test_returns_initialized_false(self):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = False
        mock_domain_mod.validator_registry = MagicMock()
        mock_domain_mod.validator_registry.list_domains.return_value = []

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = None
        mock_safety_mod._SAFETY_SINGLETON_READY = False

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        assert status["initialized"] is False
        assert status["constraints_count"] == 0
        assert status["properties_count"] == 0
        assert status["invariants_count"] == 0
        assert status["constraints"] == []

    @pytest.mark.asyncio
    async def test_validator_id_is_none(self):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = False
        mock_domain_mod.validator_registry = MagicMock()
        mock_domain_mod.validator_registry.list_domains.return_value = []

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = None
        mock_safety_mod._SAFETY_SINGLETON_READY = False

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        assert status["validator_id"] is None


# -------------------------------------------------------------------
# 5. GET /status -- Degraded Domain Validators
# -------------------------------------------------------------------


class TestGetSafetyStatusDegraded:
    @pytest.mark.asyncio
    async def test_domain_registry_error_handled(self):
        """When validator_registry.list_domains() throws, we get empty domains."""
        from vulcan.safety.safety_status_endpoint import get_safety_status

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = True
        mock_domain_mod.validator_registry = MagicMock()
        mock_domain_mod.validator_registry.list_domains.side_effect = RuntimeError("boom")

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = None
        mock_safety_mod._SAFETY_SINGLETON_READY = True

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        assert status["domains_registered"] == []
        assert status["domains_count"] == 0


# -------------------------------------------------------------------
# 6. GET /status -- Internal Error -> HTTPException
# -------------------------------------------------------------------


class TestGetSafetyStatusError:
    @pytest.mark.asyncio
    async def test_import_error_raises_http_500(self):
        """If the lazy imports fail entirely, we get HTTPException(500)."""
        from fastapi import HTTPException

        from vulcan.safety.safety_status_endpoint import get_safety_status

        # Simulate the lazy import of domain_validators failing
        with patch.dict("sys.modules", {"vulcan.safety.domain_validators": None}):
            # The function does `from .domain_validators import ...`
            # which will fail when the module is None in sys.modules.
            with pytest.raises(HTTPException) as exc_info:
                await get_safety_status()
            assert exc_info.value.status_code == 500


# -------------------------------------------------------------------
# 7. POST /initialize -- Success
# -------------------------------------------------------------------


class TestInitializeSafetySuccess:
    @pytest.mark.asyncio
    async def test_returns_status_initialized(self):
        from vulcan.safety.safety_status_endpoint import initialize_safety

        mock_validator = MagicMock()

        mock_safety_mod = MagicMock()
        mock_safety_mod.initialize_all_safety_components.return_value = mock_validator

        with patch.dict(
            "sys.modules",
            {"vulcan.safety.safety_validator": mock_safety_mod},
        ):
            result = await initialize_safety()

        assert result["status"] == "initialized"
        assert "validator_id" in result
        assert result["message"] == "Safety system initialized successfully"


# -------------------------------------------------------------------
# 8. POST /initialize -- Failure
# -------------------------------------------------------------------


class TestInitializeSafetyFailure:
    @pytest.mark.asyncio
    async def test_raises_http_500_on_error(self):
        from fastapi import HTTPException

        from vulcan.safety.safety_status_endpoint import initialize_safety

        mock_safety_mod = MagicMock()
        mock_safety_mod.initialize_all_safety_components.side_effect = RuntimeError(
            "init failed"
        )

        with patch.dict(
            "sys.modules",
            {"vulcan.safety.safety_validator": mock_safety_mod},
        ):
            with pytest.raises(HTTPException) as exc_info:
                await initialize_safety()
            assert exc_info.value.status_code == 500
            assert "init failed" in str(exc_info.value.detail)


# -------------------------------------------------------------------
# 9. Status Response Shape Validation
# -------------------------------------------------------------------


class TestStatusResponseShape:
    """Verify the response dict always has the expected top-level keys."""

    EXPECTED_KEYS = {
        "initialized",
        "validator_id",
        "domain_validators_initialized",
        "constraints_count",
        "properties_count",
        "invariants_count",
        "constraints",
        "properties",
        "invariants",
        "domains_registered",
        "domains_count",
    }

    @pytest.mark.asyncio
    async def test_all_keys_present_when_initialized(self):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        v = MagicMock()
        v._dedup_constraints = set()
        v._dedup_properties = set()
        v._dedup_invariants = set()

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = True
        mock_domain_mod.validator_registry.list_domains.return_value = []

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = v
        mock_safety_mod._SAFETY_SINGLETON_READY = True

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        missing = self.EXPECTED_KEYS - set(status.keys())
        assert not missing, f"Missing keys: {missing}"

    @pytest.mark.asyncio
    async def test_all_keys_present_when_uninitialised(self):
        from vulcan.safety.safety_status_endpoint import get_safety_status

        mock_domain_mod = MagicMock()
        mock_domain_mod._DOMAIN_VALIDATORS_INIT_DONE = False
        mock_domain_mod.validator_registry.list_domains.return_value = []

        mock_safety_mod = MagicMock()
        mock_safety_mod._SAFETY_SINGLETON_BUNDLE = None
        mock_safety_mod._SAFETY_SINGLETON_READY = False

        with patch.dict(
            "sys.modules",
            {
                "vulcan.safety.domain_validators": mock_domain_mod,
                "vulcan.safety.safety_validator": mock_safety_mod,
            },
        ):
            status = await get_safety_status()

        missing = self.EXPECTED_KEYS - set(status.keys())
        assert not missing, f"Missing keys: {missing}"
