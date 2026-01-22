"""
Test suite for admin service management endpoints.

Tests the ability to stop and start individual services via the admin API.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncServiceManagerStopStart:
    """Test AsyncServiceManager stop_service and start_service methods."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        mock = MagicMock()
        mock.routes = []
        return mock

    @pytest.fixture
    def service_manager(self):
        """Create an AsyncServiceManager instance for testing."""
        # Import here to avoid module-level import issues
        from src.full_platform import AsyncServiceManager
        return AsyncServiceManager()

    @pytest.fixture
    def registered_service(self, service_manager):
        """Set up a registered service for testing."""
        # Add a mock service to the manager
        service_manager.services["test_service"] = {
            "name": "test_service",
            "mounted": True,
            "mount_path": "/test",
            "health_path": "/test/health",
            "app": MagicMock(),
            "import_success": True,
            "import_path": "test.module",
            "docs_url": "/test/docs",
        }
        return service_manager

    @pytest.mark.asyncio
    async def test_stop_service_success(self, registered_service, mock_app):
        """Test successfully stopping a mounted service."""
        # Add a route that matches the mount path
        mock_route = MagicMock()
        mock_route.path = "/test"
        mock_app.routes = [mock_route]

        result = await registered_service.stop_service(mock_app, "test_service")

        assert result["success"] is True
        assert result["service"] == "test_service"
        assert result["status"] == "stopped"
        assert "stopped_at" in result
        assert registered_service.services["test_service"]["mounted"] is False

    @pytest.mark.asyncio
    async def test_stop_service_not_found(self, service_manager, mock_app):
        """Test stopping a service that doesn't exist."""
        result = await service_manager.stop_service(mock_app, "nonexistent_service")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_services" in result

    @pytest.mark.asyncio
    async def test_stop_service_already_stopped(self, service_manager, mock_app):
        """Test stopping a service that's already stopped."""
        service_manager.services["stopped_service"] = {
            "name": "stopped_service",
            "mounted": False,
            "mount_path": "/stopped",
            "import_success": True,
        }

        result = await service_manager.stop_service(mock_app, "stopped_service")

        assert result["success"] is False
        assert "not currently mounted" in result["error"].lower()
        assert result["status"] == "already_stopped"

    @pytest.mark.asyncio
    async def test_start_service_success(self, service_manager, mock_app):
        """Test successfully starting a stopped service."""
        service_manager.services["stopped_service"] = {
            "name": "stopped_service",
            "mounted": False,
            "mount_path": "/service",
            "health_path": "/service/health",
            "app": MagicMock(),
            "import_success": True,
            "import_path": "test.module",
            "docs_url": "/service/docs",
            "use_wsgi": False,
        }

        result = await service_manager.start_service(mock_app, "stopped_service")

        assert result["success"] is True
        assert result["service"] == "stopped_service"
        assert result["status"] == "running"
        assert "started_at" in result
        assert service_manager.services["stopped_service"]["mounted"] is True

    @pytest.mark.asyncio
    async def test_start_service_not_found(self, service_manager, mock_app):
        """Test starting a service that doesn't exist."""
        result = await service_manager.start_service(mock_app, "nonexistent_service")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_start_service_already_running(self, registered_service, mock_app):
        """Test starting a service that's already running."""
        result = await registered_service.start_service(mock_app, "test_service")

        assert result["success"] is False
        assert "already running" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_start_service_import_failed(self, service_manager, mock_app):
        """Test starting a service that failed to import."""
        service_manager.services["failed_service"] = {
            "name": "failed_service",
            "mounted": False,
            "import_success": False,
            "error": "Module not found",
        }

        result = await service_manager.start_service(mock_app, "failed_service")

        assert result["success"] is False
        assert "cannot be started" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_service_details_existing(self, registered_service):
        """Test getting details of an existing service."""
        details = await registered_service.get_service_details("test_service")

        assert details is not None
        assert details["name"] == "test_service"
        assert details["mounted"] is True
        assert details["mount_path"] == "/test"

    @pytest.mark.asyncio
    async def test_get_service_details_not_found(self, service_manager):
        """Test getting details of a non-existent service."""
        details = await service_manager.get_service_details("nonexistent")

        assert details is None


class TestAdminVerifyAccess:
    """Test _verify_admin_access function."""

    def test_verify_with_valid_api_key(self):
        """Test authentication with valid API key."""
        from src.full_platform import _verify_admin_access, settings

        # Mock settings to have a known API key
        with patch.object(settings, 'api_key', 'test-api-key-12345'):
            result = _verify_admin_access('test-api-key-12345', None)
            assert result is True

    def test_verify_with_invalid_api_key(self):
        """Test authentication with invalid API key."""
        from src.full_platform import _verify_admin_access, settings

        with patch.object(settings, 'api_key', 'correct-key'):
            result = _verify_admin_access('wrong-key', None)
            assert result is False

    def test_verify_with_no_credentials(self):
        """Test authentication with no credentials."""
        from src.full_platform import _verify_admin_access

        result = _verify_admin_access(None, None)
        assert result is False


class TestAdminEndpointsIntegration:
    """Integration tests for admin service management endpoints."""

    @pytest.fixture
    def mock_service_manager(self):
        """Create a mock service manager."""
        manager = AsyncMock()
        manager.get_service_status = AsyncMock(return_value={
            "vulcan": {"mounted": True, "mount_path": "/vulcan"},
            "registry": {"mounted": False, "mount_path": "/registry"},
        })
        manager.get_service_details = AsyncMock(return_value={
            "name": "vulcan",
            "mounted": True,
            "mount_path": "/vulcan",
        })
        manager.stop_service = AsyncMock(return_value={
            "success": True,
            "service": "vulcan",
            "status": "stopped",
        })
        manager.start_service = AsyncMock(return_value={
            "success": True,
            "service": "vulcan",
            "status": "running",
        })
        return manager

    def test_admin_endpoints_exist(self):
        """Verify admin endpoints are registered."""
        from src.full_platform import app

        # Get all route paths
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]

        # Check that admin endpoints exist
        assert "/admin/services" in route_paths
        assert "/admin/services/{service_name}" in route_paths
        assert "/admin/services/{service_name}/stop" in route_paths
        assert "/admin/services/{service_name}/start" in route_paths

    def test_admin_list_services_requires_auth(self):
        """Test that list services requires authentication."""
        from fastapi.testclient import TestClient
        from src.full_platform import app

        client = TestClient(app, raise_server_exceptions=False)

        # Request without credentials should return 401
        response = client.get("/admin/services")
        assert response.status_code == 401

    def test_admin_stop_service_requires_auth(self):
        """Test that stop service requires authentication."""
        from fastapi.testclient import TestClient
        from src.full_platform import app

        client = TestClient(app, raise_server_exceptions=False)

        # Request without credentials should return 401
        response = client.post("/admin/services/vulcan/stop")
        assert response.status_code == 401

    def test_admin_start_service_requires_auth(self):
        """Test that start service requires authentication."""
        from fastapi.testclient import TestClient
        from src.full_platform import app

        client = TestClient(app, raise_server_exceptions=False)

        # Request without credentials should return 401
        response = client.post("/admin/services/vulcan/start")
        assert response.status_code == 401
