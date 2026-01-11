"""Test suite for arena/http_session.py - HTTP session lifecycle and cleanup"""

import asyncio
import atexit
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================
# CLEANUP REGISTRATION TESTS
# ============================================================


class TestCleanupRegistration:
    """Test atexit cleanup registration."""

    def test_cleanup_registration(self):
        """Verify atexit cleanup is registered."""
        # Import the module to trigger atexit registration
        import vulcan.arena.http_session as http_session_module
        
        # Check that _sync_cleanup is registered with atexit
        # We can't directly check atexit handlers, but we can verify the function exists
        assert hasattr(http_session_module, '_sync_cleanup')
        assert callable(http_session_module._sync_cleanup)

    @patch('vulcan.arena.http_session.close_http_session')
    @patch('vulcan.arena.http_session._http_session')
    def test_sync_cleanup_with_running_loop(self, mock_session, mock_close):
        """Test sync cleanup when event loop is running."""
        from vulcan.arena.http_session import _sync_cleanup
        
        # Setup mock session
        mock_session_obj = MagicMock()
        mock_session_obj.closed = False
        
        # Mock the global session
        with patch('vulcan.arena.http_session._http_session', mock_session_obj):
            # Mock event loop that is running
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            
            with patch('asyncio.get_event_loop', return_value=mock_loop):
                _sync_cleanup()
                
                # Should schedule cleanup as a task
                mock_loop.create_task.assert_called_once()

    @patch('vulcan.arena.http_session.close_http_session')
    @patch('vulcan.arena.http_session._http_session')
    def test_sync_cleanup_with_stopped_loop(self, mock_session, mock_close):
        """Test sync cleanup when event loop is stopped."""
        from vulcan.arena.http_session import _sync_cleanup
        
        # Setup mock session
        mock_session_obj = MagicMock()
        mock_session_obj.closed = False
        
        # Create a real event loop for testing
        loop = asyncio.new_event_loop()
        
        # Mock close_http_session as a coroutine
        async def mock_close_coro():
            pass
        
        with patch('vulcan.arena.http_session._http_session', mock_session_obj):
            with patch('vulcan.arena.http_session.close_http_session', return_value=mock_close_coro()):
                with patch('asyncio.get_event_loop', return_value=loop):
                    # Loop is not running, so run_until_complete should be called
                    _sync_cleanup()
        
        loop.close()

    def test_sync_cleanup_no_session(self):
        """Test sync cleanup when no session exists."""
        from vulcan.arena.http_session import _sync_cleanup
        
        with patch('vulcan.arena.http_session._http_session', None):
            # Should not raise exception
            _sync_cleanup()

    def test_sync_cleanup_no_event_loop(self):
        """Test sync cleanup when no event loop available."""
        from vulcan.arena.http_session import _sync_cleanup
        
        mock_session_obj = MagicMock()
        mock_session_obj.closed = False
        
        with patch('vulcan.arena.http_session._http_session', mock_session_obj):
            with patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
                # Should handle RuntimeError gracefully
                _sync_cleanup()


# ============================================================
# SESSION LIFECYCLE TESTS
# ============================================================


class TestSessionLifecycle:
    """Test session creation and cleanup."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Verify session creation and cleanup."""
        # Reset session
        import vulcan.arena.http_session as http_session_module
        http_session_module._http_session = None
        
        # Skip if aiohttp not available
        if not http_session_module.AIOHTTP_AVAILABLE:
            pytest.skip("aiohttp not available")
        
        from vulcan.arena.http_session import (
            close_http_session,
            get_http_session,
            is_session_active,
        )
        
        # Initially no session
        assert not is_session_active()
        
        # Get session - should create it
        session = await get_http_session()
        assert session is not None
        assert is_session_active()
        
        # Get session again - should return same instance
        session2 = await get_http_session()
        assert id(session) == id(session2)
        
        # Close session
        await close_http_session()
        assert not is_session_active()

    @pytest.mark.asyncio
    async def test_session_recreation_after_close(self):
        """Verify session can be recreated after closing."""
        import vulcan.arena.http_session as http_session_module
        
        # Skip if aiohttp not available
        if not http_session_module.AIOHTTP_AVAILABLE:
            pytest.skip("aiohttp not available")
        
        from vulcan.arena.http_session import (
            close_http_session,
            get_http_session,
        )
        
        # Create and close session
        session1 = await get_http_session()
        await close_http_session()
        
        # Create new session
        session2 = await get_http_session()
        
        # Should be different instances
        assert id(session1) != id(session2)

    @pytest.mark.asyncio
    async def test_get_pool_config(self):
        """Test getting pool configuration."""
        import vulcan.arena.http_session as http_session_module
        
        # Skip if aiohttp not available
        if not http_session_module.AIOHTTP_AVAILABLE:
            pytest.skip("aiohttp not available")
        
        from vulcan.arena.http_session import get_pool_config
        
        config = get_pool_config()
        
        assert 'pool_limit' in config
        assert 'pool_limit_per_host' in config
        assert 'dns_cache_ttl' in config
        assert 'total_timeout' in config
        assert 'connect_timeout' in config
        assert 'read_timeout' in config
        assert 'session_active' in config
        
        # Verify config values are reasonable
        assert config['pool_limit'] > 0
        assert config['pool_limit_per_host'] > 0
        assert config['total_timeout'] > 0
