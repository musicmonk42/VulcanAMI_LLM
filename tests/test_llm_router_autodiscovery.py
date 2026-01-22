# ============================================================
# VULCAN-AGI LLM Query Router Auto-Discovery Tests
# ============================================================
# Tests for auto-discovery of LLM client and late-binding
# functionality added to fix the router initialization issue.
#
# VERSION HISTORY:
#     1.0.0 - Initial test suite for auto-discovery
# ============================================================

"""
Unit tests for LLM Query Router auto-discovery and late-binding.

These tests verify:
1. Auto-discovery of LLM client from multiple sources
2. Priority order of auto-discovery attempts
3. Late-binding via set_llm_client() method
4. Fallback behavior when no LLM client is found
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the modules under test
from src.vulcan.routing.llm_router import (
    LLMQueryRouter,
    get_llm_router,
    _discover_llm_client,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.chat = MagicMock(return_value='{"destination": "world_model", "confidence": 0.9}')
    return client


@pytest.fixture(autouse=True)
def reset_router_singleton():
    """Reset the router singleton before each test."""
    # Force new router for each test to avoid state pollution
    import src.vulcan.routing.llm_router as router_module
    router_module._llm_router_instance = None
    yield
    router_module._llm_router_instance = None


# ============================================================
# AUTO-DISCOVERY TESTS
# ============================================================


class TestAutoDiscovery:
    """Tests for LLM client auto-discovery."""
    
    def test_discover_from_singletons(self, mock_llm_client):
        """Should discover LLM client from singletons.get_llm_client()."""
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=mock_llm_client):
            from src.vulcan.routing.llm_router import _discover_llm_client
            
            client = _discover_llm_client()
            assert client is mock_llm_client
    
    def test_discover_from_hybrid_executor(self, mock_llm_client):
        """Should discover LLM client from HybridLLMExecutor.local_llm."""
        mock_executor = MagicMock()
        mock_executor.local_llm = mock_llm_client
        
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=mock_executor):
                from src.vulcan.routing.llm_router import _discover_llm_client
                
                client = _discover_llm_client()
                assert client is mock_llm_client
    
    def test_discover_from_main_global(self, mock_llm_client):
        """Should discover LLM client from main.global_llm_client."""
        mock_main = MagicMock()
        mock_main.global_llm_client = mock_llm_client
        
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=None):
                with patch.dict('sys.modules', {'vulcan.main': mock_main}):
                    from src.vulcan.routing.llm_router import _discover_llm_client
                    
                    client = _discover_llm_client()
                    assert client is mock_llm_client
    
    def test_discovery_priority_order(self, mock_llm_client):
        """Should try sources in priority order: singletons > hybrid > main."""
        singleton_client = MagicMock(name="singleton")
        hybrid_client = MagicMock(name="hybrid")
        main_client = MagicMock(name="main")
        
        mock_executor = MagicMock()
        mock_executor.local_llm = hybrid_client
        
        mock_main = MagicMock()
        mock_main.global_llm_client = main_client
        
        # Singletons returns a client - should use it and not try other sources
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=singleton_client):
            with patch('vulcan.llm.get_hybrid_executor', return_value=mock_executor):
                with patch.dict('sys.modules', {'vulcan.main': mock_main}):
                    from src.vulcan.routing.llm_router import _discover_llm_client
                    
                    client = _discover_llm_client()
                    assert client is singleton_client
    
    def test_discover_returns_none_when_unavailable(self):
        """Should return None when no LLM client is available."""
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=None):
                from src.vulcan.routing.llm_router import _discover_llm_client
                
                client = _discover_llm_client()
                assert client is None
    
    def test_discover_handles_import_errors(self):
        """Should handle import errors gracefully."""
        with patch('vulcan.reasoning.singletons.get_llm_client', side_effect=ImportError("Module not found")):
            from src.vulcan.routing.llm_router import _discover_llm_client
            
            # Should not raise, should return None
            client = _discover_llm_client()
            assert client is None
    
    def test_discover_handles_attribute_errors(self):
        """Should handle missing attributes gracefully."""
        mock_executor = MagicMock()
        # local_llm attribute doesn't exist
        del mock_executor.local_llm
        
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=mock_executor):
                from src.vulcan.routing.llm_router import _discover_llm_client
                
                client = _discover_llm_client()
                assert client is None


# ============================================================
# GET_LLM_ROUTER AUTO-DISCOVERY TESTS
# ============================================================


class TestGetLLMRouterAutoDiscovery:
    """Tests for get_llm_router() auto-discovery integration."""
    
    def test_get_llm_router_auto_discovers(self, mock_llm_client):
        """get_llm_router() should auto-discover LLM client when not provided."""
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=mock_llm_client):
            router = get_llm_router(force_new=True)
            assert router.llm_client is mock_llm_client
    
    def test_get_llm_router_uses_provided_client(self, mock_llm_client):
        """get_llm_router() should use provided client over auto-discovery."""
        provided_client = MagicMock(name="provided")
        
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=mock_llm_client):
            router = get_llm_router(llm_client=provided_client, force_new=True)
            assert router.llm_client is provided_client
    
    def test_get_llm_router_handles_no_client(self):
        """get_llm_router() should handle case when no client is discovered."""
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=None):
                router = get_llm_router(force_new=True)
                assert router.llm_client is None


# ============================================================
# SET_LLM_CLIENT TESTS
# ============================================================


class TestSetLLMClient:
    """Tests for set_llm_client() late-binding method."""
    
    def test_set_llm_client_updates_client(self, mock_llm_client):
        """set_llm_client() should update the LLM client."""
        router = LLMQueryRouter(llm_client=None)
        assert router.llm_client is None
        
        router.set_llm_client(mock_llm_client)
        assert router.llm_client is mock_llm_client
    
    def test_set_llm_client_enables_llm_classification(self, mock_llm_client):
        """After set_llm_client(), router should use LLM classification."""
        router = LLMQueryRouter(llm_client=None)
        
        # Before: should use fallback
        result1 = router.route("test query")
        assert result1.source in ["fallback", "guard", "cache"]
        
        # Set LLM client
        router.set_llm_client(mock_llm_client)
        router.cache.clear()  # Clear cache to force new classification
        
        # After: should use LLM (if not caught by guards)
        result2 = router.route("What is the meaning of life?")
        # This specific query should not be caught by guards
        assert result2.source in ["llm", "cache"]
    
    def test_set_llm_client_to_none_disables_llm(self, mock_llm_client):
        """set_llm_client(None) should disable LLM classification."""
        router = LLMQueryRouter(llm_client=mock_llm_client)
        assert router.llm_client is mock_llm_client
        
        router.set_llm_client(None)
        assert router.llm_client is None
    
    def test_set_llm_client_replaces_existing_client(self):
        """set_llm_client() should replace existing client."""
        client1 = MagicMock(name="client1")
        client2 = MagicMock(name="client2")
        
        router = LLMQueryRouter(llm_client=client1)
        assert router.llm_client is client1
        
        router.set_llm_client(client2)
        assert router.llm_client is client2


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for auto-discovery and late-binding."""
    
    def test_router_created_without_client_then_updated(self, mock_llm_client):
        """Router created without client can be updated later."""
        # Simulate early initialization without LLM
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=None):
            with patch('vulcan.llm.get_hybrid_executor', return_value=None):
                router = get_llm_router(force_new=True)
                assert router.llm_client is None
        
        # Later, when LLM becomes available
        router.set_llm_client(mock_llm_client)
        assert router.llm_client is mock_llm_client
        
        # Router should now use LLM classification
        router.cache.clear()
        result = router.route("What is quantum entanglement?")
        # Should either be from LLM or a guard (not fallback)
        assert result.source in ["llm", "guard", "cache"]
    
    def test_singleton_preserves_client_across_calls(self, mock_llm_client):
        """Singleton should preserve LLM client across get_llm_router() calls."""
        with patch('vulcan.reasoning.singletons.get_llm_client', return_value=mock_llm_client):
            router1 = get_llm_router(force_new=True)
            router2 = get_llm_router()
            
            assert router1 is router2
            assert router1.llm_client is mock_llm_client
            assert router2.llm_client is mock_llm_client


# ============================================================
# MODULE EXPORTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
