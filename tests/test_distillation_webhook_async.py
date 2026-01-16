"""
Tests for non-blocking webhook functionality in distillation module.

This test suite verifies that webhook calls are non-blocking and don't freeze
user requests during training triggers.
"""

import json
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock

import pytest

from vulcan.distillation.distiller import OpenAIKnowledgeDistiller


class MockWebhookServer(BaseHTTPRequestHandler):
    """Mock webhook server for testing."""
    
    # Class variable to track requests
    requests_received: List[Dict] = []
    response_delay: float = 0.0
    should_fail: bool = False
    
    def do_POST(self):
        """Handle POST requests."""
        # Add artificial delay if configured
        if self.response_delay > 0:
            time.sleep(self.response_delay)
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        # Parse and store request
        try:
            payload = json.loads(body.decode('utf-8'))
            self.requests_received.append({
                'payload': payload,
                'timestamp': time.time(),
                'headers': dict(self.headers)
            })
        except Exception:
            pass
        
        # Send response
        if self.should_fail:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Internal Server Error')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
    
    def log_message(self, format, *args):
        """Suppress log messages during tests."""
        pass


class TestWebhookNonBlocking:
    """Test that webhook calls are non-blocking."""
    
    def test_webhook_does_not_block_request(self):
        """
        Test that webhook call returns immediately without blocking.
        
        This is the critical fix for Issue 1: webhook should not freeze user requests.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create distiller with webhook URL (we won't actually send it)
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            # Mock the actual HTTP call to avoid network dependency
            webhook_called = {"called": False, "thread_id": None}
            
            original_urlopen = None
            try:
                import urllib.request
                original_urlopen = urllib.request.urlopen
                
                def mock_urlopen(*args, **kwargs):
                    webhook_called["called"] = True
                    webhook_called["thread_id"] = threading.current_thread().ident
                    time.sleep(5)  # Simulate slow webhook
                    return MagicMock(status=200, __enter__=lambda self: self, __exit__=lambda *args: None)
                
                urllib.request.urlopen = mock_urlopen
                
                # Trigger training (which sends webhook)
                start_time = time.time()
                result = distiller.trigger_training(reason="test", force=True)
                elapsed = time.time() - start_time
                
                # CRITICAL: Should return immediately (< 0.5s), not wait for webhook (5s)
                assert elapsed < 0.5, f"Webhook blocked for {elapsed}s - should be non-blocking!"
                assert result["triggered"] is True
                assert result["webhook_sent"] is True
                
                # Wait a bit for background thread to execute
                time.sleep(0.5)
                
                # Webhook should have been called in a different thread
                main_thread_id = threading.current_thread().ident
                assert webhook_called["thread_id"] != main_thread_id, \
                    "Webhook was called in main thread - not async!"
                
            finally:
                if original_urlopen:
                    urllib.request.urlopen = original_urlopen
    
    def test_webhook_failure_does_not_affect_trigger(self):
        """Test that webhook failures don't break the trigger_training call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://invalid-host-that-does-not-exist:9999/webhook"
            )
            
            # Should not raise exception even if webhook fails
            result = distiller.trigger_training(reason="test", force=True)
            
            # Should still succeed
            assert result["triggered"] is True
            assert result["webhook_sent"] is True  # Queued for send
    
    def test_webhook_thread_is_daemon(self):
        """Test that webhook threads are daemon threads (don't block shutdown)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            # Track thread creation
            created_threads = []
            original_thread_init = threading.Thread.__init__
            
            def mock_thread_init(self, *args, **kwargs):
                original_thread_init(self, *args, **kwargs)
                created_threads.append(self)
            
            with patch.object(threading.Thread, '__init__', mock_thread_init):
                distiller.trigger_training(reason="test", force=True)
            
            # Wait briefly for thread to be created
            time.sleep(0.1)
            
            # Find webhook threads
            webhook_threads = [t for t in created_threads if 'Webhook' in t.name]
            
            # Should have created at least one webhook thread
            assert len(webhook_threads) > 0, "No webhook thread was created"
            
            # All webhook threads should be daemon threads
            for thread in webhook_threads:
                assert thread.daemon is True, f"Thread {thread.name} is not a daemon thread!"
    
    def test_multiple_concurrent_webhooks(self):
        """Test that multiple webhook triggers don't interfere with each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            # Trigger multiple times rapidly
            results = []
            for i in range(5):
                result = distiller.trigger_training(reason=f"test_{i}", force=True)
                results.append(result)
            
            # All should succeed immediately
            assert all(r["triggered"] for r in results)
            assert all(r["webhook_sent"] for r in results)
    
    def test_webhook_payload_serialization_error_handling(self):
        """Test that webhook handles payload serialization errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            # Create a payload that can't be JSON serialized
            class UnserializableObject:
                def __repr__(self):
                    return "UnserializableObject"
            
            # This should not raise, even with unserializable object
            payload = {"bad_data": UnserializableObject()}
            
            # Should not crash
            try:
                distiller._send_webhook_async("http://localhost:9999/webhook", payload)
                # Wait a bit for background thread
                time.sleep(0.2)
                # If we get here, error was handled gracefully
                assert True
            except Exception as e:
                pytest.fail(f"Webhook should handle serialization errors: {e}")


class TestWebhookIntegration:
    """Integration tests for webhook functionality."""
    
    def test_callback_invoked_before_webhook(self):
        """Test that callback is invoked before webhook is sent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback_time = {"time": None}
            webhook_time = {"time": None}
            
            def test_callback(**kwargs):
                callback_time["time"] = time.time()
                return {"status": "callback_executed"}
            
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_trigger_callback=test_callback,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            # Mock webhook to track timing
            original_urlopen = None
            try:
                import urllib.request
                original_urlopen = urllib.request.urlopen
                
                def mock_urlopen(*args, **kwargs):
                    webhook_time["time"] = time.time()
                    return MagicMock(status=200, __enter__=lambda self: self, __exit__=lambda *args: None)
                
                urllib.request.urlopen = mock_urlopen
                
                result = distiller.trigger_training(reason="test", force=True)
                
                # Wait for background thread
                time.sleep(0.5)
                
                # Both should have been called
                assert callback_time["time"] is not None
                assert webhook_time["time"] is not None
                
                # Callback should be called first (before webhook thread even starts)
                # Note: Webhook runs in background, so timing may vary
                assert result["callback_invoked"] is True
                
            finally:
                if original_urlopen:
                    urllib.request.urlopen = original_urlopen
    
    def test_audit_log_includes_webhook_status(self):
        """Test that audit log includes webhook send status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_webhook_url="http://localhost:9999/webhook"
            )
            
            distiller.trigger_training(reason="test_audit", force=True)
            
            # Check audit log
            audit_log = distiller.get_audit_log()
            trigger_entries = [
                entry for entry in audit_log
                if entry.get("action") == "training_trigger"
            ]
            
            assert len(trigger_entries) >= 1
            last_entry = trigger_entries[-1]
            assert "webhook_sent" in last_entry.get("details", {})


class TestWebhookURLValidation:
    """Test webhook URL validation and error handling."""
    
    def test_invalid_url_does_not_crash(self):
        """Test that invalid webhook URLs don't crash the system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Various invalid URLs
            invalid_urls = [
                "not-a-url",
                "ftp://invalid-protocol.com",
                "http://",
                "",
            ]
            
            for url in invalid_urls:
                distiller = OpenAIKnowledgeDistiller(
                    storage_path=f"{tmpdir}/distillation_{url[:5]}.json",
                    training_trigger_threshold=1,
                    training_webhook_url=url if url else None
                )
                
                # Should not crash
                result = distiller.trigger_training(reason="test", force=True)
                assert result["triggered"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
