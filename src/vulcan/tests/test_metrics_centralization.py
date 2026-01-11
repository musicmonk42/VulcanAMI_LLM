"""
Test suite for metrics centralization refactoring.

This test validates that all endpoints properly import and use metrics
from the centralized vulcan.metrics module instead of creating inline metrics.
"""

import pytest


class TestMetricsCentralization:
    """Test that all endpoints use centralized metrics."""
    
    def test_execution_endpoint_uses_centralized_metrics(self):
        """Verify execution.py imports metrics from vulcan.metrics."""
        # Import the module
        from vulcan.endpoints import execution
        
        # Verify the module imports are present
        assert hasattr(execution, 'step_counter'), "step_counter should be imported"
        assert hasattr(execution, 'step_duration'), "step_duration should be imported"
        assert hasattr(execution, 'active_requests'), "active_requests should be imported"
        assert hasattr(execution, 'error_counter'), "error_counter should be imported"
        
        # Check that metrics are from vulcan.metrics (not inline created)
        from vulcan.metrics import step_counter, step_duration, active_requests, error_counter
        assert execution.step_counter is step_counter
        assert execution.step_duration is step_duration
        assert execution.active_requests is active_requests
        assert execution.error_counter is error_counter
    
    def test_memory_endpoint_uses_centralized_metrics(self):
        """Verify memory.py imports error_counter from vulcan.metrics."""
        from vulcan.endpoints import memory
        from vulcan.metrics import error_counter
        
        assert hasattr(memory, 'error_counter'), "error_counter should be imported"
        assert memory.error_counter is error_counter
    
    def test_planning_endpoint_uses_centralized_metrics(self):
        """Verify planning.py imports error_counter from vulcan.metrics."""
        from vulcan.endpoints import planning
        from vulcan.metrics import error_counter
        
        assert hasattr(planning, 'error_counter'), "error_counter should be imported"
        assert planning.error_counter is error_counter
    
    def test_status_endpoint_uses_centralized_metrics(self):
        """Verify status.py imports error_counter from vulcan.metrics."""
        from vulcan.endpoints import status
        from vulcan.metrics import error_counter
        
        assert hasattr(status, 'error_counter'), "error_counter should be imported"
        assert status.error_counter is error_counter
    
    def test_self_improvement_endpoint_uses_centralized_metrics(self):
        """Verify self_improvement.py imports error_counter from vulcan.metrics."""
        from vulcan.endpoints import self_improvement
        from vulcan.metrics import error_counter
        
        assert hasattr(self_improvement, 'error_counter'), "error_counter should be imported"
        assert self_improvement.error_counter is error_counter
    
    def test_unified_chat_endpoint_uses_centralized_metrics(self):
        """Verify unified_chat.py imports error_counter from vulcan.metrics (critical bug fix)."""
        from vulcan.endpoints import unified_chat
        from vulcan.metrics import error_counter
        
        # This was the critical bug - error_counter was used but not imported
        assert hasattr(unified_chat, 'error_counter'), "error_counter should be imported"
        assert unified_chat.error_counter is error_counter
    
    def test_collective_imports_improvement_metrics(self):
        """Verify collective.py imports self-improvement metrics."""
        from vulcan.orchestrator import collective
        
        # Check that METRICS_AVAILABLE flag is set
        assert hasattr(collective, 'METRICS_AVAILABLE'), "METRICS_AVAILABLE should be defined"
        
        # If metrics are available, verify they're imported
        if collective.METRICS_AVAILABLE:
            assert hasattr(collective, 'improvement_attempts')
            assert hasattr(collective, 'improvement_successes')
            assert hasattr(collective, 'improvement_failures')
            assert hasattr(collective, 'improvement_cost')


class TestMetricsConsistentNaming:
    """Test that all metrics use the vulcan_* prefix."""
    
    def test_all_metrics_have_vulcan_prefix(self):
        """Verify all centralized metrics use vulcan_* naming convention."""
        from vulcan.metrics import (
            step_counter,
            step_duration,
            active_requests,
            error_counter,
            improvement_attempts,
            improvement_successes,
            improvement_failures,
            improvement_cost,
        )
        
        # All metrics should have the vulcan_ prefix in their name
        assert step_counter.name == "vulcan_steps_total"
        assert step_duration.name == "vulcan_step_duration_seconds"
        assert active_requests.name == "vulcan_active_requests"
        assert error_counter.name == "vulcan_errors_total"
        assert improvement_attempts.name == "vulcan_improvement_attempts_total"
        assert improvement_successes.name == "vulcan_improvement_successes_total"
        assert improvement_failures.name == "vulcan_improvement_failures_total"
        assert improvement_cost.name == "vulcan_improvement_cost_usd_total"


class TestMetricsGracefulDegradation:
    """Test that metrics degrade gracefully when Prometheus is unavailable."""
    
    def test_metrics_work_without_prometheus(self):
        """Verify metrics module provides mock objects when prometheus_client unavailable."""
        from vulcan.metrics import (
            step_counter,
            error_counter,
            MockMetric,
            PROMETHEUS_AVAILABLE,
        )
        
        # Even if Prometheus is not available, metrics should exist
        assert step_counter is not None
        assert error_counter is not None
        
        # If not available, they should be MockMetric instances
        if not PROMETHEUS_AVAILABLE:
            assert isinstance(step_counter, MockMetric)
            assert isinstance(error_counter, MockMetric)
            
            # Mock metrics should support standard operations without errors
            step_counter.inc()
            error_counter.labels(error_type="test").inc()
            
            # MockMetric should return itself for labels
            labeled_counter = error_counter.labels(error_type="test")
            assert labeled_counter is error_counter
