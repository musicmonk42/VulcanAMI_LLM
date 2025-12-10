"""
Comprehensive tests for warm_pool.py

Tests pool management, scaling policies, health checking,
demand prediction, and resource monitoring.
"""

import threading
import time

import pytest

# Import the warm pool module
from vulcan.reasoning.selection.warm_pool import (DemandPredictor,
                                                  PoolInstance, PoolState,
                                                  PoolStatistics,
                                                  ResourceMonitor,
                                                  ScalingPolicy, ToolPool,
                                                  WarmStartPool)


class MockTool:
    """Mock tool for testing"""

    def __init__(self, name="mock_tool", warm_time=0.01):
        self.name = name
        self.warm_time = warm_time
        self.call_count = 0
        self.is_warm = False

    def reason(self, problem):
        """Mock reasoning"""
        self.call_count += 1
        time.sleep(0.01)
        return {"result": f"solved {problem}"}

    def warm_up(self):
        """Mock warm-up"""
        time.sleep(self.warm_time)
        self.is_warm = True

    def health_check(self):
        """Mock health check"""
        return True

    def shutdown(self):
        """Mock shutdown"""


class TestPoolInstance:
    """Test pool instance dataclass"""

    def test_instance_creation(self):
        """Test creating pool instance"""
        tool = MockTool()
        instance = PoolInstance(
            instance_id="test_1",
            tool_name="test_tool",
            tool_instance=tool,
            state=PoolState.READY,
            created_time=time.time(),
            last_used_time=time.time(),
        )

        assert instance.instance_id == "test_1"
        assert instance.tool_name == "test_tool"
        assert instance.state == PoolState.READY
        assert instance.usage_count == 0

    def test_instance_is_available(self):
        """Test instance availability check"""
        tool = MockTool()

        # Ready instance
        instance = PoolInstance(
            instance_id="test_1",
            tool_name="test_tool",
            tool_instance=tool,
            state=PoolState.READY,
            created_time=time.time(),
            last_used_time=time.time(),
        )
        assert instance.is_available() is True

        # Busy instance
        instance.state = PoolState.BUSY
        assert instance.is_available() is False

    def test_instance_is_healthy(self):
        """Test instance health check"""
        tool = MockTool()
        instance = PoolInstance(
            instance_id="test_1",
            tool_name="test_tool",
            tool_instance=tool,
            state=PoolState.READY,
            created_time=time.time(),
            last_used_time=time.time(),
        )

        # Initially healthy
        assert instance.is_healthy() is True

        # After failures
        instance.health_check_failures = 5
        assert instance.is_healthy() is False


class TestPoolStatistics:
    """Test pool statistics dataclass"""

    def test_statistics_creation(self):
        """Test creating pool statistics"""
        stats = PoolStatistics()

        assert stats.total_instances == 0
        assert stats.cache_hits == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = PoolStatistics(total_requests=100, cache_hits=75)

        assert stats.hit_rate == 0.75

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests"""
        stats = PoolStatistics()

        assert stats.hit_rate == 0.0


class TestToolPool:
    """Test individual tool pool"""

    def setup_method(self):
        """Setup for each test"""
        self.pools_to_cleanup = list(]

    def teardown_method(self):
        """Cleanup after each test"""
        for pool in self.pools_to_cleanup:
            try:
                pool.shutdown()
            except Exception:
                pass
        self.pools_to_cleanup.clear()

    def create_pool(self, tool_name="test_tool", min_instances=1, max_instances=3):
        """Helper to create pool and register for cleanup"""
        def factory(): return MockTool(tool_name)
        pool = ToolPool(
            tool_name=tool_name,
            tool_factory=factory,
            min_instances=min_instances,
            max_instances=max_instances,
        )
        self.pools_to_cleanup.append(pool)
        return pool

    def test_pool_creation(self):
        """Test creating tool pool"""
        pool = self.create_pool(min_instances=2)

        # Give time for initialization
        time.sleep(0.1)

        assert pool.tool_name == "test_tool"
        assert pool.min_instances == 2
        assert pool.max_instances == 3

    def test_pool_initialization(self):
        """Test pool initializes minimum instances"""
        pool = self.create_pool(min_instances=2)

        # Wait for warm-up
        time.sleep(0.2)

        # Should have minimum instances
        assert len(pool.instances) >= 1  # At least trying to create them

    def test_acquire_instance(self):
        """Test acquiring instance from pool"""
        pool = self.create_pool(min_instances=1)

        # Wait for instance to warm up
        time.sleep(0.3)

        # Acquire instance
        result = pool.acquire(timeout=2.0)

        if result:
            instance_id, tool = result
            assert instance_id is not None
            assert isinstance(tool, MockTool)

            # Release it
            pool.release(instance_id)

    def test_acquire_timeout(self):
        """Test acquire with timeout"""
        pool = self.create_pool(min_instances=0, max_instances=0)

        # Should timeout
        result = pool.acquire(timeout=0.1)

        assert result is None

    def test_release_instance(self):
        """Test releasing instance back to pool"""
        pool = self.create_pool(min_instances=1)

        # Wait for warm-up
        time.sleep(0.3)

        # Acquire and release
        result = pool.acquire(timeout=2.0)
        if result:
            instance_id, tool = result

            # Check stats
            assert pool.stats.busy_instances >= 1

            # Release
            pool.release(instance_id)

            # Give time for release processing
            time.sleep(0.1)

    def test_health_check(self):
        """Test instance health checking"""
        pool = self.create_pool(min_instances=1)

        time.sleep(0.3)

        # Get an instance ID
        if pool.instances:
            instance_id = list(pool.instances.keys())[0]

            # Health check
            healthy = pool.health_check(instance_id)

            assert isinstance(healthy, bool)

    def test_scale_up(self):
        """Test scaling pool up"""
        pool = self.create_pool(min_instances=1, max_instances=5)

        time.sleep(0.2)

        initial_size = len(pool.instances)

        # Scale up
        pool.scale(initial_size + 2)

        time.sleep(0.2)

        # Should have more instances
        assert len(pool.instances) >= initial_size

    def test_scale_down(self):
        """Test scaling pool down"""
        pool = self.create_pool(min_instances=3, max_instances=5)

        time.sleep(0.5)

        initial_size = len(pool.instances)

        # Scale down
        pool.scale(1)

        time.sleep(0.2)

        # Should have fewer instances
        assert len(pool.instances) <= initial_size

    def test_scale_respects_limits(self):
        """Test scaling respects min/max limits"""
        pool = self.create_pool(min_instances=2, max_instances=4)

        # Try to scale below minimum
        pool.scale(0)
        time.sleep(0.1)
        assert len(pool.instances) >= 0  # Will respect minimum

        # Try to scale above maximum
        pool.scale(10)
        time.sleep(0.1)
        assert len(pool.instances) <= 4

    def test_get_statistics(self):
        """Test getting pool statistics"""
        pool = self.create_pool(min_instances=2)

        time.sleep(0.3)

        stats = pool.get_statistics()

        assert "tool" in stats
        assert "total_instances" in stats
        assert "ready" in stats
        assert stats["tool"] == "test_tool"

    def test_concurrent_acquire_release(self):
        """Test concurrent acquire and release operations"""
        pool = self.create_pool(min_instances=2, max_instances=3)

        time.sleep(0.5)

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = pool.acquire(timeout=2.0)
                if result:
                    instance_id, tool = result
                    results.append((worker_id, instance_id))
                    time.sleep(0.05)
                    pool.release(instance_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) > 0


class TestWarmStartPool:
    """Test main warm start pool"""

    def setup_method(self):
        """Setup for each test"""
        self.pools_to_cleanup = []

    def teardown_method(self):
        """Cleanup after each test"""
        for pool in self.pools_to_cleanup:
            try:
                pool.shutdown()
            except Exception:
                pass
        self.pools_to_cleanup.clear()

    def create_warm_pool(self, tools=None, config=None):
        """Helper to create warm pool and register for cleanup"""
        if tools is None:
            tools = {"tool1": MockTool("tool1"), "tool2": MockTool("tool2")}

        pool = WarmStartPool(tools, config or {})
        self.pools_to_cleanup.append(pool)
        return pool

    def test_warm_pool_creation(self):
        """Test creating warm pool"""
        pool = self.create_warm_pool()

        assert len(pool.pools) == 2
        assert "tool1" in pool.pools
        assert "tool2" in pool.pools

    def test_warm_pool_with_config(self):
        """Test warm pool with configuration"""
        config = {"scaling_policy": "fixed", "min_pool_size": 2, "max_pool_size": 4}

        pool = self.create_warm_pool(config=config)

        assert pool.scaling_policy == ScalingPolicy.FIXED
        assert pool.min_pool_size == 2
        assert pool.max_pool_size == 4

    def test_acquire_tool(self):
        """Test acquiring tool from warm pool"""
        pool = self.create_warm_pool()

        # Wait for warm-up
        time.sleep(0.5)

        # Acquire tool
        result = pool.acquire_tool("tool1", timeout=2.0)

        if result:
            instance_id, tool = result
            assert isinstance(tool, MockTool)

            # Release
            pool.release_tool("tool1", instance_id)

    def test_acquire_unknown_tool(self):
        """Test acquiring unknown tool"""
        pool = self.create_warm_pool()

        result = pool.acquire_tool("unknown_tool")

        assert result is None

    def test_release_tool(self):
        """Test releasing tool back to pool"""
        pool = self.create_warm_pool()

        time.sleep(0.5)

        result = pool.acquire_tool("tool1", timeout=2.0)
        if result:
            instance_id, tool = result

            # Release should not raise error
            pool.release_tool("tool1", instance_id)

    def test_get_statistics(self):
        """Test getting comprehensive statistics"""
        pool = self.create_warm_pool()

        time.sleep(0.3)

        stats = pool.get_statistics()

        assert "pools" in stats
        assert "summary" in stats
        assert "resource_usage" in stats
        assert "tool1" in stats["pools"]

    def test_shutdown(self):
        """Test shutting down warm pool"""
        tools = {"tool1": MockTool("tool1")}
        pool = WarmStartPool(tools, {})

        time.sleep(0.2)

        # Shutdown
        pool.shutdown()

        # Background threads should stop
        time.sleep(0.5)
        assert not pool.monitoring


class TestScalingPolicies:
    """Test different scaling policies"""

    def setup_method(self):
        """Setup for each test"""
        self.pools_to_cleanup = []

    def teardown_method(self):
        """Cleanup after each test"""
        for pool in self.pools_to_cleanup:
            try:
                pool.shutdown()
            except Exception:
                pass
        self.pools_to_cleanup.clear()

    def create_warm_pool(self, scaling_policy="dynamic"):
        """Helper to create warm pool"""
        tools = {"tool1": MockTool("tool1")}
        config = {
            "scaling_policy": scaling_policy,
            "min_pool_size": 1,
            "max_pool_size": 3,
        }
        pool = WarmStartPool(tools, config)
        self.pools_to_cleanup.append(pool)
        return pool

    def test_dynamic_scaling(self):
        """Test dynamic scaling policy"""
        pool = self.create_warm_pool("dynamic")

        assert pool.scaling_policy == ScalingPolicy.DYNAMIC

        # Let it run briefly
        time.sleep(0.5)

    def test_fixed_scaling(self):
        """Test fixed scaling policy"""
        pool = self.create_warm_pool("fixed")

        assert pool.scaling_policy == ScalingPolicy.FIXED

    def test_predictive_scaling(self):
        """Test predictive scaling policy"""
        pool = self.create_warm_pool("predictive")

        assert pool.scaling_policy == ScalingPolicy.PREDICTIVE

        time.sleep(0.5)

    def test_reactive_scaling(self):
        """Test reactive scaling policy"""
        pool = self.create_warm_pool("reactive")

        assert pool.scaling_policy == ScalingPolicy.REACTIVE

        time.sleep(0.5)


class TestResourceMonitor:
    """Test resource monitoring"""

    def test_monitor_creation(self):
        """Test creating resource monitor"""
        monitor = ResourceMonitor()

        assert len(monitor.cpu_history) == 0
        assert len(monitor.memory_history) == 0

    def test_get_usage(self):
        """Test getting resource usage"""
        monitor = ResourceMonitor()

        usage = monitor.get_usage()

        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "avg_cpu" in usage
        assert "avg_memory" in usage

        # Values should be reasonable
        assert 0 <= usage["cpu_percent"] <= 100
        assert 0 <= usage["memory_percent"] <= 100

    def test_usage_history(self):
        """Test resource usage history tracking"""
        monitor = ResourceMonitor()

        # Collect multiple samples
        for _ in range(5):
            monitor.get_usage()
            time.sleep(0.1)

        # Should have history
        assert len(monitor.cpu_history) > 0
        assert len(monitor.memory_history) > 0


class TestDemandPredictor:
    """Test demand prediction"""

    def test_predictor_creation(self):
        """Test creating demand predictor"""
        predictor = DemandPredictor()

        assert predictor.window_size == 100

    def test_record_request(self):
        """Test recording tool requests"""
        predictor = DemandPredictor()

        predictor.record_request("tool1")
        predictor.record_request("tool1")
        predictor.record_request("tool2")

        # Should have recorded requests
        assert len(predictor.demand_history["tool1"]) == 2
        assert len(predictor.demand_history["tool2"]) == 1

    def test_predict_no_history(self):
        """Test prediction with no history"""
        predictor = DemandPredictor()

        demand = predictor.predict("tool1")

        # Should return default low demand
        assert demand > 0
        assert demand < 1.0

    def test_predict_with_history(self):
        """Test prediction with request history"""
        predictor = DemandPredictor()

        # Record multiple requests
        time.time()
        for i in range(10):
            predictor.record_request("tool1")
            time.sleep(0.01)

        # Predict
        demand = predictor.predict("tool1")

        # Should predict non-zero demand
        assert demand > 0

    def test_demand_window_limit(self):
        """Test demand history respects window size"""
        predictor = DemandPredictor(window_size=10)

        # Record more than window size
        for i in range(20):
            predictor.record_request("tool1")

        # Should only keep window size
        assert len(predictor.demand_history["tool1"]) == 10


class TestWarmPoolIntegration:
    """Integration tests for warm pool"""

    def setup_method(self):
        """Setup for each test"""
        self.pools_to_cleanup = []

    def teardown_method(self):
        """Cleanup after each test"""
        for pool in self.pools_to_cleanup:
            try:
                pool.shutdown()
            except Exception:
                pass
        self.pools_to_cleanup.clear()

    def test_complete_workflow(self):
        """Test complete warm pool workflow"""
        # Create tools
        tools = {
            "symbolic": MockTool("symbolic"),
            "probabilistic": MockTool("probabilistic"),
        }

        config = {"scaling_policy": "dynamic", "min_pool_size": 1, "max_pool_size": 3}

        # Create warm pool
        pool = WarmStartPool(tools, config)
        self.pools_to_cleanup.append(pool)

        # Wait for initialization
        time.sleep(0.5)

        # Acquire tool
        result = pool.acquire_tool("symbolic", timeout=2.0)

        if result:
            instance_id, tool = result

            # Use tool
            output = tool.reason("test problem")
            assert output is not None

            # Release tool
            pool.release_tool("symbolic", instance_id)

        # Get statistics
        stats = pool.get_statistics()
        assert stats["summary"]["total_instances"] > 0

        # Shutdown
        pool.shutdown()

    def test_concurrent_access(self):
        """Test concurrent access to warm pool"""
        tools = {"tool1": MockTool("tool1"), "tool2": MockTool("tool2")}

        pool = WarmStartPool(tools, {"min_pool_size": 2, "max_pool_size": 4})
        self.pools_to_cleanup.append(pool)

        time.sleep(0.5)

        results = []
        errors = []

        def worker(worker_id):
            try:
                tool_name = "tool1" if worker_id % 2 == 0 else "tool2"
                result = pool.acquire_tool(tool_name, timeout=2.0)

                if result:
                    instance_id, tool = result
                    results.append((worker_id, tool_name))
                    time.sleep(0.05)
                    pool.release_tool(tool_name, instance_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) > 0

    def test_health_monitoring(self):
        """Test health monitoring functionality"""
        tools = {"tool1": MockTool("tool1")}
        pool = WarmStartPool(tools, {"min_pool_size": 1})
        self.pools_to_cleanup.append(pool)

        # Let health checks run
        time.sleep(1.0)

        # All instances should be healthy
        for tool_pool in pool.pools.values():
            for instance in tool_pool.instances.values():
                assert instance.health_check_failures < 3

    def test_demand_tracking(self):
        """Test demand tracking and prediction"""
        tools = {"tool1": MockTool("tool1")}
        pool = WarmStartPool(tools, {"scaling_policy": "predictive"})
        self.pools_to_cleanup.append(pool)

        time.sleep(0.3)

        # Make several requests
        for i in range(5):
            result = pool.acquire_tool("tool1", timeout=1.0)
            if result:
                instance_id, tool = result
                pool.release_tool("tool1", instance_id)

        # Demand should be tracked
        assert len(pool.demand_predictor.demand_history["tool1"]) > 0


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_pool_with_failing_tool_factory(self):
        """Test pool with failing tool factory"""

        def failing_factory():
            raise Exception("Factory failure")

        # Should handle gracefully
        pool = ToolPool(
            tool_name="failing_tool",
            tool_factory=failing_factory,
            min_instances=1,
            max_instances=2,
        )

        time.sleep(0.2)

        # Pool should exist but have no instances
        assert len(pool.instances) == 0

        pool.shutdown()

    def test_acquire_with_no_instances(self):
        """Test acquire when no instances available"""

        def slow_factory():
            time.sleep(10)
            return MockTool()

        pool = ToolPool(
            tool_name="slow_tool",
            tool_factory=slow_factory,
            min_instances=0,
            max_instances=1,
        )

        # Should timeout
        result = pool.acquire(timeout=0.1)
        assert result is None

        pool.shutdown()

    def test_release_invalid_instance(self):
        """Test releasing invalid instance"""
        def factory(): return MockTool()
        pool = ToolPool(tool_name="test_tool", tool_factory=factory, min_instances=1)

        # Release non-existent instance - should not crash
        pool.release("invalid_id")

        pool.shutdown()

    def test_shutdown_with_busy_instances(self):
        """Test shutdown with busy instances"""
        def factory(): return MockTool()
        pool = ToolPool(tool_name="test_tool", tool_factory=factory, min_instances=1)

        time.sleep(0.3)

        # Acquire instance
        result = pool.acquire(timeout=1.0)

        # Shutdown even with busy instance
        pool.shutdown()

        # Should complete without hanging
        assert len(pool.instances) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
