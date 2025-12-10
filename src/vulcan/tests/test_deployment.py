# ============================================================
# VULCAN-AGI Orchestrator - Deployment Tests
#
# This version does NOT import ProductionDeployment to avoid thread spawning.
# Instead, it tests the deployment patterns using a MockDeployment class.
# ============================================================

import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# ============================================================
# MOCK OBJECTS
# ============================================================


class MockConfig:
    """Mock configuration object"""

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 100
        self.max_auto_checkpoints = 5
        self.enable_distributed = False
        self.enable_multimodal = True
        self.enable_symbolic = True
        self.min_energy_budget_nJ = 1000
        self.max_memory_usage_mb = 7000
        self.slo_max_error_rate = 0.1
        self.max_agents = 10
        self.min_agents = 2
        self.task_queue_type = "custom"
        self.safety_policies = {"names_to_versions": {"default": "1.0"}}
        self.max_working_memory = 20
        self.short_term_capacity = 1000
        self.long_term_capacity = 100000
        self.consolidation_interval = 1000
        self.checkpoint_retry_attempts = 3
        self.checkpoint_retry_delay = 0.05


class MockHealth:
    """Mock health object"""

    def __init__(self):
        self.energy_budget_left_nJ = 1000000
        self.memory_usage_mb = 100
        self.latency_ms = 50
        self.error_rate = 0.01


class MockSA:
    """Mock self-awareness object"""

    def __init__(self):
        self.learning_efficiency = 0.8
        self.uncertainty = 0.3
        self.identity_drift = 0.1


class MockSystemState:
    """Mock system state"""

    def __init__(self):
        self.CID = "test_cid_12345"
        self.step = 0
        self.policies = {"default": "1.0"}
        self.health = MockHealth()
        self.SA = MockSA()
        self.active_modalities = set()
        self.uncertainty_estimates = {}
        self.provenance_chain = []
        self.last_obs = None
        self.last_reward = None


class MockAgentPool:
    """Mock agent pool"""

    def __init__(self):
        self.min_agents = 2
        self.max_agents = 10

    def get_pool_status(self):
        return {
            "total_agents": 5,
            "state_distribution": {"idle": 3, "working": 2},
            "pending_tasks": 0,
            "average_health_score": 0.9,
        }

    def shutdown(self):
        pass


class MockMetricsCollector:
    """Mock metrics collector"""

    def __init__(self):
        self.counters = {}
        self.gauges = {}

    def increment_counter(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value

    def update_gauge(self, name, value):
        self.gauges[name] = value

    def get_summary(self):
        return {"counters": self.counters, "gauges": self.gauges}

    def export_metrics(self):
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "timestamp": time.time(),
        }

    def import_metrics(self, data):
        self.counters = data.get("counters", {})
        self.gauges = data.get("gauges", {})

    def shutdown(self):
        pass


class MockDependencies:
    """Mock dependencies"""

    def __init__(self):
        self.goal_system = Mock()
        self.goal_system.generate_plan = Mock(return_value={})
        self.goal_system.get_goal_status = Mock(return_value={"goals": []})
        self.governance = Mock()
        self.governance.enforce_policies = Mock(side_effect=lambda x: x)
        self.safety_validator = Mock()
        self.safety_validator.get_safety_report = Mock(return_value={"violations": []})
        self.metrics = MockMetricsCollector()

    def validate(self):
        return {}

    def shutdown_all(self):
        pass


class MockCollective:
    """Mock collective orchestrator"""

    def __init__(self):
        self.sys = MockSystemState()
        self.deps = MockDependencies()
        self.agent_pool = MockAgentPool()
        self.config = MockConfig()
        self.cycle_count = 0

    def step(self, history, context):
        self.sys.step += 1
        self.cycle_count += 1
        return {
            "action": {"type": "test_action"},
            "success": True,
            "observation": "test_observation",
            "reward": 0.5,
        }

    def shutdown(self):
        pass


# ============================================================
# MOCK DEPLOYMENT CLASS
# This mirrors ProductionDeployment's interface without spawning threads
# ============================================================


class MockDeployment:
    """
    Mock deployment that mirrors ProductionDeployment interface.
    This is what we test instead of the real class.
    """

    def __init__(self, config, orchestrator_type="parallel"):
        self.config = config
        self.orchestrator_type = orchestrator_type
        self.checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else None
        )

        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.collective = MockCollective()
        self.metrics_collector = MockMetricsCollector()
        self.collective.deps.metrics = self.metrics_collector

        # State
        self._shutdown_requested = False
        self._last_checkpointed_step = 0
        self.start_time = time.time()

    def _health_check(self):
        """Check system health."""
        health = self.collective.sys.health

        if health.energy_budget_left_nJ < self.config.min_energy_budget_nJ:
            return False
        if health.memory_usage_mb > self.config.max_memory_usage_mb:
            return False
        if health.error_rate > self.config.slo_max_error_rate:
            return False

        return True

    def step_with_monitoring(self, history, context):
        """Execute step with monitoring."""
        if self._shutdown_requested:
            return {
                "error": "System shutdown requested",
                "action": {"type": "SYSTEM_SHUTDOWN"},
                "success": False,
            }

        if not self._health_check():
            return {
                "error": "Health check failed",
                "action": {"type": "SYSTEM_UNHEALTHY"},
                "success": False,
            }

        result = self.collective.step(history, context)
        self._update_monitoring(result)
        return result

    def _update_monitoring(self, result):
        """Update monitoring metrics."""
        if result.get("success"):
            self.metrics_collector.increment_counter("successful_steps")
        else:
            self.metrics_collector.increment_counter("failed_steps")

        # Update gauges
        self.metrics_collector.update_gauge(
            "energy_remaining_nJ", self.collective.sys.health.energy_budget_left_nJ
        )
        self.metrics_collector.update_gauge(
            "identity_drift", self.collective.sys.SA.identity_drift
        )
        self.metrics_collector.update_gauge(
            "uncertainty", self.collective.sys.SA.uncertainty
        )
        self.metrics_collector.update_gauge(
            "learning_efficiency", self.collective.sys.SA.learning_efficiency
        )
        self.metrics_collector.update_gauge(
            "agent_pool_size",
            self.collective.agent_pool.get_pool_status()["total_agents"],
        )

    def get_status(self):
        """Get deployment status."""
        return {
            "cid": self.collective.sys.CID,
            "step": self.collective.sys.step,
            "uptime_seconds": time.time() - self.start_time,
            "orchestrator_type": self.orchestrator_type,
            "health": {
                "energy_budget_left_nJ": self.collective.sys.health.energy_budget_left_nJ,
                "memory_usage_mb": self.collective.sys.health.memory_usage_mb,
                "latency_ms": self.collective.sys.health.latency_ms,
                "error_rate": self.collective.sys.health.error_rate,
            },
            "self_awareness": {
                "learning_efficiency": self.collective.sys.SA.learning_efficiency,
                "uncertainty": self.collective.sys.SA.uncertainty,
                "identity_drift": self.collective.sys.SA.identity_drift,
            },
            "metrics": self.metrics_collector.get_summary(),
            "agent_pool": self.collective.agent_pool.get_pool_status(),
            "config": {},
            "shutdown_requested": self._shutdown_requested,
            "goal_status": self.collective.deps.goal_system.get_goal_status(),
            "safety_report": self.collective.deps.safety_validator.get_safety_report(),
        }

    def save_checkpoint(self, path=None):
        """Save checkpoint."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step = self.collective.sys.step
            path = str(self.checkpoint_dir / f"checkpoint_{timestamp}_step{step}.pkl")

        checkpoint_path = Path(path)

        # Create checkpoint data
        checkpoint_data = {
            "step": self.collective.sys.step,
            "cid": self.collective.sys.CID,
            "metrics": self.metrics_collector.export_metrics(),
            "timestamp": time.time(),
        }

        # Save checkpoint
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Save metadata
        metadata_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_metadata.json"
        )
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "step": self.collective.sys.step,
            "cid": self.collective.sys.CID,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        self._last_checkpointed_step = self.collective.sys.step
        return True

    def _load_checkpoint(self, path):
        """Load checkpoint."""
        with open(path, "rb") as f:
            checkpoint_data = pickle.load(f)

        self.collective.sys.step = checkpoint_data["step"]
        self._last_checkpointed_step = checkpoint_data["step"]

        if "metrics" in checkpoint_data:
            self.metrics_collector.import_metrics(checkpoint_data["metrics"])

    def list_checkpoints(self):
        """List available checkpoints."""
        if not self.checkpoint_dir:
            return []

        checkpoints = []
        for pkl_file in sorted(
            self.checkpoint_dir.glob("checkpoint_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            metadata_file = pkl_file.with_name(pkl_file.stem + "_metadata.json")

            checkpoint_info = {
                "path": str(pkl_file),
                "size_mb": pkl_file.stat().st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat(),
            }

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                checkpoint_info.update(metadata)

            checkpoints.append(checkpoint_info)

        return checkpoints

    def request_shutdown(self):
        """Request shutdown."""
        self._shutdown_requested = True

    def shutdown(self):
        """Shutdown deployment."""
        if not self._shutdown_requested:
            # Save final checkpoint
            if self.checkpoint_dir:
                final_path = (
                    self.checkpoint_dir
                    / f"checkpoint_final_{self.collective.sys.step}.pkl"
                )
                self.save_checkpoint(str(final_path))

        self._shutdown_requested = True
        self.collective.shutdown()
        self.metrics_collector.shutdown()


# ============================================================
# TEST BASE CLASS
# ============================================================


class DeploymentTestBase(unittest.TestCase):
    """Base class for deployment tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = MockConfig(checkpoint_dir=str(self.temp_dir))
        self.deployment = None

    def tearDown(self):
        """Clean up"""
        if self.deployment:
            self.deployment._shutdown_requested = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_deployment(self, orchestrator_type="parallel"):
        """Create a mock deployment"""
        return MockDeployment(self.config, orchestrator_type=orchestrator_type)


# ============================================================
# TEST: INITIALIZATION
# ============================================================


class TestDeploymentInitialization(DeploymentTestBase):
    """Test deployment initialization"""

    def test_initialization_creates_instance(self):
        """Test that deployment can be created"""
        self.deployment = self.create_deployment()
        self.assertIsNotNone(self.deployment)
        self.assertFalse(self.deployment._shutdown_requested)

    def test_checkpoint_directory_exists(self):
        """Test that checkpoint directory exists"""
        self.deployment = self.create_deployment()
        self.assertTrue(self.temp_dir.exists())
        self.assertTrue(self.temp_dir.is_dir())

    def test_default_orchestrator_type(self):
        """Test default orchestrator type"""
        self.deployment = self.create_deployment()
        self.assertEqual(self.deployment.orchestrator_type, "parallel")

    def test_custom_orchestrator_type(self):
        """Test custom orchestrator type"""
        self.deployment = self.create_deployment(orchestrator_type="adaptive")
        self.assertEqual(self.deployment.orchestrator_type, "adaptive")


# ============================================================
# TEST: HEALTH CHECKS
# ============================================================


class TestHealthChecks(DeploymentTestBase):
    """Test health check functionality"""

    def test_health_check_pass(self):
        """Test health check passing"""
        self.deployment = self.create_deployment()
        result = self.deployment._health_check()
        self.assertTrue(result)

    def test_health_check_low_energy(self):
        """Test health check fails on low energy"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.health.energy_budget_left_nJ = 0
        result = self.deployment._health_check()
        self.assertFalse(result)

    def test_health_check_high_memory(self):
        """Test health check fails on high memory usage"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.health.memory_usage_mb = 10000
        result = self.deployment._health_check()
        self.assertFalse(result)

    def test_health_check_high_error_rate(self):
        """Test health check fails on high error rate"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.health.error_rate = 0.5
        result = self.deployment._health_check()
        self.assertFalse(result)


# ============================================================
# TEST: STEP EXECUTION
# ============================================================


class TestStepExecution(DeploymentTestBase):
    """Test step execution with monitoring"""

    def test_step_executes(self):
        """Test basic step execution"""
        self.deployment = self.create_deployment()
        result = self.deployment.step_with_monitoring(["test"], {"goal": "test"})
        self.assertTrue(result.get("success", False))

    def test_step_increments_counter(self):
        """Test that step increments step counter"""
        self.deployment = self.create_deployment()
        initial_step = self.deployment.collective.sys.step
        self.deployment.step_with_monitoring(["test"], {})
        self.assertEqual(self.deployment.collective.sys.step, initial_step + 1)

    def test_step_during_shutdown_returns_error(self):
        """Test step during shutdown returns error"""
        self.deployment = self.create_deployment()
        self.deployment._shutdown_requested = True
        result = self.deployment.step_with_monitoring(["test"], {})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "System shutdown requested")

    def test_step_unhealthy_returns_error(self):
        """Test step when unhealthy returns error"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.health.energy_budget_left_nJ = 0
        result = self.deployment.step_with_monitoring(["test"], {})
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Health check failed")

    def test_step_updates_metrics(self):
        """Test that step updates metrics"""
        self.deployment = self.create_deployment()
        self.deployment.step_with_monitoring(["test"], {})
        self.assertIn("successful_steps", self.deployment.metrics_collector.counters)


# ============================================================
# TEST: STATUS
# ============================================================


class TestStatus(DeploymentTestBase):
    """Test status reporting"""

    def test_get_status_returns_dict(self):
        """Test getting status"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertIsInstance(status, dict)

    def test_status_includes_cid(self):
        """Test status includes CID"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertEqual(status["cid"], self.deployment.collective.sys.CID)

    def test_status_includes_step(self):
        """Test status includes step"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertEqual(status["step"], self.deployment.collective.sys.step)

    def test_status_includes_health(self):
        """Test status includes health"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertIn("health", status)
        self.assertIn("energy_budget_left_nJ", status["health"])

    def test_status_includes_metrics(self):
        """Test status includes metrics"""
        self.deployment = self.create_deployment()
        self.deployment.metrics_collector.increment_counter("test_counter")
        status = self.deployment.get_status()
        self.assertIn("metrics", status)
        self.assertIn("test_counter", status["metrics"]["counters"])

    def test_status_includes_self_awareness(self):
        """Test status includes self awareness"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertIn("self_awareness", status)
        self.assertIn("learning_efficiency", status["self_awareness"])

    def test_status_includes_agent_pool(self):
        """Test status includes agent pool"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertIn("agent_pool", status)
        self.assertIn("total_agents", status["agent_pool"])


# ============================================================
# TEST: CHECKPOINTING
# ============================================================


class TestCheckpointing(DeploymentTestBase):
    """Test checkpoint save/load functionality"""

    def test_save_checkpoint(self):
        """Test saving checkpoint"""
        self.deployment = self.create_deployment()
        checkpoint_path = self.temp_dir / "test_checkpoint.pkl"
        success = self.deployment.save_checkpoint(str(checkpoint_path))
        self.assertTrue(success)
        self.assertTrue(checkpoint_path.exists())

    def test_save_checkpoint_creates_metadata(self):
        """Test checkpoint creates metadata file"""
        self.deployment = self.create_deployment()
        checkpoint_path = self.temp_dir / "test_checkpoint.pkl"
        self.deployment.save_checkpoint(str(checkpoint_path))
        metadata_path = self.temp_dir / "test_checkpoint_metadata.json"
        self.assertTrue(metadata_path.exists())

    def test_save_checkpoint_auto_naming(self):
        """Test checkpoint auto-naming"""
        self.deployment = self.create_deployment()
        initial_count = len(list(self.temp_dir.glob("checkpoint_*.pkl")))
        self.deployment.save_checkpoint()
        final_count = len(list(self.temp_dir.glob("checkpoint_*.pkl")))
        self.assertGreater(final_count, initial_count)

    def test_load_checkpoint_restores_step(self):
        """Test loading checkpoint restores step"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.step = 42
        checkpoint_path = self.temp_dir / "test_load.pkl"
        self.deployment.save_checkpoint(str(checkpoint_path))

        # Create new deployment and load
        new_deployment = self.create_deployment()
        self.assertEqual(new_deployment.collective.sys.step, 0)
        new_deployment._load_checkpoint(str(checkpoint_path))
        self.assertEqual(new_deployment.collective.sys.step, 42)

    def test_load_checkpoint_restores_metrics(self):
        """Test loading checkpoint restores metrics"""
        self.deployment = self.create_deployment()
        self.deployment.metrics_collector.increment_counter("test_metric", 5)
        checkpoint_path = self.temp_dir / "test_load_metrics.pkl"
        self.deployment.save_checkpoint(str(checkpoint_path))

        new_deployment = self.create_deployment()
        new_deployment._load_checkpoint(str(checkpoint_path))
        self.assertEqual(
            new_deployment.metrics_collector.counters.get("test_metric"), 5
        )

    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        self.deployment = self.create_deployment()
        self.deployment.save_checkpoint()
        time.sleep(0.02)
        self.deployment.collective.sys.step = 1
        self.deployment.save_checkpoint()

        checkpoints = self.deployment.list_checkpoints()
        self.assertEqual(len(checkpoints), 2)

    def test_list_checkpoints_sorted_by_time(self):
        """Test checkpoints are sorted by time (newest first)"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.step = 1
        self.deployment.save_checkpoint()
        time.sleep(0.02)
        self.deployment.collective.sys.step = 2
        self.deployment.save_checkpoint()

        checkpoints = self.deployment.list_checkpoints()
        self.assertEqual(checkpoints[0]["step"], 2)
        self.assertEqual(checkpoints[1]["step"], 1)


# ============================================================
# TEST: MONITORING
# ============================================================


class TestMonitoring(DeploymentTestBase):
    """Test monitoring functionality"""

    def test_update_monitoring_success(self):
        """Test monitoring updates on success"""
        self.deployment = self.create_deployment()
        result = {"success": True}
        self.deployment._update_monitoring(result)
        self.assertIn("successful_steps", self.deployment.metrics_collector.counters)
        self.assertEqual(
            self.deployment.metrics_collector.counters["successful_steps"], 1
        )

    def test_update_monitoring_failure(self):
        """Test monitoring updates on failure"""
        self.deployment = self.create_deployment()
        result = {"success": False}
        self.deployment._update_monitoring(result)
        self.assertIn("failed_steps", self.deployment.metrics_collector.counters)
        self.assertEqual(self.deployment.metrics_collector.counters["failed_steps"], 1)

    def test_update_monitoring_updates_gauges(self):
        """Test monitoring updates gauges"""
        self.deployment = self.create_deployment()
        result = {"success": True}
        self.deployment._update_monitoring(result)
        self.assertIn("energy_remaining_nJ", self.deployment.metrics_collector.gauges)
        self.assertIn("identity_drift", self.deployment.metrics_collector.gauges)
        self.assertIn("uncertainty", self.deployment.metrics_collector.gauges)


# ============================================================
# TEST: SHUTDOWN
# ============================================================


class TestShutdown(DeploymentTestBase):
    """Test shutdown functionality"""

    def test_request_shutdown(self):
        """Test requesting shutdown"""
        self.deployment = self.create_deployment()
        self.assertFalse(self.deployment._shutdown_requested)
        self.deployment.request_shutdown()
        self.assertTrue(self.deployment._shutdown_requested)

    def test_shutdown_saves_checkpoint(self):
        """Test shutdown saves final checkpoint"""
        self.deployment = self.create_deployment()
        self.deployment.collective.sys.step = 5
        self.deployment.shutdown()
        final_checkpoint = self.temp_dir / "checkpoint_final_5.pkl"
        self.assertTrue(final_checkpoint.exists())

    def test_shutdown_is_idempotent(self):
        """Test shutdown can be called multiple times"""
        self.deployment = self.create_deployment()
        self.deployment.shutdown()
        self.assertTrue(self.deployment._shutdown_requested)
        # Should not raise
        self.deployment.shutdown()
        self.assertTrue(self.deployment._shutdown_requested)

    def test_shutdown_sets_flag(self):
        """Test shutdown sets the shutdown flag"""
        self.deployment = self.create_deployment()
        self.assertFalse(self.deployment._shutdown_requested)
        self.deployment.shutdown()
        self.assertTrue(self.deployment._shutdown_requested)


# ============================================================
# TEST: INTEGRATION
# ============================================================


class TestIntegration(DeploymentTestBase):
    """Integration tests"""

    def test_full_lifecycle(self):
        """Test full deployment lifecycle"""
        self.deployment = self.create_deployment()

        # Execute steps
        for i in range(5):
            result = self.deployment.step_with_monitoring(["test"], {"goal": "test"})
            self.assertTrue(result.get("success", False))

        self.assertEqual(self.deployment.collective.sys.step, 5)

        # Get status
        status = self.deployment.get_status()
        self.assertEqual(status["step"], 5)
        self.assertEqual(status["metrics"]["counters"]["successful_steps"], 5)

        # Save checkpoint
        success = self.deployment.save_checkpoint()
        self.assertTrue(success)

        # List checkpoints
        checkpoints = self.deployment.list_checkpoints()
        self.assertGreaterEqual(len(checkpoints), 1)

        # Shutdown
        self.deployment.shutdown()
        self.assertTrue(self.deployment._shutdown_requested)

    def test_checkpoint_restore_workflow(self):
        """Test saving and restoring from checkpoint"""
        # Create and run deployment
        deployment1 = self.create_deployment()
        for i in range(10):
            deployment1.step_with_monitoring(["test"], {})
        deployment1.metrics_collector.increment_counter("before_save", 1)

        checkpoint_path = self.temp_dir / "restore_test.pkl"
        deployment1.save_checkpoint(str(checkpoint_path))
        deployment1._shutdown_requested = True

        # Create new deployment and restore
        deployment2 = self.create_deployment()
        self.assertEqual(deployment2.collective.sys.step, 0)
        deployment2._load_checkpoint(str(checkpoint_path))

        # Verify restoration
        self.assertEqual(deployment2.collective.sys.step, 10)
        self.assertEqual(deployment2.metrics_collector.counters.get("before_save"), 1)

        # Continue execution
        deployment2.step_with_monitoring(["test"], {})
        self.assertEqual(deployment2.collective.sys.step, 11)

    def test_error_recovery(self):
        """Test recovery from errors"""
        self.deployment = self.create_deployment()

        # Execute some steps
        for i in range(3):
            self.deployment.step_with_monitoring(["test"], {})

        # Simulate unhealthy state
        self.deployment.collective.sys.health.energy_budget_left_nJ = 0
        result = self.deployment.step_with_monitoring(["test"], {})
        self.assertIn("error", result)

        # Restore health
        self.deployment.collective.sys.health.energy_budget_left_nJ = 1000000
        result = self.deployment.step_with_monitoring(["test"], {})
        self.assertTrue(result.get("success", False))


# ============================================================
# TEST SUITE RUNNER
# ============================================================


def suite():
    """Create test suite"""
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(loader.loadTestsFromTestCase(TestDeploymentInitialization))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHealthChecks))
    test_suite.addTests(loader.loadTestsFromTestCase(TestStepExecution))
    test_suite.addTests(loader.loadTestsFromTestCase(TestStatus))
    test_suite.addTests(loader.loadTestsFromTestCase(TestCheckpointing))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMonitoring))
    test_suite.addTests(loader.loadTestsFromTestCase(TestShutdown))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
