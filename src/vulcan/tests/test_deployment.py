# ============================================================
# VULCAN-AGI Orchestrator - Deployment Tests 
# Avoids thread spawning by delaying imports and using simpler mocks
# ============================================================

import unittest
import sys
import time
import pickle
import json
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============================================================
# MOCK OBJECTS (Standalone - no vulcan imports)
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
        self.safety_policies = {'names_to_versions': {'default': '1.0'}}
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
        self.policies = {'default': '1.0'}
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
            'total_agents': 5,
            'state_distribution': {'idle': 3, 'working': 2},
            'pending_tasks': 0,
            'average_health_score': 0.9
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
        return {'counters': self.counters, 'gauges': self.gauges}

    def export_metrics(self):
        return {'counters': self.counters, 'gauges': self.gauges, 'timestamp': time.time()}

    def import_metrics(self, data):
        self.counters = data.get('counters', {})
        self.gauges = data.get('gauges', {})

    def shutdown(self):
        pass


class MockDependencies:
    """Mock dependencies"""
    def __init__(self):
        self.goal_system = Mock()
        self.goal_system.generate_plan = Mock(return_value={})
        self.goal_system.get_goal_status = Mock(return_value={'goals': []})
        self.governance = Mock()
        self.governance.enforce_policies = Mock(side_effect=lambda x: x)
        self.safety_validator = Mock()
        self.safety_validator.get_safety_report = Mock(return_value={'violations': []})
        self.metrics = MockMetricsCollector()

    def validate(self):
        # Return empty dict - no missing dependencies
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
            'action': {'type': 'test_action'},
            'success': True,
            'observation': 'test_observation',
            'reward': 0.5
        }

    def shutdown(self):
        pass


# ============================================================
# BASE TEST CLASS - Handles import with mocking
# ============================================================

class DeploymentTestBase(unittest.TestCase):
    """Base class that handles safe importing and deployment creation"""
    
    @classmethod
    def setUpClass(cls):
        """Import ProductionDeployment with thread-spawning components mocked"""
        # These patches prevent thread spawning during import
        cls._patches = [
            patch.dict('sys.modules', {
                'vulcan.safety.rollback_audit': MagicMock(),
                'vulcan.orchestrator.metrics': MagicMock(),
                'vulcan.safety.governance_alignment': MagicMock(),
                'vulcan.learning.parameter_history': MagicMock(),
            })
        ]
        for p in cls._patches:
            p.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop patches"""
        for p in cls._patches:
            try:
                p.stop()
            except:
                pass

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = MockConfig(checkpoint_dir=str(self.temp_dir))
        self.deployment = None

    def tearDown(self):
        """Clean up"""
        if self.deployment:
            self.deployment._shutdown_requested = True  # Prevent real shutdown
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_deployment(self):
        """Create a deployment with all heavy components mocked"""
        from vulcan.orchestrator.deployment import ProductionDeployment
        
        with patch.object(ProductionDeployment, 'initialize'):
            deployment = ProductionDeployment(self.config)
        
        # Assign mock components
        deployment.collective = MockCollective()
        deployment.metrics_collector = MockMetricsCollector()
        deployment.collective.deps.metrics = deployment.metrics_collector
        deployment.unified_runtime = None
        deployment.checkpoint_dir = self.temp_dir
        deployment._last_checkpointed_step = 0
        deployment._shutdown_requested = False
        deployment.start_time = time.time()
        
        return deployment


# ============================================================
# TEST: INITIALIZATION
# ============================================================

class TestDeploymentInitialization(DeploymentTestBase):
    """Test ProductionDeployment initialization"""

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
        from vulcan.orchestrator.deployment import ProductionDeployment
        
        with patch.object(ProductionDeployment, 'initialize'):
            deployment = ProductionDeployment(self.config)
        
        self.assertEqual(deployment.orchestrator_type, "parallel")
        self.deployment = deployment

    def test_custom_orchestrator_type(self):
        """Test custom orchestrator type"""
        from vulcan.orchestrator.deployment import ProductionDeployment
        
        with patch.object(ProductionDeployment, 'initialize'):
            deployment = ProductionDeployment(self.config, orchestrator_type="adaptive")
        
        self.assertEqual(deployment.orchestrator_type, "adaptive")
        self.deployment = deployment


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
        result = self.deployment.step_with_monitoring(["test"], {'goal': 'test'})
        self.assertTrue(result.get('success', False))

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
        self.assertIn('error', result)

    def test_step_updates_metrics(self):
        """Test that step updates metrics"""
        self.deployment = self.create_deployment()
        self.deployment.step_with_monitoring(["test"], {})
        self.assertIn('successful_steps', self.deployment.metrics_collector.counters)


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
        self.assertEqual(status['cid'], self.deployment.collective.sys.CID)

    def test_status_includes_step(self):
        """Test status includes step"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertEqual(status['step'], self.deployment.collective.sys.step)

    def test_status_includes_health(self):
        """Test status includes health"""
        self.deployment = self.create_deployment()
        status = self.deployment.get_status()
        self.assertIn('health', status)
        self.assertIn('energy_budget_left_nJ', status['health'])

    def test_status_includes_metrics(self):
        """Test status includes metrics"""
        self.deployment = self.create_deployment()
        self.deployment.metrics_collector.increment_counter("test_counter")
        status = self.deployment.get_status()
        self.assertIn('metrics', status)
        self.assertIn('test_counter', status['metrics']['counters'])


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

    def test_list_checkpoints(self):
        """Test listing checkpoints"""
        self.deployment = self.create_deployment()
        self.deployment.save_checkpoint()
        time.sleep(0.02)
        self.deployment.collective.sys.step = 1
        self.deployment.save_checkpoint()
        
        checkpoints = self.deployment.list_checkpoints()
        self.assertEqual(len(checkpoints), 2)


# ============================================================
# TEST: MONITORING
# ============================================================

class TestMonitoring(DeploymentTestBase):
    """Test monitoring functionality"""

    def test_update_monitoring_success(self):
        """Test monitoring updates on success"""
        self.deployment = self.create_deployment()
        result = {'success': True}
        self.deployment._update_monitoring(result)
        self.assertIn('successful_steps', self.deployment.metrics_collector.counters)

    def test_update_monitoring_failure(self):
        """Test monitoring updates on failure"""
        self.deployment = self.create_deployment()
        result = {'success': False}
        self.deployment._update_monitoring(result)
        self.assertIn('failed_steps', self.deployment.metrics_collector.counters)

    def test_update_monitoring_updates_gauges(self):
        """Test monitoring updates gauges"""
        self.deployment = self.create_deployment()
        result = {'success': True}
        self.deployment._update_monitoring(result)
        self.assertIn('energy_remaining_nJ', self.deployment.metrics_collector.gauges)


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
            result = self.deployment.step_with_monitoring(["test"], {'goal': 'test'})
            self.assertTrue(result.get('success', False))
        
        self.assertEqual(self.deployment.collective.sys.step, 5)
        
        # Get status
        status = self.deployment.get_status()
        self.assertEqual(status['step'], 5)
        
        # Save checkpoint
        success = self.deployment.save_checkpoint()
        self.assertTrue(success)
        
        # Shutdown
        self.deployment.shutdown()
        self.assertTrue(self.deployment._shutdown_requested)

    def test_checkpoint_restore_workflow(self):
        """Test saving and restoring from checkpoint"""
        # Create and run deployment
        deployment1 = self.create_deployment()
        for i in range(10):
            deployment1.step_with_monitoring(["test"], {})
        deployment1.metrics_collector.increment_counter('before_save', 1)
        
        checkpoint_path = self.temp_dir / "restore_test.pkl"
        deployment1.save_checkpoint(str(checkpoint_path))
        deployment1._shutdown_requested = True
        
        # Create new deployment and restore
        deployment2 = self.create_deployment()
        self.assertEqual(deployment2.collective.sys.step, 0)
        deployment2._load_checkpoint(str(checkpoint_path))
        
        self.assertEqual(deployment2.collective.sys.step, 10)
        self.assertEqual(deployment2.metrics_collector.counters.get('before_save'), 1)


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


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
