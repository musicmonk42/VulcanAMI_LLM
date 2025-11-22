"""
Comprehensive Test Suite for Dynamic Architecture Controller
============================================================

Tests all major functionality including:
- Architecture modifications (add/remove heads, layers, connections)
- Snapshot management and rollback
- Validation and constraints
- Consensus approval
- Performance metrics
- State persistence
- Thread safety
- Error handling
"""

import sys
import unittest
import tempfile
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add the uploads directory to path
sys.path.insert(0, '/mnt/user-data/uploads')

from dynamic_architecture import (
    DynamicArchitecture,
    ArchChangeResult,
    Constraints,
    DynamicArchConfig,
    ChangeType,
    SnapshotPolicy,
    ValidationLevel,
    ArchitectureStats,
    ValidationResult,
    create_default_controller,
    create_strict_controller
)


class TestDynamicArchitectureBasics(unittest.TestCase):
    """Test basic initialization and configuration."""
    
    def test_initialization_no_model(self):
        """Test initialization without a model."""
        arch = DynamicArchitecture()
        self.assertIsNone(arch.model)
        self.assertIsInstance(arch._shadow_layers, list)
        self.assertEqual(len(arch._shadow_layers), 0)
    
    def test_initialization_with_model(self):
        """Test initialization with a mock model."""
        mock_model = Mock()
        mock_model.layers = [{'id': 'layer_0'}, {'id': 'layer_1'}]
        
        arch = DynamicArchitecture(model=mock_model)
        self.assertEqual(arch.model, mock_model)
    
    def test_initialization_with_constraints(self):
        """Test initialization with custom constraints."""
        constraints = Constraints(
            max_heads_per_layer=64,
            min_heads_per_layer=2,
            require_consensus=True
        )
        
        arch = DynamicArchitecture(constraints=constraints)
        self.assertEqual(arch.constraints.max_heads_per_layer, 64)
        self.assertEqual(arch.constraints.min_heads_per_layer, 2)
        self.assertTrue(arch.constraints.require_consensus)
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = DynamicArchConfig(
            max_snapshots=50,
            enable_auto_rollback=False,
            enable_validation=True
        )
        
        arch = DynamicArchitecture(config=config)
        self.assertEqual(arch.config.max_snapshots, 50)
        self.assertFalse(arch.config.enable_auto_rollback)
        self.assertTrue(arch.config.enable_validation)


class TestHeadManagement(unittest.TestCase):
    """Test attention head management operations."""
    
    def setUp(self):
        """Set up test architecture with shadow layers."""
        self.arch = DynamicArchitecture()
        # Initialize shadow layers
        self.arch._shadow_layers = [
            {
                'id': 'layer_0',
                'heads': [
                    {'id': 'head_0', 'd_k': 64, 'd_v': 64},
                    {'id': 'head_1', 'd_k': 64, 'd_v': 64}
                ]
            },
            {
                'id': 'layer_1',
                'heads': [
                    {'id': 'head_0', 'd_k': 64, 'd_v': 64}
                ]
            }
        ]
    
    def test_add_head_success(self):
        """Test adding a head to a layer."""
        result = self.arch.add_head(0, {'d_k': 128, 'd_v': 128})
        self.assertTrue(result)
        
        # Verify head was added
        heads = self.arch.list_heads(0)
        self.assertEqual(len(heads), 3)
        self.assertEqual(heads[-1]['d_k'], 128)
    
    def test_add_head_invalid_layer(self):
        """Test adding head to invalid layer index."""
        result = self.arch.add_head(99, {'d_k': 64})
        self.assertFalse(result)
    
    def test_add_head_exceeds_max(self):
        """Test adding head when max heads exceeded."""
        self.arch.constraints.max_heads_per_layer = 2
        result = self.arch.add_head(0, {'d_k': 64})
        self.assertFalse(result)
    
    def test_remove_head_success(self):
        """Test removing a head from a layer."""
        result = self.arch.remove_head(0, 0)
        self.assertTrue(result)
        
        # Verify head was removed
        heads = self.arch.list_heads(0)
        self.assertEqual(len(heads), 1)
    
    def test_remove_head_invalid_layer(self):
        """Test removing head from invalid layer."""
        result = self.arch.remove_head(99, 0)
        self.assertFalse(result)
    
    def test_remove_head_invalid_index(self):
        """Test removing head with invalid index."""
        result = self.arch.remove_head(0, 99)
        self.assertFalse(result)
    
    def test_remove_head_below_minimum(self):
        """Test removing head below minimum constraint."""
        self.arch.constraints.min_heads_per_layer = 2
        result = self.arch.remove_head(0, 0)
        self.assertFalse(result)
    
    def test_modify_head_success(self):
        """Test modifying head configuration."""
        result = self.arch.modify_head(0, 0, {'d_k': 256, 'd_v': 256})
        self.assertTrue(result)
        
        heads = self.arch.list_heads(0)
        self.assertEqual(heads[0]['d_k'], 256)
        self.assertEqual(heads[0]['d_v'], 256)
    
    def test_list_heads(self):
        """Test listing heads for a layer."""
        heads = self.arch.list_heads(0)
        self.assertEqual(len(heads), 2)
        self.assertIn('id', heads[0])


class TestLayerManagement(unittest.TestCase):
    """Test layer management operations."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'type': 'transformer'},
            {'id': 'layer_1', 'type': 'transformer'}
        ]
    
    def test_add_layer_success(self):
        """Test adding a new layer."""
        result = self.arch.add_layer(1, {'id': 'new_layer', 'type': 'transformer'})
        self.assertTrue(result)
        
        state = self.arch.get_state()
        self.assertEqual(len(state['layers']), 3)
    
    def test_add_layer_exceeds_max(self):
        """Test adding layer when max exceeded."""
        self.arch.constraints.max_layers = 2
        result = self.arch.add_layer(2, {'id': 'new_layer'})
        self.assertFalse(result)
    
    def test_remove_layer_success(self):
        """Test removing a layer."""
        result = self.arch.remove_layer(0)
        self.assertTrue(result)
        
        state = self.arch.get_state()
        self.assertEqual(len(state['layers']), 1)
    
    def test_remove_layer_below_minimum(self):
        """Test removing layer below minimum."""
        self.arch.constraints.min_layers = 2
        result = self.arch.remove_layer(0)
        self.assertFalse(result)


class TestConnectionManagement(unittest.TestCase):
    """Test connection/edge management."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._graph_edges = set()
    
    def test_add_connection_success(self):
        """Test adding a connection."""
        result = self.arch.add_connection('node_a', 'node_b', {'weight': 1.0})
        self.assertTrue(result)
        self.assertIn(('node_a', 'node_b'), self.arch._graph_edges)
    
    def test_add_connection_duplicate(self):
        """Test adding duplicate connection."""
        self.arch.add_connection('node_a', 'node_b')
        result = self.arch.add_connection('node_a', 'node_b')
        # Should still succeed (idempotent)
        self.assertTrue(result)
    
    def test_prune_connection_success(self):
        """Test pruning a connection."""
        self.arch.add_connection('node_a', 'node_b')
        result = self.arch.prune_connection('node_a', 'node_b')
        self.assertTrue(result)
        self.assertNotIn(('node_a', 'node_b'), self.arch._graph_edges)
    
    def test_prune_nonexistent_connection(self):
        """Test pruning non-existent connection."""
        result = self.arch.prune_connection('node_x', 'node_y')
        # Should fail gracefully
        self.assertFalse(result)


class TestSnapshotManagement(unittest.TestCase):
    """Test snapshot and rollback functionality."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'heads': [{'id': 'head_0'}]}
        ]
    
    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshots_before = len(self.arch._snapshots)
        
        # Trigger a change that creates a snapshot
        self.arch.add_head(0, {'d_k': 64})
        
        snapshots_after = len(self.arch._snapshots)
        self.assertGreater(snapshots_after, snapshots_before)
    
    def test_list_snapshots(self):
        """Test listing snapshots."""
        # Create some changes to generate snapshots
        self.arch.add_head(0, {'d_k': 64})
        self.arch.add_head(0, {'d_k': 128})
        
        snapshots = self.arch.list_snapshots()
        self.assertGreater(len(snapshots), 0)
        self.assertIn('snapshot_id', snapshots[0].__dict__)
        self.assertIn('timestamp', snapshots[0].__dict__)
    
    def test_rollback_to_snapshot(self):
        """Test rolling back to a snapshot."""
        # Get initial state
        initial_heads = len(self.arch.list_heads(0))
        
        # Make a change
        self.arch.add_head(0, {'d_k': 64})
        snapshots = self.arch.list_snapshots()
        
        if snapshots:
            # Rollback to first snapshot
            snapshot_id = snapshots[0].snapshot_id
            result = self.arch.rollback_to_snapshot(snapshot_id)
            self.assertTrue(result)
    
    def test_rollback_invalid_snapshot(self):
        """Test rollback with invalid snapshot ID."""
        result = self.arch.rollback_to_snapshot('invalid_snapshot_id')
        self.assertFalse(result)
    
    def test_snapshot_policy_enforcement(self):
        """Test snapshot retention policy."""
        self.arch.config.max_snapshots = 3
        
        # Create more snapshots than the limit
        for i in range(5):
            self.arch.add_head(0, {'d_k': 64 + i})
        
        # Should not exceed max
        self.assertLessEqual(len(self.arch._snapshots), 3)


class TestValidation(unittest.TestCase):
    """Test validation and constraint checking."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'heads': [{'id': 'head_0'}]}
        ]
    
    def test_validate_architecture_basic(self):
        """Test basic architecture validation."""
        result = self.arch.validate_architecture()
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.valid)
    
    def test_proposal_validation_success(self):
        """Test valid proposal validation."""
        proposal = {
            'type': 'add_head',
            'layer_idx': 0,
            'head_cfg': {'d_k': 64}
        }
        
        errors = self.arch._validate_proposal(proposal)
        self.assertEqual(len(errors), 0)
    
    def test_proposal_validation_missing_layer_idx(self):
        """Test proposal validation with missing layer_idx."""
        proposal = {
            'type': 'add_head',
            'head_cfg': {'d_k': 64}
        }
        
        errors = self.arch._validate_proposal(proposal)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('layer_idx' in err for err in errors))
    
    def test_proposal_validation_invalid_layer_idx(self):
        """Test proposal validation with invalid layer_idx."""
        proposal = {
            'type': 'add_head',
            'layer_idx': 999,
            'head_cfg': {'d_k': 64}
        }
        
        errors = self.arch._validate_proposal(proposal)
        self.assertGreater(len(errors), 0)


class TestApplyChange(unittest.TestCase):
    """Test the apply_change method with proposals."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'heads': [{'id': 'head_0'}]}
        ]
    
    def test_apply_change_add_head(self):
        """Test applying add_head proposal."""
        proposal = {
            'type': 'add_head',
            'layer_idx': 0,
            'head_cfg': {'d_k': 64}
        }
        
        result = self.arch.apply_change(proposal)
        self.assertIsInstance(result, ArchChangeResult)
        self.assertTrue(result.ok)
    
    def test_apply_change_modify_head(self):
        """Test applying modify_head proposal."""
        proposal = {
            'type': 'modify_head',
            'layer_idx': 0,
            'head_idx': 0,
            'new_cfg': {'d_k': 128}
        }
        
        result = self.arch.apply_change(proposal)
        self.assertTrue(result.ok)
    
    def test_apply_change_invalid_proposal(self):
        """Test applying invalid proposal."""
        proposal = {
            'type': 'add_head',
            # Missing layer_idx
        }
        
        result = self.arch.apply_change(proposal)
        self.assertFalse(result.ok)
        self.assertGreater(len(result.validation_errors), 0)
    
    def test_apply_change_with_consensus(self):
        """Test apply_change with consensus approval."""
        mock_consensus = Mock()
        mock_consensus.approve = Mock(return_value=True)
        
        self.arch.consensus = mock_consensus
        self.arch.config.enable_consensus = True
        
        proposal = {
            'type': 'add_head',
            'layer_idx': 0,
            'head_cfg': {'d_k': 64}
        }
        
        result = self.arch.apply_change(proposal)
        self.assertTrue(result.ok)
        mock_consensus.approve.assert_called_once()
    
    def test_apply_change_consensus_denied(self):
        """Test apply_change when consensus denies."""
        mock_consensus = Mock()
        mock_consensus.approve = Mock(return_value=False)
        
        self.arch.consensus = mock_consensus
        self.arch.config.enable_consensus = True
        self.arch.constraints.require_consensus = True
        
        proposal = {
            'type': 'add_head',
            'layer_idx': 0,
            'head_cfg': {'d_k': 64}
        }
        
        result = self.arch.apply_change(proposal)
        self.assertFalse(result.ok)


class TestObservabilityAndAudit(unittest.TestCase):
    """Test observability and audit logging."""
    
    def test_observability_recording(self):
        """Test that observability events are recorded."""
        mock_obs = Mock()
        mock_obs.record = Mock()
        
        arch = DynamicArchitecture(observability_manager=mock_obs)
        arch._shadow_layers = [{'id': 'layer_0', 'heads': []}]
        
        arch.add_head(0, {'d_k': 64})
        
        # Should have called observability
        self.assertGreater(mock_obs.record.call_count, 0)
    
    def test_audit_logging(self):
        """Test that audit events are logged."""
        mock_audit = Mock()
        mock_audit.append = Mock()
        
        arch = DynamicArchitecture(audit_log=mock_audit)
        arch._shadow_layers = [{'id': 'layer_0', 'heads': []}]
        
        arch.add_head(0, {'d_k': 64})
        
        # Should have logged to audit
        self.assertGreater(mock_audit.append.call_count, 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics tracking."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'heads': [{'id': 'head_0'}]}
        ]
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Make some changes
        self.arch.add_head(0, {'d_k': 64})
        self.arch.add_head(0, {'d_k': 128})
        
        metrics = self.arch.get_performance_metrics()
        
        self.assertIn('total_changes', metrics)
        self.assertIn('successful_changes', metrics)
        self.assertIn('avg_change_time', metrics)
        self.assertGreater(metrics['total_changes'], 0)
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        self.arch.add_head(0, {'d_k': 64})
        
        metrics_before = self.arch.get_performance_metrics()
        self.assertGreater(metrics_before['total_changes'], 0)
        
        self.arch.reset_metrics()
        
        metrics_after = self.arch.get_performance_metrics()
        self.assertEqual(metrics_after['total_changes'], 0)


class TestStatePersistence(unittest.TestCase):
    """Test state save/load functionality."""
    
    def setUp(self):
        """Set up test architecture."""
        self.arch = DynamicArchitecture()
        self.arch._shadow_layers = [
            {'id': 'layer_0', 'heads': [{'id': 'head_0', 'd_k': 64}]}
        ]
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_path = self.temp_file.name
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temp file."""
        Path(self.temp_path).unlink(missing_ok=True)
    
    def test_save_state(self):
        """Test saving architecture state."""
        self.arch.save_state(self.temp_path)
        
        # Verify file exists and is valid JSON
        self.assertTrue(Path(self.temp_path).exists())
        
        with open(self.temp_path, 'r') as f:
            state = json.load(f)
        
        self.assertIn('layers', state)
        self.assertEqual(len(state['layers']), 1)
    
    def test_load_state(self):
        """Test loading architecture state."""
        # Save state first
        self.arch.save_state(self.temp_path)
        
        # Create new architecture and load
        new_arch = DynamicArchitecture()
        new_arch.load_state(self.temp_path)
        
        # Verify state was loaded
        state = new_arch.get_state()
        self.assertEqual(len(state['layers']), 1)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of concurrent operations."""
    
    def test_concurrent_add_head(self):
        """Test concurrent head additions."""
        arch = DynamicArchitecture()
        arch._shadow_layers = [
            {'id': 'layer_0', 'heads': []}
        ]
        
        results = []
        
        def add_heads():
            for i in range(10):
                result = arch.add_head(0, {'d_k': 64})
                results.append(result)
        
        threads = [threading.Thread(target=add_heads) for _ in range(3)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All operations should have succeeded
        self.assertTrue(all(results))
    
    def test_concurrent_snapshots(self):
        """Test concurrent snapshot operations."""
        arch = DynamicArchitecture()
        arch._shadow_layers = [{'id': 'layer_0', 'heads': []}]
        
        def make_changes():
            for i in range(5):
                arch.add_head(0, {'d_k': 64 + i})
                time.sleep(0.01)
        
        threads = [threading.Thread(target=make_changes) for _ in range(2)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have created snapshots without errors
        snapshots = arch.list_snapshots()
        self.assertGreater(len(snapshots), 0)


class TestArchitectureStats(unittest.TestCase):
    """Test architecture statistics."""
    
    def test_get_stats(self):
        """Test getting architecture statistics."""
        arch = DynamicArchitecture()
        arch._shadow_layers = [
            {
                'id': 'layer_0',
                'type': 'transformer',
                'heads': [{'id': 'h0'}, {'id': 'h1'}]
            },
            {
                'id': 'layer_1',
                'type': 'transformer',
                'heads': [{'id': 'h0'}]
            }
        ]
        
        stats = arch.get_stats()
        
        self.assertIsInstance(stats, ArchitectureStats)
        self.assertEqual(stats.num_layers, 2)
        self.assertEqual(stats.num_heads, 3)
        self.assertGreater(stats.avg_heads_per_layer, 0)


class TestSnapshotDiff(unittest.TestCase):
    """Test snapshot comparison functionality."""
    
    def test_diff_snapshots(self):
        """Test diffing two snapshots."""
        arch = DynamicArchitecture()
        arch._shadow_layers = [{'id': 'layer_0', 'heads': []}]
        
        # Create first snapshot
        arch.add_head(0, {'d_k': 64})
        snapshots1 = arch.list_snapshots()
        
        # Create second snapshot
        arch.add_head(0, {'d_k': 128})
        snapshots2 = arch.list_snapshots()
        
        if len(snapshots2) >= 2:
            snap_id1 = snapshots2[0].snapshot_id
            snap_id2 = snapshots2[1].snapshot_id
            
            diff = arch.diff_snapshots(snap_id1, snap_id2)
            self.assertIsInstance(diff, dict)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_default_controller(self):
        """Test creating default controller."""
        ctrl = create_default_controller()
        self.assertIsInstance(ctrl, DynamicArchitecture)
        self.assertIsNotNone(ctrl.constraints)
        self.assertIsNotNone(ctrl.config)
    
    def test_create_strict_controller(self):
        """Test creating strict controller."""
        ctrl = create_strict_controller()
        self.assertIsInstance(ctrl, DynamicArchitecture)
        self.assertEqual(ctrl.constraints.max_heads_per_layer, 64)
        self.assertEqual(ctrl.constraints.min_heads_per_layer, 2)
        self.assertTrue(ctrl.constraints.enforce_dag)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDynamicArchitectureBasics,
        TestHeadManagement,
        TestLayerManagement,
        TestConnectionManagement,
        TestSnapshotManagement,
        TestValidation,
        TestApplyChange,
        TestObservabilityAndAudit,
        TestPerformanceMetrics,
        TestStatePersistence,
        TestThreadSafety,
        TestArchitectureStats,
        TestSnapshotDiff,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    sys.exit(0 if result.wasSuccessful() else 1)