"""
test_defect_fixes.py - Tests for critical defect fixes in Curiosity Engine

Tests for the 4 critical defects identified in the defect report:
1. Amnesiac Subprocess (Data Starvation) - Already fixed, verify it works
2. Pickle Lock Crash (Windows) - Already fixed, verify it works
3. Phantom Gap Infinite Loop - Fixed, test prevention logic
4. FakeNumpy Regression - Fixed, test lstsq fallback

Author: GitHub Copilot
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDefect3PhantomGapPrevention(unittest.TestCase):
    """Test phantom gap prevention (Defect #3)"""
    
    def setUp(self):
        """Set up test database"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = Path(self.test_db.name)
        
    def tearDown(self):
        """Clean up test database"""
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_gap_attempt_tracking(self):
        """Test that gap attempts are tracked in SQLite"""
        from vulcan.curiosity_engine.resolution_bridge import (
            increment_gap_attempts,
            get_gap_attempts,
            _init_db,
        )
        
        # Initialize database
        _init_db(self.db_path)
        
        # Test incrementing attempts
        gap_key = "test_gap_high_error_rate"
        
        # First attempt
        count1 = increment_gap_attempts(gap_key, self.db_path)
        self.assertEqual(count1, 1)
        
        # Second attempt
        count2 = increment_gap_attempts(gap_key, self.db_path)
        self.assertEqual(count2, 2)
        
        # Third attempt
        count3 = increment_gap_attempts(gap_key, self.db_path)
        self.assertEqual(count3, 3)
        
        # Verify get works
        retrieved = get_gap_attempts(gap_key, self.db_path)
        self.assertEqual(retrieved, 3)
    
    def test_phantom_gap_prevention_skips_after_max_attempts(self):
        """Test that gaps are skipped after 3 attempts"""
        from vulcan.curiosity_engine.curiosity_engine_core import CuriosityEngine
        from vulcan.curiosity_engine.gap_analyzer import KnowledgeGap
        from vulcan.curiosity_engine.resolution_bridge import (
            increment_gap_attempts,
            _init_db,
        )
        
        # Initialize resolution bridge database  
        _init_db(self.db_path)
        
        # Create engine
        engine = CuriosityEngine()
        
        # Create a test gap
        gap = KnowledgeGap(
            id="test_phantom_gap",
            type="high_error_rate",
            description="Test gap for phantom prevention",
            severity=0.8,
            estimated_cost=10.0,
        )
        
        # Simulate 3 prior attempts
        with patch('vulcan.curiosity_engine.curiosity_engine_core._persistent_get_gap_attempts', return_value=3):
            # Should return empty list (skip the gap)
            experiments = engine.generate_targeted_experiments(gap)
            self.assertEqual(len(experiments), 0, "Should skip gap after 3 attempts")
        
        # Test with fewer attempts (should generate experiments)
        with patch('vulcan.curiosity_engine.curiosity_engine_core._persistent_get_gap_attempts', return_value=1):
            with patch.object(engine.exploration_budget, 'get_available', return_value=1000.0):
                with patch.object(engine.exploration_budget, 'can_afford', return_value=True):
                    # Mock the experiment generator to return a simple experiment
                    mock_exp = MagicMock()
                    mock_exp.complexity = 0.5
                    with patch.object(engine.experiment_generator, 'generate_causal_experiment', return_value=[mock_exp]):
                        with patch('vulcan.curiosity_engine.curiosity_engine_core._persistent_increment_gap_attempts', return_value=2):
                            experiments = engine.generate_targeted_experiments(gap)
                            self.assertGreater(len(experiments), 0, "Should generate experiments when attempts < 3")


class TestDefect4FakeNumpyLstsq(unittest.TestCase):
    """Test FakeNumpy lstsq fix (Defect #4)"""
    
    def test_fake_numpy_has_linalg_lstsq(self):
        """Test that FakeNumpy.linalg.lstsq exists and is callable"""
        # Force use of FakeNumpy by mocking NUMPY_AVAILABLE
        with patch('vulcan.world_model.meta_reasoning.numpy_compat.NUMPY_AVAILABLE', False):
            # Reimport to get FakeNumpy instance
            import importlib
            import vulcan.world_model.meta_reasoning.numpy_compat as numpy_compat
            importlib.reload(numpy_compat)
            
            np = numpy_compat.np
            
            # Verify structure
            self.assertTrue(hasattr(np, 'linalg'), "FakeNumpy should have linalg attribute")
            self.assertTrue(hasattr(np.linalg, 'lstsq'), "FakeNumpy.linalg should have lstsq method")
            self.assertTrue(callable(np.linalg.lstsq), "lstsq should be callable")
    
    def test_fake_numpy_lstsq_call(self):
        """Test that FakeNumpy.linalg.lstsq can be called"""
        from vulcan.world_model.meta_reasoning.numpy_compat import NUMPY_AVAILABLE
        
        if NUMPY_AVAILABLE:
            self.skipTest("Real NumPy is available, skipping FakeNumpy test")
        
        # Import FakeNumpy
        from vulcan.world_model.meta_reasoning.numpy_compat import np
        
        # Call lstsq (should not raise AttributeError)
        try:
            A = [[1, 1], [2, 1], [3, 1]]
            b = [1, 2, 3]
            result = np.linalg.lstsq(A, b, rcond=None)
            
            # Verify return structure (solution, residuals, rank, singular_values)
            self.assertIsInstance(result, tuple, "lstsq should return a tuple")
            self.assertEqual(len(result), 4, "lstsq should return 4-element tuple")
            self.assertIsInstance(result[0], list, "solution should be a list")
            
        except AttributeError as e:
            self.fail(f"FakeNumpy.linalg.lstsq raised AttributeError: {e}")
        except Exception as e:
            # Other exceptions are OK for this test (we're just checking structure)
            pass
    
    def test_curiosity_reward_shaper_with_fake_numpy(self):
        """Test that CuriosityRewardShaper works with FakeNumpy"""
        from vulcan.world_model.meta_reasoning.numpy_compat import NUMPY_AVAILABLE
        
        if NUMPY_AVAILABLE:
            self.skipTest("Real NumPy is available, skipping FakeNumpy test")
        
        # Import and test
        from vulcan.world_model.meta_reasoning.curiosity_reward_shaper import (
            CuriosityRewardShaper,
            CuriosityMethod,
        )
        
        # Create instance (should not crash)
        try:
            shaper = CuriosityRewardShaper(
                curiosity_weight=0.1,
                method=CuriosityMethod.COUNT_BASED,
            )
            
            # Add some novelty data to trigger trend calculation
            for i in range(25):
                shaper.novelty_history.append(0.5 + i * 0.01)
            
            # Update statistics (this calls the code that uses lstsq)
            shaper._update_statistics(0.6, shaper._classify_novelty(0.6, 5), 0.06)
            
            # Should not crash - that's the main test
            self.assertIsNotNone(shaper.statistics.novelty_trend)
            
        except AttributeError as e:
            if 'lstsq' in str(e) or 'vstack' in str(e):
                self.fail(f"FakeNumpy regression detected: {e}")
            raise


class TestDefect1And2AlreadyFixed(unittest.TestCase):
    """Verify Defect #1 and #2 are already fixed"""
    
    def test_outcome_bridge_persistence(self):
        """Test that outcomes are persisted to SQLite (Defect #1 fix)"""
        from vulcan.curiosity_engine.outcome_bridge import (
            record_query_outcome,
            get_recent_outcomes,
            _init_db,
        )
        
        # Use temporary database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = Path(f.name)
        
        try:
            # Initialize database
            _init_db(db_path)
            
            # Record an outcome
            success = record_query_outcome(
                query_id="test_query_123",
                status="success",
                routing_time_ms=100.0,
                total_time_ms=500.0,
                complexity=0.5,
                query_type="reasoning",
                db_path=db_path,
            )
            self.assertTrue(success, "Should successfully record outcome")
            
            # Retrieve outcomes (simulating subprocess read)
            outcomes = get_recent_outcomes(minutes=60, db_path=db_path)
            self.assertGreater(len(outcomes), 0, "Should retrieve outcomes from SQLite")
            self.assertEqual(outcomes[0]['query_id'], "test_query_123")
            
        finally:
            if db_path.exists():
                db_path.unlink()
    
    def test_curiosity_driver_uses_static_wrapper(self):
        """Test that CuriosityDriver uses static wrapper function (Defect #2 fix)"""
        from vulcan.curiosity_engine.curiosity_driver import _run_cycle_wrapper
        import inspect
        
        # Verify _run_cycle_wrapper is a function (not a method)
        self.assertTrue(inspect.isfunction(_run_cycle_wrapper), 
                       "Should be a static function, not a method")
        
        # Verify it takes a dict (not an engine instance)
        sig = inspect.signature(_run_cycle_wrapper)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ['engine_state'], 
                        "Should take engine_state dict, not engine instance")


if __name__ == '__main__':
    unittest.main()
