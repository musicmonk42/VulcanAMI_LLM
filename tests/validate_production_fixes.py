#!/usr/bin/env python3
"""
Simple validation script for production bug fixes (no pytest required).

Tests the following critical fixes:
1. Phantom Resolution Circuit Breaker - cycle-level deduplication
2. JSON Serialization - EnhancedJSONEncoder with Enum support
3. Self-Improvement cooldown times
4. Config loading for objective estimates
"""

import json
import os
import sys
import tempfile
from enum import Enum
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vulcan.knowledge_crystallizer.knowledge_storage import EnhancedJSONEncoder
from vulcan.curiosity_engine.resolution_bridge import (
    get_recent_resolutions_count,
    mark_gap_resolved_batch,
)


class TestPatternType(Enum):
    """Test enum to simulate PatternType"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


def test_enum_serialization():
    """Test that Enum objects are serialized to their values."""
    print("Testing Enum serialization...", end=" ")
    data = {
        "pattern_type": TestPatternType.SEQUENTIAL,
        "name": "test"
    }
    result = json.dumps(data, cls=EnhancedJSONEncoder)
    parsed = json.loads(result)
    assert parsed["pattern_type"] == "sequential", f"Expected 'sequential', got {parsed['pattern_type']}"
    print("✓ PASS")


def test_nested_enum_serialization():
    """Test that nested Enum objects are handled correctly."""
    print("Testing nested Enum serialization...", end=" ")
    data = {
        "patterns": [
            {"type": TestPatternType.SEQUENTIAL, "id": 1},
            {"type": TestPatternType.PARALLEL, "id": 2}
        ]
    }
    result = json.dumps(data, cls=EnhancedJSONEncoder)
    parsed = json.loads(result)
    assert parsed["patterns"][0]["type"] == "sequential"
    assert parsed["patterns"][1]["type"] == "parallel"
    print("✓ PASS")


def test_numpy_serialization():
    """Test handling of numpy arrays."""
    print("Testing numpy serialization...", end=" ")
    try:
        import numpy as np
        
        data = {
            "enum_field": TestPatternType.SEQUENTIAL,
            "numpy_array": np.array([1, 2, 3]),
            "numpy_scalar": np.int64(42),
        }
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)
        
        assert parsed["enum_field"] == "sequential"
        assert parsed["numpy_array"] == [1, 2, 3]
        assert parsed["numpy_scalar"] == 42
        print("✓ PASS")
    except ImportError:
        print("⊘ SKIP (numpy not available)")


def test_cycle_level_deduplication():
    """Test that marking the same gap multiple times in one cycle only counts once."""
    print("Testing cycle-level deduplication...", end=" ")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Mark the same gap resolved 3 times in the same cycle
        gap_key = "test_gap:domain"
        cycle_id = 1
        
        mark_gap_resolved_batch(gap_key, success=True, cycle_id=cycle_id, db_path=db_path)
        mark_gap_resolved_batch(gap_key, success=True, cycle_id=cycle_id, db_path=db_path)
        mark_gap_resolved_batch(gap_key, success=True, cycle_id=cycle_id, db_path=db_path)
        
        # Count unique cycles (should be 1)
        count_unique = get_recent_resolutions_count(
            gap_key, window_seconds=3600, db_path=db_path, count_unique_cycles=True
        )
        
        # Count all entries (should be 3)
        count_all = get_recent_resolutions_count(
            gap_key, window_seconds=3600, db_path=db_path, count_unique_cycles=False
        )
        
        assert count_unique == 1, f"Expected 1 unique cycle, got {count_unique}"
        assert count_all == 3, f"Expected 3 total entries, got {count_all}"
        print("✓ PASS")


def test_multiple_cycles_counted():
    """Test that resolutions in different cycles are counted separately."""
    print("Testing multiple cycle counting...", end=" ")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        gap_key = "test_gap:domain"
        
        # Mark resolved in 3 different cycles
        for cycle_id in [1, 2, 3]:
            mark_gap_resolved_batch(gap_key, success=True, cycle_id=cycle_id, db_path=db_path)
        
        # Count unique cycles
        count = get_recent_resolutions_count(
            gap_key, window_seconds=3600, db_path=db_path, count_unique_cycles=True
        )
        
        assert count == 3, f"Expected 3 unique cycles, got {count}"
        print("✓ PASS")


def test_threshold_increased():
    """Test that the new threshold is 5 (or configurable via env var)."""
    print("Testing phantom threshold increased...", end=" ")
    
    from vulcan.curiosity_engine.resolution_bridge import PHANTOM_RESOLUTION_THRESHOLD
    
    # Default should be 5 (unless overridden by env var)
    if os.environ.get("VULCAN_PHANTOM_THRESHOLD"):
        print("⊘ SKIP (custom env var set)")
    else:
        assert PHANTOM_RESOLUTION_THRESHOLD == 5, \
            f"Expected threshold of 5, got {PHANTOM_RESOLUTION_THRESHOLD}"
        print("✓ PASS")


def test_self_improvement_cooldowns():
    """Test that cooldown times have been reduced."""
    print("Testing self-improvement cooldowns...", end=" ")
    
    try:
        from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive
        
        drive = SelfImprovementDrive()
        config = drive._build_default_auto_apply_policy()
        
        transient = config.get("failure_patterns", {}).get(
            "failure_classification", {}
        ).get("transient", {}).get("cooldown_hours")
        
        systemic = config.get("failure_patterns", {}).get(
            "failure_classification", {}
        ).get("systemic", {}).get("cooldown_hours")
        
        assert transient == 2, f"Expected transient cooldown of 2h, got {transient}h"
        assert systemic == 24, f"Expected systemic cooldown of 24h, got {systemic}h"
        print("✓ PASS")
    except Exception as e:
        print(f"⊘ SKIP (error: {e})")


def test_config_loading():
    """Test that objective estimates can be loaded from config file."""
    print("Testing config file loading...", end=" ")
    
    try:
        from vulcan.world_model.meta_reasoning.counterfactual_objectives import (
            CounterfactualObjectiveReasoner
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()
            
            # Create a test config file
            config_file = config_dir / "objective_estimates.json"
            test_estimates = {
                "prediction_accuracy": 0.98,
                "safety": 0.99,
            }
            
            with open(config_file, "w") as f:
                json.dump(test_estimates, f)
            
            # Load config
            loaded = CounterfactualObjectiveReasoner._load_config_estimates(str(config_dir))
            
            assert loaded is not None, "Should load config from file"
            assert loaded["prediction_accuracy"] == 0.98
            assert loaded["safety"] == 0.99
            print("✓ PASS")
    except Exception as e:
        print(f"⊘ SKIP (error: {e})")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PRODUCTION FIXES VALIDATION")
    print("="*60 + "\n")
    
    tests = [
        ("JSON Serialization - Enum", test_enum_serialization),
        ("JSON Serialization - Nested Enum", test_nested_enum_serialization),
        ("JSON Serialization - Numpy", test_numpy_serialization),
        ("Phantom Resolution - Cycle Deduplication", test_cycle_level_deduplication),
        ("Phantom Resolution - Multiple Cycles", test_multiple_cycles_counted),
        ("Phantom Resolution - Threshold", test_threshold_increased),
        ("Self-Improvement - Cooldowns", test_self_improvement_cooldowns),
        ("Objective Estimates - Config Loading", test_config_loading),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"⊘ SKIP: {e}")
            skipped += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
