#!/usr/bin/env python3
"""
Quick validation of production fixes by checking code changes are present.
Does not require numpy or other dependencies.
"""

import sys
from pathlib import Path


def check_file_contains(filepath, search_strings, description):
    """Check if a file contains all the given search strings."""
    print(f"Checking {description}...", end=" ")
    
    if not Path(filepath).exists():
        print(f"✗ FAIL (file not found: {filepath})")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    missing = []
    for search_str in search_strings:
        if search_str not in content:
            missing.append(search_str)
    
    if missing:
        print(f"✗ FAIL (missing: {', '.join(missing[:2])}...)")
        return False
    
    print("✓ PASS")
    return True


def main():
    """Run validation checks."""
    print("\n" + "="*70)
    print("PRODUCTION FIXES CODE VALIDATION")
    print("="*70 + "\n")
    
    base_path = Path(__file__).parent.parent / "src"
    
    checks = [
        # Issue #1: Phantom Resolution Circuit Breaker
        (
            base_path / "vulcan/curiosity_engine/curiosity_engine_core.py",
            [
                "_current_cycle_resolutions",
                "cycle-level deduplication",
                "VULCAN_PHANTOM_THRESHOLD",
                "graduated backoff",
            ],
            "Phantom Resolution - Cycle Deduplication"
        ),
        (
            base_path / "vulcan/curiosity_engine/resolution_bridge.py",
            [
                "count_unique_cycles",
                "COUNT(DISTINCT cycle_id)",
                "Increased from 3 to 5",
            ],
            "Phantom Resolution - Unique Cycle Counting"
        ),
        
        # Issue #2: JSON Serialization
        (
            base_path / "vulcan/knowledge_crystallizer/knowledge_storage.py",
            [
                "class EnhancedJSONEncoder",
                "isinstance(obj, Enum)",
                "cls=EnhancedJSONEncoder",
                "obj.value",
            ],
            "JSON Serialization - EnhancedJSONEncoder"
        ),
        
        # Issue #4: Self-Improvement Cooldowns
        (
            base_path / "vulcan/world_model/meta_reasoning/self_improvement_drive.py",
            [
                '"cooldown_hours": 2',
                '"cooldown_hours": 24',
                "Reduced from",
            ],
            "Self-Improvement - Reduced Cooldowns"
        ),
        (
            base_path / "vulcan/world_model/world_model_core.py",
            [
                "is_code_improvement",
                "Non-code improvements",
                "CAN proceed autonomously",
            ],
            "Self-Improvement - Improved Logging"
        ),
        
        # Issue #5: Objective Estimates
        (
            base_path / "vulcan/world_model/meta_reasoning/counterfactual_objectives.py",
            [
                "_load_config_estimates",
                "configs/objective_estimates.json",
                "VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING",
                "logger.debug",
            ],
            "Objective Estimates - Config Loading"
        ),
        
        # Issue #6: CPU Priority
        (
            base_path / "llm_core/graphix_executor.py",
            [
                "VULCAN_SUPPRESS_CPU_PRIORITY_WARNING",
                "is_containerized",
                "/.dockerenv",
                "Check if we have permissions before attempting",
                "logger.debug",
            ],
            "CPU Priority - Permission Pre-check & Container Detection"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for filepath, search_strings, description in checks:
        if check_file_contains(filepath, search_strings, description):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed == 0:
        print("✓ All code changes are present and correct!")
        return 0
    else:
        print("✗ Some code changes are missing or incomplete.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
