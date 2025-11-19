#!/usr/bin/env python3
"""
Diagnostic script to verify self-healing/self-improvement setup

This script checks if the WorldModel self-healing system is properly configured
and provides actionable recommendations if issues are found.

Usage:
    python scripts/check_self_healing.py

Or from anywhere:
    python -m scripts.check_self_healing
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

def clear_bytecode_cache():
    """Clear all bytecode cache files"""
    print("Clearing bytecode cache...")
    cache_cleared = 0
    
    # Clear __pycache__ directories
    for pycache_dir in repo_root.rglob("__pycache__"):
        try:
            import shutil
            shutil.rmtree(pycache_dir)
            cache_cleared += 1
        except Exception as e:
            print(f"  Warning: Could not remove {pycache_dir}: {e}")
    
    # Clear .pyc files
    for pyc_file in repo_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            cache_cleared += 1
        except Exception as e:
            print(f"  Warning: Could not remove {pyc_file}: {e}")
    
    print(f"✓ Cleared {cache_cleared} cached files/directories\n")


def check_imports():
    """Check if required modules can be imported"""
    print("Checking imports...")
    issues = []
    
    try:
        from vulcan.world_model import world_model_core
        print("  ✓ vulcan.world_model.world_model_core imported")
    except ImportError as e:
        issues.append(f"Cannot import world_model_core: {e}")
        print(f"  ✗ Cannot import world_model_core: {e}")
        return issues
    
    try:
        from vulcan.world_model.world_model_core import WorldModel
        print("  ✓ WorldModel class imported")
    except ImportError as e:
        issues.append(f"Cannot import WorldModel: {e}")
        print(f"  ✗ Cannot import WorldModel: {e}")
        return issues
    
    try:
        from vulcan.world_model.meta_reasoning import SelfImprovementDrive
        print("  ✓ SelfImprovementDrive imported")
    except ImportError as e:
        issues.append(f"Cannot import SelfImprovementDrive: {e}")
        print(f"  ✗ Cannot import SelfImprovementDrive: {e}")
    
    print()
    return issues


def check_worldmodel_methods():
    """Check if WorldModel has required methods"""
    print("Checking WorldModel methods...")
    issues = []
    
    try:
        from vulcan.world_model.world_model_core import WorldModel
        
        required_methods = [
            '_handle_improvement_alert',
            '_check_improvement_approval',
            'start_autonomous_improvement',
            'stop_autonomous_improvement',
            'get_improvement_status',
            'report_error',
            'update_performance_metric'
        ]
        
        for method_name in required_methods:
            if hasattr(WorldModel, method_name):
                method = getattr(WorldModel, method_name)
                if callable(method):
                    print(f"  ✓ {method_name}() exists and is callable")
                else:
                    issues.append(f"{method_name} exists but is not callable")
                    print(f"  ✗ {method_name} exists but is not callable")
            else:
                issues.append(f"WorldModel missing method: {method_name}")
                print(f"  ✗ {method_name} is MISSING")
        
    except Exception as e:
        issues.append(f"Error checking methods: {e}")
        print(f"  ✗ Error: {e}")
    
    print()
    return issues


def run_module_diagnostics():
    """Run the built-in module diagnostics"""
    print("Running module diagnostics...\n")
    
    try:
        from vulcan.world_model.world_model_core import (
            print_diagnostics,
            print_self_healing_diagnostics
        )
        
        print_diagnostics()
        print_self_healing_diagnostics()
        
    except Exception as e:
        print(f"✗ Error running module diagnostics: {e}")
        return False
    
    return True


def main():
    """Main diagnostic routine"""
    print("=" * 70)
    print("VULCAN-AGI Self-Healing System Diagnostic Tool")
    print("=" * 70)
    print()
    
    # Step 1: Clear bytecode cache
    clear_bytecode_cache()
    
    # Step 2: Check imports
    import_issues = check_imports()
    if import_issues:
        print("=" * 70)
        print("✗ CRITICAL: Import issues detected")
        print("=" * 70)
        for issue in import_issues:
            print(f"  - {issue}")
        print("\nPlease resolve import issues before continuing.")
        print("You may need to install dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Step 3: Check WorldModel methods
    method_issues = check_worldmodel_methods()
    
    # Step 4: Run module diagnostics
    diagnostics_ok = run_module_diagnostics()
    
    # Summary
    print("=" * 70)
    if not method_issues and diagnostics_ok:
        print("✓ ALL CHECKS PASSED - Self-healing system is properly configured")
        print("=" * 70)
        print("\nYour system is ready for self-improvement/self-healing.")
        print("To enable it in your config, set:")
        print("  enable_self_improvement: true")
        print("Or set environment variable:")
        print("  export VULCAN_ENABLE_SELF_IMPROVEMENT=1")
        return 0
    else:
        print("✗ ISSUES DETECTED")
        print("=" * 70)
        
        if method_issues:
            print("\nMethod issues:")
            for issue in method_issues:
                print(f"  - {issue}")
        
        if not diagnostics_ok:
            print("\n  - Module diagnostics failed")
        
        print("\nRecommended actions:")
        print("  1. Ensure you're on the latest code:")
        print("     git pull origin main")
        print("  2. Clear cache again (this script already did it once):")
        print("     find . -type d -name '__pycache__' -exec rm -rf {} +")
        print("  3. Restart Python and re-run this script")
        print("  4. If issues persist, check for:")
        print("     - Merge conflicts")
        print("     - Modified files not committed")
        print("     - Import path issues")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
