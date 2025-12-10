#!/usr/bin/env python3
"""
Clear Python bytecode cache

This script removes all __pycache__ directories and .pyc files
to ensure you're running the latest code.

Usage:
    python scripts/clear_cache.py
"""

import shutil
import sys
from pathlib import Path


def main():
    """Clear all bytecode cache"""
    repo_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Clearing Python Bytecode Cache")
    print("=" * 60)
    print()
    
    cache_count = 0
    pyc_count = 0
    
    # Clear __pycache__ directories
    print("Removing __pycache__ directories...")
    for pycache_dir in repo_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            cache_count += 1
            print(f"  Removed: {pycache_dir.relative_to(repo_root)}")
        except Exception as e:
            print(f"  Warning: Could not remove {pycache_dir}: {e}")
    
    # Clear .pyc files
    print("\nRemoving .pyc files...")
    for pyc_file in repo_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_count += 1
            print(f"  Removed: {pyc_file.relative_to(repo_root)}")
        except Exception as e:
            print(f"  Warning: Could not remove {pyc_file}: {e}")
    
    print()
    print("=" * 60)
    print(f"✓ Cache cleared successfully")
    print(f"  - {cache_count} __pycache__ directories removed")
    print(f"  - {pyc_count} .pyc files removed")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Restart your Python process")
    print("  2. Re-import your modules")
    print("  3. Run: python scripts/check_self_healing.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
