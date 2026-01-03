#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Dependency Verification Tests.

This module verifies that expected dependencies are available for performance
tests. It ensures the test suite doesn't silently degrade when optional
dependencies are missing.

Usage:
    pytest tests/perf/test_deps.py -v
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List, Tuple

import pytest

from .conftest import PERF_LITE_DEPS, PERF_FULL_DEPS, get_available_deps


# ============================================================
# DEPENDENCY MAPPINGS
# ============================================================

# Map module names to their pip package names
MODULE_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "faiss": "faiss-cpu",
    "sentence_transformers": "sentence-transformers",
}


def get_package_name(module_name: str) -> str:
    """Get pip package name for a module."""
    return MODULE_TO_PACKAGE.get(module_name, module_name)


# ============================================================
# DEPENDENCY TESTS
# ============================================================

@pytest.mark.perf
class TestDependencies:
    """
    Tests for verifying perf test dependencies.
    """
    
    def test_psutil_available(self):
        """
        Test that psutil is available for memory tracking.
        
        psutil is critical for RSS memory tracking in boundedness tests.
        """
        try:
            import psutil
            
            # Verify basic functionality
            process = psutil.Process()
            mem_info = process.memory_info()
            assert hasattr(mem_info, "rss"), "psutil memory_info missing RSS"
            
            print(f"✓ psutil {psutil.__version__} available")
            print(f"  Current RSS: {mem_info.rss / 1024 / 1024:.2f} MB")
            
        except ImportError:
            pytest.fail(
                "psutil is required for performance tests. "
                "Install with: pip install psutil"
            )
    
    def test_perf_lite_deps(self, available_deps: Dict[str, bool]):
        """
        Test that perf-lite dependencies are available.
        
        These are the minimal dependencies for CI performance tests.
        """
        missing = []
        available = []
        
        for dep in PERF_LITE_DEPS:
            if available_deps.get(dep, False):
                available.append(dep)
            else:
                missing.append(dep)
        
        print(f"\nPerf-lite dependencies ({len(PERF_LITE_DEPS)} total):")
        for dep in available:
            print(f"  ✓ {dep}")
        for dep in missing:
            print(f"  ✗ {dep} (install: pip install {get_package_name(dep)})")
        
        # For CI, we require core deps but allow optional ones to be missing
        critical_deps = {"psutil"}
        critical_missing = critical_deps & set(missing)
        
        if critical_missing:
            pytest.fail(
                f"Critical perf-lite dependencies missing: {critical_missing}"
            )
        
        if missing:
            pytest.skip(
                f"Optional perf-lite dependencies missing: {missing}. "
                f"Install with: pip install -e '.[perf-lite]'"
            )
    
    def test_perf_full_deps(self, available_deps: Dict[str, bool]):
        """
        Test perf-full dependencies (for nightly/extended tests).
        
        These include heavy ML dependencies like torch.
        """
        missing = []
        available = []
        
        for dep in PERF_FULL_DEPS:
            if available_deps.get(dep, False):
                available.append(dep)
            else:
                missing.append(dep)
        
        heavy_deps = {"torch", "sentence_transformers", "faiss"}
        
        print(f"\nPerf-full dependencies ({len(PERF_FULL_DEPS)} total):")
        for dep in available:
            marker = "🔥" if dep in heavy_deps else "✓"
            print(f"  {marker} {dep}")
        for dep in missing:
            marker = "⚡" if dep in heavy_deps else "✗"
            print(f"  {marker} {dep} (install: pip install {get_package_name(dep)})")
        
        # perf-full is optional, just report status
        if missing:
            missing_heavy = heavy_deps & set(missing)
            missing_light = set(missing) - heavy_deps
            
            msg_parts = []
            if missing_heavy:
                msg_parts.append(f"Heavy deps: {missing_heavy}")
            if missing_light:
                msg_parts.append(f"Light deps: {missing_light}")
            
            print(f"\n  Note: {'; '.join(msg_parts)}")
            print(f"  Install all with: pip install -e '.[perf-full]'")
    
    def test_dep_versions(self):
        """
        Report versions of available dependencies.
        """
        deps_to_check = [
            ("psutil", "psutil"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("sklearn", "scikit-learn"),
            ("torch", "torch"),
        ]
        
        print("\nDependency versions:")
        for module_name, display_name in deps_to_check:
            try:
                mod = importlib.import_module(module_name)
                version = getattr(mod, "__version__", "unknown")
                print(f"  {display_name}: {version}")
            except ImportError:
                print(f"  {display_name}: not installed")


# ============================================================
# REASONER AVAILABILITY TESTS
# ============================================================

@pytest.mark.perf
class TestReasonerAvailability:
    """
    Tests for VULCAN reasoner component availability.
    
    Ensures tests don't silently fall back to mocks without warning.
    """
    
    def test_graphix_executor_available(self):
        """
        Test GraphixExecutor availability.
        """
        try:
            from src.vulcan.reasoning.graphix_executor import GraphixExecutor
            print("✓ GraphixExecutor available")
        except ImportError as e:
            print(f"⚠ GraphixExecutor not available: {e}")
            print("  Tests will use mock reasoner")
    
    def test_agent_pool_available(self):
        """
        Test AgentPoolManager availability.
        """
        try:
            from src.vulcan.orchestrator.agent_pool import AgentPoolManager
            print("✓ AgentPoolManager available")
        except ImportError as e:
            print(f"⚠ AgentPoolManager not available: {e}")
            print("  Concurrency tests will use mock components")
    
    def test_world_model_available(self):
        """
        Test WorldModel availability.
        """
        try:
            from src.vulcan.world_model.world_model import WorldModel
            print("✓ WorldModel available")
        except ImportError as e:
            print(f"⚠ WorldModel not available: {e}")
    
    def test_vulcan_package_installed(self):
        """
        Test that vulcan-agi package is installed.
        """
        try:
            import src.vulcan
            print("✓ src.vulcan package accessible")
            
            # Check for key submodules
            submodules = [
                "src.vulcan.reasoning",
                "src.vulcan.orchestrator",
                "src.vulcan.llm",
            ]
            
            for submod in submodules:
                try:
                    importlib.import_module(submod)
                    print(f"  ✓ {submod}")
                except ImportError as e:
                    print(f"  ⚠ {submod}: {e}")
                    
        except ImportError as e:
            print(f"⚠ VULCAN package not found: {e}")
            print("  Install with: pip install -e .")


# ============================================================
# SUMMARY FIXTURE
# ============================================================

@pytest.fixture(scope="module", autouse=True)
def print_dep_summary(available_deps):
    """
    Print dependency summary at end of module.
    """
    yield
    
    total = len(available_deps)
    available = sum(1 for v in available_deps.values() if v)
    
    print("\n" + "=" * 60)
    print("DEPENDENCY SUMMARY")
    print("=" * 60)
    print(f"Available: {available}/{total}")
    
    if available < total:
        missing = [k for k, v in available_deps.items() if not v]
        print(f"Missing: {', '.join(missing)}")
    
    print("=" * 60)
