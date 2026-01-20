#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check test that runs first (test_000_) to ensure basic imports work.
This catches import-time issues before pytest-xdist workers spawn.
"""
import os
import sys


def test_import_src():
    """Test that src can be imported without hanging."""
    import src

    assert src is not None


def test_import_src_vulcan():
    """Test that src.vulcan can be imported without hanging."""
    from src import vulcan

    assert vulcan is not None


def test_no_blocking_imports():
    """Ensure imports don't create blocking background threads."""
    import threading

    # Count threads before import
    initial_thread_count = threading.active_count()

    # Import a potentially problematic module
    from src.vulcan import orchestrator

    # Count threads after import
    final_thread_count = threading.active_count()

    # Allow some thread creation but not excessive
    # More than 10 new threads suggests background services started
    assert (
        final_thread_count - initial_thread_count < 10
    ), f"Import created {final_thread_count - initial_thread_count} threads, suggesting background services started"


def test_ci_mode_detected():
    """Verify CI mode is properly detected to disable problematic features."""
    assert (
        os.environ.get("CI") == "true"
        or os.environ.get("GITHUB_ACTIONS") is not None
        or os.environ.get("PYTEST_RUNNING") == "1"
    ), "CI mode should be detected in GitHub Actions"
