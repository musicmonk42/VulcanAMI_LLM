#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Installation Verification Script.

This module provides comprehensive verification of VULCAN package installation,
ensuring all critical components can be imported and are functional.

The script performs the following checks:
    1. Verifies 'import src' works correctly
    2. Validates all core VULCAN modules can be imported
    3. Checks critical component availability
    4. Reports detailed status for debugging

Usage:
    Command line:
        python scripts/verify_installation.py
        python scripts/verify_installation.py --verbose
        python scripts/verify_installation.py --json

    After pip installation:
        vulcan-verify
        vulcan-verify --verbose

Exit Codes:
    0: All checks passed
    1: One or more checks failed
    2: Invalid arguments or runtime error

Example:
    >>> from scripts.verify_installation import verify_installation
    >>> results = verify_installation()
    >>> print(f"Passed: {results['passed']}, Failed: {results['failed']}")

Author: VULCAN-AGI Team
License: MIT
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ImportResult:
    """Result of an import test."""

    module_path: str
    description: str
    success: bool
    error_message: Optional[str] = None
    import_time_ms: float = 0.0


@dataclass
class VerificationReport:
    """Complete verification report."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: List[ImportResult] = field(default_factory=list)
    overall_success: bool = False
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "overall_success": self.overall_success,
            "total_time_ms": self.total_time_ms,
            "results": [
                {
                    "module": r.module_path,
                    "description": r.description,
                    "success": r.success,
                    "error": r.error_message,
                    "time_ms": r.import_time_ms,
                }
                for r in self.results
            ],
        }


# =============================================================================
# Test Configuration
# =============================================================================

# Module test cases: (module_path, description, is_critical)
# Critical modules cause verification failure if they can't be imported
TEST_CASES: List[Tuple[str, str, bool]] = [
    # Core src module - CRITICAL
    ("src", "Base src module", True),
    # VULCAN core package - CRITICAL
    ("src.vulcan", "VULCAN core package", True),
    # Orchestrator components - CRITICAL for stress tests
    ("src.vulcan.orchestrator", "Orchestrator module", True),
    ("src.vulcan.orchestrator.agent_pool", "Agent Pool", True),
    # Reasoning components - CRITICAL for stress tests
    ("src.vulcan.reasoning", "Reasoning module", True),
    ("src.vulcan.reasoning.selection", "Selection module", False),
    ("src.vulcan.reasoning.selection.tool_selector", "Tool Selector", True),
    # LLM components - CRITICAL for stress tests
    ("src.vulcan.llm", "LLM module", True),
    ("src.vulcan.llm.hybrid_executor", "Hybrid LLM Executor", True),
    # Safety and governance
    ("src.vulcan.safety", "Safety module", False),
    ("src.vulcan.safety.safety_validator", "Safety Validator", False),
    # Memory and routing
    ("src.vulcan.memory", "Memory module", False),
    ("src.vulcan.routing", "Routing module", False),
    # Additional components
    ("src.vulcan.alignment", "Alignment module", False),
    ("src.vulcan.governance", "Governance module", False),
]


# =============================================================================
# Core Functions
# =============================================================================


def test_import(module_path: str, description: str) -> ImportResult:
    """
    Test if a module can be imported.

    Args:
        module_path: Dotted path to the module (e.g., "src.vulcan").
        description: Human-readable description of what's being tested.

    Returns:
        ImportResult: Object containing test results and timing.
    """
    start_time = time.perf_counter()

    try:
        importlib.import_module(module_path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ImportResult(
            module_path=module_path,
            description=description,
            success=True,
            import_time_ms=elapsed_ms,
        )
    except ImportError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ImportResult(
            module_path=module_path,
            description=description,
            success=False,
            error_message=str(e),
            import_time_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ImportResult(
            module_path=module_path,
            description=description,
            success=False,
            error_message=f"Unexpected error: {type(e).__name__}: {e}",
            import_time_ms=elapsed_ms,
        )


def verify_installation(
    verbose: bool = False,
    critical_only: bool = False,
) -> VerificationReport:
    """
    Run all verification checks and return a report.

    Args:
        verbose: If True, print detailed progress information.
        critical_only: If True, only test critical modules.

    Returns:
        VerificationReport: Complete report of all test results.
    """
    start_time = time.perf_counter()
    report = VerificationReport()

    # Filter test cases if needed
    test_cases = TEST_CASES
    if critical_only:
        test_cases = [(m, d, c) for m, d, c in TEST_CASES if c]

    report.total_tests = len(test_cases)

    for module_path, description, is_critical in test_cases:
        result = test_import(module_path, description)
        report.results.append(result)

        if result.success:
            report.passed += 1
            if verbose:
                logger.info(f"✓ {description} ({result.import_time_ms:.1f}ms)")
        else:
            report.failed += 1
            if verbose:
                logger.error(f"✗ {description}")
                if result.error_message:
                    logger.error(f"  Error: {result.error_message}")

    report.total_time_ms = (time.perf_counter() - start_time) * 1000

    # Determine overall success - all critical modules must pass
    critical_failures = [
        r
        for (m, d, c), r in zip(test_cases, report.results)
        if c and not r.success
    ]
    report.overall_success = len(critical_failures) == 0

    return report


def print_report(report: VerificationReport, verbose: bool = False) -> None:
    """
    Print a formatted verification report to stdout.

    Args:
        report: The verification report to print.
        verbose: If True, print detailed results for each test.
    """
    print()
    print("=" * 60)
    print("VULCAN Installation Verification Report")
    print("=" * 60)
    print()
    print(f"Timestamp:      {report.timestamp}")
    print(f"Python Version: {report.python_version}")
    print(f"Total Time:     {report.total_time_ms:.1f}ms")
    print()

    if verbose:
        print("-" * 60)
        print("Test Results:")
        print("-" * 60)
        for result in report.results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.description}")
            if not result.success and result.error_message:
                # Truncate long error messages
                error = result.error_message
                if len(error) > 70:
                    error = error[:67] + "..."
                print(f"    Error: {error}")
        print()

    print("-" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed:      {report.passed}")
    print(f"Failed:      {report.failed}")
    print()

    if report.overall_success:
        print("✓ VERIFICATION PASSED")
        print("  All critical VULCAN components are available.")
        print("  The system is ready for testing.")
    else:
        print("✗ VERIFICATION FAILED")
        print("  One or more critical components could not be imported.")
        print()
        print("  To fix this issue:")
        print("    1. Ensure you're in the repository root directory")
        print("    2. Run: pip install -e .")
        print("    3. Run this verification script again")

    print("=" * 60)
    print()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Verify VULCAN installation and component availability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_installation.py
  python scripts/verify_installation.py --verbose
  python scripts/verify_installation.py --json
  python scripts/verify_installation.py --critical-only

Exit Codes:
  0: All critical checks passed
  1: One or more critical checks failed
  2: Invalid arguments or runtime error
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed progress and results",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )

    parser.add_argument(
        "-c",
        "--critical-only",
        action="store_true",
        help="Only test critical modules (faster)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output, only return exit code",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the verification script.

    Returns:
        int: Exit code (0 for success, 1 for failure, 2 for error).
    """
    try:
        args = parse_args()

        # Run verification
        report = verify_installation(
            verbose=args.verbose and not args.quiet and not args.json,
            critical_only=args.critical_only,
        )

        # Output results
        if not args.quiet:
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print_report(report, verbose=args.verbose)

        # Return appropriate exit code
        return 0 if report.overall_success else 1

    except KeyboardInterrupt:
        logger.info("\nVerification cancelled by user")
        return 2
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
