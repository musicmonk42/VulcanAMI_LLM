#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Performance Comparison Script.

This script compares current performance test results against baseline values
and fails CI if regressions are detected. It generates comprehensive markdown
and JSON diff reports for analysis and debugging.

Architecture:
    The comparison uses a threshold-based approach where each metric type
    (latency, throughput) has its own comparison logic and configurable
    thresholds. Results are aggregated into a structured report that can
    be used by CI systems for pass/fail determination.

Key Features:
    - Threshold-based regression detection with configurable limits
    - Support for latency (lower-is-better) and throughput (higher-is-better) metrics
    - Concurrency-level specific comparisons
    - Markdown and JSON report generation
    - CI-friendly exit codes

Configuration:
    Environment Variables:
        PERF_MAX_P95_REGRESSION_PCT: Maximum p95 latency regression % (default: 25)
        PERF_MAX_RPS_REGRESSION_PCT: Maximum throughput regression % (default: 25)

Usage:
    Basic comparison::

        python scripts/compare_perf.py --results perf/results.json --baseline perf/baseline.json

    With custom thresholds::

        python scripts/compare_perf.py --p95-threshold 15 --rps-threshold 20

    Informational run (no CI failure)::

        python scripts/compare_perf.py --no-fail

Exit Codes:
    0: No regressions detected (or --no-fail specified)
    1: Regressions detected or errors occurred
    2: Invalid arguments or configuration

Example Output:
    ============================================================
    VULCAN Performance Comparison
    ============================================================
    ✓ Loaded results: perf/results.json
    ✓ Loaded baseline: perf/baseline.json

    Comparing results against baseline...
      ✓ latency_p95: 0.0523s (+4.2% vs baseline)
      ✓ throughput_qps: 18.52 qps (-2.1% vs baseline)
      ✓ concurrency_10_p95: 0.0891s (+8.3% vs baseline)

    Result: ✓ All 3 metrics within thresholds

Author:
    VULCAN-AGI Performance Engineering Team

Version:
    1.0.0

License:
    Proprietary - VULCAN AGI Project

See Also:
    - perf/baseline.json: Performance baseline definitions
    - tests/perf/test_perf_smoke.py: Performance test implementation
    - docs/perf.md: Performance testing documentation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================

__version__: Final[str] = "1.0.0"
__author__: Final[str] = "VULCAN-AGI Performance Engineering Team"

# Default threshold values (can be overridden via environment or CLI)
DEFAULT_P95_REGRESSION_PCT: Final[float] = float(
    os.environ.get("PERF_MAX_P95_REGRESSION_PCT", "25")
)
DEFAULT_RPS_REGRESSION_PCT: Final[float] = float(
    os.environ.get("PERF_MAX_RPS_REGRESSION_PCT", "25")
)

# Default file paths
DEFAULT_BASELINE_PATH: Final[str] = "perf/baseline.json"
DEFAULT_RESULTS_PATH: Final[str] = "perf/results.json"
DEFAULT_OUTPUT_PATH: Final[str] = "perf/comparison.md"

# Validation bounds
MIN_THRESHOLD_PCT: Final[float] = 0.0
MAX_THRESHOLD_PCT: Final[float] = 1000.0


# ============================================================
# ENUMERATIONS
# ============================================================

class MetricType(Enum):
    """
    Type of performance metric for comparison direction.

    LATENCY: Lower values are better (e.g., response time)
    THROUGHPUT: Higher values are better (e.g., requests/second)
    """
    LATENCY = auto()
    THROUGHPUT = auto()


class ComparisonStatus(Enum):
    """
    Status of a metric comparison.

    PASSED: Metric within acceptable threshold
    FAILED: Metric exceeded threshold (regression)
    SKIPPED: Metric comparison skipped (no baseline)
    ERROR: Comparison could not be performed
    """
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ERROR = auto()

    def __str__(self) -> str:
        """Return human-readable status."""
        return self.name


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class ComparisonError(Exception):
    """Base exception for comparison errors."""
    pass


class InvalidBaselineError(ComparisonError):
    """Raised when baseline data is invalid or corrupted."""
    pass


class InvalidResultsError(ComparisonError):
    """Raised when results data is invalid or corrupted."""
    pass


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class ComparisonResult:
    """
    Immutable result of comparing a single metric.

    Attributes:
        metric_name: Identifier for the metric being compared.
        metric_type: Type of metric (latency or throughput).
        baseline_value: Expected baseline value for the metric.
        current_value: Actual measured value from results.
        threshold_pct: Allowed deviation threshold as percentage.
        change_pct: Actual percentage change from baseline.
        status: Comparison status (passed, failed, skipped, error).
        message: Human-readable description of the result.

    Thread Safety:
        This class is immutable and thread-safe.
    """
    metric_name: str
    metric_type: MetricType
    baseline_value: float
    current_value: float
    threshold_pct: float
    change_pct: float
    status: ComparisonStatus
    message: str

    @property
    def passed(self) -> bool:
        """Check if comparison passed."""
        return self.status == ComparisonStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if comparison failed (regression detected)."""
        return self.status == ComparisonStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields.
        """
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.name,
            "baseline_value": round(self.baseline_value, 6),
            "current_value": round(self.current_value, 6),
            "threshold_pct": round(self.threshold_pct, 2),
            "change_pct": round(self.change_pct, 2),
            "status": self.status.name,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class ComparisonReport:
    """
    Complete comparison report aggregating all metric results.

    Attributes:
        timestamp: ISO8601 timestamp when report was generated.
        baseline_path: Path to baseline file used.
        results_path: Path to results file compared.
        baseline_version: Version of baseline schema.
        comparisons: List of individual metric comparisons.
        passed: Overall pass/fail status.
        summary: Human-readable summary of results.
        metadata: Additional metadata about the comparison.

    Thread Safety:
        Not thread-safe for modification. Use locking if concurrent
        access is needed.
    """
    timestamp: str
    baseline_path: str
    results_path: str
    baseline_version: str = "unknown"
    comparisons: List[ComparisonResult] = field(default_factory=list)
    passed: bool = True
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_comparison(self, result: ComparisonResult) -> None:
        """
        Add a comparison result to the report.

        Updates overall pass/fail status based on the result.

        Args:
            result: ComparisonResult to add.
        """
        self.comparisons.append(result)
        if result.failed:
            self.passed = False

    def get_passed_count(self) -> int:
        """Get count of passed comparisons."""
        return sum(1 for c in self.comparisons if c.passed)

    def get_failed_count(self) -> int:
        """Get count of failed comparisons."""
        return sum(1 for c in self.comparisons if c.failed)

    def get_skipped_count(self) -> int:
        """Get count of skipped comparisons."""
        return sum(1 for c in self.comparisons if c.status == ComparisonStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Complete dictionary representation.
        """
        return {
            "version": __version__,
            "timestamp": self.timestamp,
            "baseline_path": self.baseline_path,
            "results_path": self.results_path,
            "baseline_version": self.baseline_version,
            "overall_passed": self.passed,
            "counts": {
                "total": len(self.comparisons),
                "passed": self.get_passed_count(),
                "failed": self.get_failed_count(),
                "skipped": self.get_skipped_count(),
            },
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================
# COMPARISON LOGIC
# ============================================================

def load_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a JSON file with error handling.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
        PermissionError: If file is not readable.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"Loaded JSON from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied reading {file_path}: {e}")
        raise


def compare_metric(
    baseline_value: float,
    current_value: float,
    threshold_pct: float,
    metric_name: str,
    metric_type: MetricType,
) -> ComparisonResult:
    """
    Compare a metric value against baseline.

    For LATENCY metrics, regression = current > baseline by threshold.
    For THROUGHPUT metrics, regression = current < baseline by threshold.

    Args:
        baseline_value: Expected baseline value.
        current_value: Actual measured value.
        threshold_pct: Allowed deviation as percentage (always positive).
        metric_name: Identifier for the metric.
        metric_type: Type of metric for comparison direction.

    Returns:
        ComparisonResult with status and details.

    Example:
        >>> result = compare_metric(
        ...     baseline_value=0.050,
        ...     current_value=0.055,
        ...     threshold_pct=25.0,
        ...     metric_name="latency_p95",
        ...     metric_type=MetricType.LATENCY,
        ... )
        >>> print(result.status)
        ComparisonStatus.PASSED
    """
    # Handle edge case of zero/negative baseline
    if baseline_value <= 0:
        return ComparisonResult(
            metric_name=metric_name,
            metric_type=metric_type,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold_pct=threshold_pct,
            change_pct=0.0,
            status=ComparisonStatus.SKIPPED,
            message=f"⊘ {metric_name}: No valid baseline for comparison",
        )

    # Calculate percentage change
    change_pct = ((current_value - baseline_value) / baseline_value) * 100

    # Determine pass/fail based on metric type
    if metric_type == MetricType.LATENCY:
        # For latency: positive change (slower) is bad
        passed = change_pct <= threshold_pct
        unit = "s"
        direction_word = "slower" if change_pct > 0 else "faster"
    else:  # THROUGHPUT
        # For throughput: negative change (lower) is bad
        passed = change_pct >= -threshold_pct
        unit = "qps"
        direction_word = "higher" if change_pct > 0 else "lower"

    status = ComparisonStatus.PASSED if passed else ComparisonStatus.FAILED

    # Generate message
    if passed:
        message = (
            f"✓ {metric_name}: {current_value:.4f}{unit} "
            f"({change_pct:+.1f}% {direction_word})"
        )
    else:
        message = (
            f"✗ {metric_name}: {current_value:.4f}{unit} "
            f"({change_pct:+.1f}% exceeds ±{threshold_pct}% threshold)"
        )

    return ComparisonResult(
        metric_name=metric_name,
        metric_type=metric_type,
        baseline_value=baseline_value,
        current_value=current_value,
        threshold_pct=threshold_pct,
        change_pct=change_pct,
        status=status,
        message=message,
    )


def run_comparison(
    results: Dict[str, Any],
    baseline: Dict[str, Any],
    p95_threshold: float,
    rps_threshold: float,
) -> ComparisonReport:
    """
    Run comprehensive comparison of results against baseline.

    Compares all available metrics including:
    - Summary latency and throughput
    - Per-concurrency-level metrics

    Args:
        results: Loaded results data from test run.
        baseline: Loaded baseline data.
        p95_threshold: Threshold for p95 latency regression %.
        rps_threshold: Threshold for throughput regression %.

    Returns:
        ComparisonReport with all comparisons and summary.

    Raises:
        InvalidBaselineError: If baseline structure is invalid.
        InvalidResultsError: If results structure is invalid.
    """
    report = ComparisonReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        baseline_path=DEFAULT_BASELINE_PATH,
        results_path=DEFAULT_RESULTS_PATH,
        baseline_version=baseline.get("version", "unknown"),
        metadata={
            "p95_threshold_pct": p95_threshold,
            "rps_threshold_pct": rps_threshold,
            "comparison_engine_version": __version__,
        },
    )

    # Get baseline thresholds (may override defaults)
    baseline_thresholds = baseline.get("thresholds", {})
    p95_threshold = baseline_thresholds.get("max_p95_regression_pct", p95_threshold)
    rps_threshold = baseline_thresholds.get("max_rps_regression_pct", rps_threshold)

    # Update metadata with effective thresholds
    report.metadata["effective_p95_threshold_pct"] = p95_threshold
    report.metadata["effective_rps_threshold_pct"] = rps_threshold

    # Get benchmark baselines
    benchmarks = baseline.get("benchmarks", {})

    # Extract results summary if available
    summary = results.get("summary", {})

    # Compare summary latency metrics
    if "latency_p95" in summary:
        e2e = benchmarks.get("end_to_end", {}).get("pipeline", {})
        baseline_p95 = e2e.get("p95_latency_max_seconds", 1.0)

        result = compare_metric(
            baseline_value=baseline_p95,
            current_value=summary["latency_p95"],
            threshold_pct=p95_threshold,
            metric_name="latency_p95",
            metric_type=MetricType.LATENCY,
        )
        report.add_comparison(result)

    # Compare summary throughput if available
    if "throughput_qps" in summary:
        e2e = benchmarks.get("end_to_end", {}).get("pipeline", {})
        baseline_qps = e2e.get("min_throughput_qps", 2.0)

        result = compare_metric(
            baseline_value=baseline_qps,
            current_value=summary.get("throughput_qps", 0.0),
            threshold_pct=rps_threshold,
            metric_name="throughput_qps",
            metric_type=MetricType.THROUGHPUT,
        )
        report.add_comparison(result)

    # Compare concurrency-level results if available
    for test_result in results.get("results", []):
        if "concurrency" not in test_result:
            continue

        concurrency = test_result["concurrency"]

        # Get baseline for this concurrency level
        threadpool = benchmarks.get("concurrency", {}).get("threadpool", {})
        baseline_key = f"concurrency_{concurrency}"
        baseline_data = threadpool.get(baseline_key, {})

        if not baseline_data:
            logger.debug(f"No baseline for {baseline_key}, skipping")
            continue

        # Compare latency for this concurrency level
        if "latency_p95" in test_result:
            baseline_p95 = baseline_data.get("max_latency_p95_seconds", 5.0)

            result = compare_metric(
                baseline_value=baseline_p95,
                current_value=test_result["latency_p95"],
                threshold_pct=p95_threshold,
                metric_name=f"concurrency_{concurrency}_p95",
                metric_type=MetricType.LATENCY,
            )
            report.add_comparison(result)

        # Compare throughput for this concurrency level
        if "throughput_qps" in test_result:
            baseline_qps = baseline_data.get("min_throughput_qps", 5.0)

            result = compare_metric(
                baseline_value=baseline_qps,
                current_value=test_result["throughput_qps"],
                threshold_pct=rps_threshold,
                metric_name=f"concurrency_{concurrency}_throughput",
                metric_type=MetricType.THROUGHPUT,
            )
            report.add_comparison(result)

    # Generate summary
    total_count = len(report.comparisons)
    passed_count = report.get_passed_count()
    failed_count = report.get_failed_count()
    skipped_count = report.get_skipped_count()

    if total_count == 0:
        report.summary = "⊘ No metrics available for comparison"
    elif report.passed:
        report.summary = f"✓ All {passed_count} metrics within thresholds"
        if skipped_count > 0:
            report.summary += f" ({skipped_count} skipped)"
    else:
        report.summary = f"✗ {failed_count}/{total_count} metrics exceeded thresholds"

    return report


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_markdown_report(report: ComparisonReport) -> str:
    """
    Generate comprehensive markdown report from comparison results.

    Creates a well-formatted markdown document suitable for GitHub
    Actions summary or PR comments.

    Args:
        report: ComparisonReport to format.

    Returns:
        Markdown-formatted string.
    """
    lines: List[str] = [
        "# Performance Comparison Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Baseline Version:** {report.baseline_version}",
        f"**Status:** {'✓ PASSED' if report.passed else '✗ FAILED'}",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "### Metrics Overview",
        "",
        f"- **Total Comparisons:** {len(report.comparisons)}",
        f"- **Passed:** {report.get_passed_count()}",
        f"- **Failed:** {report.get_failed_count()}",
        f"- **Skipped:** {report.get_skipped_count()}",
        "",
        "## Detailed Results",
        "",
        "| Metric | Type | Baseline | Current | Change | Threshold | Status |",
        "|--------|------|----------|---------|--------|-----------|--------|",
    ]

    for comp in report.comparisons:
        status_icon = {
            ComparisonStatus.PASSED: "✓",
            ComparisonStatus.FAILED: "✗",
            ComparisonStatus.SKIPPED: "⊘",
            ComparisonStatus.ERROR: "⚠",
        }.get(comp.status, "?")

        metric_type = "Latency" if comp.metric_type == MetricType.LATENCY else "Throughput"

        lines.append(
            f"| {comp.metric_name} | "
            f"{metric_type} | "
            f"{comp.baseline_value:.4f} | "
            f"{comp.current_value:.4f} | "
            f"{comp.change_pct:+.1f}% | "
            f"±{comp.threshold_pct:.0f}% | "
            f"{status_icon} |"
        )

    # Add failed metrics section if any
    failed_comparisons = [c for c in report.comparisons if c.failed]
    if failed_comparisons:
        lines.extend([
            "",
            "## ⚠️ Regressions Detected",
            "",
            "The following metrics exceeded their thresholds:",
            "",
        ])
        for comp in failed_comparisons:
            lines.append(f"- **{comp.metric_name}:** {comp.message}")

    # Add metadata section
    if report.metadata:
        lines.extend([
            "",
            "## Configuration",
            "",
        ])
        for key, value in sorted(report.metadata.items()):
            lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def save_report(
    report: ComparisonReport,
    output_path: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Save comparison report in both markdown and JSON formats.

    Creates parent directories if needed.

    Args:
        report: ComparisonReport to save.
        output_path: Base path for output (used for markdown, .json appended for JSON).

    Returns:
        Tuple of (markdown_path, json_path).

    Raises:
        OSError: If files cannot be written.
    """
    md_path = Path(output_path)
    json_path = md_path.with_suffix(".json")

    # Ensure parent directory exists
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and save markdown
    markdown_content = generate_markdown_report(report)
    md_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Markdown report saved to: {md_path}")

    # Save JSON
    json_content = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    json_path.write_text(json_content, encoding="utf-8")
    logger.info(f"JSON report saved to: {json_path}")

    return md_path, json_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        Parsed arguments namespace.

    Raises:
        SystemExit: If arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Compare performance results against baseline for regression detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with defaults
  python scripts/compare_perf.py

  # Specify custom files
  python scripts/compare_perf.py --results perf/results.json --baseline perf/baseline.json

  # Use stricter thresholds
  python scripts/compare_perf.py --p95-threshold 10 --rps-threshold 15

  # Informational run (don't fail CI)
  python scripts/compare_perf.py --no-fail

Environment Variables:
  PERF_MAX_P95_REGRESSION_PCT  - Default p95 latency threshold (default: 25)
  PERF_MAX_RPS_REGRESSION_PCT  - Default throughput threshold (default: 25)
        """,
    )

    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        metavar="PATH",
        help=f"Path to results JSON file (default: {DEFAULT_RESULTS_PATH})",
    )

    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_PATH,
        metavar="PATH",
        help=f"Path to baseline JSON file (default: {DEFAULT_BASELINE_PATH})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        metavar="PATH",
        help=f"Path for markdown report output (default: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=DEFAULT_P95_REGRESSION_PCT,
        metavar="PCT",
        help=f"Maximum p95 latency regression %% (default: {DEFAULT_P95_REGRESSION_PCT})",
    )

    parser.add_argument(
        "--rps-threshold",
        type=float,
        default=DEFAULT_RPS_REGRESSION_PCT,
        metavar="PCT",
        help=f"Maximum throughput regression %% (default: {DEFAULT_RPS_REGRESSION_PCT})",
    )

    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Don't exit with failure code on regression (for informational runs)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Validate thresholds
    if args.p95_threshold < MIN_THRESHOLD_PCT or args.p95_threshold > MAX_THRESHOLD_PCT:
        parser.error(
            f"--p95-threshold must be between {MIN_THRESHOLD_PCT} and {MAX_THRESHOLD_PCT}"
        )

    if args.rps_threshold < MIN_THRESHOLD_PCT or args.rps_threshold > MAX_THRESHOLD_PCT:
        parser.error(
            f"--rps-threshold must be between {MIN_THRESHOLD_PCT} and {MAX_THRESHOLD_PCT}"
        )

    return args


def main() -> int:
    """
    Main entry point for performance comparison.

    Returns:
        Exit code:
        - 0: No regressions detected (or --no-fail specified)
        - 1: Regressions detected or errors occurred
    """
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("VULCAN Performance Comparison")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()

    # Load results file
    try:
        if not os.path.exists(args.results):
            print(f"⚠ Results file not found: {args.results}")
            print("  Run performance tests first to generate results.")
            print("  Example: pytest tests/perf/ -m perf")
            return 0  # Not a failure - just no results yet

        results = load_json_file(args.results)
        print(f"✓ Loaded results: {args.results}")

    except (json.JSONDecodeError, PermissionError) as e:
        print(f"✗ Error loading results: {e}")
        return 1

    # Load baseline file
    try:
        if not os.path.exists(args.baseline):
            print(f"⚠ Baseline file not found: {args.baseline}")
            print("  Create baseline first or use existing perf/baseline.json")
            return 0  # Not a failure - just no baseline yet

        baseline = load_json_file(args.baseline)
        print(f"✓ Loaded baseline: {args.baseline}")

    except (json.JSONDecodeError, PermissionError) as e:
        print(f"✗ Error loading baseline: {e}")
        return 1

    # Run comparison
    print("\nComparing results against baseline...")
    print(f"  Latency threshold: ±{args.p95_threshold}%")
    print(f"  Throughput threshold: ±{args.rps_threshold}%")
    print()

    try:
        report = run_comparison(
            results=results,
            baseline=baseline,
            p95_threshold=args.p95_threshold,
            rps_threshold=args.rps_threshold,
        )
    except (InvalidBaselineError, InvalidResultsError) as e:
        print(f"✗ Comparison error: {e}")
        return 1

    # Print results
    for comp in report.comparisons:
        print(f"  {comp.message}")

    print()
    print(f"Result: {report.summary}")

    # Save reports
    try:
        md_path, json_path = save_report(report, args.output)
        print()
        print(f"📄 Markdown report: {md_path}")
        print(f"📊 JSON report: {json_path}")
    except OSError as e:
        print(f"⚠ Warning: Failed to save reports: {e}")
        # Continue - don't fail just because we couldn't save reports

    # Determine exit code
    if args.no_fail:
        return 0

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
