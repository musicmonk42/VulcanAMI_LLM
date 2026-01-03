#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VULCAN Performance Comparison Script.

Compares current performance test results against baseline and fails CI if
regressions are detected. Generates a markdown diff report.

Usage:
    python scripts/compare_perf.py --results perf/results.json --baseline perf/baseline.json

Environment Variables:
    PERF_MAX_P95_REGRESSION_PCT: Max allowed p95 latency regression (default: 25)
    PERF_MAX_RPS_REGRESSION_PCT: Max allowed throughput regression (default: 25)

Exit Codes:
    0: No regressions detected
    1: Regressions detected or errors occurred
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_P95_REGRESSION_PCT = float(os.environ.get("PERF_MAX_P95_REGRESSION_PCT", "25"))
DEFAULT_RPS_REGRESSION_PCT = float(os.environ.get("PERF_MAX_RPS_REGRESSION_PCT", "25"))
DEFAULT_BASELINE_PATH = "perf/baseline.json"
DEFAULT_RESULTS_PATH = "perf/results.json"
DEFAULT_OUTPUT_PATH = "perf/comparison.md"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ComparisonResult:
    """Result of comparing a single metric."""
    metric_name: str
    baseline_value: float
    current_value: float
    threshold_pct: float
    change_pct: float
    passed: bool
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "baseline": self.baseline_value,
            "current": self.current_value,
            "change_pct": round(self.change_pct, 2),
            "threshold_pct": self.threshold_pct,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class ComparisonReport:
    """Full comparison report."""
    timestamp: str
    baseline_path: str
    results_path: str
    comparisons: List[ComparisonResult] = field(default_factory=list)
    passed: bool = True
    summary: str = ""
    
    def add_comparison(self, result: ComparisonResult) -> None:
        self.comparisons.append(result)
        if not result.passed:
            self.passed = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "baseline_path": self.baseline_path,
            "results_path": self.results_path,
            "overall_passed": self.passed,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.summary,
        }


# ============================================================
# COMPARISON LOGIC
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_latency(
    baseline_value: float,
    current_value: float,
    threshold_pct: float,
    metric_name: str,
) -> ComparisonResult:
    """
    Compare latency metrics (lower is better).
    
    Regression = current > baseline by more than threshold_pct.
    """
    if baseline_value <= 0:
        return ComparisonResult(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold_pct=threshold_pct,
            change_pct=0.0,
            passed=True,
            message="No baseline for comparison",
        )
    
    change_pct = ((current_value - baseline_value) / baseline_value) * 100
    passed = change_pct <= threshold_pct
    
    if passed:
        message = f"✓ {metric_name}: {current_value:.4f}s ({change_pct:+.1f}% vs baseline)"
    else:
        message = f"✗ {metric_name}: {current_value:.4f}s ({change_pct:+.1f}% > {threshold_pct}% threshold)"
    
    return ComparisonResult(
        metric_name=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        threshold_pct=threshold_pct,
        change_pct=change_pct,
        passed=passed,
        message=message,
    )


def compare_throughput(
    baseline_value: float,
    current_value: float,
    threshold_pct: float,
    metric_name: str,
) -> ComparisonResult:
    """
    Compare throughput metrics (higher is better).
    
    Regression = current < baseline by more than threshold_pct.
    """
    if baseline_value <= 0:
        return ComparisonResult(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold_pct=threshold_pct,
            change_pct=0.0,
            passed=True,
            message="No baseline for comparison",
        )
    
    change_pct = ((current_value - baseline_value) / baseline_value) * 100
    # For throughput, negative change is bad
    passed = change_pct >= -threshold_pct
    
    if passed:
        message = f"✓ {metric_name}: {current_value:.2f} qps ({change_pct:+.1f}% vs baseline)"
    else:
        message = f"✗ {metric_name}: {current_value:.2f} qps ({change_pct:+.1f}% < -{threshold_pct}% threshold)"
    
    return ComparisonResult(
        metric_name=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        threshold_pct=threshold_pct,
        change_pct=change_pct,
        passed=passed,
        message=message,
    )


def run_comparison(
    results: Dict[str, Any],
    baseline: Dict[str, Any],
    p95_threshold: float,
    rps_threshold: float,
) -> ComparisonReport:
    """
    Run full comparison of results against baseline.
    """
    report = ComparisonReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        baseline_path="perf/baseline.json",
        results_path="perf/results.json",
    )
    
    # Get baseline thresholds (may override defaults)
    baseline_thresholds = baseline.get("thresholds", {})
    p95_threshold = baseline_thresholds.get("max_p95_regression_pct", p95_threshold)
    rps_threshold = baseline_thresholds.get("max_rps_regression_pct", rps_threshold)
    
    # Get benchmark baselines
    benchmarks = baseline.get("benchmarks", {})
    
    # Extract results summary if available
    summary = results.get("summary", {})
    
    # Compare latency metrics
    if "latency_p95" in summary:
        # Get baseline p95 from end_to_end or concurrency sections
        e2e = benchmarks.get("end_to_end", {}).get("pipeline", {})
        baseline_p95 = e2e.get("p95_latency_max_seconds", 1.0)
        
        result = compare_latency(
            baseline_value=baseline_p95,
            current_value=summary["latency_p95"],
            threshold_pct=p95_threshold,
            metric_name="latency_p95",
        )
        report.add_comparison(result)
    
    # Compare throughput if available
    if "throughput_qps" in summary:
        e2e = benchmarks.get("end_to_end", {}).get("pipeline", {})
        baseline_qps = e2e.get("min_throughput_qps", 2.0)
        
        result = compare_throughput(
            baseline_value=baseline_qps,
            current_value=summary.get("throughput_qps", 0.0),
            threshold_pct=rps_threshold,
            metric_name="throughput_qps",
        )
        report.add_comparison(result)
    
    # Compare concurrency results if available
    for test_result in results.get("results", []):
        if "concurrency" in test_result:
            concurrency = test_result["concurrency"]
            
            # Get baseline for this concurrency level
            threadpool = benchmarks.get("concurrency", {}).get("threadpool", {})
            baseline_key = f"concurrency_{concurrency}"
            baseline_data = threadpool.get(baseline_key, {})
            
            if baseline_data and "latency_p95" in test_result:
                result = compare_latency(
                    baseline_value=baseline_data.get("max_latency_p95_seconds", 5.0),
                    current_value=test_result["latency_p95"],
                    threshold_pct=p95_threshold,
                    metric_name=f"concurrency_{concurrency}_p95",
                )
                report.add_comparison(result)
            
            if baseline_data and "throughput_qps" in test_result:
                result = compare_throughput(
                    baseline_value=baseline_data.get("min_throughput_qps", 5.0),
                    current_value=test_result["throughput_qps"],
                    threshold_pct=rps_threshold,
                    metric_name=f"concurrency_{concurrency}_throughput",
                )
                report.add_comparison(result)
    
    # Generate summary
    passed_count = sum(1 for c in report.comparisons if c.passed)
    total_count = len(report.comparisons)
    
    if total_count == 0:
        report.summary = "No metrics to compare"
    elif report.passed:
        report.summary = f"✓ All {total_count} metrics within thresholds"
    else:
        failed_count = total_count - passed_count
        report.summary = f"✗ {failed_count}/{total_count} metrics exceeded thresholds"
    
    return report


def generate_markdown_report(report: ComparisonReport) -> str:
    """Generate markdown report from comparison results."""
    lines = [
        "# Performance Comparison Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Status:** {'✓ PASSED' if report.passed else '✗ FAILED'}",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Details",
        "",
        "| Metric | Baseline | Current | Change | Threshold | Status |",
        "|--------|----------|---------|--------|-----------|--------|",
    ]
    
    for comp in report.comparisons:
        status = "✓" if comp.passed else "✗"
        lines.append(
            f"| {comp.metric_name} | "
            f"{comp.baseline_value:.4f} | "
            f"{comp.current_value:.4f} | "
            f"{comp.change_pct:+.1f}% | "
            f"±{comp.threshold_pct}% | "
            f"{status} |"
        )
    
    if not report.passed:
        lines.extend([
            "",
            "## Failed Metrics",
            "",
        ])
        for comp in report.comparisons:
            if not comp.passed:
                lines.append(f"- {comp.message}")
    
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare performance results against baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_PATH,
        help=f"Path to baseline JSON (default: {DEFAULT_BASELINE_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for markdown report (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=DEFAULT_P95_REGRESSION_PCT,
        help=f"Max p95 latency regression %% (default: {DEFAULT_P95_REGRESSION_PCT})",
    )
    parser.add_argument(
        "--rps-threshold",
        type=float,
        default=DEFAULT_RPS_REGRESSION_PCT,
        help=f"Max throughput regression %% (default: {DEFAULT_RPS_REGRESSION_PCT})",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=True,
        help="Exit with code 1 if regressions detected (default: True)",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Don't fail on regression (for informational runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    print("=" * 60)
    print("VULCAN Performance Comparison")
    print("=" * 60)
    
    # Load files
    try:
        if not os.path.exists(args.results):
            print(f"⚠ Results file not found: {args.results}")
            print("  Run performance tests first to generate results.")
            return 0  # Not a failure - just no results yet
        
        results = load_json(args.results)
        print(f"✓ Loaded results: {args.results}")
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        return 1
    
    try:
        if not os.path.exists(args.baseline):
            print(f"⚠ Baseline file not found: {args.baseline}")
            print("  Create baseline with: python scripts/update_perf_baseline.py")
            return 0  # Not a failure - just no baseline yet
        
        baseline = load_json(args.baseline)
        print(f"✓ Loaded baseline: {args.baseline}")
    except Exception as e:
        print(f"✗ Error loading baseline: {e}")
        return 1
    
    # Run comparison
    print("\nComparing results against baseline...")
    report = run_comparison(
        results=results,
        baseline=baseline,
        p95_threshold=args.p95_threshold,
        rps_threshold=args.rps_threshold,
    )
    
    # Print results
    print()
    for comp in report.comparisons:
        print(f"  {comp.message}")
    
    print()
    print(f"Result: {report.summary}")
    
    # Generate markdown report
    markdown = generate_markdown_report(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"\n📄 Report saved to: {args.output}")
    
    # Save JSON report
    json_output = output_path.with_suffix(".json")
    json_output.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"📊 JSON saved to: {json_output}")
    
    # Determine exit code
    if args.no_fail:
        return 0
    
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
