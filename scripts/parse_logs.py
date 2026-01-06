#!/usr/bin/env python3
"""
Parse logs for error-budget enforcement and emit a concise summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List


DEFAULT_PATTERNS = [
    "No language generation backend available",
    "OpenAI package not installed",
    "OPENAI_API_KEY configuration",
]


def load_metrics(metrics_path: str) -> Dict:
    if not metrics_path or not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_log_file(path: str) -> Dict[str, any]:
    errors = []
    warns = 0
    total_errors = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ERROR" in line:
                total_errors += 1
                errors.append(line.strip())
            if "WARN" in line or "WARNING" in line:
                warns += 1

    top_errors = Counter(errors).most_common(10)
    return {
        "total_errors": total_errors,
        "total_warns": warns,
        "top_errors": top_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Log parser with error budgets")
    parser.add_argument("--log-path", required=True, help="Path to log file")
    parser.add_argument("--summary-output", default="artifacts/log_summary.txt")
    parser.add_argument("--json-output", default="artifacts/log_summary.json")
    parser.add_argument("--metrics-json", default=None, help="Optional metrics json to merge")
    parser.add_argument("--error-budget-total", type=int, default=20, help="Maximum allowed ERROR lines")
    parser.add_argument("--error-budget-patterns", type=int, default=0, help="Maximum allowed matches for critical patterns")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional pattern to treat as critical (can be repeated)",
    )
    args = parser.parse_args()

    patterns = DEFAULT_PATTERNS + args.pattern
    data = parse_log_file(args.log_path)
    matches: Dict[str, int] = {}
    with open(args.log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pat in patterns:
                if re.search(pat, line):
                    matches[pat] = matches.get(pat, 0) + 1

    metrics = load_metrics(args.metrics_json)

    summary = {
        "log_path": args.log_path,
        "total_errors": data["total_errors"],
        "total_warns": data["total_warns"],
        "top_errors": data["top_errors"],
        "pattern_matches": matches,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"Errors: {data['total_errors']} (budget {args.error_budget_total})",
        f"Warnings: {data['total_warns']}",
    ]
    for pat, count in matches.items():
        lines.append(f"Pattern '{pat}': {count} (budget {args.error_budget_patterns})")
    lines.append("Top errors:")
    for err, count in data["top_errors"]:
        lines.append(f"  {count}x {err}")

    if metrics:
        lines.append("Load metrics:")
        summary_fields = metrics.get("summary") or metrics
        for key in ["total_requests", "success_rate", "latency_p50_ms", "latency_p95_ms", "error_count"]:
            if key in summary_fields:
                lines.append(f"  {key}: {summary_fields[key]}")

    with open(args.summary_output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)

    fail = False
    if data["total_errors"] > args.error_budget_total:
        fail = True
    if any(count > args.error_budget_patterns for count in matches.values()):
        fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
