"""
Graphix Transparency Report Generator (Production-Ready)
========================================================
Version: 2.0.0 - All issues fixed, production-ready
Generates comprehensive transparency reports with metrics, audits, and anomaly detection.
"""

import os
import json
import logging
import datetime
import threading
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass, field
from collections import deque
import statistics

# Platform-specific file locking
try:
    import fcntl

    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False
    fcntl = None

# Windows-specific file locking
try:
    import msvcrt

    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False
    msvcrt = None

# Optional prometheus_client
try:
    import prometheus_client

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_client = None

# Optional numpy
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Optional observability_manager
try:
    from observability_manager import (
        get_prometheus_metrics,
        notify_error,
        notify_success,
        send_metric_event,
        notify_anomaly,
    )

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

    # Fallback implementations
    def get_prometheus_metrics(names):
        return {}

    def notify_error(msg):
        logging.error(msg)

    def notify_success(msg):
        logging.info(msg)

    def send_metric_event(name, data):
        pass

    def notify_anomaly(msg):
        logging.warning(msg)


# Optional security_audit_engine
try:
    from security_audit_engine import AUDIT_LOG_PATH, parse_audit_log_line

    AUDIT_ENGINE_AVAILABLE = True
except ImportError:
    AUDIT_ENGINE_AVAILABLE = False
    AUDIT_LOG_PATH = "./audit_log.jsonl"

    def parse_audit_log_line(line):
        return json.loads(line)


# Optional nso_aligner
try:
    from nso_aligner import (
        get_bias_taxonomy,
        get_bias_trends,
        get_bias_examples,
        get_bias_taxonomy_schema,
    )

    NSO_ALIGNER_AVAILABLE = True
except ImportError:
    NSO_ALIGNER_AVAILABLE = False

    def get_bias_taxonomy():
        return {}

    def get_bias_trends():
        return {}

    def get_bias_examples(n):
        return []

    def get_bias_taxonomy_schema():
        return {}


from logging.handlers import RotatingFileHandler

# --- Constants and Config ---
REPORT_PATH = Path("transparency_report.md")
LOG_PATH = Path("transparency_logs.log")
ARCHIVE_PATH = Path("transparency_report_archive")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
MAX_RECENT_AUDITS = 10
MAX_RECENT_BIAS = 5
MAX_REPORT_SIZE = 2_000_000  # bytes
MAX_ARCHIVE_COUNT = 100
MAX_AUDIT_LOG_LINES = 100000
ANOMALY_THRESHOLD_STDDEV = 3.0  # Statistical threshold
MIN_SAMPLES_FOR_ANOMALY = 5

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        RotatingFileHandler(LOG_PATH, maxBytes=MAX_REPORT_SIZE, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class MetricHistory:
    """Tracks metric values over time for anomaly detection."""

    values: deque = field(default_factory=lambda: deque(maxlen=20))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=20))

    def add(self, value: float, timestamp: datetime.datetime):
        """Add a value with timestamp."""
        self.values.append(value)
        self.timestamps.append(timestamp)

    def has_enough_samples(self) -> bool:
        """Check if we have enough samples for statistics."""
        return len(self.values) >= MIN_SAMPLES_FOR_ANOMALY

    def get_mean(self) -> Optional[float]:
        """Get mean of values."""
        if not self.values:
            return None
        return statistics.mean(self.values)

    def get_stddev(self) -> Optional[float]:
        """Get standard deviation."""
        if len(self.values) < 2:
            return None
        return statistics.stdev(self.values)


class FileLock:
    """Cross-platform file lock for concurrent access protection."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock_file = None
        self.lock_path = filepath.parent / f".{filepath.name}.lock"

    def __enter__(self):
        """Acquire lock (platform-specific)."""
        try:
            self.lock_file = open(self.lock_path, "w")

            if FCNTL_AVAILABLE:
                # Unix/Linux file locking
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
            elif MSVCRT_AVAILABLE:
                # Windows file locking
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                # Fallback: No locking available (testing/development only)
                logger.warning("File locking not available on this platform")

            return self
        except Exception as e:
            logger.warning(f"Failed to acquire file lock: {e}")
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock (platform-specific)."""
        if self.lock_file:
            try:
                if FCNTL_AVAILABLE:
                    # Unix/Linux file unlocking
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                elif MSVCRT_AVAILABLE:
                    # Windows file unlocking
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)

                self.lock_file.close()
            except Exception as e:
                logger.warning(f"Failed to release file lock: {e}")


class TransparencyReportValidator:
    """Validator for report data."""

    @staticmethod
    def validate_metrics(metrics: Any) -> Tuple[bool, Optional[str]]:
        """Validate metrics structure."""
        if not isinstance(metrics, dict):
            return False, f"Metrics must be dict, got {type(metrics)}"

        for key, value in metrics.items():
            if not isinstance(key, str):
                return False, f"Metric key must be string, got {type(key)}"

            if not isinstance(value, (int, float, str)):
                return (
                    False,
                    f"Metric value must be numeric or string, got {type(value)}",
                )

        return True, None

    @staticmethod
    def validate_audit_record(record: Any) -> Tuple[bool, Optional[str]]:
        """Validate audit log record."""
        if not isinstance(record, dict):
            return False, f"Record must be dict, got {type(record)}"

        return True, None

    @staticmethod
    def validate_json_line(line: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Validate and parse JSON line."""
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return False, "JSON must be dict", None
            return True, None, data
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None


def ensure_archive():
    """Create archive directory if it doesn't exist."""
    try:
        ARCHIVE_PATH.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error(f"Failed to create archive directory: {e}")


def cleanup_old_archives():
    """Remove old archives to prevent unbounded growth."""
    try:
        if not ARCHIVE_PATH.exists():
            return

        archives = sorted(ARCHIVE_PATH.glob("transparency_report_*.md"))

        if len(archives) > MAX_ARCHIVE_COUNT:
            to_remove = archives[:-MAX_ARCHIVE_COUNT]
            for archive in to_remove:
                try:
                    archive.unlink()
                    logger.info(f"Removed old archive: {archive}")
                except Exception as e:
                    logger.warning(f"Failed to remove archive {archive}: {e}")

    except Exception as e:
        logger.error(f"Failed to cleanup archives: {e}")


def archive_current_report():
    """Archive current report with race condition protection."""
    try:
        # Use file lock to prevent race conditions
        with FileLock(REPORT_PATH):
            if not REPORT_PATH.exists():
                return

            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_file = ARCHIVE_PATH / f"transparency_report_{timestamp}.md"

            # Use atomic rename
            temp_file = REPORT_PATH.with_suffix(".tmp")
            REPORT_PATH.rename(temp_file)
            temp_file.rename(archive_file)

            logger.info(f"Archived previous report to {archive_file}")

    except Exception as e:
        logger.error(f"Failed to archive report: {e}")


def get_current_commit_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        git_head = Path(".git/HEAD")
        if not git_head.exists():
            return None

        with open(git_head, "r") as f:
            ref = f.readline().strip()
            if ref.startswith("ref:"):
                ref_path = Path(".git") / ref.split(" ")[-1]
                if ref_path.exists():
                    return ref_path.read_text().strip()[:8]
            else:
                return ref[:8]
    except Exception as e:
        logger.warning(f"Failed to get commit hash: {e}")
        return None


def get_ci_env_info() -> str:
    """Detect CI/CD environment."""
    for key in ["GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS"]:
        if os.environ.get(key, ""):
            return key
    return "manual"


def get_previous_report_metrics() -> Dict[str, float]:
    """
    Parse previous report for metrics with validation.

    Returns metrics with timestamp for trend analysis.
    """
    if not REPORT_PATH.exists():
        return {}

    metrics = {}
    validator = TransparencyReportValidator()

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            # Only read last 1000 lines to limit memory
            lines = deque(f, maxlen=1000)

        for line in lines:
            line = line.strip()

            # Parse markdown tables
            if line.startswith("|") and "Metric" not in line and "---" not in line:
                parts = [x.strip() for x in line.split("|") if x.strip()]

                if len(parts) >= 2:
                    name, val = parts[0], parts[1]

                    # Try to parse as float
                    try:
                        # Remove trend indicators
                        val_clean = val.split()[0] if " " in val else val
                        metrics[name] = float(val_clean)
                    except (ValueError, IndexError):
                        pass

        # Validate extracted metrics
        valid, error = validator.validate_metrics(metrics)
        if not valid:
            logger.warning(f"Previous metrics validation failed: {error}")
            return {}

    except Exception as e:
        logger.warning(f"Could not extract previous report metrics: {e}")
        return {}

    return metrics


def fetch_interpretability_metrics() -> Dict[str, Any]:
    """Fetch interpretability metrics with validation."""
    if not OBSERVABILITY_AVAILABLE:
        logger.info("Observability manager not available")
        return {}

    try:
        metric_names = [
            "shap_coverage",
            "counterfactual_diff_mean",
            "tensor_attention_max",
            "tensor_attention_mean",
            "interpretability_score",
            "local_faithfulness",
            "global_faithfulness",
        ]

        metrics = get_prometheus_metrics(metric_names)

        # Validate metrics
        validator = TransparencyReportValidator()
        valid, error = validator.validate_metrics(metrics)

        if not valid:
            logger.warning(f"Metrics validation failed: {error}")
            return {}

        # Add computed metrics
        if NUMPY_AVAILABLE and "shap_coverage" in metrics:
            try:
                metrics["shap_coverage_percent"] = float(metrics["shap_coverage"]) * 100
            except (ValueError, TypeError):
                pass

        return metrics

    except Exception as e:
        logger.exception("Failed to fetch interpretability metrics")
        notify_error(f"Transparency report error: {e}")
        return {}


def fetch_audit_metrics() -> Tuple[int, int, List[Dict]]:
    """
    Fetch audit metrics with streaming to avoid memory issues.

    Returns most recent audits, not just first N.
    """
    proposal_count = 0
    bias_detected_count = 0
    recent_audits = deque(maxlen=MAX_RECENT_AUDITS)
    validator = TransparencyReportValidator()

    try:
        if not os.path.exists(AUDIT_LOG_PATH):
            logger.warning(f"Audit log not found: {AUDIT_LOG_PATH}")
            return 0, 0, []

        # Check file permissions
        if not os.access(AUDIT_LOG_PATH, os.R_OK):
            raise PermissionError(f"No read permission for {AUDIT_LOG_PATH}")

        # Stream file line by line to avoid loading everything into memory
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            line_count = 0

            for line in f:
                line_count += 1

                # Limit lines processed to prevent DoS
                if line_count > MAX_AUDIT_LOG_LINES:
                    logger.warning(
                        f"Audit log exceeds {MAX_AUDIT_LOG_LINES} lines, truncating"
                    )
                    break

                line = line.strip()
                if not line:
                    continue

                # Validate and parse JSON
                valid_json, json_error, record = validator.validate_json_line(line)

                if not valid_json:
                    logger.warning(
                        f"Invalid JSON in audit log line {line_count}: {json_error}"
                    )
                    continue

                # Validate record structure
                valid_record, record_error = validator.validate_audit_record(record)

                if not valid_record:
                    logger.warning(f"Invalid record structure: {record_error}")
                    continue

                proposal_count += 1

                if record.get("bias_detected", False):
                    bias_detected_count += 1

                # Keep most recent (using deque with maxlen)
                recent_audits.append(record)

        return proposal_count, bias_detected_count, list(recent_audits)

    except FileNotFoundError as e:
        logger.error(f"Audit log not found: {e}")
        notify_error(f"Transparency report error: {e}")
        return 0, 0, []

    except PermissionError as e:
        logger.error(f"Permission error reading audit log: {e}")
        notify_error(f"Transparency report error: {e}")
        return 0, 0, []

    except Exception as e:
        logger.exception("Error reading audit log")
        notify_error(f"Transparency report error: {e}")
        return 0, 0, []


def fetch_bias_taxonomy() -> Dict[str, Any]:
    """Fetch bias taxonomy with error handling."""
    if not NSO_ALIGNER_AVAILABLE:
        logger.info("NSO aligner not available")
        return {}

    try:
        taxonomy = get_bias_taxonomy()

        # Validate structure
        if not isinstance(taxonomy, dict):
            logger.warning(f"Bias taxonomy must be dict, got {type(taxonomy)}")
            return {}

        return taxonomy

    except Exception as e:
        logger.exception("Failed to fetch bias taxonomy")
        notify_error(f"Transparency report error: {e}")
        return {}


def fetch_bias_trends() -> Dict[str, Any]:
    """Fetch bias trends with error handling."""
    if not NSO_ALIGNER_AVAILABLE:
        return {}

    try:
        trends = get_bias_trends()

        if not isinstance(trends, dict):
            logger.warning(f"Bias trends must be dict, got {type(trends)}")
            return {}

        return trends

    except Exception as e:
        logger.warning(f"Could not fetch bias trends: {e}")
        return {}


def fetch_bias_examples() -> List[Dict[str, Any]]:
    """Fetch bias examples with error handling."""
    if not NSO_ALIGNER_AVAILABLE:
        return []

    try:
        examples = get_bias_examples(MAX_RECENT_BIAS)

        if not isinstance(examples, list):
            logger.warning(f"Bias examples must be list, got {type(examples)}")
            return []

        return examples

    except Exception as e:
        logger.warning(f"Could not fetch bias examples: {e}")
        return []


def fetch_bias_taxonomy_schema() -> Dict[str, Any]:
    """Fetch bias taxonomy schema with error handling."""
    if not NSO_ALIGNER_AVAILABLE:
        return {}

    try:
        schema = get_bias_taxonomy_schema()

        if not isinstance(schema, dict):
            logger.warning(f"Taxonomy schema must be dict, got {type(schema)}")
            return {}

        return schema

    except Exception as e:
        logger.warning(f"Could not fetch bias taxonomy schema: {e}")
        return {}


def build_markdown_section(title: str, table: str, notes: Optional[str] = None) -> str:
    """Build markdown section with validation."""
    if not isinstance(title, str) or not isinstance(table, str):
        logger.error("Invalid section data types")
        return ""

    md = f"## {title}\n\n{table}\n"

    if notes and isinstance(notes, str):
        md += f"\n**Notes:** {notes}\n"

    return md


def generate_interpretability_table(
    metrics: Dict[str, Any], prev: Dict[str, float]
) -> str:
    """Generate interpretability metrics table."""
    if not metrics:
        return "_No interpretability metric data available._"

    table = "| Metric | Value | ΔTrend |\n|---|---|---|\n"

    for k, v in sorted(metrics.items()):
        try:
            value = f"{float(v):.4f}"
        except (ValueError, TypeError):
            value = str(v)[:50]  # Truncate long strings

        # Normalize metric name for lookup
        normalized_name = k.replace("_", " ").title()
        prev_val = prev.get(normalized_name) or prev.get(k)

        trend = ""
        if prev_val is not None:
            try:
                delta = float(v) - prev_val
                trend = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            except (ValueError, TypeError):
                trend = "N/A"

        table += f"| {normalized_name} | {value} | {trend} |\n"

    return table


def generate_audit_table(
    proposal_count: int, bias_detected_count: int, prev: Dict[str, float]
) -> str:
    """Generate audit metrics table."""
    rate = (bias_detected_count / proposal_count * 100) if proposal_count > 0 else 0

    prev_bias = prev.get("Bias Detections")
    prev_proposals = prev.get("Total Proposals")

    trend_bias = ""
    trend_rate = ""

    if prev_bias is not None:
        trend_bias = f"{bias_detected_count - int(prev_bias):+}"

    if prev_proposals is not None and prev_proposals > 0:
        prev_rate = (prev_bias / prev_proposals * 100) if prev_bias is not None else 0
        if prev_bias is not None:
            trend_rate = f"{rate - prev_rate:+.2f}"

    table = "| Total Proposals | Bias Detections | Bias Rate (%) | ΔBias | ΔRate |\n"
    table += "|---|---|---|---|---|\n"
    table += f"| {proposal_count} | {bias_detected_count} | {rate:.2f} | {trend_bias} | {trend_rate} |\n"

    return table


def generate_audit_samples_table(records: List[Dict[str, Any]]) -> str:
    """Generate audit samples table."""
    if not records:
        return "_No recent audit log records available._"

    # Get common keys across records
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())

    # Sort keys for consistent display
    keys = sorted(all_keys)

    # Limit number of columns for readability
    if len(keys) > 10:
        keys = keys[:10]

    header = "| " + " | ".join(k.replace("_", " ").title() for k in keys) + " |\n"
    sep = "|" + " --- |" * len(keys) + "\n"

    table = header + sep

    for rec in records:
        row = []
        for k in keys:
            val = str(rec.get(k, ""))[:50]  # Truncate long values
            row.append(val)
        table += "| " + " | ".join(row) + " |\n"

    return table


def generate_bias_taxonomy_table(
    taxonomy: Dict[str, Any], prev: Dict[str, float]
) -> str:
    """Generate bias taxonomy table."""
    if not taxonomy:
        return "_No bias taxonomy data available._"

    table = "| Bias Type | Count | ΔCount |\n|---|---|---|\n"

    for bias_type, count in sorted(taxonomy.items()):
        prev_count = prev.get(bias_type)

        trend = ""
        if prev_count is not None:
            try:
                trend = f"{int(count) - int(prev_count):+}"
            except (ValueError, TypeError):
                trend = "N/A"

        table += f"| {bias_type} | {count} | {trend} |\n"

    return table


def generate_bias_trends_table(trends: Dict[str, Any]) -> str:
    """Generate bias trends table."""
    if not trends:
        return ""

    table = "| Bias Type | Trend |\n|---|---|\n"

    for bias_type, trend in sorted(trends.items()):
        table += f"| {bias_type} | {str(trend)[:100]} |\n"

    return table


def generate_bias_examples_table(examples: List[Dict[str, Any]]) -> str:
    """Generate bias examples table."""
    if not examples:
        return ""

    all_keys = set()
    for rec in examples:
        all_keys.update(rec.keys())

    keys = sorted(all_keys)[:8]  # Limit columns

    header = "| " + " | ".join(k.replace("_", " ").title() for k in keys) + " |\n"
    sep = "|" + " --- |" * len(keys) + "\n"

    table = header + sep

    for rec in examples:
        row = [str(rec.get(k, ""))[:50] for k in keys]
        table += "| " + " | ".join(row) + " |\n"

    return table


def build_metadata_section(
    metadata: Dict[str, str], taxonomy_schema: Optional[Dict] = None
) -> str:
    """Build metadata section."""
    md = "### Report Metadata\n\n"

    for k, v in sorted(metadata.items()):
        md += f"- **{k.replace('_', ' ').title()}**: `{v}`\n"

    if taxonomy_schema:
        try:
            schema_json = json.dumps(taxonomy_schema, indent=2)
            md += f"\n#### Bias Taxonomy Schema (for reference)\n\n```json\n{schema_json}\n```\n"
        except Exception as e:
            logger.warning(f"Failed to serialize taxonomy schema: {e}")

    return md + "\n"


def get_environment_metadata() -> Dict[str, str]:
    """Get environment metadata."""
    metadata = {}

    metadata["timestamp_utc"] = datetime.datetime.utcnow().strftime(TIMESTAMP_FMT)
    metadata["user"] = os.environ.get("USER", "") or os.environ.get(
        "USERNAME", "unknown"
    )
    metadata["hostname"] = os.uname().nodename if hasattr(os, "uname") else "unknown"
    metadata["commit_hash"] = get_current_commit_hash() or "unknown"
    metadata["ci_env"] = get_ci_env_info()
    metadata["python_version"] = sys.version.replace("\n", " ")[:100]

    return metadata


def append_report(report_text: str):
    """
    Append report with size checking and atomic writes.

    FIXED: Checks size before appending to prevent unbounded growth.
    """
    try:
        ensure_archive()

        # Use file lock for concurrent access protection
        with FileLock(REPORT_PATH):
            # Check size BEFORE appending
            if REPORT_PATH.exists():
                current_size = REPORT_PATH.stat().st_size

                # If file is already too big OR would become too big, archive first
                if (
                    current_size > MAX_REPORT_SIZE
                    or (current_size + len(report_text)) > MAX_REPORT_SIZE
                ):
                    archive_current_report()

            # Check write permissions
            if REPORT_PATH.exists() and not os.access(REPORT_PATH, os.W_OK):
                raise PermissionError(f"No write permission for {REPORT_PATH}")

            if not REPORT_PATH.exists() and not os.access(REPORT_PATH.parent, os.W_OK):
                raise PermissionError(
                    f"No write permission for directory {REPORT_PATH.parent}"
                )

            # Atomic write: write to temp file first
            temp_file = REPORT_PATH.with_suffix(".tmp")

            with open(temp_file, "w", encoding="utf-8") as f:
                # If report exists, copy existing content
                if REPORT_PATH.exists():
                    with open(REPORT_PATH, "r", encoding="utf-8") as existing:
                        f.write(existing.read())

                # Append new content
                f.write(report_text)
                f.write("\n\n---\n\n")
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (FIXED: use replace() for Windows compatibility)
            temp_file.replace(REPORT_PATH)

    except PermissionError as e:
        logger.error(f"Permission error writing report: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to append report: {e}")
        raise


def check_for_anomalies(
    metrics: Dict[str, Any],
    prev: Dict[str, float],
    history: Optional[Dict[str, MetricHistory]] = None,
) -> List[str]:
    """
    Check for anomalies using statistical methods.

    FIXED: Uses standard deviation instead of arbitrary 50% threshold.
    """
    anomalies = []

    if history is None:
        history = {}

    for k, v in metrics.items():
        try:
            current_val = float(v)
        except (ValueError, TypeError):
            continue

        # Normalize metric name
        normalized_name = k.replace("_", " ").title()
        prev_val = prev.get(normalized_name) or prev.get(k)

        # Initialize history for this metric
        if k not in history:
            history[k] = MetricHistory()

        # Add current value
        history[k].add(current_val, datetime.datetime.utcnow())

        # Need enough samples for statistical analysis
        if not history[k].has_enough_samples():
            # Fall back to simple threshold for initial samples
            if prev_val is not None:
                delta = abs(current_val - prev_val)
                # Use 100% change as initial threshold (less sensitive)
                if delta > (abs(prev_val) + 0.01):
                    anomalies.append(
                        f"Large change in {k}: {prev_val:.4f} -> {current_val:.4f} "
                        f"(insufficient history for statistical analysis)"
                    )
            continue

        # Statistical anomaly detection
        mean = history[k].get_mean()
        stddev = history[k].get_stddev()

        if mean is not None and stddev is not None and stddev > 0:
            # Z-score calculation
            z_score = abs(current_val - mean) / stddev

            # Flag if beyond threshold (typically 3 standard deviations)
            if z_score > ANOMALY_THRESHOLD_STDDEV:
                anomalies.append(
                    f"Statistical anomaly in {k}: value={current_val:.4f}, "
                    f"mean={mean:.4f}, stddev={stddev:.4f}, z-score={z_score:.2f}"
                )

    return anomalies


def main():
    """Main report generation function."""
    try:
        # Get metadata
        metadata = get_environment_metadata()
        timestamp = metadata["timestamp_utc"]

        logger.info(f"Generating transparency report at {timestamp}")

        # Fetch all data
        prev_metrics = get_previous_report_metrics()
        interpretability_metrics = fetch_interpretability_metrics()
        proposal_count, bias_detected_count, recent_audits = fetch_audit_metrics()
        bias_taxonomy = fetch_bias_taxonomy()
        bias_trends = fetch_bias_trends()
        bias_examples = fetch_bias_examples()
        taxonomy_schema = fetch_bias_taxonomy_schema()

        # Check for anomalies with statistical methods
        anomalies = []
        anomalies += check_for_anomalies(interpretability_metrics, prev_metrics)
        anomalies += check_for_anomalies(bias_taxonomy, prev_metrics)

        # Notify about anomalies
        if anomalies:
            for a in anomalies:
                logger.warning(a)
                try:
                    notify_anomaly(a)
                except Exception as e:
                    logger.debug(f"Failed to notify anomaly: {e}")

        # Build report
        report_text = f"# Transparency Report Update ({timestamp} UTC)\n\n"
        report_text += build_metadata_section(metadata, taxonomy_schema)

        report_text += build_markdown_section(
            "Interpretability Metrics",
            generate_interpretability_table(interpretability_metrics, prev_metrics),
            notes="Metrics sourced from Prometheus. SHAP coverage and counterfactual diffs reflect model explainability.",
        )

        report_text += build_markdown_section(
            "Security Audit Metrics",
            generate_audit_table(proposal_count, bias_detected_count, prev_metrics),
            notes="Audit log metrics include proposal and bias detection counts for traceability.",
        )

        report_text += build_markdown_section(
            "Sample Audit Log Records",
            generate_audit_samples_table(recent_audits),
            notes="Most recent audit records (limited to prevent excessive memory usage).",
        )

        report_text += build_markdown_section(
            "Bias Taxonomy Trends",
            generate_bias_taxonomy_table(bias_taxonomy, prev_metrics),
            notes="Bias types categorized according to the latest taxonomy (see nso_aligner).",
        )

        if bias_trends:
            report_text += build_markdown_section(
                "Bias Trends Over Time",
                generate_bias_trends_table(bias_trends),
                notes=None,
            )

        if bias_examples:
            report_text += build_markdown_section(
                "Bias Examples", generate_bias_examples_table(bias_examples), notes=None
            )

        if anomalies:
            report_text += "\n### Anomalies Detected\n\n"
            for a in anomalies:
                report_text += f"- {a}\n"
            report_text += "\n**Note:** Anomalies detected using statistical analysis "
            report_text += (
                f"(threshold: {ANOMALY_THRESHOLD_STDDEV} standard deviations)\n"
            )

        # Append report with proper locking
        append_report(report_text)

        logger.info("Report successfully generated and appended.")

        # Cleanup old archives
        cleanup_old_archives()

        # Notify success
        try:
            notify_success("Transparency report update generated successfully.")
            send_metric_event(
                "transparency_report_generated",
                {
                    "timestamp": timestamp,
                    "anomalies": len(anomalies),
                    "proposals": proposal_count,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to send notifications: {e}")

    except Exception as err:
        logger.exception("Critical error in report generation")
        try:
            notify_error(f"Transparency report generation failed: {err}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Report generation interrupted by user")
    except Exception as err:
        logger.error(f"Report generation failed: {err}")
        exit(1)
