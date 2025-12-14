import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    graphviz = None
    GRAPHVIZ_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Module-level constants for resource management
MAX_LOG_DIR_SIZE_MB = 1000  # Maximum size of log directory in MB
MAX_DASHBOARD_AGE_DAYS = 30  # Keep dashboards for 30 days
MAX_PLOT_AGE_DAYS = 7  # Keep plots for 7 days
MIN_FREE_DISK_MB = 100  # Minimum free disk space required
MAX_TENSOR_SIZE = 10000  # Maximum tensor dimension
MAX_TENSOR_ELEMENTS = 100_000_000  # Maximum total elements in tensor


class ObservabilityManager:
    """
    Centralizes Prometheus metrics and auto-generates Grafana dashboards with alerting.
    Supports advanced metrics like latency histograms and error rates for robust monitoring.

    Features:
    - Thread-safe file operations
    - Input validation for all metrics
    - Automatic log rotation and cleanup
    - Disk space monitoring
    - Configurable alert notifications
    """

    # Class-level constants (reference module constants for backwards compatibility)
    MAX_LOG_DIR_SIZE_MB = MAX_LOG_DIR_SIZE_MB
    MAX_DASHBOARD_AGE_DAYS = MAX_DASHBOARD_AGE_DAYS
    MAX_PLOT_AGE_DAYS = MAX_PLOT_AGE_DAYS
    MIN_FREE_DISK_MB = MIN_FREE_DISK_MB
    MAX_TENSOR_SIZE = MAX_TENSOR_SIZE
    MAX_TENSOR_ELEMENTS = MAX_TENSOR_ELEMENTS

    def __init__(
        self,
        log_dir: str = "observability_logs",
        notification_channels: Optional[List[str]] = None,
        enable_cleanup: bool = True,
    ):
        """
        Initialize the ObservabilityManager.

        Args:
            log_dir: Directory for storing logs and exports
            notification_channels: List of Grafana notification channel UIDs
            enable_cleanup: Enable automatic cleanup of old files
        """
        self.registry = CollectorRegistry()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Notification configuration
        self.notification_channels = notification_channels or []

        # Thread safety
        self.file_lock = threading.RLock()
        self.metrics_lock = threading.RLock()

        # Cleanup configuration
        self.enable_cleanup = enable_cleanup
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=1)

        # Initialize metrics with thread safety
        with self.metrics_lock:
            self.metrics = {
                # Existing metrics
                "tensor_attention": Gauge(
                    "tensor_attention",
                    "Attention weight for tensor features",
                    ["tensor_id"],
                    registry=self.registry,
                ),
                "audit_events": Counter(
                    "audit_events_total",
                    "Total audit events",
                    ["event_type"],
                    registry=self.registry,
                ),
                "bias_detected": Counter(
                    "bias_detected_total",
                    "Bias detected events",
                    registry=self.registry,
                ),
                "explainability_score": Gauge(
                    "explainability_score",
                    "Explainability score for tensor",
                    ["tensor_id"],
                    registry=self.registry,
                ),
                "counterfactual_diff": Gauge(
                    "counterfactual_diff",
                    "Counterfactual difference",
                    ["tensor_id"],
                    registry=self.registry,
                ),
                # Expanded Metrics
                "execution_latency": Histogram(
                    "execution_latency_seconds",
                    "Latency of component execution",
                    ["component"],
                    registry=self.registry,
                    buckets=[
                        0.001,
                        0.005,
                        0.01,
                        0.05,
                        0.1,
                        0.25,
                        0.5,
                        1,
                        2.5,
                        5,
                        10,
                        30,
                        60,
                    ],
                ),
                "execution_errors": Counter(
                    "execution_errors_total",
                    "Total execution errors",
                    ["component", "error_type"],
                    registry=self.registry,
                ),
                "disk_usage_bytes": Gauge(
                    "disk_usage_bytes",
                    "Disk space used by observability logs",
                    registry=self.registry,
                ),
                "cleanup_operations": Counter(
                    "cleanup_operations_total",
                    "Total cleanup operations performed",
                    ["operation_type"],
                    registry=self.registry,
                ),
            }

        self.logger = logging.getLogger("ObservabilityManager")
        self.logger.info(f"ObservabilityManager initialized with log_dir={log_dir}")

        # Perform initial cleanup if enabled
        if self.enable_cleanup:
            self._periodic_cleanup()

    def _check_disk_space(self) -> bool:
        """
        Check if sufficient disk space is available.

        Returns:
            True if sufficient space, False otherwise
        """
        try:
            stat = shutil.disk_usage(self.log_dir)
            free_mb = stat.free / (1024 * 1024)

            if free_mb < MIN_FREE_DISK_MB:
                self.logger.error(
                    f"Insufficient disk space: {free_mb:.2f}MB free, "
                    f"minimum required: {MIN_FREE_DISK_MB}MB"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to check disk space: {e}")
            return False

    def _get_directory_size_mb(self) -> float:
        """Calculate total size of log directory in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.log_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)

            size_mb = total_size / (1024 * 1024)

            # Update metric
            with self.metrics_lock:
                self.metrics["disk_usage_bytes"].set(total_size)

            return size_mb

        except Exception as e:
            self.logger.error(f"Failed to calculate directory size: {e}")
            return 0.0

    def _periodic_cleanup(self):
        """Perform periodic cleanup of old files."""
        now = datetime.now()

        # Handle both datetime and float (timestamp) for backwards compatibility
        if isinstance(self.last_cleanup, (int, float)):
            last_cleanup_dt = datetime.fromtimestamp(self.last_cleanup)
        else:
            last_cleanup_dt = self.last_cleanup

        # Check if cleanup is needed
        if now - last_cleanup_dt < self.cleanup_interval:
            return

        self.last_cleanup = now

        try:
            # Check directory size
            dir_size_mb = self._get_directory_size_mb()
            self.logger.info(f"Log directory size: {dir_size_mb:.2f}MB")

            # Clean old dashboards
            cleaned_dashboards = self._cleanup_old_files(
                pattern="*.json", max_age_days=MAX_DASHBOARD_AGE_DAYS
            )

            # Clean old plots
            cleaned_plots = self._cleanup_old_files(
                pattern="semantic_map_*.png", max_age_days=MAX_PLOT_AGE_DAYS
            )

            if cleaned_dashboards > 0 or cleaned_plots > 0:
                self.logger.info(
                    f"Cleanup complete: removed {cleaned_dashboards} dashboards, "
                    f"{cleaned_plots} plots"
                )

                with self.metrics_lock:
                    self.metrics["cleanup_operations"].labels(
                        operation_type="scheduled"
                    ).inc()

            # If still over size limit, perform aggressive cleanup
            if dir_size_mb > MAX_LOG_DIR_SIZE_MB:
                self.logger.warning(
                    f"Directory size {dir_size_mb:.2f}MB exceeds limit "
                    f"{MAX_LOG_DIR_SIZE_MB}MB, performing aggressive cleanup"
                )
                self._aggressive_cleanup()

        except Exception as e:
            self.logger.error(f"Periodic cleanup failed: {e}", exc_info=True)

    def _cleanup_old_files(self, pattern: str, max_age_days: int) -> int:
        """
        Remove files older than max_age_days matching pattern.

        Args:
            pattern: Glob pattern for files to clean
            max_age_days: Maximum age in days

        Returns:
            Number of files removed
        """
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(days=max_age_days)

        try:
            for filepath in self.log_dir.glob(pattern):
                if filepath.is_file():
                    file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

                    if file_mtime < cutoff_time:
                        with self.file_lock:
                            filepath.unlink()
                            cleaned_count += 1
                            self.logger.debug(f"Removed old file: {filepath.name}")

        except Exception as e:
            self.logger.error(f"Cleanup of {pattern} failed: {e}")

        return cleaned_count

    def _aggressive_cleanup(self):
        """Perform aggressive cleanup when size limit is exceeded."""
        try:
            # Remove oldest files first until under limit
            all_files = []

            for filepath in self.log_dir.glob("*"):
                if filepath.is_file():
                    all_files.append((filepath.stat().st_mtime, filepath))

            # Sort by modification time (oldest first)
            all_files.sort()

            removed_count = 0
            for mtime, filepath in all_files:
                current_size = self._get_directory_size_mb()

                if current_size <= MAX_LOG_DIR_SIZE_MB * 0.8:  # Target 80% of limit
                    break

                with self.file_lock:
                    filepath.unlink()
                    removed_count += 1
                    self.logger.debug(f"Aggressively removed: {filepath.name}")

            if removed_count > 0:
                self.logger.warning(f"Aggressive cleanup removed {removed_count} files")

                with self.metrics_lock:
                    self.metrics["cleanup_operations"].labels(
                        operation_type="aggressive"
                    ).inc()

        except Exception as e:
            self.logger.error(f"Aggressive cleanup failed: {e}", exc_info=True)

    def _validate_tensor(self, tensor: np.ndarray, tensor_id: str) -> bool:
        """
        Validate tensor input.

        Args:
            tensor: Numpy array to validate
            tensor_id: Identifier for logging

        Returns:
            True if valid, False otherwise
        """
        # Check if it's actually a numpy array
        if not isinstance(tensor, np.ndarray):
            self.logger.error(
                f"Invalid tensor {tensor_id}: expected numpy.ndarray, "
                f"got {type(tensor).__name__}"
            )
            return False

        # Check dimensions
        if tensor.ndim > 3:
            self.logger.error(
                f"Invalid tensor {tensor_id}: dimensionality {tensor.ndim} > 3 not supported"
            )
            return False

        # Check size constraints
        if any(dim > MAX_TENSOR_SIZE for dim in tensor.shape):
            self.logger.error(
                f"Invalid tensor {tensor_id}: dimension {tensor.shape} exceeds "
                f"maximum {MAX_TENSOR_SIZE}"
            )
            return False

        # Check total elements (using >= to make MAX_TENSOR_ELEMENTS exclusive)
        total_elements = tensor.size
        if total_elements >= MAX_TENSOR_ELEMENTS:
            self.logger.error(
                f"Invalid tensor {tensor_id}: {total_elements} elements exceeds "
                f"maximum {MAX_TENSOR_ELEMENTS}"
            )
            return False

        # Check for NaN or Inf
        if not np.isfinite(tensor).all():
            self.logger.warning(f"Tensor {tensor_id} contains NaN or Inf values")
            # Don't reject, just warn

        return True

    def plot_semantic_map(
        self,
        attn_tensor: np.ndarray,
        labels: Optional[List[str]] = None,
        out_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a Graphviz plot for an attention/semantic map.

        Args:
            attn_tensor: Square attention matrix (n x n)
            labels: Optional labels for nodes
            out_path: Optional output path for the plot

        Returns:
            Path to generated plot, or None if failed
        """
        if not GRAPHVIZ_AVAILABLE:
            self.logger.warning("Graphviz not installed; skipping plot.")
            return None

        # Validate tensor
        if not self._validate_tensor(attn_tensor, "semantic_map"):
            return None

        # Validate square matrix
        if attn_tensor.ndim != 2:
            self.logger.error(f"Attention tensor must be 2D, got {attn_tensor.ndim}D")
            return None

        if attn_tensor.shape[0] != attn_tensor.shape[1]:
            self.logger.error(
                f"Attention tensor must be square, got shape {attn_tensor.shape}"
            )
            return None

        n = attn_tensor.shape[0]

        # Validate or generate labels
        if labels is not None:
            if len(labels) != n:
                self.logger.error(
                    f"Label count {len(labels)} does not match tensor size {n}"
                )
                return None
        else:
            labels = [f"n{i}" for i in range(n)]

        # Check disk space before creating plot
        if not self._check_disk_space():
            self.logger.error("Insufficient disk space to create plot")
            return None

        try:
            dot = graphviz.Digraph(format="png")

            # Add nodes
            for i, lbl in enumerate(labels):
                dot.node(str(i), str(lbl))

            # Add edges with weights
            for i in range(n):
                for j in range(n):
                    weight = float(attn_tensor[i, j])

                    # Skip very low weights for clarity and performance
                    if abs(weight) > 0.01:
                        penwidth = max(0.5, min(abs(weight) * 3, 10.0))
                        dot.edge(
                            str(i),
                            str(j),
                            label=f"{weight:.2f}",
                            penwidth=str(penwidth),
                        )

            # Generate output path
            if out_path is None:
                timestamp = int(time.time() * 1000)
                out_path = str(self.log_dir / f"semantic_map_{timestamp}.png")

            # Render with file lock
            with self.file_lock:
                base_path = os.path.splitext(out_path)[0]
                dot.render(filename=base_path, cleanup=True, view=False)

            self.logger.info(f"Semantic map plotted to {out_path}")

            # Trigger cleanup if enabled
            if self.enable_cleanup:
                self._periodic_cleanup()

            return out_path

        except Exception as e:
            self.logger.error(f"Failed to render Graphviz plot: {e}", exc_info=True)
            return None

    def log_tensor_semantics(self, tensor: np.ndarray, tensor_id: Optional[str] = None):
        """
        Log tensor attention/explainability to Prometheus.

        Args:
            tensor: Numpy array containing tensor data
            tensor_id: Optional identifier for the tensor
        """
        # Generate ID if not provided
        if tensor_id is None:
            tensor_id = f"tensor_{id(tensor)}"

        # Validate tensor
        if not self._validate_tensor(tensor, tensor_id):
            self.logger.error(f"Skipping metrics for invalid tensor {tensor_id}")
            return

        try:
            with self.metrics_lock:
                if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
                    # Square matrix - compute mean attention weight
                    mean_weight = float(np.mean(np.abs(tensor)))
                    self.metrics["tensor_attention"].labels(tensor_id).set(mean_weight)
                    self.logger.debug(
                        f"Logged attention for {tensor_id}: {mean_weight:.4f}"
                    )

                elif tensor.ndim == 1:
                    # 1D tensor - compute explainability score
                    score = float(np.sum(np.abs(tensor)))
                    self.metrics["explainability_score"].labels(tensor_id).set(score)
                    self.logger.debug(
                        f"Logged explainability for {tensor_id}: {score:.4f}"
                    )

                else:
                    # Multi-dimensional tensor - compute general score
                    score = float(np.mean(np.abs(tensor)))
                    self.metrics["explainability_score"].labels(tensor_id).set(score)
                    self.logger.debug(
                        f"Logged general score for {tensor_id}: {score:.4f}"
                    )

            self.logger.info(f"Logged tensor semantics for {tensor_id}")

        except Exception as e:
            self.logger.error(
                f"Failed to log tensor semantics for {tensor_id}: {e}", exc_info=True
            )

    def log_audit_event(self, event_type: str = "generic"):
        """
        Increment audit event counter with a specific type.

        Args:
            event_type: Type of audit event
        """
        try:
            with self.metrics_lock:
                self.metrics["audit_events"].labels(event_type=event_type).inc()
            self.logger.info(f"Audit event logged: {event_type}")

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}", exc_info=True)

    def log_bias_detected(self):
        """Increment bias detection counter."""
        try:
            with self.metrics_lock:
                self.metrics["bias_detected"].inc()
            self.logger.warning("Bias detected event logged.")

        except Exception as e:
            self.logger.error(f"Failed to log bias detection: {e}", exc_info=True)

    def log_counterfactual_diff(self, tensor_id: str, diff: float):
        """
        Log counterfactual trace result.

        Args:
            tensor_id: Identifier for the tensor
            diff: Counterfactual difference value
        """
        # Validate diff value
        if not isinstance(diff, (int, float)):
            self.logger.error(
                f"Invalid diff value for {tensor_id}: expected number, "
                f"got {type(diff).__name__}"
            )
            return

        if not np.isfinite(diff):
            self.logger.error(
                f"Invalid diff value for {tensor_id}: {diff} is not finite"
            )
            return

        try:
            with self.metrics_lock:
                self.metrics["counterfactual_diff"].labels(tensor_id).set(float(diff))
            self.logger.info(f"Counterfactual diff logged for {tensor_id}: {diff:.4f}")

        except Exception as e:
            self.logger.error(
                f"Failed to log counterfactual diff for {tensor_id}: {e}", exc_info=True
            )

    def log_execution_latency(self, component: str, duration_seconds: float):
        """
        Observe execution latency for a component.

        Args:
            component: Name of the component
            duration_seconds: Execution duration in seconds
        """
        # Validate duration
        if not isinstance(duration_seconds, (int, float)):
            self.logger.error(
                f"Invalid duration for {component}: expected number, "
                f"got {type(duration_seconds).__name__}"
            )
            return

        if duration_seconds < 0:
            self.logger.error(
                f"Invalid duration for {component}: {duration_seconds} < 0"
            )
            return

        if not np.isfinite(duration_seconds):
            self.logger.error(
                f"Invalid duration for {component}: {duration_seconds} is not finite"
            )
            return

        try:
            with self.metrics_lock:
                self.metrics["execution_latency"].labels(component=component).observe(
                    duration_seconds
                )
            self.logger.info(
                f"Logged latency for '{component}': {duration_seconds:.4f}s"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to log execution latency for {component}: {e}", exc_info=True
            )

    def log_error(self, component: str, error_type: str = "generic_error"):
        """
        Increment error counter for a component.

        Args:
            component: Name of the component
            error_type: Type of error
        """
        try:
            with self.metrics_lock:
                self.metrics["execution_errors"].labels(
                    component=component, error_type=error_type
                ).inc()
            self.logger.error(f"Logged error for '{component}', type: '{error_type}'")

        except Exception as e:
            self.logger.error(
                f"Failed to log error for {component}: {e}", exc_info=True
            )

    def export_dashboard(self, dashboard_name: str = "graphix_dashboard") -> str:
        """
        Export Prometheus metrics as an advanced Grafana dashboard JSON with alerts.

        Args:
            dashboard_name: Name for the dashboard

        Returns:
            Path to the exported dashboard JSON file
        """
        # Check disk space
        if not self._check_disk_space():
            self.logger.error("Insufficient disk space to export dashboard")
            raise IOError("Insufficient disk space")

        # Build notification configuration
        notifications = []
        for channel_uid in self.notification_channels:
            notifications.append({"uid": channel_uid})

        dashboard = {
            "__inputs": [],
            "__requires": [],
            "annotations": {"list": []},
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [
                {
                    "type": "timeseries",
                    "title": "95th Percentile Execution Latency",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, sum(rate(execution_latency_seconds_bucket[5m])) by (le, component))",
                            "legendFormat": "{{component}}",
                            "refId": "A",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": None, "color": "green"},
                                    {"value": 1, "color": "yellow"},
                                    {"value": 5, "color": "red"},
                                ],
                            },
                        }
                    },
                },
                {
                    "type": "timeseries",
                    "title": "Execution Error Rate (per second)",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "sum(rate(execution_errors_total[5m])) by (component)",
                            "legendFormat": "{{component}}",
                            "refId": "A",
                        }
                    ],
                    "alert": {
                        "name": "High Execution Error Rate",
                        "message": "The error rate for component '{{ $labels.component }}' is above 0.1 per second for more than 1 minute.",
                        "for": "1m",
                        "every": "1m",
                        "noDataState": "ok",
                        "executionErrorState": "alerting",
                        "conditions": [
                            {
                                "type": "query",
                                "evaluator": {"type": "gt", "params": [0.1]},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "last"},
                            }
                        ],
                        "notifications": notifications,
                    },
                    "fieldConfig": {
                        "defaults": {
                            "unit": "ops",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": None, "color": "green"},
                                    {"value": 0.05, "color": "yellow"},
                                    {"value": 0.1, "color": "red"},
                                ],
                            },
                        }
                    },
                },
                {
                    "type": "stat",
                    "title": "Total Audit Events",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "sum(audit_events_total) by (event_type)",
                            "refId": "A",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "short"}},
                },
                {
                    "type": "stat",
                    "title": "Total Bias Detections",
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
                    "targets": [{"expr": "bias_detected_total", "refId": "A"}],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": None, "color": "green"},
                                    {"value": 1, "color": "yellow"},
                                    {"value": 10, "color": "red"},
                                ],
                            },
                        }
                    },
                },
                {
                    "type": "table",
                    "title": "Tensor Semantics",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "explainability_score or tensor_attention",
                            "format": "table",
                            "refId": "A",
                        }
                    ],
                },
                {
                    "type": "timeseries",
                    "title": "Disk Usage",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 12},
                    "targets": [
                        {
                            "expr": "disk_usage_bytes",
                            "legendFormat": "Disk Usage",
                            "refId": "A",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "bytes",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": None, "color": "green"},
                                    {"value": 800000000, "color": "yellow"},
                                    {"value": 1000000000, "color": "red"},
                                ],
                            },
                        }
                    },
                },
                {
                    "type": "stat",
                    "title": "Cleanup Operations",
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 12},
                    "targets": [
                        {
                            "expr": "sum(cleanup_operations_total) by (operation_type)",
                            "refId": "A",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "short"}},
                },
            ],
            "refresh": "10s",
            "schemaVersion": 36,
            "style": "dark",
            "tags": ["graphix", "autogenerated"],
            "templating": {"list": []},
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "timezone": "browser",
            "title": dashboard_name,
            "uid": None,
            "version": 1,
        }

        out_path = self.log_dir / f"{dashboard_name}.json"

        try:
            with self.file_lock:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(dashboard, f, indent=2)

            self.logger.info(f"Exported Grafana dashboard JSON to {out_path}")

            # Trigger cleanup if enabled
            if self.enable_cleanup:
                self._periodic_cleanup()

            return str(out_path)

        except Exception as e:
            self.logger.error(f"Failed to export dashboard: {e}", exc_info=True)
            raise

    def get_prometheus_metrics(self) -> bytes:
        """
        Returns metrics in Prometheus text format for scraping.

        Returns:
            Metrics in Prometheus exposition format
        """
        try:
            with self.metrics_lock:
                return generate_latest(self.registry)
        except Exception as e:
            self.logger.error(
                f"Failed to generate Prometheus metrics: {e}", exc_info=True
            )
            return b""

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics about the observability manager.

        Returns:
            Dictionary with current statistics
        """
        try:
            dir_size_mb = self._get_directory_size_mb()

            # Count files by type
            dashboard_count = len(list(self.log_dir.glob("*.json")))
            plot_count = len(list(self.log_dir.glob("semantic_map_*.png")))

            # Disk space info
            stat = shutil.disk_usage(self.log_dir)
            free_mb = stat.free / (1024 * 1024)

            return {
                "log_dir": str(self.log_dir),
                "dir_size_mb": dir_size_mb,
                "dashboard_count": dashboard_count,
                "plot_count": plot_count,
                "free_disk_mb": free_mb,
                "cleanup_enabled": self.enable_cleanup,
                "last_cleanup": (
                    self.last_cleanup.isoformat()
                    if isinstance(self.last_cleanup, datetime)
                    else datetime.fromtimestamp(self.last_cleanup).isoformat()
                ),
                "notification_channels": len(self.notification_channels),
            }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {}

    def shutdown(self):
        """Perform cleanup before shutdown."""
        self.logger.info("Shutting down ObservabilityManager...")

        try:
            # Final cleanup
            if self.enable_cleanup:
                self._periodic_cleanup()

            self.logger.info("Shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize with notification channels
    obs = ObservabilityManager(
        log_dir="observability_logs",
        notification_channels=["slack-alerts", "email-oncall"],
        enable_cleanup=True,
    )

    print("\n=== ObservabilityManager Demo ===\n")

    # Log existing and new metrics
    print("1. Logging tensor semantics...")
    tensor = np.random.rand(5, 5)
    obs.log_tensor_semantics(tensor, tensor_id="t_map_1")

    print("\n2. Creating semantic map plot...")
    img_path = obs.plot_semantic_map(tensor, labels=["A", "B", "C", "D", "E"])
    if img_path:
        print(f"   Semantic map saved to: {img_path}")

    print("\n3. Logging various events...")
    obs.log_audit_event("test_event")
    obs.log_audit_event("security_check")
    obs.log_bias_detected()
    obs.log_counterfactual_diff("tensor1", 0.12)

    print("\n4. Logging execution metrics...")
    obs.log_execution_latency("graph_execution", 0.155)
    obs.log_execution_latency("graph_execution", 0.170)
    obs.log_execution_latency("llm_call", 2.451)
    obs.log_error("llm_call", "api_timeout")
    obs.log_error("graph_parser", "syntax_error")

    print("\n5. Testing input validation...")
    # These should fail validation
    obs.log_tensor_semantics("not_a_tensor", "invalid_1")  # Wrong type
    obs.log_tensor_semantics(np.array([np.inf, np.nan]), "invalid_2")  # Invalid values
    obs.log_execution_latency("test", -1.0)  # Negative duration
    obs.log_counterfactual_diff("test", float("inf"))  # Infinite value

    print("\n6. Exporting dashboard...")
    dashboard_json = obs.export_dashboard("graphix_monitoring")
    print(f"   Dashboard exported to: {dashboard_json}")

    print("\n7. Getting statistics...")
    stats = obs.get_stats()
    print(f"   Directory size: {stats['dir_size_mb']:.2f}MB")
    print(f"   Dashboard count: {stats['dashboard_count']}")
    print(f"   Plot count: {stats['plot_count']}")
    print(f"   Free disk space: {stats['free_disk_mb']:.2f}MB")

    print("\n8. Prometheus metrics output:")
    print("-" * 60)
    metrics_output = obs.get_prometheus_metrics().decode()
    # Print first 20 lines
    for i, line in enumerate(metrics_output.split("\n")[:20]):
        print(line)
    print("... (truncated)")
    print("-" * 60)

    print("\n9. Shutting down...")
    obs.shutdown()

    print("\n=== Demo Complete ===\n")
