"""
Graphix Drift Detector (Production-Ready)
==========================================
Version: 2.0.0 - All issues fixed, realignment implemented
Tracks embedding drift, detects anomalies, and performs automatic realignment.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# FAISS import with enhanced configuration
try:
    from src.utils.faiss_config import initialize_faiss

    faiss, FAISS_AVAILABLE, _ = initialize_faiss()
except ImportError:
    # Fallback: try direct import if config module not available
    try:
        import faiss

        FAISS_AVAILABLE = True
    except ImportError:
        # FAISS truly not available
        FAISS_AVAILABLE = False
        faiss = None

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # No-op metrics
    class _NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def time(self):
            class _Timer:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def __call__(self, func):
                    return func

            return _Timer()

    Gauge = Counter = Histogram = _NoOpMetric  # type: ignore

logger = logging.getLogger(__name__)


# Metrics - Handle duplicate registration gracefully with singleton pattern
class _MetricsRegistry:
    """Singleton registry for DriftDetector metrics to prevent duplicate registration."""

    _instance = None
    _initialized = False
    _drift_value_gauge = None
    _drift_events = None
    _drift_checks = None
    _drift_latency = None
    _realignment_operations = None
    _validation_errors = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize metrics once at class level
        if not _MetricsRegistry._initialized and PROMETHEUS_AVAILABLE:
            try:
                _MetricsRegistry._drift_value_gauge = Gauge(
                    "drift_detector_drift_value", "Current mean embedding drift"
                )
                _MetricsRegistry._drift_events = Counter(
                    "drift_detector_realignments_total",
                    "Total drift-triggered realignments",
                )
                _MetricsRegistry._drift_checks = Counter(
                    "drift_detector_checks_total", "Total drift checks performed"
                )
                _MetricsRegistry._drift_latency = Histogram(
                    "drift_detector_check_latency_seconds",
                    "Latency of drift detection checks",
                )
                _MetricsRegistry._realignment_operations = Counter(
                    "drift_detector_realignment_operations",
                    "Realignment operations performed",
                )
                _MetricsRegistry._validation_errors = Counter(
                    "drift_detector_validation_errors", "Validation errors encountered"
                )
                _MetricsRegistry._initialized = True
                logger.debug("DriftDetector metrics initialized")
            except ValueError as e:
                # Metrics already registered from previous import
                if "Duplicated timeseries" in str(e):
                    logger.debug("Metrics already registered, retrieving from registry")
                    # Try to retrieve from registry
                    from prometheus_client import REGISTRY

                    for collector in list(REGISTRY._collector_to_names.keys()):
                        if hasattr(collector, "_name"):
                            if collector._name == "drift_detector_drift_value":
                                _MetricsRegistry._drift_value_gauge = collector
                            elif collector._name == "drift_detector_realignments_total":
                                _MetricsRegistry._drift_events = collector
                            elif collector._name == "drift_detector_checks_total":
                                _MetricsRegistry._drift_checks = collector
                            elif (
                                collector._name
                                == "drift_detector_check_latency_seconds"
                            ):
                                _MetricsRegistry._drift_latency = collector
                            elif (
                                collector._name
                                == "drift_detector_realignment_operations"
                            ):
                                _MetricsRegistry._realignment_operations = collector
                            elif collector._name == "drift_detector_validation_errors":
                                _MetricsRegistry._validation_errors = collector
                    _MetricsRegistry._initialized = True
                else:
                    raise
        elif not PROMETHEUS_AVAILABLE and not _MetricsRegistry._initialized:
            # Note: Use _NoOpMetric instances instead of None when Prometheus is not available
            # This prevents AttributeError when using @drift_latency.time() decorator
            _MetricsRegistry._drift_value_gauge = _NoOpMetric()
            _MetricsRegistry._drift_events = _NoOpMetric()
            _MetricsRegistry._drift_checks = _NoOpMetric()
            _MetricsRegistry._drift_latency = _NoOpMetric()
            _MetricsRegistry._realignment_operations = _NoOpMetric()
            _MetricsRegistry._validation_errors = _NoOpMetric()
            _MetricsRegistry._initialized = True

    @property
    def drift_value_gauge(self):
        return _MetricsRegistry._drift_value_gauge

    @property
    def drift_events(self):
        return _MetricsRegistry._drift_events

    @property
    def drift_checks(self):
        return _MetricsRegistry._drift_checks

    @property
    def drift_latency(self):
        return _MetricsRegistry._drift_latency

    @property
    def realignment_operations(self):
        return _MetricsRegistry._realignment_operations

    @property
    def validation_errors(self):
        return _MetricsRegistry._validation_errors


# Create singleton instance
_metrics = _MetricsRegistry()

# Module-level references for backward compatibility
drift_value_gauge = _metrics.drift_value_gauge
drift_events = _metrics.drift_events
drift_checks = _metrics.drift_checks
drift_latency = _metrics.drift_latency
realignment_operations = _metrics.realignment_operations
validation_errors = _metrics.validation_errors

# Constants
MIN_DRIFT_THRESHOLD = 0.0
MAX_DRIFT_THRESHOLD = 2.0
MIN_HISTORY_SIZE = 1
MAX_HISTORY_SIZE = 1000
MAX_EMBEDDINGS = 100000
REALIGNMENT_METHODS = ["center", "pca", "procrustes"]


@dataclass
class DriftMetrics:
    """Drift metrics for a single check."""

    timestamp: datetime
    drift: float
    avg_drift: float
    embeddings_count: int
    dimension: int
    realignment_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift": self.drift,
            "avg_drift": self.avg_drift,
            "embeddings_count": self.embeddings_count,
            "dimension": self.dimension,
            "realignment_triggered": self.realignment_triggered,
        }


class DriftDetector:
    """
    Production-ready drift detector with:
    - Thread-safe operations
    - Input validation
    - Actual realignment implementation
    - FAISS-based similarity analysis
    - Comprehensive metrics
    - Proper cleanup
    """

    def __init__(
        self,
        dim: int = 128,
        drift_threshold: float = 0.1,
        history: int = 5,
        realignment_method: str = "center",
    ):
        """
        Initialize drift detector.

        Args:
            dim: Embedding dimension (must be > 0)
            drift_threshold: Drift threshold for triggering realignment (0-2)
            history: Number of drift measurements to track (1-1000)
            realignment_method: Method for realignment ("center", "pca", "procrustes")

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate dimension
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")

        # Validate drift threshold
        if not isinstance(drift_threshold, (int, float)):
            raise ValueError(
                f"drift_threshold must be numeric, got {type(drift_threshold)}"
            )

        if not (MIN_DRIFT_THRESHOLD <= drift_threshold <= MAX_DRIFT_THRESHOLD):
            raise ValueError(
                f"drift_threshold must be in [{MIN_DRIFT_THRESHOLD}, {MAX_DRIFT_THRESHOLD}], "
                f"got {drift_threshold}"
            )

        # Validate history
        if not isinstance(history, int) or history < MIN_HISTORY_SIZE:
            raise ValueError(f"history must be >= {MIN_HISTORY_SIZE}, got {history}")

        if history > MAX_HISTORY_SIZE:
            logger.warning(f"history {history} exceeds max {MAX_HISTORY_SIZE}, capping")
            history = MAX_HISTORY_SIZE

        # Validate realignment method
        if realignment_method not in REALIGNMENT_METHODS:
            raise ValueError(
                f"realignment_method must be one of {REALIGNMENT_METHODS}, "
                f"got {realignment_method}"
            )

        # Configuration
        self.dim = dim
        self.drift_threshold = float(drift_threshold)
        self.history_size = history
        self.realignment_method = realignment_method

        # Thread safety
        self.lock = threading.RLock()

        # State
        self.previous_embeddings: Optional[np.ndarray] = None
        self.reference_embeddings: Optional[np.ndarray] = None  # For realignment
        self.drift_history: deque = deque(maxlen=history)
        self.metrics_history: List[DriftMetrics] = []

        # FAISS index for similarity search
        self.faiss_index = faiss.IndexFlatIP(dim)

        # Statistics
        self.total_checks = 0
        self.total_realignments = 0
        self.last_realignment: Optional[datetime] = None

        logger.info(
            f"DriftDetector initialized: dim={dim}, threshold={drift_threshold}, "
            f"history={history}, method={realignment_method}"
        )

    def _validate_embeddings(
        self, embeddings: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate embeddings array.

        Returns:
            (is_valid, error_message) tuple
        """
        if embeddings is None:
            return False, "Embeddings cannot be None"

        if not isinstance(embeddings, np.ndarray):
            return False, f"Embeddings must be ndarray, got {type(embeddings)}"

        if embeddings.ndim != 2:
            return False, f"Embeddings must be 2D, got {embeddings.ndim}D"

        if embeddings.shape[0] == 0:
            return False, "Embeddings array is empty"

        if embeddings.shape[1] != self.dim:
            return False, (
                f"Embeddings dimension mismatch: expected {self.dim}, "
                f"got {embeddings.shape[1]}"
            )

        if embeddings.shape[0] > MAX_EMBEDDINGS:
            return False, (
                f"Too many embeddings: {embeddings.shape[0]} > {MAX_EMBEDDINGS}"
            )

        # Check for NaN or Inf
        if not np.isfinite(embeddings).all():
            return False, "Embeddings contain NaN or Inf values"

        return True, None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length with proper zero handling.

        Args:
            embeddings: Input embeddings

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Handle zero vectors
        zero_mask = norms.squeeze() < 1e-10
        if np.any(zero_mask):
            logger.warning(f"Found {np.sum(zero_mask)} zero or near-zero vectors")
            # Replace zero vectors with small random vectors
            embeddings = embeddings.copy()
            embeddings[zero_mask] = np.random.randn(np.sum(zero_mask), self.dim) * 1e-6
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add epsilon to prevent division by zero
        norms = norms + 1e-8

        return embeddings / norms

    @drift_latency.time()
    def track_drift(self, embeddings: np.ndarray) -> float:
        """
        Compute mean cosine drift between previous and current embeddings.

        Args:
            embeddings: Embeddings array of shape (n_agents, dim)

        Returns:
            Drift value (cosine distance)

        Raises:
            ValueError: If embeddings are invalid
        """
        # Validate input
        valid, error = self._validate_embeddings(embeddings)
        if not valid:
            validation_errors.inc()
            raise ValueError(f"Invalid embeddings: {error}")

        with self.lock:
            drift_checks.inc()
            self.total_checks += 1

            current = self._normalize(embeddings)

            # First batch - initialize
            if self.previous_embeddings is None:
                self.previous_embeddings = current.copy()
                self.reference_embeddings = current.copy()
                drift_value_gauge.set(0.0)

                # Add to drift history
                self.drift_history.append(0.0)

                # Record metrics
                metrics = DriftMetrics(
                    timestamp=datetime.utcnow(),
                    drift=0.0,
                    avg_drift=0.0,
                    embeddings_count=current.shape[0],
                    dimension=self.dim,
                )
                self.metrics_history.append(metrics)

                logger.info("[DriftDetector] First embedding batch; drift=0.0")
                return 0.0

            prev = self.previous_embeddings

            # Handle shape mismatch
            n_current = current.shape[0]
            n_prev = prev.shape[0]

            if n_current != n_prev:
                logger.warning(
                    f"[DriftDetector] Shape mismatch: current={n_current}, "
                    f"previous={n_prev}, using min={min(n_current, n_prev)}"
                )

            n = min(n_current, n_prev)
            current_aligned = current[:n]
            prev_aligned = prev[:n]

            # Compute cosine similarities
            sims = np.sum(current_aligned * prev_aligned, axis=1)

            # Cosine distance
            drift = float(np.mean(1.0 - sims))

            # Update history
            self.drift_history.append(drift)

            # Update state
            self.previous_embeddings = current.copy()

            # Update metrics
            drift_value_gauge.set(drift)

            # Record detailed metrics
            metrics = DriftMetrics(
                timestamp=datetime.utcnow(),
                drift=drift,
                avg_drift=self.mean_drift(),
                embeddings_count=n,
                dimension=self.dim,
            )
            self.metrics_history.append(metrics)

            logger.debug(f"[DriftDetector] Drift computed: {drift:.4f}")

            return drift

    def mean_drift(self) -> float:
        """
        Get moving average drift.

        Returns:
            Mean drift over history window
        """
        with self.lock:
            if not self.drift_history:
                return 0.0
            return float(np.mean(list(self.drift_history)))

    def realign_embeddings(
        self, embeddings: np.ndarray, method: Optional[str] = None
    ) -> np.ndarray:
        """
        Realign embeddings to reduce drift.

        Args:
            embeddings: Embeddings to realign
            method: Realignment method (uses instance default if None)

        Returns:
            Realigned embeddings

        Raises:
            ValueError: If embeddings invalid or method unknown
        """
        # Validate input
        valid, error = self._validate_embeddings(embeddings)
        if not valid:
            raise ValueError(f"Invalid embeddings: {error}")

        method = method or self.realignment_method

        if method not in REALIGNMENT_METHODS:
            raise ValueError(f"Unknown realignment method: {method}")

        with self.lock:
            if self.reference_embeddings is None:
                logger.warning("No reference embeddings, using current as reference")
                self.reference_embeddings = embeddings.copy()
                return embeddings

            if method == "center":
                # Center-based realignment
                realigned = self._realign_center(embeddings)
            elif method == "pca":
                # PCA-based realignment
                realigned = self._realign_pca(embeddings)
            elif method == "procrustes":
                # Procrustes alignment
                realigned = self._realign_procrustes(embeddings)
            else:
                realigned = embeddings  # Fallback

            # Update statistics
            self.total_realignments += 1
            self.last_realignment = datetime.utcnow()
            realignment_operations.inc()

            logger.info(f"Realigned embeddings using method: {method}")

            return realigned

    def _realign_center(self, embeddings: np.ndarray) -> np.ndarray:
        """Center-based realignment (subtract mean, match reference scale)."""
        # Center current embeddings
        current_mean = np.mean(embeddings, axis=0)
        centered = embeddings - current_mean

        # Match reference scale
        if self.reference_embeddings is not None:
            ref_mean = np.mean(self.reference_embeddings, axis=0)
            ref_std = np.std(self.reference_embeddings, axis=0) + 1e-8
            current_std = np.std(centered, axis=0) + 1e-8

            # Scale to match reference
            scaled = centered * (ref_std / current_std)

            # Shift to reference center
            realigned = scaled + ref_mean
        else:
            realigned = centered

        return realigned

    def _realign_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA-based realignment to principal axes of reference."""
        if self.reference_embeddings is None:
            return embeddings

        # Compute PCA of reference
        ref_centered = self.reference_embeddings - np.mean(
            self.reference_embeddings, axis=0
        )
        ref_cov = np.cov(ref_centered, rowvar=False)
        ref_eigvals, ref_eigvecs = np.linalg.eigh(ref_cov)

        # Sort by eigenvalues
        idx = np.argsort(ref_eigvals)[::-1]
        ref_eigvecs = ref_eigvecs[:, idx]

        # Project current embeddings
        curr_centered = embeddings - np.mean(embeddings, axis=0)
        projected = curr_centered @ ref_eigvecs

        # Project back
        realigned = projected @ ref_eigvecs.T
        realigned += np.mean(self.reference_embeddings, axis=0)

        return realigned

    def _realign_procrustes(self, embeddings: np.ndarray) -> np.ndarray:
        """Procrustes alignment to reference embeddings."""
        if self.reference_embeddings is None:
            return embeddings

        # Use min size
        n = min(embeddings.shape[0], self.reference_embeddings.shape[0])
        X = embeddings[:n]
        Y = self.reference_embeddings[:n]

        # Center both
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # Compute optimal rotation via SVD
        H = X_centered.T @ Y_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply transformation to all embeddings
        realigned = (embeddings - X_mean) @ R + Y_mean

        return realigned

    def realign_if_drift(
        self, embeddings: np.ndarray, agents: List[str]
    ) -> Dict[str, Any]:
        """
        Check drift and perform realignment if needed.

        Args:
            embeddings: Current embeddings
            agents: Agent identifiers

        Returns:
            Dictionary with drift info and realigned embeddings

        Raises:
            ValueError: If inputs invalid
        """
        # Validate inputs
        valid, error = self._validate_embeddings(embeddings)
        if not valid:
            raise ValueError(f"Invalid embeddings: {error}")

        if not isinstance(agents, list):
            raise ValueError("agents must be a list")

        if len(agents) != embeddings.shape[0]:
            raise ValueError(
                f"Agent count mismatch: {len(agents)} agents but "
                f"{embeddings.shape[0]} embeddings"
            )

        with self.lock:
            # Compute drift
            drift = self.track_drift(embeddings)
            avg_drift = self.mean_drift()

            # Check if realignment needed
            realignment_needed = avg_drift > self.drift_threshold

            if realignment_needed:
                drift_events.inc()
                logger.warning(
                    f"[DriftDetector] Drift threshold exceeded "
                    f"({avg_drift:.4f} > {self.drift_threshold}), "
                    f"triggering realignment for {len(agents)} agents"
                )

                # Perform realignment
                realigned_embeddings = self.realign_embeddings(embeddings)

                # Update state with realigned embeddings
                self.previous_embeddings = self._normalize(realigned_embeddings)

                # Update metrics
                if self.metrics_history:
                    self.metrics_history[-1].realignment_triggered = True

                agents_to_realign = agents.copy()
            else:
                realigned_embeddings = embeddings
                agents_to_realign = []

            result = {
                "drift": float(drift),
                "current_drift": float(drift),  # Add this for compatibility
                "avg_drift": float(avg_drift),
                "mean_drift": float(avg_drift),  # Add this for compatibility
                "realignment_needed": realignment_needed,
                "agents_to_realign": agents_to_realign,
                "realigned_embeddings": realigned_embeddings,
                "total_realignments": self.total_realignments,
                "last_realignment": (
                    self.last_realignment.isoformat() if self.last_realignment else None
                ),
            }

            return result

    def faiss_pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity using FAISS.

        Args:
            embeddings: Embeddings array of shape (n_agents, dim)

        Returns:
            Similarity matrix of shape (n_agents, n_agents)

        Raises:
            ValueError: If embeddings invalid
        """
        # Validate input
        valid, error = self._validate_embeddings(embeddings)
        if not valid:
            raise ValueError(f"Invalid embeddings: {error}")

        with self.lock:
            # Normalize embeddings
            embs = self._normalize(embeddings).astype("float32")

            # Compute pairwise similarities directly using dot products
            # This is more reliable than FAISS search for similarity matrix
            sims = embs @ embs.T

            logger.debug(f"Computed similarity matrix for {len(embs)} embeddings")

            return sims

    def find_drift_outliers(
        self, embeddings: np.ndarray, threshold: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find outlier embeddings based on similarity to reference.

        Args:
            embeddings: Current embeddings
            threshold: Outlier threshold (std deviations)

        Returns:
            (outlier_indices, outlier_scores) tuple

        Raises:
            ValueError: If embeddings invalid
        """
        # Validate input
        valid, error = self._validate_embeddings(embeddings)
        if not valid:
            raise ValueError(f"Invalid embeddings: {error}")

        with self.lock:
            if self.reference_embeddings is None:
                logger.warning("No reference embeddings for outlier detection")
                return np.array([]), np.array([])

            # Normalize both
            current = self._normalize(embeddings)
            reference = self._normalize(self.reference_embeddings)

            # Use FAISS to find nearest reference for each current
            temp_index = faiss.IndexFlatIP(self.dim)
            temp_index.add(reference.astype("float32"))

            sims, _ = temp_index.search(current.astype("float32"), 1)
            similarities = sims.squeeze()

            # Compute outlier scores (lower similarity = higher score)
            scores = 1.0 - similarities

            # Find outliers using z-score
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            if std_score < 1e-8:
                # No variation
                return np.array([]), np.array([])

            z_scores = (scores - mean_score) / std_score
            outlier_mask = z_scores > threshold

            outlier_indices = np.where(outlier_mask)[0]
            outlier_scores = scores[outlier_mask]

            logger.info(
                f"Found {len(outlier_indices)} outliers (threshold={threshold} std)"
            )

            return outlier_indices, outlier_scores

    def drift_report(self) -> Dict[str, Any]:
        """
        Get comprehensive drift report.

        Returns:
            Report dictionary with statistics and history
        """
        with self.lock:
            report = {
                "current_drift": (
                    float(self.drift_history[-1]) if self.drift_history else 0.0
                ),
                "mean_drift": self.mean_drift(),
                "drift_history": list(self.drift_history),
                "threshold": self.drift_threshold,
                "dimension": self.dim,
                "history_size": self.history_size,
                "total_checks": self.total_checks,
                "total_realignments": self.total_realignments,
                "last_realignment": (
                    self.last_realignment.isoformat() if self.last_realignment else None
                ),
                "realignment_method": self.realignment_method,
                "has_reference": self.reference_embeddings is not None,
                "metrics_count": len(self.metrics_history),
            }

            return report

    def get_detailed_metrics(self) -> List[Dict[str, Any]]:
        """Get detailed metrics history."""
        with self.lock:
            return [m.to_dict() for m in self.metrics_history]

    def reset(self):
        """Reset detector state (keeps configuration)."""
        with self.lock:
            self.previous_embeddings = None
            self.reference_embeddings = None
            self.drift_history.clear()
            self.metrics_history.clear()
            self.faiss_index.reset()
            self.total_checks = 0
            self.total_realignments = 0
            self.last_realignment = None

            logger.info("DriftDetector state reset")

    def cleanup(self):
        """
        Cleanup resources (FAISS index).
        Call this when done to free memory.
        """
        with self.lock:
            try:
                self.faiss_index.reset()
                self.previous_embeddings = None
                self.reference_embeddings = None
                logger.info("DriftDetector cleaned up")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# Demo and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Drift Detector - Production Demo")
    print("=" * 60)

    # Create detector
    detector = DriftDetector(
        dim=128, drift_threshold=0.15, history=5, realignment_method="procrustes"
    )

    # Test 1: Basic drift tracking
    print("\n1. Basic Drift Tracking")
    np.random.seed(42)

    embeddings1 = np.random.randn(10, 128)
    drift1 = detector.track_drift(embeddings1)
    print(f"   First drift: {drift1:.4f} (should be 0)")

    embeddings2 = embeddings1 + np.random.randn(10, 128) * 0.1
    drift2 = detector.track_drift(embeddings2)
    print(f"   Second drift: {drift2:.4f}")

    embeddings3 = embeddings1 + np.random.randn(10, 128) * 0.3
    drift3 = detector.track_drift(embeddings3)
    print(f"   Third drift: {drift3:.4f}")
    print(f"   Mean drift: {detector.mean_drift():.4f}")

    # Test 2: Realignment
    print("\n2. Drift Detection and Realignment")

    # Create high drift
    embeddings_drifted = embeddings1 + np.random.randn(10, 128) * 0.5
    agents = [f"agent_{i}" for i in range(10)]

    result = detector.realign_if_drift(embeddings_drifted, agents)

    print(f"   Drift: {result['drift']:.4f}")
    print(f"   Avg drift: {result['avg_drift']:.4f}")
    print(f"   Realignment needed: {result['realignment_needed']}")
    print(f"   Agents to realign: {len(result['agents_to_realign'])}")

    if result["realignment_needed"]:
        print(f"   Realigned embeddings shape: {result['realigned_embeddings'].shape}")

    # Test 3: FAISS similarity
    print("\n3. FAISS Pairwise Similarity")

    test_embs = np.random.randn(5, 128)
    sim_matrix = detector.faiss_pairwise_similarity(test_embs)

    print(f"   Similarity matrix shape: {sim_matrix.shape}")
    print(f"   Diagonal (self-similarity): {np.diag(sim_matrix)}")

    # Test 4: Outlier detection
    print("\n4. Outlier Detection")

    # Create some outliers
    normal_embs = np.random.randn(20, 128)
    outlier_embs = np.random.randn(3, 128) * 3  # Much larger variance
    all_embs = np.vstack([normal_embs, outlier_embs])

    detector.reset()
    detector.track_drift(normal_embs)  # Set reference

    outlier_idx, outlier_scores = detector.find_drift_outliers(all_embs, threshold=2.0)

    print(f"   Total embeddings: {len(all_embs)}")
    print(f"   Outliers found: {len(outlier_idx)}")
    print(f"   Outlier indices: {outlier_idx}")
    print(f"   Outlier scores: {outlier_scores}")

    # Test 5: Validation
    print("\n5. Input Validation")

    try:
        # Wrong dimension
        bad_embs = np.random.randn(10, 64)
        detector.track_drift(bad_embs)
        print("   Dimension validation: FAILED (should raise)")
    except ValueError as e:
        print(f"   Dimension validation: PASSED ({str(e)[:50]}...)")

    try:
        # Empty array
        detector.track_drift(np.array([]).reshape(0, 128))
        print("   Empty validation: FAILED (should raise)")
    except ValueError as e:
        print(f"   Empty validation: PASSED ({str(e)[:50]}...)")

    # Test 6: Report
    print("\n6. Drift Report")

    report = detector.drift_report()
    print(f"   Current drift: {report['current_drift']:.4f}")
    print(f"   Mean drift: {report['mean_drift']:.4f}")
    print(f"   Total checks: {report['total_checks']}")
    print(f"   Total realignments: {report['total_realignments']}")
    print(f"   Has reference: {report['has_reference']}")

    # Test 7: Cleanup
    print("\n7. Cleanup")
    detector.cleanup()
    print("   Cleanup complete")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
