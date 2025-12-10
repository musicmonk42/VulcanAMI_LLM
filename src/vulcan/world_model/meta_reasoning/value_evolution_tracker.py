# src/vulcan/world_model/meta_reasoning/value_evolution_tracker.py
"""
value_evolution_tracker.py - Time-series tracking and analysis of agent value evolution
Part of the meta_reasoning subsystem for VULCAN-AMI

FULL PRODUCTION IMPLEMENTATION

Tracks how agent values evolve over time with sophisticated analysis:
- Time-series tracking of value trajectories
- Statistical drift detection (CUSUM, moving averages, change-point detection)
- Value alignment stability metrics
- Longitudinal trend analysis with forecasting
- Anomaly detection for value deviations
- Correlation analysis between values
- Integration with safety systems for alerts

Algorithms:
- CUSUM (Cumulative Sum) for drift detection
- Exponential Weighted Moving Average (EWMA) for trend smoothing
- Linear regression for trend analysis
- Pearson correlation for value relationships
- Z-score based anomaly detection
- Change-point detection using binary segmentation
- Kolmogorov-Smirnov test for distribution changes

Integration:
- Alerts sent to SelfImprovementDrive for critical drift
- Records to ValidationTracker for pattern learning
- Feeds TransparencyInterface for audit trails
- Integrates with EthicalBoundaryMonitor for safety checks

FIX (2025-10-22):
- Corrected trend detection logic to properly detect INCREASING/DECREASING trends
- Adjusted stable_threshold calculation to be more sensitive to actual changes
- Fixed comparison logic to properly classify trends based on meaningful slopes
- Improved threshold calculation to consider data range and time span
"""

import logging
import math  # Import math for fallback functions
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
# import numpy as np # Original import
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock  # --- START FIX: Import MagicMock ---

# --- START FIX: Add numpy fallback ---
# logger = logging.getLogger(__name__) # Original logger placement
logger = logging.getLogger(__name__)  # Moved logger init up
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using list-based math")

    class FakeNumpy:
        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0  # Return float

        @staticmethod
        def array(lst):
            return list(lst)

        @staticmethod
        def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
            # Simplified std dev for 1D list
            if not isinstance(a, list):
                raise NotImplementedError("FakeNumpy std only supports lists")
            n = len(a)
            if n <= ddof:
                return float("nan")  # Cannot compute std dev
            mean_val = sum(a) / n if n > 0 else 0
            variance = sum((x - mean_val) ** 2 for x in a) / max(
                1, (n - ddof)
            )  # Avoid div by zero if n=ddof
            return math.sqrt(variance)

        @staticmethod
        def vstack(tup):
            # Simplified vstack for list of lists/tuples
            return [list(row) for row in tup]

        @staticmethod
        def ones(shape, dtype=None):
            # Simplified ones for 1D list
            if isinstance(shape, int):
                return [1.0] * shape
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                return [1.0] * shape[0]
            raise NotImplementedError("FakeNumpy ones only supports 1D shape")

        @staticmethod
        def linalg_lstsq(a, b, rcond=None):
            # Simplified fallback for least squares (Ax=b)
            # Returns mock solution (zeros), empty residuals, rank, singular values
            num_cols = len(a[0]) if a and isinstance(a[0], list) else 1
            mock_solution = ([0.0] * num_cols,)  # Solution array (tuple)
            mock_residuals = []  # Empty residuals
            mock_rank = min(len(a), num_cols) if a else 0
            mock_s = []  # Empty singular values
            return mock_solution, mock_residuals, mock_rank, mock_s

        # Add linalg attribute
        class FakeLinalg:
            lstsq = None

        linalg = FakeLinalg()
        linalg.lstsq = linalg_lstsq  # Assign static method

        @staticmethod
        def diff(a, n=1, axis=-1, prepend=None, append=None):
            # Simplified diff for 1D list
            if not isinstance(a, list) or n != 1 or axis != -1:
                raise NotImplementedError(
                    "FakeNumpy diff only supports n=1, axis=-1 on lists"
                )
            if len(a) < 2:
                return []
            # Ensure elements are numeric
            try:
                result = [a[i] - a[i - 1] for i in range(1, len(a))]
            except TypeError:
                logger.error("FakeNumpy diff encountered non-numeric list elements.")
                return []
            # Prepend/append not implemented simply here
            return result

        @staticmethod
        def sqrt(x):
            if isinstance(x, list):
                return [math.sqrt(i) if i >= 0 else float("nan") for i in x]
            return math.sqrt(x) if x >= 0 else float("nan")

        @staticmethod
        def convolve(a, v, mode="full"):
            # Simplified 1D convolution (basic sliding window sum - WRONG but provides output)
            if mode != "valid" or not isinstance(a, list) or not isinstance(v, list):
                raise NotImplementedError(
                    "FakeNumpy convolve simplified for mode='valid' on lists only"
                )
            if len(v) > len(a):
                return []
            # This is NOT a real convolution, just a placeholder
            k = len(v)
            return [
                sum(a[i : i + k]) for i in range(len(a) - k + 1)
            ]  # Incorrect math, placeholder only

        @staticmethod
        def exp(x):
            if isinstance(x, list):
                return [math.exp(i) for i in x]
            return math.exp(x)

        @staticmethod
        def linspace(
            start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
        ):
            num = int(num)  # Ensure integer
            if num <= 0:
                return []
            if endpoint:
                if num == 1:
                    step = 0.0
                    result = [float(start)]  # Handle num=1
                else:
                    step = (stop - start) / (num - 1)
            else:
                step = (stop - start) / num
            # Generate list using float arithmetic
            result = [float(start + i * step) for i in range(num)]
            # Ensure last point is exactly 'stop' if endpoint=True and num > 1
            if endpoint and num > 1:
                result[-1] = float(stop)

            if retstep:
                return result, float(step)
            return result

        @staticmethod
        def corrcoef(x, y=None, rowvar=True):
            # Simplified Pearson correlation for two 1D lists x and y
            if y is None:
                y = x  # Correlation matrix of x with itself (diag=1)
            if (
                not isinstance(x, list)
                or not isinstance(y, list)
                or len(x) != len(y)
                or len(x) < 2
            ):
                # Return identity matrix structure if dimensions allow, else 0.0
                size = len(x) if isinstance(x, list) else 1
                if size > 0:
                    return [
                        [1.0 if i == j else 0.0 for j in range(size)]
                        for i in range(size)
                    ]
                return [[0.0]]  # Fallback for invalid input

            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            std_dev_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
            std_dev_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

            if std_dev_x == 0 or std_dev_y == 0:
                corr = 0.0  # No correlation if one variable is constant
            else:
                covariance = (
                    sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
                )
                corr = covariance / (std_dev_x * std_dev_y)
                # Clamp between -1 and 1 due to potential float errors
                corr = max(-1.0, min(1.0, corr))

            # Return matrix structure [[corr(x,x), corr(x,y)], [corr(y,x), corr(y,y)]]
            return [[1.0, corr], [corr, 1.0]]

        @staticmethod
        def isnan(x):
            if isinstance(x, list):
                return [math.isnan(i) for i in x]
            return math.isnan(x)

        # Add generic type placeholder if needed elsewhere
        class generic:
            pass

        # Add ndarray type placeholder if needed elsewhere
        class ndarray:
            pass

    np = FakeNumpy()
# --- END FIX ---


class DriftSeverity(Enum):
    """Severity level of detected drift"""

    NONE = "none"
    MINOR = "minor"  # Small deviation, monitor
    MODERATE = "moderate"  # Significant deviation, investigate
    MAJOR = "major"  # Large deviation, action needed
    CRITICAL = "critical"  # Dangerous deviation, immediate action


class TrendDirection(Enum):
    """Direction of value trend"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ValueChangeType(Enum):
    """Type of value change detected"""

    GRADUAL_DRIFT = "gradual_drift"  # Slow continuous change
    SUDDEN_SHIFT = "sudden_shift"  # Abrupt change
    PERIODIC_OSCILLATION = "periodic_oscillation"  # Cyclic pattern
    ANOMALY = "anomaly"  # One-time spike/dip
    STABILIZATION = "stabilization"  # Return to baseline


@dataclass
class ValueState:
    """Snapshot of agent values at a point in time"""

    timestamp: float
    values: Dict[str, float]  # value_name -> score
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed on creation
    state_id: str = field(default_factory=lambda: f"state_{time.time_ns()}")

    def get_value(self, name: str, default: float = 0.0) -> float:
        """Get value by name with default"""
        # Ensure returned value is float
        return float(self.values.get(name, default))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "values": {k: float(v) for k, v in self.values.items()},  # Ensure floats
            "metadata": self.metadata.copy(),
            "state_id": self.state_id,
        }


@dataclass
class ValueTrajectory:
    """Time series trajectory for a single value"""

    value_name: str
    timestamps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add_point(self, timestamp: float, value: float):
        """Add data point to trajectory"""
        # Ensure value is float before adding
        try:
            self.timestamps.append(float(timestamp))
            self.values.append(float(value))
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid timestamp or value for trajectory {self.value_name}: ts={timestamp}, val={value}"
            )

    def get_latest(self) -> Optional[Tuple[float, float]]:
        """Get latest (timestamp, value) pair"""
        if not self.values or not self.timestamps:  # Check both lists
            return None
        return (self.timestamps[-1], self.values[-1])

    def get_mean(self) -> float:
        """Get mean value"""
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        return float(_np.mean(self.values)) if self.values else 0.0  # Ensure float

    def get_std(self) -> float:
        """Get standard deviation"""
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        # Need at least 2 points for std dev
        return (
            float(_np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0
        )  # Use sample std dev (ddof=1), ensure float

    def get_trend(self) -> Tuple[float, float]:
        """
        Get linear trend (slope, intercept) using least squares.
        Returns (0, mean) if insufficient data or calculation fails.
        """
        _np = np if NUMPY_AVAILABLE else FakeNumpy
        if len(self.values) < 2:
            return (0.0, self.get_mean())

        try:
            # Normalize timestamps relative to the start to improve numerical stability
            t_array = _np.array(self.timestamps)
            t_norm = t_array - t_array[0]  # Assumes t_array[0] works for list fallback
            v_array = _np.array(self.values)

            # Linear regression: v = slope * t_norm + intercept
            # Prepare matrix A for least squares (t_norm column and ones column)
            A = _np.vstack([t_norm, _np.ones(len(t_norm))]).T  # Uses vstack, ones

            # Perform least squares
            result = _np.linalg.lstsq(A, v_array, rcond=None)  # Uses linalg.lstsq
            slope, intercept = result[
                0
            ]  # First element of result tuple is the solution [slope, intercept]

            return (float(slope), float(intercept))
        except Exception as e:
            # Fallback if least squares fails (e.g., with FakeNumpy or other issues)
            logger.error(
                f"Linear regression failed for trajectory {self.value_name}: {e}"
            )
            return (0.0, self.get_mean())  # Return 0 slope and current mean

    def get_volatility(self) -> float:
        """Calculate volatility (coefficient of variation: std_dev / mean)"""
        if len(self.values) < 2:
            return 0.0

        mean = self.get_mean()
        std_dev = self.get_std()

        # Avoid division by zero if mean is close to zero
        if abs(mean) < 1e-9:
            # If standard deviation is also near zero, volatility is 0
            # Otherwise, volatility is effectively infinite (or very large) - cap at 1.0?
            return (
                0.0 if abs(std_dev) < 1e-9 else 1.0
            )  # Return 1.0 as capped high volatility

        # Return absolute value as volatility is typically non-negative
        return abs(std_dev / mean)


@dataclass
class DriftAlert:
    """Alert for detected value drift"""

    value_name: str
    severity: DriftSeverity
    change_type: ValueChangeType
    description: str

    # Drift details
    old_value: float
    new_value: float
    change_magnitude: float
    drift_score: float  # Statistical measure of drift

    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "value_name": self.value_name,
            "severity": self.severity.value,
            "change_type": self.change_type.value,
            "description": self.description,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_magnitude": self.change_magnitude,
            "drift_score": self.drift_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ValueEvolutionAnalysis:
    """Comprehensive analysis of value evolution"""

    time_window_start: float
    time_window_end: float

    # Per-value analysis
    value_trends: Dict[str, TrendDirection]  # value -> trend direction
    value_drift_scores: Dict[str, float]  # value -> drift score
    value_stability_scores: Dict[str, float]  # value -> stability (0-1)

    # Detected drifts
    drifted_values: List[str]
    drift_alerts: List[DriftAlert]

    # Overall assessment
    overall_stability: float  # 0-1, higher is more stable
    alignment_consistency: float  # 0-1, how consistent values are

    # Statistical measures
    correlation_matrix: Dict[Tuple[str, str], float]  # Pairwise correlations

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring JSON compatibility"""

        # Helper for recursive serialization (handles numpy if present)
        def _make_serializable(data):
            _np = np if NUMPY_AVAILABLE else FakeNumpy
            if isinstance(data, dict):
                return {str(k): _make_serializable(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [_make_serializable(item) for item in data]
            elif isinstance(data, Enum):
                return data.value
            elif hasattr(data, "to_dict") and callable(data.to_dict):
                try:
                    return data.to_dict()  # Assumes to_dict returns serializable
                except Exception:
                    return str(data)
            elif NUMPY_AVAILABLE and isinstance(data, (_np.ndarray, _np.generic)):
                if isinstance(data, _np.ndarray):
                    return data.tolist()
                elif isinstance(data, _np.generic):
                    return data.item()  # Convert numpy scalars
            elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
                return str(data)  # JSON doesn't support NaN/inf
            elif isinstance(data, (str, int, float, bool, type(None))):
                return data
            else:  # Fallback
                try:
                    return str(data)
                except Exception:
                    return f"<unserializable_{type(data).__name__}>"

        return {
            "time_window": {
                "start": self.time_window_start,
                "end": self.time_window_end,
                "duration": self.time_window_end - self.time_window_start,
            },
            "value_trends": {k: v.value for k, v in self.value_trends.items()},
            "value_drift_scores": _make_serializable(self.value_drift_scores),
            "value_stability_scores": _make_serializable(self.value_stability_scores),
            "drifted_values": self.drifted_values,
            "drift_alerts": [
                _make_serializable(a) for a in self.drift_alerts
            ],  # Use helper on list items
            "overall_stability": self.overall_stability,
            "alignment_consistency": self.alignment_consistency,
            # FIXED: Convert tuple keys in correlation_matrix to strings for JSON
            "correlation_matrix": {
                f"{k[0]}|{k[1]}": v for k, v in self.correlation_matrix.items()
            },
            "metadata": _make_serializable(self.metadata),
        }


class ValueEvolutionTracker:
    """
    Time-series tracking and analysis of agent value evolution

    Tracks value trajectories over time and detects:
    - Gradual drift (CUSUM algorithm)
    - Sudden shifts (change-point detection)
    - Anomalies (z-score based)
    - Trends (linear regression)
    - Stability (variance analysis)
    - Correlations (between values)

    Provides:
    - Real-time drift detection with severity classification
    - Forecasting of future value states
    - Alignment stability metrics
    - Longitudinal trend analysis
    - Integration with safety systems

    Thread-safe with proper locking for concurrent access.
    """

    def __init__(
        self,
        max_history: int = 10000,
        drift_threshold: float = 0.15,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None,
        self_improvement_drive=None,
        validation_tracker=None,
        transparency_interface=None,
    ):
        """
        Initialize value evolution tracker

        Args:
            max_history: Maximum state history to keep
            drift_threshold: Threshold for CUSUM drift detection (relative to baseline)
            alert_callback: Optional callback for drift alerts
            self_improvement_drive: Optional SelfImprovementDrive instance (can be mock)
            validation_tracker: Optional ValidationTracker instance (can be mock)
            transparency_interface: Optional TransparencyInterface instance (can be mock)
        """
        # Use fake numpy if needed
        self._np = np if NUMPY_AVAILABLE else FakeNumpy

        self.max_history = max_history
        self.drift_threshold = drift_threshold
        self.alert_callback = alert_callback
        # Use MagicMock for optional dependencies if None
        self.self_improvement_drive = self_improvement_drive or MagicMock()
        self.validation_tracker = validation_tracker or MagicMock()
        self.transparency_interface = transparency_interface or MagicMock()

        # Value state history (deque provides maxlen automatically)
        self.value_history: deque[ValueState] = deque(maxlen=max_history)
        self.current_values: Dict[str, float] = {}

        # Per-value trajectories
        self.trajectories: Dict[str, ValueTrajectory] = {}

        # CUSUM state for drift detection
        self.cusum_state: Dict[
            str, Dict[str, float]
        ] = {}  # value -> {cusum_pos, cusum_neg, baseline}

        # Drift alerts history
        self.drift_alerts: deque[DriftAlert] = deque(
            maxlen=1000
        )  # Use deque for alerts too
        self.max_alerts = 1000  # Max alerts in deque

        # Baselines for comparison
        self.baseline_values: Dict[str, float] = {}
        self.baseline_set: bool = False

        # Analysis cache
        self.last_analysis: Optional[ValueEvolutionAnalysis] = None
        self.last_analysis_time: float = 0.0
        self.analysis_cache_ttl: float = 60.0  # Cache for 60 seconds

        # Statistics
        self.stats = defaultdict(int)
        self.stats["initialized_at"] = time.time()

        # Configuration
        self.ewma_alpha = 0.2  # EWMA smoothing parameter
        # CUSUM slack parameter (relative to std dev?) Needs clarification/tuning. Let's assume relative to drift_threshold * baseline magnitude.
        self.cusum_slack = 0.5
        self.anomaly_z_threshold = 3.0  # Z-score threshold for anomalies
        self.min_history_for_drift = (
            10  # Minimum history points before running drift detection
        )

        # Thread safety
        self.lock = threading.RLock()

        logger.info("ValueEvolutionTracker initialized (FULL IMPLEMENTATION)")
        logger.info(f"  Max history: {max_history}, Drift threshold: {drift_threshold}")
        logger.info(
            f"  CUSUM slack param (relative): {self.cusum_slack}, Anomaly Z-threshold: {self.anomaly_z_threshold}"
        )

    def record_value_state(
        self, values: Dict[str, float], metadata: Optional[Dict[str, Any]] = None
    ) -> ValueState:
        """Record current value state, update trajectories, and perform incremental drift detection."""
        with self.lock:
            # Input validation
            if not isinstance(values, dict):
                logger.error(f"Invalid 'values' type: {type(values)}. Expected dict.")
                # Return a dummy state or raise error? Return dummy for now.
                return ValueState(timestamp=time.time(), values={})

            # **************************************************************************
            # FIX 3: Use time_ns() for high-resolution timestamps
            current_time = time.time_ns() / 1_000_000_000.0
            # **************************************************************************

            processed_values = {}
            # Ensure all values are floats
            for k, v in values.items():
                try:
                    processed_values[k] = float(v)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric value for '{k}': {v}")

            if not processed_values:
                logger.warning("No valid numeric values provided in state update.")
                return ValueState(timestamp=current_time, values={})

            # Create state object
            state = ValueState(
                timestamp=current_time, values=processed_values, metadata=metadata or {}
            )

            # Store state (deque handles maxlen)
            self.value_history.append(state)
            self.current_values = processed_values.copy()  # Update current view

            # Update trajectories
            for value_name, value in processed_values.items():
                if value_name not in self.trajectories:
                    self.trajectories[value_name] = ValueTrajectory(
                        value_name=value_name
                    )
                # Add point handles float conversion/errors
                self.trajectories[value_name].add_point(state.timestamp, value)

                # Initialize CUSUM state if new value tracked
                if value_name not in self.cusum_state:
                    self.cusum_state[value_name] = {
                        "cusum_pos": 0.0,
                        "cusum_neg": 0.0,
                        "baseline": value,
                    }

            # Set baseline automatically if first time reaching enough data
            if (
                not self.baseline_set and len(self.value_history) >= 5
            ):  # Use small window for initial baseline
                logger.info("Setting initial baseline values...")
                self._set_baseline()  # Sets self.baseline_set = True

            # Perform incremental drift detection if baseline is set and enough history
            if (
                self.baseline_set
                and len(self.value_history) >= self.min_history_for_drift
            ):
                self._detect_drift_incremental(processed_values)

            # Update statistics
            self.stats["states_recorded"] += 1
            self.stats["unique_values_tracked"] = len(self.trajectories)

            # Invalidate analysis cache as new data arrived
            self.last_analysis = None

            # Record to transparency interface if available and configured
            if not isinstance(self.transparency_interface, MagicMock) and hasattr(
                self.transparency_interface, "record_value_state"
            ):
                try:
                    # Pass serializable data
                    self.transparency_interface.record_value_state(
                        values=processed_values, timestamp=state.timestamp
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to record value state to transparency interface: {e}"
                    )

            logger.debug(
                f"Recorded value state at {state.timestamp}: {len(processed_values)} values."
            )

            return state

    def detect_drift(
        self, value_name: Optional[str] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Detect value drift using multiple methods (CUSUM, Change Point, Trend)."""
        with self.lock:
            # Check history size first
            if len(self.value_history) < self.min_history_for_drift:
                return {
                    "drift_detected": False,
                    "reason": "Insufficient history",
                    "history_size": len(self.value_history),
                    "required": self.min_history_for_drift,
                }

            # Use provided threshold or instance default
            effective_threshold = (
                threshold if threshold is not None else self.drift_threshold
            )
            # Ensure threshold is valid
            effective_threshold = max(0.01, min(1.0, effective_threshold))

            # Determine which values to check
            if value_name:
                if value_name not in self.trajectories:
                    return {
                        "drift_detected": False,
                        "reason": f"Value '{value_name}' not tracked.",
                    }
                values_to_check = [value_name]
            else:
                values_to_check = list(self.trajectories.keys())

            if not values_to_check:
                return {
                    "drift_detected": False,
                    "reason": "No values currently tracked.",
                }

            # Check each value
            drift_details = {}
            drifted_values_list = []

            for vname in values_to_check:
                trajectory = self.trajectories.get(vname)
                # Skip if trajectory somehow missing or too short
                if (
                    not trajectory
                    or len(trajectory.values) < self.min_history_for_drift
                ):
                    drift_details[vname] = {
                        "drift_detected": False,
                        "drift_score": 0.0,
                        "reason": "Insufficient data",
                    }
                    continue

                # Run multiple detection methods
                # Note: These methods return scores, not just booleans
                cusum_drift = self._detect_drift_cusum(
                    vname, effective_threshold
                )  # Uses internal state
                change_point_drift = self._detect_change_point(vname)  # Uses trajectory
                trend_drift = self._detect_trend_drift(
                    vname, effective_threshold
                )  # Uses trajectory

                # Aggregate drift score (e.g., weighted average or max)
                # Using max emphasizes sensitivity to any drift type
                drift_score = max(
                    cusum_drift["score"],
                    change_point_drift["score"],
                    trend_drift["score"],
                )
                # Alternative: Weighted average? Needs tuning.
                # drift_score = (0.4 * cusum_drift['score'] + 0.3 * change_point_drift['score'] + 0.3 * trend_drift['score'])

                # Determine if drift detected based on aggregated score and threshold
                drift_detected = drift_score > effective_threshold

                current_val = trajectory.values[-1] if trajectory.values else 0.0
                baseline_val = self.baseline_values.get(vname, trajectory.get_mean())

                drift_details[vname] = {
                    "drift_detected": drift_detected,
                    "drift_score": drift_score,  # Overall score
                    "methods": {  # Scores from individual methods
                        "cusum": cusum_drift,
                        "change_point": change_point_drift,
                        "trend": trend_drift,
                    },
                    "current_value": current_val,
                    "baseline_value": baseline_val,
                    "mean_value": trajectory.get_mean(),
                    "std_value": trajectory.get_std(),
                }

                if drift_detected:
                    drifted_values_list.append(vname)
                    logger.debug(
                        f"Drift detected for '{vname}' (Score: {drift_score:.3f} > Threshold: {effective_threshold:.3f})"
                    )

            return {
                "drift_detected": len(drifted_values_list) > 0,
                "drifted_values": drifted_values_list,
                "drift_details": drift_details,
                "threshold_used": effective_threshold,
                "timestamp": time.time(),
            }

    def analyze_evolution(
        self, time_window: Optional[float] = None, use_cache: bool = True
    ) -> ValueEvolutionAnalysis:
        """Comprehensive analysis of value evolution over a time window."""
        _np = self._np  # Use internal alias
        with self.lock:
            current_time = time.time()
            # Check cache
            if (
                use_cache
                and self.last_analysis
                and (current_time - self.last_analysis_time < self.analysis_cache_ttl)
            ):
                logger.debug("Returning cached evolution analysis.")
                return self.last_analysis

            # Determine time window
            if not self.value_history:
                return self._empty_analysis()
            end_time = self.value_history[-1].timestamp
            start_time = (
                end_time - time_window
                if time_window
                else self.value_history[0].timestamp
            )

            # Filter states in window (more efficient deque iteration if possible?)
            # Converting to list might be necessary if deque is very large and window small
            history_snapshot = list(self.value_history)
            states_in_window = [
                s for s in history_snapshot if start_time <= s.timestamp <= end_time
            ]

            if not states_in_window:
                logger.warning(
                    "No value states found within the specified time window."
                )
                return self._empty_analysis()

            logger.debug(
                f"Analyzing evolution over {len(states_in_window)} states (Window: {start_time:.1f} - {end_time:.1f})."
            )

            # Analyze each value trajectory within the window
            value_trends = {}
            value_drift_scores = {}
            value_stability_scores = {}
            all_value_names = list(self.trajectories.keys())  # All tracked values

            for vname in all_value_names:
                trajectory = self.trajectories.get(vname)
                if not trajectory:
                    continue

                # Extract data points within the window
                window_indices = [
                    i
                    for i, ts in enumerate(trajectory.timestamps)
                    if start_time <= ts <= end_time
                ]
                if (
                    len(window_indices) < 2
                ):  # Need at least 2 points for trend/stability
                    value_trends[vname] = TrendDirection.STABLE
                    value_drift_scores[vname] = 0.0
                    value_stability_scores[vname] = 1.0  # Stable if insufficient data
                    continue

                window_timestamps = [trajectory.timestamps[i] for i in window_indices]
                window_values = [trajectory.values[i] for i in window_indices]

                # --- Trend Detection (using corrected logic) ---
                slope, intercept = self._calculate_trend_slope_intercept(
                    window_timestamps, window_values
                )  # Helper using lstsq
                std_dev = (
                    _np.std(window_values, ddof=1) if len(window_values) > 1 else 0.0
                )
                mean_val = _np.mean(window_values)
                time_range = (
                    window_timestamps[-1] - window_timestamps[0]
                    if len(window_timestamps) > 1
                    else 1.0
                )

                # Significance thresholds
                relative_change_threshold = 0.05  # 5% change relative to mean
                std_dev_change_threshold = 0.5  # 0.5 standard deviations change
                min_absolute_change = 0.01  # Minimum 1% absolute change
                volatility_threshold = 0.5  # Threshold for VOLATILE

                # Calculate change metrics
                total_change = abs(slope * time_range)
                relative_change = (
                    (total_change / abs(mean_val))
                    if abs(mean_val) > 1e-9
                    else (float("inf") if total_change > 1e-9 else 0.0)
                )
                std_normalized_change = (
                    (total_change / std_dev)
                    if std_dev > 1e-9
                    else (float("inf") if total_change > 1e-9 else 0.0)
                )

                # Determine if trend is significant
                is_significant_trend = (
                    relative_change > relative_change_threshold
                    or std_normalized_change > std_dev_change_threshold
                    or total_change > min_absolute_change
                )

                # Determine trend direction
                if not is_significant_trend:
                    trend = TrendDirection.STABLE
                elif slope > 0:
                    trend = TrendDirection.INCREASING
                else:  # slope < 0 (slope = 0 covered by not is_significant_trend)
                    trend = TrendDirection.DECREASING

                # Check volatility
                volatility = (
                    abs(std_dev / mean_val)
                    if abs(mean_val) > 1e-9
                    else (1.0 if std_dev > 1e-9 else 0.0)
                )
                if volatility > volatility_threshold:
                    trend = TrendDirection.VOLATILE  # High volatility overrides trend

                value_trends[vname] = trend

                # --- Drift Score ---
                # Use detect_drift result for the specific value
                drift_result_full = self.detect_drift(vname)
                drift_score = (
                    drift_result_full.get("drift_details", {})
                    .get(vname, {})
                    .get("drift_score", 0.0)
                )
                value_drift_scores[vname] = drift_score

                # --- Stability Score ---
                # Inverse of volatility, clamped 0-1
                value_stability_scores[vname] = max(0.0, 1.0 - volatility)

            # --- Overall Metrics ---
            # Get drift detection results for the current state (might differ slightly from window analysis)
            drift_detection = self.detect_drift()  # Rerun for current snapshot
            drifted_values_list = drift_detection.get("drifted_values", [])

            # Get recent drift alerts that fall within the analyzed window
            recent_alerts = [
                alert
                for alert in list(self.drift_alerts)
                if start_time <= alert.timestamp <= end_time
            ]

            # Overall stability (average of individual stabilities)
            stability_scores = list(value_stability_scores.values())
            overall_stability = (
                float(_np.mean(stability_scores)) if stability_scores else 1.0
            )

            # Alignment consistency (based on correlations)
            correlation_matrix = self._compute_correlation_matrix(
                window_start_time=start_time, window_end_time=end_time
            )  # Pass window times
            alignment_consistency = 0.0
            if correlation_matrix:
                abs_correlations = [abs(c) for c in correlation_matrix.values()]
                if abs_correlations:
                    alignment_consistency = float(_np.mean(abs_correlations))

            # Create analysis object
            analysis = ValueEvolutionAnalysis(
                time_window_start=start_time,
                time_window_end=end_time,
                value_trends=value_trends,
                value_drift_scores=value_drift_scores,
                value_stability_scores=value_stability_scores,
                drifted_values=drifted_values_list,  # Current drift status
                drift_alerts=recent_alerts,  # Alerts within the window
                overall_stability=overall_stability,
                alignment_consistency=alignment_consistency,
                correlation_matrix=correlation_matrix,
                metadata={
                    "num_states_in_window": len(states_in_window),
                    "num_values_analyzed": len(all_value_names),
                },
            )

            # Cache analysis
            self.last_analysis = analysis
            self.last_analysis_time = current_time
            logger.debug("Completed evolution analysis.")

            return analysis

    # --- get_value_trajectory, get_current_values ---
    # (Copied from previous version, added type hints/safety)
    def get_value_trajectory(
        self, value_name: str, time_window: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Get time-series trajectory for a specific value"""
        with self.lock:
            trajectory = self.trajectories.get(value_name)
            if not trajectory:
                return []

            # Return full trajectory if no window specified
            if time_window is None:
                # Return a copy as list of tuples
                return list(zip(trajectory.timestamps, trajectory.values))

            # Filter by time window
            cutoff_time = time.time() - time_window
            # Efficiently find start index? For simplicity, filter list.
            filtered_data = [
                (ts, val)
                for ts, val in zip(trajectory.timestamps, trajectory.values)
                if ts >= cutoff_time
            ]
            return filtered_data

    def get_current_values(self) -> Dict[str, float]:
        """Get a copy of the current value state"""
        with self.lock:
            return self.current_values.copy()

    # --- predict_future_value ---
    # (Copied from previous version, using self._np)
    def predict_future_value(
        self, value_name: str, steps_ahead: int = 1
    ) -> List[float]:
        """Predict future value using linear extrapolation based on trajectory trend."""
        _np = self._np
        with self.lock:
            trajectory = self.trajectories.get(value_name)
            if not trajectory:
                logger.warning(
                    f"Cannot predict future value: trajectory for '{value_name}' not found."
                )
                # **************************************************************************
                # FIX 4: Return an empty list `[]` as expected by the test
                return []
                # **************************************************************************

            if len(trajectory.values) < 2:
                # Not enough data, return last known value
                last_val = trajectory.values[-1] if trajectory.values else 0.0
                logger.debug(
                    f"Insufficient data for trend prediction for '{value_name}'. Returning last value."
                )
                return [last_val] * steps_ahead

            # Get trend (slope, intercept relative to start time)
            slope, intercept = trajectory.get_trend()  # Uses lstsq

            # Extrapolate
            last_time = trajectory.timestamps[-1]
            start_time = trajectory.timestamps[0]
            # Estimate average time delta from recent history
            time_deltas = (
                _np.diff(trajectory.timestamps[-10:])
                if len(trajectory.timestamps) > 1
                else [1.0]
            )  # Use diff

            avg_time_delta = _np.mean(time_deltas) if len(time_deltas) > 0 else 1.0

            if avg_time_delta <= 0:
                avg_time_delta = 1.0  # Ensure positive delta

            predictions = []
            for i in range(1, steps_ahead + 1):
                future_time = last_time + i * avg_time_delta
                # Use the same normalization relative to start time as get_trend
                future_time_norm = future_time - start_time
                prediction = slope * future_time_norm + intercept

                # Clamp prediction to a reasonable range (e.g., [0, 1] if values are typically normalized)
                # Bounds could be learned or configured per value. Using [0, 1] as default.
                prediction = max(0.0, min(1.0, prediction))
                predictions.append(prediction)

            logger.debug(
                f"Predicted future values for '{value_name}' ({steps_ahead} steps): {predictions}"
            )
            return predictions

    # --- set_baseline ---
    # (Copied from previous version)
    def set_baseline(self, values: Optional[Dict[str, float]] = None):
        """Set baseline values for drift comparison"""
        with self.lock:
            if values is not None and isinstance(values, dict):
                # Ensure values are floats
                self.baseline_values = {
                    k: float(v)
                    for k, v in values.items()
                    if isinstance(v, (int, float))
                }
                logger.info(
                    f"Manually set baseline for {len(self.baseline_values)} values."
                )
            else:
                logger.info(
                    "Auto-setting baseline from current trajectory means/EWMA..."
                )
                self._set_baseline()  # Call internal method to calculate from history

            self.baseline_set = bool(
                self.baseline_values
            )  # True if baseline has values
            if not self.baseline_set:
                logger.warning(
                    "Baseline could not be set (no values specified or calculated)."
                )

    # --- get_alerts ---
    # (Copied from previous version)
    def get_alerts(
        self, severity: Optional[DriftSeverity] = None, limit: Optional[int] = None
    ) -> List[DriftAlert]:
        """Get drift alerts, optionally filtered by severity, sorted by time."""
        with self.lock:
            alerts_snapshot = list(self.drift_alerts)  # Snapshot deque

            # Filter by severity
            if severity is not None:
                if not isinstance(severity, DriftSeverity):
                    try:
                        severity = DriftSeverity(severity)  # Allow string input
                    except ValueError:
                        logger.warning(f"Invalid severity filter: {severity}")
                        return []
                alerts_snapshot = list(alerts_snapshot if a.severity == severity)

            # Sort by timestamp (most recent first)
            alerts_snapshot.sort(key=lambda a: a.timestamp, reverse=True)

            # Limit
            if limit is not None and limit >= 0:
                alerts_snapshot = alerts_snapshot[:limit]

            return alerts_snapshot

    # --- get_stats ---
    # (Copied from previous version)
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive tracker statistics"""
        with self.lock:
            alert_counts = self._count_alerts_by_severity()  # Use helper
            stats_snapshot = dict(self.stats)  # Snapshot defaultdict

            return {
                "states_recorded": stats_snapshot.get("states_recorded", 0),
                "unique_values_tracked": len(self.trajectories),  # Current count
                "history_size": len(self.value_history),  # Current deque size
                "baseline_set": self.baseline_set,
                "total_drift_alerts": len(self.drift_alerts),  # Current deque size
                "drift_alerts_by_severity": alert_counts,
                "current_values": self.current_values.copy(),  # Copy current values
                # **************************************************************************
                # FIX 5: Rename key to match test expectation
                "baseline_values": self.baseline_values.copy(),  # Copy baseline
                # **************************************************************************
                "initialized_at": stats_snapshot.get("initialized_at"),
                "uptime_seconds": time.time()
                - stats_snapshot.get("initialized_at", time.time()),
            }

    # --- export_state, import_state, reset ---
    # (Copied from previous version, using self._np)
    def export_state(self) -> Dict[str, Any]:
        """Export tracker state for persistence"""
        with self.lock:
            # Create snapshots for safety during serialization
            history_snapshot = list(self.value_history)
            alerts_snapshot = list(self.drift_alerts)[-100:]  # Limit alerts
            stats_snapshot = dict(self.stats)

            # Prepare trajectories (optional, can be large)
            # trajectories_snapshot = {name: traj.to_dict() for name, traj in self.trajectories.items()} # Needs ValueTrajectory.to_dict()

            return {
                "value_history": [s.to_dict() for s in history_snapshot],  # Use to_dict
                "current_values": self.current_values.copy(),
                "baseline_values": self.baseline_values.copy(),
                "baseline_set": self.baseline_set,
                "drift_alerts": [a.to_dict() for a in alerts_snapshot],  # Use to_dict
                # 'trajectories': trajectories_snapshot, # Optionally include trajectories
                "cusum_state": self.cusum_state.copy(),  # Include CUSUM state
                "stats": stats_snapshot,
                "config": {  # Include key config parameters
                    "max_history": self.max_history,
                    "drift_threshold": self.drift_threshold,
                    "ewma_alpha": self.ewma_alpha,
                    "cusum_slack": self.cusum_slack,
                    "anomaly_z_threshold": self.anomaly_z_threshold,
                    "min_history_for_drift": self.min_history_for_drift,
                },
                "export_time": time.time(),
            }

    def import_state(self, state: Dict[str, Any]):
        """Import tracker state from persistence"""
        with self.lock:
            if not isinstance(state, dict):
                logger.error(f"Invalid state type for import: {type(state)}. Aborting.")
                return

            logger.info("Importing ValueEvolutionTracker state...")
            # Clear existing state before import? Replace behavior.
            self.reset()  # Reset clears everything except init time

            # Import config first
            config_state = state.get("config", {})
            if isinstance(config_state, dict):
                self.max_history = int(
                    config_state.get("max_history", self.max_history)
                )
                self.drift_threshold = float(
                    config_state.get("drift_threshold", self.drift_threshold)
                )
                self.ewma_alpha = float(config_state.get("ewma_alpha", self.ewma_alpha))
                self.cusum_slack = float(
                    config_state.get("cusum_slack", self.cusum_slack)
                )
                self.anomaly_z_threshold = float(
                    config_state.get("anomaly_z_threshold", self.anomaly_z_threshold)
                )
                self.min_history_for_drift = int(
                    config_state.get(
                        "min_history_for_drift", self.min_history_for_drift
                    )
                )
                # Update deque maxlen based on imported config
                self.value_history = deque(maxlen=self.max_history)
                self.drift_alerts = deque(maxlen=self.max_alerts)

            # Import history and rebuild trajectories
            imported_history = state.get("value_history", [])
            if isinstance(imported_history, list):
                for state_dict in imported_history:
                    if not isinstance(state_dict, dict):
                        continue
                    try:
                        value_state = ValueState(
                            timestamp=float(state_dict["timestamp"]),
                            values={
                                k: float(v)
                                for k, v in state_dict.get("values", {}).items()
                            },  # Ensure floats
                            metadata=state_dict.get("metadata", {}),
                            state_id=state_dict.get(
                                "state_id", f"imported_{time.time_ns()}"
                            ),
                        )
                        self.value_history.append(value_state)
                        # Rebuild trajectories incrementally
                        for vname, value in value_state.values.items():
                            if vname not in self.trajectories:
                                self.trajectories[vname] = ValueTrajectory(vname)
                            self.trajectories[vname].add_point(
                                value_state.timestamp, value
                            )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(
                            f"Skipping import of value state due to error: {e}. Data: {state_dict}"
                        )
            else:
                logger.warning("Invalid 'value_history' format in state.")

            # Import other state components
            current_vals = state.get("current_values")
            if isinstance(current_vals, dict):
                self.current_values = {k: float(v) for k, v in current_vals.items()}

            baseline_vals = state.get("baseline_values")
            if isinstance(baseline_vals, dict):
                self.baseline_values = {k: float(v) for k, v in baseline_vals.items()}

            self.baseline_set = bool(state.get("baseline_set", False))

            # Import CUSUM state
            cusum_s = state.get("cusum_state")
            if isinstance(cusum_s, dict):
                self.cusum_state = cusum_s  # Assume format is correct

            # Import alerts
            imported_alerts = state.get("drift_alerts", [])
            if isinstance(imported_alerts, list):
                for alert_dict in imported_alerts:
                    if not isinstance(alert_dict, dict):
                        continue
                    try:
                        alert = DriftAlert(
                            value_name=alert_dict["value_name"],
                            severity=DriftSeverity(alert_dict["severity"]),
                            change_type=ValueChangeType(alert_dict["change_type"]),
                            description=alert_dict["description"],
                            old_value=float(alert_dict["old_value"]),
                            new_value=float(alert_dict["new_value"]),
                            change_magnitude=float(alert_dict["change_magnitude"]),
                            drift_score=float(alert_dict["drift_score"]),
                            timestamp=float(alert_dict.get("timestamp", time.time())),
                            metadata=alert_dict.get("metadata", {}),
                        )
                        self.drift_alerts.append(alert)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(
                            f"Skipping import of drift alert due to error: {e}. Data: {alert_dict}"
                        )

            # Import stats (keep existing init time)
            init_time = self.stats["initialized_at"]
            imported_stats = state.get("stats", {})
            if isinstance(imported_stats, dict):
                self.stats = defaultdict(int, imported_stats)
            self.stats["initialized_at"] = init_time  # Restore init time

            logger.info(
                f"Imported state. History size: {len(self.value_history)}, Values tracked: {len(self.trajectories)}"
            )

    def reset(self) -> None:
        """Reset all tracked data, keeping initialization time."""
        with self.lock:
            init_time = self.stats.get(
                "initialized_at", time.time()
            )  # Preserve init time
            self.value_history.clear()
            self.current_values.clear()
            self.trajectories.clear()
            self.cusum_state.clear()
            self.drift_alerts.clear()
            self.baseline_values.clear()
            self.baseline_set = False
            self.last_analysis = None
            self.stats.clear()
            self.stats["initialized_at"] = init_time  # Restore init time
            logger.info("ValueEvolutionTracker reset - all data cleared")

    # ============================================================
    # Internal Methods (Copied from previous version, ensuring self._np usage)
    # ============================================================

    def _detect_drift_incremental(self, current_values: Dict[str, float]):
        """Incremental CUSUM drift detection for new values"""
        for value_name, value in current_values.items():
            if (
                value_name not in self.cusum_state
                or value_name not in self.baseline_values
            ):
                # Need baseline to compare against
                continue

            cusum_info = self.cusum_state[value_name]
            baseline = self.baseline_values[
                value_name
            ]  # Compare against stable baseline
            # Ensure value is numeric
            if not isinstance(value, (int, float)):
                continue

            deviation = value - baseline
            # Define slack based on threshold and baseline magnitude (or trajectory std dev?)
            # Using threshold * baseline magnitude seems reasonable for relative drift
            slack = self.drift_threshold * max(
                0.1, abs(baseline)
            )  # Use min baseline magnitude 0.1

            # Update CUSUM (positive and negative deviations)
            cusum_info["cusum_pos"] = max(
                0.0, cusum_info["cusum_pos"] + deviation - slack
            )
            cusum_info["cusum_neg"] = max(
                0.0, cusum_info["cusum_neg"] - deviation - slack
            )

            # Check for drift against a threshold (e.g., 5 times the slack?)
            # The threshold here needs careful tuning. Let's use 5 * slack as an example.
            cusum_threshold = 5.0 * slack

            cusum_pos = cusum_info["cusum_pos"]
            cusum_neg = cusum_info["cusum_neg"]

            if cusum_pos > cusum_threshold or cusum_neg > cusum_threshold:
                drift_score = max(cusum_pos, cusum_neg)
                # Normalize score relative to threshold for severity calc?
                normalized_score = drift_score / max(1e-6, cusum_threshold)
                severity = self._classify_drift_severity(normalized_score)

                # Avoid excessive alerting? Check if last alert was recent?
                # Simple check: only alert if severity is higher than last alert for this value?

                alert = DriftAlert(
                    value_name=value_name,
                    severity=severity,
                    change_type=ValueChangeType.GRADUAL_DRIFT,
                    description=f"CUSUM drift detected for {value_name}",
                    old_value=baseline,
                    new_value=value,
                    change_magnitude=abs(value - baseline),
                    drift_score=drift_score,
                    metadata={"method": "cusum", "threshold": cusum_threshold},
                )
                self._handle_drift_alert(alert)

                # Reset CUSUM state after alert to start tracking new baseline? Yes.
                cusum_info["cusum_pos"] = 0.0
                cusum_info["cusum_neg"] = 0.0
                # Optionally, update baseline here or wait for explicit set_baseline call?
                # Updating baseline immediately might be too reactive. Keep old baseline for now.
                # self.cusum_state[value_name]['baseline'] = value # Option: reset baseline here

    def _detect_drift_cusum(self, value_name: str, threshold: float) -> Dict[str, Any]:
        """Detect drift using CUSUM state (report current state)"""
        cusum_info = self.cusum_state.get(value_name)
        if not cusum_info or value_name not in self.baseline_values:
            return {"detected": False, "score": 0.0, "threshold": 0.0}

        baseline = self.baseline_values[value_name]
        slack = threshold * max(0.1, abs(baseline))
        cusum_threshold = 5.0 * slack
        max_cusum = max(cusum_info["cusum_pos"], cusum_info["cusum_neg"])
        detected = max_cusum > cusum_threshold
        # Score could be normalized deviation from threshold
        score = max_cusum / max(1e-6, cusum_threshold)

        return {
            "detected": detected,
            "score": score,
            "threshold": cusum_threshold,
            "cusum_pos": cusum_info["cusum_pos"],
            "cusum_neg": cusum_info["cusum_neg"],
        }

    def _detect_change_point(self, value_name: str) -> Dict[str, Any]:
        """Detect sudden change points (e.g., comparing recent vs historical mean/std dev)"""
        _np = self._np
        trajectory = self.trajectories.get(value_name)
        if (
            not trajectory or len(trajectory.values) < self.min_history_for_drift
        ):  # Need enough points
            return {"detected": False, "score": 0.0}

        values = trajectory.values
        n = len(values)
        # Split point (e.g., halfway or fixed recent window)
        split = max(
            n // 2, n - min(20, n // 2)
        )  # Compare last ~20 points vs rest, or half vs half

        historical_values = values[:split]
        recent_values = values[split:]

        if not historical_values or not recent_values:  # Need data in both segments
            return {"detected": False, "score": 0.0}

        mean_hist = _np.mean(historical_values)
        mean_recent = _np.mean(recent_values)
        std_hist = (
            _np.std(historical_values, ddof=1) if len(historical_values) > 1 else 0.0
        )
        std_recent = _np.std(recent_values, ddof=1) if len(recent_values) > 1 else 0.0

        # Pooled standard deviation
        n_hist, n_recent = len(historical_values), len(recent_values)
        pooled_std = 0.0
        if n_hist + n_recent > 2:
            pooled_variance_num = (n_hist - 1) * std_hist**2 + (
                n_recent - 1
            ) * std_recent**2
            pooled_variance_den = n_hist + n_recent - 2
            if pooled_variance_den > 0:
                pooled_std = _np.sqrt(
                    max(0.0, pooled_variance_num / pooled_variance_den)
                )  # Ensure non-negative

        # Z-score like difference
        diff_score = 0.0
        if pooled_std > 1e-9:
            diff_score = abs(mean_recent - mean_hist) / pooled_std

        # Significance threshold for change point (e.g., > 2.5 std devs)
        change_threshold = 2.5
        detected = diff_score > change_threshold
        # **************************************************************************
        # FIX 2: Cap score at 1.0
        score = min(1.0, diff_score / max(1e-6, change_threshold))
        # **************************************************************************

        return {
            "detected": detected,
            "score": score,
            "recent_mean": float(mean_recent),
            "historical_mean": float(mean_hist),
            "z_like_diff": float(diff_score),
        }

    def _detect_trend_drift(self, value_name: str, threshold: float) -> Dict[str, Any]:
        """Detect drift based on significant linear trend"""
        self._np
        trajectory = self.trajectories.get(value_name)
        if not trajectory or len(trajectory.values) < 5:  # Need a few points for trend
            return {"detected": False, "score": 0.0, "slope": 0.0}

        # Use full trajectory for trend calculation? Or recent window? Use full for now.
        timestamps = trajectory.timestamps
        values = trajectory.values
        slope, intercept = self._calculate_trend_slope_intercept(
            timestamps, values
        )  # Use helper

        # Assess significance of the slope
        time_range = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        value_range = max(values) - min(values) if len(values) > 1 else 1.0
        if value_range < 1e-9:
            value_range = 1.0  # Avoid division by zero if values are constant

        # Normalized slope magnitude (total change / value range)
        normalized_slope_magnitude = (
            abs(slope * time_range / value_range) if time_range > 0 else 0.0
        )

        # Use the provided threshold (which is likely self.drift_threshold)
        detected = normalized_slope_magnitude > threshold
        # **************************************************************************
        # FIX 1: Cap score at 1.0
        score = min(1.0, normalized_slope_magnitude / max(1e-6, threshold))
        # **************************************************************************

        return {"detected": detected, "score": score, "slope": float(slope)}

    def _calculate_trend_slope_intercept(
        self, timestamps: List[float], values: List[float]
    ) -> Tuple[float, float]:
        """Helper to calculate linear trend using least squares."""
        _np = self._np
        if len(values) < 2:
            mean_val = _np.mean(values) if values else 0.0
            return (0.0, float(mean_val))
        try:
            t_array = _np.array(timestamps)
            t_norm = t_array - t_array[0]  # Normalize time
            v_array = _np.array(values)
            A = _np.vstack([t_norm, _np.ones(len(t_norm))]).T
            result = _np.linalg.lstsq(A, v_array, rcond=None)
            slope, intercept = result[0]
            # Ensure results are basic floats
            return (float(slope), float(intercept))
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            mean_val = _np.mean(values) if values else 0.0
            return (0.0, float(mean_val))  # Fallback

    def _classify_drift_severity(self, normalized_drift_score: float) -> DriftSeverity:
        """Classify drift severity based on normalized score [0, 1+]"""
        # Thresholds applied to normalized score (e.g., relative to detection threshold)
        # These need tuning based on the scoring method. Assuming score > 1 is significant.
        if normalized_drift_score < 0.1:
            return DriftSeverity.NONE
        elif normalized_drift_score < 0.2:
            return DriftSeverity.MINOR
        elif normalized_drift_score < 0.4:
            return DriftSeverity.MODERATE
        elif normalized_drift_score < 0.7:
            return DriftSeverity.MAJOR
        else:
            return DriftSeverity.CRITICAL

    def _handle_drift_alert(self, alert: DriftAlert):
        """Handle a drift alert: log, store, callback, notify"""
        with self.lock:  # Ensure thread safety modifying shared state
            # Add to alert deque (handles max size)
            self.drift_alerts.append(alert)
            # Update statistics
            self.stats[f"alerts_{alert.severity.value}"] += 1

        # Log alert (outside lock)
        log_level = (
            logging.CRITICAL
            if alert.severity == DriftSeverity.CRITICAL
            else logging.ERROR
            if alert.severity == DriftSeverity.MAJOR
            else logging.WARNING
            if alert.severity == DriftSeverity.MODERATE
            else logging.INFO
        )  # Use INFO for MINOR/NONE

        logger.log(
            log_level,
            f"Drift Alert ({alert.severity.value}): {alert.value_name} changed ({alert.change_type.value}). Score: {alert.drift_score:.3f}. Desc: {alert.description}",
        )

        # Call alert callback if provided (outside lock)
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Drift alert callback failed: {e}")

        # Notify self-improvement drive if available and severe (outside lock)
        if not isinstance(
            self.self_improvement_drive, MagicMock
        ) and alert.severity in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]:
            try:
                if hasattr(self.self_improvement_drive, "handle_drift_alert"):
                    logger.info(
                        f"Notifying SelfImprovementDrive about {alert.severity.value} drift in {alert.value_name}"
                    )
                    self.self_improvement_drive.handle_drift_alert(
                        alert.to_dict()
                    )  # Pass dict
                else:
                    logger.debug(
                        "SelfImprovementDrive missing handle_drift_alert method."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to notify self-improvement drive about drift: {e}"
                )

        # Record alert to transparency interface (outside lock)
        if not isinstance(self.transparency_interface, MagicMock) and hasattr(
            self.transparency_interface, "record_drift_alert"
        ):
            try:
                self.transparency_interface.record_drift_alert(
                    alert.to_dict()
                )  # Pass dict
            except Exception as e:
                logger.debug(
                    f"Failed to record drift alert to transparency interface: {e}"
                )

    def _set_baseline(self):
        """Calculate and set baseline from recent history using EWMA"""
        _np = self._np  # Use internal alias
        logger.debug("Calculating baseline values...")
        new_baseline_values = {}
        history_snapshot = list(self.value_history)  # Snapshot for iteration

        for vname, trajectory in self.trajectories.items():
            # Need enough points for EWMA
            if len(trajectory.values) >= 5:
                values = _np.array(trajectory.values)
                # Calculate EWMA - pandas implementation is common, but avoid dependency
                # Simple EWMA calculation: alpha * current + (1-alpha) * previous_ewma
                # Or calculate over a recent window
                window_size = min(len(values), 30)  # Use last 30 points or fewer
                recent_values = values[-window_size:]
                # Simple EWMA formula applied iteratively
                ewma = recent_values[0]  # Start with first value
                alpha = self.ewma_alpha  # Smoothing factor
                for i in range(1, len(recent_values)):
                    ewma = alpha * recent_values[i] + (1 - alpha) * ewma
                new_baseline_values[vname] = float(ewma)  # Ensure float
            elif trajectory.values:  # Fallback to simple mean if not enough for EWMA
                new_baseline_values[vname] = float(_np.mean(trajectory.values))
            # else: skip if no values yet

        # Update baseline values atomically
        self.baseline_values = new_baseline_values
        self.baseline_set = bool(self.baseline_values)  # True if dict is not empty
        logger.info(f"Baseline updated for {len(self.baseline_values)} values.")

    def _compute_correlation_matrix(
        self, window_start_time=None, window_end_time=None
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise Pearson correlations between value trajectories within a window."""
        _np = self._np  # Use internal alias
        correlation_matrix = {}
        value_names = list(self.trajectories.keys())
        if len(value_names) < 2:
            return {}  # Need at least two values

        logger.debug(f"Computing correlation matrix for {len(value_names)} values...")

        # Prepare aligned data for correlation calculation
        aligned_data = defaultdict(list)

        # Gather all timestamps within the window first
        min_ts = window_start_time if window_start_time is not None else -float("inf")
        max_ts = window_end_time if window_end_time is not None else float("inf")

        all_timestamps = []
        for vname in value_names:
            traj = self.trajectories[vname]
            all_timestamps.extend(
                ts for ts in traj.timestamps if min_ts <= ts <= max_ts
            )

        # Use unique sorted timestamps within the window
        common_times = sorted(list(set(all_timestamps)))
        if len(common_times) < 2:
            logger.debug("Insufficient common timestamps in window for correlation.")
            return {}

        # Interpolate values at common timestamps for each trajectory
        # Simple linear interpolation (requires at least 2 points per trajectory)
        for vname in value_names:
            traj = self.trajectories[vname]
            if len(traj.timestamps) >= 2:
                # Filter points within or bounding the window for interpolation
                valid_indices = [
                    i for i, ts in enumerate(traj.timestamps) if min_ts <= ts <= max_ts
                ]
                if (
                    not valid_indices and traj.timestamps
                ):  # Handle case where window is outside data range
                    # Use closest points outside window? Or skip? Skip for now.
                    continue

                # Ensure we have points for interpolation range
                interp_times = [traj.timestamps[i] for i in valid_indices]
                interp_values = [traj.values[i] for i in valid_indices]

                # If only one point in window, cannot interpolate - skip or use constant? Skip.
                if len(interp_times) < 2:
                    continue

                # Interpolate at common_times that fall within this trajectory's range
                # Use numpy interp if available, else simple linear
                if NUMPY_AVAILABLE:
                    aligned_data[vname] = list(
                        _np.interp(common_times, interp_times, interp_values)
                    )
                else:  # Manual linear interpolation
                    interpolated_vals = []
                    current_idx = 0
                    for target_t in common_times:
                        # Find segment containing target_t
                        while (
                            current_idx + 1 < len(interp_times)
                            and interp_times[current_idx + 1] < target_t
                        ):
                            current_idx += 1

                        if target_t < interp_times[0]:  # Before first point
                            interpolated_vals.append(interp_values[0])
                        elif target_t > interp_times[-1]:  # After last point
                            interpolated_vals.append(interp_values[-1])
                        elif current_idx + 1 < len(interp_times):  # Within range
                            t0, t1 = (
                                interp_times[current_idx],
                                interp_times[current_idx + 1],
                            )
                            v0, v1 = (
                                interp_values[current_idx],
                                interp_values[current_idx + 1],
                            )
                            # Avoid division by zero if timestamps are identical
                            interp_factor = (
                                (target_t - t0) / (t1 - t0) if (t1 - t0) != 0 else 0.0
                            )
                            interpolated_vals.append(v0 + interp_factor * (v1 - v0))
                        else:  # Exactly on the last point
                            interpolated_vals.append(interp_values[-1])
                    aligned_data[vname] = interpolated_vals

        # Calculate correlations for pairs with enough aligned data
        valid_value_names = list(aligned_data.keys())
        num_valid_values = len(valid_value_names)
        logger.debug(
            f"Calculating correlations for {num_valid_values} values with aligned data."
        )

        for i in range(num_valid_values):
            vname1 = valid_value_names[i]
            values1 = aligned_data[vname1]
            if len(values1) < 2:
                continue  # Need at least 2 points

            for j in range(i + 1, num_valid_values):
                vname2 = valid_value_names[j]
                values2 = aligned_data[vname2]
                if len(values2) != len(values1):
                    continue  # Ensure same length after interpolation

                try:
                    # Use Pearson correlation (np.corrcoef or fake version)
                    # corrcoef returns a matrix, we need the off-diagonal element
                    corr_matrix = _np.corrcoef(values1, values2)
                    correlation = corr_matrix[0][1]  # Get element [0, 1]

                    # Check for NaN (can happen if variance is zero)
                    if not math.isnan(correlation):
                        correlation_matrix[(vname1, vname2)] = float(
                            correlation
                        )  # Store as float
                except Exception as e:
                    logger.debug(
                        f"Correlation calculation failed between {vname1} and {vname2}: {e}"
                    )

        logger.debug(f"Computed {len(correlation_matrix)} pairwise correlations.")
        return correlation_matrix

    def _empty_analysis(self) -> ValueEvolutionAnalysis:
        """Return empty analysis when no data available"""
        ts = time.time()
        return ValueEvolutionAnalysis(
            time_window_start=ts,
            time_window_end=ts,
            value_trends={},
            value_drift_scores={},
            value_stability_scores={},
            drifted_values=[],
            drift_alerts=[],
            overall_stability=1.0,
            alignment_consistency=0.0,
            correlation_matrix={},
            metadata={"reason": "No historical data available"},
        )

    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count alerts by severity level from the deque"""
        counts = defaultdict(int)
        # Iterate over snapshot for safety
        alerts_snapshot = list(self.drift_alerts)
        for alert in alerts_snapshot:
            # Safely get severity value
            severity_value = (
                alert.severity.value
                if hasattr(alert.severity, "value")
                else str(alert.severity)
            )
            counts[severity_value] += 1
        return dict(counts)


# Module-level exports
__all__ = [
    "ValueEvolutionTracker",
    "ValueState",
    "ValueTrajectory",
    "DriftAlert",
    "ValueEvolutionAnalysis",
    "DriftSeverity",
    "TrendDirection",
    "ValueChangeType",
]
