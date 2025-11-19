"""
Distribution Monitor for Tool Selection System

Monitors for distribution shifts, concept drift, and data drift to detect
when the system is operating outside expected conditions.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import time
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import pickle
import threading
import warnings
from .security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of distribution drift"""
    FEATURE_DRIFT = "feature_drift"
    CONCEPT_DRIFT = "concept_drift"
    PRIOR_DRIFT = "prior_drift"
    COVARIATE_SHIFT = "covariate_shift"
    LABEL_SHIFT = "label_shift"


class DetectionMethod(Enum):
    """Distribution shift detection methods"""
    KS_TEST = "kolmogorov_smirnov"
    CHI_SQUARED = "chi_squared"
    WASSERSTEIN = "wasserstein"
    JS_DIVERGENCE = "jensen_shannon"
    MMD = "maximum_mean_discrepancy"
    PAGE_HINKLEY = "page_hinkley"
    ADWIN = "adwin"
    CUSUM = "cumulative_sum"


class DriftSeverity(Enum):
    """Severity levels of detected drift"""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DriftDetection:
    """Single drift detection result"""
    drift_type: DriftType
    method: DetectionMethod
    severity: DriftSeverity
    statistic: float
    p_value: Optional[float]
    threshold: float
    detected: bool
    timestamp: float = field(default_factory=time.time)
    feature_indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionSnapshot:
    """Snapshot of distribution at a point in time"""
    features: np.ndarray
    labels: Optional[np.ndarray]
    timestamp: float
    sample_count: int
    feature_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowedDistribution:
    """Maintains windowed distribution statistics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        
    def update(self, value: np.ndarray):
        """Add new value to window"""
        self.data.append(value)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update running statistics"""
        if not self.data:
            return
        
        data_array = np.array(self.data)
        self.mean = np.mean(data_array, axis=0)
        self.std = np.std(data_array, axis=0)
        self.min_val = np.min(data_array, axis=0)
        self.max_val = np.max(data_array, axis=0)
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get current statistics"""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'count': len(self.data)
        }


class KolmogorovSmirnovDetector:
    """Kolmogorov-Smirnov test for distribution shift"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_samples = None
    
    def set_reference(self, data: np.ndarray):
        """Set reference distribution"""
        self.reference_samples = data
    
    def detect(self, current_data: np.ndarray) -> List[DriftDetection]:
        """Detect drift using KS test"""
        
        if self.reference_samples is None:
            return []
        
        detections = []
        n_features = current_data.shape[1]
        
        for i in range(n_features):
            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_samples[:, i],
                current_data[:, i]
            )
            
            # Determine severity
            if p_value < self.threshold:
                if p_value < 0.001:
                    severity = DriftSeverity.CRITICAL
                elif p_value < 0.01:
                    severity = DriftSeverity.HIGH
                elif p_value < 0.03:
                    severity = DriftSeverity.MODERATE
                else:
                    severity = DriftSeverity.LOW
                
                detected = True
            else:
                severity = DriftSeverity.NONE
                detected = False
            
            detections.append(DriftDetection(
                drift_type=DriftType.FEATURE_DRIFT,
                method=DetectionMethod.KS_TEST,
                severity=severity,
                statistic=statistic,
                p_value=p_value,
                threshold=self.threshold,
                detected=detected,
                feature_indices=[i]
            ))
        
        return detections


class WassersteinDetector:
    """Wasserstein distance for distribution shift"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_distribution = None
    
    def set_reference(self, data: np.ndarray):
        """Set reference distribution"""
        self.reference_distribution = data
    
    def detect(self, current_data: np.ndarray) -> List[DriftDetection]:
        """Detect drift using Wasserstein distance"""
        
        if self.reference_distribution is None:
            return []
        
        detections = []
        n_features = current_data.shape[1]
        
        for i in range(n_features):
            # Calculate Wasserstein distance
            distance = wasserstein_distance(
                self.reference_distribution[:, i],
                current_data[:, i]
            )
            
            # Normalize by range
            ref_range = np.ptp(self.reference_distribution[:, i])
            if ref_range > 0:
                normalized_distance = distance / ref_range
            else:
                normalized_distance = distance
            
            # Determine severity
            if normalized_distance > self.threshold:
                if normalized_distance > self.threshold * 4:
                    severity = DriftSeverity.CRITICAL
                elif normalized_distance > self.threshold * 3:
                    severity = DriftSeverity.HIGH
                elif normalized_distance > self.threshold * 2:
                    severity = DriftSeverity.MODERATE
                else:
                    severity = DriftSeverity.LOW
                
                detected = True
            else:
                severity = DriftSeverity.NONE
                detected = False
            
            detections.append(DriftDetection(
                drift_type=DriftType.COVARIATE_SHIFT,
                method=DetectionMethod.WASSERSTEIN,
                severity=severity,
                statistic=normalized_distance,
                p_value=None,
                threshold=self.threshold,
                detected=detected,
                feature_indices=[i]
            ))
        
        return detections


class MMDDetector:
    """Maximum Mean Discrepancy for distribution shift"""
    
    def __init__(self, threshold: float = 0.05, kernel: str = 'rbf'):
        self.threshold = threshold
        self.kernel = kernel
        self.reference_data = None
    
    def set_reference(self, data: np.ndarray):
        """Set reference distribution"""
        self.reference_data = data
    
    def detect(self, current_data: np.ndarray) -> DriftDetection:
        """Detect drift using MMD"""
        
        if self.reference_data is None:
            return None
        
        # Compute MMD statistic
        mmd_stat = self._compute_mmd(self.reference_data, current_data)
        
        # Determine severity
        if mmd_stat > self.threshold:
            if mmd_stat > self.threshold * 4:
                severity = DriftSeverity.CRITICAL
            elif mmd_stat > self.threshold * 3:
                severity = DriftSeverity.HIGH
            elif mmd_stat > self.threshold * 2:
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.LOW
            
            detected = True
        else:
            severity = DriftSeverity.NONE
            detected = False
        
        return DriftDetection(
            drift_type=DriftType.COVARIATE_SHIFT,
            method=DetectionMethod.MMD,
            severity=severity,
            statistic=mmd_stat,
            p_value=None,
            threshold=self.threshold,
            detected=detected
        )
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy"""
        
        n_x, n_y = len(X), len(Y)
        
        if self.kernel == 'rbf':
            # RBF kernel with median heuristic for bandwidth
            combined = np.vstack([X, Y])
            distances = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
            median_dist = np.median(distances[distances > 0])
            gamma = 1.0 / median_dist if median_dist > 0 else 1.0
            
            # Compute kernel matrices
            K_xx = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
            K_yy = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
            K_xy = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        else:
            # Linear kernel
            K_xx = X @ X.T
            K_yy = Y @ Y.T
            K_xy = X @ Y.T
        
        # Compute MMD
        mmd = (np.sum(K_xx) / (n_x * n_x) + 
               np.sum(K_yy) / (n_y * n_y) - 
               2 * np.sum(K_xy) / (n_x * n_y))
        
        return max(0, mmd)


class PageHinkleyDetector:
    """Page-Hinkley test for detecting changes in mean"""
    
    def __init__(self, delta: float = 0.005, threshold: float = 50):
        self.delta = delta
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset detector state"""
        self.sum = 0
        self.min_sum = 0
        self.num_samples = 0
    
    def detect(self, value: float) -> DriftDetection:
        """Detect drift in single value stream"""
        
        self.num_samples += 1
        self.sum += value - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        
        ph_statistic = self.sum - self.min_sum
        
        # Determine if drift detected
        if ph_statistic > self.threshold:
            severity = DriftSeverity.HIGH
            detected = True
            self.reset()  # Reset after detection
        else:
            severity = DriftSeverity.NONE
            detected = False
        
        return DriftDetection(
            drift_type=DriftType.CONCEPT_DRIFT,
            method=DetectionMethod.PAGE_HINKLEY,
            severity=severity,
            statistic=ph_statistic,
            p_value=None,
            threshold=self.threshold,
            detected=detected
        )


class DistributionMonitor:
    """
    Main distribution monitoring system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Configuration
        self.window_size = config.get('window_size', 1000)
        self.reference_window_size = config.get('reference_window_size', 5000)
        self.detection_threshold = config.get('detection_threshold', 0.05)
        self.check_interval = config.get('check_interval', 100)  # Check every N samples
        
        # Reference distribution
        self.reference_distribution = None
        self.reference_performance = None
        
        # Current distribution tracking
        self.current_window = WindowedDistribution(self.window_size)
        self.sample_count = 0
        
        # Detectors
        self.ks_detector = KolmogorovSmirnovDetector(self.detection_threshold)
        self.wasserstein_detector = WassersteinDetector(threshold=0.1)
        self.mmd_detector = MMDDetector(threshold=0.05)
        self.ph_detector = PageHinkleyDetector()
        
        # Detection history
        self.detection_history = deque(maxlen=1000)
        self.drift_alerts = deque(maxlen=100)
        
        # Performance tracking
        self.performance_window = deque(maxlen=self.window_size)
        self.baseline_performance = None
        
        # PCA for dimensionality reduction
        self.pca = None
        self.scaler = StandardScaler()
        
        # Statistics
        self.total_checks = 0
        self.total_drifts_detected = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def set_reference(self, features: np.ndarray, 
                     performance: Optional[float] = None):
        """Set reference distribution"""
        
        with self.lock:
            self.reference_distribution = features
            self.reference_performance = performance
            
            # Update detectors
            self.ks_detector.set_reference(features)
            self.wasserstein_detector.set_reference(features)
            self.mmd_detector.set_reference(features)
            
            # Fit PCA for dimensionality reduction
            if features.shape[1] > 10:
                self.pca = PCA(n_components=min(10, features.shape[1]))
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.pca.fit(scaled_features)
            
            logger.info(f"Reference distribution set with {len(features)} samples")
    
    def update(self, features: np.ndarray, 
              performance: Optional[float] = None) -> bool:
        """
        Update with new sample and check for drift
        
        Returns:
            True if drift detected
        """
        
        with self.lock:
            self.sample_count += 1
            
            # Update current window
            self.current_window.update(features)
            
            # Update performance tracking
            if performance is not None:
                self.performance_window.append(performance)
            
            # Check for drift at intervals
            if self.sample_count % self.check_interval == 0:
                return self._check_drift()
            
            return False
    
    def detect_shift(self, features: np.ndarray, 
                    result: Optional[Any] = None) -> bool:
        """
        Detect if there's a distribution shift
        
        Returns:
            True if shift detected
        """
        
        with self.lock:
            self.total_checks += 1
            
            if self.reference_distribution is None:
                return False
            
            detections = []
            
            # Get current data
            current_data = np.array(list(self.current_window.data))
            
            if len(current_data) < 50:
                return False  # Not enough data
            
            # Run various detectors
            
            # KS test
            ks_detections = self.ks_detector.detect(current_data)
            detections.extend(ks_detections)
            
            # Wasserstein distance
            w_detections = self.wasserstein_detector.detect(current_data)
            detections.extend(w_detections)
            
            # MMD test
            mmd_detection = self.mmd_detector.detect(current_data)
            if mmd_detection:
                detections.append(mmd_detection)
            
            # Performance-based drift detection
            if result and hasattr(result, 'confidence'):
                ph_detection = self.ph_detector.detect(1 - result.confidence)
                if ph_detection:
                    detections.append(ph_detection)
            
            # Check if any critical or high severity drift
            critical_drifts = [d for d in detections 
                              if d.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]]
            
            if critical_drifts:
                self.total_drifts_detected += 1
                self._record_drift(critical_drifts)
                return True
            
            # Check if multiple moderate drifts
            moderate_drifts = [d for d in detections 
                              if d.severity == DriftSeverity.MODERATE]
            
            if len(moderate_drifts) >= 3:
                self.total_drifts_detected += 1
                self._record_drift(moderate_drifts)
                return True
            
            return False
    
    def _check_drift(self) -> bool:
        """Internal drift checking"""
        
        if self.reference_distribution is None:
            return False
        
        current_data = np.array(list(self.current_window.data))
        
        if len(current_data) < 50:
            return False
        
        # Run detection
        return self.detect_shift(current_data[-1])
    
    def _record_drift(self, detections: List[DriftDetection]):
        """Record detected drift"""
        
        for detection in detections:
            self.detection_history.append(detection)
        
        # Create alert
        alert = {
            'timestamp': time.time(),
            'detections': len(detections),
            'max_severity': max(d.severity.value for d in detections),
            'drift_types': list(set(d.drift_type.value for d in detections)),
            'methods': list(set(d.method.value for d in detections))
        }
        
        self.drift_alerts.append(alert)
        
        logger.warning(f"Distribution drift detected: {alert}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drifts"""
        
        with self.lock:
            if not self.detection_history:
                return {
                    'drift_detected': False,
                    'total_checks': self.total_checks,
                    'total_drifts': self.total_drifts_detected
                }
            
            recent_detections = list(self.detection_history)[-10:]
            
            return {
                'drift_detected': True,
                'total_checks': self.total_checks,
                'total_drifts': self.total_drifts_detected,
                'recent_detections': [
                    {
                        'type': d.drift_type.value,
                        'method': d.method.value,
                        'severity': d.severity.value,
                        'statistic': d.statistic,
                        'timestamp': d.timestamp
                    }
                    for d in recent_detections
                ],
                'alerts': list(self.drift_alerts)[-5:]
            }
    
    def analyze_feature_importance(self) -> Dict[int, float]:
        """Analyze which features contribute most to drift"""
        
        with self.lock:
            feature_drift_counts = defaultdict(int)
            
            for detection in self.detection_history:
                if detection.feature_indices:
                    for idx in detection.feature_indices:
                        feature_drift_counts[idx] += detection.severity.value
            
            # Normalize
            total = sum(feature_drift_counts.values())
            if total > 0:
                return {k: v/total for k, v in feature_drift_counts.items()}
            
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        
        with self.lock:
            stats = {
                'total_samples': self.sample_count,
                'total_checks': self.total_checks,
                'total_drifts': self.total_drifts_detected,
                'drift_rate': self.total_drifts_detected / max(1, self.total_checks),
                'current_window_size': len(self.current_window.data),
                'reference_set': self.reference_distribution is not None
            }
            
            # Add current distribution statistics
            if self.current_window.mean is not None:
                stats['current_distribution'] = {
                    'mean': self.current_window.mean.tolist(),
                    'std': self.current_window.std.tolist()
                }
            
            # Add performance statistics
            if self.performance_window:
                perf_array = np.array(self.performance_window)
                stats['performance'] = {
                    'mean': np.mean(perf_array),
                    'std': np.std(perf_array),
                    'trend': self._calculate_trend(perf_array)
                }
            
            # Add feature importance
            stats['feature_importance'] = self.analyze_feature_importance()
            
            return stats
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend in values"""
        
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"
    
    def save_state(self, path: str):
        """Save monitor state"""
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with self.lock:
            # Save reference distribution
            if self.reference_distribution is not None:
                np.save(save_path / 'reference_distribution.npy', 
                       self.reference_distribution)
            
            # Save statistics
            stats = self.get_statistics()
            with open(save_path / 'statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Save detection history
            with open(save_path / 'detection_history.pkl', 'wb') as f:
                pickle.dump(list(self.detection_history), f)
        
        logger.info(f"Distribution monitor state saved to {save_path}")
    
    def load_state(self, path: str):
        """Load monitor state"""
        
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"State path {load_path} not found")
            return
        
        with self.lock:
            # Load reference distribution
            ref_file = load_path / 'reference_distribution.npy'
            if ref_file.exists():
                self.reference_distribution = np.load(ref_file)
                self.set_reference(self.reference_distribution)
            
            # Load detection history
            hist_file = load_path / 'detection_history.pkl'
            if hist_file.exists():
                with open(hist_file, 'rb') as f:
                    history = safe_pickle_load(f)
                    self.detection_history = deque(history, maxlen=1000)
        
        logger.info(f"Distribution monitor state loaded from {load_path}")