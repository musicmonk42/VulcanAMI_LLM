"""
Data Quality Score (DQS) System

This module provides comprehensive data quality scoring with support for
multiple scoring models, historical tracking, anomaly detection, and reporting.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DQSComponents:
    """
    Components that contribute to the Data Quality Score

    Attributes:
        pii_confidence: Confidence that data contains PII (0-1, higher = more PII)
        graph_completeness: Completeness of graph structure (0-1, higher = more complete)
        syntactic_completeness: Syntactic correctness and completeness (0-1)
    """

    pii_confidence: float
    graph_completeness: float
    syntactic_completeness: float

    def __post_init__(self):
        """Validate component values"""
        for field_name in [
            "pii_confidence",
            "graph_completeness",
            "syntactic_completeness",
        ]:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "pii_confidence": self.pii_confidence,
            "graph_completeness": self.graph_completeness,
            "syntactic_completeness": self.syntactic_completeness,
        }


@dataclass
class DQSResult:
    """
    Result of a DQS computation

    Attributes:
        score: Final DQS score (0-1)
        components: Individual component scores
        gate_decision: Gate decision (allow, quarantine, reject)
        timestamp: When the score was computed
        metadata: Additional metadata
    """

    score: float
    components: DQSComponents
    gate_decision: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization"""
        return {
            "score": self.score,
            "components": self.components.to_dict(),
            "gate_decision": self.gate_decision,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> DQSResult:
        """Create from dictionary"""
        return cls(
            score=data["score"],
            components=DQSComponents(**data["components"]),
            gate_decision=data["gate_decision"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


def compute_dqs(
    comp: DQSComponents, weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute Data Quality Score from components.

    The default formula emphasizes graph completeness and syntactic completeness
    while penalizing PII presence.

    Args:
        comp: DQS components
        weights: Optional custom weights (default: pii=0.3, graph=0.4, syntactic=0.3)

    Returns:
        DQS score between 0 and 1
    """
    if weights is None:
        weights = {
            "pii_confidence": 0.3,
            "graph_completeness": 0.4,
            "syntactic_completeness": 0.3,
        }

    # PII confidence is inverted (higher PII = lower quality)
    pii_weight = weights.get("pii_confidence", 0.3)
    graph_weight = weights.get("graph_completeness", 0.4)
    syntactic_weight = weights.get("syntactic_completeness", 0.3)

    score = (
        pii_weight * (1 - comp.pii_confidence)
        + graph_weight * comp.graph_completeness
        + syntactic_weight * comp.syntactic_completeness
    )

    return max(0.0, min(1.0, score))  # Clamp to [0, 1]


def compute_dqs_v2(
    comp: DQSComponents,
    weights: Optional[Dict[str, float]] = None,
    pii_penalty_factor: float = 1.5,
) -> float:
    """
    Enhanced DQS computation with non-linear PII penalty.

    This version applies a stronger penalty for high PII confidence.

    Args:
        comp: DQS components
        weights: Optional custom weights
        pii_penalty_factor: Multiplier for PII penalty (>1 = harsher penalty)

    Returns:
        DQS score between 0 and 1
    """
    if weights is None:
        weights = {
            "pii_confidence": 0.3,
            "graph_completeness": 0.4,
            "syntactic_completeness": 0.3,
        }

    # Non-linear PII penalty
    pii_penalty = comp.pii_confidence**pii_penalty_factor
    pii_weight = weights.get("pii_confidence", 0.3)
    graph_weight = weights.get("graph_completeness", 0.4)
    syntactic_weight = weights.get("syntactic_completeness", 0.3)

    score = (
        pii_weight * (1 - pii_penalty)
        + graph_weight * comp.graph_completeness
        + syntactic_weight * comp.syntactic_completeness
    )

    return max(0.0, min(1.0, score))


def gate(dqs: float, reject_below: float, quarantine_below: float) -> str:
    """
    Make a gate decision based on DQS score.

    Args:
        dqs: DQS score
        reject_below: Reject threshold
        quarantine_below: Quarantine threshold

    Returns:
        "reject", "quarantine", or "allow"
    """
    if dqs < reject_below:
        return "reject"
    if dqs < quarantine_below:
        return "quarantine"
    return "allow"


class DQSScorer:
    """
    Advanced DQS scorer with multiple scoring models and configuration.

    This class provides:
    - Multiple scoring algorithms
    - Configurable weights and thresholds
    - Batch scoring capabilities
    - Score normalization
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        reject_below: float = 0.3,
        quarantine_below: float = 0.4,
        model: str = "default",
    ):
        """
        Initialize DQS scorer.

        Args:
            weights: Component weights
            reject_below: Rejection threshold
            quarantine_below: Quarantine threshold
            model: Scoring model ("default", "v2", "strict")
        """
        self.weights = weights or {
            "pii_confidence": 0.3,
            "graph_completeness": 0.4,
            "syntactic_completeness": 0.3,
        }
        self.reject_below = reject_below
        self.quarantine_below = quarantine_below
        self.model = model

        # Validate thresholds
        if not 0 <= reject_below <= 1:
            raise ValueError("reject_below must be between 0 and 1")
        if not 0 <= quarantine_below <= 1:
            raise ValueError("quarantine_below must be between 0 and 1")
        if reject_below > quarantine_below:
            raise ValueError("reject_below must be <= quarantine_below")

        logger.info(
            f"Initialized DQS Scorer with model={model}, "
            f"reject={reject_below}, quarantine={quarantine_below}"
        )

    def score(self, comp: DQSComponents) -> DQSResult:
        """
        Score a data item.

        Args:
            comp: DQS components

        Returns:
            DQSResult with score and gate decision
        """
        if self.model == "v2":
            score = compute_dqs_v2(comp, self.weights)
        elif self.model == "strict":
            score = compute_dqs_v2(comp, self.weights, pii_penalty_factor=2.0)
        else:
            score = compute_dqs(comp, self.weights)

        decision = gate(score, self.reject_below, self.quarantine_below)

        result = DQSResult(score=score, components=comp, gate_decision=decision)

        logger.debug(f"Scored item: {score:.3f} -> {decision}")

        return result

    def score_batch(self, items: List[DQSComponents]) -> List[DQSResult]:
        """
        Score multiple items in batch.

        Args:
            items: List of DQS components

        Returns:
            List of DQS results
        """
        results = [self.score(comp) for comp in items]
        logger.info(f"Scored batch of {len(results)} items")
        return results

    def get_config(self) -> Dict[str, any]:
        """Get current scorer configuration"""
        return {
            "model": self.model,
            "weights": self.weights,
            "reject_below": self.reject_below,
            "quarantine_below": self.quarantine_below,
        }


class DQSTracker:
    """
    Track DQS scores over time for analysis and anomaly detection.

    This class maintains a history of scores and provides:
    - Statistical analysis
    - Trend detection
    - Anomaly detection
    - Reporting
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize DQS tracker.

        Args:
            max_history: Maximum number of scores to keep in memory
        """
        self.max_history = max_history
        self.history: deque[DQSResult] = deque(maxlen=max_history)
        self.total_scored = 0
        self.decision_counts: Dict[str, int] = {
            "allow": 0,
            "quarantine": 0,
            "reject": 0,
        }

        logger.info(f"Initialized DQS Tracker with max_history={max_history}")

    def record(self, result: DQSResult) -> None:
        """
        Record a DQS result.

        Args:
            result: DQS result to record
        """
        self.history.append(result)
        self.total_scored += 1
        self.decision_counts[result.gate_decision] += 1

        logger.debug(
            f"Recorded DQS result: {result.score:.3f} -> {result.gate_decision}"
        )

    def record_batch(self, results: List[DQSResult]) -> None:
        """
        Record multiple DQS results.

        Args:
            results: List of DQS results
        """
        for result in results:
            self.record(result)

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistical summary of tracked scores.

        Returns:
            Dictionary with statistics
        """
        if not self.history:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "stddev": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        scores = [r.score for r in self.history]

        return {
            "count": len(scores),
            "total_scored": self.total_scored,
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stddev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "decision_counts": self.decision_counts.copy(),
        }

    def get_recent_trend(self, window_size: int = 100) -> str:
        """
        Detect recent trend in DQS scores.

        Args:
            window_size: Number of recent scores to analyze

        Returns:
            "improving", "stable", or "degrading"
        """
        if len(self.history) < window_size:
            return "insufficient_data"

        recent = list(self.history)[-window_size:]
        first_half = [r.score for r in recent[: window_size // 2]]
        second_half = [r.score for r in recent[window_size // 2 :]]

        mean_first = statistics.mean(first_half)
        mean_second = statistics.mean(second_half)

        diff = mean_second - mean_first

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"

    def detect_anomalies(self, threshold_stddev: float = 3.0) -> List[DQSResult]:
        """
        Detect anomalous scores using statistical methods.

        Args:
            threshold_stddev: Number of standard deviations for anomaly

        Returns:
            List of anomalous results
        """
        if len(self.history) < 10:
            return []

        scores = [r.score for r in self.history]
        mean = statistics.mean(scores)
        stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        if stddev == 0:
            return []

        anomalies = []
        for result in self.history:
            z_score = abs((result.score - mean) / stddev)
            if z_score > threshold_stddev:
                anomalies.append(result)

        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def get_percentile(self, percentile: float) -> float:
        """
        Get score at given percentile.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Score at that percentile
        """
        if not self.history:
            return 0.0

        scores = sorted([r.score for r in self.history])
        index = int(len(scores) * percentile / 100)
        return scores[min(index, len(scores) - 1)]

    def generate_report(self) -> str:
        """
        Generate a text report of DQS statistics.

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        trend = self.get_recent_trend()

        report = f"""
DQS Tracking Report
===================

Summary Statistics:
  Total Scored: {stats["total_scored"]}
  History Size: {stats["count"]}
  Mean Score: {stats["mean"]:.3f}
  Median Score: {stats["median"]:.3f}
  Std Dev: {stats["stddev"]:.3f}
  Range: [{stats["min"]:.3f}, {stats["max"]:.3f}]

Gate Decisions:
  Allowed: {stats["decision_counts"]["allow"]} ({stats["decision_counts"]["allow"] / max(1, stats["total_scored"]) * 100:.1f}%)
  Quarantined: {stats["decision_counts"]["quarantine"]} ({stats["decision_counts"]["quarantine"] / max(1, stats["total_scored"]) * 100:.1f}%)
  Rejected: {stats["decision_counts"]["reject"]} ({stats["decision_counts"]["reject"] / max(1, stats["total_scored"]) * 100:.1f}%)

Recent Trend: {trend.upper()}

Percentiles:
  P10: {self.get_percentile(10):.3f}
  P25: {self.get_percentile(25):.3f}
  P50: {self.get_percentile(50):.3f}
  P75: {self.get_percentile(75):.3f}
  P90: {self.get_percentile(90):.3f}
  P95: {self.get_percentile(95):.3f}
  P99: {self.get_percentile(99):.3f}
"""
        return report

    def save(self, path: Path) -> None:
        """
        Save tracking history to file.

        Args:
            path: Path to save to
        """
        data = {
            "total_scored": self.total_scored,
            "decision_counts": self.decision_counts,
            "history": [r.to_dict() for r in self.history],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved DQS history to {path}")

    @classmethod
    def load(cls, path: Path) -> DQSTracker:
        """
        Load tracking history from file.

        Args:
            path: Path to load from

        Returns:
            Restored DQSTracker
        """
        with open(path, "r") as f:
            data = json.load(f)

        tracker = cls()
        tracker.total_scored = data["total_scored"]
        tracker.decision_counts = data["decision_counts"]
        tracker.history = deque(
            [DQSResult.from_dict(r) for r in data["history"]],
            maxlen=tracker.max_history,
        )

        logger.info(f"Loaded DQS history from {path}")
        return tracker


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Basic DQS Computation ===")
    comp = DQSComponents(
        pii_confidence=0.2, graph_completeness=0.8, syntactic_completeness=0.9
    )

    score = compute_dqs(comp)
    decision = gate(score, 0.3, 0.4)
    print(f"Components: {comp.to_dict()}")
    print(f"DQS Score: {score:.3f}")
    print(f"Gate Decision: {decision}")

    print("\n=== Testing DQS Scorer ===")
    scorer = DQSScorer(model="v2")
    result = scorer.score(comp)
    print(f"Result: {result.score:.3f} -> {result.gate_decision}")

    print("\n=== Testing DQS Tracker ===")
    tracker = DQSTracker()

    # Simulate scoring multiple items
    for i in range(100):
        import random

        test_comp = DQSComponents(
            pii_confidence=random.uniform(0, 0.5),
            graph_completeness=random.uniform(0.5, 1.0),
            syntactic_completeness=random.uniform(0.6, 1.0),
        )
        test_result = scorer.score(test_comp)
        tracker.record(test_result)

    # Generate report
    print(tracker.generate_report())

    # Detect anomalies
    anomalies = tracker.detect_anomalies(threshold_stddev=2.0)
    print(f"\nDetected {len(anomalies)} anomalies")
