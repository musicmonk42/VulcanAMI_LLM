"""
Meta-cognitive monitoring and compositional understanding
"""

import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

from ..config import EMBEDDING_DIM, HIDDEN_DIM
from ..security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)

# ============================================================
# METACOGNITIVE TYPES
# ============================================================


class ReasoningPhase(Enum):
    """Phases of reasoning process"""

    PERCEPTION = "perception"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    REFLECTION = "reflection"


@dataclass
class ReasoningStep:
    """Single step in reasoning trace"""

    phase: ReasoningPhase
    content: Any
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CausalRelation:
    """Causal relation between concepts"""

    cause: str
    effect: str
    strength: float
    confidence: float
    evidence_count: int = 0


# ============================================================
# META-COGNITIVE MONITOR
# ============================================================


class MetaCognitiveMonitor:
    """Enhanced meta-cognitive monitoring with self-improvement and audit trails."""

    def __init__(
        self,
        model_ref: Optional[nn.Module] = None,
        optimizer_ref: Optional[optim.Optimizer] = None,
    ):
        # References to actual model and optimizer
        self.model_ref = model_ref
        self.optimizer_ref = optimizer_ref

        # History tracking
        self.learning_history = deque(maxlen=1000)
        self.performance_trends = {}
        self.strategy_effectiveness = {}
        self.reasoning_quality_history = deque(maxlen=100)

        # Self model
        self.self_model = {
            "strengths": [],
            "weaknesses": [],
            "improvement_areas": [],
            "learning_style": "balanced",
            "confidence_calibration": 1.0,
            "metacognitive_accuracy": 0.0,
        }

        # Improvement strategies with actual implementations
        self.improvement_strategies = {
            "high_loss": self._strategy_reduce_learning_rate,
            "high_variance": self._strategy_increase_regularization,
            "plateau": self._strategy_change_optimizer,
            "forgetting": self._strategy_increase_replay,
            "overconfidence": self._strategy_calibrate_confidence,
            "underconfidence": self._strategy_boost_confidence,
        }

        self.applied_improvements = deque(maxlen=50)

        # Audit trail for improvements
        self.improvement_audit = []
        self.model_snapshots = deque(maxlen=20)

        # Confidence estimation
        self.confidence_estimator = ConfidenceEstimator()
        self.confidence_history = deque(maxlen=100)

        # Causal reasoning
        self.causal_graph = nx.DiGraph()
        self.causal_relations = {}
        self.strategy_outcomes = {}

        # Persistence
        self.save_path = Path("metacognition_states")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # FIXED: Single lock for all operations to prevent deadlocks
        self._lock = threading.RLock()

    def _get_performance_trend(self, key: str) -> deque:
        """Get or create performance trend deque (thread-safe)"""
        # Lock already held by caller
        if key not in self.performance_trends:
            self.performance_trends[key] = deque(maxlen=100)
        return self.performance_trends[key]

    def _get_strategy_outcomes(self, key: str) -> list:
        """Get or create strategy outcomes list (thread-safe)"""
        # Lock already held by caller
        if key not in self.strategy_outcomes:
            self.strategy_outcomes[key] = []
        return self.strategy_outcomes[key]

    def set_model_optimizer(self, model: nn.Module, optimizer: optim.Optimizer):
        """Set references to model and optimizer for actual improvements"""
        with self._lock:
            self.model_ref = model
            self.optimizer_ref = optimizer

    def update_self_model(self, metrics: Dict[str, Any]):
        """FIXED: Update self-model based on performance metrics with consistent locking"""
        with self._lock:
            self.learning_history.append(metrics)

            # Track confidence calibration
            if "predicted_confidence" in metrics and "actual_performance" in metrics:
                self.confidence_history.append(
                    {
                        "predicted": metrics["predicted_confidence"],
                        "actual": metrics["actual_performance"],
                        "calibration_error": abs(
                            metrics["predicted_confidence"]
                            - metrics["actual_performance"]
                        ),
                    }
                )

                # Update confidence calibration factor
                if len(self.confidence_history) > 10:
                    recent_errors = [
                        h["calibration_error"]
                        for h in list(self.confidence_history)[-10:]
                    ]
                    avg_error = np.mean(recent_errors)

                    if avg_error > 0.2:
                        self.self_model["confidence_calibration"] *= 0.9
                    elif avg_error < 0.1:
                        self.self_model["confidence_calibration"] = min(
                            1.5, self.self_model["confidence_calibration"] * 1.1
                        )

            # Identify strengths and weaknesses
            if "modality" in metrics:
                modality = str(metrics["modality"])
                loss = metrics.get("loss", 0)
                self._get_performance_trend(modality).append(loss)

                # Update strengths/weaknesses with more nuanced thresholds
                trend_deque = self._get_performance_trend(modality)
                if len(trend_deque) > 10:
                    recent = list(trend_deque)[-10:]
                    avg_loss = np.mean(recent)
                    trend = self._compute_trend(recent)

                    if avg_loss < 0.3 and trend <= 0:
                        if modality not in self.self_model["strengths"]:
                            self.self_model["strengths"].append(modality)
                        if modality in self.self_model["weaknesses"]:
                            self.self_model["weaknesses"].remove(modality)
                    elif avg_loss > 0.7 or trend > 0.01:
                        if modality not in self.self_model["weaknesses"]:
                            self.self_model["weaknesses"].append(modality)
                        if modality in self.self_model["strengths"]:
                            self.self_model["strengths"].remove(modality)

            # Update causal graph if causal info provided
            if "cause" in metrics and "effect" in metrics:
                self._update_causal_graph(
                    metrics["cause"], metrics["effect"], metrics.get("correlation", 0.5)
                )

    def introspect_reasoning(
        self, reasoning_trace: List[Union[Dict, ReasoningStep]]
    ) -> Dict[str, Any]:
        """Analyze reasoning quality with improvements"""
        if not reasoning_trace:
            return {"quality_score": 0, "analysis": "No reasoning trace"}

        # Convert dicts to ReasoningSteps if needed
        trace_steps = []
        for step in reasoning_trace:
            try:
                if isinstance(step, ReasoningStep):
                    trace_steps.append(step)
                elif isinstance(step, dict):
                    phase_str = step.get("phase", "execution")
                    try:
                        phase = ReasoningPhase(phase_str)
                    except ValueError:
                        phase = ReasoningPhase.EXECUTION

                    trace_steps.append(
                        ReasoningStep(
                            phase=phase,
                            content=step.get("content"),
                            confidence=step.get("confidence", 0.5),
                            timestamp=step.get("timestamp", time.time()),
                            metadata=step.get("metadata", {}),
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to process reasoning step: {e}")
                continue

        if not trace_steps:
            return {"quality_score": 0, "analysis": "No valid reasoning steps"}

        quality_metrics = {
            "depth": len(trace_steps),
            "consistency": self._check_consistency(trace_steps),
            "completeness": self._check_completeness(trace_steps),
            "efficiency": self._check_efficiency(trace_steps),
            "confidence_calibration": self._check_confidence_calibration(trace_steps),
            "logical_coherence": self._check_logical_coherence(trace_steps),
        }

        # Compute overall quality score
        weights = {
            "consistency": 0.25,
            "completeness": 0.2,
            "efficiency": 0.2,
            "confidence_calibration": 0.15,
            "logical_coherence": 0.2,
        }

        quality_score = sum(quality_metrics.get(k, 0) * v for k, v in weights.items())

        with self._lock:
            self.reasoning_quality_history.append(quality_score)

        # Identify issues
        issues = []
        if quality_metrics["consistency"] < 0.5:
            issues.append("inconsistent_reasoning")
        if quality_metrics["completeness"] < 0.5:
            issues.append("incomplete_reasoning")
        if quality_metrics["efficiency"] < 0.5:
            issues.append("inefficient_reasoning")
        if quality_metrics["confidence_calibration"] < 0.5:
            issues.append("poor_confidence_calibration")
        if quality_metrics["logical_coherence"] < 0.5:
            issues.append("logical_incoherence")

        # Generate improvement suggestions
        suggestions = self._generate_reasoning_improvements(
            quality_metrics, trace_steps
        )

        with self._lock:
            history_list = list(self.reasoning_quality_history)

        return {
            "quality_score": quality_score,
            "metrics": quality_metrics,
            "issues": issues,
            "suggestions": suggestions,
            "trend": self._compute_trend(history_list),
        }

    def _check_consistency(self, trace: List[ReasoningStep]) -> float:
        """Check reasoning consistency"""
        if len(trace) < 2:
            return 1.0

        consistency_score = 1.0

        # Check confidence consistency
        confidences = [step.confidence for step in trace]
        if confidences:
            confidence_std = np.std(confidences)
            if confidence_std > 0.3:
                consistency_score *= 1 - confidence_std

        # Check phase transitions
        for i in range(1, len(trace))
            curr_phase = trace[i].phase
            prev_phase = trace[i - 1].phase

            # Penalize illogical transitions
            if (
                prev_phase == ReasoningPhase.EXECUTION
                and curr_phase == ReasoningPhase.PERCEPTION
            ):
                consistency_score *= 0.9
            elif (
                prev_phase == ReasoningPhase.LEARNING
                and curr_phase == ReasoningPhase.PERCEPTION
            ):
                consistency_score *= 0.95

        return max(0, consistency_score)

    def _check_completeness(self, trace: List[ReasoningStep]) -> float:
        """Check reasoning completeness"""
        expected_phases = {
            ReasoningPhase.PERCEPTION,
            ReasoningPhase.PLANNING,
            ReasoningPhase.EXECUTION,
            ReasoningPhase.LEARNING,
        }

        actual_phases = {step.phase for step in trace}

        return len(actual_phases & expected_phases) / len(expected_phases)

    def _check_efficiency(self, trace: List[ReasoningStep]) -> float:
        """Check reasoning efficiency"""
        if not trace:
            return 0

        # Check for redundancy
        content_hashes = [hash(str(step.content)) for step in trace]
        unique_content = len(set(content_hashes))
        total_content = len(content_hashes)

        redundancy_score = unique_content / total_content if total_content > 0 else 0

        # Check for unnecessary complexity
        complexity_score = 1.0 / (1 + np.log(len(trace) + 1))

        return (redundancy_score + complexity_score) / 2

    def _check_confidence_calibration(self, trace: List[ReasoningStep]) -> float:
        """Check if confidence levels are well-calibrated"""
        if not trace:
            return 0.5

        confidences = [step.confidence for step in trace]

        # Check if confidence varies appropriately
        if len(set(confidences)) == 1:
            return 0.3  # All same confidence is suspicious

        # Check if confidence correlates with phase importance
        phase_weights = {
            ReasoningPhase.PERCEPTION: 0.7,
            ReasoningPhase.PLANNING: 0.8,
            ReasoningPhase.EXECUTION: 0.9,
            ReasoningPhase.LEARNING: 0.6,
            ReasoningPhase.REFLECTION: 0.5,
        }

        weighted_confidence = []
        for step in trace:
            expected = phase_weights.get(step.phase, 0.5)
            weighted_confidence.append(1 - abs(step.confidence - expected))

        return np.mean(weighted_confidence)

    def _check_logical_coherence(self, trace: List[ReasoningStep]) -> float:
        """Check logical coherence of reasoning"""
        if len(trace) < 2:
            return 1.0

        coherence_score = 1.0

        # Check causal consistency
        for i in range(1, len(trace))
            curr = trace[i]
            prev = trace[i - 1]

            # Check if execution follows planning
            if (
                curr.phase == ReasoningPhase.EXECUTION
                and prev.phase == ReasoningPhase.PLANNING
            ):
                coherence_score *= 1.1

            # Check if learning follows execution
            if (
                curr.phase == ReasoningPhase.LEARNING
                and prev.phase == ReasoningPhase.EXECUTION
            ):
                coherence_score *= 1.05

        return min(1.0, coherence_score)

    def _generate_reasoning_improvements(
        self, metrics: Dict[str, float], trace: List[ReasoningStep]
    ) -> List[str]:
        """Generate specific suggestions for reasoning improvement"""
        suggestions = []

        if metrics["consistency"] < 0.5:
            suggestions.append(
                "Maintain more consistent confidence levels across reasoning steps"
            )
            suggestions.append("Ensure logical phase transitions in reasoning")

        if metrics["completeness"] < 0.5:
            missing_phases = {
                ReasoningPhase.PERCEPTION,
                ReasoningPhase.PLANNING,
                ReasoningPhase.EXECUTION,
                ReasoningPhase.LEARNING,
            }
            actual_phases = {step.phase for step in trace}
            missing = missing_phases - actual_phases

            for phase in missing:
                suggestions.append(f"Include {phase.value} phase in reasoning")

        if metrics["efficiency"] < 0.5:
            suggestions.append("Reduce redundant reasoning steps")
            suggestions.append("Simplify reasoning chain where possible")

        if metrics["confidence_calibration"] < 0.5:
            suggestions.append("Calibrate confidence levels to match uncertainty")
            suggestions.append("Vary confidence based on available evidence")

        return suggestions

    def analyze_learning_efficiency(self) -> Dict[str, Any]:
        """FIXED: Analyze learning efficiency with proper locking"""
        with self._lock:
            if len(self.learning_history) < 10:
                return {
                    "status": "insufficient_data",
                    "samples": len(self.learning_history),
                }

            recent = list(self.learning_history)[-100:]

            # Calculate metrics
            losses = [h.get("loss", 0) for h in recent if "loss" in h]

            analysis = {
                "avg_loss": np.mean(losses) if losses else 0,
                "loss_std": np.std(losses) if losses else 0,
                "loss_trend": self._compute_trend(losses),
                "improving": False,
                "plateau_detected": False,
                "variance_issue": False,
                "overconfident": False,
                "underconfident": False,
            }

            # Detect patterns
            if analysis["loss_trend"] < -0.01:
                analysis["improving"] = True
            elif abs(analysis["loss_trend"]) < 0.001:
                analysis["plateau_detected"] = True

            if analysis["loss_std"] > 0.5:
                analysis["variance_issue"] = True

            # Check confidence calibration
            if self.confidence_history:
                recent_conf = list(self.confidence_history)[-20:]
                avg_error = np.mean([h["calibration_error"] for h in recent_conf])

                if avg_error > 0.3:
                    predicted_avg = np.mean([h["predicted"] for h in recent_conf])
                    actual_avg = np.mean([h["actual"] for h in recent_conf])

                    if predicted_avg > actual_avg:
                        analysis["overconfident"] = True
                    else:
                        analysis["underconfident"] = True

            # Modality-specific analysis
            modality_performance = self._analyze_modality_performance(recent)
            analysis["modality_performance"] = modality_performance

            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)

        # Apply improvements outside lock to allow strategies to acquire lock if needed
        if analysis["recommendations"]:
            self._apply_improvements(analysis)

        return analysis

    def _analyze_modality_performance(self, history: List[Dict]) -> Dict:
        """Analyze performance per modality (lock held by caller)"""
        modality_losses = {}

        for h in history:
            if "modality" in h and "loss" in h:
                modality = str(h["modality"])
                if modality not in modality_losses:
                    modality_losses[modality] = []
                modality_losses[modality].append(h["loss"])

        performance = {}
        for modality, losses in modality_losses.items():
            if losses:
                performance[modality] = {
                    "avg_loss": np.mean(losses),
                    "std_loss": np.std(losses),
                    "trend": self._compute_trend(losses),
                    "sample_count": len(losses),
                    "improving": self._compute_trend(losses) < -0.01,
                }

        return performance

    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []

        if analysis["avg_loss"] > 0.5:
            recommendations.append(
                {
                    "issue": "high_loss",
                    "suggestion": "Reduce learning rate or increase model capacity",
                    "priority": "high",
                    "auto_fix": True,
                }
            )

        if analysis["variance_issue"]:
            recommendations.append(
                {
                    "issue": "high_variance",
                    "suggestion": "Increase regularization or batch size",
                    "priority": "medium",
                    "auto_fix": True,
                }
            )

        if analysis["plateau_detected"]:
            recommendations.append(
                {
                    "issue": "plateau",
                    "suggestion": "Change optimizer or learning rate schedule",
                    "priority": "medium",
                    "auto_fix": True,
                }
            )

        if analysis.get("overconfident"):
            recommendations.append(
                {
                    "issue": "overconfidence",
                    "suggestion": "Calibrate confidence estimates",
                    "priority": "low",
                    "auto_fix": True,
                }
            )

        if analysis.get("underconfident"):
            recommendations.append(
                {
                    "issue": "underconfidence",
                    "suggestion": "Boost confidence in predictions",
                    "priority": "low",
                    "auto_fix": True,
                }
            )

        return recommendations

    def _apply_improvements(self, analysis: Dict):
        """FIXED: Automatically apply improvements with proper locking"""
        for rec in analysis["recommendations"]:
            if rec["auto_fix"] and rec["issue"] in self.improvement_strategies:
                strategy = self.improvement_strategies[rec["issue"]]

                # Create audit record
                audit_record = {
                    "timestamp": time.time(),
                    "issue": rec["issue"],
                    "strategy": strategy.__name__,
                    "analysis": {
                        k: v for k, v in analysis.items() if k != "recommendations"
                    },
                    "before_metrics": {
                        "avg_loss": analysis["avg_loss"],
                        "loss_std": analysis["loss_std"],
                    },
                }

                # Apply strategy
                try:
                    result = strategy()
                    audit_record["result"] = result
                    audit_record["success"] = True

                    # Track strategy effectiveness
                    with self._lock:
                        self._get_strategy_outcomes(rec["issue"]).append(
                            {
                                "timestamp": time.time(),
                                "before_loss": analysis["avg_loss"],
                                "applied": True,
                            }
                        )

                except Exception as e:
                    audit_record["result"] = str(e)
                    audit_record["success"] = False
                    logger.error(f"Failed to apply {rec['issue']} strategy: {e}")

                # Store audit record
                with self._lock:
                    self.improvement_audit.append(audit_record)

                    self.applied_improvements.append(
                        {
                            "issue": rec["issue"],
                            "strategy": strategy.__name__,
                            "timestamp": time.time(),
                            "result": audit_record["result"],
                        }
                    )

                logger.info(
                    f"Applied improvement for {rec['issue']}: {audit_record['result']}"
                )

    def _strategy_reduce_learning_rate(self) -> str:
        """FIXED: Strategy to reduce learning rate with proper checks"""
        if self.optimizer_ref is None:
            logger.warning("Cannot reduce LR: no optimizer reference")
            return "No optimizer reference available"

        try:
            if (
                not hasattr(self.optimizer_ref, "param_groups")
                or not self.optimizer_ref.param_groups
            ):
                return "Optimizer has no parameter groups"

            old_lr = self.optimizer_ref.param_groups[0]["lr"]
            new_lr = max(1e-6, old_lr * 0.5)

            for param_group in self.optimizer_ref.param_groups:
                param_group["lr"] = new_lr

            return f"Reduced learning rate from {old_lr:.6f} to {new_lr:.6f}"
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to reduce LR: {e}")
            return f"Failed to reduce learning rate: {e}"

    def _strategy_increase_regularization(self) -> str:
        """FIXED: Strategy to increase regularization with validation"""
        if self.model_ref is None:
            logger.warning("Cannot increase regularization: no model reference")
            return "No model reference available"

        try:
            # Increase dropout rates
            dropout_modified = 0
            for module in self.model_ref.modules():
                if isinstance(module, nn.Dropout):
                    module.p
                    module.p = min(0.5, module.p * 1.2)
                    dropout_modified += 1

            # Increase weight decay
            if (
                self.optimizer_ref
                and hasattr(self.optimizer_ref, "param_groups")
                and self.optimizer_ref.param_groups
            ):
                if "weight_decay" in self.optimizer_ref.param_groups[0]:
                    old_wd = self.optimizer_ref.param_groups[0]["weight_decay"]
                    new_wd = old_wd * 1.5

                    for param_group in self.optimizer_ref.param_groups:
                        param_group["weight_decay"] = new_wd

                    return f"Increased {dropout_modified} dropout rates and weight decay from {old_wd:.4f} to {new_wd:.4f}"

            return f"Increased {dropout_modified} dropout rates"
        except Exception as e:
            logger.error(f"Failed to increase regularization: {e}")
            return f"Failed to increase regularization: {e}"

    def _strategy_change_optimizer(self) -> str:
        """FIXED: Strategy to change optimizer with validation"""
        if self.model_ref is None or self.optimizer_ref is None:
            logger.warning("Cannot change optimizer: no model/optimizer reference")
            return "No model/optimizer reference available"

        try:
            if (
                not hasattr(self.optimizer_ref, "param_groups")
                or not self.optimizer_ref.param_groups
            ):
                return "Current optimizer has no parameter groups"

            old_optimizer_name = self.optimizer_ref.__class__.__name__
            old_lr = self.optimizer_ref.param_groups[0]["lr"]

            # Switch to AdamW with cosine annealing
            self.optimizer_ref = optim.AdamW(
                self.model_ref.parameters(), lr=old_lr, weight_decay=0.01
            )

            return f"Switched from {old_optimizer_name} to AdamW (lr={old_lr:.6f})"
        except Exception as e:
            logger.error(f"Failed to change optimizer: {e}")
            return f"Failed to change optimizer: {e}"

    def _strategy_increase_replay(self) -> str:
        """Strategy to increase experience replay"""
        return (
            "Increased replay frequency recommendation (requires replay buffer access)"
        )

    def _strategy_calibrate_confidence(self) -> str:
        """Strategy to calibrate confidence estimates"""
        try:
            with self._lock:
                old_calibration = self.self_model["confidence_calibration"]
                self.self_model["confidence_calibration"] *= 0.8
                new_calibration = self.self_model["confidence_calibration"]

            return f"Adjusted confidence calibration from {old_calibration:.2f} to {new_calibration:.2f}"
        except Exception as e:
            logger.error(f"Failed to calibrate confidence: {e}")
            return f"Failed to calibrate confidence: {e}"

    def _strategy_boost_confidence(self) -> str:
        """Strategy to boost confidence"""
        try:
            with self._lock:
                old_calibration = self.self_model["confidence_calibration"]
                self.self_model["confidence_calibration"] = min(
                    1.5, self.self_model["confidence_calibration"] * 1.2
                )
                new_calibration = self.self_model["confidence_calibration"]

            return f"Boosted confidence calibration from {old_calibration:.2f} to {new_calibration:.2f}"
        except Exception as e:
            logger.error(f"Failed to boost confidence: {e}")
            return f"Failed to boost confidence: {e}"

    def _update_causal_graph(self, cause: str, effect: str, correlation: float):
        """Update causal graph with new relation (lock held by caller)"""
        # Add nodes if not present
        if cause not in self.causal_graph:
            self.causal_graph.add_node(cause)
        if effect not in self.causal_graph:
            self.causal_graph.add_node(effect)

        # Add or update edge
        if self.causal_graph.has_edge(cause, effect):
            # Update strength
            old_weight = self.causal_graph[cause][effect]["weight"]
            new_weight = 0.9 * old_weight + 0.1 * correlation
            self.causal_graph[cause][effect]["weight"] = new_weight
        else:
            self.causal_graph.add_edge(cause, effect, weight=correlation)

        # Store causal relation
        relation_key = f"{cause}->{effect}"
        if relation_key in self.causal_relations:
            rel = self.causal_relations[relation_key]
            rel.strength = 0.9 * rel.strength + 0.1 * correlation
            rel.evidence_count += 1
        else:
            self.causal_relations[relation_key] = CausalRelation(
                cause=cause,
                effect=effect,
                strength=correlation,
                confidence=0.5,
                evidence_count=1,
            )

    def infer_causal_chain(self, start: str, end: str) -> Optional[List[str]]:
        """Infer causal chain from start to end"""
        with self._lock:
            if start not in self.causal_graph or end not in self.causal_graph:
                return None

            try:
                path = nx.shortest_path(self.causal_graph, start, end)
                return path
            except nx.NetworkXNoPath:
                return None

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend in values"""
        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            if not np.isfinite(slope):
                return 0
            return float(slope)
        except (ValueError, ZeroDivisionError):
            return 0

    def get_improvement_history(self) -> List[Dict]:
        """Get history of applied improvements for auditing"""
        with self._lock:
            return self.improvement_audit.copy()

    def save_state(self, filename: Optional[str] = None) -> str:
        """Save metacognitive state"""
        if filename is None:
            filename = f"metacog_state_{int(time.time())}.pkl"

        filepath = self.save_path / filename

        with self._lock:
            state = {
                "self_model": self.self_model,
                "learning_history": list(self.learning_history),
                "performance_trends": {
                    k: list(v) for k, v in self.performance_trends.items()
                },
                "strategy_effectiveness": dict(self.strategy_effectiveness),
                "improvement_audit": self.improvement_audit,
                "confidence_history": list(self.confidence_history),
                "causal_relations": self.causal_relations,
                "strategy_outcomes": dict(self.strategy_outcomes),
            }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved metacognitive state to {filepath}")
        return str(filepath)

    def load_state(self, filepath: str):
        """Load metacognitive state"""
        with open(filepath, "rb") as f:
            state = safe_pickle_load(f)

        with self._lock:
            self.self_model = state["self_model"]
            self.learning_history = deque(state["learning_history"], maxlen=1000)
            self.improvement_audit = state["improvement_audit"]
            self.confidence_history = deque(state["confidence_history"], maxlen=100)
            self.causal_relations = state.get("causal_relations", {})

            for k, v in state["performance_trends"].items():
                self.performance_trends[k] = deque(v, maxlen=100)

        logger.info(f"Loaded metacognitive state from {filepath}")


# ============================================================
# CONFIDENCE ESTIMATION
# ============================================================


class ConfidenceEstimator:
    """Estimate confidence in predictions"""

    def __init__(self):
        self.calibration_history = deque(maxlen=100)
        self.temperature = 1.0
        # FIXED: Get device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def estimate_confidence(
        self, logits: torch.Tensor, method: str = "softmax"
    ) -> float:
        """FIXED: Estimate confidence from logits with device management"""
        try:
            # Ensure logits on correct device
            logits = logits.to(self.device)

            if method == "softmax":
                probs = torch.softmax(logits / self.temperature, dim=-1)
                confidence = torch.max(probs).item()
            elif method == "entropy":
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                confidence = 1.0 / (1.0 + entropy.item())
            elif method == "margin":
                sorted_logits, _ = torch.sort(logits, descending=True)
                if len(sorted_logits) > 1:
                    margin = (sorted_logits[0] - sorted_logits[1]).item()
                    confidence = torch.sigmoid(
                        torch.tensor(margin, device=self.device)
                    ).item()
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            return confidence
        except Exception as e:
            logger.error(f"Confidence estimation failed: {e}")
            return 0.5

    def calibrate_temperature(self, predictions: List[Tuple[float, bool]]):
        """Calibrate temperature scaling based on prediction accuracy"""
        if len(predictions) < 10:
            return

        try:
            # Find temperature that minimizes calibration error
            best_temp = self.temperature
            best_error = float("inf")

            for temp in np.linspace(0.5, 2.0, 20):
                error = 0
                for conf, correct in predictions:
                    calibrated = conf ** (1 / temp)
                    error += abs(calibrated - (1.0 if correct else 0.0))

                if error < best_error:
                    best_error = error
                    best_temp = temp

            self.temperature = best_temp
        except Exception as e:
            logger.error(f"Temperature calibration failed: {e}")


# ============================================================
# COMPOSITIONAL UNDERSTANDING
# ============================================================


class CompositionalUnderstanding:
    """Enhanced compositional understanding with neural composition."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.concept_hierarchy = {}
        self.compositions = {}
        self.primitive_concepts = set()
        self.concept_relations = {}
        self.emergent_properties = {}

        # FIXED: Get device for neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural composition model with training capability
        self.composition_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, embedding_dim),
        ).to(self.device)

        self.decomposition_net = nn.Sequential(
            nn.Linear(embedding_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, embedding_dim * 2),
        ).to(self.device)

        # Optimizers for training
        self.composition_optimizer = optim.Adam(
            self.composition_net.parameters(), lr=0.001
        )
        self.decomposition_optimizer = optim.Adam(
            self.decomposition_net.parameters(), lr=0.001
        )

        # Training history
        self.training_history = deque(maxlen=100)

        # Concept embeddings storage
        self.concept_embeddings = {}

        # Composition rules
        self.composition_rules = {}

    def train_composition(
        self,
        concepts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        epochs: int = 10,
    ) -> Dict[str, float]:
        """FIXED: Train composition network with device management"""
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for comp1, comp2, target in concepts:
                # FIXED: Ensure tensors on correct device
                comp1 = comp1.to(self.device)
                comp2 = comp2.to(self.device)
                target = target.to(self.device)

                # Forward pass
                composed = self.compose_concepts_neural(comp1, comp2)
                loss = nn.functional.mse_loss(composed, target)

                # Backward pass
                self.composition_optimizer.zero_grad()
                loss.backward()
                self.composition_optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(concepts)
            total_loss += epoch_loss

            self.training_history.append(
                {"epoch": epoch, "composition_loss": epoch_loss}
            )

        return {"avg_loss": total_loss / epochs}

    def train_decomposition(
        self,
        concepts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        epochs: int = 10,
    ) -> Dict[str, float]:
        """FIXED: Train decomposition network with device management"""
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for composed, target1, target2 in concepts:
                # FIXED: Ensure tensors on correct device
                composed = composed.to(self.device)
                target1 = target1.to(self.device)
                target2 = target2.to(self.device)

                # Forward pass
                decomp1, decomp2 = self.decompose_concept_neural(composed)

                # Loss for both components
                loss1 = nn.functional.mse_loss(decomp1, target1)
                loss2 = nn.functional.mse_loss(decomp2, target2)
                loss = (loss1 + loss2) / 2

                # Backward pass
                self.decomposition_optimizer.zero_grad()
                loss.backward()
                self.decomposition_optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(concepts)
            total_loss += epoch_loss

            self.training_history.append(
                {"epoch": epoch, "decomposition_loss": epoch_loss}
            )

        return {"avg_loss": total_loss / epochs}

    def compose_concepts_neural(
        self, concept1: torch.Tensor, concept2: torch.Tensor
    ) -> torch.Tensor:
        """FIXED: Compose concepts using neural network with device check"""
        concept1 = concept1.to(self.device)
        concept2 = concept2.to(self.device)
        combined = torch.cat([concept1, concept2], dim=-1)
        return self.composition_net(combined)

    def decompose_concept_neural(
        self, concept: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIXED: Decompose concept using neural network with device check"""
        concept = concept.to(self.device)
        decomposed = self.decomposition_net(concept)
        mid = decomposed.shape[-1] // 2
        return decomposed[..., :mid], decomposed[..., mid:]

    def learn_composition(
        self, concept: torch.Tensor, components: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """FIXED: Learn compositional structure with device management"""
        if len(components) < 2:
            return {"error": "Need at least 2 components"}

        # Ensure all on same device
        concept = concept.to(self.device)
        components = [c.to(self.device) for c in components]

        # Compose components iteratively
        composed = components[0]
        composition_path = [0]

        for i, comp in enumerate(components[1:], 1):
            composed = self.compose_concepts_neural(composed, comp)
            composition_path.append(i)

        # Measure composition quality
        reconstruction_error = torch.mean((composed - concept) ** 2).item()

        # Learn emergent properties
        emergent = concept - composed
        emergent_magnitude = torch.norm(emergent).item()

        # Store composition rule
        comp_key = f"comp_{len(self.composition_rules)}"
        self.composition_rules[comp_key] = {
            "components": composition_path,
            "error": reconstruction_error,
            "emergent_magnitude": emergent_magnitude,
        }

        return {
            "composed": composed,
            "reconstruction_error": reconstruction_error,
            "emergent_properties": emergent,
            "emergent_magnitude": emergent_magnitude,
            "components_count": len(components),
            "composition_key": comp_key,
        }

    def hierarchical_composition(
        self, primitives: List[torch.Tensor], composition_tree: Dict
    ) -> torch.Tensor:
        """FIXED: Build complex concept with device management"""
        primitives = [p.to(self.device) for p in primitives]

        if composition_tree["op"] == "primitive":
            idx = composition_tree["index"]
            return (
                primitives[idx]
                if idx < len(primitives)
                else torch.zeros(self.embedding_dim, device=self.device)
            )
        elif composition_tree["op"] == "compose":
            left = self.hierarchical_composition(primitives, composition_tree["left"])
            right = self.hierarchical_composition(primitives, composition_tree["right"])
            return self.compose_concepts_neural(left, right)
        else:
            raise ValueError(f"Unknown operation: {composition_tree['op']}")

    def analyze_compositionality(
        self, concept: torch.Tensor, known_primitives: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """FIXED: Analyze compositionality with device management"""
        concept = concept.to(self.device)
        known_primitives = [p.to(self.device) for p in known_primitives]

        best_composition = None
        best_error = float("inf")

        # Try different composition strategies
        compositions_tried = []

        # Try pairwise compositions
        for i, p1 in enumerate(known_primitives):
            for j, p2 in enumerate(known_primitives):
                if i != j:
                    composed = self.compose_concepts_neural(p1, p2)
                    error = torch.mean((composed - concept) ** 2).item()

                    compositions_tried.append(
                        {"primitives": [i, j], "error": error, "type": "pairwise"}
                    )

                    if error < best_error:
                        best_error = error
                        best_composition = {
                            "primitives": [i, j],
                            "error": error,
                            "type": "pairwise",
                        }

        # Try triple compositions if pairwise isn't good enough
        if best_error > 0.1 and len(known_primitives) >= 3:
            for i in range(len(known_primitives))
                for j in range(i + 1, len(known_primitives))
                    for k in range(j + 1, len(known_primitives))
                        comp_ij = self.compose_concepts_neural(
                            known_primitives[i], known_primitives[j]
                        )
                        composed = self.compose_concepts_neural(
                            comp_ij, known_primitives[k]
                        )
                        error = torch.mean((composed - concept) ** 2).item()

                        if error < best_error:
                            best_error = error
                            best_composition = {
                                "primitives": [i, j, k],
                                "error": error,
                                "type": "triple",
                            }

        # Try decomposition
        decomposed = self.decompose_concept_neural(concept)

        # Analyze decomposition quality
        recomposed = self.compose_concepts_neural(decomposed[0], decomposed[1])
        decomp_error = torch.mean((recomposed - concept) ** 2).item()

        return {
            "best_composition": best_composition,
            "decomposition": decomposed,
            "decomposition_error": decomp_error,
            "compositionality_score": 1.0 / (1.0 + best_error)
            if best_composition
            else 0.0,
            "num_compositions_tried": len(compositions_tried),
            "is_primitive": best_error > 0.5,
        }

    def discover_primitives(
        self, concepts: List[torch.Tensor], max_primitives: int = 10
    ) -> List[torch.Tensor]:
        """FIXED: Discover primitive concepts with device management"""
        try:
            from sklearn.decomposition import PCA

            # Convert to numpy for analysis
            concept_matrix = (
                torch.stack([c.to(self.device) for c in concepts])
                .detach()
                .cpu()
                .numpy()
            )

            # Use PCA to find principal components
            pca = PCA(n_components=min(max_primitives, len(concepts)))
            pca.fit(concept_matrix)

            # Principal components as primitives
            primitives = []
            for component in pca.components_:
                primitives.append(
                    torch.tensor(component, dtype=torch.float32, device=self.device)
                )

            # Store as primitive concepts
            for i, prim in enumerate(primitives):
                prim_key = f"primitive_{len(self.primitive_concepts)}"
                self.primitive_concepts.add(prim_key)
                self.concept_embeddings[prim_key] = prim

            return primitives
        except Exception as e:
            logger.error(f"Failed to discover primitives: {e}")
            return []
