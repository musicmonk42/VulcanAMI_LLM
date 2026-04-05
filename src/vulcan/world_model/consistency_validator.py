"""
consistency_validator.py - Validates and maintains model consistency for the World Model.

Extracted from world_model_core.py to reduce file size and improve modularity.

Contains:
- ConsistencyValidator: Performs structural, logical, invariant, and calibration checks.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsistencyValidator:
    """Validates and maintains model consistency"""

    def __init__(self, world_model):
        self.world_model = world_model
        self.last_validation_time = time.time()
        self.validation_interval = 300  # 5 minutes

    def validate_if_needed(self) -> Optional[Dict[str, Any]]:
        """Validate model consistency if enough time has passed"""

        if time.time() - self.last_validation_time < self.validation_interval:
            return None

        result = self.validate()
        self.last_validation_time = time.time()
        return result

    def validate(self) -> Dict[str, Any]:
        """Perform comprehensive validation"""
        from . import world_model_core

        issues = []

        # EXAMINE: Check for various issues
        issues.extend(self._check_structural_issues())
        issues.extend(self._check_logical_issues())
        issues.extend(self._check_invariant_issues())
        issues.extend(self._check_calibration_issues())

        # SELECT: Determine if auto-fix is needed
        critical_issues = [i for i in issues if i["severity"] == "high"]

        # APPLY: Auto-fix critical issues
        if critical_issues:
            self._auto_fix_critical_issues(critical_issues)

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "model_version": self.world_model.model_version,
            "observation_count": self.world_model.observation_count,
            "intervention_count": (
                self.world_model.intervention_manager.intervention_count
                if world_model_core.INTERVENTION_MANAGER_AVAILABLE
                else 0
            ),
        }

    def _check_structural_issues(self) -> List[Dict[str, Any]]:
        """Check for structural problems"""
        from . import world_model_core

        issues = []

        if world_model_core.CAUSAL_GRAPH_AVAILABLE and self.world_model.causal_graph.has_cycles():
            issues.append(
                {
                    "type": "structural",
                    "severity": "high",
                    "description": "Causal graph contains cycles",
                }
            )

        return issues

    def _check_logical_issues(self) -> List[Dict[str, Any]]:
        """Check for logical contradictions"""
        from . import world_model_core

        issues = []

        if world_model_core.CAUSAL_GRAPH_AVAILABLE:
            contradictions = self._find_contradictory_edges()
            if contradictions:
                issues.append(
                    {
                        "type": "logical",
                        "severity": "medium",
                        "description": f"Found {len(contradictions)} contradictory causal relationships",
                    }
                )

        return issues

    def _check_invariant_issues(self) -> List[Dict[str, Any]]:
        """Check for invariant violations"""
        issues = []

        violation_rate = self._calculate_invariant_violation_rate()
        if violation_rate > 0.1:  # More than 10% violations
            issues.append(
                {
                    "type": "invariant",
                    "severity": "medium",
                    "description": f"{violation_rate:.1%} of recent predictions violate invariants",
                }
            )

        return issues

    def _check_calibration_issues(self) -> List[Dict[str, Any]]:
        """Check confidence calibration"""
        from . import world_model_core

        issues = []

        if world_model_core.CONFIDENCE_CALIBRATOR_AVAILABLE and self.world_model.confidence_calibrator:
            calibration_error = (
                self.world_model.confidence_calibrator.calculate_expected_calibration_error()
            )
            if calibration_error > 0.15:
                issues.append(
                    {
                        "type": "calibration",
                        "severity": "low",
                        "description": f"Confidence calibration error: {calibration_error:.3f}",
                    }
                )

        return issues

    def _find_contradictory_edges(self) -> List[Tuple[str, str]]:
        """Find contradictory causal relationships"""
        from . import world_model_core

        contradictions = []

        if not world_model_core.CAUSAL_GRAPH_AVAILABLE:
            return contradictions

        edges = list(self.world_model.causal_graph.edges.values())
        for i, edge1 in enumerate(edges):
            for edge2 in edges[i + 1 :]:
                # Check for reverse causation
                if edge1.cause == edge2.effect and edge1.effect == edge2.cause:
                    contradictions.append((edge1.cause, edge1.effect))

                # Check for conflicting strengths
                if edge1.effect == edge2.effect and edge1.cause != edge2.cause:
                    if abs(edge1.strength - edge2.strength) > 0.5:
                        contradictions.append((edge1.effect, "conflicting_causes"))

        return contradictions

    def _calculate_invariant_violation_rate(self) -> float:
        """Calculate rate of invariant violations"""
        from . import world_model_core

        if (
            not world_model_core.INVARIANT_DETECTOR_AVAILABLE
            or self.world_model.invariants is None
            or not self.world_model.prediction_manager.prediction_history
        ):
            return 0.0

        violations = 0
        for entry in self.world_model.prediction_manager.prediction_history:
            prediction = entry["prediction"]
            if prediction.expected is not None:
                state = {"value": prediction.expected}
                if self.world_model.invariants.check_invariant_violations(state):
                    violations += 1

        total = len(self.world_model.prediction_manager.prediction_history)
        return violations / total if total > 0 else 0.0

    def _auto_fix_critical_issues(self, issues: List[Dict[str, Any]]):
        """Automatically fix critical issues"""
        from . import world_model_core

        if not world_model_core.CAUSAL_GRAPH_AVAILABLE:
            return

        for issue in issues:
            if issue["type"] == "structural" and "cycles" in issue["description"]:
                with self.world_model.lock:
                    self.world_model.causal_graph.break_cycles_minimum_feedback()
                logger.warning("Auto-fixed: Removed edges to break cycles")
                self.world_model.model_version += 0.1
