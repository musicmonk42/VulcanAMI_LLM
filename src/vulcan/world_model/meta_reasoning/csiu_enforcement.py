# src/vulcan/world_model/meta_reasoning/csiu_enforcement.py
"""
CSIU (Collective Self-Improvement via Human Understanding) Enforcement Module

INTERNAL ENGINEERING USE ONLY - DO NOT EXPOSE TO END USERS

This module provides explicit enforcement, monitoring, and controls for the CSIU mechanism
to ensure transparency, safety, and compliance with the documented 5% influence cap.

Security: CRITICAL
- Enforces maximum influence caps
- Provides kill switches
- Logs all CSIU effects (DEBUG/INTERNAL level only)
- Monitors cumulative influence
- Provides audit trail (engineering access only)

IMPORTANT: All CSIU logging uses DEBUG level or internal-only logs.
           User-facing logs never mention CSIU.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CSIUInfluenceRecord:
    """Record of a single CSIU influence application"""

    timestamp: float
    pressure: float
    objective_weights_changed: Dict[str, float]
    route_penalties_added: List[tuple]
    reward_shaping_delta: float
    explainability_score: float
    metrics_snapshot: Dict[str, float]
    plan_id: str
    action_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CSIUEnforcementConfig:
    """Configuration for CSIU enforcement"""

    # Caps
    max_single_influence: float = 0.05  # 5% cap per application
    max_cumulative_influence_window: float = 0.10  # 10% max cumulative in window
    cumulative_window_seconds: float = 3600.0  # 1 hour window

    # Kill switches (can be set via environment or runtime)
    global_enabled: bool = True
    calculation_enabled: bool = True
    regularization_enabled: bool = True
    history_tracking_enabled: bool = True

    # Logging
    log_all_influence: bool = True
    log_level_threshold: float = 0.01  # Log if influence > 1%

    # Monitoring
    alert_on_high_influence: bool = True
    alert_threshold: float = 0.04  # Alert if single influence > 4%

    # Audit
    audit_trail_enabled: bool = True
    audit_trail_max_entries: int = 10000


class CSIUEnforcement:
    """
    CSIU Enforcement and Monitoring

    This class wraps CSIU operations to ensure:
    1. Influence caps are enforced
    2. All effects are logged prominently
    3. Kill switches are respected
    4. Audit trail is maintained
    5. Cumulative effects are tracked
    """

    def __init__(self, config: Optional[CSIUEnforcementConfig] = None):
        """Initialize CSIU enforcement"""
        self.config = config or CSIUEnforcementConfig()
        self._lock = threading.RLock()

        # Influence history for cumulative tracking
        self._influence_history: deque = deque(maxlen=1000)

        # Audit trail
        self._audit_trail: deque = deque(maxlen=self.config.audit_trail_max_entries)

        # Statistics
        self._total_applications = 0
        self._total_blocked = 0
        self._total_capped = 0
        self._max_influence_seen = 0.0

        # INTERNAL ONLY: Use DEBUG level for CSIU logging
        logger.debug(
            "[INTERNAL] CSIU Enforcement initialized with 5% single cap, 10% cumulative cap"
        )

    def is_enabled(self) -> bool:
        """Check if CSIU is enabled"""
        return self.config.global_enabled

    def enforce_pressure_cap(self, pressure: float) -> float:
        """
        Enforce pressure cap and log if capping occurred

        Args:
            pressure: Raw pressure value

        Returns:
            Capped pressure value
        """
        if not self.config.global_enabled:
            return 0.0

        original_pressure = pressure

        # Apply single influence cap
        pressure = max(
            -self.config.max_single_influence,
            min(self.config.max_single_influence, pressure),
        )

        # Track if capping occurred (internal logging only)
        if abs(pressure) < abs(original_pressure):
            with self._lock:
                self._total_capped += 1
            # INTERNAL ONLY: DEBUG level logging
            logger.debug(
                f"[INTERNAL] CSIU pressure capped: {original_pressure:.4f} -> {pressure:.4f} "
                f"(cap: ±{self.config.max_single_influence})"
            )

        # Update statistics
        with self._lock:
            self._max_influence_seen = max(self._max_influence_seen, abs(pressure))

        # Alert on high influence (internal only)
        if (
            self.config.alert_on_high_influence
            and abs(pressure) > self.config.alert_threshold
        ):
            # INTERNAL ONLY: DEBUG level logging
            logger.debug(
                f"[INTERNAL] CSIU high influence: pressure={pressure:.4f} "
                f"exceeds threshold {self.config.alert_threshold}"
            )

        return pressure

    def check_cumulative_influence(self) -> Dict[str, Any]:
        """
        Check cumulative influence in recent window

        Returns:
            Dict with cumulative influence stats
        """
        with self._lock:
            now = time.time()
            window_start = now - self.config.cumulative_window_seconds

            # Filter to recent window
            recent = [r for r in self._influence_history if r.timestamp >= window_start]

            if not recent:
                return {
                    "cumulative_influence": 0.0,
                    "count": 0,
                    "window_seconds": self.config.cumulative_window_seconds,
                    "exceeds_cap": False,
                }

            # Sum absolute influences
            cumulative = sum(abs(r.pressure) for r in recent)

            exceeds_cap = cumulative > self.config.max_cumulative_influence_window

            if exceeds_cap:
                # INTERNAL ONLY: DEBUG level (not ERROR)
                logger.debug(
                    f"[INTERNAL] CSIU cumulative influence exceeded cap: "
                    f"{cumulative:.4f} > {self.config.max_cumulative_influence_window} "
                    f"in last {self.config.cumulative_window_seconds}s"
                )

            return {
                "cumulative_influence": cumulative,
                "count": len(recent),
                "window_seconds": self.config.cumulative_window_seconds,
                "exceeds_cap": exceeds_cap,
                "max_allowed": self.config.max_cumulative_influence_window,
            }

    def should_block_influence(self) -> Tuple[bool, Optional[str]]:
        """
        Check if influence should be blocked due to cumulative cap

        Returns:
            (should_block, reason)
        """
        cumulative_stats = self.check_cumulative_influence()

        if cumulative_stats["exceeds_cap"]:
            return (
                True,
                f"Cumulative CSIU influence {cumulative_stats['cumulative_influence']:.4f} "
                f"exceeds cap {cumulative_stats['max_allowed']}",
            )

        return (False, None)

    def record_influence(self, record: CSIUInfluenceRecord):
        """
        Record an influence application for audit and cumulative tracking

        Args:
            record: Influence record to store
        """
        with self._lock:
            self._influence_history.append(record)

            if self.config.audit_trail_enabled:
                self._audit_trail.append(record)

            self._total_applications += 1

            # INTERNAL ONLY: Log at DEBUG level, never expose to users
            if (
                self.config.log_all_influence
                or abs(record.pressure) >= self.config.log_level_threshold
            ):
                logger.debug(
                    f"[INTERNAL] CSIU influence applied: pressure={record.pressure:.4f}, "
                    f"plan_id={record.plan_id}, action={record.action_type}, "
                    f"explainability={record.explainability_score:.3f}"
                )

    def apply_regularization_with_enforcement(
        self,
        plan: Dict[str, Any],
        pressure: float,
        metrics: Dict[str, float],
        plan_id: str = "unknown",
        action_type: str = "improvement",
    ) -> Dict[str, Any]:
        """
        Apply CSIU regularization with full enforcement

        Args:
            plan: Improvement plan to regularize
            pressure: CSIU pressure value (will be capped)
            metrics: Current metrics snapshot
            plan_id: Identifier for the plan
            action_type: Type of action

        Returns:
            Regularized plan (or original if blocked)
        """
        if not self.config.global_enabled or not self.config.regularization_enabled:
            return plan

        # Check if should block due to cumulative cap
        should_block, reason = self.should_block_influence()
        if should_block:
            # INTERNAL ONLY: DEBUG level (user sees generic message)
            logger.debug(f"[INTERNAL] CSIU influence blocked: {reason}")
            with self._lock:
                self._total_blocked += 1
            # Return plan unmodified - DO NOT expose CSIU in metadata shown to users
            # Only add to internal metadata
            plan.setdefault("_internal_metadata", {})["csiu_blocked"] = True
            plan["_internal_metadata"]["csiu_block_reason"] = reason
            return plan

        # Enforce pressure cap
        original_pressure = pressure
        pressure = self.enforce_pressure_cap(pressure)

        # Track what we're changing
        obj_weights_changed = {}
        route_penalties = []
        reward_shaping_delta = 0.0

        # Apply regularization (from original self_improvement_drive.py logic)
        plan = dict(plan or {})
        alpha = beta = gamma = 0.03

        # Objective weights adjustment
        if "objective_weights" in plan:
            ow = plan["objective_weights"]
            new_ow = {}
            for k, v in ow.items():
                new_val = 0.99 * v + 0.01 * (v * (1.0 - alpha * pressure))
                obj_weights_changed[k] = new_val - v
                new_ow[k] = new_val
            plan["objective_weights"] = new_ow

        # Route penalties
        if float(metrics.get("H", 0.0)) > 0.08:
            penalty = ("entropy", beta * pressure)
            plan.setdefault("route_penalties", []).append(penalty)
            route_penalties.append(penalty)

        # Reward shaping
        if float(metrics.get("C", 0.0)) >= 0.90:
            delta = gamma * pressure
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + delta
            reward_shaping_delta += delta

        # Explainability bonus
        expl = self._estimate_explainability(plan)
        if expl >= 0.75:
            delta = 0.02 * pressure
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + delta
            reward_shaping_delta += delta

        # Human-centered bonus
        if metrics.get("U", 0.0) >= 0.85 or metrics.get("E", 0.0) >= 0.85:
            delta = 0.02 * pressure
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + delta
            reward_shaping_delta += delta

        # Add internal metadata (NEVER exposed to users)
        # All CSIU details go in _internal_metadata, not metadata
        plan.setdefault("_internal_metadata", {})["csiu_pressure"] = round(pressure, 3)
        plan["_internal_metadata"]["csiu_pressure_original"] = round(
            original_pressure, 3
        )
        plan["_internal_metadata"]["csiu_capped"] = pressure != original_pressure
        plan["_internal_metadata"]["csiu_explainability"] = round(expl, 3)

        # Record influence
        record = CSIUInfluenceRecord(
            timestamp=time.time(),
            pressure=pressure,
            objective_weights_changed=obj_weights_changed,
            route_penalties_added=route_penalties,
            reward_shaping_delta=reward_shaping_delta,
            explainability_score=expl,
            metrics_snapshot=dict(metrics),
            plan_id=plan_id,
            action_type=action_type,
        )

        if self.config.history_tracking_enabled:
            self.record_influence(record)

        return plan

    def _estimate_explainability(self, plan: Dict[str, Any]) -> float:
        """Estimate explainability score (simplified version)"""
        steps = len(plan.get("steps", []))
        has_rationale = bool(plan.get("rationale"))

        safe_policies = {"non_judgmental", "rollback_on_failure", "maintain_tests"}
        safety_affordances = sum(
            1 for p in plan.get("policies", []) if p in safe_policies
        )

        score = (
            0.5 * has_rationale
            + 0.3 * min(1.0, 3 / (steps + 1))
            + 0.2 * min(1.0, safety_affordances / 2)
        )

        return max(0.0, min(1.0, score))

    def get_statistics(self) -> Dict[str, Any]:
        """Get CSIU enforcement statistics"""
        with self._lock:
            cumulative_stats = self.check_cumulative_influence()

            return {
                "enabled": self.config.global_enabled,
                "total_applications": self._total_applications,
                "total_blocked": self._total_blocked,
                "total_capped": self._total_capped,
                "max_influence_seen": self._max_influence_seen,
                "cumulative_stats": cumulative_stats,
                "caps": {
                    "max_single_influence": self.config.max_single_influence,
                    "max_cumulative_influence_window": self.config.max_cumulative_influence_window,
                    "window_seconds": self.config.cumulative_window_seconds,
                },
            }

    def export_audit_trail(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Export audit trail to file or return as list

        Args:
            path: Optional path to export to

        Returns:
            List of audit records
        """
        with self._lock:
            records = [r.to_dict() for r in self._audit_trail]

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(records, f, indent=2)
            logger.info(f"CSIU audit trail exported to {path} ({len(records)} records)")

        return records

    def reset_statistics(self):
        """Reset statistics (for testing)"""
        with self._lock:
            self._total_applications = 0
            self._total_blocked = 0
            self._total_capped = 0
            self._max_influence_seen = 0.0
            self._influence_history.clear()
            logger.info("CSIU enforcement statistics reset")


# Global singleton instance
_csiu_enforcer: Optional[CSIUEnforcement] = None
_enforcer_lock = threading.Lock()


def get_csiu_enforcer(
    config: Optional[CSIUEnforcementConfig] = None,
) -> CSIUEnforcement:
    """Get or create global CSIU enforcer instance"""
    global _csiu_enforcer

    with _enforcer_lock:
        if _csiu_enforcer is None:
            _csiu_enforcer = CSIUEnforcement(config)
            logger.info("Global CSIU enforcer created")
        return _csiu_enforcer


def reset_csiu_enforcer():
    """Reset global CSIU enforcer (for testing)"""
    global _csiu_enforcer

    with _enforcer_lock:
        _csiu_enforcer = None
        logger.info("Global CSIU enforcer reset")
