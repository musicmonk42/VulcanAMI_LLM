# ============================================================
# VULCAN-AGI Experiment Trigger - Meta-Learning Experiment Generation
# ============================================================
# Enterprise-grade experiment triggering for self-improvement:
# - Telemetry threshold monitoring
# - Quality degradation detection
# - Novelty pattern detection
# - Error pattern analysis
#
# PRODUCTION-READY: Thread-safe, configurable thresholds, comprehensive logging
# META-LEARNING: Integrates with llm_meta_state.json for experiment generation
# ============================================================

"""
VULCAN Experiment Trigger

Decides when to run meta-learning experiments based on:
    - Telemetry thresholds (e.g., every 100 user interactions, every 50 AI interactions)
    - Pattern degradation detection
    - New problem types encountered
    - Error pattern clustering

When triggered, calls meta-learning system to generate and run experiments.

Features:
    - Multiple trigger conditions with configurable thresholds
    - Cooldown management to prevent experiment storms
    - Comprehensive statistics tracking
    - Callback system for experiment execution

Trigger Types:
    - telemetry_threshold: Periodic triggers based on interaction count
    - quality_degradation: Response quality drops below threshold
    - novelty_detection: High ratio of new query types
    - error_pattern: Clustering of errors in time window
    - ai_interaction_threshold: AI-specific periodic triggers
    - tournament_analysis: After arena tournaments

Thread Safety:
    All public methods are thread-safe.

Usage:
    from vulcan.routing import should_run_experiment, get_experiment_trigger
    
    # Check if experiment should run
    if should_run_experiment(telemetry_count=300, recent_patterns={}):
        run_experiments()
    
    # Get trigger for detailed operations
    trigger = get_experiment_trigger()
    proposal = trigger.check_should_experiment()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Default configuration
DEFAULT_META_STATE_PATH = Path("data/llm_meta_state.json")

# Interaction thresholds for experiments
DEFAULT_USER_INTERACTION_THRESHOLD = 100
DEFAULT_AI_INTERACTION_THRESHOLD = 50
DEFAULT_QUALITY_THRESHOLD = 0.7
DEFAULT_NOVELTY_THRESHOLD = 0.3
DEFAULT_ERROR_CLUSTER_THRESHOLD = 5
DEFAULT_ERROR_WINDOW_SECONDS = 300

# Cooldown defaults
DEFAULT_COOLDOWN_SECONDS = 300.0  # 5 minutes
TOURNAMENT_COOLDOWN_SECONDS = 600.0  # 10 minutes
QUALITY_COOLDOWN_SECONDS = 180.0  # 3 minutes

# Pattern tracking limits
MAX_RECENT_ITEMS = 200


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class ExperimentCondition:
    """
    Condition that can trigger an experiment.
    
    Attributes:
        name: Condition name
        check_fn: Function to check if condition is met
        experiment_type: Type of experiment to trigger
        priority: Priority (higher = more important)
        cooldown_seconds: Minimum time between triggers
        last_triggered: Timestamp of last trigger
    """
    name: str
    check_fn: Callable[["ExperimentTrigger"], bool]
    experiment_type: str
    priority: int = 1
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    last_triggered: float = 0.0


@dataclass
class ExperimentProposal:
    """
    Proposal for a meta-learning experiment.
    
    Attributes:
        experiment_id: Unique experiment identifier
        experiment_type: Type of experiment
        trigger_reason: Condition that triggered the experiment
        priority: Experiment priority
        parameters: Experiment parameters
        telemetry_snapshot: Snapshot of current telemetry
        timestamp: Proposal creation timestamp
    """
    experiment_id: str
    experiment_type: str
    trigger_reason: str
    priority: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    telemetry_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "trigger_reason": self.trigger_reason,
            "priority": self.priority,
            "parameters": self.parameters,
            "telemetry_snapshot": self.telemetry_snapshot,
            "timestamp": self.timestamp,
        }


# ============================================================
# EXPERIMENT TRIGGER CLASS
# ============================================================


class ExperimentTrigger:
    """
    Decides when to run meta-learning experiments.
    
    Monitors telemetry and patterns to determine when experiments
    would be beneficial for system improvement.
    
    Usage:
        trigger = ExperimentTrigger()
        
        # Record interactions
        trigger.record_interaction(
            query_type="reasoning",
            quality_score=0.8
        )
        
        # Check for experiments
        proposal = trigger.check_should_experiment()
        if proposal:
            run_experiment(proposal)
    """
    
    def __init__(
        self,
        meta_state_path: Optional[Path] = None,
        user_threshold: int = DEFAULT_USER_INTERACTION_THRESHOLD,
        ai_threshold: int = DEFAULT_AI_INTERACTION_THRESHOLD,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
        error_cluster_threshold: int = DEFAULT_ERROR_CLUSTER_THRESHOLD,
        error_window_seconds: float = DEFAULT_ERROR_WINDOW_SECONDS
    ):
        """
        Initialize the experiment trigger.
        
        Args:
            meta_state_path: Path to llm_meta_state.json
            user_threshold: Number of user interactions before triggering
            ai_threshold: Number of AI interactions before triggering
            quality_threshold: Quality score threshold for degradation detection
            novelty_threshold: Novelty ratio threshold for new problem types
            error_cluster_threshold: Number of errors to trigger investigation
            error_window_seconds: Time window for error clustering
        """
        self._meta_state_path = meta_state_path or DEFAULT_META_STATE_PATH
        self._user_threshold = user_threshold
        self._ai_threshold = ai_threshold
        self._quality_threshold = quality_threshold
        self._novelty_threshold = novelty_threshold
        self._error_cluster_threshold = error_cluster_threshold
        self._error_window_seconds = error_window_seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Counters for threshold tracking
        self._last_user_check_count = 0
        self._last_ai_check_count = 0
        self._last_check_timestamp = 0.0
        
        # Pattern tracking
        self._recent_query_types: List[str] = []
        self._recent_quality_scores: List[float] = []
        self._recent_errors: List[Dict[str, Any]] = []
        self._recent_ai_interactions: List[Dict[str, Any]] = []
        
        # Statistics
        self._stats = {
            "checks_performed": 0,
            "experiments_triggered": 0,
            "telemetry_triggers": 0,
            "ai_triggers": 0,
            "quality_triggers": 0,
            "novelty_triggers": 0,
            "error_triggers": 0,
            "tournament_triggers": 0,
        }
        
        # Set up experiment conditions
        self._conditions = self._setup_conditions()
        
        # Callback for triggering experiments
        self._experiment_callback: Optional[Callable[[ExperimentProposal], None]] = None
        
        logger.debug(
            f"ExperimentTrigger initialized with thresholds: "
            f"user={user_threshold}, ai={ai_threshold}, quality={quality_threshold}"
        )
    
    def _setup_conditions(self) -> List[ExperimentCondition]:
        """Set up experiment trigger conditions."""
        return [
            ExperimentCondition(
                name="user_telemetry_threshold",
                check_fn=self._check_user_threshold,
                experiment_type="periodic_optimization",
                priority=1,
                cooldown_seconds=600.0,
            ),
            ExperimentCondition(
                name="ai_telemetry_threshold",
                check_fn=self._check_ai_threshold,
                experiment_type="ai_performance_analysis",
                priority=2,
                cooldown_seconds=300.0,
            ),
            ExperimentCondition(
                name="quality_degradation",
                check_fn=self._check_quality_degradation,
                experiment_type="quality_recovery",
                priority=4,
                cooldown_seconds=QUALITY_COOLDOWN_SECONDS,
            ),
            ExperimentCondition(
                name="novelty_detection",
                check_fn=self._check_novelty,
                experiment_type="novelty_adaptation",
                priority=2,
                cooldown_seconds=600.0,
            ),
            ExperimentCondition(
                name="error_pattern",
                check_fn=self._check_error_pattern,
                experiment_type="error_mitigation",
                priority=5,
                cooldown_seconds=180.0,
            ),
            ExperimentCondition(
                name="tournament_analysis",
                check_fn=self._check_tournament_completed,
                experiment_type="tournament_learning",
                priority=3,
                cooldown_seconds=TOURNAMENT_COOLDOWN_SECONDS,
            ),
        ]
    
    def set_experiment_callback(self, callback: Callable[[ExperimentProposal], None]) -> None:
        """
        Set callback function for triggering experiments.
        
        Args:
            callback: Function to call with ExperimentProposal when triggered
        """
        self._experiment_callback = callback
    
    def record_interaction(
        self,
        query_type: str,
        source: str = "user",
        quality_score: Optional[float] = None,
        error_occurred: bool = False,
        error_details: Optional[Dict[str, Any]] = None,
        is_tournament: bool = False
    ) -> None:
        """
        Record an interaction for pattern tracking.
        
        Args:
            query_type: Type of the query
            source: "user" or "agent"/"arena"
            quality_score: Quality score of the response (0.0 to 1.0)
            error_occurred: Whether an error occurred
            error_details: Details about the error if any
            is_tournament: Whether this was a tournament interaction
        """
        with self._lock:
            # Track query types
            self._recent_query_types.append(query_type)
            if len(self._recent_query_types) > MAX_RECENT_ITEMS:
                self._recent_query_types.pop(0)
            
            # Track quality scores
            if quality_score is not None:
                self._recent_quality_scores.append(quality_score)
                if len(self._recent_quality_scores) > MAX_RECENT_ITEMS:
                    self._recent_quality_scores.pop(0)
            
            # Track errors
            if error_occurred:
                self._recent_errors.append({
                    "timestamp": time.time(),
                    "query_type": query_type,
                    "source": source,
                    "details": error_details or {}
                })
                if len(self._recent_errors) > MAX_RECENT_ITEMS:
                    self._recent_errors.pop(0)
            
            # Track AI interactions
            if source in ("agent", "arena") or is_tournament:
                self._recent_ai_interactions.append({
                    "timestamp": time.time(),
                    "query_type": query_type,
                    "is_tournament": is_tournament
                })
                if len(self._recent_ai_interactions) > MAX_RECENT_ITEMS:
                    self._recent_ai_interactions.pop(0)
    
    def check_should_experiment(self) -> Optional[ExperimentProposal]:
        """
        Check if an experiment should be triggered.
        
        Evaluates all conditions and returns an ExperimentProposal
        if any condition is met (respecting cooldowns).
        
        Returns:
            ExperimentProposal if experiment should run, None otherwise
        """
        with self._lock:
            self._stats["checks_performed"] += 1
        
        current_time = time.time()
        triggered_conditions = []
        
        # Check each condition
        for condition in self._conditions:
            # Check cooldown
            if current_time - condition.last_triggered < condition.cooldown_seconds:
                continue
            
            # Check condition
            try:
                if condition.check_fn(self):
                    triggered_conditions.append(condition)
            except Exception as e:
                logger.error(f"[ExperimentTrigger] Error checking {condition.name}: {e}")
        
        if not triggered_conditions:
            return None
        
        # Select highest priority condition
        triggered_conditions.sort(key=lambda c: c.priority, reverse=True)
        condition = triggered_conditions[0]
        
        # Create experiment proposal
        proposal = ExperimentProposal(
            experiment_id=f"exp_{int(current_time * 1000)}_{condition.name}",
            experiment_type=condition.experiment_type,
            trigger_reason=condition.name,
            priority=condition.priority,
            parameters=self._get_experiment_parameters(condition),
            telemetry_snapshot=self._get_telemetry_snapshot(),
        )
        
        # Update condition state
        condition.last_triggered = current_time
        
        # Update stats
        with self._lock:
            self._stats["experiments_triggered"] += 1
            stat_key = f"{condition.name.split('_')[0]}_triggers"
            if stat_key in self._stats:
                self._stats[stat_key] += 1
        
        logger.info(
            f"[ExperimentTrigger] Triggered experiment: {proposal.experiment_type} "
            f"(reason: {proposal.trigger_reason}, priority: {proposal.priority})"
        )
        
        # Call callback if set
        if self._experiment_callback:
            try:
                self._experiment_callback(proposal)
            except Exception as e:
                logger.error(f"[ExperimentTrigger] Callback error: {e}")
        
        # Write proposal to meta state
        self._write_experiment_to_meta_state(proposal)
        
        return proposal
    
    def _check_user_threshold(self, _self) -> bool:
        """Check if user telemetry count has reached threshold."""
        current_count = self._get_user_telemetry_count()
        
        intervals_last = self._last_user_check_count // self._user_threshold
        intervals_current = current_count // self._user_threshold
        
        if intervals_current > intervals_last:
            self._last_user_check_count = current_count
            return True
        
        return False
    
    def _check_ai_threshold(self, _self) -> bool:
        """Check if AI interaction count has reached threshold."""
        current_count = len(self._recent_ai_interactions)
        
        intervals_last = self._last_ai_check_count // self._ai_threshold
        intervals_current = current_count // self._ai_threshold
        
        if intervals_current > intervals_last:
            self._last_ai_check_count = current_count
            return True
        
        return False
    
    def _check_quality_degradation(self, _self) -> bool:
        """Check if response quality has degraded."""
        if len(self._recent_quality_scores) < 10:
            return False
        
        # Calculate average of recent scores
        recent = self._recent_quality_scores[-20:]
        avg_quality = sum(recent) / len(recent)
        
        return avg_quality < self._quality_threshold
    
    def _check_novelty(self, _self) -> bool:
        """Check if novel query patterns are appearing frequently."""
        if len(self._recent_query_types) < 20:
            return False
        
        # Calculate unique type ratio in recent queries
        recent = self._recent_query_types[-50:]
        unique_ratio = len(set(recent)) / len(recent)
        
        return unique_ratio > self._novelty_threshold
    
    def _check_error_pattern(self, _self) -> bool:
        """Check if error patterns indicate need for intervention."""
        if len(self._recent_errors) < 3:
            return False
        
        # Check for recent error clustering
        now = time.time()
        recent_errors = [
            e for e in self._recent_errors
            if now - e["timestamp"] < self._error_window_seconds
        ]
        
        return len(recent_errors) >= self._error_cluster_threshold
    
    def _check_tournament_completed(self, _self) -> bool:
        """Check if a tournament was recently completed."""
        if not self._recent_ai_interactions:
            return False
        
        # Check for recent tournament interactions
        now = time.time()
        recent_tournaments = [
            i for i in self._recent_ai_interactions
            if i.get("is_tournament") and now - i["timestamp"] < 60
        ]
        
        return len(recent_tournaments) >= 3
    
    def _get_user_telemetry_count(self) -> int:
        """Get current user telemetry count from meta state."""
        try:
            if self._meta_state_path.exists():
                with open(self._meta_state_path, 'r') as f:
                    state = json.load(f)
                    telemetry = state.get("objects", {}).get("telemetry", [])
                    # Count user interactions
                    return sum(1 for t in telemetry if t.get("source") == "user")
        except Exception as e:
            logger.debug(f"[ExperimentTrigger] Could not read telemetry count: {e}")
        return len(self._recent_query_types)
    
    def _get_experiment_parameters(self, condition: ExperimentCondition) -> Dict[str, Any]:
        """Get parameters for an experiment based on condition."""
        params = {
            "trigger_condition": condition.name,
            "priority": condition.priority,
            "timestamp": time.time(),
        }
        
        if "quality" in condition.name:
            if self._recent_quality_scores:
                recent = self._recent_quality_scores[-20:]
                params["current_avg_quality"] = sum(recent) / len(recent)
                params["target_quality"] = self._quality_threshold
                params["quality_trend"] = self._calculate_trend(recent)
        
        elif "error" in condition.name:
            now = time.time()
            recent_errors = [
                e for e in self._recent_errors
                if now - e["timestamp"] < self._error_window_seconds
            ]
            params["recent_error_count"] = len(recent_errors)
            params["error_types"] = list(set(
                e.get("query_type", "unknown") for e in recent_errors
            ))
        
        elif "novelty" in condition.name:
            if self._recent_query_types:
                recent = self._recent_query_types[-50:]
                params["novelty_ratio"] = len(set(recent)) / len(recent)
                params["unique_types"] = list(set(recent))
        
        elif "tournament" in condition.name:
            recent_tournaments = [
                i for i in self._recent_ai_interactions
                if i.get("is_tournament")
            ]
            params["tournament_count"] = len(recent_tournaments)
        
        return params
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 3:
            return "stable"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff = second_avg - first_avg
        if abs(diff) < 0.05:
            return "stable"
        return "improving" if diff > 0 else "declining"
    
    def _get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current telemetry state."""
        with self._lock:
            return {
                "recent_query_types": self._recent_query_types[-10:],
                "recent_quality_scores": self._recent_quality_scores[-10:],
                "recent_error_count": len([
                    e for e in self._recent_errors
                    if time.time() - e["timestamp"] < self._error_window_seconds
                ]),
                "recent_ai_interaction_count": len(self._recent_ai_interactions),
                "total_tracked_interactions": len(self._recent_query_types),
            }
    
    def _write_experiment_to_meta_state(self, proposal: ExperimentProposal) -> None:
        """Write experiment proposal to llm_meta_state.json."""
        try:
            self._meta_state_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self._meta_state_path.exists():
                with open(self._meta_state_path, 'r') as f:
                    state = json.load(f)
            else:
                state = {"objects": {"experiments": []}}
            
            if "objects" not in state:
                state["objects"] = {}
            if "experiments" not in state["objects"]:
                state["objects"]["experiments"] = []
            
            # Add experiment record
            state["objects"]["experiments"].append({
                "step": len(state["objects"]["experiments"]),
                **proposal.to_dict(),
                "status": "proposed"
            })
            
            # Write atomically
            temp_path = self._meta_state_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
            temp_path.replace(self._meta_state_path)
            
        except Exception as e:
            logger.error(f"[ExperimentTrigger] Failed to write experiment: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get experiment trigger statistics.
        
        Returns:
            Dictionary with check and trigger counts
        """
        with self._lock:
            stats = dict(self._stats)
            stats["recent_query_types_count"] = len(self._recent_query_types)
            stats["recent_quality_scores_count"] = len(self._recent_quality_scores)
            stats["recent_errors_count"] = len(self._recent_errors)
            stats["recent_ai_interactions_count"] = len(self._recent_ai_interactions)
            stats["thresholds"] = {
                "user_interaction": self._user_threshold,
                "ai_interaction": self._ai_threshold,
                "quality": self._quality_threshold,
                "novelty": self._novelty_threshold,
                "error_cluster": self._error_cluster_threshold,
            }
            return stats


# ============================================================
# SINGLETON PATTERN
# ============================================================

_global_trigger: Optional[ExperimentTrigger] = None
_trigger_lock = threading.Lock()


def get_experiment_trigger() -> ExperimentTrigger:
    """
    Get or create the global experiment trigger (thread-safe singleton).
    
    Returns:
        ExperimentTrigger instance
    """
    global _global_trigger
    
    if _global_trigger is None:
        with _trigger_lock:
            if _global_trigger is None:
                _global_trigger = ExperimentTrigger()
                logger.debug("Global ExperimentTrigger instance created")
    
    return _global_trigger


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def should_run_experiment(
    telemetry_count: int = 0,
    recent_patterns: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Determine if an experiment should be triggered.
    
    Convenience function that checks conditions.
    
    Args:
        telemetry_count: Current telemetry entry count (deprecated, uses internal tracking)
        recent_patterns: Recent pattern data (deprecated, uses internal tracking)
        
    Returns:
        True if experiment should be triggered
    """
    trigger = get_experiment_trigger()
    
    # Update trigger with pattern data if provided (for compatibility)
    if recent_patterns:
        for query_type in recent_patterns.get("query_types", []):
            trigger.record_interaction(
                query_type=query_type,
                quality_score=recent_patterns.get("avg_quality"),
                error_occurred=recent_patterns.get("error_occurred", False)
            )
    
    # Check if experiment should run
    proposal = trigger.check_should_experiment()
    return proposal is not None


def generate_experiments_from_interactions() -> List[ExperimentProposal]:
    """
    Analyze interactions and generate experiment proposals.
    
    Analyzes BOTH learning modes:
        From user interactions:
        - Which queries are hard?
        - What types of problems appear frequently?
        - Where does the model struggle?
        
        From AI interactions:
        - Which agent combinations work best?
        - Which tournament strategies win?
        - What collaboration patterns emerge?
    
    Returns:
        List of ExperimentProposal objects
    """
    trigger = get_experiment_trigger()
    proposals = []
    
    # Check all conditions and collect proposals
    for _ in range(5):  # Maximum 5 experiments per call
        proposal = trigger.check_should_experiment()
        if proposal:
            proposals.append(proposal)
        else:
            break
    
    logger.info(f"[ExperimentTrigger] Generated {len(proposals)} experiment proposals")
    return proposals
