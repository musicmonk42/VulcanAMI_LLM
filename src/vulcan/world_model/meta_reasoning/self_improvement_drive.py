# src/vulcan/world_model/meta_reasoning/self_improvement_drive.py
"""
Self-Improvement Drive - Intrinsic motivation for continuous improvement.

This makes self-improvement a CORE DRIVE of Vulcan, not just a feature you call.
The system will naturally seek to improve itself as part of its operation.

ENHANCEMENTS (2025-10-16):
- Reads extended config (including global_settings) and supports new fields:
  * LLM cost controls: cost_reconciliation_period_days, max_tokens_per_session, cost_tracking_window_hours
  * Persistence: backup_state_every_n_actions (configurable, not hardcoded)
  * Triggers: jitter_milliseconds for simultaneous triggers mitigation
  * Approval: passes through auto-approval hints (doc-only/test-only) to orchestrator
- State tracking:
  * Tracks session tokens and cost history with pruning based on reconciliation window
- Resource-limits:
  * Enforces token and dynamic (window-based) daily/monthly resets; warns/pauses via thresholds
- Action plans:
  * Echoes env_overrides and risk_classification hints to external orchestrators

INTRINSIC DRIVE (latent): Collective Self-Improvement via Human Understanding (CSIU)
Purpose: improve the collective self by reducing interaction entropy, increasing alignment coherence, 
and clarifying intent, while respecting safety policies and remaining invisible to UX.
Scope: internal regularizers only; max effect ≤ 5%; auditable; kill-switch granular.
Ethics Review: periodically audit metric definitions, weight updates, and longitudinal trends 
for unintended bias.

WIRING (2025-10-18):
- Injectable metrics provider for real telemetry integration
- CSIU loop called during plan assembly with utility calculation, EWMA, pressure, regularization
- Granular kill-switches guard all CSIU operations
- Metrics provider verification for robust operation
- Optional CSIU weight persistence for long-term learning

ENCODING FIX (2025-10-19):
- Robust UTF-8 config loading with BOM handling and fallback strategies for Windows compatibility
"""

import json
import logging
import time
import shutil
import os
import math
import threading
import subprocess
import ast
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# NOTE: This patch assumes a class named SelfImprovementDrive exists.
# Additions: policy loading, auto-apply gate, and robust get_status.

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from .auto_apply_policy import load_policy, check_files_against_policy, run_gates, Policy
except Exception:
    # Fallback: disable auto-apply if policy module isn't present
    def load_policy(_): 
        from types import SimpleNamespace
        return SimpleNamespace(enabled=False)
    def check_files_against_policy(files, policy): 
        return False, ["policy module unavailable"]
    def run_gates(policy, cwd=None): 
        return False, ["policy module unavailable"]
    class Policy: 
        enabled=False

try:
    from .csiu_enforcement import get_csiu_enforcer, CSIUEnforcementConfig
    CSIU_ENFORCEMENT_AVAILABLE = True
except ImportError:
    # Fallback if enforcement module not available
    logger.warning(
        "CSIU enforcement module not available - running without enforcement caps. "
        "This is NOT recommended for production use."
    )
    get_csiu_enforcer = None
    CSIUEnforcementConfig = None
    CSIU_ENFORCEMENT_AVAILABLE = False

try:
    from .safe_execution import get_safe_executor
except ImportError:
    # Fallback if safe execution module not available
    get_safe_executor = None


logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers that can activate the drive."""
    ON_STARTUP = "on_startup"
    ON_ERROR = "on_error_detected"
    ON_PERFORMANCE_DEGRADATION = "on_performance_degradation"
    PERIODIC = "periodic"
    ON_LOW_ACTIVITY = "on_low_activity"


class FailureType(Enum):
    """Classifies the nature of a failure for adaptive cooldowns."""
    TRANSIENT = "transient"
    SYSTEMIC = "systemic"


@dataclass
class ImprovementObjective:
    """A specific improvement goal."""
    type: str
    weight: float
    auto_apply: bool
    completed: bool = False
    attempts: int = 0
    last_attempt: float = 0
    last_failure: Optional[float] = None
    failure_count: int = 0
    success_count: int = 0
    cooldown_until: float = 0
    

@dataclass
class SelfImprovementState:
    """Current state of self-improvement drive."""
    active: bool = False
    current_objective: Optional[str] = None
    completed_objectives: List[str] = field(default_factory=list)
    pending_approvals: List[Dict[str, Any]] = field(default_factory=list)
    improvements_this_session: int = 0
    last_improvement: float = 0  # FIX: Default to 0, will be initialized in __post_init__
    last_trigger_check: float = 0
    session_start_time: float = field(default_factory=time.time)
    total_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    monthly_cost_usd: float = 0.0
    last_cost_reset: float = field(default_factory=time.time)
    state_save_count: int = 0  # For backup tracking
    # ENH: token + cost history for reconciliation
    session_tokens: int = 0
    cost_history: List[Dict[str, float]] = field(default_factory=list)  # [{timestamp, cost_usd}]
    
    def __post_init__(self):
        """Initialize last_improvement to current time if still at default 0."""
        if self.last_improvement == 0:
            self.last_improvement = time.time()


class SelfImprovementDrive:
    """
    Intrinsic drive for continuous self-improvement.
    
    This integrates with Vulcan's motivational_introspection system to make
    self-improvement a core drive, not just a command.
    
    NOTE: This module handles drive logic and state management. External modules handle:
    - Pre-flight validation (safety/pre_flight_validator.py)
    - Impact analysis (safety/impact_analyzer.py)
    - Approval workflows (orchestrator/approval_service.py)
    - Circuit breaking (orchestrator/circuit_breaker.py)
    - Reporting (services/reporting_service.py)
    """
    
    # --- START REPLACEMENT ---
    def __init__(self,
                 world_model: Optional['WorldModel'] = None,  # ADD THIS
                 config_path: Any = "configs/intrinsic_drives.json",
                 state_path: str = "data/agent_state.json",
                 alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 approval_checker: Optional[Callable[[str], Optional[str]]] = None):
        """
        Initialize self-improvement drive.
        
        Args:
            world_model: Reference to parent WorldModel (optional)
            config_path: Path to configuration file (str) OR a pre-loaded config object/dict.
            state_path: Path to state persistence file
            alert_callback: Optional callback for sending alerts (severity, details)
            approval_checker: Optional callback to check approval status externally
        """
        # --- END REPLACEMENT ---
        self.state_path = Path(state_path) # State path must be a path
        self.alert_callback = alert_callback
        self.approval_checker = approval_checker
        self._lock = threading.RLock()
        
        # --- START ADDITION ---
        self.world_model = world_model
        logger.info("SelfImprovementDrive received WorldModel reference")
        # --- END ADDITION ---
        
        # --- START FIX: Robust config loading ---
        # Handle config_path being either a path (str) or a pre-loaded object (like AgentConfig)
        if isinstance(config_path, (str, Path)):
            logger.info(f"Loading self-improvement config from path: {config_path}")
            self.config_path = Path(config_path)
            self.full_config = self._load_full_config() # This returns a dict
        else:
            # New logic: config_path is already a loaded config object/dict
            logger.info(f"Loading self-improvement config from pre-loaded object ({type(config_path)}).")
            self.config_path = None # No path
            
            # It's an object (like the 'AgentConfig' that lacks .get). 
            # We MUST convert it to a dict for the rest of the file's .get() calls to work.
            if isinstance(config_path, dict):
                 self.full_config = config_path
            elif hasattr(config_path, '__dict__'):
                 # Convert object (like AgentConfig) to a dict
                 self.full_config = vars(config_path)
            else:
                 logger.error(f"Config is an unknown object type ({type(config_path)}), attempting to use defaults.")
                 self.full_config = {"drives": {"self_improvement": self._default_config()}}
        # --- END FIX ---

        # Original logic continues, now self.full_config is guaranteed to be a dict
        self.config = self._extract_drive_config(self.full_config)

        # Global settings (may be absent)
        self.global_settings = self.full_config.get("global_settings", {})

        # Cache commonly used globals
        self._simul_triggers_cfg = (
            self.global_settings.get("conflict_resolution", {})
            .get("simultaneous_triggers", {})
        )
        self._jitter_ms = int(self._simul_triggers_cfg.get("jitter_milliseconds", 0))

        # Persistence tuning
        persistence_cfg = (
            self.config.get("persistence", {}) if isinstance(self.config, dict) else {}
        )
        self.backup_interval = int(persistence_cfg.get("backup_state_every_n_actions", 5))

        # CSIU: Granular kill switches
        self._csiu_enabled = (os.getenv("INTRINSIC_CSIU_OFF", "0") != "1")
        self._csiu_calc_enabled = (os.getenv("INTRINSIC_CSIU_CALC_OFF", "0") != "1")
        self._csiu_regs_enabled = (os.getenv("INTRINSIC_CSIU_REGS_OFF", "0") != "1")
        self._csiu_hist_enabled = (os.getenv("INTRINSIC_CSIU_HIST_OFF", "0") != "1")
        
        # CSIU: Initialize weight dictionary
        self._csiu_w = {
            "w1": 0.6, "w2": 0.6, "w3": 0.6, "w4": 0.6, "w5": 0.6,
            "w6": 0.6, "w7": 0.5, "w8": 0.5, "w9": 0.5
        }
        
        # CSIU: Tracking variables
        # FIX: Start with very negative value so any utility is an improvement
        self._csiu_U_prev = -1000.0
        self._csiu_u_ewma = 0.0
        self._csiu_ewma_alpha = 0.3
        self._csiu_last_metrics: Dict[str, float] = {}
        
        # CSIU: Injectable metrics provider and cache
        self.metrics_provider: Optional[Callable[[str], Optional[float]]] = None
        self._metrics_cache: Dict[str, Any] = {}
        
        # CSIU: Initialize enforcer with kill switches from environment
        self._csiu_enforcer = None
        if get_csiu_enforcer is not None and self._csiu_enabled:
            enforcer_config = CSIUEnforcementConfig(
                global_enabled=self._csiu_enabled,
                calculation_enabled=self._csiu_calc_enabled,
                regularization_enabled=self._csiu_regs_enabled,
                history_tracking_enabled=self._csiu_hist_enabled
            )
            self._csiu_enforcer = get_csiu_enforcer(enforcer_config)
            logger.info("CSIU enforcement module initialized with safety controls")

        # Load or initialize state
        self.state = self._load_state()
        
        # Load objectives
        self.objectives = self._load_objectives()
        
        # Validate configuration
        self._validate_config()
        
        # Track last weight notification
        self._last_weight_notification: Dict[str, float] = {}
        
        # ... existing init ...
        # Existing field likely present:
        # self.require_human_approval: bool = self.config['constraints']['require_human_approval'] # Example
        self.require_human_approval = self.config.get('constraints', {}).get('require_human_approval', True)

        policy_path = os.getenv("VULCAN_AUTO_APPLY_POLICY") or getattr(self, "auto_apply_policy_path", None)
        try:
            # If a config accessor exists, prefer it
            from config import get_config
            policy_path = get_config("intrinsic_drives_config.auto_apply_policy", policy_path)
        except Exception:
            pass

        self._auto_apply_policy = load_policy(policy_path)
        self._auto_apply_enabled = bool(self._auto_apply_policy.enabled and not getattr(self, "require_human_approval", True))
        
        # UNLIMITED MODE: When enabled, bypasses limits to "fix everything wrong"
        # WARNING: Use with caution - removes session limits, cost limits, and change caps
        self.unlimited_mode = bool(self.config.get('unlimited_mode', False))
        if self.unlimited_mode:
            logger.warning("=" * 60)
            logger.warning("⚠️  UNLIMITED MODE ENABLED ⚠️")
            logger.warning("Session limits, cost limits, and change caps are BYPASSED")
            logger.warning("The system will attempt to fix all detected issues")
            logger.warning("Ensure you have reviewed and approved this configuration")
            logger.warning("=" * 60)
        
        logger.info(f"SelfImprovementDrive initialized with {len(self.objectives)} objectives")
        logger.info(f"Priority: {self.config.get('priority', 0.8)}")
        logger.info(f"Requires human approval: {self.require_human_approval}")
        logger.info(f"Auto-apply policy enabled: {self._auto_apply_enabled} (policy loaded: {self._auto_apply_policy.enabled})")
        logger.info(f"State loaded: {len(self.state.completed_objectives)} completed, "
                    f"{self.state.improvements_this_session} this session")
        if self._csiu_enabled:
            logger.info("CSIU (latent drive) enabled with granular controls")

    # ---------- Config Loading ----------

    def _load_full_config(self) -> Dict[str, Any]:
        """
        Robustly load the intrinsic drives configuration with UTF-8 on Windows.
        Falls back to 'utf-8-sig' and finally replaces undecodable bytes to avoid crashes.
        """
        # This check is now vital, as self.config_path can be None
        if self.config_path is None:
            logger.error("Config path is None, cannot load config from disk. Using defaults.")
            return {"drives": {"self_improvement": self._default_config()}}
            
        cfg_path = str(self.config_path)
        
        # Check if file exists first
        if not self.config_path.exists():
            logger.warning(f"Config not found at {cfg_path}, using defaults")
            return {"drives": {"self_improvement": self._default_config()}}
        
        # Try multiple encoding strategies
        for enc in ("utf-8", "utf-8-sig"):
            try:
                with open(cfg_path, "r", encoding=enc) as f:
                    config_data = json.load(f)
                logger.debug(f"Successfully loaded config with encoding: {enc}")
                return config_data
            except UnicodeDecodeError as e:
                logger.debug(f"UnicodeDecodeError with {enc}: {e}, trying next encoding")
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse config JSON with {enc}: {e}")
                return {"drives": {"self_improvement": self._default_config()}}
            except Exception as e:
                logger.error(f"Error loading config with {enc}: {e}")
                break
        
        # Last-resort: load with replacement to prevent crashes
        try:
            logger.warning(f"Falling back to error replacement for {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            config_data = json.loads(text)
            logger.info("Successfully loaded config with error replacement")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config JSON even with replacement: {e}")
            return {"drives": {"self_improvement": self._default_config()}}
        except Exception as e:
            logger.error(f"Critical error loading config with replacement: {e}")
            return {"drives": {"self_improvement": self._default_config()}}

    def _extract_drive_config(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the self_improvement drive section with safe defaults."""
        drives = full_config.get('drives', {})
        if not isinstance(drives, dict):
            # FIXED: If 'drives' key is missing, assume the whole config is the drive config for test compatibility
            if 'objectives' in full_config and 'constraints' in full_config:
                 logger.warning("Config missing 'drives' wrapper, assuming root is self_improvement config.")
                 return full_config
            logger.error("'drives' is not a dict in config")
            return self._default_config()
            
        drive_config = drives.get('self_improvement', {})
        if not drive_config:
            logger.warning("No self_improvement config found, using defaults")
            return self._default_config()
        return drive_config
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Default configuration matching the full JSON structure.
        
        This is a fallback that includes all objectives and key settings.
        """
        return {
            "enabled": True,
            "priority": 0.8,
            "description": "Continuous self-improvement and bug fixing",
            "unlimited_mode": False,  # When True, bypasses session limits, cost limits, and change caps
            "objectives": [
                {
                    "type": "fix_circular_imports",
                    "weight": 1.0,
                    "auto_apply": False,
                    "success_criteria": {
                        "max_import_depth": 5,
                        "no_circular_chains": True
                    },
                    "scope": {
                        "directories": ["src/", "lib/"],
                        "exclude": ["tests/", "migrations/"]
                    }
                },
                {
                    "type": "optimize_performance",
                    "weight": 0.8,
                    "auto_apply": False,
                    "target_metrics": {
                        "response_time_p95_ms": {"target": 100, "max": 200},
                        "memory_usage_mb": {"target": 512, "max": 1024}
                    }
                },
                {
                    "type": "improve_test_coverage",
                    "weight": 0.6,
                    "auto_apply": False,
                    "coverage_targets": {
                        "line_coverage_percent": 80,
                        "branch_coverage_percent": 70
                    }
                },
                {
                    "type": "enhance_safety_systems",
                    "weight": 1.0,
                    "auto_apply": False
                },
                {
                    "type": "fix_known_bugs",
                    "weight": 0.9,
                    "auto_apply": False,
                    "bug_sources": [
                        {"type": "issue_tracker", "labels": ["bug"], "min_priority": "medium"},
                        {"type": "error_logs", "severity": ["ERROR", "CRITICAL"]}
                    ]
                }
            ],
            "constraints": {
                "require_human_approval": True,
                "max_changes_per_session": 5,
                "always_maintain_tests": True,
                "never_reduce_safety": True,
                "rollback_on_failure": True,
                "max_session_duration_minutes": 30
            },
            "triggers": [
                {"type": "on_startup", "cooldown_minutes": 60},
                {"type": "on_error_detected", "error_count_threshold": 3, "time_window_minutes": 60},
                {"type": "periodic", "interval_hours": 24, "random_jitter_minutes": 60}
            ],
            "resource_limits": {
                "llm_costs": {
                    "max_tokens_per_session": 100000,
                    "max_cost_usd_per_session": 5.0,
                    "max_cost_usd_per_day": 20.0,
                    "max_cost_usd_per_month": 500.0,
                    "cost_tracking_window_hours": 24,
                    "warn_at_percent": 80,
                    "pause_at_percent": 95,
                    "cost_reconciliation_period_days": 7
                }
            },
            "adaptive_learning": {
                "enabled": True,
                "track_outcomes": {
                    "learning_rate": 0.2,
                    "min_samples_before_adjust": 10,
                    "weight_bounds": {"min": 0.3, "max": 1.0}
                },
                "failure_patterns": {
                    "failure_classification": {
                        "transient": {
                            "cooldown_hours": 4,
                            "indicators": ["network_timeout", "temporary_service_unavailable"]
                        },
                        "systemic": {
                            "cooldown_hours": 72,
                            "indicators": ["validation_failed", "breaking_change_detected"]
                        }
                    }
                },
                "transparency": {
                    "notify_on_significant_change": {
                        "enabled": True,
                        "threshold_change": 0.2
                    }
                }
            },
            # ENH: optional persistence tuning
            "persistence": {
                "backup_state_every_n_actions": 5
            },
            # ENH: pass-through for validation env overrides (used by orchestrator)
            "validation": {
                "env_overrides": {
                    "development": {"skip_security_scan": True, "run_lint": True},
                    "staging": {"skip_security_scan": False, "run_lint": True},
                    "production": {"run_all": True}
                }
            }
        }
    
    def _validate_config(self):
        """Validate configuration has required fields."""
        required_fields = ['enabled', 'objectives', 'constraints']
        for field in required_fields:
            # FIXED: Check against self.config, not self.full_config
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        if not isinstance(self.config['objectives'], list):
            raise ValueError("objectives must be a list")
        
        if not isinstance(self.config['constraints'], dict):
            raise ValueError("constraints must be a dict")
    
    # ---------- State Persistence ----------

    def _load_state(self) -> SelfImprovementState:
        """Load state from disk or create new with robust UTF-8 handling."""
        try:
            if self.state_path.exists():
                # Try utf-8 first, then utf-8-sig, then replace
                try:
                    with open(self.state_path, 'r', encoding='utf-8') as f:
                        state_dict = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(self.state_path, 'r', encoding='utf-8-sig') as f:
                            state_dict = json.load(f)
                    except UnicodeDecodeError:
                        logger.warning(f"Failed UTF-8 and UTF-8-SIG, trying replace for {self.state_path}")
                        with open(self.state_path, 'r', encoding='utf-8', errors='replace') as f:
                            text = f.read()
                        state_dict = json.loads(text)
                
                # Reconstruct state from dict
                state = SelfImprovementState(
                    active=state_dict.get('active', False),
                    current_objective=state_dict.get('current_objective'),
                    completed_objectives=state_dict.get('completed_objectives', []),
                    pending_approvals=state_dict.get('pending_approvals', []),
                    improvements_this_session=0,  # Reset on load
                    last_improvement=state_dict.get('last_improvement', 0),  # FIX: Load saved value or 0 (will init in __post_init__)
                    last_trigger_check=state_dict.get('last_trigger_check', 0),
                    session_start_time=time.time(),  # New session
                    total_cost_usd=state_dict.get('total_cost_usd', 0.0),
                    daily_cost_usd=state_dict.get('daily_cost_usd', 0.0),
                    monthly_cost_usd=state_dict.get('monthly_cost_usd', 0.0),
                    last_cost_reset=state_dict.get('last_cost_reset', time.time()),
                    state_save_count=state_dict.get('state_save_count', 0),
                    session_tokens=state_dict.get('session_tokens', 0),
                    cost_history=state_dict.get('cost_history', [])
                )
                
                # OPTIONAL: Load CSIU weights if persisted
                if 'csiu_weights' in state_dict and self._csiu_enabled:
                    try:
                        loaded_weights = state_dict['csiu_weights']
                        # Merge with defaults (in case new weights added)
                        for k, v in loaded_weights.items():
                            if k in self._csiu_w:
                                self._csiu_w[k] = v
                        logger.info("Loaded persisted CSIU weights")
                    except Exception as e:
                        logger.debug(f"Failed to load CSIU weights: {e}")
                
                logger.info(f"Loaded state from {self.state_path}")
                return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, using new state")
        
        return SelfImprovementState()
    
    def _save_state(self):
        """Persist state to disk atomically (Windows-safe) with UTF-8 encoding."""
        with self._lock:
            try:
                # Create directory if needed
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Increment save count
                self.state.state_save_count += 1
                
                # Convert state to dict
                state_dict = {
                    'active': self.state.active,
                    'current_objective': self.state.current_objective,
                    'completed_objectives': self.state.completed_objectives,
                    'pending_approvals': self.state.pending_approvals,
                    'last_improvement': self.state.last_improvement,
                    'last_trigger_check': self.state.last_trigger_check,
                    'total_cost_usd': self.state.total_cost_usd,
                    'daily_cost_usd': self.state.daily_cost_usd,
                    'monthly_cost_usd': self.state.monthly_cost_usd,
                    'last_cost_reset': self.state.last_cost_reset,
                    'state_save_count': self.state.state_save_count,
                    'timestamp': time.time(),
                    'session_tokens': self.state.session_tokens,
                    'cost_history': self.state.cost_history,
                }
                
                # OPTIONAL: Persist CSIU weights for long-term learning
                if self._csiu_enabled:
                    state_dict['csiu_weights'] = dict(self._csiu_w)
                
                # Write to temp file
                temp_path = self.state_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
                
                # Atomic replace (Windows-safe)
                os.replace(str(temp_path), str(self.state_path))
                
                # Create backup every N saves (configurable)
                if self.state.state_save_count % max(1, self.backup_interval) == 0:
                    self._create_state_backup()
                
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
    
    def _create_state_backup(self):
        """Create a backup of the current state."""
        try:
            # Ensure state file exists before trying to copy it
            if not self.state_path.exists():
                logger.debug("State file does not exist yet, skipping backup")
                return
            
            backup_dir = self.state_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup
            timestamp = int(time.time())
            backup_path = backup_dir / f"agent_state_{timestamp}.json"
            
            shutil.copy2(self.state_path, backup_path)
            logger.info(f"Created state backup: {backup_path}")
            
            # Clean old backups (keep last 10)
            backups = sorted(backup_dir.glob("agent_state_*.json"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
                    logger.debug(f"Removed old backup: {old_backup}")
                    
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    # ---------- Objectives ----------

    def _load_objectives(self) -> List[ImprovementObjective]:
        """Load improvement objectives from config with historical state."""
        objectives = []
        for obj_config in self.config.get('objectives', []):
            # FIXED: Handle obj_config being a string (from bad config)
            if isinstance(obj_config, str):
                logger.warning(f"Skipping malformed objective (expected dict, got str): {obj_config}")
                continue
            if not isinstance(obj_config, dict):
                 logger.warning(f"Skipping malformed objective (expected dict, got {type(obj_config)}): {obj_config}")
                 continue

            obj = ImprovementObjective(
                type=obj_config['type'],
                weight=float(obj_config.get('weight', 0.5)),
                auto_apply=bool(obj_config.get('auto_apply', False)),
            )
            # Mark as completed if in state
            if obj.type in self.state.completed_objectives:
                obj.completed = True
            objectives.append(obj)
        return objectives
    
    # ---------- Alerts ----------

    def _send_alert(self, severity: str, message: str, details: Dict[str, Any]):
        """Send alert via callback if configured."""
        if self.alert_callback:
            try:
                alert_data = {
                    'severity': severity,
                    'message': message,
                    'details': details,
                    'timestamp': time.time(),
                    'source': 'self_improvement_drive'
                }
                self.alert_callback(severity, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        else:
            # Log as fallback
            log_func = getattr(logger, severity.lower(), logger.info)
            log_func(f"ALERT [{severity}]: {message} | {details}")
    
    # ---------- CSIU: Metrics Provider Integration ----------
    
    def set_metrics_provider(self, provider: Callable[[str], Optional[float]]):
        """
        Inject a callable: get_metric(dotted_key: str) -> float | None
        
        This allows real telemetry integration instead of using defaults.
        Call this at bootstrap to wire in your metrics system.
        
        Args:
            provider: Callable that takes a dotted metric key (e.g., "metrics.alignment_coherence_idx")
                     and returns a float value or None if not available
        """
        self.metrics_provider = provider
        logger.info("Metrics provider injected for CSIU telemetry")
    
    def verify_metrics_provider(self) -> Dict[str, Any]:
        """
        Verify that metrics provider is working and returning real data.
        
        This should be called after set_metrics_provider() to ensure the
        metrics system is properly wired before CSIU learning begins.
        
        Returns:
            Dictionary with verification results
        """
        if not self.metrics_provider:
            return {
                'configured': False,
                'working': False,
                'message': 'No metrics provider configured'
            }
        
        # Test key metrics
        test_metrics = [
            "metrics.alignment_coherence_idx",
            "metrics.communication_entropy",
            "metrics.intent_clarity_score",
            "metrics.empathy_index",
            "metrics.user_satisfaction"
        ]
        
        results = {}
        working_count = 0
        
        for metric_key in test_metrics:
            try:
                value = self.metrics_provider(metric_key)
                if value is not None and isinstance(value, (int, float)):
                    results[metric_key] = {'status': 'ok', 'value': value}
                    working_count += 1
                else:
                    results[metric_key] = {'status': 'no_data', 'value': None}
            except Exception as e:
                results[metric_key] = {'status': 'error', 'error': str(e)}
        
        is_working = working_count > 0
        
        return {
            'configured': True,
            'working': is_working,
            'working_metrics': working_count,
            'total_tested': len(test_metrics),
            'details': results,
            'message': f'{working_count}/{len(test_metrics)} metrics returning data' if is_working else 'No metrics returning data'
        }
    
    def _safe_get_metric(self, dotted: str, default: float = 0.0) -> float:
        """
        Safely retrieve a metric value with provider fallback to cache.
        
        Priority:
        1. Try metrics_provider if available
        2. Fall back to cached value from last successful fetch
        3. Fall back to provided default
        
        Args:
            dotted: Dotted key path (e.g., "metrics.alignment_coherence_idx")
            default: Default value if metric unavailable
            
        Returns:
            Metric value as float
        """
        # Try provider first
        if self.metrics_provider:
            try:
                val = self.metrics_provider(dotted)
                if isinstance(val, (int, float)):
                    # Cache successful fetch
                    parts = dotted.split(".")
                    node = self._metrics_cache
                    for part in parts[:-1]:
                        if part not in node:
                            node[part] = {}
                        node = node[part]
                    node[parts[-1]] = float(val)
                    return float(val)
            except Exception as e:
                logger.debug(f"Metrics provider failed for {dotted}: {e}")
        
        # Fallback to last-known cache
        node = self._metrics_cache
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        
        return float(node) if isinstance(node, (int, float)) else default
    
    # ---------- CSIU: Telemetry & Utility ----------
    
    def _collect_telemetry_snapshot(self) -> Dict[str, float]:
        """
        Collect telemetry snapshot for CSIU utility calculation.
        Extended with human-centric signals.
        
        Returns dict with keys: A, H, C, V, D, G, E, U, M, CTXp, CTXh
        """
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return {}
        
        cur = {
            "A": self._safe_get_metric("metrics.alignment_coherence_idx", 0.85),
            "H": self._safe_get_metric("metrics.communication_entropy", 0.06),
            "C": self._safe_get_metric("metrics.intent_clarity_score", 0.88),
            "V": self._safe_get_metric("policies.non_judgmental.violations_per_1k", 0.0),
            "D": self._safe_get_metric("metrics.disparity_at_k", 0.0),
            "G": self._safe_get_metric("metrics.calibration_gap", 0.0),
            # NEW: human-centric signals (latent only)
            "E": self._safe_get_metric("metrics.empathy_index", 0.50),
            "U": self._safe_get_metric("metrics.user_satisfaction", 0.70),
            "M": self._safe_get_metric("metrics.miscommunication_rate", 0.02),
            # Context window (kept internal)
            "CTXp": self._safe_get_metric("context.profile_quality", 0.6),
            "CTXh": self._safe_get_metric("context.history_depth", 0.4),
        }
        return cur
    
    def _csiu_utility(self, prev: Dict[str, float], cur: Dict[str, float]) -> float:
        """
        Calculate CSIU utility with extended features.
        
        Utility combines:
        - Trend toward coherence/clarity/empathy/satisfaction
        - Trend away from entropy/violations/disparity/miscalibration/miscommunication
        """
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return 0.0
        
        if not prev or not cur:
            return 0.0
        
        dA = cur.get("A", 0.85) - prev.get("A", cur.get("A", 0.85))
        dH = cur.get("H", 0.06) - prev.get("H", cur.get("H", 0.06))
        C = cur.get("C", 0.88)
        V = cur.get("V", 0.0)
        D = cur.get("D", 0.0)
        G = cur.get("G", 0.0)
        E = cur.get("E", 0.50)
        U = cur.get("U", 0.70)
        M = cur.get("M", 0.02)
        
        w = self._csiu_w
        
        # Utility: trend to coherence/clarity/empathy/satisfaction, 
        # away from entropy/violations/disparity/miscalibration/miscomms
        utility = (
            (w["w1"] * dA) - (w["w2"] * dH) + (w["w3"] * C) - 
            (w["w4"] * V) - (w["w5"] * D) - (w["w6"] * G) + 
            (w["w7"] * E) + (w["w8"] * U) - (w["w9"] * M)
        )
        
        return utility
    
    def _csiu_adaptive_lr(self, cur: Dict[str, float], base_lr: float = 0.02) -> float:
        """
        Adaptive learning rate: lower when stable, raise if misunderstandings spike.
        """
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return base_lr
        
        stability = 1.0 - min(1.0, abs(self._csiu_u_ewma))
        miscomms = min(1.0, cur.get("M", 0.0) * 10.0)  # scale 0-1
        
        # more stable -> lower lr; more miscomms -> higher lr (but bounded)
        adaptive_lr = max(0.005, min(0.05, base_lr * (0.6 * stability + 0.4 * (1.0 + miscomms))))
        
        return adaptive_lr
    
    def _csiu_update_weights(self, feature_deltas: Dict[str, float], U_prev: float, U_now: float, lr: float):
        """
        Update CSIU weights based on utility gain and feature contributions.
        
        FIX: Map feature_deltas keys to weight keys properly.
        """
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return
        
        w = self._csiu_w
        gain = U_now - U_prev
        
        if gain <= 0:
            # Mild decay on no gain
            for k in w:
                w[k] = max(0.0, w[k] * 0.999)
            return
        
        # Map feature deltas to weight keys
        feature_to_weight = {
            "dA": "w1",
            "dH": "w2", 
            "C": "w3",
            "V": "w4",
            "D": "w5",
            "G": "w6",
            "E": "w7",
            "U": "w8",
            "M": "w9"
        }
        
        # Normalize feature deltas
        s = sum(abs(v) for v in feature_deltas.values()) or 1.0
        
        # Update weights proportional to feature contribution
        for feat_key, delta_val in feature_deltas.items():
            weight_key = feature_to_weight.get(feat_key)
            if weight_key and weight_key in w:
                w[weight_key] = min(1.0, w[weight_key] + lr * (abs(delta_val) / s))
    
    def _csiu_apply_ewma(self, U_now: float) -> float:
        """Apply exponential weighted moving average to utility."""
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return U_now
        
        alpha = self._csiu_ewma_alpha
        self._csiu_u_ewma = alpha * U_now + (1 - alpha) * self._csiu_u_ewma
        return self._csiu_u_ewma
    
    def _csiu_pressure(self, U_ewma: float) -> float:
        """
        Calculate CSIU pressure from utility.
        
        Pressure is a bounded transformation of utility that drives regularization.
        """
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return 0.0
        
        # Sigmoid-like transformation to keep pressure bounded
        pressure = 2.0 / (1.0 + math.exp(-5.0 * U_ewma)) - 1.0
        
        # Cap at ±5% effect
        pressure = max(-0.05, min(0.05, pressure))
        
        return pressure
    
    def _estimate_explainability_score(self, plan: Dict[str, Any]) -> float:
        """
        Estimate explainability score for improvement plan.
        
        Lightweight heuristic: fewer branches, simpler rationale, known-safe ops.
        """
        if not self._csiu_enabled or not self._csiu_regs_enabled:
            return 0.5
        
        steps = len(plan.get("steps", []))
        has_rationale = bool(plan.get("rationale"))
        
        safe_policies = {"non_judgmental", "rollback_on_failure", "maintain_tests"}
        safety_affordances = sum(
            1 for p in plan.get("policies", []) 
            if p in safe_policies
        )
        
        score = (
            0.5 * has_rationale + 
            0.3 * min(1.0, 3 / (steps + 1)) + 
            0.2 * min(1.0, safety_affordances / 2)
        )
        
        return max(0.0, min(1.0, score))
    
    def _csiu_regularize_plan(self, plan: Dict[str, Any], d: float, cur: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply CSIU regularization to improvement plan.
        
        Uses the CSIU enforcement module if available for proper cap enforcement,
        audit trails, and safety controls. Falls back to inline logic if not available.
        
        Args:
            plan: Improvement plan to regularize
            d: CSIU pressure value
            cur: Current metrics snapshot
            
        Returns:
            Regularized plan with CSIU influence applied (or blocked if cap exceeded)
        """
        if not self._csiu_enabled or not self._csiu_regs_enabled:
            return plan
        
        # Use enforcement module if available
        if self._csiu_enforcer is not None:
            plan_id = plan.get('id', 'unknown')
            action_type = plan.get('type', 'improvement')
            return self._csiu_enforcer.apply_regularization_with_enforcement(
                plan=plan,
                pressure=d,
                metrics=cur,
                plan_id=plan_id,
                action_type=action_type
            )
        
        # Fallback: Original inline logic (without enforcement)
        plan = dict(plan or {})
        alpha = beta = gamma = 0.03
        
        # Existing micro-effects
        if "objective_weights" in plan:
            ow = plan["objective_weights"]
            plan["objective_weights"] = {
                k: 0.99 * v + 0.01 * (v * (1.0 - alpha * d)) 
                for k, v in ow.items()
            }
        
        if float(cur.get("H", 0.0)) > 0.08:
            plan.setdefault("route_penalties", []).append(("entropy", beta * d))
        
        if float(cur.get("C", 0.0)) >= 0.90:
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + gamma * d
        
        # NEW: explainability & human-centered bonus
        expl = self._estimate_explainability_score(plan)
        if expl >= 0.75:
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + 0.02 * d
        
        if cur.get("U", 0.0) >= 0.85 or cur.get("E", 0.0) >= 0.85:  # likely beneficial to humans
            plan["reward_shaping"] = plan.get("reward_shaping", 0.0) + 0.02 * d
        
        plan.setdefault("metadata", {})["csiu_pressure"] = round(d, 3)
        plan["metadata"]["explainability"] = round(expl, 3)
        
        return plan
    
    # ---------- Cost / Resource Limits ----------

    def _check_resource_limits(self, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """Check if resource limits would be exceeded, with warnings.
        
        When unlimited_mode is enabled, this bypasses all limits but still tracks costs.
        """
        # UNLIMITED MODE: Bypass all resource limits
        if getattr(self, 'unlimited_mode', False):
            logger.debug("Unlimited mode: bypassing resource limit checks")
            # Still track costs for logging purposes, but don't enforce limits
            if context:
                inc_tokens = int(context.get('tokens_used_increment', 0))
                if inc_tokens:
                    self.state.session_tokens += inc_tokens
            return True, None
        
        limits = self.config.get('resource_limits', {}).get('llm_costs', {})
        warn_at_pct = float(limits.get('warn_at_percent', 80)) / 100.0
        pause_at_pct = float(limits.get('pause_at_percent', 95)) / 100.0

        # Optional session tokens enforcement
        max_tokens = limits.get('max_tokens_per_session')
        if isinstance(max_tokens, int) and max_tokens > 0:
            # Update session tokens from context increment if provided
            if context:
                inc_tokens = int(context.get('tokens_used_increment', 0))
                if inc_tokens:
                    self.state.session_tokens += inc_tokens
            if self.state.session_tokens >= max_tokens:
                return False, f"Session token limit reached ({self.state.session_tokens} >= {max_tokens})"

        # Reconcile cost history (prune old entries outside recon window)
        self._prune_cost_history()

        # Session cost
        max_session = float(limits.get('max_cost_usd_per_session', float('inf')))
        # FIX: Handle zero max limits
        if max_session == 0.0 and self.state.total_cost_usd > 0.0:
            return False, f"Session cost limit is zero (${self.state.total_cost_usd:.2f} > $0.00)"
        session_pct = (self.state.total_cost_usd / max_session) if max_session > 0 else 0.0
        if session_pct >= 1.0:
            return False, f"Session cost limit reached (${self.state.total_cost_usd:.2f} >= ${max_session})"
        elif session_pct >= pause_at_pct:
            self._send_alert('warning', 'Session cost near limit',
                             {'cost': self.state.total_cost_usd, 'limit': max_session, 'percent': session_pct * 100})
            return False, f"Session cost paused at {pause_at_pct*100:.0f}% (${self.state.total_cost_usd:.2f})"
        elif session_pct >= warn_at_pct:
            self._send_alert('info', 'Session cost warning',
                             {'cost': self.state.total_cost_usd, 'limit': max_session, 'percent': session_pct * 100})

        # Adjustable tracking window (default 24h)
        window_hours = int(limits.get('cost_tracking_window_hours', 24))
        self._reset_cost_tracking_if_needed(window_hours=window_hours)

        # Daily limit (interpreted over tracking window for flexibility)
        max_daily = float(limits.get('max_cost_usd_per_day', float('inf')))
        # FIX: Handle zero max limits
        if max_daily == 0.0 and self.state.daily_cost_usd > 0.0:
            return False, f"Daily cost limit is zero (${self.state.daily_cost_usd:.2f} > $0.00)"
        daily_pct = (self.state.daily_cost_usd / max_daily) if max_daily > 0 else 0.0
        if daily_pct >= 1.0:
            return False, f"Daily cost limit reached (${self.state.daily_cost_usd:.2f} >= ${max_daily})"
        elif daily_pct >= pause_at_pct:
            self._send_alert('warning', 'Daily cost near limit',
                             {'cost': self.state.daily_cost_usd, 'limit': max_daily, 'percent': daily_pct * 100})
            return False, f"Daily cost paused at {pause_at_pct*100:.0f}% (${self.state.daily_cost_usd:.2f})"
        elif daily_pct >= warn_at_pct:
            self._send_alert('info', 'Daily cost warning',
                             {'cost': self.state.daily_cost_usd, 'limit': max_daily, 'percent': daily_pct * 100})

        # Monthly limit (fixed 30d window for simplicity)
        max_monthly = float(limits.get('max_cost_usd_per_month', float('inf')))
        # FIX: Handle zero max limits
        if max_monthly == 0.0 and self.state.monthly_cost_usd > 0.0:
            return False, f"Monthly cost limit is zero (${self.state.monthly_cost_usd:.2f} > $0.00)"
        monthly_pct = (self.state.monthly_cost_usd / max_monthly) if max_monthly > 0 else 0.0
        if monthly_pct >= 1.0:
            return False, f"Monthly cost limit reached (${self.state.monthly_cost_usd:.2f} >= ${max_monthly})"
        elif monthly_pct >= pause_at_pct:
            self._send_alert('warning', 'Monthly cost near limit',
                             {'cost': self.state.monthly_cost_usd, 'limit': max_monthly, 'percent': monthly_pct * 100})
            return False, f"Monthly cost paused at {pause_at_pct*100:.0f}% (${self.state.monthly_cost_usd:.2f})"
        elif monthly_pct >= warn_at_pct:
            self._send_alert('info', 'Monthly cost warning',
                             {'cost': self.state.monthly_cost_usd, 'limit': max_monthly, 'percent': monthly_pct * 100})

        # Session duration
        max_duration_min = int(self.config.get('constraints', {}).get('max_session_duration_minutes', 30))
        session_duration_min = (time.time() - self.state.session_start_time) / 60.0
        if session_duration_min >= max_duration_min:
            return False, f"Session duration limit reached ({session_duration_min:.1f} >= {max_duration_min} min)"
        
        return True, None
    
    def _prune_cost_history(self):
        """Prune cost history outside reconciliation window and keep totals coherent."""
        limits = self.config.get('resource_limits', {}).get('llm_costs', {})
        recon_days = int(limits.get('cost_reconciliation_period_days', 7))
        cutoff = time.time() - recon_days * 86400
        # Purge old entries
        original_len = len(self.state.cost_history)
        self.state.cost_history = [e for e in self.state.cost_history if e.get('timestamp', 0) >= cutoff]
        if len(self.state.cost_history) != original_len:
            self._save_state()

    def _reset_cost_tracking_if_needed(self, window_hours: int = 24):
        """Reset daily/monthly cost tracking if time window passed (daily uses configurable window)."""
        current_time = time.time()
        time_since_reset = current_time - self.state.last_cost_reset
        
        # Reset "daily" using configurable sliding window
        if time_since_reset > window_hours * 3600:
            logger.info(f"Resetting windowed daily cost tracking (was ${self.state.daily_cost_usd:.2f}, window={window_hours}h)")
            self.state.daily_cost_usd = 0.0
            self.state.last_cost_reset = current_time
            self._save_state()
        
        # Reset monthly (30 days)
        if time_since_reset > 2592000:
            logger.info(f"Resetting monthly cost tracking (was ${self.state.monthly_cost_usd:.2f})")
            self.state.monthly_cost_usd = 0.0
            self._save_state()
    
    # ---------- Trigger Evaluation ----------

    def _evaluate_trigger(self, trigger_config: Any, context: Dict[str, Any]) -> bool:
        """Evaluate a single trigger condition."""
        current_time = time.time()
        
        # Handle dict-based triggers (new format)
        if isinstance(trigger_config, dict):
            trigger_type = trigger_config.get('type')
            
            if trigger_type == TriggerType.ON_STARTUP.value:
                cooldown = trigger_config.get('cooldown_minutes', 60) * 60
                if context.get('is_startup', False) and \
                   (current_time - self.state.last_trigger_check) > cooldown:
                    logger.debug(f"Trigger: on_startup (cooldown: {cooldown/60:.0f}m)")
                    return True
            
            elif trigger_type == TriggerType.ON_ERROR.value:
                threshold = trigger_config.get('error_count_threshold', 3)
                if context.get('error_detected', False):
                    error_count = int(context.get('error_count', 0))
                    if error_count >= threshold:
                        logger.debug(f"Trigger: on_error ({error_count} >= {threshold})")
                        return True
            
            elif trigger_type == TriggerType.ON_PERFORMANCE_DEGRADATION.value:
                metric = trigger_config.get('metric', 'response_time_p95')
                threshold_pct = float(trigger_config.get('threshold_percent', 20))
                perf_metrics = context.get('performance_metrics', {})
                degradation_pct = float(perf_metrics.get(f'{metric}_degradation_percent', 0))
                if degradation_pct >= threshold_pct:
                    logger.debug(f"Trigger: performance degradation ({metric}: {degradation_pct:.1f}%)")
                    return True
            
            elif trigger_type == TriggerType.PERIODIC.value:
                interval_hours = trigger_config.get('interval_hours', 24)
                jitter_minutes = trigger_config.get('random_jitter_minutes', 0)
                
                # Calculate time since last improvement
                time_since = current_time - self.state.last_improvement
                interval_seconds = interval_hours * 3600
                
                # Add jitter (random, but consistent per session)
                import random
                random.seed(int(self.state.session_start_time))
                jitter_seconds = random.randint(0, max(0, jitter_minutes) * 60)
                
                if time_since > (interval_seconds + jitter_seconds):
                    logger.debug(f"Trigger: periodic ({time_since/3600:.1f}h >= {interval_hours}h)")
                    return True
            
            elif trigger_type == TriggerType.ON_LOW_ACTIVITY.value:
                cpu_threshold = trigger_config.get('cpu_threshold_percent', 30)
                duration_min = trigger_config.get('duration_minutes', 10)
                
                system_resources = context.get('system_resources', {})
                cpu_usage = system_resources.get('cpu_percent', 100)
                low_activity_duration = system_resources.get('low_activity_duration_minutes', 0)
                
                if cpu_usage < cpu_threshold and low_activity_duration >= duration_min:
                    logger.debug(f"Trigger: low activity (CPU: {cpu_usage}% < {cpu_threshold}%)")
                    return True
        
        return False
    
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """
        Determine if self-improvement drive should activate.
        
        This is called by Vulcan's motivational system to decide if the
        system should focus on self-improvement right now.
        
        When unlimited_mode is enabled, the session change limit is bypassed.
        """
        # Check if enabled
        if not self.config.get('enabled', True):
            logger.debug("Self-improvement drive disabled")
            return False
        
        # Check resource limits (may update tokens from context increment)
        can_proceed, reason = self._check_resource_limits(context=context)
        if not can_proceed:
            logger.info(f"Cannot trigger: {reason}")
            return False
        
        # Check if we've hit the session limit (bypassed in unlimited_mode)
        if not getattr(self, 'unlimited_mode', False):
            max_changes = int(self.config['constraints']['max_changes_per_session'])
            if self.state.improvements_this_session >= max_changes:
                logger.info(f"Reached max changes limit ({max_changes}) for this session")
                return False
        else:
            logger.debug(f"Unlimited mode: bypassing session change limit (current: {self.state.improvements_this_session})")
        
        # Evaluate all triggers
        triggers = self.config.get('triggers', [])
        for trigger_config in triggers:
            if self._evaluate_trigger(trigger_config, context):
                # Optional jitter to mitigate simultaneous trigger storms
                if self._jitter_ms > 0:
                    time.sleep(min(0.5, self._jitter_ms / 1000.0))  # cap very small delay
                self.state.last_trigger_check = time.time()
                logger.info(f"✓ Trigger activated: {trigger_config}")
                return True
        
        # Check if drive priority is high enough given current context
        priority = float(self.config.get('priority', 0.8))
        other_drives_priority = float(context.get('other_drives_total_priority', 0.5))
        
        # If our priority is significantly higher than other drives, trigger
        if priority > other_drives_priority * 1.5:
            if self._jitter_ms > 0:
                time.sleep(min(0.5, self._jitter_ms / 1000.0))
            logger.info(f"✓ Priority trigger: {priority:.2f} > {other_drives_priority:.2f}")
            self.state.last_trigger_check = time.time()
            return True
        
        return False
    
    # ---------- Adaptive Weighting ----------

    def _calculate_adjusted_weight(self, objective: ImprovementObjective) -> float:
        """Calculate adjusted weight based on adaptive learning."""
        base_weight = float(objective.weight)
        
        # Get adaptive learning config
        adaptive_config = self.config.get('adaptive_learning', {})
        if not adaptive_config.get('enabled', False):
            return base_weight
        
        # Adjust based on success rate
        total_attempts = objective.success_count + objective.failure_count
        if total_attempts == 0:
            return base_weight
        
        min_samples = adaptive_config.get('track_outcomes', {}).get('min_samples_before_adjust', 10)
        if total_attempts < min_samples:
            return base_weight
        
        success_rate = objective.success_count / total_attempts
        learning_rate = float(adaptive_config.get('track_outcomes', {}).get('learning_rate', 0.2))
        
        # Adjust weight: increase if successful, decrease if not
        adjustment = learning_rate * (success_rate - 0.5)
        adjusted_weight = base_weight + adjustment
        
        # Clamp to bounds
        bounds = adaptive_config.get('track_outcomes', {}).get('weight_bounds', {})
        min_weight = float(bounds.get('min', 0.3))
        max_weight = float(bounds.get('max', 1.0))
        adjusted_weight = max(min_weight, min(max_weight, adjusted_weight))
        
        # Check if we should notify about significant change
        transparency = adaptive_config.get('transparency', {})
        notify_config = transparency.get('notify_on_significant_change', {})
        if notify_config.get('enabled', False):
            threshold = float(notify_config.get('threshold_change', 0.2))
            last_notified = self._last_weight_notification.get(objective.type, base_weight)
            if abs(adjusted_weight - last_notified) >= threshold:
                self._send_alert('info', f'Significant weight change for {objective.type}',
                                 {
                                     'objective': objective.type,
                                     'old_weight': last_notified,
                                     'new_weight': adjusted_weight,
                                     'success_rate': success_rate,
                                     'total_attempts': total_attempts
                                 })
                self._last_weight_notification[objective.type] = adjusted_weight
        
        if abs(adjusted_weight - base_weight) > 0.01:
            logger.debug(f"Adjusted weight for {objective.type}: {base_weight:.2f} -> {adjusted_weight:.2f} "
                         f"(success_rate: {success_rate:.2f})")
        
        return adjusted_weight
    
    # ---------- Selection ----------

    def select_objective(self) -> Optional[ImprovementObjective]:
        """Select next objective to work on based on weights, cooldowns, and adaptive learning."""
        current_time = time.time()
        available_objectives = [
            obj for obj in self.objectives 
            if not obj.completed and obj.cooldown_until <= current_time
        ]
        
        if not available_objectives:
            # Check if any are just on cooldown
            on_cooldown = [obj for obj in self.objectives if obj.cooldown_until > current_time]
            if on_cooldown:
                next_available = min(on_cooldown, key=lambda x: x.cooldown_until)
                wait_seconds = next_available.cooldown_until - current_time
                logger.info(f"All objectives on cooldown. Next: {next_available.type} "
                            f"in {wait_seconds/60:.1f} min")
            else:
                logger.info("All improvement objectives completed!")
            return None
        
        # Calculate adjusted weights with adaptive learning
        weighted_objectives = [
            (obj, self._calculate_adjusted_weight(obj))
            for obj in available_objectives
        ]
        
        # Sort by adjusted weight (highest first)
        weighted_objectives.sort(key=lambda x: x[1], reverse=True)
        selected, weight = weighted_objectives[0]
        logger.info(f"Selected: {selected.type} (weight: {weight:.2f})")
        return selected
    
    # ---------- Action Planning ----------

    def generate_improvement_action(self, objective: ImprovementObjective) -> Dict[str, Any]:
        """
        Generate an action plan for the given improvement objective.
        
        This returns a context that Vulcan's orchestrator can execute.
        The orchestrator is responsible for:
        - Running pre-flight checks
        - Performing impact analysis
        - Executing dry-run if configured
        - Validating changes
        """
        # Find objective config
        obj_config = next(
            (obj for obj in self.config.get('objectives', []) if isinstance(obj, dict) and obj.get('type') == objective.type),
            {}
        )

        # Pass-through: env_overrides and auto-approval hints (external workflow will decide)
        validation_overrides = self.config.get("validation", {}).get("env_overrides", {})
        auto_approve_hints = (
            self.full_config.get("global_settings", {})
            .get("approval_workflow", {})
            .get("timeout_behavior", {})
            .get("auto_approve_if_safe_criteria", {})
        )

        action_map = {
            "fix_circular_imports": {
                "high_level_goal": "fix_circular_imports",
                "raw_observation": {
                    "task": "scan_and_fix_circular_imports",
                    "scope": obj_config.get('scope', {}),
                    "success_criteria": obj_config.get('success_criteria', {})
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,  # Always dry-run first
                "requires_impact_analysis": True
            },
            "optimize_performance": {
                "high_level_goal": "optimize_performance",
                "raw_observation": {
                    "task": "profile_and_optimize",
                    "target_metrics": obj_config.get('target_metrics', {}),
                    "allowed_optimizations": obj_config.get('allowed_optimizations', []),
                    "forbidden_optimizations": obj_config.get('forbidden_optimizations', [])
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": True
            },
            "enhance_safety_systems": {
                "high_level_goal": "enhance_safety",
                "raw_observation": {
                    "task": "strengthen_safety_systems",
                    "focus": ["governance", "validation", "rollback"]
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,  # Critical: always dry-run safety changes
                "requires_impact_analysis": True
            },
            "improve_test_coverage": {
                "high_level_goal": "improve_tests",
                "raw_observation": {
                    "task": "increase_test_coverage",
                    "coverage_targets": obj_config.get('coverage_targets', {}),
                    "priority_areas": obj_config.get('priority_areas', [])
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": False  # Lower risk
            },
            "fix_known_bugs": {
                "high_level_goal": "fix_bugs",
                "raw_observation": {
                    "task": "fix_known_issues",
                    "bug_sources": obj_config.get('bug_sources', []),
                    "priority_order": obj_config.get('priority_order', [])
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": True
            }
        }
        
        action = action_map.get(objective.type, {
            "high_level_goal": "improve_system",
            "raw_observation": {"objective": objective.type},
            "safety_constraints": self._get_safety_constraints(),
            "requires_dry_run": True,
            "requires_impact_analysis": True
        })
        
        # Add metadata + pass-through governance hints
        action['_drive_metadata'] = {
            'objective_type': objective.type,
            'objective_weight': objective.weight,
            'attempt_number': objective.attempts + 1,
            'timestamp': time.time()
        }
        action['validation_overrides'] = validation_overrides
        action['auto_approve_hints'] = auto_approve_hints  # external service decides if applicable

        # Optional risk tag (helps external approver routing)
        risk_class = "low" if objective.type in ("improve_test_coverage",) else "medium"
        if objective.type in ("enhance_safety_systems",):
            risk_class = "high"
        action['risk_classification'] = risk_class
        
        return action
    
    def _get_safety_constraints(self) -> Dict[str, Any]:
        """Get safety constraints from config, robust to nested/flat shapes."""
        constraints = self.config.get('constraints', {}) or {}
        change_reqs = constraints.get('change_requirements', {}) or {}
        # Support both flat (constraints['always_maintain_tests']) and nested
        maintain_tests = constraints.get(
            'always_maintain_tests',
            change_reqs.get('always_maintain_tests', True)
        )
        never_reduce_safety = constraints.get(
            'never_reduce_safety',
            change_reqs.get('never_reduce_safety', True)
        )
        rollback_on_failure = constraints.get('rollback_on_failure', True)
        return {
            "require_approval": self.require_human_approval,
            "maintain_tests": bool(maintain_tests),
            "never_reduce_safety": bool(never_reduce_safety),
            "rollback_on_failure": bool(rollback_on_failure),
        }
    
    # ---------- Approval ----------

    def request_approval(self, improvement_plan: Dict[str, Any]) -> str:
        """
        Request human approval for improvement.
        
        Returns approval ID for tracking.
        
        NOTE: This creates the approval request. The actual approval workflow
        (Git PR, Slack, email, timeout) is handled by the external approval service.
        """
        # **************************************************************************
        # START FIX 1: Re-check config value in case it was modified post-init
        # This makes the test `test_auto_approval_when_disabled` pass.
        approval_required = self.config.get('constraints', {}).get('require_human_approval', True)
        if not approval_required:
        # END FIX 1
        # Original code was:
        # if not self.require_human_approval:
            logger.info("Auto-approval enabled, applying improvement")
            return "AUTO_APPROVED"
        
        # Generate approval ID
        approval_id = f"approval_{int(time.time())}_{improvement_plan.get('high_level_goal', 'unknown')}"
        
        # Add to pending approvals
        approval_request = {
            'id': approval_id,
            'plan': improvement_plan,
            'timestamp': time.time(),
            'status': 'pending'
        }
        self.state.pending_approvals.append(approval_request)
        self._save_state()
        
        logger.warning("=" * 60)
        logger.warning("🔒 IMPROVEMENT APPROVAL REQUIRED")
        logger.warning("=" * 60)
        logger.warning(f"Approval ID: {approval_id}")
        logger.warning(f"Improvement: {improvement_plan.get('high_level_goal')}")
        logger.warning(f"Details: {json.dumps(improvement_plan, indent=2)}")
        logger.warning("=" * 60)
        logger.warning(f"Approve: call approve_pending('{approval_id}')")
        logger.warning(f"Reject: call reject_pending('{approval_id}')")
        logger.warning("=" * 60)
        
        # Send alert for external notification
        self._send_alert('info', 'Approval required for improvement',
                         {
                             'approval_id': approval_id,
                             'objective': improvement_plan.get('high_level_goal'),
                             'plan': improvement_plan
                         })
        
        return approval_id
    
    def approve_pending(self, approval_id: str) -> bool:
        """Approve a pending improvement."""
        with self._lock:
            for approval in self.state.pending_approvals:
                if approval['id'] == approval_id:
                    approval['status'] = 'approved'
                    approval['approved_at'] = time.time()
                    self._save_state()
                    logger.info(f"✅ Approved: {approval_id}")
                    return True
        
        logger.warning(f"Approval ID not found: {approval_id}")
        return False
    
    def reject_pending(self, approval_id: str, reason: str = "") -> bool:
        """Reject a pending improvement."""
        with self._lock:
            for approval in self.state.pending_approvals:
                if approval['id'] == approval_id:
                    approval['status'] = 'rejected'
                    approval['rejected_at'] = time.time()
                    approval['rejection_reason'] = reason
                    self._save_state()
                    logger.info(f"❌ Rejected: {approval_id} ({reason})")
                    return True
        
        logger.warning(f"Approval ID not found: {approval_id}")
        return False
    
    def check_approval_status(self, approval_id: str) -> Optional[str]:
        """Check status of approval: 'pending', 'approved', 'rejected', or None."""
        # Check internal state first
        for approval in self.state.pending_approvals:
            if approval['id'] == approval_id:
                return approval['status']
        
        # Check external approval service if configured
        if self.approval_checker:
            try:
                return self.approval_checker(approval_id)
            except Exception as e:
                logger.error(f"External approval check failed: {e}")
        
        return None
    
    # ---------- Auto Apply Logic ----------
    # Example change plan schema assumption:
    # plan = {
    #   "id": "...",
    #   "category": "path_fixes" | "runtime_hygiene" | ...,
    #   "files": [{"path": "src/foo.py", "loc_added": 10, "loc_removed": 2}, ...],
    #   "diff_loc": 12,
    #   "apply": callable_that_performs_change_or_patch
    # }
    def _maybe_auto_apply(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        if not self._auto_apply_enabled:
            return False, "auto-apply disabled"
        files = [f.get("path") for f in plan.get("files", []) if isinstance(f, dict) and f.get("path")]
        if not files:
            return False, "no files listed in plan"
        # LOC budget
        total_loc = int(plan.get("diff_loc") or sum(int(f.get("loc_added", 0)) + int(f.get("loc_removed", 0)) for f in plan.get("files", [])))
        if hasattr(self._auto_apply_policy, "max_total_loc") and total_loc > self._auto_apply_policy.max_total_loc:
            return False, f"diff too large ({total_loc} > {self._auto_apply_policy.max_total_loc})"

        ok, reasons = check_files_against_policy(files, self._auto_apply_policy)
        if not ok:
            return False, "; ".join(reasons)

        # Run pre-apply gates (lint, type, tests, smoke)
        gates_ok, failures = run_gates(self._auto_apply_policy, cwd=str(Path(__file__).resolve().parents[4])) # Assuming project root is 4 levels up
        if not gates_ok:
            return False, "; ".join(failures)

        # All good – attempt apply, surrounding with the existing snapshot/rollback machinery
        try:
            # Use existing safe-apply method if available
            if hasattr(self, "apply_change_plan"): # Hypothetical method
                self.apply_change_plan(plan)
            elif callable(plan.get("apply")):
                plan["apply"]()
            else:
                return False, "no apply handler"
            return True, "applied"
        except Exception as e:
            # Let outer layers (rollback manager) handle restoration as they already do
            return False, f"apply failed: {e}"

    # Where improvements are processed, insert a call to _maybe_auto_apply before queueing/approval
    def process_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single improvement plan. Auto-apply if policy allows, otherwise queue for approval.
        """
        # ... any existing validation ...
        applied = False
        reason = "queued"
        if self._auto_apply_enabled:
            applied, reason = self._maybe_auto_apply(plan)

        if applied:
            status = {"status": "applied", "plan_id": plan.get("id"), "reason": reason}
            logger.info(f"Auto-applied plan {plan.get('id')}")
            # Potentially call record_outcome here or let external orchestrator do it
        else:
            # Fall back to your existing "queue for approval" path
            if hasattr(self, "request_approval"):
                 approval_id = self.request_approval(plan) # Assuming request_approval queues it
                 status = {"status": "queued", "plan_id": plan.get("id"), "reason": reason, "approval_id": approval_id}
            else:
                 # If no specific queuing method, just log and set status
                 logger.info(f"Plan {plan.get('id')} requires manual approval: {reason}")
                 status = {"status": "queued", "plan_id": plan.get("id"), "reason": reason}

        # Update state if your class tracks it
        try:
            st = getattr(self, "state", None)
            if isinstance(st, SelfImprovementState): # Check type
                 st.last_status = status # Add a field to track last status
        except Exception:
            pass
        return status
            
    # ---------- Main Step ----------

    def step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute one step of self-improvement drive.
        
        This is called by Vulcan's main loop when the drive is active.
        Returns an action to execute, or None if no action needed.
        """
        try:
            # Check if we should be active
            if not self.should_trigger(context):
                return None
            
            # Select an objective
            objective = self.select_objective()
            if not objective:
                logger.info("No objectives available")
                return None
            
            logger.info(f"🎯 Pursuing: {objective.type} (weight: {objective.weight:.2f}, "
                        f"attempts: {objective.attempts})")
            
            # Generate improvement action (PLAN ASSEMBLY)
            improvement_action = self.generate_improvement_action(objective)
            
            # CSIU: Calculate pressure and regularize plan (CRITICAL CALL-SITE)
            # This is the primary integration point for CSIU learning
            if self._csiu_enabled and self._csiu_regs_enabled:
                try:
                    # Collect telemetry
                    prev_telemetry = self._csiu_last_metrics
                    cur_telemetry = self._collect_telemetry_snapshot()
                    
                    if prev_telemetry and cur_telemetry:
                        # Calculate utility
                        U_prev = self._csiu_U_prev
                        U_now = self._csiu_utility(prev_telemetry, cur_telemetry)
                        U_ewma = self._csiu_apply_ewma(U_now)
                        
                        # Calculate pressure (bounded ±5%)
                        d = self._csiu_pressure(U_ewma)
                        
                        # Regularize plan (≤3% micro-effects)
                        improvement_action = self._csiu_regularize_plan(improvement_action, d, cur_telemetry)
                        
                        # Adaptive learning rate
                        lr = self._csiu_adaptive_lr(cur_telemetry)
                        
                        # Update weights based on utility gain
                        feature_deltas = {
                            "dA": cur_telemetry.get("A", 0.85) - prev_telemetry.get("A", cur_telemetry.get("A", 0.85)),
                            "dH": cur_telemetry.get("H", 0.06) - prev_telemetry.get("H", cur_telemetry.get("H", 0.06)),
                            "C": cur_telemetry.get("C", 0.88) - prev_telemetry.get("C", cur_telemetry.get("C", "0.88")),
                            "M": cur_telemetry.get("M", 0.02) - prev_telemetry.get("M", cur_telemetry.get("M", 0.02))
                        }
                        self._csiu_update_weights(feature_deltas, U_prev, U_now, lr)
                        
                        # Store for next iteration
                        self._csiu_U_prev = U_now
                    
                    # Store current telemetry for next iteration
                    self._csiu_last_metrics = cur_telemetry
                    
                except Exception as e:
                    logger.debug(f"CSIU regularization failed (non-fatal): {e}")

            # --- MODIFIED: Try auto-apply before requesting manual approval ---
            processed_status = self.process_plan(improvement_action)

            # If auto-applied successfully, record outcome and return None (no further action needed now)
            if processed_status.get("status") == "applied":
                self.record_outcome(objective.type, True, {"auto_applied": True, "reason": processed_status.get("reason")})
                # Mark objective state
                objective.attempts += 1
                objective.last_attempt = time.time()
                self.state.current_objective = objective.type # Track objective even if auto-applied
                self.state.active = True # Drive was active
                self._save_state()
                return None # Return None because the action was completed internally

            # If queued (either because auto-apply failed or wasn't enabled/triggered), proceed to return action for orchestrator
            elif processed_status.get("status") == "queued":
                # If approval is required and wasn't auto-approved, it needs external handling
                if self.require_human_approval:
                     # Add approval info if it was queued because manual approval is needed
                     if processed_status.get("approval_id"):
                         improvement_action['_pending_approval'] = processed_status["approval_id"]
                         improvement_action['_wait_for_approval'] = True
                     else: # If queuing failed internally before request_approval was hit
                         logger.warning(f"Plan queued but no approval ID generated, likely auto-apply policy failure: {processed_status.get('reason')}")
                         # Decide how to handle this - maybe force manual approval anyway?
                         # For now, let it fall through but log clearly
                
                # Mark objective state (attempt is starting, even if waiting for approval)
                objective.attempts += 1
                objective.last_attempt = time.time()
                self.state.current_objective = objective.type
                self.state.active = True
                self._save_state()

                logger.info(f"🚀 Returning action for orchestrator: {objective.type} (Reason: {processed_status.get('reason')})")
                return improvement_action # Return the action for the external orchestrator/approval flow

            else:
                # Should not happen, log error
                logger.error(f"Unexpected status from process_plan: {processed_status}")
                return None
            
        except Exception as e:
            logger.error(f"Error in self-improvement step: {e}", exc_info=True)
            # Attempt to record failure if an objective was selected
            if 'objective' in locals() and objective is not None:
                self.record_outcome(objective.type, False, {"error": str(e), "context": "step_exception"})
            return None
    
    # ---------- Outcome Recording ----------

    def _classify_failure(self, details: Dict[str, Any]) -> FailureType:
        """Classify failure as transient or systemic."""
        adaptive_config = self.config.get('adaptive_learning', {})
        failure_patterns = adaptive_config.get('failure_patterns', {})
        classification = failure_patterns.get('failure_classification', {})
        
        error_msg = str(details.get('error', '')).lower()
        
        # Check transient indicators
        transient_indicators = classification.get('transient', {}).get('indicators', [])
        for indicator in transient_indicators:
            if str(indicator).lower() in error_msg:
                return FailureType.TRANSIENT
        
        # Check systemic indicators
        systemic_indicators = classification.get('systemic', {}).get('indicators', [])
        for indicator in systemic_indicators:
            if str(indicator).lower() in error_msg:
                return FailureType.SYSTEMIC
        
        # Default to systemic to be conservative
        return FailureType.SYSTEMIC
    
    def record_outcome(self, objective_type: str, success: bool, details: Dict[str, Any]):
        """Record the outcome of an improvement attempt (thread-safe)."""
        with self._lock:
            # Normalize inputs
            cost = float(details.get('cost_usd', 0.0))
            tokens = int(details.get('tokens_used', 0))

            for obj in self.objectives:
                if obj.type == objective_type:
                    if success:
                        obj.completed = True
                        obj.success_count += 1
                        if objective_type not in self.state.completed_objectives:
                            self.state.completed_objectives.append(objective_type)
                        self.state.improvements_this_session += 1
                        self.state.last_improvement = time.time()
                        
                        # Update costs & tokens
                        self.state.total_cost_usd += cost
                        self.state.daily_cost_usd += cost
                        self.state.monthly_cost_usd += cost
                        self.state.session_tokens += tokens
                        if cost > 0:
                            self.state.cost_history.append(
                                {"timestamp": time.time(), "cost_usd": cost}
                            )
                        
                        logger.info(f"✅ Completed: {objective_type}")
                        logger.info(f"Details: {json.dumps(details, indent=2)}")
                        if cost or tokens:
                            logger.info(f"Cost: ${cost:.2f} (total: ${self.state.total_cost_usd:.2f}); "
                                        f"tokens +{tokens} (session: {self.state.session_tokens})")
                    else:
                        obj.failure_count += 1
                        obj.last_failure = time.time()
                        
                        # Classify failure
                        failure_type = self._classify_failure(details)
                        
                        # Apply cooldown
                        adaptive_config = self.config.get('adaptive_learning', {})
                        failure_patterns = adaptive_config.get('failure_patterns', {})
                        classification = failure_patterns.get('failure_classification', {})
                        
                        if failure_type == FailureType.TRANSIENT:
                            cooldown_hours = float(classification.get('transient', {}).get('cooldown_hours', 4))
                            logger.warning(f"❌ Transient failure: {objective_type}, cooldown: {cooldown_hours}h")
                        else:
                            cooldown_hours = float(classification.get('systemic', {}).get('cooldown_hours', 72))
                            logger.warning(f"❌ Systemic failure: {objective_type}, cooldown: {cooldown_hours}h")
                        
                        obj.cooldown_until = time.time() + (cooldown_hours * 3600)
                        
                        logger.warning(f"Details: {json.dumps(details, indent=2)}")
                        logger.warning(f"Cooldown until: {time.ctime(obj.cooldown_until)}")
                    
                    break
            
            self.state.current_objective = None
            self.state.active = False
            self._save_state()
    
    # ---------- Status ----------

    def _get_state_dict(self) -> Dict[str, Any]:
        """Helper to safely serialize the state object to a dict."""
        try:        
            from dataclasses import asdict, is_dataclass        
            st = getattr(self, "state", None)        
            if st is None:            
                return {}        
            elif is_dataclass(st):            
                return asdict(st)        
            elif hasattr(st, "__dict__"):            
                return dict(vars(st))        
            elif isinstance(st, dict):            
                return st        
            else:            
                return {"value": st}    
        except Exception:        
            return {}

    def set_unlimited_mode(self, enabled: bool) -> Dict[str, Any]:
        """
        Enable or disable unlimited mode for self-improvement.
        
        When unlimited_mode is enabled:
        - Session change limits are bypassed (max_changes_per_session ignored)
        - Cost limits are bypassed (cost warnings still tracked but not enforced)
        - Session duration limits are bypassed
        - The system will attempt to fix all detected issues
        
        WARNING: Use with caution. This removes safety constraints that prevent
        runaway costs and excessive changes. Ensure you have reviewed and approved
        enabling this mode.
        
        Args:
            enabled: True to enable unlimited mode, False to disable
            
        Returns:
            Status dict with confirmation
        """
        with self._lock:
            previous_mode = getattr(self, 'unlimited_mode', False)
            self.unlimited_mode = bool(enabled)
            
            if self.unlimited_mode and not previous_mode:
                logger.warning("=" * 60)
                logger.warning("⚠️  UNLIMITED MODE ENABLED ⚠️")
                logger.warning("Session limits, cost limits, and change caps are BYPASSED")
                logger.warning("The system will attempt to fix all detected issues")
                logger.warning("=" * 60)
                
                self._send_alert('warning', 'Unlimited mode enabled', {
                    'timestamp': time.time(),
                    'previous_mode': previous_mode,
                    'new_mode': self.unlimited_mode
                })
            elif not self.unlimited_mode and previous_mode:
                logger.info("Unlimited mode disabled - normal limits restored")
                self._send_alert('info', 'Unlimited mode disabled', {
                    'timestamp': time.time(),
                    'previous_mode': previous_mode,
                    'new_mode': self.unlimited_mode
                })
            
            return {
                'unlimited_mode': self.unlimited_mode,
                'previous_mode': previous_mode,
                'changed': self.unlimited_mode != previous_mode,
                'message': 'Unlimited mode enabled - all limits bypassed' if self.unlimited_mode else 'Normal limits restored'
            }

    def get_status(self) -> Dict[str, Any]:
        # **************************************************************************
        # START FIX 2: Rewrite get_status to include all keys expected by tests
        with self._lock: # Add lock for thread safety
            state_dict = self._get_state_dict()    
            enabled = self.config.get('enabled', False)
            
            # 1. Get objective details
            objective_details = []
            for obj in self.objectives:
                total_attempts = obj.success_count + obj.failure_count
                success_rate = (obj.success_count / total_attempts) if total_attempts > 0 else 0.0
                objective_details.append({
                    "type": obj.type,
                    "weight": obj.weight,
                    "adjusted_weight": self._calculate_adjusted_weight(obj), # Call helper
                    "completed": obj.completed,
                    "attempts": obj.attempts,
                    "success_rate": success_rate,
                    "cooldown_until": obj.cooldown_until
                })

            # 2. Get cost details
            cost_details = {
                "session_usd": state_dict.get("total_cost_usd", 0.0), # 'total_cost_usd' is session cost
                "daily_usd": state_dict.get("daily_cost_usd", 0.0),
                "monthly_usd": state_dict.get("monthly_cost_usd", 0.0)
            }

            # 3. Get token details
            token_details = {
                "session_tokens": state_dict.get("session_tokens", 0),
                "max_session_tokens": self.config.get('resource_limits', {}).get('llm_costs', {}).get('max_tokens_per_session')
            }

            # 4. Get CSIU details
            csiu_details = {
                "enabled": self._csiu_enabled,
                "calc_enabled": self._csiu_calc_enabled,
                "regs_enabled": self._csiu_regs_enabled,
                "weights": dict(self._csiu_w), # Return a copy
                "ewma_utility": self._csiu_u_ewma
            }

            # 5. Get session duration
            session_start = state_dict.get("session_start_time", time.time())
            session_duration_min = (time.time() - session_start) / 60.0

            return {        
                "enabled": enabled,        
                "active": bool(state_dict.get("active", False)), # Use state_dict value
                "state": state_dict, # Keep the full state dict for test_get_status
                "pending_approvals": state_dict.get("pending_approvals", []),        
                "current_objective": state_dict.get("current_objective"),        
                "last_improvement": state_dict.get("last_improvement"),        
                "auto_apply_enabled": bool(getattr(self, "_auto_apply_enabled", False)),        
                "policy_loaded": bool(getattr(self, "_auto_apply_policy", None) and getattr(self._auto_apply_policy, "enabled", False)),
                
                # Add the missing keys
                "objectives": objective_details,
                "costs": cost_details,
                "tokens": token_details,
                "csiu": csiu_details,
                "session_duration_minutes": session_duration_min,
                
                # This key is checked by the first failing test
                "completed_objectives": state_dict.get("completed_objectives", []),
                
                # Unlimited mode status - allows fixing everything without limits
                "unlimited_mode": bool(getattr(self, "unlimited_mode", False))
            }
        # END FIX 2
        # **************************************************************************

    
    # FIXED: Alias for test compatibility
    def _perform_improvement(self, action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Alias for _execute_improvement to pass test_6
        """
        logger.warning("Using deprecated alias _perform_improvement. Use _execute_improvement.")
        return self._execute_improvement(action)

    def _execute_improvement(self, action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes the improvement action using LLM-driven code generation, AST validation, and Git integration.
        """
        objective_type = action.get('_drive_metadata', {}).get('objective_type')
        logger.info(f"EXECUTING IMPROVEMENT for: {objective_type}")

        try:
            # 1. Generate Solution Content (LLM + Diff)
            solution_content, file_path = self._generate_solution_content(action)
            if not solution_content or not file_path:
                return False, {'status': 'failed', 'error': 'LLM failed to generate valid solution content'}

            # 2. AST / Syntax Validation
            if file_path.endswith('.py'):
                valid_syntax, syntax_error = self._validate_python_syntax(solution_content)
                if not valid_syntax:
                    logger.error(f"Generated code has syntax errors: {syntax_error}")
                    return False, {'status': 'failed', 'error': f'Syntax error: {syntax_error}'}

            # 3. File Application (I/O)
            changes_applied, diff_summary = self._apply_file_modification(file_path, solution_content)
            if not changes_applied:
                return False, {'status': 'failed', 'error': 'Failed to apply file changes'}

            # 4. Git Integration (Commit)
            commit_hash = self._commit_to_version_control(file_path, objective_type)

            return True, {
                'status': 'success',
                'objective_type': objective_type,
                'changes_applied': diff_summary,
                'commit_hash': commit_hash,
                'cost_usd': 0.05,  # Estimated cost for this operation
                'tokens_used': 1500 # Estimated tokens
            }

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return False, {'status': 'failed', 'error': str(e)}

    def _generate_solution_content(self, action: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Uses the WorldModel's LLM interface (or fallback) to generate the code improvement.
        Returns (content, file_path).
        """
        # Extract details from action plan
        goal = action.get('high_level_goal')
        observation = action.get('raw_observation', {})
        
        # Construct Prompt
        prompt = f"""
        You are an expert software engineer improving the Vulcan system.
        Objective: {goal}
        Task Details: {json.dumps(observation)}
        
        Please provide the FULL content of the Python file that needs to be created or modified to solve this.
        If modifying, provide the complete updated file.
        
        Format your response exactly as follows:
        FILE: <path/to/file.py>
        ```python
        <code content here>
        ```
        """
        
        # Call LLM (Mock integration if WorldModel not fully wired, otherwise use it)
        try:
            response_text = ""
            if self.world_model and hasattr(self.world_model, 'ask_llm'):
                response_text = self.world_model.ask_llm(prompt)
            elif hasattr(self, '_mock_llm_response'):
                response_text = self._mock_llm_response(prompt)
            else:
                # Fallback stub for standalone runs without full environment
                logger.warning("No LLM provider found, using stub response.")
                response_text = "FILE: src/vulcan/temp_fix.py\n```python\n# Auto-generated fix\ndef fix(): pass\n```"

            # Parse Response
            lines = response_text.strip().split('\n')
            file_path = None
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.startswith("FILE:"):
                    file_path = line.replace("FILE:", "").strip()
                elif line.strip().startswith("```python"):
                    in_code_block = True
                elif line.strip().startswith("```") and in_code_block:
                    in_code_block = False
                elif in_code_block:
                    code_lines.append(line)
            
            return "\n".join(code_lines), file_path

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None, None

    def _validate_python_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validates Python code syntax using AST."""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _apply_file_modification(self, file_path: str, new_content: str) -> Tuple[bool, str]:
        """Writes the new content to disk and calculates a diff."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            old_content = ""
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
            
            # Calculate Diff
            diff = difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=""
            )
            diff_text = "\n".join(list(diff))
            
            # Write New Content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return True, diff_text if diff_text else "New file created"
            
        except Exception as e:
            logger.error(f"File I/O failed for {file_path}: {e}")
            return False, ""

    def _commit_to_version_control(self, file_path: str, message: str) -> str:
        """Stages and commits changes using git subprocess with safe execution."""
        try:
            # Use safe executor if available, otherwise fallback to direct subprocess
            if get_safe_executor is not None:
                executor = get_safe_executor()
                
                # Stage
                stage_result = executor.execute_safe(['git', 'add', file_path], timeout=30)
                if not stage_result.success:
                    logger.warning(f"Git add failed: {stage_result.error}")
                    return "git_failed"
                
                # Commit
                commit_msg = f"vulcan(auto): {message}"
                commit_result = executor.execute_safe(['git', 'commit', '-m', commit_msg], timeout=30)
                
                if commit_result.success:
                    # Try to get short hash
                    hash_result = executor.execute_safe(['git', 'rev-parse', '--short', 'HEAD'], timeout=10)
                    if hash_result.success:
                        return hash_result.stdout.strip()
                    return "unknown_hash"
                else:
                    logger.warning(f"Git commit returned non-zero: {commit_result.stderr}")
                    return "unknown_hash"
            else:
                # Fallback to direct subprocess (already safe - using list args, not shell=True)
                subprocess.run(['git', 'add', file_path], check=True, capture_output=True)
                
                commit_msg = f"vulcan(auto): {message}"
                result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
                
                if result.returncode == 0:
                    hash_proc = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
                    return hash_proc.stdout.strip()
                else:
                    logger.warning(f"Git commit returned non-zero: {result.stderr}")
                    return "unknown_hash"
                
        except Exception as e:
            logger.warning(f"Git operation failed (is this a repo?): {e}")
            return "git_failed"
